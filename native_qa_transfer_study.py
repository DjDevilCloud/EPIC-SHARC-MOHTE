"""Reproducible gates for the bounded native document-QA study.

No label-derived intervention is used by normal inference. The oracle below
is confined to this evaluator and deliberately leaves the learned copy gate
unchanged. A failed prerequisite prevents the learning phases from starting.
"""
import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from data import TextWindowDataset, build_collate_fn
from native_qa_review import reviewed_indices, qa_fingerprint
from native_question_memory import NativeQuestionState
from source_answer_span import CompleteAnswerSpanPredictor
from train import load_bundle_from_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'review_artifacts/native_qa_transfer_v1'
SEEDS = (47, 83)
BASELINE_HASHES = {47: 'a0c3ecefe77c0a1346063c63990f442c264ce7e1f1f801df63f528073b3e0358',
                   83: 'e1093251728cf100e32b75f2eb7718bc8e43a20a836005614affd3ca0206c35f'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def prompt(row):
    return f"Context:\n{row['context']}\nQuestion:\n{row['question']}"


def generation_fields(t, row):
    p = t.prepare_generation_hierarchy(prompt(row))
    return torch.tensor([p.token_ids]), {k: torch.tensor([getattr(p, k)]) for k in
        ('signature_ids', 'signature_level_ids', 'signature_relation_ids', 'parent_signature_ids', 'signature_family_ids')}


def generate(m, t, row, oracle=False):
    ids, tracks = generation_fields(t, row)
    original = m.native_question_memory.forward
    def oracle_readout(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('return_readout'):
            hidden, states, records = result
            replaced = []
            for r, pos, q, current, plan, emitted in records:
                injected = dict(plan, candidates=[dict(tokens=CompleteAnswerSpanPredictor.payload(t, row['answer']))],
                                probabilities=q.new_ones(1), log_probabilities=q.new_zeros(1))
                replaced.append((r, pos, q, current, injected, emitted))
            return hidden, states, replaced
        return result
    options = dict(max_new_tokens=max(64, len(CompleteAnswerSpanPredictor.payload(t, row['answer'])) + 2),
        min_new_tokens=0, temperature=0., top_k=1, top_p=1., repetition_penalty=1.,
        no_repeat_ngram_size=0, use_speculative_decoding=False,
        suppressed_token_ids=t.generation_suppressed_token_ids(), token_signature_lookup=t.signature_lookup_by_token_id(),
        token_family_lookup=t.signature_family_lookup_by_token_id(), token_level_lookup=t.signature_level_lookup_by_token_id(),
        token_relation_lookup=t.signature_relation_lookup_by_token_id())
    with torch.no_grad(), patch.object(m.native_question_memory, 'forward', side_effect=oracle_readout if oracle else original):
        emitted = m.generate(ids, **tracks, **options)[0, ids.shape[1]:].tolist()
    return dict(raw=t.decode(emitted, clean_text=False, collapse_structure=False), tokens=emitted,
                exact=t.decode(emitted, clean_text=False, collapse_structure=False).strip() == row['answer'],
                ended=bool(emitted and emitted[-1] == t.eos_id))


def source_packet(m, t, row):
    ids, _ = generation_fields(t, row)
    mem = m.native_question_memory
    with torch.no_grad():
        _, states = mem(ids, torch.zeros(1, ids.shape[1], m.cfg.d_model), m.construction_embedding, t, diagnostics=True)
    state = states[0]
    q, recalled, prob, plan = state.cached_packet
    gold = NativeQuestionState()
    for token in t.encode('<BOI>Context:\n' + row['answer'] + '<EOI>', add_special_tokens=False):
        gold, _ = mem._observe(gold, token, t)
    source, target = [w for w, _ in state.source], [w for w, _ in gold.source]
    starts = [i for i in range(len(source) - len(target) + 1) if source[i:i + len(target)] == target]
    candidates = mem.span_readout.candidates(state, t, starts)
    with torch.no_grad():
        scores = mem.span_readout.score_candidates(candidates, q, plan['word_vectors'], plan['start_log_probabilities'], plan['marker_words']) if candidates else q.new_empty(0)
    order = plan['start_log_probabilities'].argsort(descending=True).tolist()
    selected = plan['candidates'][int(plan['probabilities'].argmax())]
    ledger = dict(fingerprint=qa_fingerprint(row), source_words=len(source), words_seen=state.source_seen,
        representable=any(c['text'] == row['answer'] for c in candidates),
        start_rank=min((order.index(i) + 1 for i in starts), default=None),
        shortlist_inclusion=any(c['text'] == row['answer'] for c in plan['candidates']),
        gold_start_extent=bool(candidates and candidates[int(scores.argmax())]['text'] == row['answer']),
        proposal=selected['text'], proposal_correct=selected['text'] == row['answer'])
    return state, plan, ledger


def cached_live_check(m, t, row, state, plan):
    mem = m.native_question_memory
    with torch.no_grad():
        cached = mem.decision_packet(plan['legacy_question'], plan['base_word_vectors'],
            mem.value(plan['base_word_vectors']), plan['legacy_start_logits'], plan['marker_words'],
            plan['semantic_inputs'], lambda selected: mem.span_readout.candidates(state, t, selected),
            plan['source_words'], diagnostics=True)
    live = state.cached_packet
    differences = []
    for a, b in zip(live[:3], cached[:3]):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-5)
        differences.append(float((a - b).abs().max()))
    if plan['candidates'] != cached[3]['candidates']:
        raise AssertionError('Cached/live candidate identities differ')
    torch.testing.assert_close(plan['log_probabilities'], cached[3]['log_probabilities'], rtol=2e-4, atol=3e-5)
    return max(differences)


def full_incremental_check(m, t, row):
    dataset = TextWindowDataset([f"<BOI>{prompt(row)}<EOI><BOO>{row['answer']}<EOO>"], t,
                               seq_len=2048, max_samples=0, preserve_grounded_qa_context=True)
    assert len(dataset) == 1
    fields = ('input_ids', 'labels', 'signature_ids', 'signature_level_ids', 'signature_relation_ids',
              'parent_signature_ids', 'signature_family_ids', 'hierarchy_vectors', 'loss_mask')
    b = dict(zip(fields, build_collate_fn(t.pad_id, t.signature_pad_id)([dataset[0]])))
    prefix, _ = generation_fields(t, row)
    split = prefix.shape[1]
    assert torch.equal(prefix, b['input_ids'][:, :split])
    tracks = {k: v for k, v in b.items() if k not in ('input_ids', 'labels', 'loss_mask', 'hierarchy_vectors')}
    with torch.no_grad():
        full = m(b['input_ids'], **tracks, collect_telemetry=False)
        prefix_out = m(prefix, **{k: v[:, :split] for k, v in tracks.items()}, collect_telemetry=False)
        torch.testing.assert_close(full.logits[:, split - 1], prefix_out.logits[:, -1], rtol=2e-4, atol=3e-5)
        output = prefix_out
        maximum = float((full.logits[:, split - 1] - output.logits[:, -1]).abs().max())
        for i in range(split, b['input_ids'].shape[1]):
            _, _, output = m.forward_incremental(b['input_ids'][:, i:i+1],
                **{k: v[:, i:i+1] for k, v in tracks.items()}, position_index=i,
                slot_state=output.slot_state, signature_lattice_state=output.signature_lattice_state,
                runtime_signature_state=output.runtime_signature_state, bounded_identity_state=output.bounded_identity_state)
            torch.testing.assert_close(full.logits[:, i], output.logits[:, -1], rtol=2e-4, atol=3e-5)
            maximum = max(maximum, float((full.logits[:, i] - output.logits[:, -1]).abs().max()))
    return maximum


def reviewed_diagnostics():
    manifest = json.loads((ROOT / 'review_artifacts/native_real_qa_comparison/manifest.json').read_text())
    review_path = ROOT / 'review_artifacts/native_retention_repair/annotation_audit.json'
    audit = json.loads(review_path.read_text())
    assert audit['source_manifest_sha256'] == digest(ROOT / 'review_artifacts/native_real_qa_comparison/manifest.json')
    rows = []
    for split, records in manifest['splits'].items():
        for i in reviewed_indices(records, audit['reviews'][split]):
            rows.append(dict(records[i], diagnostic_split=split, original_index=i))
    counts = dict(original=dict(audit['original_counts']), reviewed=dict(audit['retained_counts']), review_sha256=digest(review_path))
    broader_path = OUT / 'data/broader_review.json'
    if broader_path.exists():
        broader = json.loads((ROOT / 'review_artifacts/native_owner_markers/seed47/manifest.json').read_text())['groups']['broader_qa']
        reviews = json.loads(broader_path.read_text())['reviews']
        for i in reviewed_indices(broader, reviews):
            rows.append(dict(broader[i], diagnostic_split='broader', original_index=i))
        counts['original']['broader'] = len(broader)
        counts['reviewed']['broader'] = len(rows) - sum(audit['retained_counts'].values())
        counts['broader_review_sha256'] = digest(broader_path)
    return rows, counts


def contract(seed):
    folder = OUT / f'seed{seed}'
    source = ROOT / f'review_artifacts/native_retained_decisions_counterfactual/seed{seed}/candidate/model.pt'
    baseline = ROOT / f'review_artifacts/native_owner_markers/seed{seed}/candidate/model.pt'
    assert digest(baseline) == BASELINE_HASHES[seed]
    m, t, _ = load_bundle_from_checkpoint(source, device='cpu', load_training_state=False)
    m.eval(); m._prismal_tokenizer = t
    before = {k: v.clone() for k, v in m.state_dict().items()}
    m.cfg.qa_readout_route = 'native'
    route = folder / 'route_native'
    save_checkpoint(m, route, tokenizer=t, metrics=dict(experimental=True, accepted=False, phase='execution_contract'))
    reloaded, rt, cfg = load_bundle_from_checkpoint(route / 'model.pt', device='cpu', load_training_state=False)
    reloaded.eval(); reloaded._prismal_tokenizer = rt
    assert cfg.qa_readout_route == 'native' and reloaded.bounded_identity_readout is not None
    assert all(torch.equal(v, reloaded.state_dict()[k]) for k, v in before.items())
    rows, counts = reviewed_diagnostics()
    # Historical native-path raw outputs are an independent route migration reference.
    original = ROOT / f'review_artifacts/native_retained_decisions_counterfactual/seed{seed}'
    reference = {}
    for path in original.glob('generation_candidate*.json'):
        for group in json.loads(path.read_text()).values():
            for r in group['outputs']:
                reference[qa_fingerprint(r)] = r['raw']
    report_path = folder / 'contract.json'
    previous_report = json.loads(report_path.read_text()) if report_path.exists() else None
    checked_code = {name:digest(ROOT/name) for name in ('model.py', 'native_question_memory.py', 'native_span_readout.py', 'native_semantic_alignment.py', 'native_qa_transfer_study.py')}
    report = dict(seed=seed, source=str(source), source_sha256=digest(source), baseline_sha256=digest(baseline), complete=False,
        counts=counts, runtime=dict(python=platform.python_version(), torch=torch.__version__, device='cpu', threads=1),
        parameters_buffers_unchanged=True, routes_saved=True, rows=[], errors=[], passed=False, checks_code_sha256=checked_code)
    if previous_report:
        report['resumed_contract_report_sha256'] = digest(report_path)
    previous_rows = {r.get('fingerprint'):r for r in previous_report['rows']} if previous_report and previous_report['source_sha256'] == digest(source) and previous_report.get('checks_code_sha256') == checked_code else {}
    for i, row in enumerate(rows):
        if qa_fingerprint(row) in previous_rows and 'error' not in previous_rows[qa_fingerprint(row)]:
            result = previous_rows[qa_fingerprint(row)]
            report['rows'].append(result)
            if not result.get('historical_raw_match') or not result.get('oracle_exact_payload'):
                report['errors'].append(dict(index=i, question=row['question'], resumed_failed_row=True))
            continue
        result = dict(question=row['question'], answer=row['answer'], diagnostic_split=row['diagnostic_split'],
                      original_index=row['original_index'])
        try:
            state, plan, ledger = source_packet(reloaded, rt, row)
            result.update(ledger)
            result['cached_live_maximum_difference'] = cached_live_check(reloaded, rt, row, state, plan)
            result['full_incremental_maximum_difference'] = full_incremental_check(reloaded, rt, row)
            normal = generate(reloaded, rt, row)
            result['raw'] = normal
            result['historical_raw_match'] = normal['raw'] == reference[qa_fingerprint(row)]
            oracle = generate(reloaded, rt, row, oracle=True)
            result['oracle'] = oracle
            result['oracle_exact_payload'] = tuple(oracle['tokens']) == CompleteAnswerSpanPredictor.payload(rt, row['answer'])
            if not result['historical_raw_match'] or not result['oracle_exact_payload']:
                report['errors'].append(dict(index=i, question=row['question'], historical_match=result['historical_raw_match'], oracle_exact=result['oracle_exact_payload']))
        except (AssertionError, ValueError, RuntimeError, KeyError) as exc:
            result['error'] = str(exc)
            report['errors'].append(dict(index=i, question=row['question'], error=str(exc)))
        report['rows'].append(result)
        dump(folder / 'contract.json', report)
        print('CONTRACT', seed, i + 1, len(rows), len(report['errors']), row['question'], flush=True)
    report['passed'] = not report['errors'] and len(report['rows']) == len(rows)
    report['complete'] = True
    changed = [k for k, value in reloaded.state_dict().items() if not torch.equal(value, before[k])]
    report['runtime_changed_buffers_parameters'] = changed
    if changed:
        report['errors'].append(dict(runtime_state_changed=changed))
        report['passed'] = False
    assert digest(baseline) == BASELINE_HASHES[seed]
    dump(folder / 'contract.json', report)
    print('CONTRACT_COMPLETE', seed, report['passed'], len(report['errors']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('contract', 'status'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'contract':
        for seed in ((args.seed,) if args.seed else SEEDS):
            contract(seed)
    else:
        reports = {str(seed): json.loads((OUT / f'seed{seed}/contract.json').read_text()) for seed in SEEDS}
        passed = all(r.get('complete', len(r['rows']) == 89) and r['passed'] for r in reports.values())
        inventory_path = OUT / 'data/review_candidates.json'
        inventory = json.loads(inventory_path.read_text())['inventory'] if inventory_path.exists() else None
        other_path = OUT / 'data/other_local_shards.json'
        other = json.loads(other_path.read_text()) if other_path.exists() else {}
        possible_extra_groups = sum(v['parsed_document_qa_rows'] for v in other.values())
        group_upper_bound = inventory['unseen_eligible_groups_including_single_question'] + possible_extra_groups if inventory else None
        data_gate = False  # Only a complete, fingerprinted reviewed manifest can open this gate.
        manifest_path = OUT / 'data/frozen_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            data_gate = True
            from native_qa_transfer_data import validate_review
            groups = []
            for split, n, g in (('train', 128, 64), ('development', 64, 32), ('test', 64, 32)):
                records = manifest['splits'][split]
                assert len(records) == n and len({r['source_group'] for r in records}) >= g
                for row in records:
                    validate_review(row, row['review'])
                groups.append({r['source_group'] for r in records})
            assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        result = dict(accepted=False, baseline_replaced=False, execution_gate_passed=passed, data_gate_passed=data_gate,
                      data_inventory=inventory, other_local_shards=other, source_group_upper_bound=group_upper_bound,
                      required_source_groups=128, stop_reason=('Insufficient fresh annotated source groups' if group_upper_bound is not None and group_upper_bound < 128 else 'Reviewed data not frozen') if not data_gate else ('Execution contract failed' if not passed else None),
                      learning_updates=0, test_evaluations=0,
                      stages={name:'not_run_prerequisite_failed' for name in ('source_control', 'source_structure', 'source_question_attention', 'fusion', 'extent', 'joint', 'locked_test')},
                      training_authorized_by_gates=passed and data_gate, seeds={s:dict(passed=r['passed'], errors=r['errors'],
                      completed=len(r['rows']), counts=r['counts']) for s,r in reports.items()})
        dump(OUT / 'acceptance.json', result)
        protocol = dict(version=1, seeds=list(SEEDS), qa_readout_route='native',
            dataset_questions=dict(train=128, development=64, test=64),
            minimum_source_groups=dict(train=64, development=32, test=32),
            shingle_containment_threshold=.8, pretrained_encoder=False, expand_width=False, expand_memory=False,
            source_arms=['control', 'structure', 'question_attention'], source_updates=1600, fusion_updates=800,
            extent_updates=800, joint_updates=1600, joint_evaluation_interval=200,
            source_batch=dict(new=8, reviewed_legacy=2, native_replay=2, controlled_replay=2),
            source_gate=dict(training_top1_fraction=.95, development_top1=32, development_top16=58, fresh_value_starts=16),
            extent_gate=dict(development_gold_start_extent=58),
            transfer_gate=dict(development_exact=32, locked_test_exact=32, improvement_over_control=8),
            retention=dict(reviewed_legacy=58, native_replay=60, controlled_training=288, fresh_values=16,
                           owner_gap0=36, owner_gap24=36, prior_layouts=32, each_grammar=16),
            stop_on_prerequisite_failure=True, no_new_study_test_evaluations=True,
            code_sha256={name:digest(ROOT/name) for name in ('model.py', 'config.py', 'native_question_memory.py',
                'native_semantic_alignment.py', 'native_span_readout.py', 'native_qa_transfer_study.py', 'native_qa_transfer_data.py',
                'review_artifacts/prepare_native_transfer_review.py', 'review_artifacts/review_native_transfer_broader.py')})
        dump(OUT / 'protocol.json', protocol)
        result['protocol_sha256'] = digest(OUT / 'protocol.json')
        result['evidence_sha256'] = {str(path.relative_to(OUT)):digest(path) for path in
            (OUT / 'data/review_candidates.json', OUT / 'data/exclusion_snapshot.json', OUT / 'data/broader_review.json',
             OUT / 'seed47/contract.json', OUT / 'seed83/contract.json') if path.exists()}
        result['tests'] = {}
        for name in ('native_transfer_final_tests.log', 'native_transfer_followup_tests.log'):
            log = ROOT / 'review_artifacts' / name
            content = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
            ran = re.search(r'Ran (\d+) tests?', content)
            success = re.search(r'^OK(?: \(skipped=(\d+)\))?$', content, re.MULTILINE)
            skipped = int(success.group(1) or 0) if success else None
            result['tests'][name] = dict(ran=int(ran.group(1)) if ran else None, passed=(int(ran.group(1))-skipped if ran and success else None),
                                          skipped=skipped, ok=bool(success), sha256=digest(log) if log.exists() else None)
        dump(OUT / 'acceptance.json', result)
        lines = ['arm,seed47_updates,seed83_updates,status'] + [f'{name},0,0,not_run_data_gate_failed' for name in ('control','structure','question_attention','fusion','extent','joint')]
        (OUT / 'comparison.csv').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
