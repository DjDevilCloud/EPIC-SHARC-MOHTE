"""Validate the frozen passage benchmark without evaluating a model on it."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from data import parse_diverse_qa_record
from native_qa_review import qa_fingerprint
from native_qa_transfer_data import fixed_passages, group_sources, validate_review
from native_question_memory import NativeQuestionState
from source_answer_span import CompleteAnswerSpanPredictor
from train import load_bundle_from_checkpoint
from review_native_qa_passages import DATA, ROOT, materialized_reviews


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    import pyarrow.parquet as pq
    torch.set_num_threads(1)
    candidates = json.loads((DATA/'candidates.json').read_text())
    bundle = json.loads((DATA/'reviews.json').read_text())
    manifest = json.loads((DATA/'frozen_manifest.json').read_text())
    exclusion_path = ROOT/'review_artifacts/native_qa_transfer_v1/data/exclusion_snapshot.json'
    exclusion = json.loads(exclusion_path.read_text())
    source = Path(r'D:\AAISourceStart\SourceALLM\Datasets\NVIDIAPT\DiverseQA\part_000000.parquet')
    assert sha(source) == candidates['source_sha256'] == manifest['source_sha256']
    assert sha(exclusion_path) == candidates['exclusion_sha256'] == manifest['exclusion_sha256']
    assert sha(DATA/'candidates.json') == bundle['candidate_sha256']
    canonical = dict(candidates, manual_family_links=bundle.get('family_links', []))
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest() == manifest['candidate_fingerprint']
    reviews = materialized_reviews(candidates, bundle)
    usable_families=set()
    for document in candidates['documents']:
        retained=[r for r in document['records'] if reviews.get(qa_fingerprint(r),{}).get('decision')=='retain']
        for i,a in enumerate(retained):
            if any(a['context']==b['context'] and a['answer']!=b['answer'] for b in retained[i+1:]):
                usable_families.add(reviews[qa_fingerprint(a)].get('related_family',document['source_group']))
                break
    originals = {}
    for row in pq.read_table(source).to_pylist():
        parsed = parse_diverse_qa_record(row['text'])
        if parsed:
            assert row['id'] not in originals
            originals[row['id']] = parsed
    inventory = {qa_fingerprint(r): r for d in candidates['documents'] for r in d['records']}
    counts = {}; owners = {}; all_rows = []
    for split, expected in [('train',128), ('development',64), ('test',64)]:
        rows = manifest['splits'][split]
        assert len(rows) == expected
        groups = {}
        for row in rows:
            fingerprint = qa_fingerprint(row)
            original = inventory[fingerprint]
            review = reviews[fingerprint]
            assert row['review'] == review and review['decision'] == 'retain'
            validate_review(row, review)
            assert row['source_group'] == review.get('related_family', original['source_group'])
            assert row['passage_source_group'] == original['source_group']
            assert row['source_id'] not in exclusion['source_ids']
            parent, pairs = originals[row['source_id']]
            assert (row['question'],row['answer']) in pairs
            assert hashlib.sha256(parent.encode()).hexdigest() == row['parent_sha256']
            start, stop = row['parent_offsets']
            assert parent[start:stop] == row['context']
            parts = fixed_passages(parent)
            offsets = [(p['start'],p['stop']) for p in parts]
            assert (start,stop) in offsets
            assert hashlib.sha256(json.dumps(offsets).encode()).hexdigest() == row['partition_sha256']
            assert len(row['context'].split()) <= 160
            a,b = review['answer_offsets']
            assert parent[start+a:start+b] == row['answer']
            group = row['source_group']
            assert group not in owners or owners[group] == split
            owners[group] = split
            groups.setdefault(group,[]).append(row)
        assert len(groups) == expected//2
        for pair in groups.values():
            assert len(pair) == 2 and pair[0]['context'] == pair[1]['context']
            assert pair[0]['answer'] != pair[1]['answer']
            assert pair[0]['question'] != pair[1]['question']
        counts[split] = dict(questions=len(rows),source_families=len(groups))
        all_rows.extend(rows)
    # Both full parent texts and fixed passages must remain separate from prior
    # exposure and other allocated families under the specified containment rule.
    contexts = [r['context'] for r in all_rows] + [originals[r['source_id']][0] for r in all_rows]
    ids = [r['source_group'] for r in all_rows]*2
    labels = group_sources(contexts + exclusion['contexts'], source_ids=ids+[None]*len(exclusion['contexts']))
    exposed = set(labels[len(contexts):]); components = {}
    for family,label in zip(ids,labels):
        assert label not in exposed, 'Selected family overlaps an old context'
        components.setdefault(label,set()).add(family)
    assert all(len(families)==1 for families in components.values()), 'Allocated families must be merged before splitting'
    geometry = {}
    for seed in (47,83):
        checkpoint = ROOT/f'review_artifacts/native_retained_decisions_counterfactual/seed{seed}/candidate/model.pt'
        model,tokenizer,_ = load_bundle_from_checkpoint(checkpoint,device='cpu',load_training_state=False)
        mem = model.native_question_memory
        max_tokens = max_source = max_question = 0
        for row in all_rows:
            prompt = f"Context:\n{row['context']}\nQuestion:\n{row['question']}"
            tokens = tokenizer.prepare_generation_hierarchy(prompt).token_ids
            total = len(tokens)+len(CompleteAnswerSpanPredictor.payload(tokenizer,row['answer']))
            assert total<=2048
            state=NativeQuestionState()
            for token in tokens:state,_=mem._observe(state,token,tokenizer)
            def observe(text):
                result=NativeQuestionState()
                for token in tokenizer.encode('<BOI>Context:\n'+text+'<EOI>',add_special_tokens=False):
                    result,_=mem._observe(result,token,tokenizer)
                return result
            question=observe(row['question']); gold=observe(row['answer'])
            assert state.source_seen<=512 and question.source_seen<=32
            words=[w for w,_ in state.source]; target=[w for w,_ in gold.source]
            assert 0<len(target)<=16
            starts=[i for i in range(len(words)-len(target)+1) if words[i:i+len(target)]==target]
            assert any(c['text']==row['answer'] for c in mem.span_readout.candidates(state,tokenizer,starts))
            max_tokens=max(max_tokens,total);max_source=max(max_source,state.source_seen);max_question=max(max_question,question.source_seen)
        geometry[str(seed)]=dict(questions=len(all_rows),exact_spans=len(all_rows),max_tokens=max_tokens,max_source_words=max_source,max_question_words=max_question,checkpoint_sha256=sha(checkpoint))
        print('GEOMETRY_PASS',seed,len(all_rows),flush=True)
        del model
    expected_hashes={47:'a0c3ecefe77c0a1346063c63990f442c264ce7e1f1f801df63f528073b3e0358',83:'e1093251728cf100e32b75f2eb7718bc8e43a20a836005614affd3ca0206c35f'}
    for seed,expected in expected_hashes.items():
        assert sha(ROOT/f'review_artifacts/native_owner_markers/seed{seed}/candidate/model.pt')==expected
    files=['native_qa_transfer_data.py','prepare_native_qa_passages.py','review_native_qa_passages.py','validate_native_qa_passages.py','data.py','native_qa_review.py','source_answer_span.py','native_question_memory.py','native_span_readout.py','train.py','config.py','model.py','native_semantic_alignment.py']
    report=dict(passed=True,task=manifest['task'],scope='Fixed answer-blind passages; this does not pass the original full-document data gate.',splits=counts,geometry=geometry,
        review=dict(decisions=len(bundle['decisions']),usable_source_families=len(usable_families),selected_source_families=len(owners),manual_family_links=bundle.get('family_links',[]),exposure_exclusions=bundle.get('exposed_groups',{})),
        model_training_started=False,test_model_evaluations=0,model_accepted=False,remaining_gate='Source-learning study has not run.',
        source_family_uncertainty=candidates['provenance_limitations'],reviewed_by='assistant; no independent human review claimed',
        baseline_hashes_unchanged=True,full_document_gate_passed=False,
        hashes={name:sha(ROOT/name) for name in files},artifact_hashes={name:sha(DATA/name) for name in ['candidates.json','reviews.json','frozen_manifest.json']},
        runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__))
    frozen_sources=DATA.parent/'repro_sources';frozen_sources.mkdir(exist_ok=True)
    for name in files:(frozen_sources/name).write_bytes((ROOT/name).read_bytes())
    (DATA.parent/'data_gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    try:
        validate()
    except Exception as error:
        # A failed rerun must not leave a stale positive gate on disk.
        (DATA.parent/'data_gate.json').write_text(json.dumps(dict(
            passed=False,model_accepted=False,error=f'{type(error).__name__}: {error}'
        ),indent=2),encoding='utf-8')
        raise
