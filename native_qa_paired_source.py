"""Separate, bounded paired-question source-objective pilot; no locked-test inference.

The preceding source study remains closed. This runner changes supervision,
not the model, tokenizer, decoding, memory limits, or accepted checkpoints.
"""
import argparse
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import native_qa_source_study as old
from train import save_checkpoint

OUT = old.OUT.parent / 'paired_source_pilot'
UPDATES = 400


def pairs(xs):
    grouped = defaultdict(list)
    for i, x in enumerate(xs):
        grouped[x['row']['context']].append(i)
    result = {}
    for indices in grouped.values():
        if len(indices) != 2:
            raise ValueError('Every passage must have exactly two contrasting questions')
        a, b = indices
        if xs[a]['words'] != xs[b]['words'] or torch.any(xs[a]['positive'] & xs[b]['positive']):
            raise ValueError('Paired supervision requires identical source positions and disjoint gold starts')
        result[a] = b
        result[b] = a
    return result


def assignment_margin(left, right, positive_left, positive_right):
    """Correct versus swapped assignment, cancelling question-independent salience.

    Labels are confined to training/evaluation. Normal inference uses the
    unchanged source scorer and never receives its paired question or labels.
    """
    if left.shape != right.shape or left.shape != positive_left.shape or left.shape != positive_right.shape:
        raise ValueError('Aligned source positions required')
    if positive_left.dtype != torch.bool or positive_right.dtype != torch.bool:
        raise ValueError('Boolean labels required')
    if not positive_left.any() or not positive_right.any() or (positive_left & positive_right).any():
        raise ValueError('Nonempty disjoint answers required')
    return (left[positive_left].logsumexp(0) - left[positive_right].logsumexp(0)
            + right[positive_right].logsumexp(0) - right[positive_left].logsumexp(0))


@torch.no_grad()
def contrast(model, xs):
    mates = pairs(xs)
    logits = [model.native_question_memory.score_sources(x['q'], x['prior'], x['inputs'])['fused'] for x in xs]
    rows = []
    for a, b in mates.items():
        if a > b:
            continue
        pa, pb = xs[a]['positive'], xs[b]['positive']
        la, lb = logits[a], logits[b]
        margin = float(assignment_margin(la, lb, pa, pb))
        rows.append(dict(indices=[a, b], source_group=xs[a]['row'].get('source_group'),
                         assignment_margin=margin, assignment_correct=margin > 0,
                         assignment_tied=margin == 0,
                         question_changes_top1=int(la.argmax()) != int(lb.argmax()),
                         both_top1_correct=bool(pa[la.argmax()] and pb[lb.argmax()]),
                         forced_choice_correct=int(la[pa].logsumexp(0) > la[pb].logsumexp(0))
                         + int(lb[pb].logsumexp(0) > lb[pa].logsumexp(0))))
    return dict(pairs=len(rows), assignment_correct=sum(r['assignment_correct'] for r in rows),
                ties=sum(r['assignment_tied'] for r in rows),
                question_changes_top1=sum(r['question_changes_top1'] for r in rows),
                both_top1_correct=sum(r['both_top1_correct'] for r in rows),
                forced_choice_correct=sum(r['forced_choice_correct'] for r in rows),
                forced_choice_denominator=2 * len(rows), rows=rows)


def packets(seed):
    path = old.OUT / f'seed{seed}/packets.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['experiment_sha256'] == old.sha(old.OUT / 'frozen_experiment.json')
    return saved['packets']


def diagnose():
    old.verify()
    result = {}
    for seed in (47, 83):
        cache = packets(seed)
        for stage in ('initial', 'final'):
            path = old.OUT / f'seed{seed}/structure_only/{stage}/model.pt'
            model, _ = old.load(seed, 'structure_only', path)
            result[f'{seed}_{stage}'] = {k: contrast(model, cache[k]) for k in ('new', 'development')}
            print('CONTRAST', seed, stage, {k: {a:b for a,b in v.items() if a != 'rows'} for k,v in result[f'{seed}_{stage}'].items()}, flush=True)
    old.dump(OUT / 'diagnosis.json', result)


def freeze():
    metadata = old.verify()
    protocol = dict(version=1, status='separate_predeclared_pilot', updates=UPDATES, seeds=[47,83],
        arms=['matched_control','paired_assignment'], pair_loss_weight=1.,
        hypothesis='Explicit correct-versus-swapped question/answer assignment reduces question-independent source salience and transfers to held-out source groups.',
        source_study_sha256=old.sha(old.OUT/'frozen_experiment.json'),
        source_schedule=metadata['schedules']['source'][:UPDATES],
        sampling='Same 8 new + 2 legacy + 2 native + 2 controlled slots; each new slot averages its own and its paired question CE. Both arms see identical questions and forwards.',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.01,clip=1.),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),device='cpu',threads=1),
        selection='Final update only; report both seeds, no extension or locked-test evaluation.',
        pilot_signal='Both seeds improve development top1 by at least 4/64 and assignment accuracy by at least 4/32 over matched control; all 16 fresh starts must remain correct.',
        acceptance='Pilot signal alone cannot accept a model. Existing full source and retention gates still apply before further stages.',
        code_hashes={name:old.sha(old.ROOT/name) for name in old.CODE + ('native_qa_paired_source.py',)},
        packet_hashes={str(s):old.sha(old.OUT/f'seed{s}/packets.pt') for s in (47,83)},
        initial_hashes={str(s):old.sha(old.OUT/f'seed{s}/structure_only/initial/model.pt') for s in (47,83)},
        locked_test_predictions=0)
    path=OUT/'protocol.json'
    if path.exists():
        assert old.read(path)==protocol, 'Pilot protocol is frozen'
    else:
        old.dump(path,protocol)
        (OUT/'code').mkdir(exist_ok=True)
        for name in protocol['code_hashes']:
            (OUT/'code'/name).write_bytes((old.ROOT/name).read_bytes())
    return protocol


def train(seed, arm):
    protocol=freeze()
    assert arm in protocol['arms']
    dest=OUT/f'seed{seed}'/arm
    if (dest/'result.json').exists():
        print('ALREADY_COMPLETE',seed,arm,flush=True)
        return
    assert not (dest/'final/model.pt').exists(), 'Do not overwrite a completed candidate'
    cache=packets(seed);mates=pairs(cache['new'])
    model,t=old.load(seed,'structure_only',old.OUT/f'seed{seed}/structure_only/initial/model.pt')
    initial={k:v.clone() for k,v in model.state_dict().items()}
    model.requires_grad_(False);encoder=model.native_question_memory.semantic_alignment;encoder.requires_grad_(True)
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=.001,weight_decay=.01)
    torch.manual_seed(seed)
    start=time.monotonic();history=[]
    for step,slots in enumerate(protocol['source_schedule'],1):
        xs=[cache[k][i] for k,i in slots]
        partners=[cache['new'][mates[i]] for k,i in slots if k=='new']
        optimizer.zero_grad(set_to_none=True)
        ds=old.batch_scores(model,xs+partners)
        terms=[];paired=[];j=0
        for d,x,(kind,index) in zip(ds[:len(xs)],xs,slots):
            term=old.objective(d,x,kind!='new')
            if kind=='new':
                other=partners[j];od=ds[len(xs)+j];j+=1
                term=(term+old.objective(od,other,False))/2
                paired.append(F.softplus(-assignment_margin(d['fused'],od['fused'],x['positive'],other['positive'])))
            terms.append(term)
        loss=torch.stack(terms).mean()
        if arm=='paired_assignment':
            loss=loss+torch.stack(paired).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%100==0:
            print('PAIRED_STEP',seed,arm,step,'loss',round(float(loss.detach()),5),'seconds',round(time.monotonic()-start,1),flush=True)
            history.append(dict(step=step,loss=float(loss.detach())))
    changed=[k for k,v in model.state_dict().items() if not torch.equal(initial[k],v)]
    assert changed and all(k.startswith('native_question_memory.semantic_alignment.') for k in changed)
    summary,ledger=old.evaluate(model,cache)
    contrasts={k:contrast(model,cache[k]) for k in ('new','development')}
    save_checkpoint(model,dest/'final',tokenizer=t,metrics=dict(experimental=True,accepted=False,updates=UPDATES,arm=arm))
    reloaded,rt=old.load(seed,'structure_only',dest/'final/model.pt')
    for k,v in model.state_dict().items():
        assert torch.equal(v,reloaded.state_dict()[k]), k
    again,again_ledger=old.evaluate(reloaded,cache)
    for kind,rows in ledger.items():
        for a,b in zip(rows,again_ledger[kind]):
            assert a['shortlist']==b['shortlist'] and a['selected_start']==b['selected_start']
        assert summary[kind]['top1']==again[kind]['top1'] and summary[kind]['top16']==again[kind]['top16']
        assert math.isclose(summary[kind]['source_loss'],again[kind]['source_loss'],rel_tol=2e-4,abs_tol=3e-5)
    # Recompute features from actual input, compare all development source scores
    # and candidate identities with cached fitting, and check ordinary generation
    # reload on two contrasting pairs. These are parity checks, not answer success.
    maximum=0.
    for x in cache['development']:
        live=old.packet(reloaded,rt,x['row'])
        a=old.decision(reloaded,x)[3];b=old.decision(reloaded,live)[3]
        torch.testing.assert_close(a['start_log_probabilities'],b['start_log_probabilities'],rtol=2e-4,atol=3e-5)
        assert a['candidates']==b['candidates']
        maximum=max(maximum,float((a['start_log_probabilities']-b['start_log_probabilities']).abs().max()))
    greedy=[]
    for x in cache['development'][:4]:
        answer=old.generate(model,t,x['row']);assert answer==old.generate(reloaded,rt,x['row'])
        greedy.append(dict(question=x['row']['question'],answer=answer,gold=x['row']['answer']))
    result=dict(seed=seed,arm=arm,updates=UPDATES,summary=again,ledger=again_ledger,contrasts=contrasts,
                protected_parameters_bitwise_equal=True,changed_keys=changed,history=history,
                checkpoint_sha256=old.sha(dest/'final/model.pt'),protocol_sha256=old.sha(OUT/'protocol.json'),
                live_cached_all_development_candidates_identical=True,maximum_live_cached_logit_difference=maximum,
                reload_state_bitwise_equal=True,greedy_reload_checks=greedy,locked_test_predictions=0,accepted=False)
    old.dump(dest/'result.json',result)
    old.verify()
    print('PAIRED_COMPLETE',seed,arm,again['development'],flush=True)


def report():
    freeze();rows=[];signals=[]
    for seed in (47,83):
        results={a:old.read(OUT/f'seed{seed}/{a}/result.json') for a in ('matched_control','paired_assignment')}
        for arm,r in results.items():
            assert old.sha(OUT/f'seed{seed}/{arm}/final/model.pt')==r['checkpoint_sha256']
            rows.append(dict(seed=seed,arm=arm,summary=r['summary'],contrasts={k:{a:b for a,b in v.items() if a!='rows'} for k,v in r['contrasts'].items()}))
        c,p=results['matched_control'],results['paired_assignment']
        delta=p['summary']['development']['top1']-c['summary']['development']['top1']
        contrast_delta=p['contrasts']['development']['assignment_correct']-c['contrasts']['development']['assignment_correct']
        signals.append(dict(seed=seed,top1_delta=delta,assignment_delta=contrast_delta,
                            pilot_signal=delta>=4 and contrast_delta>=4 and p['summary']['fresh']['top1']==16))
    passed=all(s['pilot_signal'] for s in signals)
    result=dict(status='PILOT_SIGNAL_PASSED' if passed else 'STOPPED_PAIRED_HYPOTHESIS_NOT_SUPPORTED',
                rows=rows,signals=signals,model_accepted=False,baseline_changed=False,locked_test_predictions=0,
                note='Source-only objective pilot; extent, emission and full raw retention were not trained or certified.')
    old.dump(OUT/'acceptance.json',result)
    lines=['# Paired-question source-objective pilot','',result['status'],'',
           '| Seed | Arm | Train starts /128 | Dev starts /64 | Dev top16 /64 | Dev assignment /32 | Fresh starts /16 |',
           '|---|---|---|---|---|---|---|']
    for r in rows:
        s=r['summary'];d=r['contrasts']['development']
        lines.append(f"| {r['seed']} | {r['arm']} | {s['new']['top1']} | {s['development']['top1']} | {s['development']['top16']} | {d['assignment_correct']} | {s['fresh']['top1']} |")
    lines+=['','Assignment accuracy measures correct versus swapped assignments between two annotated answers; it is an oracle-restricted diagnostic, not inference success.',
            'Final-update comparison only. Both arms used identical initial checkpoints, example schedules and paired forwards. Only the paired-assignment loss differed.',
            'The accepted baseline and prior failed study are unchanged. Locked test predictions: zero.',
            'Training/development data remain the previously reviewed passage benchmark, with the same provenance limitations. No full-document QA claim.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(result['status'],signals,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['diagnose','freeze','train','report'])
    parser.add_argument('--seed',type=int,choices=[47,83]);parser.add_argument('--arm',choices=['matched_control','paired_assignment'])
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='diagnose':diagnose()
    elif args.phase=='freeze':freeze()
    elif args.phase=='train':
        if args.seed is None or args.arm is None:parser.error('train requires --seed and --arm')
        train(args.seed,args.arm)
    else:report()
