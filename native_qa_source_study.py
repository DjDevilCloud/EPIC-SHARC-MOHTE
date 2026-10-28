"""Bounded source-only passage-QA study. Never reads locked test examples."""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from native_qa_review import qa_fingerprint, reviewed_indices
from native_question_memory import NativeQuestionState
from native_semantic_alignment import NativeSemanticAlignment
from native_training_objectives import preserve_decision_margin
from native_qa_transfer_study import generate, full_incremental_check, BASELINE_HASHES
from train import load_bundle_from_checkpoint, save_checkpoint

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'review_artifacts/native_qa_transfer_passages_v2/source_study'
DATA=OUT.parent/'data'
ARMS=('control','structure_only','question_attention_only')
CODE=('native_qa_source_study.py','native_question_memory.py','native_semantic_alignment.py',
      'native_span_readout.py','native_training_objectives.py','model.py','config.py','native_qa_transfer_study.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def read(path):return json.loads(Path(path).read_text())
def row_key(row):return row['context'],row['question'],row['answer'],row.get('style',0)
def prompt(row):
    style=row.get('style',0)
    if style==1:return f"CONTEXT:\n{row['context']}\nQUESTION:  {row['question']}"
    if style==2:return f"Context:\t{row['context']}\nQuestion:    {row['question']}"
    return f"Context:\n{row['context']}\nQuestion:\n{row['question']}"


def pools(seed):
    # Only these two new splits are loaded into fitting packets or evaluation.
    manifest=read(DATA/'frozen_manifest.json')
    real=read(ROOT/'review_artifacts/native_real_qa_comparison/manifest.json')
    audit=read(ROOT/'review_artifacts/native_retention_repair/annotation_audit.json')
    assert sha(ROOT/'review_artifacts/native_real_qa_comparison/manifest.json')==audit['source_manifest_sha256']
    legacy=[real['splits']['train'][i] for i in reviewed_indices(real['splits']['train'],audit['reviews']['train'])]
    previous=read(ROOT/f'review_artifacts/native_owner_markers/seed{seed}/manifest.json')
    native=read(ROOT/'review_artifacts/native_span_stable_gate_20261014/manifest.json')['native_span_training']
    old=previous['training']+previous['replay']; lookup={}
    for i,row in enumerate(old):lookup.setdefault(row_key(row),i)
    native_indices=[lookup[row_key(r)] for r in native]
    native=[old[i] for i in native_indices]
    # Keep the existing 288 controlled + 32 layout presentation pool. Four
    # layout entries intentionally repeat native QA inputs; they are not extra
    # independent examples and do not enter the new transfer denominators.
    controlled=[r for i,r in enumerate(old) if i not in set(native_indices)]
    assert (len(legacy),len(native),len(controlled))==(58,60,320)
    return dict(new=manifest['splits']['train'],development=manifest['splits']['development'],
                legacy=legacy,native=native,controlled=controlled,fresh=previous['groups']['fresh_values'])


def schedule(lengths,updates,seed=20261016):
    rng=random.Random(seed);orders={k:list(range(n)) for k,n in lengths.items()};cursors={k:0 for k in lengths}
    for order in orders.values():rng.shuffle(order)
    result=[]
    for _ in range(updates):
        step=[]
        for kind,count in [('new',8),('legacy',2),('native',2),('controlled',2)]:
            for _ in range(count):
                if cursors[kind]==len(orders[kind]):rng.shuffle(orders[kind]);cursors[kind]=0
                step.append([kind,orders[kind][cursors[kind]]]);cursors[kind]+=1
        result.append(step)
    return result


def freeze():
    gate=read(OUT.parent/'data_gate.json');assert gate['passed']
    for name,expected in gate['artifact_hashes'].items():assert sha(DATA/name)==expected
    for name,expected in gate['hashes'].items():
        # Two encoder/scoring files acquire a batch execution helper in this
        # study; new parity checks below validate it before optimization.
        if name not in ('native_question_memory.py','native_semantic_alignment.py'):assert sha(ROOT/name)==expected,name
    p=pools(47);other=pools(83)
    assert {k:[row_key(r) for r in rows] for k,rows in p.items()}=={k:[row_key(r) for r in rows] for k,rows in other.items()}
    lengths={k:len(p[k]) for k in ('new','legacy','native','controlled')}
    source_schedule=schedule(lengths,1600);fusion_schedule=schedule(lengths,800,20261017)
    metadata=dict(version=1,task='passage_grounded_extractive_qa',seeds=[47,83],arms=list(ARMS),
        updates=1600,fusion_updates=800,learning_rate=.001,weight_decay=.01,gradient_clip=1.,
        objective='All-source gold-start negative log mass. Replay adds source KL and capped gold margin only when teacher top1 is correct; incorrect teacher decisions receive gold supervision only.',
        optimizer='AdamW',threads=1,qa_readout_route='native',test_predictions_allowed=False,
        replay_overlap_note='Four of the 32 layout presentations duplicate native QA inputs; controlled pool is 288 tasks plus 32 layout presentations, not 320 independent documents.',
        pool_fingerprints={k:[qa_fingerprint(r) for r in rows] for k,rows in p.items()},
        file_hashes={str(path.relative_to(ROOT)):sha(path) for path in [DATA/'frozen_manifest.json',DATA/'reviews.json',
            ROOT/'review_artifacts/native_retention_repair/annotation_audit.json',ROOT/'review_artifacts/native_span_stable_gate_20261014/manifest.json']},
        checkpoint_hashes={str(seed):sha(ROOT/f'review_artifacts/native_retained_decisions_counterfactual/seed{seed}/candidate/model.pt') for seed in (47,83)},
        code_hashes={name:sha(ROOT/name) for name in CODE},runtime=dict(python=platform.python_version(),torch=str(torch.__version__),device='cpu'),
        schedules=dict(source=source_schedule,fusion=fusion_schedule))
    target=OUT/'frozen_experiment.json'
    if target.exists():assert read(target)==metadata,'Frozen experiment changed'
    else:dump(target,metadata)
    snapshots=OUT/'code';snapshots.mkdir(exist_ok=True)
    for name in CODE:
        target=snapshots/name
        if target.exists():assert sha(target)==metadata['code_hashes'][name]
        else:target.write_bytes((ROOT/name).read_bytes())
    print('EXPERIMENT_FROZEN',sha(OUT/'frozen_experiment.json'),flush=True)
    return metadata


def verify():
    metadata=read(OUT/'frozen_experiment.json')
    for name,expected in metadata['code_hashes'].items():assert sha(ROOT/name)==expected,('Code changed',name)
    for name,expected in metadata['file_hashes'].items():assert sha(ROOT/name)==expected,('Input changed',name)
    for seed,expected in metadata['checkpoint_hashes'].items():assert sha(ROOT/f'review_artifacts/native_retained_decisions_counterfactual/seed{seed}/candidate/model.pt')==expected
    for seed,expected in BASELINE_HASHES.items():assert sha(ROOT/f'review_artifacts/native_owner_markers/seed{seed}/candidate/model.pt')==expected
    return metadata


def load(seed,arm,checkpoint=None):
    torch.manual_seed(seed)
    path=checkpoint or ROOT/f'review_artifacts/native_retained_decisions_counterfactual/seed{seed}/candidate/model.pt'
    model,t,_=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
    model.eval();model.requires_grad_(False);model._prismal_tokenizer=t;model.cfg.qa_readout_route='native'
    if checkpoint is None and arm!='control':
        old=model.native_question_memory.semantic_alignment
        layer=NativeSemanticAlignment(model.cfg.d_model,prior_scaling=old.prior_scale is not None,
            structure=arm=='structure_only',question_attention=arm=='question_attention_only')
        missing,unexpected=layer.load_state_dict(old.state_dict(),strict=False)
        assert not unexpected
        expected={'structure.weight'} if arm=='structure_only' else {'question_residual.weight','question_attention.in_proj_weight','question_attention.in_proj_bias','question_attention.out_proj.weight','question_attention.out_proj.bias'}
        assert set(missing)==expected
        model.native_question_memory.semantic_alignment=layer
        model.cfg.native_question_memory_semantic_structure=arm=='structure_only'
        model.cfg.native_question_memory_question_attention=arm=='question_attention_only'
    return model,t


@torch.no_grad()
def packet(model,t,row):
    mem=model.native_question_memory
    ids=torch.tensor([t.prepare_generation_hierarchy(prompt(row)).token_ids])
    _,states=mem(ids,torch.zeros(1,ids.shape[1],model.cfg.d_model),model.construction_embedding,t,diagnostics=True)
    state=states[0];plan=state.cached_packet[3];assert plan is not None
    gold=NativeQuestionState()
    for token in t.encode('<BOI>Context:\n'+row['answer']+'<EOI>',add_special_tokens=False):gold,_=mem._observe(gold,token,t)
    words=[w for w,_ in state.source];target=[w for w,_ in gold.source]
    starts=[i for i in range(len(words)-len(target)+1) if words[i:i+len(target)]==target]
    assert starts and any(c['text']==row['answer'] for c in mem.span_readout.candidates(state,t,starts)),row['question']
    inputs=dict(plan['semantic_inputs'])
    inputs['structure']=NativeSemanticAlignment.observed_structure(state,t,inputs['units'])
    bank=mem.span_readout.candidates(state,t,list(range(len(words))));by_start={i:[] for i in range(len(words))}
    for c in bank:by_start[c['start']].append(c)
    # Matching lexical words alone can label the wrong occurrence positive
    # when punctuation/currency differs. Require an exact producible payload.
    starts=[i for i in starts if any(c['text']==row['answer'] for c in by_start[i])]
    assert starts, 'A gold source start must support the complete exact answer'
    positive=torch.zeros(len(words),dtype=torch.bool);positive[starts]=True
    result=dict(row=row,q=plan['legacy_question'],prior=plan['legacy_start_logits'],inputs=inputs,
        vectors=plan['base_word_vectors'],values=mem.value(plan['base_word_vectors']),markers=plan['marker_words'],words=words,
        by_start=by_start,positive=positive,teacher=plan['start_log_probabilities'].detach(),
        initial_candidates=plan['candidates'],
        teacher_correct=bool(positive[int(plan['start_log_probabilities'].argmax())]))
    question,recalled,prob,cached=decision(model,result)
    for a,b in zip(state.cached_packet[:3],(question,recalled,prob)):torch.testing.assert_close(a,b,rtol=2e-4,atol=3e-5)
    assert plan['candidates']==cached['candidates']
    torch.testing.assert_close(plan['log_probabilities'],cached['log_probabilities'],rtol=2e-4,atol=3e-5)
    return result


def decision(model,x):
    return model.native_question_memory.decision_packet(x['q'],x['vectors'],x['values'],x['prior'],x['markers'],x['inputs'],
        lambda selected:[c for i in selected for c in x['by_start'][i]],x['words'],diagnostics=True)


def batch_scores(model,xs):
    return model.native_question_memory.score_sources_many([x['q'] for x in xs],[x['prior'] for x in xs],[x['inputs'] for x in xs])


def objective(details,x,replay):
    logp=details['fused'].log_softmax(0)
    loss=-logp[x['positive']].logsumexp(0)
    if replay and x['teacher_correct']:
        loss=loss+F.kl_div(logp,x['teacher'].exp(),reduction='sum')+preserve_decision_margin(logp,x['teacher'],x['positive'])
    return loss


@torch.no_grad()
def evaluate(model,packets):
    summary={};ledger={}
    for kind,xs in packets.items():
        rows=[]
        for offset in range(0,len(xs),14):
            # Discrete evaluation always uses the ordinary inference scorer.
            # Tiny batch arithmetic differences may reorder nearly tied starts.
            for x in xs[offset:offset+14]:
                details=model.native_question_memory.score_sources(x['q'],x['prior'],x['inputs'])
                logp=details['fused'].log_softmax(0);order=logp.argsort(descending=True).tolist()
                probabilities=details['fused'].softmax(0)
                selected=probabilities.topk(min(16,len(probabilities))).indices
                ranks=[order.index(i)+1 for i in x['positive'].nonzero().flatten().tolist()]
                rows.append(dict(fingerprint=qa_fingerprint(x['row']),question=x['row']['question'],answer=x['row']['answer'],source_group=x['row'].get('source_group'),
                    representable=True,start_rank=min(ranks),start_top1=bool(x['positive'][int(probabilities.argmax())]),
                    start_top16=bool(x['positive'][selected].any()),selected_start=int(probabilities.argmax()),shortlist=selected.tolist(),
                    source_loss=float(-logp[x['positive']].logsumexp(0)),teacher_top1_correct=x['teacher_correct']))
        summary[kind]=dict(n=len(rows),top1=sum(r['start_top1'] for r in rows),top16=sum(r['start_top16'] for r in rows),source_loss=sum(r['source_loss'] for r in rows)/len(rows))
        ledger[kind]=rows
    return summary,ledger


def preflight(seed):
    metadata=verify();folder=OUT/f'seed{seed}';data=pools(seed)
    parent,t=load(seed,'control');mem=parent.native_question_memory
    originals={k:v.clone() for k,v in parent.state_dict().items()}
    cache_path=folder/'packets.pt'
    if cache_path.exists():
        saved=torch.load(cache_path,map_location='cpu',weights_only=True)
        assert saved['experiment_sha256']==sha(OUT/'frozen_experiment.json')
        packets=saved['packets']
    else:
        packets={}
        for kind,rows in data.items():
            packets[kind]=[]
            for i,row in enumerate(rows):
                packets[kind].append(packet(parent,t,row))
                if (i+1)%32==0 or i+1==len(rows):print('PREFLIGHT_PACKETS',seed,kind,i+1,len(rows),flush=True)
        folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(experiment_sha256=sha(OUT/'frozen_experiment.json'),packets=packets),cache_path)
    assert all(torch.equal(v,parent.state_dict()[k]) for k,v in originals.items())
    result=dict(passed=False,seed=seed,experiment_sha256=sha(OUT/'frozen_experiment.json'),arms={},oracle=[],prefix=[])
    # Existing complete oracle and prefix diagnostic coverage stays required.
    previous=read(ROOT/f'review_artifacts/native_qa_transfer_v1/seed{seed}/contract.json')
    assert previous['passed'] and previous['complete'] and len(previous['rows'])==103
    assert all(r['oracle_exact_payload'] and r['historical_raw_match'] for r in previous['rows'])
    old_code=read(ROOT/'review_artifacts/native_qa_transfer_v1/protocol.json')['code_sha256']
    for name in ('model.py','config.py','native_span_readout.py','native_qa_transfer_study.py'):assert sha(ROOT/name)==old_code[name]
    # New passage forms: normal generation, prefix execution and exact proposal
    # injection are checked before learning, with the learned copy gate intact.
    checks=[data['new'][i] for i in (0,1,32,33)]+data['development'][:2]
    for row in checks:
        result['prefix'].append(dict(fingerprint=qa_fingerprint(row),maximum_difference=full_incremental_check(parent,t,row)))
        normal=generate(parent,t,row);oracle=generate(parent,t,row,oracle=True)
        from source_answer_span import CompleteAnswerSpanPredictor
        assert tuple(oracle['tokens'])==CompleteAnswerSpanPredictor.payload(t,row['answer'])
        result['oracle'].append(dict(fingerprint=qa_fingerprint(row),exact_payload=True,termination=oracle['ended'],normal=normal))
    for arm in ARMS:
        model,at=load(seed,arm)
        mem=model.native_question_memory
        initial=folder/arm/'initial'
        if not initial.exists():save_checkpoint(model,initial,tokenizer=at,metrics=dict(experimental=True,accepted=False,updates=0,arm=arm))
        reloaded,rt=load(seed,arm,initial/'model.pt')
        assert reloaded.cfg.qa_readout_route=='native' and reloaded.bounded_identity_readout is not None
        assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
        maximum=0.;batch_order_differences=0
        all_x=[x for xs in packets.values() for x in xs]
        with torch.no_grad():
            for offset in range(0,len(all_x),14):
                xs=all_x[offset:offset+14];batched=batch_scores(model,xs)
                for x,b in zip(xs,batched):
                    single=mem.score_sources(x['q'],x['prior'],x['inputs'])['fused']
                    torch.testing.assert_close(single,b['fused'],rtol=2e-4,atol=3e-5)
                    torch.testing.assert_close(single.log_softmax(0),x['teacher'],rtol=2e-4,atol=3e-5)
                    batch_order_differences+=int(single.topk(min(16,len(single))).indices.tolist()!=b['fused'].topk(min(16,len(single))).indices.tolist())
                    # Candidate identities are checked on the actual shared
                    # inference/cached decision path, not a batch-loss tensor.
                    question,recalled,prob,cached_plan=decision(model,x)
                    assert x['initial_candidates']==cached_plan['candidates']
                    maximum=max(maximum,float((single-b['fused']).abs().max()))
            for row,reference in zip(checks,result['oracle']):
                live_packet=packet(model,at,row)
                key=qa_fingerprint(row)
                frozen_packet=next(x for x in all_x if qa_fingerprint(x['row'])==key)
                assert decision(model,live_packet)[3]['candidates']==decision(model,frozen_packet)[3]['candidates']
                assert generate(reloaded,rt,row)==reference['normal']
        result['arms'][arm]=dict(cached_live_and_batched_source_agree=True,identical_live_cached_candidates=True,reload_and_greedy_agree=True,maximum_source_difference=maximum,batch_loss_top16_order_differences=batch_order_differences,batch_used_for_ranking=False)
    result['passed']=True
    torch.save(dict(experiment_sha256=sha(OUT/'frozen_experiment.json'),packets=packets),folder/'packets.pt')
    dump(folder/'preflight.json',result)
    print('PREFLIGHT_PASS',seed,result['arms'],flush=True)


def fit(seed,arm):
    metadata=verify();folder=OUT/f'seed{seed}';pre=read(folder/'preflight.json');assert pre['passed']
    assert pre['experiment_sha256']==sha(OUT/'frozen_experiment.json')
    cached=torch.load(folder/'packets.pt',map_location='cpu',weights_only=True)
    assert cached['experiment_sha256']==sha(OUT/'frozen_experiment.json')
    packets=cached['packets'];dest=folder/arm;resume=dest/'resume.pt'
    model,t=load(seed,arm,dest/'initial/model.pt')
    if arm=='normalized_fusion':raise ValueError('Use fusion phase')
    trainable=model.native_question_memory.semantic_alignment
    model.requires_grad_(False);trainable.requires_grad_(True)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(trainable.parameters(),lr=metadata['learning_rate'],weight_decay=metadata['weight_decay'])
    history=[];start=0
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        assert saved['experiment_sha256']==sha(OUT/'frozen_experiment.json')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);start=saved['step'];history=saved['history']
    else:
        scores,ledger=evaluate(model,packets);history.append(dict(step=0,scores=scores));dump(dest/'initial_sources.json',dict(summary=scores,ledger=ledger))
    started=time.monotonic()
    for step in range(start+1,1601):
        chosen=metadata['schedules']['source'][step-1];xs=[packets[k][i] for k,i in chosen]
        optimizer.zero_grad(set_to_none=True)
        details=batch_scores(model,xs)
        loss=torch.stack([objective(d,x,k!='new') for d,x,(k,_) in zip(details,xs,chosen)]).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite source objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==start+1 or step%50==0:print('SOURCE_STEP',seed,arm,step,'loss',round(float(loss.detach()),5),'seconds',round(time.monotonic()-started,1),flush=True)
        if step%200==0 or step==1600:
            scores,ledger=evaluate(model,packets);history.append(dict(step=step,scores=scores));dump(dest/'history.json',history)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),step=step,history=history,experiment_sha256=sha(OUT/'frozen_experiment.json')),resume)
            print('SOURCE_EVALUATION',seed,arm,step,scores,flush=True)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(initial[k],v)]
    assert all(k.startswith('native_question_memory.semantic_alignment.') for k in changed),changed
    save_checkpoint(model,dest/'final',tokenizer=t,metrics=dict(experimental=True,accepted=False,updates=1600,arm=arm))
    check,ct=load(seed,arm,dest/'final/model.pt')
    scores,ledger=evaluate(check,packets)
    expected=evaluate(model,packets)[0];assert scores==expected
    dump(dest/'source_results.json',dict(seed=seed,arm=arm,updates=1600,summary=scores,ledger=ledger,changed_keys=changed,protected_parameters_bitwise_equal=True,checkpoint_sha256=sha(dest/'final/model.pt'),locked_test_predictions=0))
    verify()
    print('SOURCE_ARM_COMPLETE',seed,arm,scores['development'],flush=True)


def select():
    verify();rows={}
    for arm in ARMS:
        rows[arm]={str(seed):read(OUT/f'seed{seed}/{arm}/source_results.json') for seed in (47,83)}
        assert all(r['updates']==1600 and r['protected_parameters_bitwise_equal'] for r in rows[arm].values())
    def key(arm):
        dev=[r['summary']['development'] for r in rows[arm].values()]
        return min(d['top1'] for d in dev),min(d['top16'] for d in dev),-max(d['source_loss'] for d in dev)
    chosen=max(ARMS,key=key)
    result=dict(selected_encoder=chosen,selection_key=key(chosen),criteria='Worst-seed development top1, then worst-seed top16, then worst-seed source loss',
        source_gate_not_yet_final='One prescribed fusion comparison remains',arms={a:{s:r['summary'] for s,r in values.items()} for a,values in rows.items()},locked_test_predictions=0)
    dump(OUT/'encoder_selection.json',result)
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arm','seed','updates','training_top1','development_top1','development_top16','development_source_loss','fresh_top1'])
        for arm,values in rows.items():
            for seed,r in values.items():
                s=r['summary'];writer.writerow([arm,seed,r['updates'],s['new']['top1'],s['development']['top1'],s['development']['top16'],s['development']['source_loss'],s['fresh']['top1']])
    print('ENCODER_SELECTION',chosen,result['selection_key'],flush=True)


@torch.no_grad()
def failure_ledger(model,t,packets,raw=False):
    results={}
    for kind in ('new','development','fresh'):
        rows=[]
        for x in packets[kind]:
            _,_,_,plan=decision(model,x)
            scores=plan['start_log_probabilities'];order=scores.argsort(descending=True).tolist()
            starts=x['positive'].nonzero().flatten().tolist()
            gold=[c for i in starts for c in x['by_start'][i]]
            extent=model.native_question_memory.span_readout.score_candidates(gold,
                model.native_question_memory.decision_packet(x['q'],x['vectors'],x['values'],x['prior'],x['markers'],x['inputs'])[0],
                plan['word_vectors'],scores,x['markers'])
            proposal=plan['candidates'][int(plan['probabilities'].argmax())]['text']
            answer=generate(model,t,x['row']) if raw else None
            rows.append(dict(fingerprint=qa_fingerprint(x['row']),question=x['row']['question'],answer=x['row']['answer'],
                source_group=x['row'].get('source_group'),representable=True,start_rank=min(order.index(i)+1 for i in starts),
                shortlist_inclusion=any(i in plan['selected_starts'] for i in starts),
                gold_start_extent=gold[int(extent.argmax())]['text']==x['row']['answer'],
                complete_proposal=proposal,complete_proposal_correct=proposal==x['row']['answer'],raw_answer=answer,
                raw_status='evaluated_normal_native_route' if raw else 'not_evaluated_source_only_stage',
                source_scores={k:v.tolist() for k,v in plan['diagnostics'].items() if k.endswith('_logits') and v is not None},
                shortlist=plan['selected_starts'],endpoint_probabilities=plan['probabilities'].tolist()))
        results[kind]=dict(denominator=len(rows),rows=rows)
    return results


def fusion(seed):
    metadata=verify();selected=read(OUT/'encoder_selection.json')['selected_encoder'];folder=OUT/f'seed{seed}';dest=folder/'normalized_fusion'
    packets=torch.load(folder/'packets.pt',map_location='cpu',weights_only=True)['packets']
    model,t=load(seed,selected,folder/selected/'final/model.pt');old=model.native_question_memory.semantic_alignment
    layer=NativeSemanticAlignment(model.cfg.d_model,prior_scaling=old.prior_scale is not None,
        structure=old.structure is not None,question_attention=old.question_attention is not None,normalized_fusion=True)
    missing,unexpected=layer.load_state_dict(old.state_dict(),strict=False)
    assert not unexpected and all(k=='temperatures' or k.startswith('fusion_gate.') for k in missing)
    model.native_question_memory.semantic_alignment=layer;model.cfg.native_question_memory_normalized_fusion=True
    model.requires_grad_(False);layer.fusion_gate.requires_grad_(True);layer.temperatures.requires_grad_(True)
    before={k:v.clone() for k,v in model.state_dict().items()}
    parameters=[p for p in layer.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=metadata['learning_rate'],weight_decay=metadata['weight_decay'])
    resume=dest/'resume.pt';start=0;history=[]
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True);assert saved['experiment_sha256']==sha(OUT/'frozen_experiment.json')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);start=saved['step'];history=saved['history']
    for step in range(start+1,801):
        chosen=metadata['schedules']['fusion'][step-1];xs=[packets[k][i] for k,i in chosen]
        optimizer.zero_grad(set_to_none=True)
        # Encoder is frozen here; its contextual states are not optimized.
        details=batch_scores(model,xs)
        loss=torch.stack([objective(d,x,k!='new') for d,x,(k,_) in zip(details,xs,chosen)]).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        if step==start+1 or step%50==0:print('FUSION_STEP',seed,step,float(loss.detach()),flush=True)
        if step%200==0:
            scores,ledger=evaluate(model,packets);history.append(dict(step=step,scores=scores));dump(dest/'history.json',history)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),step=step,history=history,experiment_sha256=sha(OUT/'frozen_experiment.json')),resume)
            print('FUSION_EVALUATION',seed,step,scores,flush=True)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(before[k],v)]
    assert all(k.startswith('native_question_memory.semantic_alignment.fusion_gate.') or k=='native_question_memory.semantic_alignment.temperatures' for k in changed)
    save_checkpoint(model,dest/'final',tokenizer=t,metrics=dict(experimental=True,accepted=False,updates=800,selected_encoder=selected))
    reloaded,rt=load(seed,selected,dest/'final/model.pt');scores,ledger=evaluate(reloaded,packets)
    assert scores==evaluate(model,packets)[0]
    dump(dest/'source_results.json',dict(seed=seed,arm='normalized_fusion',updates=800,selected_encoder=selected,summary=scores,ledger=ledger,changed_keys=changed,protected_parameters_bitwise_equal=True,checkpoint_sha256=sha(dest/'final/model.pt'),locked_test_predictions=0))
    print('FUSION_COMPLETE',seed,scores['development'],flush=True)


def decide():
    verify();selected=read(OUT/'encoder_selection.json')['selected_encoder'];options={}
    for arm in (selected,'normalized_fusion'):
        seeds={}
        for seed in (47,83):
            result=read(OUT/f'seed{seed}/{arm}/source_results.json');s=result['summary']
            checks=dict(training_top1=s['new']['top1']/s['new']['n']>=.95,development_top1=s['development']['top1']>=32,
                development_top16=s['development']['top16']>=58,fresh_starts=s['fresh']['top1']==16,protected_parameters=result['protected_parameters_bitwise_equal'])
            seeds[str(seed)]=dict(passed=all(checks.values()),checks=checks,summary=s)
        options[arm]=dict(passed=all(r['passed'] for r in seeds.values()),seeds=seeds)
    passed=[a for a,r in options.items() if r['passed']]
    chosen=max(passed,key=lambda a:(min(r['summary']['development']['top1'] for r in options[a]['seeds'].values()),min(r['summary']['development']['top16'] for r in options[a]['seeds'].values()))) if passed else None
    result=dict(status='SOURCE_GATE_PASSED' if chosen else 'STOPPED_SOURCE_GATE_FAILED',selected_encoder=selected,source_candidate=chosen,
        options=options,source_gate_passed=bool(chosen),model_accepted=False,accepted_baseline_changed=False,
        source_updates_per_arm=1600,fusion_updates=800,locked_test_model_evaluations=0,extent_updates=0,joint_updates=0,
        failed_hypothesis=None if chosen else 'The prescribed encoder and fusion changes did not meet the passage source-transfer gate in both seeds. No extension or additional branch is authorized by this protocol.',
        manifest_sha256=sha(DATA/'frozen_manifest.json'),experiment_sha256=sha(OUT/'frozen_experiment.json'))
    dump(OUT/'acceptance.json',result)
    # Preserve the data-only decision history; report the new stage separately.
    dump(OUT.parent/'source_acceptance.json',result)
    with (OUT/'comparison.csv').open('a',newline='') as f:
        writer=csv.writer(f)
        for seed in (47,83):
            r=read(OUT/f'seed{seed}/normalized_fusion/source_results.json');s=r['summary']
            writer.writerow(['normalized_fusion',seed,800,s['new']['top1'],s['development']['top1'],s['development']['top16'],s['development']['source_loss'],s['fresh']['top1']])
    print('SOURCE_DECISION',json.dumps(result),flush=True)


def ledgers(seed):
    verify();packets=torch.load(OUT/f'seed{seed}/packets.pt',map_location='cpu',weights_only=True)['packets']
    for arm in ARMS+('normalized_fusion',):
        selected=read(OUT/'encoder_selection.json')['selected_encoder'] if arm=='normalized_fusion' else arm
        folder=OUT/f'seed{seed}/{arm}';model,t=load(seed,selected,folder/'final/model.pt')
        # Detailed raw QA is measured for the selected encoder and its fusion
        # comparison. Other arm ledgers explicitly retain unmeasured raw fields.
        result=failure_ledger(model,t,packets,raw=arm in (read(OUT/'encoder_selection.json')['selected_encoder'],'normalized_fusion'))
        dump(folder/'failure_ledger.json',result)
        print('LEDGER_COMPLETE',seed,arm,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('freeze','preflight','fit','select','fusion','decide','ledgers'))
    parser.add_argument('--seed',type=int,choices=(47,83));parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='freeze':freeze()
    elif args.phase=='preflight':preflight(args.seed)
    elif args.phase=='fit':fit(args.seed,args.arm)
    elif args.phase=='select':select()
    elif args.phase=='fusion':fusion(args.seed)
    elif args.phase=='ledgers':ledgers(args.seed)
    else:decide()


if __name__=='__main__':main()
