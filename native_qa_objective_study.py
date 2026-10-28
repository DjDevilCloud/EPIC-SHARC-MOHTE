"""Gated V4 objective-first study. Test split is never loaded for predictions."""
import argparse
import json
import math
import random
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

import native_qa_source_study as previous
from native_qa_multianswer import OUT as DATA,ROOT,read,sha,write_new,verify_overlay,verify_history,qa_fingerprint,score_text
from native_qa_functional_replay import (source_supervision,gold_mass_loss,protected_decision,
    protected_gate,protected_tokens,cached_emission)
from native_semantic_alignment import NativeSemanticAlignment
from native_question_memory import NativeQuestionState
from native_qa_transfer_study import generation_fields
from source_answer_span import CompleteAnswerSpanPredictor
from data import TextWindowDataset,build_collate_fn
from train import load_bundle_from_checkpoint,save_checkpoint,generate_text

OUT=DATA/'study'
ARMS=('control','branch_consistent')
CODE=('native_qa_objective_study.py','native_qa_functional_replay.py','native_qa_multianswer.py',
      'qa_generation.py','native_question_memory.py','native_semantic_alignment.py','native_span_readout.py',
      'config.py','model.py','train.py','cli.py','data.py','source_answer_span.py','native_training_objectives.py')
CODE=tuple(sorted(set(CODE)|{p.name for p in ROOT.glob('*.py')}))


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def load(seed,path=None):
    torch.manual_seed(seed)
    path=path or ROOT/f'review_artifacts/native_retained_decisions_counterfactual/seed{seed}/candidate/model.pt'
    model,t,_=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
    model.eval();model.requires_grad_(False);model._prismal_tokenizer=t
    model.cfg.qa_readout_route='native';model.cfg.qa_generation_profile='native_qa_eval_v1'
    return model,t


def generate(model,t,row):
    return generate_text(model,t,previous.prompt(row),torch.device('cpu'),return_generation_record=True)


def decision(model,x):
    return previous.decision(model,x)


@torch.no_grad()
def observed_packet(model,t,row):
    ids=torch.tensor([t.prepare_generation_hierarchy(previous.prompt(row)).token_ids])
    _,states=model.native_question_memory(ids,torch.zeros(1,ids.shape[1],model.cfg.d_model),
        model.construction_embedding,t,diagnostics=True)
    return states[0].cached_packet


def endpoint_groups(model,x,question,plan):
    memory=model.native_question_memory;groups=[]
    for start in x['positive'].nonzero().flatten().tolist():
        candidates=x['by_start'][start]
        energies=memory.span_readout.score_candidates(candidates,question,plan['word_vectors'],
                    torch.zeros_like(x['prior']),x['markers'])
        positive=torch.tensor([c['text'] in x.get('accepted_answers',(x['row']['answer'],)) for c in candidates],dtype=torch.bool)
        groups.append((energies,positive))
    return groups


def gate_logit(model,question,plan):
    return model.native_question_memory.span_readout.copy_gate(torch.cat((plan['copy_context'],question))).squeeze()


def replay(model,x):
    question,recalled,_,plan=decision(model,x)
    teacher=x['functional_teacher']
    # The caller already includes gold source supervision in both arms.
    current=plan['start_log_probabilities']
    source=protected_decision(current,teacher['source'],x['positive'])-gold_mass_loss(current,x['positive'])
    groups=endpoint_groups(model,x,question,plan)
    extent=torch.stack([protected_decision(logits,old,positive) for (logits,positive),old in zip(groups,teacher['extent'])]).mean()
    gate=protected_gate(gate_logit(model,question,plan),teacher['gate'],True)
    logits=cached_emission(model,x['emission'],question,recalled,plan)
    tokens=protected_tokens(logits,teacher['tokens'],x['emission']['labels'],x['emission']['mask'])
    return source+extent+gate+tokens,dict(source=source,extent=extent,gate=gate,tokens=tokens)


@torch.no_grad()
def evaluate(model,packets):
    # Canonical measurement must match a reloaded inference model, including
    # parameter flags that can select different recurrent kernels near ties.
    parameters=list(model.parameters());flags=[p.requires_grad for p in parameters]
    try:
        model.requires_grad_(False)
        summary,ledger=previous.evaluate(model,packets)
    finally:
        for parameter,flag in zip(parameters,flags):parameter.requires_grad_(flag)
    for kind,xs in packets.items():
        old_correct=0
        for x,row in zip(xs,ledger[kind]):
            old_correct+=bool(x['original_positive'][row['selected_start']])
        summary[kind]['historical_answer_string_top1']=old_correct
        summary[kind]['label_policy']='reviewed_supported_copyable_answers' if kind in ('new','development') else 'existing_replay_labels'
    return summary,ledger


def cache_inputs(seed):
    cache_path=previous.OUT/f'seed{seed}/packets.pt'
    original=torch.load(cache_path,map_location='cpu',weights_only=True)
    assert original['experiment_sha256']==sha(previous.OUT/'frozen_experiment.json')
    packets=original['packets'];overlay=verify_overlay()
    binding=read(DATA/'native_bindings.json')['seeds'][str(seed)]['bindings']
    for kind,xs in packets.items():
        for x in xs:
            x['original_positive']=x['positive'].clone()
            if kind in ('new','development'):
                key=qa_fingerprint(x['row']);entry=overlay['decisions'][key]
                x['accepted_answers']=tuple(a['text'] for a in entry['answers'])
                x['positive']=torch.zeros_like(x['positive'])
                for occurrence in binding[key]['occurrences']:x['positive'][occurrence['start']]=True
                x['unavailable_supported_occurrences']=binding[key]['unavailable_supported_occurrences']
            else:x['accepted_answers']=(x['row']['answer'],)
            x['teacher_correct']=bool(x['positive'][x['teacher'].argmax()])
    old_path=ROOT/f'review_artifacts/native_decision_packets_seed{seed}.pt'
    old=torch.load(old_path,map_location='cpu',weights_only=True)
    assert old['metadata']['source_sha256']==previous.BASELINE_HASHES[seed]
    lookup={previous.row_key(x['r']):x for pool in old['packets'] for x in pool}
    for kind in ('legacy','native','controlled'):
        for x in packets[kind]:
            cached=lookup[previous.row_key(x['row'])]
            x['emission']={k:cached[k].detach() for k in ('hidden','residual_mask','labels','mask')}
            x['emission']['backbone_hidden']=x['emission'].pop('hidden')
            x['emission']['emitted']=cached['emitted']
    return packets,dict(source_packets=sha(cache_path),frozen_emission_packets=sha(old_path),
                         emission_metadata=old['metadata'])


@torch.no_grad()
def readout_oracle(model,t,row,answers):
    """Exact production state-machine rollout against adversarial fallback.

    Stable gates make this stronger than a particular backbone distribution:
    put virtually all native probability on a wrong token at every step.
    Full-model oracle/entrypoint checks are separately run on representative cases.
    """
    mem=model.native_question_memory;assert mem.span_readout.stable_gate
    ids=torch.tensor([t.prepare_generation_hierarchy(previous.prompt(row)).token_ids])
    _,states,records=mem(ids,torch.zeros(1,ids.shape[1],model.cfg.d_model),model.construction_embedding,t,return_readout=True)
    assert records
    initial=records[-1]
    for answer in answers:
        expected=CompleteAnswerSpanPredictor.payload(t,answer)
        states_at=states;record=initial
        for index,target in enumerate(expected):
            _,_,q,current,plan,emitted=record
            injected=dict(plan,candidates=[dict(tokens=expected)],probabilities=q.new_ones(1),log_probabilities=q.new_zeros(1))
            native=torch.zeros(1,1,model.vocab_size);native[0,0,(target+1)%model.vocab_size]=1000.
            mixed=mem.span_readout.mix(native,[(0,0,q,current,injected,emitted)])
            assert int(mixed.argmax(-1))==target,(row['question'],answer,index)
            _,states_at,following=mem(torch.tensor([[target]]),torch.zeros(1,1,model.cfg.d_model),model.construction_embedding,t,states_at,return_readout=True)
            if index+1<len(expected):
                assert following;record=following[-1]
            else:assert target==t.eos_id and not states_at[0].emitting


def full_batch(model,t,row):
    text=f"<BOI>{previous.prompt(row)}<EOI><BOO>{row['answer']}<EOO>"
    ds=TextWindowDataset([text],t,seq_len=2048,max_samples=0,preserve_grounded_qa_context=True)
    assert len(ds)==1
    fields=('input_ids','labels','signature_ids','signature_level_ids','signature_relation_ids','parent_signature_ids','signature_family_ids','hierarchy_vectors','loss_mask')
    batch=dict(zip(fields,build_collate_fn(t.pad_id,t.signature_pad_id)([ds[0]])))
    positions=(batch['loss_mask'][0]*batch['labels'][0].ne(t.pad_id)).nonzero().flatten()
    with torch.no_grad():
        out=model(**{k:v for k,v in batch.items() if k!='labels'},collect_telemetry=False)
    return out,batch,positions


def preflight(seed):
    verify_history();gate=read(DATA/'data_gate.json');assert gate['passed']
    for name,digest in gate['artifact_hashes'].items():assert sha(DATA/name)==digest
    model,t=load(seed);initial={k:v.clone() for k,v in model.state_dict().items()}
    packets,binding=cache_inputs(seed);maximum=0.;checked=0
    with torch.no_grad():
        for kind,xs in packets.items():
            for x in xs:
                a,b=decision(model,x),observed_packet(model,t,x['row'])
                for left,right in zip(a[:3],b[:3]):torch.testing.assert_close(left,right,rtol=2e-4,atol=3e-5)
                assert a[3]['candidates']==b[3]['candidates']
                torch.testing.assert_close(a[3]['log_probabilities'],b[3]['log_probabilities'],rtol=2e-4,atol=3e-5)
                maximum=max(maximum,float((a[3]['start_log_probabilities']-b[3]['start_log_probabilities']).abs().max()))
                if kind in ('legacy','native','controlled'):
                    q,recalled,_,plan=a
                    x['functional_teacher']=dict(source=plan['start_log_probabilities'].detach(),
                        extent=[logits.detach() for logits,_ in endpoint_groups(model,x,q,plan)],
                        gate=gate_logit(model,q,plan).detach(),tokens=cached_emission(model,x['emission'],q,recalled,plan).detach())
                checked+=1
            print('PREFLIGHT_SOURCES',seed,kind,len(xs),flush=True)
        # All 192 reviewed cases/211 answer references receive an exact readout
        # oracle test; labels never enter ordinary source selection or emission.
        for kind in ('new','development'):
            for x in packets[kind]:readout_oracle(model,t,x['row'],x['accepted_answers'])
        print('PREFLIGHT_READOUT_ORACLES',seed,192,flush=True)
        # Full cached/live token parity, including a long answer and each style.
        samples=[('legacy',0),('legacy',32),('native',0),('native',59),('controlled',0),('controlled',-1)]
        emission_checks=[]
        for kind,index in samples:
            x=packets[kind][index];full,batch,positions=full_batch(model,t,x['row'])
            torch.testing.assert_close(batch['labels'][0,positions],x['emission']['labels'],rtol=0,atol=0)
            teacher=x['functional_teacher']['tokens']
            torch.testing.assert_close(teacher,full.logits[0,positions],rtol=2e-4,atol=3e-5)
            emission_checks.append(dict(pool=kind,index=index,maximum_difference=float((teacher-full.logits[0,positions]).abs().max())))
        prefix_checks=[];generation_checks=[]
        for kind,index in [('new',0),('new',65),('development',20),('native',0)]:
            row=packets[kind][index]['row']
            prefix_checks.append(previous.full_incremental_check(model,t,row))
            result=generate(model,t,row);assert result['canonical']
            # API and direct model entrypoint must use identical input and settings.
            ids,tracks=generation_fields(t,row)
            direct=model.generate(ids,**tracks,suppressed_token_ids=t.generation_suppressed_token_ids(),
                token_signature_lookup=t.signature_lookup_by_token_id(),token_family_lookup=t.signature_family_lookup_by_token_id(),
                token_level_lookup=t.signature_level_lookup_by_token_id(),token_relation_lookup=t.signature_relation_lookup_by_token_id())
            assert direct[0,len(ids[0]):].tolist()==result['tokens']
            generation_checks.append(dict(pool=kind,index=index,tokens=result['tokens'],ended=result['ended']))
        # Causal future-answer substitution: only shared observed prefix logits count.
        row=packets['legacy'][0]['row'];a,ab,_=full_batch(model,t,row)
        changed=dict(row,answer='Different unrelated continuation.')
        b,bb,_=full_batch(model,t,changed)
        left,right=ab['input_ids'][0],bb['input_ids'][0]
        n=next((i for i in range(min(len(left),len(right))) if left[i]!=right[i]),min(len(left),len(right)))
        torch.testing.assert_close(a.logits[:,:n],b.logits[:,:n],rtol=2e-4,atol=3e-5)
    changed=[k for k,v in model.state_dict().items() if k not in initial or not torch.equal(initial[k],v)]
    assert not changed,changed
    folder=OUT/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    torch.save(dict(packets=packets,input_binding=binding),folder/'packets.pt')
    save_checkpoint(model,folder/'initial',tokenizer=t,metrics=dict(experimental=True,accepted=False,updates=0))
    reloaded,rt=load(seed,folder/'initial/model.pt')
    assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
    for kind,index in [('new',0),('development',20),('native',0)]:
        assert generate(reloaded,rt,packets[kind][index]['row'])==generate(model,t,packets[kind][index]['row'])
    scores,ledger=evaluate(reloaded,packets)
    report=dict(passed=True,seed=seed,source_packets=checked,maximum_source_difference=maximum,
        all_reviewed_readout_oracles=192,accepted_reference_count=sum(len(x['accepted_answers']) for k in ('new','development') for x in packets[k]),
        readout_oracle_scope='Production native state machine and mixture against maximally adversarial fallback; full model ordinary generation sampled separately.',
        emission_checks=emission_checks,full_incremental_differences=prefix_checks,generation_checks=generation_checks,
        future_answer_substitution_prefix_logits_equal=True,weights_buffers_unchanged=True,reload_state_bitwise_equal=True,
        initial_sources=scores,initial_ledger=ledger,locked_test_predictions=0)
    write_new(folder/'preflight.json',report)
    print('PREFLIGHT_PASS',seed,flush=True)


def freeze():
    verify_history();gate=read(DATA/'data_gate.json');assert gate['passed']
    packets={}
    for seed in (47,83):
        folder=OUT/f'seed{seed}'
        assert read(folder/'preflight.json')['passed']
        packets[str(seed)]={name:sha(folder/name) for name in ('packets.pt','preflight.json','initial/model.pt')}
    assert read(DATA/'implementation_test_run.json')['success']
    lengths={k:len(v) for k,v in previous.pools(47).items() if k in ('new','legacy','native','controlled')}
    import platform
    metadata=dict(version=4,seeds=[47,83],arms=list(ARMS),
        source_updates=1600,fusion_updates=800,extent_updates=800,joint_updates=1600,maximum_total_updates=19200,
        source_objectives=dict(control='CE(fused supported copyable starts)',branch_consistent='mean(CE(fused), CE(semantic-only))'),
        replay='Unit-weight correct-teacher source/extent preservation, stable gate protection and per-token correct-teacher KL/gold fallback, averaged per example.',
        fusion_experts='corrected_legacy_v1',optimizer=dict(name='AdamW',lr=.001,weight_decay=.01,clip=1.),
        sampling=dict(new=8,legacy=2,native=2,controlled=2),
        schedules={name:previous.schedule(lengths,updates,seed) for name,updates,seed in (
            ('source',1600,20261016),('fusion',800,20261017),('extent',800,20261018),('joint',1600,20261019))},
        source_gate=dict(train_top1=122,development_top1=32,development_top16=58,fresh_top1=16,both_seeds=True),
        source_selection='Final update only; per arm unchanged/learned fusion by worst-seed dev top1, then top16, then loss.',
        later_stage_gates=dict(extent=58,development_raw=32,improvement_over_retained_control=8,locked_test_raw=32),
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),device='cpu',dtype='float32',threads=1),
        qa_readout_route='native',qa_generation_profile='native_qa_eval_v1',locked_test_predictions=0,
        code_hashes={n:sha(ROOT/n) for n in CODE},input_hashes={n:sha(DATA/n) for n in ('data_gate.json','reviewed_answers.json','native_bindings.json')},
        seeds_inputs=packets)
    target=OUT/'frozen_experiment.json'
    if target.exists():assert read(target)==metadata,'Frozen experiment changed'
    else:
        write_new(target,metadata)
        for name in CODE:
            dest=OUT/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as stream:stream.write((ROOT/name).read_bytes())
    return metadata


def verify():
    metadata=read(OUT/'frozen_experiment.json');verify_history()
    for name,expected in metadata['code_hashes'].items():assert sha(ROOT/name)==expected,('Study code changed',name)
    for name,expected in metadata['input_hashes'].items():assert sha(DATA/name)==expected,('Input changed',name)
    for seed,paths in metadata['seeds_inputs'].items():
        for name,expected in paths.items():assert sha(OUT/f'seed{seed}'/name)==expected
    return metadata


def source_fit(seed,arm,fusion=False):
    metadata=verify();assert arm in ARMS
    folder=OUT/f'seed{seed}';dest=folder/arm/('fusion' if fusion else 'source')
    if (dest/'results.json').exists():raise FileExistsError('Do not rerun a completed stage')
    cached=torch.load(folder/'packets.pt',map_location='cpu',weights_only=True);packets=cached['packets']
    if fusion:
        model,t=load(seed,folder/arm/'source/final/model.pt')
        old=model.native_question_memory.semantic_alignment
        layer=NativeSemanticAlignment(model.cfg.d_model,prior_scaling=old.prior_scale is not None,
            structure=old.structure is not None,question_attention=old.question_attention is not None,
            normalized_fusion=True,fusion_experts='corrected_legacy_v1')
        missing,unexpected=layer.load_state_dict(old.state_dict(),strict=False)
        assert not unexpected and all(k=='temperatures' or k.startswith('fusion_gate.') for k in missing)
        model.native_question_memory.semantic_alignment=layer
        model.cfg.native_question_memory_normalized_fusion=True;model.cfg.native_question_memory_fusion_experts='corrected_legacy_v1'
        model.requires_grad_(False);layer.fusion_gate.requires_grad_(True);layer.temperatures.requires_grad_(True)
        allowed=lambda k:k.startswith('native_question_memory.semantic_alignment.fusion_gate.') or k=='native_question_memory.semantic_alignment.temperatures'
    else:
        model,t=load(seed,folder/'initial/model.pt')
        model.native_question_memory.semantic_alignment.requires_grad_(True)
        allowed=lambda k:k.startswith('native_question_memory.semantic_alignment.')
    initial={k:v.clone() for k,v in model.state_dict().items()}
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01)
    stage='fusion' if fusion else 'source';budget=800 if fusion else 1600
    start=0;history=[];clipped=0
    resume=dest/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        assert saved['experiment_sha256']==sha(OUT/'frozen_experiment.json') and saved['step']<=budget
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        start=saved['step'];history=saved['history'];clipped=saved['clipped']
    started=time.monotonic()
    for step in range(start+1,budget+1):
        chosen=metadata['schedules'][stage][step-1];xs=[packets[k][i] for k,i in chosen]
        optimizer.zero_grad(set_to_none=True)
        details=previous.batch_scores(model,xs)
        supervised=torch.stack([gold_mass_loss(d['fused'],x['positive']) if fusion else source_supervision(d,x['positive'],arm) for d,x in zip(details,xs)]).mean()
        protected=[];components={k:[] for k in ('source','extent','gate','tokens')}
        for (kind,index),x in zip(chosen,xs):
            if kind!='new':
                loss,parts=replay(model,x);protected.append(loss)
                for k,value in parts.items():components[k].append(value)
        # Six replay slots contribute at the same per-example weight as the
        # eight new slots; do not renormalize replay to a separate full batch.
        retention=torch.stack(protected).sum()/len(xs)
        loss=supervised+retention
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
        gradients=None
        if step%200==0:
            a=torch.autograd.grad(supervised,parameters,retain_graph=True,allow_unused=True)
            b=torch.autograd.grad(retention,parameters,retain_graph=True,allow_unused=True)
            left=torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for v,p in zip(a,parameters)])
            right=torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for v,p in zip(b,parameters)])
            gradients=dict(source_norm=float(left.norm()),replay_norm=float(right.norm()),
                cosine=float(F.cosine_similarity(left[None],right[None])))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1.)
        optimizer.step()
        if step%50==0 or step==start+1:
            print('OBJECTIVE_STEP',seed,arm,stage,step,'loss',round(float(loss.detach()),5),'seconds',round(time.monotonic()-started,1),flush=True)
        if step%200==0:
            scores,ledger=evaluate(model,packets)
            history.append(dict(step=step,summary=scores,loss=float(loss.detach()),gradients=gradients,clipping_fraction=clipped/step,
                replay_components={k:float(torch.stack(v).mean().detach()) for k,v in components.items()}))
            dest.mkdir(parents=True,exist_ok=True)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),step=step,
                history=history,clipped=clipped,experiment_sha256=sha(OUT/'frozen_experiment.json')),resume)
            dump(dest/'history.json',history)
            print('OBJECTIVE_EVALUATION',seed,arm,stage,step,scores['development'],flush=True)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])]
    assert all(allowed(k) for k in changed),changed
    save_checkpoint(model,dest/'final',tokenizer=t,metrics=dict(experimental=True,accepted=False,arm=arm,stage=stage,updates=budget))
    reloaded,rt=load(seed,dest/'final/model.pt')
    assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
    before,before_ledger=evaluate(model,packets);scores,ledger=evaluate(reloaded,packets)
    for kind in scores:
        assert scores[kind]['top1']==before[kind]['top1'] and scores[kind]['top16']==before[kind]['top16']
        assert all(a['shortlist']==b['shortlist'] for a,b in zip(ledger[kind],before_ledger[kind]))
        assert math.isclose(scores[kind]['source_loss'],before[kind]['source_loss'],rel_tol=2e-4,abs_tol=3e-5)
    write_new(dest/'results.json',dict(seed=seed,arm=arm,stage=stage,updates=budget,summary=scores,ledger=ledger,
        protected_parameters_bitwise_equal=True,changed_keys=changed,checkpoint_sha256=sha(dest/'final/model.pt'),
        history=history,locked_test_predictions=0,accepted=False))
    verify()
    print('OBJECTIVE_STAGE_COMPLETE',seed,arm,stage,scores['development'],flush=True)


def decide():
    verify();selected={};table=[]
    for arm in ARMS:
        options={stage:{str(seed):read(OUT/f'seed{seed}/{arm}/{stage}/results.json') for seed in (47,83)} for stage in ('source','fusion')}
        def key(stage):
            ds=[r['summary']['development'] for r in options[stage].values()]
            return min(x['top1'] for x in ds),min(x['top16'] for x in ds),-max(x['source_loss'] for x in ds)
        stage=max(options,key=key)
        selected[arm]=dict(stage=stage,seeds={seed:r['summary'] for seed,r in options[stage].items()})
        for name,values in options.items():
            for seed,r in values.items():table.append(dict(arm=arm,stage=name,seed=int(seed),updates=r['updates'],summary=r['summary']))
    candidate=selected['branch_consistent'];checks={}
    for seed,scores in candidate['seeds'].items():
        checks[seed]=dict(training=scores['new']['top1']>=122,development=scores['development']['top1']>=32,
            shortlist=scores['development']['top16']>=58,fresh=scores['fresh']['top1']==16)
    passed=all(all(v.values()) for v in checks.values())
    result=dict(status='SOURCE_GATE_PASSED_EXTENT_PENDING' if passed else 'STOPPED_SOURCE_GATE_FAILED',source_gate_passed=passed,
        accepted=False,baseline_replaced=False,selected=selected,checks=checks,comparison=table,
        updates_completed=9600,locked_test_predictions=0,extent_updates=0,joint_updates=0)
    write_new(OUT/'acceptance.json',result)
    print(result['status'],checks,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','freeze','source','fusion','decide'])
    parser.add_argument('--seed',type=int,choices=[47,83]);parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.phase=='preflight':
        if args.seed is None:parser.error('preflight needs --seed')
        preflight(args.seed)
    elif args.phase=='freeze':freeze()
    elif args.phase in ('source','fusion'):
        if args.seed is None or args.arm is None:parser.error('fit needs --seed and --arm')
        source_fit(args.seed,args.arm,args.phase=='fusion')
    else:decide()


if __name__=='__main__':main()
