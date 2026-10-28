"""V5 parameter-matched early/late question-conditioning experiment.

Reuses the sealed V4 optimization/execution engine; no second scoring formula.
Historical V4 verification resolves its archived production code explicitly.
"""
import argparse
import importlib.util
import math
import platform
import random
from pathlib import Path

import torch

import native_qa_objective_study as engine
from native_qa_multianswer import ROOT, OUT as DATA, read, sha, write_new, verify_history as earlier_history
from native_qa_functional_replay import source_supervision
from native_semantic_alignment import NativeSemanticAlignment

OUT=ROOT/'review_artifacts/native_qa_question_context_v5'
ARMS=('late_context','early_context')
STAGES=dict(late_context='post_context',early_context='pre_context')
ACTIVE_ARM='late_context'
BASE_LOAD=engine.load


def verify_history():
    earlier_history()
    seal=read(OUT/'history/seal.json')
    old=ROOT/seal['v4_root']
    assert sha(old/'final_artifact_index.json')==seal['v4_index_sha256']
    index=read(old/'final_artifact_index.json')
    for name,expected in index['artifacts'].items():
        assert sha(old/name)==expected,('Historical V4 evidence changed',name)
    for name,expected in seal['code_hashes'].items():
        assert sha(OUT/'history/v4_code'/name)==expected,('Archived V4 code changed',name)
    return True


def load(seed,path=None):
    model,t=BASE_LOAD(seed,path)
    old=model.native_question_memory.semantic_alignment
    if old.question_attention is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261024+seed)
            layer=NativeSemanticAlignment(model.cfg.d_model,
                prior_scaling=old.prior_scale is not None,structure=old.structure is not None,
                question_attention=True,normalized_fusion=old.fusion_gate is not None,
                fusion_experts=old.fusion_experts)
        missing,extra=layer.load_state_dict(old.state_dict(),strict=False)
        assert not extra and all(k.startswith(('question_attention.','question_residual.')) for k in missing)
        assert len(missing)==5
        model.native_question_memory.semantic_alignment=layer
        model.cfg.native_question_memory_question_attention=True
    if path is None or Path(path)==OUT/f'seed{seed}/initial/model.pt':
        stage=STAGES[ACTIVE_ARM]
        model.cfg.native_question_memory_question_attention_stage=stage
        model.native_question_memory.semantic_alignment.question_attention_stage=stage
    model.eval();model.requires_grad_(False)
    return model,t


def bind_engine():
    engine.OUT=OUT
    engine.ARMS=ARMS
    engine.load=load
    engine.verify_history=verify_history
    engine.verify=verify
    # Exactly the prior control objective in both arms: do not combine the
    # failed independent-branch objective with this architecture hypothesis.
    engine.source_supervision=lambda details,positive,arm:source_supervision(details,positive,'control')


@torch.no_grad()
def preflight(seed):
    assert not (OUT/f'seed{seed}/preflight.json').exists(),'Do not overwrite a completed preflight'
    engine.preflight(seed)
    packets=torch.load(OUT/f'seed{seed}/packets.pt',map_location='cpu',weights_only=True)['packets']
    model,t=load(seed,OUT/f'seed{seed}/initial/model.pt')
    old=read(DATA/f'study/seed{seed}/preflight.json')
    initial=read(OUT/f'seed{seed}/preflight.json')
    for kind,counts in old['initial_sources'].items():
        assert initial['initial_sources'][kind]['top1']==counts['top1']
        assert initial['initial_sources'][kind]['top16']==counts['top16']
    layer=model.native_question_memory.semantic_alignment
    before={k:v.clone() for k,v in model.state_dict().items()}
    # Both placements must start with bitwise-identical source states and
    # inference proposals; no favorable source decisions are selected here.
    zero_checks=0
    for xs in packets.values():
        for x in xs:
            layer.question_attention_stage='post_context';late=engine.decision(model,x)
            layer.question_attention_stage='pre_context';early=engine.decision(model,x)
            for a,b in zip(late[:3],early[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            assert late[3]['candidates']==early[3]['candidates']
            torch.testing.assert_close(late[3]['log_probabilities'],early[3]['log_probabilities'],rtol=0,atol=0)
            zero_checks+=1
    # An active, nonzero early residual must also agree with ordinary execution.
    # This probe is discarded, never saved as a training parent.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+712)
        layer.question_residual.weight.normal_(std=.02)
    maximum=0.;active_checks=0
    model.cfg.native_question_memory_question_attention_stage='pre_context'
    for kind,xs in packets.items():
        for x in xs:
            cached=engine.decision(model,x);live=engine.observed_packet(model,t,x['row'])
            for a,b in zip(cached[:3],live[:3]):torch.testing.assert_close(a,b,rtol=2e-4,atol=3e-5)
            assert cached[3]['candidates']==live[3]['candidates']
            torch.testing.assert_close(cached[3]['log_probabilities'],live[3]['log_probabilities'],rtol=2e-4,atol=3e-5)
            maximum=max(maximum,float((cached[3]['start_log_probabilities']-live[3]['start_log_probabilities']).abs().max()))
            active_checks+=1
        print('CONTEXT_ACTIVE_PARITY',seed,kind,len(xs),flush=True)
    emission=[]
    for kind,index in [('legacy',0),('legacy',32),('native',0),('controlled',0)]:
        x=packets[kind][index];q,recalled,_,plan=engine.decision(model,x)
        cached=engine.cached_emission(model,x['emission'],q,recalled,plan)
        full,batch,positions=engine.full_batch(model,t,x['row'])
        torch.testing.assert_close(cached,full.logits[0,positions],rtol=2e-4,atol=3e-5)
        emission.append(dict(pool=kind,index=index,maximum_difference=float((cached-full.logits[0,positions]).abs().max())))
    prefix=[engine.previous.full_incremental_check(model,t,packets[k][i]['row'])
            for k,i in [('new',0),('development',20),('native',0)]]
    model.load_state_dict(before)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    write_new(OUT/f'seed{seed}/conditioning_preflight.json',dict(passed=True,
        identical_initial_source_cases=zero_checks,active_early_cached_live_cases=active_checks,
        maximum_source_difference=maximum,active_early_emission=emission,
        active_early_incremental_differences=prefix,probe_discarded=True,
        original_parent_initial_source_counts_unchanged=True,optimizer_updates=0,locked_test_predictions=0))
    print('CONTEXT_PREFLIGHT_PASS',seed,flush=True)


def freeze():
    verify_history()
    assert read(OUT/'implementation_test_run.json')['success']
    source=read(DATA/'study/frozen_experiment.json')
    inputs={}
    for seed in (47,83):
        folder=OUT/f'seed{seed}'
        assert read(folder/'preflight.json')['passed'] and read(folder/'conditioning_preflight.json')['passed']
        inputs[str(seed)]={name:sha(folder/name) for name in ('packets.pt','preflight.json','conditioning_preflight.json','initial/model.pt')}
    names=sorted(p.name for p in ROOT.glob('*.py'))
    metadata=dict(version=5,seeds=[47,83],arms=list(ARMS),source_updates=1600,
        maximum_total_updates=6400,fusion_updates=0,extent_updates=0,joint_updates=0,
        hypothesis='Question evidence before source contextualization improves held-out supported-start selection versus parameter-matched post-context attention.',
        source_objectives={arm:'CE(fused supported copyable starts)' for arm in ARMS},
        attention_placement=STAGES,
        early_mapping='Duplicate W lexical queries to 2W; attend to 2W contextual question states; project 2W residual and average its two W halves before source GRU.',
        replay=source['replay'],optimizer=source['optimizer'],sampling=source['sampling'],
        schedules=dict(source=source['schedules']['source']),source_gate=source['source_gate'],
        selection='Final update only; no intermediate selection, fusion fitting, extensions or additional arms.',
        architecture_attribution='Identical attention parameter shapes, seed-specific initial tensors, parent encoder weights and all other components; only placement differs.',
        downstream='Source success is required before any separately specified extent/joint/test continuation. No transfer acceptance from this source study.',
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),device='cpu',dtype='float32',threads=1),
        code_hashes={name:sha(ROOT/name) for name in names},
        verification_hashes={name:sha(ROOT/name) for name in ('tests/test_native_question_context.py',
            'tests/test_native_qa_context_study.py','NATIVE_QA_QUESTION_CONTEXT.md',
            str((OUT/'implementation_tests.log').relative_to(ROOT)))},
        input_hashes={name:sha(DATA/name) for name in ('data_gate.json','reviewed_answers.json','native_bindings.json')},
        seeds_inputs=inputs,implementation_test_sha256=sha(OUT/'implementation_test_run.json'),
        historical_seal_sha256=sha(OUT/'history/seal.json'),qa_readout_route='native',qa_generation_profile='native_qa_eval_v1',
        locked_test_predictions=0,baseline_replaced=False)
    write_new(OUT/'frozen_experiment.json',metadata)
    for name in names:
        dest=OUT/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write((ROOT/name).read_bytes())
    print('CONTEXT_EXPERIMENT_FROZEN',metadata['maximum_total_updates'],flush=True)


def verify():
    metadata=read(OUT/'frozen_experiment.json');verify_history()
    for name,expected in metadata['code_hashes'].items():assert sha(ROOT/name)==expected,('Study code changed',name)
    for name,expected in metadata['input_hashes'].items():assert sha(DATA/name)==expected,('Data changed',name)
    for seed,paths in metadata['seeds_inputs'].items():
        for name,expected in paths.items():assert sha(OUT/f'seed{seed}'/name)==expected
    assert sha(OUT/'implementation_test_run.json')==metadata['implementation_test_sha256']
    assert sha(OUT/'history/seal.json')==metadata['historical_seal_sha256']
    for name,expected in metadata['verification_hashes'].items():assert sha(ROOT/name)==expected,('Verification artifact changed',name)
    return metadata


def fit(seed,arm):
    global ACTIVE_ARM
    ACTIVE_ARM=arm
    engine.source_fit(seed,arm,fusion=False)


def decide():
    verify();selected={};comparison=[]
    for arm in ARMS:
        values={}
        for seed in (47,83):
            result=read(OUT/f'seed{seed}/{arm}/source/results.json')
            assert result['updates']==1600 and result['protected_parameters_bitwise_equal']
            assert sha(OUT/f'seed{seed}/{arm}/source/final/model.pt')==result['checkpoint_sha256']
            values[str(seed)]=result['summary']
            comparison.append(dict(arm=arm,seed=seed,updates=1600,summary=result['summary']))
        selected[arm]=dict(stage='source',seeds=values)
    checks={seed:dict(training=s['new']['top1']>=122,development=s['development']['top1']>=32,
        shortlist=s['development']['top16']>=58,fresh=s['fresh']['top1']==16)
        for seed,s in selected['early_context']['seeds'].items()}
    passed=all(all(c.values()) for c in checks.values())
    write_new(OUT/'acceptance.json',dict(status='SOURCE_GATE_PASSED_DOWNSTREAM_PENDING' if passed else 'STOPPED_SOURCE_GATE_FAILED',
        source_gate_passed=passed,accepted=False,baseline_replaced=False,selected=selected,checks=checks,
        comparison=comparison,updates_completed=6400,locked_test_predictions=0,extent_updates=0,joint_updates=0,
        fusion_updates=0,full_raw_retention_certified=False))
    print('CONTEXT_DECISION',passed,checks,flush=True)


def report():
    verify();decision=read(OUT/'acceptance.json')
    spec=importlib.util.spec_from_file_location('v4_ledger_helpers',DATA/'report_source_study.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    contrasts={};uncertainty={};deltas={}
    for seed in (47,83):
        packets=torch.load(OUT/f'seed{seed}/packets.pt',map_location='cpu',weights_only=True)['packets']
        measured={}
        for arm in ARMS:
            model,t=load(seed,OUT/f'seed{seed}/{arm}/source/final/model.pt')
            assert model.cfg.native_question_memory_question_attention_stage==STAGES[arm]
            ledger,branches=helpers.failure_ledger(model,packets)
            paired=helpers.contrast(ledger['development']['rows'])
            write_new(OUT/f'seed{seed}/{arm}/source/failure_ledger.json',dict(seed=seed,arm=arm,groups=ledger,
                branch_diagnostics=branches,contrasting_development_questions=paired,replay_proxy=helpers.replay_proxy(model,packets),
                semantic_score_role='Residual correction in both arms; no standalone probability calibration claim.',
                optimization_updates_by_reporter=0,locked_test_predictions=0))
            contrasts[f'{seed}/{arm}']=paired;measured[arm]=ledger['development']['rows']
            print('CONTEXT_LEDGER',seed,arm,flush=True)
        grouped={}
        for a,b in zip(measured['late_context'],measured['early_context']):
            assert a['fingerprint']==b['fingerprint']
            grouped.setdefault(a['source_group'],[]).append(int(b['start_top1'])-int(a['start_top1']))
        assert len(grouped)==32 and all(len(v)==2 for v in grouped.values())
        ds=[sum(v)/2 for v in grouped.values()];rng=random.Random(20261024)
        samples=sorted(sum(rng.choice(ds) for _ in ds)/32 for _ in range(10000))
        uncertainty[str(seed)]=dict(early_minus_late=sum(ds)/32,bootstrap_95_interval=[samples[249],samples[9749]],groups=32,
            caveat='Observed source-family uncertainty only; development was previously inspected and authoritative parent provenance is unavailable.')
    write_new(OUT/'uncertainty.json',uncertainty)
    write_new(OUT/'contrasting_questions.json',contrasts)
    lines=['# V5 question-conditioned source contextualization','',decision['status'],'',
        '| Arm | Seed | Training /128 | Dev starts /64 | Dev top16 /64 | Fresh /16 |',
        '|---|---|---|---|---|---|']
    for row in decision['comparison']:
        s=row['summary'];lines.append(f"| {row['arm']} | {row['seed']} | {s['new']['top1']} | {s['development']['top1']} | {s['development']['top16']} | {s['fresh']['top1']} |")
    lines+=['','Both arms have identical parameter shapes and initialization; only attention placement differs.',
        'Each ran 1600 source updates with the corrected-distribution objective and identical functional replay. Total: 6400 updates. No fusion fitting.',
        'All 192 reviewed questions, 211 literal accepted references and source bindings were reused unchanged. No predictions selected annotations.',
        'Full execution preflights include 646 zero-initialized parity cases and 646 active early-conditioning cached/live cases per seed; oracle and ordinary-generation scopes are recorded separately.',
        'Gold-start extent remains auxiliary; raw answers and termination are unmeasured at this source gate. Cached teacher-forced replay is not full retention.',
        'See contrasting_questions.json for both-question success, uncertainty.json for paired source-family bootstrap intervals, and each checkpoint ledger for stage failures.',
        'No locked-test predictions, extent updates or joint updates were made. Accepted baselines and prior study artifacts remain unchanged.',
        'Review limitations remain: assistant annotations, finite literal references, missing authoritative provenance and disclosed question-provided qualifiers.',
        'Source success would authorize planning the downstream gated study, not acceptance as a general QA model. Failure ends this architecture-placement study without more arms or updates.']
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    verify()
    write_new(OUT/'artifact_audit.json',dict(passed=True,baseline_changed=False,accepted=False,
        optimizer_updates=6400,locked_test_predictions=0,extent_updates=0,joint_updates=0,
        artifact_hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print('CONTEXT_REPORT_COMPLETE',decision['status'],uncertainty,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['preflight','freeze','source','decide','report','verify'])
    parser.add_argument('--seed',type=int,choices=[47,83]);parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();torch.set_num_threads(1);bind_engine()
    if args.phase=='preflight':
        if args.seed is None:parser.error('preflight requires --seed')
        preflight(args.seed)
    elif args.phase=='freeze':freeze()
    elif args.phase=='source':
        if args.seed is None or args.arm is None:parser.error('source requires --seed and --arm')
        fit(args.seed,args.arm)
    elif args.phase=='decide':decide()
    elif args.phase=='report':report()
    else:verify();print('CONTEXT_FROZEN_INPUTS_VERIFIED')


if __name__=='__main__':main()
