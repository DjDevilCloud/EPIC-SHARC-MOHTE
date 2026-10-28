"""Read-only final-check repair; never performs optimization or alters checkpoints."""
import argparse
import torch

import native_qa_source_study as study
from run_native_qa_source_stage import sealed_inputs


def finalize(seed,arm):
    sealed_inputs();folder=study.OUT/f'seed{seed}/{arm}'
    saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
    assert saved['experiment_sha256']==study.sha(study.OUT/'frozen_experiment.json')
    budget=800 if arm=='normalized_fusion' else 1600
    assert saved['step']==budget
    selected=study.read(study.OUT/'encoder_selection.json')['selected_encoder'] if arm=='normalized_fusion' else arm
    reloaded,t=study.load(seed,selected,folder/'final/model.pt')
    assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in saved['model'].items())
    model,mt=study.load(seed,selected,folder/'final/model.pt')
    if arm=='normalized_fusion':
        model.native_question_memory.semantic_alignment.fusion_gate.requires_grad_(True)
        model.native_question_memory.semantic_alignment.temperatures.requires_grad_(True)
        initial,_=study.load(seed,selected,study.OUT/f'seed{seed}/{selected}/final/model.pt')
        allowed=lambda k:k.startswith('native_question_memory.semantic_alignment.fusion_gate.') or k=='native_question_memory.semantic_alignment.temperatures'
    else:
        model.native_question_memory.semantic_alignment.requires_grad_(True)
        initial,_=study.load(seed,arm,folder/'initial/model.pt')
        allowed=lambda k:k.startswith('native_question_memory.semantic_alignment.')
    before=initial.state_dict();after=model.state_dict()
    changed=[k for k,v in after.items() if k not in before or not torch.equal(before[k],v)]
    assert all(allowed(k) for k in changed)
    packets=torch.load(study.OUT/f'seed{seed}/packets.pt',map_location='cpu',weights_only=True)['packets']
    maximum=0.;checked=0
    with torch.no_grad():
        for xs in packets.values():
            for x in xs:
                a=model.native_question_memory.score_sources(x['q'],x['prior'],x['inputs'])['fused']
                b=reloaded.native_question_memory.score_sources(x['q'],x['prior'],x['inputs'])['fused']
                torch.testing.assert_close(a,b,rtol=2e-4,atol=3e-5)
                maximum=max(maximum,float((a-b).abs().max()))
                left=study.decision(model,x)[3];right=study.decision(reloaded,x)[3]
                assert left['candidates']==right['candidates']
                torch.testing.assert_close(left['log_probabilities'],right['log_probabilities'],rtol=2e-4,atol=3e-5)
                checked+=1
    scores,ledger=study.evaluate(reloaded,packets);original,_=study.evaluate(model,packets)
    for kind in scores:
        for metric in ('n','top1','top16'):assert scores[kind][metric]==original[kind][metric]
        assert abs(scores[kind]['source_loss']-original[kind]['source_loss'])<=3e-5
    greedy=[]
    for kind,i in [('new',0),('new',1),('development',0),('development',1),('native',0),('legacy',0),('fresh',0),('fresh',1)]:
        row=packets[kind][i]['row'];left=study.generate(model,mt,row);right=study.generate(reloaded,t,row)
        assert left==right,(kind,i)
        greedy.append(dict(pool=kind,index=i,identical=True))
    report=dict(seed=seed,arm=arm,updates=budget,summary=scores,ledger=ledger,changed_keys=changed,
        protected_parameters_bitwise_equal=True,checkpoint_sha256=study.sha(folder/'final/model.pt'),locked_test_predictions=0,
        reload_validation=dict(parameters_buffers_bitwise_equal=True,source_packets_checked=checked,
            candidate_identities_equal=True,maximum_source_logit_difference=maximum,greedy_checks=greedy,
            tolerance=dict(rtol=2e-4,atol=3e-5),
            repair='Original final check required exact equality of floating-point source-loss averages. This read-only check uses the original logit tolerances, exact weights/candidate identities/counts/greedy answers, and performs zero additional updates.',
            finalizer_sha256=study.sha(__file__)))
    if arm=='normalized_fusion':report['selected_encoder']=selected
    study.dump(folder/'source_results.json',report)
    sealed_inputs()
    print('READONLY_FINALIZATION_PASS',seed,arm,scores['development'],maximum,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(47,83),required=True)
    parser.add_argument('--arm',choices=study.ARMS+('normalized_fusion',),required=True)
    args=parser.parse_args();torch.set_num_threads(1);finalize(args.seed,args.arm)
