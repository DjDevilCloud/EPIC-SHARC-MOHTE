"""Read-only branch attribution and live/cached final development verification."""
import torch
import native_qa_source_study as study
from native_question_memory import NativeQuestionState
from run_native_qa_source_stage import sealed_inputs


@torch.no_grad()
def diagnose():
    sealed_inputs();selected=study.read(study.OUT/'encoder_selection.json')['selected_encoder'];result={}
    for seed in (47,83):
        rows=study.pools(seed)['development']
        for arm in (selected,'normalized_fusion'):
            folder=study.OUT/f'seed{seed}/{arm}';model,t=study.load(seed,selected,folder/'final/model.pt');mem=model.native_question_memory
            expected=study.read(folder/'source_results.json')['ledger']['development'];cases=[]
            counts={key:dict(top1=0,top16=0) for key in ('legacy','semantic','fused')}
            for row,reference in zip(rows,expected):
                ids=torch.tensor([t.prepare_generation_hierarchy(study.prompt(row)).token_ids])
                _,states=mem(ids,torch.zeros(1,ids.shape[1],model.cfg.d_model),model.construction_embedding,t,diagnostics=True)
                state=states[0];plan=state.cached_packet[3];gold=NativeQuestionState()
                for token in t.encode('<BOI>Context:\n'+row['answer']+'<EOI>',add_special_tokens=False):gold,_=mem._observe(gold,token,t)
                words=[w for w,_ in state.source];target=[w for w,_ in gold.source]
                matches=[i for i in range(len(words)-len(target)+1) if words[i:i+len(target)]==target]
                candidates=mem.span_readout.candidates(state,t,matches)
                starts={c['start'] for c in candidates if c['text']==row['answer']};assert starts
                branches={}
                for key in counts:
                    scores=plan['diagnostics'][key+'_start_logits'];prob=scores.softmax(0)
                    top=int(prob.argmax());short=prob.topk(min(16,len(prob))).indices.tolist()
                    if key=='fused':
                        assert short==plan['selected_starts']
                        assert short==reference['shortlist']
                    values=dict(top1=top in starts,top16=bool(starts.intersection(short)),selected=top)
                    for metric in ('top1','top16'):counts[key][metric]+=values[metric]
                    branches[key]=values
                assert branches['fused']['top1']==reference['start_top1']
                assert branches['fused']['top16']==reference['start_top16']
                assert branches['fused']['selected']==reference['selected_start']
                cases.append(dict(fingerprint=study.qa_fingerprint(row),source_group=row['source_group'],branches=branches,
                                  both_individual_branches_miss_top16=not(branches['legacy']['top16'] or branches['semantic']['top16'])))
            result[f'{arm}/seed{seed}']=dict(denominator=64,live_cached_decisions_agree=64,full_shortlist_identity_agreement=64,counts=counts,cases=cases,
                neither_branch_top16=sum(r['both_individual_branches_miss_top16'] for r in cases),
                note='Standalone legacy/semantic rankings are evaluator diagnostics, not an added trained arm or normal inference success.')
            print('LIVE_BRANCH_DIAGNOSTIC',seed,arm,counts,flush=True)
    study.dump(study.OUT/'branch_diagnostics.json',dict(task='passage_grounded_extractive_qa',optimization_updates=0,test_predictions=0,results=result))
    sealed_inputs()


if __name__=='__main__':torch.set_num_threads(1);diagnose()
