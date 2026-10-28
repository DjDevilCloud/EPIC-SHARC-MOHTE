"""Attribute recorded proposal/raw disagreements without interventions or updates."""
import torch
import native_qa_source_study as study
from source_answer_span import CompleteAnswerSpanPredictor
from run_native_qa_source_stage import sealed_inputs


@torch.no_grad()
def trace():
    sealed_inputs();selected=study.read(study.OUT/'encoder_selection.json')['selected_encoder'];result=[]
    for seed in (47,83):
        for arm in (selected,'normalized_fusion'):
            path=study.OUT/f'seed{seed}/{arm}/failure_ledger.json'
            if not path.exists():continue
            ledger=study.read(path);model,t=study.load(seed,selected,path.parent/'final/model.pt');mem=model.native_question_memory
            rows={study.qa_fingerprint(r):r for kind,rs in study.pools(seed).items() for r in rs}
            for kind,group in ledger.items():
                for case in group['rows']:
                    if not case['complete_proposal_correct'] or case['raw_answer'] is None or case['raw_answer']['exact']:continue
                    row=rows[case['fingerprint']];ids=torch.tensor([t.prepare_generation_hierarchy(study.prompt(row)).token_ids])
                    _,states=mem(ids,torch.zeros(1,ids.shape[1],model.cfg.d_model),model.construction_embedding,t)
                    q,_,_,plan=states[0].cached_packet
                    assert mem.span_readout.stable_gate and mem.span_readout.conditional_copy
                    expected=CompleteAnswerSpanPredictor.payload(t,row['answer']);actual=case['raw_answer']['tokens']
                    diff=next(i for i,(a,b) in enumerate(zip(expected,actual)) if a!=b)
                    prefix=tuple(actual[:diff]);support=[];targets=[]
                    for i,c in enumerate(plan['candidates']):
                        if len(c['tokens'])>diff and c['tokens'][:diff]==prefix:
                            support.append(i);targets.append(c['tokens'][diff])
                    weights=plan['log_probabilities'][support];normalizer=weights.logsumexp(0)
                    target_ids=torch.tensor(targets)
                    correct=float((weights[target_ids==expected[diff]].logsumexp(0)-normalizer).exp())
                    wrong=float((weights[target_ids==actual[diff]].logsumexp(0)-normalizer).exp())
                    surviving=float(normalizer.exp())
                    gate_logit=mem.span_readout.copy_gate(torch.cat((plan['copy_context'],q))).squeeze().double()
                    gate=float(gate_logit.sigmoid());native_bound=float((-gate_logit).sigmoid())
                    result.append(dict(seed=seed,arm=arm,split=kind,fingerprint=case['fingerprint'],question=row['question'],answer=row['answer'],
                        raw=case['raw_answer']['raw'],first_divergent_token_index=diff,correct_next_token=t.construction_units[expected[diff]].text,
                        actual_next_token=t.construction_units[actual[diff]].text,copy_gate=gate,surviving_prefix_mass=surviving,
                        conditional_correct_token_mass=correct,conditional_actual_token_mass=wrong,native_fallback_upper_bound=native_bound,
                        copy_mass_margin_exceeds_all_possible_native_fallback=gate*(wrong-correct)>native_bound,
                        interpretation='A highest-probability complete proposal need not win greedy token-mixture decoding: several lower-probability proposals can pool their next-token mass. No oracle proposal was inserted.'))
    study.dump(study.OUT/'emission_disagreements.json',dict(optimization_updates=0,interventions=0,test_predictions=0,cases=result))
    sealed_inputs();print('EMISSION_TRACES',len(result),flush=True)


if __name__=='__main__':torch.set_num_threads(1);trace()
