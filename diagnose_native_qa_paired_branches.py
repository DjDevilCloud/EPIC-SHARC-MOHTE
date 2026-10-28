"""Post-hoc branch attribution; annotations never enter inference scoring."""
import torch

import native_qa_source_study as old
import native_qa_paired_source as pilot


@torch.no_grad()
def main():
    pilot.freeze();result={}
    for seed in (47,83):
        cache=pilot.packets(seed)
        for arm in ('matched_control','paired_assignment'):
            model,_=old.load(seed,'structure_only',pilot.OUT/f'seed{seed}/{arm}/final/model.pt')
            branches={k:dict(top1=0,top16=0,n=64) for k in ('legacy','semantic','fused')}
            rows=[]
            for x in cache['development']:
                details=model.native_question_memory.score_sources(x['q'],x['prior'],x['inputs'])
                row=dict(question=x['row']['question'],answer=x['row']['answer'],prior_scale=float(details['prior_scale']))
                for k in branches:
                    p=details[k].softmax(0)
                    correct=bool(x['positive'][p.argmax()])
                    branches[k]['top1']+=int(correct)
                    branches[k]['top16']+=int(x['positive'][p.topk(min(16,len(p))).indices].any())
                    row[k]=dict(top1=int(p.argmax()),correct=correct)
                rows.append(row)
            result[f'{seed}_{arm}']=dict(branches=branches,
                semantic_correct_fused_wrong=sum(r['semantic']['correct'] and not r['fused']['correct'] for r in rows),
                either_branch_correct=sum(r['semantic']['correct'] or r['legacy']['correct'] for r in rows),rows=rows)
            print('PAIRED_BRANCHES',seed,arm,branches,flush=True)
    old.dump(pilot.OUT/'branch_diagnostics.json',result)


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
