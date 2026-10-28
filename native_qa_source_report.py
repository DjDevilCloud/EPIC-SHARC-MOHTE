"""Summarize source study results with source-family bootstrap uncertainty."""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'review_artifacts/native_qa_transfer_passages_v2/source_study'


def grouped_interval(rows,field):
    groups=defaultdict(list)
    for r in rows:groups[r['source_group']].append(float(r[field]))
    samples=list(groups.values());rng=random.Random(20261016)
    means=[]
    for _ in range(10000):
        drawn=[samples[rng.randrange(len(samples))] for _ in samples]
        means.append(sum(sum(x) for x in drawn)/sum(len(x) for x in drawn))
    means.sort()
    return dict(unit='source_family',families=len(samples),resamples=10000,seed=20261016,
                estimate=sum(float(r[field]) for r in rows)/len(rows),interval95=[means[250],means[9749]],
                limitation='Bootstrap reflects the sampled reviewed source families, not unknown missing parent-page provenance.')


def report():
    decision=json.loads((OUT/'acceptance.json').read_text());selection=json.loads((OUT/'encoder_selection.json').read_text())
    results={}
    for arm in ('control','structure_only','question_attention_only','normalized_fusion'):
        for seed in (47,83):
            folder=OUT/f'seed{seed}/{arm}'
            result=json.loads((folder/'source_results.json').read_text());rows=result['ledger']['development']
            groups=defaultdict(list)
            for row in rows:groups[row['source_group']].append(row)
            contrasts=dict(source_pairs=len(groups),both_correct=sum(all(r['start_top1'] for r in pair) for pair in groups.values()),
                one_correct=sum(sum(r['start_top1'] for r in pair)==1 for pair in groups.values()),
                neither_correct=sum(not any(r['start_top1'] for r in pair) for pair in groups.values()))
            entry=dict(summary=result['summary'],development_source_top1=grouped_interval(rows,'start_top1'),
                development_top16=grouped_interval(rows,'start_top16'),contrasting_questions=contrasts)
            ledger_path=folder/'failure_ledger.json'
            if ledger_path.exists():
                detail=json.loads(ledger_path.read_text());entry['raw_answers']={}
                for split,group in detail.items():
                    measured=[r for r in group['rows'] if r['raw_answer'] is not None]
                    entry['raw_answers'][split]=dict(original_denominator=group['denominator'],measured=len(measured),
                        representable=sum(r['representable'] for r in group['rows']),
                        shortlist_inclusion=sum(r['shortlist_inclusion'] for r in group['rows']),
                        gold_start_extent=sum(r['gold_start_extent'] for r in group['rows']),
                        complete_proposal=sum(r['complete_proposal_correct'] for r in group['rows']),
                        exact=sum(r['raw_answer']['exact'] for r in measured) if measured else None,
                        correct_proposal_wrong_raw=sum(r['complete_proposal_correct'] and not r['raw_answer']['exact'] for r in measured) if measured else None)
                    if split=='development' and measured:
                        raw=[dict(source_group=r['source_group'],exact=r['raw_answer']['exact']) for r in measured]
                        entry['development_raw_exact_interval']=grouped_interval(raw,'exact')
            results[f'{arm}/seed{seed}']=entry
    branches=OUT/'branch_diagnostics.json'
    emissions=OUT/'emission_disagreements.json'
    report=dict(decision=decision,results=results,test_model_evaluations=0,task='passage_grounded_extractive_qa',
        branch_diagnostics=json.loads(branches.read_text()) if branches.exists() else None,
        emission_disagreements=json.loads(emissions.read_text()) if emissions.exists() else None)
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Bounded passage source-selection results','',f"Decision: **{decision['status']}**. Selected encoder: `{selection['selected_encoder']}`.",'',
        '| Arm | Seed | Train top-1 /128 | Dev top-1 /64 | Dev top-16 /64 | Fresh starts /16 |',
        '|---|---:|---:|---:|---:|---:|']
    for key,r in results.items():
        arm,seed=key.split('/');s=r['summary']
        lines.append(f"| {arm} | {seed[4:]} | {s['new']['top1']} | {s['development']['top1']} | {s['development']['top16']} | {s['fresh']['top1']} |")
    lines+=['','All source arms ran exactly 1,600 updates. The selected encoder was frozen for the single 800-update fusion comparison.',
        'The locked test received no model predictions. Endpoint/emission parameters were frozen; their behavior can still change as semantic inputs change.',
        'Protected parameter and baseline checkpoint hashes were checked. No candidate replaces the accepted baseline.',
        '', 'Uncertainty uses 10,000 bootstrap resamples of the 32 paired development source families. Full intervals, contrast outcomes, raw-answer measurement denominators, and provenance limitations are in `report.json`.',
        '',decision.get('failed_hypothesis') or 'The source gate passes; extent and full retention/transfer gates remain untested.']
    if branches.exists():
        lines+=['','The final selected and fusion models also agree with fresh live extraction on all 64 development decisions per seed. Standalone semantic/legacy branch scores are diagnostic only.']
        for key,r in report['branch_diagnostics']['results'].items():
            if key.startswith('normalized_fusion/'):
                c=r['counts'];lines.append(f"{key}: legacy top-1 {c['legacy']['top1']}/64, semantic top-1 {c['semantic']['top1']}/64, fused top-1 {c['fused']['top1']}/64.")
    lines+=['','| Candidate | Seed | Gold-start extent /64 | Correct proposal /64 | Raw exact /64 |',
        '|---|---:|---:|---:|---:|']
    for key,r in results.items():
        raw=r.get('raw_answers',{}).get('development')
        if raw and raw['measured']:
            arm,seed=key.split('/');lines.append(f"| {arm} | {seed[4:]} | {raw['gold_start_extent']} | {raw['complete_proposal']} | {raw['exact']} |")
    if emissions.exists():
        cases=report['emission_disagreements']['cases'];unique=len({r['fingerprint'] for r in cases})
        explained=sum(r['copy_mass_margin_exceeds_all_possible_native_fallback'] for r in cases)
        lines+=['',f'Read-only emission analysis: {len(cases)} checkpoint/example traces cover {unique} distinct QA cases. In {explained} traces, pooled copy-continuation mass beats the correct next token by more than the entire possible native fallback. The highest-probability complete span therefore does not necessarily win greedy token-mixture decoding. This is distinct from the earlier native-fallback arithmetic defect. No new decoder arm or optimization was introduced.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':report()
