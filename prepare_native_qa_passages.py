"""Build answer-blind passage candidates; never grants semantic review approval."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from data import parse_diverse_qa_record
from native_qa_review import qa_fingerprint
from native_qa_transfer_data import fixed_passages, group_sources, normalized, regroup_passage_candidates
from native_question_memory import NativeQuestionState
from source_answer_span import CompleteAnswerSpanPredictor
from train import load_bundle_from_checkpoint

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'review_artifacts/native_qa_transfer_passages_v2/data'


def main():
    import pyarrow.parquet as pq
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'candidates.json').exists():
        raise FileExistsError('Candidate inventory already exists; reviewed inputs must not be replaced')
    source=Path(r'D:\AAISourceStart\SourceALLM\Datasets\NVIDIAPT\DiverseQA\part_000000.parquet')
    snapshot=ROOT/'review_artifacts/native_qa_transfer_v1/data/exclusion_snapshot.json'
    exclusion=json.loads(snapshot.read_text())
    old=exclusion['contexts']; old_ids=set(exclusion['source_ids'])
    rows=[]
    for r in pq.read_table(source).to_pylist():
        parsed=parse_diverse_qa_record(r['text'])
        if parsed:
            context,pairs=parsed
            rows.append(dict(context=context,pairs=pairs,source_id=r['id']))
    # Group ORIGINAL documents first, including excluded documents. Passage count
    # cannot create additional families or reopen a previously exposed family.
    family=group_sources([r['context'] for r in rows]+old,source_ids=[r['source_id'] for r in rows]+[None]*len(old))
    excluded=set(family[len(rows):])|{g for r,g in zip(rows,family) if r['source_id'] in old_ids}
    m,t,_=load_bundle_from_checkpoint(ROOT/'review_artifacts/native_retained_decisions_counterfactual/seed47/candidate/model.pt',device='cpu',load_training_state=False)
    mem=m.native_question_memory; groups={}; rejects=Counter(); partition_hashes={}; parent_inventory={}
    for ri,(row,g) in enumerate(zip(rows,family)):
        if g in excluded:continue
        text=row['context']; source_hash=hashlib.sha256(text.encode()).hexdigest()
        parts=fixed_passages(text)
        # These offsets are fixed before accessing any question or answer.
        partition_hashes[source_hash]=hashlib.sha256(json.dumps([(p['start'],p['stop']) for p in parts]).encode()).hexdigest()
        parent_inventory[row['source_id']]=dict(group=g,sha256=source_hash,words=len(text.split()),passage_offsets=[(p['start'],p['stop']) for p in parts])
        for part in parts:
            if len(part['context'].split())<8:continue
            for question,answer in row['pairs']:
                if not answer or answer not in part['context'] or '\n' in answer or len(answer.split())>16:continue
                prompt=f"Context:\n{part['context']}\nQuestion:\n{question}"
                ids=t.prepare_generation_hierarchy(prompt).token_ids
                if len(ids)+len(CompleteAnswerSpanPredictor.payload(t,answer))>2048:
                    rejects['tokens']+=1;continue
                state=NativeQuestionState()
                for token in ids:state,_=mem._observe(state,token,t)
                question_state=NativeQuestionState()
                for token in t.encode('<BOI>Context:\n'+question+'<EOI>',add_special_tokens=False):question_state,_=mem._observe(question_state,token,t)
                if state.source_seen>512 or question_state.source_seen>32:
                    rejects['memory']+=1;continue
                gold=NativeQuestionState()
                for token in t.encode('<BOI>Context:\n'+answer+'<EOI>',add_special_tokens=False):gold,_=mem._observe(gold,token,t)
                source_words=[w for w,_ in state.source];target=[w for w,_ in gold.source]
                starts=[i for i in range(len(source_words)-len(target)+1) if source_words[i:i+len(target)]==target]
                if not any(c['text']==answer for c in mem.span_readout.candidates(state,t,starts)):
                    rejects['span']+=1;continue
                r=dict(context=part['context'],question=question,answer=answer,source_id=row['source_id'],source_group=g,
                       parent_sha256=source_hash,parent_offsets=[part['start'],part['stop']],partition_sha256=partition_hashes[source_hash])
                entry=groups.setdefault(g,dict(source_group=g,records=[]))
                if not any(qa_fingerprint(r)==qa_fingerprint(x) for x in entry['records']):entry['records'].append(r)
        if ri%250==0:print('PARTITION_PROGRESS',ri,len(groups),flush=True)
    # Prioritize two distinct answers within ONE input passage. This supplies a
    # question-dependence check without choosing a passage around either label.
    paired=[]
    for group in groups.values():
        by_context={}
        for r in group['records']:by_context.setdefault(r['context'],[]).append(r)
        eligible=[rs for rs in by_context.values() if len({r['answer'] for r in rs})>=2]
        if eligible:
            # Preserve the original inventory's stable pre-group record order.
            # regroup_passage_candidates below chooses the actual review
            # passage by text hash after closing source families.
            eligible.sort(key=lambda rs:len(rs[0]['context'].split()))
            first=eligible[0]
            group['records']=first+[r for r in group['records'] if r not in first]
            paired.append(group)
    paired.sort(key=lambda g:hashlib.sha256(('native-passages-v2-review:'+g['source_group']).encode()).hexdigest())
    result=dict(version=2,task='passage_grounded_extractive_qa',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        exclusion_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        provenance_limitations='Parent IDs/text/observed URLs are grouped conservatively; authoritative parent-page provenance is unavailable.',
        partition=dict(max_words=160,min_boundary_words=64,overlap=0,answer_blind=True,parent_grouping_before_partition=True),
        inventory=dict(fresh_groups_with_eligible_qa=len(groups),paired_groups=len(paired),eligible_questions=sum(len(g['records']) for g in groups.values()),rejected=dict(rejects)),
        documents=paired,parent_inventory=parent_inventory,code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('native_qa_transfer_data.py','prepare_native_qa_passages.py')})
    result=regroup_passage_candidates(result,old)
    (OUT/'candidates.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['inventory']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT,help='New output directory; existing candidates are never overwritten')
    args=parser.parse_args()
    OUT=args.output
    main()
