"""Record explicit passage reviews and freeze/validate the separate v2 dataset."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from native_qa_review import qa_fingerprint
from native_qa_transfer_data import validate_review, freeze_dataset

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'review_artifacts/native_qa_transfer_passages_v2/data'


def add_review_batch(batch):
    path=DATA/'candidates.json'
    candidates=json.loads(path.read_text())
    fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    review_path=DATA/'reviews.json'
    result=json.loads(review_path.read_text()) if review_path.exists() else dict(candidate_sha256=fingerprint,reviewer='assistant',decisions={},group_notes={})
    assert result['candidate_sha256']==fingerprint, 'Candidate file changed after review began'
    for group_index,entry in batch.items():
        document=candidates['documents'][int(group_index)]
        for action in ('keep','reject'):
            for record_index,reason in entry.get(action,{}).items():
                row=document['records'][int(record_index)]
                key=qa_fingerprint(row)
                pos=row['context'].index(row['answer'])
                review=dict(fingerprint=key,decision='retain' if action=='keep' else 'quarantine',reason=reason,
                    support=[0,len(row['context'])],answer_offsets=[pos,pos+len(row['answer'])],
                    group_index=int(group_index),record_index=int(record_index))
                validate_review(row,review)
                if key in result['decisions']:assert result['decisions'][key]==review, 'Review decision already recorded'
                result['decisions'][key]=review
        if entry.get('note'):result['group_notes'][str(group_index)]=entry['note']
    review_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    effective=materialized_reviews(candidates,result)
    approved=[];seen_families=set()
    for i,d in enumerate(candidates['documents']):
        if str(i) in result.get('exposed_groups',{}):continue
        rows=[r for r in d['records'] if effective.get(qa_fingerprint(r),{}).get('decision')=='retain']
        family=next((effective[qa_fingerprint(r)].get('related_family',d['source_group']) for r in rows),d['source_group'])
        if family not in seen_families and any(a['context']==b['context'] and a['answer']!=b['answer'] for j,a in enumerate(rows) for b in rows[j+1:]):
            approved.append(i);seen_families.add(family)
    print('REVIEW_PROGRESS',len(approved),'/128 groups',len(result['decisions']),'decisions')
    return approved


def add_family_link(indices, reason):
    path=DATA/'reviews.json';result=json.loads(path.read_text())
    result.setdefault('family_links',[]).append(dict(groups=indices,reason=reason))
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')


def exclude_exposed_group(index, reason):
    path=DATA/'reviews.json';result=json.loads(path.read_text())
    result.setdefault('exposed_groups',{})[str(index)]=reason
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')


def materialized_reviews(candidates, bundle):
    parents=list(range(len(candidates['documents'])))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for link in bundle.get('family_links',[]):
        for i in link['groups'][1:]:parents[find(i)]=find(link['groups'][0])
    families={}
    for i,document in enumerate(candidates['documents']):families.setdefault(find(i),[]).append(document['source_group'])
    result=json.loads(json.dumps(bundle['decisions']))
    excluded_roots={find(int(i)) for i in bundle.get('exposed_groups',{})}
    for i,document in enumerate(candidates['documents']):
        family=families[find(i)]
        if len(family)>1:
            alias=hashlib.sha256(('manual-family:'+':'.join(sorted(family))).encode()).hexdigest()
            for record in document['records']:
                key=qa_fingerprint(record)
                if key in result:result[key]['related_family']=alias
        if find(i) in excluded_roots:
            for record in document['records']:
                key=qa_fingerprint(record)
                if key in result:
                    result[key]['decision']='quarantine'
                    result[key]['reason']='Related source family was previously exposed; excluded from the fresh benchmark.'
    return result


def show(start,count):
    candidates=json.loads((DATA/'candidates.json').read_text())
    for i,d in enumerate(candidates['documents'][start:start+count],start):
        context=d['records'][0]['context']
        print('\nGROUP',i,'PARENT',d['records'][0]['source_id']);print(context)
        for j,row in enumerate(d['records']):
            if row['context']==context:print(j,'Q:',row['question'],'A:',row['answer'])


def freeze():
    c=DATA/'candidates.json';r=DATA/'reviews.json'
    candidates=json.loads(c.read_text());reviews=json.loads(r.read_text())
    assert reviews['candidate_sha256']==hashlib.sha256(c.read_bytes()).hexdigest()
    output=DATA/'frozen_manifest.json'
    candidates['manual_family_links']=reviews.get('family_links',[])
    manifest=freeze_dataset(candidates,materialized_reviews(candidates,reviews),output)
    print('FROZEN', {k:len(v) for k,v in manifest['splits'].items()})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('show','progress','freeze'));p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=16);a=p.parse_args()
    if a.phase=='show':show(a.start,a.count)
    elif a.phase=='progress':add_review_batch({})
    else:freeze()
