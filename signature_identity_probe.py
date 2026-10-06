"""Inference-only interventions on complete prompt signature identities."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from tiny_learning_control import TRACKS


@torch.no_grad()
def run_probe(model,tokenizer,samples,rows,held_samples,held_rows,device,max_new_tokens,output):
    model.eval()
    other=tokenizer.signature_special_ids["<OTHER>"]
    other_family=tokenizer.signature_family_id_by_signature_id[other]
    word_ids={idx for idx,code in tokenizer._signature_id_to_code.items()
              if code.startswith(("word|","code|")) and "|st=" in code}
    line_ids={idx for idx,code in tokenizer._signature_id_to_code.items() if code.startswith("line|")}
    special=set(tokenizer.special_tokens.values())-set(tokenizer._byte_fallback_ids.values())
    boi,eoi=tokenizer.special_tokens["<BOI>"],tokenizer.special_tokens["<EOI>"]
    prefixes=[tokenizer.prepare_generation_hierarchy(r["prompt"]) for r in rows]
    groups={}
    for index,prefix in enumerate(prefixes):
        groups.setdefault(len(prefix.token_ids),[]).append(index)
    donors={index:indices[(offset+1)%len(indices)] for indices in groups.values() for offset,index in enumerate(indices)}
    donor_line={index:next(int(sig) for token,sig in zip(prefixes[donor].token_ids,prefixes[donor].signature_ids)
                          if token==tokenizer.special_tokens["<LINE>"]) for index,donor in donors.items()}

    def batch_of(selected):
        return {name:torch.nn.utils.rnn.pad_sequence([getattr(s,name) for s in selected],batch_first=True).to(device)
                for name in ("input_ids","labels","loss_mask",*TRACKS)}

    def modify(values,mode,indices=None):
        result={name:value.clone() for name,value in values.items() if name!="hierarchy_vectors"}
        change=dict(signature_ids=0,parent_signature_ids=0,signature_family_ids=0)
        for row_index in range(result["input_ids"].size(0)):
            ids=result["input_ids"][row_index].tolist()
            left,right=ids.index(boi)+1,ids.index(eoi)
            selected=word_ids if mode=="word_fallback" else line_ids if mode in {"line_fallback","line_swap"} else word_ids|line_ids
            if mode=="normal":
                selected=set()
            replacement=donor_line[indices[row_index]] if mode=="line_swap" else other
            for track in ("signature_ids","parent_signature_ids"):
                for position in range(left,right):
                    if int(result[track][row_index,position]) in selected:
                        result[track][row_index,position]=replacement
                        if track=="signature_ids":
                            result["signature_family_ids"][row_index,position]=(
                                tokenizer.signature_family_id_by_signature_id[replacement] if mode=="line_swap" else other_family)
            for name in change:
                change[name]+=int(result[name][row_index].ne(values[name][row_index]).sum())
            for name in TRACKS:
                assert torch.equal(result[name][row_index,:left],values[name][row_index,:left])
                assert torch.equal(result[name][row_index,right:],values[name][row_index,right:])
        for name in ("input_ids","signature_level_ids","signature_relation_ids","labels","loss_mask"):
            if name in values:
                assert torch.equal(result[name],values[name]),name
        return result,change

    def metrics(selected,mode,training=True):
        total,content_total,loss,content_loss,correct,supervised_correct=0,0,0.,0.,0,0
        error_records=[]
        changes=dict(signature_ids=0,parent_signature_ids=0,signature_family_ids=0)
        for start in range(0,len(selected),8):
            original=batch_of(selected[start:start+8])
            values,change=modify(original,mode,list(range(start,min(start+8,len(selected)))) if training else None)
            for name in changes:
                changes[name]+=change[name]
            _,out=model.compute_loss(**values,collect_telemetry=False)
            nll=F.cross_entropy(out.logits.transpose(1,2),values["labels"],reduction="none")
            mask=values["loss_mask"].gt(0)&values["labels"].ne(tokenizer.pad_id)
            content=mask&~torch.isin(values["labels"],torch.tensor(sorted(special),device=device))
            total+=int(mask.sum());content_total+=int(content.sum())
            loss+=float(nll[mask].sum());content_loss+=float(nll[content].sum())
            correct+=int(out.logits.argmax(-1)[content].eq(values["labels"][content]).sum())
            predictions=out.logits.argmax(-1)
            supervised_correct+=int(predictions[mask].eq(values["labels"][mask]).sum())
            for row in range(mask.size(0)):
                errors=mask[row]&predictions[row].ne(values["labels"][row])
                if errors.any():
                    position=int(errors.nonzero()[0])
                    expected=int(values["labels"][row,position]);predicted=int(predictions[row,position])
                    error_records.append(dict(record_index=start+row,supervised_errors=int(errors.sum()),
                                              content_errors=int((errors&content[row]).sum()),
                                              first_position=position,expected_token=expected,predicted_token=predicted,
                                              expected_unit=tokenizer.construction_units[expected].text,
                                              predicted_unit=tokenizer.construction_units[predicted].text))
        return dict(ce=loss/total,content_ce=content_loss/content_total,content_accuracy=correct/content_total,
                    supervised_accuracy=supervised_correct/total,teacher_error_records=error_records,
                    changed_fields=changes,content_tokens=content_total)

    def generate(prefix,mode,index):
        values={"input_ids":torch.tensor([prefix.token_ids],device=device)}
        values.update({name:torch.tensor([track],device=device) for name,track in zip(TRACKS,prefix.as_tuple()[1:])})
        values,changes=modify(values,mode,[index])
        ids=model.generate(**values,max_new_tokens=max_new_tokens,min_new_tokens=0,temperature=0.,
            repetition_penalty=1.,no_repeat_ngram_size=0,beam_size=1,use_speculative_decoding=False,
            suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
            token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
            token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
            token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
            token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
        )[0,len(prefix.token_ids):].tolist()
        return tokenizer.decode(ids,clean_text=False).strip(),ids,sum(changes.values()) > 0

    def coverage(selected):
        counts=dict(line_frames=0,line_fallback_frames=0,parents=0,other_parents=0,known_word_parents=0,known_line_parents=0)
        for sample in selected:
            ids=sample.input_ids.tolist();left,right=ids.index(boi)+1,ids.index(eoi)
            for position in range(left,right):
                sig,parent=int(sample.signature_ids[position]),int(sample.parent_signature_ids[position])
                counts["parents"]+=1;counts["other_parents"]+=int(parent==other)
                counts["known_word_parents"]+=int(parent in word_ids)
                counts["known_line_parents"]+=int(parent in line_ids)
                if ids[position] in {tokenizer.special_tokens["<LINE>"],tokenizer.special_tokens["<EOL>"]}:
                    counts["line_frames"]+=1;counts["line_fallback_frames"]+=int(sig==other)
        return counts

    results=dict(protocol="Complete prompt word/line IDs replaced; text, levels, relations, positions and answer fields fixed",
                 coverage=dict(train=coverage(samples),held_out=coverage(held_samples)),conditions={})
    for mode in ("normal","word_fallback","line_fallback","both_fallback","line_swap"):
        train=metrics(samples,mode)
        held=metrics(held_samples,mode,False) if mode!="line_swap" else None
        outputs=[]
        for index,(row,prefix) in enumerate(zip(rows,prefixes)):
            text,ids,active=generate(prefix,mode,index)
            outputs.append(dict(prompt=row["prompt"],expected=row["answer"],generated=text,token_ids=ids,
                exact=text==row["answer"],donor_index=donors[index] if mode=="line_swap" else None,
                active_intervention=active,
                donor_exact=text==rows[donors[index]]["answer"] if mode=="line_swap" and active and donors[index]!=index else None))
        record=dict(train=train,held_out=held,exact=sum(r["exact"] for r in outputs),
                    donor_exact=sum(bool(r["donor_exact"]) for r in outputs),outputs=outputs,
                    active_examples=sum(r["active_intervention"] for r in outputs),
                    active_intervention=sum(train["changed_fields"].values()) > 0)
        results["conditions"][mode]=record
        (Path(output)/"signature_identity_probe.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(dict(signature_probe=mode,train=train,held_out=held,exact=record["exact"],donor_exact=record["donor_exact"])),flush=True)
    results["baseline_all_exact"] = results["conditions"]["normal"]["exact"] == len(rows)
    (Path(output)/"signature_identity_probe.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    return results
