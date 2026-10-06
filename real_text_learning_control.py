"""Real-text construction and prompt-selection controls with a fixed representation."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import PrismalWaveConfig
from data import PrismalTokenizer, _build_window_samples_from_text
from model import PrismalWaveModel
from train import _infer_runtime_sizes_from_state, resolve_runtime_config, save_checkpoint
from tiny_learning_control import TRACKS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--answer-words", type=int, choices=(3,12), default=3)
    parser.add_argument("--records", type=int, choices=(8,16,64), default=8)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--full-budget", action="store_true")
    parser.add_argument("--lr-taper-from-step", type=int, default=0)
    parser.add_argument("--final-lr", type=float, default=0.0001)
    parser.add_argument("--signature-probe", action="store_true")
    parser.add_argument("--refit-structured", action="store_true")
    parser.add_argument("--word-profile-dropout", type=float, default=0.0)
    parser.add_argument("--line-profile-dropout", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("review_artifacts/real_text_control"))
    parser.add_argument("--save-final-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    assert 0.0 <= args.word_profile_dropout <= 1.0
    assert 0.0 <= args.line_profile_dropout <= 1.0
    assert args.lr_taper_from_step >= 0 and args.final_lr > 0.0
    assert not args.lr_taper_from_step or args.lr_taper_from_step < args.steps
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    root = Path("review_artifacts/nvidiapt_hqs_probe")
    payload = torch.load(root / "best/model.pt", map_location="cpu", weights_only=False)
    tokenizer = PrismalTokenizer.from_state_dict(payload["tokenizer_state"])
    tokenizer.refresh_construction_index()
    cfg = PrismalWaveConfig.from_dict(payload["config"])
    cfg = _infer_runtime_sizes_from_state(cfg,payload["model_state"],tokenizer=None)
    cfg.lr, cfg.batch_size, cfg.optimizer = 0.001, 8, "adamw"
    source_split = json.loads((root / "split.json").read_text(encoding="utf-8"))
    source_rows = source_split["train"]
    skipped_duplicate_prompts = []
    rows = [dict(record, answer=" ".join(record["answer"].split()[:args.answer_words])) for record in source_rows
            if len(tokenizer.prepare_generation_hierarchy(record["prompt"]).token_ids) == 25]
    assert len(rows) == 8 and len({r["answer"] for r in rows}) == 8
    if args.records >= 16:
        added = [dict(record,answer=" ".join(record["answer"].split()[:args.answer_words])) for record in source_rows
                 if len(tokenizer.prepare_generation_hierarchy(record["prompt"]).token_ids)==28][:8]
        assert len(added)==8
        rows.extend(added)
    held_rows = source_split["held_out"] if args.records>=16 else []
    if args.refit_structured:
        assert args.records >= 16 and args.answer_words == 12
        # Freeze source selection before the new vocabulary changes segmentation.
        rows = json.loads(Path("review_artifacts/real_text_control_16_records/manifest.json").read_text(encoding="utf-8"))["rows"]
        if args.records == 64:
            existing = {r["source_hash"] for r in rows}
            prompts = {r["prompt"] for r in rows}
            for r in source_rows:
                if len(rows) == 64:
                    break
                if r["source_hash"] in existing:
                    continue
                if r["prompt"] in prompts:
                    skipped_duplicate_prompts.append(dict(source_id=r["source_id"],prompt=r["prompt"]))
                    continue
                rows.append(r)
                prompts.add(r["prompt"])
        tokenizer = PrismalTokenizer()
        tokenizer.learn_from_texts(
            [f"<BOI>{r['prompt']}<EOI><BOO>{r['answer']}<EOO>" for r in source_rows],
            max_new_tokens=256, max_word_tokens=128, min_frequency=2)
        for name in ("vocab_size", "signature_vocab_size", "signature_level_vocab_size",
                     "signature_relation_vocab_size", "signature_bucket_vocab_size"):
            setattr(cfg, name, 0)
        cfg = resolve_runtime_config(cfg, tokenizer)
    count = len(rows)
    assert count==args.records and len({r["answer"] for r in rows})==count
    assert len({r["prompt"] for r in rows}) == count
    assert not ({r["source_hash"] for r in rows} & {r["source_hash"] for r in held_rows})
    manifest = dict(source="NVIDIAPT/HQS; prior record manifest", rows=rows,
                    skipped_duplicate_prompts=skipped_duplicate_prompts,
                    prompt_tokens=[len(tokenizer.prepare_generation_hierarchy(r["prompt"]).token_ids) for r in rows],
                    answer_words=args.answer_words,held_out=held_rows,
                    tokenizer="Fresh fitting on the original 96 training records" if args.refit_structured else "Frozen prior HQS tokenizer; no fitting or extension",
                    word_profile_dropout=args.word_profile_dropout,
                    line_profile_dropout=args.line_profile_dropout,
                    lr_taper_from_step=args.lr_taper_from_step,
                    final_lr=args.final_lr if args.lr_taper_from_step else None,
                    tokenizer_sizes=dict(tokens=tokenizer.vocab_size,signatures=tokenizer.signature_vocab_size,
                                         families=tokenizer.signature_family_vocab_size),
                    note="Conditional memorization control, not a held-out generalization test.")
    (args.output / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    torch.manual_seed(31)
    device = torch.device(args.device)
    # Config and vocabulary are unchanged; weights and optimizer are fresh.
    model = PrismalWaveModel(cfg).to(device)
    model.prepare_capacity_for_tokenizer(tokenizer)
    model.set_capacity_growth_locked(True)
    model._prismal_tokenizer = tokenizer
    def sample_of(record):
        windows = _build_window_samples_from_text(tokenizer,
            f"<BOI>{record['prompt']}<EOI><BOO>{record['answer']}<EOO>",seq_len=256,max_samples=1,
            hierarchy_vector_dtype="float32")
        assert len(windows)==1 and int(windows[0].labels[-1])==tokenizer.eos_id
        return windows[0]
    samples = [sample_of(r) for r in rows]
    held_samples = [sample_of(r) for r in held_rows]
    def batch_of(selected):
        return {name:torch.nn.utils.rnn.pad_sequence([getattr(s,name) for s in selected],
             batch_first=True).to(device)
             for name in ("input_ids","labels","loss_mask","hierarchy_vectors",*TRACKS)}
    batch = batch_of(samples)
    from word_profile_control import unavailable_prompt_profiles
    profile_sampler = random.Random(917)
    line_profile_sampler = random.Random(918)
    profile_dropout_count = 0
    profile_exposures = dict(available=0, word_only=0, line_only=0, both=0)
    per_record_profiles = [dict(available=0, word_only=0, line_only=0, both=0) for _ in rows]
    prefixes = [tokenizer.prepare_generation_hierarchy(r["prompt"]) for r in rows]
    max_new_tokens = max(48,max(s.input_ids.numel()+1-len(p.token_ids) for s,p in zip(samples,prefixes))+16)
    for s,prefix,record in zip(samples,prefixes,rows):
        prefix_len = len(prefix.token_ids)
        assert s.input_ids[:prefix_len].tolist() == prefix.token_ids
        for name, track in zip(TRACKS,prefix.as_tuple()[1:]):
            assert getattr(s,name)[:prefix_len].tolist() == track
        ids = tokenizer.encode(record["answer"],add_special_tokens=False)
        decoded = tokenizer.decode(ids,clean_text=False).strip()
        assert decoded == record["answer"], f"Answer tokenization changed text: {decoded!r} != {record['answer']!r}"
    special = set(tokenizer.special_tokens.values())-set(tokenizer._byte_fallback_ids.values())
    special_tensor = torch.tensor(sorted(special),device=device)
    space_id = tokenizer.special_tokens["<SPACE>"]
    space_frame = tokenizer.output_hierarchy_frame(
        [tokenizer.special_tokens["<BOO>"],tokenizer.special_tokens["<LINE>"]],space_id)
    neutral = dict(zip(TRACKS,space_frame))
    boi, eoi = tokenizer.special_tokens["<BOI>"],tokenizer.special_tokens["<EOI>"]
    line_level = tokenizer.signature_level_to_id["line"]
    line_relation = tokenizer.signature_relation_to_id["containment"]
    line_family = tokenizer.signature_family_to_id["line"]

    def neutralize(values):
        values = {name:tensor.clone() for name,tensor in values.items() if name != "hierarchy_vectors"}
        for row in range(values["input_ids"].size(0)):
            ids = values["input_ids"][row].tolist()
            left, right = ids.index(boi)+1, ids.index(eoi)
            for index in range(left,right):
                original = ids[index]
                if original in {tokenizer.special_tokens["<LINE>"],tokenizer.special_tokens["<EOL>"]}:
                    frame = dict(zip(TRACKS,(tokenizer.signature_blo_id,line_level,line_relation,
                                            tokenizer.signature_blo_id,line_family)))
                else:
                    values["input_ids"][row,index] = space_id
                    frame = neutral
                for name,value in frame.items():
                    values[name][row,index] = value
        return values

    erased_batch = neutralize(batch)
    # Token/vector reconstruction has the same fixed checkpoint normalization.
    normal_eval_batch = {name:value for name,value in batch.items() if name != "hierarchy_vectors"}
    groups = {}
    for index,prefix in enumerate(prefixes):
        groups.setdefault(len(prefix.token_ids),[]).append(index)
    for length,indices in groups.items():
        for name in ("input_ids",*TRACKS):
            assert torch.equal(erased_batch[name][indices,:length],
                               erased_batch[name][indices[0]:indices[0]+1,:length].expand(len(indices),-1))

    @torch.no_grad()
    def score(values):
        _,out = model.compute_loss(**values,collect_telemetry=False)
        mask = values["loss_mask"].gt(0) & values["labels"].ne(tokenizer.pad_id)
        content = mask & ~torch.isin(values["labels"],special_tensor)
        nll = F.cross_entropy(out.logits.transpose(1,2),values["labels"],reduction="none")
        return dict(ce=float(out.ce_loss),content_ce=float(nll[content].mean()),
            content_accuracy=float(out.logits.argmax(-1)[content].eq(values["labels"][content]).float().mean()),
            supervised_tokens=int(mask.sum()),content_tokens=int(content.sum()))

    def score_all(selected):
        metrics=[score({k:v for k,v in batch_of(selected[start:start+8]).items() if k!="hierarchy_vectors"})
                 for start in range(0,len(selected),8)]
        total=sum(m["supervised_tokens"] for m in metrics)
        content_total=sum(m["content_tokens"] for m in metrics)
        return dict(ce=sum(m["ce"]*m["supervised_tokens"] for m in metrics)/total,
            content_ce=sum(m["content_ce"]*m["content_tokens"] for m in metrics)/content_total,
            content_accuracy=sum(m["content_accuracy"]*m["content_tokens"] for m in metrics)/content_total,
            supervised_tokens=total,content_tokens=content_total)

    @torch.no_grad()
    def generate(prefix, erased=False):
        values = {"input_ids":torch.tensor([prefix.token_ids],device=device)}
        values.update({name:torch.tensor([track],device=device) for name,track in zip(TRACKS,prefix.as_tuple()[1:])})
        if erased:
            values = neutralize(values)
        else:
            values["hierarchy_vectors"] = torch.tensor([prefix.hierarchy_vectors],device=device)
        ids = model.generate(**values,max_new_tokens=max_new_tokens,min_new_tokens=0,
            temperature=0.0,repetition_penalty=1.0,no_repeat_ngram_size=0,beam_size=1,
            use_speculative_decoding=False,suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
            token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
            token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
            token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
            token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
        )[0,len(prefix.token_ids):].tolist()
        return tokenizer.decode(ids,clean_text=False).strip(),ids

    start = time.monotonic()
    evaluations, gradient_sums = [], {}
    validation_curve = []
    best_validation = None
    best_validation_state = None

    @torch.no_grad()
    def validate(step, held=None):
        nonlocal best_validation, best_validation_state
        if not args.validation_every or not held_samples:
            return
        model.eval()
        held = held if held is not None else score_all(held_samples)
        point = dict(step=step, exposure_per_record=step*8/count, **held)
        validation_curve.append(point)
        if best_validation is None or point["content_ce"] < best_validation["content_ce"]:
            best_validation = point
            best_validation_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        (args.output/"validation_curve.json").write_text(json.dumps(dict(points=validation_curve,best=best_validation),indent=2),encoding="utf-8")
        print(json.dumps(dict(validation=point,best_validation_step=best_validation["step"])),flush=True)

    @torch.no_grad()
    def finish_best_validation():
        if best_validation_state is None:
            return
        model.load_state_dict(best_validation_state)
        model.eval()
        examples=[]
        for r in held_rows[:3]:
            text,ids=generate(tokenizer.prepare_generation_hierarchy(r["prompt"]))
            examples.append(dict(prompt=r["prompt"],expected=r["answer"],generated=text,token_ids=ids))
        result=dict(selected=best_validation,train=score_all(samples),held_out=score_all(held_samples),
                    examples=examples,selection="Lowest held-out content CE at monitored updates; weights kept only in memory")
        (args.output/"best_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")

    def save_final_checkpoint():
        if args.save_final_checkpoint is None:
            return
        model.eval()
        final_eval = evaluations[-1]
        metrics = dict(step=final_eval["step"], exact=final_eval["exact"], records=count,
            train_content_ce=final_eval["matched"]["content_ce"],
            held_out_content_ce=(final_eval["held_out"]["content_ce"] if final_eval["held_out"] else None),
            seed=31, optimizer="adamw", initial_lr=0.001, weight_decay=0.01,
            lr_taper_from_step=args.lr_taper_from_step, final_lr=args.final_lr,
            word_profile_dropout=args.word_profile_dropout,
            line_profile_dropout=args.line_profile_dropout,
            exposure_per_record=final_eval["exposure_per_record"],
            torus_chunk_len=model.cfg.torus_chunk_len,
            signature_lattice_chunk_len=model.cfg.signature_lattice_chunk_len)
        path = save_checkpoint(model,args.save_final_checkpoint,tokenizer=tokenizer,
                               config=model.cfg,metrics=metrics)
        print(json.dumps(dict(final_checkpoint=str(path),metrics=metrics)),flush=True)
    @torch.no_grad()
    def evaluate(step):
        model.eval()
        matched, erased = score_all(samples), score(erased_batch)
        results = []
        for record,prefix in zip(rows,prefixes):
            text,ids = generate(prefix)
            results.append(dict(prompt=record["prompt"],expected=record["answer"],generated=text,
                exact=text==record["answer"],first_word_exact=text.split()[:1]==record["answer"].split()[:1],token_ids=ids))
        erased_results = []
        for length,indices in groups.items():
            text,ids=generate(prefixes[indices[0]],True)
            erased_results.append(dict(prompt_tokens=length,generated=text,token_ids=ids,
                exact=sum(text==rows[index]["answer"] for index in indices)))
        held_out = score_all(held_samples) if held_samples else None
        validate(step,held_out)
        held_generation=[]
        if held_rows and step in {0,args.steps}:
            for r in held_rows[:3]:
                text,ids=generate(tokenizer.prepare_generation_hierarchy(r["prompt"]))
                held_generation.append(dict(prompt=r["prompt"],expected=r["answer"],generated=text,token_ids=ids))
        record = dict(step=step,matched=matched,erased=erased,
            erased_minus_matched_content_ce=erased["content_ce"]-matched["content_ce"],
            exact=sum(r["exact"] for r in results),first_word_exact=sum(r["first_word_exact"] for r in results),
            erased_exact=sum(r["exact"] for r in erased_results),erased_results=erased_results,
            held_out=held_out,held_generation=held_generation,
            exposure_per_record=step*8/count,results=results,elapsed_seconds=time.monotonic()-start)
        evaluations.append(record)
        (args.output / "results.json").write_text(json.dumps(dict(
            manifest=manifest,config=model.cfg.to_dict(),seed=31,optimizer="adamw",lr=0.001,
            lr_taper_from_step=args.lr_taper_from_step,final_lr=args.final_lr if args.lr_taper_from_step else None,
            weight_decay=0.01,
            parameters=sum(p.numel() for p in model.parameters()),evaluations=evaluations,max_new_tokens=max_new_tokens,
            gradients=gradient_sums,device=str(device),
            unavailable_word_profile_exposures=profile_dropout_count,
            profile_exposures=profile_exposures,per_record_profiles=per_record_profiles,
            profile_sampler_seeds=dict(word=917,line=918),
            max_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else 0,
        ),ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({key:value for key,value in record.items() if key not in {"results","erased_results","held_generation"}},ensure_ascii=True),flush=True)
        return record

    evaluate(0)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    passes = 0
    order=list(range(count))
    sampler=random.Random(31)
    exposure_counts=[0]*count
    for step in range(1,args.steps+1):
        model.train()
        if args.lr_taper_from_step and step > args.lr_taper_from_step:
            progress = (step - args.lr_taper_from_step) / (args.steps - args.lr_taper_from_step)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
            current_lr = args.final_lr + (0.001 - args.final_lr) * cosine
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        if count==8:
            selected=list(range(8))
            train_batch=batch
        else:
            offset=((step-1)*8)%count
            if offset==0:
                sampler.shuffle(order)
            selected=order[offset:offset+8]
            train_batch=batch_of([samples[i] for i in selected])
        for index in selected:
            exposure_counts[index]+=1
        word_mask = [profile_sampler.random() < args.word_profile_dropout for _ in selected] if args.word_profile_dropout else [False]*len(selected)
        line_mask = [line_profile_sampler.random() < args.line_profile_dropout for _ in selected] if args.line_profile_dropout else [False]*len(selected)
        for index, word, line in zip(selected, word_mask, line_mask):
            mode = "both" if word and line else "word_only" if word else "line_only" if line else "available"
            profile_exposures[mode] += 1
            per_record_profiles[index][mode] += 1
        profile_dropout_count += sum(word_mask)
        if args.word_profile_dropout or args.line_profile_dropout:
            train_batch = unavailable_prompt_profiles(train_batch, tokenizer, word_mask, line_mask)
        loss,_ = model.compute_loss(**train_batch,collect_telemetry=False)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        for name,p in (("torus_write",model.torus_core.write_delta_proj.weight),
                       ("lattice_value",model.signature_lattice_attention.v_proj.weight)):
            gradient_sums[name] = float(p.grad.abs().sum()) if p.grad is not None else 0.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0:
            print(json.dumps(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),
                                  elapsed_seconds=time.monotonic()-start)),flush=True)
        if args.validation_every and step % args.validation_every == 0 and step % (50*count//8) != 0 and step != args.steps:
            validate(step)
        if step % (50*count//8) == 0 or step==args.steps:
            record = evaluate(step)
            passes = passes+1 if record["exact"]==count and record["matched"]["ce"]<0.1 else 0
            if passes>=2 and (count==8 or step>=400) and (not args.full_budget or step==args.steps):
                assert record["erased_exact"]<=len(groups)
                assert all(v>0 for v in gradient_sums.values())
                (args.output / "success.json").write_text(json.dumps(dict(success=True,step=step,
                    consecutive_passing_evaluations=passes,matched_exact=count,erased_exact=record["erased_exact"],
                    exposure_counts=exposure_counts,
                    no_saved_checkpoint=True,note="Real-text conditional memorization, not generalization."),indent=2),encoding="utf-8")
                print("PASS: real word construction and equal-length prompt-conditioned selection",flush=True)
                if args.signature_probe:
                    from signature_identity_probe import run_probe
                    run_probe(model,tokenizer,samples,rows,held_samples,held_rows,device,max_new_tokens,args.output)
                save_final_checkpoint()
                finish_best_validation()
                return 0
    print("INCOMPLETE: bounded control failed to meet both gates",flush=True)
    if args.signature_probe:
        from signature_identity_probe import run_probe
        run_probe(model,tokenizer,samples,rows,held_samples,held_rows,device,max_new_tokens,args.output)
    save_final_checkpoint()
    finish_best_validation()
    return 1


if __name__=="__main__":
    raise SystemExit(main())
