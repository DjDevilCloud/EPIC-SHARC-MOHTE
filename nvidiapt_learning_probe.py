"""Small record-disjoint NVIDIA text-continuation learning experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from config import PrismalWaveConfig
from data import PrismalTokenizer, _build_window_samples_from_text
from model import PrismalWaveModel
from train import resolve_runtime_config, save_checkpoint, load_bundle_from_checkpoint
from tiny_learning_control import TRACKS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=Path(r"D:\AAISourceStart\SourceALLM\Datasets\NVIDIAPT\HQS\part_000000.parquet"))
    parser.add_argument("--output", type=Path, default=Path("review_artifacts/nvidiapt_hqs_probe"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(23)
    torch.set_num_threads(1)
    rng = random.Random(23)
    device = torch.device("cuda")
    records = pq.read_table(args.source, columns=["id", "text"]).to_pylist()
    candidates, seen = [], set()
    for record in records:
        text = str(record["text"]).strip()
        # Keep a short natural-text prefix and its actual following words.
        words = text.split()
        if len(words) < 24 or any(marker in text[:200] for marker in ("```", "<extra_id_", "\\[", "$$")):
            continue
        if sum(ch.isalpha() for ch in text[:150]) < 80:
            continue
        prefix = " ".join(words[:5])
        answer = " ".join(words[5:17])
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        full_hash = hashlib.sha256(normalized.encode()).hexdigest()
        pair_hash = hashlib.sha256((prefix + " " + answer).lower().encode()).hexdigest()
        if full_hash in seen or pair_hash in seen:
            continue
        seen.update((full_hash, pair_hash))
        candidates.append(dict(source_id=record["id"], source_hash=full_hash,
                               pair_hash=pair_hash, prompt=prefix, answer=answer))
    rng.shuffle(candidates)
    selected = candidates[:120]
    if len(selected) < 120:
        raise RuntimeError(f"Not enough usable source records: {len(selected)}")
    train_records, val_records = selected[:96], selected[96:]
    assert not ({r['source_hash'] for r in train_records} & {r['source_hash'] for r in val_records})
    assert not ({r['pair_hash'] for r in train_records} & {r['pair_hash'] for r in val_records})
    manifest = dict(source=str(args.source), task="Five source words as prompt, next twelve words as continuation",
                    seed=23, train=train_records, held_out=val_records,
                    notes="Whitespace normalized; no paraphrase or fabricated targets. Split before tokenizer fitting.")
    (args.output / "split.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    text_of = lambda r: f"<BOI>{r['prompt']}<EOI><BOO>{r['answer']}<EOO>"
    tokenizer = PrismalTokenizer()
    tokenizer.learn_from_texts([text_of(r) for r in train_records], max_new_tokens=256,
                              max_word_tokens=128, min_frequency=2)
    cfg = PrismalWaveConfig()
    overrides = dict(
        d_model=64, n_emitters=16, n_slots=16, top_k_emitters=4, top_k_slots=2,
        ff_mult=2, factorized_embedding_dim=32, torus_depth=2, torus_height=2,
        torus_width=2, torus_chunk_len=1, signature_lattice_chunk_len=1,
        signature_lattice_dim=32, hierarchical_precision_enabled=False,
        hierarchical_precision_accumulator_dtype="float32", hierarchy_vector_dtype="float32",
        use_turbo_quantization=False, use_torchao_weight_only=False,
        use_torchao_embedding_weight_only=False, use_gate=False, use_gatetrain=False,
        use_fullgatetrain=False, use_learned_residency_head=False,
        use_residency_with_reinforcement=False, use_contrastive_routing=False,
        use_signature_lattice_attention=True, use_token_memory_cross_attention=False,
        use_hmote=False, hierarchical_nest_depth=1, use_recursive_hmoe=False,
        use_fixed_point_solver=False, dropout=0.0, path_noise=0.0,
        disable_auxlosses=True, position_embedding_init_size=256, use_fst=False,
        use_torus_race_lanes=False, use_speculative_decoding=False,
        use_token_superposition_training=False,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    model = PrismalWaveModel(resolve_runtime_config(cfg, tokenizer)).to(device)
    model.cfg.optimizer = "adamw"
    model.cfg.lr = 0.001
    model._prismal_tokenizer = tokenizer
    def make_samples(rows):
        samples = []
        for record in rows:
            windows = _build_window_samples_from_text(tokenizer, text_of(record), seq_len=512,
                        max_samples=1, hierarchy_vector_dtype="float32")
            assert len(windows) == 1
            sample = windows[0]
            assert sample.input_ids.numel() < 256, "Record exceeds bounded full-context probe length"
            prompt = tokenizer.prepare_generation_hierarchy(record["prompt"])
            assert sample.input_ids[:len(prompt.token_ids)].tolist() == prompt.token_ids
            for name, track in zip(TRACKS, prompt.as_tuple()[1:]):
                assert getattr(sample, name)[:len(track)].tolist() == track
            samples.append(sample)
        return samples
    train_samples, val_samples = make_samples(train_records), make_samples(val_records)
    def batch_of(samples):
        return {name: torch.nn.utils.rnn.pad_sequence([getattr(s, name) for s in samples],
                    batch_first=True).to(device)
                for name in ("input_ids", "labels", "loss_mask", "hierarchy_vectors", *TRACKS)}
    structure = set(tokenizer.special_tokens.values())
    structure.difference_update(tokenizer._byte_fallback_ids.values())
    content_mask = lambda batch: batch["loss_mask"].gt(0) & batch["labels"].ne(tokenizer.pad_id) & ~torch.isin(batch["labels"], torch.tensor(sorted(structure), device=device))
    counts = Counter()
    for sample in train_samples:
        counts.update(int(label) for label, mask in zip(sample.labels, sample.loss_mask) if float(mask) > 0)
    probs = torch.ones(tokenizer.vocab_size, device=device)
    for token, count in counts.items():
        probs[token] += count
    unigram_nll = -(probs / probs.sum()).log()
    start, evaluations = time.monotonic(), []
    best_ce, best_state, best_step = float("inf"), None, 0

    @torch.no_grad()
    def metrics(samples):
        sums = dict(loss=0.0, count=0, content_loss=0.0, content_count=0,
                    correct=0, unigram_loss=0.0, unigram_content_loss=0.0)
        for offset in range(0, len(samples), 8):
            batch = batch_of(samples[offset:offset+8])
            _, output = model.compute_loss(**batch, collect_telemetry=False)
            nll = F.cross_entropy(output.logits.transpose(1,2), batch["labels"], reduction="none")
            mask = batch["loss_mask"].gt(0) & batch["labels"].ne(tokenizer.pad_id)
            content = content_mask(batch)
            sums["loss"] += float(nll[mask].sum())
            sums["content_loss"] += float(nll[content].sum())
            sums["count"] += int(mask.sum())
            sums["content_count"] += int(content.sum())
            sums["correct"] += int(output.logits.argmax(-1)[content].eq(batch["labels"][content]).sum())
            sums["unigram_loss"] += float(unigram_nll[batch["labels"]][mask].sum())
            sums["unigram_content_loss"] += float(unigram_nll[batch["labels"]][content].sum())
        return dict(ce=sums["loss"]/sums["count"], content_ce=sums["content_loss"]/sums["content_count"],
                    content_accuracy=sums["correct"]/sums["content_count"],
                    unigram_ce=sums["unigram_loss"]/sums["count"],
                    unigram_content_ce=sums["unigram_content_loss"]/sums["content_count"],
                    supervised_tokens=sums["count"], content_tokens=sums["content_count"])

    @torch.no_grad()
    def continuations(rows):
        results = []
        for record in rows[:3]:
            bundle = tokenizer.prepare_generation_hierarchy(record["prompt"])
            metadata = {name: torch.tensor([track], device=device) for name, track in zip(TRACKS, bundle.as_tuple()[1:])}
            ids = model.generate(torch.tensor([bundle.token_ids], device=device), **metadata,
                hierarchy_vectors=torch.tensor([bundle.hierarchy_vectors], device=device),
                max_new_tokens=48, min_new_tokens=0, temperature=0.0, repetition_penalty=1.0,
                no_repeat_ngram_size=0, beam_size=1, use_speculative_decoding=False,
                suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
                token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
                token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
                token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
                token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
            )[0, len(bundle.token_ids):].tolist()
            text = tokenizer.decode(ids, clean_text=False).strip()
            results.append(dict(prompt=record["prompt"], reference=record["answer"], generated=text,
                                cleaned_generated=tokenizer.decode(ids).strip(), token_ids=ids))
        return results

    def evaluate(step):
        nonlocal best_ce, best_state, best_step
        model.eval()
        result = dict(step=step, train=metrics(train_samples), held_out=metrics(val_samples),
                      train_generation=continuations(train_records), held_out_generation=continuations(val_records),
                      elapsed_seconds=time.monotonic()-start)
        evaluations.append(result)
        if step > 0 and result["held_out"]["content_ce"] < best_ce:
            best_ce, best_step = result["held_out"]["content_ce"], step
            best_state = {name: tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}
        print(json.dumps(result, ensure_ascii=True), flush=True)
        (args.output / "results.json").write_text(json.dumps(dict(
            source=str(args.source), task=manifest["task"], seed=23, training_records=96, held_out_records=24,
            optimizer="adamw", learning_rate=0.001, weight_decay=0.01, grad_clip=1.0,
            config=model.cfg.to_dict(), parameters=sum(p.numel() for p in model.parameters()),
            lengths=dict(train=[s.input_ids.numel() for s in train_samples], held_out=[s.input_ids.numel() for s in val_samples]),
            evaluations=evaluations, best_step=best_step, best_held_out_content_ce=best_ce,
            max_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20,
        ), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    evaluate(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    for step in range(1, args.steps+1):
        model.train()
        batch = batch_of(rng.sample(train_samples, args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        loss,_ = model.compute_loss(**batch, collect_telemetry=False)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0:
            print(json.dumps(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
                                  elapsed_seconds=time.monotonic()-start)), flush=True)
        if step % 50 == 0 or step == args.steps:
            evaluate(step)
    model.load_state_dict(best_state)
    model.eval()
    best = metrics(val_samples)
    # A single checkpoint for this bounded experiment; selection uses held-out content CE.
    checkpoint = save_checkpoint(model, args.output / "best", tokenizer=tokenizer, metrics=dict(
        best_step=best_step, held_out_ce=best["ce"], held_out_content_ce=best["content_ce"]))
    restored, restored_tokenizer, _ = load_bundle_from_checkpoint(checkpoint, device=device)
    probe_batch = batch_of(val_samples[:1])
    with torch.no_grad():
        _, original = model.compute_loss(**probe_batch, collect_telemetry=False)
        restored.eval()
        _, loaded = restored.compute_loss(**probe_batch, collect_telemetry=False)
        torch.testing.assert_close(original.logits, loaded.logits, rtol=1e-4, atol=1e-5)
    summary = dict(best_step=best_step, best_held_out=best, checkpoint=str(checkpoint),
                   checkpoint_round_trip=True,
                   below_unigram=best["content_ce"] < best["unigram_content_ce"],
                   improved_from_initial=best["content_ce"] < evaluations[0]["held_out"]["content_ce"],
                   note="Held-out records used for checkpoint selection; no independent final test split.")
    (args.output / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
