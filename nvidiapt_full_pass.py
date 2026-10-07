"""Fresh-weight, one-epoch pass across every NVIDIAPT sample category."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch

from config import PrismalWaveConfig
from data import (PrismalTokenizer, StreamingTextCorpusDataset, _build_window_samples_from_text,
                  iter_text_corpus)
from model import PrismalWaveModel
from train import (build_train_val_dataloaders, generate_text, load_bundle_from_checkpoint,
                   resolve_runtime_config, save_checkpoint, train_model)


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "review_artifacts" / "nvidiapt_full_pass"
SOURCE = RUN / "all_categories_shuffled.jsonl"
TEMPLATE_CHECKPOINT = ROOT / "checkpoints" / "nvidia_tiny_prototype" / "run16_cosine_taper_64" / "model.pt"
SAVE_DIR = ROOT / "checkpoints" / "nvidia_tiny_prototype" / "full_nvidiapt_all_categories"
TRACKS = ("signature_ids", "signature_level_ids", "signature_relation_ids",
          "parent_signature_ids", "signature_family_ids")
SEED = 31
VAL_FRACTION = 0.10
BATCH_SIZE = max(1, int(os.environ.get("NVIDIAPT_BATCH_SIZE", "32") or 32))
MAX_NEW_TOKENS = 8192  # tokenizer construction-unit cap
MAX_GENERATION_TOKENS = 24
MAX_WORD_TOKENS = 4096
SEQ_LEN = 0
WINDOW_STRIDE = 256
INCLUDE_STRUCTURE_STARTS = False


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIAPT full pass requires the available RTX 4070 SUPER CUDA device.")
    device = torch.device("cuda")
    template = torch.load(TEMPLATE_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = PrismalWaveConfig.from_dict(template["config"])
    # Preserve the already-tested model and one-token hierarchy contract while
    # fitting every lexical/signature capacity from this run's training split.
    for name in ("vocab_size", "signature_vocab_size", "signature_level_vocab_size",
                 "signature_relation_vocab_size", "signature_bucket_vocab_size"):
        setattr(cfg, name, 0)
    cfg.hierarchy_vector_normalization = {}
    cfg.optimizer = "adamw"
    cfg.lr = 0.001
    cfg.max_samples = 0
    cfg.use_gradient_accumulation = False
    cfg.gradient_accumulation_steps = 1
    if os.environ.get("NVIDIAPT_BF16", "0").strip().lower() in {"1", "true", "yes"}:
        # The verified tiny-control template intentionally used FP32. Allow a
        # separately measured CUDA BF16 profile without changing that baseline.
        cfg.hierarchical_precision_enabled = True
        cfg.hierarchical_precision_root_dtype = "bfloat16"
        cfg.hierarchical_precision_mid_dtype = "bfloat16"
        cfg.hierarchical_precision_leaf_dtype = "bfloat16"
        cfg.hierarchical_precision_fallback_dtype = "bfloat16"
        cfg.hierarchical_precision_accumulator_dtype = "bfloat16"
    assert cfg.torus_chunk_len == cfg.signature_lattice_chunk_len == 1

    tokenizer = PrismalTokenizer(
        use_pronunciation_signatures=cfg.use_pronunciation_signatures,
        hierarchy_vector_low_rank_enabled=cfg.hierarchy_vector_low_rank_enabled,
        hierarchy_vector_low_rank_dim=cfg.hierarchy_vector_low_rank_dim,
    )
    split_reference = StreamingTextCorpusDataset(
        SOURCE, tokenizer, seq_len=0, split="train", val_fraction=VAL_FRACTION, seed=SEED,
        hierarchy_vector_dtype=cfg.hierarchy_vector_dtype,
    )
    print(json.dumps(dict(stage="tokenizer_fit_start", source=str(SOURCE), seed=SEED)), flush=True)
    tokenizer.learn_from_texts(
        (text for index, text in enumerate(iter_text_corpus(SOURCE))
         if split_reference._include_record(index)),
        max_new_tokens=MAX_NEW_TOKENS, min_frequency=2,
        max_word_tokens=MAX_WORD_TOKENS, max_line_tokens=0, max_signature_tokens=0,
    )
    tokenizer.refresh_construction_index()
    tokenizer_sizes = dict(tokens=tokenizer.vocab_size, signatures=tokenizer.signature_vocab_size,
                           levels=tokenizer.signature_level_vocab_size,
                           relations=tokenizer.signature_relation_vocab_size,
                           families=tokenizer.signature_family_vocab_size,
                           hierarchy_normalization=tokenizer.hierarchy_normalization_capacities)
    print(json.dumps(dict(stage="tokenizer_fit_done", **tokenizer_sizes)), flush=True)

    train_ref = StreamingTextCorpusDataset(
        SOURCE, tokenizer, seq_len=0, split="train", val_fraction=VAL_FRACTION, seed=SEED,
        hierarchy_vector_dtype=cfg.hierarchy_vector_dtype,
    )
    val_ref = StreamingTextCorpusDataset(
        SOURCE, tokenizer, seq_len=0, split="val", val_fraction=VAL_FRACTION, seed=SEED,
        hierarchy_vector_dtype=cfg.hierarchy_vector_dtype,
    )
    window_counts = Counter()
    record_counts = Counter()
    category_counts: dict[str, Counter[str]] = {}
    count_started = time.monotonic()
    print(json.dumps(dict(stage="window_count_start")), flush=True)
    with SOURCE.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            category = str(record["category"])
            text = str(record["text"])
            split_name = "val" if val_ref._include_record(index) else "train"
            record_counts[split_name] += 1
            category_counts.setdefault(category, Counter())[split_name] += 1
            windows = _build_window_samples_from_text(
                tokenizer, text, seq_len=SEQ_LEN, max_samples=0,
                window_stride=WINDOW_STRIDE,
                include_structure_starts=INCLUDE_STRUCTURE_STARTS,
                hierarchy_vector_dtype=cfg.hierarchy_vector_dtype,
            )
            window_counts[split_name] += len(windows)
            category_counts[category][f"{split_name}_windows"] += len(windows)
            if (index + 1) % 1000 == 0:
                print(json.dumps(dict(stage="window_count", records=index + 1,
                    train_windows=window_counts["train"], val_windows=window_counts["val"],
                    elapsed_seconds=time.monotonic() - count_started)), flush=True)
    train_windows, val_windows = window_counts["train"], window_counts["val"]
    if not train_windows or not val_windows:
        raise RuntimeError(f"Invalid split sizes: {dict(window_counts)}")
    planned_steps = math.ceil(train_windows / BATCH_SIZE)
    requested_step_cap = max(0, int(os.environ.get("NVIDIAPT_MAX_STEPS", "0") or 0))
    run_steps = min(planned_steps, requested_step_cap) if requested_step_cap else planned_steps
    (RUN / "training_manifest.json").write_text(json.dumps(dict(
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        records=sum(record_counts.values()), record_counts=dict(record_counts),
        window_counts=dict(window_counts), planned_optimizer_batches=planned_steps,
        category_counts={key: dict(value) for key, value in category_counts.items()},
        seed=SEED, validation_fraction=VAL_FRACTION, tokenizer_sizes=tokenizer_sizes,
        generation_max_new_tokens=MAX_GENERATION_TOKENS,
        precision=dict(enabled=cfg.hierarchical_precision_enabled,
                       root=cfg.hierarchical_precision_root_dtype,
                       mid=cfg.hierarchical_precision_mid_dtype,
                       leaf=cfg.hierarchical_precision_leaf_dtype),
        max_new_tokens=MAX_NEW_TOKENS, max_word_tokens=MAX_WORD_TOKENS,
        sequence_window=dict(chunk_tokens=256, stride_tokens=WINDOW_STRIDE,
                             extra_structure_starts=INCLUDE_STRUCTURE_STARTS,
                             note="Full token coverage with long-answer anchor windows; hierarchy fields preserved."),
        model_template=str(TEMPLATE_CHECKPOINT), fresh_model_weights=True,
        torus_chunk_len=cfg.torus_chunk_len,
        signature_lattice_chunk_len=cfg.signature_lattice_chunk_len,
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(stage="window_count_done", train_windows=train_windows,
        val_windows=val_windows, planned_batches=planned_steps,
        elapsed_seconds=time.monotonic() - count_started)), flush=True)

    runtime_cfg = resolve_runtime_config(cfg, tokenizer)
    model = PrismalWaveModel(runtime_cfg).to(device)
    model.prepare_capacity_for_tokenizer(tokenizer)
    model.set_capacity_growth_locked(True)
    model._prismal_tokenizer = tokenizer
    train_loader, val_loader = build_train_val_dataloaders(
        SOURCE, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        max_samples=0, val_max_samples=0, val_fraction=VAL_FRACTION,
        seed=SEED, streaming=True, window_stride=WINDOW_STRIDE,
        include_structure_starts=INCLUDE_STRUCTURE_STARTS,
        hierarchy_vector_dtype=cfg.hierarchy_vector_dtype,
    )
    print(json.dumps(dict(stage="training_start", device=torch.cuda.get_device_name(device),
        architecture=dict(d_model=runtime_cfg.d_model, n_layers=runtime_cfg.n_layers,
                          torus_chunk_len=runtime_cfg.torus_chunk_len,
                          signature_lattice_chunk_len=runtime_cfg.signature_lattice_chunk_len),
        batch_size=BATCH_SIZE, planned_batches=planned_steps, run_batches=run_steps,
        bounded_profile=run_steps < planned_steps)), flush=True)
    start = time.monotonic()
    metrics = train_model(
        model, train_loader, device, cfg=runtime_cfg, optimizer_name="adamw",
        epochs=1, steps=run_steps, lr=0.001, grad_clip=1.0,
        progress=True, val_loader=val_loader if run_steps >= planned_steps else None,
        diagnostic_interval=100, use_amp=True,
    )
    elapsed = time.monotonic() - start
    actual_steps = int(metrics["steps"])
    if actual_steps != run_steps:
        raise RuntimeError(f"Expected {run_steps} training batches, got {actual_steps}.")
    metrics.update(dict(
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        record_counts=dict(record_counts), window_counts=dict(window_counts),
        category_counts={key: dict(value) for key, value in category_counts.items()},
        planned_batches=planned_steps, fresh_weights=True, seed=SEED,
        tokenizer_sizes=tokenizer_sizes, device=torch.cuda.get_device_name(device),
        max_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
        torus_chunk_len=runtime_cfg.torus_chunk_len,
        signature_lattice_chunk_len=runtime_cfg.signature_lattice_chunk_len,
        elapsed_full_pass_seconds=elapsed,
    ))
    if run_steps < planned_steps:
        print(json.dumps(dict(stage="bounded_profile_done", completed_batches=actual_steps,
            planned_batches=planned_steps, elapsed_seconds=elapsed,
            final_ce_loss=metrics.get("final_ce_loss"),
            max_memory_allocated_bytes=metrics.get("max_memory_allocated_bytes"))), flush=True)
        return
    model.eval()
    checkpoint = save_checkpoint(model, SAVE_DIR, tokenizer=tokenizer,
                                 config=runtime_cfg, metrics=metrics)
    print(json.dumps(dict(stage="checkpoint_saved", path=str(checkpoint), metrics={
        key: metrics[key] for key in ("steps", "epochs", "final_ce_loss", "avg_ce_loss",
            "val_ce_loss", "best_val_loss", "elapsed_full_pass_seconds",
            "max_memory_allocated_bytes", "param_count") if key in metrics
    })), flush=True)

    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    loaded, loaded_tokenizer, loaded_cfg = load_bundle_from_checkpoint(
        checkpoint, device=device, load_training_state=False)
    loaded.set_capacity_growth_locked(True)
    assert loaded_cfg.torus_chunk_len == loaded_cfg.signature_lattice_chunk_len == 1
    examples: dict[str, dict[str, str]] = {}
    with SOURCE.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            category = str(record["category"])
            if category in examples:
                continue
            text = str(record["text"])
            if "<BOO>" in text:
                prompt = text.split("<BOO>", 1)[0] + "<BOO>"
            elif "output:" in text.lower():
                offset = text.lower().find("output:") + len("output:")
                prompt = text[:offset]
            else:
                prompt = text[:160]
            examples[category] = dict(prompt=prompt, expected_record_prefix=text[:240])
            if len(examples) == len(category_counts):
                break
    generations = []
    for category, item in sorted(examples.items()):
        generated = generate_text(
            loaded, loaded_tokenizer, item["prompt"], device, max_new_tokens=MAX_GENERATION_TOKENS,
            min_new_tokens=0, top_k=1, top_p=1.0, temperature=0.0,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        generations.append(dict(category=category, prompt=item["prompt"],
                                generated=generated, expected_record_prefix=item["expected_record_prefix"]))
    result = dict(checkpoint=str(checkpoint), fresh_load=True,
        categories=sorted(category_counts), record_counts=dict(record_counts),
        window_counts=dict(window_counts), generations=generations,
        exact_one_epoch=actual_steps == planned_steps,
        torus_chunk_len=loaded_cfg.torus_chunk_len,
        signature_lattice_chunk_len=loaded_cfg.signature_lattice_chunk_len,
        training_metrics=metrics)
    (RUN / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(stage="complete", checkpoint=str(checkpoint), categories=len(category_counts),
        train_records=record_counts["train"], val_records=record_counts["val"],
        train_windows=train_windows, val_windows=val_windows, full_pass_steps=actual_steps,
        train_ce=metrics.get("final_ce_loss"), validation_ce=metrics.get("val_ce_loss"),
        generated_categories=len(generations), torus_chunk_len=loaded_cfg.torus_chunk_len,
        signature_lattice_chunk_len=loaded_cfg.signature_lattice_chunk_len)), flush=True)


if __name__ == "__main__":
    main()
