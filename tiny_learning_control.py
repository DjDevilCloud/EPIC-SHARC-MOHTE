"""Bounded fresh learning control for the causal hierarchy and torus/lattice."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from config import PrismalWaveConfig
from data import PrismalTokenizer, _build_window_samples_from_text
from model import PrismalWaveModel
from train import resolve_runtime_config


PAIRS = [
    ("red?", "qzxalpha."),
    ("blue?", "qzxbeta."),
    ("lines?", "one.\ntwo."),
    ("mark?", "café 🔍."),
]
TRACKS = ("signature_ids", "signature_level_ids", "signature_relation_ids",
          "parent_signature_ids", "signature_family_ids")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("review_artifacts/tiny_learning_control"))
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    tokenizer = PrismalTokenizer()
    texts = [f"<BOI>{prompt}<EOI><BOO>{answer}<EOO>" for prompt, answer in PAIRS]
    # Force rare shared-prefix words through construction units, not a whole-answer token.
    tokenizer.learn_from_texts(texts, max_new_tokens=1, min_frequency=1, max_word_tokens=1)
    for _, answer in PAIRS:
        encoded_answer = tokenizer.encode(answer, add_special_tokens=False)
        assert tokenizer.decode(encoded_answer).strip() == answer, "Control answer does not survive tokenization"
    cfg = PrismalWaveConfig()
    overrides = dict(
        d_model=32, n_emitters=8, n_slots=8, top_k_emitters=2, top_k_slots=2,
        ff_mult=2, factorized_embedding_dim=16, torus_depth=2, torus_height=2,
        torus_width=2, torus_chunk_len=1, signature_lattice_chunk_len=1,
        signature_lattice_dim=16, hierarchical_precision_enabled=False,
        hierarchical_precision_accumulator_dtype="float32", hierarchy_vector_dtype="float32",
        use_turbo_quantization=False, use_torchao_weight_only=False,
        use_torchao_embedding_weight_only=False, use_gate=False, use_gatetrain=False,
        use_fullgatetrain=False, use_learned_residency_head=False,
        use_residency_with_reinforcement=False, use_contrastive_routing=False,
        use_signature_lattice_attention=True, use_token_memory_cross_attention=False,
        use_hmote=False, hierarchical_nest_depth=1, use_recursive_hmoe=False,
        use_fixed_point_solver=False, dropout=0.0, path_noise=0.0,
        disable_auxlosses=True, position_embedding_init_size=128, use_fst=False,
        use_torus_race_lanes=False, use_speculative_decoding=False,
        use_token_superposition_training=False,
    )
    for name, value in overrides.items():
        if not hasattr(cfg, name):
            raise ValueError(f"Unknown configuration field: {name}")
        setattr(cfg, name, value)
    model = PrismalWaveModel(resolve_runtime_config(cfg, tokenizer)).to(device)
    model._prismal_tokenizer = tokenizer
    samples = [_build_window_samples_from_text(tokenizer, text, seq_len=128,
                hierarchy_vector_dtype="float32")[0] for text in texts]
    length = max(int(sample.input_ids.ne(tokenizer.pad_id).sum()) for sample in samples)
    batch = {name: torch.nn.utils.rnn.pad_sequence(
                [getattr(sample, name)[:length] for sample in samples], batch_first=True).to(device)
             for name in ("input_ids", "labels", "loss_mask", "hierarchy_vectors", *TRACKS)}
    for (prompt, _), sample in zip(PAIRS, samples):
        bundle = tokenizer.prepare_generation_hierarchy(prompt)
        assert sample.input_ids[:len(bundle.token_ids)].tolist() == bundle.token_ids, "Prompt framing mismatch"
        for name, expected in zip(TRACKS, bundle.as_tuple()[1:]):
            assert getattr(sample, name)[:len(expected)].tolist() == expected, f"Prompt {name} mismatch"

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        loss, output = model.compute_loss(**batch, collect_telemetry=False)
        mask = batch["loss_mask"].gt(0) & batch["labels"].ne(tokenizer.pad_id)
        accuracy = float(output.logits.argmax(-1)[mask].eq(batch["labels"][mask]).float().mean())
        results = []
        for prompt, answer in PAIRS:
            bundle = tokenizer.prepare_generation_hierarchy(prompt)
            metadata = {name: torch.tensor([track], device=device) for name, track in zip(TRACKS, bundle.as_tuple()[1:])}
            generated = model.generate(
                torch.tensor([bundle.token_ids], device=device), **metadata,
                hierarchy_vectors=torch.tensor([bundle.hierarchy_vectors], device=device),
                max_new_tokens=32, min_new_tokens=0, temperature=0.0,
                repetition_penalty=1.0, no_repeat_ngram_size=0, beam_size=1,
                suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
                token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
                token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
                token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
                token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
                use_speculative_decoding=False,
            )[0, len(bundle.token_ids):].tolist()
            decoded = tokenizer.decode(generated).strip()
            results.append(dict(prompt=prompt, expected=answer, generated=decoded,
                                token_ids=generated, exact=decoded == answer))
        record = dict(step=step, ce=float(output.ce_loss), accuracy=accuracy,
                      exact=sum(result["exact"] for result in results), results=results,
                      elapsed_seconds=time.monotonic() - start)
        evaluations.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
        (args.output / "results.json").write_text(json.dumps(dict(
            pairs=PAIRS, config=model.cfg.to_dict(), seed=17, device=str(device),
            parameters=sum(p.numel() for p in model.parameters()),
            evaluations=evaluations, learning_rate=args.lr,
            max_gpu_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
        ), ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    start = time.monotonic()
    evaluations = []
    evaluate(0)  # Initialize lazy modules before constructing the optimizer.
    observed_gradient_sums = {}
    consecutive_passes = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(**batch, collect_telemetry=False)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        for name, parameter in (
            ("torus_write", model.torus_core.write_delta_proj.weight),
            ("lattice_value", model.signature_lattice_attention.v_proj.weight),
        ):
            observed_gradient_sums[name] = float(parameter.grad.detach().abs().sum()) if parameter.grad is not None else 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0:
            print(json.dumps(dict(step=step, train_loss=float(loss.detach()), grad_norm=float(grad_norm),
                                  elapsed_seconds=time.monotonic() - start)), flush=True)
        if step % 50 == 0 or step == args.steps:
            result = evaluate(step)
            consecutive_passes = consecutive_passes + 1 if result["exact"] == len(PAIRS) and result["ce"] < 0.1 else 0
            if consecutive_passes >= 2:
                assert all(value > 0 for value in observed_gradient_sums.values()), observed_gradient_sums
                (args.output / "success.json").write_text(json.dumps(dict(
                    success=True, consecutive_passing_evaluations=consecutive_passes,
                    final_gradient_sums=observed_gradient_sums, final_step=step,
                    note="Memorization control only; no held-out generalization claim or saved weight checkpoint.",
                ), indent=2), encoding="utf-8")
                print("PASS: all four distinct answers learned under greedy decoding", flush=True)
                return 0
    print("INCOMPLETE: bounded control did not meet all success criteria", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
