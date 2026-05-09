# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run small generation ablations across EPIC-SHARC MOHTE checkpoints."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import torch

try:
    from .train import load_bundle_from_checkpoint, resolve_device
except ImportError:  # pragma: no cover - supports direct script launching.
    from train import load_bundle_from_checkpoint, resolve_device


def _resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            return model_path
    return checkpoint_path


def _format_snippet(text: str, width: int = 160) -> str:
    compact = " ".join(text.replace("\r", "\n").split())
    return compact[:width]


@contextmanager
def _temporarily_disable_sharc(model) -> Iterator[None]:
    cfg = model.cfg
    original = {
        "use_race_lanes": getattr(model, "use_race_lanes", False),
        "use_signature_lattice_attention": getattr(cfg, "use_signature_lattice_attention", True),
        "use_signature_lattice_generation_cache": getattr(cfg, "use_signature_lattice_generation_cache", True),
        "signature_lattice_attention": getattr(model, "signature_lattice_attention", None),
    }
    try:
        model.use_race_lanes = False
        cfg.use_torus_race_lanes = False
        cfg.use_signature_lattice_attention = False
        cfg.use_signature_lattice_generation_cache = False
        model.signature_lattice_attention = None
        yield
    finally:
        model.use_race_lanes = original["use_race_lanes"]
        cfg.use_torus_race_lanes = original["use_race_lanes"]
        cfg.use_signature_lattice_attention = original["use_signature_lattice_attention"]
        cfg.use_signature_lattice_generation_cache = original["use_signature_lattice_generation_cache"]
        model.signature_lattice_attention = original["signature_lattice_attention"]


def _build_prompt_tensors(tokenizer, prompt: str, template_prompt: bool, device: torch.device) -> Dict[str, torch.Tensor]:
    prompt_text = prompt.strip()
    if template_prompt and not any(marker in prompt_text.lower() for marker in ("instruction:", "response:")):
        prompt_text = f"Instruction: {prompt_text}\nResponse: "
    bundle = tokenizer.prepare_generation_hierarchy(prompt_text)
    token_ids, sig_ids, level_ids, rel_ids, parent_ids, family_ids = bundle.as_tuple()
    return {
        "prompt_text": prompt_text,
        "input_ids": torch.tensor([token_ids], dtype=torch.long, device=device),
        "signature_ids": torch.tensor([sig_ids], dtype=torch.long, device=device),
        "signature_level_ids": torch.tensor([level_ids], dtype=torch.long, device=device),
        "signature_relation_ids": torch.tensor([rel_ids], dtype=torch.long, device=device),
        "parent_signature_ids": torch.tensor([parent_ids], dtype=torch.long, device=device),
        "signature_family_ids": torch.tensor([family_ids], dtype=torch.long, device=device),
    }


def _run_variant(
    model,
    tokenizer,
    prompt: str,
    *,
    template_prompt: bool,
    max_new_tokens: int,
    min_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    beam_size: int,
    use_speculative_decoding: bool,
    speculative_draft_tokens: int,
    speculative_temperature: float,
    sharc_enabled: bool,
) -> Dict[str, object]:
    model_device = next(model.parameters()).device
    tensors = _build_prompt_tensors(tokenizer, prompt, template_prompt, model_device)
    common_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        beam_size=beam_size,
        use_speculative_decoding=use_speculative_decoding,
        speculative_draft_tokens=speculative_draft_tokens,
        speculative_temperature=speculative_temperature,
        token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
        token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
        token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
        token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
        suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
    )

    if sharc_enabled:
        generated = model.generate(
            tensors["input_ids"],
            signature_family_ids=tensors["signature_family_ids"],
            signature_ids=tensors["signature_ids"],
            signature_level_ids=tensors["signature_level_ids"],
            signature_relation_ids=tensors["signature_relation_ids"],
            parent_signature_ids=tensors["parent_signature_ids"],
            **common_kwargs,
        )
    else:
        with _temporarily_disable_sharc(model):
            generated = model.generate(
                tensors["input_ids"],
                signature_family_ids=tensors["signature_family_ids"],
                signature_ids=tensors["signature_ids"],
                signature_level_ids=tensors["signature_level_ids"],
                signature_relation_ids=tensors["signature_relation_ids"],
                parent_signature_ids=tensors["parent_signature_ids"],
                **common_kwargs,
            )

    generated_ids = generated[0].tolist()
    prompt_len = len(tensors["input_ids"][0])
    continuation_ids = generated_ids[prompt_len:]
    continuation_text = tokenizer.decode(continuation_ids)
    return {
        "prompt_text": tensors["prompt_text"],
        "generated_text": continuation_text,
        "generated_snippet": _format_snippet(continuation_text),
        "generated_token_count": len(continuation_ids),
        "generated_token_ids": continuation_ids,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare EPIC-SHARC MOHTE generation settings side by side.")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint directory or model.pt path")
    parser.add_argument("--prompt", required=True, help="Prompt to run through every checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--speculative-draft-tokens", type=int, default=2)
    parser.add_argument("--speculative-temperature", type=float, default=0.0)
    parser.add_argument("--template-prompt", dest="template_prompt", action="store_true")
    parser.add_argument("--no-template-prompt", dest="template_prompt", action="store_false")
    parser.set_defaults(template_prompt=True)
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON instead of tables")
    return parser


def _mode_rows() -> Sequence[tuple[str, bool, bool]]:
    return (
        ("spec_on__sharc_on", True, True),
        ("spec_off__sharc_on", False, True),
        ("spec_on__sharc_off", True, False),
        ("spec_off__sharc_off", False, False),
    )


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = resolve_device(args.device)

    results: List[Dict[str, object]] = []
    for checkpoint in args.checkpoints:
        checkpoint_path = _resolve_checkpoint_path(checkpoint)
        model, tokenizer, raw_cfg = load_bundle_from_checkpoint(checkpoint_path, device=device)
        model = model.to(device)
        checkpoint_result = {
            "checkpoint": str(checkpoint_path),
            "optimizer": getattr(raw_cfg, "optimizer", getattr(model.cfg, "optimizer", "unknown")),
            "use_speculative_decoding_default": bool(getattr(model.cfg, "use_speculative_decoding", False)),
            "use_signature_lattice_attention_default": bool(getattr(model.cfg, "use_signature_lattice_attention", False)),
            "modes": [],
        }
        for mode_name, spec_enabled, sharc_enabled in _mode_rows():
            mode_result = _run_variant(
                model,
                tokenizer,
                args.prompt,
                template_prompt=args.template_prompt,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                beam_size=args.beam_size,
                use_speculative_decoding=spec_enabled,
                speculative_draft_tokens=args.speculative_draft_tokens,
                speculative_temperature=args.speculative_temperature,
                sharc_enabled=sharc_enabled,
            )
            checkpoint_result["modes"].append(
                {
                    "mode": mode_name,
                    "speculative": spec_enabled,
                    "sharc": sharc_enabled,
                    "snippet": mode_result["generated_snippet"],
                    "tokens": mode_result["generated_token_count"],
                    "token_ids": mode_result["generated_token_ids"],
                }
            )
        results.append(checkpoint_result)

    if args.json_output:
        print(json.dumps(results, indent=2))
        return 0

    for checkpoint_result in results:
        print(f"Checkpoint: {checkpoint_result['checkpoint']}")
        print(
            f"  optimizer={checkpoint_result['optimizer']} "
            f"default_spec={checkpoint_result['use_speculative_decoding_default']} "
            f"default_sharc={checkpoint_result['use_signature_lattice_attention_default']}"
        )
        print("| mode | spec | sharc | tokens | snippet |")
        print("| --- | --- | --- | --- | --- |")
        for mode in checkpoint_result["modes"]:
            print(
                f"| {mode['mode']} | {mode['speculative']} | {mode['sharc']} | {mode['tokens']} | "
                f"{mode['snippet'].replace('|', '\\|')} |"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
