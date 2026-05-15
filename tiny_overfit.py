# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

"""Run a tiny overfit probe on the checked-in mini overfit fixture."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CLI_PATH = ROOT / "cli.py"
DEFAULT_DATA = ROOT / "BaseData" / "mini_overfit" / "mini_overfit.jsonl"
DEFAULT_SAVE_ROOT = ROOT / "checkpoints" / "mini_overfit"
DEFAULT_PROMPT = "What is a cat?"
DEFAULT_RESPONSE = "meow"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the mini overfit harness on the checked-in fixture.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--diagnostic-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--min-token-frequency", type=int, default=2)
    parser.add_argument("--tokenizer-workers", type=int, default=1)
    parser.add_argument("--use-gradient-accumulation", dest="use_gradient_accumulation", action="store_true")
    parser.add_argument("--no-gradient-accumulation", dest="use_gradient_accumulation", action="store_false")
    parser.set_defaults(use_gradient_accumulation=False)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--post-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--post-response", default=DEFAULT_RESPONSE)
    parser.add_argument("--post-max-new-tokens", type=int, default=64)
    parser.add_argument("--post-min-new-tokens", type=int, default=1)
    parser.add_argument("--post-temperature", type=float, default=0.0)
    parser.add_argument("--post-top-k", type=int, default=0)
    parser.add_argument("--post-top-p", type=float, default=1.0)
    parser.add_argument("--post-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--post-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--post-beam-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-template-prompt", dest="template_prompt", action="store_false")
    parser.set_defaults(template_prompt=False)
    parser.add_argument("--dry-run", action="store_true", help="Print the train command without running it.")
    return parser


def _timestamped_save_dir(save_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return save_root / stamp


def build_train_command(args: argparse.Namespace, save_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(CLI_PATH),
        "train",
        "--data",
        str(args.data),
        "--save-dir",
        str(save_dir),
        "--steps",
        str(args.steps),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--max-samples",
        str(args.max_samples),
        "--val-fraction",
        str(args.val_fraction),
        "--diagnostic-interval",
        str(args.diagnostic_interval),
        "--lr",
        str(args.lr),
        "--min-token-frequency",
        str(args.min_token_frequency),
        "--tokenizer-workers",
        str(args.tokenizer_workers),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--no-gradient-accumulation" if not args.use_gradient_accumulation else "--use-gradient-accumulation",
        "--post-prompt",
        args.post_prompt,
        "--post-output",
        str(save_dir / "prompt.txt"),
        "--post-max-new-tokens",
        str(args.post_max_new_tokens),
        "--post-min-new-tokens",
        str(args.post_min_new_tokens),
        "--post-temperature",
        str(args.post_temperature),
        "--post-top-k",
        str(args.post_top_k),
        "--post-top-p",
        str(args.post_top_p),
        "--post-repetition-penalty",
        str(args.post_repetition_penalty),
        "--post-no-repeat-ngram-size",
        str(args.post_no_repeat_ngram_size),
        "--post-beam-size",
        str(args.post_beam_size),
        "--post-no-template-prompt",
    ]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.data.exists():
        raise FileNotFoundError(f"Mini overfit dataset not found: {args.data}")

    save_dir = _timestamped_save_dir(args.save_root)
    save_dir.mkdir(parents=True, exist_ok=True)
    command = build_train_command(args, save_dir)

    if args.dry_run:
        print(json.dumps({"save_dir": str(save_dir), "command": command}, indent=2))
        return 0

    print(f"[mini-overfit] dataset: {args.data}")
    print(f"[mini-overfit] save_dir: {save_dir}")
    subprocess.run(command, check=True)

    metrics_path = save_dir / "metrics.json"
    prompt_path = save_dir / "prompt.txt"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
    continuation = prompt_text.split("\n\n", 1)[-1].split("\nPrompt token IDs:", 1)[0].strip().lower()
    expected_tokens = re.findall(r"[a-z0-9']+", args.post_response.lower())
    generated_tokens = re.findall(r"[a-z0-9']+", continuation)
    overlap = sum(min(Counter(expected_tokens)[tok], Counter(generated_tokens)[tok]) for tok in set(expected_tokens))
    token_f1 = (2 * overlap) / max(len(expected_tokens) + len(generated_tokens), 1)

    summary = {
        "final_total_loss": metrics.get("final_total_loss"),
        "final_ce_loss": metrics.get("final_ce_loss"),
        "val_total_loss": metrics.get("val_total_loss"),
        "token_f1": token_f1,
        "expected_response": args.post_response,
        "prompt_path": str(prompt_path),
        "post_preview": " ".join(prompt_text.split())[:240],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
