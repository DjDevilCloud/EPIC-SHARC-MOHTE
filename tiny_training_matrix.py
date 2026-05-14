from __future__ import annotations

"""Tiny training matrix runner for routing-stability checks."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
from statistics import fmean
from typing import Any, Iterable, Sequence

try:
    from .config import PrismalWaveConfig
    from .train import PrismalTokenizer, PrismalWaveModel, build_dataloader, generate_text, load_bundle_from_checkpoint, resolve_device, run_benchmark
except ImportError:  # pragma: no cover - supports direct script launching.
    from config import PrismalWaveConfig
    from train import PrismalTokenizer, PrismalWaveModel, build_dataloader, generate_text, load_bundle_from_checkpoint, resolve_device, run_benchmark


ROOT = Path(__file__).resolve().parent
CLI_PATH = ROOT / "cli.py"
COMPARE_PATH = ROOT / "compare_generation_ablation.py"
PRETOKENIZED_ROOT = ROOT / "pretokenized"
DEFAULT_PROMPT = "What is a cat?"
DEFAULT_RUN_ROOT = ROOT / "checkpoints" / "tiny_training_matrix" / datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_SMOKE_RUN_ROOT = ROOT / "checkpoints" / "miniqa100_smoke" / datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_BENCHMARK_DATA = PRETOKENIZED_ROOT / "DictWords_synthetic_sentences"
DEFAULT_MINIQA100_DATA = ROOT / "BaseData" / "miniqa100" / "miniqa100.jsonl"
DEFAULT_STAGE1_MAX_SAMPLES = 64
DEFAULT_STAGE2_MAX_SAMPLES = 256
DEFAULT_BATCH_SIZE = 10
DEFAULT_BENCHMARK_STEPS = 8
DEFAULT_BENCHMARK_BATCH_SIZE = 10
DEFAULT_BENCHMARK_MAX_SAMPLES = 64
DEFAULT_SMOKE_BATCH_SIZE = 4
DEFAULT_SMOKE_MAX_SAMPLES = 32
DEFAULT_SMOKE_HOLDOUT_COUNT = 5
DEFAULT_SMOKE_MAX_NEW_TOKENS = 32
DEFAULT_SMOKE_MIN_NEW_TOKENS = 1
DEFAULT_SMOKE_TOP_K = 8
DEFAULT_SMOKE_TOP_P = 0.92
DEFAULT_SMOKE_TEMPERATURE = 0.15
DEFAULT_SMOKE_REPETITION_PENALTY = 1.1
DEFAULT_SMOKE_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_SMOKE_BEAM_SIZE = 2
DEFAULT_DECODE_RUN_ROOT = ROOT / "checkpoints" / "miniqa100_decode" / datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_COHERENCY_RUN_ROOT = ROOT / "checkpoints" / "miniqa100_coherency" / datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_TIERS_RUN_ROOT = ROOT / "checkpoints" / "miniqa100_tiers" / datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_RESIDENCY_RUN_ROOT = ROOT / "checkpoints" / "miniqa100_residency" / datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_COHERENCY_MAX_SAMPLES = 53
DEFAULT_TIERS_MAX_SAMPLES = 53
DEFAULT_RESIDENCY_MAX_SAMPLES = 53
DEFAULT_MINIQA100_BASELINE_CHECKPOINT = ROOT / "checkpoints" / "miniqa100_smoke" / "20260513_203641" / "baseline"
DEFAULT_COMPARE_PROMPTS = (
    "What is a cat?",
    "Explain torus routing in one sentence.",
    "Summarize the routing behavior in one sentence.",
)

FIXED_TRAIN_ARGS: tuple[str, ...] = (
    "--epochs",
    "1",
    "--steps",
    "0",
    "--tokenizer-workers",
    "1",
    "--no-dataset-streaming",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    max_samples: int


@dataclass(frozen=True)
class VariantSpec:
    name: str
    extra_args: tuple[str, ...] = ()


@dataclass
class RunRecord:
    stage: str
    dataset: str
    variant: str
    data_path: str
    save_dir: str
    status: str
    command: list[str]
    max_samples: int
    batch_size: int
    metrics: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    prompt_preview: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class QAExample:
    index: int
    instruction: str
    context: str
    response: str
    category: str = ""


@dataclass
class SmokeGenerationRecord:
    variant: str
    index: int
    prompt: str
    reference: str
    generated_text: str
    generated_answer: str
    normalized_reference: str
    normalized_generated: str
    exact_match: bool
    token_overlap: float
    token_precision: float
    token_recall: float
    unique_token_ratio: float
    repetition_ratio: float
    token_count: int
    char_count: int
    token_soup: bool
    readable: bool


@dataclass
class SmokeVariantRecord:
    variant: str
    settings_source: str
    save_dir: str
    source_data_path: str
    train_data_path: str
    holdout_data_path: str
    train_mode: str
    status: str
    command: list[str]
    train_metrics: dict[str, Any] | None = None
    train_config: dict[str, Any] | None = None
    holdout_examples: list[dict[str, Any]] | None = None
    generations: list[dict[str, Any]] | None = None
    eval_summary: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class DecodeVariantSpec:
    name: str
    max_new_tokens: int = DEFAULT_SMOKE_MAX_NEW_TOKENS
    min_new_tokens: int = DEFAULT_SMOKE_MIN_NEW_TOKENS
    top_k: int = DEFAULT_SMOKE_TOP_K
    top_p: float = DEFAULT_SMOKE_TOP_P
    temperature: float = DEFAULT_SMOKE_TEMPERATURE
    repetition_penalty: float = DEFAULT_SMOKE_REPETITION_PENALTY
    no_repeat_ngram_size: int = DEFAULT_SMOKE_NO_REPEAT_NGRAM_SIZE
    beam_size: int = DEFAULT_SMOKE_BEAM_SIZE
    speculative_draft_tokens: int = 2
    speculative_temperature: float = 0.0
    template_prompt: bool = False


@dataclass
class DecodeVariantRecord:
    variant: str
    checkpoint_path: str
    settings_source: str
    save_dir: str
    source_data_path: str
    holdout_data_path: str
    status: str
    command: list[str]
    settings: dict[str, Any]
    holdout_examples: list[dict[str, Any]] | None = None
    generations: list[dict[str, Any]] | None = None
    eval_summary: dict[str, Any] | None = None
    error: str | None = None


STAGE1_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("DictWords_synthetic_sentences", PRETOKENIZED_ROOT / "DictWords_synthetic_sentences", DEFAULT_STAGE1_MAX_SAMPLES),
    DatasetSpec("dolly-15k", PRETOKENIZED_ROOT / "dolly-15k", DEFAULT_STAGE1_MAX_SAMPLES),
)

STAGE2_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("CombinedNVIDIAPT", PRETOKENIZED_ROOT / "CombinedNVIDIAPT", DEFAULT_STAGE2_MAX_SAMPLES),
    DatasetSpec("WordNet", PRETOKENIZED_ROOT / "WordNet", DEFAULT_STAGE2_MAX_SAMPLES),
)

MATRIX_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        "baseline",
        (
            "--torus-local-field-radius",
            "2",
            "--torus-relay-write-radius",
            "2",
            "--torus-inner-temperature",
            "0.20",
            "--torus-outer-temperature",
            "1.00",
        ),
    ),
    VariantSpec(
        "wider_support",
        (
            "--torus-local-field-radius",
            "3",
            "--torus-relay-write-radius",
            "3",
            "--torus-inner-temperature",
            "0.20",
            "--torus-outer-temperature",
            "1.00",
        ),
    ),
    VariantSpec(
        "softer_temperature",
        (
            "--torus-local-field-radius",
            "2",
            "--torus-relay-write-radius",
            "2",
            "--torus-inner-temperature",
            "0.35",
            "--torus-outer-temperature",
            "1.25",
        ),
    ),
)

DECODE_VARIANTS: tuple[DecodeVariantSpec, ...] = (
    DecodeVariantSpec("baseline"),
    DecodeVariantSpec(
        "strict_beam",
        top_k=8,
        top_p=0.88,
        temperature=0.08,
        repetition_penalty=1.15,
        no_repeat_ngram_size=5,
        beam_size=4,
    ),
    DecodeVariantSpec(
        "clean_greedy",
        top_k=0,
        top_p=1.0,
        temperature=0.05,
        repetition_penalty=1.20,
        no_repeat_ngram_size=6,
        beam_size=1,
    ),
    DecodeVariantSpec(
        "freer_sample",
        top_k=12,
        top_p=0.97,
        temperature=0.20,
        repetition_penalty=1.06,
        no_repeat_ngram_size=4,
        beam_size=2,
    ),
)

COHERENCY_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("baseline", ()),
    VariantSpec(
        "router_cooler",
        (
            "--router-temperature",
            "0.50",
            "--signature-temperature",
            "0.45",
        ),
    ),
    VariantSpec(
        "lattice_focus",
        (
            "--signature-lattice-weight",
            "0.36",
            "--signature-lattice-candidates",
            "16",
            "--signature-lattice-decay",
            "0.90",
        ),
    ),
)

RESIDENCY_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("baseline", ()),
    VariantSpec(
        "learned_residency_weight_half",
        ("--learned-residency-weight", "0.05"),
    ),
    VariantSpec(
        "learned_residency_weight_bump",
        ("--learned-residency-weight", "0.15"),
    ),
    VariantSpec(
        "torus_global_bus_decay_half",
        ("--torus-global-bus-decay", "0.46"),
    ),
    VariantSpec(
        "torus_global_bus_decay_bump",
        ("--torus-global-bus-decay", "1.0"),
    ),
    VariantSpec(
        "torus_global_bus_write_scale_half",
        ("--torus-global-bus-write-scale", "0.16"),
    ),
    VariantSpec(
        "torus_global_bus_write_scale_bump",
        ("--torus-global-bus-write-scale", "0.48"),
    ),
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return slug or "run"


def _ensure_dataset_paths(datasets: Iterable[DatasetSpec]) -> None:
    missing = [dataset.path for dataset in datasets if not dataset.path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset path(s): " + ", ".join(str(path) for path in missing))


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        model_path = path / "model.pt"
        if model_path.exists():
            return model_path
    return path


def _build_train_command(
    *,
    dataset: DatasetSpec,
    variant: VariantSpec,
    save_dir: Path,
    batch_size: int,
    max_samples: int,
    extra_args: Sequence[str] = (),
    continue_checkpoint: Path | None = None,
    post_prompt: str | None = DEFAULT_PROMPT,
    post_output: Path | None = None,
    python_exe: str | None = None,
) -> list[str]:
    executable = python_exe or sys.executable
    command = [executable, "-u", str(CLI_PATH), "train"]
    if continue_checkpoint is not None:
        command.extend(["--continue-checkpoint", str(continue_checkpoint)])
    command.extend(
        [
            "--data",
            str(dataset.path),
            "--save-dir",
            str(save_dir),
            "--batch-size",
            str(batch_size),
            "--max-samples",
            str(max_samples),
        ]
    )
    command.extend(FIXED_TRAIN_ARGS)
    command.extend(extra_args)
    command.extend(variant.extra_args)
    if post_prompt is not None:
        command.extend(["--post-prompt", post_prompt])
        command.extend(["--post-output", str(post_output or (save_dir / "prompt.txt"))])
    return command


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_prompt_preview(path: Path, *, width: int = 240) -> str | None:
    if not path.exists():
        return None
    text = " ".join(path.read_text(encoding="utf-8").split())
    return text[:width]


def _collect_record(
    *,
    stage: str,
    dataset: DatasetSpec,
    variant: VariantSpec,
    save_dir: Path,
    command: list[str],
    batch_size: int,
    max_samples: int,
    status: str,
    error: str | None = None,
) -> RunRecord:
    return RunRecord(
        stage=stage,
        dataset=dataset.name,
        variant=variant.name,
        data_path=str(dataset.path),
        save_dir=str(save_dir),
        status=status,
        command=command,
        max_samples=max_samples,
        batch_size=batch_size,
        metrics=_read_json(save_dir / "metrics.json"),
        config=_read_json(save_dir / "config.json"),
        prompt_preview=_read_prompt_preview(save_dir / "prompt.txt"),
        error=error,
    )


def _run_training(
    *,
    stage: str,
    dataset: DatasetSpec,
    variant: VariantSpec,
    run_root: Path,
    batch_size: int,
    max_samples: int,
    python_exe: str | None = None,
    dry_run: bool = False,
) -> RunRecord:
    save_dir = run_root / stage / _slugify(dataset.name) / variant.name
    save_dir.mkdir(parents=True, exist_ok=True)
    command = _build_train_command(
        dataset=dataset,
        variant=variant,
        save_dir=save_dir,
        batch_size=batch_size,
        max_samples=max_samples,
        python_exe=python_exe,
    )

    if dry_run:
        print(" ".join(command))
        return _collect_record(
            stage=stage,
            dataset=dataset,
            variant=variant,
            save_dir=save_dir,
            command=command,
            batch_size=batch_size,
            max_samples=max_samples,
            status="dry-run",
        )

    print(f"[matrix] {stage}/{dataset.name}/{variant.name}")
    status = "ok"
    error = None
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        status = "failed"
        error = f"returncode={exc.returncode}"
        print(f"[matrix] {stage}/{dataset.name}/{variant.name} failed: {error}")
    return _collect_record(
        stage=stage,
        dataset=dataset,
        variant=variant,
        save_dir=save_dir,
        command=command,
        batch_size=batch_size,
        max_samples=max_samples,
        status=status,
        error=error,
    )


def _metric_value(record: RunRecord, *names: str) -> float | None:
    if not record.metrics:
        return None
    for name in names:
        value = record.metrics.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _mean_metric(records: Sequence[RunRecord], *names: str) -> float:
    values = []
    for record in records:
        if record.status != "ok":
            continue
        value = _metric_value(record, *names)
        if value is not None:
            values.append(value)
    return fmean(values) if values else float("nan")


def _aggregate_records(records: Sequence[RunRecord]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.status != "ok":
            continue
        grouped.setdefault(record.variant, []).append(record)

    aggregates: dict[str, dict[str, float]] = {}
    for variant, variant_records in grouped.items():
        aggregates[variant] = {
            "val_total": _mean_metric(variant_records, "val_total_loss", "val_loss"),
            "val_ce": _mean_metric(variant_records, "val_ce_loss"),
            "val_aux": _mean_metric(variant_records, "val_aux_loss"),
            "usage_conc": _mean_metric(variant_records, "avg_emitter_usage_concentration"),
            "eff_count": _mean_metric(variant_records, "avg_emitter_cell_effective_count"),
            "cell_breadth": _mean_metric(variant_records, "avg_emitter_cell_breadth"),
            "cell_cov": _mean_metric(variant_records, "avg_emitter_cell_coverage_loss"),
            "clip_groups": _mean_metric(variant_records, "stability_clipped_groups"),
            "repairs": _mean_metric(variant_records, "stability_repaired_tensors"),
        }
    return aggregates


def _rank_variants(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    aggregates = _aggregate_records(records)
    ranking: list[dict[str, Any]] = []
    for variant, metrics in aggregates.items():
        ranking.append(
            {
                "variant": variant,
                "mean_val_total": metrics["val_total"],
                "mean_val_ce": metrics["val_ce"],
                "mean_usage_conc": metrics["usage_conc"],
                "mean_eff_count": metrics["eff_count"],
                "mean_cell_cov": metrics["cell_cov"],
            }
        )
    ranking.sort(key=lambda row: (row["mean_val_total"], row["mean_val_ce"], row["variant"]))
    return ranking


def _write_summary(stage: str, run_root: Path, records: Sequence[RunRecord]) -> Path:
    payload = {
        "stage": stage,
        "run_root": str(run_root),
        "records": [asdict(record) for record in records],
        "ranking": _rank_variants(records),
        "aggregates": _aggregate_records(records),
    }
    output_path = run_root / f"{stage}_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md_path = run_root / f"{stage}_summary.md"
    lines = [
        f"# {stage.title()} Summary",
        "",
        "| dataset | variant | status | val_total | val_ce | val_aux | usage_conc | eff_count | cell_breadth | cell_cov | clip_groups | repairs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        val_total = _metric_value(record, "val_total_loss", "val_loss")
        val_ce = _metric_value(record, "val_ce_loss")
        val_aux = _metric_value(record, "val_aux_loss")
        usage_conc = _metric_value(record, "avg_emitter_usage_concentration")
        eff_count = _metric_value(record, "avg_emitter_cell_effective_count")
        cell_breadth = _metric_value(record, "avg_emitter_cell_breadth")
        cell_cov = _metric_value(record, "avg_emitter_cell_coverage_loss")
        clip_groups = _metric_value(record, "stability_clipped_groups")
        repairs = _metric_value(record, "stability_repaired_tensors")
        lines.append(
            "| {dataset} | {variant} | {status} | {val_total} | {val_ce} | {val_aux} | {usage_conc} | {eff_count} | {cell_breadth} | {cell_cov} | {clip_groups} | {repairs} |".format(
                dataset=record.dataset,
                variant=record.variant,
                status=record.status,
                val_total=f"{val_total:.4f}" if val_total is not None else "n/a",
                val_ce=f"{val_ce:.4f}" if val_ce is not None else "n/a",
                val_aux=f"{val_aux:.4f}" if val_aux is not None else "n/a",
                usage_conc=f"{usage_conc:.4f}" if usage_conc is not None else "n/a",
                eff_count=f"{eff_count:.2f}" if eff_count is not None else "n/a",
                cell_breadth=f"{cell_breadth:.4f}" if cell_breadth is not None else "n/a",
                cell_cov=f"{cell_cov:.4f}" if cell_cov is not None else "n/a",
                clip_groups=f"{clip_groups:.0f}" if clip_groups is not None else "n/a",
                repairs=f"{repairs:.0f}" if repairs is not None else "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            "| variant | mean_val_total | mean_val_ce | mean_usage_conc | mean_eff_count | mean_cell_cov |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in _rank_variants(records):
        lines.append(
            "| {variant} | {mean_val_total:.4f} | {mean_val_ce:.4f} | {mean_usage_conc:.4f} | {mean_eff_count:.2f} | {mean_cell_cov:.4f} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _select_top_variants(summary_path: Path, top_n: int) -> list[str]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ranking = payload.get("ranking", [])
    selected = [str(row["variant"]) for row in ranking[: max(1, top_n)]]
    if not selected:
        raise ValueError(f"No variant ranking found in {summary_path}")
    return selected


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_jsonl_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _select_holdout_indices(total: int, holdout_count: int) -> list[int]:
    if total <= 0 or holdout_count <= 0:
        return []
    count = min(total, holdout_count)
    if count == 1:
        return [total // 2]
    indices: list[int] = []
    seen: set[int] = set()
    for slot in range(count):
        idx = int(round(slot * (total - 1) / max(count - 1, 1)))
        idx = max(0, min(total - 1, idx))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    if len(indices) < count:
        for idx in range(total):
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
            if len(indices) >= count:
                break
    return sorted(indices[:count])


def _qa_example_from_payload(payload: dict[str, Any], *, index: int) -> QAExample:
    instruction = str(payload.get("instruction") or payload.get("question") or payload.get("prompt") or "").strip()
    context = str(payload.get("context") or payload.get("input") or "").strip()
    response = str(payload.get("response") or payload.get("answer") or payload.get("output") or "").strip()
    category = str(payload.get("category") or "").strip()
    return QAExample(index=index, instruction=instruction, context=context, response=response, category=category)


def _load_miniqa100_examples(path: Path) -> list[QAExample]:
    return [_qa_example_from_payload(payload, index=index) for index, payload in enumerate(_load_jsonl_records(path))]


def _split_miniqa100_examples(examples: Sequence[QAExample], holdout_count: int) -> tuple[list[QAExample], list[QAExample], list[int]]:
    holdout_indices = _select_holdout_indices(len(examples), holdout_count)
    holdout_lookup = set(holdout_indices)
    train_examples = [example for example in examples if example.index not in holdout_lookup]
    holdout_examples = [example for example in examples if example.index in holdout_lookup]
    return train_examples, holdout_examples, holdout_indices


def _example_to_payload(example: QAExample) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "instruction": example.instruction,
        "context": example.context,
        "response": example.response,
    }
    if example.category:
        payload["category"] = example.category
    payload["index"] = example.index
    return payload


def _format_miniqa100_prompt(example: QAExample) -> str:
    parts = [f"Instruction: {example.instruction.strip()}"]
    if example.context.strip():
        parts.append(f"Context: {example.context.strip()}")
    parts.append("Response:")
    return "\n".join(parts)


def _extract_generated_answer(generated_text: str) -> str:
    body, _, _ = generated_text.partition("\nPrompt token IDs:")
    if "\n\n" not in body:
        return ""
    _, _, continuation = body.rpartition("\n\n")
    return continuation.strip()


_QA_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _normalize_answer(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9'\s]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _tokenize_answer(text: str) -> list[str]:
    return _QA_TOKEN_RE.findall(text.lower())


def _score_qa_generation(*, reference: str, generated: str) -> dict[str, Any]:
    reference_norm = _normalize_answer(reference)
    generated_norm = _normalize_answer(generated)
    reference_tokens = _tokenize_answer(reference_norm)
    generated_tokens = _tokenize_answer(generated_norm)
    reference_counts = Counter(reference_tokens)
    generated_counts = Counter(generated_tokens)
    overlap = sum(min(reference_counts[token], generated_counts[token]) for token in reference_counts)
    precision = overlap / len(generated_tokens) if generated_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    token_overlap = (2.0 * precision * recall / (precision + recall)) if precision and recall else 0.0
    unique_ratio = (len(set(generated_tokens)) / len(generated_tokens)) if generated_tokens else 0.0
    repetition_ratio = 1.0 - unique_ratio if generated_tokens else 0.0
    exact_match = bool(reference_norm) and generated_norm == reference_norm
    substring_match = bool(reference_norm) and (reference_norm in generated_norm or generated_norm in reference_norm)
    alpha_token_count = sum(1 for token in generated_tokens if any(ch.isalpha() for ch in token))
    token_soup = bool(generated_tokens) and (
        (len(generated_tokens) >= 10 and unique_ratio < 0.45)
        or repetition_ratio > 0.70
        or alpha_token_count <= 1
    )
    readable = bool(generated_tokens) and len(generated_tokens) >= 2 and not token_soup
    return {
        "normalized_reference": reference_norm,
        "normalized_generated": generated_norm,
        "exact_match": exact_match,
        "substring_match": substring_match,
        "token_overlap": token_overlap,
        "token_precision": precision,
        "token_recall": recall,
        "unique_token_ratio": unique_ratio,
        "repetition_ratio": repetition_ratio,
        "token_count": len(generated_tokens),
        "char_count": len(generated),
        "token_soup": token_soup,
        "readable": readable,
    }


def _miniqa100_default_config() -> dict[str, Any]:
    return PrismalWaveConfig().to_dict()


def _miniqa100_baseline_extra_args() -> tuple[str, ...]:
    return ()


def _miniqa100_decode_variants() -> tuple[DecodeVariantSpec, ...]:
    return DECODE_VARIANTS


def _format_float_arg(value: float) -> str:
    return f"{value:.6g}"


def _miniqa100_tier_variants() -> tuple[VariantSpec, ...]:
    default_cfg = PrismalWaveConfig()
    sweep_specs = [
        ("hierarchical_tier_char_weight", "--hierarchical-tier-char-weight", default_cfg.hierarchical_tier_char_weight),
        ("hierarchical_tier_piece_weight", "--hierarchical-tier-piece-weight", default_cfg.hierarchical_tier_piece_weight),
        ("hierarchical_tier_word_weight", "--hierarchical-tier-word-weight", default_cfg.hierarchical_tier_word_weight),
        ("hierarchical_tier_phrase_weight", "--hierarchical-tier-phrase-weight", default_cfg.hierarchical_tier_phrase_weight),
        ("hierarchical_tier_line_weight", "--hierarchical-tier-line-weight", default_cfg.hierarchical_tier_line_weight),
        ("hierarchical_tier_special_weight", "--hierarchical-tier-special-weight", default_cfg.hierarchical_tier_special_weight),
        ("hierarchical_leaf_char_boost", "--hierarchical-leaf-char-boost", default_cfg.hierarchical_leaf_char_boost),
        ("hierarchical_leaf_piece_boost", "--hierarchical-leaf-piece-boost", default_cfg.hierarchical_leaf_piece_boost),
    ]
    variants: list[VariantSpec] = [VariantSpec("baseline", ())]
    for name, flag, value in sweep_specs:
        half_value = max(1e-6, value * 0.5)
        bump_value = value * 1.5
        variants.append(VariantSpec(f"{_slugify(name)}_half", (flag, _format_float_arg(half_value))))
        variants.append(VariantSpec(f"{_slugify(name)}_bump", (flag, _format_float_arg(bump_value))))
    return tuple(variants)


def _smoke_variant_records_to_summary(records: Sequence[SmokeVariantRecord]) -> dict[str, Any]:
    by_variant: dict[str, list[SmokeVariantRecord]] = {}
    for record in records:
        by_variant.setdefault(record.variant, []).append(record)

    summary_variants: list[dict[str, Any]] = []
    for variant, variant_records in by_variant.items():
        eval_summaries = [record.eval_summary or {} for record in variant_records if record.status == "ok"]
        train_metrics = [record.train_metrics or {} for record in variant_records if record.status == "ok"]
        summary_variants.append(
            {
                "variant": variant,
                "status": variant_records[-1].status,
                "train_mode": variant_records[-1].train_mode,
                "mean_val_total": fmean([float(metrics.get("val_total_loss", metrics.get("val_loss", float("nan")))) for metrics in train_metrics]) if train_metrics else float("nan"),
                "mean_val_ce": fmean([float(metrics.get("val_ce_loss", float("nan"))) for metrics in train_metrics]) if train_metrics else float("nan"),
                "mean_val_aux": fmean([float(metrics.get("val_aux_loss", float("nan"))) for metrics in train_metrics]) if train_metrics else float("nan"),
                "mean_match_signal": fmean([float(summary.get("mean_match_signal", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_token_f1": fmean([float(summary.get("mean_token_f1", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_readable_rate": fmean([float(summary.get("readable_rate", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "token_soup_rate": fmean([float(summary.get("token_soup_rate", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
            }
        )
    summary_variants.sort(key=lambda row: (row["variant"]))
    baseline = next((row for row in summary_variants if row["variant"] == "baseline"), None)
    comparison: dict[str, Any] = {}
    if baseline:
        for row in summary_variants:
            if row["variant"] == "baseline":
                continue
            comparison[row["variant"]] = {
                "val_total_delta": row["mean_val_total"] - baseline["mean_val_total"],
                "val_ce_delta": row["mean_val_ce"] - baseline["mean_val_ce"],
                "val_aux_delta": row["mean_val_aux"] - baseline["mean_val_aux"],
                "match_signal_delta": row["mean_match_signal"] - baseline["mean_match_signal"],
                "token_f1_delta": row["mean_token_f1"] - baseline["mean_token_f1"],
                "readable_rate_delta": row["mean_readable_rate"] - baseline["mean_readable_rate"],
                "token_soup_rate_delta": row["token_soup_rate"] - baseline["token_soup_rate"],
            }
    return {
        "variants": summary_variants,
        "comparison": comparison,
    }


def _write_smoke_report(
    *,
    run_root: Path,
    summary: dict[str, Any],
    records: Sequence[SmokeVariantRecord],
    report_name: str = "smoke_report.md",
    title: str = "MiniQA100 Smoke Summary",
) -> Path:
    lines = [
        f"# {title}",
        "",
        f"Run root: `{run_root}`",
        "",
        "| variant | mode | val_total | val_ce | val_aux | match | token_f1 | readable | token_soup |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.get("variants", []):
        lines.append(
            "| {variant} | {train_mode} | {mean_val_total:.4f} | {mean_val_ce:.4f} | {mean_val_aux:.4f} | {mean_match_signal:.4f} | {mean_token_f1:.4f} | {mean_readable_rate:.2f} | {token_soup_rate:.2f} |".format(
                **row
            )
        )
    if summary.get("comparison"):
        lines.extend(
            [
                "",
                "## Comparison",
                "",
            ]
        )
        for variant, deltas in summary["comparison"].items():
            lines.append(f"### {variant}")
            for key, value in deltas.items():
                lines.append(f"- {key}: {value:.4f}")
    lines.extend(
        [
            "",
            "## Runs",
            "",
        ]
    )
    for record in records:
        lines.append(f"- {record.variant}: {record.status} ({record.train_mode}) -> {record.save_dir}")
    report_path = run_root / report_name
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _decode_variant_spec_to_dict(spec: DecodeVariantSpec) -> dict[str, Any]:
    return asdict(spec)


def _decode_variant_specs_to_summary(records: Sequence[DecodeVariantRecord]) -> dict[str, Any]:
    by_variant: dict[str, list[DecodeVariantRecord]] = {}
    for record in records:
        by_variant.setdefault(record.variant, []).append(record)

    summary_variants: list[dict[str, Any]] = []
    for variant, variant_records in by_variant.items():
        eval_summaries = [record.eval_summary or {} for record in variant_records if record.status == "ok"]
        summary_variants.append(
            {
                "variant": variant,
                "status": variant_records[-1].status,
                "checkpoint_path": variant_records[-1].checkpoint_path,
                "mean_match_signal": fmean([float(summary.get("mean_match_signal", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_token_f1": fmean([float(summary.get("mean_token_f1", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_token_precision": fmean([float(summary.get("mean_token_precision", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_token_recall": fmean([float(summary.get("mean_token_recall", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_unique_token_ratio": fmean([float(summary.get("mean_unique_token_ratio", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_repetition_ratio": fmean([float(summary.get("mean_repetition_ratio", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_token_count": fmean([float(summary.get("mean_token_count", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_char_count": fmean([float(summary.get("mean_char_count", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "mean_readable_rate": fmean([float(summary.get("readable_rate", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "token_soup_rate": fmean([float(summary.get("token_soup_rate", float("nan"))) for summary in eval_summaries]) if eval_summaries else float("nan"),
                "settings": variant_records[-1].settings,
            }
        )
    summary_variants.sort(key=lambda row: (row["variant"]))
    baseline = next((row for row in summary_variants if row["variant"] == "baseline"), None)
    comparison: dict[str, Any] = {}
    if baseline:
        for row in summary_variants:
            if row["variant"] == "baseline":
                continue
            comparison[row["variant"]] = {
                "match_signal_delta": row["mean_match_signal"] - baseline["mean_match_signal"],
                "token_f1_delta": row["mean_token_f1"] - baseline["mean_token_f1"],
                "token_precision_delta": row["mean_token_precision"] - baseline["mean_token_precision"],
                "token_recall_delta": row["mean_token_recall"] - baseline["mean_token_recall"],
                "readable_rate_delta": row["mean_readable_rate"] - baseline["mean_readable_rate"],
                "token_soup_rate_delta": row["token_soup_rate"] - baseline["token_soup_rate"],
                "token_count_delta": row["mean_token_count"] - baseline["mean_token_count"],
                "char_count_delta": row["mean_char_count"] - baseline["mean_char_count"],
            }
    ranking = sorted(
        summary_variants,
        key=lambda row: (
            -float(row["mean_match_signal"]),
            -float(row["mean_token_f1"]),
            -float(row["mean_readable_rate"]),
            float(row["token_soup_rate"]),
            float(row["mean_repetition_ratio"]),
            row["variant"],
        ),
    )
    return {
        "variants": summary_variants,
        "comparison": comparison,
        "ranking": ranking,
    }


def _write_decode_report(
    *,
    run_root: Path,
    summary: dict[str, Any],
    records: Sequence[DecodeVariantRecord],
) -> Path:
    lines = [
        "# MiniQA100 Decode Coherency Summary",
        "",
        f"Run root: `{run_root}`",
        "",
        "| variant | checkpoint | temp | top_p | top_k | beam | rep | no_repeat | match | token_f1 | readable | token_soup |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    record_lookup = {record.variant: record for record in records}
    for row in summary.get("variants", []):
        record = record_lookup.get(row["variant"])
        settings = record.settings if record else {}
        lines.append(
            "| {variant} | {checkpoint} | {temperature:.2f} | {top_p:.2f} | {top_k} | {beam_size} | {repetition_penalty:.2f} | {no_repeat_ngram_size} | {mean_match_signal:.4f} | {mean_token_f1:.4f} | {mean_readable_rate:.2f} | {token_soup_rate:.2f} |".format(
                checkpoint=(record.checkpoint_path if record else ""),
                temperature=float(settings.get("temperature", float("nan"))),
                top_p=float(settings.get("top_p", float("nan"))),
                top_k=int(settings.get("top_k", 0)),
                beam_size=int(settings.get("beam_size", 0)),
                repetition_penalty=float(settings.get("repetition_penalty", float("nan"))),
                no_repeat_ngram_size=int(settings.get("no_repeat_ngram_size", 0)),
                **row,
            )
        )
    if summary.get("comparison"):
        lines.extend(["", "## Comparison", ""])
        for variant, deltas in summary["comparison"].items():
            lines.append(f"### {variant}")
            for key, value in deltas.items():
                lines.append(f"- {key}: {value:.4f}")
    lines.extend(["", "## Runs", ""])
    for record in records:
        lines.append(f"- {record.variant}: {record.status} -> {record.save_dir}")
    report_path = run_root / "decode_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _run_decode_variant(
    *,
    source_data_path: Path,
    settings_source: str,
    checkpoint_path: Path,
    model: PrismalWaveModel,
    tokenizer: PrismalTokenizer,
    holdout_examples: Sequence[QAExample],
    variant: DecodeVariantSpec,
    run_root: Path,
    python_exe: str | None,
    device: str,
    dry_run: bool,
) -> DecodeVariantRecord:
    save_dir = run_root / variant.name
    save_dir.mkdir(parents=True, exist_ok=True)
    holdout_data_path = run_root / "data" / "holdout.jsonl"
    holdout_payloads = [_example_to_payload(example) for example in holdout_examples]
    _write_jsonl_records(holdout_data_path, holdout_payloads)
    settings = _decode_variant_spec_to_dict(variant)
    executable = python_exe or sys.executable
    command = [
        executable,
        str(ROOT / "tiny_training_matrix.py"),
        "decode",
        "--checkpoint",
        str(checkpoint_path),
        "--data",
        str(source_data_path),
        "--holdout-count",
        str(len(holdout_examples)),
        "--variant",
        variant.name,
    ]
    if dry_run:
        print(" ".join(command))
        return DecodeVariantRecord(
            variant=variant.name,
            checkpoint_path=str(checkpoint_path),
            settings_source=settings_source,
            save_dir=str(save_dir),
            source_data_path=str(source_data_path),
            holdout_data_path=str(holdout_data_path),
            status="dry-run",
            command=command,
            settings=settings,
            holdout_examples=[asdict(example) for example in holdout_examples],
        )

    eval_records: list[dict[str, Any]] = []
    for example in holdout_examples:
        prompt = _format_miniqa100_prompt(example)
        generated_text = generate_text(
            model,
            tokenizer,
            prompt,
            resolve_device(device),
            max_new_tokens=variant.max_new_tokens,
            min_new_tokens=variant.min_new_tokens,
            top_k=variant.top_k,
            top_p=variant.top_p,
            temperature=variant.temperature,
            repetition_penalty=variant.repetition_penalty,
            no_repeat_ngram_size=variant.no_repeat_ngram_size,
            beam_size=variant.beam_size,
            speculative_draft_tokens=variant.speculative_draft_tokens,
            speculative_temperature=variant.speculative_temperature,
            template_prompt=variant.template_prompt,
        )
        generated_answer = _extract_generated_answer(generated_text)
        score = _score_qa_generation(reference=example.response, generated=generated_answer)
        row = {
            "variant": variant.name,
            "index": example.index,
            "prompt": prompt,
            "reference": example.response,
            "generated_answer": generated_answer,
            "generated_text": generated_text,
            **score,
            "match_signal": max(1.0 if score["exact_match"] or score["substring_match"] else 0.0, float(score["token_overlap"])),
        }
        eval_records.append(row)

    summary = {
        "variant": variant.name,
        "num_examples": len(eval_records),
        "mean_match_signal": fmean([float(row["match_signal"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_token_f1": fmean([float(row["token_overlap"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_token_precision": fmean([float(row["token_precision"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_token_recall": fmean([float(row["token_recall"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_unique_token_ratio": fmean([float(row["unique_token_ratio"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_repetition_ratio": fmean([float(row["repetition_ratio"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_token_count": fmean([float(row["token_count"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_char_count": fmean([float(row["char_count"]) for row in eval_records]) if eval_records else float("nan"),
        "readable_rate": fmean([1.0 if row["readable"] else 0.0 for row in eval_records]) if eval_records else float("nan"),
        "token_soup_rate": fmean([1.0 if row["token_soup"] else 0.0 for row in eval_records]) if eval_records else float("nan"),
    }

    _write_jsonl_records(save_dir / "decode_generations.jsonl", eval_records)
    (save_dir / "decode_eval.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return DecodeVariantRecord(
        variant=variant.name,
        checkpoint_path=str(checkpoint_path),
        settings_source=settings_source,
        save_dir=str(save_dir),
        source_data_path=str(source_data_path),
        holdout_data_path=str(holdout_data_path),
        status="ok",
        command=command,
        settings=settings,
        holdout_examples=[asdict(example) for example in holdout_examples],
        generations=eval_records,
        eval_summary=summary,
    )


def _run_miniqa100_decode(
    *,
    run_root: Path,
    checkpoint_path: Path,
    source_data_path: Path,
    holdout_count: int,
    python_exe: str | None,
    device: str,
    dry_run: bool,
    variant_names: Sequence[str] | None = None,
) -> list[DecodeVariantRecord]:
    _ensure_dataset_paths((DatasetSpec("miniqa100", source_data_path, holdout_count),))
    resolved_checkpoint = _resolve_checkpoint_path(checkpoint_path)
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint path: {resolved_checkpoint}")
    run_root.mkdir(parents=True, exist_ok=True)
    examples = _load_miniqa100_examples(source_data_path)
    _, holdout_examples, holdout_indices = _split_miniqa100_examples(examples, holdout_count)
    selected_names = [str(name) for name in variant_names] if variant_names else [variant.name for variant in DECODE_VARIANTS]
    variant_lookup = {variant.name: variant for variant in DECODE_VARIANTS}
    missing = [name for name in selected_names if name not in variant_lookup]
    if missing:
        raise ValueError(f"Unknown decode variant name(s): {', '.join(missing)}")
    selected_variants = [variant_lookup[name] for name in selected_names]
    model, tokenizer, _cfg = load_bundle_from_checkpoint(resolved_checkpoint, device=resolve_device(device))
    records = [
        _run_decode_variant(
            source_data_path=source_data_path,
            settings_source="decode.default",
            checkpoint_path=resolved_checkpoint,
            model=model,
            tokenizer=tokenizer,
            holdout_examples=holdout_examples,
            variant=variant,
            run_root=run_root,
            python_exe=python_exe,
            device=device,
            dry_run=dry_run,
        )
        for variant in selected_variants
    ]
    summary = {
        "settings_source": "decode.default",
        "checkpoint_path": str(resolved_checkpoint),
        "source_data_path": str(source_data_path),
        "holdout_examples": len(holdout_examples),
        "holdout_indices": holdout_indices,
        "records": [asdict(record) for record in records],
        "summary": _decode_variant_specs_to_summary(records),
    }
    (run_root / "decode_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_decode_report(run_root=run_root, summary=summary["summary"], records=records)
    return records


def _run_smoke_variant(
    *,
    source_data_path: Path,
    settings_source: str,
    train_examples: Sequence[QAExample],
    holdout_examples: Sequence[QAExample],
    variant: VariantSpec,
    baseline_extra_args: Sequence[str],
    run_root: Path,
    python_exe: str | None,
    batch_size: int,
    max_samples: int,
    gradient_accumulation_steps: int | None,
    device: str,
    dry_run: bool,
    artifact_prefix: str = "smoke",
) -> SmokeVariantRecord:
    save_dir = run_root / variant.name
    save_dir.mkdir(parents=True, exist_ok=True)
    train_data_path = run_root / "data" / "train.jsonl"
    holdout_data_path = run_root / "data" / "holdout.jsonl"
    train_mode = "fresh-train"
    train_payloads = [_example_to_payload(example) for example in train_examples]
    holdout_payloads = [_example_to_payload(example) for example in holdout_examples]
    _write_jsonl_records(train_data_path, train_payloads)
    _write_jsonl_records(holdout_data_path, holdout_payloads)
    dataset = DatasetSpec("miniqa100", train_data_path, max_samples)
    extra_args = list(baseline_extra_args)
    if gradient_accumulation_steps is not None:
        extra_args.extend(["--gradient-accumulation-steps", str(gradient_accumulation_steps)])
    command = _build_train_command(
        dataset=dataset,
        variant=variant,
        save_dir=save_dir,
        batch_size=batch_size,
        max_samples=max_samples,
        extra_args=extra_args,
        post_prompt=None,
        python_exe=python_exe,
    )

    if dry_run:
        print(" ".join(command))
        return SmokeVariantRecord(
            variant=variant.name,
            settings_source=settings_source,
            save_dir=str(save_dir),
            source_data_path=str(source_data_path),
            train_data_path=str(train_data_path),
            holdout_data_path=str(holdout_data_path),
            train_mode=train_mode,
            status="dry-run",
            command=command,
            holdout_examples=[asdict(example) for example in holdout_examples],
        )

    status = "ok"
    error: str | None = None
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        status = "failed"
        error = f"returncode={exc.returncode}"

    train_metrics = _read_json(save_dir / "metrics.json")
    train_config = _read_json(save_dir / "config.json")
    if status != "ok":
        return SmokeVariantRecord(
            variant=variant.name,
            settings_source=settings_source,
            save_dir=str(save_dir),
            source_data_path=str(source_data_path),
            train_data_path=str(train_data_path),
            holdout_data_path=str(holdout_data_path),
            train_mode=train_mode,
            status=status,
            command=command,
            train_metrics=train_metrics,
            train_config=train_config,
            holdout_examples=[asdict(example) for example in holdout_examples],
            error=error,
        )

    checkpoint_file = _resolve_checkpoint_path(save_dir)
    model, tokenizer, _cfg = load_bundle_from_checkpoint(checkpoint_file, device=resolve_device(device))
    eval_records: list[dict[str, Any]] = []
    eval_scores: list[dict[str, Any]] = []
    for example in holdout_examples:
        prompt = _format_miniqa100_prompt(example)
        generated_text = generate_text(
            model,
            tokenizer,
            prompt,
            resolve_device(device),
            max_new_tokens=DEFAULT_SMOKE_MAX_NEW_TOKENS,
            min_new_tokens=DEFAULT_SMOKE_MIN_NEW_TOKENS,
            top_k=DEFAULT_SMOKE_TOP_K,
            top_p=DEFAULT_SMOKE_TOP_P,
            temperature=DEFAULT_SMOKE_TEMPERATURE,
            repetition_penalty=DEFAULT_SMOKE_REPETITION_PENALTY,
            no_repeat_ngram_size=DEFAULT_SMOKE_NO_REPEAT_NGRAM_SIZE,
            beam_size=DEFAULT_SMOKE_BEAM_SIZE,
            template_prompt=False,
        )
        generated_answer = _extract_generated_answer(generated_text)
        score = _score_qa_generation(reference=example.response, generated=generated_answer)
        row = {
            "variant": variant.name,
            "index": example.index,
            "prompt": prompt,
            "reference": example.response,
            "generated_answer": generated_answer,
            "generated_text": generated_text,
            **score,
            "match_signal": max(1.0 if score["exact_match"] or score["substring_match"] else 0.0, float(score["token_overlap"])),
        }
        eval_records.append(row)
        eval_scores.append(row)

    summary = {
        "variant": variant.name,
        "num_examples": len(eval_records),
        "mean_match_signal": fmean([float(row["match_signal"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_token_f1": fmean([float(row["token_overlap"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_token_precision": fmean([float(row["token_precision"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_token_recall": fmean([float(row["token_recall"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_unique_token_ratio": fmean([float(row["unique_token_ratio"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_repetition_ratio": fmean([float(row["repetition_ratio"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_token_count": fmean([float(row["token_count"]) for row in eval_records]) if eval_records else float("nan"),
        "mean_char_count": fmean([float(row["char_count"]) for row in eval_records]) if eval_records else float("nan"),
        "readable_rate": fmean([1.0 if row["readable"] else 0.0 for row in eval_records]) if eval_records else float("nan"),
        "token_soup_rate": fmean([1.0 if row["token_soup"] else 0.0 for row in eval_records]) if eval_records else float("nan"),
    }

    _write_jsonl_records(save_dir / f"{artifact_prefix}_generations.jsonl", eval_records)
    (save_dir / f"{artifact_prefix}_eval.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return SmokeVariantRecord(
        variant=variant.name,
        settings_source=settings_source,
        save_dir=str(save_dir),
        source_data_path=str(source_data_path),
        train_data_path=str(train_data_path),
        holdout_data_path=str(holdout_data_path),
        train_mode=train_mode,
        status=status,
        command=command,
        train_metrics=train_metrics,
        train_config=train_config,
        holdout_examples=[asdict(example) for example in holdout_examples],
        generations=eval_records,
        eval_summary=summary,
        error=error,
    )


def _run_miniqa100_smoke(
    *,
    run_root: Path,
    batch_size: int,
    max_samples: int,
    gradient_accumulation_steps: int | None,
    holdout_count: int,
    source_data_path: Path,
    python_exe: str | None,
    device: str,
    dry_run: bool,
) -> list[SmokeVariantRecord]:
    _ensure_dataset_paths((DatasetSpec("miniqa100", source_data_path, max_samples),))
    run_root.mkdir(parents=True, exist_ok=True)
    examples = _load_miniqa100_examples(source_data_path)
    train_examples, holdout_examples, holdout_indices = _split_miniqa100_examples(examples, holdout_count)
    default_config = _miniqa100_default_config()
    baseline_extra_args = _miniqa100_baseline_extra_args()
    smoke_variants = (
        VariantSpec("baseline", ()),
    )
    records = [
        _run_smoke_variant(
            source_data_path=source_data_path,
            settings_source="PrismalWaveConfig.defaults",
            train_examples=train_examples,
            holdout_examples=holdout_examples,
            variant=variant,
            baseline_extra_args=baseline_extra_args,
            run_root=run_root,
            python_exe=python_exe,
            batch_size=batch_size,
            max_samples=max_samples,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
            dry_run=dry_run,
        )
        for variant in smoke_variants
    ]
    summary = {
        "settings_source": "PrismalWaveConfig.defaults",
        "default_config": default_config,
        "source_data_path": str(source_data_path),
        "train_examples": len(train_examples),
        "holdout_examples": len(holdout_examples),
        "holdout_indices": holdout_indices,
        "records": [asdict(record) for record in records],
        "summary": _smoke_variant_records_to_summary(records),
    }
    (run_root / "smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_smoke_report(run_root=run_root, summary=summary["summary"], records=records, report_name="smoke_report.md")
    return records


def _run_miniqa100_coherency(
    *,
    run_root: Path,
    batch_size: int,
    max_samples: int,
    holdout_count: int,
    source_data_path: Path,
    python_exe: str | None,
    device: str,
    dry_run: bool,
) -> list[SmokeVariantRecord]:
    _ensure_dataset_paths((DatasetSpec("miniqa100", source_data_path, max_samples),))
    run_root.mkdir(parents=True, exist_ok=True)
    examples = _load_miniqa100_examples(source_data_path)
    train_examples, holdout_examples, holdout_indices = _split_miniqa100_examples(examples, holdout_count)
    default_config = _miniqa100_default_config()
    coherency_variants = COHERENCY_VARIANTS
    records = [
        _run_smoke_variant(
            source_data_path=source_data_path,
            settings_source="PrismalWaveConfig.defaults",
            train_examples=train_examples,
            holdout_examples=holdout_examples,
            variant=variant,
            baseline_extra_args=(),
            run_root=run_root,
            python_exe=python_exe,
            batch_size=batch_size,
            max_samples=max_samples,
            device=device,
            dry_run=dry_run,
            artifact_prefix="coherency",
        )
        for variant in coherency_variants
    ]
    summary = {
        "settings_source": "PrismalWaveConfig.defaults",
        "default_config": default_config,
        "source_data_path": str(source_data_path),
        "train_examples": len(train_examples),
        "holdout_examples": len(holdout_examples),
        "holdout_indices": holdout_indices,
        "records": [asdict(record) for record in records],
        "summary": _smoke_variant_records_to_summary(records),
    }
    (run_root / "coherency_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_smoke_report(run_root=run_root, summary=summary["summary"], records=records, report_name="coherency_report.md", title="MiniQA100 Router/Lattice Coherency Summary")
    return records


def _run_miniqa100_tiers(
    *,
    run_root: Path,
    batch_size: int,
    max_samples: int,
    holdout_count: int,
    source_data_path: Path,
    python_exe: str | None,
    device: str,
    dry_run: bool,
) -> list[SmokeVariantRecord]:
    _ensure_dataset_paths((DatasetSpec("miniqa100", source_data_path, max_samples),))
    run_root.mkdir(parents=True, exist_ok=True)
    examples = _load_miniqa100_examples(source_data_path)
    train_examples, holdout_examples, holdout_indices = _split_miniqa100_examples(examples, holdout_count)
    default_config = _miniqa100_default_config()
    tier_variants = _miniqa100_tier_variants()
    records = [
        _run_smoke_variant(
            source_data_path=source_data_path,
            settings_source="PrismalWaveConfig.defaults",
            train_examples=train_examples,
            holdout_examples=holdout_examples,
            variant=variant,
            baseline_extra_args=(),
            run_root=run_root,
            python_exe=python_exe,
            batch_size=batch_size,
            max_samples=max_samples,
            device=device,
            dry_run=dry_run,
            artifact_prefix="tiers",
        )
        for variant in tier_variants
    ]
    summary = {
        "settings_source": "PrismalWaveConfig.defaults",
        "default_config": default_config,
        "source_data_path": str(source_data_path),
        "train_examples": len(train_examples),
        "holdout_examples": len(holdout_examples),
        "holdout_indices": holdout_indices,
        "records": [asdict(record) for record in records],
        "summary": _smoke_variant_records_to_summary(records),
    }
    (run_root / "tiers_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_smoke_report(
        run_root=run_root,
        summary=summary["summary"],
        records=records,
        report_name="tiers_report.md",
        title="MiniQA100 Hierarchical Tier Sweep Summary",
    )
    return records


def _run_miniqa100_residency(
    *,
    run_root: Path,
    batch_size: int,
    max_samples: int,
    holdout_count: int,
    source_data_path: Path,
    python_exe: str | None,
    device: str,
    dry_run: bool,
) -> list[SmokeVariantRecord]:
    _ensure_dataset_paths((DatasetSpec("miniqa100", source_data_path, max_samples),))
    run_root.mkdir(parents=True, exist_ok=True)
    examples = _load_miniqa100_examples(source_data_path)
    train_examples, holdout_examples, holdout_indices = _split_miniqa100_examples(examples, holdout_count)
    default_config = _miniqa100_default_config()
    records = [
        _run_smoke_variant(
            source_data_path=source_data_path,
            settings_source="PrismalWaveConfig.defaults",
            train_examples=train_examples,
            holdout_examples=holdout_examples,
            variant=variant,
            baseline_extra_args=(),
            run_root=run_root,
            python_exe=python_exe,
            batch_size=batch_size,
            max_samples=max_samples,
            device=device,
            dry_run=dry_run,
            artifact_prefix="residency",
        )
        for variant in RESIDENCY_VARIANTS
    ]
    summary = {
        "settings_source": "PrismalWaveConfig.defaults",
        "default_config": default_config,
        "source_data_path": str(source_data_path),
        "train_examples": len(train_examples),
        "holdout_examples": len(holdout_examples),
        "holdout_indices": holdout_indices,
        "records": [asdict(record) for record in records],
        "summary": _smoke_variant_records_to_summary(records),
    }
    (run_root / "residency_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_smoke_report(
        run_root=run_root,
        summary=summary["summary"],
        records=records,
        report_name="residency_report.md",
        title="MiniQA100 Residency and Bus Sweep Summary",
    )
    return records


def _run_stage(
    *,
    stage: str,
    datasets: Sequence[DatasetSpec],
    variants: Sequence[VariantSpec],
    run_root: Path,
    batch_size: int,
    max_samples: int,
    python_exe: str | None = None,
    dry_run: bool = False,
) -> list[RunRecord]:
    _ensure_dataset_paths(datasets)
    records: list[RunRecord] = []
    for dataset in datasets:
        for variant in variants:
            records.append(
                _run_training(
                    stage=stage,
                    dataset=dataset,
                    variant=variant,
                    run_root=run_root,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    python_exe=python_exe,
                    dry_run=dry_run,
                )
            )
    _write_summary(stage, run_root, records)
    return records


def _run_compare_generation(
    *,
    checkpoints: Sequence[Path],
    prompts: Sequence[str],
    output_root: Path,
    device: str = "auto",
    max_new_tokens: int = 32,
    min_new_tokens: int = 1,
    top_k: int = 8,
    top_p: float = 0.92,
    temperature: float = 0.15,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 4,
    beam_size: int = 2,
    speculative_draft_tokens: int = 2,
    speculative_temperature: float = 0.0,
    template_prompt: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        command = [
            sys.executable,
            str(COMPARE_PATH),
            *[str(checkpoint) for checkpoint in checkpoints],
            "--prompt",
            prompt,
            "--device",
            device,
            "--max-new-tokens",
            str(max_new_tokens),
            "--min-new-tokens",
            str(min_new_tokens),
            "--top-k",
            str(top_k),
            "--top-p",
            str(top_p),
            "--temperature",
            str(temperature),
            "--repetition-penalty",
            str(repetition_penalty),
            "--no-repeat-ngram-size",
            str(no_repeat_ngram_size),
            "--beam-size",
            str(beam_size),
            "--speculative-draft-tokens",
            str(speculative_draft_tokens),
            "--speculative-temperature",
            str(speculative_temperature),
            "--json",
        ]
        command.append("--template-prompt" if template_prompt else "--no-template-prompt")
        if dry_run:
            print(" ".join(command))
            results.append({"prompt": prompt, "status": "dry-run", "command": command})
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        result_payload: dict[str, Any] = {
            "prompt": prompt,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.stdout:
            try:
                result_payload["json"] = json.loads(completed.stdout)
            except json.JSONDecodeError:
                pass
        results.append(result_payload)
    (output_root / "compare_generation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _run_checkpoint_benchmark(
    *,
    checkpoint: Path,
    data_path: Path,
    device: str,
    steps: int,
    batch_size: int,
    max_samples: int,
    seq_len: int = 0,
) -> dict[str, Any]:
    model, tokenizer, _raw_cfg = load_bundle_from_checkpoint(checkpoint, device=resolve_device(device))
    benchmark_device = next(model.parameters()).device
    dataloader = build_dataloader(
        data_path,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        max_samples=max_samples,
        shuffle=False,
    )
    return run_benchmark(model, dataloader, benchmark_device, steps=steps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tiny routing-stability training matrices.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT, help="Directory for matrix run artifacts")
    common.add_argument("--python", default="", help="Python executable to use for training subprocesses")
    common.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    common.add_argument("--device", default="auto", help="Device to use for smoke evaluation")

    screen = sub.add_parser("screen", parents=[common], help="Run the stage-1 screen on the smallest datasets")
    screen.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    screen.add_argument("--max-samples", type=int, default=DEFAULT_STAGE1_MAX_SAMPLES)

    confirm = sub.add_parser("confirm", parents=[common], help="Run the stage-2 confirmation pass on the broader datasets")
    confirm.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    confirm.add_argument("--max-samples", type=int, default=DEFAULT_STAGE2_MAX_SAMPLES)
    confirm.add_argument("--variants", nargs="*", default=[])
    confirm.add_argument("--from-summary", type=Path, default=None, help="screen_summary.json path to rank from")
    confirm.add_argument("--top", type=int, default=2, help="Number of variants to select when ranking from a summary")

    followup = sub.add_parser("followup", parents=[common], help="Run compare-generation and benchmark checks on winner checkpoints")
    followup.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    followup.add_argument("--prompts", nargs="*", default=list(DEFAULT_COMPARE_PROMPTS))
    followup.add_argument("--benchmark-data", type=Path, default=DEFAULT_BENCHMARK_DATA)
    followup.add_argument("--benchmark-steps", type=int, default=DEFAULT_BENCHMARK_STEPS)
    followup.add_argument("--benchmark-batch-size", type=int, default=DEFAULT_BENCHMARK_BATCH_SIZE)
    followup.add_argument("--benchmark-max-samples", type=int, default=DEFAULT_BENCHMARK_MAX_SAMPLES)
    followup.add_argument("--benchmark-device", default="auto")
    followup.add_argument("--benchmark-seq-len", type=int, default=0)

    smoke = sub.add_parser("smoke", parents=[common], help="Run the miniqa100 coherence smoke test")
    smoke.set_defaults(run_root=DEFAULT_SMOKE_RUN_ROOT)
    smoke.add_argument("--data", type=Path, default=DEFAULT_MINIQA100_DATA)
    smoke.add_argument("--batch-size", type=int, default=DEFAULT_SMOKE_BATCH_SIZE)
    smoke.add_argument("--max-samples", type=int, default=DEFAULT_SMOKE_MAX_SAMPLES)
    smoke.add_argument("--gradient-accumulation-steps", type=int, default=None)
    smoke.add_argument("--holdout-count", type=int, default=DEFAULT_SMOKE_HOLDOUT_COUNT)

    coherency = sub.add_parser("coherency", parents=[common], help="Run the miniqa100 signature-lattice/router-temperature training sweep")
    coherency.set_defaults(run_root=DEFAULT_COHERENCY_RUN_ROOT)
    coherency.add_argument("--data", type=Path, default=DEFAULT_MINIQA100_DATA)
    coherency.add_argument("--batch-size", type=int, default=DEFAULT_SMOKE_BATCH_SIZE)
    coherency.add_argument("--max-samples", type=int, default=DEFAULT_COHERENCY_MAX_SAMPLES)
    coherency.add_argument("--holdout-count", type=int, default=DEFAULT_SMOKE_HOLDOUT_COUNT)

    tiers = sub.add_parser("tiers", parents=[common], help="Run the miniqa100 hierarchical tier-weight sweep")
    tiers.set_defaults(run_root=DEFAULT_TIERS_RUN_ROOT)
    tiers.add_argument("--data", type=Path, default=DEFAULT_MINIQA100_DATA)
    tiers.add_argument("--batch-size", type=int, default=DEFAULT_SMOKE_BATCH_SIZE)
    tiers.add_argument("--max-samples", type=int, default=DEFAULT_TIERS_MAX_SAMPLES)
    tiers.add_argument("--holdout-count", type=int, default=DEFAULT_SMOKE_HOLDOUT_COUNT)

    residency = sub.add_parser("residency", parents=[common], help="Run the miniqa100 learned residency and bus sweep")
    residency.set_defaults(run_root=DEFAULT_RESIDENCY_RUN_ROOT)
    residency.add_argument("--data", type=Path, default=DEFAULT_MINIQA100_DATA)
    residency.add_argument("--batch-size", type=int, default=DEFAULT_SMOKE_BATCH_SIZE)
    residency.add_argument("--max-samples", type=int, default=DEFAULT_RESIDENCY_MAX_SAMPLES)
    residency.add_argument("--holdout-count", type=int, default=DEFAULT_SMOKE_HOLDOUT_COUNT)

    decode = sub.add_parser("decode", parents=[common], help="Run decode-only coherency sweeps on the miniqa100 baseline checkpoint")
    decode.set_defaults(run_root=DEFAULT_DECODE_RUN_ROOT)
    decode.add_argument("--data", type=Path, default=DEFAULT_MINIQA100_DATA)
    decode.add_argument("--checkpoint", type=Path, default=DEFAULT_MINIQA100_BASELINE_CHECKPOINT)
    decode.add_argument("--holdout-count", type=int, default=DEFAULT_SMOKE_HOLDOUT_COUNT)
    decode.add_argument("--variants", nargs="*", default=[])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_root = args.run_root
    python_exe = args.python.strip() or None
    dry_run = bool(args.dry_run)
    run_root.mkdir(parents=True, exist_ok=True)

    if args.command == "screen":
        _run_stage(
            stage="screen",
            datasets=STAGE1_DATASETS,
            variants=MATRIX_VARIANTS,
            run_root=run_root,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
            python_exe=python_exe,
            dry_run=dry_run,
        )
        return 0

    if args.command == "confirm":
        if args.variants:
            selected_variants = [str(item) for item in args.variants]
        elif args.from_summary is not None:
            selected_variants = _select_top_variants(args.from_summary, int(args.top))
        else:
            raise ValueError("confirm requires either --variants or --from-summary")

        variant_lookup = {variant.name: variant for variant in MATRIX_VARIANTS}
        missing = [variant for variant in selected_variants if variant not in variant_lookup]
        if missing:
            raise ValueError(f"Unknown variant name(s): {', '.join(missing)}")
        chosen_variants = [variant_lookup[name] for name in selected_variants]
        _run_stage(
            stage="confirm",
            datasets=STAGE2_DATASETS,
            variants=chosen_variants,
            run_root=run_root,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
            python_exe=python_exe,
            dry_run=dry_run,
        )
        return 0

    if args.command == "followup":
        checkpoints = [_resolve_checkpoint_path(path) for path in args.checkpoints]
        followup_root = run_root / "followup"
        compare_results = _run_compare_generation(
            checkpoints=checkpoints,
            prompts=list(args.prompts),
            output_root=followup_root,
            dry_run=dry_run,
        )

        benchmark_metrics: dict[str, Any] = {"dry_run": True} if dry_run else {}
        if not dry_run:
            _ensure_dataset_paths((DatasetSpec(args.benchmark_data.name, args.benchmark_data, int(args.benchmark_max_samples)),))
            for checkpoint in checkpoints:
                benchmark_metrics[str(checkpoint)] = _run_checkpoint_benchmark(
                    checkpoint=checkpoint,
                    data_path=args.benchmark_data,
                    device=str(args.benchmark_device),
                    steps=int(args.benchmark_steps),
                    batch_size=int(args.benchmark_batch_size),
                    max_samples=int(args.benchmark_max_samples),
                    seq_len=int(args.benchmark_seq_len),
                )
            (followup_root / "benchmark_summary.json").write_text(
                json.dumps(benchmark_metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        (followup_root / "followup_summary.json").write_text(
            json.dumps({"compare": compare_results, "benchmark": benchmark_metrics}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return 0

    if args.command == "smoke":
        records = _run_miniqa100_smoke(
            run_root=run_root,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
            gradient_accumulation_steps=(
                int(args.gradient_accumulation_steps)
                if getattr(args, "gradient_accumulation_steps", None) is not None
                else None
            ),
            holdout_count=int(args.holdout_count),
            source_data_path=Path(args.data),
            python_exe=python_exe,
            device=str(args.device),
            dry_run=dry_run,
        )
        print(f"[matrix] smoke summary written to {run_root / 'smoke_summary.json'}")
        print(f"[matrix] smoke report written to {run_root / 'smoke_report.md'}")
        for record in records:
            print(f"[matrix] smoke/{record.variant}: {record.status} ({record.train_mode})")
        return 0

    if args.command == "coherency":
        records = _run_miniqa100_coherency(
            run_root=run_root,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
            holdout_count=int(args.holdout_count),
            source_data_path=Path(args.data),
            python_exe=python_exe,
            device=str(args.device),
            dry_run=dry_run,
        )
        print(f"[matrix] coherency summary written to {run_root / 'coherency_summary.json'}")
        print(f"[matrix] coherency report written to {run_root / 'coherency_report.md'}")
        for record in records:
            print(f"[matrix] coherency/{record.variant}: {record.status} ({record.train_mode})")
        return 0

    if args.command == "tiers":
        records = _run_miniqa100_tiers(
            run_root=run_root,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
            holdout_count=int(args.holdout_count),
            source_data_path=Path(args.data),
            python_exe=python_exe,
            device=str(args.device),
            dry_run=dry_run,
        )
        print(f"[matrix] tiers summary written to {run_root / 'tiers_summary.json'}")
        print(f"[matrix] tiers report written to {run_root / 'tiers_report.md'}")
        for record in records:
            print(f"[matrix] tiers/{record.variant}: {record.status} ({record.train_mode})")
        return 0

    if args.command == "residency":
        records = _run_miniqa100_residency(
            run_root=run_root,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
            holdout_count=int(args.holdout_count),
            source_data_path=Path(args.data),
            python_exe=python_exe,
            device=str(args.device),
            dry_run=dry_run,
        )
        print(f"[matrix] residency summary written to {run_root / 'residency_summary.json'}")
        print(f"[matrix] residency report written to {run_root / 'residency_report.md'}")
        for record in records:
            print(f"[matrix] residency/{record.variant}: {record.status} ({record.train_mode})")
        return 0

    if args.command == "decode":
        records = _run_miniqa100_decode(
            run_root=run_root,
            checkpoint_path=Path(args.checkpoint),
            source_data_path=Path(args.data),
            holdout_count=int(args.holdout_count),
            python_exe=python_exe,
            device=str(args.device),
            dry_run=dry_run,
            variant_names=[str(item) for item in args.variants] if args.variants else None,
        )
        print(f"[matrix] decode summary written to {run_root / 'decode_summary.json'}")
        print(f"[matrix] decode report written to {run_root / 'decode_report.md'}")
        for record in records:
            print(f"[matrix] decode/{record.variant}: {record.status}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
