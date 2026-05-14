# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

"""Training and benchmarking helpers for Prismal Torus."""

import math
import json
import hashlib
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from .config import PrismalWaveConfig, save_config
    from .data import (
        ByteTokenizer,
        PrismalTokenizer,
        MemmapTokenDataset,
        StreamingTextCorpusDataset,
        TextWindowDataset,
        build_collate_fn,
        _find_answer_start,
        load_text_corpus,
        iter_text_corpus,
        split_text_window_dataset,
    )
    from .muon_optim import (
        MuonAdamW,
        PrecisionAdaptiveHierarchicalOptimizer,
        _nested_learning_beta_for,
        _nested_learning_interval_for,
        split_muon_parameter_groups,
        split_precision_adaptive_parameter_groups,
    )
    from .hierarchical_precision import HierarchicalPrecisionPolicy
    from .hierarchical_precision import attach_precision_policy
    from .model import PrismalWaveModel
    from .quantization import refresh_quantized_caches
except ImportError:  # pragma: no cover - supports direct script launching.
    from config import PrismalWaveConfig, save_config
    from data import (
        ByteTokenizer,
        PrismalTokenizer,
        MemmapTokenDataset,
        StreamingTextCorpusDataset,
        TextWindowDataset,
        build_collate_fn,
        _find_answer_start,
        load_text_corpus,
        iter_text_corpus,
        split_text_window_dataset,
    )
    from muon_optim import (
        MuonAdamW,
        PrecisionAdaptiveHierarchicalOptimizer,
        _nested_learning_beta_for,
        _nested_learning_interval_for,
        split_muon_parameter_groups,
        split_precision_adaptive_parameter_groups,
    )
    from hierarchical_precision import HierarchicalPrecisionPolicy
    from hierarchical_precision import attach_precision_policy
    from model import PrismalWaveModel
    from quantization import refresh_quantized_caches


def _supervised_tokenizer_texts(texts: Iterable[str]) -> Iterable[str]:
    for text in texts:
        if not isinstance(text, str):
            continue
        answer_start = _find_answer_start(text)
        if answer_start is None:
            yield text
            continue
        answer_text = text[answer_start:].strip()
        if answer_text:
            yield answer_text


def _limit_iterable(items: Iterable[str], max_items: int) -> Iterable[str]:
    if max_items <= 0:
        yield from items
        return
    for index, item in enumerate(items):
        if index >= max_items:
            break
        yield item


def _tokenizer_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None and str(cache_dir).strip():
        return Path(cache_dir)
    return Path(tempfile.gettempdir()) / "epicsharcmohte_tokenizer_cache"


def _tokenizer_source_fingerprint(source: str | Path) -> Dict[str, object]:
    path = Path(source)
    resolved = str(path.resolve())
    if not path.exists():
        return {"kind": "missing", "path": resolved}
    if path.is_file():
        stat = path.stat()
        return {
            "kind": "file",
            "path": resolved,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    if path.is_dir():
        entries: list[dict[str, object]] = []
        for subpath in sorted(path.rglob("*")):
            if not subpath.is_file():
                continue
            suffix = subpath.suffix.lower()
            if suffix not in {".txt", ".md", ".markdown", ".rst", ".jsonl", ".parquet"}:
                continue
            stat = subpath.stat()
            entries.append(
                {
                    "path": str(subpath.relative_to(path)),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        return {"kind": "dir", "path": resolved, "entries": entries}
    stat = path.stat()
    return {
        "kind": "other",
        "path": resolved,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _resolve_tokenizer_workers(source: str | Path, requested_workers: int) -> int:
    return max(1, int(requested_workers) if int(requested_workers) > 0 else 1)


def _tokenizer_cache_key(
    base_tokenizer_fingerprint: str,
    source: str | Path,
    *,
    max_new_tokens: int,
    min_frequency: int,
    max_word_tokens: int,
    max_line_tokens: int,
    max_signature_tokens: int,
    max_source_samples: int,
    supervised_only: bool,
    use_pronunciation_signatures: bool,
) -> str:
    payload = {
        "cache_version": 1,
        "base_tokenizer": base_tokenizer_fingerprint,
        "source": _tokenizer_source_fingerprint(source),
        "settings": {
            "max_new_tokens": int(max_new_tokens),
            "min_frequency": int(min_frequency),
            "max_word_tokens": int(max_word_tokens),
            "max_line_tokens": int(max_line_tokens),
            "max_signature_tokens": int(max_signature_tokens),
            "max_source_samples": int(max_source_samples),
            "supervised_only": bool(supervised_only),
            "use_pronunciation_signatures": bool(use_pronunciation_signatures),
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tokenizer_state_fingerprint(tokenizer: PrismalTokenizer) -> str:
    blob = json.dumps(tokenizer.to_state_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tokenizer_cache_path(
    base_tokenizer_fingerprint: str,
    source: str | Path,
    *,
    cache_dir: str | Path | None,
    max_new_tokens: int,
    min_frequency: int,
    max_word_tokens: int,
    max_line_tokens: int,
    max_signature_tokens: int,
    max_source_samples: int,
    supervised_only: bool,
    use_pronunciation_signatures: bool,
) -> Path:
    cache_root = _tokenizer_cache_dir(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = _tokenizer_cache_key(
        base_tokenizer_fingerprint,
        source,
        max_new_tokens=max_new_tokens,
        min_frequency=min_frequency,
        max_word_tokens=max_word_tokens,
        max_line_tokens=max_line_tokens,
        max_signature_tokens=max_signature_tokens,
        max_source_samples=max_source_samples,
        supervised_only=supervised_only,
        use_pronunciation_signatures=use_pronunciation_signatures,
    )
    name = Path(source).stem or "source"
    return cache_root / f"{name}_{digest}.json"


@dataclass
class ProgressTracker:
    total_epochs: int
    batches_per_epoch: int
    total_batches: int
    streaming_mode: bool = False
    start_time: float = field(default_factory=time.perf_counter)
    last_print_time: float = field(default_factory=time.perf_counter)

    def should_print(self, epoch_idx: int, batch_idx: int, force: bool = False) -> bool:
        if force:
            return True
        if batch_idx in (1, self.batches_per_epoch):
            return True
        now = time.perf_counter()
        return (now - self.last_print_time) >= 10.0

    def emit(
        self,
        *,
        epoch_idx: int,
        batch_idx: int,
        completed_batches: int | None = None,
        total_loss: float,
        ce_loss: float,
        aux_loss: float,
        signature_agreement: float,
        entropy: float,
        raw_active_emitters: float,
        soft_active_emitters: float | None = None,
        effective_emitters: float | None = None,
        normalized_breadth: float | None = None,
        soft_breadth: float | None = None,
        balance_loss: float | None = None,
        emitter_cell_coverage: float | None = None,
        usage_entropy: float | None = None,
        usage_concentration: float | None = None,
        torus_coverage: float | None = None,
        stability_text: str = "",
        timings: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> None:
        if not self.should_print(epoch_idx, batch_idx, force=force):
            return
        now = time.perf_counter()
        done_batches = completed_batches if completed_batches is not None else (epoch_idx - 1) * self.batches_per_epoch + batch_idx
        elapsed = max(now - self.start_time, 1e-6)
        rate = done_batches / elapsed
        timing_text = _format_timing_breakdown(timings or {})
        usage_text = ""
        if usage_entropy is not None:
            usage_text = f" usage={usage_entropy:.4f}"
        if balance_loss is not None:
            usage_text += f" bal={balance_loss:.4f}"
        if torus_coverage is not None:
            usage_text += f" torus_cov={torus_coverage:.4f}"
        if normalized_breadth is not None:
            usage_text += f" cell_breadth={normalized_breadth:.4f}"
        if soft_active_emitters is not None:
            usage_text += f" soft_active={soft_active_emitters:.2f}"
        if soft_breadth is not None:
            usage_text += f" soft_cell_breadth={soft_breadth:.4f}"
        if usage_concentration is not None:
            usage_text += f" usage_conc={usage_concentration:.4f}"
        if effective_emitters is not None:
            usage_text += f" eff_count={effective_emitters:.2f}"
        if emitter_cell_coverage is not None:
            usage_text += f" cell_cov={emitter_cell_coverage:.4f}"
        stability_suffix = f" {stability_text}" if stability_text else ""
        if self.streaming_mode:
            print(
                f"[Prismal] stream epoch {epoch_idx} "
                f"batches_seen={done_batches} total={total_loss:.4f} ce={ce_loss:.4f} aux={aux_loss:.4f} "
                f"sig={signature_agreement:.4f} entropy={entropy:.4f} raw={raw_active_emitters:.2f} "
                f"rate={rate:.2f} batches/s{usage_text}{stability_suffix}{timing_text}",
                flush=True,
            )
            self.last_print_time = now
            return
        print(
            f"[Prismal] epoch {epoch_idx}/{self.total_epochs} "
            f"batch {batch_idx}/{self.batches_per_epoch} "
            f"({100.0 * done_batches / max(self.total_batches, 1):.1f}%) total={total_loss:.4f} ce={ce_loss:.4f} aux={aux_loss:.4f} "
            f"sig={signature_agreement:.4f} entropy={entropy:.4f} raw={raw_active_emitters:.2f} "
            f"rate={rate:.2f} batches/s eta={max(self.total_batches - done_batches, 0) / max(rate, 1e-6) / 60.0:.1f}m{usage_text}{stability_suffix}{timing_text}",
            flush=True,
        )
        self.last_print_time = now

    def estimate_rate(self, epoch_idx: int, batch_idx: int) -> tuple[float, float]:
        done_batches = (epoch_idx - 1) * self.batches_per_epoch + batch_idx
        elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        rate = done_batches / elapsed
        remaining = max(self.total_batches - done_batches, 0)
        return rate, remaining / max(rate, 1e-6)


def resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _unwrap_model(model: PrismalWaveModel) -> PrismalWaveModel:
    return getattr(model, "_orig_mod", model)


def _precision_context(
    precision_policy: HierarchicalPrecisionPolicy | None,
    device: torch.device,
    *,
    use_amp: bool,
):
    if precision_policy is not None:
        return precision_policy.training_context(device, enabled=bool(precision_policy.enabled))
    return torch.autocast("cuda", enabled=bool(use_amp)) if device.type == "cuda" else nullcontext()


def _amp_scaler_enabled(
    precision_policy: HierarchicalPrecisionPolicy | None,
    device: torch.device,
    *,
    use_amp: bool,
) -> bool:
    if not use_amp or device.type != "cuda":
        return False
    if precision_policy is None or not precision_policy.enabled:
        return True
    dtype_name = str(getattr(precision_policy, "root_compute_dtype", "")).strip().lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return False
    return True


def _apply_nested_optimizer_settings(optimizer: torch.optim.Optimizer, cfg: PrismalWaveConfig) -> None:
    if not isinstance(optimizer, PrecisionAdaptiveHierarchicalOptimizer):
        return

    nested_enabled = bool(getattr(cfg, "use_nested_learning", False))
    nest_depth = max(1, int(getattr(cfg, "hierarchical_nest_depth", 1)))
    local_interval = int(getattr(cfg, "nested_learning_local_interval", 1))
    mid_interval = int(getattr(cfg, "nested_learning_mid_interval", 2))
    global_interval = int(getattr(cfg, "nested_learning_global_interval", 4))
    local_beta = float(getattr(cfg, "nested_learning_local_ema_beta", 0.90))
    mid_beta = float(getattr(cfg, "nested_learning_mid_ema_beta", 0.95))
    global_beta = float(getattr(cfg, "nested_learning_global_ema_beta", 0.99))

    for group in optimizer.param_groups:
        tier = str(group.get("hierarchy_tier", "mid")).lower()
        group["nested_learning_enabled"] = nested_enabled
        group["nested_update_interval"] = int(
            _nested_learning_interval_for(
                tier,
                hierarchical_nest_depth=nest_depth,
                local_interval=local_interval,
                mid_interval=mid_interval,
                global_interval=global_interval,
            )
        )
        group["nested_update_ema_beta"] = float(
            _nested_learning_beta_for(
                tier,
                local_beta=local_beta,
                mid_beta=mid_beta,
                global_beta=global_beta,
            )
        )
        if nested_enabled and "_nested_group_step" not in group:
            group["_nested_group_step"] = 0


def _build_optimizer(
    model: PrismalWaveModel,
    *,
    optimizer_name: str,
    base_lr: float,
    cfg: PrismalWaveConfig,
    precision_policy: HierarchicalPrecisionPolicy | None = None,
) -> torch.optim.Optimizer:
    name = optimizer_name.strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=base_lr)
    if name == "hierarchical":
        param_groups = split_precision_adaptive_parameter_groups(
            model.named_parameters(),
            base_lr=float(base_lr),
            muon_lr=float(getattr(cfg, "muon_lr", 0.02)),
            muon_weight_decay=float(getattr(cfg, "muon_weight_decay", 0.01)),
            momentum_beta=float(getattr(cfg, "muon_momentum_beta", 0.95)),
            nesterov=bool(getattr(cfg, "muon_nesterov", True)),
            ns_steps=int(getattr(cfg, "muon_ns_steps", 5)),
            extra_scale_factor=float(getattr(cfg, "muon_extra_scale_factor", 1.0)),
            scalar_optimizer=str(getattr(cfg, "muon_scalar_optimizer", "adamw")),
            nested_learning_enabled=bool(getattr(cfg, "use_nested_learning", False)),
            hierarchical_nest_depth=int(getattr(cfg, "hierarchical_nest_depth", 1)),
            nested_learning_local_interval=int(getattr(cfg, "nested_learning_local_interval", 1)),
            nested_learning_mid_interval=int(getattr(cfg, "nested_learning_mid_interval", 2)),
            nested_learning_global_interval=int(getattr(cfg, "nested_learning_global_interval", 4)),
            nested_learning_local_ema_beta=float(getattr(cfg, "nested_learning_local_ema_beta", 0.90)),
            nested_learning_mid_ema_beta=float(getattr(cfg, "nested_learning_mid_ema_beta", 0.95)),
            nested_learning_global_ema_beta=float(getattr(cfg, "nested_learning_global_ema_beta", 0.99)),
            precision_policy=precision_policy,
        )
        if not param_groups:
            return torch.optim.AdamW(model.parameters(), lr=base_lr)
        optimizer = PrecisionAdaptiveHierarchicalOptimizer(param_groups)
        _apply_nested_optimizer_settings(optimizer, cfg)
        return optimizer
    if name != "muon":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    param_groups = split_muon_parameter_groups(
        model.named_parameters(),
        muon_lr=float(getattr(cfg, "muon_lr", 0.02)),
        scalar_lr=float(base_lr),
        muon_weight_decay=float(getattr(cfg, "muon_weight_decay", 0.01)),
        scalar_weight_decay=0.01,
        momentum_beta=float(getattr(cfg, "muon_momentum_beta", 0.95)),
        nesterov=bool(getattr(cfg, "muon_nesterov", True)),
        ns_steps=int(getattr(cfg, "muon_ns_steps", 5)),
        extra_scale_factor=float(getattr(cfg, "muon_extra_scale_factor", 1.0)),
        scalar_optimizer=str(getattr(cfg, "muon_scalar_optimizer", "adamw")),
    )
    if not param_groups:
        return torch.optim.AdamW(model.parameters(), lr=base_lr)
    return MuonAdamW(param_groups)


def maybe_compile_model(model: PrismalWaveModel, *, enabled: bool = True) -> PrismalWaveModel:
    if not enabled or not torch.cuda.is_available() or not hasattr(torch, "compile"):
        return model
    if hasattr(model, "_orig_mod"):
        return model
    try:
        print("[Prismal] compiling model with reduce-overhead...", flush=True)
        return torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        print(f"[Prismal] compile skipped: {exc}", flush=True)
        return model


def build_tokenizer_from_source(
    source: str | Path,
    *,
    max_new_tokens: int = 0,
    min_frequency: int = 2,
    max_word_tokens: int = 0,
    max_line_tokens: int = 0,
    max_signature_tokens: int = 0,
    max_source_samples: int = 0,
    supervised_only: bool = True,
    use_pronunciation_signatures: bool = True,
    tokenizer_workers: int = 1,
    tokenizer_cache_dir: str | Path | None = None,
    tokenizer: PrismalTokenizer | None = None,
) -> PrismalTokenizer:
    tokenizer = tokenizer or PrismalTokenizer(use_pronunciation_signatures=use_pronunciation_signatures)
    base_fingerprint = _tokenizer_state_fingerprint(tokenizer)
    cache_path = _tokenizer_cache_path(
        base_fingerprint,
        source,
        cache_dir=tokenizer_cache_dir,
        max_new_tokens=max_new_tokens,
        min_frequency=min_frequency,
        max_word_tokens=max_word_tokens,
        max_line_tokens=max_line_tokens,
        max_signature_tokens=max_signature_tokens,
        max_source_samples=max_source_samples,
        supervised_only=supervised_only,
        use_pronunciation_signatures=use_pronunciation_signatures,
    )
    if cache_path.exists():
        try:
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached_payload, dict):
                if cached_payload.get("base_tokenizer") != base_fingerprint:
                    raise ValueError("tokenizer base fingerprint mismatch")
                tokenizer_state = cached_payload.get("tokenizer_state")
                if isinstance(tokenizer_state, dict):
                    tokenizer = PrismalTokenizer.from_state_dict(tokenizer_state)
                    print(f"[Prismal] loaded tokenizer cache: {cache_path}", flush=True)
                    return tokenizer
        except Exception as exc:
            print(f"[Prismal] tokenizer cache ignored: {exc}", flush=True)

    texts = _limit_iterable(iter_text_corpus(source), max_source_samples)
    tokenizer_texts = _supervised_tokenizer_texts(texts) if supervised_only else texts
    tokenizer_workers = _resolve_tokenizer_workers(source, tokenizer_workers)
    tokenizer.learn_from_texts(
        tokenizer_texts,
        max_new_tokens=max_new_tokens,
        min_frequency=min_frequency,
        max_word_tokens=max_word_tokens,
        max_line_tokens=max_line_tokens,
        max_signature_tokens=max_signature_tokens,
        workers=tokenizer_workers,
    )
    try:
        cache_payload = {
            "cache_version": 1,
            "source": str(Path(source).resolve()),
            "base_tokenizer": base_fingerprint,
            "settings": {
                "max_new_tokens": int(max_new_tokens),
                "min_frequency": int(min_frequency),
                "max_word_tokens": int(max_word_tokens),
                "max_line_tokens": int(max_line_tokens),
                "max_signature_tokens": int(max_signature_tokens),
                "max_source_samples": int(max_source_samples),
                "supervised_only": bool(supervised_only),
                "use_pronunciation_signatures": bool(use_pronunciation_signatures),
            },
            "tokenizer_state": tokenizer.to_state_dict(),
        }
        cache_path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[Prismal] saved tokenizer cache: {cache_path}", flush=True)
    except Exception as exc:
        print(f"[Prismal] tokenizer cache save skipped: {exc}", flush=True)
    return tokenizer


def _find_jsonl_source(path: Path) -> Optional[Path]:
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return path
    if path.is_dir():
        jsonl_files = sorted(path.rglob("*.jsonl"))
        if len(jsonl_files) == 1:
            return jsonl_files[0]
    return None


def _find_pretokenized_root(source: str | Path) -> Optional[Path]:
    path = Path(source)
    if path.is_dir() and (path / "meta.json").exists():
        return path
    if path.is_file():
        # Respect explicit file selections. A sibling pretokenized bundle can be
        # useful when the user points at the dataset directory, but if they
        # selected a concrete JSONL/parquet file we should treat that as the raw
        # source and not silently swap in adjacent memmaps with a different vocab.
        return None
    candidates: list[Path] = []
    if path.is_dir():
        candidates.extend([path / "pretokenized", path / "pretokenized_windows"])
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "meta.json").exists():
            return candidate
    return None


def build_dataloader(
    texts_or_source: Sequence[str] | str | Path,
    tokenizer: PrismalTokenizer,
    seq_len: int,
    batch_size: int,
    max_samples: int = 1000,
    shuffle: bool = True,
) -> DataLoader:
    if isinstance(texts_or_source, (str, Path)):
        source = Path(texts_or_source)
        pretokenized_root = _find_pretokenized_root(source)
        if pretokenized_root is not None:
            dataset = MemmapTokenDataset(pretokenized_root, split="all", max_samples=max_samples)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
            )
        dataset = StreamingTextCorpusDataset(source, tokenizer, seq_len=seq_len, max_samples=max_samples, split="all")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
        )
    else:
        texts = texts_or_source
    dataset = TextWindowDataset(texts, tokenizer, seq_len=seq_len, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=len(dataset) >= batch_size,
        collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
    )


def build_train_val_dataloaders(
    texts_or_source: Sequence[str] | str | Path,
    tokenizer: ByteTokenizer,
    seq_len: int,
    batch_size: int,
    *,
    val_fraction: float = 0.1,
    max_samples: int = 1000,
    seed: int = 42,
    streaming: bool = True,
) -> tuple[DataLoader, DataLoader]:
    if isinstance(texts_or_source, (str, Path)):
        source = Path(texts_or_source)
        pretokenized_root = _find_pretokenized_root(source)
        if pretokenized_root is not None:
            train_dataset = MemmapTokenDataset(
                pretokenized_root,
                split="train",
                val_fraction=val_fraction,
                seed=seed,
                max_samples=max_samples,
            )
            val_dataset = MemmapTokenDataset(
                pretokenized_root,
                split="val",
                val_fraction=val_fraction,
                seed=seed,
                max_samples=max_samples,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=len(train_dataset) >= batch_size,
                collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
            )
            return train_loader, val_loader
        if not bool(streaming):
            texts = load_text_corpus(source)
            dataset = TextWindowDataset(texts, tokenizer, seq_len=seq_len, max_samples=max_samples)
            train_dataset, val_dataset = split_text_window_dataset(dataset, val_fraction=val_fraction, seed=seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=len(train_dataset) >= batch_size,
                collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
            )
            return train_loader, val_loader
        train_dataset = StreamingTextCorpusDataset(
            source,
            tokenizer,
            seq_len=seq_len,
            max_samples=max_samples,
            split="train",
            val_fraction=val_fraction,
            seed=seed,
        )
        val_dataset = StreamingTextCorpusDataset(
            source,
            tokenizer,
            seq_len=seq_len,
            max_samples=max_samples,
            split="val",
            val_fraction=val_fraction,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
        )
        return train_loader, val_loader
    else:
        texts = texts_or_source
    dataset = TextWindowDataset(texts, tokenizer, seq_len=seq_len, max_samples=max_samples)
    train_dataset, val_dataset = split_text_window_dataset(dataset, val_fraction=val_fraction, seed=seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= batch_size,
        collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=build_collate_fn(tokenizer.pad_id, tokenizer.signature_pad_id),
    )
    return train_loader, val_loader


def save_checkpoint(
    model: PrismalWaveModel,
    save_dir: str | Path,
    *,
    tokenizer: Optional[PrismalTokenizer] = None,
    config: Optional[PrismalWaveConfig] = None,
    metrics: Optional[Dict[str, float]] = None,
    training_state: Optional[Dict[str, object]] = None,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = _unwrap_model(model)
    # Save the actual runtime architecture. The raw CLI/config object can differ
    # after tokenizer resolution or resume loading, which makes size sweeps look
    # like they had no effect when inspecting checkpoints.
    cfg = PrismalWaveConfig.from_dict(model.cfg.to_dict())
    if config is not None:
        for field_name in (
            "optimizer",
            "lr",
            "use_bitsandbytes_leaf_precision",
            "bitsandbytes_leaf_precision_mode",
            "bitsandbytes_leaf_quant_type",
            "bitsandbytes_leaf_compute_dtype",
            "training_finite_guard_enabled",
            "inference_finite_guard_enabled",
            "grad_clip_muon",
            "grad_clip_scalar",
            "grad_clip_rowwise",
            "hierarchical_precision_enabled",
            "hierarchical_precision_root_dtype",
            "hierarchical_precision_mid_dtype",
            "hierarchical_precision_leaf_dtype",
            "hierarchical_precision_fallback_dtype",
            "hierarchical_precision_accumulator_dtype",
            "hierarchical_precision_allow_float8_leaf",
            "quantization_aware_training",
            "qat_start_fraction",
            "qat_ramp_fraction",
            "use_token_superposition_training",
            "token_superposition_bag_size",
            "token_superposition_phase_fraction",
            "muon_lr",
            "muon_weight_decay",
            "muon_momentum_beta",
            "muon_nesterov",
            "muon_ns_steps",
            "muon_extra_scale_factor",
            "muon_scalar_optimizer",
        ):
            if hasattr(config, field_name):
                setattr(cfg, field_name, getattr(config, field_name))
    saved_cfg = PrismalWaveConfig.from_dict(cfg.to_dict())
    saved_cfg.vocab_size = 0
    saved_cfg.signature_vocab_size = 0
    saved_cfg.signature_level_vocab_size = 0
    saved_cfg.signature_relation_vocab_size = 0
    saved_cfg.signature_bucket_vocab_size = 0
    state = model.state_dict()
    position_weight = state.get("position_embedding.embedding.weight")
    if isinstance(position_weight, torch.Tensor):
        saved_cfg.position_embedding_init_size = max(
            int(getattr(saved_cfg, "position_embedding_init_size", 0)),
            int(position_weight.shape[0]),
        )
    payload = {
        "model_state": state,
        "config": saved_cfg.to_dict(),
        "metrics": metrics or {},
    }
    checkpoint_training_state = training_state
    if checkpoint_training_state is None:
        checkpoint_training_state = getattr(model, "_prismal_training_state", None)
    if isinstance(checkpoint_training_state, dict) and checkpoint_training_state:
        payload["training_state"] = checkpoint_training_state
    if tokenizer is not None:
        payload["tokenizer_state"] = tokenizer.to_state_dict()
    ckpt_path = save_dir / "model.pt"
    torch.save(payload, ckpt_path)
    save_config(saved_cfg, save_dir / "config.json")
    if metrics is not None:
        (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return ckpt_path


def _accumulate_timings(bucket: Dict[str, float], route_stats: Dict[str, torch.Tensor]) -> None:
    for key, value in route_stats.items():
        if not key.startswith("timing_") or key.endswith("_total_ms") or key.endswith("_step_ms"):
            continue
        if torch.is_tensor(value):
            bucket[key] = bucket.get(key, 0.0) + float(value.detach().mean().item())
        else:
            bucket[key] = bucket.get(key, 0.0) + float(value)


def _extract_timings(route_stats: Dict[str, torch.Tensor], *, include_totals: bool = False) -> Dict[str, float]:
    timings: Dict[str, float] = {}
    for key, value in route_stats.items():
        if not key.startswith("timing_") or (not include_totals and (key.endswith("_total_ms") or key.endswith("_step_ms"))):
            continue
        if torch.is_tensor(value):
            timings[key] = float(value.detach().mean().item())
        else:
            timings[key] = float(value)
    return timings


def _format_timing_group(label: str, timings: Dict[str, float], keys: Sequence[tuple[str, str]]) -> tuple[str, set[str]]:
    parts: list[str] = []
    used: set[str] = set()
    for short_name, timing_key in keys:
        value = float(timings.get(timing_key, 0.0))
        if value <= 0.0:
            continue
        parts.append(f"{short_name}={value:.1f}ms")
        used.add(timing_key)
    if not parts:
        return "", used
    return f"{label}(" + " / ".join(parts) + ")", used


def _format_timing_breakdown(timings: Dict[str, float]) -> str:
    if not timings:
        return ""
    groups: list[str] = []
    used: set[str] = set()

    encode_group, encode_used = _format_timing_group(
        "encode",
        timings,
        [
            ("total", "timing_encode_ms"),
            ("embed", "timing_encode_embed_ms"),
            ("condition", "timing_encode_condition_ms"),
            ("registry", "timing_encode_registry_ms"),
        ],
    )
    if encode_group:
        groups.append(encode_group)
        used.update(encode_used)

    token_memory_group, token_memory_used = _format_timing_group(
        "token_memory",
        timings,
        [
            ("total", "timing_token_memory_total_ms"),
            ("query", "timing_token_memory_query_ms"),
            ("select", "timing_token_memory_select_ms"),
            ("project", "timing_token_memory_project_ms"),
            ("append", "timing_token_memory_append_ms"),
        ],
    )
    if token_memory_group:
        groups.append(token_memory_group)
        used.update(token_memory_used)

    path_group, path_used = _format_timing_group(
        "path",
        timings,
        [
            ("total", "timing_path_total_ms"),
            ("core", "timing_path_core_ms"),
            ("overlay", "timing_path_overlay_ms"),
            ("finalize", "timing_path_finalize_ms"),
            ("aggregate", "timing_path_aggregate_ms"),
            ("loop", "timing_path_loop_ms"),
            ("lane_select", "timing_lane_select_ms"),
        ],
    )
    if path_group:
        groups.append(path_group)
        used.update(path_used)

    residual = [
        (key, value)
        for key, value in timings.items()
        if key not in used and value > 0.0
    ]
    residual.sort(key=lambda item: item[1], reverse=True)
    if residual:
        tail = "other(" + " / ".join(f"{key.removeprefix('timing_')}={value:.1f}ms" for key, value in residual[:4]) + ")"
        groups.append(tail)

    return " " + " ".join(groups) if groups else ""


def _stat_value(route_stats: Dict[str, torch.Tensor], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = route_stats.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            return float(value.detach().mean().item())
        return float(value)
    return float(default)


def _checkpoint_mismatch_is_tolerable(missing: Sequence[str], unexpected: Sequence[str]) -> bool:
    tolerated_missing_exact = {
        "signature_embedding.weight",
    }
    tolerated_unexpected_exact = {
        "signature_parent_embedding.weight",
    }
    tolerated_missing_prefixes = (
        "learned_residency_head.",
        "router.",
    )
    tolerated_unexpected_prefixes = (
        "learned_residency_head.",
        "router.",
    )
    tolerated_quantization_suffixes = (
        ".absmax",
        ".quant_map",
        ".nested_absmax",
        ".nested_quant_map",
    )
    tolerated_recursive_prefixes = (
        "torus_core.recursive_",
        "recursive_",
    )

    def _is_tolerated(name: str, exact: set[str]) -> bool:
        if name in exact:
            return True
        if any(name.startswith(prefix) for prefix in tolerated_missing_prefixes):
            return True
        if any(name.startswith(prefix) for prefix in tolerated_unexpected_prefixes):
            return True
        if "quant_state.bitsandbytes__" in name:
            return True
        if any(name.endswith(suffix) for suffix in tolerated_quantization_suffixes):
            return True
        return any(name.startswith(prefix) for prefix in tolerated_recursive_prefixes)

    return all(_is_tolerated(name, tolerated_missing_exact) for name in missing) and all(
        _is_tolerated(name, tolerated_unexpected_exact) for name in unexpected
    )


def _optimizer_group_grad_clip_cap(group: dict, cfg: PrismalWaveConfig, fallback_grad_clip: float) -> float:
    group_name = str(group.get("parameter_role", "")).lower()
    update_rule = str(group.get("update_rule", "")).lower()
    if group.get("use_muon", False) or update_rule == "muon":
        cap = float(getattr(cfg, "grad_clip_muon", 0.0))
    elif update_rule == "rowwise" or group_name in {"table", "tensor"}:
        cap = float(getattr(cfg, "grad_clip_rowwise", 0.0))
    else:
        cap = float(getattr(cfg, "grad_clip_scalar", 0.0))
    if cap <= 0.0 and fallback_grad_clip > 0.0:
        return float(fallback_grad_clip)
    return max(0.0, float(cap))


def _optimizer_gradients_are_finite(optimizer: torch.optim.Optimizer) -> bool:
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if not isinstance(param, torch.Tensor) or param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                grad = grad.coalesce().values()
            if not torch.isfinite(grad).all():
                return False
    return True


def _clip_optimizer_group_gradients(
    optimizer: torch.optim.Optimizer,
    cfg: PrismalWaveConfig,
    *,
    fallback_grad_clip: float,
) -> tuple[int, float]:
    clipped_groups = 0
    total_norm = 0.0
    for group in optimizer.param_groups:
        params = [param for param in group.get("params", []) if isinstance(param, torch.Tensor) and param.grad is not None]
        if not params:
            continue
        cap = _optimizer_group_grad_clip_cap(group, cfg, fallback_grad_clip)
        if cap <= 0.0:
            continue
        clipped_groups += 1
        norm = torch.nn.utils.clip_grad_norm_(params, cap)
        total_norm += float(norm.detach().item() if torch.is_tensor(norm) else norm)
    return clipped_groups, total_norm


def _optimizer_state_is_shape_compatible(optimizer: torch.optim.Optimizer) -> bool:
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if not isinstance(param, torch.Tensor):
                continue
            state = optimizer.state.get(param)
            if not state:
                continue
            for value in state.values():
                if not torch.is_tensor(value):
                    continue
                if value.ndim == 0:
                    continue
                if value.shape != param.shape:
                    return False
    return True


def maybe_compile_model(model: PrismalWaveModel, *, enabled: bool = True) -> PrismalWaveModel:
    if not enabled or not torch.cuda.is_available() or not hasattr(torch, "compile"):
        return model
    if hasattr(model, "_orig_mod"):
        return model
    try:
        print("[Prismal] compiling model with reduce-overhead...", flush=True)
        return torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        print(f"[Prismal] compile skipped: {exc}", flush=True)
        return model


def resolve_runtime_config(cfg: PrismalWaveConfig, tokenizer: PrismalTokenizer) -> PrismalWaveConfig:
    runtime_cfg = PrismalWaveConfig.from_dict(cfg.to_dict())
    runtime_cfg.base_vocab_size = tokenizer.base_vocab_size
    if getattr(runtime_cfg, "use_hmote", False) or getattr(runtime_cfg, "use_recursive_hmoe", False):
        runtime_cfg.use_torus_core = True
    if runtime_cfg.vocab_size <= 0:
        runtime_cfg.vocab_size = tokenizer.vocab_size
    if runtime_cfg.signature_vocab_size <= 0:
        runtime_cfg.signature_vocab_size = tokenizer.signature_vocab_size
    if runtime_cfg.signature_level_vocab_size <= 0:
        runtime_cfg.signature_level_vocab_size = tokenizer.signature_level_vocab_size
    if runtime_cfg.signature_relation_vocab_size <= 0:
        runtime_cfg.signature_relation_vocab_size = tokenizer.signature_relation_vocab_size
    if runtime_cfg.signature_bucket_vocab_size <= 0:
        runtime_cfg.signature_bucket_vocab_size = max(8, tokenizer.signature_family_vocab_size)
    return runtime_cfg


def _token_superposition_phase_active(
    cfg: PrismalWaveConfig,
    *,
    resume_global_step: int,
    optimizer_step: int,
    scheduler_total_steps: int,
) -> bool:
    if not bool(getattr(cfg, "use_token_superposition_training", False)):
        return False
    bag_size = max(1, int(getattr(cfg, "token_superposition_bag_size", 1)))
    if bag_size <= 1:
        return False
    phase_fraction = float(getattr(cfg, "token_superposition_phase_fraction", 0.0))
    if phase_fraction <= 0.0:
        return False
    total_planned_steps = max(1, int(resume_global_step) + int(scheduler_total_steps))
    current_step = int(resume_global_step) + int(optimizer_step)
    return float(current_step) < phase_fraction * float(total_planned_steps)


def _infer_runtime_sizes_from_state(
    cfg: PrismalWaveConfig,
    state: Dict[str, torch.Tensor],
    tokenizer: Optional[PrismalTokenizer] = None,
) -> PrismalWaveConfig:
    """Recover dynamically grown vocab sizes from a checkpoint's tensor shapes."""
    runtime_cfg = PrismalWaveConfig.from_dict(cfg.to_dict())
    if tokenizer is not None:
        runtime_cfg.base_vocab_size = tokenizer.base_vocab_size

    def _first_dim(*keys: str) -> Optional[int]:
        for key in keys:
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return int(value.shape[0])
        return None

    vocab_weight = state.get("construction_head.weight")
    if isinstance(vocab_weight, torch.Tensor):
        runtime_cfg.vocab_size = max(int(getattr(runtime_cfg, "vocab_size", 0)), int(vocab_weight.shape[0]))
    elif tokenizer is not None and runtime_cfg.vocab_size <= 0:
        runtime_cfg.vocab_size = tokenizer.vocab_size

    family_dim = _first_dim(
        "registry.family_embedding.weight",
        "registry.family_embedding._base_weight_dense",
        "registry.family_embedding._base_weight_radii",
    )
    if family_dim is not None:
        runtime_cfg.signature_vocab_size = max(int(getattr(runtime_cfg, "signature_vocab_size", 0)), family_dim)
    elif tokenizer is not None and runtime_cfg.signature_vocab_size <= 0:
        runtime_cfg.signature_vocab_size = tokenizer.signature_vocab_size

    level_dim = _first_dim(
        "registry.level_embedding.weight",
        "registry.level_embedding._base_weight_dense",
        "registry.level_embedding._base_weight_radii",
    )
    if level_dim is not None:
        runtime_cfg.signature_level_vocab_size = max(int(getattr(runtime_cfg, "signature_level_vocab_size", 0)), level_dim)
    elif tokenizer is not None and runtime_cfg.signature_level_vocab_size <= 0:
        runtime_cfg.signature_level_vocab_size = tokenizer.signature_level_vocab_size

    relation_dim = _first_dim(
        "registry.relation_embedding.weight",
        "registry.relation_embedding._base_weight_dense",
        "registry.relation_embedding._base_weight_radii",
    )
    if relation_dim is not None:
        runtime_cfg.signature_relation_vocab_size = max(int(getattr(runtime_cfg, "signature_relation_vocab_size", 0)), relation_dim)
    elif tokenizer is not None and runtime_cfg.signature_relation_vocab_size <= 0:
        runtime_cfg.signature_relation_vocab_size = tokenizer.signature_relation_vocab_size

    bucket_dim = _first_dim(
        "signature_neighborhood_embedding.weight",
        "signature_neighborhood_embedding._base_weight_dense",
        "signature_neighborhood_embedding._base_weight_radii",
        "signature_token_head.weight",
        "signature_token_head._base_weight_dense",
        "signature_token_head._base_weight_radii",
    )
    if bucket_dim is not None:
        runtime_cfg.signature_bucket_vocab_size = max(int(getattr(runtime_cfg, "signature_bucket_vocab_size", 0)), bucket_dim)
    elif tokenizer is not None and runtime_cfg.signature_bucket_vocab_size <= 0:
        runtime_cfg.signature_bucket_vocab_size = max(8, tokenizer.signature_family_vocab_size)
    return runtime_cfg


def load_model_from_checkpoint(checkpoint_path: str | Path, device: str | torch.device | None = None) -> PrismalWaveModel:
    device_obj = resolve_device(device)
    payload = torch.load(checkpoint_path, map_location=device_obj)
    cfg_payload = payload.get("config")
    if not isinstance(cfg_payload, dict):
        raise ValueError("Checkpoint missing config.")
    cfg = PrismalWaveConfig.from_dict(cfg_payload)
    tokenizer_state = payload.get("tokenizer_state")
    if isinstance(tokenizer_state, dict):
        tokenizer = PrismalTokenizer.from_state_dict(tokenizer_state)
    else:
        tokenizer = PrismalTokenizer()
    tokenizer.refresh_construction_index()
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint missing model_state.")
    position_weight = state.get("position_embedding.embedding.weight")
    if isinstance(position_weight, torch.Tensor):
        cfg.position_embedding_init_size = max(
            int(getattr(cfg, "position_embedding_init_size", 0)),
            int(position_weight.shape[0]),
        )
    runtime_cfg = resolve_runtime_config(cfg, tokenizer)
    runtime_cfg = _infer_runtime_sizes_from_state(runtime_cfg, state, tokenizer)
    if isinstance(state, dict) and not any(str(key).startswith("torus_core.") for key in state.keys()):
        runtime_cfg.use_torus_core = False
        runtime_cfg.Torus_SHARC_Router = False
        runtime_cfg.use_torus_sharc_router = False
    model = PrismalWaveModel(runtime_cfg).to(device_obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not _checkpoint_mismatch_is_tolerable(missing, unexpected):
        raise ValueError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")
    training_state = payload.get("training_state")
    if isinstance(training_state, dict):
        setattr(model, "_prismal_training_state", training_state)
        if hasattr(model, "configure_precision"):
            model.configure_precision(device_obj, enabled=False, checkpoint_precision_state=training_state)
    elif hasattr(model, "configure_precision"):
        model.configure_precision(device_obj, enabled=False)
    refresh_quantized_caches(model)
    model.eval()
    return model


def load_bundle_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    *,
    load_training_state: bool = True,
) -> tuple[PrismalWaveModel, PrismalTokenizer, PrismalWaveConfig]:
    device_obj = resolve_device(device)
    payload = torch.load(checkpoint_path, map_location=device_obj)
    cfg_payload = payload.get("config")
    if not isinstance(cfg_payload, dict):
        raise ValueError("Checkpoint missing config.")
    cfg = PrismalWaveConfig.from_dict(cfg_payload)
    tokenizer_state = payload.get("tokenizer_state")
    if isinstance(tokenizer_state, dict):
        tokenizer = PrismalTokenizer.from_state_dict(tokenizer_state)
    else:
        tokenizer = PrismalTokenizer()
    tokenizer.refresh_construction_index()
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint missing model_state.")
    position_weight = state.get("position_embedding.embedding.weight")
    if isinstance(position_weight, torch.Tensor):
        cfg.position_embedding_init_size = max(
            int(getattr(cfg, "position_embedding_init_size", 0)),
            int(position_weight.shape[0]),
        )
    runtime_cfg = PrismalWaveConfig.from_dict(cfg.to_dict())
    if getattr(runtime_cfg, "use_hmote", False) or getattr(runtime_cfg, "use_recursive_hmoe", False):
        runtime_cfg.use_torus_core = True
    runtime_cfg = _infer_runtime_sizes_from_state(runtime_cfg, state, tokenizer=None)
    if isinstance(state, dict) and not any(str(key).startswith("torus_core.") for key in state.keys()):
        runtime_cfg.use_torus_core = False
        runtime_cfg.Torus_SHARC_Router = False
        runtime_cfg.use_torus_sharc_router = False
    model = PrismalWaveModel(runtime_cfg).to(device_obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not _checkpoint_mismatch_is_tolerable(missing, unexpected):
        raise ValueError(f"Checkpoint mismatch. missing={missing} unexpected={unexpected}")

    if tokenizer.vocab_size > model.vocab_size:
        print(
            f"[Prismal] expanding checkpoint vocab from {model.vocab_size} to {tokenizer.vocab_size} "
            "to match the loaded tokenizer",
            flush=True,
        )
        model.resize_vocab(tokenizer.vocab_size)

    if hasattr(model, "registry") and hasattr(model.registry, "_ensure_capacity"):
        target_family_vocab = max(int(getattr(model.registry, "family_vocab_size", 0)), int(getattr(tokenizer, "signature_vocab_size", 0)))
        target_level_vocab = max(int(getattr(model.registry.level_embedding, "num_embeddings", 0)), int(getattr(tokenizer, "signature_level_vocab_size", 0)))
        target_relation_vocab = max(int(getattr(model.registry.relation_embedding, "num_embeddings", 0)), int(getattr(tokenizer, "signature_relation_vocab_size", 0)))
        if (
            target_family_vocab > model.registry.family_vocab_size
            or target_level_vocab > model.registry.level_embedding.num_embeddings
            or target_relation_vocab > model.registry.relation_embedding.num_embeddings
        ):
            model.registry._ensure_capacity(
                max(0, target_family_vocab - 1),
                max(0, target_relation_vocab - 1),
                max(0, target_level_vocab - 1),
            )
            model.signature_vocab_size = max(model.signature_vocab_size, target_family_vocab)
            model.signature_level_vocab_size = max(model.signature_level_vocab_size, target_level_vocab)
            model.signature_relation_vocab_size = max(model.signature_relation_vocab_size, target_relation_vocab)
            model.cfg.signature_vocab_size = model.signature_vocab_size
            model.cfg.signature_level_vocab_size = model.signature_level_vocab_size
            model.cfg.signature_relation_vocab_size = model.signature_relation_vocab_size

    training_state = payload.get("training_state")
    if load_training_state and isinstance(training_state, dict):
        setattr(model, "_prismal_training_state", training_state)
    if hasattr(model, "configure_precision"):
        if load_training_state and isinstance(training_state, dict):
            model.configure_precision(device_obj, enabled=False, checkpoint_precision_state=training_state)
        else:
            model.configure_precision(device_obj, enabled=False)
    refresh_quantized_caches(model)
    model.eval()
    return model, tokenizer, cfg


@torch.no_grad()
def evaluate_model(
    model: PrismalWaveModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool | None = None,
) -> Dict[str, float]:
    def _set_force_aux_stats(module: torch.nn.Module, enabled: bool, seen: Optional[set[int]] = None) -> None:
        if seen is None:
            seen = set()
        module_id = id(module)
        if module_id in seen:
            return
        seen.add(module_id)
        if hasattr(module, "force_aux_stats"):
            setattr(module, "force_aux_stats", enabled)
        for child in module.children():
            _set_force_aux_stats(child, enabled, seen)

    runtime_model = _unwrap_model(model)
    precision_policy = getattr(runtime_model, "precision_policy", None)
    if hasattr(runtime_model, "configure_precision"):
        runtime_model.configure_precision(
            device,
            enabled=bool(precision_policy is not None and precision_policy.enabled),
        )
        precision_policy = getattr(runtime_model, "precision_policy", None)
    was_training = model.training
    force_aux_stats = bool(getattr(model, "force_aux_stats", False))
    model.eval()
    _set_force_aux_stats(model, True)
    if use_amp is None:
        use_amp = device.type == "cuda"
    losses = []
    ce_losses = []
    aux_losses = []
    sig_agree = []
    entropy = []
    active = []
    soft_active = []
    family_active = []
    family_unique = []
    family_bank = []
    family_capacity = []
    family_budget = []
    family_hit_rate = []
    family_gate = []
    cell_breadth = []
    cell_soft_breadth = []
    cell_effective = []
    cell_coverage = []
    usage_entropy = []
    usage_concentration = []
    try:
        for input_ids, labels, signature_ids, signature_level_ids, signature_relation_ids, parent_signature_ids, signature_family_ids, loss_mask in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            signature_ids = signature_ids.to(device)
            signature_level_ids = signature_level_ids.to(device)
            signature_relation_ids = signature_relation_ids.to(device)
            parent_signature_ids = parent_signature_ids.to(device)
            signature_family_ids = signature_family_ids.to(device)
            loss_mask = loss_mask.to(device)
            context = _precision_context(precision_policy, device, use_amp=bool(use_amp))
            with context:
                loss, output = model.compute_loss(
                    input_ids,
                    labels,
                    signature_ids=signature_ids,
                    signature_level_ids=signature_level_ids,
                    signature_relation_ids=signature_relation_ids,
                    parent_signature_ids=parent_signature_ids,
                    signature_family_ids=signature_family_ids,
                    loss_mask=loss_mask,
                )
            losses.append(float(loss.item()))
            ce_losses.append(float(output.ce_loss.item()))
            aux_losses.append(float(output.aux_loss.item()))
            sig_agree.append(float(output.route_stats["signature_agreement"].mean().item()))
            entropy.append(float(output.route_stats["avg_entropy"].item()))
            active.append(_stat_value(output.route_stats, "emitter_cell_occupancy", "avg_active_emitters"))
            soft_active.append(_stat_value(output.route_stats, "avg_emitter_cell_soft_occupancy", "emitter_cell_soft_occupancy"))
            family_active.append(_stat_value(output.route_stats, "specialist_family_specialist_active_count", "family_specialist_active_count"))
            family_unique.append(_stat_value(output.route_stats, "specialist_family_specialist_unique_families", "family_specialist_unique_families"))
            family_bank.append(_stat_value(output.route_stats, "specialist_family_specialist_bank_size", "family_specialist_bank_size"))
            family_capacity.append(_stat_value(output.route_stats, "specialist_family_specialist_capacity", "family_specialist_capacity"))
            family_budget.append(_stat_value(output.route_stats, "specialist_family_specialist_budget", "family_specialist_budget"))
            family_hit_rate.append(_stat_value(output.route_stats, "specialist_family_specialist_hit_rate", "family_specialist_hit_rate"))
            family_gate.append(_stat_value(output.route_stats, "specialist_family_specialist_gate_mean", "family_specialist_gate_mean"))
            cell_breadth.append(_stat_value(output.route_stats, "emitter_cell_breadth"))
            cell_soft_breadth.append(_stat_value(output.route_stats, "avg_emitter_cell_soft_breadth", "emitter_cell_soft_breadth"))
            cell_effective.append(_stat_value(output.route_stats, "emitter_cell_effective_count", "avg_emitter_topk_effective_count"))
            cell_coverage.append(_stat_value(output.route_stats, "emitter_cell_coverage_loss", "torus_coverage_loss"))
            usage_entropy.append(_stat_value(output.route_stats, "emitter_usage_entropy"))
            usage_concentration.append(_stat_value(output.route_stats, "emitter_usage_concentration"))
        return {
            "loss": sum(losses) / max(len(losses), 1),
            "ce_loss": sum(ce_losses) / max(len(ce_losses), 1),
            "aux_loss": sum(aux_losses) / max(len(aux_losses), 1),
            "signature_agreement": sum(sig_agree) / max(len(sig_agree), 1),
            "avg_entropy": sum(entropy) / max(len(entropy), 1),
            "avg_active_emitters": sum(active) / max(len(active), 1),
            "avg_emitter_cell_occupancy": sum(active) / max(len(active), 1),
            "avg_emitter_cell_soft_occupancy": sum(soft_active) / max(len(soft_active), 1),
            "avg_family_specialist_active_count": sum(family_active) / max(len(family_active), 1),
            "avg_family_specialist_unique_families": sum(family_unique) / max(len(family_unique), 1),
            "avg_family_specialist_bank_size": sum(family_bank) / max(len(family_bank), 1),
            "avg_family_specialist_capacity": sum(family_capacity) / max(len(family_capacity), 1),
            "avg_family_specialist_budget": sum(family_budget) / max(len(family_budget), 1),
            "avg_family_specialist_hit_rate": sum(family_hit_rate) / max(len(family_hit_rate), 1),
            "avg_family_specialist_gate_mean": sum(family_gate) / max(len(family_gate), 1),
            "avg_emitter_cell_breadth": sum(cell_breadth) / max(len(cell_breadth), 1),
            "avg_emitter_cell_soft_breadth": sum(cell_soft_breadth) / max(len(cell_soft_breadth), 1),
            "avg_emitter_cell_effective_count": sum(cell_effective) / max(len(cell_effective), 1),
            "avg_emitter_cell_coverage_loss": sum(cell_coverage) / max(len(cell_coverage), 1),
            "avg_emitter_usage_entropy": sum(usage_entropy) / max(len(usage_entropy), 1),
            "avg_emitter_usage_concentration": sum(usage_concentration) / max(len(usage_concentration), 1),
        }
    finally:
        _set_force_aux_stats(model, force_aux_stats)
        if was_training:
            model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def train_model(
    model: PrismalWaveModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    cfg: PrismalWaveConfig | None = None,
    optimizer_name: str = "hierarchical",
    epochs: int = 0,
    steps: int,
    lr: float,
    grad_clip: float = 1.0,
    grad_accum_steps: int | None = None,
    minutes: float | None = None,
    progress: bool = True,
    val_loader: Optional[DataLoader] = None,
    diagnostic_interval: int = 20,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    cfg = cfg or PrismalWaveConfig()
    runtime_model = _unwrap_model(model)
    precision_policy = getattr(runtime_model, "precision_policy", None)
    optimizer_name_norm = optimizer_name.strip().lower()
    runtime_model.cfg.optimizer = optimizer_name_norm
    runtime_model.cfg.lr = float(lr)
    for field_name in (
        "muon_lr",
        "muon_weight_decay",
        "muon_momentum_beta",
        "muon_nesterov",
        "muon_ns_steps",
        "muon_extra_scale_factor",
        "muon_scalar_optimizer",
        "use_bitsandbytes_leaf_precision",
        "bitsandbytes_leaf_precision_mode",
        "bitsandbytes_leaf_quant_type",
        "bitsandbytes_leaf_compute_dtype",
        "use_gradient_accumulation",
        "gradient_accumulation_steps",
        "quantization_aware_training",
        "qat_start_fraction",
        "qat_ramp_fraction",
        "use_token_superposition_training",
        "token_superposition_bag_size",
        "token_superposition_phase_fraction",
    ):
        if hasattr(cfg, field_name):
            setattr(runtime_model.cfg, field_name, getattr(cfg, field_name))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)
        except Exception:
            pass
    optimizer = _build_optimizer(
        runtime_model,
        optimizer_name=optimizer_name_norm,
        base_lr=lr,
        cfg=cfg,
        precision_policy=getattr(runtime_model, "precision_policy", None),
    )
    if hasattr(runtime_model, "set_capacity_growth_locked"):
        runtime_model.set_capacity_growth_locked(True)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and device.type == "cuda"))
    def _current_scaler_enabled() -> bool:
        return _amp_scaler_enabled(precision_policy, device, use_amp=use_amp)
    use_grad_accum = bool(getattr(cfg, "use_gradient_accumulation", False))
    if grad_accum_steps is not None:
        requested_grad_accum_steps = max(1, int(grad_accum_steps))
        use_grad_accum = requested_grad_accum_steps > 1
    else:
        requested_grad_accum_steps = max(1, int(getattr(cfg, "gradient_accumulation_steps", 1)))
    grad_accum_steps = max(1, requested_grad_accum_steps if use_grad_accum else 1)
    resume_state = getattr(runtime_model, "_prismal_training_state", None)
    resume_global_step = 0
    optimizer_state_loaded = False
    if isinstance(resume_state, dict):
        resume_global_step = int(resume_state.get("global_step", 0) or 0)
        resume_optimizer_name = str(resume_state.get("optimizer_name", "")).strip().lower()
        resume_optimizer_state = resume_state.get("optimizer_state")
        if resume_optimizer_name == optimizer_name_norm and isinstance(resume_optimizer_state, dict):
            try:
                optimizer.load_state_dict(resume_optimizer_state)
                _apply_nested_optimizer_settings(optimizer, cfg)
                if _optimizer_state_is_shape_compatible(optimizer):
                    optimizer_state_loaded = True
                else:
                    print(
                        "[Prismal] optimizer state restore skipped: loaded state tensor shapes do not match current parameters",
                        flush=True,
                    )
                    optimizer.state.clear()
            except Exception as exc:
                print(f"[Prismal] optimizer state restore skipped: {exc}", flush=True)
        elif resume_optimizer_state is not None:
            print(
                f"[Prismal] skipping optimizer state restore because checkpoint optimizer={resume_optimizer_name or 'unknown'} "
                f"and current optimizer={optimizer_name_norm}",
                flush=True,
            )
    if isinstance(resume_state, dict) and scaler_enabled:
        resume_scaler_state = resume_state.get("scaler_state")
        if isinstance(resume_scaler_state, dict):
            try:
                scaler.load_state_dict(resume_scaler_state)
            except Exception as exc:
                print(f"[Prismal] scaler state restore skipped: {exc}", flush=True)
    if hasattr(runtime_model, "configure_precision"):
        runtime_model.configure_precision(
            device,
            enabled=bool(precision_policy is not None and precision_policy.enabled),
            checkpoint_precision_state=resume_state if isinstance(resume_state, dict) else None,
        )
    precision_policy = getattr(runtime_model, "precision_policy", None)
    diagnostic_interval = max(1, int(diagnostic_interval))
    loss_sum: Optional[float] = None
    ce_loss_sum: Optional[float] = None
    aux_loss_sum: Optional[float] = None
    best_loss_tensor: Optional[float] = None
    last_loss_tensor: Optional[float] = None
    last_ce_loss_tensor: Optional[float] = None
    last_aux_loss_tensor: Optional[float] = None
    route_agreement_sum: Optional[float] = None
    route_entropy_sum: Optional[float] = None
    route_active_sum: Optional[float] = None
    route_soft_active_sum: Optional[float] = None
    route_family_active_sum: Optional[float] = None
    route_family_unique_sum: Optional[float] = None
    route_family_bank_sum: Optional[float] = None
    route_family_capacity_sum: Optional[float] = None
    route_family_budget_sum: Optional[float] = None
    route_family_hit_rate_sum: Optional[float] = None
    route_family_gate_sum: Optional[float] = None
    route_usage_entropy_sum: Optional[float] = None
    route_usage_concentration_sum: Optional[float] = None
    route_cell_breadth_sum: Optional[float] = None
    route_cell_soft_breadth_sum: Optional[float] = None
    route_cell_effective_sum: Optional[float] = None
    route_cell_coverage_sum: Optional[float] = None
    gatetrain_hit_sum: Optional[float] = None
    gatetrain_miss_sum: Optional[float] = None
    gatetrain_churn_sum: Optional[float] = None
    gatetrain_predicted_tiles_sum: Optional[float] = None
    gatetrain_confidence_sum: Optional[float] = None
    gatetrain_latency_saved_sum: Optional[float] = None
    gatetrain_plan_time_sum: Optional[float] = None
    gatetrain_lead_time_sum: Optional[float] = None
    gatetrain_full_scope_sum: Optional[float] = None
    diagnostic_count = 0
    stability_nonfinite_loss_batches = 0
    stability_nonfinite_grad_batches = 0
    stability_skipped_optimizer_steps = 0
    stability_repaired_tensors = 0
    stability_clipped_groups = 0
    stability_clipped_steps = 0
    last_stability_text = ""
    val_losses = []
    val_agreements = []
    last_val_metrics: Optional[Dict[str, float]] = None
    timing_totals: Dict[str, float] = {}
    start = time.perf_counter()
    step = 0
    limit_seconds = minutes * 60.0 if minutes is not None else None
    epoch_count = max(0, int(epochs))
    try:
        batches_per_epoch = max(1, len(dataloader))
    except TypeError:
        batches_per_epoch = max(1, int(steps) if steps and steps > 0 else 1)
    streaming_mode = isinstance(getattr(dataloader, "dataset", None), torch.utils.data.IterableDataset)
    step_limit = int(steps) if steps and steps > 0 else (200 if epoch_count <= 0 else 0)
    epoch_batches = epoch_count * batches_per_epoch if epoch_count > 0 else step_limit
    total_batches = min(epoch_batches, step_limit) if step_limit > 0 and epoch_count > 0 else epoch_batches
    total_steps = max(1, step_limit if step_limit > 0 else epoch_batches)
    if epoch_count > 0:
        total_steps = max(total_steps, epoch_batches)
    scheduler_total_steps = max(1, math.ceil(total_steps / grad_accum_steps))
    scheduler_warmup_steps = max(1, min(int(scheduler_total_steps * 0.1), max(1, scheduler_total_steps - 1)))
    if optimizer_state_loaded and isinstance(resume_state, dict):
        scheduler_total_steps = int(resume_state.get("scheduler_total_steps", scheduler_total_steps) or scheduler_total_steps)
        scheduler_warmup_steps = int(resume_state.get("scheduler_warmup_steps", scheduler_warmup_steps) or scheduler_warmup_steps)
    scheduler = None
    if scheduler_total_steps > 1:
        warmup_steps = max(1, min(int(scheduler_warmup_steps), max(1, scheduler_total_steps - 1)))

        def _lr_factor(step_idx: int) -> float:
            if step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            progress = float(step_idx - warmup_steps) / float(max(1, scheduler_total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_factor)
        if optimizer_state_loaded and isinstance(resume_state, dict):
            resume_scheduler_state = resume_state.get("scheduler_state")
            if isinstance(resume_scheduler_state, dict):
                try:
                    scheduler.load_state_dict(resume_scheduler_state)
                except Exception as exc:
                    print(f"[Prismal] scheduler state restore skipped: {exc}", flush=True)
    tracker = ProgressTracker(
        total_epochs=max(1, epoch_count if epoch_count > 0 else 1),
        batches_per_epoch=batches_per_epoch,
        total_batches=max(1, total_batches),
        streaming_mode=streaming_mode,
    )
    qat_enabled = bool(
        getattr(runtime_model.cfg, "quantization_aware_training", False)
        and getattr(runtime_model, "use_turbo_quantization", False)
        and precision_policy is not None
        and precision_policy.enabled
    )
    qat_start_fraction = float(getattr(runtime_model.cfg, "qat_start_fraction", 0.65))
    qat_ramp_fraction = float(getattr(runtime_model.cfg, "qat_ramp_fraction", 0.20))
    qat_last_policy_state: Optional[Dict[str, object]] = precision_policy.to_state_dict() if qat_enabled and precision_policy is not None else None

    def _maybe_update_qat_precision(step_index: int) -> None:
        nonlocal precision_policy, qat_last_policy_state
        if not qat_enabled or precision_policy is None:
            return
        absolute_step = resume_global_step + int(step_index)
        progress = float(absolute_step) / float(max(resume_global_step + scheduler_total_steps, 1))
        scheduled_policy = precision_policy.progressive_qat_policy(
            progress,
            start_fraction=qat_start_fraction,
            ramp_fraction=qat_ramp_fraction,
        )
        scheduled_state = scheduled_policy.to_state_dict()
        if qat_last_policy_state == scheduled_state:
            return
        qat_last_policy_state = scheduled_state
        precision_policy = scheduled_policy
        runtime_model.precision_policy = scheduled_policy
        attach_precision_policy(runtime_model, scheduled_policy)
        runtime_model.configure_precision(device, enabled=True)

    optimizer.zero_grad(set_to_none=True)
    accumulation_step = 0
    optimizer_step = 0

    def _finish_optimizer_step() -> tuple[bool, int]:
        nonlocal accumulation_step, optimizer_step
        nonlocal stability_skipped_optimizer_steps, stability_nonfinite_grad_batches
        nonlocal stability_clipped_groups, stability_clipped_steps
        nonlocal last_stability_text
        if accumulation_step <= 0:
            return False, 0
        should_step_scheduler = True
        step_scaler_enabled = _current_scaler_enabled()
        if step_scaler_enabled:
            scaler.unscale_(optimizer)
        if not _optimizer_gradients_are_finite(optimizer):
            should_step_scheduler = False
            stability_skipped_optimizer_steps += 1
            stability_nonfinite_grad_batches += 1
            last_stability_text = "grad_nan"
        clipped_groups = 0
        if should_step_scheduler:
            clipped_groups, _total_norm = _clip_optimizer_group_gradients(
                optimizer,
                runtime_model.cfg,
                fallback_grad_clip=float(grad_clip),
            )
            if clipped_groups > 0:
                stability_clipped_groups += clipped_groups
                stability_clipped_steps += 1
        if should_step_scheduler:
            if step_scaler_enabled:
                scaler.step(optimizer)
            else:
                optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        if step_scaler_enabled:
            scaler.update()
        if scheduler is not None and should_step_scheduler:
            scheduler.step()
        if should_step_scheduler:
            optimizer_step += 1
        optimizer.zero_grad(set_to_none=True)
        accumulation_step = 0
        return should_step_scheduler, clipped_groups

    def _epoch_validation_points(num_batches: int) -> list[int]:
        """Return the batch indices where validation should run within an epoch."""
        if num_batches <= 0:
            return []
        return [num_batches]

    def _run_validation(label: str) -> None:
        nonlocal last_val_metrics
        if val_loader is None:
            return
        val_metrics = evaluate_model(model, val_loader, device, use_amp=use_amp)
        last_val_metrics = val_metrics
        val_losses.append(float(val_metrics["loss"]))
        val_agreements.append(float(val_metrics["signature_agreement"]))
        if progress:
            val_raw_active = float(val_metrics.get("avg_active_emitters", 0.0))
            val_soft_active = float(val_metrics.get("avg_emitter_cell_soft_occupancy", 0.0))
            val_effective = float(val_metrics.get("avg_emitter_cell_effective_count", val_raw_active))
            val_breadth = float(val_metrics.get("avg_emitter_cell_breadth", 0.0))
            val_soft_breadth = float(val_metrics.get("avg_emitter_cell_soft_breadth", 0.0))
            val_balance = float(val_metrics.get("avg_emitter_cell_coverage_loss", 0.0))
            val_usage_entropy = float(val_metrics.get("avg_emitter_usage_entropy", 0.0))
            val_usage_concentration = float(val_metrics.get("avg_emitter_usage_concentration", 0.0))
            torus_core = getattr(model, "torus_core", None)
            val_local_r = int(getattr(torus_core, "local_field_radius", getattr(model, "local_field_radius", 0)))
            val_scout_r = int(getattr(torus_core, "scout_read_radius", getattr(model, "scout_read_radius", 0)))
            val_relay_r = int(getattr(torus_core, "relay_write_radius", getattr(model, "relay_write_radius", 0)))
            print(
                f"[Prismal] {label} "
                f"val_total={val_metrics['loss']:.4f} val_ce={val_metrics.get('ce_loss', 0.0):.4f} val_aux={val_metrics.get('aux_loss', 0.0):.4f} "
                f"val_sig={val_metrics['signature_agreement']:.4f} "
                f"val_active={val_raw_active:.2f} "
                f"val_soft_active={val_soft_active:.2f} "
                f"val_eff_count={val_effective:.2f} "
                f"val_cell_breadth={val_breadth:.4f} "
                f"val_soft_cell_breadth={val_soft_breadth:.4f} "
                f"val_cell_cov={val_balance:.4f} "
                f"val_usage_entropy={val_usage_entropy:.4f} "
                f"val_usage_conc={val_usage_concentration:.4f} "
                f"geom_local={val_local_r} "
                f"geom_scout={val_scout_r} "
                f"geom_relay={val_relay_r}",
                flush=True,
            )

    def _train_batch(
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        signature_ids: torch.Tensor,
        signature_level_ids: torch.Tensor,
        signature_relation_ids: torch.Tensor,
        parent_signature_ids: torch.Tensor,
        signature_family_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        epoch_idx: int,
        batch_idx: int,
    ) -> None:
        nonlocal step
        nonlocal accumulation_step
        nonlocal optimizer_step
        nonlocal last_stability_text
        nonlocal stability_nonfinite_loss_batches, stability_nonfinite_grad_batches
        nonlocal stability_skipped_optimizer_steps, stability_repaired_tensors
        nonlocal stability_clipped_groups, stability_clipped_steps
        nonlocal loss_sum, ce_loss_sum, aux_loss_sum, best_loss_tensor, last_loss_tensor, last_ce_loss_tensor, last_aux_loss_tensor
        nonlocal route_agreement_sum, route_entropy_sum, route_active_sum
        nonlocal route_soft_active_sum, route_usage_entropy_sum, route_usage_concentration_sum
        nonlocal route_family_active_sum, route_family_unique_sum, route_family_bank_sum
        nonlocal route_family_capacity_sum, route_family_budget_sum, route_family_hit_rate_sum, route_family_gate_sum
        nonlocal route_cell_breadth_sum, route_cell_soft_breadth_sum, route_cell_effective_sum, route_cell_coverage_sum, diagnostic_count
        nonlocal gatetrain_hit_sum, gatetrain_miss_sum, gatetrain_churn_sum, gatetrain_predicted_tiles_sum
        nonlocal gatetrain_confidence_sum, gatetrain_latency_saved_sum, gatetrain_plan_time_sum, gatetrain_lead_time_sum
        nonlocal gatetrain_full_scope_sum
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        signature_ids = signature_ids.to(device)
        signature_level_ids = signature_level_ids.to(device)
        signature_relation_ids = signature_relation_ids.to(device)
        parent_signature_ids = parent_signature_ids.to(device)
        signature_family_ids = signature_family_ids.to(device)
        loss_mask = loss_mask.to(device)
        token_superposition_bag_size = 1
        if _token_superposition_phase_active(
            cfg,
            resume_global_step=resume_global_step,
            optimizer_step=optimizer_step,
            scheduler_total_steps=scheduler_total_steps,
        ):
            token_superposition_bag_size = max(1, int(getattr(cfg, "token_superposition_bag_size", 1)))

        if accumulation_step == 0:
            _maybe_update_qat_precision(resume_global_step + optimizer_step)
        next_step = step + 1
        should_collect = (
            next_step == 1
            or next_step % diagnostic_interval == 0
            or batch_idx == 1
            or batch_idx == batches_per_epoch
            or (step_limit > 0 and next_step == step_limit)
        )
        context = _precision_context(precision_policy, device, use_amp=use_amp)
        with context:
            loss, output = model.compute_loss(
                input_ids,
                labels,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                signature_family_ids=signature_family_ids,
                loss_mask=loss_mask,
                superposition_bag_size=token_superposition_bag_size,
                collect_telemetry=should_collect,
            )
        repair_count = 0
        stability_value = output.route_stats.get("stability_nonfinite_repair_count")
        if torch.is_tensor(stability_value):
            repair_count = int(max(0.0, float(stability_value.detach().float().mean().item())))
        elif stability_value is not None:
            repair_count = int(max(0.0, float(stability_value)))
        if repair_count > 0:
            stability_repaired_tensors += repair_count
        is_finite_loss = bool(torch.isfinite(loss.detach()).all().item())
        last_stability_text = f"repairs={repair_count}"
        if not is_finite_loss:
            stability_nonfinite_loss_batches += 1
            stability_skipped_optimizer_steps += 1
            last_stability_text = f"loss_nan repairs={repair_count}"
            optimizer.zero_grad(set_to_none=True)
            accumulation_step = 0
            step += 1
            return
        scaled_loss = loss / float(grad_accum_steps)
        step_scaler_enabled = _current_scaler_enabled()
        if step_scaler_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        accumulation_step += 1
        should_step_scheduler = False
        clipped_groups = 0
        if accumulation_step >= grad_accum_steps:
            should_step_scheduler, clipped_groups = _finish_optimizer_step()
            if clipped_groups > 0:
                last_stability_text = f"repairs={repair_count} clip_groups={clipped_groups}"

        step += 1
        loss_value = float(loss.detach().item())
        ce_value = float(output.ce_loss.detach().item())
        aux_value = float(output.aux_loss.detach().item())
        last_loss_tensor = loss_value
        last_ce_loss_tensor = ce_value
        last_aux_loss_tensor = aux_value
        loss_sum = loss_value if loss_sum is None else loss_sum + loss_value
        ce_loss_sum = ce_value if ce_loss_sum is None else ce_loss_sum + ce_value
        aux_loss_sum = aux_value if aux_loss_sum is None else aux_loss_sum + aux_value
        best_loss_tensor = loss_value if best_loss_tensor is None else min(best_loss_tensor, loss_value)

        if should_collect:
            sig_value = float(output.route_stats["signature_agreement"].mean().detach().item())
            ent_value = float(output.route_stats["avg_entropy"].detach().item())
            act_value = float(output.route_stats.get("emitter_cell_occupancy", output.route_stats.get("avg_active_emitters", torch.tensor(0.0, device=device))).detach().float().mean().item())
            soft_act_value = float(output.route_stats.get("avg_emitter_cell_soft_occupancy", output.route_stats.get("emitter_cell_soft_occupancy", torch.tensor(0.0, device=device))).detach().float().mean().item())
            breadth_value = float(output.route_stats.get("emitter_cell_breadth", torch.tensor(0.0, device=device)).detach().float().mean().item())
            soft_breadth_value = float(output.route_stats.get("avg_emitter_cell_soft_breadth", output.route_stats.get("emitter_cell_soft_breadth", torch.tensor(0.0, device=device))).detach().float().mean().item())
            family_active_value = float(output.route_stats.get("specialist_family_specialist_active_count", output.route_stats.get("family_specialist_active_count", torch.tensor(0.0, device=device))).detach().float().mean().item())
            family_unique_value = float(output.route_stats.get("specialist_family_specialist_unique_families", output.route_stats.get("family_specialist_unique_families", torch.tensor(0.0, device=device))).detach().float().mean().item())
            family_bank_value = float(output.route_stats.get("specialist_family_specialist_bank_size", output.route_stats.get("family_specialist_bank_size", torch.tensor(0.0, device=device))).detach().float().mean().item())
            family_capacity_value = float(output.route_stats.get("specialist_family_specialist_capacity", output.route_stats.get("family_specialist_capacity", torch.tensor(0.0, device=device))).detach().float().mean().item())
            family_budget_value = float(output.route_stats.get("specialist_family_specialist_budget", output.route_stats.get("family_specialist_budget", torch.tensor(0.0, device=device))).detach().float().mean().item())
            family_hit_rate_value = float(output.route_stats.get("specialist_family_specialist_hit_rate", output.route_stats.get("family_specialist_hit_rate", torch.tensor(0.0, device=device))).detach().float().mean().item())
            family_gate_value = float(output.route_stats.get("specialist_family_specialist_gate_mean", output.route_stats.get("family_specialist_gate_mean", torch.tensor(0.0, device=device))).detach().float().mean().item())
            effective_value = float(output.route_stats.get(
                "emitter_cell_effective_count",
                output.route_stats.get("avg_emitter_topk_effective_count", torch.tensor(0.0, device=device)),
            ).detach().float().mean().item())
            coverage_value = float(output.route_stats.get(
                "emitter_cell_coverage_loss",
                output.route_stats.get("torus_coverage_loss", torch.tensor(0.0, device=device)),
            ).detach().float().mean().item())
            usage_entropy_value = float(output.route_stats.get("emitter_usage_entropy", torch.tensor(0.0, device=device)).detach().float().mean().item())
            usage_concentration_value = float(output.route_stats.get("emitter_usage_concentration", torch.tensor(0.0, device=device)).detach().float().mean().item())
            route_agreement_sum = sig_value if route_agreement_sum is None else route_agreement_sum + sig_value
            route_entropy_sum = ent_value if route_entropy_sum is None else route_entropy_sum + ent_value
            route_active_sum = act_value if route_active_sum is None else route_active_sum + act_value
            route_soft_active_sum = soft_act_value if route_soft_active_sum is None else route_soft_active_sum + soft_act_value
            route_family_active_sum = family_active_value if route_family_active_sum is None else route_family_active_sum + family_active_value
            route_family_unique_sum = family_unique_value if route_family_unique_sum is None else route_family_unique_sum + family_unique_value
            route_family_bank_sum = family_bank_value if route_family_bank_sum is None else route_family_bank_sum + family_bank_value
            route_family_capacity_sum = family_capacity_value if route_family_capacity_sum is None else route_family_capacity_sum + family_capacity_value
            route_family_budget_sum = family_budget_value if route_family_budget_sum is None else route_family_budget_sum + family_budget_value
            route_family_hit_rate_sum = family_hit_rate_value if route_family_hit_rate_sum is None else route_family_hit_rate_sum + family_hit_rate_value
            route_family_gate_sum = family_gate_value if route_family_gate_sum is None else route_family_gate_sum + family_gate_value
            route_cell_breadth_sum = breadth_value if route_cell_breadth_sum is None else route_cell_breadth_sum + breadth_value
            route_cell_soft_breadth_sum = soft_breadth_value if route_cell_soft_breadth_sum is None else route_cell_soft_breadth_sum + soft_breadth_value
            route_cell_effective_sum = effective_value if route_cell_effective_sum is None else route_cell_effective_sum + effective_value
            route_cell_coverage_sum = coverage_value if route_cell_coverage_sum is None else route_cell_coverage_sum + coverage_value
            route_usage_entropy_sum = usage_entropy_value if route_usage_entropy_sum is None else route_usage_entropy_sum + usage_entropy_value
            route_usage_concentration_sum = usage_concentration_value if route_usage_concentration_sum is None else route_usage_concentration_sum + usage_concentration_value
            if getattr(runtime_model, "use_gatetrain", False):
                gatetrain_hit_sum = _stat_value(output.route_stats, "gatetrain_hit_count") if gatetrain_hit_sum is None else gatetrain_hit_sum + _stat_value(output.route_stats, "gatetrain_hit_count")
                gatetrain_miss_sum = _stat_value(output.route_stats, "gatetrain_miss_count") if gatetrain_miss_sum is None else gatetrain_miss_sum + _stat_value(output.route_stats, "gatetrain_miss_count")
                gatetrain_churn_sum = _stat_value(output.route_stats, "gatetrain_tile_churn") if gatetrain_churn_sum is None else gatetrain_churn_sum + _stat_value(output.route_stats, "gatetrain_tile_churn")
                gatetrain_predicted_tiles_sum = _stat_value(output.route_stats, "gatetrain_predicted_tiles") if gatetrain_predicted_tiles_sum is None else gatetrain_predicted_tiles_sum + _stat_value(output.route_stats, "gatetrain_predicted_tiles")
                gatetrain_confidence_sum = _stat_value(output.route_stats, "gatetrain_confidence") if gatetrain_confidence_sum is None else gatetrain_confidence_sum + _stat_value(output.route_stats, "gatetrain_confidence")
                gatetrain_latency_saved_sum = _stat_value(output.route_stats, "gatetrain_latency_saved_ms") if gatetrain_latency_saved_sum is None else gatetrain_latency_saved_sum + _stat_value(output.route_stats, "gatetrain_latency_saved_ms")
                gatetrain_plan_time_sum = _stat_value(output.route_stats, "gatetrain_plan_time_ms") if gatetrain_plan_time_sum is None else gatetrain_plan_time_sum + _stat_value(output.route_stats, "gatetrain_plan_time_ms")
                gatetrain_lead_time_sum = _stat_value(output.route_stats, "gatetrain_lead_time_ms") if gatetrain_lead_time_sum is None else gatetrain_lead_time_sum + _stat_value(output.route_stats, "gatetrain_lead_time_ms")
                gatetrain_full_scope_sum = _stat_value(output.route_stats, "gatetrain_full_scope") if gatetrain_full_scope_sum is None else gatetrain_full_scope_sum + _stat_value(output.route_stats, "gatetrain_full_scope")
            diagnostic_count += 1
            _accumulate_timings(timing_totals, output.route_stats)

        if progress and should_collect:
            sig = float(output.route_stats["signature_agreement"].mean().item())
            ent = float(output.route_stats["avg_entropy"].item())
            act = _stat_value(output.route_stats, "emitter_cell_occupancy", "avg_active_emitters")
            soft_act = _stat_value(output.route_stats, "avg_emitter_cell_soft_occupancy", "emitter_cell_soft_occupancy")
            breadth = _stat_value(output.route_stats, "emitter_cell_breadth")
            soft_breadth = _stat_value(output.route_stats, "avg_emitter_cell_soft_breadth", "emitter_cell_soft_breadth")
            coverage_loss = _stat_value(output.route_stats, "emitter_cell_coverage_loss", "torus_coverage_loss")
            usage_entropy = float(output.route_stats["emitter_usage_entropy"].item()) if "emitter_usage_entropy" in output.route_stats else None
            usage_concentration = float(output.route_stats["emitter_usage_concentration"].item()) if "emitter_usage_concentration" in output.route_stats else None
            effective_count = _stat_value(output.route_stats, "emitter_cell_effective_count", "avg_emitter_topk_effective_count")
            balance_loss = float(output.route_stats["emitter_balance_loss"].item()) if "emitter_balance_loss" in output.route_stats else None
            torus_coverage = float(output.route_stats["torus_coverage_loss"].item()) if "torus_coverage_loss" in output.route_stats else None
            batch_timings = _extract_timings(output.route_stats, include_totals=True)
            tracker.emit(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                completed_batches=step,
                total_loss=float(loss.item()),
                ce_loss=ce_value,
                aux_loss=aux_value,
                signature_agreement=sig,
                entropy=ent,
                raw_active_emitters=act,
                soft_active_emitters=soft_act,
                effective_emitters=effective_count,
                normalized_breadth=breadth,
                soft_breadth=soft_breadth,
                balance_loss=balance_loss,
                emitter_cell_coverage=coverage_loss,
                usage_entropy=usage_entropy,
                usage_concentration=usage_concentration,
                torus_coverage=torus_coverage,
                stability_text=last_stability_text,
                timings=batch_timings,
            )

    if epoch_count > 0:
        for epoch_idx in range(1, epoch_count + 1):
            if limit_seconds is not None and step > 0 and (time.perf_counter() - start) >= limit_seconds:
                break
            if step_limit > 0 and step >= step_limit:
                break
            validation_points = [] if streaming_mode else _epoch_validation_points(batches_per_epoch)
            validation_point_index = 0
            iterator = iter(dataloader)
            for batch_idx, (
                input_ids,
                labels,
                signature_ids,
                signature_level_ids,
                signature_relation_ids,
                parent_signature_ids,
                signature_family_ids,
                loss_mask,
            ) in enumerate(iterator, start=1):
                if limit_seconds is not None and step > 0 and (time.perf_counter() - start) >= limit_seconds:
                    break
                if step_limit > 0 and step >= step_limit:
                    break
                _train_batch(
                    input_ids,
                    labels,
                    signature_ids,
                    signature_level_ids,
                    signature_relation_ids,
                    parent_signature_ids,
                    signature_family_ids,
                    loss_mask,
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                )
                while validation_point_index < len(validation_points) and batch_idx >= validation_points[validation_point_index]:
                    _finish_optimizer_step()
                    _run_validation(f"epoch {epoch_idx}/{epoch_count} @ end")
                    validation_point_index += 1
            _finish_optimizer_step()
    else:
        iterator = iter(dataloader)
        batch_idx = 0
        while True:
            if limit_seconds is not None and step > 0 and (time.perf_counter() - start) >= limit_seconds:
                break
            if minutes is None and step_limit > 0 and step >= step_limit:
                break

            try:
                input_ids, labels, signature_ids, signature_level_ids, signature_relation_ids, parent_signature_ids, signature_family_ids, loss_mask = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                input_ids, labels, signature_ids, signature_level_ids, signature_relation_ids, parent_signature_ids, signature_family_ids, loss_mask = next(iterator)

            batch_idx = (batch_idx % batches_per_epoch) + 1
            epoch_idx = min(epoch_count if epoch_count > 0 else 1, ((step // batches_per_epoch) + 1))
            _train_batch(
                input_ids,
                labels,
                signature_ids,
                signature_level_ids,
                signature_relation_ids,
                parent_signature_ids,
                signature_family_ids,
                loss_mask,
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
            )
        _finish_optimizer_step()

    if epoch_count <= 0 or streaming_mode:
        _run_validation("final")

    elapsed = time.perf_counter() - start
    runtime_model._prismal_training_state = {
        "optimizer_name": optimizer_name_norm,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "global_step": int(resume_global_step + optimizer_step),
        "microbatch_step": int(step),
        "grad_accum_steps": int(grad_accum_steps),
        "scheduler_total_steps": int(scheduler_total_steps),
        "scheduler_warmup_steps": int(scheduler_warmup_steps),
        "use_amp": bool(use_amp),
        "precision_policy_state": getattr(getattr(runtime_model, "precision_policy", None), "to_state_dict", lambda: None)(),
        "precision_tier_map": getattr(runtime_model, "_precision_tier_map", []),
    }
    if hasattr(runtime_model, "configure_precision"):
        runtime_model.configure_precision(device, enabled=False, checkpoint_precision_state=runtime_model._prismal_training_state)
    def _tensor_float(value: Optional[float], default: float = float("nan")) -> float:
        return float(value) if value is not None else default
    def safe_avg(value: Optional[float], default: float = 0.0) -> float:
        if value is None:
            return float(default)
        value = float(value)
        if math.isnan(value):
            return float(default)
        return value / diag_den
    step_den = max(step, 1)
    def safe_step_avg(value: Optional[float], default: float = 0.0) -> float:
        if value is None:
            return float(default)
        value = float(value)
        if math.isnan(value):
            return float(default)
        return value / step_den
    diag_den = max(diagnostic_count, 1)
    return {
        "steps": float(step),
        "epochs": float(epoch_count),
        "elapsed_seconds": float(elapsed),
        "steps_per_second": float(step / max(elapsed, 1e-6)),
        "final_total_loss": _tensor_float(last_loss_tensor),
        "final_ce_loss": _tensor_float(last_ce_loss_tensor),
        "final_aux_loss": _tensor_float(last_aux_loss_tensor),
        "best_total_loss": _tensor_float(best_loss_tensor),
        "avg_total_loss": safe_step_avg(loss_sum),
        "avg_ce_loss": safe_step_avg(ce_loss_sum),
        "avg_aux_loss": safe_step_avg(aux_loss_sum),
        "val_total_loss": float(val_losses[-1]) if val_losses else float("nan"),
        "val_ce_loss": float(last_val_metrics["ce_loss"]) if last_val_metrics is not None else float("nan"),
        "val_aux_loss": float(last_val_metrics["aux_loss"]) if last_val_metrics is not None else float("nan"),
        "best_val_loss": float(min(val_losses)) if val_losses else float("nan"),
        "final_loss": _tensor_float(last_loss_tensor),
        "best_loss": _tensor_float(best_loss_tensor),
        "val_loss": float(val_losses[-1]) if val_losses else float("nan"),
        "val_signature_agreement": float(val_agreements[-1]) if val_agreements else float("nan"),
        "signature_agreement": safe_avg(route_agreement_sum),
        "avg_entropy": safe_avg(route_entropy_sum),
        "avg_active_emitters": safe_avg(route_active_sum),
        "avg_emitter_cell_occupancy": safe_avg(route_active_sum),
        "avg_emitter_cell_soft_occupancy": safe_avg(route_soft_active_sum),
        "avg_family_specialist_active_count": safe_avg(route_family_active_sum),
        "avg_family_specialist_unique_families": safe_avg(route_family_unique_sum),
        "avg_family_specialist_bank_size": safe_avg(route_family_bank_sum),
        "avg_family_specialist_capacity": safe_avg(route_family_capacity_sum),
        "avg_family_specialist_budget": safe_avg(route_family_budget_sum),
        "avg_family_specialist_hit_rate": safe_avg(route_family_hit_rate_sum),
        "avg_family_specialist_gate_mean": safe_avg(route_family_gate_sum),
        "avg_emitter_cell_breadth": safe_avg(route_cell_breadth_sum),
        "avg_emitter_cell_soft_breadth": safe_avg(route_cell_soft_breadth_sum),
        "avg_emitter_cell_effective_count": safe_avg(route_cell_effective_sum),
        "avg_emitter_cell_coverage_loss": safe_avg(route_cell_coverage_sum),
        "avg_emitter_usage_entropy": safe_avg(route_usage_entropy_sum),
        "avg_emitter_usage_concentration": safe_avg(route_usage_concentration_sum),
        "avg_gatetrain_hit_count": safe_avg(gatetrain_hit_sum),
        "avg_gatetrain_miss_count": safe_avg(gatetrain_miss_sum),
        "avg_gatetrain_tile_churn": safe_avg(gatetrain_churn_sum),
        "avg_gatetrain_predicted_tiles": safe_avg(gatetrain_predicted_tiles_sum),
        "avg_gatetrain_confidence": safe_avg(gatetrain_confidence_sum),
        "avg_gatetrain_latency_saved_ms": safe_avg(gatetrain_latency_saved_sum),
        "avg_gatetrain_plan_time_ms": safe_avg(gatetrain_plan_time_sum),
        "avg_gatetrain_lead_time_ms": safe_avg(gatetrain_lead_time_sum),
        "avg_gatetrain_full_scope": safe_avg(gatetrain_full_scope_sum),
        "gatetrain_hit_rate": float(
            (gatetrain_hit_sum or 0.0) / max((gatetrain_hit_sum or 0.0) + (gatetrain_miss_sum or 0.0), 1e-6)
        )
        if gatetrain_hit_sum is not None or gatetrain_miss_sum is not None
        else float("nan"),
        "stability_nonfinite_loss_batches": float(stability_nonfinite_loss_batches),
        "stability_nonfinite_grad_batches": float(stability_nonfinite_grad_batches),
        "stability_skipped_optimizer_steps": float(stability_skipped_optimizer_steps),
        "stability_repaired_tensors": float(stability_repaired_tensors),
        "stability_clipped_groups": float(stability_clipped_groups),
        "stability_clipped_steps": float(stability_clipped_steps),
        "diagnostic_batches": float(diagnostic_count),
        "timing_breakdown_ms_total": dict(sorted(timing_totals.items())),
        "timing_breakdown_ms_per_step": {
            key: float(value / max(step, 1)) for key, value in sorted(timing_totals.items())
        },
        "param_count": float(sum(p.numel() for p in model.parameters())),
    }


@torch.no_grad()
def run_benchmark(
    model: PrismalWaveModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    steps: int = 16,
) -> Dict[str, float]:
    model.eval()
    iterator = iter(dataloader)
    losses = []
    path_diversity = []
    signature_agreement = []
    active_emitters = []
    cell_breadth = []
    cell_soft_breadth = []
    cell_effective = []
    cell_coverage = []
    route_entropy = []
    timing_totals: Dict[str, float] = {}
    start = time.perf_counter()

    for _ in range(max(1, steps)):
        try:
            input_ids, labels, signature_ids, signature_level_ids, signature_relation_ids, parent_signature_ids, signature_family_ids, loss_mask = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            input_ids, labels, signature_ids, signature_level_ids, signature_relation_ids, parent_signature_ids, signature_family_ids, loss_mask = next(iterator)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        signature_ids = signature_ids.to(device)
        signature_level_ids = signature_level_ids.to(device)
        signature_relation_ids = signature_relation_ids.to(device)
        parent_signature_ids = parent_signature_ids.to(device)
        signature_family_ids = signature_family_ids.to(device)
        loss_mask = loss_mask.to(device)
        loss, output = model.compute_loss(
            input_ids,
            labels,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            signature_family_ids=signature_family_ids,
            loss_mask=loss_mask,
        )
        losses.append(float(loss.item()))
        path_diversity.append(float(output.route_stats["pairwise_diversity"].item()))
        signature_agreement.append(float(output.route_stats["signature_agreement"].mean().item()))
        active_emitters.append(_stat_value(output.route_stats, "emitter_cell_occupancy", "avg_active_emitters"))
        cell_breadth.append(_stat_value(output.route_stats, "emitter_cell_breadth"))
        cell_soft_breadth.append(_stat_value(output.route_stats, "emitter_cell_soft_breadth", "emitter_cell_breadth"))
        cell_effective.append(_stat_value(output.route_stats, "emitter_cell_effective_count", "avg_emitter_topk_effective_count"))
        cell_coverage.append(_stat_value(output.route_stats, "emitter_cell_coverage_loss", "torus_coverage_loss"))
        route_entropy.append(float(output.route_stats["avg_entropy"].item()))
        _accumulate_timings(timing_totals, output.route_stats)

    elapsed = time.perf_counter() - start
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "steps": float(max(1, steps)),
        "elapsed_seconds": float(elapsed),
        "steps_per_second": float(max(1, steps) / max(elapsed, 1e-6)),
        "path_diversity": float(sum(path_diversity) / max(len(path_diversity), 1)),
        "signature_agreement": float(sum(signature_agreement) / max(len(signature_agreement), 1)),
        "avg_active_emitters": float(sum(active_emitters) / max(len(active_emitters), 1)),
        "avg_emitter_cell_occupancy": float(sum(active_emitters) / max(len(active_emitters), 1)),
        "avg_emitter_cell_breadth": float(sum(cell_breadth) / max(len(cell_breadth), 1)),
        "avg_emitter_cell_soft_breadth": float(sum(cell_soft_breadth) / max(len(cell_soft_breadth), 1)),
        "avg_emitter_cell_effective_count": float(sum(cell_effective) / max(len(cell_effective), 1)),
        "avg_emitter_cell_coverage_loss": float(sum(cell_coverage) / max(len(cell_coverage), 1)),
        "avg_entropy": float(sum(route_entropy) / max(len(route_entropy), 1)),
        "timing_breakdown_ms_total": dict(sorted(timing_totals.items())),
        "timing_breakdown_ms_per_step": {
            key: float(value / max(steps, 1)) for key, value in sorted(timing_totals.items())
        },
        "param_count": float(sum(p.numel() for p in model.parameters())),
    }


def _format_generation_prompt(prompt: str, *, template_prompt: bool = False) -> str:
    prompt = prompt.strip()
    lowered = prompt.lower()
    if any(marker in lowered for marker in ("instruction:", "response:", "prompt:", "context:", "answer:", "output:", "completion:", "target:")):
        if lowered.endswith(("response:", "answer:", "output:", "completion:", "target:")):
            return prompt + " "
        return prompt
    if template_prompt:
        return f"Instruction: {prompt}\nResponse: "
    return prompt


@torch.no_grad()
def generate_text(
    model: PrismalWaveModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 64,
    min_new_tokens: int = 24,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.08,
    no_repeat_ngram_size: int = 4,
    beam_size: int = 1,
    use_speculative_decoding: Optional[bool] = None,
    speculative_draft_tokens: Optional[int] = None,
    speculative_temperature: Optional[float] = None,
    template_prompt: bool = False,
) -> str:
    model.eval()
    prompt = _format_generation_prompt(prompt, template_prompt=template_prompt)
    prompt_bundle = tokenizer.prepare_generation_hierarchy(prompt)
    (
        prompt_ids,
        prompt_signature_ids,
        prompt_signature_level_ids,
        prompt_signature_relation_ids,
        prompt_parent_signature_ids,
        prompt_signature_family_ids,
    ) = prompt_bundle.as_tuple()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    signature_ids = torch.tensor([prompt_signature_ids], dtype=torch.long, device=device)
    signature_level_ids = torch.tensor([prompt_signature_level_ids], dtype=torch.long, device=device)
    signature_relation_ids = torch.tensor([prompt_signature_relation_ids], dtype=torch.long, device=device)
    parent_signature_ids = torch.tensor([prompt_parent_signature_ids], dtype=torch.long, device=device)
    signature_family_ids = torch.tensor([prompt_signature_family_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        signature_family_ids=signature_family_ids,
        signature_ids=signature_ids,
        signature_level_ids=signature_level_ids,
        signature_relation_ids=signature_relation_ids,
        parent_signature_ids=parent_signature_ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        top_k=top_k,
        token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
        token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
        token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
        token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
        suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        beam_size=beam_size,
        use_speculative_decoding=False if use_speculative_decoding is None else bool(use_speculative_decoding),
        speculative_draft_tokens=speculative_draft_tokens,
        speculative_temperature=speculative_temperature,
    )
    generated_ids = generated[0].tolist()
    continuation = generated_ids[len(prompt_ids):]
    prompt_text = prompt
    continuation_text = tokenizer.decode(continuation)
    lines: list[str] = []
    if prompt_text:
        lines.append(prompt_text)
    if continuation_text:
        lines.append("")
        lines.append(continuation_text)
    lines.append("")
    lines.append(f"Prompt token IDs: {prompt_ids}")
    lines.append(f"Generated token IDs: {generated_ids}")
    lines.append(f"Continuation token IDs: {continuation}")
    return "\n".join(lines).strip()
