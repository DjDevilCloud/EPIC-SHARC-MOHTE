# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

"""Utilities for building pretokenized memmap datasets from txt/jsonl/parquet corpora."""

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np

try:
    from .data import (
        PrismalTokenizer,
        StreamingTextCorpusDataset,
        _build_window_samples_from_text,
        _clean_record_value,
        _compose_record_text,
        stream_pretokenized_windows,
    )
    from .train import build_tokenizer_from_source
except ImportError:  # pragma: no cover - supports direct script launching.
    from data import (
        PrismalTokenizer,
        StreamingTextCorpusDataset,
        _build_window_samples_from_text,
        _clean_record_value,
        _compose_record_text,
        stream_pretokenized_windows,
    )
    from train import build_tokenizer_from_source


@dataclass(frozen=True)
class _ShardSpec:
    kind: str
    source: str
    start: int = 0
    end: int = 0
    files: tuple[str, ...] = ()


class _ShardSampleIterable:
    """Re-iterable shard source that can be consumed twice by the writer."""

    def __init__(
        self,
        tokenizer: PrismalTokenizer,
        shard_spec: _ShardSpec,
        *,
        seq_len: int,
        max_samples: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.shard_spec = shard_spec
        self.seq_len = max(0, int(seq_len))
        self.max_samples = max(0, int(max_samples))

    def __iter__(self) -> Iterator[Any]:
        yielded = 0
        for text in _iter_shard_texts(self.shard_spec):
            merged = text.strip()
            if not merged:
                continue
            remaining = self.max_samples - yielded if self.max_samples else 0
            windows = _build_window_samples_from_text(
                self.tokenizer,
                merged,
                seq_len=self.seq_len,
                max_samples=max(0, remaining),
            )
            for sample in windows:
                yield sample
                yielded += 1
                if self.max_samples and yielded >= self.max_samples:
                    return


def _supported_corpus_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for subpath in sorted(root.rglob("*")):
        if not subpath.is_file():
            continue
        if subpath.suffix.lower() in {".txt", ".md", ".markdown", ".rst", ".jsonl", ".parquet"}:
            files.append(subpath)
    return files


def _file_partition_ranges(items: Sequence[Any], shard_count: int) -> list[Sequence[Any]]:
    shard_count = max(1, int(shard_count))
    if shard_count <= 1 or len(items) <= 1:
        return [tuple(items)]
    shard_count = min(shard_count, len(items))
    total_weight = len(items)
    base = total_weight // shard_count
    extra = total_weight % shard_count
    ranges: list[Sequence[Any]] = []
    start = 0
    for index in range(shard_count):
        stop = start + base + (1 if index < extra else 0)
        ranges.append(tuple(items[start:stop]))
        start = stop
    return [group for group in ranges if group]


def _jsonl_byte_ranges(path: Path, shard_count: int) -> list[tuple[int, int]]:
    shard_count = max(1, int(shard_count))
    if shard_count <= 1:
        return [(0, path.stat().st_size)]
    size = path.stat().st_size
    shard_count = min(shard_count, max(1, size))
    boundaries = [0]
    with path.open("rb") as f:
        for shard_index in range(1, shard_count):
            target = size * shard_index // shard_count
            if target <= 0:
                boundaries.append(0)
                continue
            if target >= size:
                boundaries.append(size)
                continue
            f.seek(target - 1)
            if f.read(1) == b"\n":
                boundaries.append(target)
                continue
            f.seek(target)
            f.readline()
            boundaries.append(f.tell())
    boundaries.append(size)
    ranges: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right > left:
            ranges.append((left, right))
    if not ranges:
        ranges.append((0, size))
    return ranges


def _iter_jsonl_texts_byte_range(path: Path, start: int, end: int) -> Iterator[str]:
    with path.open("rb") as f:
        if start > 0:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.seek(start)
                f.readline()
        else:
            f.seek(0)

        while True:
            line_start = f.tell()
            if line_start >= end:
                break
            raw = f.readline()
            if not raw:
                break
            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except UnicodeDecodeError:
                line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                merged = _compose_record_text(payload)
                if merged:
                    yield merged
            elif isinstance(payload, str):
                merged = payload.strip()
                if merged:
                    yield merged
            else:
                merged = _clean_record_value(payload)
                if merged:
                    yield merged


def _iter_shard_texts(shard_spec: _ShardSpec) -> Iterator[str]:
    source = Path(shard_spec.source)
    if shard_spec.kind == "jsonl_range":
        yield from _iter_jsonl_texts_byte_range(source, shard_spec.start, shard_spec.end)
        return
    if shard_spec.kind == "files":
        for file_path_str in shard_spec.files:
            yield from _iter_text_source(Path(file_path_str))
        return
    yield from _iter_text_source(source)


def _iter_text_source(path: Path) -> Iterator[str]:
    if path.is_dir():
        for subpath in sorted(path.rglob("*")):
            if not subpath.is_file():
                continue
            suffix = subpath.suffix.lower()
            if suffix in {".txt", ".md", ".markdown", ".rst"}:
                text = subpath.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    yield text
            elif suffix == ".jsonl":
                yield from _iter_text_source(subpath)
            elif suffix == ".parquet":
                try:
                    from .data import _iter_parquet_texts
                except ImportError:  # pragma: no cover - supports direct script launching.
                    from data import _iter_parquet_texts

                yield from _iter_parquet_texts(subpath)
        return

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_texts_byte_range(path, 0, path.stat().st_size)
        return
    if suffix == ".parquet":
        try:
            from .data import _iter_parquet_texts
        except ImportError:  # pragma: no cover - supports direct script launching.
            from data import _iter_parquet_texts

        yield from _iter_parquet_texts(path)
        return

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield text


def _build_shard_specs(source: Path, workers: int) -> list[_ShardSpec]:
    workers = max(1, int(workers))
    if source.is_dir():
        files = _supported_corpus_files(source)
        if len(files) <= 1 or workers <= 1:
            return [_ShardSpec(kind="files", source=str(source), files=tuple(str(file_path) for file_path in files))]
        groups = _file_partition_ranges(files, workers)
        return [
            _ShardSpec(kind="files", source=str(source), files=tuple(str(file_path) for file_path in group))
            for group in groups
        ]

    suffix = source.suffix.lower()
    if suffix == ".jsonl" and workers > 1:
        ranges = _jsonl_byte_ranges(source, workers)
        return [
            _ShardSpec(kind="jsonl_range", source=str(source), start=start, end=end)
            for start, end in ranges
        ]

    return [_ShardSpec(kind="single", source=str(source))]


def _resolve_worker_count(source: Path, requested_workers: int) -> int:
    requested_workers = int(requested_workers)
    if requested_workers > 0:
        return requested_workers
    cpu_count = max(1, os.cpu_count() or 1)
    if source.is_dir():
        files = _supported_corpus_files(source)
        if len(files) <= 1:
            return 1
        return max(1, min(cpu_count, len(files), 8))
    if source.suffix.lower() == ".jsonl" and source.is_file():
        try:
            size = source.stat().st_size
        except OSError:
            return 1
        if size >= 64 * 1024 * 1024:
            return max(1, min(cpu_count, 8))
    return 1


def _worker_pretokenize_shard(
    tokenizer_state: dict[str, Any],
    shard_spec: _ShardSpec,
    *,
    seq_len: int,
    max_samples: int,
    output_dir: str | Path,
) -> str:
    tokenizer = PrismalTokenizer.from_state_dict(tokenizer_state)
    shard_dir = Path(output_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    sample_iterable = _ShardSampleIterable(
        tokenizer,
        shard_spec,
        seq_len=seq_len,
        max_samples=max_samples,
    )
    stream_pretokenized_windows(
        sample_iterable,
        shard_dir,
        seq_len=seq_len,
        metadata={
            "source": shard_spec.source,
            "shard_kind": shard_spec.kind,
            "shard_start": int(shard_spec.start),
            "shard_end": int(shard_spec.end),
            "shard_files": list(shard_spec.files),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "max_samples": int(max_samples),
        },
    )
    return str(shard_dir)


def _merge_pretokenized_shards(
    shard_dirs: Sequence[Path],
    output_dir: str | Path,
    *,
    seq_len: int,
    metadata: Optional[dict[str, Any]] = None,
    max_samples: int = 0,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_plans: list[tuple[Path, dict[str, Any], int, int]] = []
    remaining = max(0, int(max_samples)) or None

    for shard_dir in shard_dirs:
        meta_path = shard_dir / "meta.json"
        if not meta_path.exists():
            continue
        shard_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample_lengths = np.load(shard_dir / "sample_lengths.npy", mmap_mode="r")
        shard_samples = int(shard_meta.get("num_samples", len(sample_lengths)))
        if shard_samples <= 0:
            continue
        keep = shard_samples if remaining is None else min(shard_samples, remaining)
        if keep <= 0:
            break
        token_count = int(np.asarray(sample_lengths[:keep]).sum())
        shard_plans.append((shard_dir, shard_meta, keep, token_count))
        if remaining is not None:
            remaining -= keep
            if remaining <= 0:
                break

    if not shard_plans:
        raise ValueError("Cannot save an empty pretokenized dataset.")

    total_samples = sum(keep for _, _, keep, _ in shard_plans)
    total_tokens = sum(token_count for _, _, _, token_count in shard_plans)
    first_meta = shard_plans[0][1]
    field_names = list(first_meta.get("field_names", [])) or [
        "input_ids",
        "labels",
        "signature_ids",
        "signature_level_ids",
        "signature_relation_ids",
        "parent_signature_ids",
        "signature_family_ids",
        "loss_mask",
    ]

    offsets = np.zeros(total_samples, dtype=np.int64)
    lengths = np.zeros(total_samples, dtype=np.int64)
    field_dtypes = {
        "input_ids": np.int64,
        "labels": np.int64,
        "signature_ids": np.int64,
        "signature_level_ids": np.int64,
        "signature_relation_ids": np.int64,
        "parent_signature_ids": np.int64,
        "signature_family_ids": np.int64,
        "loss_mask": np.float32,
    }
    field_memmaps: dict[str, np.ndarray] = {
        field_name: np.lib.format.open_memmap(
            output_path / f"{field_name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(total_tokens,),
        )
        for field_name, dtype in field_dtypes.items()
    }

    sample_cursor = 0
    token_cursor = 0
    for shard_dir, _, keep, token_count in shard_plans:
        shard_lengths = np.load(shard_dir / "sample_lengths.npy", mmap_mode="r")
        for local_index in range(keep):
            sample_len = int(shard_lengths[local_index])
            offsets[sample_cursor] = token_cursor
            lengths[sample_cursor] = sample_len
            sample_cursor += 1
        for field_name in field_names:
            shard_field = np.load(shard_dir / f"{field_name}.npy", mmap_mode="r")
            if field_name == "loss_mask":
                field_memmaps[field_name][token_cursor : token_cursor + token_count] = np.asarray(
                    shard_field[:token_count],
                    dtype=np.float32,
                )
            else:
                field_memmaps[field_name][token_cursor : token_cursor + token_count] = np.asarray(
                    shard_field[:token_count],
                    dtype=np.int64,
                )
        token_cursor += token_count

    for mmap in field_memmaps.values():
        mmap.flush()

    np.save(output_path / "sample_offsets.npy", offsets)
    np.save(output_path / "sample_lengths.npy", lengths)

    payload: dict[str, Any] = {
        "format_version": 1,
        "seq_len": int(seq_len),
        "num_samples": int(total_samples),
        "field_names": field_names,
        "sharded": True,
        "num_shards": int(len(shard_plans)),
    }
    if metadata:
        payload.update(metadata)
    (output_path / "meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def pretokenize_corpus(
    source_path: str | Path,
    tokenizer: PrismalTokenizer,
    *,
    seq_len: int = 512,
    max_samples: int = 0,
    output_dir: str | Path = "pretokenized",
    workers: int = 1,
) -> Path:
    """Tokenize a corpus once and write memmapped training windows."""
    source = Path(source_path)
    requested_workers = _resolve_worker_count(source, workers)
    shard_specs = _build_shard_specs(source, requested_workers)
    if len(shard_specs) <= 1:
        second_pass = StreamingTextCorpusDataset(source, tokenizer, seq_len=seq_len, max_samples=max_samples, split="all")
        return stream_pretokenized_windows(
            second_pass,
            output_dir,
            seq_len=seq_len,
            metadata={
                "source": str(source),
                "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
                "max_samples": int(max_samples),
                "workers": 1,
            },
        )

    token_state = tokenizer.to_state_dict()
    output_path = Path(output_dir)
    temp_root = Path(tempfile.mkdtemp(prefix="pretokenize_shards_", dir=str(output_path.parent)))
    shard_dirs: list[Path] = []
    print(f"[Prismal] sharded pretokenize: {len(shard_specs)} workers", flush=True)
    try:
        per_shard_max = 0
        if max_samples > 0 and len(shard_specs) > 0:
            per_shard_max = max(1, math.ceil(max_samples / len(shard_specs)))

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(shard_specs)) as executor:
            futures = []
            for shard_index, shard_spec in enumerate(shard_specs):
                shard_dir = temp_root / f"shard_{shard_index:04d}"
                shard_dirs.append(shard_dir)
                shard_dir.mkdir(parents=True, exist_ok=True)
                futures.append(
                    executor.submit(
                        _worker_pretokenize_shard,
                        token_state,
                        shard_spec,
                        seq_len=seq_len,
                        max_samples=per_shard_max,
                        output_dir=str(shard_dir),
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                shard_path = Path(future.result())
                print(f"[Prismal] shard complete: {shard_path.name}", flush=True)

        merged = _merge_pretokenized_shards(
            shard_dirs,
            output_dir,
            seq_len=seq_len,
            metadata={
                "source": str(source),
                "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
                "max_samples": int(max_samples),
                "workers": int(len(shard_specs)),
            },
            max_samples=max_samples,
        )
        print(f"[Prismal] merged {len(shard_dirs)} shards into {merged}", flush=True)
        return merged
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize a txt/jsonl/parquet corpus into memmap windows. Try demo/corpus/ for the shipped sample."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a txt, md, jsonl, parquet file, or directory; demo/corpus/ is the shipped quickstart sample",
    )
    parser.add_argument("--output-dir", default="pretokenized")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1, help="Parallel shard workers to use")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-word-tokens", type=int, default=0)
    parser.add_argument("--max-line-tokens", type=int, default=0)
    parser.add_argument("--max-signature-tokens", type=int, default=0)
    parser.add_argument("--supervised-only", dest="supervised_only", action="store_true")
    parser.add_argument("--full-text", dest="supervised_only", action="store_false")
    parser.set_defaults(supervised_only=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tokenizer = build_tokenizer_from_source(
        args.data,
        max_new_tokens=args.max_new_tokens,
        min_frequency=args.min_frequency,
        max_word_tokens=args.max_word_tokens,
        max_line_tokens=args.max_line_tokens,
        max_signature_tokens=args.max_signature_tokens,
        supervised_only=args.supervised_only,
        tokenizer_workers=args.workers,
    )
    output_path = pretokenize_corpus(
        args.data,
        tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(f"Saved pretokenized dataset to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
