# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

"""Hierarchical byte/token data utilities for the EPIC-SHARC MOHTE architecture."""

from collections import Counter
import concurrent.futures
import itertools
import os
from dataclasses import dataclass, asdict, field
import hashlib
import json
import re
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


WORD_RE = re.compile(r"[A-Za-z0-9_']+|[^\w\s]", re.UNICODE)
SEGMENT_RE = re.compile(r"[A-Za-z0-9_']+|\s+|[^\w\s]", re.UNICODE)
ANSWER_MARKER_RE = re.compile(r"\b(response|answer|output|completion|target)\s*:\s*", re.IGNORECASE)
CONTROL_LINE_RE = re.compile(
    r"^(instruction|context|prompt|input|question|response|answer|output|completion|target|text)\s*:",
    re.IGNORECASE,
)
MAX_DYNAMIC_LINE_TOKEN_CHARS = 240
CONTROL_TOKEN_TEXTS = {
    "instruction",
    "context",
    "prompt",
    "input",
    "question",
    "response",
    "answer",
    "output",
    "completion",
    "target",
}
COMMON_DYNAMIC_TOKEN_TEXTS = CONTROL_TOKEN_TEXTS | {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "many",
    "may",
    "more",
    "most",
    "not",
    "of",
    "on",
    "one",
    "or",
    "other",
    "she",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


SIGNATURE_LEVEL_IDS: Dict[str, int] = {
    "pad": 0,
    "char": 1,
    "piece": 2,
    "word": 3,
    "phrase": 4,
    "line": 5,
    "special": 6,
}

SIGNATURE_RELATION_IDS: Dict[str, int] = {
    "pad": 0,
    "exact": 1,
    "prefix": 2,
    "suffix": 3,
    "continuation": 4,
    "adjacency": 5,
    "containment": 6,
    "special": 7,
}

DEFAULT_HIERARCHY_VECTOR_DIM = 12


def _build_hierarchy_vector_tensor(
    token_ids: torch.Tensor,
    signature_ids: torch.Tensor,
    signature_level_ids: torch.Tensor,
    signature_relation_ids: torch.Tensor,
    parent_signature_ids: torch.Tensor,
    signature_family_ids: torch.Tensor,
    *,
    token_vocab_size: int,
    signature_vocab_size: int,
    level_vocab_size: int,
    relation_vocab_size: int,
    family_vocab_size: int,
) -> torch.Tensor:
    token_ids = token_ids.to(dtype=torch.float32)
    signature_ids = signature_ids.to(dtype=torch.float32)
    signature_level_ids = signature_level_ids.to(dtype=torch.float32)
    signature_relation_ids = signature_relation_ids.to(dtype=torch.float32)
    parent_signature_ids = parent_signature_ids.to(dtype=torch.float32)
    signature_family_ids = signature_family_ids.to(dtype=torch.float32)

    def _norm(values: torch.Tensor, denom: int) -> torch.Tensor:
        return values / max(float(denom), 1.0)

    token_norm = _norm(token_ids, token_vocab_size)
    signature_norm = _norm(signature_ids, signature_vocab_size)
    family_norm = _norm(signature_family_ids, family_vocab_size)
    level_norm = _norm(signature_level_ids, level_vocab_size)
    relation_norm = _norm(signature_relation_ids, relation_vocab_size)
    parent_norm = _norm(parent_signature_ids, signature_vocab_size)
    delta_norm = _norm(signature_ids - parent_signature_ids, signature_vocab_size)
    special_flag = (
        (signature_level_ids.eq(float(SIGNATURE_LEVEL_IDS["special"])))
        | (signature_relation_ids.eq(float(SIGNATURE_RELATION_IDS["special"])))
    ).to(dtype=torch.float32)
    line_flag = signature_level_ids.eq(float(SIGNATURE_LEVEL_IDS["line"])).to(dtype=torch.float32)
    word_flag = (
        signature_level_ids.eq(float(SIGNATURE_LEVEL_IDS["word"]))
        | signature_level_ids.eq(float(SIGNATURE_LEVEL_IDS["phrase"]))
    ).to(dtype=torch.float32)
    piece_flag = (
        signature_level_ids.eq(float(SIGNATURE_LEVEL_IDS["piece"]))
        | signature_level_ids.eq(float(SIGNATURE_LEVEL_IDS["char"]))
    ).to(dtype=torch.float32)
    structural_flag = (
        signature_relation_ids.eq(float(SIGNATURE_RELATION_IDS["adjacency"]))
        | signature_relation_ids.eq(float(SIGNATURE_RELATION_IDS["containment"]))
    ).to(dtype=torch.float32)

    return torch.stack(
        [
            token_norm,
            signature_norm,
            family_norm,
            level_norm,
            relation_norm,
            parent_norm,
            delta_norm,
            special_flag,
            line_flag,
            word_flag,
            piece_flag,
            structural_flag,
        ],
        dim=-1,
    )


@dataclass
class DynamicToken:
    text: str
    kind: str
    frequency: int = 0
    signature: str = ""


@dataclass
class SignatureToken:
    code: str
    kind: str
    frequency: int = 0
    family: str = ""


@dataclass
class ConstructionUnit:
    text: str
    kind: str
    render: str
    signature: str = ""
    pronunciation: str = ""


@dataclass
class ConstructionFrame:
    unit_id: int
    signature_neighborhood_id: int
    level_id: int
    relation_id: int
    parent_signature_id: int
    signature_family_id: int


@dataclass
class HierarchyEncoding:
    token_ids: List[int]
    signature_ids: List[int]
    signature_level_ids: List[int]
    signature_relation_ids: List[int]
    parent_signature_ids: List[int]
    signature_family_ids: List[int]
    hierarchy_vectors: List[List[float]] = field(default_factory=list)

    def validate(self, *, context: str = "hierarchy") -> None:
        lengths = {
            len(self.token_ids),
            len(self.signature_ids),
            len(self.signature_level_ids),
            len(self.signature_relation_ids),
            len(self.parent_signature_ids),
            len(self.signature_family_ids),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"{context} sequences must stay aligned; got lengths "
                f"token={len(self.token_ids)}, signature={len(self.signature_ids)}, "
                f"level={len(self.signature_level_ids)}, relation={len(self.signature_relation_ids)}, "
                f"parent={len(self.parent_signature_ids)}, family={len(self.signature_family_ids)}"
            )
        if self.hierarchy_vectors:
            if len(self.hierarchy_vectors) != len(self.token_ids):
                raise ValueError(
                    f"{context} hierarchy vectors must align with tokens; got "
                    f"vectors={len(self.hierarchy_vectors)} token={len(self.token_ids)}"
                )
            vector_lengths = {len(vector) for vector in self.hierarchy_vectors}
            if len(vector_lengths) != 1:
                raise ValueError(f"{context} hierarchy vectors must all share the same width; got {sorted(vector_lengths)}")

    def as_tuple(self) -> tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
        self.validate()
        return (
            self.token_ids,
            self.signature_ids,
            self.signature_level_ids,
            self.signature_relation_ids,
            self.parent_signature_ids,
            self.signature_family_ids,
        )

    def compact_tuple(self) -> tuple[List[int], List[List[float]]]:
        self.validate()
        return self.token_ids, self.hierarchy_vectors

    def prepend(
        self,
        *,
        token_id: int,
        signature_id: int,
        level_id: int,
        relation_id: int,
        parent_id: int,
        family_id: int,
    ) -> "HierarchyEncoding":
        vector_width = len(self.hierarchy_vectors[0]) if self.hierarchy_vectors else DEFAULT_HIERARCHY_VECTOR_DIM
        return HierarchyEncoding(
            token_ids=[token_id] + self.token_ids,
            signature_ids=[signature_id] + self.signature_ids,
            signature_level_ids=[level_id] + self.signature_level_ids,
            signature_relation_ids=[relation_id] + self.signature_relation_ids,
            parent_signature_ids=[parent_id] + self.parent_signature_ids,
            signature_family_ids=[family_id] + self.signature_family_ids,
            hierarchy_vectors=[[0.0] * vector_width] + self.hierarchy_vectors,
        )

    def trim_trailing_tokens(self, token_ids: set[int]) -> "HierarchyEncoding":
        if not token_ids:
            return self
        trim_len = len(self.token_ids)
        while trim_len > 0 and self.token_ids[trim_len - 1] in token_ids:
            trim_len -= 1
        if trim_len == len(self.token_ids):
            return self
        return HierarchyEncoding(
            token_ids=self.token_ids[:trim_len],
            signature_ids=self.signature_ids[:trim_len],
            signature_level_ids=self.signature_level_ids[:trim_len],
            signature_relation_ids=self.signature_relation_ids[:trim_len],
            parent_signature_ids=self.parent_signature_ids[:trim_len],
            signature_family_ids=self.signature_family_ids[:trim_len],
            hierarchy_vectors=self.hierarchy_vectors[:trim_len],
        )


def _clean_record_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(parts).strip()
    return str(value).strip()


def _compose_record_text(payload: Dict[str, Any]) -> str:
    raw_text = _clean_record_value(payload.get("text") or payload.get("content"))
    if len(payload) == 1 and raw_text:
        return raw_text

    structured_fields = [
        ("Instruction", ("instruction",)),
        ("Context", ("context",)),
        ("Input", ("input",)),
        ("Prompt", ("prompt",)),
        ("Question", ("question",)),
        ("Response", ("response", "output", "answer", "completion", "target")),
    ]
    input_pieces: List[str] = []
    output_pieces: List[str] = []
    used_keys: set[str] = set()
    for label, keys in structured_fields:
        value = ""
        for key in keys:
            candidate = _clean_record_value(payload.get(key))
            if candidate:
                value = candidate
                used_keys.add(key)
                break
        if not value:
            continue
        if label == "Response":
            output_pieces.append(f"{label}: {value}")
        else:
            input_pieces.append(f"{label}: {value}")

    if raw_text:
        if not input_pieces and not output_pieces:
            return raw_text
        if output_pieces and not input_pieces:
            input_pieces.append(f"Text: {raw_text}")
        elif input_pieces:
            input_pieces.append(f"Text: {raw_text}")

    if input_pieces or output_pieces:
        pieces: List[str] = []
        if input_pieces:
            pieces.append("<BOI>")
            pieces.extend(input_pieces)
            pieces.append("<EOI>")
        if output_pieces:
            pieces.append("<BOO>")
            pieces.extend(output_pieces)
            pieces.append("<EOO>")
        return "\n".join(pieces).strip()

    fallback_keys = [key for key in payload.keys() if key not in used_keys]
    for key in fallback_keys:
        value = _clean_record_value(payload.get(key))
        if value:
            return f"{key}: {value}" if key else value

    return raw_text


def iter_text_corpus(source: str | Path) -> Iterator[str]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Text source not found: {source}")

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
                yield from iter_text_corpus(subpath)
            elif suffix == ".parquet":
                yield from _iter_parquet_texts(subpath)
        return

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_texts(path)
        return
    if suffix == ".parquet":
        yield from _iter_parquet_texts(path)
        return

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        yield text


def _chunk_text_iterable(texts: Iterable[str], chunk_size: int) -> Iterator[list[str]]:
    chunk_size = max(1, int(chunk_size))
    chunk: list[str] = []
    for text in texts:
        chunk.append(text)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _scan_tokenizer_learning_chunk(tokenizer_state: dict[str, Any], texts: Sequence[str]) -> tuple[Counter[str], Counter[str]]:
    tokenizer = PrismalTokenizer.from_state_dict(tokenizer_state)
    return tokenizer._collect_construction_learning_counts(texts)


def _iter_jsonl_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                merged = _compose_record_text(payload)
                if merged:
                    yield merged
            elif isinstance(payload, str) and payload.strip():
                yield payload.strip()


def _iter_parquet_texts(path: Path) -> Iterator[str]:
    projected_columns = [
        "text",
        "content",
        "body",
        "document",
        "instruction",
        "context",
        "input",
        "prompt",
        "response",
        "output",
        "answer",
        "completion",
        "target",
    ]

    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        pq = None

    if pq is not None:
        parquet_file = pq.ParquetFile(path)
        available_columns = [name for name in projected_columns if name in parquet_file.schema.names]
        if available_columns:
            # Project just the likely text columns first. Most corpora only need a
            # small subset of the row, so avoiding full-row dict materialization
            # makes tokenizer bootstrap much cheaper on large parquet files.
            if len(available_columns) == 1 and available_columns[0] in {"text", "content", "body", "document"}:
                column_name = available_columns[0]
                for batch in parquet_file.iter_batches(columns=[column_name]):
                    for value in batch.column(0).to_pylist():
                        merged = _clean_record_value(value)
                        if merged:
                            yield merged
                return

            for batch in parquet_file.iter_batches(columns=available_columns):
                columns = {name: batch.column(index).to_pylist() for index, name in enumerate(available_columns)}
                for row_index in range(batch.num_rows):
                    payload = {name: values[row_index] for name, values in columns.items()}
                    merged = _compose_record_text(payload)
                    if merged:
                        yield merged
            return

        for batch in parquet_file.iter_batches():
            for payload in batch.to_pylist():
                merged = _compose_record_text(payload)
                if merged:
                    yield merged
        return

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency already installed in this workspace
        raise ImportError("Reading parquet corpora requires pandas with a parquet engine installed.") from exc

    df = pd.read_parquet(path)
    if df.empty:
        return

    available_columns = [name for name in projected_columns if name in df.columns]
    if len(available_columns) == 1 and available_columns[0] in {"text", "content", "body", "document"}:
        for value in df[available_columns[0]].tolist():
            merged = _clean_record_value(value)
            if merged:
                yield merged
        return

    # `itertuples()` keeps the scan much lighter while preserving the same payload shape.
    columns = list(df.columns)
    for row in df.itertuples(index=False, name=None):
        payload = {key: value for key, value in zip(columns, row)}
        merged = _compose_record_text(payload)
        if merged:
            yield merged


class PrismalTokenizer:
    """Byte tokenizer with a handcrafted base alphabet plus learned construction units."""

    base_vocab_size: int = 0
    codec_version: int = 5
    _COMMON_WORDS_AS_WHOLE_UNITS = COMMON_DYNAMIC_TOKEN_TEXTS | {"than", "then", "them", "there", "are"}
    _CONSTRUCTION_PIECES: tuple[str, ...] = (
        "tion",
        "sion",
        "ment",
        "ness",
        "able",
        "ible",
        "ing",
        "ers",
        "est",
        "ed",
        "er",
        "ly",
        "es",
        "ar",
        "re",
        "ra",
        "th",
        "he",
        "in",
        "an",
        "en",
        "nd",
        "ng",
        "st",
        "sh",
        "ch",
        "ph",
        "wh",
        "qu",
        "ck",
        "ai",
        "ay",
        "ee",
        "ea",
        "oo",
        "ou",
        "ow",
        "oi",
        "oy",
        "ie",
        "ei",
        "or",
        "ur",
        "ir",
        "al",
        "le",
        "ve",
        "ce",
        "se",
        "de",
        "ne",
        "me",
        "ri",
        "ro",
        "ic",
        "io",
        "is",
        "as",
        "at",
        "to",
        "of",
        "it",
        "ha",
        "hi",
        "co",
    )
    _PUNCT_UNITS: tuple[str, ...] = (
        ".",
        ",",
        "?",
        "!",
        ":",
        ";",
        "'",
        '"',
        "-",
        "_",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "/",
        "\\",
        "@",
        "#",
        "$",
        "%",
        "&",
        "*",
        "+",
        "=",
        "<",
        ">",
    )

    def __init__(self, *, use_pronunciation_signatures: bool = True) -> None:
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.use_pronunciation_signatures = bool(use_pronunciation_signatures)

        self.dynamic_tokens: List[DynamicToken] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {
            self.pad_id: "<PAD>",
            self.bos_id: "<BOS>",
            self.eos_id: "<EOS>",
        }
        self.token_kind_by_id: Dict[int, str] = {
            self.pad_id: "special",
            self.bos_id: "special",
            self.eos_id: "special",
        }
        self.token_frequency_by_id: Dict[int, int] = {
            self.pad_id: 0,
            self.bos_id: 0,
            self.eos_id: 0,
        }
        self.signature_by_id: Dict[int, str] = {
            self.pad_id: "pad",
            self.bos_id: "bos",
            self.eos_id: "eos",
        }
        self.skip_decode_ids = {self.pad_id, self.bos_id}

        self.signature_pad_id = 0
        self.signature_bos_id = 1
        self.signature_eos_id = 2
        self.signature_boi_id = 8
        self.signature_eoi_id = 9
        self.signature_boo_id = 10
        self.signature_eoo_id = 11
        self.signature_bop_id = 12
        self.signature_eop_id = 13
        self.signature_blo_id = 14
        self.signature_special_ids = {
            "<SIGPAD>": self.signature_pad_id,
            "<SIGBOS>": self.signature_bos_id,
            "<SIGEOS>": self.signature_eos_id,
            "<LINE>": 3,
            "<EOL>": 4,
            "<SPACE>": 5,
            "<PUNCT>": 6,
            "<OTHER>": 7,
            "<SIGBOI>": self.signature_boi_id,
            "<SIGEOI>": self.signature_eoi_id,
            "<SIGBOO>": self.signature_boo_id,
            "<SIGEOO>": self.signature_eoo_id,
            "<SIGBOP>": self.signature_bop_id,
            "<SIGEOP>": self.signature_eop_id,
            "<SIGBLO>": self.signature_blo_id,
        }
        self._signature_id_to_code: Dict[int, str] = {
            self.signature_pad_id: "<SIGPAD>",
            self.signature_bos_id: "<SIGBOS>",
            self.signature_eos_id: "<SIGEOS>",
            3: "<LINE>",
            4: "<EOL>",
            5: "<SPACE>",
            6: "<PUNCT>",
            7: "<OTHER>",
            self.signature_boi_id: "<SIGBOI>",
            self.signature_eoi_id: "<SIGEOI>",
            self.signature_boo_id: "<SIGBOO>",
            self.signature_eoo_id: "<SIGEOO>",
            self.signature_bop_id: "<SIGBOP>",
            self.signature_eop_id: "<SIGEOP>",
            self.signature_blo_id: "<SIGBLO>",
        }
        self.signature_kind_by_id: Dict[int, str] = {
            self.signature_pad_id: "special",
            self.signature_bos_id: "special",
            self.signature_eos_id: "special",
            3: "structure",
            4: "structure",
            5: "structure",
            6: "structure",
            7: "fallback",
            self.signature_boi_id: "boundary",
            self.signature_eoi_id: "boundary",
            self.signature_boo_id: "boundary",
            self.signature_eoo_id: "boundary",
            self.signature_bop_id: "boundary",
            self.signature_eop_id: "boundary",
            self.signature_blo_id: "line",
        }
        self.signature_frequency_by_id: Dict[int, int] = {idx: 0 for idx in self._signature_id_to_code}
        self.signature_tokens: List[SignatureToken] = []
        self.signature_to_id: Dict[str, int] = {code: idx for idx, code in self._signature_id_to_code.items()}
        self.signature_family_to_id: Dict[str, int] = {
            "pad": 0,
            "special": 1,
            "line": 2,
            "fallback": 3,
            "boundary": 4,
        }
        self._include_boundary_family = True
        self.signature_family_by_id: Dict[int, str] = {
            idx: family for family, idx in self.signature_family_to_id.items()
        }
        self.signature_family_id_by_signature_id: Dict[int, int] = {
            idx: self.signature_family_to_id.get(self._signature_family_key(code), self.signature_family_to_id["fallback"])
            for idx, code in self._signature_id_to_code.items()
        }
        self.signature_family_id_by_signature_id[self.signature_boo_id] = self.signature_family_to_id["boundary"]
        self.signature_family_id_by_signature_id[self.signature_eoo_id] = self.signature_family_to_id["boundary"]
        self.signature_family_id_by_signature_id[self.signature_boi_id] = self.signature_family_to_id["boundary"]
        self.signature_family_id_by_signature_id[self.signature_eoi_id] = self.signature_family_to_id["boundary"]
        self.signature_family_id_by_signature_id[self.signature_bop_id] = self.signature_family_to_id["boundary"]
        self.signature_family_id_by_signature_id[self.signature_eop_id] = self.signature_family_to_id["boundary"]
        self.signature_family_id_by_signature_id[self.signature_blo_id] = self.signature_family_to_id["line"]
        self.token_signature_id_by_id: Dict[int, int] = {
            self.pad_id: self.signature_pad_id,
            self.bos_id: self.signature_bos_id,
            self.eos_id: self.signature_eos_id,
        }
        self._signature_trie: Dict[str, dict] = {}
        self.signature_level_to_id: Dict[str, int] = dict(SIGNATURE_LEVEL_IDS)
        self.signature_relation_to_id: Dict[str, int] = dict(SIGNATURE_RELATION_IDS)

        self.special_tokens = {
            "<BOI>": self.add_token("<BOI>", kind="structure", frequency=1, signature="boundary"),
            "<EOI>": self.add_token("<EOI>", kind="structure", frequency=1, signature="boundary"),
            "<BOO>": self.add_token("<BOO>", kind="structure", frequency=1, signature="boundary"),
            "<EOO>": self.add_token("<EOO>", kind="structure", frequency=1, signature="boundary"),
            "<BOP>": self.add_token("<BOP>", kind="structure", frequency=1, signature="boundary"),
            "<EOP>": self.add_token("<EOP>", kind="structure", frequency=1, signature="boundary"),
            "<BLO>": self.add_token("<BLO>", kind="structure", frequency=1, signature="line"),
            "<LINE>": self.add_token("<LINE>", kind="structure", frequency=1, signature="line"),
            "<EOL>": self.add_token("<EOL>", kind="structure", frequency=1, signature="newline"),
            "<SIG:OTHER>": self.add_token("<SIG:OTHER>", kind="signature", frequency=1, signature="other"),
        }
        for special_text in ("<BOI>", "<EOI>", "<BOO>", "<BOP>", "<EOP>"):
            self.skip_decode_ids.add(self.special_tokens[special_text])

        self._trie: Dict[str, dict] = {}
        self._sorted_dynamic_tokens: List[str] = []
        self.construction_units: List[ConstructionUnit] = []
        self.construction_text_to_id: Dict[str, int] = {}
        self._construction_piece_ids: Dict[str, int] = {}
        self._byte_fallback_ids: Dict[int, int] = {}
        self._install_construction_vocabulary()

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.dynamic_tokens)

    @property
    def signature_vocab_size(self) -> int:
        return max(self.signature_to_id.values(), default=0) + 1

    @property
    def signature_level_vocab_size(self) -> int:
        return max(self.signature_level_to_id.values(), default=0) + 1

    @property
    def signature_relation_vocab_size(self) -> int:
        return max(self.signature_relation_to_id.values(), default=0) + 1

    @property
    def signature_family_vocab_size(self) -> int:
        return max(self.signature_family_to_id.values(), default=0) + 1

    def _install_construction_vocabulary(self) -> None:
        """Install the v2 compact construction alphabet."""

        self.dynamic_tokens = []
        self.construction_units = []
        self.construction_text_to_id = {}
        self._construction_piece_ids = {}
        self._byte_fallback_ids = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_kind_by_id = {}
        self.token_frequency_by_id = {}
        self.signature_by_id = {}
        self.token_signature_id_by_id = {}
        self.special_tokens = {}
        self.skip_decode_ids = {self.pad_id, self.bos_id}
        self._trie = {}
        self._sorted_dynamic_tokens = []

        def add_unit(text: str, kind: str, *, render: str, signature: str = "") -> int:
            unit_id = len(self.construction_units)
            unit = ConstructionUnit(text=text, kind=kind, render=render, signature=signature or kind)
            self.construction_units.append(unit)
            self.id_to_token[unit_id] = text
            self.token_kind_by_id[unit_id] = kind
            self.token_frequency_by_id[unit_id] = 1
            self.signature_by_id[unit_id] = unit.signature
            self.token_to_id[text.lower()] = unit_id
            self.construction_text_to_id[text] = unit_id
            self.token_signature_id_by_id[unit_id] = self.signature_id_for_code(unit.signature)
            return unit_id

        # Fixed ids for core specials.
        self.pad_id = add_unit("<PAD>", "special", render="", signature="pad")
        self.bos_id = add_unit("<BOS>", "special", render="", signature="bos")
        self.eos_id = add_unit("<EOS>", "special", render="", signature="eos")
        self.special_tokens = {
            "<BOI>": add_unit("<BOI>", "structure", render="", signature="boundary"),
            "<EOI>": add_unit("<EOI>", "structure", render="", signature="boundary"),
            "<BOO>": add_unit("<BOO>", "structure", render="", signature="boundary"),
            "<EOO>": add_unit("<EOO>", "structure", render="", signature="boundary"),
            "<BOP>": add_unit("<BOP>", "structure", render="", signature="boundary"),
            "<EOP>": add_unit("<EOP>", "structure", render="", signature="boundary"),
            "<BLO>": add_unit("<BLO>", "structure", render="", signature="line"),
            "<LINE>": add_unit("<LINE>", "structure", render="\n", signature="line"),
            "<EOL>": add_unit("<EOL>", "structure", render="\n", signature="newline"),
            "<SPACE>": add_unit("<SPACE>", "space", render=" ", signature="<SPACE>"),
            "<TAB>": add_unit("<TAB>", "space", render="\t", signature="<SPACE>"),
            "<CAP>": add_unit("<CAP>", "case", render="", signature="case|cap"),
            "<UPPER>": add_unit("<UPPER>", "case", render="", signature="case|upper"),
            "<SIG:OTHER>": add_unit("<SIG:OTHER>", "signature", render="", signature="other"),
        }
        self.skip_decode_ids.update(
            {
                self.special_tokens["<BOI>"],
                self.special_tokens["<EOI>"],
                self.special_tokens["<BOO>"],
                self.special_tokens["<CAP>"],
                self.special_tokens["<UPPER>"],
                self.special_tokens["<SIG:OTHER>"],
            }
        )

        for ch in "abcdefghijklmnopqrstuvwxyz":
            add_unit(ch, "char", render=ch, signature=self._construction_signature_code(ch, "char"))
        for digit in "0123456789":
            add_unit(digit, "digit", render=digit, signature=self._construction_signature_code(digit, "digit"))
        for punct in self._PUNCT_UNITS:
            add_unit(punct, "punct", render=punct, signature=self._construction_signature_code(punct, "punct"))

        for piece in sorted(dict.fromkeys(self._CONSTRUCTION_PIECES), key=lambda value: (-len(value), value)):
            if piece in self._COMMON_WORDS_AS_WHOLE_UNITS:
                continue
            if piece in self.construction_text_to_id:
                continue
            unit_id = add_unit(piece, "piece", render=piece, signature=self._construction_signature_code(piece, "piece"))
            self._construction_piece_ids[piece] = unit_id
            node = self._trie
            for ch in piece:
                node = node.setdefault(ch, {})
            node["$"] = unit_id

        for byte_value in range(256):
            text = f"<BYTE:{byte_value:02x}>"
            render = text
            unit_id = add_unit(text, "byte", render=render, signature=self._byte_signature_code(byte_value))
            self._byte_fallback_ids[byte_value] = unit_id

        self.base_vocab_size = len(self.construction_units)
        for unit_id, unit in enumerate(self.construction_units):
            if unit.kind in {"char", "digit", "punct", "space", "case", "piece", "byte"}:
                self.token_signature_id_by_id[unit_id] = self.signature_id_for_code(unit.signature)
        self._rebuild_signature_index()

    def _append_construction_unit(
        self,
        text: str,
        *,
        kind: str,
        render: str | None = None,
        signature: str = "",
        pronunciation: str = "",
    ) -> int:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            raise ValueError("Construction unit text cannot be empty.")
        existing = self.construction_text_to_id.get(normalized)
        if existing is not None:
            return existing

        unit_id = len(self.construction_units)
        render_text = normalized if render is None else render
        unit = ConstructionUnit(
            text=normalized,
            kind=kind,
            render=render_text,
            signature=signature or kind,
            pronunciation=pronunciation,
        )
        self.construction_units.append(unit)
        self.id_to_token[unit_id] = normalized
        self.token_kind_by_id[unit_id] = kind
        self.token_frequency_by_id[unit_id] = max(1, self.token_frequency_by_id.get(unit_id, 1))
        self.signature_by_id[unit_id] = unit.signature
        self.token_to_id[normalized.lower()] = unit_id
        self.construction_text_to_id[normalized] = unit_id
        self.token_signature_id_by_id[unit_id] = self.signature_id_for_code(unit.signature)
        if kind == "piece":
            self._construction_piece_ids[normalized] = unit_id
            node = self._trie
            for ch in normalized.lower():
                node = node.setdefault(ch, {})
            node["$"] = unit_id
        elif kind == "byte":
            match = re.fullmatch(r"<BYTE:([0-9a-fA-F]{2})>", normalized)
            if match:
                self._byte_fallback_ids[int(match.group(1), 16)] = unit_id
        self.base_vocab_size = len(self.construction_units)
        return unit_id

    def _collect_construction_unit_counts(self, texts: Iterable[str]) -> Counter[str]:
        candidate_counts: Counter[str] = Counter()
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            for raw_line in text.splitlines() or [text]:
                line = re.sub(r"\s+", " ", raw_line.strip())
                if not line:
                    continue
                for segment_match in SEGMENT_RE.finditer(line):
                    segment = segment_match.group(0)
                    if not self._is_word_like(segment):
                        continue
                    normalized = segment.lower().strip()
                    if normalized in self._COMMON_WORDS_AS_WHOLE_UNITS:
                        continue
                    if normalized in {"www", "http", "https"}:
                        continue
                    if not self._token_quality_ok(normalized, kind="word"):
                        continue
                    if normalized in self.construction_text_to_id:
                        continue
                    candidate_counts[normalized] += 1
        return candidate_counts

    def _learn_construction_units(
        self,
        texts: Iterable[str],
        *,
        max_new_tokens: int = 0,
        min_frequency: int = 1,
        max_word_tokens: int = 0,
        candidate_counts: Counter[str] | None = None,
    ) -> None:
        if candidate_counts is None:
            candidate_counts = self._collect_construction_unit_counts(texts)

        if not candidate_counts:
            return

        max_new_tokens = max(0, int(max_new_tokens))
        min_frequency = max(1, int(min_frequency))
        max_word_tokens = max(0, int(max_word_tokens))
        max_new_tokens_limit = None if max_new_tokens <= 0 else max_new_tokens
        max_word_tokens_limit = None if max_word_tokens <= 0 else max_word_tokens

        def sort_key(item: tuple[str, int]) -> tuple[int, int, str]:
            word, freq = item
            return (freq, len(word), word)

        learned = 0
        primary_candidates = sorted(
            ((word, freq) for word, freq in candidate_counts.items() if freq >= min_frequency),
            key=sort_key,
            reverse=True,
        )
        fallback_candidates = sorted(
            ((word, freq) for word, freq in candidate_counts.items() if freq < min_frequency),
            key=sort_key,
            reverse=True,
        )
        for word, freq in primary_candidates + fallback_candidates:
            if max_new_tokens_limit is not None and learned >= max_new_tokens_limit:
                break
            if max_word_tokens_limit is not None and learned >= max_word_tokens_limit:
                break
            if word in self.construction_text_to_id:
                continue
            self._append_construction_unit(
                word,
                kind="piece",
                render=word,
                signature=self._construction_signature_code(word, "piece"),
                pronunciation=self._pronunciation_code(word),
            )
            self.token_frequency_by_id[self.construction_text_to_id[word]] = freq
            learned += 1

    def _restore_construction_vocabulary(self, construction_units: Sequence[object]) -> None:
        """Restore the learned construction alphabet from a checkpoint payload."""

        self.dynamic_tokens = []
        self.construction_units = []
        self.construction_text_to_id = {}
        self._construction_piece_ids = {}
        self._byte_fallback_ids = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_kind_by_id = {}
        self.token_frequency_by_id = {}
        self.signature_by_id = {}
        self.token_signature_id_by_id = {}
        self.special_tokens = {}
        self.skip_decode_ids = {self.pad_id, self.bos_id}
        self._trie = {}
        self._sorted_dynamic_tokens = []

        for item in construction_units:
            if not isinstance(item, dict):
                continue
            unit = ConstructionUnit(
                text=str(item.get("text", "")),
                kind=str(item.get("kind", "char")),
                render=str(item.get("render", "")),
                signature=str(item.get("signature", "")),
                pronunciation=str(item.get("pronunciation", "")),
            )
            unit_id = len(self.construction_units)
            if not unit.pronunciation and unit.kind in {"word", "phrase", "piece"} and unit.text:
                unit.pronunciation = self._pronunciation_code(unit.text)
            self.construction_units.append(unit)
            self.id_to_token[unit_id] = unit.text
            self.token_kind_by_id[unit_id] = unit.kind
            self.token_frequency_by_id[unit_id] = 1
            self.signature_by_id[unit_id] = unit.signature
            self.token_to_id[unit.text.lower()] = unit_id
            self.construction_text_to_id[unit.text] = unit_id
            self.token_signature_id_by_id[unit_id] = self.signature_id_for_code(unit.signature)

        self.base_vocab_size = len(self.construction_units)
        if self.base_vocab_size == 0:
            self._rebuild_signature_index()
            return

        for special_text in ("<BOI>", "<EOI>", "<BOO>", "<EOO>", "<BOP>", "<EOP>", "<BLO>", "<LINE>", "<EOL>", "<SPACE>", "<TAB>", "<CAP>", "<UPPER>", "<SIG:OTHER>"):
            token_id = self.construction_text_to_id.get(special_text)
            if token_id is not None:
                self.special_tokens[special_text] = token_id
        self.skip_decode_ids = {self.pad_id, self.bos_id}
        for special_text in ("<BOI>", "<EOI>", "<BOO>", "<EOO>", "<BOP>", "<EOP>", "<CAP>", "<UPPER>", "<SIG:OTHER>"):
            token_id = self.special_tokens.get(special_text)
            if token_id is not None:
                self.skip_decode_ids.add(token_id)

        self._include_boundary_family = any(unit.text in {"<BOI>", "<EOI>", "<BOO>", "<EOO>", "<BOP>", "<EOP>"} for unit in self.construction_units)
        for unit_id, unit in enumerate(self.construction_units):
            if unit.kind == "piece":
                self._construction_piece_ids[unit.text] = unit_id
                node = self._trie
                for ch in unit.text:
                    node = node.setdefault(ch, {})
                node["$"] = unit_id
            elif unit.kind == "byte":
                match = re.fullmatch(r"<BYTE:([0-9a-fA-F]{2})>", unit.text)
                if match:
                    self._byte_fallback_ids[int(match.group(1), 16)] = unit_id

        self._rebuild_signature_index()
        for unit_id, unit in enumerate(self.construction_units):
            self.token_signature_id_by_id[unit_id] = self.signature_id_for_code(unit.signature)

    def _rebuild_index(self) -> None:
        self.token_to_id.clear()
        self._trie = {}
        self._sorted_dynamic_tokens = sorted((tok.text for tok in self.dynamic_tokens), key=len, reverse=True)
        for idx, token in enumerate(self.dynamic_tokens, start=self.base_vocab_size):
            self.token_to_id[token.text] = idx
            self.id_to_token[idx] = token.text
            self.token_kind_by_id[idx] = token.kind
            self.token_frequency_by_id[idx] = token.frequency
            self.signature_by_id[idx] = token.signature
            if token.signature and token.signature in self.signature_to_id:
                self.token_signature_id_by_id[idx] = self.signature_to_id[token.signature]
            elif token.kind == "line":
                self.token_signature_id_by_id[idx] = self.signature_id_for_line(token.text)
            elif token.kind in {"word", "phrase"}:
                self.token_signature_id_by_id[idx] = self.signature_id_for_word(token.text)
            else:
                self.token_signature_id_by_id[idx] = self.signature_special_ids["<OTHER>"]
            node = self._trie
            for ch in token.text.lower():
                node = node.setdefault(ch, {})
            node["$"] = idx

    def _rebuild_signature_index(self) -> None:
        self.signature_to_id = {code: idx for idx, code in self._signature_id_to_code.items()}
        self._signature_trie = {}
        self.signature_family_to_id = {
            "pad": 0,
            "special": 1,
            "line": 2,
            "fallback": 3,
        }
        if getattr(self, "_include_boundary_family", True):
            self.signature_family_to_id["boundary"] = len(self.signature_family_to_id)
        self.signature_family_by_id = {idx: family for family, idx in self.signature_family_to_id.items()}
        self.signature_family_id_by_signature_id = {}
        for token in self.signature_tokens:
            if token.code in self.signature_to_id:
                continue
            idx = len(self.signature_to_id)
            self.signature_to_id[token.code] = idx
            self._signature_id_to_code[idx] = token.code
            self.signature_kind_by_id[idx] = token.kind
            self.signature_frequency_by_id[idx] = token.frequency
            family_key = token.family or self._signature_family_key(token.code)
            family_id = self.signature_family_to_id.get(family_key)
            if family_id is None:
                family_id = len(self.signature_family_to_id)
                self.signature_family_to_id[family_key] = family_id
                self.signature_family_by_id[family_id] = family_key
            self.signature_family_id_by_signature_id[idx] = family_id
            node = self._signature_trie
            for ch in token.code.lower():
                node = node.setdefault(ch, {})
            node["$"] = idx
        for idx, code in self._signature_id_to_code.items():
            family_key = self._signature_family_key(code)
            family_id = self.signature_family_to_id.get(family_key)
            if family_id is None:
                family_id = len(self.signature_family_to_id)
                self.signature_family_to_id[family_key] = family_id
                self.signature_family_by_id[family_id] = family_key
            self.signature_family_id_by_signature_id[idx] = family_id

    @staticmethod
    def _is_word_like(text: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_']*", text))

    @staticmethod
    def _char_class(ch: str) -> str:
        if ch.lower() in "aeiouy":
            return "V"
        if ch.isalpha():
            return "C"
        if ch.isdigit():
            return "D"
        if ch.isspace():
            return "S"
        return "P"

    @classmethod
    def _shape_code(cls, text: str) -> str:
        collapsed: List[str] = []
        prev = ""
        for ch in text:
            tag = cls._char_class(ch)
            if tag != prev:
                collapsed.append(tag)
                prev = tag
        return "".join(collapsed)[:24] or "OTHER"

    @staticmethod
    def _vowel_profile(text: str) -> str:
        profile = [ch.lower() for ch in text if ch.lower() in "aeiouy"]
        if not profile:
            return "0"
        compact: List[str] = []
        prev = ""
        for ch in profile:
            if ch != prev:
                compact.append(ch)
                prev = ch
        return "".join(compact)[:12]

    @staticmethod
    def _pronunciation_code(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9']", "", text.lower())
        if not normalized:
            return "phon|empty"
        digraphs = (
            ("augh", "af"),
            ("ough", "off"),
            ("eigh", "ay"),
            ("igh", "ay"),
            ("tion", "shun"),
            ("sion", "zhun"),
            ("tch", "ch"),
            ("dge", "j"),
            ("ph", "f"),
            ("qu", "kw"),
            ("ck", "k"),
            ("wh", "w"),
            ("wr", "r"),
            ("kn", "n"),
            ("gn", "n"),
            ("x", "ks"),
        )
        for source, target in digraphs:
            normalized = normalized.replace(source, target)

        sounds: List[str] = []
        prev = ""
        for ch in normalized:
            if ch in "aeiouy":
                tag = "V"
            elif ch.isdigit():
                tag = ch
            elif ch == "'":
                continue
            else:
                tag = ch
            if tag != prev:
                sounds.append(tag)
                prev = tag
        if not sounds:
            sounds = ["0"]
        return f"phon|sk={''.join(sounds)[:20]}|len={len(normalized)}"

    @staticmethod
    def _stem_heuristic(word: str) -> str:
        stem = word.lower().strip()
        suffixes = (
            "ingly",
            "edly",
            "ing",
            "edly",
            "ed",
            "er",
            "est",
            "ly",
            "ies",
            "ied",
            "es",
            "s",
            "ment",
            "ness",
            "able",
            "ible",
            "tion",
            "sion",
            "al",
            "ity",
            "ize",
        )
        for suffix in suffixes:
            if len(stem) > len(suffix) + 2 and stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _word_signature_code(self, word: str) -> str:
        normalized = word.lower().strip()
        shape = self._shape_code(normalized)
        vowel_profile = self._vowel_profile(normalized)
        pronunciation = self._pronunciation_code(normalized) if self.use_pronunciation_signatures else "phon|off"
        stem = self._stem_heuristic(normalized)
        prefix = normalized[:3] if normalized else ""
        suffix = normalized[-3:] if len(normalized) >= 3 else normalized
        token_kind = "code" if any(ch.isdigit() or ch == "_" for ch in normalized) else "word"
        return f"{token_kind}|sh={shape}|v={vowel_profile}|{pronunciation}|st={stem}|pre={prefix}|suf={suffix}|len={len(normalized)}"

    def _line_signature_code(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        if not normalized:
            return "line|empty"
        indent = len(text) - len(text.lstrip(" \t"))
        word_count = len(WORD_RE.findall(normalized))
        punct_count = sum(1 for ch in normalized if not ch.isalnum() and not ch.isspace())
        code_like = any(ch.isdigit() or ch == "_" for ch in normalized) or punct_count > max(1, len(normalized) // 6)
        mode = "code" if code_like else "text"
        shape = self._shape_code(normalized)
        vowel_profile = self._vowel_profile(normalized)
        return f"line|mode={mode}|ind={indent}|wc={word_count}|pc={punct_count}|sh={shape}|v={vowel_profile}|len={len(normalized)}"

    def _segment_signature_code(self, segment: str, *, line_code: str, kind: str) -> str:
        normalized = segment.lower().strip()
        if not normalized:
            return "<OTHER>"
        if kind == "space":
            return "<SPACE>"
        if kind == "punct":
            return f"punct|ch={normalized[:4]}|len={len(normalized)}"
        if kind == "word":
            return self._word_signature_code(normalized)
        return f"{kind}|line={line_code[:48]}|sh={self._shape_code(normalized)}|len={len(normalized)}"

    def _byte_signature_code(self, byte_value: int) -> str:
        byte_value = int(byte_value) & 0xFF
        ch = bytes([byte_value]).decode("utf-8", errors="ignore")
        if not ch:
            return f"byte|hex={byte_value:02x}|class=nonutf8"
        if ch.isspace():
            return "<SPACE>"
        if ch.isalpha():
            lower = ch.lower()
            case = "upper" if ch.isupper() else "lower"
            return f"char|ch={lower}|case={case}|class=alpha"
        if ch.isdigit():
            return f"char|ch={ch}|class=digit"
        return f"char|ch={ch}|class=punct"

    def _construction_signature_code(self, text: str, kind: str) -> str:
        normalized = text.lower()
        if kind == "space":
            return "<SPACE>"
        if kind == "case":
            return f"case|mode={normalized.strip('<>').lower()}"
        if kind == "byte":
            return f"byte|unit={normalized}"
        if kind == "punct":
            return f"punct|ch={normalized[:4]}|len={len(normalized)}"
        shape = self._shape_code(normalized)
        vowels = self._vowel_profile(normalized)
        pronunciation = self._pronunciation_code(normalized) if self.use_pronunciation_signatures else "phon|off"
        if kind == "piece":
            edge = "rime" if normalized and normalized[0] in "aeiouy" else "onset"
            return f"piece|edge={edge}|sh={shape}|v={vowels}|{pronunciation}|len={len(normalized)}"
        if kind == "digit":
            return "digit|class=numeric"
        return f"char|ch={normalized}|class={'alpha' if normalized.isalpha() else kind}"

    def _unit_id_for_text(self, text: str) -> int | None:
        return self.construction_text_to_id.get(text)

    def _byte_fallback_unit(self, byte_value: int) -> int:
        byte_value = int(byte_value) & 0xFF
        return self._byte_fallback_ids.get(byte_value, self.special_tokens["<SIG:OTHER>"])

    def _decompose_word(self, word: str) -> List[tuple[int, str, str]]:
        pieces: List[tuple[int, str, str]] = []
        if not word:
            return pieces
        if len(word) > 1 and word.isupper():
            pieces.append((self.special_tokens["<UPPER>"], "case", ""))
        elif word[0].isupper():
            pieces.append((self.special_tokens["<CAP>"], "case", ""))

        lowered = word.lower()
        i = 0
        while i < len(lowered):
            match_text = ""
            match_id: int | None = None
            node = self._trie
            j = i
            while j < len(lowered):
                ch = lowered[j]
                if ch not in node:
                    break
                node = node[ch]
                j += 1
                candidate_id = node.get("$")
                if isinstance(candidate_id, int):
                    candidate = lowered[i:j]
                    if candidate not in self._COMMON_WORDS_AS_WHOLE_UNITS:
                        match_text = candidate
                        match_id = candidate_id
            if match_id is not None and match_text:
                pieces.append((match_id, "piece", match_text))
                i += len(match_text)
                continue

            ch = lowered[i]
            unit_id = self._unit_id_for_text(ch)
            if unit_id is not None:
                kind = "digit" if ch.isdigit() else "char" if ch.isalpha() else "punct"
                pieces.append((unit_id, kind, ch))
            else:
                for b in word[i].encode("utf-8", errors="ignore"):
                    pieces.append((self._byte_fallback_unit(b), "byte", f"<BYTE:{b:02x}>"))
            i += 1
        return pieces

    def _relation_for_piece(self, piece_index: int, piece_count: int) -> int:
        if piece_count <= 1:
            return self.signature_relation_to_id["exact"]
        if piece_index == 0:
            return self.signature_relation_to_id["prefix"]
        if piece_index == piece_count - 1:
            return self.signature_relation_to_id["suffix"]
        return self.signature_relation_to_id["continuation"]

    @staticmethod
    def _signature_family_key(signature: str) -> str:
        if signature in {
            "<SIGBOI>",
            "<SIGEOI>",
            "<SIGBOO>",
            "<SIGEOO>",
            "<SIGBOP>",
            "<SIGEOP>",
            "<BOI>",
            "<EOI>",
            "<BOO>",
            "<EOO>",
            "<BOP>",
            "<EOP>",
        }:
            return "boundary"
        if signature in {"<SIGBLO>", "<BLO>"}:
            return "line"
        parts = signature.split("|")
        if len(parts) >= 2:
            return "|".join(parts[:2])
        return signature

    def signature_id_for_code(self, code: str) -> int:
        normalized = code.strip()
        return self.signature_to_id.get(normalized, self.signature_special_ids["<OTHER>"])

    def signature_family_id_for_code(self, code: str) -> int:
        normalized = code.strip()
        signature_id = self.signature_id_for_code(normalized)
        return self.signature_family_id_by_signature_id.get(
            signature_id,
            self.signature_family_to_id["fallback"],
        )

    def signature_id_for_word(self, word: str) -> int:
        return self.signature_id_for_code(self._word_signature_code(word))

    def signature_id_for_line(self, line: str) -> int:
        return self.signature_id_for_code(self._line_signature_code(line))

    def signature_id_for_segment(self, segment: str, *, line_code: str, kind: str) -> int:
        return self.signature_id_for_code(self._segment_signature_code(segment, line_code=line_code, kind=kind))

    def _line_signature(self, text: str) -> str:
        """Backward-compatible alias for the richer line signature code."""

        return self._line_signature_code(text)

    def _signature_level_id(self, *, kind: str, is_word_piece: bool, is_line: bool, is_structure: bool) -> int:
        if is_line:
            return self.signature_level_to_id["line"]
        if is_structure:
            return self.signature_level_to_id["piece"]
        if kind == "phrase":
            return self.signature_level_to_id["phrase"]
        if kind == "word":
            return self.signature_level_to_id["word"]
        if is_word_piece:
            return self.signature_level_to_id["piece"]
        return self.signature_level_to_id["char"]

    def _signature_relation_id(
        self,
        *,
        kind: str,
        relation: str,
        is_line: bool,
        is_structure: bool,
    ) -> int:
        if is_line:
            return self.signature_relation_to_id["containment"]
        if is_structure:
            return self.signature_relation_to_id["adjacency"]
        return self.signature_relation_to_id.get(relation, self.signature_relation_to_id["continuation"])

    def _encode_construction_hierarchy(
        self,
        text: str,
        add_special_tokens: bool = True,
        span_role: str = "input",
    ) -> tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
        token_ids: List[int] = []
        signature_ids: List[int] = []
        signature_level_ids: List[int] = []
        signature_relation_ids: List[int] = []
        parent_signature_ids: List[int] = []
        signature_family_ids: List[int] = []

        def append_frame(unit_id: int, sig_id: int, level_id: int, relation_id: int, parent_id: int) -> None:
            token_ids.append(unit_id)
            signature_ids.append(sig_id)
            signature_level_ids.append(level_id)
            signature_relation_ids.append(relation_id)
            parent_signature_ids.append(parent_id)
            signature_family_ids.append(
                self.signature_family_id_by_signature_id.get(sig_id, self.signature_family_to_id["fallback"])
            )

        span_role = "output" if str(span_role).lower() == "output" else "input"
        if add_special_tokens:
            append_frame(
                self.bos_id,
                self.signature_bos_id,
                self.signature_level_to_id["special"],
                self.signature_relation_to_id["special"],
                self.signature_bos_id,
            )
            start_text = "<BOO>" if span_role == "output" else "<BOI>"
            start_sig = self.signature_boo_id if span_role == "output" else self.signature_boi_id
            append_frame(
                self.special_tokens[start_text],
                start_sig,
                self.signature_level_to_id["special"],
                self.signature_relation_to_id["special"],
                start_sig,
            )

        lines = text.splitlines(keepends=True) or [text]
        for raw_line in lines:
            line = raw_line.rstrip("\n")
            if not (line or raw_line.endswith("\n")):
                continue
            if not line.strip():
                blo_token = self.special_tokens.get("<BLO>")
                if blo_token is not None:
                    append_frame(
                        blo_token,
                        self.signature_blo_id,
                        self.signature_level_to_id["special"],
                        self.signature_relation_to_id["special"],
                        self.signature_blo_id,
                    )
                    signature_family_ids[-1] = self.signature_family_to_id["line"]
                continue
            line_code = self._line_signature_code(line)
            line_sig_id = self.signature_id_for_code(line_code)
            line_family_id = self.signature_family_id_for_code(line_code)
            append_frame(
                self.special_tokens["<LINE>"],
                line_sig_id,
                self.signature_level_to_id["line"],
                self.signature_relation_to_id["containment"],
                line_sig_id,
            )
            signature_family_ids[-1] = line_family_id

            for segment_match in SEGMENT_RE.finditer(line):
                segment = segment_match.group(0)
                if segment.isspace():
                    for ch in segment:
                        unit_id = self.special_tokens["<TAB>"] if ch == "\t" else self.special_tokens["<SPACE>"]
                        sig_id = self.signature_special_ids["<SPACE>"]
                        append_frame(
                            unit_id,
                            sig_id,
                            self.signature_level_to_id["piece"],
                            self.signature_relation_to_id["adjacency"],
                            line_sig_id,
                        )
                    continue

                if self._is_word_like(segment):
                    word_sig_id = self.signature_id_for_word(segment)
                    word_pieces = self._decompose_word(segment)
                    renderable = [(unit_id, kind, text_piece) for unit_id, kind, text_piece in word_pieces if kind != "case"]
                    render_count = max(1, len(renderable))
                    render_index = 0
                    for unit_id, kind, text_piece in word_pieces:
                        unit = self.construction_units[unit_id]
                        sig_code = unit.signature or self._construction_signature_code(text_piece or unit.text, kind)
                        sig_id = self.signature_id_for_code(sig_code)
                        if kind == "case":
                            relation_id = self.signature_relation_to_id["prefix"]
                            level_id = self.signature_level_to_id["piece"]
                        else:
                            relation_id = self._relation_for_piece(render_index, render_count)
                            level_id = self.signature_level_to_id["piece"] if kind in {"piece", "byte"} else self.signature_level_to_id["char"]
                            render_index += 1
                        append_frame(unit_id, sig_id, level_id, relation_id, word_sig_id)
                    continue

                for ch in segment:
                    unit_id = self._unit_id_for_text(ch)
                    if unit_id is None:
                        byte_values = ch.encode("utf-8", errors="ignore")
                        for b in byte_values:
                            unit_id = self._byte_fallback_unit(b)
                            sig_id = self.signature_id_for_code(self._byte_signature_code(b))
                            append_frame(
                                unit_id,
                                sig_id,
                                self.signature_level_to_id["char"],
                                self.signature_relation_to_id["adjacency"],
                                line_sig_id,
                            )
                        continue
                    unit = self.construction_units[unit_id]
                    sig_id = self.signature_id_for_code(unit.signature)
                    append_frame(
                        unit_id,
                        sig_id,
                        self.signature_level_to_id["piece"],
                        self.signature_relation_to_id["adjacency"],
                        line_sig_id,
                    )

            append_frame(
                self.special_tokens["<EOL>"],
                line_sig_id,
                self.signature_level_to_id["line"],
                self.signature_relation_to_id["containment"],
                line_sig_id,
            )
            signature_family_ids[-1] = line_family_id

        if add_special_tokens:
            end_text = "<EOO>" if span_role == "output" else "<EOI>"
            end_sig = self.signature_eoo_id if span_role == "output" else self.signature_eoi_id
            append_frame(
                self.special_tokens[end_text],
                end_sig,
                self.signature_level_to_id["special"],
                self.signature_relation_to_id["special"],
                end_sig,
            )
            append_frame(
                self.eos_id,
                self.signature_eos_id,
                self.signature_level_to_id["special"],
                self.signature_relation_to_id["special"],
                self.signature_eos_id,
            )
        return token_ids, signature_ids, signature_level_ids, signature_relation_ids, parent_signature_ids, signature_family_ids

    def encode_hierarchy_bundle(self, text: str, add_special_tokens: bool = True, span_role: str = "input") -> HierarchyEncoding:
        token_ids: List[int] = []
        signature_ids: List[int] = []
        signature_level_ids: List[int] = []
        signature_relation_ids: List[int] = []
        parent_signature_ids: List[int] = []
        signature_family_ids: List[int] = []
        span_role = "output" if str(span_role).lower() == "output" else "input"

        if add_special_tokens:
            token_ids.append(self.bos_id)
            signature_ids.append(self.signature_bos_id)
            signature_level_ids.append(self.signature_level_to_id["special"])
            signature_relation_ids.append(self.signature_relation_to_id["special"])
            parent_signature_ids.append(self.signature_bos_id)
            signature_family_ids.append(self.signature_family_to_id["special"])

            start_text = "<BOO>" if span_role == "output" else "<BOI>"
            start_sig = self.signature_boo_id if span_role == "output" else self.signature_boi_id
            if start_text in self.special_tokens:
                token_ids.append(self.special_tokens[start_text])
                signature_ids.append(start_sig)
                signature_level_ids.append(self.signature_level_to_id["special"])
                signature_relation_ids.append(self.signature_relation_to_id["special"])
                parent_signature_ids.append(start_sig)
                signature_family_ids.append(self.signature_family_to_id["boundary"])

        lines = text.splitlines(keepends=True) or [text]
        for raw_line in lines:
            line = raw_line.rstrip("\n")
            if not (line or raw_line.endswith("\n")):
                continue
            if not line.strip():
                blo_token = self.special_tokens.get("<BLO>")
                if blo_token is not None:
                    token_ids.append(blo_token)
                    signature_ids.append(self.signature_blo_id)
                    signature_level_ids.append(self.signature_level_to_id["special"])
                    signature_relation_ids.append(self.signature_relation_to_id["special"])
                    parent_signature_ids.append(self.signature_blo_id)
                    signature_family_ids.append(self.signature_family_to_id["line"])
                continue

            line_code = self._line_signature_code(line)
            line_sig_id = self.signature_id_for_code(line_code)
            token_ids.append(self.special_tokens["<LINE>"])
            signature_ids.append(line_sig_id)
            signature_level_ids.append(self.signature_level_to_id["line"])
            signature_relation_ids.append(self.signature_relation_to_id["containment"])
            parent_signature_ids.append(line_sig_id)
            signature_family_ids.append(self.signature_family_id_for_code(line_code))

            for segment_match in SEGMENT_RE.finditer(line):
                segment = segment_match.group(0)
                if segment.isspace():
                    pairs = self._encode_segment_with_hierarchy(
                        segment,
                        line_code=line_code,
                        kind="space",
                        parent_sig_id=line_sig_id,
                        segment_sig_id=self.signature_special_ids["<SPACE>"],
                    )
                elif self._is_word_like(segment):
                    pairs = self._encode_segment_with_hierarchy(
                        segment,
                        line_code=line_code,
                        kind="word",
                        parent_sig_id=self.signature_id_for_word(segment),
                        segment_sig_id=self.signature_id_for_word(segment),
                    )
                else:
                    pairs = self._encode_segment_with_hierarchy(
                        segment,
                        line_code=line_code,
                        kind="punct",
                        parent_sig_id=line_sig_id,
                        segment_sig_id=self.signature_special_ids["<PUNCT>"],
                    )
                for token_id, sig_piece, level_id, relation_id, parent_id in pairs:
                    token_ids.append(token_id)
                    signature_ids.append(sig_piece)
                    signature_level_ids.append(level_id)
                    signature_relation_ids.append(relation_id)
                    parent_signature_ids.append(parent_id)
                    signature_family_ids.append(self.signature_family_id_for_code(self._signature_id_to_code.get(sig_piece, line_code)))

            token_ids.append(self.special_tokens["<EOL>"])
            signature_ids.append(line_sig_id)
            signature_level_ids.append(self.signature_level_to_id["line"])
            signature_relation_ids.append(self.signature_relation_to_id["containment"])
            parent_signature_ids.append(line_sig_id)
            signature_family_ids.append(self.signature_family_id_for_code(line_code))

        if add_special_tokens:
            end_text = "<EOO>" if span_role == "output" else "<EOI>"
            end_sig = self.signature_eoo_id if span_role == "output" else self.signature_eoi_id
            if end_text in self.special_tokens:
                token_ids.append(self.special_tokens[end_text])
                signature_ids.append(end_sig)
                signature_level_ids.append(self.signature_level_to_id["special"])
                signature_relation_ids.append(self.signature_relation_to_id["special"])
                parent_signature_ids.append(end_sig)
                signature_family_ids.append(self.signature_family_to_id["boundary"])

            token_ids.append(self.eos_id)
            signature_ids.append(self.signature_eos_id)
            signature_level_ids.append(self.signature_level_to_id["special"])
            signature_relation_ids.append(self.signature_relation_to_id["special"])
            parent_signature_ids.append(self.signature_eos_id)
            signature_family_ids.append(self.signature_family_to_id["special"])

        hierarchy_vectors = _build_hierarchy_vector_tensor(
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(signature_ids, dtype=torch.long),
            torch.tensor(signature_level_ids, dtype=torch.long),
            torch.tensor(signature_relation_ids, dtype=torch.long),
            torch.tensor(parent_signature_ids, dtype=torch.long),
            torch.tensor(signature_family_ids, dtype=torch.long),
            token_vocab_size=max(self.vocab_size, 1),
            signature_vocab_size=max(self.signature_vocab_size, 1),
            level_vocab_size=max(self.signature_level_vocab_size, 1),
            relation_vocab_size=max(self.signature_relation_vocab_size, 1),
            family_vocab_size=max(self.signature_family_vocab_size, 1),
        ).tolist()
        bundle = HierarchyEncoding(
            token_ids=token_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            signature_family_ids=signature_family_ids,
            hierarchy_vectors=hierarchy_vectors,
        )
        bundle.validate(context="encode_hierarchy")
        return bundle

    def encode_hierarchy(
        self,
        text: str,
        add_special_tokens: bool = True,
        span_role: str = "input",
    ) -> tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
        return self.encode_hierarchy_bundle(text, add_special_tokens=add_special_tokens, span_role=span_role).as_tuple()

    def prepare_generation_hierarchy(self, text: str) -> HierarchyEncoding:
        bundle = self.encode_hierarchy_bundle(text, add_special_tokens=False)
        token_ids = [self.bos_id, self.special_tokens["<BOI>"], *bundle.token_ids]
        signature_ids = [self.signature_bos_id, self.signature_boi_id, *bundle.signature_ids]
        signature_level_ids = [
            self.signature_level_to_id["special"],
            self.signature_level_to_id["special"],
            *bundle.signature_level_ids,
        ]
        signature_relation_ids = [
            self.signature_relation_to_id["special"],
            self.signature_relation_to_id["special"],
            *bundle.signature_relation_ids,
        ]
        parent_signature_ids = [self.signature_bos_id, self.signature_boi_id, *bundle.parent_signature_ids]
        signature_family_ids = [
            self.signature_family_to_id["special"],
            self.signature_family_to_id["boundary"],
            *bundle.signature_family_ids,
        ]
        token_ids.append(self.special_tokens["<EOI>"])
        signature_ids.append(self.signature_eoi_id)
        signature_level_ids.append(self.signature_level_to_id["special"])
        signature_relation_ids.append(self.signature_relation_to_id["special"])
        parent_signature_ids.append(self.signature_eoi_id)
        signature_family_ids.append(self.signature_family_to_id["boundary"])
        token_ids.append(self.special_tokens["<BOO>"])
        signature_ids.append(self.signature_boo_id)
        signature_level_ids.append(self.signature_level_to_id["special"])
        signature_relation_ids.append(self.signature_relation_to_id["special"])
        parent_signature_ids.append(self.signature_boo_id)
        signature_family_ids.append(self.signature_family_to_id["boundary"])
        hierarchy_vectors = _build_hierarchy_vector_tensor(
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(signature_ids, dtype=torch.long),
            torch.tensor(signature_level_ids, dtype=torch.long),
            torch.tensor(signature_relation_ids, dtype=torch.long),
            torch.tensor(parent_signature_ids, dtype=torch.long),
            torch.tensor(signature_family_ids, dtype=torch.long),
            token_vocab_size=max(self.vocab_size, 1),
            signature_vocab_size=max(self.signature_vocab_size, 1),
            level_vocab_size=max(self.signature_level_vocab_size, 1),
            relation_vocab_size=max(self.signature_relation_vocab_size, 1),
            family_vocab_size=max(self.signature_family_vocab_size, 1),
        ).tolist()
        prompt_bundle = HierarchyEncoding(
            token_ids=token_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            signature_family_ids=signature_family_ids,
            hierarchy_vectors=hierarchy_vectors,
        )
        prompt_bundle.validate(context="generation prompt")
        return prompt_bundle

    def encode_with_signatures(self, text: str, add_special_tokens: bool = True) -> tuple[List[int], List[int]]:
        token_ids, signature_ids, _, _, _, _ = self.encode_hierarchy(text, add_special_tokens=add_special_tokens)
        return token_ids, signature_ids

    def encode_with_hierarchy_vectors(
        self, text: str, add_special_tokens: bool = True, span_role: str = "input"
    ) -> tuple[List[int], List[List[float]]]:
        bundle = self.encode_hierarchy_bundle(text, add_special_tokens=add_special_tokens, span_role=span_role)
        return bundle.token_ids, bundle.hierarchy_vectors

    def add_token(self, text: str, *, kind: str = "word", frequency: int = 1, signature: str = "") -> int:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            raise ValueError("Token text cannot be empty.")
        existing = self.token_to_id.get(normalized)
        if existing is not None:
            self.token_frequency_by_id[existing] = self.token_frequency_by_id.get(existing, 0) + max(1, int(frequency))
            for token in self.dynamic_tokens:
                if token.text == normalized:
                    token.frequency = self.token_frequency_by_id[existing]
                    break
            return existing

        token = DynamicToken(text=normalized, kind=kind, frequency=max(1, int(frequency)), signature=signature or kind)
        self.dynamic_tokens.append(token)
        self._rebuild_index()
        self.token_signature_id_by_id[self.token_to_id[normalized]] = (
            self.signature_id_for_word(normalized) if kind in {"word", "phrase"} else self.signature_id_for_code(signature or kind)
        )
        return self.token_to_id[normalized]

    @staticmethod
    def _token_quality_ok(text: str, *, kind: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        if not normalized:
            return False
        if kind in {"word", "phrase", "line"} and len(normalized) < 2:
            return False
        if kind in {"word", "phrase"} and not any(ch.isalnum() for ch in normalized):
            return False
        if kind == "word" and normalized in COMMON_DYNAMIC_TOKEN_TEXTS:
            return False
        if kind == "phrase":
            phrase_words = [word for word in re.findall(r"[A-Za-z0-9_']+", normalized)]
            if not phrase_words:
                return False
            if phrase_words[0] in COMMON_DYNAMIC_TOKEN_TEXTS or phrase_words[-1] in COMMON_DYNAMIC_TOKEN_TEXTS:
                return False
            if all(word in COMMON_DYNAMIC_TOKEN_TEXTS for word in phrase_words):
                return False

        compact = re.sub(r"\s+", "", normalized)
        if len(compact) >= 4:
            counts = Counter(compact)
            dominant = max(counts.values()) / max(len(compact), 1)
            if dominant >= 0.75 and any(ch.isalnum() for ch in compact):
                return False
            if re.fullmatch(r"(.)\1+", compact):
                return False
            if re.search(r"([A-Za-z0-9])\1\1\1+", compact):
                return False
            if re.search(r"([A-Za-z0-9])\1\1", compact):
                allow_short_repeats = {"www", "http", "https", "ttt"}
                if compact not in allow_short_repeats and not compact.startswith(("www.", "http://", "https://")):
                    if kind in {"word", "phrase"}:
                        return False
        if kind in {"word", "phrase"}:
            alpha_num = sum(ch.isalnum() for ch in normalized)
            if alpha_num / max(len(normalized), 1) < 0.5:
                return False
        if kind == "phrase" and (not normalized[0].isalnum() or not normalized[-1].isalnum()):
            return False
        return True

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        token_ids, _ = self.encode_with_signatures(text, add_special_tokens=add_special_tokens)
        return token_ids

    def _encode_segment_with_hierarchy(
        self,
        text: str,
        *,
        line_code: str,
        kind: str,
        parent_sig_id: int,
        segment_sig_id: int,
    ) -> List[tuple[int, int, int, int, int]]:
        pairs: List[tuple[int, int, int, int, int]] = []
        if kind == "word":
            word_pieces = self._decompose_word(text)
            renderable = [(unit_id, piece_kind, text_piece) for unit_id, piece_kind, text_piece in word_pieces if piece_kind != "case"]
            render_count = max(1, len(renderable))
            render_index = 0
            for unit_id, piece_kind, text_piece in word_pieces:
                unit = self.construction_units[unit_id]
                sig_code = unit.signature or self._construction_signature_code(text_piece or unit.text, piece_kind)
                sig_id = self.signature_id_for_code(sig_code)
                if piece_kind == "case":
                    relation_id = self.signature_relation_to_id["prefix"]
                    level_id = self.signature_level_to_id["piece"]
                else:
                    relation_id = self._relation_for_piece(render_index, render_count)
                    level_id = self.signature_level_to_id["piece"] if piece_kind in {"piece", "byte"} else self.signature_level_to_id["char"]
                    render_index += 1
                pairs.append(
                    (
                        unit_id,
                        sig_id,
                        level_id,
                        relation_id,
                        parent_sig_id,
                    )
                )
            return pairs

        i = 0
        while i < len(text):
            node = self._trie
            match_id = None
            match_len = 0
            j = i
            while j < len(text):
                ch = text[j].lower()
                if ch not in node:
                    break
                node = node[ch]
                j += 1
                if "$" in node:
                    match_id = node["$"]
                    match_len = j - i
            if match_id is not None and match_len > 0:
                if isinstance(match_id, dict):
                    match_id = match_id.get("$", self.signature_special_ids["<OTHER>"])
                if not isinstance(match_id, int):
                    match_id = self.signature_special_ids["<OTHER>"]
                level_id = self._signature_level_id(kind=kind, is_word_piece=False, is_line=False, is_structure=kind in {"space", "punct"})
                relation_id = self._signature_relation_id(
                    kind=kind,
                    relation="exact" if kind == "word" else "adjacency" if kind == "punct" else "adjacency",
                    is_line=False,
                    is_structure=kind in {"space", "punct"},
                )
                if kind == "word" and match_len < len(text):
                    relation_id = self.signature_relation_to_id["containment"]
                pairs.append((match_id, segment_sig_id, level_id, relation_id, parent_sig_id))
                i += match_len
                continue

            if kind == "space":
                unit_id = self.special_tokens["<TAB>"] if text[i] == "\t" else self.special_tokens["<SPACE>"]
                piece_sig_id = self.signature_special_ids["<SPACE>"]
                relation = "adjacency"
                pairs.append(
                    (
                        unit_id,
                        piece_sig_id,
                        self._signature_level_id(kind="space", is_word_piece=False, is_line=False, is_structure=True),
                        self._signature_relation_id(kind="space", relation=relation, is_line=False, is_structure=True),
                        parent_sig_id,
                    )
                )
                i += 1
                continue

            lookup_text = text[i].lower() if kind == "word" else text[i]
            unit_id = self._unit_id_for_text(lookup_text)
            if unit_id is not None:
                unit = self.construction_units[unit_id]
                piece_sig_id = self.signature_id_for_code(unit.signature)
                if kind == "word" and len(text) > 1:
                    if i == 0:
                        relation = "prefix"
                    elif i == len(text) - 1:
                        relation = "suffix"
                    else:
                        relation = "continuation"
                else:
                    relation = "exact"
                pairs.append(
                    (
                        unit_id,
                        piece_sig_id,
                        self._signature_level_id(kind=unit.kind, is_word_piece=kind == "word", is_line=False, is_structure=False),
                        self._signature_relation_id(kind=unit.kind, relation=relation, is_line=False, is_structure=False),
                        parent_sig_id,
                    )
                )
                i += 1
                continue

            byte_vals = text[i].encode("utf-8", errors="ignore")
            byte_count = max(1, len(byte_vals))
            for byte_index, b in enumerate(byte_vals):
                unit_id = self._byte_fallback_unit(b)
                piece_sig_id = self.signature_id_for_code(self._byte_signature_code(b))
                if kind == "word" and len(text) > 1:
                    if i == 0 and byte_index == 0:
                        relation = "prefix"
                    elif i == len(text) - 1 and byte_index == byte_count - 1:
                        relation = "suffix"
                    else:
                        relation = "continuation"
                elif byte_count == 1:
                    relation = "exact"
                elif byte_index == 0:
                    relation = "prefix"
                elif byte_index == byte_count - 1:
                    relation = "suffix"
                else:
                    relation = "continuation"
                pairs.append(
                    (
                        unit_id,
                        piece_sig_id,
                        self._signature_level_id(kind="byte", is_word_piece=True, is_line=False, is_structure=False),
                        self._signature_relation_id(kind="byte", relation=relation, is_line=False, is_structure=False),
                        parent_sig_id,
                    )
                )
            i += 1
        return pairs

    @staticmethod
    def _cleanup_decoded_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
        text = re.sub(r"\b([A-Za-z0-9]+)(?:[\s,;:.'\"!?-]+\1\b){2,}", r"\1 \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(www|http|https)(?:[\s,;:.'\"!?-]+\1\b){1,}", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _decode_construction(
        self,
        token_ids: Sequence[int],
        *,
        collapse_structure: bool = True,
        clean_text: bool = True,
    ) -> str:
        if not token_ids:
            return ""

        parts: List[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.skip_decode_ids:
                continue
            if token_id == self.eos_id:
                break

            unit = self.construction_units[token_id] if 0 <= token_id < len(self.construction_units) else None
            token_text = self.id_to_token.get(token_id, f"<UNK:{token_id}>")

            if unit is not None:
                if unit.text == "<BLO>":
                    parts.append("\n\n")
                    continue
                if unit.text in {"<BOO>", "<EOO>", "<BOP>", "<EOP>", "<CAP>", "<UPPER>", "<SIG:OTHER>"}:
                    continue
                render = getattr(unit, "render", "") or getattr(unit, "text", "")
                if render:
                    parts.append(render)
                continue

            if token_text and not token_text.startswith("<BYTE:"):
                parts.append(token_text)

        decoded = "".join(parts)
        if collapse_structure:
            decoded = re.sub(r"\n{3,}", "\n\n", decoded)
            decoded = re.sub(r" +", " ", decoded)
            decoded = decoded.strip()
        return self._cleanup_decoded_text(decoded) if clean_text else decoded

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        collapse_structure: bool = True,
        clean_text: bool = True,
    ) -> str:
        """Decode token IDs back to clean text using the render field."""

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        parts: List[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.skip_decode_ids:
                continue
            if token_id == self.eos_id:
                break
            if 0 <= token_id < len(self.construction_units):
                unit = self.construction_units[token_id]
                if unit.text == "<BLO>":
                    parts.append("\n\n")
                else:
                    parts.append(unit.render)
            else:
                parts.append(f"<UNK:{token_id}>")

        decoded = "".join(parts)
        if collapse_structure:
            decoded = re.sub(r"\n{3,}", "\n\n", decoded)
            decoded = re.sub(r" +", " ", decoded)
            decoded = decoded.strip()
        return self._cleanup_decoded_text(decoded) if clean_text else decoded

    def refresh_construction_index(self) -> None:
        """Rebuild decode-facing construction metadata after restore or mutation."""

        for unit in self.construction_units:
            if unit.kind == "byte":
                unit.render = unit.text
            elif not getattr(unit, "render", ""):
                unit.render = unit.text
            if not getattr(unit, "pronunciation", "") and unit.kind in {"word", "phrase", "piece"} and unit.text:
                unit.pronunciation = self._pronunciation_code(unit.text)
        self._rebuild_signature_index()

    def _collect_construction_signature_counts(self, texts: Iterable[str]) -> Counter[str]:
        signature_counts: Counter[str] = Counter()
        for unit_id, unit in enumerate(self.construction_units):
            if unit.signature:
                signature_counts[unit.signature] += max(1, self.token_frequency_by_id.get(unit_id, 1))

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            for raw_line in text.splitlines() or [text]:
                line = re.sub(r"\s+", " ", raw_line.strip())
                if not line:
                    continue
                line_code = self._line_signature_code(line)
                signature_counts[line_code] += 1
                for segment_match in SEGMENT_RE.finditer(line):
                    segment = segment_match.group(0)
                    if segment.isspace():
                        signature_counts["<SPACE>"] += len(segment)
                    elif self._is_word_like(segment):
                        signature_counts[self._word_signature_code(segment)] += 1
                        for _unit_id, kind, text_piece in self._decompose_word(segment):
                            if kind == "case":
                                signature_counts["case|mode=cap"] += 1
                            elif text_piece:
                                signature_counts[self._construction_signature_code(text_piece, kind)] += 1
                        else:
                            for ch in segment:
                                if ch in self.construction_text_to_id:
                                    unit = self.construction_units[self.construction_text_to_id[ch]]
                                    signature_counts[unit.signature] += 1
                                else:
                                    for b in ch.encode("utf-8", errors="ignore"):
                                        signature_counts[self._byte_signature_code(b)] += 1
        return signature_counts

    def _collect_construction_learning_counts(self, texts: Iterable[str]) -> tuple[Counter[str], Counter[str]]:
        candidate_counts: Counter[str] = Counter()
        signature_counts: Counter[str] = Counter()
        for unit_id, unit in enumerate(self.construction_units):
            if unit.signature:
                signature_counts[unit.signature] += max(1, self.token_frequency_by_id.get(unit_id, 1))

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            for raw_line in text.splitlines() or [text]:
                line = re.sub(r"\s+", " ", raw_line.strip())
                if not line:
                    continue
                line_code = self._line_signature_code(line)
                signature_counts[line_code] += 1
                for segment_match in SEGMENT_RE.finditer(line):
                    segment = segment_match.group(0)
                    if segment.isspace():
                        signature_counts["<SPACE>"] += len(segment)
                    elif self._is_word_like(segment):
                        normalized = segment.lower().strip()
                        if (
                            normalized
                            and normalized not in self._COMMON_WORDS_AS_WHOLE_UNITS
                            and normalized not in {"www", "http", "https"}
                            and self._token_quality_ok(normalized, kind="word")
                            and normalized not in self.construction_text_to_id
                        ):
                            candidate_counts[normalized] += 1
                        signature_counts[self._word_signature_code(segment)] += 1
                        for _unit_id, kind, text_piece in self._decompose_word(segment):
                            if kind == "case":
                                signature_counts["case|mode=cap"] += 1
                            elif text_piece:
                                signature_counts[self._construction_signature_code(text_piece, kind)] += 1
                    else:
                        for ch in segment:
                            if ch in self.construction_text_to_id:
                                unit = self.construction_units[self.construction_text_to_id[ch]]
                                signature_counts[unit.signature] += 1
                            else:
                                for b in ch.encode("utf-8", errors="ignore"):
                                    signature_counts[self._byte_signature_code(b)] += 1

        return candidate_counts, signature_counts

    def _learn_construction_signatures(
        self,
        texts: Iterable[str],
        *,
        max_signature_tokens: int = 0,
        signature_counts: Counter[str] | None = None,
    ) -> None:
        if signature_counts is None:
            signature_counts = self._collect_construction_signature_counts(texts)

        signature_limit = None if max_signature_tokens is None or int(max_signature_tokens) <= 0 else int(max_signature_tokens)
        self.signature_tokens = []
        for code, freq in signature_counts.most_common(signature_limit):
            if code in self.signature_to_id:
                continue
            self.signature_tokens.append(
                SignatureToken(code=code, kind="construction", frequency=freq, family=self._signature_family_key(code))
            )
        self._rebuild_signature_index()
        for unit_id, unit in enumerate(self.construction_units):
            self.token_signature_id_by_id[unit_id] = self.signature_id_for_code(unit.signature)

    def learn_from_texts(
        self,
        texts: Iterable[str],
        *,
        max_new_tokens: int = 0,
        min_frequency: int = 2,
        max_word_tokens: int = 0,
        max_line_tokens: int = 0,
        max_signature_tokens: int = 0,
        workers: int = 0,
    ) -> None:
        del max_line_tokens
        max_new_tokens = max(0, int(max_new_tokens))
        max_word_tokens = max(0, int(max_word_tokens))
        requested_workers = max(0, int(workers))
        if requested_workers > 1:
            tokenizer_state = self.to_state_dict()
            chunk_size = max(16, 64 // min(requested_workers, 8))
            unit_counts: Counter[str] = Counter()
            signature_counts: Counter[str] = Counter()
            print(f"[Prismal] tokenizer learning: {requested_workers} workers", flush=True)
            with concurrent.futures.ProcessPoolExecutor(max_workers=requested_workers) as executor:
                futures = [
                    executor.submit(_scan_tokenizer_learning_chunk, tokenizer_state, chunk)
                    for chunk in _chunk_text_iterable(texts, chunk_size)
                ]
                for future in concurrent.futures.as_completed(futures):
                    chunk_units, chunk_signatures = future.result()
                    unit_counts.update(chunk_units)
                    signature_counts.update(chunk_signatures)
        else:
            unit_counts, signature_counts = self._collect_construction_learning_counts(texts)

        self._learn_construction_units(
            (),
            max_new_tokens=max_new_tokens,
            min_frequency=min_frequency,
            max_word_tokens=max_word_tokens,
            candidate_counts=unit_counts,
        )
        self._learn_construction_signatures(
            (),
            max_signature_tokens=max_signature_tokens,
            signature_counts=signature_counts,
        )
        return

        word_counts: Counter[str] = Counter()
        bigram_counts: Counter[str] = Counter()
        line_counts: Counter[str] = Counter()
        signature_counts: Counter[str] = Counter()
        byte_signature_counts: Counter[str] = Counter()
        word_signature_counts: Counter[str] = Counter()
        line_signature_counts: Counter[str] = Counter()

        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            lines = text.splitlines() or [text]
            for raw_line in lines:
                line = re.sub(r"\s+", " ", raw_line.strip())
                if not line:
                    continue
                if len(line) <= MAX_DYNAMIC_LINE_TOKEN_CHARS and not CONTROL_LINE_RE.match(line):
                    line_counts[line] += 1
                line_code = self._line_signature_code(line)
                line_signature_counts[line_code] += 1
                for ch in raw_line:
                    for byte_value in ch.encode("utf-8", errors="ignore"):
                        byte_signature_counts[self._byte_signature_code(byte_value)] += 1
                words = [m.group(0) for m in WORD_RE.finditer(line)]
                for word in words:
                    if len(word) >= 2:
                        word_counts[word] += 1
                        word_signature_counts[self._word_signature_code(word)] += 1
                for left, right in zip(words, words[1:]):
                    bigram = f"{left} {right}"
                    if len(bigram) >= 5:
                        bigram_counts[bigram] += 1

        signature_counts.update(word_signature_counts)
        signature_counts.update(byte_signature_counts)
        signature_counts.update(line_signature_counts)

        candidates: List[DynamicToken] = []

        def add_candidates(items: Sequence[tuple[str, int]], kind: str, signature_prefix: str = "") -> None:
            for text, freq in items:
                if max_new_tokens_limit is not None and len(candidates) >= max_new_tokens_limit:
                    return
                if freq < min_frequency:
                    continue
                normalized = re.sub(r"\s+", " ", text.strip())
                if not normalized or normalized in self.token_to_id:
                    continue
                if not self._token_quality_ok(normalized, kind=kind):
                    continue
                sig = signature_prefix or kind
                candidates.append(DynamicToken(text=normalized, kind=kind, frequency=freq, signature=sig))

        signature_limit = None if max_signature_tokens is None or int(max_signature_tokens) <= 0 else int(max_signature_tokens)
        signature_tokens = [(key, freq) for key, freq in signature_counts.most_common(signature_limit) if key]
        line_tokens = line_counts.most_common(max_line_tokens)
        word_tokens = word_counts.most_common(max_word_tokens)
        bigram_tokens = bigram_counts.most_common(max_word_tokens // 2)

        # Hierarchy: structural signatures first, then line-level chunks, then words/phrases.
        add_candidates(line_tokens, "line", "line")
        add_candidates(bigram_tokens, "phrase", "phrase")
        add_candidates(word_tokens, "word", "word")

        family_counts: Counter[str] = Counter(self._signature_family_key(token.signature) for token in candidates)
        family_cap = None if max_new_tokens_limit is None else max(4, int(max_new_tokens_limit * 0.05))
        filtered_candidates: List[DynamicToken] = []
        family_used: Counter[str] = Counter()
        kind_cap = None if max_new_tokens_limit is None else {
            "word": max(8, int(max_new_tokens_limit * 0.5)),
            "phrase": max(8, int(max_new_tokens_limit * 0.3)),
            "line": max(4, int(max_new_tokens_limit * 0.2)),
        }
        kind_used: Counter[str] = Counter()

        for token in sorted(candidates, key=lambda tok: (tok.frequency, len(tok.text)), reverse=True):
            if max_new_tokens_limit is not None and len(self.dynamic_tokens) >= max_new_tokens_limit:
                break
            family_key = self._signature_family_key(token.signature)
            if family_cap is not None and family_counts[family_key] > family_cap and family_used[family_key] >= family_cap:
                continue
            if kind_cap is not None and token.kind in kind_cap and kind_used[token.kind] >= kind_cap[token.kind]:
                continue
            filtered_candidates.append(token)
            family_used[family_key] += 1
            kind_used[token.kind] += 1

        for token in filtered_candidates:
            if max_new_tokens_limit is not None and len(self.dynamic_tokens) >= max_new_tokens_limit:
                break
            self.dynamic_tokens.append(token)

        self._rebuild_index()
        self.signature_tokens = []
        for code, freq in signature_counts.most_common(signature_limit):
            if code in self.signature_to_id:
                continue
            self.signature_tokens.append(
                SignatureToken(code=code, kind="hierarchical", frequency=freq, family=code.split("|", 1)[0])
            )
        self._rebuild_signature_index()

    def to_state_dict(self) -> Dict[str, object]:
        return {
            "codec_version": self.codec_version,
            "base_vocab_size": self.base_vocab_size,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "use_pronunciation_signatures": self.use_pronunciation_signatures,
            "construction_units": [asdict(unit) for unit in self.construction_units],
            "dynamic_tokens": [],
            "signature_tokens": [asdict(tok) for tok in self.signature_tokens],
        }

    @classmethod
    def from_state_dict(cls, payload: Dict[str, object]) -> "PrismalTokenizer":
        tokenizer = cls(
            use_pronunciation_signatures=bool(payload.get("use_pronunciation_signatures", True)),
        )
        tokenizer.pad_id = int(payload.get("pad_id", tokenizer.pad_id))
        tokenizer.bos_id = int(payload.get("bos_id", tokenizer.bos_id))
        tokenizer.eos_id = int(payload.get("eos_id", tokenizer.eos_id))
        if int(payload.get("codec_version", tokenizer.codec_version)) >= 2:
            construction_units = payload.get("construction_units", [])
            if isinstance(construction_units, (list, tuple)):
                tokenizer._restore_construction_vocabulary(construction_units)
            tokenizer.signature_tokens = []
            for item in payload.get("signature_tokens", []):
                if isinstance(item, dict):
                    tokenizer.signature_tokens.append(
                        SignatureToken(
                            code=str(item.get("code", "")),
                            kind=str(item.get("kind", "construction")),
                            frequency=int(item.get("frequency", 0)),
                            family=str(item.get("family", "")),
                        )
                    )
            tokenizer._rebuild_signature_index()
            for unit_id, unit in enumerate(tokenizer.construction_units):
                tokenizer.token_signature_id_by_id[unit_id] = tokenizer.signature_id_for_code(unit.signature)
            return tokenizer

        tokenizer.base_vocab_size = int(payload.get("base_vocab_size", tokenizer.base_vocab_size))
        tokenizer.pad_id = int(payload.get("pad_id", tokenizer.pad_id))
        tokenizer.bos_id = int(payload.get("bos_id", tokenizer.bos_id))
        tokenizer.eos_id = int(payload.get("eos_id", tokenizer.eos_id))
        tokenizer.dynamic_tokens = []
        for item in payload.get("dynamic_tokens", []):
            if isinstance(item, dict):
                tokenizer.dynamic_tokens.append(
                    DynamicToken(
                        text=str(item.get("text", "")).lower(),
                        kind=str(item.get("kind", "word")),
                        frequency=int(item.get("frequency", 0)),
                        signature=str(item.get("signature", "")),
                    )
                )
        has_boundary_tokens = any(token.text in {"<boi>", "<eoi>", "<boo>", "<eoo>"} for token in tokenizer.dynamic_tokens)
        tokenizer._include_boundary_family = has_boundary_tokens
        if not has_boundary_tokens:
            for sig_id in (
                tokenizer.signature_boi_id,
                tokenizer.signature_eoi_id,
                tokenizer.signature_boo_id,
                tokenizer.signature_eoo_id,
            ):
                code = tokenizer._signature_id_to_code.pop(sig_id, None)
                if code is not None:
                    tokenizer.signature_to_id.pop(code, None)
                    tokenizer.signature_special_ids.pop(code, None)
                tokenizer.signature_kind_by_id.pop(sig_id, None)
                tokenizer.signature_frequency_by_id.pop(sig_id, None)
                tokenizer.signature_family_id_by_signature_id.pop(sig_id, None)
        tokenizer.signature_tokens = []
        for item in payload.get("signature_tokens", []):
            if isinstance(item, dict):
                tokenizer.signature_tokens.append(
                    SignatureToken(
                        code=str(item.get("code", "")),
                        kind=str(item.get("kind", "hierarchical")),
                        frequency=int(item.get("frequency", 0)),
                        family=str(item.get("family", "")),
                    )
                )
        tokenizer.special_tokens = {}
        tokenizer.id_to_token = {
            tokenizer.pad_id: "<PAD>",
            tokenizer.bos_id: "<BOS>",
            tokenizer.eos_id: "<EOS>",
        }
        tokenizer.token_kind_by_id = {
            tokenizer.pad_id: "special",
            tokenizer.bos_id: "special",
            tokenizer.eos_id: "special",
        }
        tokenizer.token_frequency_by_id = {
            tokenizer.pad_id: 0,
            tokenizer.bos_id: 0,
            tokenizer.eos_id: 0,
        }
        tokenizer.signature_by_id = {
            tokenizer.pad_id: "pad",
            tokenizer.bos_id: "bos",
            tokenizer.eos_id: "eos",
        }
        tokenizer._rebuild_index()
        for special_text in ("<BOI>", "<EOI>", "<BOO>", "<EOO>", "<BOP>", "<EOP>", "<BLO>", "<LINE>", "<EOL>", "<SIG:OTHER>"):
            token_id = tokenizer.token_to_id.get(special_text.lower())
            if token_id is not None:
                tokenizer.special_tokens[special_text] = token_id
        for special_text, kind, signature in (
            ("<BOI>", "structure", "boundary"),
            ("<EOI>", "structure", "boundary"),
            ("<LINE>", "structure", "line"),
            ("<EOL>", "structure", "newline"),
            ("<BOP>", "structure", "boundary"),
            ("<EOP>", "structure", "boundary"),
            ("<BLO>", "structure", "line"),
            ("<SIG:OTHER>", "signature", "other"),
        ):
            if special_text not in tokenizer.special_tokens:
                tokenizer.special_tokens[special_text] = tokenizer.add_token(
                    special_text,
                    kind=kind,
                    frequency=1,
                    signature=signature,
                )
        tokenizer.skip_decode_ids = {tokenizer.pad_id, tokenizer.bos_id}
        for special_text in ("<BOI>", "<EOI>", "<BOO>", "<EOO>", "<BOP>", "<EOP>", "<SIG:OTHER>"):
            token_id = tokenizer.special_tokens.get(special_text)
            if token_id is not None:
                tokenizer.skip_decode_ids.add(token_id)
        blo_id = tokenizer.special_tokens.get("<BLO>")
        if blo_id is not None:
            tokenizer.skip_decode_ids.discard(blo_id)
        tokenizer._rebuild_signature_index()
        return tokenizer

    def signature_lookup_by_token_id(self) -> Dict[int, int]:
        lookup = dict(self.token_signature_id_by_id)
        lookup[self.pad_id] = self.signature_pad_id
        lookup[self.bos_id] = self.signature_bos_id
        lookup[self.eos_id] = self.signature_eos_id
        for special_text, signature_id in (
            ("<BOI>", self.signature_boi_id),
            ("<EOI>", self.signature_eoi_id),
            ("<BOO>", self.signature_boo_id),
            ("<EOO>", self.signature_eoo_id),
            ("<BOP>", self.signature_bop_id),
            ("<EOP>", self.signature_eop_id),
            ("<BLO>", self.signature_blo_id),
        ):
            token_id = self.special_tokens.get(special_text)
            if token_id is not None:
                lookup[token_id] = signature_id
        return lookup

    def signature_level_lookup_by_token_id(self) -> Dict[int, int]:
        lookup: Dict[int, int] = {}
        for token_id, kind in self.token_kind_by_id.items():
            token_text = self.id_to_token.get(token_id, "")
            if token_text in {"<BOI>", "<EOI>", "<BOO>", "<EOO>", "<BOP>", "<EOP>"}:
                lookup[token_id] = self.signature_level_to_id["special"]
                continue
            if kind == "special":
                lookup[token_id] = self.signature_level_to_id["special"]
            elif kind in {"structure", "line"}:
                lookup[token_id] = self.signature_level_to_id["line"]
            elif kind == "phrase":
                lookup[token_id] = self.signature_level_to_id["phrase"]
            elif kind == "word":
                lookup[token_id] = self.signature_level_to_id["word"]
            elif kind in {"piece", "case", "space", "byte"}:
                lookup[token_id] = self.signature_level_to_id["piece"]
            else:
                lookup[token_id] = self.signature_level_to_id["char"]
        return lookup

    def signature_relation_lookup_by_token_id(self) -> Dict[int, int]:
        lookup: Dict[int, int] = {}
        for token_id, kind in self.token_kind_by_id.items():
            token_text = self.id_to_token.get(token_id, "")
            if token_text in {"<BOI>", "<EOI>", "<BOO>", "<EOO>", "<BOP>", "<EOP>"}:
                lookup[token_id] = self.signature_relation_to_id["special"]
                continue
            if kind == "special":
                lookup[token_id] = self.signature_relation_to_id["special"]
            elif kind in {"structure", "line", "phrase"}:
                lookup[token_id] = self.signature_relation_to_id["containment"]
            elif kind in {"space", "punct"}:
                lookup[token_id] = self.signature_relation_to_id["adjacency"]
            elif kind == "word":
                lookup[token_id] = self.signature_relation_to_id["exact"]
            else:
                lookup[token_id] = self.signature_relation_to_id["continuation"]
        return lookup

    def signature_family_lookup_by_token_id(self) -> Dict[int, int]:
        lookup = {
            token_id: self.signature_family_id_by_signature_id.get(
                signature_id,
                self.signature_family_to_id["fallback"],
            )
            for token_id, signature_id in self.token_signature_id_by_id.items()
        }
        lookup[self.pad_id] = self.signature_family_to_id["pad"]
        lookup[self.bos_id] = self.signature_family_to_id["special"]
        lookup[self.eos_id] = self.signature_family_to_id["special"]
        for special_text in ("<BOI>", "<EOI>", "<BOO>", "<EOO>", "<BOP>", "<EOP>"):
            token_id = self.special_tokens.get(special_text)
            if token_id is not None:
                lookup[token_id] = self.signature_family_to_id["boundary"]
        blo_id = self.special_tokens.get("<BLO>")
        if blo_id is not None:
            lookup[blo_id] = self.signature_family_to_id["line"]
        return lookup

    def hierarchy_vector_lookup_by_token_id(self) -> Dict[int, List[float]]:
        lookup: Dict[int, List[float]] = {}
        signature_lookup = self.signature_lookup_by_token_id()
        level_lookup = self.signature_level_lookup_by_token_id()
        relation_lookup = self.signature_relation_lookup_by_token_id()
        family_lookup = self.signature_family_lookup_by_token_id()
        for token_id, signature_id in signature_lookup.items():
            level_id = level_lookup.get(token_id, self.signature_level_to_id["char"])
            relation_id = relation_lookup.get(token_id, self.signature_relation_to_id["continuation"])
            family_id = family_lookup.get(token_id, self.signature_family_to_id["fallback"])
            lookup[token_id] = _build_hierarchy_vector_tensor(
                torch.tensor([token_id], dtype=torch.long),
                torch.tensor([signature_id], dtype=torch.long),
                torch.tensor([level_id], dtype=torch.long),
                torch.tensor([relation_id], dtype=torch.long),
                torch.tensor([signature_id], dtype=torch.long),
                torch.tensor([family_id], dtype=torch.long),
                token_vocab_size=max(self.vocab_size, 1),
                signature_vocab_size=max(self.signature_vocab_size, 1),
                level_vocab_size=max(self.signature_level_vocab_size, 1),
                relation_vocab_size=max(self.signature_relation_vocab_size, 1),
                family_vocab_size=max(self.signature_family_vocab_size, 1),
            )[0].tolist()
        return lookup

    def generation_suppressed_token_ids(self) -> List[int]:
        suppressed = [self.pad_id, self.bos_id]
        allowed_punct = {".", ",", "?", "!", ":", ";", "'", '"', "-", "_", "(", ")"}
        for unit_id, unit in enumerate(self.construction_units):
            if unit.text == "<BLO>":
                continue
            if unit.kind in {"special", "structure", "signature", "byte"} and unit_id != self.eos_id:
                suppressed.append(unit_id)
            elif unit.kind == "punct" and unit.text not in allowed_punct:
                suppressed.append(unit_id)
        return sorted(set(token_id for token_id in suppressed if token_id != self.eos_id))


def load_text_corpus(source: str | Path) -> List[str]:
    return list(iter_text_corpus(source))


@dataclass
class WindowSample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    signature_ids: torch.Tensor
    signature_level_ids: torch.Tensor
    signature_relation_ids: torch.Tensor
    parent_signature_ids: torch.Tensor
    signature_family_ids: torch.Tensor
    hierarchy_vectors: torch.Tensor
    loss_mask: torch.Tensor

    def __post_init__(self) -> None:
        seq_lengths = {
            int(self.input_ids.shape[0]),
            int(self.labels.shape[0]),
            int(self.signature_ids.shape[0]),
            int(self.signature_level_ids.shape[0]),
            int(self.signature_relation_ids.shape[0]),
            int(self.parent_signature_ids.shape[0]),
            int(self.signature_family_ids.shape[0]),
            int(self.hierarchy_vectors.shape[0]),
            int(self.loss_mask.shape[0]),
        }
        lengths = seq_lengths
        if len(lengths) != 1:
            raise ValueError(
                "WindowSample tensors must stay aligned; got lengths "
                f"input={self.input_ids.shape[0]}, labels={self.labels.shape[0]}, "
                f"signature={self.signature_ids.shape[0]}, level={self.signature_level_ids.shape[0]}, "
                f"relation={self.signature_relation_ids.shape[0]}, parent={self.parent_signature_ids.shape[0]}, "
                f"family={self.signature_family_ids.shape[0]}, hierarchy={self.hierarchy_vectors.shape[0]}, "
                f"mask={self.loss_mask.shape[0]}"
            )
        if self.hierarchy_vectors.dim() != 2:
            raise ValueError(
                f"WindowSample hierarchy_vectors must be 2D, got shape {tuple(self.hierarchy_vectors.shape)}"
            )
        if int(self.hierarchy_vectors.shape[1]) <= 0:
            raise ValueError(
                f"WindowSample hierarchy_vectors must have a positive width, got shape {tuple(self.hierarchy_vectors.shape)}"
            )


def _find_answer_start(text: str) -> int | None:
    matches = list(ANSWER_MARKER_RE.finditer(text))
    if not matches:
        return None
    marker = matches[0]
    start = marker.end()
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def _build_loss_mask(tokenizer: PrismalTokenizer, text: str, encoded_tokens: Sequence[int]) -> List[float]:
    answer_start = _find_answer_start(text)
    if answer_start is None:
        return [1.0] * len(encoded_tokens)

    prefix_text = text[:answer_start]
    prefix_bundle = tokenizer.encode_hierarchy_bundle(prefix_text, add_special_tokens=False)
    prefix_bundle = prefix_bundle.trim_trailing_tokens(
        {
            token_id
            for token_id in {
                tokenizer.special_tokens.get("<BOI>"),
                tokenizer.special_tokens.get("<EOI>"),
                tokenizer.special_tokens.get("<BOO>"),
                tokenizer.special_tokens.get("<EOO>"),
                tokenizer.special_tokens.get("<BOP>"),
                tokenizer.special_tokens.get("<EOP>"),
                tokenizer.special_tokens.get("<BLO>"),
                tokenizer.special_tokens.get("<EOL>"),
                tokenizer.special_tokens.get("<LINE>"),
            }
            if token_id is not None
        }
    )
    prefix_len = min(len(prefix_bundle.token_ids), len(encoded_tokens))
    if prefix_len <= 0 or prefix_len >= len(encoded_tokens):
        return [1.0] * len(encoded_tokens)
    return [0.0] * prefix_len + [1.0] * (len(encoded_tokens) - prefix_len)


def _build_window_samples_from_text(
    tokenizer: PrismalTokenizer,
    merged: str,
    *,
    seq_len: int,
    max_samples: int = 0,
) -> List[WindowSample]:
    samples: List[WindowSample] = []
    seq_len = max(0, int(seq_len))
    encoded, encoded_signatures, encoded_levels, encoded_relations, encoded_parents, encoded_families = tokenizer.encode_hierarchy(
        merged,
        add_special_tokens=False,
    )
    if not any(marker in merged for marker in ("<BOI>", "<EOI>", "<BOO>", "<EOO>")):
        boi_token = tokenizer.special_tokens.get("<BOI>")
        eoi_token = tokenizer.special_tokens.get("<EOI>")
        if boi_token is not None and eoi_token is not None:
            encoded = [boi_token] + encoded + [eoi_token]
            encoded_signatures = [tokenizer.signature_boi_id] + encoded_signatures + [tokenizer.signature_eoi_id]
            encoded_levels = [
                tokenizer.signature_level_to_id["special"],
                *encoded_levels,
                tokenizer.signature_level_to_id["special"],
            ]
            encoded_relations = [
                tokenizer.signature_relation_to_id["special"],
                *encoded_relations,
                tokenizer.signature_relation_to_id["special"],
            ]
            encoded_parents = [tokenizer.signature_boi_id] + encoded_parents + [tokenizer.signature_eoi_id]
            encoded_families = [
                tokenizer.signature_family_to_id["boundary"],
                *encoded_families,
                tokenizer.signature_family_to_id["boundary"],
            ]
    if len(encoded) < 4:
        (
            encoded,
            encoded_signatures,
            encoded_levels,
            encoded_relations,
            encoded_parents,
            encoded_families,
        ) = tokenizer.encode_hierarchy(merged * 4, add_special_tokens=False)
    if len(encoded) < 2:
        return samples

    hierarchy_vectors = _build_hierarchy_vector_tensor(
        torch.tensor(encoded, dtype=torch.long),
        torch.tensor(encoded_signatures, dtype=torch.long),
        torch.tensor(encoded_levels, dtype=torch.long),
        torch.tensor(encoded_relations, dtype=torch.long),
        torch.tensor(encoded_parents, dtype=torch.long),
        torch.tensor(encoded_families, dtype=torch.long),
        token_vocab_size=max(tokenizer.vocab_size, 1),
        signature_vocab_size=max(tokenizer.signature_vocab_size, 1),
        level_vocab_size=max(tokenizer.signature_level_vocab_size, 1),
        relation_vocab_size=max(tokenizer.signature_relation_vocab_size, 1),
        family_vocab_size=max(tokenizer.signature_family_vocab_size, 1),
    )

    token_loss_mask = _build_loss_mask(tokenizer, merged, encoded)
    unsupervised_token_ids = {
        tokenizer.special_tokens.get("<BOI>"),
        tokenizer.special_tokens.get("<EOI>"),
        tokenizer.special_tokens.get("<BOO>"),
        tokenizer.special_tokens.get("<EOO>"),
        tokenizer.special_tokens.get("<BOP>"),
        tokenizer.special_tokens.get("<EOP>"),
        tokenizer.special_tokens.get("<BLO>"),
        tokenizer.special_tokens.get("<LINE>"),
        tokenizer.special_tokens.get("<EOL>"),
        tokenizer.special_tokens.get("<SIG:OTHER>"),
    }
    token_loss_mask = [
        0.0 if token_id in unsupervised_token_ids else float(mask)
        for token_id, mask in zip(encoded, token_loss_mask)
    ]

    def append_window(start: int, chunk_len: int) -> bool:
        start = max(0, int(start))
        chunk_len = max(2, int(chunk_len))
        chunk = encoded[start : start + chunk_len]
        sig_chunk = encoded_signatures[start : start + chunk_len]
        lvl_chunk = encoded_levels[start : start + chunk_len]
        rel_chunk = encoded_relations[start : start + chunk_len]
        parent_chunk = encoded_parents[start : start + chunk_len]
        family_chunk = encoded_families[start : start + chunk_len]
        mask_chunk = token_loss_mask[start : start + chunk_len]
        if len(chunk) < 2:
            return False
        if not any(float(value) > 0.0 for value in mask_chunk):
            return False
        ids = [tokenizer.bos_id] + chunk + [tokenizer.eos_id]
        sig_ids = [tokenizer.signature_bos_id] + sig_chunk + [tokenizer.signature_eos_id]
        lvl_ids = [tokenizer.signature_level_to_id["special"]] + lvl_chunk + [tokenizer.signature_level_to_id["special"]]
        rel_ids = [tokenizer.signature_relation_to_id["special"]] + rel_chunk + [tokenizer.signature_relation_to_id["special"]]
        parent_ids = [tokenizer.signature_bos_id] + parent_chunk + [tokenizer.signature_eos_id]
        family_ids = [tokenizer.signature_family_to_id["special"]] + family_chunk + [tokenizer.signature_family_to_id["special"]]
        eos_loss = 1.0 if start + len(chunk) >= len(encoded) else 0.0
        loss_mask = [0.0] + mask_chunk + [eos_loss]
        hierarchy_vectors = _build_hierarchy_vector_tensor(
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(sig_ids, dtype=torch.long),
            torch.tensor(lvl_ids, dtype=torch.long),
            torch.tensor(rel_ids, dtype=torch.long),
            torch.tensor(parent_ids, dtype=torch.long),
            torch.tensor(family_ids, dtype=torch.long),
            token_vocab_size=max(tokenizer.vocab_size, 1),
            signature_vocab_size=max(tokenizer.signature_vocab_size, 1),
            level_vocab_size=max(tokenizer.signature_level_vocab_size, 1),
            relation_vocab_size=max(tokenizer.signature_relation_vocab_size, 1),
            family_vocab_size=max(tokenizer.signature_family_vocab_size, 1),
        )
        samples.append(
            WindowSample(
                input_ids=torch.tensor(ids[:-1], dtype=torch.long),
                labels=torch.tensor(ids[1:], dtype=torch.long),
                signature_ids=torch.tensor(sig_ids[:-1], dtype=torch.long),
                signature_level_ids=torch.tensor(lvl_ids[:-1], dtype=torch.long),
                signature_relation_ids=torch.tensor(rel_ids[:-1], dtype=torch.long),
                parent_signature_ids=torch.tensor(parent_ids[:-1], dtype=torch.long),
                signature_family_ids=torch.tensor(family_ids[:-1], dtype=torch.long),
                hierarchy_vectors=hierarchy_vectors[:-1].to(dtype=torch.float32),
                loss_mask=torch.tensor(loss_mask[1:], dtype=torch.float32),
            )
        )
        return True

    if seq_len <= 0:
        chunk_len = max(64, min(len(encoded), 256))
        stride = max(1, chunk_len // 2)
        supervised_indices = [idx for idx, mask in enumerate(token_loss_mask) if float(mask) > 0.0]
        anchored_starts: set[int] = set()
        if supervised_indices:
            answer_token_start = supervised_indices[0]
            if answer_token_start >= chunk_len - 8:
                answer_context_tail = 128
                answer_chunk_len = min(len(encoded), max(chunk_len, min(1024, answer_token_start + answer_context_tail)))
                if answer_chunk_len >= len(encoded) or answer_token_start + answer_context_tail <= answer_chunk_len:
                    answer_start = 0
                else:
                    answer_start = max(0, answer_token_start - max(0, answer_chunk_len - answer_context_tail))
                if append_window(answer_start, answer_chunk_len):
                    anchored_starts.add(answer_start)
                    if max_samples and len(samples) >= max_samples:
                        return samples
        for start in range(0, max(1, len(encoded) - 1), stride):
            if start in anchored_starts:
                continue
            append_window(start, chunk_len)
            if max_samples and len(samples) >= max_samples:
                break
    else:
        chunk_len = max(4, seq_len - 1)
        stride = max(1, chunk_len // 2)
        for start in range(0, max(1, len(encoded) - 1), stride):
            append_window(start, chunk_len)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


class TextWindowDataset(Dataset):
    """Hierarchical token next-token dataset."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: PrismalTokenizer,
        seq_len: int,
        stride: int | None = None,
        max_samples: int = 0,
        sample_seed: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = max(0, int(seq_len))
        if stride is not None:
            self.stride = max(1, int(stride))
        elif self.seq_len > 0:
            self.stride = max(1, self.seq_len // 2)
        else:
            self.stride = 0
        self.sample_seed = sample_seed
        self.samples: List[WindowSample] = []
        saw_any_text = False

        for text in texts:
            if not isinstance(text, str):
                continue
            merged = text.strip()
            if not merged:
                continue
            saw_any_text = True
            windows = _build_window_samples_from_text(
                tokenizer,
                merged,
                seq_len=self.seq_len,
                max_samples=max(0, int(max_samples) - len(self.samples)) if max_samples else 0,
            )
            self.samples.extend(windows)
            if max_samples and len(self.samples) >= max_samples:
                break

        if not self.samples:
            reason = "source contained no text" if not saw_any_text else "source produced no supervised token windows"
            raise ValueError(f"TextWindowDataset is empty: {reason}.")

        if max_samples and self.sample_seed is not None and len(self.samples) > max_samples:
            rng = random.Random(self.sample_seed)
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> WindowSample:
        return self.samples[idx]


class StreamingTextCorpusDataset(IterableDataset):
    """Stream text records into token windows without materializing the corpus."""

    def __init__(
        self,
        source: str | Path,
        tokenizer: PrismalTokenizer,
        *,
        seq_len: int,
        max_samples: int = 0,
        split: str = "all",
        val_fraction: float = 0.1,
        seed: int = 42,
        sample_seed: int | None = None,
        shuffle_buffer_size: int | None = None,
    ) -> None:
        self.source = Path(source)
        self.tokenizer = tokenizer
        self.seq_len = max(0, int(seq_len))
        self.max_samples = max(0, int(max_samples))
        self.split = split
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.sample_seed = int(sample_seed) if sample_seed is not None else random.SystemRandom().randrange(1 << 63)
        self.shuffle_buffer_size = max(
            1,
            int(shuffle_buffer_size) if shuffle_buffer_size is not None else min(max(self.max_samples, 1), 64) if self.max_samples else 1024,
        )

    def _include_record(self, record_index: int) -> bool:
        if self.split == "all":
            return True
        if self.split not in {"train", "val"}:
            raise ValueError("split must be 'all', 'train', or 'val'.")
        fraction = min(max(self.val_fraction, 0.0), 1.0)
        if fraction <= 0.0:
            return self.split != "val"
        if fraction >= 1.0:
            return self.split == "val"
        key = f"{self.seed}:{record_index}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") / float(1 << 64)
        if self.split == "val":
            return bucket < fraction
        return bucket >= fraction

    def _windows_for_line(self, line: str, remaining: int = 0) -> List[WindowSample]:
        merged = line.strip()
        if not merged:
            return []
        return _build_window_samples_from_text(
            self.tokenizer,
            merged,
            seq_len=self.seq_len,
            max_samples=remaining,
        )

    def __iter__(self) -> Iterator[WindowSample]:
        if self.max_samples > 0:
            rng = random.Random(self.sample_seed)
            buffer: List[WindowSample] = []
            yielded = 0

            def _emit_buffered() -> Iterator[WindowSample]:
                while buffer and yielded < self.max_samples:
                    index = rng.randrange(len(buffer))
                    yield buffer.pop(index)

            def _ingest_records(include_filter: bool) -> None:
                nonlocal yielded
                for record_index, text in enumerate(iter_text_corpus(self.source)):
                    if include_filter and not self._include_record(record_index):
                        continue
                    for sample in self._windows_for_line(text):
                        buffer.append(sample)
                        if len(buffer) < self.shuffle_buffer_size:
                            continue
                        index = rng.randrange(len(buffer))
                        yield_sample = buffer.pop(index)
                        yield yield_sample
                        yielded += 1
                        if yielded >= self.max_samples:
                            return

            for sample in _ingest_records(include_filter=True):
                yield sample
            if yielded == 0 and self.split == "train":
                buffer.clear()
                yielded = 0
                for sample in _ingest_records(include_filter=False):
                    yield sample
            if yielded < self.max_samples:
                for sample in _emit_buffered():
                    yield sample
                    yielded += 1
                    if yielded >= self.max_samples:
                        return
            return

        yielded = 0
        for record_index, text in enumerate(iter_text_corpus(self.source)):
            if not self._include_record(record_index):
                continue
            windows = self._windows_for_line(text)
            for sample in windows:
                yield sample
                yielded += 1
        if yielded == 0 and self.split == "train":
            for text in iter_text_corpus(self.source):
                windows = self._windows_for_line(text)
                for sample in windows:
                    yield sample
                    yielded += 1
        return

    @staticmethod
    def _find_answer_start(text: str) -> int | None:
        return _find_answer_start(text)

    @staticmethod
    def _build_loss_mask(tokenizer: PrismalTokenizer, text: str, encoded_tokens: Sequence[int]) -> List[float]:
        return _build_loss_mask(tokenizer, text, encoded_tokens)


class StreamingJSONLDataset(StreamingTextCorpusDataset):
    """Backward-compatible alias for the generic streaming text corpus dataset."""


def _window_sample_field_names() -> List[str]:
    return [
        "input_ids",
        "labels",
        "signature_ids",
        "signature_level_ids",
        "signature_relation_ids",
        "parent_signature_ids",
        "signature_family_ids",
        "hierarchy_vectors",
        "loss_mask",
    ]


def _window_sample_numpy(sample: WindowSample, field_name: str) -> np.ndarray:
    tensor = getattr(sample, field_name)
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected tensor field {field_name}, got {type(tensor).__name__}")
    return tensor.detach().cpu().numpy()


def save_pretokenized_windows(
    samples: Sequence[WindowSample],
    output_dir: str | Path,
    *,
    seq_len: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    return stream_pretokenized_windows(samples, output_dir, seq_len=seq_len, metadata=metadata)


def stream_pretokenized_windows(
    samples: Iterable[WindowSample],
    output_dir: str | Path,
    *,
    seq_len: int,
    metadata: Optional[Dict[str, Any]] = None,
    num_samples: Optional[int] = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    samples_iter, samples_for_write = itertools.tee(samples)

    if num_samples is None:
        sample_count = 0
        total_tokens = 0
        for sample in samples_iter:
            sample_count += 1
            total_tokens += int(sample.input_ids.numel())
        if sample_count == 0:
            raise ValueError("Cannot save an empty pretokenized dataset.")
        num_samples = sample_count
    else:
        num_samples = max(0, int(num_samples))
        if num_samples == 0:
            raise ValueError("Cannot save an empty pretokenized dataset.")
        total_tokens = 0
        for sample in samples_iter:
            total_tokens += int(sample.input_ids.numel())

    offsets = np.zeros(num_samples, dtype=np.int64)
    lengths = np.zeros(num_samples, dtype=np.int64)
    hierarchy_vector_dim = DEFAULT_HIERARCHY_VECTOR_DIM
    if num_samples > 0:
        preview_iter, samples_for_write = itertools.tee(samples_for_write)
        first_sample = next(preview_iter, None)
        if first_sample is not None:
            hierarchy_vector_dim = int(first_sample.hierarchy_vectors.shape[-1])
    field_dtypes = {
        "input_ids": np.int64,
        "labels": np.int64,
        "signature_ids": np.int64,
        "signature_level_ids": np.int64,
        "signature_relation_ids": np.int64,
        "parent_signature_ids": np.int64,
        "signature_family_ids": np.int64,
        "hierarchy_vectors": np.float32,
        "loss_mask": np.float32,
    }
    field_memmaps: Dict[str, np.ndarray] = {
        field_name: np.lib.format.open_memmap(
            output_path / f"{field_name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(total_tokens, hierarchy_vector_dim) if field_name == "hierarchy_vectors" else (total_tokens,),
        )
        for field_name, dtype in field_dtypes.items()
    }

    cursor = 0
    written = 0
    for sample in samples_for_write:
        if written >= num_samples:
            break
        sample_len = int(sample.input_ids.numel())
        offsets[written] = cursor
        lengths[written] = sample_len
        for field_name in _window_sample_field_names():
            array = _window_sample_numpy(sample, field_name)
            if field_name == "hierarchy_vectors":
                if array.shape != (sample_len, hierarchy_vector_dim):
                    raise ValueError(
                        "hierarchy_vectors must match sample length and shared width; "
                        f"got {tuple(array.shape)} expected {(sample_len, hierarchy_vector_dim)}"
                    )
                field_memmaps[field_name][cursor : cursor + sample_len, :] = array.astype(np.float32, copy=False)
                continue
            flat = array.reshape(-1)
            if field_name != "loss_mask" and flat.dtype != np.int64:
                flat = flat.astype(np.int64, copy=False)
            if field_name == "loss_mask" and flat.dtype != np.float32:
                flat = flat.astype(np.float32, copy=False)
            field_memmaps[field_name][cursor : cursor + sample_len] = flat
        cursor += sample_len
        written += 1

    if written == 0:
        raise ValueError("Cannot save an empty pretokenized dataset.")

    for mmap in field_memmaps.values():
        mmap.flush()

    np.save(output_path / "sample_offsets.npy", offsets)
    np.save(output_path / "sample_lengths.npy", lengths)

    payload: Dict[str, Any] = {
        "format_version": 1,
        "seq_len": int(seq_len),
        "num_samples": int(written),
        "field_names": _window_sample_field_names(),
        "hierarchy_vector_dim": int(hierarchy_vector_dim),
    }
    if metadata:
        payload.update(metadata)
    (output_path / "meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


class MemmapTokenDataset(Dataset):
    """Random-access dataset backed by pretokenized memory-mapped arrays."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        split: str = "all",
        val_fraction: float = 0.1,
        seed: int = 42,
        max_samples: int = 0,
        sample_seed: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.meta_path = self.root_dir / "meta.json"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Pretokenized metadata not found: {self.meta_path}")
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.seq_len = int(self.meta.get("seq_len", 0))
        self.total_samples = int(self.meta.get("num_samples", 0))
        self.split = split
        self.val_fraction = float(val_fraction)
        self.seed = int(seed)
        self.max_samples = max(0, int(max_samples))
        self.sample_seed = int(sample_seed) if sample_seed is not None else random.SystemRandom().randrange(1 << 63)
        self.offsets = np.load(self.root_dir / "sample_offsets.npy", mmap_mode="r")
        self.lengths = np.load(self.root_dir / "sample_lengths.npy", mmap_mode="r")
        self.fields = {
            "input_ids": np.load(self.root_dir / "input_ids.npy", mmap_mode="r"),
            "labels": np.load(self.root_dir / "labels.npy", mmap_mode="r"),
            "signature_ids": np.load(self.root_dir / "signature_ids.npy", mmap_mode="r"),
            "signature_level_ids": np.load(self.root_dir / "signature_level_ids.npy", mmap_mode="r"),
            "signature_relation_ids": np.load(self.root_dir / "signature_relation_ids.npy", mmap_mode="r"),
            "parent_signature_ids": np.load(self.root_dir / "parent_signature_ids.npy", mmap_mode="r"),
            "signature_family_ids": np.load(self.root_dir / "signature_family_ids.npy", mmap_mode="r"),
            "hierarchy_vectors": np.load(self.root_dir / "hierarchy_vectors.npy", mmap_mode="r")
            if (self.root_dir / "hierarchy_vectors.npy").exists()
            else None,
            "loss_mask": np.load(self.root_dir / "loss_mask.npy", mmap_mode="r"),
        }
        self.indices = self._build_indices()

    def _include_sample(self, sample_index: int) -> bool:
        if self.split == "all":
            return True
        if self.split not in {"train", "val"}:
            raise ValueError("split must be 'all', 'train', or 'val'.")
        fraction = min(max(self.val_fraction, 0.0), 1.0)
        if fraction <= 0.0:
            return self.split != "val"
        if fraction >= 1.0:
            return self.split == "val"
        key = f"{self.seed}:{sample_index}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") / float(1 << 64)
        if self.split == "val":
            return bucket < fraction
        return bucket >= fraction

    def _build_indices(self) -> List[int]:
        selected = [idx for idx in range(self.total_samples) if self._include_sample(idx)]
        if self.max_samples:
            rng = random.Random(self.sample_seed)
            rng.shuffle(selected)
            selected = selected[: self.max_samples]
        return selected

    def __len__(self) -> int:
        return len(self.indices)

    def _slice_field(self, field_name: str, start: int, end: int, *, dtype: torch.dtype) -> torch.Tensor:
        field = self.fields[field_name]
        return torch.from_numpy(np.asarray(field[start:end]).copy()).to(dtype=dtype)

    def __getitem__(self, idx: int) -> WindowSample:
        sample_index = self.indices[idx]
        start = int(self.offsets[sample_index])
        length = int(self.lengths[sample_index])
        end = start + length
        hierarchy_vectors_field = self.fields.get("hierarchy_vectors")
        if hierarchy_vectors_field is not None:
            hierarchy_vectors = torch.from_numpy(np.asarray(hierarchy_vectors_field[start:end]).copy()).to(dtype=torch.float32)
        else:
            input_ids = self._slice_field("input_ids", start, end, dtype=torch.long)
            signature_ids = self._slice_field("signature_ids", start, end, dtype=torch.long)
            signature_level_ids = self._slice_field("signature_level_ids", start, end, dtype=torch.long)
            signature_relation_ids = self._slice_field("signature_relation_ids", start, end, dtype=torch.long)
            parent_signature_ids = self._slice_field("parent_signature_ids", start, end, dtype=torch.long)
            signature_family_ids = self._slice_field("signature_family_ids", start, end, dtype=torch.long)
            hierarchy_vectors = _build_hierarchy_vector_tensor(
                input_ids,
                signature_ids,
                signature_level_ids,
                signature_relation_ids,
                parent_signature_ids,
                signature_family_ids,
                token_vocab_size=max(int(self.meta.get("token_vocab_size", int(input_ids.max().item()) + 1 if input_ids.numel() > 0 else 1)), 1),
                signature_vocab_size=max(int(self.meta.get("signature_vocab_size", int(signature_ids.max().item()) + 1 if signature_ids.numel() > 0 else 1)), 1),
                level_vocab_size=max(int(self.meta.get("signature_level_vocab_size", int(signature_level_ids.max().item()) + 1 if signature_level_ids.numel() > 0 else 1)), 1),
                relation_vocab_size=max(int(self.meta.get("signature_relation_vocab_size", int(signature_relation_ids.max().item()) + 1 if signature_relation_ids.numel() > 0 else 1)), 1),
                family_vocab_size=max(int(self.meta.get("signature_family_vocab_size", int(signature_family_ids.max().item()) + 1 if signature_family_ids.numel() > 0 else 1)), 1),
            )
        return WindowSample(
            input_ids=self._slice_field("input_ids", start, end, dtype=torch.long),
            labels=self._slice_field("labels", start, end, dtype=torch.long),
            signature_ids=self._slice_field("signature_ids", start, end, dtype=torch.long),
            signature_level_ids=self._slice_field("signature_level_ids", start, end, dtype=torch.long),
            signature_relation_ids=self._slice_field("signature_relation_ids", start, end, dtype=torch.long),
            parent_signature_ids=self._slice_field("parent_signature_ids", start, end, dtype=torch.long),
            signature_family_ids=self._slice_field("signature_family_ids", start, end, dtype=torch.long),
            hierarchy_vectors=hierarchy_vectors,
            loss_mask=self._slice_field("loss_mask", start, end, dtype=torch.float32),
        )


def split_text_window_dataset(
    dataset: TextWindowDataset,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[torch.utils.data.Subset[TextWindowDataset], torch.utils.data.Subset[TextWindowDataset]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    total = len(dataset)
    if total < 2:
        raise ValueError("Dataset must contain at least 2 samples for a validation split.")
    val_size = max(1, int(round(total * val_fraction)))
    train_size = max(1, total - val_size)
    if train_size + val_size > total:
        val_size = total - train_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)


def build_collate_fn(
    pad_id: int,
    signature_pad_id: int = 0,
) -> Callable[
    [Sequence[WindowSample]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    def collate(
        batch: Sequence[WindowSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(sample.input_ids.numel() for sample in batch)
        hierarchy_dim = max(sample.hierarchy_vectors.size(-1) for sample in batch)
        inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        signatures = torch.full((len(batch), max_len), signature_pad_id, dtype=torch.long)
        signature_levels = torch.full((len(batch), max_len), SIGNATURE_LEVEL_IDS["pad"], dtype=torch.long)
        signature_relations = torch.full((len(batch), max_len), SIGNATURE_RELATION_IDS["pad"], dtype=torch.long)
        parent_signatures = torch.full((len(batch), max_len), signature_pad_id, dtype=torch.long)
        signature_families = torch.full((len(batch), max_len), 0, dtype=torch.long)
        hierarchy_vectors = torch.zeros((len(batch), max_len, hierarchy_dim), dtype=torch.float32)
        loss_masks = torch.zeros((len(batch), max_len), dtype=torch.float32)
        for i, sample in enumerate(batch):
            n = sample.input_ids.numel()
            inputs[i, :n] = sample.input_ids
            labels[i, : sample.labels.numel()] = sample.labels
            signatures[i, : sample.signature_ids.numel()] = sample.signature_ids
            signature_levels[i, : sample.signature_level_ids.numel()] = sample.signature_level_ids
            signature_relations[i, : sample.signature_relation_ids.numel()] = sample.signature_relation_ids
            parent_signatures[i, : sample.parent_signature_ids.numel()] = sample.parent_signature_ids
            signature_families[i, : sample.signature_family_ids.numel()] = sample.signature_family_ids
            hierarchy_vectors[i, : sample.hierarchy_vectors.size(0), : sample.hierarchy_vectors.size(1)] = sample.hierarchy_vectors
            loss_masks[i, : sample.loss_mask.numel()] = sample.loss_mask
        return inputs, labels, signatures, signature_levels, signature_relations, parent_signatures, signature_families, hierarchy_vectors, loss_masks

    return collate


class PrismalConstructionCodec(PrismalTokenizer):
    """Named v2 construction codec; kept separate for clearer architecture docs."""


# CLI-compatible aliases now use the v2 construction codec.
ByteTokenizer = PrismalConstructionCodec
