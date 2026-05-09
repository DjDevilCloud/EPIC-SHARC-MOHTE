"""EPIC-SHARC MOHTE package."""

# SPDX-License-Identifier: AGPL-3.0-or-later
from .config import PrismalWaveConfig
from .data import ByteTokenizer, TextWindowDataset, load_text_corpus
from .model import PrismalRecursiveTorusCore, PrismalWaveModel, PrismalWaveOutput
from .train import (
    evaluate_model,
    generate_text,
    load_model_from_checkpoint,
    run_benchmark,
    save_checkpoint,
    train_model,
)

__version__ = "0.1.0"

__all__ = [
    "ByteTokenizer",
    "TextWindowDataset",
    "PrismalWaveConfig",
    "PrismalRecursiveTorusCore",
    "PrismalWaveModel",
    "PrismalWaveOutput",
    "evaluate_model",
    "generate_text",
    "load_model_from_checkpoint",
    "load_text_corpus",
    "run_benchmark",
    "save_checkpoint",
    "train_model",
    "__version__",
]
