"""Smoke tests for EPIC-SHARC MOHTE."""

# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PrismalWaveConfig
from data import PrismalTokenizer
from model import PrismalWaveModel
from train import load_model_from_checkpoint, save_checkpoint
import epic_sharc_mohte as package


class SmokeTests(unittest.TestCase):
    def _make_small_config(self, tokenizer: PrismalTokenizer) -> PrismalWaveConfig:
        cfg = PrismalWaveConfig()
        cfg.base_vocab_size = tokenizer.base_vocab_size
        cfg.vocab_size = tokenizer.vocab_size
        cfg.signature_vocab_size = tokenizer.signature_vocab_size
        cfg.signature_level_vocab_size = tokenizer.signature_level_vocab_size
        cfg.signature_relation_vocab_size = tokenizer.signature_relation_vocab_size
        cfg.signature_bucket_vocab_size = tokenizer.signature_family_vocab_size
        cfg.d_model = 32
        cfg.ff_mult = 2
        cfg.n_layers = 1
        cfg.n_emitters = 8
        cfg.n_slots = 8
        cfg.n_paths = 1
        cfg.top_k_emitters = 2
        cfg.top_k_slots = 2
        cfg.use_factorized_embedding = True
        cfg.factorized_embedding_dim = 16
        cfg.use_torus_core = False
        cfg.use_hmote = False
        cfg.use_recursive_hmoe = False
        cfg.use_signature_lattice_attention = False
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32
        return cfg

    def test_package_version(self) -> None:
        self.assertEqual(package.__version__, "0.1.0")

    def test_tokenizer_roundtrip(self) -> None:
        tokenizer = PrismalTokenizer()
        text = "Hello world."
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(token_ids)
        self.assertIn("hello", decoded.lower())
        self.assertIn("world", decoded.lower())

    def test_hierarchy_alignment(self) -> None:
        tokenizer = PrismalTokenizer()
        bundle = tokenizer.encode_hierarchy_bundle("One small paragraph.\n\nAnother line.", add_special_tokens=True)
        lengths = {
            len(bundle.token_ids),
            len(bundle.signature_ids),
            len(bundle.signature_level_ids),
            len(bundle.signature_relation_ids),
            len(bundle.parent_signature_ids),
            len(bundle.signature_family_ids),
        }
        self.assertEqual(len(lengths), 1)
        self.assertGreater(len(bundle.token_ids), 0)

    def test_model_forward(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._make_small_config(tokenizer)
        model = PrismalWaveModel(cfg)
        bundle = tokenizer.encode_hierarchy_bundle("Hello world.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)

        output = model(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )

        self.assertEqual(tuple(output.logits.shape[:2]), tuple(input_ids.shape))
        self.assertEqual(output.logits.shape[-1], cfg.vocab_size)

    def test_checkpoint_roundtrip(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._make_small_config(tokenizer)
        model = PrismalWaveModel(cfg)
        model.eval()
        bundle = tokenizer.encode_hierarchy_bundle("Hello world.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = save_checkpoint(model, tmpdir, tokenizer=tokenizer, config=cfg)
            loaded = load_model_from_checkpoint(checkpoint_path, device="cpu")
            loaded.eval()

        original = model(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )
        restored = loaded(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )

        self.assertTrue(torch.allclose(original.logits, restored.logits, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
