import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PrismalWaveConfig
from data import PrismalTokenizer
from model import PrismalWaveModel


class SmokeTests(unittest.TestCase):
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

    def test_token_memory_forward_and_copy_bias(self) -> None:
        tokenizer = PrismalTokenizer()
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
        cfg.use_torus_core = True
        cfg.use_hmote = False
        cfg.use_recursive_hmoe = False
        cfg.use_signature_lattice_attention = False
        cfg.use_token_memory_cross_attention = True
        cfg.use_token_memory_generation_cache = True
        cfg.token_memory_window = 6
        cfg.token_memory_top_k = 3
        cfg.token_memory_weight = 0.5
        cfg.token_memory_copy_bias = 1.0
        cfg.token_memory_rare_token_cutoff = 2
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32

        model = PrismalWaveModel(cfg)
        model.eval()
        attention = model.token_memory_attention
        self.assertIsNotNone(attention)
        assert attention is not None
        attention.eval()
        for parameter in attention.parameters():
            parameter.data.zero_()

        bundle = tokenizer.encode_hierarchy_bundle("alpha beta alpha gamma alpha", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)
        token_counts = torch.bincount(input_ids[0], minlength=cfg.vocab_size)
        repeated_candidates = torch.nonzero(token_counts > 1, as_tuple=False).flatten()
        repeated_token_id = int(repeated_candidates[0].item()) if repeated_candidates.numel() > 0 else int(input_ids[0, 0].item())
        unique_candidates = torch.nonzero(token_counts == 1, as_tuple=False).flatten()
        unique_token_id = int(unique_candidates[0].item()) if unique_candidates.numel() > 0 else repeated_token_id

        output = model(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )

        self.assertIsNotNone(output.token_memory_state)
        self.assertIn("token_memory_copy_logits", output.route_stats)
        copy_logits = output.route_stats["token_memory_copy_logits"]
        self.assertEqual(copy_logits.shape[-1], cfg.vocab_size)
        repeated_score = float(copy_logits[0, repeated_token_id].item())
        unique_score = float(copy_logits[0, unique_token_id].item())
        self.assertGreater(repeated_score, unique_score)

        step_output = model.forward_incremental(
            input_ids[:, -1:],
            signature_family_ids=signature_family_ids[:, -1:],
            signature_ids=signature_ids[:, -1:],
            signature_level_ids=signature_level_ids[:, -1:],
            signature_relation_ids=signature_relation_ids[:, -1:],
            parent_signature_ids=parent_signature_ids[:, -1:],
            slot_state=None,
            signature_lattice_state=None,
            token_memory_state=output.token_memory_state,
            position_index=input_ids.size(1) - 1,
        )
        self.assertEqual(step_output[0].shape[-1], cfg.vocab_size)
        self.assertIsNotNone(step_output[2].token_memory_state)

    def test_token_memory_disabled_shape_stability(self) -> None:
        tokenizer = PrismalTokenizer()
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
        cfg.use_token_memory_cross_attention = False
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32

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
        self.assertIsNone(output.token_memory_state)


if __name__ == "__main__":
    unittest.main()
