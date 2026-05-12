import sys
import unittest
import tempfile
import sys
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PrismalWaveConfig
from data import PrismalTokenizer
from model import PrismalWaveModel
from train import build_train_val_dataloaders, train_model


class SmokeTests(unittest.TestCase):
    def test_tokenizer_roundtrip(self) -> None:
        tokenizer = PrismalTokenizer()
        text = "Hello world."
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(token_ids)
        self.assertIn("hello", decoded.lower())
        self.assertIn("world", decoded.lower())

    def test_tokenizer_unicode_byte_fallback(self) -> None:
        tokenizer = PrismalTokenizer()
        text = "Café 漢字 😀"
        bundle = tokenizer.encode_hierarchy_bundle(text, add_special_tokens=True)
        lengths = {
            len(bundle.token_ids),
            len(bundle.signature_ids),
            len(bundle.signature_level_ids),
            len(bundle.signature_relation_ids),
            len(bundle.parent_signature_ids),
            len(bundle.signature_family_ids),
        }
        self.assertEqual(len(lengths), 1)
        self.assertTrue(any(tokenizer.token_kind_by_id.get(token_id) == "byte" for token_id in bundle.token_ids))

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
        cfg.hierarchical_nest_depth = 2
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

    def test_emitter_router_respects_hierarchy_ids(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = PrismalWaveConfig()
        cfg.base_vocab_size = tokenizer.base_vocab_size
        cfg.vocab_size = tokenizer.vocab_size
        cfg.signature_vocab_size = tokenizer.signature_vocab_size
        cfg.signature_level_vocab_size = tokenizer.signature_level_vocab_size
        cfg.signature_relation_vocab_size = tokenizer.signature_relation_vocab_size
        cfg.signature_bucket_vocab_size = tokenizer.signature_family_vocab_size
        cfg.d_model = 8
        cfg.ff_mult = 2
        cfg.n_layers = 1
        cfg.n_emitters = 4
        cfg.n_slots = 4
        cfg.n_paths = 1
        cfg.top_k_emitters = 1
        cfg.top_k_slots = 1
        cfg.use_factorized_embedding = True
        cfg.factorized_embedding_dim = 8
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
        cfg.torus_weight = 0.0
        cfg.frequency_weight = 0.0
        cfg.emitter_neighbor_weight = 0.0
        cfg.emitter_hierarchy_score_weight = 1.0

        model = PrismalWaveModel(cfg)
        router = model.router
        self.assertIsNotNone(router)
        assert router is not None
        router.eval()

        def _zero_linear(module: torch.nn.Module) -> None:
            weight = getattr(module, "weight", None)
            if weight is not None:
                weight.data.zero_()
            bias = getattr(module, "bias", None)
            if bias is not None:
                bias.data.zero_()

        def _identity_linear(module: torch.nn.Module) -> None:
            _zero_linear(module)
            weight = getattr(module, "weight", None)
            if weight is not None and weight.ndim == 2 and weight.shape[0] == weight.shape[1]:
                weight.data.copy_(torch.eye(weight.shape[0]))

        with torch.no_grad():
            router.emitter_bank.zero_()
            router.operator_hierarchy_bank.zero_()
            router.operator_hierarchy_bank[0, 0] = 8.0
            router.operator_hierarchy_bank[1, 1] = 8.0
            router.operator_hierarchy_bank[2, 0] = -8.0
            router.operator_hierarchy_bank[3, 1] = -8.0
            router.emitter_phase.zero_()
            router.emitter_frequency.fill_(1.0)
            router.path_basis.zero_()
            router.slot_seed.zero_()
            router.signature_embedding.weight.zero_()
            router.signature_embedding.weight[3, 0] = 8.0
            router.signature_embedding.weight[4, 1] = 8.0
            router.family_embedding.weight.zero_()
            router.level_embedding.weight.zero_()
            router.relation_embedding.weight.zero_()
            router.parent_embedding.weight.zero_()
            _identity_linear(router.query_proj)
            _identity_linear(router.value_proj)
            _identity_linear(router.signature_proj)
            _identity_linear(router.hierarchy_proj)
            _identity_linear(router.phase_proj)
            _identity_linear(router.frequency_proj)
            _identity_linear(router.route_gate)
            _identity_linear(router.out_proj)
            _identity_linear(router.slot_query)
            _identity_linear(router.slot_value)
            _identity_linear(router.path_coord_proj)

        hidden = torch.zeros(1, 1, cfg.d_model)
        slots = torch.zeros(1, cfg.n_slots, cfg.d_model)
        shared_ids = torch.zeros(1, 1, dtype=torch.long)

        _, _, stats_a = router.route(
            hidden,
            slots,
            signature_family_ids=shared_ids,
            signature_ids=torch.tensor([[3]], dtype=torch.long),
            signature_level_ids=shared_ids,
            signature_relation_ids=shared_ids,
            parent_signature_ids=shared_ids,
            path_index=0,
            layer_index=0,
        )
        _, _, stats_b = router.route(
            hidden,
            slots,
            signature_family_ids=shared_ids,
            signature_ids=torch.tensor([[4]], dtype=torch.long),
            signature_level_ids=shared_ids,
            signature_relation_ids=shared_ids,
            parent_signature_ids=shared_ids,
            path_index=0,
            layer_index=0,
        )

        top_a = int(stats_a["emitter_top_idx"][0, 0, 0].item())
        top_b = int(stats_b["emitter_top_idx"][0, 0, 0].item())
        self.assertEqual(top_a, 0)
        self.assertEqual(top_b, 1)
        self.assertNotEqual(top_a, top_b)

    def test_hierarchy_growth_is_blocked_during_training(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = PrismalWaveConfig()
        cfg.base_vocab_size = tokenizer.base_vocab_size
        cfg.vocab_size = tokenizer.vocab_size
        cfg.signature_vocab_size = tokenizer.signature_vocab_size
        cfg.signature_level_vocab_size = tokenizer.signature_level_vocab_size
        cfg.signature_relation_vocab_size = tokenizer.signature_relation_vocab_size
        cfg.signature_bucket_vocab_size = tokenizer.signature_family_vocab_size
        cfg.d_model = 8
        cfg.ff_mult = 2
        cfg.n_layers = 1
        cfg.n_emitters = 4
        cfg.n_slots = 4
        cfg.n_paths = 1
        cfg.top_k_emitters = 1
        cfg.top_k_slots = 1
        cfg.use_factorized_embedding = True
        cfg.factorized_embedding_dim = 8
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
        model.train()
        router = model.router
        self.assertIsNotNone(router)
        assert router is not None
        model.set_capacity_growth_locked(True)

        hidden = torch.zeros(1, 1, cfg.d_model)
        slots = torch.zeros(1, cfg.n_slots, cfg.d_model)
        growing_ids = torch.tensor([[999]], dtype=torch.long)

        with self.assertRaises(RuntimeError):
            router.route(
                hidden,
                slots,
                signature_family_ids=growing_ids,
                signature_ids=growing_ids,
                signature_level_ids=growing_ids,
                signature_relation_ids=growing_ids,
                parent_signature_ids=growing_ids,
                path_index=0,
                layer_index=0,
            )

    def test_registry_growth_is_blocked_during_training(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = PrismalWaveConfig()
        cfg.base_vocab_size = tokenizer.base_vocab_size
        cfg.vocab_size = tokenizer.vocab_size
        cfg.signature_vocab_size = tokenizer.signature_vocab_size
        cfg.signature_level_vocab_size = tokenizer.signature_level_vocab_size
        cfg.signature_relation_vocab_size = tokenizer.signature_relation_vocab_size
        cfg.signature_bucket_vocab_size = tokenizer.signature_family_vocab_size
        cfg.d_model = 8
        cfg.ff_mult = 2
        cfg.n_layers = 1
        cfg.n_emitters = 4
        cfg.n_slots = 4
        cfg.n_paths = 1
        cfg.top_k_emitters = 1
        cfg.top_k_slots = 1
        cfg.use_factorized_embedding = True
        cfg.factorized_embedding_dim = 8
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
        model.train()

        growing_ids = torch.tensor([[999]], dtype=torch.long)
        model.set_capacity_growth_locked(True)
        with self.assertRaises(RuntimeError):
            model.registry.observe(
                family_ids=growing_ids,
                level_ids=growing_ids,
                relation_ids=growing_ids,
                parent_ids=growing_ids,
            )

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
        cfg.profile_runtime = True
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
        self.assertIn("timing_token_memory_total_ms", output.route_stats)
        self.assertIn("timing_token_memory_query_ms", output.route_stats)
        self.assertIn("timing_encode_embed_ms", output.route_stats)
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

    def test_anchor_rail_forces_literal_copy_through_beam_search(self) -> None:
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
        cfg.token_memory_window = 32
        cfg.token_memory_top_k = 2
        cfg.token_memory_weight = 0.5
        cfg.token_memory_copy_bias = 0.0
        cfg.token_memory_copy_min_confidence = 0.0
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32

        model = PrismalWaveModel(cfg)
        model.eval()

        prompt = 'please repeat the id "zx-19k" exactly.'
        prompt_bundle = tokenizer.prepare_generation_hierarchy(prompt)
        literal_bundle = tokenizer.encode_hierarchy_bundle('"zx-19k"', add_special_tokens=False)
        self.assertIn('"zx-19k"', prompt)
        self.assertGreater(len(literal_bundle.token_ids), 0)

        input_ids = torch.tensor([prompt_bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([prompt_bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([prompt_bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([prompt_bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([prompt_bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([prompt_bundle.signature_family_ids], dtype=torch.long)

        anchor_state = model.token_memory_attention.init_state(1, input_ids.device, torch.float32)
        literal_ids = torch.tensor(literal_bundle.token_ids, dtype=torch.long)
        literal_len = literal_ids.numel()
        anchor_state.token_ids[0, :literal_len] = literal_ids
        anchor_state.lengths[0] = literal_len
        anchor_state.anchor_token_ids[0, :literal_len] = literal_ids
        anchor_state.anchor_span_ids[0, :literal_len] = 1
        anchor_state.anchor_offsets[0, :literal_len] = torch.arange(literal_len, dtype=torch.long)
        anchor_state.anchor_lengths[0, :literal_len] = literal_len
        anchor_state.anchor_tags[0, :literal_len] = 0xA1B2C3D4
        anchor_state.anchor_flags[0, :literal_len] = 1
        anchor_state.anchor_span_starts[0, :literal_len] = 0
        anchor_state.anchor_cursor_active[0] = True
        anchor_state.anchor_cursor_pos[0] = 0
        anchor_state.anchor_cursor_span_id[0] = 1
        anchor_state.anchor_cursor_offset[0] = 0
        anchor_state.anchor_cursor_length[0] = literal_len
        anchor_state.anchor_cursor_tag[0] = 0xA1B2C3D4

        generated = model.generate(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            max_new_tokens=1,
            min_new_tokens=1,
            beam_size=2,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            use_speculative_decoding=False,
            token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
            token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
            token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
            token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
            suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
            token_memory_state=anchor_state,
        )

        self.assertEqual(int(generated[0, input_ids.size(1)].item()), int(literal_ids[0].item()))

    def test_anchor_rail_forces_bare_identifier_copy_through_beam_search(self) -> None:
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
        cfg.token_memory_window = 32
        cfg.token_memory_top_k = 2
        cfg.token_memory_weight = 0.5
        cfg.token_memory_copy_bias = 0.0
        cfg.token_memory_copy_min_confidence = 0.0
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32

        model = PrismalWaveModel(cfg)
        model.eval()

        prompt = "please repeat the id zx-19k exactly."
        prompt_bundle = tokenizer.prepare_generation_hierarchy(prompt)
        literal_bundle = tokenizer.encode_hierarchy_bundle("zx-19k", add_special_tokens=False)
        self.assertIn("zx-19k", prompt)
        self.assertGreater(len(literal_bundle.token_ids), 0)

        input_ids = torch.tensor([prompt_bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([prompt_bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([prompt_bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([prompt_bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([prompt_bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([prompt_bundle.signature_family_ids], dtype=torch.long)

        anchor_state = model.token_memory_attention.init_state(1, input_ids.device, torch.float32)
        literal_ids = torch.tensor(literal_bundle.token_ids, dtype=torch.long)
        literal_len = literal_ids.numel()
        anchor_state.token_ids[0, :literal_len] = literal_ids
        anchor_state.lengths[0] = literal_len
        anchor_state.anchor_token_ids[0, :literal_len] = literal_ids
        anchor_state.anchor_span_ids[0, :literal_len] = 1
        anchor_state.anchor_offsets[0, :literal_len] = torch.arange(literal_len, dtype=torch.long)
        anchor_state.anchor_lengths[0, :literal_len] = literal_len
        anchor_state.anchor_tags[0, :literal_len] = 0x0BADC0DE
        anchor_state.anchor_flags[0, :literal_len] = 1
        anchor_state.anchor_span_starts[0, :literal_len] = 0
        anchor_state.anchor_cursor_active[0] = True
        anchor_state.anchor_cursor_pos[0] = 0
        anchor_state.anchor_cursor_span_id[0] = 1
        anchor_state.anchor_cursor_offset[0] = 0
        anchor_state.anchor_cursor_length[0] = literal_len
        anchor_state.anchor_cursor_tag[0] = 0x0BADC0DE

        generated = model.generate(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            max_new_tokens=1,
            min_new_tokens=1,
            beam_size=2,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            use_speculative_decoding=False,
            token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
            token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
            token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
            token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
            suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
            token_memory_state=anchor_state,
        )

        self.assertEqual(int(generated[0, input_ids.size(1)].item()), int(literal_ids[0].item()))

    def test_token_copy_aliases_sync_to_canonical_names(self) -> None:
        cfg = PrismalWaveConfig(
            token_copy_window=12,
            token_copy_top_k=4,
            token_copy_weight=0.31,
            token_copy_bias_strength=0.92,
            token_copy_rare_token_cutoff=3,
            token_copy_min_confidence=0.27,
            use_token_copy_cross_attention=False,
            use_token_copy_generation_cache=False,
        )

        self.assertEqual(cfg.token_memory_window, 12)
        self.assertEqual(cfg.token_memory_top_k, 4)
        self.assertAlmostEqual(cfg.token_memory_weight, 0.31)
        self.assertAlmostEqual(cfg.token_memory_copy_bias, 0.92)
        self.assertEqual(cfg.token_memory_rare_token_cutoff, 3)
        self.assertAlmostEqual(cfg.token_memory_copy_min_confidence, 0.27)
        self.assertFalse(cfg.use_token_memory_cross_attention)
        self.assertFalse(cfg.use_token_memory_generation_cache)

    def test_hierarchical_leaf_torus_size_is_preserved_and_validated(self) -> None:
        with self.assertRaises(ValueError):
            PrismalWaveConfig(hierarchical_leaf_torus_size=1)

        tokenizer = PrismalTokenizer()
        cfg = PrismalWaveConfig(hierarchical_leaf_torus_size=6)
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
        cfg.hierarchical_nest_depth = 2
        cfg.use_signature_lattice_attention = False
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32

        model = PrismalWaveModel(cfg)
        self.assertIsNotNone(model.token_hierarchy)
        assert model.token_hierarchy is not None
        self.assertEqual(model.token_hierarchy.leaf_size, 6)

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

    def test_dataset_streaming_toggle_uses_in_memory_loader(self) -> None:
        tokenizer = PrismalTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "tiny.txt"
            source.write_text("Alpha beta gamma.\n\nDelta epsilon zeta.", encoding="utf-8")
            train_loader, val_loader = build_train_val_dataloaders(
                source,
                tokenizer,
                seq_len=16,
                batch_size=2,
                max_samples=0,
                val_fraction=0.5,
                seed=7,
                streaming=False,
            )

        self.assertFalse(isinstance(getattr(train_loader, "dataset", None), torch.utils.data.IterableDataset))
        self.assertFalse(isinstance(getattr(val_loader, "dataset", None), torch.utils.data.IterableDataset))

    def test_gradient_accumulation_defers_optimizer_step(self) -> None:
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int):
                value = torch.tensor([index], dtype=torch.long)
                mask = torch.tensor([1.0], dtype=torch.float32)
                return value, value, value, value, value, value, value, mask

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.cfg = PrismalWaveConfig()
                self.cfg.use_gradient_accumulation = True
                self.cfg.gradient_accumulation_steps = 2
                self.precision_policy = None
                self.use_turbo_quantization = False

            def set_capacity_growth_locked(self, locked: bool) -> None:
                return None

            def configure_precision(self, *args, **kwargs) -> None:
                return None

            def compute_loss(self, *args, **kwargs):
                loss = (self.weight - 2.0).pow(2)
                value = self.weight.detach().new_tensor(0.5)
                route_stats = {
                    "signature_agreement": value.unsqueeze(0),
                    "avg_entropy": value,
                    "avg_active_emitters": value,
                    "avg_emitter_cell_soft_occupancy": value,
                    "emitter_cell_breadth": value,
                    "avg_emitter_cell_soft_breadth": value,
                    "family_specialist_active_count": value,
                    "family_specialist_unique_families": value,
                    "family_specialist_bank_size": value,
                    "family_specialist_capacity": value,
                    "family_specialist_budget": value,
                    "family_specialist_hit_rate": value,
                    "family_specialist_gate_mean": value,
                    "avg_emitter_topk_effective_count": value,
                    "torus_coverage_loss": value,
                    "emitter_usage_entropy": value,
                    "emitter_usage_concentration": value,
                }
                return loss, SimpleNamespace(ce_loss=loss.detach(), aux_loss=value, route_stats=route_stats)

        dataset = DummyDataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        model = DummyModel()
        original_step = torch.optim.AdamW.step
        step_calls = {"count": 0}

        def wrapped_step(self, *args, **kwargs):
            step_calls["count"] += 1
            return original_step(self, *args, **kwargs)

        with mock.patch.object(torch.optim.AdamW, "step", wrapped_step):
            train_model(
                model,
                loader,
                torch.device("cpu"),
                cfg=model.cfg,
                optimizer_name="adamw",
                epochs=1,
                steps=0,
                lr=1e-3,
                grad_clip=0.0,
                progress=False,
                val_loader=None,
                diagnostic_interval=999,
                use_amp=False,
            )

        self.assertEqual(step_calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
