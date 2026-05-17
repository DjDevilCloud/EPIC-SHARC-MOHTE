import sys
import unittest
import tempfile
import math
import sys
import re
from collections import Counter
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PrismalWaveConfig
from config import load_config
from cli import build_parser, _build_config, _resolve_tokenizer_bootstrap
from data import PrismalTokenizer, iter_text_corpus
from model import PrismalTorusCore, PrismalWaveModel
from muon_optim import PrecisionAdaptiveHierarchicalOptimizer
from quantization import QuantizationConfig
from train import build_train_val_dataloaders, generate_text, load_model_from_checkpoint, save_checkpoint, train_model, _clip_optimizer_group_gradients, _token_superposition_phase_active


class SmokeTests(unittest.TestCase):
    def _build_gate_cfg(
        self,
        tokenizer: PrismalTokenizer,
        *,
        use_gate: bool,
        use_gatetrain: bool = False,
        use_fullgatetrain: bool = False,
        use_hmote: bool = False,
        use_torus_core: bool = False,
        use_learned_residency_head: bool = False,
        use_residency_with_reinforcement: bool = False,
        use_contrastive_routing: bool = False,
        use_contrastive_routing_signature_neighborhood: bool = False,
        use_contrastive_routing_temporal: bool = False,
        use_contrastive_routing_residency: bool = False,
        use_contrastive_routing_cross_view: bool = False,
        use_contrastive_routing_self_contrast: bool = False,
    ) -> PrismalWaveConfig:
        cfg = PrismalWaveConfig()
        cfg.base_vocab_size = tokenizer.base_vocab_size
        cfg.vocab_size = tokenizer.vocab_size
        cfg.signature_vocab_size = tokenizer.signature_vocab_size
        cfg.signature_level_vocab_size = tokenizer.signature_level_vocab_size
        cfg.signature_relation_vocab_size = tokenizer.signature_relation_vocab_size
        cfg.signature_bucket_vocab_size = tokenizer.signature_family_vocab_size
        cfg.d_model = 16
        cfg.ff_mult = 2
        cfg.n_layers = 1
        cfg.n_emitters = 8
        cfg.n_slots = 8
        cfg.n_paths = 1
        cfg.top_k_emitters = 2
        cfg.top_k_slots = 2
        cfg.use_factorized_embedding = True
        cfg.factorized_embedding_dim = 8
        cfg.use_torus_core = use_torus_core
        cfg.Torus_SHARC_Router = use_torus_core
        cfg.use_hmote = use_hmote
        cfg.hmote_branching = 4
        cfg.use_recursive_hmoe = False
        cfg.hierarchical_nest_depth = 2
        cfg.use_signature_lattice_attention = False
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32
        cfg.use_gate = use_gate
        cfg.gate_residency_budget = 4
        cfg.gate_prefetch_horizon = 1
        cfg.gate_tile_granularity = 2
        cfg.gate_offload_to_cpu = False
        cfg.gate_fallback_on_miss = True
        cfg.use_gatetrain = use_gatetrain
        cfg.use_fullgatetrain = use_fullgatetrain
        cfg.gatetrain_residency_budget = 4
        cfg.gatetrain_prefetch_horizon = 1
        cfg.gatetrain_tile_granularity = 2
        cfg.gatetrain_offload_to_cpu = False
        cfg.gatetrain_fallback_on_miss = True
        cfg.use_learned_residency_head = use_learned_residency_head
        cfg.residency_head_layers = 1
        cfg.residency_head_hidden_dim = 32
        cfg.learned_residency_weight = 0.1
        cfg.use_residency_with_reinforcement = use_residency_with_reinforcement
        cfg.use_contrastive_routing = use_contrastive_routing
        cfg.contrastive_routing_weight = 0.1
        cfg.contrastive_routing_temperature = 0.1
        cfg.contrastive_routing_hard_negatives = False
        cfg.use_contrastive_routing_signature_neighborhood = use_contrastive_routing_signature_neighborhood
        cfg.use_contrastive_routing_temporal = use_contrastive_routing_temporal
        cfg.use_contrastive_routing_residency = use_contrastive_routing_residency
        cfg.use_contrastive_routing_cross_view = use_contrastive_routing_cross_view
        cfg.use_contrastive_routing_self_contrast = use_contrastive_routing_self_contrast
        return cfg

    def _build_small_stability_cfg(self, tokenizer: PrismalTokenizer) -> PrismalWaveConfig:
        cfg = PrismalWaveConfig()
        cfg.base_vocab_size = tokenizer.base_vocab_size
        cfg.vocab_size = tokenizer.vocab_size
        cfg.signature_vocab_size = tokenizer.signature_vocab_size
        cfg.signature_level_vocab_size = tokenizer.signature_level_vocab_size
        cfg.signature_relation_vocab_size = tokenizer.signature_relation_vocab_size
        cfg.signature_bucket_vocab_size = tokenizer.signature_family_vocab_size
        cfg.d_model = 16
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
        cfg.Torus_SHARC_Router = False
        cfg.use_hmote = False
        cfg.use_recursive_hmoe = False
        cfg.use_signature_lattice_attention = False
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 16
        cfg.training_finite_guard_enabled = True
        cfg.inference_finite_guard_enabled = True
        cfg.grad_clip_muon = 0.5
        cfg.grad_clip_scalar = 0.75
        cfg.grad_clip_rowwise = 0.6
        return cfg

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

    def test_pretokenized_bootstrap_uses_bundle_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.jsonl"
            source.write_text("{}", encoding="utf-8")
            meta_path = root / "meta.json"
            meta_path.write_text(
                "{\"source\": \"%s\", \"vocab_size\": 12345}" % source.as_posix(),
                encoding="utf-8",
            )

            (
                tokenizer_source,
                tokenizer_max_source_samples,
                tokenizer_max_new_tokens,
                tokenizer_min_frequency,
                tokenizer_max_line_tokens,
                tokenizer_max_signature_tokens,
                tokenizer_supervised_only,
                meta_vocab_size,
            ) = _resolve_tokenizer_bootstrap(
                root,
                max_source_samples=99,
                max_new_tokens=77,
                min_frequency=9,
                max_line_tokens=11,
                max_signature_tokens=13,
                supervised_only=False,
            )

            self.assertEqual(Path(tokenizer_source), source)
            self.assertEqual(tokenizer_max_source_samples, 0)
            self.assertEqual(tokenizer_max_new_tokens, 0)
            self.assertEqual(tokenizer_min_frequency, 2)
            self.assertEqual(tokenizer_max_line_tokens, 0)
            self.assertEqual(tokenizer_max_signature_tokens, 0)
            self.assertTrue(tokenizer_supervised_only)
            self.assertEqual(meta_vocab_size, 12345)

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
        cfg.Torus_SHARC_Router = False
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

    def test_gate_config_roundtrip(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=True)
        payload = cfg.to_dict()
        clone = PrismalWaveConfig.from_dict(payload)
        self.assertTrue(clone.use_gate)
        self.assertEqual(clone.gate_residency_budget, cfg.gate_residency_budget)
        self.assertEqual(clone.gate_prefetch_horizon, cfg.gate_prefetch_horizon)
        self.assertEqual(clone.gate_tile_granularity, cfg.gate_tile_granularity)
        self.assertFalse(clone.gate_offload_to_cpu)
        self.assertTrue(clone.gate_fallback_on_miss)

    def test_gatetrain_config_roundtrip(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_gatetrain=True)
        payload = cfg.to_dict()
        clone = PrismalWaveConfig.from_dict(payload)
        self.assertTrue(clone.use_gatetrain)
        self.assertEqual(clone.gatetrain_residency_budget, cfg.gatetrain_residency_budget)
        self.assertEqual(clone.gatetrain_prefetch_horizon, cfg.gatetrain_prefetch_horizon)
        self.assertEqual(clone.gatetrain_tile_granularity, cfg.gatetrain_tile_granularity)
        self.assertFalse(clone.gatetrain_offload_to_cpu)
        self.assertTrue(clone.gatetrain_fallback_on_miss)

    def test_full_gatetrain_config_roundtrip(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_gatetrain=True, use_fullgatetrain=True)
        payload = cfg.to_dict()
        clone = PrismalWaveConfig.from_dict(payload)
        self.assertTrue(clone.use_fullgatetrain)
        self.assertTrue(clone.use_gatetrain)

    def test_torus_sharc_router_requires_torus_core(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_small_stability_cfg(tokenizer)
        cfg.Torus_SHARC_Router = True
        cfg.use_torus_core = False
        with self.assertRaises(ValueError):
            PrismalWaveModel(cfg)

    def test_torus_patch_linear_indices_match_legacy_gather_and_scatter(self) -> None:
        cfg = PrismalWaveConfig()
        cfg.d_model = 8
        cfg.n_paths = 1
        cfg.torus_depth = 4
        cfg.torus_height = 5
        cfg.torus_width = 6
        cfg.torus_local_field_radius = 1
        cfg.torus_scout_read_radius = 1
        cfg.torus_relay_write_radius = 1
        cfg.torus_transport_interval = 3
        core = PrismalTorusCore(cfg)

        batch_size = 2
        field_dim = 4
        field = torch.randn(batch_size, core.depth, core.height, core.width, field_dim)
        center_z = torch.tensor([0, core.depth - 1], dtype=torch.long)
        center_y = torch.tensor([1, 0], dtype=torch.long)
        center_x = torch.tensor([2, core.width - 1], dtype=torch.long)
        patch_idx = core._patch_linear_indices(center_z, center_y, center_x, core.local_offsets)

        gathered_linear = core._gather_patch_from_linear_idx(field, patch_idx)
        gathered_legacy = core._gather_patch(field, center_z, center_y, center_x, core.local_field_radius)
        self.assertTrue(torch.allclose(gathered_linear, gathered_legacy))

        update = torch.randn(batch_size, patch_idx.size(1), field_dim)
        linear_field = field.clone().reshape(batch_size, -1, field_dim)
        linear_field.scatter_add_(1, patch_idx.unsqueeze(-1).expand(-1, -1, field_dim), update)
        linear_field = linear_field.view(batch_size, core.depth, core.height, core.width, field_dim)

        offsets = core.local_offsets.to(field.device)
        zz = torch.remainder(center_z.unsqueeze(1) + offsets[:, 0].view(1, -1), core.depth)
        yy = torch.remainder(center_y.unsqueeze(1) + offsets[:, 1].view(1, -1), core.height)
        xx = torch.remainder(center_x.unsqueeze(1) + offsets[:, 2].view(1, -1), core.width)
        legacy_idx = (zz * (core.height * core.width) + yy * core.width + xx).unsqueeze(-1)
        legacy_field = field.clone().reshape(batch_size, -1, field_dim)
        legacy_field.scatter_add_(1, legacy_idx.expand(-1, -1, field_dim), update)
        legacy_field = legacy_field.view(batch_size, core.depth, core.height, core.width, field_dim)

        self.assertTrue(torch.allclose(linear_field, legacy_field))

    def test_torus_chunked_step_returns_direct_stats_for_single_step_chunk(self) -> None:
        cfg = PrismalWaveConfig()
        cfg.d_model = 8
        cfg.n_paths = 1
        cfg.torus_depth = 4
        cfg.torus_height = 5
        cfg.torus_width = 6
        cfg.torus_local_field_radius = 1
        cfg.torus_scout_read_radius = 1
        cfg.torus_relay_write_radius = 1
        cfg.torus_transport_interval = 3
        core = PrismalTorusCore(cfg)

        hidden_chunk = torch.randn(1, 2, cfg.d_model)
        field_state = torch.zeros(1, core.depth, core.height, core.width, cfg.d_model)
        step_output = torch.randn(1, cfg.d_model)
        step_stats = {
            "torus_entropy": torch.tensor(1.25),
            "recursive_depth": torch.tensor(3.0),
            "recursive_aux_loss": torch.tensor(0.5),
            "recursive_child_count": torch.tensor(2.0),
        }

        def fake_step(step_hidden: torch.Tensor, current_state: torch.Tensor, **kwargs):
            return step_output, current_state, step_stats

        with mock.patch.object(core, "step", side_effect=fake_step) as patched_step:
            chunk_output, next_state, chunk_stats = core.chunked_step(hidden_chunk, field_state)

        expected_output = hidden_chunk + (step_output - hidden_chunk[:, 0, :]).unsqueeze(1)
        self.assertTrue(torch.allclose(chunk_output, expected_output))
        self.assertEqual(patched_step.call_count, 1)
        self.assertIsNot(chunk_stats, step_stats)
        self.assertEqual(set(chunk_stats.keys()), set(step_stats.keys()))
        self.assertTrue(torch.equal(chunk_stats["recursive_depth"], step_stats["recursive_depth"]))
        self.assertTrue(torch.equal(chunk_stats["recursive_aux_loss"], step_stats["recursive_aux_loss"]))
        self.assertTrue(torch.equal(chunk_stats["recursive_child_count"], step_stats["recursive_child_count"]))
        self.assertTrue(torch.equal(chunk_stats["torus_entropy"], step_stats["torus_entropy"]))
        self.assertIsNotNone(next_state)

    def test_learned_residency_config_roundtrip(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(
            tokenizer,
            use_gate=False,
            use_learned_residency_head=True,
            use_residency_with_reinforcement=True,
        )
        payload = cfg.to_dict()
        clone = PrismalWaveConfig.from_dict(payload)
        self.assertTrue(clone.use_learned_residency_head)
        self.assertTrue(clone.use_residency_with_reinforcement)
        self.assertEqual(clone.residency_head_layers, cfg.residency_head_layers)
        self.assertEqual(clone.residency_head_hidden_dim, cfg.residency_head_hidden_dim)
        self.assertEqual(clone.learned_residency_weight, cfg.learned_residency_weight)

    def test_learned_residency_head_constructs_when_enabled(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_learned_residency_head=True)
        model = PrismalWaveModel(cfg)
        self.assertIsNotNone(model.learned_residency_head)
        self.assertTrue(model.use_learned_residency_head)
        self.assertEqual(model.residency_head_layers, 1)

    def test_contrastive_routing_config_roundtrip(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(
            tokenizer,
            use_gate=False,
            use_contrastive_routing=True,
            use_contrastive_routing_signature_neighborhood=True,
            use_contrastive_routing_temporal=True,
            use_contrastive_routing_cross_view=True,
        )
        payload = cfg.to_dict()
        clone = PrismalWaveConfig.from_dict(payload)
        self.assertTrue(clone.use_contrastive_routing)
        self.assertTrue(clone.use_contrastive_routing_signature_neighborhood)
        self.assertTrue(clone.use_contrastive_routing_temporal)
        self.assertTrue(clone.use_contrastive_routing_cross_view)
        self.assertFalse(clone.use_contrastive_routing_residency)
        self.assertFalse(clone.use_contrastive_routing_self_contrast)
        self.assertEqual(clone.contrastive_routing_weight, cfg.contrastive_routing_weight)
        self.assertEqual(clone.contrastive_routing_temperature, cfg.contrastive_routing_temperature)

    def test_contrastive_routing_disabled_is_noop(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_contrastive_routing=False)
        model = PrismalWaveModel(cfg)
        bundle = tokenizer.encode_hierarchy_bundle("Contrastive routing remains off.", add_special_tokens=True)
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
        self.assertTrue(torch.isfinite(output.aux_loss))
        self.assertEqual(float(output.route_stats["contrastive_routing_enabled"].item()), 0.0)
        self.assertEqual(float(output.route_stats["contrastive_routing_loss"].item()), 0.0)

    def test_contrastive_routing_enabled_emits_losses(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(
            tokenizer,
            use_gate=True,
            use_gatetrain=True,
            use_learned_residency_head=True,
            use_contrastive_routing=True,
            use_contrastive_routing_signature_neighborhood=True,
            use_contrastive_routing_temporal=True,
            use_contrastive_routing_residency=True,
            use_contrastive_routing_cross_view=True,
            use_contrastive_routing_self_contrast=True,
        )
        cfg.n_paths = 2
        cfg.hierarchical_nest_depth = 1
        model = PrismalWaveModel(cfg)
        bundle = tokenizer.encode_hierarchy_bundle("Contrastive routing should light up.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids, bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids, bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids, bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids, bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids, bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids, bundle.signature_family_ids], dtype=torch.long)
        output = model(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )
        self.assertTrue(torch.isfinite(output.aux_loss))
        self.assertEqual(float(output.route_stats["contrastive_routing_enabled"].item()), 1.0)
        self.assertIn("contrastive_routing_signature_neighborhood_loss", output.route_stats)
        self.assertIn("contrastive_routing_temporal_loss", output.route_stats)
        self.assertIn("contrastive_routing_residency_loss", output.route_stats)
        self.assertIn("contrastive_routing_cross_view_loss", output.route_stats)
        self.assertIn("contrastive_routing_self_contrast_loss", output.route_stats)
        self.assertIn("aux_signature_loss_term", output.route_stats)
        self.assertIn("aux_routing_entropy_term", output.route_stats)
        self.assertIn("aux_loss_total", output.route_stats)
        self.assertTrue(torch.isclose(output.route_stats["aux_loss_total"], output.aux_loss.detach(), atol=1e-6).item())
        self.assertTrue(torch.isfinite(output.route_stats["contrastive_routing_loss"]))

    def test_contrastive_routing_cli_flags_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--data",
            "demo/corpus",
            "--save-dir",
            "tmp/run",
        ])
        self.assertFalse(args.use_contrastive_routing)
        self.assertFalse(args.use_contrastive_routing_signature_neighborhood)
        args = parser.parse_args([
            "train",
            "--data",
            "demo/corpus",
            "--save-dir",
            "tmp/run",
            "--use-contrastive-routing",
            "--use-contrastive-routing-temporal",
            "--use-contrastive-routing-cross-view",
        ])
        self.assertTrue(args.use_contrastive_routing)
        self.assertTrue(args.use_contrastive_routing_temporal)
        self.assertTrue(args.use_contrastive_routing_cross_view)

    def test_nested_learning_cli_flags_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--data",
            "demo/corpus",
            "--save-dir",
            "tmp/run",
            "--use-nested-learning",
            "--nested-learning-global-interval",
            "6",
            "--nested-learning-mid-ema-beta",
            "0.91",
        ])
        cfg = _build_config(args)
        self.assertTrue(args.use_nested_learning)
        self.assertTrue(cfg.use_nested_learning)
        self.assertEqual(cfg.nested_learning_global_interval, 6)
        self.assertAlmostEqual(cfg.nested_learning_mid_ema_beta, 0.91)

    def test_token_superposition_config_roundtrip_and_checkpoint_persistence(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_small_stability_cfg(tokenizer)
        cfg.use_token_superposition_training = True
        cfg.token_superposition_bag_size = 8
        cfg.token_superposition_phase_fraction = 0.25
        payload = cfg.to_dict()
        clone = PrismalWaveConfig.from_dict(payload)
        self.assertTrue(clone.use_token_superposition_training)
        self.assertEqual(clone.token_superposition_bag_size, cfg.token_superposition_bag_size)
        self.assertEqual(clone.token_superposition_phase_fraction, cfg.token_superposition_phase_fraction)

        model = PrismalWaveModel(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, tmpdir, config=cfg)
            saved_cfg = load_config(Path(tmpdir) / "config.json")
        self.assertTrue(saved_cfg.use_token_superposition_training)
        self.assertEqual(saved_cfg.token_superposition_bag_size, cfg.token_superposition_bag_size)
        self.assertEqual(saved_cfg.token_superposition_phase_fraction, cfg.token_superposition_phase_fraction)

    def test_token_superposition_cli_flags_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--data",
            "demo/corpus",
            "--save-dir",
            "tmp/run",
        ])
        self.assertFalse(args.use_token_superposition_training)
        self.assertEqual(args.token_superposition_bag_size, PrismalWaveConfig().token_superposition_bag_size)

        args = parser.parse_args([
            "train",
            "--data",
            "demo/corpus",
            "--save-dir",
            "tmp/run",
            "--use-token-superposition-training",
            "--token-superposition-bag-size",
            "4",
            "--token-superposition-phase-fraction",
            "0.4",
        ])
        self.assertTrue(args.use_token_superposition_training)
        self.assertEqual(args.token_superposition_bag_size, 4)
        self.assertAlmostEqual(args.token_superposition_phase_fraction, 0.4)

    def test_token_superposition_phase_schedule_is_resume_safe(self) -> None:
        cfg = PrismalWaveConfig(
            use_token_superposition_training=True,
            token_superposition_bag_size=8,
            token_superposition_phase_fraction=0.5,
        )
        self.assertTrue(
            _token_superposition_phase_active(
                cfg,
                resume_global_step=2,
                optimizer_step=0,
                scheduler_total_steps=8,
            )
        )
        self.assertFalse(
            _token_superposition_phase_active(
                cfg,
                resume_global_step=2,
                optimizer_step=3,
                scheduler_total_steps=8,
            )
        )
        self.assertFalse(
            _token_superposition_phase_active(
                PrismalWaveConfig(use_token_superposition_training=False),
                resume_global_step=0,
                optimizer_step=0,
                scheduler_total_steps=8,
            )
        )

    def test_token_superposition_compute_loss_bags_odd_length_batches(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_small_stability_cfg(tokenizer)
        cfg.use_token_superposition_training = True
        cfg.token_superposition_bag_size = 3
        cfg.token_superposition_phase_fraction = 1.0
        model = PrismalWaveModel(cfg)
        model.train()

        bundle = tokenizer.encode_hierarchy_bundle("Token superposition smoke test.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)

        pad_id = tokenizer.pad_id
        target_len = input_ids.size(1) + (1 if input_ids.size(1) % 2 == 0 else 2)
        pad_width = target_len - input_ids.size(1)
        if pad_width > 0:
            pad_tokens = torch.full((1, pad_width), pad_id, dtype=torch.long)
            pad_mask = torch.zeros((1, pad_width), dtype=torch.float32)
            input_ids = torch.cat([input_ids, pad_tokens], dim=1)
            signature_ids = torch.cat([signature_ids, torch.zeros_like(pad_tokens)], dim=1)
            signature_level_ids = torch.cat([signature_level_ids, torch.zeros_like(pad_tokens)], dim=1)
            signature_relation_ids = torch.cat([signature_relation_ids, torch.zeros_like(pad_tokens)], dim=1)
            parent_signature_ids = torch.cat([parent_signature_ids, torch.zeros_like(pad_tokens)], dim=1)
            signature_family_ids = torch.cat([signature_family_ids, torch.zeros_like(pad_tokens)], dim=1)
            loss_mask = torch.cat([torch.ones((1, input_ids.size(1) - pad_width), dtype=torch.float32), pad_mask], dim=1)
        else:
            loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

        labels = input_ids.clone()
        loss, output = model.compute_loss(
            input_ids,
            labels,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            signature_family_ids=signature_family_ids,
            loss_mask=loss_mask,
            superposition_bag_size=3,
        )

        self.assertTrue(torch.isfinite(loss))
        expected_bag_count = math.ceil(input_ids.size(1) / 3)
        self.assertEqual(tuple(output.logits.shape[:2]), (1, expected_bag_count))
        self.assertEqual(float(output.route_stats["token_superposition_active"].item()), 1.0)
        self.assertEqual(int(output.route_stats["token_superposition_bag_size"].item()), 3)
        self.assertEqual(int(output.route_stats["token_superposition_bag_count"].item()), expected_bag_count)

    def test_stability_config_roundtrip_and_checkpoint_persistence(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_small_stability_cfg(tokenizer)
        payload = cfg.to_dict()
        clone = PrismalWaveConfig.from_dict(payload)
        self.assertTrue(clone.training_finite_guard_enabled)
        self.assertTrue(clone.inference_finite_guard_enabled)
        self.assertEqual(clone.grad_clip_muon, cfg.grad_clip_muon)
        self.assertEqual(clone.grad_clip_scalar, cfg.grad_clip_scalar)
        self.assertEqual(clone.grad_clip_rowwise, cfg.grad_clip_rowwise)

        model = PrismalWaveModel(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, tmpdir, config=cfg)
            saved_cfg = load_config(Path(tmpdir) / "config.json")
        self.assertTrue(saved_cfg.training_finite_guard_enabled)
        self.assertTrue(saved_cfg.inference_finite_guard_enabled)
        self.assertEqual(saved_cfg.grad_clip_muon, cfg.grad_clip_muon)
        self.assertEqual(saved_cfg.grad_clip_scalar, cfg.grad_clip_scalar)
        self.assertEqual(saved_cfg.grad_clip_rowwise, cfg.grad_clip_rowwise)

    def test_leaf_precision_mode_preserves_fp4(self) -> None:
        cfg = PrismalWaveConfig()
        self.assertEqual(cfg.bitsandbytes_leaf_precision_mode, "fp4")
        self.assertEqual(cfg.bitsandbytes_leaf_quant_type, "fp4")
        self.assertEqual(cfg.bitsandbytes_leaf_compute_dtype, "bfloat16")

        quant_cfg = QuantizationConfig(bitsandbytes_leaf_precision_mode="fp4", bitsandbytes_leaf_quant_type="fp4")
        self.assertEqual(quant_cfg.bitsandbytes_leaf_precision_mode, "fp4")
        self.assertEqual(quant_cfg.bitsandbytes_leaf_quant_type, "fp4")

    def test_torus_checkpoint_without_router_weights_loads(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_gatetrain=True, use_fullgatetrain=True, use_hmote=True, use_torus_core=True)
        model = PrismalWaveModel(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, tmpdir, config=cfg)
            checkpoint_path = Path(tmpdir) / "model.pt"
            payload = torch.load(checkpoint_path, map_location="cpu")
            payload["model_state"] = {
                key: value for key, value in payload["model_state"].items() if not key.startswith("router.")
            }
            torch.save(payload, checkpoint_path)
            loaded = load_model_from_checkpoint(checkpoint_path, device="cpu")

        self.assertTrue(loaded.use_torus_core)
        self.assertIsNotNone(loaded.router)

    def test_gate_disabled_matches_baseline_logits(self) -> None:
        tokenizer = PrismalTokenizer()
        torch.manual_seed(1234)
        baseline_cfg = self._build_gate_cfg(tokenizer, use_gate=False)
        gated_cfg = self._build_gate_cfg(tokenizer, use_gate=True)
        base_model = PrismalWaveModel(baseline_cfg)
        gated_model = PrismalWaveModel(gated_cfg)
        gated_model.load_state_dict(base_model.state_dict(), strict=False)
        base_model.eval()
        gated_model.eval()
        bundle = tokenizer.encode_hierarchy_bundle("Gate parity check.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)

        base_output = base_model(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )
        gated_output = gated_model(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )

        torch.testing.assert_close(gated_output.logits, base_output.logits, rtol=0.0, atol=0.0)
        self.assertIn("gate_predicted_tiles", gated_output.route_stats)

    def test_gate_planner_selects_expected_tiles(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=True, use_hmote=True)
        model = PrismalWaveModel(cfg)
        self.assertIsNotNone(model.gate_controller)
        assert model.gate_controller is not None

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        signature_family_ids = torch.tensor([[1, 2]], dtype=torch.long)
        route_stats = {
            "selected_path_index": torch.tensor([3], dtype=torch.long),
            "emitter_top_idx": torch.tensor([[[1, 7]]], dtype=torch.long),
            "signature_agreement": torch.tensor(0.92),
            "avg_entropy": torch.tensor(0.25),
        }

        plan = model.gate_controller.plan(
            input_ids=input_ids,
            signature_family_ids=signature_family_ids,
            route_stats=route_stats,
            path_index=3,
            position_index=0,
        )

        self.assertEqual(plan.family_ids, (1, 2))
        self.assertEqual(plan.expert_ids, (3,))
        self.assertEqual(plan.emitter_tile_ids, (1, 2, 0, 3))
        self.assertFalse(plan.signature_lattice_hot)
        self.assertFalse(plan.token_memory_hot)
        self.assertGreater(plan.confidence, 0.0)

    def test_gate_planner_blends_learned_tiles(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=True, use_learned_residency_head=True, use_hmote=True)
        cfg.gate_residency_budget = 6
        model = PrismalWaveModel(cfg)
        assert model.gate_controller is not None
        route_stats = {
            "selected_path_index": torch.tensor([3], dtype=torch.long),
            "emitter_top_idx": torch.tensor([[[1, 7]]], dtype=torch.long),
            "learned_residency_top_tiles": torch.tensor([[4, 5]], dtype=torch.long),
            "learned_residency_confidence": torch.tensor([0.91]),
            "signature_agreement": torch.tensor(0.92),
            "avg_entropy": torch.tensor(0.25),
        }
        plan = model.gate_controller.plan(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            signature_family_ids=torch.tensor([[1, 2]], dtype=torch.long),
            route_stats=route_stats,
            path_index=3,
            position_index=0,
        )
        self.assertIn(4, plan.emitter_tile_ids)
        self.assertIn(5, plan.emitter_tile_ids)
        self.assertGreaterEqual(plan.confidence, 0.91)

    def test_gatetrain_planner_selects_expected_tiles(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_gatetrain=True, use_hmote=True)
        model = PrismalWaveModel(cfg)
        self.assertIsNotNone(model.gatetrain_controller)
        assert model.gatetrain_controller is not None

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        signature_family_ids = torch.tensor([[2, 3]], dtype=torch.long)
        route_stats = {
            "selected_path_index": torch.tensor([5], dtype=torch.long),
            "emitter_top_idx": torch.tensor([[[2, 9]]], dtype=torch.long),
            "signature_agreement": torch.tensor(0.88),
            "avg_entropy": torch.tensor(0.33),
        }

        plan = model.gatetrain_controller.plan(
            input_ids=input_ids,
            signature_family_ids=signature_family_ids,
            route_stats=route_stats,
            path_index=5,
            position_index=0,
        )
        record = model.gatetrain_controller.record(route_stats, plan=plan)
        expert_count = len(getattr(getattr(model, "token_hierarchy", None), "experts", [])) or 1

        self.assertEqual(plan.family_ids, (2, 3))
        self.assertEqual(plan.expert_ids, (5 % expert_count,))
        self.assertEqual(plan.emitter_tile_ids, (2, 3, 1, 4))
        self.assertIn("gatetrain_predicted_tiles", record)
        self.assertIn("gatetrain_batch_hit", record)
        self.assertIn("gatetrain_batch_miss", record)

    def test_learned_residency_training_option_a(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_gatetrain=True, use_learned_residency_head=True, use_hmote=True)
        model = PrismalWaveModel(cfg)
        model.train()
        bundle = tokenizer.encode_hierarchy_bundle("Training gate residency smoke.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        labels = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

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

        self.assertTrue(torch.is_tensor(loss))
        self.assertIn("learned_residency_loss", output.route_stats)
        self.assertIn("learned_residency_top_tiles", output.route_stats)
        self.assertEqual(float(output.route_stats["learned_residency_mode"].item()), 0.0)
        self.assertGreaterEqual(float(output.route_stats["learned_residency_loss"].item()), 0.0)

    def test_learned_residency_training_option_b(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(
            tokenizer,
            use_gate=False,
            use_gatetrain=True,
            use_learned_residency_head=True,
            use_residency_with_reinforcement=True,
            use_hmote=True,
        )
        model = PrismalWaveModel(cfg)
        model.train()
        bundle = tokenizer.encode_hierarchy_bundle("Training gate residency smoke.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        labels = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

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

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(float(output.route_stats["learned_residency_mode"].item()), 1.0)
        self.assertIn("learned_residency_reward", output.route_stats)
        self.assertIn("learned_residency_loss", output.route_stats)

    def test_full_gatetrain_planner_residents_all_routed_banks(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_gatetrain=True, use_fullgatetrain=True, use_hmote=True)
        model = PrismalWaveModel(cfg)
        assert model.gatetrain_controller is not None
        controller = model.gatetrain_controller
        plan = controller.plan(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            signature_family_ids=torch.tensor([[2, 3]], dtype=torch.long),
            route_stats={
                "selected_path_index": torch.tensor([5], dtype=torch.long),
                "emitter_top_idx": torch.tensor([[[2, 9]]], dtype=torch.long),
                "signature_agreement": torch.tensor(0.88),
                "avg_entropy": torch.tensor(0.33),
            },
            path_index=5,
            position_index=0,
        )
        record = controller.record(
            {
                "selected_path_index": torch.tensor([5], dtype=torch.long),
                "emitter_top_idx": torch.tensor([[[2, 9]]], dtype=torch.long),
                "signature_agreement": torch.tensor(0.88),
                "avg_entropy": torch.tensor(0.33),
            },
            plan=plan,
        )
        family_count = len(getattr(model, "family_specialists", []))
        expert_count = len(getattr(getattr(model, "token_hierarchy", None), "experts", []))
        self.assertTrue(plan.full_scope)
        self.assertEqual(plan.family_ids, tuple(range(family_count)))
        self.assertEqual(plan.expert_ids, tuple(range(expert_count)))
        self.assertIn("gatetrain_full_scope", record)
        self.assertIn("gatetrain_router_hot", record)
        self.assertIn("gatetrain_torus_hot", record)

    def test_full_gatetrain_planner_counts_tiles_without_router(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(
            tokenizer,
            use_gate=False,
            use_gatetrain=True,
            use_fullgatetrain=True,
            use_hmote=True,
            use_torus_core=True,
        )
        cfg.Torus_SHARC_Router = False
        model = PrismalWaveModel(cfg)
        assert model.gatetrain_controller is not None
        controller = model.gatetrain_controller

        plan = controller.plan(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            signature_family_ids=torch.tensor([[2, 3]], dtype=torch.long),
            route_stats={
                "selected_path_index": torch.tensor([5], dtype=torch.long),
                "emitter_top_idx": torch.tensor([[[2, 9]]], dtype=torch.long),
                "signature_agreement": torch.tensor(0.88),
                "avg_entropy": torch.tensor(0.33),
            },
            path_index=5,
            position_index=0,
        )
        record = controller.record(
            {
                "selected_path_index": torch.tensor([5], dtype=torch.long),
                "emitter_top_idx": torch.tensor([[[2, 9]]], dtype=torch.long),
                "signature_agreement": torch.tensor(0.88),
                "avg_entropy": torch.tensor(0.33),
            },
            plan=plan,
        )

        self.assertTrue(plan.full_scope)
        self.assertGreater(plan.predicted_tiles, 0)
        self.assertGreater(float(record["gatetrain_predicted_tiles"].item()), 0.0)
        self.assertEqual(float(record["gatetrain_hit_count"].item()), 1.0)
        self.assertEqual(float(record["gatetrain_batch_hit"].item()), 1.0)
        self.assertEqual(float(record["gatetrain_batch_miss"].item()), 0.0)

    def test_full_gatetrain_planner_counts_tiles_with_torus_router(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(
            tokenizer,
            use_gate=False,
            use_gatetrain=True,
            use_fullgatetrain=True,
            use_hmote=True,
            use_torus_core=True,
        )
        model = PrismalWaveModel(cfg)
        assert model.gatetrain_controller is not None
        self.assertIsNotNone(model.router)
        controller = model.gatetrain_controller

        plan = controller.plan(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            signature_family_ids=torch.tensor([[2, 3]], dtype=torch.long),
            route_stats={
                "selected_path_index": torch.tensor([5], dtype=torch.long),
                "emitter_top_idx": torch.tensor([[[2, 9]]], dtype=torch.long),
                "signature_agreement": torch.tensor(0.88),
                "avg_entropy": torch.tensor(0.33),
            },
            path_index=5,
            position_index=0,
        )
        record = controller.record(
            {
                "selected_path_index": torch.tensor([5], dtype=torch.long),
                "emitter_top_idx": torch.tensor([[[2, 9]]], dtype=torch.long),
                "signature_agreement": torch.tensor(0.88),
                "avg_entropy": torch.tensor(0.33),
            },
            plan=plan,
        )

        self.assertTrue(plan.full_scope)
        self.assertGreater(plan.predicted_tiles, 0)
        self.assertGreater(float(record["gatetrain_predicted_tiles"].item()), 0.0)
        self.assertEqual(float(record["gatetrain_hit_count"].item()), 1.0)
        self.assertEqual(float(record["gatetrain_batch_hit"].item()), 1.0)
        self.assertEqual(float(record["gatetrain_batch_miss"].item()), 0.0)

    def test_gatetrain_training_records_residency_metrics(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_gate_cfg(tokenizer, use_gate=False, use_gatetrain=True)
        model = PrismalWaveModel(cfg)
        model.train()
        bundle = tokenizer.encode_hierarchy_bundle("Training gate residency smoke.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        labels = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

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

        self.assertTrue(torch.is_tensor(loss))
        self.assertIn("gatetrain_predicted_tiles", output.route_stats)
        self.assertIn("gatetrain_batch_hit", output.route_stats)
        self.assertIn("gatetrain_batch_miss", output.route_stats)
        self.assertGreaterEqual(float(output.route_stats["gatetrain_predicted_tiles"].item()), 0.0)
        self.assertGreaterEqual(float(output.route_stats["gatetrain_confidence"].item()), 0.0)

    def test_gatetrain_train_model_surfaces_metrics(self) -> None:
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
                self.cfg = PrismalWaveConfig(use_gatetrain=True)
                self.cfg.use_fullgatetrain = True
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
                    "gatetrain_hit_count": value,
                    "gatetrain_miss_count": value,
                    "gatetrain_tile_churn": value,
                    "gatetrain_predicted_tiles": value,
                    "gatetrain_confidence": value,
                    "gatetrain_latency_saved_ms": value,
                    "gatetrain_plan_time_ms": value,
                    "gatetrain_lead_time_ms": value,
                    "gatetrain_full_scope": value,
                    "gatetrain_batch_hit": value,
                    "gatetrain_batch_miss": value.new_tensor(0.0),
                }
                return loss, SimpleNamespace(ce_loss=loss.detach(), aux_loss=value, route_stats=route_stats)

        dataset = DummyDataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        model = DummyModel()

        metrics = train_model(
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

        self.assertIn("avg_gatetrain_confidence", metrics)
        self.assertIn("avg_gatetrain_predicted_tiles", metrics)
        self.assertIn("gatetrain_hit_rate", metrics)
        self.assertIn("avg_gatetrain_full_scope", metrics)

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
        cfg.Torus_SHARC_Router = False
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

    def test_emitter_router_casts_slot_cache_to_query_dtype(self) -> None:
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
        cfg.Torus_SHARC_Router = False
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

        model = PrismalWaveModel(cfg)
        router = model.router
        self.assertIsNotNone(router)
        assert router is not None
        router.eval()

        class _CastToBf16(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.to(torch.bfloat16)

        router.slot_query = _CastToBf16()
        hidden = torch.zeros(1, 1, cfg.d_model)
        slots = torch.zeros(1, cfg.n_slots, cfg.d_model, dtype=torch.float32)
        shared_ids = torch.zeros(1, 1, dtype=torch.long)

        _, updated_slots, stats = router.route(
            hidden,
            slots,
            signature_family_ids=shared_ids,
            signature_ids=shared_ids,
            signature_level_ids=shared_ids,
            signature_relation_ids=shared_ids,
            parent_signature_ids=shared_ids,
            path_index=0,
            layer_index=0,
        )

        self.assertEqual(updated_slots.dtype, hidden.dtype)
        self.assertTrue(torch.isfinite(stats["emitter_entropy"]))

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
        cfg.Torus_SHARC_Router = False
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
        cfg.Torus_SHARC_Router = False
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
        self.assertFalse(cfg.use_path_logits)
        cfg.use_path_logits = True
        cfg.use_token_memory_cross_attention = True
        cfg.use_token_memory_generation_cache = True
        cfg.token_memory_window = 6
        cfg.token_memory_top_k = 3
        cfg.token_memory_weight = 0.5
        cfg.token_memory_copy_bias = 1.0
        cfg.token_memory_rare_token_cutoff = 2
        cfg.profile_runtime = True
        cfg.profile_vram = True
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
        self.assertIn("vram_profile_enabled", output.route_stats)
        self.assertIn("vram_encode_alloc_delta_mb", output.route_stats)
        self.assertIn("vram_token_memory_alloc_delta_mb", output.route_stats)
        self.assertIn("vram_path_core_alloc_delta_mb", output.route_stats)
        self.assertGreaterEqual(float(output.route_stats["vram_encode_alloc_delta_mb"].item()), 0.0)
        self.assertGreaterEqual(float(output.route_stats["vram_token_memory_alloc_delta_mb"].item()), 0.0)
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

    def test_torus_single_path_fast_path_matches_explicit_path_index(self) -> None:
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
        self.assertFalse(cfg.use_path_logits)
        cfg.use_path_logits = True
        cfg.use_token_memory_cross_attention = True
        cfg.use_token_memory_generation_cache = True
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32
        cfg.profile_runtime = True

        model = PrismalWaveModel(cfg)
        model.eval()

        bundle = tokenizer.encode_hierarchy_bundle("single path fast path check", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)

        default_output = model(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )
        self.assertEqual(default_output.path_logits.shape[1], 1)
        self.assertEqual(default_output.route_stats["path_weights"].shape[-1], 1)
        torch.testing.assert_close(default_output.route_stats["selected_path_index"], torch.zeros(1, dtype=torch.long))
        self.assertIn("timing_path_aggregate_ms", default_output.route_stats)
        self.assertTrue(torch.isfinite(default_output.route_stats["timing_path_aggregate_ms"]))

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
        cfg.Torus_SHARC_Router = False
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
        cfg.Torus_SHARC_Router = False
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

    def test_mini_overfit_fixture_loads(self) -> None:
        fixture = ROOT / "BaseData" / "mini_overfit" / "mini_overfit.jsonl"
        texts = list(iter_text_corpus(fixture))
        self.assertGreaterEqual(len(texts), 8)
        self.assertTrue(any("cat" in text.lower() for text in texts))

    def test_mini_overfit_training_can_memorize_fixture(self) -> None:
        fixture = ROOT / "BaseData" / "mini_overfit" / "mini_overfit.jsonl"
        tokenizer = PrismalTokenizer()
        train_loader, val_loader = build_train_val_dataloaders(
            fixture,
            tokenizer,
            seq_len=128,
            batch_size=1,
            max_samples=0,
            val_fraction=0.1,
            seed=7,
            streaming=False,
        )

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
        cfg.Torus_SHARC_Router = False
        cfg.use_hmote = False
        cfg.use_recursive_hmoe = False
        cfg.use_signature_lattice_attention = False
        cfg.use_gate = False
        cfg.use_gatetrain = False
        cfg.use_fullgatetrain = False
        cfg.use_learned_residency_head = False
        cfg.use_residency_with_reinforcement = False
        cfg.use_token_memory_cross_attention = False
        cfg.use_token_copy_cross_attention = False
        cfg.use_contrastive_routing = False
        cfg.use_turbo_quantization = False
        cfg.use_bitsandbytes_leaf_precision = False
        cfg.use_speculative_decoding = False
        cfg.use_gradient_checkpointing = False
        cfg.use_gradient_accumulation = False
        cfg.gradient_accumulation_steps = 1
        cfg.dropout = 0.0
        cfg.position_embedding_init_size = 32
        cfg.routing_entropy_weight = 0.0
        cfg.diversity_weight = 0.0
        cfg.torus_active_balance_weight = 0.0
        cfg.emitter_balance_weight = 0.0
        cfg.emitter_mixture_weight = 0.0
        cfg.signature_loss_weight = 0.0
        cfg.signature_level_loss_weight = 0.0
        cfg.signature_relation_loss_weight = 0.0
        cfg.signature_contrastive_weight = 0.0
        cfg.contrastive_routing_weight = 0.0
        cfg.recursive_hmoe_balance_weight = 0.0
        cfg.recursive_hmoe_child_mixture_weight = 0.0
        cfg.recursive_hmoe_agreement_weight = 0.0
        cfg.recursive_aux_weight = 0.0
        cfg.recursive_hidden_mix_scale = 0.0
        cfg.recursive_field_mix_scale = 0.0
        cfg.path_entropy_penalty = 0.0

        model = PrismalWaveModel(cfg)
        metrics = train_model(
            model,
            train_loader,
            torch.device("cpu"),
            cfg=cfg,
            optimizer_name="adamw",
            epochs=0,
            steps=1000,
            lr=2e-3,
            grad_clip=0.0,
            progress=False,
            val_loader=val_loader,
            diagnostic_interval=999,
            use_amp=False,
        )

        self.assertLess(metrics["final_ce_loss"], 0.5)
        self.assertLess(metrics["val_total_loss"], 3.0)

        prompt = "What is a cat?"
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            torch.device("cpu"),
            max_new_tokens=48,
            min_new_tokens=1,
            top_k=0,
            top_p=1.0,
            temperature=0.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            beam_size=2,
            use_speculative_decoding=False,
            template_prompt=False,
        )
        continuation = generated.split("\n\n", 1)[-1].split("\nPrompt token IDs:", 1)[0].strip().lower()
        expected = "A cat is a small furry animal that says meow and likes to chase laser pointers."
        expected_tokens = re.findall(r"[a-z0-9']+", expected.lower())
        generated_tokens = re.findall(r"[a-z0-9']+", continuation)
        overlap = sum(min(Counter(expected_tokens)[tok], Counter(generated_tokens)[tok]) for tok in set(expected_tokens))
        f1 = (2 * overlap) / max(len(expected_tokens) + len(generated_tokens), 1)
        self.assertGreaterEqual(f1, 0.35)
        self.assertIn("meow", continuation)

    def test_nested_learning_optimizer_cadence_and_state_roundtrip(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        group = {
            "params": [param],
            "update_rule": "adamw",
            "parameter_role": "scalar",
            "hierarchy_tier": "global",
            "state_dtype": torch.float32,
            "lr": 0.1,
            "weight_decay": 0.0,
            "betas": (0.0, 0.0),
            "eps": 1e-8,
            "nested_learning_enabled": True,
            "nested_update_interval": 2,
            "nested_update_ema_beta": 0.5,
        }
        optimizer = PrecisionAdaptiveHierarchicalOptimizer([group])

        param.grad = torch.tensor([1.0])
        optimizer.step()
        self.assertAlmostEqual(float(param.item()), 1.0, places=6)
        state = optimizer.state[param]
        self.assertEqual(int(state["nested_window_count"]), 1)
        self.assertTrue(torch.allclose(state["nested_grad_buffer"], torch.tensor([1.0])))

        saved_state = optimizer.state_dict()
        restored_param = torch.nn.Parameter(torch.tensor([1.0]))
        restored = PrecisionAdaptiveHierarchicalOptimizer(
            [
                {
                    "params": [restored_param],
                    "update_rule": "adamw",
                    "parameter_role": "scalar",
                    "hierarchy_tier": "global",
                    "state_dtype": torch.float32,
                    "lr": 0.1,
                    "weight_decay": 0.0,
                    "betas": (0.0, 0.0),
                    "eps": 1e-8,
                    "nested_learning_enabled": True,
                    "nested_update_interval": 2,
                    "nested_update_ema_beta": 0.5,
                }
            ]
        )
        restored.load_state_dict(saved_state)
        restored_state = restored.state[restored_param]
        self.assertEqual(int(restored.param_groups[0]["_nested_group_step"]), 1)
        self.assertEqual(int(restored_state["nested_window_count"]), 1)
        self.assertTrue(torch.allclose(restored_state["nested_grad_buffer"], torch.tensor([1.0])))

        param.grad = torch.tensor([1.0])
        optimizer.step()
        self.assertLess(float(param.item()), 1.0)
        self.assertEqual(int(state["nested_window_count"]), 0)
        self.assertTrue(torch.allclose(state["nested_grad_buffer"], torch.zeros_like(state["nested_grad_buffer"])))

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

    def test_per_group_gradient_clipping_uses_group_caps(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_small_stability_cfg(tokenizer)
        muon_param = torch.nn.Parameter(torch.ones(2, 2))
        scalar_param = torch.nn.Parameter(torch.ones(4))
        rowwise_param = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = torch.optim.SGD(
            [
                {"params": [muon_param], "use_muon": True, "parameter_role": "matrix", "update_rule": "muon", "lr": 1.0},
                {"params": [scalar_param], "parameter_role": "scalar", "update_rule": "adamw", "lr": 1.0},
                {"params": [rowwise_param], "parameter_role": "table", "update_rule": "rowwise", "lr": 1.0},
            ],
            lr=1.0,
        )
        muon_param.grad = torch.full_like(muon_param, 10.0)
        scalar_param.grad = torch.full_like(scalar_param, 10.0)
        rowwise_param.grad = torch.full_like(rowwise_param, 10.0)

        clipped_groups, total_norm = _clip_optimizer_group_gradients(optimizer, cfg, fallback_grad_clip=0.0)

        self.assertEqual(clipped_groups, 3)
        self.assertTrue(math.isfinite(total_norm))
        self.assertLessEqual(float(muon_param.grad.norm().item()), cfg.grad_clip_muon + 1e-5)
        self.assertLessEqual(float(scalar_param.grad.norm().item()), cfg.grad_clip_scalar + 1e-5)
        self.assertLessEqual(float(rowwise_param.grad.norm().item()), cfg.grad_clip_rowwise + 1e-5)

    def test_train_model_skips_nan_loss_batch_and_recovers(self) -> None:
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 2

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
                self.cfg.training_finite_guard_enabled = True
                self.cfg.inference_finite_guard_enabled = True
                self.precision_policy = None
                self.use_turbo_quantization = False
                self._calls = 0

            def set_capacity_growth_locked(self, locked: bool) -> None:
                return None

            def configure_precision(self, *args, **kwargs) -> None:
                return None

            def compute_loss(self, *args, **kwargs):
                self._calls += 1
                value = self.weight.detach().new_tensor(0.5)
                if self._calls == 1:
                    loss = self.weight * torch.tensor(float("nan"), device=self.weight.device)
                else:
                    loss = (self.weight - 2.0).pow(2)
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
                    "stability_nonfinite_repair_count": value.new_tensor(0.0),
                }
                return loss, SimpleNamespace(ce_loss=loss.detach(), aux_loss=value, route_stats=route_stats)

        dataset = DummyDataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        model = DummyModel()

        metrics = train_model(
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

        self.assertEqual(metrics["stability_nonfinite_loss_batches"], 1.0)
        self.assertGreaterEqual(metrics["stability_skipped_optimizer_steps"], 1.0)
        self.assertTrue(math.isfinite(model.weight.item()))
        self.assertNotEqual(model.weight.item(), 1.0)

    def test_generation_guard_recovers_from_nan_logits(self) -> None:
        tokenizer = PrismalTokenizer()
        cfg = self._build_small_stability_cfg(tokenizer)
        model = PrismalWaveModel(cfg)
        model.eval()
        bundle = tokenizer.encode_hierarchy_bundle("Guard the decoder.", add_special_tokens=True)
        input_ids = torch.tensor([bundle.token_ids], dtype=torch.long)
        signature_ids = torch.tensor([bundle.signature_ids], dtype=torch.long)
        signature_level_ids = torch.tensor([bundle.signature_level_ids], dtype=torch.long)
        signature_relation_ids = torch.tensor([bundle.signature_relation_ids], dtype=torch.long)
        parent_signature_ids = torch.tensor([bundle.parent_signature_ids], dtype=torch.long)
        signature_family_ids = torch.tensor([bundle.signature_family_ids], dtype=torch.long)

        original_forward = model.forward

        def noisy_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            output.logits = torch.full_like(output.logits, float("nan"))
            return output

        model.forward = noisy_forward  # type: ignore[assignment]
        generated = model.generate(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            max_new_tokens=2,
            min_new_tokens=1,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            beam_size=1,
            use_speculative_decoding=False,
            token_signature_lookup=tokenizer.signature_lookup_by_token_id(),
            token_family_lookup=tokenizer.signature_family_lookup_by_token_id(),
            token_level_lookup=tokenizer.signature_level_lookup_by_token_id(),
            token_relation_lookup=tokenizer.signature_relation_lookup_by_token_id(),
            suppressed_token_ids=tokenizer.generation_suppressed_token_ids(),
        )

        self.assertGreater(generated.size(1), input_ids.size(1))
        self.assertTrue(torch.isfinite(generated.float()).all())


if __name__ == "__main__":
    unittest.main()
