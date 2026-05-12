"""CLI for the standalone EPIC-SHARC MOHTE prototype."""

# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

try:
    from .config import PrismalWaveConfig
    from .data import ByteTokenizer
    from .model import PrismalWaveModel
    from .train import (
        build_dataloader,
        build_train_val_dataloaders,
        build_tokenizer_from_source,
        generate_text,
        resolve_runtime_config,
        resolve_device,
        load_bundle_from_checkpoint,
        maybe_compile_model,
        run_benchmark,
        save_checkpoint,
        train_model,
    )
except ImportError:  # pragma: no cover - supports direct script launching.
    from config import PrismalWaveConfig
    from data import ByteTokenizer
    from model import PrismalWaveModel
    from train import (
        build_dataloader,
        build_train_val_dataloaders,
        build_tokenizer_from_source,
        generate_text,
        resolve_runtime_config,
        resolve_device,
        load_bundle_from_checkpoint,
        maybe_compile_model,
        run_benchmark,
        save_checkpoint,
        train_model,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="EPIC-SHARC MOHTE",
        description="Standalone Prismal Torus prototype. Quickstart sample data lives in demo/corpus/",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    default_cfg = PrismalWaveConfig()

    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--d-model", type=int, default=default_cfg.d_model)
        p.add_argument("--n-layers", type=int, default=default_cfg.n_layers)
        p.add_argument("--n-emitters", type=int, default=default_cfg.n_emitters)
        p.add_argument("--n-slots", type=int, default=default_cfg.n_slots)
        p.add_argument("--n-paths", type=int, default=default_cfg.n_paths)
        p.add_argument("--top-k-emitters", type=int, default=default_cfg.top_k_emitters)
        p.add_argument("--top-k-slots", type=int, default=default_cfg.top_k_slots)
        p.add_argument("--max-seq-len", type=int, default=default_cfg.max_seq_len)
        p.add_argument("--position-embedding-init-size", type=int, default=default_cfg.position_embedding_init_size)
        p.add_argument("--dropout", type=float, default=default_cfg.dropout)
        p.add_argument("--ff-mult", type=int, default=default_cfg.ff_mult)
        p.add_argument("--no-factorized-embedding", action="store_true")
        p.add_argument("--factorized-embedding-dim", type=int, default=default_cfg.factorized_embedding_dim)
        p.add_argument("--optimizer", type=str, default=None, choices=("adamw", "muon", "hierarchical"))
        p.add_argument("--muon-lr", type=float, default=default_cfg.muon_lr)
        p.add_argument("--muon-weight-decay", type=float, default=default_cfg.muon_weight_decay)
        p.add_argument("--muon-momentum-beta", type=float, default=default_cfg.muon_momentum_beta)
        p.add_argument("--muon-nesterov", dest="muon_nesterov", action="store_true")
        p.add_argument("--no-muon-nesterov", dest="muon_nesterov", action="store_false")
        p.set_defaults(muon_nesterov=default_cfg.muon_nesterov)
        p.add_argument("--muon-ns-steps", type=int, default=default_cfg.muon_ns_steps)
        p.add_argument("--muon-extra-scale-factor", type=float, default=default_cfg.muon_extra_scale_factor)
        p.add_argument("--muon-scalar-optimizer", type=str, default=default_cfg.muon_scalar_optimizer)
        p.add_argument("--max-signature-tokens", type=int, default=0)
        p.add_argument("--max-line-tokens", type=int, default=0)
        p.add_argument("--tokenizer-full-text", action="store_true")
        p.add_argument("--use-turbo-quantization", dest="use_turbo_quantization", action="store_true")
        p.add_argument("--no-turbo-quantization", dest="use_turbo_quantization", action="store_false")
        p.set_defaults(use_turbo_quantization=default_cfg.use_turbo_quantization)
        p.add_argument("--turbo-quantization-bits", type=int, default=default_cfg.turbo_quantization_bits)
        p.add_argument("--turbo-quantization-method", type=str, default=default_cfg.turbo_quantization_method)
        p.add_argument("--use-bitsandbytes-leaf-precision", dest="use_bitsandbytes_leaf_precision", action="store_true")
        p.add_argument("--no-bitsandbytes-leaf-precision", dest="use_bitsandbytes_leaf_precision", action="store_false")
        p.set_defaults(use_bitsandbytes_leaf_precision=default_cfg.use_bitsandbytes_leaf_precision)
        p.add_argument("--bitsandbytes-leaf-precision-mode", type=str, default=default_cfg.bitsandbytes_leaf_precision_mode)
        p.add_argument("--bitsandbytes-leaf-quant-type", type=str, default=default_cfg.bitsandbytes_leaf_quant_type)
        p.add_argument("--bitsandbytes-leaf-compute-dtype", type=str, default=default_cfg.bitsandbytes_leaf_compute_dtype)
        p.add_argument("--quantization-aware-training", dest="quantization_aware_training", action="store_true")
        p.add_argument("--no-quantization-aware-training", dest="quantization_aware_training", action="store_false")
        p.set_defaults(quantization_aware_training=default_cfg.quantization_aware_training)
        p.add_argument("--qat-start-fraction", type=float, default=default_cfg.qat_start_fraction)
        p.add_argument("--qat-ramp-fraction", type=float, default=default_cfg.qat_ramp_fraction)
        p.add_argument("--use-torus-core", dest="use_torus_core", action="store_true")
        p.add_argument("--no-torus-core", dest="use_torus_core", action="store_false")
        p.set_defaults(use_torus_core=default_cfg.use_torus_core)
        p.add_argument("--use-hmote", dest="use_hmote", action="store_true")
        p.add_argument("--no-hmote", dest="use_hmote", action="store_false")
        p.set_defaults(use_hmote=default_cfg.use_hmote)
        p.add_argument("--use-recursive-hmoe", dest="use_recursive_hmoe", action="store_true")
        p.add_argument("--no-recursive-hmoe", dest="use_recursive_hmoe", action="store_false")
        p.set_defaults(use_recursive_hmoe=default_cfg.use_recursive_hmoe)
        p.add_argument("--use-gradient-checkpointing", dest="use_gradient_checkpointing", action="store_true")
        p.add_argument("--no-gradient-checkpointing", dest="use_gradient_checkpointing", action="store_false")
        p.set_defaults(use_gradient_checkpointing=default_cfg.use_gradient_checkpointing)
        p.add_argument("--hmote-depth", type=int, default=default_cfg.hmote_depth)
        p.add_argument("--hmote-branching", type=int, default=default_cfg.hmote_branching)
        p.add_argument("--recursive-hmoe-depth", type=int, default=default_cfg.recursive_hmoe_depth)
        p.add_argument("--recursive-hmoe-branching", type=int, default=default_cfg.recursive_hmoe_branching)
        p.add_argument("--recursive-hmoe-coarse-top-k", type=int, default=default_cfg.recursive_hmoe_coarse_top_k)
        p.add_argument("--recursive-hmoe-fine-top-k", type=int, default=default_cfg.recursive_hmoe_fine_top_k)
        p.add_argument("--recursive-hmoe-child-torus-depth", type=int, default=default_cfg.recursive_hmoe_child_torus_depth)
        p.add_argument("--recursive-hmoe-child-torus-height", type=int, default=default_cfg.recursive_hmoe_child_torus_height)
        p.add_argument("--recursive-hmoe-child-torus-width", type=int, default=default_cfg.recursive_hmoe_child_torus_width)
        p.add_argument("--recursive-hmoe-balance-weight", type=float, default=default_cfg.recursive_hmoe_balance_weight)
        p.add_argument("--recursive-hmoe-child-mixture-weight", type=float, default=default_cfg.recursive_hmoe_child_mixture_weight)
        p.add_argument("--recursive-hmoe-agreement-weight", type=float, default=default_cfg.recursive_hmoe_agreement_weight)
        p.add_argument("--hierarchical-nest-depth", type=int, default=default_cfg.hierarchical_nest_depth)
        p.add_argument("--hierarchical-child-torus-scale", type=float, default=default_cfg.hierarchical_child_torus_scale)
        p.add_argument("--hierarchical-leaf-torus-size", type=int, default=default_cfg.hierarchical_leaf_torus_size)
        p.add_argument("--hierarchical-d-model-scale", type=float, default=default_cfg.hierarchical_d_model_scale)
        p.add_argument("--hierarchical-min-d-model", type=int, default=default_cfg.hierarchical_min_d_model)
        p.add_argument("--hierarchical-level-d-models", type=str, default=default_cfg.hierarchical_level_d_models)
        p.add_argument("--hierarchical-torus-depth-scale", type=float, default=default_cfg.hierarchical_torus_depth_scale)
        p.add_argument("--hierarchical-recursive-depth-scale", type=float, default=default_cfg.hierarchical_recursive_depth_scale)
        p.add_argument("--hierarchical-fixed-point-scale", type=float, default=default_cfg.hierarchical_fixed_point_scale)
        p.add_argument("--hierarchical-precision-enabled", dest="hierarchical_precision_enabled", action="store_true")
        p.add_argument("--no-hierarchical-precision-enabled", dest="hierarchical_precision_enabled", action="store_false")
        p.set_defaults(hierarchical_precision_enabled=default_cfg.hierarchical_precision_enabled)
        p.add_argument("--hierarchical-precision-root-dtype", type=str, default=default_cfg.hierarchical_precision_root_dtype)
        p.add_argument("--hierarchical-precision-mid-dtype", type=str, default=default_cfg.hierarchical_precision_mid_dtype)
        p.add_argument("--hierarchical-precision-leaf-dtype", type=str, default=default_cfg.hierarchical_precision_leaf_dtype)
        p.add_argument("--hierarchical-precision-fallback-dtype", type=str, default=default_cfg.hierarchical_precision_fallback_dtype)
        p.add_argument("--hierarchical-precision-accumulator-dtype", type=str, default=default_cfg.hierarchical_precision_accumulator_dtype)
        p.add_argument("--hierarchical-precision-allow-float8-leaf", dest="hierarchical_precision_allow_float8_leaf", action="store_true")
        p.add_argument("--no-hierarchical-precision-allow-float8-leaf", dest="hierarchical_precision_allow_float8_leaf", action="store_false")
        p.set_defaults(hierarchical_precision_allow_float8_leaf=default_cfg.hierarchical_precision_allow_float8_leaf)
        p.add_argument("--hierarchical-byte-tier", dest="hierarchical_byte_tier", action="store_true")
        p.add_argument("--no-hierarchical-byte-tier", dest="hierarchical_byte_tier", action="store_false")
        p.set_defaults(hierarchical_byte_tier=default_cfg.hierarchical_byte_tier)
        p.add_argument("--per-family-torus-enabled", dest="per_family_torus_enabled", action="store_true")
        p.add_argument("--no-per-family-torus-enabled", dest="per_family_torus_enabled", action="store_false")
        p.set_defaults(per_family_torus_enabled=default_cfg.per_family_torus_enabled)
        p.add_argument("--per-family-torus-scale", type=float, default=default_cfg.per_family_torus_scale)
        p.add_argument("--family-specialist-d-model", type=int, default=default_cfg.family_specialist_d_model)
        p.add_argument("--family-specialist-gate-threshold", type=float, default=default_cfg.family_specialist_gate_threshold)
        p.add_argument("--leaf-cell-enabled", dest="leaf_cell_enabled", action="store_true")
        p.add_argument("--no-leaf-cell-enabled", dest="leaf_cell_enabled", action="store_false")
        p.set_defaults(leaf_cell_enabled=default_cfg.leaf_cell_enabled)
        p.add_argument("--leaf-cell-dim", type=int, default=default_cfg.leaf_cell_dim)
        p.add_argument("--leaf-router-confidence-threshold", type=float, default=default_cfg.leaf_router_confidence_threshold)
        p.add_argument("--max-families-per-nest", type=int, default=default_cfg.max_families_per_nest)
        p.add_argument("--family-specialist-bank-size", type=int, default=default_cfg.family_specialist_bank_size)
        p.add_argument("--use-mixture-of-torus", dest="use_mixture_of_torus", action="store_true")
        p.add_argument("--no-use-mixture-of-torus", dest="use_mixture_of_torus", action="store_false")
        p.set_defaults(use_mixture_of_torus=default_cfg.use_mixture_of_torus)
        p.add_argument("--mot-num-experts", type=int, default=default_cfg.mot_num_experts)
        p.add_argument("--mot-expert-scale", type=float, default=default_cfg.mot_expert_scale)
        p.add_argument("--mot-routing-temperature", type=float, default=default_cfg.mot_routing_temperature)
        p.add_argument("--use-topk-mot", dest="use_topk_mot", action="store_true")
        p.add_argument("--no-topk-mot", dest="use_topk_mot", action="store_false")
        p.set_defaults(use_topk_mot=default_cfg.use_topk_mot)
        p.add_argument("--mot-top-k", type=int, default=default_cfg.mot_top_k)
        p.add_argument("--use-signature-lattice-attention", dest="use_signature_lattice_attention", action="store_true")
        p.add_argument("--no-signature-lattice-attention", dest="use_signature_lattice_attention", action="store_false")
        p.set_defaults(use_signature_lattice_attention=default_cfg.use_signature_lattice_attention)
        p.add_argument("--signature-lattice-dim", type=int, default=default_cfg.signature_lattice_dim)
        p.add_argument("--signature-lattice-buckets", type=int, default=default_cfg.signature_lattice_buckets)
        p.add_argument("--signature-lattice-candidates", type=int, default=default_cfg.signature_lattice_candidates)
        p.add_argument("--signature-lattice-weight", type=float, default=default_cfg.signature_lattice_weight)
        p.add_argument("--signature-lattice-decay", type=float, default=default_cfg.signature_lattice_decay)
        p.add_argument("--signature-lattice-chunk-len", type=int, default=default_cfg.signature_lattice_chunk_len)
        p.add_argument("--use-signature-lattice-generation-cache", dest="use_signature_lattice_generation_cache", action="store_true")
        p.add_argument("--no-signature-lattice-generation-cache", dest="use_signature_lattice_generation_cache", action="store_false")
        p.set_defaults(use_signature_lattice_generation_cache=default_cfg.use_signature_lattice_generation_cache)
        p.add_argument("--use-token-memory-cross-attention", dest="use_token_memory_cross_attention", action="store_true")
        p.add_argument("--no-token-memory-cross-attention", dest="use_token_memory_cross_attention", action="store_false")
        p.set_defaults(use_token_memory_cross_attention=default_cfg.use_token_memory_cross_attention)
        p.add_argument("--use-token-memory-generation-cache", dest="use_token_memory_generation_cache", action="store_true")
        p.add_argument("--no-token-memory-generation-cache", dest="use_token_memory_generation_cache", action="store_false")
        p.set_defaults(use_token_memory_generation_cache=default_cfg.use_token_memory_generation_cache)
        p.add_argument("--token-memory-window", type=int, default=default_cfg.token_memory_window)
        p.add_argument("--token-memory-top-k", type=int, default=default_cfg.token_memory_top_k)
        p.add_argument("--token-memory-weight", type=float, default=default_cfg.token_memory_weight)
        p.add_argument("--token-memory-copy-bias", type=float, default=default_cfg.token_memory_copy_bias)
        p.add_argument("--token-memory-rare-token-cutoff", type=int, default=default_cfg.token_memory_rare_token_cutoff)
        p.add_argument("--token-memory-copy-min-confidence", type=float, default=default_cfg.token_memory_copy_min_confidence)
        p.add_argument("--use-token-copy-cross-attention", dest="use_token_copy_cross_attention", action="store_true")
        p.add_argument("--no-token-copy-cross-attention", dest="use_token_copy_cross_attention", action="store_false")
        p.set_defaults(use_token_copy_cross_attention=default_cfg.use_token_copy_cross_attention)
        p.add_argument("--use-token-copy-generation-cache", dest="use_token_copy_generation_cache", action="store_true")
        p.add_argument("--no-token-copy-generation-cache", dest="use_token_copy_generation_cache", action="store_false")
        p.set_defaults(use_token_copy_generation_cache=default_cfg.use_token_copy_generation_cache)
        p.add_argument("--token-copy-window", type=int, default=default_cfg.token_copy_window)
        p.add_argument("--token-copy-top-k", type=int, default=default_cfg.token_copy_top_k)
        p.add_argument("--token-copy-weight", type=float, default=default_cfg.token_copy_weight)
        p.add_argument("--token-copy-bias-strength", type=float, default=default_cfg.token_copy_bias_strength)
        p.add_argument("--token-copy-rare-token-cutoff", type=int, default=default_cfg.token_copy_rare_token_cutoff)
        p.add_argument("--token-copy-min-confidence", type=float, default=default_cfg.token_copy_min_confidence)
        p.add_argument("--use-pronunciation-signatures", dest="use_pronunciation_signatures", action="store_true")
        p.add_argument("--no-pronunciation-signatures", dest="use_pronunciation_signatures", action="store_false")
        p.set_defaults(use_pronunciation_signatures=default_cfg.use_pronunciation_signatures)
        p.add_argument("--use-speculative-decoding", dest="use_speculative_decoding", action="store_true")
        p.add_argument("--no-speculative-decoding", dest="use_speculative_decoding", action="store_false")
        p.set_defaults(use_speculative_decoding=default_cfg.use_speculative_decoding)
        p.add_argument("--speculative-draft-tokens", type=int, default=default_cfg.speculative_draft_tokens)
        p.add_argument("--speculative-temperature", type=float, default=default_cfg.speculative_temperature)
        p.add_argument("--torus-depth", type=int, default=default_cfg.torus_depth)
        p.add_argument("--torus-height", type=int, default=default_cfg.torus_height)
        p.add_argument("--torus-width", type=int, default=default_cfg.torus_width)
        p.add_argument("--torus-local-field-radius", type=int, default=default_cfg.torus_local_field_radius)
        p.add_argument("--torus-global-bus-slots", type=int, default=default_cfg.torus_global_bus_slots)
        p.add_argument("--torus-global-bus-decay", type=float, default=default_cfg.torus_global_bus_decay)
        p.add_argument("--torus-global-bus-write-scale", type=float, default=default_cfg.torus_global_bus_write_scale)
        p.add_argument("--torus-write-radius", type=int, default=default_cfg.torus_write_radius)
        p.add_argument("--torus-scout-read-radius", type=int, default=default_cfg.torus_scout_read_radius)
        p.add_argument("--torus-read-radius", type=int, dest="torus_scout_read_radius", help=argparse.SUPPRESS)
        p.add_argument("--torus-transport", type=float, default=default_cfg.torus_transport)
        p.add_argument("--torus-write-strength", type=float, default=default_cfg.torus_write_strength)
        p.add_argument("--torus-relay-write-radius", type=int, default=default_cfg.torus_relay_write_radius)
        p.add_argument("--torus-inner-temperature", type=float, default=default_cfg.torus_inner_temperature)
        p.add_argument("--torus-outer-temperature", type=float, default=default_cfg.torus_outer_temperature)
        p.add_argument("--torus-relay-interval", type=int, default=default_cfg.torus_relay_interval)
        p.add_argument("--torus-transport-interval", type=int, default=default_cfg.torus_transport_interval)
        p.add_argument("--torus-primary-temperature", type=float, default=default_cfg.torus_primary_temperature)
        p.add_argument("--torus-relay-temperature-1", type=float, default=default_cfg.torus_relay_temperature_1)
        p.add_argument("--torus-relay-temperature-2", type=float, default=default_cfg.torus_relay_temperature_2)
        p.add_argument("--torus-relay-temperature-3", type=float, default=default_cfg.torus_relay_temperature_3)
        p.add_argument("--torus-activity-threshold", type=float, default=default_cfg.torus_activity_threshold)
        p.add_argument("--torus-active-target-fraction", type=float, default=default_cfg.torus_active_target_fraction)
        p.add_argument("--torus-active-balance-weight", type=float, default=default_cfg.torus_active_balance_weight)
        p.add_argument("--use-torus-race-lanes", dest="use_torus_race_lanes", action="store_true")
        p.add_argument("--no-torus-race-lanes", dest="use_torus_race_lanes", action="store_false")
        p.set_defaults(use_torus_race_lanes=default_cfg.use_torus_race_lanes)
        p.add_argument("--torus-lane-count", type=int, default=default_cfg.torus_lane_count)
        p.add_argument("--torus-scout-density", type=float, default=default_cfg.torus_scout_density)
        p.add_argument("--torus-lane-select-threshold-1", type=float, default=default_cfg.torus_lane_select_threshold_1)
        p.add_argument("--torus-lane-select-threshold-2", type=float, default=default_cfg.torus_lane_select_threshold_2)
        p.add_argument("--torus-lane-relay-hop-spacing", type=int, default=default_cfg.torus_lane_relay_hop_spacing)
        p.add_argument("--generation-lap-cap", type=int, default=default_cfg.generation_lap_cap)
        p.add_argument("--generation-lap-token-cap", type=int, default=default_cfg.generation_lap_token_cap)
        p.add_argument("--torus-chunk-len", type=int, default=default_cfg.torus_chunk_len)
        p.add_argument("--use-fixed-point-solver", dest="use_fixed_point_solver", action="store_true")
        p.add_argument("--no-fixed-point-solver", dest="use_fixed_point_solver", action="store_false")
        p.set_defaults(use_fixed_point_solver=default_cfg.use_fixed_point_solver)
        p.add_argument("--use-chunk-solver-training", dest="use_chunk_solver_training", action="store_true")
        p.add_argument("--no-chunk-solver-training", dest="use_chunk_solver_training", action="store_false")
        p.set_defaults(use_chunk_solver_training=default_cfg.use_chunk_solver_training)
        p.add_argument("--chunk-solver-training-iterations", type=int, default=default_cfg.chunk_solver_training_iterations)
        p.add_argument("--chunk-solver-training-relaxation", type=float, default=default_cfg.chunk_solver_training_relaxation)
        p.add_argument("--chunk-solver-training-substeps", type=int, default=default_cfg.chunk_solver_training_substeps)
        p.add_argument("--chunk-solver-training-audit-every", type=int, default=default_cfg.chunk_solver_training_audit_every)
        p.add_argument("--fixed-point-iterations", type=int, default=default_cfg.fixed_point_iterations)
        p.add_argument("--fixed-point-tolerance", type=float, default=default_cfg.fixed_point_tolerance)
        p.add_argument("--fixed-point-relaxation", type=float, default=default_cfg.fixed_point_relaxation)
        p.add_argument("--signature-vocab-size", type=int, default=default_cfg.signature_vocab_size)
        p.add_argument("--signature-level-vocab-size", type=int, default=default_cfg.signature_level_vocab_size)
        p.add_argument("--signature-relation-vocab-size", type=int, default=default_cfg.signature_relation_vocab_size)
        p.add_argument("--signature-bucket-vocab-size", type=int, default=default_cfg.signature_bucket_vocab_size)
        p.add_argument("--emitter-family-share", type=float, default=default_cfg.emitter_family_share)
        p.add_argument("--emitter-level-share", type=float, default=default_cfg.emitter_level_share)
        p.add_argument("--emitter-relation-share", type=float, default=default_cfg.emitter_relation_share)
        p.add_argument("--emitter-parent-share", type=float, default=default_cfg.emitter_parent_share)
        p.add_argument("--emitter-balance-weight", type=float, default=default_cfg.emitter_balance_weight)
        p.add_argument("--emitter-mixture-target-count", type=float, default=default_cfg.emitter_mixture_target_count)
        p.add_argument("--emitter-mixture-weight", type=float, default=default_cfg.emitter_mixture_weight)
        p.add_argument("--emitter-birth-threshold", type=float, default=default_cfg.emitter_birth_threshold)
        p.add_argument("--emitter-promotion-threshold", type=float, default=default_cfg.emitter_promotion_threshold)
        p.add_argument("--torus-write-family-floor", type=float, default=default_cfg.torus_write_family_floor)
        p.add_argument("--torus-read-family-floor", type=float, default=default_cfg.torus_read_family_floor)
        p.add_argument("--profile-runtime", dest="profile_runtime", action="store_true")
        p.add_argument("--no-profile-runtime", dest="profile_runtime", action="store_false")
        p.set_defaults(profile_runtime=default_cfg.profile_runtime)

    train_p = sub.add_parser("train", help="Train on a text corpus")
    train_p.add_argument(
        "--data",
        required=True,
        help="File or directory of txt/jsonl/parquet data; try demo/corpus/ for the shipped quickstart sample",
    )
    train_p.add_argument("--save-dir", required=True)
    train_p.add_argument("--resume-checkpoint", default="", help="Path to an existing model.pt to continue training from")
    train_p.add_argument("--continue-checkpoint", default="", help="Path to an existing model.pt to initialize weights from without resuming optimizer state")
    train_p.add_argument("--steps", type=int, default=0)
    train_p.add_argument("--epochs", type=int, default=0)
    train_p.add_argument("--minutes", type=float, default=None)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--seq-len", type=int, default=0)
    train_p.add_argument("--lr", type=float, default=default_cfg.lr)
    train_p.add_argument("--max-samples", type=int, default=default_cfg.max_samples)
    train_p.add_argument("--max-new-tokens", type=int, default=0)
    train_p.add_argument("--min-token-frequency", type=int, default=2)
    train_p.add_argument("--dataset-streaming", dest="dataset_streaming", action="store_true")
    train_p.add_argument("--no-dataset-streaming", dest="dataset_streaming", action="store_false")
    train_p.set_defaults(dataset_streaming=True)
    train_p.add_argument(
        "--tokenizer-cache-dir",
        default="",
        help="Optional directory for tokenizer cache files; defaults to the system temp cache",
    )
    train_p.add_argument("--tokenizer-workers", type=int, default=1, help="Worker count for tokenizer bootstrapping")
    train_p.add_argument("--post-prompt", default="", help="Prompt to generate after training")
    train_p.add_argument("--post-output", default="", help="Where to save the generated output txt")
    train_p.add_argument("--post-max-new-tokens", type=int, default=80)
    train_p.add_argument("--post-min-new-tokens", type=int, default=1)
    train_p.add_argument("--post-temperature", type=float, default=0.9)
    train_p.add_argument("--post-top-k", type=int, default=8)
    train_p.add_argument("--post-top-p", type=float, default=0.92)
    train_p.add_argument("--post-repetition-penalty", type=float, default=1.1)
    train_p.add_argument("--post-no-repeat-ngram-size", type=int, default=4)
    train_p.add_argument("--post-beam-size", type=int, default=1)
    train_p.add_argument(
        "--post-template-prompt",
        dest="post_template_prompt",
        action="store_true",
        help='Wrap only the post-run generation prompt as "Instruction: ...\\nResponse: ".',
    )
    train_p.add_argument("--post-no-template-prompt", dest="post_template_prompt", action="store_false")
    train_p.set_defaults(post_template_prompt=False)
    train_p.add_argument("--torch-compile", dest="torch_compile", action="store_true")
    train_p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    train_p.set_defaults(torch_compile=True)
    train_p.add_argument("--amp", dest="amp", action="store_true")
    train_p.add_argument("--no-amp", dest="amp", action="store_false")
    train_p.set_defaults(amp=True)
    train_p.add_argument("--val-fraction", type=float, default=0.1)
    train_p.add_argument("--diagnostic-interval", type=int, default=20)
    add_model_args(train_p)

    infer_p = sub.add_parser("infer", help="Generate from a checkpoint")
    infer_p.add_argument("--checkpoint", required=True)
    infer_p.add_argument("--prompt", required=True)
    infer_p.add_argument("--max-new-tokens", type=int, default=64)
    infer_p.add_argument("--min-new-tokens", type=int, default=1)
    infer_p.add_argument("--temperature", type=float, default=0.9)
    infer_p.add_argument("--top-k", type=int, default=8)
    infer_p.add_argument("--top-p", type=float, default=0.92)
    infer_p.add_argument("--repetition-penalty", type=float, default=1.1)
    infer_p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    infer_p.add_argument("--beam-size", type=int, default=1)
    infer_p.add_argument("--use-speculative-decoding", dest="use_speculative_decoding", action="store_true")
    infer_p.add_argument("--no-speculative-decoding", dest="use_speculative_decoding", action="store_false")
    infer_p.set_defaults(use_speculative_decoding=default_cfg.use_speculative_decoding)
    infer_p.add_argument("--speculative-draft-tokens", type=int, default=default_cfg.speculative_draft_tokens)
    infer_p.add_argument("--speculative-temperature", type=float, default=default_cfg.speculative_temperature)
    infer_p.add_argument(
        "--template-prompt",
        dest="template_prompt",
        action="store_true",
        help='Wrap plain inference prompts as "Instruction: ...\\nResponse: ".',
    )
    infer_p.add_argument("--no-template-prompt", dest="template_prompt", action="store_false")
    infer_p.set_defaults(template_prompt=False)
    infer_p.add_argument("--torch-compile", dest="torch_compile", action="store_true")
    infer_p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    infer_p.set_defaults(torch_compile=True)

    bench_p = sub.add_parser("benchmark", help="Benchmark training/inference throughput")
    bench_p.add_argument(
        "--data",
        required=True,
        help="File or directory of txt/jsonl/parquet data; try demo/corpus/ for the shipped quickstart sample",
    )
    bench_p.add_argument("--steps", type=int, default=16)
    bench_p.add_argument("--batch-size", type=int, default=8)
    bench_p.add_argument("--seq-len", type=int, default=128)
    bench_p.add_argument("--max-samples", type=int, default=default_cfg.max_samples)
    bench_p.add_argument("--max-new-tokens", type=int, default=0)
    bench_p.add_argument("--min-token-frequency", type=int, default=2)
    bench_p.add_argument(
        "--tokenizer-cache-dir",
        default="",
        help="Optional directory for tokenizer cache files; defaults to the system temp cache",
    )
    bench_p.add_argument("--torch-compile", dest="torch_compile", action="store_true")
    bench_p.add_argument("--no-torch-compile", dest="torch_compile", action="store_false")
    bench_p.set_defaults(torch_compile=True)
    add_model_args(bench_p)

    gui_p = sub.add_parser("gui", help="Launch the Tkinter trainer UI")

    return parser


def _build_config(args: argparse.Namespace, tokenizer: ByteTokenizer | None = None) -> PrismalWaveConfig:
    default_cfg = PrismalWaveConfig()
    cfg = PrismalWaveConfig(
        base_vocab_size=getattr(tokenizer, "base_vocab_size", PrismalWaveConfig.base_vocab_size),
        vocab_size=0,
        max_samples=getattr(args, "max_samples", PrismalWaveConfig.max_samples),
        lr=getattr(args, "lr", PrismalWaveConfig.lr),
        optimizer=getattr(args, "optimizer", None) if getattr(args, "optimizer", None) is not None else default_cfg.optimizer,
        muon_lr=getattr(args, "muon_lr", default_cfg.muon_lr),
        muon_weight_decay=getattr(args, "muon_weight_decay", default_cfg.muon_weight_decay),
        muon_momentum_beta=getattr(args, "muon_momentum_beta", default_cfg.muon_momentum_beta),
        muon_nesterov=getattr(args, "muon_nesterov", default_cfg.muon_nesterov),
        muon_ns_steps=getattr(args, "muon_ns_steps", default_cfg.muon_ns_steps),
        muon_extra_scale_factor=getattr(args, "muon_extra_scale_factor", default_cfg.muon_extra_scale_factor),
        muon_scalar_optimizer=getattr(args, "muon_scalar_optimizer", default_cfg.muon_scalar_optimizer),
        signature_vocab_size=getattr(args, "signature_vocab_size", 0),
        signature_level_vocab_size=getattr(args, "signature_level_vocab_size", 0),
        signature_relation_vocab_size=getattr(args, "signature_relation_vocab_size", 0),
        signature_bucket_vocab_size=getattr(args, "signature_bucket_vocab_size", 0),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_emitters=args.n_emitters,
        n_slots=args.n_slots,
        n_paths=args.n_paths,
        top_k_emitters=args.top_k_emitters,
        top_k_slots=args.top_k_slots,
        max_seq_len=args.max_seq_len,
        position_embedding_init_size=args.position_embedding_init_size,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
        use_factorized_embedding=not args.no_factorized_embedding,
        factorized_embedding_dim=args.factorized_embedding_dim,
        use_turbo_quantization=args.use_turbo_quantization,
        turbo_quantization_bits=args.turbo_quantization_bits,
        turbo_quantization_method=args.turbo_quantization_method,
        use_bitsandbytes_leaf_precision=args.use_bitsandbytes_leaf_precision,
        bitsandbytes_leaf_precision_mode=args.bitsandbytes_leaf_precision_mode,
        bitsandbytes_leaf_quant_type=args.bitsandbytes_leaf_quant_type,
        bitsandbytes_leaf_compute_dtype=args.bitsandbytes_leaf_compute_dtype,
        quantization_aware_training=args.quantization_aware_training,
        qat_start_fraction=args.qat_start_fraction,
        qat_ramp_fraction=args.qat_ramp_fraction,
        use_torus_core=args.use_torus_core,
        use_hmote=getattr(args, "use_hmote", default_cfg.use_hmote),
        use_recursive_hmoe=args.use_recursive_hmoe,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        hmote_depth=getattr(args, "hmote_depth", default_cfg.hmote_depth),
        hmote_branching=getattr(args, "hmote_branching", default_cfg.hmote_branching),
        recursive_hmoe_depth=args.recursive_hmoe_depth,
        recursive_hmoe_branching=args.recursive_hmoe_branching,
        recursive_hmoe_coarse_top_k=args.recursive_hmoe_coarse_top_k,
        recursive_hmoe_fine_top_k=args.recursive_hmoe_fine_top_k,
        recursive_hmoe_child_torus_depth=args.recursive_hmoe_child_torus_depth,
        recursive_hmoe_child_torus_height=args.recursive_hmoe_child_torus_height,
        recursive_hmoe_child_torus_width=args.recursive_hmoe_child_torus_width,
        recursive_hmoe_balance_weight=args.recursive_hmoe_balance_weight,
        recursive_hmoe_child_mixture_weight=args.recursive_hmoe_child_mixture_weight,
        recursive_hmoe_agreement_weight=args.recursive_hmoe_agreement_weight,
        hierarchical_nest_depth=args.hierarchical_nest_depth if hasattr(args, "hierarchical_nest_depth") else default_cfg.hierarchical_nest_depth,
        hierarchical_child_torus_scale=args.hierarchical_child_torus_scale if hasattr(args, "hierarchical_child_torus_scale") else default_cfg.hierarchical_child_torus_scale,
        hierarchical_leaf_torus_size=args.hierarchical_leaf_torus_size if hasattr(args, "hierarchical_leaf_torus_size") else default_cfg.hierarchical_leaf_torus_size,
        hierarchical_d_model_scale=getattr(args, "hierarchical_d_model_scale", default_cfg.hierarchical_d_model_scale),
        hierarchical_min_d_model=getattr(args, "hierarchical_min_d_model", default_cfg.hierarchical_min_d_model),
        hierarchical_level_d_models=getattr(args, "hierarchical_level_d_models", default_cfg.hierarchical_level_d_models),
        hierarchical_torus_depth_scale=getattr(args, "hierarchical_torus_depth_scale", default_cfg.hierarchical_torus_depth_scale),
        hierarchical_recursive_depth_scale=getattr(args, "hierarchical_recursive_depth_scale", default_cfg.hierarchical_recursive_depth_scale),
        hierarchical_fixed_point_scale=getattr(args, "hierarchical_fixed_point_scale", default_cfg.hierarchical_fixed_point_scale),
        hierarchical_precision_enabled=getattr(args, "hierarchical_precision_enabled", default_cfg.hierarchical_precision_enabled),
        hierarchical_precision_root_dtype=getattr(args, "hierarchical_precision_root_dtype", default_cfg.hierarchical_precision_root_dtype),
        hierarchical_precision_mid_dtype=getattr(args, "hierarchical_precision_mid_dtype", default_cfg.hierarchical_precision_mid_dtype),
        hierarchical_precision_leaf_dtype=getattr(args, "hierarchical_precision_leaf_dtype", default_cfg.hierarchical_precision_leaf_dtype),
        hierarchical_precision_fallback_dtype=getattr(args, "hierarchical_precision_fallback_dtype", default_cfg.hierarchical_precision_fallback_dtype),
        hierarchical_precision_accumulator_dtype=getattr(args, "hierarchical_precision_accumulator_dtype", default_cfg.hierarchical_precision_accumulator_dtype),
        hierarchical_precision_allow_float8_leaf=getattr(args, "hierarchical_precision_allow_float8_leaf", default_cfg.hierarchical_precision_allow_float8_leaf),
        hierarchical_byte_tier=getattr(args, "hierarchical_byte_tier", default_cfg.hierarchical_byte_tier),
        per_family_torus_enabled=getattr(args, "per_family_torus_enabled", default_cfg.per_family_torus_enabled),
        per_family_torus_scale=getattr(args, "per_family_torus_scale", default_cfg.per_family_torus_scale),
        family_specialist_d_model=getattr(args, "family_specialist_d_model", default_cfg.family_specialist_d_model),
        family_specialist_gate_threshold=getattr(args, "family_specialist_gate_threshold", default_cfg.family_specialist_gate_threshold),
        leaf_cell_enabled=getattr(args, "leaf_cell_enabled", default_cfg.leaf_cell_enabled),
        leaf_cell_dim=getattr(args, "leaf_cell_dim", default_cfg.leaf_cell_dim),
        leaf_router_confidence_threshold=getattr(args, "leaf_router_confidence_threshold", default_cfg.leaf_router_confidence_threshold),
        max_families_per_nest=getattr(args, "max_families_per_nest", default_cfg.max_families_per_nest),
        family_specialist_bank_size=getattr(args, "family_specialist_bank_size", default_cfg.family_specialist_bank_size),
        use_mixture_of_torus=getattr(args, "use_mixture_of_torus", default_cfg.use_mixture_of_torus),
        mot_num_experts=getattr(args, "mot_num_experts", default_cfg.mot_num_experts),
        mot_expert_scale=getattr(args, "mot_expert_scale", default_cfg.mot_expert_scale),
        mot_routing_temperature=getattr(args, "mot_routing_temperature", default_cfg.mot_routing_temperature),
        use_topk_mot=getattr(args, "use_topk_mot", default_cfg.use_topk_mot),
        mot_top_k=getattr(args, "mot_top_k", default_cfg.mot_top_k),
        use_signature_lattice_attention=getattr(args, "use_signature_lattice_attention", default_cfg.use_signature_lattice_attention),
        signature_lattice_dim=getattr(args, "signature_lattice_dim", default_cfg.signature_lattice_dim),
        signature_lattice_buckets=getattr(args, "signature_lattice_buckets", default_cfg.signature_lattice_buckets),
        signature_lattice_candidates=getattr(args, "signature_lattice_candidates", default_cfg.signature_lattice_candidates),
        signature_lattice_weight=getattr(args, "signature_lattice_weight", default_cfg.signature_lattice_weight),
        signature_lattice_decay=getattr(args, "signature_lattice_decay", default_cfg.signature_lattice_decay),
        signature_lattice_chunk_len=getattr(args, "signature_lattice_chunk_len", default_cfg.signature_lattice_chunk_len),
        use_signature_lattice_generation_cache=getattr(args, "use_signature_lattice_generation_cache", default_cfg.use_signature_lattice_generation_cache),
        use_token_memory_cross_attention=getattr(args, "use_token_memory_cross_attention", default_cfg.use_token_memory_cross_attention),
        use_token_memory_generation_cache=getattr(args, "use_token_memory_generation_cache", default_cfg.use_token_memory_generation_cache),
        token_memory_window=getattr(args, "token_memory_window", default_cfg.token_memory_window),
        token_memory_top_k=getattr(args, "token_memory_top_k", default_cfg.token_memory_top_k),
        token_memory_weight=getattr(args, "token_memory_weight", default_cfg.token_memory_weight),
        token_memory_copy_bias=getattr(args, "token_memory_copy_bias", default_cfg.token_memory_copy_bias),
        token_memory_rare_token_cutoff=getattr(args, "token_memory_rare_token_cutoff", default_cfg.token_memory_rare_token_cutoff),
        token_memory_copy_min_confidence=getattr(args, "token_memory_copy_min_confidence", default_cfg.token_memory_copy_min_confidence),
        use_token_copy_cross_attention=getattr(args, "use_token_copy_cross_attention", default_cfg.use_token_copy_cross_attention),
        use_token_copy_generation_cache=getattr(args, "use_token_copy_generation_cache", default_cfg.use_token_copy_generation_cache),
        token_copy_window=getattr(args, "token_copy_window", default_cfg.token_copy_window),
        token_copy_top_k=getattr(args, "token_copy_top_k", default_cfg.token_copy_top_k),
        token_copy_weight=getattr(args, "token_copy_weight", default_cfg.token_copy_weight),
        token_copy_bias_strength=getattr(args, "token_copy_bias_strength", default_cfg.token_copy_bias_strength),
        token_copy_rare_token_cutoff=getattr(args, "token_copy_rare_token_cutoff", default_cfg.token_copy_rare_token_cutoff),
        token_copy_min_confidence=getattr(args, "token_copy_min_confidence", default_cfg.token_copy_min_confidence),
        use_pronunciation_signatures=getattr(args, "use_pronunciation_signatures", default_cfg.use_pronunciation_signatures),
        use_speculative_decoding=args.use_speculative_decoding,
        speculative_draft_tokens=args.speculative_draft_tokens,
        speculative_temperature=args.speculative_temperature,
        torus_depth=args.torus_depth,
        torus_height=args.torus_height,
        torus_width=args.torus_width,
        torus_local_field_radius=args.torus_local_field_radius,
        torus_global_bus_slots=args.torus_global_bus_slots,
        torus_global_bus_decay=args.torus_global_bus_decay,
        torus_global_bus_write_scale=args.torus_global_bus_write_scale,
        torus_write_radius=args.torus_write_radius,
        torus_scout_read_radius=args.torus_scout_read_radius,
        torus_transport=args.torus_transport,
        torus_write_strength=args.torus_write_strength,
        torus_relay_write_radius=args.torus_relay_write_radius,
        torus_inner_temperature=args.torus_inner_temperature,
        torus_outer_temperature=args.torus_outer_temperature,
        torus_relay_interval=args.torus_relay_interval,
        torus_transport_interval=args.torus_transport_interval,
        torus_primary_temperature=args.torus_primary_temperature,
        torus_relay_temperature_1=args.torus_relay_temperature_1,
        torus_relay_temperature_2=args.torus_relay_temperature_2,
        torus_relay_temperature_3=args.torus_relay_temperature_3,
        torus_activity_threshold=args.torus_activity_threshold,
        torus_active_target_fraction=args.torus_active_target_fraction,
        torus_active_balance_weight=args.torus_active_balance_weight,
        use_torus_race_lanes=args.use_torus_race_lanes,
        torus_lane_count=args.torus_lane_count,
        torus_scout_density=args.torus_scout_density,
        torus_lane_select_threshold_1=args.torus_lane_select_threshold_1,
        torus_lane_select_threshold_2=args.torus_lane_select_threshold_2,
        torus_lane_relay_hop_spacing=args.torus_lane_relay_hop_spacing,
        generation_lap_cap=args.generation_lap_cap,
        generation_lap_token_cap=args.generation_lap_token_cap,
        torus_chunk_len=args.torus_chunk_len,
        use_fixed_point_solver=args.use_fixed_point_solver,
        use_chunk_solver_training=args.use_chunk_solver_training,
        chunk_solver_training_iterations=args.chunk_solver_training_iterations,
        chunk_solver_training_relaxation=args.chunk_solver_training_relaxation,
        chunk_solver_training_substeps=args.chunk_solver_training_substeps,
        chunk_solver_training_audit_every=args.chunk_solver_training_audit_every,
        fixed_point_iterations=args.fixed_point_iterations,
        fixed_point_tolerance=args.fixed_point_tolerance,
        fixed_point_relaxation=args.fixed_point_relaxation,
        emitter_family_share=args.emitter_family_share,
        emitter_level_share=args.emitter_level_share,
        emitter_relation_share=args.emitter_relation_share,
        emitter_parent_share=args.emitter_parent_share,
        emitter_balance_weight=args.emitter_balance_weight,
        emitter_mixture_target_count=args.emitter_mixture_target_count,
        emitter_mixture_weight=args.emitter_mixture_weight,
        emitter_birth_threshold=args.emitter_birth_threshold,
        emitter_promotion_threshold=args.emitter_promotion_threshold,
        torus_write_family_floor=args.torus_write_family_floor,
        torus_read_family_floor=args.torus_read_family_floor,
        profile_runtime=args.profile_runtime,
    )
    return cfg


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    default_cfg = PrismalWaveConfig()
    if (
        hasattr(args, "torus_local_field_radius")
        and args.torus_local_field_radius == default_cfg.torus_local_field_radius
        and getattr(args, "torus_write_radius", default_cfg.torus_write_radius) != default_cfg.torus_write_radius
    ):
        args.torus_local_field_radius = args.torus_write_radius
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = ByteTokenizer()

    if args.command == "infer":
        model, tokenizer, _raw_cfg = load_bundle_from_checkpoint(args.checkpoint, device=device)
        # load_bundle_from_checkpoint already resolves the runtime config.
        model = model.to(device)
        model = maybe_compile_model(model, enabled=args.torch_compile)
        text = generate_text(
            model,
            tokenizer,
            args.prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            beam_size=args.beam_size,
            use_speculative_decoding=args.use_speculative_decoding,
            speculative_draft_tokens=args.speculative_draft_tokens,
            speculative_temperature=args.speculative_temperature,
            template_prompt=args.template_prompt,
        )
        print(text)
        return 0

    if args.command == "gui":
        try:
            from .gui import launch_gui
        except ImportError:  # pragma: no cover - supports direct script launching.
            from gui import launch_gui

        launch_gui()
        return 0

    if args.command == "train":
        continue_checkpoint = args.continue_checkpoint.strip()
        resume_checkpoint = args.resume_checkpoint.strip()
        if continue_checkpoint and resume_checkpoint:
            raise ValueError("Use only one of --resume-checkpoint or --continue-checkpoint.")
        if continue_checkpoint:
            model, tokenizer, raw_cfg = load_bundle_from_checkpoint(continue_checkpoint, device=device, load_training_state=False)
            model = model.to(device)
            tokenizer = build_tokenizer_from_source(
                args.data,
                max_new_tokens=args.max_new_tokens,
                min_frequency=args.min_token_frequency,
                max_line_tokens=args.max_line_tokens,
                max_signature_tokens=args.max_signature_tokens,
                max_source_samples=args.max_samples,
                supervised_only=not args.tokenizer_full_text,
                use_pronunciation_signatures=args.use_pronunciation_signatures,
                tokenizer_workers=args.tokenizer_workers,
                tokenizer_cache_dir=getattr(args, "tokenizer_cache_dir", ""),
                tokenizer=tokenizer,
            )
            if tokenizer.vocab_size > model.vocab_size:
                print(
                    f"[Prismal] expanding checkpoint vocab from {model.vocab_size} to {tokenizer.vocab_size} "
                    "to match the continued tokenizer",
                    flush=True,
                )
                model.resize_vocab(tokenizer.vocab_size)
            raw_cfg.lr = getattr(args, "lr", raw_cfg.lr)
            if getattr(args, "optimizer", None) is not None:
                raw_cfg.optimizer = getattr(args, "optimizer", raw_cfg.optimizer)
            raw_cfg.muon_lr = getattr(args, "muon_lr", raw_cfg.muon_lr)
            raw_cfg.muon_weight_decay = getattr(args, "muon_weight_decay", raw_cfg.muon_weight_decay)
            raw_cfg.muon_momentum_beta = getattr(args, "muon_momentum_beta", raw_cfg.muon_momentum_beta)
            raw_cfg.muon_nesterov = getattr(args, "muon_nesterov", raw_cfg.muon_nesterov)
            raw_cfg.muon_ns_steps = getattr(args, "muon_ns_steps", raw_cfg.muon_ns_steps)
            raw_cfg.muon_extra_scale_factor = getattr(args, "muon_extra_scale_factor", raw_cfg.muon_extra_scale_factor)
            raw_cfg.muon_scalar_optimizer = getattr(args, "muon_scalar_optimizer", raw_cfg.muon_scalar_optimizer)
            model = maybe_compile_model(model, enabled=args.torch_compile)
            print(
                "[Prismal] initialized checkpoint weights "
                f"d={model.cfg.d_model} torus={model.cfg.torus_depth}x{model.cfg.torus_height}x{model.cfg.torus_width} "
                f"emitters={model.cfg.n_emitters} slots={model.cfg.n_slots} vocab={model.cfg.vocab_size}",
                flush=True,
            )
            arch_fields = (
                "d_model",
                "n_layers",
                "n_emitters",
                "n_slots",
                "n_paths",
                "top_k_emitters",
                "top_k_slots",
                "torus_depth",
                "torus_height",
                "torus_width",
            )
            ignored_arch_overrides = [
                f"{name}={getattr(args, name)} -> checkpoint {getattr(model.cfg, name)}"
                for name in arch_fields
                if hasattr(args, name) and getattr(args, name) != getattr(model.cfg, name)
            ]
            if ignored_arch_overrides:
                print(
                    "[Prismal] continue keeps checkpoint architecture; ignored requested architecture values: "
                    + ", ".join(ignored_arch_overrides),
                    flush=True,
                )
        elif resume_checkpoint:
            model, tokenizer, raw_cfg = load_bundle_from_checkpoint(resume_checkpoint, device=device)
            model = model.to(device)
            tokenizer = build_tokenizer_from_source(
                args.data,
                max_new_tokens=args.max_new_tokens,
                min_frequency=args.min_token_frequency,
                max_line_tokens=args.max_line_tokens,
                max_signature_tokens=args.max_signature_tokens,
                max_source_samples=args.max_samples,
                supervised_only=not args.tokenizer_full_text,
                use_pronunciation_signatures=args.use_pronunciation_signatures,
                tokenizer_workers=args.tokenizer_workers,
                tokenizer_cache_dir=getattr(args, "tokenizer_cache_dir", ""),
                tokenizer=tokenizer,
            )
            if tokenizer.vocab_size > model.vocab_size:
                print(
                    f"[Prismal] expanding checkpoint vocab from {model.vocab_size} to {tokenizer.vocab_size} "
                    "to match the resumed tokenizer",
                    flush=True,
                )
                model.resize_vocab(tokenizer.vocab_size)
            raw_cfg.lr = getattr(args, "lr", raw_cfg.lr)
            if getattr(args, "optimizer", None) is not None:
                raw_cfg.optimizer = getattr(args, "optimizer", raw_cfg.optimizer)
            raw_cfg.muon_lr = getattr(args, "muon_lr", raw_cfg.muon_lr)
            raw_cfg.muon_weight_decay = getattr(args, "muon_weight_decay", raw_cfg.muon_weight_decay)
            raw_cfg.muon_momentum_beta = getattr(args, "muon_momentum_beta", raw_cfg.muon_momentum_beta)
            raw_cfg.muon_nesterov = getattr(args, "muon_nesterov", raw_cfg.muon_nesterov)
            raw_cfg.muon_ns_steps = getattr(args, "muon_ns_steps", raw_cfg.muon_ns_steps)
            raw_cfg.muon_extra_scale_factor = getattr(args, "muon_extra_scale_factor", raw_cfg.muon_extra_scale_factor)
            raw_cfg.muon_scalar_optimizer = getattr(args, "muon_scalar_optimizer", raw_cfg.muon_scalar_optimizer)
            model = maybe_compile_model(model, enabled=args.torch_compile)
            print(
                "[Prismal] resumed checkpoint architecture "
                f"d={model.cfg.d_model} torus={model.cfg.torus_depth}x{model.cfg.torus_height}x{model.cfg.torus_width} "
                f"emitters={model.cfg.n_emitters} slots={model.cfg.n_slots} vocab={model.cfg.vocab_size}",
                flush=True,
            )
            arch_fields = (
                "d_model",
                "n_layers",
                "n_emitters",
                "n_slots",
                "n_paths",
                "top_k_emitters",
                "top_k_slots",
                "torus_depth",
                "torus_height",
                "torus_width",
            )
            ignored_arch_overrides = [
                f"{name}={getattr(args, name)} -> checkpoint {getattr(model.cfg, name)}"
                for name in arch_fields
                if hasattr(args, name) and getattr(args, name) != getattr(model.cfg, name)
            ]
            if ignored_arch_overrides:
                print(
                    "[Prismal] resume keeps checkpoint architecture; ignored requested architecture values: "
                    + ", ".join(ignored_arch_overrides),
                    flush=True,
                )
        else:
            tokenizer_source = args.data
            tokenizer_max_source_samples = args.max_samples
            tokenizer_max_new_tokens = args.max_new_tokens
            tokenizer_min_frequency = args.min_token_frequency
            tokenizer_max_line_tokens = args.max_line_tokens
            tokenizer_max_signature_tokens = args.max_signature_tokens
            tokenizer_supervised_only = not args.tokenizer_full_text
            meta_vocab_size = 0
            data_path = Path(args.data)
            if data_path.is_dir():
                meta_path = data_path / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        meta = {}
                    source_path = meta.get("source")
                    if isinstance(source_path, str) and source_path.strip():
                        tokenizer_source = source_path
                        meta_vocab_size = int(meta.get("vocab_size", 0) or 0)
                        # Pretokenized roots should reuse the recipe that produced them rather than
                        # the GUI's current frequency sweep settings.
                        tokenizer_max_source_samples = 0
                        tokenizer_max_new_tokens = 0
                        tokenizer_min_frequency = 2
                        tokenizer_max_line_tokens = 0
                        tokenizer_max_signature_tokens = 0
                        tokenizer_supervised_only = True
            tokenizer = build_tokenizer_from_source(
                tokenizer_source,
                max_new_tokens=tokenizer_max_new_tokens,
                min_frequency=tokenizer_min_frequency,
                max_line_tokens=tokenizer_max_line_tokens,
                max_signature_tokens=tokenizer_max_signature_tokens,
                max_source_samples=tokenizer_max_source_samples,
                supervised_only=tokenizer_supervised_only,
                use_pronunciation_signatures=args.use_pronunciation_signatures,
                tokenizer_workers=args.tokenizer_workers,
                tokenizer_cache_dir=getattr(args, "tokenizer_cache_dir", ""),
            )
            raw_cfg = _build_config(args, tokenizer)
            if meta_vocab_size > 0:
                raw_cfg.vocab_size = meta_vocab_size
            runtime_cfg = resolve_runtime_config(raw_cfg, tokenizer)
            model = PrismalWaveModel(runtime_cfg).to(device)
            model = maybe_compile_model(model, enabled=args.torch_compile)
            print(
                "[Prismal] fresh architecture "
                f"d={runtime_cfg.d_model} torus={runtime_cfg.torus_depth}x{runtime_cfg.torus_height}x{runtime_cfg.torus_width} "
                f"emitters={runtime_cfg.n_emitters} slots={runtime_cfg.n_slots} vocab={runtime_cfg.vocab_size}",
                flush=True,
        )
            print(
                "[Prismal] family specialists "
                f"enabled={bool(getattr(runtime_cfg, 'per_family_torus_enabled', True))} "
                f"nest={int(getattr(runtime_cfg, 'hierarchical_nest_depth', getattr(runtime_cfg, 'hmote_depth', 1)))} "
                f"recursive={int(getattr(runtime_cfg, 'recursive_hmoe_depth', getattr(runtime_cfg, 'hmote_depth', 1)))} "
                f"max_per_nest={int(getattr(runtime_cfg, 'max_families_per_nest', 1))} "
                f"bank={int(getattr(runtime_cfg, 'family_specialist_bank_size', 1))} "
                f"budget={int(getattr(runtime_cfg, 'family_budget', 1))} "
                f"gate={float(getattr(runtime_cfg, 'family_specialist_gate_threshold', 0.0)):.3f}",
                flush=True,
            )
        model = maybe_compile_model(model, enabled=args.torch_compile)
        train_loader, val_loader = build_train_val_dataloaders(
            args.data,
            tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            val_fraction=args.val_fraction,
            seed=args.seed,
            streaming=getattr(args, "dataset_streaming", True),
        )
        metrics = train_model(
            model,
            train_loader,
            device,
            cfg=raw_cfg,
            optimizer_name=raw_cfg.optimizer,
            epochs=args.epochs,
            steps=args.steps,
            lr=raw_cfg.lr,
            minutes=args.minutes,
            progress=True,
            val_loader=val_loader,
            diagnostic_interval=args.diagnostic_interval,
            use_amp=args.amp,
        )
        save_checkpoint(model, args.save_dir, tokenizer=tokenizer, config=raw_cfg, metrics=metrics)

        if args.post_prompt.strip():
            model = maybe_compile_model(model, enabled=args.torch_compile)
            generated = generate_text(
                model,
                tokenizer,
                args.post_prompt.strip(),
                device,
                max_new_tokens=args.post_max_new_tokens,
                min_new_tokens=args.post_min_new_tokens,
                top_k=args.post_top_k,
                top_p=args.post_top_p,
                temperature=args.post_temperature,
                repetition_penalty=args.post_repetition_penalty,
                no_repeat_ngram_size=args.post_no_repeat_ngram_size,
                beam_size=args.post_beam_size,
                use_speculative_decoding=args.use_speculative_decoding,
                speculative_draft_tokens=args.speculative_draft_tokens,
                speculative_temperature=args.speculative_temperature,
                template_prompt=args.post_template_prompt,
            )
            if args.post_output.strip():
                Path(args.post_output).write_text(generated, encoding="utf-8")
                print(f"[Prismal] saved generated text to {args.post_output}")
            else:
                print(generated)

        print(json.dumps(metrics, indent=2))
        return 0

    if args.command == "benchmark":
        tokenizer = build_tokenizer_from_source(
            args.data,
            max_new_tokens=args.max_new_tokens,
            min_frequency=args.min_token_frequency,
            max_line_tokens=args.max_line_tokens,
            max_signature_tokens=args.max_signature_tokens,
            max_source_samples=args.max_samples,
            supervised_only=not args.tokenizer_full_text,
            use_pronunciation_signatures=args.use_pronunciation_signatures,
            tokenizer_workers=args.tokenizer_workers,
            tokenizer_cache_dir=getattr(args, "tokenizer_cache_dir", ""),
        )
        raw_cfg = _build_config(args, tokenizer)
        runtime_cfg = resolve_runtime_config(raw_cfg, tokenizer)
        model = PrismalWaveModel(runtime_cfg).to(device)
        model = maybe_compile_model(model, enabled=args.torch_compile)
        dataloader = build_dataloader(
            args.data,
            tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            shuffle=False,
        )
        metrics = run_benchmark(model, dataloader, device, steps=args.steps)
        print(json.dumps(metrics, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
