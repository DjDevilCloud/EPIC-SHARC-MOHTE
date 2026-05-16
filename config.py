# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

"""Configuration for the standalone Prismal Torus prototype."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict


@dataclass
class PrismalWaveConfig:
    base_vocab_size: int = 387
    vocab_size: int = 0  # resolved final vocab size after learned hierarchical tokens
    signature_vocab_size: int = 0
    signature_level_vocab_size: int = 0
    signature_relation_vocab_size: int = 0
    signature_bucket_vocab_size: int = 0
    max_samples: int = 0
    lr: float = 0.0008
    optimizer: str = "hierarchical"
    use_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 4
    training_finite_guard_enabled: bool = True
    inference_finite_guard_enabled: bool = True
    grad_clip_muon: float = 0.25
    grad_clip_scalar: float = 0.2
    grad_clip_rowwise: float = 0.45
    hierarchical_precision_enabled: bool = True
    hierarchical_precision_root_dtype: str = "fp4"
    hierarchical_precision_mid_dtype: str = "fp4"
    hierarchical_precision_leaf_dtype: str = "fp4"
    hierarchical_precision_fallback_dtype: str = "fp4"
    hierarchical_precision_accumulator_dtype: str = "fp4"
    hierarchical_precision_allow_float8_leaf: bool = True
    muon_lr: float = 0.008
    muon_weight_decay: float = 0.01
    muon_momentum_beta: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 8
    muon_extra_scale_factor: float = 0.5
    muon_scalar_optimizer: str = "adamw"
    use_nested_learning: bool = True
    nested_learning_local_interval: int = 2
    nested_learning_mid_interval: int = 4
    nested_learning_global_interval: int = 6
    nested_learning_local_ema_beta: float = 0.90
    nested_learning_mid_ema_beta: float = 0.95
    nested_learning_global_ema_beta: float = 0.99
    d_model: int = 128
    n_layers: int = 2
    n_emitters: int = 16384
    n_slots: int = 2048
    n_paths: int = 1
    top_k_emitters: int = 256
    top_k_slots: int = 128
    max_seq_len: int = 0
    position_embedding_init_size: int = 128
    dropout: float = 0.005
    ff_mult: int = 4
    use_factorized_embedding: bool = True
    factorized_embedding_dim: int = 64
    use_turbo_quantization: bool = False
    turbo_quantization_bits: int = 3
    turbo_quantization_method: str = "turbo"
    use_bitsandbytes_leaf_precision: bool = True
    bitsandbytes_leaf_precision_mode: str = "fp4"
    bitsandbytes_leaf_quant_type: str = "fp4"
    bitsandbytes_leaf_compute_dtype: str = "fp4"
    quantization_aware_training: bool = False
    qat_start_fraction: float = 0.65
    qat_ramp_fraction: float = 0.20
    use_torus_core: bool = True
    Torus_SHARC_Router: bool = True
    use_hmote: bool = True
    use_recursive_hmoe: bool = False
    use_gradient_checkpointing: bool = False
    hmote_depth: int = 1
    hmote_branching: int = 1
    hierarchical_nest_depth: int = 1
    hierarchical_child_torus_scale: float = 1.00
    hierarchical_leaf_torus_size: int = 20
    hierarchical_byte_tier: bool = True
    hierarchical_d_model_scale: float = 1.0
    hierarchical_min_d_model: int = 256
    hierarchical_level_d_models: str = ""
    hierarchical_torus_depth_scale: float = 0.75
    hierarchical_recursive_depth_scale: float = 0.75
    hierarchical_fixed_point_scale: float = 0.75
    use_learned_hierarchy_embeddings: bool = True
    learned_hierarchy_vector_dim: int = 128
    learned_hierarchy_vector_scale: float = 0.85
    per_family_torus_enabled: bool = True
    per_family_torus_scale: float = 0.85
    family_specialist_d_model: int = 256
    family_specialist_gate_threshold: float = 0.02
    leaf_cell_enabled: bool = True
    leaf_cell_dim: int = 64
    leaf_router_confidence_threshold: float = 0.75
    max_families_per_nest: int = 16
    family_budget: int = 32
    family_specialist_bank_size: int = 8
    use_mixture_of_torus: bool = True
    mot_num_experts: int = 2
    mot_expert_scale: float = 0.05
    mot_routing_temperature: float = 0.55
    use_topk_mot: bool = True
    mot_top_k: int = 8
    use_signature_lattice_attention: bool = True
    use_path_logits: bool = False
    signature_lattice_dim: int = 256
    signature_lattice_buckets: int = 16
    signature_lattice_candidates: int = 128
    signature_lattice_weight: float = 0.32
    signature_lattice_decay: float = 0.85
    signature_lattice_chunk_len: int = 32
    use_signature_lattice_generation_cache: bool = True
    use_token_superposition_training: bool = False
    token_superposition_bag_size: int = 2
    token_superposition_phase_fraction: float = 0.30
    use_gate: bool = True
    gate_residency_budget: int = 4
    gate_prefetch_horizon: int = 2
    gate_tile_granularity: int = 4
    gate_offload_to_cpu: bool = False
    gate_fallback_on_miss: bool = True
    use_gatetrain: bool = True
    use_fullgatetrain: bool = True
    gatetrain_residency_budget: int = 4
    gatetrain_prefetch_horizon: int = 2
    gatetrain_tile_granularity: int = 4
    gatetrain_offload_to_cpu: bool = False
    gatetrain_fallback_on_miss: bool = True
    use_learned_residency_head: bool = False
    residency_head_layers: int = 1
    residency_head_hidden_dim: int = 64
    learned_residency_weight: float = 0.25
    use_residency_with_reinforcement: bool = False
    use_token_memory_cross_attention: bool = False
    use_token_memory_generation_cache: bool = False
    token_memory_window: int = 96
    token_memory_top_k: int = 4
    token_memory_weight: float = 0.18
    token_memory_copy_bias: float = 0.75
    token_memory_rare_token_cutoff: int = 2
    token_memory_copy_min_confidence: float = 0.35
    use_token_copy_cross_attention: bool = False
    use_token_copy_generation_cache: bool = False
    token_copy_window: int = 96
    token_copy_top_k: int = 4
    token_copy_weight: float = 0.18
    token_copy_bias_strength: float = 0.75
    token_copy_rare_token_cutoff: int = 2
    token_copy_min_confidence: float = 0.35
    use_pronunciation_signatures: bool = False
    hierarchical_tier_char_weight: float = 1.00
    hierarchical_tier_piece_weight: float = 0.25
    hierarchical_tier_word_weight: float = 0.85
    hierarchical_tier_phrase_weight: float = 0.15
    hierarchical_tier_line_weight: float = 0.05
    hierarchical_tier_special_weight: float = 0.15
    hierarchical_leaf_char_boost: float = 1.05
    hierarchical_leaf_piece_boost: float = 1.05
    recursive_hmoe_depth: int = 1
    recursive_hmoe_branching: int = 1
    recursive_hmoe_coarse_top_k: int = 12
    recursive_hmoe_fine_top_k: int = 12
    recursive_hmoe_child_torus_depth: int = 1
    recursive_hmoe_child_torus_height: int = 1
    recursive_hmoe_child_torus_width: int = 1
    recursive_hmoe_balance_weight: float = 0.28
    recursive_hmoe_child_mixture_weight: float = 0.12
    recursive_hmoe_agreement_weight: float = 0.12
    recursive_aux_weight: float = 0.05
    recursive_hidden_mix_scale: float = 0.05
    recursive_field_mix_scale: float = 0.05
    torus_depth: int = 20
    torus_height: int = 20
    torus_width: int = 20
    torus_local_field_radius: int = 2
    torus_global_bus_slots: int = 24
    torus_global_bus_decay: float = 0.92
    torus_global_bus_write_scale: float = 0.32
    torus_write_radius: int = 2
    torus_scout_read_radius: int = 2
    torus_transport: float = 0.65
    torus_write_strength: float = 0.85
    torus_relay_write_radius: int = 4
    torus_inner_temperature: float = 0.05
    torus_outer_temperature: float = 1.00
    torus_relay_interval: int = 12
    torus_transport_interval: int = 4
    torus_primary_temperature: float = 0.95
    torus_relay_temperature_1: float = 0.20
    torus_relay_temperature_2: float = 0.40
    torus_relay_temperature_3: float = 0.80
    torus_activity_threshold: float = 0.001
    torus_active_target_fraction: float = 0.48
    torus_active_balance_weight: float = 0.02
    use_torus_race_lanes: bool = True
    torus_lane_count: int = 1
    torus_scout_density: float = 0.85
    torus_lane_select_threshold_1: float = 0.45
    torus_lane_select_threshold_2: float = 0.70
    torus_lane_relay_hop_spacing: int = 6
    generation_lap_cap: int = 4
    generation_lap_token_cap: int = 48
    torus_chunk_len: int = 32
    use_fixed_point_solver: bool = False
    use_chunk_solver_training: bool = False
    chunk_solver_training_iterations: int = 1
    chunk_solver_training_relaxation: float = 1.0
    chunk_solver_training_substeps: int = 8
    chunk_solver_training_audit_every: int = 18
    fixed_point_iterations: int = 1
    fixed_point_tolerance: float = 1e-4
    fixed_point_relaxation: float = 0.002
    emitter_family_share: float = 0.001
    emitter_level_share: float = 0.005
    emitter_relation_share: float = 0.001
    emitter_parent_share: float = 0.25
    emitter_hierarchy_score_weight: float = 0.75
    emitter_balance_weight: float = 0.45
    emitter_mixture_target_count: float = 32768
    emitter_mixture_weight: float = 0.45
    emitter_birth_threshold: float = 0.0001
    emitter_promotion_threshold: float = 0.85
    seed_all_emitter_families: bool = True
    emitter_seed_activity: float = 1.0
    torus_write_family_floor: float = 0.12
    torus_read_family_floor: float = 0.25
    profile_runtime: bool = False
    router_temperature: float = 1.00
    signature_temperature: float = 0.65
    path_noise: float = 0.015
    torus_weight: float = 0.35
    frequency_weight: float = 0.75
    signature_loss_weight: float = 0.15
    signature_level_loss_weight: float = 0.35
    signature_relation_loss_weight: float = 0.35
    signature_contrastive_weight: float = 0.15
    use_contrastive_routing: bool = False
    contrastive_routing_weight: float = 0.75
    contrastive_routing_temperature: float = 0.65
    contrastive_routing_hard_negatives: bool = False
    use_contrastive_routing_signature_neighborhood: bool = False
    use_contrastive_routing_temporal: bool = False
    use_contrastive_routing_residency: bool = False
    use_contrastive_routing_cross_view: bool = False
    use_contrastive_routing_self_contrast: bool = False
    routing_entropy_weight: float = 0.25
    diversity_weight: float = 0.15
    emitter_neighbor_weight: float = 0.035
    resonance_pull_weight: float = 0.045
    beam_signature_weight: float = 0.35
    beam_coherence_weight: float = 0.45
    path_entropy_penalty: float = 0.15
    memory_momentum: float = 0.85
    use_speculative_decoding: bool = False
    speculative_draft_tokens: int = 6
    speculative_temperature: float = 0.01
    validator_temperature: float = 0.15
    emitter_grid_height: int = 0
    emitter_grid_width: int = 0
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def __post_init__(self) -> None:
        def _sync_alias(canonical_name: str, alias_name: str) -> None:
            fields = type(self).__dataclass_fields__
            canonical_default = fields[canonical_name].default
            alias_default = fields[alias_name].default
            canonical_value = getattr(self, canonical_name)
            alias_value = getattr(self, alias_name)
            if canonical_value != canonical_default:
                setattr(self, alias_name, canonical_value)
            elif alias_value != alias_default:
                setattr(self, canonical_name, alias_value)
                setattr(self, alias_name, alias_value)
            else:
                setattr(self, alias_name, canonical_value)

        _sync_alias("use_token_memory_cross_attention", "use_token_copy_cross_attention")
        _sync_alias("use_token_memory_generation_cache", "use_token_copy_generation_cache")
        _sync_alias("token_memory_window", "token_copy_window")
        _sync_alias("token_memory_top_k", "token_copy_top_k")
        _sync_alias("token_memory_weight", "token_copy_weight")
        _sync_alias("token_memory_copy_bias", "token_copy_bias_strength")
        _sync_alias("token_memory_rare_token_cutoff", "token_copy_rare_token_cutoff")
        _sync_alias("token_memory_copy_min_confidence", "token_copy_min_confidence")
        self.Torus_SHARC_Router = bool(getattr(self, "Torus_SHARC_Router", True))
        self.use_torus_sharc_router = self.Torus_SHARC_Router
        if self.hmote_depth <= 0:
            self.hmote_depth = 1
        if self.hmote_branching <= 0:
            self.hmote_branching = 1
        if self.position_embedding_init_size <= 0:
            self.position_embedding_init_size = 512
        if self.hierarchical_nest_depth < 1:
            self.hierarchical_nest_depth = 1
        if self.hierarchical_child_torus_scale <= 0:
            self.hierarchical_child_torus_scale = 0.5
        if self.hierarchical_leaf_torus_size < 2:
            raise ValueError("hierarchical_leaf_torus_size must be at least 2")
        if self.hierarchical_d_model_scale <= 0:
            self.hierarchical_d_model_scale = 0.5
        if self.hierarchical_min_d_model < 1:
            self.hierarchical_min_d_model = 64
        if self.hierarchical_torus_depth_scale <= 0:
            self.hierarchical_torus_depth_scale = 0.75
        if self.hierarchical_recursive_depth_scale <= 0:
            self.hierarchical_recursive_depth_scale = 0.8
        if self.hierarchical_fixed_point_scale <= 0:
            self.hierarchical_fixed_point_scale = 0.7
        if self.learned_hierarchy_vector_dim < 4:
            self.learned_hierarchy_vector_dim = 4
        if self.learned_hierarchy_vector_scale <= 0:
            self.learned_hierarchy_vector_scale = 0.25
        if self.per_family_torus_scale <= 0:
            self.per_family_torus_scale = 0.25
        if self.family_specialist_d_model < 1:
            self.family_specialist_d_model = 256
        if self.family_specialist_gate_threshold < 0.0:
            self.family_specialist_gate_threshold = 0.08
        if self.leaf_cell_dim < 1:
            self.leaf_cell_dim = 64
        if self.leaf_router_confidence_threshold < 0.0:
            self.leaf_router_confidence_threshold = 0.0
        if self.max_families_per_nest < 1:
            self.max_families_per_nest = 1
        if self.family_budget < self.max_families_per_nest:
            self.family_budget = self.max_families_per_nest
        if self.family_specialist_bank_size < 1:
            self.family_specialist_bank_size = 8
        if self.recursive_aux_weight < 0.0:
            self.recursive_aux_weight = 0.0
        if self.recursive_hidden_mix_scale < 0.0:
            self.recursive_hidden_mix_scale = 0.0
        if self.recursive_field_mix_scale < 0.0:
            self.recursive_field_mix_scale = 0.0
        self.use_gradient_accumulation = bool(self.use_gradient_accumulation)
        if self.gradient_accumulation_steps < 1:
            self.gradient_accumulation_steps = 1
        self.training_finite_guard_enabled = bool(self.training_finite_guard_enabled)
        self.inference_finite_guard_enabled = bool(self.inference_finite_guard_enabled)
        if self.grad_clip_muon < 0.0:
            self.grad_clip_muon = 0.0
        if self.grad_clip_scalar < 0.0:
            self.grad_clip_scalar = 0.0
        if self.grad_clip_rowwise < 0.0:
            self.grad_clip_rowwise = 0.0
        if self.mot_num_experts < 1:
            self.mot_num_experts = 1
        if self.mot_expert_scale <= 0:
            self.mot_expert_scale = 0.65
        if self.mot_routing_temperature <= 0:
            self.mot_routing_temperature = 0.3
        self.use_topk_mot = bool(self.use_topk_mot)
        if self.mot_top_k < 1:
            self.mot_top_k = 1
        if self.emitter_hierarchy_score_weight < 0.0:
            self.emitter_hierarchy_score_weight = 0.0
        self.use_signature_lattice_attention = bool(self.use_signature_lattice_attention)
        self.use_signature_lattice_generation_cache = bool(self.use_signature_lattice_generation_cache)
        self.use_token_superposition_training = bool(self.use_token_superposition_training)
        if self.token_superposition_bag_size < 1:
            self.token_superposition_bag_size = 1
        if self.token_superposition_phase_fraction < 0.0:
            self.token_superposition_phase_fraction = 0.0
        if self.token_superposition_phase_fraction > 1.0:
            self.token_superposition_phase_fraction = 1.0
        self.use_gate = bool(self.use_gate)
        if self.gate_residency_budget < 1:
            self.gate_residency_budget = 1
        if self.gate_prefetch_horizon < 1:
            self.gate_prefetch_horizon = 1
        if self.gate_tile_granularity < 1:
            self.gate_tile_granularity = 1
        self.gate_offload_to_cpu = bool(self.gate_offload_to_cpu)
        self.gate_fallback_on_miss = bool(self.gate_fallback_on_miss)
        self.use_gatetrain = bool(self.use_gatetrain)
        self.use_fullgatetrain = bool(self.use_fullgatetrain)
        if self.use_fullgatetrain:
            self.use_gatetrain = True
        if self.gatetrain_residency_budget < 1:
            self.gatetrain_residency_budget = 1
        if self.gatetrain_prefetch_horizon < 1:
            self.gatetrain_prefetch_horizon = 1
        if self.gatetrain_tile_granularity < 1:
            self.gatetrain_tile_granularity = 1
        self.gatetrain_offload_to_cpu = bool(self.gatetrain_offload_to_cpu)
        self.gatetrain_fallback_on_miss = bool(self.gatetrain_fallback_on_miss)
        self.use_learned_residency_head = bool(self.use_learned_residency_head)
        self.use_residency_with_reinforcement = bool(self.use_residency_with_reinforcement)
        if self.use_residency_with_reinforcement:
            self.use_learned_residency_head = True
        if self.residency_head_layers < 1:
            self.residency_head_layers = 1
        if self.residency_head_hidden_dim < 1:
            self.residency_head_hidden_dim = 256
        if self.learned_residency_weight < 0.0:
            self.learned_residency_weight = 0.0
        self.use_token_memory_cross_attention = bool(self.use_token_memory_cross_attention)
        self.use_token_memory_generation_cache = bool(self.use_token_memory_generation_cache)
        self.use_token_copy_cross_attention = bool(self.use_token_copy_cross_attention)
        self.use_token_copy_generation_cache = bool(self.use_token_copy_generation_cache)
        if self.signature_lattice_dim < 1:
            self.signature_lattice_dim = 256
        if self.signature_lattice_buckets < 1:
            self.signature_lattice_buckets = 512
        if self.signature_lattice_candidates < 1:
            self.signature_lattice_candidates = 8
        if self.signature_lattice_weight < 0.0:
            self.signature_lattice_weight = 0.0
        if self.signature_lattice_decay < 0.0:
            self.signature_lattice_decay = 0.0
        if self.signature_lattice_decay > 1.0:
            self.signature_lattice_decay = 1.0
        if self.signature_lattice_chunk_len < 1:
            self.signature_lattice_chunk_len = 8
        self.use_contrastive_routing = bool(self.use_contrastive_routing)
        self.contrastive_routing_hard_negatives = bool(self.contrastive_routing_hard_negatives)
        self.use_contrastive_routing_signature_neighborhood = bool(self.use_contrastive_routing_signature_neighborhood)
        self.use_contrastive_routing_temporal = bool(self.use_contrastive_routing_temporal)
        self.use_contrastive_routing_residency = bool(self.use_contrastive_routing_residency)
        self.use_contrastive_routing_cross_view = bool(self.use_contrastive_routing_cross_view)
        self.use_contrastive_routing_self_contrast = bool(self.use_contrastive_routing_self_contrast)
        if self.contrastive_routing_weight < 0.0:
            self.contrastive_routing_weight = 0.0
        if self.contrastive_routing_temperature <= 0.0:
            self.contrastive_routing_temperature = 0.1
        if self.token_memory_window < 1:
            self.token_memory_window = 32
        if self.token_memory_top_k < 1:
            self.token_memory_top_k = 4
        if self.token_memory_weight < 0.0:
            self.token_memory_weight = 0.0
        if self.token_memory_copy_bias < 0.0:
            self.token_memory_copy_bias = 0.0
        if self.token_memory_rare_token_cutoff < 0:
            self.token_memory_rare_token_cutoff = 0
        if self.token_memory_copy_min_confidence < 0.0:
            self.token_memory_copy_min_confidence = 0.0
        if self.token_memory_copy_min_confidence > 1.0:
            self.token_memory_copy_min_confidence = 1.0
        self.token_copy_window = self.token_memory_window
        self.token_copy_top_k = self.token_memory_top_k
        self.token_copy_weight = self.token_memory_weight
        self.token_copy_bias_strength = self.token_memory_copy_bias
        self.token_copy_rare_token_cutoff = self.token_memory_rare_token_cutoff
        self.token_copy_min_confidence = self.token_memory_copy_min_confidence
        self.use_token_copy_cross_attention = self.use_token_memory_cross_attention
        self.use_token_copy_generation_cache = self.use_token_memory_generation_cache
        if self.path_entropy_penalty < 0.0:
            self.path_entropy_penalty = 0.0
        if self.torus_local_field_radius < 1:
            self.torus_local_field_radius = 1
        if self.torus_global_bus_slots < 1:
            self.torus_global_bus_slots = 1
        if self.torus_global_bus_decay < 0.0:
            self.torus_global_bus_decay = 0.0
        if self.torus_global_bus_decay > 1.0:
            self.torus_global_bus_decay = 1.0
        if self.torus_global_bus_write_scale < 0.0:
            self.torus_global_bus_write_scale = 0.0
        if self.torus_write_radius < 1:
            self.torus_write_radius = 1
        if self.torus_scout_read_radius < 1:
            self.torus_scout_read_radius = 1
        if self.torus_relay_write_radius < 1:
            self.torus_relay_write_radius = 1
        if self.hierarchical_tier_char_weight <= 0:
            self.hierarchical_tier_char_weight = 1.0
        if self.hierarchical_tier_piece_weight <= 0:
            self.hierarchical_tier_piece_weight = 0.82
        if self.hierarchical_tier_word_weight <= 0:
            self.hierarchical_tier_word_weight = 0.62
        if self.hierarchical_tier_phrase_weight <= 0:
            self.hierarchical_tier_phrase_weight = 0.44
        if self.hierarchical_tier_line_weight <= 0:
            self.hierarchical_tier_line_weight = 0.26
        if self.hierarchical_tier_special_weight <= 0:
            self.hierarchical_tier_special_weight = 0.16
        if self.hierarchical_leaf_char_boost <= 0:
            self.hierarchical_leaf_char_boost = 1.10
        if self.hierarchical_leaf_piece_boost <= 0:
            self.hierarchical_leaf_piece_boost = 1.10
        if self.recursive_hmoe_depth <= 0:
            self.recursive_hmoe_depth = 1
        if self.recursive_hmoe_branching <= 0:
            self.recursive_hmoe_branching = 1
        if self.recursive_hmoe_coarse_top_k <= 0:
            self.recursive_hmoe_coarse_top_k = 1
        if self.recursive_hmoe_fine_top_k <= 0:
            self.recursive_hmoe_fine_top_k = 1
        if self.speculative_draft_tokens <= 0:
            self.speculative_draft_tokens = 1
        if self.speculative_temperature < 0.0:
            self.speculative_temperature = 0.0
        self.use_hmote = bool(self.use_hmote)
        self.use_bitsandbytes_leaf_precision = bool(self.use_bitsandbytes_leaf_precision)
        def _normalize_choice(value: object, *, fallback: str, allowed: set[str]) -> str:
            text = str(value or "").strip().lower()
            if text in allowed:
                return text
            return fallback

        # Keep these values safe even if they arrive from an old config file or a
        # boolean flag that was serialized into JSON as `false`.
        self.bitsandbytes_leaf_precision_mode = _normalize_choice(
            self.bitsandbytes_leaf_precision_mode,
            fallback="fp4",
            allowed={"int4", "fp4"},
        )
        self.bitsandbytes_leaf_quant_type = _normalize_choice(
            self.bitsandbytes_leaf_quant_type,
            fallback="nf4",
            allowed={"fp4", "nf4"},
        )
        self.bitsandbytes_leaf_compute_dtype = _normalize_choice(
            self.bitsandbytes_leaf_compute_dtype,
            fallback="bfloat16",
            allowed={"bfloat16", "bf16", "float16", "fp16"},
        )
        if self.use_hmote:
            self.use_recursive_hmoe = True
            self.use_mixture_of_torus = True
            self.recursive_hmoe_depth = max(1, self.hmote_depth)
            self.recursive_hmoe_branching = max(1, self.hmote_branching)
            self.mot_num_experts = max(1, self.hmote_branching)
        self.hierarchical_precision_enabled = bool(self.hierarchical_precision_enabled)
        self.hierarchical_precision_root_dtype = str(self.hierarchical_precision_root_dtype).strip().lower() or "bf16"
        self.hierarchical_precision_mid_dtype = str(self.hierarchical_precision_mid_dtype).strip().lower() or "fp16"
        self.hierarchical_precision_leaf_dtype = str(self.hierarchical_precision_leaf_dtype).strip().lower() or "float8_e4m3fn"
        self.hierarchical_precision_fallback_dtype = str(self.hierarchical_precision_fallback_dtype).strip().lower() or "bf16"
        self.hierarchical_precision_accumulator_dtype = str(self.hierarchical_precision_accumulator_dtype).strip().lower() or "bf16"
        self.hierarchical_precision_allow_float8_leaf = bool(self.hierarchical_precision_allow_float8_leaf)
        self.use_learned_hierarchy_embeddings = bool(self.use_learned_hierarchy_embeddings)
        self.learned_hierarchy_vector_dim = int(self.learned_hierarchy_vector_dim)
        self.learned_hierarchy_vector_scale = float(self.learned_hierarchy_vector_scale)
        self.quantization_aware_training = bool(self.quantization_aware_training)
        self.use_nested_learning = bool(self.use_nested_learning)
        if self.nested_learning_local_interval < 1:
            self.nested_learning_local_interval = 1
        if self.nested_learning_mid_interval < 1:
            self.nested_learning_mid_interval = 2
        if self.nested_learning_global_interval < 1:
            self.nested_learning_global_interval = 4
        if self.nested_learning_local_ema_beta < 0.0:
            self.nested_learning_local_ema_beta = 0.0
        if self.nested_learning_local_ema_beta >= 1.0:
            self.nested_learning_local_ema_beta = 0.90
        if self.nested_learning_mid_ema_beta < 0.0:
            self.nested_learning_mid_ema_beta = 0.0
        if self.nested_learning_mid_ema_beta >= 1.0:
            self.nested_learning_mid_ema_beta = 0.95
        if self.nested_learning_global_ema_beta < 0.0:
            self.nested_learning_global_ema_beta = 0.0
        if self.nested_learning_global_ema_beta >= 1.0:
            self.nested_learning_global_ema_beta = 0.99
        if self.qat_start_fraction < 0.0:
            self.qat_start_fraction = 0.0
        if self.qat_start_fraction > 1.0:
            self.qat_start_fraction = 1.0
        if self.qat_ramp_fraction < 0.0:
            self.qat_ramp_fraction = 0.0
        if self.qat_ramp_fraction > 1.0:
            self.qat_ramp_fraction = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PrismalWaveConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in payload.items() if k in fields}
        if "Torus_SHARC_Router" not in data and "use_torus_sharc_router" in payload:
            data["Torus_SHARC_Router"] = payload["use_torus_sharc_router"]
        if "torus_local_field_radius" not in data and "torus_write_radius" in payload:
            data["torus_local_field_radius"] = payload["torus_write_radius"]
        if "torus_scout_read_radius" not in data and "torus_read_radius" in payload:
            data["torus_scout_read_radius"] = payload["torus_read_radius"]
        if "use_hmote" not in data:
            legacy_hmote = bool(payload.get("use_recursive_hmoe", False)) and bool(payload.get("use_mixture_of_torus", False))
            data["use_hmote"] = legacy_hmote
        if "hmote_depth" not in data and "recursive_hmoe_depth" in payload:
            data["hmote_depth"] = payload["recursive_hmoe_depth"]
        if "hmote_branching" not in data:
            if "recursive_hmoe_branching" in payload:
                data["hmote_branching"] = payload["recursive_hmoe_branching"]
            elif "mot_num_experts" in payload:
                data["hmote_branching"] = payload["mot_num_experts"]
        return cls(**data)


def save_config(cfg: PrismalWaveConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")


def load_config(path: str | Path) -> PrismalWaveConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    return PrismalWaveConfig.from_dict(payload)
