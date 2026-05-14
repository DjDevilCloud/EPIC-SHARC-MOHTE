# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

"""Standalone Prismal Torus model."""

from dataclasses import dataclass, field
from collections import deque
from contextlib import nullcontext
import math
import warnings
import time
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .config import PrismalWaveConfig
    from .data import SIGNATURE_LEVEL_IDS, SIGNATURE_RELATION_IDS
    from .hierarchical_precision import HierarchicalPrecisionPolicy, HierarchicalPrecisionSpec, attach_precision_policy, current_precision_spec, dtype_name
    from .quantization import QuantizationConfig, create_quantized_embedding, create_quantized_linear
except ImportError:  # pragma: no cover - supports direct script launching.
    from config import PrismalWaveConfig
    from data import SIGNATURE_LEVEL_IDS, SIGNATURE_RELATION_IDS
    from hierarchical_precision import HierarchicalPrecisionPolicy, HierarchicalPrecisionSpec, attach_precision_policy, current_precision_spec, dtype_name
    from quantization import QuantizationConfig, create_quantized_embedding, create_quantized_linear


def _assign_dense_weight(module: nn.Module, dense_weight: torch.Tensor) -> None:
    setter = getattr(module, "set_base_weight", None)
    if callable(setter):
        setter(dense_weight)
        return
    with torch.no_grad():
        module.weight.copy_(dense_weight)


def _sync_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_stage(
    enabled: bool,
    device: torch.device,
    timings: Dict[str, float],
    key: str,
    fn,
):
    if not enabled:
        return fn()
    _sync_for_timing(device)
    start = time.perf_counter()
    result = fn()
    _sync_for_timing(device)
    timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start) * 1000.0
    return result


def _effective_count_from_weights(weights: torch.Tensor) -> torch.Tensor:
    """Return the entropy-derived effective count for a categorical mixture."""
    if weights.numel() == 0:
        return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
    entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
    return torch.exp(entropy)


def _mixture_loss_from_effective_count(effective_count: torch.Tensor, target_count: float) -> torch.Tensor:
    target = max(float(target_count), 1.0)
    target_tensor = torch.tensor(target, device=effective_count.device, dtype=effective_count.dtype)
    return torch.relu(target_tensor - effective_count.mean()) / target


def _scalar_stat_tensor(value: torch.Tensor | float | int, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().to(device=device)
        return value.detach().reshape(-1).mean().to(device=device)
    return torch.tensor(float(value), device=device)


def _repair_finite_tensor(
    tensor: torch.Tensor,
    *,
    fallback: Optional[torch.Tensor] = None,
    allow_negative_inf: bool = False,
) -> Tuple[torch.Tensor, int]:
    if not torch.is_tensor(tensor):
        return tensor, 0
    if tensor.numel() == 0:
        return tensor, 0
    if allow_negative_inf:
        bad_mask = torch.isnan(tensor) | torch.isposinf(tensor)
    else:
        bad_mask = ~torch.isfinite(tensor)
    if not bool(bad_mask.any().item()):
        return tensor, 0
    if fallback is None:
        replacement = torch.zeros_like(tensor)
    else:
        replacement = fallback.to(device=tensor.device, dtype=tensor.dtype)
        if replacement.shape != tensor.shape:
            replacement = replacement.expand_as(tensor)
    repaired = torch.where(bad_mask, replacement, tensor)
    return repaired, int(bad_mask.sum().item())


def _validate_aligned_signature_tensors(
    input_ids: torch.Tensor,
    *,
    signature_family_ids: Optional[torch.Tensor] = None,
    signature_ids: Optional[torch.Tensor] = None,
    signature_level_ids: Optional[torch.Tensor] = None,
    signature_relation_ids: Optional[torch.Tensor] = None,
    parent_signature_ids: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    context: str,
) -> None:
    expected_shape = tuple(input_ids.shape)
    if input_ids.dim() < 2:
        raise ValueError(f"{context}: input_ids must be at least 2D, got shape {expected_shape}")
    checks = (
        ("signature_family_ids", signature_family_ids),
        ("signature_ids", signature_ids),
        ("signature_level_ids", signature_level_ids),
        ("signature_relation_ids", signature_relation_ids),
        ("parent_signature_ids", parent_signature_ids),
        ("loss_mask", loss_mask),
    )
    for name, tensor in checks:
        if tensor is None:
            continue
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{context}: {name} must match input_ids shape {expected_shape}, got {tuple(tensor.shape)}"
            )


@dataclass
class GateResidencyPlan:
    family_ids: Tuple[int, ...] = ()
    expert_ids: Tuple[int, ...] = ()
    emitter_tile_ids: Tuple[int, ...] = ()
    signature_lattice_hot: bool = False
    token_memory_hot: bool = False
    full_scope: bool = False
    router_hot: bool = False
    torus_hot: bool = False
    confidence: float = 0.0
    hit_budget: int = 0
    miss_budget: int = 0
    predicted_tiles: int = 0


@dataclass
class SuperpositionBatch:
    input_ids: torch.Tensor
    token_groups: torch.Tensor
    signature_family_ids: Optional[torch.Tensor]
    signature_ids: Optional[torch.Tensor]
    signature_level_ids: Optional[torch.Tensor]
    signature_relation_ids: Optional[torch.Tensor]
    parent_signature_ids: Optional[torch.Tensor]
    forward_loss_mask: Optional[torch.Tensor]
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    bag_size: int


class LearnedResidencyHead(nn.Module):
    """Small MLP that predicts resident emitter tiles from pooled hidden state."""

    def __init__(
        self,
        d_model: int,
        tile_vocab_size: int,
        *,
        hidden_dim: int = 256,
        layers: int = 1,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.tile_vocab_size = max(1, int(tile_vocab_size))
        self.hidden_dim = max(1, int(hidden_dim))
        self.layers = max(1, int(layers))
        self.quantization_config = quantization_config or QuantizationConfig()
        self.input_proj = create_quantized_linear(
            d_model,
            self.hidden_dim,
            bias=True,
            quantization_config=self.quantization_config,
        )
        self.hidden_layers = nn.ModuleList(
            create_quantized_linear(
                self.hidden_dim,
                self.hidden_dim,
                bias=True,
                quantization_config=self.quantization_config,
            )
            for _ in range(max(0, self.layers - 1))
        )
        self.output_proj = create_quantized_linear(
            self.hidden_dim,
            self.tile_vocab_size,
            bias=True,
            quantization_config=self.quantization_config,
        )

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = hidden.mean(dim=1) if hidden.dim() == 3 else hidden
        x = self.input_proj(pooled)
        x = F.gelu(x)
        for layer in self.hidden_layers:
            x = F.gelu(layer(x))
        tile_logits = self.output_proj(x)
        tile_probs = F.softmax(tile_logits, dim=-1)
        confidence = tile_probs.amax(dim=-1)
        return {
            "pooled_hidden": pooled,
            "tile_logits": tile_logits,
            "tile_probs": tile_probs,
            "confidence": confidence,
        }


class GateResidencyController:
    """Heuristic residency planner for routed-module paging."""

    def __init__(
        self,
        model: "PrismalWaveModel",
        *,
        enabled_attr: str = "use_gate",
        config_prefix: str = "gate",
        stat_prefix: str = "gate",
    ) -> None:
        self.model = model
        self.cfg = model.cfg
        self.enabled = bool(getattr(self.cfg, enabled_attr, False))
        self.full_scope = bool(getattr(self.cfg, "use_fullgatetrain", False))
        self.config_prefix = str(config_prefix)
        self.stat_prefix = str(stat_prefix)
        self.residency_budget = max(1, int(self._cfg("residency_budget", 6)))
        self.prefetch_horizon = max(1, int(self._cfg("prefetch_horizon", 2)))
        self.tile_granularity = max(1, int(self._cfg("tile_granularity", 4)))
        self.offload_to_cpu = bool(self._cfg("offload_to_cpu", False))
        self.fallback_on_miss = bool(self._cfg("fallback_on_miss", True))
        self.route_history: Deque[Dict[str, float]] = deque(maxlen=max(8, self.residency_budget * 8))
        self.last_plan: Optional[GateResidencyPlan] = None
        self.last_route_stats: Optional[Dict[str, torch.Tensor]] = None
        self.last_plan_time_ms: float = 0.0
        self.last_latency_ms: float = 0.0
        self.hit_count: int = 0
        self.miss_count: int = 0
        self.churn_count: int = 0
        self._emitter_tiles_on_device: Tuple[int, ...] = ()

    def _cfg(self, suffix: str, default: object) -> object:
        prefix_name = f"{self.config_prefix}_{suffix}"
        if hasattr(self.cfg, prefix_name):
            return getattr(self.cfg, prefix_name)
        fallback_name = f"gate_{suffix}"
        if hasattr(self.cfg, fallback_name):
            return getattr(self.cfg, fallback_name)
        return default

    def _stat_name(self, suffix: str) -> str:
        return f"{self.stat_prefix}_{suffix}" if self.stat_prefix else suffix

    def _unique_ids(self, values: Sequence[int], *, budget: Optional[int] = None) -> Tuple[int, ...]:
        seen: set[int] = set()
        ordered: List[int] = []
        limit = self.residency_budget if budget is None else max(1, int(budget))
        for value in values:
            item = int(value)
            if item < 0 or item in seen:
                continue
            seen.add(item)
            ordered.append(item)
            if len(ordered) >= limit:
                break
        return tuple(ordered)

    def _route_history_summary(self) -> Dict[str, float]:
        if not self.route_history:
            return {"signature_agreement": 0.0, "avg_entropy": 0.0, "selected_path_index": 0.0}
        agreement = sum(item.get("signature_agreement", 0.0) for item in self.route_history) / float(len(self.route_history))
        entropy = sum(item.get("avg_entropy", 0.0) for item in self.route_history) / float(len(self.route_history))
        selected = self.route_history[-1].get("selected_path_index", 0.0)
        return {
            "signature_agreement": float(agreement),
            "avg_entropy": float(entropy),
            "selected_path_index": float(selected),
        }

    def _route_top_tiles(self, route_stats: Optional[Dict[str, torch.Tensor]]) -> Tuple[int, ...]:
        if not route_stats:
            return ()
        top_idx = route_stats.get("emitter_top_idx")
        if top_idx is None or not torch.is_tensor(top_idx) or top_idx.numel() == 0:
            return ()
        flat = top_idx.detach().long().reshape(-1).tolist()
        tiles: List[int] = []
        for value in flat:
            base_tile = int(value) // self.tile_granularity
            for horizon_offset in range(self.prefetch_horizon):
                tiles.append(base_tile + horizon_offset)
        return self._unique_ids(tiles, budget=self.residency_budget)

    def _history_tiles(self) -> Tuple[int, ...]:
        if not self.route_history:
            return ()
        tiles: List[int] = []
        for entry in reversed(self.route_history):
            selected = int(entry.get("selected_path_index", 0.0))
            tiles.append(max(0, selected // max(1, self.tile_granularity)))
            if len(tiles) >= self.prefetch_horizon:
                break
        return self._unique_ids(tiles, budget=self.residency_budget)

    def _family_seed_tiles(self, family_ids: Optional[torch.Tensor]) -> Tuple[int, ...]:
        if family_ids is None or not torch.is_tensor(family_ids) or family_ids.numel() == 0:
            return ()
        bank_size = max(1, int(getattr(self.model, "signature_bucket_vocab_size", 1)))
        family_values = family_ids.detach().long().reshape(-1).tolist()
        tiles = [int(value) % bank_size for value in family_values]
        return self._unique_ids(tiles, budget=self.residency_budget)

    def _path_seed_experts(self, route_stats: Optional[Dict[str, torch.Tensor]], path_index: Optional[int]) -> Tuple[int, ...]:
        experts = getattr(getattr(self.model, "token_hierarchy", None), "experts", None)
        if experts is None:
            return ()
        expert_count = len(experts)
        if expert_count <= 0:
            return ()
        if route_stats is not None and "selected_path_index" in route_stats:
            selected = route_stats["selected_path_index"]
            if torch.is_tensor(selected) and selected.numel() > 0:
                values = [int(value) % expert_count for value in selected.detach().long().reshape(-1).tolist()]
                return self._unique_ids(values, budget=self.residency_budget)
        if path_index is not None:
            return (int(path_index) % expert_count,)
        return (0,)

    def _history_seed_experts(self) -> Tuple[int, ...]:
        experts = getattr(getattr(self.model, "token_hierarchy", None), "experts", None)
        if experts is None:
            return ()
        expert_count = len(experts)
        if expert_count <= 0 or not self.route_history:
            return ()
        recent = [int(entry.get("selected_path_index", 0.0)) % expert_count for entry in list(self.route_history)[-self.prefetch_horizon :]]
        return self._unique_ids(recent, budget=self.residency_budget)

    def _learned_residency_tiles(self, route_stats: Optional[Dict[str, torch.Tensor]]) -> Tuple[int, ...]:
        if route_stats is None:
            return ()
        tile_ids = route_stats.get("learned_residency_top_tiles")
        if tile_ids is None or not torch.is_tensor(tile_ids) or tile_ids.numel() == 0:
            return ()
        flat = tile_ids.detach().long().reshape(-1).tolist()
        return self._unique_ids(flat, budget=self.residency_budget)

    def _learned_residency_confidence(self, route_stats: Optional[Dict[str, torch.Tensor]]) -> float:
        if route_stats is None:
            return 0.0
        confidence = route_stats.get("learned_residency_confidence")
        if confidence is None or not torch.is_tensor(confidence) or confidence.numel() == 0:
            return 0.0
        return float(confidence.detach().float().mean().clamp(0.0, 1.0).item())

    def _all_family_ids(self) -> Tuple[int, ...]:
        specialists = getattr(self.model, "family_specialists", None)
        if specialists is None:
            return ()
        return tuple(range(len(specialists)))

    def _all_expert_ids(self) -> Tuple[int, ...]:
        experts = getattr(getattr(self.model, "token_hierarchy", None), "experts", None)
        if experts is None:
            return ()
        return tuple(range(len(experts)))

    def _all_emitter_tiles(self) -> Tuple[int, ...]:
        router = getattr(self.model, "router", None)
        tile_size = max(1, self.tile_granularity)
        emitter_count = int(getattr(self.model.cfg, "n_emitters", 0))
        if router is not None:
            bank = getattr(router, "emitter_bank", None)
            if torch.is_tensor(bank) and bank.numel() > 0:
                emitter_count = bank.size(0)
        if emitter_count <= 0:
            return ()
        total_tiles = max(1, math.ceil(emitter_count / tile_size))
        return tuple(range(total_tiles))

    def _prefetch_module(self, module: nn.Module, device: torch.device) -> None:
        module.to(device)

    def _move_module_family(self, module_list: nn.ModuleList, resident_indices: Sequence[int], device: torch.device) -> None:
        resident = {int(index) for index in resident_indices}
        cpu_device = torch.device("cpu")
        for index, module in enumerate(module_list):
            target_device = device if index in resident else cpu_device
            module.to(target_device)

    def _emitter_tiles_to_bank(self, bank: torch.Tensor, tile_ids: Sequence[int]) -> torch.Tensor:
        if bank.numel() == 0:
            return bank
        tile_size = max(1, self.tile_granularity)
        tiles = self._unique_ids(tile_ids, budget=self.residency_budget)
        if not tiles:
            return bank
        slices: List[torch.Tensor] = []
        total_tiles = max(1, math.ceil(bank.size(0) / tile_size))
        for tile_id in tiles:
            start = max(0, int(tile_id) * tile_size)
            end = min(bank.size(0), start + tile_size)
            if start >= bank.size(0):
                continue
            slices.append(bank[start:end])
        if not slices:
            return bank
        resident = torch.cat(slices, dim=0)
        if resident.size(0) >= bank.size(0):
            return resident[: bank.size(0)]
        if self.fallback_on_miss and len(tiles) < total_tiles:
            return bank
        return resident

    def plan(
        self,
        *,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor],
        route_stats: Optional[Dict[str, torch.Tensor]],
        path_index: Optional[int],
        position_index: int,
    ) -> GateResidencyPlan:
        if not self.enabled:
            return GateResidencyPlan()
        if self.full_scope:
            family_tiles = self._all_family_ids()
            expert_ids = self._all_expert_ids()
            predicted_tiles = self._all_emitter_tiles()
            router_hot = bool(getattr(self.model, "router", None) is not None)
            torus_hot = bool(getattr(self.model, "torus_core", None) is not None)
            lattice_hot = bool(getattr(self.model, "signature_lattice_attention", None) is not None)
            token_memory_hot = bool(getattr(self.model, "token_memory_attention", None) is not None)
            return GateResidencyPlan(
                family_ids=family_tiles,
                expert_ids=expert_ids,
                emitter_tile_ids=predicted_tiles,
                signature_lattice_hot=lattice_hot,
                token_memory_hot=token_memory_hot,
                full_scope=True,
                router_hot=router_hot,
                torus_hot=torus_hot,
                confidence=1.0,
                hit_budget=len(predicted_tiles),
                miss_budget=0,
                predicted_tiles=len(predicted_tiles),
            )
        family_tiles = self._family_seed_tiles(signature_family_ids)
        route_tiles = self._route_top_tiles(route_stats)
        history_tiles = self._history_tiles()
        learned_tiles = self._learned_residency_tiles(route_stats)
        expert_ids = self._unique_ids(
            (*self._path_seed_experts(route_stats, path_index), *self._history_seed_experts()),
            budget=self.residency_budget,
        )
        predicted_tiles = self._unique_ids((*family_tiles, *route_tiles, *history_tiles, *learned_tiles), budget=self.residency_budget)
        lattice_hot = bool(getattr(self.model, "signature_lattice_attention", None) is not None)
        token_memory_hot = bool(getattr(self.model, "token_memory_attention", None) is not None)
        history_summary = self._route_history_summary()
        confidence = 0.35
        if route_stats is not None:
            entropy = route_stats.get("avg_entropy")
            agreement = route_stats.get("signature_agreement")
            if torch.is_tensor(agreement):
                confidence = float(agreement.detach().float().mean().clamp(0.0, 1.0).item())
            if torch.is_tensor(entropy):
                confidence = max(confidence, float((1.0 / (1.0 + entropy.detach().float().mean())).clamp(0.0, 1.0).item()))
        confidence = max(confidence, history_summary["signature_agreement"])
        confidence = max(confidence, max(0.0, 1.0 - history_summary["avg_entropy"]))
        confidence = max(confidence, self._learned_residency_confidence(route_stats))
        confidence = max(0.0, min(1.0, confidence))
        return GateResidencyPlan(
            family_ids=family_tiles,
            expert_ids=expert_ids,
            emitter_tile_ids=predicted_tiles,
            signature_lattice_hot=lattice_hot,
            token_memory_hot=token_memory_hot,
            full_scope=False,
            router_hot=bool(getattr(self.model, "router", None) is not None),
            torus_hot=bool(getattr(self.model, "torus_core", None) is not None),
            confidence=confidence,
            hit_budget=len(predicted_tiles),
            miss_budget=max(0, self.residency_budget - len(predicted_tiles)),
            predicted_tiles=len(predicted_tiles),
        )

    def apply(
        self,
        plan: GateResidencyPlan,
        *,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        model = self.model
        stats: Dict[str, torch.Tensor] = {}
        if plan.full_scope and model.router is not None:
            self._prefetch_module(model.router, device)
        if plan.full_scope and model.torus_core is not None:
            self._prefetch_module(model.torus_core, device)
        if plan.full_scope and getattr(model, "token_hierarchy", None) is not None:
            self._prefetch_module(model.token_hierarchy, device)
        if plan.full_scope and model.signature_lattice_attention is not None:
            self._prefetch_module(model.signature_lattice_attention, device)
        if plan.full_scope and model.token_memory_attention is not None:
            self._prefetch_module(model.token_memory_attention, device)
        if self.offload_to_cpu and not plan.full_scope and model.signature_lattice_attention is not None:
            target = device if plan.signature_lattice_hot else torch.device("cpu")
            self._prefetch_module(model.signature_lattice_attention, target)
        if self.offload_to_cpu and not plan.full_scope and model.token_memory_attention is not None:
            target = device if plan.token_memory_hot else torch.device("cpu")
            self._prefetch_module(model.token_memory_attention, target)
        if self.offload_to_cpu and getattr(model, "token_hierarchy", None) is not None:
            hierarchy = model.token_hierarchy
            if hasattr(hierarchy, "gate_apply_residency"):
                if plan.full_scope:
                    family_ids = self._all_family_ids()
                    expert_ids = self._all_expert_ids()
                else:
                    family_ids = plan.family_ids
                    expert_ids = plan.expert_ids
                stats.update(hierarchy.gate_apply_residency(device=device, family_ids=family_ids, expert_ids=expert_ids, offload_to_cpu=not plan.full_scope))
        if self.offload_to_cpu and not plan.full_scope and model.router is not None:
            model.router.emitter_bank.data = model.router.emitter_bank.data.to(device="cpu")
            model.router.operator_hierarchy_bank.data = model.router.operator_hierarchy_bank.data.to(device="cpu")
            emitter_bank = model.router.emitter_bank.detach().to(device="cpu")
            hierarchy_bank = model.router.operator_hierarchy_bank.detach().to(device="cpu")
            if plan.emitter_tile_ids:
                emitter_bank = self._emitter_tiles_to_bank(emitter_bank, plan.emitter_tile_ids).to(device=device)
                hierarchy_bank = self._emitter_tiles_to_bank(hierarchy_bank, plan.emitter_tile_ids).to(device=device)
            else:
                emitter_bank = emitter_bank.to(device=device)
                hierarchy_bank = hierarchy_bank.to(device=device)
            stats[self._stat_name("emitter_tiles")] = torch.tensor(float(len(plan.emitter_tile_ids)), device=device)
            stats[self._stat_name("emitter_override_rows")] = torch.tensor(float(emitter_bank.size(0)), device=device)
            stats[f"_{self.stat_prefix}_emitter_bank_override"] = emitter_bank
            stats[f"_{self.stat_prefix}_operator_bank_override"] = hierarchy_bank
        else:
            stats[self._stat_name("emitter_tiles")] = torch.tensor(float(len(plan.emitter_tile_ids)), device=device)
        stats[self._stat_name("full_scope")] = torch.tensor(1.0 if plan.full_scope else 0.0, device=device)
        stats[self._stat_name("router_hot")] = torch.tensor(1.0 if plan.router_hot else 0.0, device=device)
        stats[self._stat_name("torus_hot")] = torch.tensor(1.0 if plan.torus_hot else 0.0, device=device)
        self._emitter_tiles_on_device = plan.emitter_tile_ids
        return stats

    def record(self, route_stats: Dict[str, torch.Tensor], *, plan: Optional[GateResidencyPlan] = None) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            return {}
        previous_plan = self.last_plan
        plan = plan or previous_plan or GateResidencyPlan()
        summary: Dict[str, torch.Tensor] = {}
        self.last_route_stats = dict(route_stats)
        self.route_history.append(
            {
                "signature_agreement": float(route_stats.get("signature_agreement", torch.tensor(0.0)).detach().float().mean().item())
                if "signature_agreement" in route_stats and torch.is_tensor(route_stats["signature_agreement"])
                else 0.0,
                "avg_entropy": float(route_stats.get("avg_entropy", torch.tensor(0.0)).detach().float().mean().item())
                if "avg_entropy" in route_stats and torch.is_tensor(route_stats["avg_entropy"])
                else 0.0,
                "selected_path_index": float(route_stats.get("selected_path_index", torch.tensor(0.0)).detach().float().mean().item())
                if "selected_path_index" in route_stats and torch.is_tensor(route_stats["selected_path_index"])
                else 0.0,
            }
        )
        gate_hit = 1 if plan.predicted_tiles > 0 and plan.confidence >= 0.5 else 0
        gate_miss = 1 - gate_hit
        self.hit_count += gate_hit
        self.miss_count += gate_miss
        if previous_plan is not None and plan.emitter_tile_ids != previous_plan.emitter_tile_ids:
            self.churn_count += 1
        device = next(self.model.parameters()).device
        summary[self._stat_name("hit_count")] = torch.tensor(float(self.hit_count), device=device)
        summary[self._stat_name("miss_count")] = torch.tensor(float(self.miss_count), device=device)
        summary[self._stat_name("tile_churn")] = torch.tensor(float(self.churn_count), device=device)
        summary[self._stat_name("predicted_tiles")] = torch.tensor(float(plan.predicted_tiles), device=device)
        summary[self._stat_name("confidence")] = torch.tensor(float(plan.confidence), device=device)
        summary[self._stat_name("latency_saved_ms")] = torch.tensor(
            float(plan.predicted_tiles) * float(plan.confidence) * 0.1,
            device=device,
        )
        summary[self._stat_name("plan_time_ms")] = torch.tensor(float(self.last_plan_time_ms), device=device)
        summary[self._stat_name("lead_time_ms")] = torch.tensor(
            float(max(self.prefetch_horizon, 1)) * float(plan.confidence) * 0.1,
            device=device,
        )
        summary[self._stat_name("batch_hit")] = torch.tensor(float(gate_hit), device=device)
        summary[self._stat_name("batch_miss")] = torch.tensor(float(gate_miss), device=device)
        summary[self._stat_name("full_scope")] = torch.tensor(1.0 if plan.full_scope else 0.0, device=device)
        summary[self._stat_name("router_hot")] = torch.tensor(1.0 if plan.router_hot else 0.0, device=device)
        summary[self._stat_name("torus_hot")] = torch.tensor(1.0 if plan.torus_hot else 0.0, device=device)
        self.last_plan = plan
        return summary


@dataclass
class PrismalTorusState:
    field: torch.Tensor
    bus: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        return self.field.shape

    def to(self, *args, **kwargs) -> "PrismalTorusState":
        return PrismalTorusState(field=self.field.to(*args, **kwargs), bus=self.bus.to(*args, **kwargs))

    def clone(self) -> "PrismalTorusState":
        return PrismalTorusState(field=self.field.clone(), bus=self.bus.clone())


def _recursive_child_config(
    cfg: PrismalWaveConfig,
    *,
    depth: int,
    branching: int,
    child_torus_depth: int,
    child_torus_height: int,
    child_torus_width: int,
    child_fine_top_k: int,
) -> PrismalWaveConfig:
    payload = cfg.to_dict()
    payload.update(
        {
            "use_hmote": False,
            "use_recursive_hmoe": depth > 1,
            "hmote_depth": max(1, depth),
            "hmote_branching": max(1, branching),
            "recursive_hmoe_depth": max(1, depth),
            "recursive_hmoe_branching": max(1, branching),
            "recursive_hmoe_coarse_top_k": max(1, min(branching, int(getattr(cfg, "recursive_hmoe_coarse_top_k", 1)))),
            "recursive_hmoe_fine_top_k": max(1, child_fine_top_k),
            "recursive_hmoe_child_torus_depth": max(2, child_torus_depth),
            "recursive_hmoe_child_torus_height": max(2, child_torus_height),
            "recursive_hmoe_child_torus_width": max(2, child_torus_width),
            "torus_depth": max(2, child_torus_depth),
            "torus_height": max(2, child_torus_height),
            "torus_width": max(2, child_torus_width),
            "n_paths": max(1, child_fine_top_k),
            "use_torus_core": True,
        }
    )
    return PrismalWaveConfig.from_dict(payload)


class _FamilyTorusSpecialist(nn.Module):
    """Small torus wrapper with input/output projection around a mini core."""

    def __init__(
        self,
        parent_cfg: PrismalWaveConfig,
        mini_cfg: PrismalWaveConfig,
        *,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.parent_cfg = parent_cfg
        self.mini_cfg = mini_cfg
        self.quantization_config = quantization_config or QuantizationConfig()
        self.input_proj = create_quantized_linear(
            parent_cfg.d_model,
            mini_cfg.d_model,
            bias=False,
            quantization_config=self.quantization_config,
        )
        self.core = PrismalTorusCore(mini_cfg, quantization_config=self.quantization_config)
        self.output_proj = create_quantized_linear(
            mini_cfg.d_model,
            parent_cfg.d_model,
            bias=False,
            quantization_config=self.quantization_config,
        )
        self.context_projectors = nn.ModuleDict(
            {
                "registry": self._make_context_projector(parent_cfg.d_model, mini_cfg.d_model),
                "family": self._make_context_projector(parent_cfg.d_model, mini_cfg.d_model),
                "level": self._make_context_projector(parent_cfg.d_model, mini_cfg.d_model),
                "relation": self._make_context_projector(parent_cfg.d_model, mini_cfg.d_model),
                "parent": self._make_context_projector(parent_cfg.d_model, mini_cfg.d_model),
            }
        )

    def _make_context_projector(self, input_dim: int, output_dim: int) -> nn.Module:
        if input_dim == output_dim:
            return nn.Identity()
        return create_quantized_linear(
            input_dim,
            output_dim,
            bias=False,
            quantization_config=self.quantization_config,
        )

    def _project_context(self, context: Optional[torch.Tensor], projector: nn.Module) -> Optional[torch.Tensor]:
        if context is None or not torch.is_tensor(context) or context.numel() == 0:
            return None
        if not context.dtype.is_floating_point:
            return None
        if context.shape[-1] == self.mini_cfg.d_model:
            return context
        if context.shape[-1] == self.parent_cfg.d_model:
            return projector(context)
        return None

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        path_index: int = 0,
        step_index_offset: int = 0,
        use_solver: bool = False,
        relay_mode: bool = False,
    ) -> torch.Tensor:
        squeeze = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
            squeeze = True
        if hidden.dim() != 3:
            raise ValueError("hidden must have shape (batch, seq_len, d_model) or (batch, d_model)")

        projected = self.input_proj(hidden)
        field_state = self.core.init_state(projected.size(0), projected.device)

        step_fn = self.core.chunk_solver_step if use_solver else self.core.chunked_step
        step_kwargs = dict(
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            registry_context=self._project_context(registry_context, self.context_projectors["registry"]),
            family_context=self._project_context(family_context, self.context_projectors["family"]),
            level_context=self._project_context(level_context, self.context_projectors["level"]),
            relation_context=self._project_context(relation_context, self.context_projectors["relation"]),
            parent_context=self._project_context(parent_context, self.context_projectors["parent"]),
            path_index=path_index,
            step_index_offset=step_index_offset,
        )
        if not use_solver:
            step_kwargs["relay_mode"] = relay_mode
        specialist_hidden, _, _ = step_fn(projected, field_state, **step_kwargs)
        specialist_hidden = self.output_proj(specialist_hidden)
        if squeeze:
            specialist_hidden = specialist_hidden.squeeze(1)
        return specialist_hidden


class HierarchicalParameterNest(nn.Module):
    """Mixture of Torus (MoT) + light hierarchy — no duplication."""

    @staticmethod
    def _parse_d_model_schedule(value: object) -> List[int]:
        if value is None:
            return []
        text = str(value).replace(";", ",").replace("|", ",").strip()
        if not text:
            return []
        levels: List[int] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                dim = int(part)
            except ValueError:
                continue
            if dim > 0:
                levels.append(dim)
        return levels

    def __init__(
        self,
        cfg: PrismalWaveConfig,
        level: int = 0,
        parent: Optional["HierarchicalParameterNest"] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        shared_embedding: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.level = level
        object.__setattr__(self, "parent", parent)
        self.quantization_config = quantization_config or QuantizationConfig()
        self.use_gradient_checkpointing = bool(getattr(cfg, "use_gradient_checkpointing", True))
        self.use_hmote = bool(getattr(cfg, "use_hmote", False))
        self.hierarchical_depth = max(
            1,
            int(
                getattr(
                    cfg,
                    "hierarchical_nest_depth",
                    getattr(cfg, "hmote_depth", getattr(cfg, "recursive_hmoe_depth", 1)),
                )
            ),
        )
        self.child_torus_scale = float(getattr(cfg, "hierarchical_child_torus_scale", 0.5))
        self.leaf_size = int(getattr(cfg, "hierarchical_leaf_torus_size", 4))
        if self.leaf_size < 2:
            raise ValueError("hierarchical_leaf_torus_size must be at least 2")
        self.byte_tier = bool(getattr(cfg, "hierarchical_byte_tier", True))
        self.is_leaf_nest = self.level >= self.hierarchical_depth - 1
        dim_scale = float(getattr(cfg, "hierarchical_d_model_scale", 0.5))
        root_d_model = max(1, int(getattr(cfg, "d_model", 1)))
        min_d_model = min(max(1, int(getattr(cfg, "hierarchical_min_d_model", 64))), root_d_model)
        parent_d_model = parent.d_model if parent is not None else root_d_model
        dim_schedule = self._parse_d_model_schedule(getattr(cfg, "hierarchical_level_d_models", ""))
        if level == 0:
            scaled_d_model = root_d_model
        elif level < len(dim_schedule):
            scaled_d_model = dim_schedule[level]
        else:
            scaled_d_model = int(round(root_d_model * (dim_scale ** max(0, level))))
        self.d_model = max(min_d_model, min(parent_d_model, scaled_d_model))
        self.external_d_model = parent_d_model if parent is not None else self.d_model
        self.dimension_scale = float(self.d_model / max(root_d_model, 1))
        torus_scale = float(getattr(cfg, "hierarchical_torus_depth_scale", 0.75)) ** max(0, level)
        recursive_scale = float(getattr(cfg, "hierarchical_recursive_depth_scale", 0.8)) ** max(0, level)
        fixed_point_scale = float(getattr(cfg, "hierarchical_fixed_point_scale", 0.7)) ** max(0, level)
        self.effective_torus_depth = max(2, int(round(float(getattr(cfg, "torus_depth", 2)) * torus_scale)))
        self.effective_torus_height = max(2, int(round(float(getattr(cfg, "torus_height", self.effective_torus_depth)) * torus_scale)))
        self.effective_torus_width = max(2, int(round(float(getattr(cfg, "torus_width", self.effective_torus_depth)) * torus_scale)))
        if self.is_leaf_nest and self.byte_tier:
            self.effective_torus_depth = self.leaf_size
            self.effective_torus_height = self.leaf_size
            self.effective_torus_width = self.leaf_size
        self.effective_recursive_depth = max(
            1,
            int(round(float(getattr(cfg, "recursive_hmoe_depth", getattr(cfg, "hmote_depth", 1))) * recursive_scale)),
        )
        self.effective_fixed_point_iterations = max(
            1,
            int(round(float(getattr(cfg, "fixed_point_iterations", 1)) * fixed_point_scale)),
        )
        self.effective_chunk_solver_training_iterations = max(
            1,
            int(round(float(getattr(cfg, "chunk_solver_training_iterations", 1)) * fixed_point_scale)),
        )
        self.effective_chunk_solver_training_substeps = max(
            1,
            int(round(float(getattr(cfg, "chunk_solver_training_substeps", 1)) * fixed_point_scale)),
        )
        capacity_scale = self.dimension_scale if level > 0 else 1.0
        if level > 0:
            scaled_n_emitters = max(8, int(round(float(getattr(cfg, "n_emitters", 8)) * capacity_scale)))
            scaled_n_slots = max(4, int(round(float(getattr(cfg, "n_slots", 4)) * capacity_scale)))
            scaled_n_paths = max(1, int(round(float(getattr(cfg, "n_paths", 1)) * max(capacity_scale, 0.25))))
            scaled_top_k_emitters = max(1, min(scaled_n_emitters, int(round(float(getattr(cfg, "top_k_emitters", 1)) * capacity_scale))))
            scaled_top_k_slots = max(1, min(scaled_n_slots, int(round(float(getattr(cfg, "top_k_slots", 1)) * capacity_scale))))
        else:
            scaled_n_emitters = int(getattr(cfg, "n_emitters", 8))
            scaled_n_slots = int(getattr(cfg, "n_slots", 4))
            scaled_n_paths = int(getattr(cfg, "n_paths", 1))
            scaled_top_k_emitters = int(getattr(cfg, "top_k_emitters", 1))
            scaled_top_k_slots = int(getattr(cfg, "top_k_slots", 1))
        local_payload = cfg.to_dict()
        local_payload.update(
            {
                "d_model": self.d_model,
                "n_emitters": scaled_n_emitters,
                "n_slots": scaled_n_slots,
                "n_paths": scaled_n_paths,
                "top_k_emitters": scaled_top_k_emitters,
                "top_k_slots": scaled_top_k_slots,
                "torus_depth": self.effective_torus_depth,
                "torus_height": self.effective_torus_height,
                "torus_width": self.effective_torus_width,
                "hmote_depth": self.effective_recursive_depth,
                "recursive_hmoe_depth": self.effective_recursive_depth,
                "fixed_point_iterations": self.effective_fixed_point_iterations,
                "chunk_solver_training_iterations": self.effective_chunk_solver_training_iterations,
                "chunk_solver_training_substeps": self.effective_chunk_solver_training_substeps,
            }
        )
        self.local_cfg = PrismalWaveConfig.from_dict(local_payload)
        self.input_proj = (
            nn.Identity()
            if self.external_d_model == self.d_model
            else create_quantized_linear(
                self.external_d_model,
                self.d_model,
                bias=False,
                quantization_config=self.quantization_config,
            )
        )
        self.output_proj = (
            nn.Identity()
            if self.external_d_model == self.d_model
            else create_quantized_linear(
                self.d_model,
                self.external_d_model,
                bias=False,
                quantization_config=self.quantization_config,
            )
        )
        self.per_family_torus_enabled = bool(getattr(cfg, "per_family_torus_enabled", True))
        self.per_family_torus_scale = float(getattr(cfg, "per_family_torus_scale", 0.25))
        self.family_specialist_gate_threshold = float(getattr(cfg, "family_specialist_gate_threshold", 0.08))
        self.use_gate = bool(getattr(cfg, "use_gate", False))
        self.gate_residency_budget = max(1, int(getattr(cfg, "gate_residency_budget", 6)))
        self.gate_prefetch_horizon = max(1, int(getattr(cfg, "gate_prefetch_horizon", 2)))
        self.gate_tile_granularity = max(1, int(getattr(cfg, "gate_tile_granularity", 4)))
        self.gate_offload_to_cpu = bool(getattr(cfg, "gate_offload_to_cpu", False))
        self.gate_fallback_on_miss = bool(getattr(cfg, "gate_fallback_on_miss", True))
        self.leaf_cell_enabled = bool(getattr(cfg, "leaf_cell_enabled", True))
        self.leaf_cell_dim = max(1, int(getattr(cfg, "leaf_cell_dim", 64)))
        self.leaf_router_confidence_threshold = float(getattr(cfg, "leaf_router_confidence_threshold", 0.0))
        self.max_families_per_nest = max(1, int(getattr(cfg, "max_families_per_nest", 16)))
        self.family_specialist_bank_size = max(
            1,
            min(
                self.max_families_per_nest,
                int(getattr(cfg, "family_specialist_bank_size", min(8, self.max_families_per_nest))),
            ),
        )
        self.family_budget = max(
            self.max_families_per_nest,
            int(getattr(cfg, "family_budget", self.max_families_per_nest)),
        )
        default_family_budget = int(PrismalWaveConfig.__dataclass_fields__["family_budget"].default)
        default_max_families = int(PrismalWaveConfig.__dataclass_fields__["max_families_per_nest"].default)
        if (
            int(getattr(cfg, "family_budget", default_family_budget)) == default_family_budget
            and self.max_families_per_nest != default_max_families
        ):
            self.family_budget = self.max_families_per_nest
        # Keep the resolved runtime capacity visible to checkpoint save/load paths.
        self.cfg.max_families_per_nest = self.max_families_per_nest
        self.cfg.family_budget = self.family_budget
        self.cfg.family_specialist_bank_size = self.family_specialist_bank_size
        self.local_cfg.max_families_per_nest = self.max_families_per_nest
        self.local_cfg.family_budget = self.family_budget
        self.local_cfg.family_specialist_bank_size = self.family_specialist_bank_size
        self.branching = max(
            1,
            int(
                getattr(
                    cfg,
                    "hmote_branching",
                    getattr(cfg, "recursive_hmoe_branching", getattr(cfg, "mot_num_experts", 4)),
                )
            ),
        )
        self.use_mot = self.use_hmote
        self.mot_num_experts = max(1, self.branching)
        self.mot_expert_scale = float(getattr(cfg, "mot_expert_scale", 0.4))
        self.mot_routing_temperature = float(getattr(cfg, "mot_routing_temperature", 0.3))
        self.use_topk_mot = bool(getattr(cfg, "use_topk_mot", True))
        self.mot_top_k = max(1, int(getattr(cfg, "mot_top_k", 2)))
        self.child_mix_weight = 0.05
        self.family_specialist_weight = 0.3
        self.leaf_cell_weight = 0.4
        self.family_gate = create_quantized_linear(
            self.d_model,
            1,
            bias=True,
            quantization_config=self.quantization_config,
        )
        self.family_specialist_router = (
            create_quantized_linear(
                self.d_model,
                self.family_specialist_bank_size,
                bias=True,
                quantization_config=self.quantization_config,
            )
            if self.per_family_torus_enabled
            else None
        )
        self._leaf_level_ids = [
            int(level_id)
            for level_id in (
                SIGNATURE_LEVEL_IDS.get("char"),
                SIGNATURE_LEVEL_IDS.get("piece"),
            )
            if level_id is not None
        ]

        # Shared embedding only at top
        if level == 0 or parent is None:
            self.shared_embedding = shared_embedding or FactorizedEmbedding(
                cfg.vocab_size, cfg.d_model, cfg.factorized_embedding_dim,
                quantization_config=self.quantization_config,
            )
            recursive_depth = max(
                1,
                int(getattr(self.local_cfg, "recursive_hmoe_depth", getattr(self.local_cfg, "hmote_depth", 1))),
            )
            self.core = (
                PrismalRecursiveTorusCore(self.local_cfg, quantization_config=self.quantization_config)
                if self.use_hmote and recursive_depth > 1
                else PrismalTorusCore(self.local_cfg, quantization_config=self.quantization_config)
            )
        else:
            self.shared_embedding = parent.shared_embedding
            core_payload = self.local_cfg.to_dict()
            core_payload.update({
                "torus_depth": self.effective_torus_depth,
                "torus_height": self.effective_torus_height,
                "torus_width": self.effective_torus_width,
                "recursive_hmoe_child_torus_depth": self.effective_torus_depth,
                "recursive_hmoe_child_torus_height": self.effective_torus_height,
                "recursive_hmoe_child_torus_width": self.effective_torus_width,
                "use_hmote": False,
                "use_recursive_hmoe": False,
                "hmote_depth": 1,
                "hmote_branching": 1,
                "recursive_hmoe_depth": self.effective_recursive_depth,
                "recursive_hmoe_branching": 1,
                "use_mixture_of_torus": False,
                "fixed_point_iterations": self.effective_fixed_point_iterations,
                "chunk_solver_training_iterations": self.effective_chunk_solver_training_iterations,
                "chunk_solver_training_substeps": self.effective_chunk_solver_training_substeps,
            })
            core_cfg = PrismalWaveConfig.from_dict(core_payload)
            self.core = PrismalTorusCore(core_cfg, quantization_config=self.quantization_config)

        self.per_family_tori: Dict[str, _FamilyTorusSpecialist] = {}
        self.family_specialists = nn.ModuleList()
        if self.per_family_torus_enabled:
            for _ in range(self.family_specialist_bank_size):
                mini_cfg = self._make_mini_torus_cfg(self.local_cfg, scale=self.per_family_torus_scale)
                self.family_specialists.append(
                    _FamilyTorusSpecialist(
                        self.local_cfg,
                        mini_cfg,
                        quantization_config=self.quantization_config,
                    )
                )
        if self.level == 0 and self.per_family_torus_enabled:
            self.register_buffer(
                "family_usage_ema",
                torch.zeros(self.max_families_per_nest, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.family_usage_ema = None  # type: ignore[assignment]

        # === Mixture of Torus experts: tiny adapters only, no duplicate full cores ===
        self.experts = nn.ModuleList()
        if self.use_mot and not self.is_leaf_nest:
            for _ in range(self.mot_num_experts):
                self.experts.append(
                    nn.Sequential(
                        create_quantized_linear(self.d_model, self.d_model, quantization_config=self.quantization_config),
                        nn.GELU(),
                        create_quantized_linear(self.d_model, self.d_model, quantization_config=self.quantization_config),
                    )
                )

        # Router (very cheap)
        self.mot_router = (
            create_quantized_linear(
                self.d_model, len(self.experts) + 1, quantization_config=self.quantization_config
            )
            if self.experts
            else None
        )

        # Light per-level adapter
        self.local_adapter = create_quantized_linear(
            self.d_model, self.d_model, bias=True, quantization_config=self.quantization_config
        )
        self.tier_gate = create_quantized_linear(
            self.d_model, 1, bias=False, quantization_config=self.quantization_config
        )

        self.shared_leaf_cells: Optional[nn.ModuleList] = None
        if level == 0 or parent is None:
            self.shared_leaf_cells = nn.ModuleList()
            if self.leaf_cell_enabled:
                cell_dim = max(1, min(self.d_model, self.leaf_cell_dim))
                for _ in range(8):
                    self.shared_leaf_cells.append(
                        nn.Sequential(
                            create_quantized_linear(
                                self.d_model,
                                cell_dim,
                                quantization_config=self.quantization_config,
                                module_role="leaf",
                            ),
                            nn.GELU(),
                            create_quantized_linear(
                                cell_dim,
                                self.d_model,
                                quantization_config=self.quantization_config,
                                module_role="leaf",
                            ),
                        )
                    )
        else:
            if parent.d_model == self.d_model:
                self.shared_leaf_cells = parent.shared_leaf_cells
            else:
                self.shared_leaf_cells = nn.ModuleList()
                if self.leaf_cell_enabled:
                    cell_dim = max(1, min(self.d_model, self.leaf_cell_dim))
                    for _ in range(8):
                        self.shared_leaf_cells.append(
                            nn.Sequential(
                                create_quantized_linear(
                                    self.d_model,
                                    cell_dim,
                                    quantization_config=self.quantization_config,
                                    module_role="leaf",
                                ),
                                nn.GELU(),
                                create_quantized_linear(
                                    cell_dim,
                                    self.d_model,
                                    quantization_config=self.quantization_config,
                                    module_role="leaf",
                                ),
                            )
                        )
        self.leaf_cells = self.shared_leaf_cells if self.shared_leaf_cells is not None else nn.ModuleList()
        self.leaf_router = (
            create_quantized_linear(
                self.d_model,
                len(self.leaf_cells),
                bias=True,
                quantization_config=self.quantization_config,
                module_role="leaf",
            )
            if self.leaf_cell_enabled and len(self.leaf_cells) > 0
            else None
        )

        # Children (light hierarchy on top of MoT)
        self.child_nests = nn.ModuleList()
        if level + 1 < self.hierarchical_depth:
            for _ in range(self.branching):
                self.child_nests.append(
                    HierarchicalParameterNest(cfg, level + 1, self, quantization_config=self.quantization_config)
                )

    def _precision_path(self) -> str:
        return str(getattr(self, "_precision_module_path", f"hierarchy.level{self.level}"))

    def _resolve_precision_spec(self, device: torch.device) -> Optional[HierarchicalPrecisionSpec]:
        spec = getattr(self, "precision_spec", None)
        if isinstance(spec, HierarchicalPrecisionSpec):
            return spec
        policy = getattr(self, "precision_policy", None)
        if policy is None:
            return None
        spec = policy.resolve_for_level(
            self.level,
            self.hierarchical_depth,
            device,
            is_leaf=self.is_leaf_nest,
            module_path=self._precision_path(),
            module_kind="nest",
        )
        self.precision_spec = spec
        return spec

    def _precision_scope(self, device: torch.device, *, enabled: bool = True):
        policy = getattr(self, "precision_policy", None)
        spec = self._resolve_precision_spec(device)
        if policy is None or spec is None:
            return nullcontext()
        return policy.scope(spec, device=device, enabled=enabled)

    def _project_last_dim(
        self,
        tensor: Optional[torch.Tensor],
        *,
        target_dim: int,
        projector: nn.Module,
        source_dim: int,
    ) -> Optional[torch.Tensor]:
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return tensor
        if tensor.shape[-1] == target_dim:
            return tensor
        if tensor.shape[-1] == source_dim:
            return projector(tensor)
        return tensor.mean(dim=-1, keepdim=True).expand(*tensor.shape[:-1], target_dim)

    def _project_hidden_in(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self._project_last_dim(
            hidden,
            target_dim=self.d_model,
            projector=self.input_proj,
            source_dim=self.external_d_model,
        )
        return projected if projected is not None else hidden

    def _project_hidden_out(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self._project_last_dim(
            hidden,
            target_dim=self.external_d_model,
            projector=self.output_proj,
            source_dim=self.d_model,
        )
        return projected if projected is not None else hidden

    def _project_state_in(
        self,
        field_state: torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(field_state, PrismalTorusState):
            field = self._project_last_dim(
                field_state.field,
                target_dim=self.d_model,
                projector=self.input_proj,
                source_dim=self.external_d_model,
            )
            bus = self._project_last_dim(
                field_state.bus,
                target_dim=self.d_model,
                projector=self.input_proj,
                source_dim=self.external_d_model,
            )
            return PrismalTorusState(field=field if field is not None else field_state.field, bus=bus if bus is not None else field_state.bus)
        if isinstance(field_state, tuple) and len(field_state) == 2:
            field = self._project_last_dim(
                field_state[0],
                target_dim=self.d_model,
                projector=self.input_proj,
                source_dim=self.external_d_model,
            )
            bus = self._project_last_dim(
                field_state[1],
                target_dim=self.d_model,
                projector=self.input_proj,
                source_dim=self.external_d_model,
            )
            return (field if field is not None else field_state[0], bus if bus is not None else field_state[1])
        if isinstance(field_state, dict):
            projected = dict(field_state)
            for key in ("field", "bus"):
                value = projected.get(key)
                if torch.is_tensor(value):
                    projected_value = self._project_last_dim(
                        value,
                        target_dim=self.d_model,
                        projector=self.input_proj,
                        source_dim=self.external_d_model,
                    )
                    if projected_value is not None:
                        projected[key] = projected_value
            return projected
        projected_state = self._project_last_dim(
            field_state,
            target_dim=self.d_model,
            projector=self.input_proj,
            source_dim=self.external_d_model,
        )
        return projected_state if projected_state is not None else field_state

    def _project_context_kwargs(self, kwargs: Dict[str, object]) -> Dict[str, object]:
        projected = dict(kwargs)
        for key in ("registry_context", "family_context", "level_context", "relation_context", "parent_context"):
            value = projected.get(key)
            if torch.is_tensor(value):
                projected_value = self._project_last_dim(
                    value,
                    target_dim=self.d_model,
                    projector=self.input_proj,
                    source_dim=self.external_d_model,
                )
                if projected_value is not None:
                    projected[key] = projected_value
        return projected

    def _annotate_hierarchical_stats(self, stats: Dict[str, torch.Tensor], hidden: torch.Tensor) -> None:
        device = hidden.device
        stats["hierarchical_level"] = torch.tensor(float(self.level), device=device)
        stats["hierarchical_child_count"] = torch.tensor(float(len(self.child_nests)), device=device)
        stats["hierarchical_local_d_model"] = torch.tensor(float(self.d_model), device=device)
        stats["hierarchical_external_d_model"] = torch.tensor(float(self.external_d_model), device=device)
        stats["hierarchical_torus_depth"] = torch.tensor(float(self.effective_torus_depth), device=device)
        stats["hierarchical_torus_height"] = torch.tensor(float(self.effective_torus_height), device=device)
        stats["hierarchical_torus_width"] = torch.tensor(float(self.effective_torus_width), device=device)
        stats["hierarchical_recursive_depth"] = torch.tensor(float(self.effective_recursive_depth), device=device)
        stats["hierarchical_fixed_point_iterations"] = torch.tensor(float(self.effective_fixed_point_iterations), device=device)

    def _make_mini_torus_cfg(self, parent_cfg: PrismalWaveConfig, scale: float = 0.25) -> PrismalWaveConfig:
        mini = PrismalWaveConfig.from_dict(parent_cfg.to_dict())
        scale = float(scale) if scale > 0 else 0.25
        target_d_model = max(1, int(getattr(parent_cfg, "family_specialist_d_model", 256)))

        def _scale_int(value: int, minimum: int = 1) -> int:
            scaled = max(minimum, int(round(value * scale)))
            return max(minimum, min(int(value), scaled))

        mini.torus_depth = max(2, _scale_int(parent_cfg.torus_depth, 2))
        mini.torus_height = max(2, _scale_int(parent_cfg.torus_height, 2))
        mini.torus_width = max(2, _scale_int(parent_cfg.torus_width, 2))
        mini.d_model = max(16, min(int(parent_cfg.d_model), target_d_model))
        mini.n_emitters = max(8, _scale_int(parent_cfg.n_emitters, 8))
        mini.n_slots = max(4, _scale_int(parent_cfg.n_slots, 4))
        mini.n_paths = max(1, _scale_int(parent_cfg.n_paths, 1))
        mini.top_k_emitters = max(1, min(mini.n_emitters, _scale_int(parent_cfg.top_k_emitters, 1)))
        mini.top_k_slots = max(1, min(mini.n_slots, _scale_int(parent_cfg.top_k_slots, 1)))
        mini.use_recursive_hmoe = False
        mini.recursive_hmoe_depth = 1
        mini.recursive_hmoe_branching = 1
        mini.use_hmote = False
        mini.use_fixed_point_solver = False
        mini.use_chunk_solver_training = False
        mini.use_mixture_of_torus = False
        mini.use_torus_race_lanes = False
        return mini

    def _get_family_specialist(self, family_id: int) -> Optional[_FamilyTorusSpecialist]:
        if self.level != 0 or not self.per_family_torus_enabled or len(self.family_specialists) == 0:
            return None
        if family_id < 0 or family_id >= self.family_budget:
            return None
        key = str(int(family_id))
        if key not in self.per_family_tori:
            self.per_family_tori[key] = self.family_specialists[int(family_id) % len(self.family_specialists)]
        return self.per_family_tori[key]

    def _slice_batch_context(
        self,
        context: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Safe slicing helper used by family specialists."""
        if context is None or not torch.is_tensor(context) or context.numel() == 0:
            return None
        if context.dim() >= 1 and context.shape[0] == mask.shape[0]:
            return context[mask]
        return context

    def _batch_family_ids(
        self,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return the last-token family ids for each sample."""
        src = signature_family_ids if signature_family_ids is not None else signature_ids
        if src is None or not torch.is_tensor(src) or src.numel() == 0:
            return None
        if src.dtype.is_floating_point:
            return None
        family_ids = src.long()
        if family_ids.dim() >= 2:
            return family_ids[:, -1].reshape(-1)
        return family_ids.reshape(-1)

    def gate_plan(
        self,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        path_index: Optional[int] = None,
        route_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Tuple[int, ...] | float | bool]:
        if not self.use_gate:
            return {
                "family_ids": (),
                "expert_ids": (),
                "confidence": 0.0,
                "offload_to_cpu": False,
            }
        family_ids = self._batch_family_ids(signature_family_ids=signature_family_ids)
        family_values = family_ids.detach().long().reshape(-1).tolist() if family_ids is not None else []
        resident_families = []
        for value in family_values:
            resident_families.append(int(value) % len(self.family_specialists) if self.family_specialists else 0)
        resident_families = tuple(dict.fromkeys(resident_families))[: self.gate_residency_budget]
        expert_ids: Tuple[int, ...] = ()
        if self.experts:
            if route_stats is not None and "selected_path_index" in route_stats:
                selected = route_stats["selected_path_index"]
                if torch.is_tensor(selected) and selected.numel() > 0:
                    expert_ids = tuple(
                        dict.fromkeys(
                            [int(value) % len(self.experts) for value in selected.detach().long().reshape(-1).tolist()]
                        )
                    )[: self.gate_residency_budget]
            elif path_index is not None:
                expert_ids = (int(path_index) % len(self.experts),)
            else:
                expert_ids = (0,)
        confidence = 0.35
        if route_stats is not None and "signature_agreement" in route_stats:
            agreement = route_stats["signature_agreement"]
            if torch.is_tensor(agreement) and agreement.numel() > 0:
                confidence = float(agreement.detach().float().mean().clamp(0.0, 1.0).item())
        return {
            "family_ids": resident_families,
            "expert_ids": expert_ids,
            "confidence": confidence,
            "offload_to_cpu": self.gate_offload_to_cpu,
        }

    def gate_apply_residency(
        self,
        *,
        device: torch.device,
        family_ids: Sequence[int] = (),
        expert_ids: Sequence[int] = (),
        offload_to_cpu: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if not self.use_gate:
            return {}
        stats: Dict[str, torch.Tensor] = {}
        resident_families = {int(value) for value in family_ids}
        resident_experts = {int(value) for value in expert_ids}
        cpu_device = torch.device("cpu")
        def _module_device(module: nn.Module) -> torch.device:
            param = next(module.parameters(), None)
            return param.device if param is not None else device
        if self.family_specialists:
            for index, specialist in enumerate(self.family_specialists):
                target_device = device if index in resident_families else (cpu_device if offload_to_cpu else _module_device(specialist))
                specialist.to(target_device)
        if self.experts:
            for index, expert in enumerate(self.experts):
                target_device = device if index in resident_experts else (cpu_device if offload_to_cpu else _module_device(expert))
                expert.to(target_device)
        stats["gate_family_resident_count"] = torch.tensor(float(len(resident_families)), device=device)
        stats["gate_expert_resident_count"] = torch.tensor(float(len(resident_experts)), device=device)
        stats["gate_offload_to_cpu"] = torch.tensor(1.0 if offload_to_cpu else 0.0, device=device)
        return stats

    def _family_specialist_stats(
        self,
        hidden: torch.Tensor,
        *,
        batch_family_ids: Optional[torch.Tensor] = None,
        specialist_prob: Optional[torch.Tensor] = None,
        active_family_count: int = 0,
    ) -> Dict[str, torch.Tensor]:
        device = hidden.device
        unique_family_count = 0.0
        if batch_family_ids is not None and torch.is_tensor(batch_family_ids) and batch_family_ids.numel() > 0:
            unique_family_count = float(torch.unique(batch_family_ids.detach().long()).numel())
        gate_mean = (
            specialist_prob.mean().detach().to(device=device)
            if torch.is_tensor(specialist_prob) and specialist_prob.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        unique_families = max(unique_family_count, 1.0) if active_family_count > 0 else unique_family_count
        hit_rate = float(active_family_count / max(unique_families, 1.0)) if unique_family_count > 0 else 0.0
        bank_size = float(len(self.family_specialists))
        return {
            "family_specialist_active_count": torch.tensor(float(active_family_count), device=device),
            "family_specialist_unique_families": torch.tensor(float(unique_family_count), device=device),
            "family_specialist_bank_size": torch.tensor(bank_size, device=device),
            "family_specialist_capacity": torch.tensor(bank_size, device=device),
            "family_specialist_budget": torch.tensor(float(self.family_budget), device=device),
            "family_specialist_max_families_per_nest": torch.tensor(float(self.max_families_per_nest), device=device),
            "family_specialist_hit_rate": torch.tensor(float(hit_rate), device=device),
            "family_specialist_gate_mean": gate_mean,
        }

    def _leaf_token_mask(
        self,
        signature_level_ids: Optional[torch.Tensor],
        hidden: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return a boolean mask for char/piece tokens."""
        if not self.is_leaf_nest or not self.byte_tier:
            return None
        if signature_level_ids is None or not torch.is_tensor(signature_level_ids) or signature_level_ids.numel() == 0:
            return None
        if signature_level_ids.dtype.is_floating_point:
            return None
        mask = torch.zeros_like(signature_level_ids, dtype=torch.bool)
        for level_id in self._leaf_level_ids:
            mask = mask | (signature_level_ids == level_id)
        if hidden.dim() == 3 and mask.dim() == 2 and mask.shape[1] != hidden.shape[1]:
            mask = mask.reshape(mask.size(0), -1).any(dim=-1, keepdim=True).expand(-1, hidden.size(1))
        elif hidden.dim() == 3 and mask.dim() == 1:
            mask = mask.unsqueeze(1).expand(-1, hidden.size(1))
        elif hidden.dim() == 2 and mask.dim() >= 2:
            mask = mask.reshape(mask.size(0), -1).any(dim=-1)
        return mask

    def _apply_family_specialists(
        self,
        hidden: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        path_index: int = 0,
        step_index_offset: int = 0,
        use_solver: bool = False,
        relay_mode: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Vectorized family specialist routing with cheap gate."""
        batch_family_ids = self._batch_family_ids(signature_family_ids=signature_family_ids, signature_ids=signature_ids)
        if self.level != 0 or not self.per_family_torus_enabled or len(self.family_specialists) == 0:
            return hidden, self._family_specialist_stats(hidden, batch_family_ids=batch_family_ids)
        if batch_family_ids is None or batch_family_ids.numel() == 0:
            return hidden, self._family_specialist_stats(hidden)

        gate_logits = self.family_gate(hidden.mean(dim=1) if hidden.dim() == 3 else hidden)
        specialist_prob = torch.sigmoid(gate_logits).squeeze(-1)
        if specialist_prob.numel() == 0 or not bool((specialist_prob > self.family_specialist_gate_threshold).any().item()):
            stats = self._family_specialist_stats(
                hidden,
                batch_family_ids=batch_family_ids,
                specialist_prob=specialist_prob,
            )
            if self.family_usage_ema is not None and self.family_usage_ema.numel() > 0:
                stats["family_usage_ema_mean"] = self.family_usage_ema.mean().detach().to(device=hidden.device)
            return hidden, stats

        residual = torch.zeros_like(hidden)
        unique_families, inverse = torch.unique(batch_family_ids.detach().long(), return_inverse=True)
        active_family_count = 0
        for family_index, family_id in enumerate(unique_families.tolist()):
            family_value = int(family_id)
            specialist = self._get_family_specialist(family_value)
            if specialist is None:
                continue
            mask = inverse.eq(family_index)
            if not bool(mask.any().item()):
                continue
            active_family_count += 1
            if hidden.dim() == 3:
                router_hidden = hidden[mask].mean(dim=1)
            else:
                router_hidden = hidden[mask]
            if family_context is not None and torch.is_tensor(family_context) and family_context.numel() > 0:
                family_ctx = self._slice_batch_context(family_context, mask)
                if family_ctx is not None and family_ctx.dim() >= 2:
                    family_ctx_summary = family_ctx.mean(dim=1) if family_ctx.dim() >= 3 else family_ctx
                    router_hidden = 0.5 * router_hidden + 0.5 * family_ctx_summary
            router_logits = self.family_specialist_router(router_hidden) if self.family_specialist_router is not None else None
            if router_logits is not None and router_logits.numel() > 0:
                router_probs = torch.softmax(router_logits, dim=-1)
                specialist_index = int(router_probs.mean(dim=0).argmax().item())
                specialist = self.family_specialists[specialist_index]
            specialist_kwargs = {
                "signature_family_ids": signature_family_ids[mask] if signature_family_ids is not None else None,
                "signature_ids": signature_ids[mask] if signature_ids is not None else None,
                "signature_level_ids": signature_level_ids[mask] if signature_level_ids is not None else None,
                "signature_relation_ids": signature_relation_ids[mask] if signature_relation_ids is not None else None,
                "parent_signature_ids": parent_signature_ids[mask] if parent_signature_ids is not None else None,
                "registry_context": self._slice_batch_context(registry_context, mask),
                "family_context": self._slice_batch_context(family_context, mask),
                "level_context": self._slice_batch_context(level_context, mask),
                "relation_context": self._slice_batch_context(relation_context, mask),
                "parent_context": self._slice_batch_context(parent_context, mask),
            }
            specialist_out = specialist(
                hidden[mask],
                **specialist_kwargs,
                path_index=path_index,
                step_index_offset=step_index_offset,
                use_solver=use_solver,
                relay_mode=relay_mode,
            )
            residual[mask] = specialist_out.to(dtype=residual.dtype)
            if self.family_usage_ema is not None and family_value < self.family_usage_ema.numel():
                with torch.no_grad():
                    count = mask.float().sum().to(self.family_usage_ema.device)
                    self.family_usage_ema[family_value] = 0.95 * self.family_usage_ema[family_value] + 0.05 * count

        if active_family_count == 0:
            return hidden, {}

        mixed = hidden + self.family_specialist_weight * residual
        stats = self._family_specialist_stats(
            hidden,
            batch_family_ids=batch_family_ids,
            specialist_prob=specialist_prob,
            active_family_count=active_family_count,
        )
        if self.family_usage_ema is not None and self.family_usage_ema.numel() > 0:
            stats["family_usage_ema_mean"] = self.family_usage_ema.mean().detach().to(device=hidden.device)
        return mixed, stats

    def _apply_leaf_cells(
        self,
        hidden: torch.Tensor,
        *,
        signature_level_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Vectorized leaf cell bank with optional cheap router."""
        if not self.leaf_cell_enabled or not self.leaf_cells or len(self.leaf_cells) == 0:
            return hidden, {}
        mask = self._leaf_token_mask(signature_level_ids, hidden)
        if mask is None or not bool(mask.any().item()):
            return hidden, {}
        if self.leaf_router is not None:
            router_logits = self.leaf_router(hidden)
            router_weights = F.softmax(router_logits, dim=-1)
            router_confidence = router_weights.max(dim=-1).values
            if self.leaf_router_confidence_threshold > 0 and float(router_confidence.mean().item()) < self.leaf_router_confidence_threshold:
                stats = {
                    "leaf_cell_active_fraction": torch.tensor(0.0, device=hidden.device),
                    "leaf_cell_bank_size": torch.tensor(float(len(self.leaf_cells)), device=hidden.device),
                    "leaf_router_confidence_mean": router_confidence.mean().detach().to(device=hidden.device),
                }
                return hidden, stats
            cell_outputs = [cell(hidden) for cell in self.leaf_cells]
            stacked = torch.stack(cell_outputs, dim=-1)
            leaf_hidden = torch.sum(stacked * router_weights.unsqueeze(-2), dim=-1)
        else:
            cell_outputs = [cell(hidden) for cell in self.leaf_cells]
            leaf_hidden = torch.stack(cell_outputs, dim=0).mean(dim=0)
        if mask.dim() == hidden.dim() - 1:
            mask = mask.unsqueeze(-1).expand_as(hidden)
        elif mask.dim() == 1 and hidden.dim() == 3:
            mask = mask.unsqueeze(1).expand(-1, hidden.size(1))
        elif mask.dim() < hidden.dim():
            mask = mask.unsqueeze(-1).expand_as(hidden)
        mixed = hidden + self.leaf_cell_weight * (leaf_hidden * mask.to(dtype=hidden.dtype))
        stats = {
            "leaf_cell_active_fraction": mask.float().mean().detach(),
            "leaf_cell_bank_size": torch.tensor(float(len(self.leaf_cells)), device=hidden.device),
        }
        if self.leaf_router is not None:
            stats["leaf_router_confidence_mean"] = router_confidence.mean().detach().to(device=hidden.device)
        return mixed, stats

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            core = self._modules.get("core")
            if core is not None and hasattr(core, name):
                return getattr(core, name)
            raise

    def _pool_field_for_child(
        self,
        field_state: torch.Tensor | PrismalTorusState,
        child: "HierarchicalParameterNest",
    ) -> torch.Tensor | PrismalTorusState:
        field_tensor = field_state.field if isinstance(field_state, PrismalTorusState) else field_state
        if field_tensor.dim() != 5:
            return child._project_state_in(field_state)
        if field_tensor.shape[1:4] == (child.depth, child.height, child.width):
            return child._project_state_in(field_state)
        pooled = F.adaptive_avg_pool3d(
            field_tensor.permute(0, 4, 1, 2, 3),
            output_size=(child.depth, child.height, child.width),
        )
        pooled_field = pooled.permute(0, 2, 3, 4, 1).contiguous()
        if isinstance(field_state, PrismalTorusState):
            return child._project_state_in(PrismalTorusState(field=pooled_field, bus=field_state.bus))
        return child._project_state_in(pooled_field)

    def _resize_field_state(self, field_state: torch.Tensor, depth: int, height: int, width: int) -> torch.Tensor:
        if field_state.dim() != 5 or field_state.shape[1:4] == (depth, height, width):
            return field_state
        resized = F.interpolate(
            field_state.permute(0, 4, 1, 2, 3),
            size=(depth, height, width),
            mode="trilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 4, 1).contiguous()

    def _mot_router_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.mot_router is None:
            if hidden.dim() == 3:
                batch = hidden.size(0)
            else:
                batch = hidden.size(0)
            return torch.ones(batch, 1, device=hidden.device, dtype=hidden.dtype)
        router_in = hidden.mean(dim=1) if hidden.dim() == 3 else hidden
        logits = self.mot_router(router_in) / max(self.mot_routing_temperature, 1e-6)
        return F.softmax(logits, dim=-1)

    def _mix_mot_hidden(
        self,
        hidden: torch.Tensor,
        *,
        router_source: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_mot or not self.experts or self.mot_router is None:
            return hidden, {}

        router_input = router_source if router_source is not None else hidden
        weights = self._mot_router_weights(router_input)
        base_weight = weights[:, 0]
        if hidden.dim() == 3:
            mixed = hidden * base_weight.view(-1, 1, 1)
        elif hidden.dim() == 2:
            mixed = hidden * base_weight.view(-1, 1)
        else:
            mixed = hidden * base_weight.view(-1, *([1] * (hidden.dim() - 1)))

        stats: Dict[str, torch.Tensor] = {
            "mot_base_weight_mean": base_weight.mean().detach(),
        }
        if weights.size(-1) > 1:
            stats["mot_expert_weight_mean"] = weights[:, 1:].mean().detach()

        active_experts = len(self.experts)
        if self.use_topk_mot and weights.size(-1) > 1:
            expert_weights = weights[:, 1:]
            top_k = min(max(1, self.mot_top_k), expert_weights.size(-1))
            top_vals, top_idx = torch.topk(expert_weights, k=top_k, dim=-1)
            flat_tokens = torch.arange(hidden.size(0), device=hidden.device).repeat_interleave(top_k)
            flat_experts = top_idx.reshape(-1)
            flat_weights = top_vals.reshape(-1)
            sorted_experts, order = torch.sort(flat_experts)
            sorted_tokens = flat_tokens.index_select(0, order)
            sorted_weights = flat_weights.index_select(0, order)
            expert_counts = torch.bincount(sorted_experts, minlength=len(self.experts))
            active_experts = int((expert_counts > 0).sum().item())
            residual = torch.zeros_like(hidden)
            offset = 0
            for expert_idx, expert in enumerate(self.experts):
                count = int(expert_counts[expert_idx].item())
                if count <= 0:
                    continue
                next_offset = offset + count
                token_indices = sorted_tokens[offset:next_offset]
                route_weights = sorted_weights[offset:next_offset]
                expert_hidden = expert(hidden.index_select(0, token_indices))
                if expert_hidden.dim() == 3:
                    expert_update = route_weights.view(-1, 1, 1) * expert_hidden
                elif expert_hidden.dim() == 2:
                    expert_update = route_weights.view(-1, 1) * expert_hidden
                else:
                    expert_update = route_weights.view(-1, *([1] * (expert_hidden.dim() - 1))) * expert_hidden
                residual.index_add_(0, token_indices, expert_update.to(dtype=residual.dtype))
                offset = next_offset
            mixed = mixed + residual
        else:
            for expert_idx, expert in enumerate(self.experts):
                expert_hidden = expert(hidden)
                expert_weight = weights[:, min(expert_idx + 1, weights.size(-1) - 1)]
                if expert_hidden.dim() == 3:
                    expert_update = expert_weight.view(-1, 1, 1) * expert_hidden
                elif expert_hidden.dim() == 2:
                    expert_update = expert_weight.view(-1, 1) * expert_hidden
                else:
                    expert_update = expert_weight.view(-1, *([1] * (expert_hidden.dim() - 1))) * expert_hidden
                mixed = mixed + expert_update.to(dtype=mixed.dtype)

        stats["mot_num_experts"] = torch.tensor(float(len(self.experts)), device=hidden.device)
        stats["mot_active_experts"] = torch.tensor(float(active_experts), device=hidden.device)
        stats["mot_routing_temperature"] = torch.tensor(float(self.mot_routing_temperature), device=hidden.device)
        return mixed, stats

    def _run_mot_core(
        self,
        hidden: torch.Tensor,
        field_state: torch.Tensor,
        *,
        use_solver: bool = False,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        path_index: int = 0,
        step_index_offset: int = 0,
        relay_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.use_mot or len(self.experts) <= 1:
            if use_solver:
                return self.core.chunk_solver_step(
                    hidden,
                    field_state,
                    signature_family_ids=signature_family_ids,
                    signature_ids=signature_ids,
                    signature_level_ids=signature_level_ids,
                    signature_relation_ids=signature_relation_ids,
                    parent_signature_ids=parent_signature_ids,
                    registry_context=registry_context,
                    family_context=family_context,
                    level_context=level_context,
                    relation_context=relation_context,
                    parent_context=parent_context,
                    path_index=path_index,
                    step_index_offset=step_index_offset,
                )
            return self.core.chunked_step(
                hidden,
                field_state,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                registry_context=registry_context,
                family_context=family_context,
                level_context=level_context,
                relation_context=relation_context,
                parent_context=parent_context,
                path_index=path_index,
                step_index_offset=step_index_offset,
                relay_mode=relay_mode,
            )

        base_step = self.core.chunk_solver_step if use_solver else self.core.chunked_step
        base_kwargs = dict(
            signature_level_ids=signature_level_ids,
            registry_context=registry_context,
            family_context=family_context,
            level_context=level_context,
            relation_context=relation_context,
            parent_context=parent_context,
            path_index=path_index,
            step_index_offset=step_index_offset,
        )
        if not use_solver:
            base_kwargs["relay_mode"] = relay_mode
        base_hidden, base_field, base_stats = base_step(
            hidden,
            field_state,
            **base_kwargs,
        )
        mixed_hidden, mot_stats = self._mix_mot_hidden(base_hidden, router_source=base_hidden)
        for key, value in mot_stats.items():
            base_stats[f"mot_{key}"] = value
            if key == "mot_active_experts":
                base_stats["mot_active_experts"] = value
        return mixed_hidden, base_field, base_stats

    def init_state(self, *args, **kwargs):
        return self.core.init_state(*args, **kwargs)

    def transition(self, *args, **kwargs):
        return self.core.transition(*args, **kwargs)

    def solve_transition(self, *args, **kwargs):
        return self.core.solve_transition(*args, **kwargs)

    def solve_chunk_transition(self, *args, **kwargs):
        return self.core.solve_chunk_transition(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.core.step(*args, **kwargs)

    def _is_leaf_low_level(self, signature_level_ids: Optional[torch.Tensor]) -> bool:
        if not self.is_leaf_nest:
            return False
        if signature_level_ids is None or not torch.is_tensor(signature_level_ids) or signature_level_ids.numel() == 0:
            return not self.byte_tier
        if signature_level_ids.dtype.is_floating_point:
            return not self.byte_tier
        if not self.byte_tier:
            return True
        low_level_ids = [level_id for level_id in self._leaf_level_ids if level_id is not None]
        if not low_level_ids:
            return False
        mask = torch.zeros_like(signature_level_ids, dtype=torch.bool)
        for level_id in low_level_ids:
            mask = mask | signature_level_ids.eq(level_id)
        return bool(mask.any().item())

    def condition_hidden(
        self,
        hidden: torch.Tensor,
        *,
        signature_level_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        runtime_enabled = bool(getattr(self, "_hierarchical_precision_runtime_enabled", False))
        with self._precision_scope(hidden.device, enabled=runtime_enabled):
            hidden = self._project_hidden_in(hidden)
            return torch.tanh(self.local_adapter(hidden))

    def route_tokens(
        self,
        hidden: torch.Tensor,
        *,
        path_index: int = 0,
        layer_index: int = 0,
        signature_level_ids: Optional[torch.Tensor] = None,
        torus_center: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del path_index, layer_index, kwargs
        runtime_enabled = bool(getattr(self, "_hierarchical_precision_runtime_enabled", False))
        with self._precision_scope(hidden.device, enabled=runtime_enabled):
            local_hidden = self._project_hidden_in(hidden)
            out = self.condition_hidden(local_hidden, signature_level_ids=signature_level_ids)
            gate = torch.sigmoid(self.tier_gate(local_hidden))
            if torus_center is not None and torch.is_tensor(torus_center) and torus_center.numel() > 0:
                torus_center = self._project_hidden_in(torus_center)
                if torus_center.shape[-1] == out.shape[-1]:
                    out = out + 0.02 * torch.tanh(torus_center)
                else:
                    out = out + 0.02 * torch.tanh(torus_center).mean(dim=-1, keepdim=True).expand_as(out)
            return self._project_hidden_out(gate * out)

    def _blend_child_outputs(
        self,
        hidden: torch.Tensor,
        field_state: torch.Tensor,
        *args,
        use_solver: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        hidden = self._project_hidden_in(hidden)
        field_state = self._project_state_in(field_state)
        kwargs = self._project_context_kwargs(kwargs)
        base_hidden, base_field, base_stats = self._run_mot_core(
            hidden,
            field_state,
            use_solver=use_solver,
            **kwargs,
        )
        out_hidden, out_field, out_stats = self._apply_post_core_layers(
            base_hidden,
            base_field,
            base_stats,
            *args,
            use_solver=use_solver,
            **kwargs,
        )
        self._annotate_hierarchical_stats(out_stats, out_hidden)
        return self._project_hidden_out(out_hidden), out_field, out_stats

    def _apply_post_core_layers(
        self,
        base_hidden: torch.Tensor,
        base_field: torch.Tensor,
        base_stats: Dict[str, torch.Tensor],
        *args,
        use_solver: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        base_hidden, specialist_stats = self._apply_family_specialists(
            base_hidden,
            signature_family_ids=kwargs.get("signature_family_ids"),
            signature_ids=kwargs.get("signature_ids"),
            signature_level_ids=kwargs.get("signature_level_ids"),
            signature_relation_ids=kwargs.get("signature_relation_ids"),
            parent_signature_ids=kwargs.get("parent_signature_ids"),
            registry_context=kwargs.get("registry_context"),
            family_context=kwargs.get("family_context"),
            level_context=kwargs.get("level_context"),
            relation_context=kwargs.get("relation_context"),
            parent_context=kwargs.get("parent_context"),
            path_index=int(kwargs.get("path_index", 0)),
            step_index_offset=int(kwargs.get("step_index_offset", 0)),
            use_solver=use_solver,
            relay_mode=bool(kwargs.get("relay_mode", False)),
        )
        if specialist_stats:
            for key, value in specialist_stats.items():
                base_stats[f"specialist_{key}"] = value
        base_hidden, leaf_stats = self._apply_leaf_cells(
            base_hidden,
            signature_level_ids=kwargs.get("signature_level_ids"),
        )
        if leaf_stats:
            for key, value in leaf_stats.items():
                base_stats[f"leaf_{key}"] = value

        if not self.child_nests or not self.training:
            return base_hidden, base_field, base_stats

        child_hidden_terms: List[torch.Tensor] = []
        child_stat_terms: Dict[str, List[torch.Tensor]] = {}
        for child in self.child_nests:
            child_field_state = self._pool_field_for_child(base_field, child)
            child_hidden, _child_field, child_stats = child._blend_child_outputs(
                base_hidden,
                child_field_state,
                *args,
                use_solver=use_solver,
                **kwargs,
            )
            child_hidden_terms.append(child_hidden)
            for key, value in child_stats.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    child_stat_terms.setdefault(f"hierarchical_{key}", []).append(value.float())

        if child_hidden_terms:
            base_hidden = base_hidden + self.child_mix_weight * torch.stack(child_hidden_terms).mean(dim=0)
        for key, values in child_stat_terms.items():
            if values:
                base_stats[key] = torch.stack(values).mean()
        base_stats["hierarchical_level"] = torch.tensor(float(self.level), device=base_hidden.device)
        base_stats["hierarchical_child_count"] = torch.tensor(float(len(self.child_nests)), device=base_hidden.device)
        return base_hidden, base_field, base_stats

    def main_core_step(
        self,
        hidden_chunk: torch.Tensor,
        field_state: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return self._run_mot_core(hidden_chunk, field_state, *args, **kwargs)

    def overlay_step(
        self,
        base_hidden: torch.Tensor,
        base_field: torch.Tensor,
        base_stats: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return self._apply_post_core_layers(base_hidden, base_field, base_stats, *args, **kwargs)

    def chunked_step(
        self,
        hidden_chunk: torch.Tensor,
        field_state: torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, PrismalTorusState, Dict[str, torch.Tensor]]:
        return self._blend_child_outputs(hidden_chunk, field_state, *args, **kwargs, use_solver=False)

    def chunk_solver_step(
        self,
        hidden_chunk: torch.Tensor,
        field_state: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return self._blend_child_outputs(hidden_chunk, field_state, *args, **kwargs, use_solver=True)

    def forward(
        self,
        x: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        squeeze_output = torch.is_tensor(x) and x.dtype not in (torch.int64, torch.long) and x.dim() == 2
        signature_family_ids = kwargs.pop("signature_family_ids", signature_family_ids)
        signature_ids = kwargs.pop("signature_ids", signature_ids)
        signature_relation_ids = kwargs.pop("signature_relation_ids", signature_relation_ids)
        parent_signature_ids = kwargs.pop("parent_signature_ids", parent_signature_ids)
        path_index = int(kwargs.pop("path_index", 0))
        step_index_offset = int(kwargs.pop("step_index_offset", 0))
        relay_mode = bool(kwargs.pop("relay_mode", False))
        use_solver = bool(kwargs.pop("use_solver", False))

        if torch.is_tensor(x) and x.dtype in (torch.int64, torch.long):
            x = self.shared_embedding(x)
        runtime_enabled = bool(getattr(self, "_hierarchical_precision_runtime_enabled", False))
        with self._precision_scope(x.device, enabled=runtime_enabled):
            hidden = self.condition_hidden(x, signature_level_ids=signature_level_ids)
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(1)
            if hidden.dim() != 3:
                raise ValueError("HierarchicalParameterNest.forward expects hidden shape (batch, seq_len, d_model) or (batch, d_model)")

            field_state = self.core.init_state(hidden.size(0), hidden.device)
            hidden, _field_state, _stats = self._blend_child_outputs(
                hidden,
                field_state,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                path_index=path_index,
                step_index_offset=step_index_offset,
                relay_mode=relay_mode,
                use_solver=use_solver,
                **kwargs,
            )
            return hidden.squeeze(1) if squeeze_output else hidden


class FactorizedEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        embedding_dim: int,
        *,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.quantization_config = quantization_config or QuantizationConfig()
        self.embed = create_quantized_embedding(
            vocab_size,
            embedding_dim,
            quantization_config=self.quantization_config,
        )
        self.proj = create_quantized_linear(
            embedding_dim,
            d_model,
            bias=False,
            quantization_config=self.quantization_config,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(input_ids))

    def resize_vocab(self, new_vocab_size: int) -> None:
        old_embed = self.embed
        if new_vocab_size <= old_embed.num_embeddings:
            return
        new_embed = create_quantized_embedding(
            new_vocab_size,
            old_embed.embedding_dim,
            quantization_config=self.quantization_config,
        ).to(old_embed.weight.device)
        with torch.no_grad():
            dense_weight = new_embed.weight.detach()
            dense_weight[: old_embed.num_embeddings].copy_(old_embed.weight.detach())
            nn.init.normal_(dense_weight[old_embed.num_embeddings :], mean=0.0, std=0.02)
            _assign_dense_weight(new_embed, dense_weight)
        self.embed = new_embed


class DynamicPositionEmbedding(nn.Module):
    """Position embedding that can grow with sequence length instead of hard-capping it."""

    def __init__(
        self,
        initial_size: int,
        d_model: int,
        *,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.quantization_config = quantization_config or QuantizationConfig()
        self.embedding = self._make_embedding(max(1, int(initial_size)))

    def _make_embedding(self, size: int) -> nn.Embedding:
        size = max(1, int(size))
        if self.quantization_config is not None and self.quantization_config.enabled:
            return create_quantized_embedding(size, self.d_model, quantization_config=self.quantization_config)
        return nn.Embedding(size, self.d_model)

    def _resize(self, new_size: int) -> None:
        new_size = max(1, int(new_size))
        if new_size <= self.embedding.num_embeddings:
            return
        old_embedding = self.embedding
        new_embedding = self._make_embedding(new_size).to(old_embedding.weight.device)
        with torch.no_grad():
            dense_weight = new_embedding.weight.detach()
            dense_weight[: old_embedding.num_embeddings].copy_(old_embedding.weight.detach())
            nn.init.normal_(dense_weight[old_embedding.num_embeddings :], mean=0.0, std=0.02)
            _assign_dense_weight(new_embedding, dense_weight)
        self.embedding = new_embedding

    def ensure_capacity(self, required_size: int) -> None:
        self._resize(required_size)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)


class SignatureEmitterRegistry(nn.Module):
    """Sparse signature family/relation registry with lightweight birth and promotion rules."""

    def __init__(self, cfg: PrismalWaveConfig, quantization_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.quantization_config = quantization_config or QuantizationConfig()
        self.capacity_growth_locked = False
        self.family_share = float(getattr(cfg, "emitter_family_share", 1.0))
        self.level_share = float(getattr(cfg, "emitter_level_share", 0.35))
        self.relation_share = float(getattr(cfg, "emitter_relation_share", 0.50))
        self.parent_share = float(getattr(cfg, "emitter_parent_share", 0.75))
        self.birth_threshold = float(getattr(cfg, "emitter_birth_threshold", 1.0))
        self.promotion_threshold = float(getattr(cfg, "emitter_promotion_threshold", 4.0))
        family_cap = max(8, int(cfg.signature_vocab_size) or 8)
        level_cap = max(1, int(cfg.signature_level_vocab_size) or len(SIGNATURE_LEVEL_IDS))
        relation_cap = max(1, int(cfg.signature_relation_vocab_size) or len(SIGNATURE_RELATION_IDS))
        self.family_embedding = self._make_embedding(family_cap, cfg.d_model)
        self.level_embedding = self._make_embedding(level_cap, cfg.d_model)
        self.relation_embedding = self._make_embedding(relation_cap, cfg.d_model)
        self.parent_embedding = self._make_embedding(family_cap, cfg.d_model)
        self.family_gate_proj = create_quantized_linear(
            cfg.d_model,
            cfg.d_model,
            quantization_config=self.quantization_config,
        )
        self.register_buffer("family_activity", torch.zeros(family_cap, dtype=torch.float32), persistent=True)
        self.register_buffer("family_births", torch.zeros(family_cap, dtype=torch.float32), persistent=True)
        self.register_buffer("family_active_mask", torch.zeros(family_cap, dtype=torch.float32), persistent=True)
        self.register_buffer("level_activity", torch.zeros(level_cap, dtype=torch.float32), persistent=True)
        self.register_buffer("level_active_mask", torch.zeros(level_cap, dtype=torch.float32), persistent=True)
        self.register_buffer("relation_activity", torch.zeros(relation_cap, dtype=torch.float32), persistent=True)
        self.register_buffer("relation_active_mask", torch.zeros(relation_cap, dtype=torch.float32), persistent=True)
        self.register_buffer("parent_activity", torch.zeros(family_cap, dtype=torch.float32), persistent=True)
        self.register_buffer("parent_active_mask", torch.zeros(family_cap, dtype=torch.float32), persistent=True)
        self._seed_special_families()

    def _make_embedding(self, size: int, d_model: int) -> nn.Embedding:
        size = max(1, int(size))
        if self.quantization_config is not None and self.quantization_config.enabled:
            return create_quantized_embedding(size, d_model, quantization_config=self.quantization_config)
        return nn.Embedding(size, d_model)

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
        param = next(module.parameters())
        return param.device, param.dtype

    def _resize_embedding(self, name: str, required_size: int) -> None:
        required_size = max(1, int(required_size))
        embed: nn.Embedding = getattr(self, name)
        if required_size <= embed.num_embeddings:
            return
        device, _dtype = self._module_device_dtype(embed)
        new_embed = self._make_embedding(required_size, embed.embedding_dim).to(device)
        old_weight = getattr(embed, "weight", getattr(embed, "embedding_matrix", None))
        new_weight = getattr(new_embed, "weight", getattr(new_embed, "embedding_matrix", None))
        if old_weight is None or new_weight is None:
            raise AttributeError(f"Unsupported embedding module for resize: {type(embed).__name__}")
        with torch.no_grad():
            new_weight[: embed.num_embeddings].copy_(old_weight)
            nn.init.normal_(new_weight[embed.num_embeddings :], mean=0.0, std=0.02)
        setattr(self, name, new_embed)

    def _resize_buffer(self, name: str, required_size: int) -> None:
        required_size = max(1, int(required_size))
        old_buffer = getattr(self, name)
        if required_size <= old_buffer.numel():
            return
        new_buffer = torch.zeros(required_size, device=old_buffer.device, dtype=old_buffer.dtype)
        new_buffer[: old_buffer.numel()].copy_(old_buffer)
        self.register_buffer(name, new_buffer, persistent=True)

    def _ensure_capacity(self, family_id: int, relation_id: int, level_id: int) -> None:
        grow_family = family_id >= self.family_embedding.num_embeddings
        grow_relation = relation_id >= self.relation_embedding.num_embeddings
        grow_level = level_id >= self.level_embedding.num_embeddings
        if self.training and self.capacity_growth_locked and (grow_family or grow_relation or grow_level):
            raise RuntimeError(
                "Signature registry capacity grew after optimizer init; pre-grow the registry embeddings before training."
            )
        if family_id >= self.family_embedding.num_embeddings:
            self._resize_embedding("family_embedding", family_id + 1)
            self._resize_buffer("family_activity", family_id + 1)
            self._resize_buffer("family_births", family_id + 1)
            self._resize_buffer("family_active_mask", family_id + 1)
        if relation_id >= self.relation_embedding.num_embeddings:
            self._resize_embedding("relation_embedding", relation_id + 1)
            self._resize_buffer("relation_activity", relation_id + 1)
            self._resize_buffer("relation_active_mask", relation_id + 1)
        if level_id >= self.level_embedding.num_embeddings:
            self._resize_embedding("level_embedding", level_id + 1)
            self._resize_buffer("level_activity", level_id + 1)
        if family_id >= self.parent_embedding.num_embeddings:
            self._resize_embedding("parent_embedding", family_id + 1)
            self._resize_buffer("parent_activity", family_id + 1)
            self._resize_buffer("parent_active_mask", family_id + 1)

    def set_capacity_growth_locked(self, locked: bool = True) -> None:
        self.capacity_growth_locked = bool(locked)

    def ensure_capacity_for_sizes(self, family_vocab_size: int, level_vocab_size: int, relation_vocab_size: int) -> None:
        prev_locked = self.capacity_growth_locked
        try:
            self.capacity_growth_locked = False
            self._ensure_capacity(
                max(0, int(family_vocab_size) - 1),
                max(0, int(relation_vocab_size) - 1),
                max(0, int(level_vocab_size) - 1),
            )
        finally:
            self.capacity_growth_locked = prev_locked

    def _seed_special_families(self) -> None:
        with torch.no_grad():
            if bool(getattr(self.cfg, "seed_all_emitter_families", True)):
                seed_activity = float(getattr(self.cfg, "emitter_seed_activity", 0.5))
                self.family_active_mask.fill_(1.0)
                self.family_activity.fill_(seed_activity)
                self.family_births.zero_()
                self.parent_active_mask.fill_(1.0)
                self.parent_activity.fill_(seed_activity)
            else:
                self.family_active_mask[: min(3, self.family_active_mask.numel())] = 1.0
                self.parent_active_mask[: min(3, self.parent_active_mask.numel())] = 1.0
            self.relation_active_mask[: min(3, self.relation_active_mask.numel())] = 1.0

    @property
    def family_vocab_size(self) -> int:
        return int(self.family_embedding.num_embeddings)

    def _touch_ids(self, buffer_name: str, mask_name: str, ids: torch.Tensor) -> None:
        if ids is None or ids.numel() == 0:
            return
        flat = ids.detach().long().reshape(-1)
        flat = flat[flat.ge(0)]
        if flat.numel() == 0:
            return
        unique_ids = torch.unique(flat)
        max_id = int(unique_ids.max().item())
        if buffer_name == "family_activity":
            self._ensure_capacity(max_id, 0, 0)
        elif buffer_name == "relation_activity":
            self._ensure_capacity(0, max_id, 0)
        elif buffer_name == "parent_activity":
            self._ensure_capacity(max_id, 0, 0)
        else:
            self._ensure_capacity(0, 0, max_id)
        activity: torch.Tensor = getattr(self, buffer_name)
        active_mask: torch.Tensor = getattr(self, mask_name)
        with torch.no_grad():
            births = activity[unique_ids].eq(0).to(activity.dtype)
            activity[unique_ids] += 1.0
            active_mask[unique_ids] = 1.0
            if buffer_name == "family_activity":
                self.family_births[unique_ids] += births

    def observe(
        self,
        *,
        family_ids: Optional[torch.Tensor] = None,
        level_ids: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        parent_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self._touch_ids(
            "family_activity",
            "family_active_mask",
            family_ids if family_ids is not None else torch.tensor([], device=self.family_activity.device),
        )
        self._touch_ids(
            "level_activity",
            "level_active_mask",
            level_ids if level_ids is not None else torch.tensor([], device=self.family_activity.device),
        )
        self._touch_ids(
            "relation_activity",
            "relation_active_mask",
            relation_ids if relation_ids is not None else torch.tensor([], device=self.family_activity.device),
        )
        self._touch_ids(
            "parent_activity",
            "parent_active_mask",
            parent_ids if parent_ids is not None else torch.tensor([], device=self.family_activity.device),
        )

    def family_context(self, family_ids: torch.Tensor) -> torch.Tensor:
        family_ids = family_ids.clamp(min=0)
        if family_ids.numel() > 0:
            self._ensure_capacity(int(family_ids.max().item()), 0, 0)
        embed = self.family_embedding(family_ids)
        mask = self.family_active_mask[family_ids].to(embed.dtype).unsqueeze(-1)
        activity = self.family_activity[family_ids].to(embed.dtype).unsqueeze(-1)
        promotion = torch.where(activity >= self.promotion_threshold, torch.ones_like(activity), torch.zeros_like(activity))
        birth = torch.where(activity >= self.birth_threshold, torch.ones_like(activity), torch.zeros_like(activity))
        return embed * (0.6 + 0.4 * mask + 0.1 * promotion) + 0.01 * birth * embed

    def level_context(self, level_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if level_ids is None:
            device, dtype = self._module_device_dtype(self.family_embedding)
            return torch.zeros(0, device=device, dtype=dtype)
        if level_ids.numel() > 0:
            self._ensure_capacity(0, 0, int(level_ids.max().item()))
        embed = self.level_embedding(level_ids.clamp(min=0))
        mask = self.level_active_mask[level_ids.clamp(min=0)].to(embed.dtype).unsqueeze(-1)
        return embed * (0.5 + 0.5 * mask)

    def relation_context(self, relation_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if relation_ids is None:
            device, dtype = self._module_device_dtype(self.family_embedding)
            return torch.zeros(0, device=device, dtype=dtype)
        if relation_ids.numel() > 0:
            self._ensure_capacity(0, int(relation_ids.max().item()), 0)
        embed = self.relation_embedding(relation_ids.clamp(min=0))
        mask = self.relation_active_mask[relation_ids.clamp(min=0)].to(embed.dtype).unsqueeze(-1)
        activity = self.relation_activity[relation_ids.clamp(min=0)].to(embed.dtype).unsqueeze(-1)
        return embed * (0.7 + 0.3 * mask) + 0.01 * activity * embed

    def parent_context(self, parent_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if parent_ids is None:
            device, dtype = self._module_device_dtype(self.family_embedding)
            return torch.zeros(0, device=device, dtype=dtype)
        if parent_ids.numel() > 0:
            self._ensure_capacity(int(parent_ids.max().item()), 0, 0)
        embed = self.parent_embedding(parent_ids.clamp(min=0))
        mask = self.parent_active_mask[parent_ids.clamp(min=0)].to(embed.dtype).unsqueeze(-1)
        activity = self.parent_activity[parent_ids.clamp(min=0)].to(embed.dtype).unsqueeze(-1)
        return embed * (0.55 + 0.45 * mask) + 0.01 * activity * embed

    def compose_channels(
        self,
        reference: Optional[torch.Tensor] = None,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del reference
        family_ids = signature_family_ids if signature_family_ids is not None else signature_ids
        parent_ids = parent_signature_ids if parent_signature_ids is not None else family_ids
        channels: Dict[str, torch.Tensor] = {
            "family": self.family_context(family_ids) if family_ids is not None and family_ids.numel() > 0 else torch.zeros(0, device=self._module_device_dtype(self.family_embedding)[0], dtype=self._module_device_dtype(self.family_embedding)[1]),
            "level": self.level_context(signature_level_ids),
            "relation": self.relation_context(signature_relation_ids),
            "parent": self.parent_context(parent_ids) if parent_ids is not None and parent_ids.numel() > 0 else torch.zeros(0, device=self._module_device_dtype(self.family_embedding)[0], dtype=self._module_device_dtype(self.family_embedding)[1]),
        }
        return channels

    def compose(
        self,
        reference: Optional[torch.Tensor] = None,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        channels = self.compose_channels(
            reference=reference,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )
        tensors: List[torch.Tensor] = []
        if channels["family"].numel() > 0:
            tensors.append(self.family_share * channels["family"])
        if channels["level"].numel() > 0:
            tensors.append(self.level_share * channels["level"])
        if channels["relation"].numel() > 0:
            tensors.append(self.relation_share * channels["relation"])
        if channels["parent"].numel() > 0:
            tensors.append(self.parent_share * channels["parent"])
        if not tensors:
            if reference is not None:
                return torch.zeros_like(reference)
            device, dtype = self._module_device_dtype(self.family_embedding)
            return torch.zeros(0, device=device, dtype=dtype)
        return torch.stack(tensors).sum(dim=0)


class PrismalTorusCore(nn.Module):
    """Persistent toroidal recurrent field with circular wraparound."""

    @staticmethod
    def _coalesce_wrapped_offsets(
        offsets: Sequence[tuple[int, int, int]],
        depth: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unique_offsets: list[tuple[int, int, int]] = []
        inverse_indices: list[int] = []
        offset_to_index: Dict[tuple[int, int, int], int] = {}
        for dz, dy, dx in offsets:
            key = (dz % depth, dy % height, dx % width)
            index = offset_to_index.get(key)
            if index is None:
                index = len(unique_offsets)
                offset_to_index[key] = index
                unique_offsets.append(key)
            inverse_indices.append(index)
        unique_tensor = torch.tensor(unique_offsets, dtype=torch.long)
        inverse_tensor = torch.tensor(inverse_indices, dtype=torch.long)
        return unique_tensor, inverse_tensor

    def __init__(self, cfg: PrismalWaveConfig, quantization_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.quantization_config = quantization_config or QuantizationConfig()
        self.depth = max(2, int(getattr(cfg, "torus_depth", 8)))
        self.height = max(2, int(getattr(cfg, "torus_height", 8)))
        self.width = max(2, int(getattr(cfg, "torus_width", 8)))
        self.local_field_radius = max(1, int(getattr(cfg, "torus_local_field_radius", 1)))
        self.write_radius = self.local_field_radius
        self.scout_read_radius = max(
            1,
            int(getattr(cfg, "torus_scout_read_radius", getattr(cfg, "torus_read_radius", self.local_field_radius))),
        )
        self.read_radius = self.scout_read_radius
        self.global_bus_slots = max(1, int(getattr(cfg, "torus_global_bus_slots", 8)))
        self.global_bus_decay = min(max(float(getattr(cfg, "torus_global_bus_decay", 0.92)), 0.0), 1.0)
        self.global_bus_write_scale = max(float(getattr(cfg, "torus_global_bus_write_scale", 0.35)), 0.0)
        self.relay_write_radius = max(
            self.local_field_radius,
            int(getattr(cfg, "torus_relay_write_radius", self.local_field_radius)),
        )
        self.transport_rate = float(getattr(cfg, "torus_transport", 0.12))
        self.transport_interval = max(1, int(getattr(cfg, "torus_transport_interval", 2)))
        self.write_strength = float(getattr(cfg, "torus_write_strength", 0.35))
        self.inner_temperature = float(getattr(cfg, "torus_inner_temperature", 0.70))
        self.outer_temperature = float(getattr(cfg, "torus_outer_temperature", 1.15))
        self.activity_threshold = float(getattr(cfg, "torus_activity_threshold", 0.08))
        self.active_target_fraction = float(getattr(cfg, "torus_active_target_fraction", 0.25))
        self.active_balance_weight = float(getattr(cfg, "torus_active_balance_weight", 0.02))
        self.use_race_lanes = bool(getattr(cfg, "use_torus_race_lanes", False))
        self.lane_count = max(1, int(getattr(cfg, "torus_lane_count", max(1, cfg.n_paths))))
        self.scout_density = float(getattr(cfg, "torus_scout_density", 0.5))
        self.lane_select_threshold_1 = float(getattr(cfg, "torus_lane_select_threshold_1", 0.45))
        self.lane_select_threshold_2 = float(getattr(cfg, "torus_lane_select_threshold_2", 0.70))
        self.lane_relay_hop_spacing = max(1, int(getattr(cfg, "torus_lane_relay_hop_spacing", 4)))
        self.force_aux_stats = False
        self.use_fixed_point_solver = bool(getattr(cfg, "use_fixed_point_solver", False))
        self.fixed_point_iterations = max(1, int(getattr(cfg, "fixed_point_iterations", 4)))
        self.fixed_point_tolerance = float(getattr(cfg, "fixed_point_tolerance", 1e-4))
        self.fixed_point_relaxation = float(getattr(cfg, "fixed_point_relaxation", 0.5))
        d = cfg.d_model
        self.register_buffer(
            "coord_scale",
            torch.tensor(
                [float(max(self.depth - 1, 1)), float(max(self.height - 1, 1)), float(max(self.width - 1, 1))],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "coord_center",
            torch.tensor(
                [float(max(self.depth - 1, 1)), float(max(self.height - 1, 1)), float(max(self.width - 1, 1))],
                dtype=torch.float32,
            )
            / 2.0,
            persistent=False,
        )
        self.write_coord_proj = create_quantized_linear(d, 3, quantization_config=self.quantization_config)
        self.path_coord_proj = create_quantized_linear(d, 3, quantization_config=self.quantization_config)
        self.stencil_proj = create_quantized_linear(
            d,
            (2 * self.write_radius + 1) ** 3,
            quantization_config=self.quantization_config,
        )
        self.relay_stencil_proj = create_quantized_linear(
            d,
            max(1, (2 * self.relay_write_radius + 1) ** 3),
            quantization_config=self.quantization_config,
        )
        self.write_delta_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.write_gate = create_quantized_linear(d, 1, quantization_config=self.quantization_config)
        self.transport_gate = create_quantized_linear(d, 1, quantization_config=self.quantization_config)
        self.read_proj = create_quantized_linear(d * 3, d, quantization_config=self.quantization_config)
        self.bus_value_proj = create_quantized_linear(d * 2, d, quantization_config=self.quantization_config)
        self.bus_slot_gate_proj = create_quantized_linear(d, self.global_bus_slots, quantization_config=self.quantization_config)
        self.bus_write_proj = create_quantized_linear(d * 2, self.global_bus_slots * d, quantization_config=self.quantization_config)
        self.bus_gate_proj = create_quantized_linear(d, 1, quantization_config=self.quantization_config)
        self.registry_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.family_registry_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.level_registry_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.relation_registry_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.parent_registry_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.path_basis = nn.Parameter(torch.randn(max(1, cfg.n_paths), d) * 0.02)
        self.field_seed = nn.Parameter(torch.randn(self.depth, self.height, self.width, d) * 0.02)
        self.bus_seed = nn.Parameter(torch.randn(self.global_bus_slots, d) * 0.02)
        write_offsets = []
        read_offsets = []
        for dz in range(-self.local_field_radius, self.local_field_radius + 1):
            for dy in range(-self.local_field_radius, self.local_field_radius + 1):
                for dx in range(-self.local_field_radius, self.local_field_radius + 1):
                    write_offsets.append((dz, dy, dx))
        relay_offsets = []
        for dz in range(-self.relay_write_radius, self.relay_write_radius + 1):
            for dy in range(-self.relay_write_radius, self.relay_write_radius + 1):
                for dx in range(-self.relay_write_radius, self.relay_write_radius + 1):
                    relay_offsets.append((dz, dy, dx))
        for dz in range(-self.read_radius, self.read_radius + 1):
            for dy in range(-self.read_radius, self.read_radius + 1):
                for dx in range(-self.read_radius, self.read_radius + 1):
                    read_offsets.append((dz, dy, dx))
        write_offsets_unique, write_offset_groups = self._coalesce_wrapped_offsets(write_offsets, self.depth, self.height, self.width)
        relay_offsets_unique, relay_offset_groups = self._coalesce_wrapped_offsets(relay_offsets, self.depth, self.height, self.width)
        read_offsets_unique, read_offset_groups = self._coalesce_wrapped_offsets(read_offsets, self.depth, self.height, self.width)
        self.register_buffer("write_offsets", write_offsets_unique, persistent=False)
        self.register_buffer("write_offset_groups", write_offset_groups, persistent=False)
        self.register_buffer("relay_write_offsets", relay_offsets_unique, persistent=False)
        self.register_buffer("relay_write_offset_groups", relay_offset_groups, persistent=False)
        self.register_buffer("read_offsets", read_offsets_unique, persistent=False)
        self.register_buffer("read_offset_groups", read_offset_groups, persistent=False)
        self.register_buffer("local_offsets", write_offsets_unique, persistent=False)
        self.register_buffer("local_offset_groups", write_offset_groups, persistent=False)
        self.register_buffer("local_offset_count", torch.tensor(float(write_offsets_unique.size(0))), persistent=False)
        self.write_group_count = int(write_offsets_unique.size(0))
        self.relay_group_count = int(relay_offsets_unique.size(0))
        self.read_group_count = int(read_offsets_unique.size(0))
        transport_kernel = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
        transport_kernel[0, 0, 1, 1, 1] = 1.0
        transport_kernel[0, 0, 0, 1, 1] = 1.0
        transport_kernel[0, 0, 2, 1, 1] = 1.0
        transport_kernel[0, 0, 1, 0, 1] = 1.0
        transport_kernel[0, 0, 1, 2, 1] = 1.0
        transport_kernel[0, 0, 1, 1, 0] = 1.0
        transport_kernel[0, 0, 1, 1, 2] = 1.0
        self.register_buffer("transport_kernel", transport_kernel, persistent=False)
        self.register_buffer("transport_kernel_grouped", transport_kernel.repeat(d, 1, 1, 1, 1), persistent=False)

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        state: Optional[torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> PrismalTorusState:
        state_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(self, "precision_state_dtype", None)
        if not isinstance(state_dtype, torch.dtype):
            state_dtype = torch.float32
        field_state: Optional[torch.Tensor] = None
        bus_state: Optional[torch.Tensor] = None
        if isinstance(state, PrismalTorusState):
            field_state = state.field
            bus_state = state.bus
        elif isinstance(state, dict):
            field_state = state.get("field")
            bus_state = state.get("bus")
            carry_state = state.get("carry")
            if field_state is None and isinstance(carry_state, dict):
                field_state = carry_state.get("field")
                bus_state = carry_state.get("bus", bus_state)
        elif isinstance(state, tuple) and len(state) == 2:
            field_state, bus_state = state
        else:
            field_state = state

        if field_state is not None:
            if field_state.dim() == 4:
                field_state = field_state.unsqueeze(0)
            if field_state.size(0) != batch_size:
                field_state = field_state.expand(batch_size, -1, -1, -1, -1)
            if field_state.size(1) != self.depth:
                if field_state.size(1) == 1:
                    field_state = field_state.expand(batch_size, self.depth, -1, -1, -1)
                else:
                    expanded = field_state[:, :1].expand(batch_size, self.depth, -1, -1, -1).clone()
                    expanded[:, : min(field_state.size(1), self.depth)] = field_state[:, : min(field_state.size(1), self.depth)]
                    field_state = expanded
            if (
                field_state.device == device
                and field_state.size(0) == batch_size
                and field_state.size(1) == self.depth
                and field_state.size(2) == self.height
                and field_state.size(3) == self.width
            ):
                if field_state.dtype != state_dtype:
                    field_state = field_state.to(dtype=state_dtype)
            else:
                field_state = field_state.to(device=device, dtype=state_dtype).clone()
        else:
            field_state = self.field_seed.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(device=device, dtype=state_dtype).clone()

        if bus_state is not None:
            if bus_state.dim() == 2:
                bus_state = bus_state.unsqueeze(0)
            if bus_state.size(0) != batch_size:
                bus_state = bus_state.expand(batch_size, -1, -1)
            if bus_state.size(1) != self.global_bus_slots:
                if bus_state.size(1) == 1:
                    bus_state = bus_state.expand(batch_size, self.global_bus_slots, -1)
                else:
                    expanded_bus = bus_state[:, :1].expand(batch_size, self.global_bus_slots, -1).clone()
                    expanded_bus[:, : min(bus_state.size(1), self.global_bus_slots)] = bus_state[:, : min(bus_state.size(1), self.global_bus_slots)]
                    bus_state = expanded_bus
            if bus_state.size(-1) != field_state.size(-1):
                if bus_state.size(-1) == 1:
                    bus_state = bus_state.expand(batch_size, self.global_bus_slots, field_state.size(-1))
                else:
                    bus_state = bus_state.mean(dim=-1, keepdim=True).expand(batch_size, self.global_bus_slots, field_state.size(-1))
            if (
                bus_state.device == device
                and bus_state.size(0) == batch_size
                and bus_state.size(1) == self.global_bus_slots
            ):
                if bus_state.dtype != state_dtype:
                    bus_state = bus_state.to(dtype=state_dtype)
            else:
                bus_state = bus_state.to(device=device, dtype=state_dtype).clone()
        else:
            bus_state = self.bus_seed.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=state_dtype).clone()

        return PrismalTorusState(field=field_state, bus=bus_state)

    def _gather_patch(
        self,
        field: torch.Tensor,
        center_z: torch.Tensor,
        center_y: torch.Tensor,
        center_x: torch.Tensor,
        radius: int,
    ) -> torch.Tensor:
        batch_size = field.size(0)
        if radius == self.write_radius:
            offsets_tensor = self.write_offsets
        elif radius == self.scout_read_radius:
            offsets_tensor = self.read_offsets
        elif radius == self.relay_write_radius:
            offsets_tensor = self.relay_write_offsets
        else:
            offsets = []
            for dz in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        offsets.append((dz, dy, dx))
            offsets_tensor, _ = self._coalesce_wrapped_offsets(offsets, self.depth, self.height, self.width)
        offsets_tensor = offsets_tensor.to(field.device)
        zz = torch.remainder(center_z.unsqueeze(1) + offsets_tensor[:, 0].view(1, -1), self.depth)
        yy = torch.remainder(center_y.unsqueeze(1) + offsets_tensor[:, 1].view(1, -1), self.height)
        xx = torch.remainder(center_x.unsqueeze(1) + offsets_tensor[:, 2].view(1, -1), self.width)
        batch_idx = torch.arange(batch_size, device=field.device).unsqueeze(1).expand_as(zz)
        gathered = field[batch_idx, zz, yy, xx]
        return gathered

    def _aggregate_stencil_weights(self, stencil_weights: torch.Tensor, offset_groups: torch.Tensor, group_count: int) -> torch.Tensor:
        if stencil_weights.numel() == 0:
            return stencil_weights
        offset_groups = offset_groups.to(device=stencil_weights.device)
        if group_count <= 0:
            return stencil_weights
        if group_count == stencil_weights.size(-1):
            return stencil_weights
        reduced = stencil_weights.new_zeros(stencil_weights.size(0), group_count)
        reduced.scatter_add_(1, offset_groups.view(1, -1).expand(stencil_weights.size(0), -1), stencil_weights)
        return reduced

    def _coord_temperature(self, write_coord: torch.Tensor) -> torch.Tensor:
        coord_scale = self.coord_scale.to(write_coord.device)
        center = self.coord_center.to(write_coord.device)
        wrapped = torch.abs(write_coord - center)
        wrapped = torch.minimum(wrapped, coord_scale - wrapped)
        normalized = wrapped / center.clamp_min(1.0)
        radial = torch.sqrt(torch.clamp((normalized**2).mean(dim=-1), min=0.0, max=1.0))
        return self.inner_temperature + (self.outer_temperature - self.inner_temperature) * radial

    def _summarize_chunk_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None or context.numel() == 0:
            return None
        if context.dim() == 2:
            return context
        if context.dim() >= 3:
            if context.size(1) == 1:
                return context.squeeze(1)
            return 0.5 * context.mean(dim=1) + 0.25 * context[:, 0, :] + 0.25 * context[:, -1, :]
        return context

    def transition(
        self,
        hidden: torch.Tensor,
        field: torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        *,
        path_index: int,
        step_index: int = 0,
        relay_mode: bool = False,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PrismalTorusState, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_size, dim = hidden.shape
        core_hidden_dim = self.write_delta_proj.in_features
        if dim != core_hidden_dim:
            hidden = hidden.mean(dim=-1, keepdim=True).expand(batch_size, core_hidden_dim)
            dim = core_hidden_dim

        torus_state = self.init_state(batch_size, hidden.device, state=field)
        field_tensor = torus_state.field
        bus_tensor = torus_state.bus

        registry_context = self._summarize_chunk_context(registry_context)
        family_context = self._summarize_chunk_context(family_context)
        level_context = self._summarize_chunk_context(level_context)
        relation_context = self._summarize_chunk_context(relation_context)
        parent_context = self._summarize_chunk_context(parent_context)
        base_path = self.path_basis[path_index % self.path_basis.size(0)].view(1, -1).expand(batch_size, -1)
        family_write_floor = float(getattr(self.cfg, "torus_write_family_floor", 0.15))
        registry_bias = self.registry_proj(registry_context) if registry_context is not None and registry_context.numel() > 0 else 0.0
        family_bias = self.family_registry_proj(family_context) if family_context is not None and family_context.numel() > 0 else 0.0
        level_bias = self.level_registry_proj(level_context) if level_context is not None and level_context.numel() > 0 else 0.0
        relation_bias = self.relation_registry_proj(relation_context) if relation_context is not None and relation_context.numel() > 0 else 0.0
        parent_bias = self.parent_registry_proj(parent_context) if parent_context is not None and parent_context.numel() > 0 else 0.0
        path_vector = torch.tanh(
            base_path
            + (registry_bias if isinstance(registry_bias, torch.Tensor) else 0.0)
            + (0.25 + family_write_floor) * (family_bias if isinstance(family_bias, torch.Tensor) else 0.0)
            + 0.50 * (level_bias if isinstance(level_bias, torch.Tensor) else 0.0)
            + 0.50 * (relation_bias if isinstance(relation_bias, torch.Tensor) else 0.0)
            + 0.35 * (parent_bias if isinstance(parent_bias, torch.Tensor) else 0.0)
        )
        path_bias = self.path_coord_proj(path_vector)
        coord_scale = self.coord_scale.to(hidden.device)

        registry_hidden = hidden + (registry_bias if isinstance(registry_bias, torch.Tensor) else 0.0)
        if isinstance(family_bias, torch.Tensor):
            registry_hidden = registry_hidden + (0.35 + family_write_floor) * family_bias
        if isinstance(level_bias, torch.Tensor):
            registry_hidden = registry_hidden + 0.25 * level_bias
        if isinstance(relation_bias, torch.Tensor):
            registry_hidden = registry_hidden + 0.25 * relation_bias
        if isinstance(parent_bias, torch.Tensor):
            registry_hidden = registry_hidden + 0.20 * parent_bias
        write_coord = torch.sigmoid(self.write_coord_proj(registry_hidden) + 0.15 * path_bias) * coord_scale
        center_z = torch.remainder(write_coord[:, 0].floor().long(), self.depth)
        center_y = torch.remainder(write_coord[:, 1].floor().long(), self.height)
        center_x = torch.remainder(write_coord[:, 2].floor().long(), self.width)

        local_temperature = self._coord_temperature(write_coord).clamp_min(0.5)
        temperature_scale = local_temperature.view(batch_size, 1)
        active_radius = self.relay_write_radius if relay_mode else self.local_field_radius
        active_stencil_proj = self.relay_stencil_proj if relay_mode else self.stencil_proj
        active_offsets = self.relay_write_offsets if relay_mode else self.local_offsets
        active_offset_groups = self.relay_write_offset_groups if relay_mode else self.local_offset_groups

        stencil_logits = active_stencil_proj(registry_hidden + 0.1 * path_vector) / temperature_scale
        stencil_weights = F.softmax(stencil_logits, dim=-1)
        active_group_count = self.relay_group_count if relay_mode else self.write_group_count
        stencil_weights = self._aggregate_stencil_weights(stencil_weights, active_offset_groups, active_group_count)
        stencil_floor = 0.20 / max(stencil_weights.size(-1), 1)
        stencil_weights = stencil_weights * 0.80 + stencil_floor
        stencil_weights = stencil_weights / stencil_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        write_gate = torch.sigmoid(self.write_gate(registry_hidden)).view(batch_size, 1) * 0.65 + 0.35
        write_delta = torch.tanh(self.write_delta_proj(registry_hidden + 0.1 * path_vector))

        local_patch = self._gather_patch(field_tensor, center_z, center_y, center_x, active_radius)
        patch_context = (stencil_weights.unsqueeze(-1) * local_patch).sum(dim=1)
        if patch_context.shape[-1] != write_delta.shape[-1]:
            patch_context = patch_context.mean(dim=-1, keepdim=True).expand_as(write_delta)
        write_update = torch.tanh(write_delta + 0.25 * patch_context)
        field_dim = field_tensor.shape[-1]
        if write_update.shape[-1] != field_dim:
            write_update = write_update.mean(dim=-1, keepdim=True).expand(batch_size, field_dim)
        write_strength = self.write_strength / temperature_scale

        next_field = field_tensor.clone()
        offsets_tensor = active_offsets.to(hidden.device)
        zz = torch.remainder(center_z.unsqueeze(1) + offsets_tensor[:, 0].view(1, -1), self.depth)
        yy = torch.remainder(center_y.unsqueeze(1) + offsets_tensor[:, 1].view(1, -1), self.height)
        xx = torch.remainder(center_x.unsqueeze(1) + offsets_tensor[:, 2].view(1, -1), self.width)
        scalar = (write_strength * write_gate).unsqueeze(1) * stencil_weights.unsqueeze(-1)
        update = scalar * write_update.unsqueeze(1)
        field_flat = next_field.view(batch_size, self.depth * self.height * self.width, field_dim)
        update = update.to(dtype=field_flat.dtype)
        linear_idx = (zz * (self.height * self.width) + yy * self.width + xx).unsqueeze(-1)
        field_flat.scatter_add_(1, linear_idx.expand(-1, -1, field_dim), update)
        next_field = field_flat.view(batch_size, self.depth, self.height, self.width, field_dim)

        if step_index % self.transport_interval == 0:
            transport = (
                next_field
                + torch.roll(next_field, shifts=1, dims=1)
                + torch.roll(next_field, shifts=-1, dims=1)
                + torch.roll(next_field, shifts=1, dims=2)
                + torch.roll(next_field, shifts=-1, dims=2)
                + torch.roll(next_field, shifts=1, dims=3)
                + torch.roll(next_field, shifts=-1, dims=3)
            ) / 7.0
            transport_gate = torch.sigmoid(self.transport_gate(hidden)).view(batch_size, 1, 1, 1, 1)
            next_field = next_field + self.transport_rate * transport_gate * temperature_scale.view(batch_size, 1, 1, 1, 1) * (transport - next_field)

        read_patch = self._gather_patch(next_field, center_z, center_y, center_x, active_radius)
        local_summary = read_patch.mean(dim=1)
        if local_summary.shape[-1] != core_hidden_dim:
            local_summary = local_summary.mean(dim=-1, keepdim=True).expand(batch_size, core_hidden_dim)

        bus_source = torch.cat([registry_hidden, local_summary], dim=-1)
        bus_value = torch.tanh(self.bus_value_proj(bus_source))
        bus_slot_gate = torch.softmax(self.bus_slot_gate_proj(registry_hidden), dim=-1)
        bus_input = bus_slot_gate.unsqueeze(-1) * bus_value.unsqueeze(1)
        bus_gate = torch.sigmoid(self.bus_gate_proj(registry_hidden)).view(batch_size, 1, 1)
        next_bus = self.global_bus_decay * bus_tensor + (1.0 - self.global_bus_decay) * (
            self.global_bus_write_scale * bus_gate * bus_input
        )
        next_bus = torch.tanh(next_bus)
        bus_summary = next_bus.mean(dim=1)
        if bus_summary.shape[-1] != core_hidden_dim:
            bus_summary = bus_summary.mean(dim=-1, keepdim=True).expand(batch_size, core_hidden_dim)

        read_hidden = torch.tanh(self.read_proj(torch.cat([registry_hidden, local_summary, bus_summary], dim=-1)))
        family_read_floor = float(getattr(self.cfg, "torus_read_family_floor", 0.2))
        if isinstance(family_context, torch.Tensor) and family_context.numel() > 0:
            family_read_floor = min(max(family_read_floor, 0.0), 1.0)
            read_hidden = read_hidden * (1.0 - family_read_floor) + family_read_floor * family_context

        stencil_entropy = -(stencil_weights * torch.log(stencil_weights + 1e-8)).sum(dim=-1).mean()
        stencil_effective_count = _effective_count_from_weights(stencil_weights).mean()
        need_aux_stats = self.training or bool(getattr(self.cfg, "profile_runtime", False)) or self.force_aux_stats
        if need_aux_stats:
            stencil_usage_entropy = stencil_entropy / math.log(max(stencil_weights.size(-1), 2))
            stencil_usage_concentration = stencil_weights.square().sum(dim=-1).mean() * float(stencil_weights.size(-1))
            cell_energy = next_field.abs().mean(dim=-1)
            active_cells = (cell_energy > self.activity_threshold).float().sum(dim=(1, 2, 3)).mean()
            soft_active_cells = torch.sigmoid((cell_energy - self.activity_threshold) * 12.0).sum(dim=(1, 2, 3)).mean()
            total_cells = float(self.depth * self.height * self.width)
            active_fraction = active_cells / max(total_cells, 1.0)
            active_target_fraction = max(min(self.active_target_fraction, 1.0), 1e-3)
            soft_active_fraction = soft_active_cells / max(total_cells, 1.0)
            active_balance_loss = torch.abs(soft_active_fraction - active_target_fraction) / active_target_fraction
            mixture_target = float(getattr(self.cfg, "emitter_mixture_target_count", 2.0))
            emitter_cell_mixture_loss = _mixture_loss_from_effective_count(stencil_effective_count, mixture_target)
            stats = {
                "torus_entropy": stencil_entropy,
                "torus_activity_threshold": torch.tensor(self.activity_threshold, device=hidden.device).detach(),
                "cell_energy_mean": cell_energy.mean().detach(),
                "cell_energy_min": cell_energy.min().detach(),
                "cell_energy_max": cell_energy.max().detach(),
                "active_cells": active_cells.detach(),
                "emitter_cell_occupancy": active_cells.detach(),
                "emitter_cell_breadth": active_fraction.detach(),
                "emitter_cell_soft_occupancy": soft_active_cells.detach(),
                "emitter_cell_soft_breadth": soft_active_fraction.detach(),
                "emitter_cell_mixture_entropy": stencil_entropy,
                "emitter_cell_effective_count": stencil_effective_count,
                "emitter_cell_mixture_loss": emitter_cell_mixture_loss,
                "avg_emitter_usage_entropy": stencil_usage_entropy.detach(),
                "avg_emitter_usage_concentration": stencil_usage_concentration.detach(),
                "emitter_usage_entropy": stencil_usage_entropy.detach(),
                "emitter_usage_concentration": stencil_usage_concentration.detach(),
                "torus_coverage_loss": active_balance_loss,
                "emitter_cell_coverage_loss": active_balance_loss,
                "write_coord": write_coord.detach(),
                "torus_temperature": local_temperature.detach(),
                "global_bus_slots": torch.tensor(float(self.global_bus_slots), device=hidden.device),
                "global_bus_norm": next_bus.abs().mean().detach(),
                "active_torus_radius": torch.tensor(float(active_radius), device=hidden.device),
            }
        else:
            stats = {
                "torus_entropy": stencil_entropy.detach(),
                "active_cells": torch.tensor(0.0, device=hidden.device).detach(),
                "emitter_cell_occupancy": torch.tensor(0.0, device=hidden.device).detach(),
                "torus_coverage_loss": torch.tensor(0.0, device=hidden.device).detach(),
                "emitter_cell_coverage_loss": torch.tensor(0.0, device=hidden.device).detach(),
                "emitter_cell_effective_count": stencil_effective_count.detach(),
                "global_bus_norm": next_bus.abs().mean().detach(),
                "write_coord": write_coord.detach(),
                "torus_temperature": local_temperature.detach(),
                "global_bus_slots": torch.tensor(float(self.global_bus_slots), device=hidden.device),
                "active_torus_radius": torch.tensor(float(active_radius), device=hidden.device),
            }
        transition_state = {
            "linear": {
                "path_vector": path_vector,
                "path_bias": path_bias,
                "registry_hidden": registry_hidden,
                "write_coord": write_coord,
                "center_z": center_z,
                "center_y": center_y,
                "center_x": center_x,
                "active_radius": torch.tensor(float(active_radius), device=hidden.device),
            },
            "nonlinear": {
                "stencil_weights": stencil_weights,
                "write_gate": write_gate,
                "write_update": write_update,
                "read_hidden": read_hidden,
                "bus_gate": bus_gate,
                "transport_applied": torch.tensor(bool(step_index % self.transport_interval == 0), device=hidden.device),
            },
            "carry": {
                "field": next_field,
                "bus": next_bus,
                "local_field_radius": torch.tensor(float(self.local_field_radius), device=hidden.device),
                "active_radius": torch.tensor(float(active_radius), device=hidden.device),
                "bus_slots": torch.tensor(float(self.global_bus_slots), device=hidden.device),
                "temperature": local_temperature,
                "write_strength": write_strength,
            },
        }
        next_state = PrismalTorusState(field=next_field, bus=next_bus)
        return hidden + read_hidden, next_state, stats, transition_state

    def solve_transition(
        self,
        hidden: torch.Tensor,
        field: torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        *,
        path_index: int,
        step_index: int = 0,
        relay_mode: bool = False,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        relaxation: Optional[float] = None,
    ) -> Tuple[torch.Tensor, PrismalTorusState, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        max_iterations = max(1, int(self.fixed_point_iterations if max_iterations is None else max_iterations))
        tolerance = float(self.fixed_point_tolerance if tolerance is None else tolerance)
        relaxation = float(self.fixed_point_relaxation if relaxation is None else relaxation)
        relaxation = min(max(relaxation, 0.0), 1.0)

        current_hidden = hidden
        current_state = self.init_state(hidden.size(0), hidden.device, state=field)
        last_stats: Dict[str, torch.Tensor] = {}
        last_transition_state: Dict[str, torch.Tensor] = {}
        residual = torch.tensor(float("inf"), device=hidden.device)
        iterations_used = 0
        converged = False

        check_convergence = not self.training
        for iteration in range(max_iterations):
            next_hidden, next_field, stats, transition_state = self.transition(
                current_hidden,
                current_state,
                path_index=path_index,
                step_index=step_index,
                relay_mode=relay_mode,
                registry_context=registry_context,
                family_context=family_context,
                level_context=level_context,
                relation_context=relation_context,
                parent_context=parent_context,
            )
            repair_count = 0
            next_hidden, repaired = self._sanitize_tensor(next_hidden, fallback=current_hidden)
            repair_count += repaired
            next_field_field, repaired = self._sanitize_tensor(next_field.field, fallback=current_state.field)
            next_field_bus, repaired_bus = self._sanitize_tensor(next_field.bus, fallback=current_state.bus)
            repair_count += repaired + repaired_bus
            if repair_count > 0:
                next_field = PrismalTorusState(field=next_field_field, bus=next_field_bus)
            if current_hidden.shape[-1] != next_hidden.shape[-1]:
                current_hidden_cmp = current_hidden.mean(dim=-1, keepdim=True).expand_as(next_hidden)
            else:
                current_hidden_cmp = current_hidden
            hidden_residual = (next_hidden - current_hidden_cmp).abs().mean()
            field_residual = (next_field.field - current_state.field).abs().mean()
            bus_residual = (next_field.bus - current_state.bus).abs().mean()
            residual = hidden_residual + field_residual + bus_residual
            residual, residual_repairs = self._sanitize_tensor(residual, fallback=torch.tensor(1e6, device=hidden.device))
            repair_count += residual_repairs
            last_stats = stats
            if repair_count > 0:
                last_stats = dict(last_stats)
                last_stats["stability_nonfinite_repair_count"] = torch.tensor(float(repair_count), device=hidden.device)
            last_transition_state = transition_state
            iterations_used = iteration + 1
            if check_convergence:
                converged = float(residual.detach()) <= tolerance
                if converged:
                    current_hidden = next_hidden
                    current_state = next_field
                    break
            if relaxation >= 1.0:
                current_hidden = next_hidden
                current_state = next_field
            elif relaxation > 0.0:
                if current_hidden.shape[-1] != next_hidden.shape[-1]:
                    current_hidden = current_hidden.mean(dim=-1, keepdim=True).expand_as(next_hidden)
                current_hidden = current_hidden + relaxation * (next_hidden - current_hidden)
                current_state = PrismalTorusState(
                    field=current_state.field + relaxation * (next_field.field - current_state.field),
                    bus=current_state.bus + relaxation * (next_field.bus - current_state.bus),
                )

        solver_stats = {
            "fixed_point_iterations": torch.tensor(float(iterations_used), device=hidden.device),
            "fixed_point_residual": residual.detach(),
            "fixed_point_converged": torch.tensor(float(converged), device=hidden.device),
            "fixed_point_relaxation": torch.tensor(float(relaxation), device=hidden.device),
        }
        combined_stats = dict(last_stats)
        combined_stats.update(solver_stats)
        combined_transition_state = dict(last_transition_state)
        combined_transition_state["solver"] = solver_stats
        return current_hidden, current_state, combined_stats, combined_transition_state

    def solve_chunk_transition(
        self,
        hidden_chunk: torch.Tensor,
        field: torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        *,
        path_index: int,
        step_index: int = 0,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        relaxation: Optional[float] = None,
    ) -> Tuple[torch.Tensor, PrismalTorusState, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if hidden_chunk.dim() != 3:
            raise ValueError("hidden_chunk must have shape (batch, chunk_len, d_model)")
        max_iterations = max(1, int(self.fixed_point_iterations if max_iterations is None else max_iterations))
        tolerance = float(self.fixed_point_tolerance if tolerance is None else tolerance)
        relaxation = float(self.fixed_point_relaxation if relaxation is None else relaxation)
        relaxation = min(max(relaxation, 0.0), 1.0)

        if hidden_chunk.size(1) == 1:
            chunk_summary = hidden_chunk[:, 0, :]
        else:
            chunk_summary = 0.5 * hidden_chunk.mean(dim=1) + 0.25 * hidden_chunk[:, 0, :] + 0.25 * hidden_chunk[:, -1, :]

        registry_context = self._summarize_chunk_context(registry_context)
        family_context = self._summarize_chunk_context(family_context)

        current_hidden = chunk_summary
        current_state = self.init_state(hidden_chunk.size(0), hidden_chunk.device, state=field)
        last_stats: Dict[str, torch.Tensor] = {}
        last_transition_state: Dict[str, torch.Tensor] = {}
        residual = torch.tensor(float("inf"), device=hidden_chunk.device)
        iterations_used = 0
        converged = False

        check_convergence = not self.training
        for iteration in range(max_iterations):
            next_hidden, next_field, stats, transition_state = self.transition(
                current_hidden,
                current_state,
                path_index=path_index,
                step_index=step_index,
                relay_mode=False,
                registry_context=registry_context,
                family_context=family_context,
                level_context=level_context,
                relation_context=relation_context,
                parent_context=parent_context,
            )
            repair_count = 0
            next_hidden, repaired = self._sanitize_tensor(next_hidden, fallback=current_hidden)
            repair_count += repaired
            next_field_field, repaired = self._sanitize_tensor(next_field.field, fallback=current_state.field)
            next_field_bus, repaired_bus = self._sanitize_tensor(next_field.bus, fallback=current_state.bus)
            repair_count += repaired + repaired_bus
            if repair_count > 0:
                next_field = PrismalTorusState(field=next_field_field, bus=next_field_bus)
            if current_hidden.shape[-1] != next_hidden.shape[-1]:
                current_hidden_cmp = current_hidden.mean(dim=-1, keepdim=True).expand_as(next_hidden)
            else:
                current_hidden_cmp = current_hidden
            hidden_residual = (next_hidden - current_hidden_cmp).abs().mean()
            field_residual = (next_field.field - current_state.field).abs().mean()
            bus_residual = (next_field.bus - current_state.bus).abs().mean()
            residual = hidden_residual + field_residual + bus_residual
            residual, residual_repairs = self._sanitize_tensor(residual, fallback=torch.tensor(1e6, device=hidden_chunk.device))
            repair_count += residual_repairs
            last_stats = stats
            if repair_count > 0:
                last_stats = dict(last_stats)
                last_stats["stability_nonfinite_repair_count"] = torch.tensor(float(repair_count), device=hidden_chunk.device)
            last_transition_state = transition_state
            iterations_used = iteration + 1
            if check_convergence:
                converged = float(residual.detach()) <= tolerance
                if converged:
                    current_hidden = next_hidden
                    current_state = next_field
                    break
            if relaxation >= 1.0:
                current_hidden = next_hidden
                current_state = next_field
            elif relaxation > 0.0:
                if current_hidden.shape[-1] != next_hidden.shape[-1]:
                    current_hidden = current_hidden.mean(dim=-1, keepdim=True).expand_as(next_hidden)
                current_hidden = current_hidden + relaxation * (next_hidden - current_hidden)
                current_state = PrismalTorusState(
                    field=current_state.field + relaxation * (next_field.field - current_state.field),
                    bus=current_state.bus + relaxation * (next_field.bus - current_state.bus),
                )

        solver_stats = {
            "fixed_point_iterations": torch.tensor(float(iterations_used), device=hidden_chunk.device),
            "fixed_point_residual": residual.detach(),
            "fixed_point_converged": torch.tensor(float(converged), device=hidden_chunk.device),
            "fixed_point_relaxation": torch.tensor(float(relaxation), device=hidden_chunk.device),
        }
        combined_stats = dict(last_stats)
        combined_stats.update(solver_stats)
        combined_transition_state = dict(last_transition_state)
        combined_transition_state["solver"] = solver_stats
        combined_transition_state["chunk_context"] = current_hidden
        return current_hidden, current_state, combined_stats, combined_transition_state

    def chunked_step(
        self,
        hidden_chunk: torch.Tensor,
        field_state: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        path_index: int = 0,
        step_index_offset: int = 0,
        relay_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if hidden_chunk.dim() != 3:
            raise ValueError("hidden_chunk must have shape (batch, chunk_len, d_model)")
        batch_size, chunk_len, _ = hidden_chunk.shape
        current_state = self.init_state(batch_size, hidden_chunk.device, state=field_state)

        if chunk_len <= 1:
            step_output, current_state, stats = self.step(
                hidden_chunk[:, 0, :],
                current_state,
                path_index=path_index,
                step_index=step_index_offset,
                relay_mode=relay_mode,
                registry_context=registry_context[:, 0, :] if registry_context is not None and registry_context.dim() == 3 else registry_context,
                family_context=family_context[:, 0, :] if family_context is not None and family_context.dim() == 3 else family_context,
                level_context=level_context[:, 0, :] if level_context is not None and level_context.dim() == 3 else level_context,
                relation_context=relation_context[:, 0, :] if relation_context is not None and relation_context.dim() == 3 else relation_context,
                parent_context=parent_context[:, 0, :] if parent_context is not None and parent_context.dim() == 3 else parent_context,
            )
            return step_output.unsqueeze(1), current_state, stats

        outputs: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []
        active_terms: List[torch.Tensor] = []
        soft_active_terms: List[torch.Tensor] = []
        cell_effective_terms: List[torch.Tensor] = []
        cell_mixture_terms: List[torch.Tensor] = []
        cell_coverage_terms: List[torch.Tensor] = []
        soft_breadth_terms: List[torch.Tensor] = []
        usage_entropy_terms: List[torch.Tensor] = []
        usage_concentration_terms: List[torch.Tensor] = []
        scalar_terms: Dict[str, List[torch.Tensor]] = {}
        last_stats: Dict[str, torch.Tensor] = {}

        step_hidden = hidden_chunk[:, 0, :]
        step_registry_context = registry_context[:, 0, :] if registry_context is not None and registry_context.dim() == 3 else registry_context
        step_family_context = family_context[:, 0, :] if family_context is not None and family_context.dim() == 3 else family_context
        step_level_context = level_context[:, 0, :] if level_context is not None and level_context.dim() == 3 else level_context
        step_relation_context = relation_context[:, 0, :] if relation_context is not None and relation_context.dim() == 3 else relation_context
        step_parent_context = parent_context[:, 0, :] if parent_context is not None and parent_context.dim() == 3 else parent_context
        step_output, current_state, stats = self.step(
            step_hidden,
            current_state,
            path_index=path_index,
            step_index=step_index_offset,
            relay_mode=relay_mode,
            registry_context=step_registry_context,
            family_context=step_family_context,
            level_context=step_level_context,
            relation_context=step_relation_context,
            parent_context=step_parent_context,
        )
        chunk_output = hidden_chunk + (step_output - step_hidden).unsqueeze(1)
        entropy_terms.append(stats["torus_entropy"])
        active_terms.append(stats.get("emitter_cell_occupancy", stats["active_cells"]).float())
        if "emitter_cell_soft_occupancy" in stats:
            soft_active_terms.append(stats["emitter_cell_soft_occupancy"].float())
        if "emitter_cell_effective_count" in stats:
            cell_effective_terms.append(stats["emitter_cell_effective_count"].float())
        if "emitter_cell_mixture_loss" in stats:
            cell_mixture_terms.append(stats["emitter_cell_mixture_loss"].float())
        if "emitter_cell_coverage_loss" in stats:
            cell_coverage_terms.append(stats["emitter_cell_coverage_loss"].float())
        if "emitter_cell_soft_breadth" in stats:
            soft_breadth_terms.append(stats["emitter_cell_soft_breadth"].float())
        if "emitter_usage_entropy" in stats:
            usage_entropy_terms.append(stats["emitter_usage_entropy"].float())
        if "emitter_usage_concentration" in stats:
            usage_concentration_terms.append(stats["emitter_usage_concentration"].float())
        for key, value in stats.items():
            if key.startswith("recursive_") and torch.is_tensor(value) and value.numel() == 1:
                scalar_terms.setdefault(key, []).append(value.float())
        last_stats = stats
        chunk_stats = dict(last_stats)
        if entropy_terms:
            chunk_stats["torus_entropy"] = torch.stack(entropy_terms).mean()
        if active_terms:
            chunk_stats["emitter_cell_occupancy"] = torch.stack(active_terms).mean()
        if soft_active_terms:
            chunk_stats["emitter_cell_soft_occupancy"] = torch.stack(soft_active_terms).mean()
        if cell_effective_terms:
            chunk_stats["emitter_cell_effective_count"] = torch.stack(cell_effective_terms).mean()
        if cell_mixture_terms:
            chunk_stats["emitter_cell_mixture_loss"] = torch.stack(cell_mixture_terms).mean()
        if cell_coverage_terms:
            chunk_stats["emitter_cell_coverage_loss"] = torch.stack(cell_coverage_terms).mean()
        if soft_breadth_terms:
            chunk_stats["emitter_cell_soft_breadth"] = torch.stack(soft_breadth_terms).mean()
        if usage_entropy_terms:
            chunk_stats["emitter_usage_entropy"] = torch.stack(usage_entropy_terms).mean()
        if usage_concentration_terms:
            chunk_stats["emitter_usage_concentration"] = torch.stack(usage_concentration_terms).mean()
        for key, values in scalar_terms.items():
            if values:
                chunk_stats[key] = torch.stack(values).mean()
        return chunk_output, current_state, chunk_stats

    def chunk_solver_step(
        self,
        hidden_chunk: torch.Tensor,
        field_state: torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        path_index: int = 0,
        step_index_offset: int = 0,
    ) -> Tuple[torch.Tensor, PrismalTorusState, Dict[str, torch.Tensor]]:
        if hidden_chunk.dim() != 3:
            raise ValueError("hidden_chunk must have shape (batch, chunk_len, d_model)")
        chunk_len = hidden_chunk.size(1)
        chunk_hidden_dim = hidden_chunk.size(-1)
        substeps = max(1, min(chunk_len, int(getattr(self.cfg, "chunk_solver_training_substeps", 8))))
        micro_len = max(1, math.ceil(chunk_len / substeps))
        max_iterations = max(1, int(getattr(self.cfg, "chunk_solver_training_iterations", 1)))
        relaxation = float(getattr(self.cfg, "chunk_solver_training_relaxation", 1.0))
        audit_every = max(1, int(getattr(self.cfg, "chunk_solver_training_audit_every", 16)))

        def _anchor_context(context: Optional[torch.Tensor], start: int) -> Optional[torch.Tensor]:
            if context is None:
                return None
            if context.dim() == 3:
                if context.size(1) == 0:
                    return None
                safe_start = min(max(int(start), 0), context.size(1) - 1)
                return context[:, safe_start, :]
            return context

        def _slice_context(context: Optional[torch.Tensor], start: int, end: int) -> Optional[torch.Tensor]:
            if context is None:
                return None
            if context.dim() == 3:
                if context.size(1) == 0:
                    return None
                safe_start = min(max(int(start), 0), context.size(1))
                safe_end = min(max(int(end), safe_start), context.size(1))
                if safe_start >= safe_end:
                    return None
                return context[:, safe_start:safe_end, :]
            return context

        outputs: List[torch.Tensor] = []
        stats_terms: Dict[str, List[torch.Tensor]] = {}
        last_stats: Dict[str, torch.Tensor] = {}
        next_state = self.init_state(hidden_chunk.size(0), hidden_chunk.device, state=field_state)

        for micro_start in range(0, chunk_len, micro_len):
            micro_end = min(chunk_len, micro_start + micro_len)
            micro_hidden = hidden_chunk[:, micro_start:micro_end, :]
            micro_step_index = micro_start // micro_len
            use_exact_solver = (micro_step_index % audit_every) == 0
            if use_exact_solver:
                # Anchor the exact audit pass on the first token only. Using
                # the whole micro-chunk summary leaks future training tokens
                # into earlier label predictions, producing good val loss but
                # broken generation.
                micro_anchor = micro_hidden[:, 0, :]
                if micro_anchor.shape[-1] != self.write_delta_proj.in_features:
                    micro_anchor_solver = micro_anchor.mean(dim=-1, keepdim=True).expand(
                        micro_anchor.size(0),
                        self.write_delta_proj.in_features,
                    )
                else:
                    micro_anchor_solver = micro_anchor
                solved_hidden, next_state, stats, _transition_state = self.solve_transition(
                    micro_anchor_solver,
                    next_state,
                    path_index=path_index,
                    step_index=step_index_offset + micro_start,
                    registry_context=_anchor_context(registry_context, micro_start),
                    family_context=_anchor_context(family_context, micro_start),
                    level_context=_anchor_context(level_context, micro_start),
                    relation_context=_anchor_context(relation_context, micro_start),
                    parent_context=_anchor_context(parent_context, micro_start),
                    max_iterations=max_iterations,
                    relaxation=relaxation,
                )
                if solved_hidden.shape[-1] != chunk_hidden_dim:
                    solved_hidden = solved_hidden.mean(dim=-1, keepdim=True).expand(solved_hidden.size(0), chunk_hidden_dim)
                micro_delta = solved_hidden - micro_anchor
                outputs.append(micro_hidden + micro_delta.unsqueeze(1))
                stats["chunk_solver_audit"] = torch.tensor(1.0, device=hidden_chunk.device)
            else:
                surrogate_hidden, next_state, stats = self.chunked_step(
                    micro_hidden,
                    next_state,
                    signature_family_ids=_slice_context(signature_family_ids, micro_start, micro_end),
                    signature_ids=_slice_context(signature_ids, micro_start, micro_end),
                    signature_level_ids=_slice_context(signature_level_ids, micro_start, micro_end),
                    signature_relation_ids=_slice_context(signature_relation_ids, micro_start, micro_end),
                    parent_signature_ids=_slice_context(parent_signature_ids, micro_start, micro_end),
                    registry_context=_slice_context(registry_context, micro_start, micro_end),
                    family_context=_slice_context(family_context, micro_start, micro_end),
                    level_context=_slice_context(level_context, micro_start, micro_end),
                    relation_context=_slice_context(relation_context, micro_start, micro_end),
                    parent_context=_slice_context(parent_context, micro_start, micro_end),
                    path_index=path_index,
                    step_index_offset=step_index_offset + micro_start,
                    relay_mode=False,
                )
                outputs.append(surrogate_hidden)
                stats["chunk_solver_audit"] = torch.tensor(0.0, device=hidden_chunk.device)
            last_stats = stats
            for key, value in stats.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    stats_terms.setdefault(key, []).append(value.float())

        chunk_output = torch.cat(outputs, dim=1)
        chunk_stats = dict(last_stats)
        for key, values in stats_terms.items():
            if values:
                chunk_stats[key] = torch.stack(values).mean()
        return chunk_output, next_state, chunk_stats

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        field_key = prefix + "field_seed"
        if field_key in state_dict:
            field_seed = state_dict[field_key]
            if field_seed.dim() == 3:
                state_dict[field_key] = field_seed.unsqueeze(0).expand(self.depth, -1, -1, -1).clone()

        def _find_param_key(module: nn.Module, attr_name: str, param_name: str) -> Optional[str]:
            candidates = []
            if hasattr(module, "linear"):
                candidates.append(prefix + f"{attr_name}.linear.{param_name}")
            candidates.append(prefix + f"{attr_name}.{param_name}")
            for candidate in candidates:
                if candidate in state_dict:
                    return candidate
            return None

        for attr_name in ("write_coord_proj", "path_coord_proj"):
            module = getattr(self, attr_name)
            weight_key = _find_param_key(module, attr_name, "weight")
            bias_key = _find_param_key(module, attr_name, "bias")
            if weight_key is not None and state_dict[weight_key].shape != module.weight.shape:
                old_weight = state_dict[weight_key]
                if old_weight.shape[0] == 2 and module.weight.shape[0] == 3:
                    adapted = torch.zeros_like(module.weight)
                    adapted[:2].copy_(old_weight)
                    state_dict[weight_key] = adapted
            if bias_key is not None and module.bias is not None and state_dict[bias_key].shape != module.bias.shape:
                old_bias = state_dict[bias_key]
                if old_bias.numel() == 2 and module.bias.numel() == 3:
                    adapted = torch.zeros_like(module.bias)
                    adapted[:2].copy_(old_bias)
                    state_dict[bias_key] = adapted

        stencil_weight_key = _find_param_key(self.stencil_proj, "stencil_proj", "weight")
        stencil_bias_key = _find_param_key(self.stencil_proj, "stencil_proj", "bias")
        new_stencil = self.stencil_proj.weight.shape[0]
        if stencil_weight_key is not None and state_dict[stencil_weight_key].shape[0] != new_stencil:
            old_weight = state_dict[stencil_weight_key]
            old_side = int(round(old_weight.shape[0] ** 0.5))
            new_side = 2 * self.write_radius + 1
            if old_side * old_side == old_weight.shape[0] and new_side * new_side * new_side == new_stencil:
                adapted = torch.zeros_like(self.stencil_proj.weight)
                center_z = self.write_radius
                for old_idx in range(old_weight.shape[0]):
                    dy = old_idx // old_side - self.write_radius
                    dx = old_idx % old_side - self.write_radius
                    new_idx = (
                        (center_z + self.write_radius) * new_side * new_side
                        + (dy + self.write_radius) * new_side
                        + (dx + self.write_radius)
                    )
                    adapted[new_idx].copy_(old_weight[old_idx])
                state_dict[stencil_weight_key] = adapted
        if (
            stencil_bias_key is not None
            and self.stencil_proj.bias is not None
            and state_dict[stencil_bias_key].shape[0] != self.stencil_proj.bias.shape[0]
        ):
            old_bias = state_dict[stencil_bias_key]
            old_side = int(round(old_bias.shape[0] ** 0.5))
            new_side = 2 * self.write_radius + 1
            if old_side * old_side == old_bias.shape[0] and new_side * new_side * new_side == self.stencil_proj.bias.shape[0]:
                adapted = torch.zeros_like(self.stencil_proj.bias)
                center_z = self.write_radius
                for old_idx in range(old_bias.shape[0]):
                    dy = old_idx // old_side - self.write_radius
                    dx = old_idx % old_side - self.write_radius
                    new_idx = (
                        (center_z + self.write_radius) * new_side * new_side
                        + (dy + self.write_radius) * new_side
                        + (dx + self.write_radius)
                    )
                    adapted[new_idx] = old_bias[old_idx]
                state_dict[stencil_bias_key] = adapted

        relay_weight_key = _find_param_key(self.relay_stencil_proj, "relay_stencil_proj", "weight")
        relay_bias_key = _find_param_key(self.relay_stencil_proj, "relay_stencil_proj", "bias")
        if relay_weight_key is not None and relay_weight_key not in state_dict and stencil_weight_key is not None and stencil_weight_key in state_dict:
            old_weight = state_dict[stencil_weight_key]
            relay_weight = torch.zeros_like(self.relay_stencil_proj.weight)
            relay_weight[0].copy_(old_weight[old_weight.shape[0] // 2])
            state_dict[relay_weight_key] = relay_weight
        if relay_bias_key is not None and relay_bias_key not in state_dict and stencil_bias_key is not None and stencil_bias_key in state_dict:
            old_bias = state_dict[stencil_bias_key]
            relay_bias = torch.zeros_like(self.relay_stencil_proj.bias)
            relay_bias[0] = old_bias[old_bias.shape[0] // 2]
            state_dict[relay_bias_key] = relay_bias

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _finite_guard_enabled(self) -> bool:
        return bool(
            getattr(self.cfg, "training_finite_guard_enabled", True)
            if self.training
            else getattr(self.cfg, "inference_finite_guard_enabled", True)
        )

    def _sanitize_tensor(
        self,
        tensor: torch.Tensor,
        *,
        fallback: Optional[torch.Tensor] = None,
        allow_negative_inf: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        if not self._finite_guard_enabled():
            return tensor, 0
        return _repair_finite_tensor(tensor, fallback=fallback, allow_negative_inf=allow_negative_inf)

    def _sanitize_route_stats(
        self,
        route_stats: Dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        if not self._finite_guard_enabled():
            return route_stats, 0
        sanitized: Dict[str, torch.Tensor] = {}
        repairs = 0
        for key, value in route_stats.items():
            if torch.is_tensor(value):
                clean_value, clean_repairs = self._sanitize_tensor(value)
                sanitized[key] = clean_value
                repairs += clean_repairs
            else:
                sanitized[key] = value
        sanitized["stability_nonfinite_repair_count"] = torch.tensor(float(repairs), device=device)
        sanitized["stability_finite_guard_enabled"] = torch.tensor(1.0, device=device)
        return sanitized, repairs

    def _sanitize_sampling_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, int]:
        clean_logits, repairs = self._sanitize_tensor(logits, allow_negative_inf=True)
        if not self._finite_guard_enabled() or clean_logits.numel() == 0:
            return clean_logits, repairs
        row_has_finite = torch.isfinite(clean_logits).any(dim=-1, keepdim=True)
        if bool(row_has_finite.all().item()):
            return clean_logits, repairs
        cleaned = torch.where(row_has_finite, clean_logits, torch.zeros_like(clean_logits))
        repairs += int((~row_has_finite).sum().item())
        return cleaned, repairs

    def step(
        self,
        hidden: torch.Tensor,
        field: torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        *,
        path_index: int,
        step_index: int = 0,
        relay_mode: bool = False,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PrismalTorusState, Dict[str, torch.Tensor]]:
        profile_enabled = bool(getattr(self.cfg, "profile_runtime", False))
        timing_ms: Dict[str, float] = {}
        if profile_enabled:
            _sync_for_timing(hidden.device)
            start_total = time.perf_counter()
        next_hidden, next_field, stats, _transition_state = self.transition(
            hidden,
            field,
            path_index=path_index,
            step_index=step_index,
            relay_mode=relay_mode,
            registry_context=registry_context,
            family_context=family_context,
            level_context=level_context,
            relation_context=relation_context,
            parent_context=parent_context,
        )
        repair_count = 0
        next_hidden, repaired = self._sanitize_tensor(next_hidden, fallback=hidden)
        repair_count += repaired
        if isinstance(next_field, PrismalTorusState):
            next_field_field, repaired = self._sanitize_tensor(next_field.field, fallback=next_field.field.new_zeros(next_field.field.shape))
            next_field_bus, repaired_bus = self._sanitize_tensor(next_field.bus, fallback=next_field.bus.new_zeros(next_field.bus.shape))
            repair_count += repaired + repaired_bus
            next_field = PrismalTorusState(field=next_field_field, bus=next_field_bus)
        sanitized_stats, stat_repairs = self._sanitize_route_stats(stats, device=hidden.device)
        repair_count += stat_repairs
        sanitized_stats["stability_step_repair_count"] = torch.tensor(float(repair_count), device=hidden.device)
        if profile_enabled:
            _sync_for_timing(hidden.device)
            timing_ms["timing_torus_step_ms"] = (time.perf_counter() - start_total) * 1000.0
            for key, value in timing_ms.items():
                sanitized_stats[key] = torch.tensor(value, device=hidden.device)
        return next_hidden, next_field, sanitized_stats


class PrismalRecursiveTorusCore(PrismalTorusCore):
    """Torus core with a depth-limited recursive expert tree."""

    def __init__(self, cfg: PrismalWaveConfig, quantization_config: Optional[QuantizationConfig] = None):
        super().__init__(cfg, quantization_config=quantization_config)
        self.recursive_depth = max(1, int(getattr(cfg, "hmote_depth", getattr(cfg, "recursive_hmoe_depth", 1))))
        self.recursive_branching = max(1, int(getattr(cfg, "hmote_branching", getattr(cfg, "recursive_hmoe_branching", 1))))
        self.recursive_coarse_top_k = max(1, min(self.recursive_branching, int(getattr(cfg, "recursive_hmoe_coarse_top_k", 1))))
        self.recursive_fine_top_k = max(1, int(getattr(cfg, "recursive_hmoe_fine_top_k", 1)))
        self.recursive_balance_weight = float(getattr(cfg, "recursive_hmoe_balance_weight", 0.0))
        self.recursive_child_mixture_weight = float(getattr(cfg, "recursive_hmoe_child_mixture_weight", 0.0))
        self.recursive_agreement_weight = float(getattr(cfg, "recursive_hmoe_agreement_weight", 0.0))
        self.recursive_enabled = bool(getattr(cfg, "use_recursive_hmoe", False)) and self.recursive_depth > 1
        d = cfg.d_model
        coarse_context_dim = d * 6
        self.recursive_gate = create_quantized_linear(
            coarse_context_dim,
            self.recursive_branching,
            quantization_config=self.quantization_config,
        )
        self.recursive_context_proj = create_quantized_linear(coarse_context_dim, d, quantization_config=self.quantization_config)
        self.recursive_child_hidden_bias = create_quantized_embedding(
            self.recursive_branching,
            d,
            quantization_config=self.quantization_config,
        )
        self.recursive_child_field_bias = create_quantized_embedding(
            self.recursive_branching,
            d,
            quantization_config=self.quantization_config,
        )
        self.recursive_child_mix_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.recursive_child_agreement_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.recursive_child_cores = nn.ModuleList()
        if self.recursive_enabled:
            child_depth = max(1, self.recursive_depth - 1)
            child_cfg = _recursive_child_config(
                cfg,
                depth=child_depth,
                branching=self.recursive_branching,
                child_torus_depth=getattr(cfg, "recursive_hmoe_child_torus_depth", self.depth),
                child_torus_height=getattr(cfg, "recursive_hmoe_child_torus_height", self.height),
                child_torus_width=getattr(cfg, "recursive_hmoe_child_torus_width", self.width),
                child_fine_top_k=self.recursive_fine_top_k,
            )
            for _ in range(self.recursive_branching):
                child_core = (
                    PrismalRecursiveTorusCore(child_cfg, quantization_config=self.quantization_config)
                    if child_cfg.use_hmote and getattr(child_cfg, "hmote_depth", child_cfg.recursive_hmoe_depth) > 1
                    else PrismalTorusCore(child_cfg, quantization_config=self.quantization_config)
                )
                self.recursive_child_cores.append(child_core)

    def _recursive_context_vector(
        self,
        hidden: torch.Tensor,
        *,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, dim = hidden.shape
        zeros = torch.zeros(batch, dim, device=hidden.device, dtype=hidden.dtype)

        def _ctx(value: Optional[torch.Tensor]) -> torch.Tensor:
            if value is None or value.numel() == 0:
                return zeros
            summarized = self._summarize_chunk_context(value)
            if summarized is None:
                return zeros
            if summarized.dim() == 1:
                return summarized.view(batch, -1)
            return summarized

        return torch.cat(
            [
                hidden,
                _ctx(registry_context),
                _ctx(family_context),
                _ctx(level_context),
                _ctx(relation_context),
                _ctx(parent_context),
            ],
            dim=-1,
        )

    def _prefix_scalar_stats(
        self,
        stats: Dict[str, torch.Tensor],
        *,
        prefix: str,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        prefixed: Dict[str, torch.Tensor] = {}
        for key, value in stats.items():
            if not (torch.is_tensor(value) and value.numel() == 1):
                continue
            prefixed[f"{prefix}{key}"] = _scalar_stat_tensor(value, device=device)
        return prefixed

    def transition(
        self,
        hidden: torch.Tensor,
        field: torch.Tensor | PrismalTorusState | Dict[str, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor],
        *,
        path_index: int,
        step_index: int = 0,
        relay_mode: bool = False,
        registry_context: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PrismalTorusState, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        base_hidden, base_state, base_stats, base_transition_state = super().transition(
            hidden,
            field,
            path_index=path_index,
            step_index=step_index,
            relay_mode=relay_mode,
            registry_context=registry_context,
            family_context=family_context,
            level_context=level_context,
            relation_context=relation_context,
            parent_context=parent_context,
        )
        if not self.recursive_enabled or not self.recursive_child_cores:
            base_stats = dict(base_stats)
            base_stats.setdefault("recursive_depth", torch.tensor(float(self.recursive_depth), device=hidden.device))
            base_stats.setdefault("recursive_aux_loss", torch.tensor(0.0, device=hidden.device))
            return base_hidden, base_state, base_stats, base_transition_state

        coarse_context = self._recursive_context_vector(
            hidden,
            registry_context=registry_context,
            family_context=family_context,
            level_context=level_context,
            relation_context=relation_context,
            parent_context=parent_context,
        )
        coarse_logits = self.recursive_gate(coarse_context)
        coarse_weights = F.softmax(coarse_logits, dim=-1)
        coarse_top_k = min(self.recursive_coarse_top_k, coarse_logits.size(-1))
        coarse_top_scores, coarse_top_idx = torch.topk(coarse_weights, k=coarse_top_k, dim=-1)
        selected_weights = torch.gather(coarse_weights, 1, coarse_top_idx)
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        selected_children = torch.unique(coarse_top_idx.detach()).tolist()
        child_entropy = -(coarse_weights * torch.log(coarse_weights + 1e-8)).sum(dim=-1)
        child_effective_count = _effective_count_from_weights(coarse_weights)
        balance_loss = _mixture_loss_from_effective_count(child_effective_count, float(self.recursive_coarse_top_k))

        merged_stats = dict(base_stats)
        child_prefix = f"recursive_depth_{self.recursive_depth}_"
        merged_stats[f"{child_prefix}coarse_entropy"] = child_entropy.mean().detach()
        merged_stats[f"{child_prefix}coarse_effective_count"] = child_effective_count.mean().detach()
        merged_stats[f"{child_prefix}coarse_balance_loss"] = balance_loss.detach()
        merged_stats[f"{child_prefix}coarse_top_k"] = torch.tensor(float(coarse_top_k), device=hidden.device)
        merged_stats[f"{child_prefix}coarse_weights_mean"] = coarse_weights.mean().detach()

        field_summary = base_state.field.mean(dim=(1, 2, 3))
        coarse_context_delta = torch.tanh(self.recursive_context_proj(coarse_context))

        child_hidden_mix = torch.zeros_like(base_hidden)
        child_field_mix = torch.zeros_like(field_summary)
        child_aux_terms: List[torch.Tensor] = []
        executed_children = 0

        for child_idx in selected_children:
            child_core = self.recursive_child_cores[child_idx]
            child_matches = coarse_top_idx == child_idx
            if not child_matches.any():
                continue
            row_indices, topk_positions = torch.nonzero(child_matches, as_tuple=True)
            route_weights = selected_weights[row_indices, topk_positions]
            child_hidden_bias = self.recursive_child_hidden_bias.weight[child_idx].view(1, -1)
            child_field_bias = self.recursive_child_field_bias.weight[child_idx].view(1, 1, 1, 1, -1)
            child_path_index = int(path_index + child_idx) % max(1, int(getattr(child_core.cfg, "n_paths", 1)))

            child_hidden = base_hidden.index_select(0, row_indices)
            child_hidden = child_hidden + coarse_context_delta.index_select(0, row_indices) + child_hidden_bias.expand(row_indices.numel(), -1)
            child_state = child_core.init_state(row_indices.numel(), hidden.device, state=None)
            child_state = PrismalTorusState(
                field=child_state.field
                + field_summary.index_select(0, row_indices).view(row_indices.numel(), 1, 1, 1, -1)
                + child_field_bias,
                bus=child_state.bus,
            )
            child_hidden_out, child_field_out, child_stats, _child_transition_state = child_core.transition(
                child_hidden,
                child_state,
                path_index=child_path_index,
                step_index=step_index + child_idx,
                relay_mode=relay_mode,
                registry_context=registry_context.index_select(0, row_indices) if registry_context is not None and registry_context.size(0) == hidden.size(0) else registry_context,
                family_context=family_context.index_select(0, row_indices) if family_context is not None and family_context.size(0) == hidden.size(0) else family_context,
                level_context=level_context.index_select(0, row_indices) if level_context is not None and level_context.size(0) == hidden.size(0) else level_context,
                relation_context=relation_context.index_select(0, row_indices) if relation_context is not None and relation_context.size(0) == hidden.size(0) else relation_context,
                parent_context=parent_context.index_select(0, row_indices) if parent_context is not None and parent_context.size(0) == hidden.size(0) else parent_context,
            )
            child_hidden_contrib = (child_hidden_out * route_weights.unsqueeze(-1)).to(dtype=child_hidden_mix.dtype)
            child_field_contrib = (
                child_field_out.field.mean(dim=(1, 2, 3)) * route_weights.unsqueeze(-1)
            ).to(dtype=child_field_mix.dtype)
            child_hidden_mix.index_add_(0, row_indices, child_hidden_contrib)
            child_field_mix.index_add_(0, row_indices, child_field_contrib)
            child_aux_value = child_stats.get("recursive_aux_loss", 0.0)
            child_weight_mean = route_weights.mean()
            if torch.is_tensor(child_aux_value):
                child_aux_terms.append(child_aux_value * child_weight_mean)
            else:
                child_aux_terms.append(torch.tensor(float(child_aux_value), device=hidden.device) * child_weight_mean.detach())
            merged_stats.update(self._prefix_scalar_stats(child_stats, prefix=f"{child_prefix}child{child_idx}_", device=hidden.device))
            executed_children += 1

        child_weighted_hidden = self.recursive_child_mix_proj(child_hidden_mix)
        child_weighted_field = self.recursive_child_agreement_proj(child_field_mix)
        agreement = F.cosine_similarity(base_hidden, child_weighted_hidden, dim=-1)
        agreement_loss = torch.relu(1.0 - agreement).mean()
        child_mixture_loss = (
            torch.stack([term.reshape(1) if term.dim() == 0 else term.mean().reshape(1) for term in child_aux_terms], dim=0).mean()
            if child_aux_terms
            else torch.tensor(0.0, device=hidden.device)
        )

        recursive_aux_loss = (
            self.recursive_balance_weight * balance_loss
            + self.recursive_child_mixture_weight * child_mixture_loss
            + self.recursive_agreement_weight * agreement_loss
        )
        final_hidden = base_hidden + 0.5 * child_weighted_hidden
        final_field = base_state.field + 0.05 * child_weighted_field.view(child_weighted_field.size(0), 1, 1, 1, -1)
        final_state = PrismalTorusState(field=final_field, bus=base_state.bus)
        merged_stats[f"{child_prefix}agreement_loss"] = agreement_loss.detach()
        merged_stats[f"{child_prefix}child_mixture_loss"] = child_mixture_loss.detach()
        merged_stats[f"{child_prefix}recursive_aux_loss"] = recursive_aux_loss.detach()
        merged_stats[f"{child_prefix}child_effective_count"] = child_effective_count.mean().detach()
        merged_stats[f"{child_prefix}child_entropy"] = child_entropy.mean().detach()
        merged_stats[f"{child_prefix}child_weights_mean"] = coarse_weights.mean().detach()
        merged_stats["recursive_aux_loss"] = recursive_aux_loss
        merged_stats["recursive_balance_loss"] = balance_loss.detach()
        merged_stats["recursive_child_mixture_loss"] = child_mixture_loss.detach()
        merged_stats["recursive_agreement_loss"] = agreement_loss.detach()
        merged_stats["recursive_child_effective_count"] = child_effective_count.mean().detach()
        merged_stats["recursive_child_entropy"] = child_entropy.mean().detach()
        merged_stats["recursive_child_weights_mean"] = coarse_weights.mean().detach()
        merged_stats["recursive_depth"] = torch.tensor(float(self.recursive_depth), device=hidden.device)
        merged_stats["recursive_branching"] = torch.tensor(float(self.recursive_branching), device=hidden.device)
        merged_stats["recursive_coarse_top_k"] = torch.tensor(float(coarse_top_k), device=hidden.device)
        merged_stats["recursive_child_count"] = torch.tensor(float(executed_children), device=hidden.device)
        merged_transition_state = dict(base_transition_state)
        merged_transition_state["recursive"] = {
            "coarse_logits": coarse_logits.detach(),
            "coarse_top_idx": coarse_top_idx.detach(),
            "coarse_top_weights": coarse_weights.detach(),
        }
        return final_hidden, final_state, merged_stats, merged_transition_state


class CausalSelfAttention(nn.Module):
    """Lightweight causal token mixing for sequence-local dependencies."""

    def __init__(self, cfg: PrismalWaveConfig, quantization_config: Optional[QuantizationConfig] = None):
        super().__init__()
        d_model = cfg.d_model
        n_heads = min(4, d_model)
        while n_heads > 1 and d_model % n_heads != 0:
            n_heads -= 1
        self.n_heads = max(1, n_heads)
        self.head_dim = d_model // self.n_heads
        self.qkv = create_quantized_linear(d_model, d_model * 3, bias=False, quantization_config=quantization_config)
        self.out = create_quantized_linear(d_model, d_model, bias=False, quantization_config=quantization_config)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_dropout = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_dropout, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out(out)


@dataclass
class SignatureLatticeState:
    cache: torch.Tensor
    cache_decay_scale: torch.Tensor
    cache_decay_steps: torch.Tensor
    counts: torch.Tensor
    prev_parent_bucket: torch.Tensor
    prev_signature_bucket: torch.Tensor
    prev_family_bucket: torch.Tensor


@dataclass
class TokenMemoryState:
    token_ids: torch.Tensor
    memory_keys: torch.Tensor
    memory_values: torch.Tensor
    family_ids: torch.Tensor
    signature_ids: torch.Tensor
    level_ids: torch.Tensor
    relation_ids: torch.Tensor
    parent_ids: torch.Tensor
    lengths: torch.Tensor
    anchor_token_ids: torch.Tensor
    anchor_span_ids: torch.Tensor
    anchor_offsets: torch.Tensor
    anchor_lengths: torch.Tensor
    anchor_tags: torch.Tensor
    anchor_flags: torch.Tensor
    anchor_span_starts: torch.Tensor
    anchor_cursor_pos: torch.Tensor
    anchor_cursor_span_id: torch.Tensor
    anchor_cursor_offset: torch.Tensor
    anchor_cursor_length: torch.Tensor
    anchor_cursor_tag: torch.Tensor
    anchor_cursor_active: torch.Tensor


class SignatureLatticeAttention(nn.Module):
    """Causal token-to-signature-cache attention over structural addresses."""

    def __init__(self, cfg: PrismalWaveConfig, quantization_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.quantization_config = quantization_config or QuantizationConfig()
        d = int(cfg.d_model)
        self.r = max(1, int(getattr(cfg, "signature_lattice_dim", 256)))
        self.buckets = max(1, int(getattr(cfg, "signature_lattice_buckets", 512)))
        self.candidates = max(1, min(self.buckets, int(getattr(cfg, "signature_lattice_candidates", 8))))
        self.weight = max(0.0, float(getattr(cfg, "signature_lattice_weight", 0.35)))
        self.decay = max(0.0, min(1.0, float(getattr(cfg, "signature_lattice_decay", 0.92))))
        self.chunk_len = max(1, int(getattr(cfg, "signature_lattice_chunk_len", getattr(cfg, "torus_chunk_len", 8))))

        self.q_proj = create_quantized_linear(d, self.r, bias=False, quantization_config=self.quantization_config)
        self.v_proj = create_quantized_linear(d, self.r, bias=False, quantization_config=self.quantization_config)
        self.out_proj = create_quantized_linear(self.r, d, bias=False, quantization_config=self.quantization_config)
        self.gate = create_quantized_linear(d, 1, bias=True, quantization_config=self.quantization_config)
        level_vocab = max(8, int(getattr(cfg, "signature_level_vocab_size", 8)) or 8)
        relation_vocab = max(8, int(getattr(cfg, "signature_relation_vocab_size", 8)) or 8)
        self.level_bias = create_quantized_embedding(level_vocab, self.r, quantization_config=self.quantization_config)
        self.relation_bias = create_quantized_embedding(relation_vocab, self.r, quantization_config=self.quantization_config)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> SignatureLatticeState:
        return SignatureLatticeState(
            cache=torch.zeros(batch_size, self.buckets, self.r, device=device, dtype=dtype),
            cache_decay_scale=torch.ones(batch_size, 1, 1, device=device, dtype=dtype),
            cache_decay_steps=torch.zeros(batch_size, 1, 1, device=device, dtype=torch.long),
            counts=torch.zeros(batch_size, self.buckets, 1, device=device, dtype=dtype),
            prev_parent_bucket=torch.zeros(batch_size, device=device, dtype=torch.long),
            prev_signature_bucket=torch.zeros(batch_size, device=device, dtype=torch.long),
            prev_family_bucket=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def _coerce_state(
        self,
        state: Optional[SignatureLatticeState],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SignatureLatticeState:
        if state is None:
            return self.init_state(batch_size, device, dtype)
        cache = state.cache.to(device=device, dtype=dtype)
        cache_decay_scale = getattr(state, "cache_decay_scale", None)
        if cache_decay_scale is None:
            cache_decay_scale = torch.ones(cache.size(0), 1, 1, device=device, dtype=dtype)
        else:
            cache_decay_scale = cache_decay_scale.to(device=device, dtype=dtype)
        cache_decay_steps = getattr(state, "cache_decay_steps", None)
        if cache_decay_steps is None:
            cache_decay_steps = torch.zeros(cache.size(0), 1, 1, device=device, dtype=torch.long)
        else:
            cache_decay_steps = cache_decay_steps.to(device=device, dtype=torch.long)
        counts = state.counts.to(device=device, dtype=dtype)
        prev_parent = state.prev_parent_bucket.to(device=device, dtype=torch.long)
        prev_signature = state.prev_signature_bucket.to(device=device, dtype=torch.long)
        prev_family = state.prev_family_bucket.to(device=device, dtype=torch.long)
        if cache.size(0) != batch_size:
            cache = cache[:1].expand(batch_size, -1, -1).clone()
            cache_decay_scale = cache_decay_scale[:1].expand(batch_size, -1, -1).clone()
            cache_decay_steps = cache_decay_steps[:1].expand(batch_size, -1, -1).clone()
            counts = counts[:1].expand(batch_size, -1, -1).clone()
            prev_parent = prev_parent[:1].expand(batch_size).clone()
            prev_signature = prev_signature[:1].expand(batch_size).clone()
            prev_family = prev_family[:1].expand(batch_size).clone()
        return SignatureLatticeState(
            cache=cache,
            cache_decay_scale=cache_decay_scale,
            cache_decay_steps=cache_decay_steps,
            counts=counts,
            prev_parent_bucket=prev_parent,
            prev_signature_bucket=prev_signature,
            prev_family_bucket=prev_family,
        )

    def _bucket(self, ids: Optional[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
        if ids is None:
            return fallback
        return ids.to(device=fallback.device).long().clamp_min(0).remainder(self.buckets)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        state: Optional[SignatureLatticeState] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[SignatureLatticeState], Dict[str, torch.Tensor]]:
        batch, seq_len, _d = hidden.shape
        device = hidden.device
        dtype = getattr(self, "precision_state_dtype", hidden.dtype)
        lattice_state = self._coerce_state(state, batch, device, dtype)
        cache = lattice_state.cache
        counts = lattice_state.counts
        prev_parent = lattice_state.prev_parent_bucket
        prev_signature = lattice_state.prev_signature_bucket
        prev_family = lattice_state.prev_family_bucket

        zero_ids = torch.zeros(batch, seq_len, device=device, dtype=torch.long)
        family_bucket = self._bucket(signature_family_ids[:, :seq_len] if signature_family_ids is not None else None, zero_ids)
        sig_bucket = self._bucket(signature_ids[:, :seq_len] if signature_ids is not None else None, zero_ids)
        parent_bucket = self._bucket(parent_signature_ids[:, :seq_len] if parent_signature_ids is not None else None, family_bucket)
        level_ids = (
            signature_level_ids[:, :seq_len].to(device=device).long().clamp_min(0)
            if signature_level_ids is not None
            else zero_ids
        )
        relation_ids = (
            signature_relation_ids[:, :seq_len].to(device=device).long().clamp_min(0)
            if signature_relation_ids is not None
            else zero_ids
        )
        level_vocab = max(1, int(getattr(self.level_bias, "num_embeddings", level_ids.max().item() + 1 if level_ids.numel() else 1)))
        relation_vocab = max(1, int(getattr(self.relation_bias, "num_embeddings", relation_ids.max().item() + 1 if relation_ids.numel() else 1)))
        level_ids = level_ids.remainder(level_vocab)
        relation_ids = relation_ids.remainder(relation_vocab)
        level_relation_bucket = ((level_ids * 131) + (relation_ids * 17)).remainder(self.buckets)

        batch_idx = torch.arange(batch, device=device)
        q_all = self.q_proj(hidden)
        level_bias_all = self.level_bias(level_ids)
        relation_bias_all = self.relation_bias(relation_ids)
        outputs: List[torch.Tensor] = []
        gate_terms: List[torch.Tensor] = []
        num_chunks = (seq_len + self.chunk_len - 1) // self.chunk_len
        for chunk_index in range(num_chunks):
            chunk_start = chunk_index * self.chunk_len
            chunk_end = min(seq_len, chunk_start + self.chunk_len)
            h = hidden[:, chunk_start, :]
            q = q_all[:, chunk_start, :] + level_bias_all[:, chunk_start, :] + relation_bias_all[:, chunk_start, :]
            candidate_ids = torch.stack(
                [
                    parent_bucket[:, chunk_start],
                    family_bucket[:, chunk_start],
                    sig_bucket[:, chunk_start],
                    level_relation_bucket[:, chunk_start],
                    prev_parent,
                    prev_signature,
                    prev_family,
                    torch.zeros(batch, device=device, dtype=torch.long),
                ],
                dim=1,
            )[:, : self.candidates]
            cand = (cache[batch_idx.unsqueeze(1), candidate_ids] * lattice_state.cache_decay_scale).to(dtype=q.dtype)
            scores = torch.einsum("br,bcr->bc", q, cand) / math.sqrt(float(self.r))
            weights = F.softmax(scores, dim=-1)
            ctx = torch.einsum("bc,bcr->br", weights, cand)
            delta = self.out_proj(ctx)
            gate = torch.sigmoid(self.gate(h))
            h_out = h + self.weight * gate * delta
            chunk_output = hidden[:, chunk_start:chunk_end, :] + (h_out - h).unsqueeze(1)
            outputs.append(chunk_output)
            gate_terms.append(gate.mean().detach())

            write_value = self.v_proj(h_out).to(dtype=cache.dtype)
            write_ids = candidate_ids[:, : min(4, candidate_ids.size(1))]
            decay_scale = lattice_state.cache_decay_scale * self.decay
            update = write_value.unsqueeze(1).expand(-1, write_ids.size(1), -1) / float(write_ids.size(1))
            cache_update = update / decay_scale.clamp_min(1e-12)
            cache.scatter_add_(
                1,
                write_ids.unsqueeze(-1).expand(-1, -1, self.r),
                cache_update,
            )
            counts.scatter_add_(
                1,
                write_ids.unsqueeze(-1),
                torch.ones(batch, write_ids.size(1), 1, device=device, dtype=counts.dtype),
            )
            decay_steps = lattice_state.cache_decay_steps + 1
            lattice_state = SignatureLatticeState(
                cache=cache,
                cache_decay_scale=decay_scale,
                cache_decay_steps=decay_steps,
                counts=counts,
                prev_parent_bucket=lattice_state.prev_parent_bucket,
                prev_signature_bucket=lattice_state.prev_signature_bucket,
                prev_family_bucket=lattice_state.prev_family_bucket,
            )
            prev_parent = parent_bucket[:, chunk_end - 1]
            prev_signature = sig_bucket[:, chunk_end - 1]
            prev_family = family_bucket[:, chunk_end - 1]
            lattice_state.prev_parent_bucket = prev_parent
            lattice_state.prev_signature_bucket = prev_signature
            lattice_state.prev_family_bucket = prev_family

        next_state: Optional[SignatureLatticeState] = None
        if return_state:
            next_state = SignatureLatticeState(
                cache=cache.detach(),
                cache_decay_scale=decay_scale.detach(),
                cache_decay_steps=decay_steps.detach(),
                counts=counts.detach(),
                prev_parent_bucket=prev_parent.detach(),
                prev_signature_bucket=prev_signature.detach(),
                prev_family_bucket=prev_family.detach(),
            )
        effective_cache = cache * lattice_state.cache_decay_scale
        stats = {
            "signature_lattice_cache_norm": effective_cache.detach().float().norm(dim=-1).mean(),
            "signature_lattice_gate_mean": torch.stack(gate_terms).mean() if gate_terms else torch.tensor(0.0, device=device),
            "signature_lattice_candidate_count": torch.tensor(float(self.candidates), device=device),
            "signature_lattice_enabled": torch.tensor(1.0, device=device),
            "signature_lattice_decay_scale_mean": lattice_state.cache_decay_scale.detach().float().mean(),
            }
        return torch.cat(outputs, dim=1), next_state, stats


class TokenMemoryCrossAttention(nn.Module):
    """Bounded causal cross-attention over recent token and signature memory."""

    def __init__(self, cfg: PrismalWaveConfig, quantization_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.quantization_config = quantization_config or QuantizationConfig()
        self.d_model = int(cfg.d_model)
        self.window = max(1, int(getattr(cfg, "token_memory_window", 96)))
        self.top_k = max(1, min(self.window, int(getattr(cfg, "token_memory_top_k", 8))))
        self.weight = max(0.0, float(getattr(cfg, "token_memory_weight", 0.18)))
        self.copy_bias = max(0.0, float(getattr(cfg, "token_memory_copy_bias", 0.75)))
        self.rare_token_cutoff = max(0, int(getattr(cfg, "token_memory_rare_token_cutoff", 2)))
        self.copy_min_confidence = max(0.0, min(1.0, float(getattr(cfg, "token_memory_copy_min_confidence", 0.35))))
        self.vocab_size = max(1, int(getattr(cfg, "vocab_size", 0)) or int(getattr(cfg, "base_vocab_size", 1)) or 1)
        self.q_proj = create_quantized_linear(self.d_model, self.d_model, bias=False, quantization_config=self.quantization_config)
        self.k_proj = create_quantized_linear(self.d_model, self.d_model, bias=False, quantization_config=self.quantization_config)
        self.v_proj = create_quantized_linear(self.d_model, self.d_model, bias=False, quantization_config=self.quantization_config)
        self.out_proj = create_quantized_linear(self.d_model, self.d_model, bias=False, quantization_config=self.quantization_config)
        self.gate_proj = create_quantized_linear(self.d_model, 1, bias=True, quantization_config=self.quantization_config)
        self.context_proj = create_quantized_linear(self.d_model, self.d_model, bias=False, quantization_config=self.quantization_config)

    def _anchor_tag(
        self,
        token_id: int,
        family_id: int,
        signature_id: int,
        level_id: int,
        relation_id: int,
        parent_id: int,
    ) -> int:
        tag = int(token_id) & 0xFFFFFFFF
        tag ^= (int(signature_id) & 0xFFFF) << 1
        tag = ((tag << 5) - tag) & 0xFFFFFFFF
        tag ^= (int(level_id) & 0xFFFF) << 11
        tag = ((tag << 7) - tag) & 0xFFFFFFFF
        tag ^= (int(relation_id) & 0xFFFF) << 17
        tag = (tag + ((int(family_id) & 0xFFFF) << 3) + ((int(parent_id) & 0xFFFF) << 13)) & 0xFFFFFFFF
        return tag

    def _anchor_flags(
        self,
        token_id: int,
        family_id: int,
        signature_id: int,
        level_id: int,
        relation_id: int,
        token_count: int,
    ) -> int:
        flags = 0
        special_ids = {
            int(getattr(self.cfg, "pad_id", 0)),
            int(getattr(self.cfg, "bos_id", 1)),
            int(getattr(self.cfg, "eos_id", 2)),
        }
        if int(token_id) in special_ids:
            return 0
        if token_count <= self.rare_token_cutoff:
            flags |= 0x01
        exact_id = SIGNATURE_RELATION_IDS.get("exact")
        prefix_id = SIGNATURE_RELATION_IDS.get("prefix")
        suffix_id = SIGNATURE_RELATION_IDS.get("suffix")
        containment_id = SIGNATURE_RELATION_IDS.get("containment")
        if exact_id is not None and int(relation_id) == exact_id:
            flags |= 0x02
        if prefix_id is not None and int(relation_id) == prefix_id:
            flags |= 0x04
        if suffix_id is not None and int(relation_id) == suffix_id:
            flags |= 0x08
        if containment_id is not None and int(relation_id) == containment_id:
            flags |= 0x10
        if int(level_id) in {
            int(SIGNATURE_LEVEL_IDS.get("char", 0)),
            int(SIGNATURE_LEVEL_IDS.get("piece", 0)),
        }:
            flags |= 0x20
        if int(signature_id) != 0 and int(family_id) != 0:
            flags |= 0x40
        return flags

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> TokenMemoryState:
        pad_token = int(getattr(self.cfg, "pad_id", 0))
        window = self.window
        return TokenMemoryState(
            token_ids=torch.full((batch_size, window), pad_token, device=device, dtype=torch.long),
            memory_keys=torch.zeros(batch_size, window, self.d_model, device=device, dtype=dtype),
            memory_values=torch.zeros(batch_size, window, self.d_model, device=device, dtype=dtype),
            family_ids=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            signature_ids=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            level_ids=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            relation_ids=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            parent_ids=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            lengths=torch.zeros(batch_size, device=device, dtype=torch.long),
            anchor_token_ids=torch.full((batch_size, window), pad_token, device=device, dtype=torch.long),
            anchor_span_ids=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            anchor_offsets=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            anchor_lengths=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            anchor_tags=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            anchor_flags=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            anchor_span_starts=torch.zeros(batch_size, window, device=device, dtype=torch.long),
            anchor_cursor_pos=torch.full((batch_size,), -1, device=device, dtype=torch.long),
            anchor_cursor_span_id=torch.zeros(batch_size, device=device, dtype=torch.long),
            anchor_cursor_offset=torch.zeros(batch_size, device=device, dtype=torch.long),
            anchor_cursor_length=torch.zeros(batch_size, device=device, dtype=torch.long),
            anchor_cursor_tag=torch.zeros(batch_size, device=device, dtype=torch.long),
            anchor_cursor_active=torch.zeros(batch_size, device=device, dtype=torch.bool),
        )

    def _coerce_state(
        self,
        state: Optional[TokenMemoryState],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TokenMemoryState:
        if state is None:
            return self.init_state(batch_size, device, dtype)
        token_ids = state.token_ids.to(device=device, dtype=torch.long)
        memory_keys = state.memory_keys.to(device=device, dtype=dtype)
        memory_values = state.memory_values.to(device=device, dtype=dtype)
        family_ids = state.family_ids.to(device=device, dtype=torch.long)
        signature_ids = state.signature_ids.to(device=device, dtype=torch.long)
        level_ids = state.level_ids.to(device=device, dtype=torch.long)
        relation_ids = state.relation_ids.to(device=device, dtype=torch.long)
        parent_ids = state.parent_ids.to(device=device, dtype=torch.long)
        lengths = state.lengths.to(device=device, dtype=torch.long)
        anchor_token_ids = state.anchor_token_ids.to(device=device, dtype=torch.long)
        anchor_span_ids = state.anchor_span_ids.to(device=device, dtype=torch.long)
        anchor_offsets = state.anchor_offsets.to(device=device, dtype=torch.long)
        anchor_lengths = state.anchor_lengths.to(device=device, dtype=torch.long)
        anchor_tags = state.anchor_tags.to(device=device, dtype=torch.long)
        anchor_flags = state.anchor_flags.to(device=device, dtype=torch.long)
        anchor_span_starts = state.anchor_span_starts.to(device=device, dtype=torch.long)
        anchor_cursor_pos = state.anchor_cursor_pos.to(device=device, dtype=torch.long)
        anchor_cursor_span_id = state.anchor_cursor_span_id.to(device=device, dtype=torch.long)
        anchor_cursor_offset = state.anchor_cursor_offset.to(device=device, dtype=torch.long)
        anchor_cursor_length = state.anchor_cursor_length.to(device=device, dtype=torch.long)
        anchor_cursor_tag = state.anchor_cursor_tag.to(device=device, dtype=torch.long)
        anchor_cursor_active = state.anchor_cursor_active.to(device=device, dtype=torch.bool)
        if token_ids.size(0) != batch_size:
            token_ids = token_ids[:1].expand(batch_size, -1).clone()
            memory_keys = memory_keys[:1].expand(batch_size, -1, -1).clone()
            memory_values = memory_values[:1].expand(batch_size, -1, -1).clone()
            family_ids = family_ids[:1].expand(batch_size, -1).clone()
            signature_ids = signature_ids[:1].expand(batch_size, -1).clone()
            level_ids = level_ids[:1].expand(batch_size, -1).clone()
            relation_ids = relation_ids[:1].expand(batch_size, -1).clone()
            parent_ids = parent_ids[:1].expand(batch_size, -1).clone()
            lengths = lengths[:1].expand(batch_size).clone()
            anchor_token_ids = anchor_token_ids[:1].expand(batch_size, -1).clone()
            anchor_span_ids = anchor_span_ids[:1].expand(batch_size, -1).clone()
            anchor_offsets = anchor_offsets[:1].expand(batch_size, -1).clone()
            anchor_lengths = anchor_lengths[:1].expand(batch_size, -1).clone()
            anchor_tags = anchor_tags[:1].expand(batch_size, -1).clone()
            anchor_flags = anchor_flags[:1].expand(batch_size, -1).clone()
            anchor_span_starts = anchor_span_starts[:1].expand(batch_size, -1).clone()
            anchor_cursor_pos = anchor_cursor_pos[:1].expand(batch_size).clone()
            anchor_cursor_span_id = anchor_cursor_span_id[:1].expand(batch_size).clone()
            anchor_cursor_offset = anchor_cursor_offset[:1].expand(batch_size).clone()
            anchor_cursor_length = anchor_cursor_length[:1].expand(batch_size).clone()
            anchor_cursor_tag = anchor_cursor_tag[:1].expand(batch_size).clone()
            anchor_cursor_active = anchor_cursor_active[:1].expand(batch_size).clone()
        return TokenMemoryState(
            token_ids=token_ids,
            memory_keys=memory_keys,
            memory_values=memory_values,
            family_ids=family_ids,
            signature_ids=signature_ids,
            level_ids=level_ids,
            relation_ids=relation_ids,
            parent_ids=parent_ids,
            lengths=lengths,
            anchor_token_ids=anchor_token_ids,
            anchor_span_ids=anchor_span_ids,
            anchor_offsets=anchor_offsets,
            anchor_lengths=anchor_lengths,
            anchor_tags=anchor_tags,
            anchor_flags=anchor_flags,
            anchor_span_starts=anchor_span_starts,
            anchor_cursor_pos=anchor_cursor_pos,
            anchor_cursor_span_id=anchor_cursor_span_id,
            anchor_cursor_offset=anchor_cursor_offset,
            anchor_cursor_length=anchor_cursor_length,
            anchor_cursor_tag=anchor_cursor_tag,
            anchor_cursor_active=anchor_cursor_active,
        )

    def _combine_contexts(
        self,
        family_context: Optional[torch.Tensor],
        level_context: Optional[torch.Tensor],
        relation_context: Optional[torch.Tensor],
        parent_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        weighted: List[torch.Tensor] = []
        if family_context is not None and family_context.numel() > 0:
            weighted.append(1.0 * family_context)
        if level_context is not None and level_context.numel() > 0:
            weighted.append(0.35 * level_context)
        if relation_context is not None and relation_context.numel() > 0:
            weighted.append(0.35 * relation_context)
        if parent_context is not None and parent_context.numel() > 0:
            weighted.append(0.55 * parent_context)
        if not weighted:
            return None
        combined = weighted[0]
        for tensor in weighted[1:]:
            combined = combined + tensor
        return self.context_proj(combined)

    def _append_state(
        self,
        state: TokenMemoryState,
        *,
        token_ids: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        family_ids: Optional[torch.Tensor],
        signature_ids: Optional[torch.Tensor],
        level_ids: Optional[torch.Tensor],
        relation_ids: Optional[torch.Tensor],
        parent_ids: Optional[torch.Tensor],
    ) -> TokenMemoryState:
        token_store = state.token_ids.clone()
        key_store = state.memory_keys.clone()
        value_store = state.memory_values.clone()
        family_store = state.family_ids.clone()
        signature_store = state.signature_ids.clone()
        level_store = state.level_ids.clone()
        relation_store = state.relation_ids.clone()
        parent_store = state.parent_ids.clone()
        lengths = state.lengths.clone()
        anchor_token_store = state.anchor_token_ids.clone()
        anchor_span_store = state.anchor_span_ids.clone()
        anchor_offset_store = state.anchor_offsets.clone()
        anchor_length_store = state.anchor_lengths.clone()
        anchor_tag_store = state.anchor_tags.clone()
        anchor_flag_store = state.anchor_flags.clone()
        anchor_span_start_store = state.anchor_span_starts.clone()
        anchor_cursor_pos = state.anchor_cursor_pos.clone()
        anchor_cursor_span_id = state.anchor_cursor_span_id.clone()
        anchor_cursor_offset = state.anchor_cursor_offset.clone()
        anchor_cursor_length = state.anchor_cursor_length.clone()
        anchor_cursor_tag = state.anchor_cursor_tag.clone()
        anchor_cursor_active = state.anchor_cursor_active.clone()
        batch_size = token_ids.size(0)
        pad_token = int(getattr(self.cfg, "pad_id", 0))
        family_ids = family_ids if family_ids is not None else torch.full_like(token_ids, 0)
        signature_ids = signature_ids if signature_ids is not None else torch.full_like(token_ids, 0)
        level_ids = level_ids if level_ids is not None else torch.full_like(token_ids, 0)
        relation_ids = relation_ids if relation_ids is not None else torch.full_like(token_ids, 0)
        parent_ids = parent_ids if parent_ids is not None else torch.full_like(token_ids, 0)
        for batch_idx in range(batch_size):
            length = int(lengths[batch_idx].item())
            if length < self.window:
                slot = length
                lengths[batch_idx] = length + 1
            else:
                slot = self.window - 1
                if self.window > 1:
                    # Keep the window bounded without materializing overlapping slice clones.
                    token_store[batch_idx] = torch.roll(token_store[batch_idx], shifts=-1, dims=0)
                    key_store[batch_idx] = torch.roll(key_store[batch_idx], shifts=-1, dims=0)
                    value_store[batch_idx] = torch.roll(value_store[batch_idx], shifts=-1, dims=0)
                    family_store[batch_idx] = torch.roll(family_store[batch_idx], shifts=-1, dims=0)
                    signature_store[batch_idx] = torch.roll(signature_store[batch_idx], shifts=-1, dims=0)
                    level_store[batch_idx] = torch.roll(level_store[batch_idx], shifts=-1, dims=0)
                    relation_store[batch_idx] = torch.roll(relation_store[batch_idx], shifts=-1, dims=0)
                    parent_store[batch_idx] = torch.roll(parent_store[batch_idx], shifts=-1, dims=0)
                    anchor_token_store[batch_idx] = torch.roll(anchor_token_store[batch_idx], shifts=-1, dims=0)
                    anchor_span_store[batch_idx] = torch.roll(anchor_span_store[batch_idx], shifts=-1, dims=0)
                    anchor_offset_store[batch_idx] = torch.roll(anchor_offset_store[batch_idx], shifts=-1, dims=0)
                    anchor_length_store[batch_idx] = torch.roll(anchor_length_store[batch_idx], shifts=-1, dims=0)
                    anchor_tag_store[batch_idx] = torch.roll(anchor_tag_store[batch_idx], shifts=-1, dims=0)
                    anchor_flag_store[batch_idx] = torch.roll(anchor_flag_store[batch_idx], shifts=-1, dims=0)
                    anchor_span_start_store[batch_idx] = torch.roll(anchor_span_start_store[batch_idx], shifts=-1, dims=0)
                    if anchor_cursor_active[batch_idx]:
                        anchor_cursor_pos[batch_idx] = anchor_cursor_pos[batch_idx] - 1
                        if int(anchor_cursor_pos[batch_idx].item()) < 0:
                            anchor_cursor_active[batch_idx] = False
                            anchor_cursor_pos[batch_idx] = -1
                            anchor_cursor_span_id[batch_idx] = 0
                            anchor_cursor_offset[batch_idx] = 0
                            anchor_cursor_length[batch_idx] = 0
                            anchor_cursor_tag[batch_idx] = 0
                lengths[batch_idx] = self.window
            token_store[batch_idx, slot] = int(token_ids[batch_idx].item()) if token_ids.dim() == 2 else int(token_ids[batch_idx])
            key_store[batch_idx, slot] = key[batch_idx]
            value_store[batch_idx, slot] = value[batch_idx]
            family_store[batch_idx, slot] = int(family_ids[batch_idx].item()) if family_ids.dim() == 2 else int(family_ids[batch_idx])
            signature_store[batch_idx, slot] = int(signature_ids[batch_idx].item()) if signature_ids.dim() == 2 else int(signature_ids[batch_idx])
            level_store[batch_idx, slot] = int(level_ids[batch_idx].item()) if level_ids.dim() == 2 else int(level_ids[batch_idx])
            relation_store[batch_idx, slot] = int(relation_ids[batch_idx].item()) if relation_ids.dim() == 2 else int(relation_ids[batch_idx])
            parent_store[batch_idx, slot] = int(parent_ids[batch_idx].item()) if parent_ids.dim() == 2 else int(parent_ids[batch_idx])
            token_id = int(token_store[batch_idx, slot].item())
            family_id = int(family_store[batch_idx, slot].item())
            signature_id = int(signature_store[batch_idx, slot].item())
            level_id = int(level_store[batch_idx, slot].item())
            relation_id = int(relation_store[batch_idx, slot].item())
            parent_id = int(parent_store[batch_idx, slot].item())
            token_window = token_store[batch_idx, : lengths[batch_idx].item()]
            token_count = int(token_window.eq(token_id).sum().item()) if token_window.numel() > 0 else 0
            tag = self._anchor_tag(token_id, family_id, signature_id, level_id, relation_id, parent_id)
            anchor_flag = self._anchor_flags(
                token_id,
                family_id,
                signature_id,
                level_id,
                relation_id,
                token_count,
            )
            if anchor_flag > 0:
                prev_idx = slot - 1
                prev_continues = False
                prev_span_id = 0
                prev_offset = -1
                prev_span_start = slot
                prev_tag = 0
                if prev_idx >= 0 and anchor_flag_store[batch_idx, prev_idx].item() > 0:
                    prev_span_id = int(anchor_span_store[batch_idx, prev_idx].item())
                    prev_offset = int(anchor_offset_store[batch_idx, prev_idx].item())
                    prev_span_start = int(anchor_span_start_store[batch_idx, prev_idx].item())
                    prev_tag = int(anchor_tag_store[batch_idx, prev_idx].item())
                    prev_continues = prev_span_id > 0 and prev_tag == tag and prev_offset + 1 >= 1
                if prev_continues:
                    span_id = prev_span_id
                    offset = prev_offset + 1
                    span_start = prev_span_start
                else:
                    span_id = int(anchor_span_store[batch_idx, : slot].max().item()) + 1 if slot > 0 else 1
                    offset = 0
                    span_start = slot
                span_len = offset + 1
                anchor_token_store[batch_idx, slot] = token_store[batch_idx, slot]
                anchor_span_store[batch_idx, slot] = span_id
                anchor_offset_store[batch_idx, slot] = offset
                anchor_length_store[batch_idx, slot] = span_len
                anchor_tag_store[batch_idx, slot] = tag
                anchor_flag_store[batch_idx, slot] = anchor_flag
                anchor_span_start_store[batch_idx, slot] = span_start
            else:
                anchor_token_store[batch_idx, slot] = token_store[batch_idx, slot]
                anchor_span_store[batch_idx, slot] = 0
                anchor_offset_store[batch_idx, slot] = 0
                anchor_length_store[batch_idx, slot] = 0
                anchor_tag_store[batch_idx, slot] = 0
                anchor_flag_store[batch_idx, slot] = 0
                anchor_span_start_store[batch_idx, slot] = slot
            if token_store[batch_idx, slot].item() == pad_token and int(token_ids[batch_idx].item()) == pad_token:
                continue
        return TokenMemoryState(
            token_ids=token_store,
            memory_keys=key_store,
            memory_values=value_store,
            family_ids=family_store,
            signature_ids=signature_store,
            level_ids=level_store,
            relation_ids=relation_store,
            parent_ids=parent_store,
            lengths=lengths,
            anchor_token_ids=anchor_token_store,
            anchor_span_ids=anchor_span_store,
            anchor_offsets=anchor_offset_store,
            anchor_lengths=anchor_length_store,
            anchor_tags=anchor_tag_store,
            anchor_flags=anchor_flag_store,
            anchor_span_starts=anchor_span_start_store,
            anchor_cursor_pos=anchor_cursor_pos,
            anchor_cursor_span_id=anchor_cursor_span_id,
            anchor_cursor_offset=anchor_cursor_offset,
            anchor_cursor_length=anchor_cursor_length,
            anchor_cursor_tag=anchor_cursor_tag,
            anchor_cursor_active=anchor_cursor_active,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        family_context: Optional[torch.Tensor] = None,
        level_context: Optional[torch.Tensor] = None,
        relation_context: Optional[torch.Tensor] = None,
        parent_context: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        state: Optional[TokenMemoryState] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[TokenMemoryState], Dict[str, torch.Tensor]]:
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        if hidden.dim() != 3:
            raise ValueError("hidden must have shape (batch, seq_len, d_model) or (batch, d_model)")
        batch, seq_len, _ = hidden.shape
        device = hidden.device
        dtype = hidden.dtype if hidden.dtype.is_floating_point else torch.float32
        memory_state = self._coerce_state(state, batch, device, dtype)
        if token_ids is None:
            token_ids = torch.zeros(batch, seq_len, device=device, dtype=torch.long)
        else:
            token_ids = token_ids[:, :seq_len].to(device=device, dtype=torch.long)
        family_ids = signature_family_ids[:, :seq_len].to(device=device, dtype=torch.long) if signature_family_ids is not None else None
        signature_ids = signature_ids[:, :seq_len].to(device=device, dtype=torch.long) if signature_ids is not None else None
        level_ids = signature_level_ids[:, :seq_len].to(device=device, dtype=torch.long) if signature_level_ids is not None else None
        relation_ids = signature_relation_ids[:, :seq_len].to(device=device, dtype=torch.long) if signature_relation_ids is not None else None
        parent_ids = parent_signature_ids[:, :seq_len].to(device=device, dtype=torch.long) if parent_signature_ids is not None else None
        family_context = family_context[:, :seq_len] if family_context is not None and family_context.numel() > 0 else None
        level_context = level_context[:, :seq_len] if level_context is not None and level_context.numel() > 0 else None
        relation_context = relation_context[:, :seq_len] if relation_context is not None and relation_context.numel() > 0 else None
        parent_context = parent_context[:, :seq_len] if parent_context is not None and parent_context.numel() > 0 else None
        outputs: List[torch.Tensor] = []
        gate_terms: List[torch.Tensor] = []
        copy_conf_terms: List[torch.Tensor] = []
        copy_logits = torch.zeros(batch, self.vocab_size, device=device, dtype=dtype)
        timing_totals: Dict[str, float] = {}
        scale = math.sqrt(float(self.d_model))
        zero_long = torch.zeros(batch, device=device, dtype=torch.long)
        special_token_ids = {
            int(getattr(self.cfg, "pad_id", 0)),
            int(getattr(self.cfg, "bos_id", 1)),
            int(getattr(self.cfg, "eos_id", 2)),
        }
        for step_idx in range(seq_len):
            step_hidden = hidden[:, step_idx, :]
            step_token_ids = token_ids[:, step_idx] if token_ids.numel() > 0 else zero_long
            step_family_ids = family_ids[:, step_idx] if family_ids is not None else None
            step_signature_ids = signature_ids[:, step_idx] if signature_ids is not None else None
            step_level_ids = level_ids[:, step_idx] if level_ids is not None else None
            step_relation_ids = relation_ids[:, step_idx] if relation_ids is not None else None
            step_parent_ids = parent_ids[:, step_idx] if parent_ids is not None else None
            meta_context = self._combine_contexts(
                family_context[:, step_idx, :] if family_context is not None else None,
                level_context[:, step_idx, :] if level_context is not None else None,
                relation_context[:, step_idx, :] if relation_context is not None else None,
                parent_context[:, step_idx, :] if parent_context is not None else None,
            )
            query_input = step_hidden if meta_context is None else step_hidden + meta_context
            start_query = time.perf_counter()
            query = self.q_proj(query_input)
            if bool(getattr(self.cfg, "profile_runtime", False)):
                timing_totals["timing_token_memory_query_ms"] = timing_totals.get("timing_token_memory_query_ms", 0.0) + (
                    (time.perf_counter() - start_query) * 1000.0
                )
            memory_context = torch.zeros_like(step_hidden)
            step_copy_logits = torch.zeros(batch, self.vocab_size, device=device, dtype=dtype)
            step_confidence = torch.zeros(batch, device=device, dtype=dtype)
            batch_top_idx: List[Optional[torch.Tensor]] = [None] * batch
            batch_usable_mask: List[Optional[torch.Tensor]] = [None] * batch
            start_select = time.perf_counter()
            for batch_idx in range(batch):
                length = int(memory_state.lengths[batch_idx].item())
                if length <= 0:
                    continue
                key_bank = memory_state.memory_keys[batch_idx, :length]
                value_bank = memory_state.memory_values[batch_idx, :length]
                if key_bank.numel() == 0:
                    continue
                scores = torch.matmul(key_bank, query[batch_idx]) / max(scale, 1e-6)
                if step_family_ids is not None:
                    scores = scores + 0.12 * memory_state.family_ids[batch_idx, :length].eq(step_family_ids[batch_idx]).to(dtype)
                if step_signature_ids is not None:
                    scores = scores + 0.18 * memory_state.signature_ids[batch_idx, :length].eq(step_signature_ids[batch_idx]).to(dtype)
                if step_level_ids is not None:
                    scores = scores + 0.05 * memory_state.level_ids[batch_idx, :length].eq(step_level_ids[batch_idx]).to(dtype)
                if step_relation_ids is not None:
                    scores = scores + 0.05 * memory_state.relation_ids[batch_idx, :length].eq(step_relation_ids[batch_idx]).to(dtype)
                if step_parent_ids is not None:
                    scores = scores + 0.08 * memory_state.parent_ids[batch_idx, :length].eq(step_parent_ids[batch_idx]).to(dtype)
                k = min(self.top_k, length)
                top_scores, top_idx = torch.topk(scores, k=k)
                weights = F.softmax(top_scores, dim=-1)
                selected_values = value_bank.index_select(0, top_idx)
                memory_context[batch_idx] = torch.sum(weights.unsqueeze(-1) * selected_values, dim=0)
                step_confidence[batch_idx] = weights.max()
                if self.copy_bias > 0.0 and self.vocab_size > 0:
                    selected_token_ids = memory_state.token_ids[batch_idx, :length].index_select(0, top_idx)
                    window_tokens = memory_state.token_ids[batch_idx, :length].clamp_min(0).to(dtype=torch.long)
                    unique_tokens, unique_counts = torch.unique(window_tokens, sorted=True, return_counts=True)
                    if unique_tokens.numel() > 0:
                        lookup = torch.searchsorted(unique_tokens, selected_token_ids)
                        valid_lookup = lookup < unique_tokens.numel()
                        matched_tokens = torch.zeros_like(selected_token_ids, dtype=torch.bool)
                        if valid_lookup.any():
                            matched_lookup = lookup[valid_lookup]
                            matched_tokens[valid_lookup] = unique_tokens.index_select(0, matched_lookup).eq(selected_token_ids[valid_lookup])
                        token_counts = torch.zeros_like(selected_token_ids, dtype=unique_counts.dtype)
                        if matched_tokens.any():
                            token_counts[matched_tokens] = unique_counts.index_select(0, lookup[matched_tokens])
                    else:
                        token_counts = torch.zeros_like(selected_token_ids, dtype=torch.long)
                    rare_mask = token_counts <= self.rare_token_cutoff
                    special_mask = torch.ones_like(selected_token_ids, dtype=torch.bool)
                    for special_id in special_token_ids:
                        special_mask = special_mask & selected_token_ids.ne(special_id)
                    valid_token_mask = (selected_token_ids >= 0) & (selected_token_ids < self.vocab_size)
                    usable_mask = rare_mask & special_mask & valid_token_mask
                    if usable_mask.any():
                        step_copy_logits[batch_idx].scatter_add_(
                            0,
                            selected_token_ids[usable_mask],
                            weights[usable_mask] * self.copy_bias * step_confidence[batch_idx].clamp(min=0.0, max=1.0),
                        )
                    batch_top_idx[batch_idx] = top_idx
                    batch_usable_mask[batch_idx] = usable_mask
            if bool(getattr(self.cfg, "profile_runtime", False)):
                timing_totals["timing_token_memory_select_ms"] = timing_totals.get("timing_token_memory_select_ms", 0.0) + (
                    (time.perf_counter() - start_select) * 1000.0
                )
            for batch_idx in range(batch):
                if step_confidence[batch_idx].item() < self.copy_min_confidence:
                    continue
                if not (step_copy_logits[batch_idx].abs().sum().item() > 0.0):
                    continue
                candidate_top_idx = batch_top_idx[batch_idx]
                candidate_mask = batch_usable_mask[batch_idx]
                if candidate_top_idx is None or candidate_mask is None:
                    continue
                candidate_pos: Optional[int] = None
                for cand_idx in range(candidate_top_idx.size(0)):
                    if not bool(candidate_mask[cand_idx].item()):
                        continue
                    candidate_pos = int(candidate_top_idx[cand_idx].item())
                    break
                if candidate_pos is None:
                    continue
                span_id = int(memory_state.anchor_span_ids[batch_idx, candidate_pos].item())
                span_start = int(memory_state.anchor_span_starts[batch_idx, candidate_pos].item())
                span_len = int(memory_state.anchor_lengths[batch_idx, candidate_pos].item())
                span_tag = int(memory_state.anchor_tags[batch_idx, candidate_pos].item())
                if span_id <= 0 or span_len <= 0:
                    continue
                memory_state.anchor_cursor_active[batch_idx] = True
                memory_state.anchor_cursor_pos[batch_idx] = span_start
                memory_state.anchor_cursor_span_id[batch_idx] = span_id
                memory_state.anchor_cursor_offset[batch_idx] = 0
                memory_state.anchor_cursor_length[batch_idx] = span_len
                memory_state.anchor_cursor_tag[batch_idx] = span_tag
            start_project = time.perf_counter()
            gate = torch.sigmoid(self.gate_proj(query_input))
            delta = self.out_proj(memory_context)
            step_output = step_hidden + self.weight * gate * delta
            if bool(getattr(self.cfg, "profile_runtime", False)):
                timing_totals["timing_token_memory_project_ms"] = timing_totals.get("timing_token_memory_project_ms", 0.0) + (
                    (time.perf_counter() - start_project) * 1000.0
                )
            outputs.append(step_output.unsqueeze(1))
            gate_terms.append(gate.mean().detach())
            copy_conf_terms.append(step_confidence.mean().detach())
            copy_logits = step_copy_logits
            start_append = time.perf_counter()
            memory_state = self._append_state(
                memory_state,
                token_ids=step_token_ids,
                key=self.k_proj(query_input),
                value=self.v_proj(step_output),
                family_ids=step_family_ids,
                signature_ids=step_signature_ids,
                level_ids=step_level_ids,
                relation_ids=step_relation_ids,
                parent_ids=step_parent_ids,
            )
            if bool(getattr(self.cfg, "profile_runtime", False)):
                timing_totals["timing_token_memory_append_ms"] = timing_totals.get("timing_token_memory_append_ms", 0.0) + (
                    (time.perf_counter() - start_append) * 1000.0
                )

        next_state: Optional[TokenMemoryState] = None
        if return_state:
            next_state = TokenMemoryState(
                token_ids=memory_state.token_ids.detach(),
                memory_keys=memory_state.memory_keys.detach(),
                memory_values=memory_state.memory_values.detach(),
                family_ids=memory_state.family_ids.detach(),
                signature_ids=memory_state.signature_ids.detach(),
                level_ids=memory_state.level_ids.detach(),
                relation_ids=memory_state.relation_ids.detach(),
                parent_ids=memory_state.parent_ids.detach(),
                lengths=memory_state.lengths.detach(),
                anchor_token_ids=memory_state.anchor_token_ids.detach(),
                anchor_span_ids=memory_state.anchor_span_ids.detach(),
                anchor_offsets=memory_state.anchor_offsets.detach(),
                anchor_lengths=memory_state.anchor_lengths.detach(),
                anchor_tags=memory_state.anchor_tags.detach(),
                anchor_flags=memory_state.anchor_flags.detach(),
                anchor_span_starts=memory_state.anchor_span_starts.detach(),
                anchor_cursor_pos=memory_state.anchor_cursor_pos.detach(),
                anchor_cursor_span_id=memory_state.anchor_cursor_span_id.detach(),
                anchor_cursor_offset=memory_state.anchor_cursor_offset.detach(),
                anchor_cursor_length=memory_state.anchor_cursor_length.detach(),
                anchor_cursor_tag=memory_state.anchor_cursor_tag.detach(),
                anchor_cursor_active=memory_state.anchor_cursor_active.detach(),
            )
        stats = {
            "token_memory_enabled": torch.tensor(1.0, device=device),
            "copy_attention_enabled": torch.tensor(1.0, device=device),
            "token_memory_gate_mean": torch.stack(gate_terms).mean() if gate_terms else torch.tensor(0.0, device=device),
            "copy_attention_gate_mean": torch.stack(gate_terms).mean() if gate_terms else torch.tensor(0.0, device=device),
            "token_memory_copy_confidence": torch.stack(copy_conf_terms).mean() if copy_conf_terms else torch.tensor(0.0, device=device),
            "copy_attention_max_weight": torch.stack(copy_conf_terms).max() if copy_conf_terms else torch.tensor(0.0, device=device),
            "token_memory_memory_fill": (
                memory_state.lengths.float().mean().clamp(min=0.0) / max(float(self.window), 1.0)
            ).detach(),
            "copy_attention_memory_fill": (
                memory_state.lengths.float().mean().clamp(min=0.0) / max(float(self.window), 1.0)
            ).detach(),
            "token_memory_window": torch.tensor(float(self.window), device=device),
            "token_memory_top_k": torch.tensor(float(self.top_k), device=device),
            "token_memory_copy_logits": copy_logits.detach(),
            "copy_attention_candidate_count": torch.tensor(float(self.top_k), device=device),
        }
        if timing_totals:
            timing_totals["timing_token_memory_total_ms"] = sum(timing_totals.values())
            for key, value in timing_totals.items():
                stats[key] = torch.tensor(value, device=device)
        return torch.cat(outputs, dim=1), next_state, stats


@dataclass
class PrismalWaveOutput:
    logits: torch.Tensor
    input_signature: torch.Tensor
    output_signature: torch.Tensor
    path_logits: torch.Tensor
    route_stats: Dict[str, torch.Tensor]
    ce_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    aux_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    slot_state: Optional[torch.Tensor | PrismalTorusState] = None
    signature_lattice_state: Optional[SignatureLatticeState] = None
    token_memory_state: Optional[TokenMemoryState] = None


@dataclass
class PrismalTorusFrame:
    input_ids: torch.Tensor
    hidden: torch.Tensor
    input_signature: torch.Tensor
    signature_lattice_state: Optional[SignatureLatticeState]
    token_memory_state: Optional[TokenMemoryState]
    lattice_stats: Dict[str, torch.Tensor]
    token_memory_stats: Dict[str, torch.Tensor]
    prep_timings: Dict[str, float]
    signature_family_ids: Optional[torch.Tensor]
    signature_ids: Optional[torch.Tensor]
    signature_level_ids: Optional[torch.Tensor]
    signature_relation_ids: Optional[torch.Tensor]
    parent_signature_ids: Optional[torch.Tensor]
    family_context_seq: Optional[torch.Tensor]
    level_context_seq: Optional[torch.Tensor]
    relation_context_seq: Optional[torch.Tensor]
    parent_context_seq: Optional[torch.Tensor]
    position_offset: int


class PrismalEmitterRouter(nn.Module):
    """Reusable operator router that blends content with hierarchy-aware compatibility."""

    def __init__(self, cfg: PrismalWaveConfig, quantization_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.quantization_config = quantization_config or QuantizationConfig()
        self.capacity_growth_locked = False
        d = cfg.d_model
        self.query_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.value_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.signature_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.hierarchy_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.phase_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.frequency_proj = create_quantized_linear(d, 1, quantization_config=self.quantization_config)
        self.route_gate = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.out_proj = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.slot_query = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.slot_value = create_quantized_linear(d, d, quantization_config=self.quantization_config)
        self.path_coord_proj = create_quantized_linear(d, 2, quantization_config=self.quantization_config)
        self.emitter_bank = nn.Parameter(torch.randn(cfg.n_emitters, d) * 0.02)
        self.operator_hierarchy_bank = nn.Parameter(torch.randn(cfg.n_emitters, d) * 0.02)
        self.emitter_phase = nn.Parameter(torch.rand(cfg.n_emitters, d) * (2.0 * math.pi) - math.pi)
        self.emitter_frequency = nn.Parameter(torch.rand(cfg.n_emitters, 1) + 0.5)
        self.slot_seed = nn.Parameter(torch.randn(cfg.n_slots, d) * 0.02)
        self.path_basis = nn.Parameter(torch.randn(cfg.n_paths, d) * 0.02)
        self.signature_vocab_size = max(8, int(getattr(cfg, "signature_vocab_size", 0)) or 8)
        self.signature_bucket_vocab_size = max(8, int(getattr(cfg, "signature_bucket_vocab_size", 0)) or 8)
        self.signature_level_vocab_size = max(8, int(getattr(cfg, "signature_level_vocab_size", 0)) or len(SIGNATURE_LEVEL_IDS))
        self.signature_relation_vocab_size = max(8, int(getattr(cfg, "signature_relation_vocab_size", 0)) or len(SIGNATURE_RELATION_IDS))
        self.signature_embedding = self._make_embedding(self.signature_vocab_size, d)
        self.family_embedding = self._make_embedding(self.signature_bucket_vocab_size, d)
        self.level_embedding = self._make_embedding(self.signature_level_vocab_size, d)
        self.relation_embedding = self._make_embedding(self.signature_relation_vocab_size, d)
        self.parent_embedding = self._make_embedding(self.signature_vocab_size, d)
        grid_h = int(cfg.emitter_grid_height or round(math.sqrt(cfg.n_emitters)) or 1)
        grid_w = int(cfg.emitter_grid_width or math.ceil(cfg.n_emitters / max(grid_h, 1)) or 1)
        self.grid_height = max(1, grid_h)
        self.grid_width = max(1, grid_w)
        coords = []
        for idx in range(cfg.n_emitters):
            y = idx // self.grid_width
            x = idx % self.grid_width
            coords.append((float(y % self.grid_height), float(x)))
        self.register_buffer("emitter_grid_coords", torch.tensor(coords, dtype=torch.float32), persistent=False)

    def _make_embedding(self, size: int, d_model: int) -> nn.Embedding:
        size = max(1, int(size))
        if self.quantization_config is not None and self.quantization_config.enabled:
            return create_quantized_embedding(size, d_model, quantization_config=self.quantization_config)
        return nn.Embedding(size, d_model)

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
        param = next(module.parameters())
        return param.device, param.dtype

    def _resize_embedding(self, name: str, required_size: int) -> None:
        required_size = max(1, int(required_size))
        embed: nn.Embedding = getattr(self, name)
        if required_size <= embed.num_embeddings:
            return
        device, _dtype = self._module_device_dtype(embed)
        new_embed = self._make_embedding(required_size, embed.embedding_dim).to(device)
        old_weight = getattr(embed, "weight", getattr(embed, "embedding_matrix", None))
        new_weight = getattr(new_embed, "weight", getattr(new_embed, "embedding_matrix", None))
        if old_weight is None or new_weight is None:
            raise AttributeError(f"Unsupported embedding module for resize: {type(embed).__name__}")
        with torch.no_grad():
            new_weight[: embed.num_embeddings].copy_(old_weight)
            nn.init.normal_(new_weight[embed.num_embeddings :], mean=0.0, std=0.02)
        setattr(self, name, new_embed)

    def _ensure_hierarchy_capacity(
        self,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
    ) -> None:
        max_signature = -1
        max_family = -1
        max_level = -1
        max_relation = -1
        if signature_ids is not None and signature_ids.numel() > 0:
            max_signature = max(max_signature, int(signature_ids.detach().long().max().item()))
        if signature_family_ids is not None and signature_family_ids.numel() > 0:
            max_family = max(max_family, int(signature_family_ids.detach().long().max().item()))
        if signature_level_ids is not None and signature_level_ids.numel() > 0:
            max_level = max(max_level, int(signature_level_ids.detach().long().max().item()))
        if signature_relation_ids is not None and signature_relation_ids.numel() > 0:
            max_relation = max(max_relation, int(signature_relation_ids.detach().long().max().item()))
        if parent_signature_ids is not None and parent_signature_ids.numel() > 0:
            max_signature = max(max_signature, int(parent_signature_ids.detach().long().max().item()))
        grow_signature = max_signature >= self.signature_embedding.num_embeddings
        grow_family = max_family >= self.family_embedding.num_embeddings
        grow_level = max_level >= self.level_embedding.num_embeddings
        grow_relation = max_relation >= self.relation_embedding.num_embeddings
        if self.training and self.capacity_growth_locked and (grow_signature or grow_family or grow_level or grow_relation):
            raise RuntimeError(
                "Hierarchy embedding capacity grew after optimizer init; pre-grow the hierarchy embeddings before training."
            )
        if max_signature >= 0:
            self._resize_embedding("signature_embedding", max_signature + 1)
            self._resize_embedding("parent_embedding", max_signature + 1)
        if max_family >= 0:
            self._resize_embedding("family_embedding", max_family + 1)
        if max_level >= 0:
            self._resize_embedding("level_embedding", max_level + 1)
        if max_relation >= 0:
            self._resize_embedding("relation_embedding", max_relation + 1)

    def set_capacity_growth_locked(self, locked: bool = True) -> None:
        self.capacity_growth_locked = bool(locked)

    def ensure_hierarchy_capacity_for_sizes(
        self,
        signature_vocab_size: int,
        family_vocab_size: int,
        level_vocab_size: int,
        relation_vocab_size: int,
    ) -> None:
        prev_locked = self.capacity_growth_locked
        try:
            self.capacity_growth_locked = False
            family_cap = max(0, int(family_vocab_size) - 1)
            signature_cap = max(0, int(signature_vocab_size) - 1)
            level_cap = max(0, int(level_vocab_size) - 1)
            relation_cap = max(0, int(relation_vocab_size) - 1)
            self._ensure_hierarchy_capacity(
                signature_family_ids=torch.tensor([family_cap], device=self.emitter_bank.device, dtype=torch.long),
                signature_ids=torch.tensor([signature_cap], device=self.emitter_bank.device, dtype=torch.long),
                signature_level_ids=torch.tensor([level_cap], device=self.emitter_bank.device, dtype=torch.long),
                signature_relation_ids=torch.tensor([relation_cap], device=self.emitter_bank.device, dtype=torch.long),
                parent_signature_ids=torch.tensor([signature_cap], device=self.emitter_bank.device, dtype=torch.long),
            )
        finally:
            self.capacity_growth_locked = prev_locked

    def _hierarchy_context(
        self,
        hidden: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._ensure_hierarchy_capacity(
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )
        family_ids = signature_family_ids if signature_family_ids is not None else signature_ids
        parent_ids = parent_signature_ids if parent_signature_ids is not None else family_ids
        tensors: List[torch.Tensor] = []
        if signature_ids is not None and signature_ids.numel() > 0:
            tensors.append(self.signature_embedding(signature_ids.clamp(min=0)))
        if family_ids is not None and family_ids.numel() > 0:
            tensors.append(float(getattr(self.cfg, "emitter_family_share", 1.0)) * self.family_embedding(family_ids.clamp(min=0)))
        if signature_level_ids is not None and signature_level_ids.numel() > 0:
            tensors.append(float(getattr(self.cfg, "emitter_level_share", 0.35)) * self.level_embedding(signature_level_ids.clamp(min=0)))
        if signature_relation_ids is not None and signature_relation_ids.numel() > 0:
            tensors.append(float(getattr(self.cfg, "emitter_relation_share", 0.50)) * self.relation_embedding(signature_relation_ids.clamp(min=0)))
        if parent_ids is not None and parent_ids.numel() > 0:
            tensors.append(float(getattr(self.cfg, "emitter_parent_share", 0.75)) * self.parent_embedding(parent_ids.clamp(min=0)))
        if not tensors:
            return torch.zeros_like(hidden)
        hierarchy_context = tensors[0]
        for tensor in tensors[1:]:
            hierarchy_context = hierarchy_context + tensor
        hierarchy_context = self.hierarchy_proj(hierarchy_context)
        return F.normalize(hierarchy_context, dim=-1)

    def init_slots(
        self,
        batch_size: int,
        device: torch.device,
        slot_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state_dtype = getattr(self, "precision_state_dtype", None)
        if not isinstance(state_dtype, torch.dtype):
            state_dtype = torch.float32
        if slot_state is not None:
            if slot_state.dim() == 2:
                slot_state = slot_state.unsqueeze(0)
            if slot_state.size(0) != batch_size:
                slot_state = slot_state.expand(batch_size, -1, -1)
            return slot_state.to(device=device, dtype=state_dtype).clone()
        return self.slot_seed.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=state_dtype).clone()

    def signature(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        return F.normalize(self.signature_proj(pooled), dim=-1)

    def route(
        self,
        hidden: torch.Tensor,
        slots: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        emitter_bank_override: Optional[torch.Tensor] = None,
        operator_hierarchy_bank_override: Optional[torch.Tensor] = None,
        path_index: int,
        layer_index: int,
        torus_center: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch, seq_len, dim = hidden.shape
        state_dtype = getattr(self, "precision_state_dtype", None)
        if isinstance(state_dtype, torch.dtype) and slots.dtype != state_dtype:
            slots = slots.to(dtype=state_dtype)
        query = self.query_proj(hidden)
        value = self.value_proj(hidden)
        phase_state = torch.tanh(self.phase_proj(hidden)) * math.pi
        token_frequency = F.softplus(self.frequency_proj(hidden)) + 1e-3
        hierarchy_context = self._hierarchy_context(
            hidden,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )

        path_vector = self.path_basis[path_index % self.path_basis.size(0)]
        path_vector = torch.tanh(path_vector) * math.pi
        emitter_phase = torch.tanh(self.emitter_phase + 0.1 * path_vector.unsqueeze(0)) * math.pi
        emitter_frequency = F.softplus(self.emitter_frequency)
        emitter_coords = self.emitter_grid_coords.to(hidden.device)
        coord_scale = torch.tensor(
            [float(max(self.grid_height - 1, 1)), float(max(self.grid_width - 1, 1))],
            device=hidden.device,
        )
        path_coord = torch.sigmoid(self.path_coord_proj(path_vector.unsqueeze(0))) * coord_scale
        path_coord = path_coord.squeeze(0)

        phase_dist = (1.0 - torch.cos(phase_state.unsqueeze(2) - emitter_phase.view(1, 1, -1, dim))).mean(dim=-1)
        freq_dist = torch.abs(token_frequency - emitter_frequency.view(1, 1, -1))
        grid_h = torch.tensor(float(self.grid_height), device=hidden.device)
        grid_w = torch.tensor(float(self.grid_width), device=hidden.device)
        coord_y = torch.abs(emitter_coords[:, 0].view(1, 1, -1) - path_coord[0]).clamp(max=float(self.grid_height))
        coord_x = torch.abs(emitter_coords[:, 1].view(1, 1, -1) - path_coord[1]).clamp(max=float(self.grid_width))
        coord_y = torch.minimum(coord_y, grid_h - coord_y)
        coord_x = torch.minimum(coord_x, grid_w - coord_x)
        grid_dist = coord_y + coord_x
        path_bias = (hidden * path_vector.view(1, 1, -1)).sum(dim=-1, keepdim=True) / math.sqrt(dim)

        emitter_bank = emitter_bank_override if emitter_bank_override is not None else self.emitter_bank
        operator_hierarchy_bank = (
            operator_hierarchy_bank_override if operator_hierarchy_bank_override is not None else self.operator_hierarchy_bank
        )
        content_scores = torch.einsum(
            "btd,ed->bte",
            F.normalize(query, dim=-1, eps=1e-6),
            F.normalize(emitter_bank, dim=-1, eps=1e-6),
        ) * math.sqrt(dim)
        hierarchy_scores = torch.einsum(
            "btd,ed->bte",
            F.normalize(hierarchy_context, dim=-1, eps=1e-6),
            F.normalize(operator_hierarchy_bank, dim=-1, eps=1e-6),
        ) * math.sqrt(dim)
        hierarchy_context_norm = hierarchy_context.norm(dim=-1).mean()
        hierarchy_probs = F.softmax(hierarchy_scores / max(self.cfg.router_temperature, 1e-3), dim=-1)
        hierarchy_entropy = -(hierarchy_probs * torch.log(hierarchy_probs + 1e-8)).sum(dim=-1).mean()
        hierarchy_score_weight = float(getattr(self.cfg, "emitter_hierarchy_score_weight", 0.25))
        scores = (
            content_scores
            + hierarchy_score_weight * hierarchy_scores
            - self.cfg.torus_weight * phase_dist
            - self.cfg.frequency_weight * freq_dist
        )
        scores = scores - self.cfg.emitter_neighbor_weight * grid_dist
        scores = scores + 0.05 * path_bias
        scores = scores + 0.03 * (layer_index + 1)

        top_k_emitters = max(1, min(self.cfg.top_k_emitters, scores.size(-1)))
        top_scores, top_idx = torch.topk(scores, k=top_k_emitters, dim=-1)
        top_weights = F.softmax(top_scores / max(self.cfg.router_temperature, 1e-3), dim=-1)
        selected_emitters = torch.take_along_dim(
            emitter_bank.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1),
            top_idx.unsqueeze(-1).expand(-1, -1, -1, dim),
            dim=2,
        )
        selected_hierarchy = torch.take_along_dim(
            operator_hierarchy_bank.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1),
            top_idx.unsqueeze(-1).expand(-1, -1, -1, dim),
            dim=2,
        )
        effective_emitters = selected_emitters + 0.25 * selected_hierarchy + 0.15 * hierarchy_context.unsqueeze(2)
        emitter_context = torch.sum(top_weights.unsqueeze(-1) * effective_emitters, dim=2)
        topk_entropy = -(top_weights * torch.log(top_weights + 1e-8)).sum(dim=-1)
        topk_effective_count = torch.exp(topk_entropy)
        emitter_mixture_target = float(getattr(self.cfg, "emitter_mixture_target_count", 2.0))
        emitter_mixture_loss = torch.relu(
            torch.tensor(emitter_mixture_target, device=hidden.device) - topk_effective_count.mean()
        ) / max(emitter_mixture_target, 1.0)

        slot_query = self.slot_query(hidden)
        slot_scores = torch.einsum("btd,bsd->bts", slot_query, slots) / math.sqrt(dim)
        top_k_slots = max(1, min(self.cfg.top_k_slots, slot_scores.size(-1)))
        top_slot_scores, top_slot_idx = torch.topk(slot_scores, k=top_k_slots, dim=-1)
        top_slot_weights = F.softmax(top_slot_scores / max(self.cfg.router_temperature, 1e-3), dim=-1)
        slot_weights = torch.zeros_like(slot_scores, dtype=top_slot_weights.dtype)
        slot_weights.scatter_(-1, top_slot_idx, top_slot_weights)
        selected_slots = torch.take_along_dim(
            slots.unsqueeze(1).expand(batch, seq_len, -1, -1),
            top_slot_idx.unsqueeze(-1).expand(-1, -1, -1, dim),
            dim=2,
        )
        slot_context = torch.sum(top_slot_weights.unsqueeze(-1) * selected_slots, dim=2)

        combined_context = self.value_proj(emitter_context) + self.slot_value(slot_context) + 0.15 * hierarchy_context
        gate = torch.sigmoid(self.route_gate(hidden))
        delta = gate * self.out_proj(combined_context)

        update_source = self.slot_value(value + emitter_context + slot_context + 0.25 * hierarchy_context)
        per_token_slot_update = torch.einsum("bts,btd->btsd", slot_weights, update_source)
        cumulative_slot_update = torch.cumsum(per_token_slot_update, dim=1)
        cumulative_slot_weight = torch.cumsum(slot_weights, dim=1).unsqueeze(-1).clamp_min(1e-6)
        running_slot_update = cumulative_slot_update / cumulative_slot_weight
        final_slot_update = running_slot_update[:, -1]
        updated_slots = self.cfg.memory_momentum * slots + (1.0 - self.cfg.memory_momentum) * (
            self.slot_seed.unsqueeze(0) + final_slot_update
        )

        emitter_probs = F.softmax(scores / max(self.cfg.router_temperature, 1e-3), dim=-1)
        entropy = -(emitter_probs * torch.log(emitter_probs + 1e-8)).sum(dim=-1).mean()
        active_emitters = torch.tensor(float(torch.unique(top_idx).numel()), device=hidden.device)
        usage = torch.bincount(top_idx.reshape(-1).detach(), minlength=emitter_bank.size(0)).to(hidden.device).float()
        usage = usage / usage.sum().clamp_min(1.0)
        soft_usage = emitter_probs.mean(dim=(0, 1))
        soft_usage = soft_usage / soft_usage.sum().clamp_min(1e-8)
        usage_entropy = -(usage * torch.log(usage + 1e-8)).sum() / math.log(max(emitter_bank.size(0), 2))
        usage_concentration = soft_usage.square().sum() * float(emitter_bank.size(0))
        balance_loss = (usage_concentration - 1.0).clamp_min(0.0)

        stats = {
            "emitter_top_idx": top_idx.detach(),
            "emitter_top_weights": top_weights.detach(),
            "emitter_entropy": entropy,
            "emitter_topk_entropy": topk_entropy.mean(),
            "emitter_topk_effective_count": topk_effective_count.mean(),
            "emitter_mixture_loss": emitter_mixture_loss,
            "emitter_usage_entropy": usage_entropy.detach(),
            "emitter_usage_concentration": usage_concentration.detach(),
            "emitter_balance_loss": balance_loss,
            "emitter_hierarchy_entropy": hierarchy_entropy.detach(),
            "emitter_hierarchy_context_norm": hierarchy_context_norm.detach(),
            "emitter_hierarchy_score_weight": torch.tensor(hierarchy_score_weight, device=hidden.device).detach(),
            "slot_top_idx": top_slot_idx.detach(),
            "slot_top_weights": top_slot_weights.detach(),
            "slot_entropy": (-(slot_weights * torch.log(slot_weights + 1e-8)).sum(dim=-1).mean()).detach(),
            "active_emitters": active_emitters.detach(),
        }
        return delta, updated_slots, stats


class PrismalWaveBlock(nn.Module):
    def __init__(self, cfg: PrismalWaveConfig, quantization_config: Optional[QuantizationConfig] = None):
        super().__init__()
        d = cfg.d_model
        self.norm = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg, quantization_config=quantization_config)
        self.ff = nn.Sequential(
            create_quantized_linear(d, d * cfg.ff_mult, quantization_config=quantization_config),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            create_quantized_linear(d * cfg.ff_mult, d, quantization_config=quantization_config),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        slots: torch.Tensor,
        router: PrismalEmitterRouter,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        emitter_bank_override: Optional[torch.Tensor] = None,
        operator_hierarchy_bank_override: Optional[torch.Tensor] = None,
        path_index: int,
        layer_index: int,
        torus_center: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        normed = self.norm(hidden)
        attn_out = self.attn(normed)
        delta, updated_slots, stats = router.route(
            normed,
            slots,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            emitter_bank_override=emitter_bank_override,
            operator_hierarchy_bank_override=operator_hierarchy_bank_override,
            path_index=path_index,
            layer_index=layer_index,
            torus_center=torus_center,
        )
        hidden = hidden + self.dropout(attn_out + delta)
        hidden = hidden + self.dropout(self.ff(self.norm(hidden)))
        return hidden, updated_slots, stats


class PrismalWaveModel(nn.Module):
    def __init__(self, cfg: PrismalWaveConfig):
        super().__init__()
        self.cfg = cfg
        self.signature_level_to_id = dict(SIGNATURE_LEVEL_IDS)
        self.signature_relation_to_id = dict(SIGNATURE_RELATION_IDS)
        self.use_torus_core = bool(getattr(cfg, "use_torus_core", True))
        self.use_torus_sharc_router = bool(
            getattr(cfg, "Torus_SHARC_Router", getattr(cfg, "use_torus_sharc_router", True))
        )
        self.use_hmote = bool(getattr(cfg, "use_hmote", False))
        self.use_recursive_hmoe = bool(getattr(cfg, "use_recursive_hmoe", False)) and not self.use_hmote
        self.use_race_lanes = bool(getattr(cfg, "use_torus_race_lanes", False))
        self.use_token_memory_cross_attention = bool(getattr(cfg, "use_token_memory_cross_attention", True))
        self.lane_count = max(1, int(getattr(cfg, "torus_lane_count", max(1, cfg.n_paths))))
        self.scout_density = float(getattr(cfg, "torus_scout_density", 0.5))
        self.lane_select_threshold_1 = float(getattr(cfg, "torus_lane_select_threshold_1", 0.45))
        self.lane_select_threshold_2 = float(getattr(cfg, "torus_lane_select_threshold_2", 0.70))
        self.lane_relay_hop_spacing = max(1, int(getattr(cfg, "torus_lane_relay_hop_spacing", 4)))
        self.use_turbo_quantization = bool(getattr(cfg, "use_turbo_quantization", True))
        self.use_gradient_checkpointing = bool(getattr(cfg, "use_gradient_checkpointing", True))
        qcfg = QuantizationConfig(
            enabled=self.use_turbo_quantization,
            bits=int(getattr(cfg, "turbo_quantization_bits", 3)),
            method=str(getattr(cfg, "turbo_quantization_method", "turbo")),
            use_bitsandbytes_leaf_precision=bool(getattr(cfg, "use_bitsandbytes_leaf_precision", False)),
            bitsandbytes_leaf_precision_mode=str(getattr(cfg, "bitsandbytes_leaf_precision_mode", "int4")),
            bitsandbytes_leaf_quant_type=str(getattr(cfg, "bitsandbytes_leaf_quant_type", "nf4")),
            bitsandbytes_leaf_compute_dtype=str(getattr(cfg, "bitsandbytes_leaf_compute_dtype", "bfloat16")),
        )
        self.quantization_config = qcfg
        self.registry = SignatureEmitterRegistry(cfg, quantization_config=qcfg)
        if self.use_torus_sharc_router and not self.use_torus_core:
            warnings.warn(
                "Torus_SHARC_Router requires use_torus_core=True; torus/router augmentation cannot be enabled "
                "without the torus core.",
                stacklevel=2,
            )
            raise ValueError("Torus_SHARC_Router requires use_torus_core=True")
        resolved_vocab_size = max(1, int(getattr(cfg, "vocab_size", 0)) or int(getattr(cfg, "base_vocab_size", 1)) or 1)
        self.signature_vocab_size = max(1, int(cfg.signature_vocab_size) or 8)
        self.signature_level_vocab_size = max(1, int(cfg.signature_level_vocab_size) or len(SIGNATURE_LEVEL_IDS))
        self.signature_relation_vocab_size = max(1, int(cfg.signature_relation_vocab_size) or len(SIGNATURE_RELATION_IDS))
        bucket_vocab = int(getattr(cfg, "signature_bucket_vocab_size", 0) or 0)
        if bucket_vocab <= 0:
            bucket_vocab = max(8, self.registry.family_vocab_size)
        self.signature_bucket_vocab_size = bucket_vocab
        if cfg.use_factorized_embedding:
            self.construction_embedding = FactorizedEmbedding(
                resolved_vocab_size,
                cfg.d_model,
                cfg.factorized_embedding_dim,
                quantization_config=qcfg,
            )
        else:
            self.construction_embedding = create_quantized_embedding(
                resolved_vocab_size,
                cfg.d_model,
                quantization_config=qcfg,
            )
        self.position_embedding = DynamicPositionEmbedding(
            int(getattr(cfg, "position_embedding_init_size", 256)),
            cfg.d_model,
            quantization_config=qcfg,
        )
        self.token_hierarchy = (
            HierarchicalParameterNest(cfg, shared_embedding=self.construction_embedding, quantization_config=qcfg)
            if self.use_hmote or int(getattr(cfg, "hierarchical_nest_depth", 1)) > 1
            else None
        )
        self.signature_lattice_attention = (
            SignatureLatticeAttention(cfg, quantization_config=qcfg)
            if bool(getattr(cfg, "use_signature_lattice_attention", True))
            else None
        )
        self.token_memory_attention = (
            TokenMemoryCrossAttention(cfg, quantization_config=qcfg)
            if self.use_torus_core and self.use_token_memory_cross_attention
            else None
        )
        if self.use_torus_core:
            self.torus_core = (
                self.token_hierarchy
                if self.token_hierarchy is not None
                else (
                    PrismalRecursiveTorusCore(cfg, quantization_config=qcfg)
                    if self.use_recursive_hmoe
                    else PrismalTorusCore(cfg, quantization_config=qcfg)
                )
            )
            self.router = (
                PrismalEmitterRouter(cfg, quantization_config=qcfg)
                if self.use_torus_sharc_router
                else None
            )
            self.blocks = nn.ModuleList()
        else:
            self.torus_core = None
            self.router = PrismalEmitterRouter(cfg, quantization_config=qcfg)
            self.blocks = nn.ModuleList([PrismalWaveBlock(cfg, quantization_config=qcfg) for _ in range(cfg.n_layers)])
        self.activity_threshold = float(
            getattr(
                self.torus_core,
                "activity_threshold",
                getattr(cfg, "torus_activity_threshold", 0.08),
            )
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.signature_head = create_quantized_linear(cfg.d_model, cfg.d_model, bias=False, quantization_config=qcfg)
        self.signature_token_head = create_quantized_linear(
            cfg.d_model,
            self.signature_bucket_vocab_size,
            bias=True,
            quantization_config=qcfg,
        )
        self.signature_neighborhood_embedding = create_quantized_embedding(
            self.signature_bucket_vocab_size,
            cfg.d_model,
            quantization_config=qcfg,
        )
        self.signature_neighborhood_gate = create_quantized_linear(
            cfg.d_model * 2,
            cfg.d_model,
            bias=True,
            quantization_config=qcfg,
        )
        self.signature_level_head = create_quantized_linear(
            cfg.d_model,
            self.signature_level_vocab_size,
            bias=True,
            quantization_config=qcfg,
        )
        self.signature_relation_head = create_quantized_linear(
            cfg.d_model,
            self.signature_relation_vocab_size,
            bias=True,
            quantization_config=qcfg,
        )
        self.construction_head = create_quantized_linear(
            cfg.d_model,
            resolved_vocab_size,
            bias=False,
            quantization_config=qcfg,
        )
        if not cfg.use_factorized_embedding and not self.use_turbo_quantization:
            self.construction_head.weight = self.construction_embedding.weight
        self._vocab_size = resolved_vocab_size
        self.precision_policy = HierarchicalPrecisionPolicy.from_config(cfg)
        attach_precision_policy(self, self.precision_policy)
        self._precision_tier_map: List[Dict[str, object]] = []
        self._prismal_precision_state: Dict[str, object] = {}
        self.use_learned_residency_head = bool(
            getattr(cfg, "use_learned_residency_head", False) or getattr(cfg, "use_residency_with_reinforcement", False)
        )
        self.use_residency_with_reinforcement = bool(getattr(cfg, "use_residency_with_reinforcement", False))
        self.residency_head_layers = max(1, int(getattr(cfg, "residency_head_layers", 1)))
        self.residency_head_hidden_dim = max(1, int(getattr(cfg, "residency_head_hidden_dim", 256)))
        self.learned_residency_weight = max(0.0, float(getattr(cfg, "learned_residency_weight", 0.1)))
        self.learned_residency_head = (
            LearnedResidencyHead(
                cfg.d_model,
                self._learned_residency_tile_vocab_size(),
                hidden_dim=self.residency_head_hidden_dim,
                layers=self.residency_head_layers,
                quantization_config=qcfg,
            )
            if self.use_learned_residency_head
            else None
        )
        self.use_contrastive_routing = bool(getattr(cfg, "use_contrastive_routing", False))
        self.contrastive_routing_weight = max(0.0, float(getattr(cfg, "contrastive_routing_weight", 0.1)))
        self.contrastive_routing_temperature = max(1e-3, float(getattr(cfg, "contrastive_routing_temperature", 0.1)))
        self.contrastive_routing_hard_negatives = bool(getattr(cfg, "contrastive_routing_hard_negatives", False))
        self.use_contrastive_routing_signature_neighborhood = bool(
            getattr(cfg, "use_contrastive_routing_signature_neighborhood", False)
        )
        self.use_contrastive_routing_temporal = bool(getattr(cfg, "use_contrastive_routing_temporal", False))
        self.use_contrastive_routing_residency = bool(getattr(cfg, "use_contrastive_routing_residency", False))
        self.use_contrastive_routing_cross_view = bool(getattr(cfg, "use_contrastive_routing_cross_view", False))
        self.use_contrastive_routing_self_contrast = bool(getattr(cfg, "use_contrastive_routing_self_contrast", False))
        self.use_gate = bool(getattr(cfg, "use_gate", False))
        self.use_fullgatetrain = bool(getattr(cfg, "use_fullgatetrain", False))
        self.use_gatetrain = bool(getattr(cfg, "use_gatetrain", False)) or self.use_fullgatetrain
        self.gate_controller = GateResidencyController(self) if self.use_gate else None
        self.gatetrain_controller = (
            GateResidencyController(
                self,
                enabled_attr="use_gatetrain",
                config_prefix="gatetrain",
                stat_prefix="gatetrain",
            )
            if self.use_gatetrain
            else None
        )

    @property
    def token_embedding(self) -> nn.Module:
        return self.construction_embedding

    @token_embedding.setter
    def token_embedding(self, value: nn.Module) -> None:
        self.construction_embedding = value

    @property
    def lm_head(self) -> nn.Module:
        return self.construction_head

    @lm_head.setter
    def lm_head(self, value: nn.Module) -> None:
        self.construction_head = value

    @property
    def signature_neighborhood_head(self) -> nn.Module:
        return self.signature_token_head

    @staticmethod
    def _superposition_family_priority(family_id: int) -> int:
        if family_id == 4:
            return 3
        if family_id == 2:
            return 2
        if family_id == 1:
            return 1
        return 0

    def _select_superposition_representative(
        self,
        family_row: torch.Tensor,
        valid_row: torch.Tensor,
    ) -> int:
        valid_positions = torch.nonzero(valid_row, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            return 0

        family_values = family_row[valid_positions]
        for preferred_family_id in (4, 2, 1):
            preferred_positions = valid_positions[family_values == preferred_family_id]
            if preferred_positions.numel() > 0:
                return int(preferred_positions[0].item())

        counts: Dict[int, int] = {}
        first_positions: Dict[int, int] = {}
        for position in valid_positions.tolist():
            family_id = int(family_row[position].item())
            if family_id == 0:
                continue
            counts[family_id] = counts.get(family_id, 0) + 1
            first_positions.setdefault(family_id, position)
        if counts:
            best_family_id = max(
                counts.keys(),
                key=lambda family_id: (counts[family_id], -first_positions[family_id]),
            )
            return int(first_positions[best_family_id])
        return int(valid_positions[0].item())

    def _build_superposition_batch(
        self,
        input_ids: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        bag_size: int,
    ) -> SuperpositionBatch:
        bag_size = max(1, int(bag_size))
        batch_size, seq_len = input_ids.shape
        if seq_len <= bag_size:
            raise ValueError("superposition requires at least two bags")

        bag_count = max(1, (seq_len + bag_size - 1) // bag_size)
        target_count = max(0, bag_count - 1)

        def _pad_tensor(tensor: torch.Tensor, *, pad_value: int | float) -> torch.Tensor:
            remainder = tensor.size(1) % bag_size
            if remainder == 0:
                return tensor
            pad_len = bag_size - remainder
            pad_shape = (tensor.size(0), pad_len)
            pad = tensor.new_full(pad_shape, pad_value)
            return torch.cat([tensor, pad], dim=1)

        padded_input_ids = _pad_tensor(input_ids, pad_value=self.cfg.pad_id)
        token_groups = padded_input_ids.view(batch_size, bag_count, bag_size)
        valid_groups = token_groups.ne(self.cfg.pad_id)

        padded_family_ids = _pad_tensor(
            signature_family_ids if signature_family_ids is not None else input_ids.new_zeros(input_ids.shape),
            pad_value=0,
        ).view(batch_size, bag_count, bag_size)
        reduced_input_ids = torch.full(
            (batch_size, bag_count),
            self.cfg.pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        def _reduce_track(track: Optional[torch.Tensor], *, pad_value: int) -> Optional[torch.Tensor]:
            if track is None:
                return None
            padded = _pad_tensor(track, pad_value=pad_value).view(batch_size, bag_count, bag_size)
            reduced = torch.full((batch_size, bag_count), pad_value, dtype=track.dtype, device=track.device)
            for batch_idx in range(batch_size):
                for bag_idx in range(bag_count):
                    representative_idx = self._select_superposition_representative(
                        padded_family_ids[batch_idx, bag_idx],
                        valid_groups[batch_idx, bag_idx],
                    )
                    reduced[batch_idx, bag_idx] = padded[batch_idx, bag_idx, representative_idx]
            return reduced

        reduced_signature_family_ids = _reduce_track(signature_family_ids, pad_value=0)
        reduced_signature_ids = _reduce_track(signature_ids, pad_value=0)
        reduced_signature_level_ids = _reduce_track(signature_level_ids, pad_value=SIGNATURE_LEVEL_IDS["pad"])
        reduced_signature_relation_ids = _reduce_track(signature_relation_ids, pad_value=SIGNATURE_RELATION_IDS["pad"])
        reduced_parent_signature_ids = _reduce_track(parent_signature_ids, pad_value=0)

        for batch_idx in range(batch_size):
            for bag_idx in range(bag_count):
                representative_idx = self._select_superposition_representative(
                    padded_family_ids[batch_idx, bag_idx],
                    valid_groups[batch_idx, bag_idx],
                )
                reduced_input_ids[batch_idx, bag_idx] = token_groups[batch_idx, bag_idx, representative_idx]

        target_source_ids = input_ids[:, bag_size:]
        target_source_mask = loss_mask[:, bag_size:] if loss_mask is not None else None
        if target_source_ids.numel() == 0:
            target_ids = torch.empty((batch_size, 0, bag_size), device=input_ids.device, dtype=input_ids.dtype)
            target_mask = torch.empty((batch_size, 0, bag_size), device=input_ids.device, dtype=torch.float32)
        else:
            target_pad_len = target_count * bag_size - target_source_ids.size(1)
            if target_pad_len > 0:
                target_ids_padded = torch.cat(
                    [target_source_ids, target_source_ids.new_full((batch_size, target_pad_len), self.cfg.pad_id)],
                    dim=1,
                )
                if target_source_mask is not None:
                    target_mask_padded = torch.cat(
                        [target_source_mask, target_source_mask.new_zeros((batch_size, target_pad_len))],
                        dim=1,
                    )
                else:
                    target_mask_padded = target_source_mask
            else:
                target_ids_padded = target_source_ids
                target_mask_padded = target_source_mask
            target_ids = target_ids_padded[:, : target_count * bag_size].view(batch_size, target_count, bag_size)
            if target_mask_padded is not None:
                target_mask = target_mask_padded[:, : target_count * bag_size].view(batch_size, target_count, bag_size)
            else:
                target_mask = target_ids.ne(self.cfg.pad_id).to(dtype=torch.float32)
            target_mask = target_mask * target_ids.ne(self.cfg.pad_id).to(dtype=torch.float32)

        forward_loss_mask = None
        if target_count > 0:
            forward_loss_mask = target_mask.any(dim=-1).to(dtype=torch.float32)
            forward_loss_mask = torch.cat(
                [
                    forward_loss_mask,
                    forward_loss_mask.new_zeros((batch_size, 1)),
                ],
                dim=1,
            )

        return SuperpositionBatch(
            input_ids=reduced_input_ids,
            token_groups=token_groups,
            signature_family_ids=reduced_signature_family_ids,
            signature_ids=reduced_signature_ids,
            signature_level_ids=reduced_signature_level_ids,
            signature_relation_ids=reduced_signature_relation_ids,
            parent_signature_ids=reduced_parent_signature_ids,
            forward_loss_mask=forward_loss_mask,
            target_ids=target_ids,
            target_mask=target_mask,
            bag_size=bag_size,
        )

    def _learned_residency_tile_vocab_size(self) -> int:
        emitter_count = max(1, int(getattr(self.cfg, "n_emitters", 1)))
        tile_granularity = max(
            1,
            int(
                getattr(
                    self.cfg,
                    "gate_tile_granularity",
                    getattr(self.cfg, "gatetrain_tile_granularity", 1),
                )
            ),
        )
        return max(1, math.ceil(emitter_count / tile_granularity))

    def _contrastive_routing_outputs(
        self,
        final_hidden: torch.Tensor,
        input_signature: torch.Tensor,
        final_output_signature: torch.Tensor,
        *,
        path_signatures: Optional[Sequence[torch.Tensor]] = None,
        path_emitter_top_weights: Optional[Sequence[torch.Tensor]] = None,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        gate_runtime: Optional[Dict[str, torch.Tensor | GateResidencyPlan]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        device = final_hidden.device
        zero = torch.tensor(0.0, device=device)
        stats: Dict[str, torch.Tensor] = {
            "contrastive_routing_enabled": zero,
            "contrastive_routing_signature_neighborhood_enabled": zero,
            "contrastive_routing_temporal_enabled": zero,
            "contrastive_routing_residency_enabled": zero,
            "contrastive_routing_cross_view_enabled": zero,
            "contrastive_routing_self_contrast_enabled": zero,
            "contrastive_routing_signature_neighborhood_loss": zero,
            "contrastive_routing_temporal_loss": zero,
            "contrastive_routing_residency_loss": zero,
            "contrastive_routing_cross_view_loss": zero,
            "contrastive_routing_self_contrast_loss": zero,
            "contrastive_routing_loss": zero,
            "contrastive_routing_temperature": torch.tensor(self.contrastive_routing_temperature, device=device),
            "contrastive_routing_weight": torch.tensor(self.contrastive_routing_weight, device=device),
            "contrastive_routing_hard_negatives": torch.tensor(
                1.0 if self.contrastive_routing_hard_negatives else 0.0,
                device=device,
            ),
        }
        if not self.training or not self.use_contrastive_routing:
            return stats, zero

        temperature = max(self.contrastive_routing_temperature, 1e-3)
        total_loss = zero

        def _batch_labels(source: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if source is None or not torch.is_tensor(source) or source.numel() == 0:
                return None
            values = source.detach().long()
            if values.dim() == 1:
                return values.to(device=device)
            flattened: List[int] = []
            for row in values:
                row_values = row.reshape(-1)
                row_values = row_values[row_values > 0]
                flattened.append(int(row_values[0].item()) if row_values.numel() > 0 else -1)
            return torch.tensor(flattened, device=device, dtype=torch.long)

        def _supervised_contrastive(embeddings: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
            if labels is None or embeddings.numel() == 0:
                return zero
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            labels = labels.reshape(-1).to(device=device)
            if labels.numel() != embeddings.size(0):
                return zero
            valid = labels.ge(0)
            if not bool(valid.any().item()):
                return zero
            embeddings = embeddings[valid]
            labels = labels[valid]
            if embeddings.size(0) < 2:
                return zero
            if torch.unique(labels).numel() < 1:
                return zero
            normalized = F.normalize(embeddings, dim=-1, eps=1e-6)
            logits = normalized @ normalized.transpose(0, 1)
            logits = logits / temperature
            eye = torch.eye(logits.size(0), device=device, dtype=torch.bool)
            logits = logits.masked_fill(eye, float("-inf"))
            positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~eye
            positive_rows = positive_mask.any(dim=-1)
            if not bool(positive_rows.any().item()):
                return zero
            if self.contrastive_routing_hard_negatives:
                positive_logits = logits.masked_fill(~positive_mask, float("-inf")).max(dim=-1).values
                negative_logits = logits.masked_fill(positive_mask, float("-inf")).max(dim=-1).values
                valid_rows = torch.isfinite(positive_logits) & torch.isfinite(negative_logits)
                if not bool(valid_rows.any().item()):
                    return zero
                loss = F.softplus((negative_logits - positive_logits) / temperature)
                return loss.masked_select(valid_rows).mean()
            positive_logsumexp = torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=-1)
            all_logsumexp = torch.logsumexp(logits, dim=-1)
            valid_rows = positive_rows & torch.isfinite(positive_logsumexp) & torch.isfinite(all_logsumexp)
            if not bool(valid_rows.any().item()):
                return zero
            loss = -(positive_logsumexp - all_logsumexp)
            return loss.masked_select(valid_rows).mean()

        any_active = False

        if self.use_contrastive_routing_signature_neighborhood:
            labels = _batch_labels(signature_family_ids if signature_family_ids is not None else signature_ids)
            signature_neighborhood_loss = _supervised_contrastive(final_output_signature, labels)
            if torch.is_tensor(signature_neighborhood_loss) and signature_neighborhood_loss.numel() > 0:
                stats["contrastive_routing_signature_neighborhood_enabled"] = torch.tensor(1.0, device=device)
                stats["contrastive_routing_signature_neighborhood_loss"] = signature_neighborhood_loss.detach()
                total_loss = total_loss + signature_neighborhood_loss
                any_active = True

        if self.use_contrastive_routing_temporal and final_hidden.size(1) > 1:
            source = signature_family_ids if signature_family_ids is not None else signature_ids
            if source is not None and torch.is_tensor(source) and source.size(1) > 1:
                temporal_embeddings = final_hidden[:, :-1, :].reshape(-1, final_hidden.size(-1))
                temporal_labels = source[:, 1:].reshape(-1).detach().long()
                temporal_loss = _supervised_contrastive(temporal_embeddings, temporal_labels)
                if torch.is_tensor(temporal_loss) and temporal_loss.numel() > 0:
                    stats["contrastive_routing_temporal_enabled"] = torch.tensor(1.0, device=device)
                    stats["contrastive_routing_temporal_loss"] = temporal_loss.detach()
                    total_loss = total_loss + temporal_loss
                    any_active = True

        if self.use_contrastive_routing_residency and self.learned_residency_head is not None:
            plan = None
            if gate_runtime is not None:
                plan = gate_runtime.get("plan")
            if isinstance(plan, GateResidencyPlan) and plan.emitter_tile_ids:
                tile_logits = self.learned_residency_head(final_hidden)["tile_logits"]
                tile_count = tile_logits.size(-1)
                target_mask = torch.zeros(tile_count, device=device, dtype=torch.bool)
                for tile_id in plan.emitter_tile_ids:
                    if 0 <= int(tile_id) < tile_count:
                        target_mask[int(tile_id)] = True
                if bool(target_mask.any().item()):
                    target_mask = target_mask.unsqueeze(0).expand(tile_logits.size(0), -1)
                    positive_logits = tile_logits.masked_fill(~target_mask, float("-inf")).max(dim=-1).values
                    negative_logits = tile_logits.masked_fill(target_mask, float("-inf")).max(dim=-1).values
                    valid_rows = torch.isfinite(positive_logits) & torch.isfinite(negative_logits)
                    if bool(valid_rows.any().item()):
                        residency_loss = F.softplus((negative_logits - positive_logits) / temperature)
                        residency_loss = residency_loss.masked_select(valid_rows).mean()
                        stats["contrastive_routing_residency_enabled"] = torch.tensor(1.0, device=device)
                        stats["contrastive_routing_residency_loss"] = residency_loss.detach()
                        total_loss = total_loss + residency_loss
                        any_active = True

        if self.use_contrastive_routing_cross_view and path_signatures:
            path_signature_tensor = torch.stack(list(path_signatures), dim=1)
            if path_signature_tensor.size(1) > 1:
                view_embeddings = path_signature_tensor.reshape(-1, path_signature_tensor.size(-1))
                view_labels = torch.arange(path_signature_tensor.size(0), device=device).repeat_interleave(
                    path_signature_tensor.size(1)
                )
                cross_view_loss = _supervised_contrastive(view_embeddings, view_labels)
                if torch.is_tensor(cross_view_loss) and cross_view_loss.numel() > 0:
                    stats["contrastive_routing_cross_view_enabled"] = torch.tensor(1.0, device=device)
                    stats["contrastive_routing_cross_view_loss"] = cross_view_loss.detach()
                    total_loss = total_loss + cross_view_loss
                    any_active = True

        if self.use_contrastive_routing_self_contrast and path_emitter_top_weights:
            topk_weights = torch.stack(list(path_emitter_top_weights), dim=1)
            if topk_weights.numel() > 0 and topk_weights.size(-1) > 1:
                candidate_weights = topk_weights.mean(dim=2).clamp_min(1e-8)
                candidate_logits = torch.log(candidate_weights)
                self_contrast_targets = torch.zeros(
                    candidate_logits.size(0) * candidate_logits.size(1),
                    device=device,
                    dtype=torch.long,
                )
                self_contrast_loss = F.cross_entropy(
                    candidate_logits.reshape(-1, candidate_logits.size(-1)) / temperature,
                    self_contrast_targets,
                )
                stats["contrastive_routing_self_contrast_enabled"] = torch.tensor(1.0, device=device)
                stats["contrastive_routing_self_contrast_loss"] = self_contrast_loss.detach()
                total_loss = total_loss + self_contrast_loss
                any_active = True

        stats["contrastive_routing_enabled"] = torch.tensor(1.0 if any_active else 0.0, device=device)
        stats["contrastive_routing_loss"] = total_loss.detach()
        return stats, total_loss

    def _learned_residency_target_tiles(self, route_stats: Dict[str, torch.Tensor], *, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        tile_vocab_size = self._learned_residency_tile_vocab_size()
        top_idx = route_stats.get("emitter_top_idx")
        if top_idx is None or not torch.is_tensor(top_idx) or top_idx.numel() == 0:
            return None
        active_controller = self._active_residency_controller()
        tile_granularity = max(
            1,
            int(
                getattr(
                    self.cfg,
                    "gate_tile_granularity",
                    getattr(self.cfg, "gatetrain_tile_granularity", 1),
                )
            ),
        )
        values = top_idx.detach().long()
        if values.dim() == 0:
            values = values.view(1, 1).expand(batch_size, 1)
        elif values.size(0) == batch_size:
            values = values.reshape(batch_size, -1)
        else:
            values = values.reshape(1, -1).expand(batch_size, -1)
        prefetch_horizon = max(
            1,
            int(
                getattr(
                    active_controller,
                    "prefetch_horizon",
                    getattr(self.cfg, "gate_prefetch_horizon", getattr(self.cfg, "gatetrain_prefetch_horizon", 1)),
                )
            ),
        )
        targets = torch.zeros(batch_size, tile_vocab_size, device=device)
        for batch_idx in range(batch_size):
            tile_ids: List[int] = []
            for raw_value in values[batch_idx].reshape(-1).tolist():
                base_tile = max(0, int(raw_value) // tile_granularity)
                for offset in range(prefetch_horizon):
                    tile_ids.append(base_tile + offset)
            for tile_id in self._unique_tile_ids(tile_ids, tile_vocab_size):
                targets[batch_idx, tile_id] = 1.0
        return targets

    def _unique_tile_ids(self, values: Sequence[int], limit: int) -> Tuple[int, ...]:
        seen: set[int] = set()
        ordered: List[int] = []
        for value in values:
            item = int(value)
            if item < 0 or item >= limit or item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return tuple(ordered)

    def _learned_residency_reward(self, route_stats: Dict[str, torch.Tensor], *, batch_size: int, device: torch.device) -> torch.Tensor:
        def _stat(keys: Sequence[str]) -> Optional[torch.Tensor]:
            for key in keys:
                value = route_stats.get(key)
                if value is not None and torch.is_tensor(value):
                    return value.detach().to(device=device, dtype=torch.float32)
            return None

        hit_rate = _stat(("gatetrain_hit_rate", "gate_hit_rate"))
        if hit_rate is None:
            hit_count = _stat(("gatetrain_hit_count", "gate_hit_count"))
            miss_count = _stat(("gatetrain_miss_count", "gate_miss_count"))
            if hit_count is not None or miss_count is not None:
                hit_count = hit_count if hit_count is not None else torch.zeros(1, device=device)
                miss_count = miss_count if miss_count is not None else torch.zeros(1, device=device)
                hit_rate = hit_count / torch.clamp(hit_count + miss_count, min=1e-6)
        if hit_rate is None:
            hit_rate = torch.zeros(1, device=device)
        churn = _stat(("gatetrain_tile_churn", "gate_tile_churn"))
        if churn is None:
            churn = torch.zeros(1, device=device)
        latency_saved = _stat(("gatetrain_latency_saved_ms", "gate_latency_saved_ms"))
        if latency_saved is None:
            latency_saved = torch.zeros(1, device=device)
        reward = hit_rate.float().mean()
        reward = reward - 0.05 * torch.tanh(churn.float().mean())
        reward = reward + 0.01 * torch.tanh(latency_saved.float().mean())
        reward = reward.clamp(-1.0, 1.0)
        return reward.expand(batch_size)

    def _learned_residency_outputs(
        self,
        hidden: torch.Tensor,
        route_stats: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if not self.use_learned_residency_head or self.learned_residency_head is None:
            return {}, torch.tensor(0.0, device=hidden.device)
        head_out = self.learned_residency_head(hidden)
        tile_logits = head_out["tile_logits"]
        tile_probs = head_out["tile_probs"]
        confidence = head_out["confidence"]
        active_controller = self._active_residency_controller()
        residency_budget = int(
            getattr(
                active_controller,
                "residency_budget",
                getattr(self.cfg, "gate_residency_budget", getattr(self.cfg, "gatetrain_residency_budget", 1)),
            )
        )
        top_k = min(self.learned_residency_head.tile_vocab_size, max(1, residency_budget))
        top_tiles = torch.topk(tile_logits, k=top_k, dim=-1).indices
        targets = self._learned_residency_target_tiles(route_stats, batch_size=tile_logits.size(0), device=tile_logits.device)
        reward = self._learned_residency_reward(route_stats, batch_size=tile_logits.size(0), device=tile_logits.device)
        if targets is not None:
            base_loss = F.binary_cross_entropy_with_logits(tile_logits, targets, reduction="none")
            if self.use_residency_with_reinforcement:
                reward_scale = (1.0 + reward).clamp(0.0, 2.0).unsqueeze(-1)
                loss = (base_loss * reward_scale).mean()
            else:
                loss = base_loss.mean()
        else:
            probs = tile_probs.clamp_min(1e-8)
            if self.use_residency_with_reinforcement:
                loss = -torch.log(probs.max(dim=-1).values).mean() * (1.0 - reward.mean().clamp(-1.0, 1.0) * 0.5)
            else:
                loss = -torch.log(probs.max(dim=-1).values).mean()
        predicted = top_tiles.detach()
        outputs: Dict[str, torch.Tensor] = {
            "learned_residency_tile_logits": tile_logits.detach(),
            "learned_residency_tile_probs": tile_probs.detach(),
            "learned_residency_top_tiles": predicted,
            "learned_residency_confidence": confidence.detach(),
            "learned_residency_reward": reward.detach(),
            "learned_residency_loss": loss.detach(),
            "learned_residency_enabled": torch.tensor(1.0, device=tile_logits.device),
            "learned_residency_mode": torch.tensor(1.0 if self.use_residency_with_reinforcement else 0.0, device=tile_logits.device),
        }
        return outputs, loss

    def _finite_guard_enabled(self) -> bool:
        return bool(
            getattr(self.cfg, "training_finite_guard_enabled", True)
            if self.training
            else getattr(self.cfg, "inference_finite_guard_enabled", True)
        )

    def _sanitize_tensor(
        self,
        tensor: torch.Tensor,
        *,
        fallback: Optional[torch.Tensor] = None,
        allow_negative_inf: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        if not self._finite_guard_enabled():
            return tensor, 0
        return _repair_finite_tensor(tensor, fallback=fallback, allow_negative_inf=allow_negative_inf)

    def _sanitize_route_stats(
        self,
        route_stats: Dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        if not self._finite_guard_enabled():
            return route_stats, 0
        sanitized: Dict[str, torch.Tensor] = {}
        repairs = 0
        for key, value in route_stats.items():
            if torch.is_tensor(value):
                clean_value, clean_repairs = self._sanitize_tensor(value)
                sanitized[key] = clean_value
                repairs += clean_repairs
            else:
                sanitized[key] = value
        sanitized["stability_nonfinite_repair_count"] = torch.tensor(float(repairs), device=device)
        sanitized["stability_finite_guard_enabled"] = torch.tensor(1.0, device=device)
        return sanitized, repairs

    def _sanitize_generation_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return self._sanitize_tensor(logits, allow_negative_inf=True)

    def _sanitize_sampling_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, int]:
        clean_logits, repairs = self._sanitize_generation_logits(logits)
        if not self._finite_guard_enabled() or clean_logits.numel() == 0:
            return clean_logits, repairs
        row_has_finite = torch.isfinite(clean_logits).any(dim=-1, keepdim=True)
        if bool(row_has_finite.all().item()):
            return clean_logits, repairs
        cleaned = torch.where(row_has_finite, clean_logits, torch.zeros_like(clean_logits))
        repairs += int((~row_has_finite).sum().item())
        return cleaned, repairs

    def configure_precision(
        self,
        device: torch.device,
        *,
        enabled: Optional[bool] = None,
        checkpoint_precision_state: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        policy = getattr(self, "precision_policy", None)
        if isinstance(checkpoint_precision_state, dict):
            checkpoint_policy_state = checkpoint_precision_state.get("precision_policy_state")
            if isinstance(checkpoint_policy_state, dict):
                policy = HierarchicalPrecisionPolicy.from_state_dict(checkpoint_policy_state)
                self.precision_policy = policy
                attach_precision_policy(self, policy)
        if not isinstance(policy, HierarchicalPrecisionPolicy):
            policy = HierarchicalPrecisionPolicy.from_config(self.cfg)
            self.precision_policy = policy
            attach_precision_policy(self, policy)
        runtime_enabled = bool(policy.enabled if enabled is None else enabled)
        resolved_state_dtype = policy.state_dtype_for_device(device)
        tier_map: List[Dict[str, object]] = []

        for module_name, module in self.named_modules():
            setattr(module, "_hierarchical_precision_runtime_enabled", runtime_enabled)
            setattr(module, "precision_policy", policy)
            if isinstance(module, HierarchicalParameterNest):
                module_path = module_name or f"hierarchy.level{module.level}"
                module._precision_module_path = module_path
                spec = policy.resolve_for_level(
                    module.level,
                    module.hierarchical_depth,
                    device,
                    is_leaf=module.is_leaf_nest,
                    module_path=module_path,
                    module_kind="nest",
                )
                module.precision_spec = spec
                module.precision_state_dtype = resolved_state_dtype
                tier_map.append(spec.to_state_dict())
            elif isinstance(module, (PrismalTorusCore, PrismalRecursiveTorusCore, PrismalEmitterRouter, SignatureLatticeAttention)):
                module.precision_state_dtype = resolved_state_dtype

        root_spec = policy.resolve_for_level(
            0,
            max(1, int(getattr(self.cfg, "hierarchical_nest_depth", 1))),
            device,
            is_leaf=False,
            module_path="root",
            module_kind="root",
        )
        self.precision_spec = root_spec
        if checkpoint_precision_state is not None:
            previous_map = checkpoint_precision_state.get("precision_tier_map") if isinstance(checkpoint_precision_state, dict) else None
            if previous_map is not None and previous_map != tier_map:
                print("[Prismal] precision policy re-resolved for the current device", flush=True)

        self._precision_tier_map = tier_map
        self._prismal_precision_state = {
            "precision_policy_state": policy.to_state_dict(),
            "precision_tier_map": tier_map,
            "state_dtype": dtype_name(resolved_state_dtype),
            "runtime_enabled": runtime_enabled,
        }
        return self._prismal_precision_state

    def _prepare_gate_runtime(
        self,
        *,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor],
        path_index: Optional[int],
        position_index: int,
    ) -> Dict[str, torch.Tensor | GateResidencyPlan]:
        controller = self.gatetrain_controller if self.training else self.gate_controller
        if controller is None:
            return {}
        prior_route_stats = getattr(controller, "last_route_stats", None)
        start = time.perf_counter()
        plan = controller.plan(
            input_ids=input_ids,
            signature_family_ids=signature_family_ids,
            route_stats=prior_route_stats,
            path_index=path_index,
            position_index=position_index,
        )
        gate_stats = controller.apply(plan, device=input_ids.device)
        controller.last_plan_time_ms = (time.perf_counter() - start) * 1000.0
        gate_stats["plan"] = plan
        return gate_stats

    def _active_residency_controller(self) -> Optional[GateResidencyController]:
        if self.training:
            return self.gatetrain_controller if self.use_gatetrain else None
        return self.gate_controller if self.use_gate else None

    def _ensure_position_embedding_capacity(self, required_size: int) -> None:
        if self.cfg.max_seq_len <= 0:
            self.position_embedding.ensure_capacity(required_size)

    def _path_vectors(self, device: torch.device) -> torch.Tensor:
        if self.use_torus_core and self.torus_core is not None:
            basis = self.torus_core.path_basis
        else:
            basis = self.router.path_basis
        return torch.tanh(basis).to(device)

    def _registry_channels(
        self,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        *,
        reference: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.registry.compose_channels(
            reference=reference,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )

    def _registry_context(
        self,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        *,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.registry.compose(
            reference=reference,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
        )

    def set_capacity_growth_locked(self, locked: bool = True) -> None:
        if hasattr(self, "registry") and hasattr(self.registry, "set_capacity_growth_locked"):
            self.registry.set_capacity_growth_locked(locked)
        if hasattr(self, "router") and hasattr(self.router, "set_capacity_growth_locked"):
            self.router.set_capacity_growth_locked(locked)

    def prepare_capacity_for_tokenizer(self, tokenizer: object) -> None:
        router = getattr(self, "router", None)
        registry = getattr(self, "registry", None)
        signature_vocab_size = max(1, int(getattr(tokenizer, "signature_vocab_size", self.signature_vocab_size) or self.signature_vocab_size))
        signature_level_vocab_size = max(
            1,
            int(getattr(tokenizer, "signature_level_vocab_size", self.signature_level_vocab_size) or self.signature_level_vocab_size),
        )
        signature_relation_vocab_size = max(
            1,
            int(getattr(tokenizer, "signature_relation_vocab_size", self.signature_relation_vocab_size) or self.signature_relation_vocab_size),
        )
        signature_bucket_vocab_size = max(
            8,
            int(getattr(tokenizer, "signature_family_vocab_size", self.signature_bucket_vocab_size) or self.signature_bucket_vocab_size),
        )
        registry_old = {
            "signature_vocab_size": int(getattr(registry.family_embedding, "num_embeddings", 0)) if registry is not None else 0,
            "signature_level_vocab_size": int(getattr(registry.level_embedding, "num_embeddings", 0)) if registry is not None else 0,
            "signature_relation_vocab_size": int(getattr(registry.relation_embedding, "num_embeddings", 0)) if registry is not None else 0,
        }
        router_old = {
            "signature_vocab_size": int(getattr(getattr(router, "signature_embedding", None), "num_embeddings", 0)) if router is not None else 0,
            "signature_bucket_vocab_size": int(getattr(getattr(router, "family_embedding", None), "num_embeddings", 0)) if router is not None else 0,
            "signature_level_vocab_size": int(getattr(getattr(router, "level_embedding", None), "num_embeddings", 0)) if router is not None else 0,
            "signature_relation_vocab_size": int(getattr(getattr(router, "relation_embedding", None), "num_embeddings", 0)) if router is not None else 0,
        }
        if hasattr(self, "registry") and hasattr(self.registry, "ensure_capacity_for_sizes"):
            self.registry.ensure_capacity_for_sizes(
                signature_vocab_size,
                signature_level_vocab_size,
                signature_relation_vocab_size,
            )
        if hasattr(self, "router") and hasattr(self.router, "ensure_hierarchy_capacity_for_sizes"):
            self.router.ensure_hierarchy_capacity_for_sizes(
                signature_vocab_size,
                signature_bucket_vocab_size,
                signature_level_vocab_size,
                signature_relation_vocab_size,
            )
        if signature_bucket_vocab_size > self.signature_bucket_vocab_size:
            self.resize_signature_bucket_vocab(signature_bucket_vocab_size)
        old_bucket_vocab_size = int(self.signature_bucket_vocab_size)
        self.signature_vocab_size = max(self.signature_vocab_size, signature_vocab_size)
        self.signature_level_vocab_size = max(self.signature_level_vocab_size, signature_level_vocab_size)
        self.signature_relation_vocab_size = max(self.signature_relation_vocab_size, signature_relation_vocab_size)
        self.signature_bucket_vocab_size = max(self.signature_bucket_vocab_size, signature_bucket_vocab_size)
        self.cfg.signature_vocab_size = self.signature_vocab_size
        self.cfg.signature_level_vocab_size = self.signature_level_vocab_size
        self.cfg.signature_relation_vocab_size = self.signature_relation_vocab_size
        self.cfg.signature_bucket_vocab_size = self.signature_bucket_vocab_size
        expansions: List[str] = []
        if registry is not None:
            registry_new = {
                "signature_vocab_size": int(getattr(registry.family_embedding, "num_embeddings", 0)),
                "signature_level_vocab_size": int(getattr(registry.level_embedding, "num_embeddings", 0)),
                "signature_relation_vocab_size": int(getattr(registry.relation_embedding, "num_embeddings", 0)),
            }
            for key, old_value in registry_old.items():
                new_value = registry_new[key]
                if new_value > old_value:
                    expansions.append(f"registry.{key} {old_value}->{new_value}")
        if router is not None:
            router_new = {
                "signature_vocab_size": int(getattr(getattr(router, "signature_embedding", None), "num_embeddings", 0)),
                "signature_bucket_vocab_size": int(getattr(getattr(router, "family_embedding", None), "num_embeddings", 0)),
                "signature_level_vocab_size": int(getattr(getattr(router, "level_embedding", None), "num_embeddings", 0)),
                "signature_relation_vocab_size": int(getattr(getattr(router, "relation_embedding", None), "num_embeddings", 0)),
            }
            for key, old_value in router_old.items():
                new_value = router_new[key]
                if new_value > old_value:
                    expansions.append(f"router.{key} {old_value}->{new_value}")
        if signature_bucket_vocab_size > old_bucket_vocab_size:
            expansions.append(f"signature_bucket_vocab_size {old_bucket_vocab_size}->{signature_bucket_vocab_size}")
        if expansions:
            print("[Prismal] pre-grew transfer capacities: " + ", ".join(expansions), flush=True)
        else:
            print("[Prismal] pre-grew transfer capacities: no expansion required", flush=True)

    def _signature_family_targets(self, family_ids: torch.Tensor) -> torch.Tensor:
        bucket_vocab = max(1, int(self.signature_bucket_vocab_size))
        return family_ids.clamp(min=0).remainder(bucket_vocab)

    def _construction_logits_from_hidden(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict signature neighborhood first, then construct units from that context."""

        neighborhood_logits = self.signature_neighborhood_head(hidden)
        neighborhood_probs = F.softmax(neighborhood_logits / max(self.cfg.signature_temperature, 1e-3), dim=-1)
        neighborhood_weight = getattr(
            self.signature_neighborhood_embedding,
            "weight",
            getattr(self.signature_neighborhood_embedding, "embedding_matrix", None),
        )
        if neighborhood_weight is None:
            neighborhood_ids = torch.arange(self.signature_bucket_vocab_size, device=hidden.device)
            neighborhood_weight = self.signature_neighborhood_embedding(neighborhood_ids)
        neighborhood_context = torch.matmul(neighborhood_probs, neighborhood_weight)
        gate = torch.sigmoid(self.signature_neighborhood_gate(torch.cat([hidden, neighborhood_context], dim=-1)))
        guided_hidden = hidden + gate * neighborhood_context
        construction_logits = self.construction_head(guided_hidden)
        return construction_logits, guided_hidden, neighborhood_logits

    def _apply_token_memory_copy_bias(self, logits: torch.Tensor, output: "PrismalWaveOutput") -> torch.Tensor:
        copy_logits = output.route_stats.get("token_memory_copy_logits")
        if copy_logits is None:
            return logits
        confidence = output.route_stats.get("token_memory_copy_confidence")
        min_confidence = float(getattr(self.cfg, "token_memory_copy_min_confidence", 0.0))
        if confidence is not None and float(confidence.float().mean().item()) < min_confidence:
            return logits
        if copy_logits.shape != logits.shape:
            return logits
        blocked = copy_logits.to(device=logits.device, dtype=logits.dtype).clone()
        for token_id in {int(self.cfg.pad_id), int(self.cfg.bos_id), int(self.cfg.eos_id)}:
            if 0 <= token_id < blocked.size(-1):
                blocked[:, token_id] = 0.0
        return logits + blocked

    def _token_memory_anchor_next_ids(
        self,
        token_memory_state: Optional[TokenMemoryState],
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if token_memory_state is None:
            return None
        active = token_memory_state.anchor_cursor_active
        if active is None or not bool(active.any().item() if active.numel() > 0 else False):
            return None
        batch_size = token_memory_state.token_ids.size(0)
        next_ids = torch.full((batch_size, 1), -1, device=device, dtype=torch.long)
        cursor_pos = token_memory_state.anchor_cursor_pos.to(device=device, dtype=torch.long)
        token_ids = token_memory_state.anchor_token_ids.to(device=device, dtype=torch.long)
        for batch_idx in range(batch_size):
            if not bool(active[batch_idx].item()):
                continue
            pos = int(cursor_pos[batch_idx].item())
            if pos < 0 or pos >= token_ids.size(1):
                continue
            next_ids[batch_idx, 0] = token_ids[batch_idx, pos]
        if bool(next_ids.ge(0).any().item()):
            return next_ids
        return None

    def _advance_token_memory_anchor_state(
        self,
        token_memory_state: Optional[TokenMemoryState],
        emitted_ids: torch.Tensor,
    ) -> Optional[TokenMemoryState]:
        if token_memory_state is None:
            return None
        if emitted_ids.dim() == 1:
            emitted_ids = emitted_ids.unsqueeze(-1)
        active = token_memory_state.anchor_cursor_active.to(device=emitted_ids.device, dtype=torch.bool)
        if not bool(active.any().item() if active.numel() > 0 else False):
            return token_memory_state
        token_ids = token_memory_state.anchor_token_ids.to(device=emitted_ids.device, dtype=torch.long)
        cursor_pos = token_memory_state.anchor_cursor_pos.to(device=emitted_ids.device, dtype=torch.long).clone()
        cursor_span_id = token_memory_state.anchor_cursor_span_id.to(device=emitted_ids.device, dtype=torch.long).clone()
        cursor_offset = token_memory_state.anchor_cursor_offset.to(device=emitted_ids.device, dtype=torch.long).clone()
        cursor_length = token_memory_state.anchor_cursor_length.to(device=emitted_ids.device, dtype=torch.long).clone()
        cursor_tag = token_memory_state.anchor_cursor_tag.to(device=emitted_ids.device, dtype=torch.long).clone()
        cursor_active = active.clone()
        batch_size = emitted_ids.size(0)
        for batch_idx in range(batch_size):
            if not bool(cursor_active[batch_idx].item()):
                continue
            pos = int(cursor_pos[batch_idx].item())
            if pos < 0 or pos >= token_ids.size(1):
                cursor_active[batch_idx] = False
                cursor_pos[batch_idx] = -1
                cursor_span_id[batch_idx] = 0
                cursor_offset[batch_idx] = 0
                cursor_length[batch_idx] = 0
                cursor_tag[batch_idx] = 0
                continue
            expected = int(token_ids[batch_idx, pos].item())
            emitted = int(emitted_ids[batch_idx, 0].item())
            if emitted != expected:
                cursor_active[batch_idx] = False
                cursor_pos[batch_idx] = -1
                cursor_span_id[batch_idx] = 0
                cursor_offset[batch_idx] = 0
                cursor_length[batch_idx] = 0
                cursor_tag[batch_idx] = 0
                continue
            next_offset = int(cursor_offset[batch_idx].item()) + 1
            next_pos = pos + 1
            span_len = int(cursor_length[batch_idx].item())
            if next_offset >= span_len or next_pos >= token_ids.size(1):
                cursor_active[batch_idx] = False
                cursor_pos[batch_idx] = -1
                cursor_span_id[batch_idx] = 0
                cursor_offset[batch_idx] = 0
                cursor_length[batch_idx] = 0
                cursor_tag[batch_idx] = 0
            else:
                cursor_pos[batch_idx] = next_pos
                cursor_offset[batch_idx] = next_offset
        return TokenMemoryState(
            token_ids=token_memory_state.token_ids.to(device=emitted_ids.device, dtype=torch.long),
            memory_keys=token_memory_state.memory_keys.to(device=emitted_ids.device, dtype=token_memory_state.memory_keys.dtype),
            memory_values=token_memory_state.memory_values.to(device=emitted_ids.device, dtype=token_memory_state.memory_values.dtype),
            family_ids=token_memory_state.family_ids.to(device=emitted_ids.device, dtype=torch.long),
            signature_ids=token_memory_state.signature_ids.to(device=emitted_ids.device, dtype=torch.long),
            level_ids=token_memory_state.level_ids.to(device=emitted_ids.device, dtype=torch.long),
            relation_ids=token_memory_state.relation_ids.to(device=emitted_ids.device, dtype=torch.long),
            parent_ids=token_memory_state.parent_ids.to(device=emitted_ids.device, dtype=torch.long),
            lengths=token_memory_state.lengths.to(device=emitted_ids.device, dtype=torch.long),
            anchor_token_ids=token_memory_state.anchor_token_ids.to(device=emitted_ids.device, dtype=torch.long),
            anchor_span_ids=token_memory_state.anchor_span_ids.to(device=emitted_ids.device, dtype=torch.long),
            anchor_offsets=token_memory_state.anchor_offsets.to(device=emitted_ids.device, dtype=torch.long),
            anchor_lengths=token_memory_state.anchor_lengths.to(device=emitted_ids.device, dtype=torch.long),
            anchor_tags=token_memory_state.anchor_tags.to(device=emitted_ids.device, dtype=torch.long),
            anchor_flags=token_memory_state.anchor_flags.to(device=emitted_ids.device, dtype=torch.long),
            anchor_span_starts=token_memory_state.anchor_span_starts.to(device=emitted_ids.device, dtype=torch.long),
            anchor_cursor_pos=cursor_pos,
            anchor_cursor_span_id=cursor_span_id,
            anchor_cursor_offset=cursor_offset,
            anchor_cursor_length=cursor_length,
            anchor_cursor_tag=cursor_tag,
            anchor_cursor_active=cursor_active,
        )

    def _relay_temperature(self, step_index: int, base_temperature: float) -> float:
        base_temperature = max(1e-3, float(base_temperature))
        if step_index <= 0:
            configured = max(1e-3, float(getattr(self.cfg, "torus_primary_temperature", base_temperature)))
            return configured
        relay_temps = [
            float(getattr(self.cfg, "torus_relay_temperature_1", 0.22)),
            float(getattr(self.cfg, "torus_relay_temperature_2", 0.44)),
            float(getattr(self.cfg, "torus_relay_temperature_3", 0.88)),
        ]
        relay_temps = [max(1e-3, temp) for temp in relay_temps if temp > 0.0]
        if not relay_temps:
            return base_temperature
        return relay_temps[(step_index - 1) % len(relay_temps)]

    def _race_lane_limit(self, step_index: int) -> int:
        if not self.use_race_lanes:
            return 1
        relay_stage = max(0, int(step_index) // self.lane_relay_hop_spacing)
        return max(1, min(self.lane_count, relay_stage + 1))

    def _race_lane_temperature(self, lane_index: int, step_index: int, base_temperature: float) -> float:
        base_temperature = max(1e-3, float(base_temperature))
        if not self.use_race_lanes:
            return self._relay_temperature(step_index, base_temperature)
        if lane_index <= 0:
            configured = max(1e-3, float(getattr(self.cfg, "torus_primary_temperature", base_temperature)))
            return configured
        relay_temps = [
            float(getattr(self.cfg, "torus_relay_temperature_1", 0.22)),
            float(getattr(self.cfg, "torus_relay_temperature_2", 0.44)),
            float(getattr(self.cfg, "torus_relay_temperature_3", 0.88)),
        ]
        relay_temps = [max(1e-3, temp) for temp in relay_temps if temp > 0.0]
        if not relay_temps:
            return base_temperature
        return relay_temps[min(len(relay_temps) - 1, max(0, lane_index - 1))]

    def _race_lane_band_from_score(self, score: float) -> int:
        if score >= self.lane_select_threshold_2:
            return 0
        if score >= self.lane_select_threshold_1:
            return 1
        return 2

    def _race_scout_count(self) -> int:
        if not self.use_race_lanes:
            return 1
        return max(1, min(self.lane_count, int(round(self.lane_count * max(self.scout_density, 0.0)))))

    def _scout_race_lane(self, frame: "PrismalTorusFrame", path_count: int) -> int:
        path_count = max(1, min(int(path_count), self.lane_count))
        path_vectors = self._path_vectors(frame.hidden.device)[:path_count]
        hidden_summary = frame.hidden.mean(dim=(0, 1), keepdim=True)
        signature_summary = frame.input_signature.mean(dim=0, keepdim=True)
        scout_context = F.normalize(0.5 * hidden_summary + 0.5 * signature_summary, dim=-1)
        lane_scores = torch.matmul(scout_context, path_vectors.transpose(0, 1)).squeeze(0)
        return int(lane_scores.argmax().item())

    def _scout_race_scores(self, frame: "PrismalTorusFrame", path_count: int) -> torch.Tensor:
        path_count = max(1, min(int(path_count), self.lane_count))
        path_vectors = self._path_vectors(frame.hidden.device)[:path_count]
        hidden_summary = frame.hidden.mean(dim=(0, 1), keepdim=True)
        signature_summary = frame.input_signature.mean(dim=0, keepdim=True)
        scout_context = F.normalize(0.5 * hidden_summary + 0.5 * signature_summary, dim=-1)
        return torch.matmul(scout_context, path_vectors.transpose(0, 1)).squeeze(0)

    def _select_race_lane(self, output: "PrismalWaveOutput", step_index: int, fallback_lane: int = 0) -> Tuple[int, int, float]:
        if not self.use_race_lanes:
            lane = int(fallback_lane)
            return lane, self._race_lane_band_from_score(1.0), self._race_lane_temperature(lane, step_index, 1.0)
        path_scores = output.route_stats.get("race_scout_path_scores")
        if path_scores is None:
            path_scores = output.route_stats.get("path_scores")
        if path_scores is None:
            lane = int(fallback_lane)
            return lane, self._race_lane_band_from_score(1.0), self._race_lane_temperature(lane, step_index, 1.0)
        scores = path_scores.float().view(-1)
        if scores.numel() == 0:
            lane = int(fallback_lane)
            return lane, self._race_lane_band_from_score(1.0), self._race_lane_temperature(lane, step_index, 1.0)
        lane_limit = self._race_lane_limit(step_index)
        eligible = scores[:lane_limit]
        scout_count = min(max(1, self._race_scout_count()), eligible.numel())
        active_emitters = output.route_stats.get("path_active_emitters")
        if active_emitters is not None and active_emitters.numel() >= eligible.numel():
            active = active_emitters.float().view(-1)[:eligible.numel()]
            eligible = eligible + 0.01 * (active - active.mean())
        top_values, top_indices = torch.topk(eligible, k=scout_count)
        selected = int(top_indices[0].item())
        scout_score = float(top_values[0].item())
        band = self._race_lane_band_from_score(scout_score)
        temperature = self._race_lane_temperature(selected, step_index, 1.0)
        return selected, band, temperature

    def _encode(
        self,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        timings: Optional[Dict[str, float]] = None,
        superposition_token_groups: Optional[torch.Tensor] = None,
        superposition_bag_size: int = 1,
    ) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        position_offset = max(0, int(position_offset))
        superposition_active = superposition_token_groups is not None and int(superposition_bag_size) > 1
        if superposition_active:
            bag_size = max(1, int(superposition_bag_size))
            token_groups = superposition_token_groups
            if token_groups.dim() != 3:
                raise ValueError(f"superposition_token_groups must be 3D, got shape {tuple(token_groups.shape)}")
            if token_groups.size(0) != batch:
                raise ValueError(
                    "superposition_token_groups batch dimension must match input_ids; "
                    f"got {token_groups.size(0)} and {batch}"
                )
            seq_len = min(token_groups.size(1), self.cfg.max_seq_len) if self.cfg.max_seq_len > 0 else token_groups.size(1)
            token_groups = token_groups[:, :seq_len, :bag_size]
            position_offset = position_offset // bag_size
            if self.cfg.max_seq_len <= 0:
                self._ensure_position_embedding_capacity(position_offset + seq_len)

            def _superposed_embed() -> torch.Tensor:
                flat_ids = token_groups.contiguous().view(batch, seq_len * bag_size)
                token_embeddings = self.construction_embedding(flat_ids).view(batch, seq_len, bag_size, -1)
                valid_mask = token_groups.ne(self.cfg.pad_id).unsqueeze(-1)
                valid_mask_f = valid_mask.to(dtype=token_embeddings.dtype)
                denom = valid_mask_f.sum(dim=2).clamp_min(1.0)
                pooled = (token_embeddings * valid_mask_f).sum(dim=2) / denom
                pos = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device).unsqueeze(0)
                if self.cfg.max_seq_len > 0:
                    pos = pos.clamp(max=self.cfg.max_seq_len - 1)
                return pooled + self.position_embedding(pos)

            if timings is not None:
                hidden = _profile_stage(
                    bool(getattr(self.cfg, "profile_runtime", False)),
                    input_ids.device,
                    timings,
                    "timing_encode_embed_ms",
                    _superposed_embed,
                )
            else:
                hidden = _superposed_embed()
        else:
            if self.cfg.max_seq_len > 0:
                seq_len = min(seq_len, self.cfg.max_seq_len)
            else:
                self._ensure_position_embedding_capacity(position_offset + seq_len)
            input_ids = input_ids[:, :seq_len]
            pos = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device).unsqueeze(0)
            if self.cfg.max_seq_len > 0:
                pos = pos.clamp(max=self.cfg.max_seq_len - 1)
            if timings is not None:
                hidden = _profile_stage(
                    bool(getattr(self.cfg, "profile_runtime", False)),
                    input_ids.device,
                    timings,
                    "timing_encode_embed_ms",
                    lambda: self.construction_embedding(input_ids) + self.position_embedding(pos),
                )
            else:
                hidden = self.construction_embedding(input_ids) + self.position_embedding(pos)
        if hasattr(self.torus_core, "condition_hidden"):
            if timings is not None:
                condition_hidden = _profile_stage(
                    bool(getattr(self.cfg, "profile_runtime", False)),
                    input_ids.device,
                    timings,
                    "timing_encode_condition_ms",
                    lambda: self.torus_core.condition_hidden(
                        hidden,
                        registry_context=None,
                        family_context=signature_family_ids[:, :seq_len] if signature_family_ids is not None else None,
                        level_context=signature_level_ids[:, :seq_len] if signature_level_ids is not None else None,
                        relation_context=signature_relation_ids[:, :seq_len] if signature_relation_ids is not None else None,
                        parent_context=parent_signature_ids[:, :seq_len] if parent_signature_ids is not None else None,
                    ),
                )
            else:
                condition_hidden = self.torus_core.condition_hidden(
                    hidden,
                    registry_context=None,
                    family_context=signature_family_ids[:, :seq_len] if signature_family_ids is not None else None,
                    level_context=signature_level_ids[:, :seq_len] if signature_level_ids is not None else None,
                    relation_context=signature_relation_ids[:, :seq_len] if signature_relation_ids is not None else None,
                    parent_context=parent_signature_ids[:, :seq_len] if parent_signature_ids is not None else None,
                )
            hidden = hidden + 0.25 * condition_hidden
        if (
            signature_family_ids is not None
            or signature_ids is not None
            or signature_level_ids is not None
            or signature_relation_ids is not None
            or parent_signature_ids is not None
        ):
            if timings is not None:
                registry_context = _profile_stage(
                    bool(getattr(self.cfg, "profile_runtime", False)),
                    input_ids.device,
                    timings,
                    "timing_encode_registry_ms",
                    lambda: self._registry_context(
                        signature_family_ids=signature_family_ids[:, :seq_len] if signature_family_ids is not None else None,
                        signature_ids=signature_ids[:, :seq_len] if signature_ids is not None else None,
                        signature_level_ids=signature_level_ids[:, :seq_len] if signature_level_ids is not None else None,
                        signature_relation_ids=signature_relation_ids[:, :seq_len] if signature_relation_ids is not None else None,
                        parent_signature_ids=parent_signature_ids[:, :seq_len] if parent_signature_ids is not None else None,
                        reference=hidden,
                    ),
                )
            else:
                registry_context = self._registry_context(
                    signature_family_ids=signature_family_ids[:, :seq_len] if signature_family_ids is not None else None,
                    signature_ids=signature_ids[:, :seq_len] if signature_ids is not None else None,
                    signature_level_ids=signature_level_ids[:, :seq_len] if signature_level_ids is not None else None,
                    signature_relation_ids=signature_relation_ids[:, :seq_len] if signature_relation_ids is not None else None,
                    parent_signature_ids=parent_signature_ids[:, :seq_len] if parent_signature_ids is not None else None,
                    reference=hidden,
                )
            hidden = hidden + registry_context
        return hidden

    def _encode_step(
        self,
        input_ids: torch.Tensor,
        position_index: int,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = input_ids.size(0)
        if self.cfg.max_seq_len > 0:
            position_index = max(0, min(int(position_index), self.cfg.max_seq_len - 1))
        else:
            position_index = max(0, int(position_index))
            self._ensure_position_embedding_capacity(position_index + 1)
        pos = torch.full((batch, 1), position_index, device=input_ids.device, dtype=torch.long)
        hidden = self.construction_embedding(input_ids[:, :1]) + self.position_embedding(pos)
        if hasattr(self.torus_core, "condition_hidden"):
            hidden = hidden + 0.25 * self.torus_core.condition_hidden(
                hidden,
                registry_context=None,
                family_context=signature_family_ids[:, :1] if signature_family_ids is not None else None,
                level_context=signature_level_ids[:, :1] if signature_level_ids is not None else None,
                relation_context=signature_relation_ids[:, :1] if signature_relation_ids is not None else None,
                parent_context=parent_signature_ids[:, :1] if parent_signature_ids is not None else None,
            )
        if (
            signature_family_ids is not None
            or signature_ids is not None
            or signature_level_ids is not None
            or signature_relation_ids is not None
            or parent_signature_ids is not None
        ):
            registry_context = self._registry_context(
                signature_family_ids=signature_family_ids[:, :1] if signature_family_ids is not None else None,
                signature_ids=signature_ids[:, :1] if signature_ids is not None else None,
                signature_level_ids=signature_level_ids[:, :1] if signature_level_ids is not None else None,
                signature_relation_ids=signature_relation_ids[:, :1] if signature_relation_ids is not None else None,
                parent_signature_ids=parent_signature_ids[:, :1] if parent_signature_ids is not None else None,
                reference=hidden,
            )
            hidden = hidden + registry_context
        return hidden

    def _prepare_torus_frame(
        self,
        input_ids: torch.Tensor,
        *,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        signature_lattice_state: Optional[SignatureLatticeState] = None,
        token_memory_state: Optional[TokenMemoryState] = None,
        position_offset: int = 0,
    ) -> PrismalTorusFrame:
        profile_enabled = bool(getattr(self.cfg, "profile_runtime", False))
        prep_timings: Dict[str, float] = {}
        hidden = _profile_stage(
            profile_enabled,
            input_ids.device,
            prep_timings,
            "timing_encode_ms",
            lambda: self._encode(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                position_offset=position_offset,
                timings=prep_timings if profile_enabled else None,
            ),
        )
        batch_size, seq_len = hidden.shape[:2]
        input_ids = input_ids[:, :seq_len]
        if signature_family_ids is not None:
            signature_family_ids = signature_family_ids[:, :seq_len]
        if signature_ids is not None:
            signature_ids = signature_ids[:, :seq_len]
        if signature_level_ids is not None:
            signature_level_ids = signature_level_ids[:, :seq_len]
        if signature_relation_ids is not None:
            signature_relation_ids = signature_relation_ids[:, :seq_len]
        if parent_signature_ids is not None:
            parent_signature_ids = parent_signature_ids[:, :seq_len]

        next_signature_lattice_state: Optional[SignatureLatticeState] = signature_lattice_state
        lattice_stats: Dict[str, torch.Tensor] = {
            "signature_lattice_cache_norm": torch.tensor(0.0, device=input_ids.device),
            "signature_lattice_gate_mean": torch.tensor(0.0, device=input_ids.device),
            "signature_lattice_candidate_count": torch.tensor(0.0, device=input_ids.device),
            "signature_lattice_enabled": torch.tensor(0.0, device=input_ids.device),
        }
        if self.signature_lattice_attention is not None:
            hidden, next_signature_lattice_state, lattice_stats = _profile_stage(
                profile_enabled,
                input_ids.device,
                prep_timings,
                "timing_signature_lattice_ms",
                lambda: self.signature_lattice_attention(
                    hidden,
                    signature_family_ids=signature_family_ids,
                    signature_ids=signature_ids,
                    signature_level_ids=signature_level_ids,
                    signature_relation_ids=signature_relation_ids,
                    parent_signature_ids=parent_signature_ids,
                    state=signature_lattice_state,
                    return_state=bool(getattr(self.cfg, "use_signature_lattice_generation_cache", True))
                    and not self.training,
                ),
            )
        if self.cfg.max_seq_len <= 0:
            self._ensure_position_embedding_capacity(position_offset + seq_len)

        family_ids_for_context = signature_family_ids if signature_family_ids is not None else signature_ids
        if (
            signature_family_ids is not None
            or signature_ids is not None
            or signature_level_ids is not None
            or signature_relation_ids is not None
            or parent_signature_ids is not None
        ):
            self.registry.observe(
                family_ids=family_ids_for_context,
                level_ids=signature_level_ids,
                relation_ids=signature_relation_ids,
                parent_ids=parent_signature_ids,
            )

        family_context_seq: Optional[torch.Tensor] = None
        level_context_seq: Optional[torch.Tensor] = None
        relation_context_seq: Optional[torch.Tensor] = None
        parent_context_seq: Optional[torch.Tensor] = None
        if family_ids_for_context is not None and family_ids_for_context.numel() > 0:
            family_context_seq = self.registry.family_context(family_ids_for_context)
        if signature_level_ids is not None and signature_level_ids.numel() > 0:
            level_context_seq = self.registry.level_context(signature_level_ids)
        if signature_relation_ids is not None and signature_relation_ids.numel() > 0:
            relation_context_seq = self.registry.relation_context(signature_relation_ids)
        if parent_signature_ids is not None and parent_signature_ids.numel() > 0:
            parent_context_seq = self.registry.parent_context(parent_signature_ids)

        next_token_memory_state: Optional[TokenMemoryState] = token_memory_state
        token_memory_stats: Dict[str, torch.Tensor] = {
            "token_memory_enabled": torch.tensor(0.0, device=input_ids.device),
            "copy_attention_enabled": torch.tensor(0.0, device=input_ids.device),
            "token_memory_gate_mean": torch.tensor(0.0, device=input_ids.device),
            "copy_attention_gate_mean": torch.tensor(0.0, device=input_ids.device),
            "token_memory_copy_confidence": torch.tensor(0.0, device=input_ids.device),
            "copy_attention_max_weight": torch.tensor(0.0, device=input_ids.device),
            "token_memory_memory_fill": torch.tensor(0.0, device=input_ids.device),
            "copy_attention_memory_fill": torch.tensor(0.0, device=input_ids.device),
            "token_memory_window": torch.tensor(float(getattr(self.cfg, "token_memory_window", 0)), device=input_ids.device),
            "token_memory_top_k": torch.tensor(float(getattr(self.cfg, "token_memory_top_k", 0)), device=input_ids.device),
            "token_memory_copy_logits": torch.zeros(batch_size, self.vocab_size, device=input_ids.device, dtype=hidden.dtype),
            "copy_attention_candidate_count": torch.tensor(float(getattr(self.cfg, "token_memory_top_k", 0)), device=input_ids.device),
        }
        if self.token_memory_attention is not None:
            hidden, next_token_memory_state, token_memory_stats = _profile_stage(
                profile_enabled,
                input_ids.device,
                prep_timings,
                "timing_token_memory_ms",
                lambda: self.token_memory_attention(
                    hidden,
                    signature_family_ids=signature_family_ids,
                    signature_ids=signature_ids,
                    signature_level_ids=signature_level_ids,
                    signature_relation_ids=signature_relation_ids,
                    parent_signature_ids=parent_signature_ids,
                    family_context=family_context_seq,
                    level_context=level_context_seq,
                    relation_context=relation_context_seq,
                    parent_context=parent_context_seq,
                    token_ids=input_ids[:, :seq_len],
                    state=token_memory_state,
                    return_state=not self.training and bool(getattr(self.cfg, "use_token_memory_generation_cache", True)),
                ),
            )
        if profile_enabled:
            for key, value in token_memory_stats.items():
                if key.startswith("timing_"):
                    prep_timings[key] = prep_timings.get(key, 0.0) + float(value.detach().item() if torch.is_tensor(value) else float(value))

        input_signature = F.normalize(self.signature_head(hidden.mean(dim=1)), dim=-1)

        return PrismalTorusFrame(
            input_ids=input_ids,
            hidden=hidden,
            input_signature=input_signature,
            signature_lattice_state=next_signature_lattice_state,
            token_memory_state=next_token_memory_state,
            lattice_stats=lattice_stats,
            token_memory_stats=token_memory_stats,
            prep_timings=prep_timings,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            family_context_seq=family_context_seq,
            level_context_seq=level_context_seq,
            relation_context_seq=relation_context_seq,
            parent_context_seq=parent_context_seq,
            position_offset=position_offset,
        )

    def _forward_torus_path(
        self,
        frame: PrismalTorusFrame,
        *,
        slot_state: Optional[torch.Tensor] = None,
        return_signature_lattice_state: bool = False,
        path_index: int = 0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, float],
        Optional[SignatureLatticeState],
    ]:
        profile_enabled = bool(getattr(self.cfg, "profile_runtime", False))
        path_timings: Dict[str, float] = dict(frame.prep_timings)
        input_ids = frame.input_ids
        hidden = frame.hidden
        input_signature = frame.input_signature
        next_signature_lattice_state = frame.signature_lattice_state
        lattice_stats: Dict[str, torch.Tensor] = frame.lattice_stats
        batch_size, seq_len = input_ids.shape
        field_state = self.torus_core.init_state(batch_size, input_ids.device, state=slot_state)
        router_aug_stats: Dict[str, torch.Tensor] = {}
        if self.router is not None and self.use_torus_sharc_router:
            router_slots = self.router.init_slots(batch_size, input_ids.device)

            def _run_sharc_router() -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
                return self.router.route(
                    hidden,
                    router_slots,
                    signature_family_ids=signature_family_ids,
                    signature_ids=signature_ids,
                    signature_level_ids=signature_level_ids,
                    signature_relation_ids=signature_relation_ids,
                    parent_signature_ids=parent_signature_ids,
                    path_index=path_index,
                    layer_index=0,
                    torus_center=frame.input_signature.unsqueeze(1),
                )

            router_delta, _, router_aug_stats = _profile_stage(
                profile_enabled,
                input_ids.device,
                path_timings,
                "timing_sharc_router_ms",
                _run_sharc_router,
            )
            router_aug_stats = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in router_aug_stats.items()
            }
            hidden = hidden + 0.20 * router_delta
        outputs: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []
        active_terms: List[torch.Tensor] = []
        soft_active_terms: List[torch.Tensor] = []
        cell_effective_terms: List[torch.Tensor] = []
        cell_mixture_terms: List[torch.Tensor] = []
        cell_coverage_terms: List[torch.Tensor] = []
        path_usage_entropy_terms: List[torch.Tensor] = []
        path_usage_concentration_terms: List[torch.Tensor] = []
        soft_breadth_terms: List[torch.Tensor] = []
        cell_energy_mean_terms: List[torch.Tensor] = []
        cell_energy_min_terms: List[torch.Tensor] = []
        cell_energy_max_terms: List[torch.Tensor] = []
        global_bus_norm_terms: List[torch.Tensor] = []
        torus_threshold_terms: List[torch.Tensor] = []
        mot_active_expert_terms: List[torch.Tensor] = []
        chunk_solver_iterations: List[torch.Tensor] = []
        chunk_solver_residuals: List[torch.Tensor] = []
        chunk_solver_converged: List[torch.Tensor] = []
        recursive_aux_terms: List[torch.Tensor] = []
        recursive_stat_terms: Dict[str, List[torch.Tensor]] = {}
        family_active_terms: List[torch.Tensor] = []
        family_unique_terms: List[torch.Tensor] = []
        family_bank_terms: List[torch.Tensor] = []
        family_capacity_terms: List[torch.Tensor] = []
        family_budget_terms: List[torch.Tensor] = []
        family_hit_rate_terms: List[torch.Tensor] = []
        family_gate_terms: List[torch.Tensor] = []
        signature_family_ids = frame.signature_family_ids
        signature_ids = frame.signature_ids
        signature_level_ids = frame.signature_level_ids
        signature_relation_ids = frame.signature_relation_ids
        parent_signature_ids = frame.parent_signature_ids
        family_context_seq = frame.family_context_seq
        level_context_seq = frame.level_context_seq
        relation_context_seq = frame.relation_context_seq
        parent_context_seq = frame.parent_context_seq
        position_offset = frame.position_offset
        family_ids_for_context = signature_family_ids if signature_family_ids is not None else signature_ids

        chunk_len = max(1, int(getattr(self.cfg, "torus_chunk_len", seq_len)))
        num_chunks = (seq_len + chunk_len - 1) // chunk_len

        for chunk_index in range(num_chunks):
            chunk_start = chunk_index * chunk_len
            chunk_end = min(seq_len, chunk_start + chunk_len)
            chunk_hidden = hidden[:, chunk_start:chunk_end, :]
            signature_family_slice = signature_family_ids[:, chunk_start:chunk_end] if signature_family_ids is not None else None
            signature_ids_slice = signature_ids[:, chunk_start:chunk_end] if signature_ids is not None else None
            signature_level_slice = signature_level_ids[:, chunk_start:chunk_end] if signature_level_ids is not None else None
            signature_relation_slice = signature_relation_ids[:, chunk_start:chunk_end] if signature_relation_ids is not None else None
            parent_signature_slice = parent_signature_ids[:, chunk_start:chunk_end] if parent_signature_ids is not None else None
            chunk_family_context = (
                family_context_seq[:, chunk_start:chunk_end] if family_context_seq is not None else None
            )
            chunk_level_context = level_context_seq[:, chunk_start:chunk_end] if level_context_seq is not None else None
            chunk_relation_context = relation_context_seq[:, chunk_start:chunk_end] if relation_context_seq is not None else None
            chunk_parent_context = parent_context_seq[:, chunk_start:chunk_end] if parent_context_seq is not None else None
            use_chunk_solver = (
                bool(getattr(self.cfg, "use_fixed_point_solver", False))
                and bool(getattr(self.cfg, "use_chunk_solver_training", True))
            )
            core_step_fn = (
                self.torus_core.main_core_step
                if hasattr(self.torus_core, "main_core_step")
                else (self.torus_core.chunk_solver_step if use_chunk_solver else self.torus_core.chunked_step)
            )
            overlay_step_fn = self.torus_core.overlay_step if hasattr(self.torus_core, "overlay_step") else None

            def _run_chunk_core(
                chunk_hidden_: torch.Tensor,
                field_state_: torch.Tensor,
                family_context_: torch.Tensor,
                level_context_: torch.Tensor,
                relation_context_: torch.Tensor,
                parent_context_: torch.Tensor,
                _step_fn=core_step_fn,
                _chunk_start=chunk_start,
                _signature_family_slice=signature_family_slice,
                _signature_ids_slice=signature_ids_slice,
                _signature_level_slice=signature_level_slice,
                _signature_relation_slice=signature_relation_slice,
                _parent_signature_slice=parent_signature_slice,
            ):
                return _step_fn(
                    chunk_hidden_,
                    field_state_,
                    signature_family_ids=_signature_family_slice,
                    signature_ids=_signature_ids_slice,
                    signature_level_ids=_signature_level_slice,
                    signature_relation_ids=_signature_relation_slice,
                    parent_signature_ids=_parent_signature_slice,
                    registry_context=None,
                    family_context=family_context_ if family_context_.numel() > 0 else None,
                    level_context=level_context_ if level_context_.numel() > 0 else None,
                    relation_context=relation_context_ if relation_context_.numel() > 0 else None,
                    parent_context=parent_context_ if parent_context_.numel() > 0 else None,
                    path_index=path_index,
                    step_index_offset=_chunk_start,
                )

            def _run_chunk_overlay(
                chunk_hidden_: torch.Tensor,
                field_state_: torch.Tensor,
                chunk_stats_: Dict[str, torch.Tensor],
                _chunk_start=chunk_start,
                _signature_family_slice=signature_family_slice,
                _signature_ids_slice=signature_ids_slice,
                _signature_level_slice=signature_level_slice,
                _signature_relation_slice=signature_relation_slice,
                _parent_signature_slice=parent_signature_slice,
                _overlay_step_fn=overlay_step_fn,
            ):
                if _overlay_step_fn is None:
                    return chunk_hidden_, field_state_, chunk_stats_
                return _overlay_step_fn(
                    chunk_hidden_,
                    field_state_,
                    chunk_stats_,
                    signature_family_ids=_signature_family_slice,
                    signature_ids=_signature_ids_slice,
                    signature_level_ids=_signature_level_slice,
                    signature_relation_ids=_signature_relation_slice,
                    parent_signature_ids=_parent_signature_slice,
                    registry_context=None,
                    family_context=chunk_family_context,
                    level_context=chunk_level_context,
                    relation_context=chunk_relation_context,
                    parent_context=chunk_parent_context,
                    path_index=path_index,
                    step_index_offset=_chunk_start,
                    use_solver=use_chunk_solver,
                    relay_mode=False,
                )

            if self.training and self.use_gradient_checkpointing:
                family_arg = chunk_family_context if chunk_family_context is not None else torch.empty(0, device=input_ids.device)
                level_arg = chunk_level_context if chunk_level_context is not None else torch.empty(0, device=input_ids.device)
                relation_arg = chunk_relation_context if chunk_relation_context is not None else torch.empty(0, device=input_ids.device)
                parent_arg = chunk_parent_context if chunk_parent_context is not None else torch.empty(0, device=input_ids.device)
                if profile_enabled:
                    _sync_for_timing(input_ids.device)
                    start_chunk_core = time.perf_counter()
                core_output, field_state, core_stats = checkpoint(
                    _run_chunk_core,
                    chunk_hidden,
                    field_state,
                    family_arg,
                    level_arg,
                    relation_arg,
                    parent_arg,
                    use_reentrant=False,
                )
                if profile_enabled:
                    _sync_for_timing(input_ids.device)
                    path_timings["timing_path_core_ms"] = path_timings.get("timing_path_core_ms", 0.0) + (time.perf_counter() - start_chunk_core) * 1000.0
                    start_chunk_overlay = time.perf_counter()
                chunk_output, field_state, chunk_stats = _run_chunk_overlay(core_output, field_state, core_stats)
                if profile_enabled:
                    _sync_for_timing(input_ids.device)
                    path_timings["timing_path_overlay_ms"] = path_timings.get("timing_path_overlay_ms", 0.0) + (time.perf_counter() - start_chunk_overlay) * 1000.0
            else:
                if profile_enabled:
                    _sync_for_timing(input_ids.device)
                    start_chunk_core = time.perf_counter()
                core_output, field_state, core_stats = _run_chunk_core(
                    chunk_hidden,
                    field_state,
                    chunk_family_context if chunk_family_context is not None else torch.empty(0, device=input_ids.device),
                    chunk_level_context if chunk_level_context is not None else torch.empty(0, device=input_ids.device),
                    chunk_relation_context if chunk_relation_context is not None else torch.empty(0, device=input_ids.device),
                    chunk_parent_context if chunk_parent_context is not None else torch.empty(0, device=input_ids.device),
                )
                if profile_enabled:
                    _sync_for_timing(input_ids.device)
                    path_timings["timing_path_core_ms"] = path_timings.get("timing_path_core_ms", 0.0) + (time.perf_counter() - start_chunk_core) * 1000.0
                    start_chunk_overlay = time.perf_counter()
                chunk_output, field_state, chunk_stats = _run_chunk_overlay(core_output, field_state, core_stats)
                if profile_enabled:
                    _sync_for_timing(input_ids.device)
                    path_timings["timing_path_overlay_ms"] = path_timings.get("timing_path_overlay_ms", 0.0) + (time.perf_counter() - start_chunk_overlay) * 1000.0
            outputs.append(chunk_output)
            entropy_terms.append(chunk_stats["torus_entropy"])
            active_terms.append(chunk_stats.get("emitter_cell_occupancy", chunk_stats.get("active_cells", torch.tensor(0.0, device=input_ids.device))).float())
            if "emitter_cell_soft_occupancy" in chunk_stats:
                soft_active_terms.append(chunk_stats["emitter_cell_soft_occupancy"].float())
            if "emitter_cell_effective_count" in chunk_stats:
                cell_effective_terms.append(chunk_stats["emitter_cell_effective_count"].float())
            if "emitter_cell_mixture_loss" in chunk_stats:
                cell_mixture_terms.append(chunk_stats["emitter_cell_mixture_loss"].float())
            if "emitter_cell_coverage_loss" in chunk_stats:
                cell_coverage_terms.append(chunk_stats["emitter_cell_coverage_loss"].float())
            if "emitter_cell_soft_breadth" in chunk_stats:
                soft_breadth_terms.append(chunk_stats["emitter_cell_soft_breadth"].float())
            if "emitter_usage_entropy" in chunk_stats:
                path_usage_entropy_terms.append(chunk_stats["emitter_usage_entropy"].float())
            if "emitter_usage_concentration" in chunk_stats:
                path_usage_concentration_terms.append(chunk_stats["emitter_usage_concentration"].float())
            if "torus_activity_threshold" in chunk_stats:
                torus_threshold_terms.append(chunk_stats["torus_activity_threshold"].float())
            if "mot_active_experts" in chunk_stats:
                mot_active_expert_terms.append(chunk_stats["mot_active_experts"].float())
            elif "mot_mot_active_experts" in chunk_stats:
                mot_active_expert_terms.append(chunk_stats["mot_mot_active_experts"].float())
            if "specialist_family_specialist_active_count" in chunk_stats:
                family_active_terms.append(chunk_stats["specialist_family_specialist_active_count"].float())
            if "specialist_family_specialist_unique_families" in chunk_stats:
                family_unique_terms.append(chunk_stats["specialist_family_specialist_unique_families"].float())
            if "specialist_family_specialist_bank_size" in chunk_stats:
                family_bank_terms.append(chunk_stats["specialist_family_specialist_bank_size"].float())
            if "specialist_family_specialist_capacity" in chunk_stats:
                family_capacity_terms.append(chunk_stats["specialist_family_specialist_capacity"].float())
            if "specialist_family_specialist_budget" in chunk_stats:
                family_budget_terms.append(chunk_stats["specialist_family_specialist_budget"].float())
            if "specialist_family_specialist_hit_rate" in chunk_stats:
                family_hit_rate_terms.append(chunk_stats["specialist_family_specialist_hit_rate"].float())
            if "specialist_family_specialist_gate_mean" in chunk_stats:
                family_gate_terms.append(chunk_stats["specialist_family_specialist_gate_mean"].float())
            if "cell_energy_mean" in chunk_stats:
                cell_energy_mean_terms.append(chunk_stats["cell_energy_mean"].float())
            if "cell_energy_min" in chunk_stats:
                cell_energy_min_terms.append(chunk_stats["cell_energy_min"].float())
            if "cell_energy_max" in chunk_stats:
                cell_energy_max_terms.append(chunk_stats["cell_energy_max"].float())
            if "global_bus_norm" in chunk_stats:
                global_bus_norm_terms.append(chunk_stats["global_bus_norm"].float())
            if "fixed_point_iterations" in chunk_stats:
                chunk_solver_iterations.append(chunk_stats["fixed_point_iterations"].float())
            if "fixed_point_residual" in chunk_stats:
                chunk_solver_residuals.append(chunk_stats["fixed_point_residual"].float())
            if "fixed_point_converged" in chunk_stats:
                chunk_solver_converged.append(chunk_stats["fixed_point_converged"].float())
            if "recursive_aux_loss" in chunk_stats and torch.is_tensor(chunk_stats["recursive_aux_loss"]):
                recursive_aux_terms.append(chunk_stats["recursive_aux_loss"].float())
            for key, value in chunk_stats.items():
                if key.startswith("recursive_") and torch.is_tensor(value) and value.numel() == 1:
                    recursive_stat_terms.setdefault(key, []).append(value.detach().float().reshape(()))
            if profile_enabled:
                for key, value in chunk_stats.items():
                    if key.startswith("timing_"):
                        path_timings[key] = path_timings.get(key, 0.0) + float(value.item())

        path_hidden_state = _profile_stage(
            profile_enabled,
            input_ids.device,
            path_timings,
            "timing_path_finalize_ms",
            lambda: torch.cat(outputs, dim=1),
        )
        path_hidden_state = self.final_norm(path_hidden_state)
        logits, path_hidden_state, neighborhood_logits = self._construction_logits_from_hidden(path_hidden_state)
        output_signature = F.normalize(self.signature_head(path_hidden_state.mean(dim=1)), dim=-1)
        agreement = F.cosine_similarity(input_signature, output_signature, dim=-1)
        route_entropy = torch.stack(entropy_terms).mean() if entropy_terms else torch.tensor(0.0, device=input_ids.device)
        active_emitters = torch.stack(active_terms).mean() if active_terms else torch.tensor(0.0, device=input_ids.device)
        soft_active_emitters = torch.stack(soft_active_terms).mean() if soft_active_terms else torch.tensor(0.0, device=input_ids.device)
        cell_effective = torch.stack(cell_effective_terms).mean() if cell_effective_terms else torch.tensor(0.0, device=input_ids.device)
        cell_mixture_loss = torch.stack(cell_mixture_terms).mean() if cell_mixture_terms else torch.tensor(0.0, device=input_ids.device)
        cell_coverage_loss = torch.stack(cell_coverage_terms).mean() if cell_coverage_terms else torch.tensor(0.0, device=input_ids.device)
        usage_entropy = torch.stack(path_usage_entropy_terms).mean() if path_usage_entropy_terms else torch.tensor(0.0, device=input_ids.device)
        usage_concentration = torch.stack(path_usage_concentration_terms).mean() if path_usage_concentration_terms else torch.tensor(0.0, device=input_ids.device)
        soft_breadth = torch.stack(soft_breadth_terms).mean() if soft_breadth_terms else torch.tensor(0.0, device=input_ids.device)
        torus_activity_threshold = torch.stack(torus_threshold_terms).mean() if torus_threshold_terms else torch.tensor(float(self.torus_core.activity_threshold), device=input_ids.device)
        cell_energy_mean = torch.stack(cell_energy_mean_terms).mean() if cell_energy_mean_terms else torch.tensor(0.0, device=input_ids.device)
        cell_energy_min = torch.stack(cell_energy_min_terms).min() if cell_energy_min_terms else torch.tensor(0.0, device=input_ids.device)
        cell_energy_max = torch.stack(cell_energy_max_terms).max() if cell_energy_max_terms else torch.tensor(0.0, device=input_ids.device)
        global_bus_norm = torch.stack(global_bus_norm_terms).mean() if global_bus_norm_terms else torch.tensor(0.0, device=input_ids.device)
        recursive_aux_loss = torch.stack(recursive_aux_terms).mean() if recursive_aux_terms else torch.tensor(0.0, device=input_ids.device)
        mot_active_experts = torch.stack(mot_active_expert_terms).mean() if mot_active_expert_terms else torch.tensor(0.0, device=input_ids.device)
        family_active_count = torch.stack(family_active_terms).mean() if family_active_terms else torch.tensor(0.0, device=input_ids.device)
        family_unique_count = torch.stack(family_unique_terms).mean() if family_unique_terms else torch.tensor(0.0, device=input_ids.device)
        family_bank_size = torch.stack(family_bank_terms).mean() if family_bank_terms else torch.tensor(0.0, device=input_ids.device)
        family_capacity = torch.stack(family_capacity_terms).mean() if family_capacity_terms else torch.tensor(0.0, device=input_ids.device)
        family_budget = torch.stack(family_budget_terms).mean() if family_budget_terms else torch.tensor(0.0, device=input_ids.device)
        family_hit_rate = torch.stack(family_hit_rate_terms).mean() if family_hit_rate_terms else torch.tensor(0.0, device=input_ids.device)
        family_gate_mean = torch.stack(family_gate_terms).mean() if family_gate_terms else torch.tensor(0.0, device=input_ids.device)
        path_score = agreement - 0.15 * route_entropy
        if self.use_race_lanes:
            lane_norm = float(path_index % max(self.lane_count, 1)) / max(self.lane_count - 1, 1)
            path_score = path_score + (1.0 - lane_norm) * 0.015 - lane_norm * 0.015 * route_entropy
        if profile_enabled:
            path_timings["timing_path_total_ms"] = sum(path_timings.values())
        path_stats = {
            "active_cells": active_emitters.detach(),
            "emitter_cell_occupancy": active_emitters.detach(),
            "emitter_cell_breadth": (
                active_emitters
                / max(float(self.torus_core.depth * self.torus_core.height * self.torus_core.width), 1.0)
            ).detach(),
            "avg_emitter_cell_soft_occupancy": soft_active_emitters.detach(),
            "emitter_cell_soft_occupancy": soft_active_emitters.detach(),
            "avg_emitter_cell_soft_breadth": soft_breadth.detach(),
            "emitter_cell_soft_breadth": soft_breadth.detach(),
            "emitter_cell_effective_count": cell_effective,
            "emitter_cell_mixture_loss": cell_mixture_loss,
            "emitter_cell_coverage_loss": cell_coverage_loss,
            "torus_coverage_loss": cell_coverage_loss,
            "avg_emitter_usage_entropy": usage_entropy.detach(),
            "avg_emitter_usage_concentration": usage_concentration.detach(),
            "emitter_usage_entropy": usage_entropy.detach(),
            "emitter_usage_concentration": usage_concentration.detach(),
            "torus_activity_threshold": torus_activity_threshold,
            "cell_energy_mean": cell_energy_mean,
            "cell_energy_min": cell_energy_min,
            "cell_energy_max": cell_energy_max,
            "torus_entropy": route_entropy,
            "global_bus_norm": global_bus_norm.detach(),
            "signature_neighborhood_logits": neighborhood_logits,
            "mot_active_experts": mot_active_experts.detach(),
            "specialist_family_specialist_active_count": family_active_count.detach(),
            "specialist_family_specialist_unique_families": family_unique_count.detach(),
            "specialist_family_specialist_bank_size": family_bank_size.detach(),
            "specialist_family_specialist_capacity": family_capacity.detach(),
            "specialist_family_specialist_budget": family_budget.detach(),
            "specialist_family_specialist_hit_rate": family_hit_rate.detach(),
            "specialist_family_specialist_gate_mean": family_gate_mean.detach(),
        }
        path_stats.update(lattice_stats)
        if chunk_solver_iterations:
            path_stats["fixed_point_iterations"] = torch.stack(chunk_solver_iterations).mean().detach()
        if chunk_solver_residuals:
            path_stats["fixed_point_residual"] = torch.stack(chunk_solver_residuals).mean().detach()
        if chunk_solver_converged:
            path_stats["fixed_point_converged"] = torch.stack(chunk_solver_converged).mean().detach()
        for key, values in recursive_stat_terms.items():
            if values:
                path_stats[key] = torch.stack([v.reshape(()) for v in values]).mean().detach()
        return (
            path_hidden_state,
            logits,
            output_signature,
            field_state,
            path_score,
            route_entropy,
            active_emitters,
            recursive_aux_loss,
            path_stats,
            path_timings,
            next_signature_lattice_state,
        )

    def _forward_torus(
        self,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        slot_state: Optional[torch.Tensor] = None,
        signature_lattice_state: Optional[SignatureLatticeState] = None,
        token_memory_state: Optional[TokenMemoryState] = None,
        path_index: Optional[int] = None,
        position_offset: int = 0,
        gate_runtime: Optional[Dict[str, torch.Tensor | GateResidencyPlan]] = None,
    ) -> PrismalWaveOutput:
        profile_enabled = bool(getattr(self.cfg, "profile_runtime", False))
        timings: Dict[str, float] = {}
        gate_runtime = gate_runtime or {}
        race_enabled = self.use_race_lanes and not self.training
        path_count = self.lane_count if (race_enabled and path_index is None) else max(1, self.cfg.n_paths)
        path_indices = [path_index] if path_index is not None else list(range(path_count))
        path_hidden: List[torch.Tensor] = []
        path_logits: List[torch.Tensor] = []
        path_signatures: List[torch.Tensor] = []
        path_scores: List[torch.Tensor] = []
        path_entropy: List[torch.Tensor] = []
        path_active_emitters: List[torch.Tensor] = []
        path_soft_occupancy: List[torch.Tensor] = []
        path_soft_breadth: List[torch.Tensor] = []
        path_slot_states: List[PrismalTorusState] = []
        path_global_bus_norm: List[torch.Tensor] = []
        path_topk_entropy: List[torch.Tensor] = []
        path_topk_effective_count: List[torch.Tensor] = []
        path_mixture_loss: List[torch.Tensor] = []
        path_coverage_loss: List[torch.Tensor] = []
        path_usage_entropy: List[torch.Tensor] = []
        path_usage_concentration: List[torch.Tensor] = []
        path_recursive_aux_loss: List[torch.Tensor] = []
        path_emitter_top_weights: List[torch.Tensor] = []
        path_family_active_counts: List[torch.Tensor] = []
        path_family_unique_families: List[torch.Tensor] = []
        path_family_bank_sizes: List[torch.Tensor] = []
        path_family_capacities: List[torch.Tensor] = []
        path_family_budgets: List[torch.Tensor] = []
        path_family_hit_rates: List[torch.Tensor] = []
        path_family_gate_means: List[torch.Tensor] = []
        path_signature_lattice_cache_norm: List[torch.Tensor] = []
        path_signature_lattice_gate_mean: List[torch.Tensor] = []
        path_signature_lattice_candidate_count: List[torch.Tensor] = []
        path_signature_lattice_enabled: List[torch.Tensor] = []
        path_mot_active_experts: List[torch.Tensor] = []
        recursive_stat_terms: Dict[str, List[torch.Tensor]] = {}
        frame = self._prepare_torus_frame(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            signature_lattice_state=signature_lattice_state,
            token_memory_state=token_memory_state,
            position_offset=position_offset,
        )
        final_signature_lattice_state: Optional[SignatureLatticeState] = frame.signature_lattice_state
        race_audit = False
        scout_path_scores: Optional[torch.Tensor] = None
        race_lane_scores: Optional[torch.Tensor] = None
        selected_lane_index: Optional[int] = None
        if race_enabled and path_index is None:
            scout_path_scores = self._scout_race_scores(frame, path_count)
            race_lane_scores = scout_path_scores
            selected_lane_index = int(scout_path_scores.argmax().item())
            selected_path_index = torch.full(
                (input_ids.size(0),),
                selected_lane_index,
                device=input_ids.device,
                dtype=torch.long,
            )
            audit_period = max(1, int(self.lane_relay_hop_spacing) * 8)
            race_audit = position_offset > 0 and position_offset % audit_period == 0
            path_indices = list(range(path_count)) if race_audit else [selected_lane_index]

        for current_path_index in path_indices:
            if profile_enabled:
                _sync_for_timing(input_ids.device)
                start_path = time.perf_counter()
            (
                hidden_state,
                logits,
                output_signature,
                field_state,
                path_score,
                route_entropy,
                active_emitters,
                recursive_aux_loss,
                path_stats,
                path_timings,
                path_signature_lattice_state,
            ) = self._forward_torus_path(
                frame,
                slot_state=slot_state,
                return_signature_lattice_state=(
                    bool(getattr(self.cfg, "use_signature_lattice_generation_cache", True))
                    and not self.training
                ),
                path_index=current_path_index,
            )
            if final_signature_lattice_state is None and path_signature_lattice_state is not None:
                final_signature_lattice_state = path_signature_lattice_state
            path_hidden.append(hidden_state)
            path_logits.append(logits)
            path_signatures.append(output_signature)
            path_scores.append(path_score)
            path_entropy.append(route_entropy)
            path_active_emitters.append(active_emitters)
            if "emitter_cell_soft_occupancy" in path_stats:
                path_soft_occupancy.append(path_stats["emitter_cell_soft_occupancy"].float())
            elif "avg_emitter_cell_soft_occupancy" in path_stats:
                path_soft_occupancy.append(path_stats["avg_emitter_cell_soft_occupancy"].float())
            if "emitter_cell_soft_breadth" in path_stats:
                path_soft_breadth.append(path_stats["emitter_cell_soft_breadth"].float())
            elif "avg_emitter_cell_soft_breadth" in path_stats:
                path_soft_breadth.append(path_stats["avg_emitter_cell_soft_breadth"].float())
            if "global_bus_norm" in path_stats:
                path_global_bus_norm.append(path_stats["global_bus_norm"].float())
            if "emitter_cell_effective_count" in path_stats:
                path_topk_effective_count.append(path_stats["emitter_cell_effective_count"].float())
            elif "avg_emitter_topk_effective_count" in path_stats:
                path_topk_effective_count.append(path_stats["avg_emitter_topk_effective_count"].float())
            if "emitter_usage_entropy" in path_stats:
                path_usage_entropy.append(path_stats["emitter_usage_entropy"].float())
            elif "avg_emitter_usage_entropy" in path_stats:
                path_usage_entropy.append(path_stats["avg_emitter_usage_entropy"].float())
            if "emitter_usage_concentration" in path_stats:
                path_usage_concentration.append(path_stats["emitter_usage_concentration"].float())
            elif "avg_emitter_usage_concentration" in path_stats:
                path_usage_concentration.append(path_stats["avg_emitter_usage_concentration"].float())
            if "emitter_top_weights" in path_stats:
                path_emitter_top_weights.append(path_stats["emitter_top_weights"].float())
            if "emitter_cell_mixture_loss" in path_stats:
                path_mixture_loss.append(path_stats["emitter_cell_mixture_loss"].float())
            if "emitter_cell_coverage_loss" in path_stats:
                path_coverage_loss.append(path_stats["emitter_cell_coverage_loss"].float())
            if "signature_lattice_cache_norm" in path_stats:
                path_signature_lattice_cache_norm.append(path_stats["signature_lattice_cache_norm"].float())
            if "signature_lattice_gate_mean" in path_stats:
                path_signature_lattice_gate_mean.append(path_stats["signature_lattice_gate_mean"].float())
            if "signature_lattice_candidate_count" in path_stats:
                path_signature_lattice_candidate_count.append(path_stats["signature_lattice_candidate_count"].float())
            if "signature_lattice_enabled" in path_stats:
                path_signature_lattice_enabled.append(path_stats["signature_lattice_enabled"].float())
            if "mot_active_experts" in path_stats:
                path_mot_active_experts.append(path_stats["mot_active_experts"].float())
            if "specialist_family_specialist_active_count" in path_stats:
                path_family_active_counts.append(path_stats["specialist_family_specialist_active_count"].float())
            if "specialist_family_specialist_unique_families" in path_stats:
                path_family_unique_families.append(path_stats["specialist_family_specialist_unique_families"].float())
            if "specialist_family_specialist_bank_size" in path_stats:
                path_family_bank_sizes.append(path_stats["specialist_family_specialist_bank_size"].float())
            if "specialist_family_specialist_capacity" in path_stats:
                path_family_capacities.append(path_stats["specialist_family_specialist_capacity"].float())
            if "specialist_family_specialist_budget" in path_stats:
                path_family_budgets.append(path_stats["specialist_family_specialist_budget"].float())
            if "specialist_family_specialist_hit_rate" in path_stats:
                path_family_hit_rates.append(path_stats["specialist_family_specialist_hit_rate"].float())
            if "specialist_family_specialist_gate_mean" in path_stats:
                path_family_gate_means.append(path_stats["specialist_family_specialist_gate_mean"].float())
            for key, value in path_stats.items():
                if key.startswith("recursive_") and torch.is_tensor(value) and value.numel() == 1:
                    recursive_stat_terms.setdefault(key, []).append(value.detach().float().reshape(()))
            if torch.is_tensor(recursive_aux_loss):
                path_recursive_aux_loss.append(recursive_aux_loss.float())
            path_slot_states.append(field_state)
            if profile_enabled:
                for key, value in path_timings.items():
                    timings[key] = timings.get(key, 0.0) + float(value)
            if profile_enabled:
                _sync_for_timing(input_ids.device)
                timings["timing_path_loop_ms"] = timings.get("timing_path_loop_ms", 0.0) + (time.perf_counter() - start_path) * 1000.0

        if profile_enabled:
            _sync_for_timing(input_ids.device)
            start_aggregate = time.perf_counter()
        path_hidden_tensor = torch.stack(path_hidden, dim=1)
        path_logits_tensor = torch.stack(path_logits, dim=1)
        path_scores_tensor = torch.stack(path_scores, dim=1)
        if race_enabled and path_index is None and not race_audit and selected_lane_index is not None:
            path_weights = torch.ones(path_scores_tensor.size(0), 1, device=input_ids.device)
            final_hidden = self.final_norm(path_hidden_tensor[:, 0])
            selected_path_index = torch.full(
                (input_ids.size(0),),
                selected_lane_index,
                device=input_ids.device,
                dtype=torch.long,
            )
        elif path_index is None:
            path_weights = F.softmax(path_scores_tensor / max(self.cfg.validator_temperature, 1e-3), dim=-1)
            final_hidden = torch.einsum("bp,bptd->btd", path_weights, path_hidden_tensor)
            final_hidden = self.final_norm(final_hidden)
            selected_path_index = path_scores_tensor.argmax(dim=-1)
        else:
            path_weights = torch.ones(path_scores_tensor.size(0), 1, device=input_ids.device)
            final_hidden = self.final_norm(path_hidden_tensor[:, 0])
            selected_path_index = torch.full((input_ids.size(0),), int(path_index), device=input_ids.device, dtype=torch.long)
        final_logits, final_hidden, final_neighborhood_logits = self._construction_logits_from_hidden(final_hidden)
        if profile_enabled:
            _sync_for_timing(input_ids.device)
            timings["timing_path_aggregate_ms"] = (time.perf_counter() - start_aggregate) * 1000.0

        input_signature = frame.input_signature
        final_output_signature = F.normalize(self.signature_head(final_hidden.mean(dim=1)), dim=-1)
        signature_cosine = F.cosine_similarity(input_signature, final_output_signature, dim=-1)

        signature_loss = torch.tensor(0.0, device=input_ids.device)
        signature_accuracy = torch.tensor(0.0, device=input_ids.device)
        family_targets_source = signature_family_ids if signature_family_ids is not None else signature_ids
        if family_targets_source is not None and family_targets_source.size(1) > 1:
            signature_logits = final_neighborhood_logits[:, :-1, :]
            signature_targets = self._signature_family_targets(family_targets_source[:, 1 : 1 + signature_logits.size(1)])
            valid_signature = signature_targets.ne(0)
            if loss_mask is not None:
                valid_signature = valid_signature & loss_mask[:, : signature_logits.size(1)].to(device=input_ids.device).bool()
            if valid_signature.any():
                signature_token_losses = F.cross_entropy(
                    signature_logits.reshape(-1, signature_logits.size(-1)),
                    signature_targets.reshape(-1),
                    reduction="none",
                ).view_as(signature_targets)
                signature_loss = signature_token_losses.masked_select(valid_signature).mean()
                signature_predictions = signature_logits.argmax(dim=-1)
                signature_accuracy = (
                    signature_predictions.eq(signature_targets).masked_select(valid_signature).float().mean()
                )
        signature_level_loss = torch.tensor(0.0, device=input_ids.device)
        signature_level_accuracy = torch.tensor(0.0, device=input_ids.device)
        if signature_level_ids is not None and signature_level_ids.size(1) > 1:
            level_logits = self.signature_level_head(final_hidden[:, :-1, :])
            level_targets = signature_level_ids[:, 1 : 1 + level_logits.size(1)]
            valid_level = level_targets.ne(0)
            if loss_mask is not None:
                valid_level = valid_level & loss_mask[:, : level_logits.size(1)].to(device=input_ids.device).bool()
            if valid_level.any():
                level_token_losses = F.cross_entropy(
                    level_logits.reshape(-1, level_logits.size(-1)),
                    level_targets.reshape(-1),
                    reduction="none",
                ).view_as(level_targets)
                signature_level_loss = level_token_losses.masked_select(valid_level).mean()
                level_predictions = level_logits.argmax(dim=-1)
                signature_level_accuracy = (
                    level_predictions.eq(level_targets).masked_select(valid_level).float().mean()
                )
        signature_relation_loss = torch.tensor(0.0, device=input_ids.device)
        signature_relation_accuracy = torch.tensor(0.0, device=input_ids.device)
        if signature_relation_ids is not None and signature_relation_ids.size(1) > 1:
            relation_logits = self.signature_relation_head(final_hidden[:, :-1, :])
            relation_targets = signature_relation_ids[:, 1 : 1 + relation_logits.size(1)]
            valid_relation = relation_targets.ne(0)
            if loss_mask is not None:
                valid_relation = valid_relation & loss_mask[:, : relation_logits.size(1)].to(device=input_ids.device).bool()
            if valid_relation.any():
                relation_token_losses = F.cross_entropy(
                    relation_logits.reshape(-1, relation_logits.size(-1)),
                    relation_targets.reshape(-1),
                    reduction="none",
                ).view_as(relation_targets)
                signature_relation_loss = relation_token_losses.masked_select(valid_relation).mean()
                relation_predictions = relation_logits.argmax(dim=-1)
                signature_relation_accuracy = (
                    relation_predictions.eq(relation_targets).masked_select(valid_relation).float().mean()
                )
        signature_agreement = signature_cosine
        agreement_parts = [signature_accuracy, signature_level_accuracy, signature_relation_accuracy]
        if signature_ids is not None and signature_ids.size(1) > 1:
            signature_agreement = torch.stack(agreement_parts).mean()

        contrastive_loss = torch.tensor(0.0, device=input_ids.device)
        if final_output_signature.size(0) > 1:
            contrastive_logits = torch.matmul(
                F.normalize(input_signature, dim=-1),
                F.normalize(final_output_signature, dim=-1).transpose(0, 1),
            ) / max(self.cfg.signature_temperature, 1e-3)
            contrastive_targets = torch.arange(final_output_signature.size(0), device=input_ids.device)
            contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_targets)

        avg_entropy = torch.stack(path_entropy).mean() if path_entropy else torch.tensor(0.0, device=input_ids.device)
        avg_active = torch.stack(path_active_emitters).mean() if path_active_emitters else torch.tensor(0.0, device=input_ids.device)
        avg_soft_active = torch.stack(path_soft_occupancy).mean() if path_soft_occupancy else torch.tensor(0.0, device=input_ids.device)
        avg_soft_breadth = torch.stack(path_soft_breadth).mean() if path_soft_breadth else torch.tensor(0.0, device=input_ids.device)
        avg_usage_entropy = torch.stack(path_usage_entropy).mean() if path_usage_entropy else torch.tensor(0.0, device=input_ids.device)
        avg_usage_concentration = torch.stack(path_usage_concentration).mean() if path_usage_concentration else torch.tensor(0.0, device=input_ids.device)
        avg_mixture_loss = torch.stack(path_mixture_loss).mean() if path_mixture_loss else torch.tensor(0.0, device=input_ids.device)
        avg_coverage_loss = torch.stack(path_coverage_loss).mean() if path_coverage_loss else torch.tensor(0.0, device=input_ids.device)
        avg_recursive_aux_loss = torch.stack(path_recursive_aux_loss).mean() if path_recursive_aux_loss else torch.tensor(0.0, device=input_ids.device)
        avg_global_bus_norm = torch.stack(path_global_bus_norm).mean() if path_global_bus_norm else torch.tensor(0.0, device=input_ids.device)
        avg_family_active_count = torch.stack(path_family_active_counts).mean() if path_family_active_counts else torch.tensor(0.0, device=input_ids.device)
        avg_family_unique_families = torch.stack(path_family_unique_families).mean() if path_family_unique_families else torch.tensor(0.0, device=input_ids.device)
        avg_family_bank_size = torch.stack(path_family_bank_sizes).mean() if path_family_bank_sizes else torch.tensor(0.0, device=input_ids.device)
        avg_family_capacity = torch.stack(path_family_capacities).mean() if path_family_capacities else torch.tensor(0.0, device=input_ids.device)
        avg_family_budget = torch.stack(path_family_budgets).mean() if path_family_budgets else torch.tensor(0.0, device=input_ids.device)
        avg_family_hit_rate = torch.stack(path_family_hit_rates).mean() if path_family_hit_rates else torch.tensor(0.0, device=input_ids.device)
        avg_family_gate_mean = torch.stack(path_family_gate_means).mean() if path_family_gate_means else torch.tensor(0.0, device=input_ids.device)
        if path_index is None:
            final_field_state = torch.einsum(
                "bp,bpzxyc->bzxyc",
                path_weights,
                torch.stack([slot.field for slot in path_slot_states], dim=1),
            )
            final_bus_state = torch.einsum(
                "bp,bpsd->bsd",
                path_weights,
                torch.stack([slot.bus for slot in path_slot_states], dim=1),
            )
            final_slot_state = PrismalTorusState(field=final_field_state, bus=final_bus_state)
        else:
            final_slot_state = path_slot_states[0]

        lane_limit = self.lane_count if self.use_race_lanes else 1
        scout_count = self._race_scout_count()
        scout_score = torch.tensor(0.0, device=input_ids.device)
        lane_band = torch.tensor(0, device=input_ids.device)
        lane_temperature = torch.tensor(self._race_lane_temperature(0, position_offset, 1.0), device=input_ids.device)
        if race_enabled and path_index is None and path_scores_tensor.numel() > 0:
            if profile_enabled:
                _sync_for_timing(input_ids.device)
                start_lane = time.perf_counter()
            combined_scores = (
                race_lane_scores.detach().float()
                if race_lane_scores is not None
                else path_scores_tensor.detach().float()
            )
            if path_active_emitters:
                active_tensor = torch.stack(path_active_emitters).detach().float().view(1, -1)
                combined_scores = combined_scores + 0.01 * (active_tensor - active_tensor.mean(dim=-1, keepdim=True))
            eligible = combined_scores[:, :lane_limit]
            scout_k = min(max(1, scout_count), eligible.size(-1))
            scout_values, scout_indices = torch.topk(eligible, k=scout_k, dim=-1)
            if race_audit or selected_lane_index is None:
                selected_path_index = scout_indices[:, 0]
                scout_score = scout_values[:, 0].mean()
                lane_band = torch.tensor(
                    self._race_lane_band_from_score(float(scout_score.item())),
                    device=input_ids.device,
                )
                lane_temperature = torch.tensor(
                    self._race_lane_temperature(int(selected_path_index.view(-1)[0].item()), position_offset, 1.0),
                    device=input_ids.device,
                )
            else:
                scout_score = combined_scores[:, int(selected_lane_index)].mean()
                lane_band = torch.tensor(
                    self._race_lane_band_from_score(float(scout_score.item())),
                    device=input_ids.device,
                )
                lane_temperature = torch.tensor(
                    self._race_lane_temperature(int(selected_lane_index), position_offset, 1.0),
                    device=input_ids.device,
                )
            if profile_enabled:
                _sync_for_timing(input_ids.device)
                timings["timing_lane_select_ms"] = (time.perf_counter() - start_lane) * 1000.0
        elif race_enabled and path_index is not None:
            lane_band = torch.tensor(
                self._race_lane_band_from_score(float(path_scores_tensor[:, 0].mean().item())),
                device=input_ids.device,
            )
            lane_temperature = torch.tensor(
                self._race_lane_temperature(int(path_index), position_offset, 1.0),
                device=input_ids.device,
            )

        pairwise_diversity = torch.tensor(0.0, device=input_ids.device)
        if len(path_signatures) > 1:
            div_terms: List[torch.Tensor] = []
            for i in range(len(path_signatures)):
                for j in range(i + 1, len(path_signatures)):
                    div_terms.append(1.0 - F.cosine_similarity(path_signatures[i], path_signatures[j], dim=-1).mean())
            if div_terms:
                pairwise_diversity = torch.stack(div_terms).mean()

        route_stats = {
            "path_weights": path_weights.detach(),
            "path_scores": path_scores_tensor.detach(),
            "race_scout_path_scores": (
                race_lane_scores.detach()
                if race_lane_scores is not None
                else path_scores_tensor.detach()
            ),
            "selected_path_index": selected_path_index.detach(),
            "path_entropy": torch.stack(path_entropy).detach() if path_entropy else torch.zeros(1, device=input_ids.device).detach(),
            "path_active_emitters": torch.stack(path_active_emitters).detach() if path_active_emitters else torch.zeros(1, device=input_ids.device).detach(),
            "signature_agreement": signature_agreement.detach(),
            "signature_accuracy": signature_accuracy.detach(),
            "signature_level_accuracy": signature_level_accuracy.detach(),
            "signature_relation_accuracy": signature_relation_accuracy.detach(),
            "signature_cosine": signature_cosine.detach(),
            "signature_loss": signature_loss.detach(),
            "signature_neighborhood_logits": final_neighborhood_logits.detach(),
            "signature_neighborhood_confidence": F.softmax(
                self._sanitize_tensor(final_neighborhood_logits.detach(), fallback=torch.zeros_like(final_neighborhood_logits))[0],
                dim=-1,
            ).amax(dim=-1).mean(),
            "signature_level_loss": signature_level_loss.detach(),
            "signature_relation_loss": signature_relation_loss.detach(),
            "signature_contrastive_loss": contrastive_loss.detach(),
            "pairwise_diversity": pairwise_diversity.detach(),
            "avg_entropy": avg_entropy.detach(),
            "avg_active_emitters": avg_active.detach(),
            "emitter_cell_occupancy": avg_active.detach(),
            "avg_emitter_cell_soft_occupancy": avg_soft_active.detach(),
            "emitter_cell_soft_occupancy": avg_soft_active.detach(),
            "avg_emitter_usage_entropy": avg_usage_entropy.detach(),
            "avg_emitter_usage_concentration": avg_usage_concentration.detach(),
            "emitter_usage_entropy": avg_usage_entropy.detach(),
            "emitter_usage_concentration": avg_usage_concentration.detach(),
            "specialist_family_specialist_active_count": avg_family_active_count.detach(),
            "specialist_family_specialist_unique_families": avg_family_unique_families.detach(),
            "specialist_family_specialist_bank_size": avg_family_bank_size.detach(),
            "specialist_family_specialist_capacity": avg_family_capacity.detach(),
            "specialist_family_specialist_budget": avg_family_budget.detach(),
            "specialist_family_specialist_hit_rate": avg_family_hit_rate.detach(),
            "specialist_family_specialist_gate_mean": avg_family_gate_mean.detach(),
            "emitter_cell_breadth": (
                avg_active / max(float(self.torus_core.depth * self.torus_core.height * self.torus_core.width), 1.0)
            ).detach(),
            "avg_emitter_cell_soft_breadth": avg_soft_breadth.detach(),
            "emitter_cell_soft_breadth": avg_soft_breadth.detach(),
            "emitter_cell_effective_count": torch.stack(path_topk_effective_count).mean().detach() if path_topk_effective_count else torch.tensor(0.0, device=input_ids.device).detach(),
            "emitter_cell_mixture_loss": avg_mixture_loss.detach(),
            "emitter_cell_coverage_loss": avg_coverage_loss.detach(),
            "torus_coverage_loss": avg_coverage_loss.detach(),
            "emitter_mixture_loss": avg_mixture_loss.detach(),
            "global_bus_norm": avg_global_bus_norm.detach(),
            "signature_lattice_cache_norm": (
                torch.stack(path_signature_lattice_cache_norm).mean().detach()
                if path_signature_lattice_cache_norm
                else torch.tensor(0.0, device=input_ids.device).detach()
            ),
            "signature_lattice_gate_mean": (
                torch.stack(path_signature_lattice_gate_mean).mean().detach()
                if path_signature_lattice_gate_mean
                else torch.tensor(0.0, device=input_ids.device).detach()
            ),
            "signature_lattice_candidate_count": (
                torch.stack(path_signature_lattice_candidate_count).mean().detach()
                if path_signature_lattice_candidate_count
                else torch.tensor(0.0, device=input_ids.device).detach()
            ),
            "signature_lattice_enabled": (
                torch.stack(path_signature_lattice_enabled).mean().detach()
                if path_signature_lattice_enabled
                else torch.tensor(0.0, device=input_ids.device).detach()
            ),
            "token_memory_enabled": frame.token_memory_stats.get("token_memory_enabled", torch.tensor(0.0, device=input_ids.device)).detach(),
            "copy_attention_enabled": frame.token_memory_stats.get("copy_attention_enabled", torch.tensor(0.0, device=input_ids.device)).detach(),
            "token_memory_gate_mean": frame.token_memory_stats.get("token_memory_gate_mean", torch.tensor(0.0, device=input_ids.device)).detach(),
            "copy_attention_gate_mean": frame.token_memory_stats.get("copy_attention_gate_mean", torch.tensor(0.0, device=input_ids.device)).detach(),
            "token_memory_copy_confidence": frame.token_memory_stats.get("token_memory_copy_confidence", torch.tensor(0.0, device=input_ids.device)).detach(),
            "copy_attention_max_weight": frame.token_memory_stats.get("copy_attention_max_weight", torch.tensor(0.0, device=input_ids.device)).detach(),
            "token_memory_memory_fill": frame.token_memory_stats.get("token_memory_memory_fill", torch.tensor(0.0, device=input_ids.device)).detach(),
            "copy_attention_memory_fill": frame.token_memory_stats.get("copy_attention_memory_fill", torch.tensor(0.0, device=input_ids.device)).detach(),
            "token_memory_window": frame.token_memory_stats.get("token_memory_window", torch.tensor(0.0, device=input_ids.device)).detach(),
            "token_memory_top_k": frame.token_memory_stats.get("token_memory_top_k", torch.tensor(0.0, device=input_ids.device)).detach(),
            "copy_attention_candidate_count": frame.token_memory_stats.get(
                "copy_attention_candidate_count",
                torch.tensor(0.0, device=input_ids.device),
            ).detach(),
            "token_memory_copy_logits": frame.token_memory_stats.get(
                "token_memory_copy_logits",
                torch.zeros(input_ids.size(0), self.vocab_size, device=input_ids.device, dtype=final_logits.dtype),
            ).detach(),
            "mot_active_experts": (
                torch.stack(path_mot_active_experts).mean().detach()
                if path_mot_active_experts
                else torch.tensor(0.0, device=input_ids.device).detach()
            ),
            "race_lane_limit": torch.tensor(float(lane_limit), device=input_ids.device).detach(),
            "race_scout_count": torch.tensor(float(scout_count), device=input_ids.device).detach(),
            "race_lane_band": lane_band.detach(),
            "race_lane_temperature": lane_temperature.detach(),
            "race_scout_score": scout_score.detach(),
            "registry_active_families": torch.tensor(
                float((self.registry.family_active_mask > 0).sum().item()),
                device=input_ids.device,
            ).detach(),
            "registry_active_relations": torch.tensor(
                float((self.registry.relation_active_mask > 0).sum().item()),
                device=input_ids.device,
            ).detach(),
            "recursive_aux_loss": avg_recursive_aux_loss.detach(),
        }
        if router_aug_stats:
            route_stats.update(router_aug_stats)
            for key, value in router_aug_stats.items():
                if key.startswith("sharc_router_"):
                    continue
                route_stats[f"sharc_router_{key}"] = value.detach() if torch.is_tensor(value) else value
            route_stats["sharc_router_enabled"] = torch.tensor(1.0, device=input_ids.device)
        else:
            route_stats["sharc_router_enabled"] = torch.tensor(0.0, device=input_ids.device)
        learned_residency_stats, learned_residency_loss = self._learned_residency_outputs(final_hidden, route_stats)
        route_stats.update(learned_residency_stats)
        residency_controller = self._active_residency_controller()
        if residency_controller is not None:
            gate_stats = {key: value for key, value in gate_runtime.items() if torch.is_tensor(value) and not key.startswith("_")}
            route_stats.update(gate_stats)
            plan = gate_runtime.get("plan")
            route_stats.update(
                residency_controller.record(
                    route_stats,
                    plan=plan if isinstance(plan, GateResidencyPlan) else None,
                )
            )
        contrastive_routing_stats, contrastive_routing_loss = self._contrastive_routing_outputs(
            final_hidden,
            input_signature,
            final_output_signature,
            path_signatures=path_signatures,
            path_emitter_top_weights=path_emitter_top_weights,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            gate_runtime=gate_runtime,
        )
        route_stats.update(contrastive_routing_stats)
        route_stats, route_repairs = self._sanitize_route_stats(route_stats, device=input_ids.device)
        for key, values in recursive_stat_terms.items():
            if values and key not in route_stats:
                route_stats[key] = torch.stack([v.reshape(()) for v in values]).mean().detach()
        route_stats, recursive_repairs = self._sanitize_route_stats(route_stats, device=input_ids.device)
        route_repairs += recursive_repairs
        final_logits, logits_repairs = self._sanitize_tensor(final_logits, fallback=torch.zeros_like(final_logits))
        final_output_signature, signature_repairs = self._sanitize_tensor(
            final_output_signature,
            fallback=torch.zeros_like(final_output_signature),
        )
        path_logits_tensor, path_repairs = self._sanitize_tensor(path_logits_tensor, fallback=torch.zeros_like(path_logits_tensor))
        route_stats["stability_nonfinite_repair_count"] = torch.tensor(
            float(route_repairs + logits_repairs + signature_repairs + path_repairs),
            device=input_ids.device,
        )
        route_stats["stability_finite_guard_enabled"] = torch.tensor(1.0, device=input_ids.device)
        if profile_enabled:
            route_stats.update({key: torch.tensor(value, device=input_ids.device).detach() for key, value in timings.items()})

        aux_loss = (
            self.cfg.signature_loss_weight * signature_loss
            + getattr(self.cfg, "signature_level_loss_weight", self.cfg.signature_loss_weight) * signature_level_loss
            + getattr(self.cfg, "signature_relation_loss_weight", self.cfg.signature_loss_weight) * signature_relation_loss
            + self.cfg.signature_contrastive_weight * contrastive_loss
            + self.contrastive_routing_weight * contrastive_routing_loss
            - self.cfg.routing_entropy_weight * avg_entropy
            + float(getattr(self.cfg, "emitter_balance_weight", 0.0))
            * route_stats.get("emitter_balance_loss", torch.tensor(0.0, device=input_ids.device))
            + float(getattr(self.cfg, "emitter_mixture_weight", 0.0))
            * avg_mixture_loss
            + float(getattr(self.cfg, "torus_active_balance_weight", 0.0))
            * avg_coverage_loss
            - self.cfg.diversity_weight * pairwise_diversity
            + self.learned_residency_weight * learned_residency_loss
        )

        return PrismalWaveOutput(
            logits=final_logits,
            input_signature=input_signature,
            output_signature=final_output_signature,
            path_logits=path_logits_tensor,
            route_stats=route_stats,
            ce_loss=torch.tensor(0.0, device=input_ids.device).detach(),
            aux_loss=aux_loss,
            slot_state=final_slot_state,
            signature_lattice_state=final_signature_lattice_state,
            token_memory_state=frame.token_memory_state,
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def resize_vocab(self, new_vocab_size: int) -> None:
        new_vocab_size = int(new_vocab_size)
        if new_vocab_size <= self._vocab_size:
            return
        device = next(self.parameters()).device

        if isinstance(self.token_embedding, FactorizedEmbedding):
            self.token_embedding.resize_vocab(new_vocab_size)
        else:
            old_embed = self.token_embedding
            new_embed = create_quantized_embedding(
                new_vocab_size,
                old_embed.embedding_dim,
                quantization_config=self.quantization_config,
            ).to(device)
            with torch.no_grad():
                dense_weight = new_embed.weight.detach()
                dense_weight[: old_embed.num_embeddings].copy_(old_embed.weight.detach())
                nn.init.normal_(dense_weight[old_embed.num_embeddings :], mean=0.0, std=0.02)
                _assign_dense_weight(new_embed, dense_weight)
            self.token_embedding = new_embed

        old_head = self.lm_head
        new_head = create_quantized_linear(
            self.cfg.d_model,
            new_vocab_size,
            bias=False,
            quantization_config=self.quantization_config,
        ).to(device)
        with torch.no_grad():
            rows = min(old_head.weight.shape[0], new_vocab_size)
            dense_weight = new_head.weight.detach()
            dense_weight[:rows].copy_(old_head.weight[:rows].detach())
            if new_vocab_size > rows:
                nn.init.normal_(dense_weight[rows:], mean=0.0, std=0.02)
            _assign_dense_weight(new_head, dense_weight)
        self.construction_head = new_head
        if not self.cfg.use_factorized_embedding and not self.use_turbo_quantization:
            self.lm_head.weight = self.token_embedding.weight
        self._vocab_size = new_vocab_size
        self.cfg.vocab_size = new_vocab_size

    def resize_signature_bucket_vocab(self, new_bucket_vocab_size: int) -> None:
        new_bucket_vocab_size = int(new_bucket_vocab_size)
        if new_bucket_vocab_size <= self.signature_bucket_vocab_size:
            return
        device = next(self.parameters()).device

        old_head = self.signature_token_head
        new_head = create_quantized_linear(
            self.cfg.d_model,
            new_bucket_vocab_size,
            bias=True,
            quantization_config=self.quantization_config,
        ).to(device)
        with torch.no_grad():
            rows = min(old_head.weight.shape[0], new_bucket_vocab_size)
            dense_weight = new_head.weight.detach()
            dense_weight[:rows].copy_(old_head.weight[:rows].detach())
            if new_bucket_vocab_size > rows:
                nn.init.normal_(dense_weight[rows:], mean=0.0, std=0.02)
            if old_head.bias is not None and new_head.bias is not None:
                dense_bias = new_head.bias.detach()
                dense_bias[:rows].copy_(old_head.bias[:rows].detach())
                if new_bucket_vocab_size > rows:
                    nn.init.zeros_(dense_bias[rows:])
            _assign_dense_weight(new_head, dense_weight)
        self.signature_token_head = new_head

        old_embedding = self.signature_neighborhood_embedding
        new_embedding = create_quantized_embedding(
            new_bucket_vocab_size,
            self.cfg.d_model,
            quantization_config=self.quantization_config,
        ).to(device)
        with torch.no_grad():
            dense_weight = getattr(new_embedding, "weight", getattr(new_embedding, "embedding_matrix", None))
            old_weight = getattr(old_embedding, "weight", getattr(old_embedding, "embedding_matrix", None))
            if dense_weight is None or old_weight is None:
                raise AttributeError("Unsupported signature neighborhood embedding module for resize.")
            dense_weight[: old_embedding.num_embeddings].copy_(old_weight)
            if new_bucket_vocab_size > old_embedding.num_embeddings:
                nn.init.normal_(dense_weight[old_embedding.num_embeddings :], mean=0.0, std=0.02)
        self.signature_neighborhood_embedding = new_embedding
        self.signature_bucket_vocab_size = new_bucket_vocab_size
        self.cfg.signature_bucket_vocab_size = new_bucket_vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        slot_state: Optional[torch.Tensor] = None,
        signature_lattice_state: Optional[SignatureLatticeState] = None,
        token_memory_state: Optional[TokenMemoryState] = None,
        path_index: Optional[int] = None,
        position_index: int = 0,
        torus_center: Optional[torch.Tensor] = None,
        superposition_token_groups: Optional[torch.Tensor] = None,
        superposition_bag_size: int = 1,
    ) -> PrismalWaveOutput:
        _validate_aligned_signature_tensors(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            loss_mask=loss_mask,
            context="forward",
        )
        gate_runtime = self._prepare_gate_runtime(
            input_ids=input_ids,
            signature_family_ids=signature_family_ids,
            path_index=path_index,
            position_index=position_index,
        )
        if self.use_torus_core and self.torus_core is not None:
            return self._forward_torus(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                loss_mask=loss_mask,
                slot_state=slot_state,
                signature_lattice_state=signature_lattice_state,
                token_memory_state=token_memory_state,
                path_index=path_index,
                position_offset=position_index,
                gate_runtime=gate_runtime,
            )
        hidden = self._encode(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            position_offset=position_index,
            superposition_token_groups=superposition_token_groups,
            superposition_bag_size=superposition_bag_size,
        )
        avg_recursive_aux_loss = torch.tensor(0.0, device=input_ids.device)
        input_signature = self.router.signature(hidden)
        path_vectors = self._path_vectors(hidden.device)
        path_hidden: List[torch.Tensor] = []
        path_logits: List[torch.Tensor] = []
        path_signatures: List[torch.Tensor] = []
        path_scores: List[torch.Tensor] = []
        path_entropy: List[torch.Tensor] = []
        path_active_emitters: List[torch.Tensor] = []
        path_slot_states: List[torch.Tensor] = []
        path_topk_entropy: List[torch.Tensor] = []
        path_topk_effective_count: List[torch.Tensor] = []
        path_soft_occupancy: List[torch.Tensor] = []
        path_soft_breadth: List[torch.Tensor] = []
        path_cell_energy_mean: List[torch.Tensor] = []
        path_cell_energy_min: List[torch.Tensor] = []
        path_cell_energy_max: List[torch.Tensor] = []
        path_usage_entropy: List[torch.Tensor] = []
        path_usage_concentration: List[torch.Tensor] = []
        path_mixture_loss: List[torch.Tensor] = []
        path_coverage_loss: List[torch.Tensor] = []
        path_balance_loss: List[torch.Tensor] = []
        path_emitter_top_weights: List[torch.Tensor] = []
        gate_emitter_bank_override = gate_runtime.get("_gate_emitter_bank_override")
        gate_operator_bank_override = gate_runtime.get("_gate_operator_bank_override")

        path_indices = [path_index] if path_index is not None else list(range(self.cfg.n_paths))
        for current_path_index in path_indices:
            path_hidden_state = hidden
            slots = self.router.init_slots(hidden.size(0), hidden.device, slot_state=slot_state)
            entropy_terms: List[torch.Tensor] = []
            active_terms: List[torch.Tensor] = []
            for layer_index, block in enumerate(self.blocks):
                if self.token_hierarchy is not None:
                    path_hidden_state = path_hidden_state + 0.20 * self.token_hierarchy.route_tokens(
                        path_hidden_state,
                        path_index=current_path_index,
                        layer_index=layer_index,
                        signature_level_ids=signature_level_ids,
                        torus_center=torus_center if torus_center is not None else input_signature,
                    )
                path_hidden_state, slots, stats = block(
                    path_hidden_state,
                    slots,
                    self.router,
                    signature_family_ids=signature_family_ids,
                    signature_ids=signature_ids,
                    signature_level_ids=signature_level_ids,
                    signature_relation_ids=signature_relation_ids,
                    parent_signature_ids=parent_signature_ids,
                    emitter_bank_override=gate_emitter_bank_override if torch.is_tensor(gate_emitter_bank_override) else None,
                    operator_hierarchy_bank_override=gate_operator_bank_override if torch.is_tensor(gate_operator_bank_override) else None,
                    path_index=current_path_index,
                    layer_index=layer_index,
                    torus_center=torus_center if torus_center is not None else input_signature,
                )
                entropy_terms.append(stats["emitter_entropy"])
                active_terms.append(stats["active_emitters"].float())
                if "emitter_topk_entropy" in stats:
                    path_topk_entropy.append(stats["emitter_topk_entropy"])
                if "emitter_topk_effective_count" in stats:
                    path_topk_effective_count.append(stats["emitter_topk_effective_count"].float())
                elif "emitter_cell_effective_count" in stats:
                    path_topk_effective_count.append(stats["emitter_cell_effective_count"].float())
                if "emitter_cell_soft_occupancy" in stats:
                    path_soft_occupancy.append(stats["emitter_cell_soft_occupancy"].float())
                if "emitter_cell_soft_breadth" in stats:
                    path_soft_breadth.append(stats["emitter_cell_soft_breadth"].float())
                if "cell_energy_mean" in stats:
                    path_cell_energy_mean.append(stats["cell_energy_mean"].float())
                if "cell_energy_min" in stats:
                    path_cell_energy_min.append(stats["cell_energy_min"].float())
                if "cell_energy_max" in stats:
                    path_cell_energy_max.append(stats["cell_energy_max"].float())
                if "emitter_usage_entropy" in stats:
                    path_usage_entropy.append(stats["emitter_usage_entropy"].float())
                if "emitter_usage_concentration" in stats:
                    path_usage_concentration.append(stats["emitter_usage_concentration"].float())
                if "emitter_top_weights" in stats:
                    path_emitter_top_weights.append(stats["emitter_top_weights"].float())
                if "emitter_mixture_loss" in stats:
                    path_mixture_loss.append(stats["emitter_mixture_loss"].float())
                elif "emitter_cell_mixture_loss" in stats:
                    path_mixture_loss.append(stats["emitter_cell_mixture_loss"].float())
            if "emitter_balance_loss" in stats:
                path_balance_loss.append(stats["emitter_balance_loss"].float())
            path_hidden_state = self.final_norm(path_hidden_state)
            logits, path_hidden_state, neighborhood_logits = self._construction_logits_from_hidden(path_hidden_state)
            output_signature = F.normalize(self.signature_head(path_hidden_state.mean(dim=1)), dim=-1)
            agreement = F.cosine_similarity(input_signature, output_signature, dim=-1)
            route_entropy = torch.stack(entropy_terms).mean()
            validator_score = agreement - 0.15 * route_entropy
            path_hidden.append(path_hidden_state)
            path_logits.append(logits)
            path_signatures.append(output_signature)
            path_scores.append(validator_score)
            path_entropy.append(route_entropy)
            path_active_emitters.append(torch.stack(active_terms).mean())
            path_slot_states.append(slots)

        path_hidden_tensor = torch.stack(path_hidden, dim=1)
        path_logits_tensor = torch.stack(path_logits, dim=1)
        path_scores_tensor = torch.stack(path_scores, dim=1)
        path_scores_tensor, _ = self._sanitize_tensor(path_scores_tensor, fallback=torch.zeros_like(path_scores_tensor))
        if path_index is None:
            path_weights = F.softmax(path_scores_tensor / max(self.cfg.validator_temperature, 1e-3), dim=-1)
            final_hidden = torch.einsum("bp,bptd->btd", path_weights, path_hidden_tensor)
            final_hidden = self.final_norm(final_hidden)
            selected_path_index = path_scores_tensor.argmax(dim=-1)
        else:
            path_weights = torch.ones(path_scores_tensor.size(0), 1, device=input_ids.device)
            final_hidden = self.final_norm(path_hidden_tensor[:, 0])
            selected_path_index = torch.full((input_ids.size(0),), int(path_index), device=input_ids.device, dtype=torch.long)
        final_logits, final_hidden, final_neighborhood_logits = self._construction_logits_from_hidden(final_hidden)

        final_output_signature = F.normalize(self.signature_head(final_hidden.mean(dim=1)), dim=-1)
        signature_cosine = F.cosine_similarity(input_signature, final_output_signature, dim=-1)

        signature_loss = torch.tensor(0.0, device=input_ids.device)
        signature_accuracy = torch.tensor(0.0, device=input_ids.device)
        if signature_ids is not None and signature_ids.size(1) > 1:
            signature_logits = final_neighborhood_logits[:, :-1, :]
            signature_targets = self._signature_family_targets(signature_ids[:, 1 : 1 + signature_logits.size(1)])
            valid_signature = signature_targets.ne(0)
            if loss_mask is not None:
                valid_signature = valid_signature & loss_mask[:, : signature_logits.size(1)].to(device=input_ids.device).bool()
            if valid_signature.any():
                signature_token_losses = F.cross_entropy(
                    signature_logits.reshape(-1, signature_logits.size(-1)),
                    signature_targets.reshape(-1),
                    reduction="none",
                ).view_as(signature_targets)
                signature_loss = signature_token_losses.masked_select(valid_signature).mean()
                signature_predictions = signature_logits.argmax(dim=-1)
                signature_accuracy = (
                    signature_predictions.eq(signature_targets).masked_select(valid_signature).float().mean()
                )
        signature_level_loss = torch.tensor(0.0, device=input_ids.device)
        signature_level_accuracy = torch.tensor(0.0, device=input_ids.device)
        if signature_level_ids is not None and signature_level_ids.size(1) > 1:
            level_logits = self.signature_level_head(final_hidden[:, :-1, :])
            level_targets = signature_level_ids[:, 1 : 1 + level_logits.size(1)]
            valid_level = level_targets.ne(0)
            if loss_mask is not None:
                valid_level = valid_level & loss_mask[:, : level_logits.size(1)].to(device=input_ids.device).bool()
            if valid_level.any():
                level_token_losses = F.cross_entropy(
                    level_logits.reshape(-1, level_logits.size(-1)),
                    level_targets.reshape(-1),
                    reduction="none",
                ).view_as(level_targets)
                signature_level_loss = level_token_losses.masked_select(valid_level).mean()
                level_predictions = level_logits.argmax(dim=-1)
                signature_level_accuracy = (
                    level_predictions.eq(level_targets).masked_select(valid_level).float().mean()
                )
        signature_relation_loss = torch.tensor(0.0, device=input_ids.device)
        signature_relation_accuracy = torch.tensor(0.0, device=input_ids.device)
        if signature_relation_ids is not None and signature_relation_ids.size(1) > 1:
            relation_logits = self.signature_relation_head(final_hidden[:, :-1, :])
            relation_targets = signature_relation_ids[:, 1 : 1 + relation_logits.size(1)]
            valid_relation = relation_targets.ne(0)
            if loss_mask is not None:
                valid_relation = valid_relation & loss_mask[:, : relation_logits.size(1)].to(device=input_ids.device).bool()
            if valid_relation.any():
                relation_token_losses = F.cross_entropy(
                    relation_logits.reshape(-1, relation_logits.size(-1)),
                    relation_targets.reshape(-1),
                    reduction="none",
                ).view_as(relation_targets)
                signature_relation_loss = relation_token_losses.masked_select(valid_relation).mean()
                relation_predictions = relation_logits.argmax(dim=-1)
                signature_relation_accuracy = (
                    relation_predictions.eq(relation_targets).masked_select(valid_relation).float().mean()
                )
        signature_agreement = signature_cosine
        if signature_ids is not None and signature_ids.size(1) > 1:
            signature_agreement = torch.stack(
                [signature_accuracy, signature_level_accuracy, signature_relation_accuracy]
            ).mean()

        contrastive_loss = torch.tensor(0.0, device=input_ids.device)
        if final_output_signature.size(0) > 1:
            contrastive_logits = torch.matmul(
                F.normalize(input_signature, dim=-1),
                F.normalize(final_output_signature, dim=-1).transpose(0, 1),
            ) / max(self.cfg.signature_temperature, 1e-3)
            contrastive_targets = torch.arange(final_output_signature.size(0), device=input_ids.device)
            contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_targets)

        avg_entropy = torch.stack(path_entropy).mean()
        avg_active = torch.stack(path_active_emitters).mean()
        avg_soft_active = torch.stack(path_soft_occupancy).mean() if path_soft_occupancy else torch.tensor(0.0, device=input_ids.device)
        avg_soft_breadth = torch.stack(path_soft_breadth).mean() if path_soft_breadth else torch.tensor(0.0, device=input_ids.device)
        avg_cell_energy_mean = torch.stack(path_cell_energy_mean).mean() if path_cell_energy_mean else torch.tensor(0.0, device=input_ids.device)
        avg_cell_energy_min = torch.stack(path_cell_energy_min).min() if path_cell_energy_min else torch.tensor(0.0, device=input_ids.device)
        avg_cell_energy_max = torch.stack(path_cell_energy_max).max() if path_cell_energy_max else torch.tensor(0.0, device=input_ids.device)
        avg_usage_entropy = torch.stack(path_usage_entropy).mean() if path_usage_entropy else torch.tensor(0.0, device=input_ids.device)
        avg_usage_concentration = torch.stack(path_usage_concentration).mean() if path_usage_concentration else torch.tensor(0.0, device=input_ids.device)
        avg_mixture_loss = torch.stack(path_mixture_loss).mean() if path_mixture_loss else torch.tensor(0.0, device=input_ids.device)
        avg_balance_loss = torch.stack(path_balance_loss).mean() if path_balance_loss else torch.tensor(0.0, device=input_ids.device)
        route_denominator = max(float(getattr(self.cfg, "n_emitters", 1)), 1.0)
        if path_index is None:
            final_slot_state = torch.einsum(
                "bp,bpsd->bsd",
                path_weights,
                torch.stack(path_slot_states, dim=1),
            )
        else:
            final_slot_state = path_slot_states[0]

        if torus_center is not None:
            center = torus_center.unsqueeze(1)
            final_slot_state = self.cfg.memory_momentum * final_slot_state + (1.0 - self.cfg.memory_momentum) * (
                final_slot_state + self.cfg.resonance_pull_weight * (center - final_slot_state)
            )

        pairwise_diversity = torch.tensor(0.0, device=input_ids.device)
        if len(path_signatures) > 1:
            div_terms: List[torch.Tensor] = []
            for i in range(len(path_signatures)):
                for j in range(i + 1, len(path_signatures)):
                    div_terms.append(1.0 - F.cosine_similarity(path_signatures[i], path_signatures[j], dim=-1).mean())
            if div_terms:
                pairwise_diversity = torch.stack(div_terms).mean()

        route_stats = {
            "path_weights": path_weights.detach(),
            "path_scores": path_scores_tensor.detach(),
            "selected_path_index": selected_path_index.detach(),
            "path_entropy": torch.stack(path_entropy).detach(),
            "path_active_emitters": torch.stack(path_active_emitters).detach(),
            "signature_agreement": signature_agreement.detach(),
            "signature_accuracy": signature_accuracy.detach(),
            "signature_level_accuracy": signature_level_accuracy.detach(),
            "signature_relation_accuracy": signature_relation_accuracy.detach(),
            "signature_cosine": signature_cosine.detach(),
            "signature_loss": signature_loss.detach(),
            "signature_neighborhood_logits": final_neighborhood_logits.detach(),
            "signature_neighborhood_confidence": F.softmax(
                self._sanitize_tensor(final_neighborhood_logits.detach(), fallback=torch.zeros_like(final_neighborhood_logits))[0],
                dim=-1,
            ).amax(dim=-1).mean(),
            "signature_level_loss": signature_level_loss.detach(),
            "signature_relation_loss": signature_relation_loss.detach(),
            "signature_contrastive_loss": contrastive_loss.detach(),
            "pairwise_diversity": pairwise_diversity.detach(),
            "avg_entropy": avg_entropy.detach(),
            "avg_active_emitters": avg_active.detach(),
            "emitter_cell_occupancy": avg_active.detach(),
            "avg_emitter_cell_soft_occupancy": avg_soft_active.detach(),
            "emitter_cell_soft_occupancy": avg_soft_active.detach(),
            "emitter_cell_breadth": (
                avg_active / route_denominator
            ).detach(),
            "avg_emitter_cell_soft_breadth": avg_soft_breadth.detach(),
            "emitter_cell_soft_breadth": avg_soft_breadth.detach(),
            "avg_emitter_usage_entropy": avg_usage_entropy.detach(),
            "avg_emitter_usage_concentration": avg_usage_concentration.detach(),
            "emitter_usage_entropy": avg_usage_entropy.detach(),
            "emitter_usage_concentration": avg_usage_concentration.detach(),
            "torus_activity_threshold": torch.tensor(float(self.activity_threshold), device=input_ids.device).detach(),
            "cell_energy_mean": avg_cell_energy_mean.detach(),
            "cell_energy_min": avg_cell_energy_min.detach(),
            "cell_energy_max": avg_cell_energy_max.detach(),
            "emitter_cell_effective_count": torch.stack(path_topk_effective_count).mean().detach() if path_topk_effective_count else torch.tensor(0.0, device=input_ids.device).detach(),
            "emitter_cell_mixture_loss": avg_mixture_loss.detach(),
            "emitter_cell_coverage_loss": torch.stack(path_coverage_loss).mean().detach() if path_coverage_loss else torch.tensor(0.0, device=input_ids.device).detach(),
            "avg_emitter_topk_entropy": torch.stack(path_topk_entropy).mean().detach() if path_topk_entropy else torch.tensor(0.0, device=input_ids.device).detach(),
            "avg_emitter_topk_effective_count": torch.stack(path_topk_effective_count).mean().detach() if path_topk_effective_count else torch.tensor(0.0, device=input_ids.device).detach(),
            "emitter_mixture_loss": avg_mixture_loss.detach(),
            "emitter_balance_loss": avg_balance_loss.detach(),
            "mot_active_experts": torch.tensor(0.0, device=input_ids.device).detach(),
            "signature_lattice_cache_norm": torch.tensor(0.0, device=input_ids.device).detach(),
            "signature_lattice_gate_mean": torch.tensor(0.0, device=input_ids.device).detach(),
            "signature_lattice_candidate_count": torch.tensor(0.0, device=input_ids.device).detach(),
            "signature_lattice_enabled": torch.tensor(0.0, device=input_ids.device).detach(),
            "registry_active_families": torch.tensor(
                float((self.registry.family_active_mask > 0).sum().item()),
                device=input_ids.device,
            ).detach(),
            "registry_active_relations": torch.tensor(
                float((self.registry.relation_active_mask > 0).sum().item()),
                device=input_ids.device,
            ).detach(),
        }
        learned_residency_stats, learned_residency_loss = self._learned_residency_outputs(final_hidden, route_stats)
        route_stats.update(learned_residency_stats)
        residency_controller = self._active_residency_controller()
        if residency_controller is not None:
            gate_stats = {key: value for key, value in gate_runtime.items() if torch.is_tensor(value) and not key.startswith("_")}
            route_stats.update(gate_stats)
            plan = gate_runtime.get("plan")
            route_stats.update(
                residency_controller.record(
                    route_stats,
                    plan=plan if isinstance(plan, GateResidencyPlan) else None,
                )
            )
        contrastive_routing_stats, contrastive_routing_loss = self._contrastive_routing_outputs(
            final_hidden,
            input_signature,
            final_output_signature,
            path_signatures=path_signatures,
            path_emitter_top_weights=path_emitter_top_weights,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            gate_runtime=gate_runtime,
        )
        route_stats.update(contrastive_routing_stats)
        route_stats, route_repairs = self._sanitize_route_stats(route_stats, device=input_ids.device)

        aux_loss = (
            self.cfg.signature_loss_weight * signature_loss
            + getattr(self.cfg, "signature_level_loss_weight", self.cfg.signature_loss_weight) * signature_level_loss
            + getattr(self.cfg, "signature_relation_loss_weight", self.cfg.signature_loss_weight) * signature_relation_loss
            + self.cfg.signature_contrastive_weight * contrastive_loss
            + self.contrastive_routing_weight * contrastive_routing_loss
            - self.cfg.routing_entropy_weight * avg_entropy
            + float(getattr(self.cfg, "emitter_balance_weight", 0.0))
            * avg_balance_loss
            + float(getattr(self.cfg, "emitter_mixture_weight", 0.0))
            * avg_mixture_loss
            + float(getattr(self.cfg, "torus_active_balance_weight", 0.0))
            * route_stats.get("torus_coverage_loss", torch.tensor(0.0, device=input_ids.device))
            + avg_recursive_aux_loss
            - self.cfg.diversity_weight * pairwise_diversity
            + self.learned_residency_weight * learned_residency_loss
        )
        final_logits, logits_repairs = self._sanitize_tensor(final_logits, fallback=torch.zeros_like(final_logits))
        final_output_signature, signature_repairs = self._sanitize_tensor(
            final_output_signature,
            fallback=torch.zeros_like(final_output_signature),
        )
        path_logits_tensor, path_repairs = self._sanitize_tensor(path_logits_tensor, fallback=torch.zeros_like(path_logits_tensor))
        route_stats["stability_nonfinite_repair_count"] = torch.tensor(
            float(route_repairs + logits_repairs + signature_repairs + path_repairs),
            device=input_ids.device,
        )
        route_stats["stability_finite_guard_enabled"] = torch.tensor(1.0, device=input_ids.device)

        return PrismalWaveOutput(
            logits=final_logits,
            input_signature=input_signature,
            output_signature=final_output_signature,
            path_logits=path_logits_tensor,
            route_stats=route_stats,
            ce_loss=torch.tensor(0.0, device=input_ids.device).detach(),
            aux_loss=aux_loss,
            slot_state=final_slot_state,
            signature_lattice_state=None,
        )

    def forward_incremental(
        self,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        slot_state: Optional[torch.Tensor] = None,
        signature_lattice_state: Optional[SignatureLatticeState] = None,
        token_memory_state: Optional[TokenMemoryState] = None,
        path_index: Optional[int] = None,
        position_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, PrismalWaveOutput]:
        output = self.forward(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            slot_state=slot_state,
            signature_lattice_state=signature_lattice_state,
            token_memory_state=token_memory_state,
            path_index=path_index,
            position_index=position_index,
        )
        logits = output.logits[:, -1, :]
        next_slot_state = output.slot_state if output.slot_state is not None else slot_state
        if next_slot_state is None:
            if self.use_torus_core and self.torus_core is not None:
                next_slot_state = self.torus_core.init_state(input_ids.size(0), input_ids.device)
            else:
                next_slot_state = self.router.init_slots(input_ids.size(0), input_ids.device)
        return logits, next_slot_state, output

    def _apply_generation_safety_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Suppress universal non-output ids; codec-specific masks are passed explicitly."""
        blocked = logits.clone()
        blocked[:, self.cfg.pad_id] = float("-inf")
        blocked[:, self.cfg.bos_id] = float("-inf")
        return blocked

    def _apply_explicit_generation_mask(
        self,
        logits: torch.Tensor,
        suppressed_token_ids: Optional[Sequence[int]],
    ) -> torch.Tensor:
        if not suppressed_token_ids:
            return logits
        blocked = logits.clone()
        valid_ids = [int(token_id) for token_id in suppressed_token_ids if 0 <= int(token_id) < blocked.size(-1)]
        if valid_ids:
            blocked[:, valid_ids] = float("-inf")
        return blocked

    def _apply_signature_neighborhood_generation_bias(
        self,
        logits: torch.Tensor,
        output: PrismalWaveOutput,
        token_family_lookup: Optional[torch.Tensor],
        *,
        strength: float = 0.35,
        min_confidence: float = 0.55,
    ) -> torch.Tensor:
        if token_family_lookup is None:
            return logits
        neighborhood_logits = output.route_stats.get("signature_neighborhood_logits")
        if neighborhood_logits is None or neighborhood_logits.dim() < 3:
            return logits
        confidence = output.route_stats.get("signature_neighborhood_confidence")
        if confidence is None:
            return logits
        confidence_value = float(confidence.float().mean().item())
        if confidence_value < float(min_confidence):
            return logits
        bucket_vocab = max(1, int(self.signature_bucket_vocab_size))
        predicted = neighborhood_logits[:, -1, :].argmax(dim=-1)
        confidence_scale = max(0.0, min(1.0, (confidence_value - float(min_confidence)) / max(1.0 - float(min_confidence), 1e-6)))
        effective_strength = float(strength) * confidence_scale
        if effective_strength <= 0.0:
            return logits
        family_lookup = token_family_lookup.to(device=logits.device, dtype=torch.long)
        if family_lookup.numel() == 0:
            return logits
        family_lookup = family_lookup[: logits.size(-1)]
        if family_lookup.numel() == 0:
            return logits
        biased = logits.clone()
        family_targets = family_lookup.remainder(bucket_vocab).unsqueeze(0)
        target_family = predicted.remainder(bucket_vocab).unsqueeze(1)
        matches = family_targets.eq(target_family)
        biased = biased + effective_strength * matches.to(dtype=biased.dtype)
        return biased

    def _lookup_tensor_from_map(
        self,
        lookup: Optional[Dict[int, int]],
        *,
        device: torch.device,
        default_value: int,
        size_hint: int = 0,
    ) -> Optional[torch.Tensor]:
        if not lookup:
            return None
        max_key = max((int(key) for key in lookup.keys()), default=-1)
        size = max(int(size_hint), max_key + 1)
        if size <= 0:
            return None
        tensor = torch.full((size,), int(default_value), device=device, dtype=torch.long)
        for key, value in lookup.items():
            token_id = int(key)
            if token_id < 0:
                continue
            if token_id >= tensor.size(0):
                expanded = torch.full((token_id + 1,), int(default_value), device=device, dtype=torch.long)
                expanded[: tensor.size(0)] = tensor
                tensor = expanded
            tensor[token_id] = int(value)
        return tensor

    def _apply_construction_collapse_mask(self, logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if history.size(1) < 3:
            return logits
        blocked = logits.clone()
        for batch_idx in range(history.size(0)):
            tokens = history[batch_idx].tolist()
            tail = [int(tok) for tok in tokens[-6:] if int(tok) not in {self.cfg.pad_id, self.cfg.bos_id}]
            if len(tail) >= 3 and tail[-1] == tail[-2] == tail[-3]:
                blocked[batch_idx, tail[-1]] = float("-inf")
            if len(tail) >= 6 and tail[-2:] == tail[-4:-2] == tail[-6:-4]:
                blocked[batch_idx, tail[-1]] = float("-inf")
        return blocked

    @torch.no_grad()
    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        slot_state: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        min_new_tokens: int = 24,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.08,
        no_repeat_ngram_size: int = 4,
        beam_size: int = 2,
        token_signature_lookup: Optional[Dict[int, int]] = None,
        token_family_lookup: Optional[Dict[int, int]] = None,
        token_level_lookup: Optional[Dict[int, int]] = None,
        token_relation_lookup: Optional[Dict[int, int]] = None,
        suppressed_token_ids: Optional[Sequence[int]] = None,
        recommit_interval: int = 0,
        recommit_signature_threshold: float = 0.65,
        signature_lattice_state: Optional[SignatureLatticeState] = None,
        token_memory_state: Optional[TokenMemoryState] = None,
    ) -> torch.Tensor:
        _validate_aligned_signature_tensors(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            context="generate",
        )
        beam_size = max(1, int(beam_size))
        token_signature_lookup_tensor = self._lookup_tensor_from_map(
            token_signature_lookup,
            device=input_ids.device,
            default_value=7,
            size_hint=self.vocab_size,
        )
        token_family_lookup_tensor = self._lookup_tensor_from_map(
            token_family_lookup,
            device=input_ids.device,
            default_value=0,
            size_hint=self.vocab_size,
        )
        token_level_lookup_tensor = self._lookup_tensor_from_map(
            token_level_lookup,
            device=input_ids.device,
            default_value=self.signature_level_to_id["char"],
            size_hint=self.vocab_size,
        )
        token_relation_lookup_tensor = self._lookup_tensor_from_map(
            token_relation_lookup,
            device=input_ids.device,
            default_value=self.signature_relation_to_id["continuation"],
            size_hint=self.vocab_size,
        )

        def _lookup_next_tokens(lookup_tensor: Optional[torch.Tensor], token_ids: torch.Tensor, default_value: int) -> torch.Tensor:
            if lookup_tensor is None:
                return torch.full_like(token_ids, fill_value=default_value)
            values = lookup_tensor[token_ids.squeeze(-1).long()]
            return values.to(device=token_ids.device, dtype=token_ids.dtype).unsqueeze(-1)

        def apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            if repetition_penalty <= 1.0:
                return logits
            adjusted = logits.clone()
            for batch_idx in range(adjusted.size(0)):
                unique_tokens = torch.unique(history[batch_idx])
                for token_id in unique_tokens.tolist():
                    if token_id in {self.cfg.pad_id, self.cfg.bos_id}:
                        continue
                    value = adjusted[batch_idx, token_id]
                    adjusted[batch_idx, token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
            return adjusted

        def apply_no_repeat_ngram(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            if no_repeat_ngram_size <= 1:
                return logits
            seq_len = history.size(1)
            if seq_len < no_repeat_ngram_size - 1:
                return logits
            blocked = logits.clone()
            prefix_len = no_repeat_ngram_size - 1
            for batch_idx in range(history.size(0)):
                tokens = history[batch_idx].tolist()
                if len(tokens) < no_repeat_ngram_size:
                    continue
                ngram_map: Dict[Tuple[int, ...], set[int]] = {}
                for i in range(len(tokens) - no_repeat_ngram_size + 1):
                    prefix = tuple(tokens[i : i + prefix_len])
                    next_token = tokens[i + prefix_len]
                    ngram_map.setdefault(prefix, set()).add(next_token)
                current_prefix = tuple(tokens[-prefix_len:])
                banned = ngram_map.get(current_prefix, set())
                if banned:
                    blocked[batch_idx, list(banned)] = float("-inf")
            return blocked

        def suppress_early_eos(logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            if step_idx + 1 >= min_new_tokens:
                return logits
            blocked = logits.clone()
            blocked[:, self.cfg.eos_id] = float("-inf")
            return blocked

        def step_quality(output: PrismalWaveOutput) -> torch.Tensor:
            return output.route_stats["signature_agreement"].float() - self.cfg.beam_coherence_weight * output.route_stats[
                "avg_entropy"
            ].float()

        def _force_anchor_logits(
            logits: torch.Tensor,
            memory_state: Optional[TokenMemoryState],
        ) -> Tuple[torch.Tensor, bool]:
            anchor_next_ids = self._token_memory_anchor_next_ids(memory_state, device=logits.device)
            if anchor_next_ids is None:
                return logits, False
            active_mask = anchor_next_ids.ge(0).squeeze(-1)
            if not bool(active_mask.any().item()):
                return logits, False
            forced = logits.clone()
            for row_idx in range(forced.size(0)):
                if not bool(active_mask[row_idx].item()):
                    continue
                forced[row_idx] = float("-inf")
                forced[row_idx, int(anchor_next_ids[row_idx, 0].item())] = 0.0
            return forced, True

        generated = input_ids.clone()
        generated_families = signature_family_ids.clone() if signature_family_ids is not None else torch.zeros_like(generated)
        generated_signatures = signature_ids.clone() if signature_ids is not None else torch.zeros_like(generated)
        generated_levels = signature_level_ids.clone() if signature_level_ids is not None else torch.zeros_like(generated)
        generated_relations = signature_relation_ids.clone() if signature_relation_ids is not None else torch.zeros_like(generated)
        generated_parents = parent_signature_ids.clone() if parent_signature_ids is not None else generated_signatures.clone()
        carried_token_memory_state = token_memory_state
        anchor_token_memory_state = token_memory_state
        min_new_tokens = max(0, int(min_new_tokens))
        top_p = float(max(0.0, min(1.0, top_p)))

        probe_output = self.forward(
            generated,
            signature_family_ids=generated_families,
            signature_ids=generated_signatures,
            signature_level_ids=generated_levels,
            signature_relation_ids=generated_relations,
            parent_signature_ids=generated_parents,
            slot_state=slot_state,
            signature_lattice_state=signature_lattice_state,
            token_memory_state=carried_token_memory_state,
            path_index=None,
        )
        selected = probe_output.route_stats.get("selected_path_index")
        committed_path_index = int(selected.view(-1)[0].item()) if selected is not None else 0
        base_slots = probe_output.slot_state
        if base_slots is None:
            base_slots = self.router.init_slots(generated.size(0), generated.device)
        prompt_logits = probe_output.logits[:, -1, :]
        prompt_anchor_state = token_memory_state

        beams = [
            {
                "generated": generated,
                "families": generated_families,
                "signatures": generated_signatures,
                "levels": generated_levels,
                "relations": generated_relations,
                "parents": generated_parents,
                "slots": base_slots,
                "lattice_state": probe_output.signature_lattice_state,
                "memory_state": probe_output.token_memory_state,
                "anchor_state": prompt_anchor_state,
                "score": 0.0,
                "finished": False,
                "prefill_done": False,
                "prompt_logits": prompt_logits,
                "committed_path_index": committed_path_index,
            }
        ]

        relay_interval = max(1, int(recommit_interval if recommit_interval > 0 else getattr(self.cfg, "torus_relay_interval", 16)))
        generation_lap_cap = max(1, int(getattr(self.cfg, "generation_lap_cap", 1)))
        generation_lap_token_cap = max(1, int(getattr(self.cfg, "generation_lap_token_cap", relay_interval)))
        relay_interval = min(relay_interval, generation_lap_token_cap)
        laps_used = 0

        for step_idx in range(max_new_tokens):
            candidates: List[Dict[str, torch.Tensor | float | bool]] = []
            for beam in beams:
                if beam["finished"]:
                    candidates.append(beam)
                    continue
                if not bool(beam.get("prefill_done", False)):
                    logits, anchor_forced = _force_anchor_logits(beam["prompt_logits"], beam.get("anchor_state"))
                    next_slots = beam["slots"]
                    output = probe_output
                    next_committed_path_index = int(beam.get("committed_path_index", committed_path_index))
                    lane_band = self._race_lane_band_from_score(float(output.route_stats.get("signature_agreement", torch.tensor(1.0)).float().mean().item()))
                    lane_temperature = self._race_lane_temperature(next_committed_path_index, step_idx, temperature)
                else:
                    current_committed = int(beam.get("committed_path_index", committed_path_index))
                    if step_idx > 0 and step_idx % relay_interval == 0:
                        if laps_used >= generation_lap_cap - 1:
                            pass
                        else:
                            laps_used += 1
                            output = self.forward(
                                beam["generated"],
                                signature_family_ids=beam["families"],
                                signature_ids=beam["signatures"],
                                signature_level_ids=beam["levels"],
                                signature_relation_ids=beam["relations"],
                                parent_signature_ids=beam["parents"],
                                slot_state=None,
                                path_index=None,
                                token_memory_state=beam.get("memory_state"),
                            )
                            logits = output.logits[:, -1, :]
                            next_slots = output.slot_state if output.slot_state is not None else beam["slots"]
                            selected_path = output.route_stats.get("selected_path_index")
                            if selected_path is not None:
                                next_committed_path_index = int(selected_path.view(-1)[0].item())
                            else:
                                next_committed_path_index = current_committed
                            next_committed_path_index, lane_band, lane_temperature = self._select_race_lane(
                                output,
                                step_idx,
                                fallback_lane=next_committed_path_index,
                            )
                            logits, anchor_forced = _force_anchor_logits(logits, beam.get("anchor_state"))
                            decode_temperature = max(
                                float(temperature),
                                float(lane_temperature) if self.use_race_lanes else float(temperature),
                                1e-3,
                            )
                            logits = logits / decode_temperature
                            logits = apply_repetition_penalty(logits, beam["generated"])
                            logits = apply_no_repeat_ngram(logits, beam["generated"])
                            logits = self._apply_construction_collapse_mask(logits, beam["generated"])
                            logits = suppress_early_eos(logits, step_idx)
                            logits = self._apply_signature_neighborhood_generation_bias(logits, output, token_family_lookup_tensor)
                            logits = self._apply_token_memory_copy_bias(logits, output)
                            logits = self._apply_generation_safety_mask(logits)
                            logits = self._apply_explicit_generation_mask(logits, suppressed_token_ids)
                            if top_k > 0:
                                top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                                filtered = torch.full_like(logits, float("-inf"))
                                filtered.scatter_(1, top_idx, top_vals)
                                logits = filtered
                            if top_p < 1.0:
                                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                                sorted_logits, _ = self._sanitize_sampling_logits(sorted_logits)
                                sorted_probs = F.softmax(sorted_logits, dim=-1)
                                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                                cutoff = cumprobs > top_p
                                cutoff[..., 0] = False
                                sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
                                logits = torch.full_like(logits, float("-inf"))
                                logits.scatter_(1, sorted_idx, sorted_logits)
                            logits, _ = self._sanitize_sampling_logits(logits)
                            log_probs = F.log_softmax(logits, dim=-1)
                            candidate_k = 1 if anchor_forced else min(max(beam_size * 2, top_k if top_k > 0 else beam_size), log_probs.size(-1))
                            top_log_probs, top_token_ids = torch.topk(log_probs, k=candidate_k, dim=-1)
                            quality = float(step_quality(output).mean().item())
                            for idx in range(top_token_ids.size(-1)):
                                next_id = top_token_ids[:, idx : idx + 1]
                                next_logprob = float(top_log_probs[:, idx : idx + 1].sum().item())
                                next_sig = _lookup_next_tokens(token_signature_lookup_tensor, next_id, 7)
                                next_family = _lookup_next_tokens(token_family_lookup_tensor, next_id, 0)
                                next_level = _lookup_next_tokens(
                                    token_level_lookup_tensor,
                                    next_id,
                                    self.signature_level_to_id["char"],
                                )
                                next_relation = _lookup_next_tokens(
                                    token_relation_lookup_tensor,
                                    next_id,
                                    self.signature_relation_to_id["continuation"],
                                )
                                next_parent = next_sig.clone()
                                new_generated = torch.cat([beam["generated"], next_id], dim=-1)
                                new_families = torch.cat([beam["families"], next_family], dim=-1)
                                new_signatures = torch.cat([beam["signatures"], next_sig], dim=-1)
                                new_levels = torch.cat([beam["levels"], next_level], dim=-1)
                                new_relations = torch.cat([beam["relations"], next_relation], dim=-1)
                                new_parents = torch.cat([beam["parents"], next_parent], dim=-1)
                                finished = bool(step_idx + 1 >= min_new_tokens and torch.all(next_id.squeeze(-1) == self.cfg.eos_id))
                                next_memory_state = self._advance_token_memory_anchor_state(beam.get("anchor_state"), next_id)
                                candidates.append(
                                    {
                                        "generated": new_generated,
                                        "families": new_families,
                                        "signatures": new_signatures,
                                        "levels": new_levels,
                                        "relations": new_relations,
                                        "parents": new_parents,
                                        "slots": next_slots,
                                        "lattice_state": output.signature_lattice_state,
                                        "memory_state": next_memory_state if next_memory_state is not None else output.token_memory_state,
                                        "score": float(beam["score"]) + next_logprob + self.cfg.beam_signature_weight * quality,
                                        "finished": finished,
                                        "prefill_done": True,
                                        "prompt_logits": beam["prompt_logits"],
                                        "committed_path_index": next_committed_path_index,
                                    }
                            )
                            continue
                    if step_idx > 0 and step_idx % relay_interval == 0:
                        output = self.forward(
                            beam["generated"],
                            signature_family_ids=beam["families"],
                            signature_ids=beam["signatures"],
                            signature_level_ids=beam["levels"],
                            signature_relation_ids=beam["relations"],
                            parent_signature_ids=beam["parents"],
                            slot_state=None,
                            path_index=None,
                            token_memory_state=beam.get("memory_state"),
                        )
                        logits = output.logits[:, -1, :]
                        next_slots = output.slot_state if output.slot_state is not None else beam["slots"]
                        selected_path = output.route_stats.get("selected_path_index")
                        if selected_path is not None:
                            next_committed_path_index = int(selected_path.view(-1)[0].item())
                        else:
                            next_committed_path_index = current_committed
                        next_committed_path_index, lane_band, lane_temperature = self._select_race_lane(
                            output,
                            step_idx,
                            fallback_lane=next_committed_path_index,
                        )
                        logits, anchor_forced = _force_anchor_logits(logits, beam.get("anchor_state"))
                    else:
                        next_committed_path_index = current_committed
                        output = self.forward(
                            beam["generated"],
                            signature_family_ids=beam["families"],
                            signature_ids=beam["signatures"],
                            signature_level_ids=beam["levels"],
                            signature_relation_ids=beam["relations"],
                            parent_signature_ids=beam["parents"],
                            slot_state=None,
                            path_index=next_committed_path_index,
                        )
                        logits = output.logits[:, -1, :]
                        next_slots = output.slot_state if output.slot_state is not None else beam["slots"]
                        lane_band = self._race_lane_band_from_score(float(output.route_stats["signature_agreement"].float().mean().item()))
                        lane_temperature = self._race_lane_temperature(next_committed_path_index, step_idx, temperature)
                        logits, anchor_forced = _force_anchor_logits(logits, beam.get("anchor_state"))
                decode_temperature = max(
                    float(temperature),
                    float(lane_temperature) if self.use_race_lanes else float(temperature),
                    1e-3,
                )
                logits = logits / decode_temperature
                logits = apply_repetition_penalty(logits, beam["generated"])
                logits = apply_no_repeat_ngram(logits, beam["generated"])
                logits = self._apply_construction_collapse_mask(logits, beam["generated"])
                logits = suppress_early_eos(logits, step_idx)
                logits = self._apply_signature_neighborhood_generation_bias(logits, output, token_family_lookup_tensor)
                logits = self._apply_token_memory_copy_bias(logits, output)
                logits = self._apply_generation_safety_mask(logits)
                logits = self._apply_explicit_generation_mask(logits, suppressed_token_ids)
                if top_k > 0:
                    top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    filtered = torch.full_like(logits, float("-inf"))
                    filtered.scatter_(1, top_idx, top_vals)
                    logits = filtered
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    sorted_logits, _ = self._sanitize_sampling_logits(sorted_logits)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumprobs > top_p
                    cutoff[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, sorted_idx, sorted_logits)
                logits, _ = self._sanitize_sampling_logits(logits)
                log_probs = F.log_softmax(logits, dim=-1)
                candidate_k = 1 if anchor_forced else min(max(beam_size * 2, top_k if top_k > 0 else beam_size), log_probs.size(-1))
                top_log_probs, top_token_ids = torch.topk(log_probs, k=candidate_k, dim=-1)
                quality = float(step_quality(output).mean().item())
                for idx in range(top_token_ids.size(-1)):
                    next_id = top_token_ids[:, idx : idx + 1]
                    next_logprob = float(top_log_probs[:, idx : idx + 1].sum().item())
                    next_sig = _lookup_next_tokens(token_signature_lookup_tensor, next_id, 7)
                    next_family = _lookup_next_tokens(token_family_lookup_tensor, next_id, 0)
                    next_level = _lookup_next_tokens(
                        token_level_lookup_tensor,
                        next_id,
                        self.signature_level_to_id["char"],
                    )
                    next_relation = _lookup_next_tokens(
                        token_relation_lookup_tensor,
                        next_id,
                        self.signature_relation_to_id["continuation"],
                    )
                    next_parent = next_sig.clone()
                    new_generated = torch.cat([beam["generated"], next_id], dim=-1)
                    new_families = torch.cat([beam["families"], next_family], dim=-1)
                    new_signatures = torch.cat([beam["signatures"], next_sig], dim=-1)
                    new_levels = torch.cat([beam["levels"], next_level], dim=-1)
                    new_relations = torch.cat([beam["relations"], next_relation], dim=-1)
                    new_parents = torch.cat([beam["parents"], next_parent], dim=-1)
                    finished = bool(step_idx + 1 >= min_new_tokens and torch.all(next_id.squeeze(-1) == self.cfg.eos_id))
                    next_anchor_state = self._advance_token_memory_anchor_state(beam.get("anchor_state"), next_id)
                    next_memory_state = self._advance_token_memory_anchor_state(beam.get("anchor_state"), next_id)
                    candidates.append(
                        {
                            "generated": new_generated,
                            "families": new_families,
                            "signatures": new_signatures,
                            "levels": new_levels,
                            "relations": new_relations,
                            "parents": new_parents,
                            "slots": next_slots,
                            "lattice_state": output.signature_lattice_state,
                            "memory_state": next_memory_state if next_memory_state is not None else output.token_memory_state,
                            "anchor_state": next_anchor_state if next_anchor_state is not None else beam.get("anchor_state"),
                            "score": float(beam["score"]) + next_logprob + self.cfg.beam_signature_weight * quality,
                            "finished": finished,
                            "prefill_done": True,
                            "prompt_logits": beam["prompt_logits"],
                            "committed_path_index": next_committed_path_index,
                        }
                    )
            candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            beams = candidates[:beam_size]
            if all(bool(beam["finished"]) for beam in beams):
                break

        return beams[0]["generated"]

    @torch.no_grad()
    def _coarse_fine_speculative_generate(
        self,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        slot_state: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        min_new_tokens: int = 24,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.08,
        no_repeat_ngram_size: int = 4,
        token_signature_lookup: Optional[Dict[int, int]] = None,
        token_family_lookup: Optional[Dict[int, int]] = None,
        token_level_lookup: Optional[Dict[int, int]] = None,
        token_relation_lookup: Optional[Dict[int, int]] = None,
        suppressed_token_ids: Optional[Sequence[int]] = None,
        recommit_interval: int = 0,
        recommit_signature_threshold: float = 0.65,
        speculative_draft_tokens: int = 6,
        speculative_temperature: float = 0.0,
        signature_lattice_state: Optional[SignatureLatticeState] = None,
        token_memory_state: Optional[TokenMemoryState] = None,
    ) -> torch.Tensor:
        if speculative_draft_tokens <= 1:
            return self.generate(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                slot_state=slot_state,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                token_signature_lookup=token_signature_lookup,
                token_family_lookup=token_family_lookup,
                token_level_lookup=token_level_lookup,
                token_relation_lookup=token_relation_lookup,
                suppressed_token_ids=suppressed_token_ids,
                recommit_interval=recommit_interval,
                recommit_signature_threshold=recommit_signature_threshold,
                beam_size=1,
                use_speculative_decoding=False,
                signature_lattice_state=signature_lattice_state,
                token_memory_state=token_memory_state,
            )

        token_signature_lookup_tensor = self._lookup_tensor_from_map(
            token_signature_lookup,
            device=input_ids.device,
            default_value=7,
            size_hint=self.vocab_size,
        )
        token_family_lookup_tensor = self._lookup_tensor_from_map(
            token_family_lookup,
            device=input_ids.device,
            default_value=0,
            size_hint=self.vocab_size,
        )
        token_level_lookup_tensor = self._lookup_tensor_from_map(
            token_level_lookup,
            device=input_ids.device,
            default_value=self.signature_level_to_id["char"],
            size_hint=self.vocab_size,
        )
        token_relation_lookup_tensor = self._lookup_tensor_from_map(
            token_relation_lookup,
            device=input_ids.device,
            default_value=self.signature_relation_to_id["continuation"],
            size_hint=self.vocab_size,
        )

        def _lookup_next_tokens(lookup_tensor: Optional[torch.Tensor], token_ids: torch.Tensor, default_value: int) -> torch.Tensor:
            if lookup_tensor is None:
                return torch.full_like(token_ids, fill_value=default_value)
            values = lookup_tensor[token_ids.squeeze(-1).long()]
            return values.to(device=token_ids.device, dtype=token_ids.dtype).unsqueeze(-1)

        def _build_signature_tensors(token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            next_sig = _lookup_next_tokens(token_signature_lookup_tensor, token_ids, 7)
            next_family = _lookup_next_tokens(token_family_lookup_tensor, token_ids, 0)
            next_level = _lookup_next_tokens(token_level_lookup_tensor, token_ids, self.signature_level_to_id["char"])
            next_relation = _lookup_next_tokens(
                token_relation_lookup_tensor,
                token_ids,
                self.signature_relation_to_id["continuation"],
            )
            next_parent = next_sig.clone()
            return next_sig, next_family, next_level, next_relation, next_parent

        def apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            if repetition_penalty <= 1.0:
                return logits
            adjusted = logits.clone()
            for batch_idx in range(adjusted.size(0)):
                unique_tokens = torch.unique(history[batch_idx])
                for token_id in unique_tokens.tolist():
                    if token_id in {self.cfg.pad_id, self.cfg.bos_id}:
                        continue
                    value = adjusted[batch_idx, token_id]
                    adjusted[batch_idx, token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
            return adjusted

        def apply_no_repeat_ngram(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            if no_repeat_ngram_size <= 1:
                return logits
            seq_len = history.size(1)
            if seq_len < no_repeat_ngram_size - 1:
                return logits
            blocked = logits.clone()
            prefix_len = no_repeat_ngram_size - 1
            for batch_idx in range(history.size(0)):
                tokens = history[batch_idx].tolist()
                if len(tokens) < no_repeat_ngram_size:
                    continue
                ngram_map: Dict[Tuple[int, ...], set[int]] = {}
                for i in range(len(tokens) - no_repeat_ngram_size + 1):
                    prefix = tuple(tokens[i : i + prefix_len])
                    next_token = tokens[i + prefix_len]
                    ngram_map.setdefault(prefix, set()).add(next_token)
                current_prefix = tuple(tokens[-prefix_len:])
                banned = ngram_map.get(current_prefix, set())
                if banned:
                    blocked[batch_idx, list(banned)] = float("-inf")
            return blocked

        def suppress_early_eos(logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            if step_idx + 1 >= min_new_tokens:
                return logits
            blocked = logits.clone()
            blocked[:, self.cfg.eos_id] = float("-inf")
            return blocked

        def _apply_speculative_generation_filters(
            logits: torch.Tensor,
            history: torch.Tensor,
            output: PrismalWaveOutput,
            *,
            step_idx: int,
        ) -> torch.Tensor:
            filtered = logits
            filtered = apply_repetition_penalty(filtered, history)
            filtered = apply_no_repeat_ngram(filtered, history)
            filtered = self._apply_construction_collapse_mask(filtered, history)
            filtered = suppress_early_eos(filtered, step_idx)
            filtered = self._apply_signature_neighborhood_generation_bias(filtered, output, token_family_lookup_tensor)
            filtered = self._apply_token_memory_copy_bias(filtered, output)
            filtered = self._apply_generation_safety_mask(filtered)
            filtered = self._apply_explicit_generation_mask(filtered, suppressed_token_ids)
            if top_k > 0:
                top_vals, top_idx = torch.topk(filtered, k=min(top_k, filtered.size(-1)))
                top_filtered = torch.full_like(filtered, float("-inf"))
                top_filtered.scatter_(1, top_idx, top_vals)
                filtered = top_filtered
            filtered, _ = self._sanitize_sampling_logits(filtered)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
                sorted_logits, _ = self._sanitize_sampling_logits(sorted_logits)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumprobs > top_p
                cutoff[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
                filtered = torch.full_like(filtered, float("-inf"))
                filtered.scatter_(1, sorted_idx, sorted_logits)
            return filtered

        generated = input_ids.clone()
        generated_families = signature_family_ids.clone() if signature_family_ids is not None else torch.zeros_like(generated)
        generated_signatures = signature_ids.clone() if signature_ids is not None else torch.zeros_like(generated)
        generated_levels = signature_level_ids.clone() if signature_level_ids is not None else torch.zeros_like(generated)
        generated_relations = signature_relation_ids.clone() if signature_relation_ids is not None else torch.zeros_like(generated)
        generated_parents = parent_signature_ids.clone() if parent_signature_ids is not None else generated_signatures.clone()
        carried_slots = slot_state
        carried_lattice_state = signature_lattice_state
        carried_token_memory_state = token_memory_state
        anchor_token_memory_state = token_memory_state
        committed_path_index: Optional[int] = None
        min_new_tokens = max(0, int(min_new_tokens))
        speculative_draft_tokens = max(1, int(speculative_draft_tokens))
        speculative_temperature = max(0.0, float(speculative_temperature))
        total_target_len = generated.size(1) + max_new_tokens

        prefill_output = self.forward(
            generated,
            signature_family_ids=generated_families,
            signature_ids=generated_signatures,
            signature_level_ids=generated_levels,
            signature_relation_ids=generated_relations,
            parent_signature_ids=generated_parents,
            slot_state=carried_slots,
            signature_lattice_state=carried_lattice_state,
            token_memory_state=carried_token_memory_state,
            path_index=None,
        )
        committed_path_index = int(prefill_output.route_stats.get("selected_path_index", torch.tensor([0], device=generated.device)).view(-1)[0].item())
        committed_path_index, _, _ = self._select_race_lane(prefill_output, 0, fallback_lane=committed_path_index)
        carried_slots = prefill_output.slot_state if prefill_output.slot_state is not None else carried_slots
        carried_lattice_state = prefill_output.signature_lattice_state if prefill_output.signature_lattice_state is not None else carried_lattice_state
        carried_token_memory_state = prefill_output.token_memory_state if prefill_output.token_memory_state is not None else carried_token_memory_state
        if (
            carried_token_memory_state is not None
            and carried_token_memory_state.anchor_cursor_active.numel() > 0
            and bool(carried_token_memory_state.anchor_cursor_active.any().item())
        ):
            return self.generate(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                slot_state=slot_state,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                beam_size=1,
                token_signature_lookup=token_signature_lookup,
                token_family_lookup=token_family_lookup,
                token_level_lookup=token_level_lookup,
                token_relation_lookup=token_relation_lookup,
                suppressed_token_ids=suppressed_token_ids,
                recommit_interval=recommit_interval,
                recommit_signature_threshold=recommit_signature_threshold,
                signature_lattice_state=signature_lattice_state,
                token_memory_state=token_memory_state,
                use_speculative_decoding=False,
                speculative_draft_tokens=1,
                speculative_temperature=0.0,
            )

        while generated.size(1) < total_target_len:
            remaining = total_target_len - generated.size(1)
            draft_count = min(speculative_draft_tokens, remaining)
            if draft_count <= 0:
                break
            base_len = generated.size(1)
            draft_tokens: List[torch.Tensor] = []
            draft_signatures: List[torch.Tensor] = []
            draft_families: List[torch.Tensor] = []
            draft_levels: List[torch.Tensor] = []
            draft_relations: List[torch.Tensor] = []
            draft_parents: List[torch.Tensor] = []
            draft_slots = carried_slots
            draft_lattice_state = carried_lattice_state
            current_token = generated[:, -1:]
            current_signature = generated_signatures[:, -1:]
            current_family = generated_families[:, -1:]
            current_level = generated_levels[:, -1:]
            current_relation = generated_relations[:, -1:]
            current_parent = generated_parents[:, -1:]

            for step_idx in range(draft_count):
                logits, draft_slots, output = self.forward_incremental(
                    current_token,
                    signature_family_ids=current_family,
                    signature_ids=current_signature,
                    signature_level_ids=current_level,
                    signature_relation_ids=current_relation,
                    parent_signature_ids=current_parent,
                    slot_state=draft_slots,
                    signature_lattice_state=draft_lattice_state,
                    token_memory_state=carried_token_memory_state,
                    path_index=committed_path_index,
                    position_index=base_len - 1 + step_idx,
                )
                draft_lattice_state = output.signature_lattice_state if output.signature_lattice_state is not None else draft_lattice_state
                carried_token_memory_state = output.token_memory_state if output.token_memory_state is not None else carried_token_memory_state
                draft_logits = _apply_speculative_generation_filters(
                    logits,
                    torch.cat([generated, current_token], dim=1),
                    output,
                    step_idx=base_len + step_idx,
                )
                if speculative_temperature > 0.0:
                    draft_logits = draft_logits / max(speculative_temperature, 1e-3)
                    draft_logits, _ = self._sanitize_sampling_logits(draft_logits)
                    next_token = torch.multinomial(F.softmax(draft_logits, dim=-1), num_samples=1)
                else:
                    draft_logits, _ = self._sanitize_sampling_logits(draft_logits)
                    next_token = torch.argmax(draft_logits, dim=-1, keepdim=True)
                next_signature, next_family, next_level, next_relation, next_parent = _build_signature_tensors(next_token)
                draft_tokens.append(next_token)
                draft_signatures.append(next_signature)
                draft_families.append(next_family)
                draft_levels.append(next_level)
                draft_relations.append(next_relation)
                draft_parents.append(next_parent)
                current_token = next_token
                current_signature = next_signature
                current_family = next_family
                current_level = next_level
                current_relation = next_relation
                current_parent = next_parent

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1)
            draft_signatures_tensor = torch.cat(draft_signatures, dim=1)
            draft_families_tensor = torch.cat(draft_families, dim=1)
            draft_levels_tensor = torch.cat(draft_levels, dim=1)
            draft_relations_tensor = torch.cat(draft_relations, dim=1)
            draft_parents_tensor = torch.cat(draft_parents, dim=1)

            verify_input = torch.cat([generated, draft_tokens_tensor], dim=1)
            verify_families = torch.cat([generated_families, draft_families_tensor], dim=1)
            verify_signatures = torch.cat([generated_signatures, draft_signatures_tensor], dim=1)
            verify_levels = torch.cat([generated_levels, draft_levels_tensor], dim=1)
            verify_relations = torch.cat([generated_relations, draft_relations_tensor], dim=1)
            verify_parents = torch.cat([generated_parents, draft_parents_tensor], dim=1)
            verify_output = self.forward(
                verify_input,
                signature_family_ids=verify_families,
                signature_ids=verify_signatures,
                signature_level_ids=verify_levels,
                signature_relation_ids=verify_relations,
                parent_signature_ids=verify_parents,
                slot_state=carried_slots,
                signature_lattice_state=None,
                token_memory_state=carried_token_memory_state,
                path_index=None,
            )
            verify_start = max(0, generated.size(1) - 1)
            verify_logits = verify_output.logits[:, verify_start : verify_start + draft_count, :]
            verify_predictions = _apply_speculative_generation_filters(
                verify_logits[:, 0, :],
                generated,
                verify_output,
                step_idx=base_len,
            ).argmax(dim=-1, keepdim=True)
            if draft_count > 1:
                tail_predictions: List[torch.Tensor] = []
                running_history = generated.clone()
                for idx in range(1, draft_count):
                    step_logits = _apply_speculative_generation_filters(
                        verify_logits[:, idx, :],
                        torch.cat([running_history, draft_tokens_tensor[:, :idx]], dim=1),
                        verify_output,
                        step_idx=base_len + idx,
                    )
                    tail_predictions.append(step_logits.argmax(dim=-1, keepdim=True))
                if tail_predictions:
                    verify_predictions = torch.cat([verify_predictions] + tail_predictions, dim=1)
            matches = draft_tokens_tensor.eq(verify_predictions)
            accept_count = draft_count
            for step_idx in range(draft_count):
                if not bool(matches[:, step_idx].all()):
                    accept_count = step_idx
                    break
            if accept_count <= 0:
                accepted_tokens = verify_predictions[:, :1]
                accepted_signatures, accepted_families, accepted_levels, accepted_relations, accepted_parents = _build_signature_tensors(accepted_tokens)
                accept_count = 1
            else:
                accepted_tokens = draft_tokens_tensor[:, :accept_count]
                accepted_signatures = draft_signatures_tensor[:, :accept_count]
                accepted_families = draft_families_tensor[:, :accept_count]
                accepted_levels = draft_levels_tensor[:, :accept_count]
                accepted_relations = draft_relations_tensor[:, :accept_count]
                accepted_parents = draft_parents_tensor[:, :accept_count]

            generated = torch.cat([generated, accepted_tokens], dim=1)
            generated_families = torch.cat([generated_families, accepted_families], dim=1)
            generated_signatures = torch.cat([generated_signatures, accepted_signatures], dim=1)
            generated_levels = torch.cat([generated_levels, accepted_levels], dim=1)
            generated_relations = torch.cat([generated_relations, accepted_relations], dim=1)
            generated_parents = torch.cat([generated_parents, accepted_parents], dim=1)

            replay_slots = carried_slots
            replay_lattice_state = carried_lattice_state
            replay_path_index = committed_path_index
            for step_idx in range(accept_count):
                _, replay_slots, replay_output = self.forward_incremental(
                    generated[:, base_len + step_idx : base_len + step_idx + 1],
                    signature_family_ids=generated_families[:, base_len + step_idx : base_len + step_idx + 1],
                    signature_ids=generated_signatures[:, base_len + step_idx : base_len + step_idx + 1],
                    signature_level_ids=generated_levels[:, base_len + step_idx : base_len + step_idx + 1],
                    signature_relation_ids=generated_relations[:, base_len + step_idx : base_len + step_idx + 1],
                    parent_signature_ids=generated_parents[:, base_len + step_idx : base_len + step_idx + 1],
                    slot_state=replay_slots,
                    signature_lattice_state=replay_lattice_state,
                    token_memory_state=carried_token_memory_state,
                    path_index=replay_path_index,
                    position_index=base_len + step_idx,
                )
                replay_lattice_state = replay_output.signature_lattice_state if replay_output.signature_lattice_state is not None else replay_lattice_state
                carried_token_memory_state = replay_output.token_memory_state if replay_output.token_memory_state is not None else carried_token_memory_state
                selected = replay_output.route_stats.get("selected_path_index")
                if selected is not None:
                    replay_path_index = int(selected.view(-1)[0].item())
            carried_slots = replay_slots
            carried_lattice_state = replay_lattice_state
            committed_path_index, _, _ = self._select_race_lane(verify_output, generated.size(1), fallback_lane=replay_path_index)

            if generated.size(1) >= total_target_len:
                break
            if bool(torch.all(accepted_tokens[:, -1] == self.cfg.eos_id)) and generated.size(1) >= input_ids.size(1) + min_new_tokens:
                break

        return generated

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        superposition_bag_size: int = 1,
    ) -> Tuple[torch.Tensor, PrismalWaveOutput]:
        superposition_bag_size = max(1, int(superposition_bag_size))
        use_superposition = (
            bool(getattr(self.cfg, "use_token_superposition_training", False))
            and superposition_bag_size > 1
            and input_ids.size(1) > superposition_bag_size
        )
        if use_superposition:
            superposition = self._build_superposition_batch(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                loss_mask=loss_mask,
                bag_size=superposition_bag_size,
            )
            output = self(
                superposition.input_ids,
                signature_family_ids=superposition.signature_family_ids,
                signature_ids=superposition.signature_ids,
                signature_level_ids=superposition.signature_level_ids,
                signature_relation_ids=superposition.signature_relation_ids,
                parent_signature_ids=superposition.parent_signature_ids,
                loss_mask=superposition.forward_loss_mask,
                superposition_token_groups=superposition.token_groups,
                superposition_bag_size=superposition.bag_size,
            )
            target_count = superposition.target_ids.size(1)
            logits = output.logits[:, :target_count, :]
            repeated_logits = logits.unsqueeze(2).expand(-1, -1, superposition.target_ids.size(2), -1)
            token_losses = F.cross_entropy(
                repeated_logits.reshape(-1, repeated_logits.size(-1)),
                superposition.target_ids.reshape(-1),
                ignore_index=self.cfg.pad_id,
                reduction="none",
            ).view_as(superposition.target_ids)
            valid_tokens = superposition.target_mask.to(device=token_losses.device, dtype=token_losses.dtype)
            denom = valid_tokens.sum()
            if float(denom.item()) > 0.0:
                ce = (token_losses * valid_tokens).sum() / denom
            else:
                valid = superposition.target_ids.ne(self.cfg.pad_id)
                ce = token_losses.masked_select(valid).mean() if valid.any() else token_losses.mean()
        else:
            output = self(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                loss_mask=loss_mask,
            )
            token_losses = F.cross_entropy(
                output.logits.reshape(-1, output.logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.cfg.pad_id,
                reduction="none",
            ).view_as(labels)
            valid_tokens = labels.ne(self.cfg.pad_id)
            if loss_mask is not None:
                loss_mask = loss_mask.to(device=labels.device, dtype=token_losses.dtype)
                if loss_mask.shape != labels.shape:
                    loss_mask = loss_mask[..., : labels.size(1)]
                    if loss_mask.shape != labels.shape:
                        loss_mask = loss_mask.expand_as(labels)
                weighted_mask = loss_mask * valid_tokens.to(dtype=token_losses.dtype)
                denom = weighted_mask.sum()
                if float(denom.item()) > 0.0:
                    ce = (token_losses * weighted_mask).sum() / denom
                else:
                    ce = token_losses.masked_select(valid_tokens).mean() if valid_tokens.any() else token_losses.mean()
            else:
                ce = token_losses.masked_select(valid_tokens).mean() if valid_tokens.any() else token_losses.mean()
        output.route_stats["token_superposition_active"] = torch.tensor(
            1.0 if use_superposition else 0.0,
            device=output.logits.device,
        )
        output.route_stats["token_superposition_bag_size"] = torch.tensor(float(superposition_bag_size), device=output.logits.device)
        output.route_stats["token_superposition_bag_count"] = torch.tensor(float(output.logits.size(1)), device=output.logits.device)
        loss = ce + output.aux_loss
        output.ce_loss = ce.detach()
        return loss, output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        signature_family_ids: Optional[torch.Tensor] = None,
        signature_ids: Optional[torch.Tensor] = None,
        signature_level_ids: Optional[torch.Tensor] = None,
        signature_relation_ids: Optional[torch.Tensor] = None,
        parent_signature_ids: Optional[torch.Tensor] = None,
        slot_state: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        min_new_tokens: int = 24,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.08,
        no_repeat_ngram_size: int = 4,
        token_signature_lookup: Optional[Dict[int, int]] = None,
        token_family_lookup: Optional[Dict[int, int]] = None,
        token_level_lookup: Optional[Dict[int, int]] = None,
        token_relation_lookup: Optional[Dict[int, int]] = None,
        suppressed_token_ids: Optional[Sequence[int]] = None,
        recommit_interval: int = 0,
        recommit_signature_threshold: float = 0.65,
        beam_size: int = 1,
        use_speculative_decoding: Optional[bool] = None,
        speculative_draft_tokens: Optional[int] = None,
        speculative_temperature: Optional[float] = None,
        signature_lattice_state: Optional[SignatureLatticeState] = None,
        token_memory_state: Optional[TokenMemoryState] = None,
    ) -> torch.Tensor:
        _validate_aligned_signature_tensors(
            input_ids,
            signature_family_ids=signature_family_ids,
            signature_ids=signature_ids,
            signature_level_ids=signature_level_ids,
            signature_relation_ids=signature_relation_ids,
            parent_signature_ids=parent_signature_ids,
            context="generate",
        )
        beam_size = max(1, int(beam_size))
        speculative_enabled = self.cfg.use_speculative_decoding if use_speculative_decoding is None else bool(use_speculative_decoding)
        speculative_draft_tokens = int(speculative_draft_tokens if speculative_draft_tokens is not None else getattr(self.cfg, "speculative_draft_tokens", 1))
        speculative_temperature = float(
            speculative_temperature if speculative_temperature is not None else getattr(self.cfg, "speculative_temperature", 0.0)
        )
        anchor_rail_active = bool(
            token_memory_state is not None
            and token_memory_state.anchor_cursor_active.numel() > 0
            and bool(token_memory_state.anchor_cursor_active.any().item())
        )
        if anchor_rail_active and beam_size > 1:
            return self.generate(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                slot_state=slot_state,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                beam_size=1,
                token_signature_lookup=token_signature_lookup,
                token_family_lookup=token_family_lookup,
                token_level_lookup=token_level_lookup,
                token_relation_lookup=token_relation_lookup,
                suppressed_token_ids=suppressed_token_ids,
                recommit_interval=recommit_interval,
                recommit_signature_threshold=recommit_signature_threshold,
                signature_lattice_state=signature_lattice_state,
                token_memory_state=token_memory_state,
                use_speculative_decoding=False,
                speculative_draft_tokens=1,
                speculative_temperature=0.0,
            )
        if (
            speculative_enabled
            and speculative_draft_tokens > 1
            and self.use_torus_core
            and self.use_hmote
            and beam_size == 1
            and not anchor_rail_active
        ):
            return self._coarse_fine_speculative_generate(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                slot_state=slot_state,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                token_signature_lookup=token_signature_lookup,
                token_family_lookup=token_family_lookup,
                token_level_lookup=token_level_lookup,
                token_relation_lookup=token_relation_lookup,
                suppressed_token_ids=suppressed_token_ids,
                recommit_interval=recommit_interval,
                recommit_signature_threshold=recommit_signature_threshold,
                speculative_draft_tokens=speculative_draft_tokens,
                speculative_temperature=speculative_temperature,
                signature_lattice_state=signature_lattice_state,
                token_memory_state=token_memory_state,
            )
        if beam_size > 1:
            return self._beam_search_generate(
                input_ids,
                signature_family_ids=signature_family_ids,
                signature_ids=signature_ids,
                signature_level_ids=signature_level_ids,
                signature_relation_ids=signature_relation_ids,
                parent_signature_ids=parent_signature_ids,
                slot_state=slot_state,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                beam_size=beam_size,
                token_signature_lookup=token_signature_lookup,
                token_family_lookup=token_family_lookup,
                token_level_lookup=token_level_lookup,
                token_relation_lookup=token_relation_lookup,
                suppressed_token_ids=suppressed_token_ids,
                recommit_interval=recommit_interval,
                recommit_signature_threshold=recommit_signature_threshold,
                signature_lattice_state=signature_lattice_state,
                token_memory_state=token_memory_state,
            )
        token_signature_lookup_tensor = self._lookup_tensor_from_map(
            token_signature_lookup,
            device=input_ids.device,
            default_value=7,
            size_hint=self.vocab_size,
        )
        token_family_lookup_tensor = self._lookup_tensor_from_map(
            token_family_lookup,
            device=input_ids.device,
            default_value=0,
            size_hint=self.vocab_size,
        )
        token_level_lookup_tensor = self._lookup_tensor_from_map(
            token_level_lookup,
            device=input_ids.device,
            default_value=self.signature_level_to_id["char"],
            size_hint=self.vocab_size,
        )
        token_relation_lookup_tensor = self._lookup_tensor_from_map(
            token_relation_lookup,
            device=input_ids.device,
            default_value=self.signature_relation_to_id["continuation"],
            size_hint=self.vocab_size,
        )

        def _lookup_next_tokens(lookup_tensor: Optional[torch.Tensor], token_ids: torch.Tensor, default_value: int) -> torch.Tensor:
            if lookup_tensor is None:
                return torch.full_like(token_ids, fill_value=default_value)
            values = lookup_tensor[token_ids.squeeze(-1).long()]
            return values.to(device=token_ids.device, dtype=token_ids.dtype).unsqueeze(-1)

        self.eval()
        generated = input_ids.clone()
        generated_families = signature_family_ids.clone() if signature_family_ids is not None else torch.zeros_like(generated)
        generated_signatures = signature_ids.clone() if signature_ids is not None else torch.zeros_like(generated)
        generated_levels = signature_level_ids.clone() if signature_level_ids is not None else torch.zeros_like(generated)
        generated_relations = signature_relation_ids.clone() if signature_relation_ids is not None else torch.zeros_like(generated)
        generated_parents = parent_signature_ids.clone() if parent_signature_ids is not None else generated_signatures.clone()
        carried_slots = slot_state
        carried_lattice_state = signature_lattice_state
        carried_token_memory_state = token_memory_state
        anchor_token_memory_state = token_memory_state
        committed_path_index: Optional[int] = None
        min_new_tokens = max(0, int(min_new_tokens))
        top_p = float(max(0.0, min(1.0, top_p)))
        repetition_penalty = float(max(1.0, repetition_penalty))
        no_repeat_ngram_size = int(no_repeat_ngram_size)
        recommit_signature_threshold = float(recommit_signature_threshold)
        relay_interval = max(1, int(recommit_interval if recommit_interval > 0 else getattr(self.cfg, "torus_relay_interval", 16)))
        generation_lap_cap = max(1, int(getattr(self.cfg, "generation_lap_cap", 1)))
        generation_lap_token_cap = max(1, int(getattr(self.cfg, "generation_lap_token_cap", relay_interval)))
        relay_interval = min(relay_interval, generation_lap_token_cap)
        laps_used = 0

        def apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            if repetition_penalty <= 1.0:
                return logits
            adjusted = logits.clone()
            for batch_idx in range(adjusted.size(0)):
                unique_tokens = torch.unique(history[batch_idx])
                for token_id in unique_tokens.tolist():
                    if token_id in {self.cfg.pad_id, self.cfg.bos_id}:
                        continue
                    value = adjusted[batch_idx, token_id]
                    adjusted[batch_idx, token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
            return adjusted

        def apply_no_repeat_ngram(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            if no_repeat_ngram_size <= 1:
                return logits
            seq_len = history.size(1)
            if seq_len < no_repeat_ngram_size - 1:
                return logits
            blocked = logits.clone()
            prefix_len = no_repeat_ngram_size - 1
            for batch_idx in range(history.size(0)):
                tokens = history[batch_idx].tolist()
                if len(tokens) < no_repeat_ngram_size:
                    continue
                ngram_map: Dict[Tuple[int, ...], set[int]] = {}
                for i in range(len(tokens) - no_repeat_ngram_size + 1):
                    prefix = tuple(tokens[i : i + prefix_len])
                    next_token = tokens[i + prefix_len]
                    ngram_map.setdefault(prefix, set()).add(next_token)
                current_prefix = tuple(tokens[-prefix_len:])
                banned = ngram_map.get(current_prefix, set())
                if banned:
                    blocked[batch_idx, list(banned)] = float("-inf")
            return blocked

        def suppress_early_eos(logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            if step_idx + 1 >= min_new_tokens:
                return logits
            blocked = logits.clone()
            blocked[:, self.cfg.eos_id] = float("-inf")
            return blocked

        def _apply_speculative_generation_filters(
            logits: torch.Tensor,
            history: torch.Tensor,
            output: PrismalWaveOutput,
            *,
            step_idx: int,
        ) -> torch.Tensor:
            filtered = logits
            filtered = apply_repetition_penalty(filtered, history)
            filtered = apply_no_repeat_ngram(filtered, history)
            filtered = self._apply_construction_collapse_mask(filtered, history)
            filtered = suppress_early_eos(filtered, step_idx)
            filtered = self._apply_signature_neighborhood_generation_bias(filtered, output, token_family_lookup_tensor)
            filtered = self._apply_token_memory_copy_bias(filtered, output)
            filtered = self._apply_generation_safety_mask(filtered)
            filtered = self._apply_explicit_generation_mask(filtered, suppressed_token_ids)
            if top_k > 0:
                top_vals, top_idx = torch.topk(filtered, k=min(top_k, filtered.size(-1)))
                top_filtered = torch.full_like(filtered, float("-inf"))
                top_filtered.scatter_(1, top_idx, top_vals)
                filtered = top_filtered
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumprobs > top_p
                cutoff[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
                filtered = torch.full_like(filtered, float("-inf"))
                filtered.scatter_(1, sorted_idx, sorted_logits)
            return filtered

        prefill_output = self.forward(
            generated,
            signature_family_ids=generated_families,
            signature_ids=generated_signatures,
            signature_level_ids=generated_levels,
            signature_relation_ids=generated_relations,
            parent_signature_ids=generated_parents,
            slot_state=carried_slots,
            signature_lattice_state=carried_lattice_state,
            token_memory_state=carried_token_memory_state,
            path_index=None,
        )
        selected = prefill_output.route_stats.get("selected_path_index")
        if selected is not None:
            committed_path_index = int(selected.view(-1)[0].item())
        else:
            committed_path_index = committed_path_index if committed_path_index is not None else 0
        committed_path_index, committed_band, lane_temperature = self._select_race_lane(
            prefill_output,
            0,
            fallback_lane=committed_path_index,
        )
        carried_slots = prefill_output.slot_state if prefill_output.slot_state is not None else carried_slots
        carried_lattice_state = prefill_output.signature_lattice_state if prefill_output.signature_lattice_state is not None else carried_lattice_state
        carried_token_memory_state = prefill_output.token_memory_state if prefill_output.token_memory_state is not None else carried_token_memory_state
        output = prefill_output

        logits = prefill_output.logits[:, -1, :]
        decode_temperature = max(
            float(temperature),
            float(lane_temperature) if self.use_race_lanes else self._relay_temperature(0, temperature),
            1e-3,
        )
        logits = logits / decode_temperature
        logits = apply_repetition_penalty(logits, generated)
        logits = apply_no_repeat_ngram(logits, generated)
        logits = self._apply_construction_collapse_mask(logits, generated)
        logits = suppress_early_eos(logits, 0)
        logits = self._apply_signature_neighborhood_generation_bias(logits, prefill_output, token_family_lookup_tensor)
        logits = self._apply_token_memory_copy_bias(logits, prefill_output)
        logits = self._apply_generation_safety_mask(logits)
        logits = self._apply_explicit_generation_mask(logits, suppressed_token_ids)
        if top_k > 0:
            top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, top_idx, top_vals)
            logits = filtered
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_logits, _ = self._sanitize_sampling_logits(sorted_logits)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumprobs > top_p
            cutoff[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_idx, sorted_logits)
        logits, _ = self._sanitize_sampling_logits(logits)
        anchor_next_ids = self._token_memory_anchor_next_ids(anchor_token_memory_state, device=input_ids.device)
        if anchor_next_ids is not None:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            active_anchor_mask = anchor_next_ids.ge(0)
            if bool(active_anchor_mask.any().item()):
                next_id = next_id.clone()
                next_id[active_anchor_mask] = anchor_next_ids[active_anchor_mask]
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        anchor_token_memory_state = self._advance_token_memory_anchor_state(anchor_token_memory_state, next_id)
        if token_signature_lookup is not None:
            next_sig = _lookup_next_tokens(token_signature_lookup_tensor, next_id, 7)
        else:
            next_sig = torch.full_like(next_id, fill_value=7)
        if token_family_lookup is not None:
            next_family = _lookup_next_tokens(token_family_lookup_tensor, next_id, 0)
        else:
            next_family = torch.zeros_like(next_id)
        if token_level_lookup is not None:
            next_level = _lookup_next_tokens(token_level_lookup_tensor, next_id, self.signature_level_to_id["char"])
        else:
            next_level = torch.full_like(next_id, fill_value=self.signature_level_to_id["char"])
        if token_relation_lookup is not None:
            next_relation = _lookup_next_tokens(
                token_relation_lookup_tensor,
                next_id,
                self.signature_relation_to_id["continuation"],
            )
        else:
            next_relation = torch.full_like(next_id, fill_value=self.signature_relation_to_id["continuation"])
        next_parent = next_sig.clone()
        generated = torch.cat([generated, next_id], dim=-1)
        generated_families = torch.cat([generated_families, next_family], dim=-1)
        generated_signatures = torch.cat([generated_signatures, next_sig], dim=-1)
        generated_levels = torch.cat([generated_levels, next_level], dim=-1)
        generated_relations = torch.cat([generated_relations, next_relation], dim=-1)
        generated_parents = torch.cat([generated_parents, next_parent], dim=-1)
        if max_new_tokens <= 1 or (1 >= min_new_tokens and torch.all(next_id.squeeze(-1) == self.cfg.eos_id)):
            return generated

        for step_idx in range(1, max_new_tokens):
            relay_needed = step_idx % relay_interval == 0
            if not relay_needed and "signature_agreement" in output.route_stats:
                relay_needed = float(output.route_stats["signature_agreement"].float().mean().item()) < recommit_signature_threshold
            if relay_needed and laps_used >= generation_lap_cap - 1:
                relay_needed = False
            current_path_index = None if relay_needed else committed_path_index
            logits, carried_slots, output = self.forward_incremental(
                generated[:, -1:],
                signature_family_ids=generated_families[:, -1:],
                signature_ids=generated_signatures[:, -1:],
                signature_level_ids=generated_levels[:, -1:],
                signature_relation_ids=generated_relations[:, -1:],
                parent_signature_ids=generated_parents[:, -1:],
                slot_state=carried_slots,
                signature_lattice_state=carried_lattice_state,
                token_memory_state=carried_token_memory_state,
                path_index=current_path_index,
                position_index=generated.size(1) - 1,
            )
            carried_lattice_state = output.signature_lattice_state if output.signature_lattice_state is not None else carried_lattice_state
            carried_token_memory_state = output.token_memory_state if output.token_memory_state is not None else carried_token_memory_state
            selected = output.route_stats.get("selected_path_index")
            if selected is not None:
                committed_path_index = int(selected.view(-1)[0].item())
            committed_path_index, committed_band, lane_temperature = self._select_race_lane(
                output,
                step_idx,
                fallback_lane=committed_path_index,
            )
            if relay_needed:
                laps_used += 1
            decode_temperature = max(
                float(temperature),
                float(lane_temperature) if self.use_race_lanes else self._relay_temperature(step_idx, temperature),
                1e-3,
            )
            logits = logits / decode_temperature
            logits = apply_repetition_penalty(logits, generated)
            logits = apply_no_repeat_ngram(logits, generated)
            logits = self._apply_construction_collapse_mask(logits, generated)
            logits = suppress_early_eos(logits, step_idx)
            logits = self._apply_signature_neighborhood_generation_bias(logits, output, token_family_lookup)
            logits = self._apply_token_memory_copy_bias(logits, output)
            logits = self._apply_generation_safety_mask(logits)
            logits = self._apply_explicit_generation_mask(logits, suppressed_token_ids)
            if top_k > 0:
                top_vals, top_idx = torch.topk(logits, k=min(top_k, logits.size(-1)))
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, top_idx, top_vals)
                logits = filtered
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_logits, _ = self._sanitize_sampling_logits(sorted_logits)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumprobs > top_p
                cutoff[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)
            logits, _ = self._sanitize_sampling_logits(logits)
            anchor_next_ids = self._token_memory_anchor_next_ids(anchor_token_memory_state, device=input_ids.device)
            if anchor_next_ids is not None:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                active_anchor_mask = anchor_next_ids.ge(0)
                if bool(active_anchor_mask.any().item()):
                    next_id = next_id.clone()
                    next_id[active_anchor_mask] = anchor_next_ids[active_anchor_mask]
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            anchor_token_memory_state = self._advance_token_memory_anchor_state(anchor_token_memory_state, next_id)
            if token_signature_lookup is not None:
                next_sig = _lookup_next_tokens(token_signature_lookup_tensor, next_id, 7)
            else:
                next_sig = torch.full_like(next_id, fill_value=7)
            if token_family_lookup is not None:
                next_family = _lookup_next_tokens(token_family_lookup_tensor, next_id, 0)
            else:
                next_family = torch.zeros_like(next_id)
            if token_level_lookup is not None:
                next_level = _lookup_next_tokens(token_level_lookup_tensor, next_id, self.signature_level_to_id["char"])
            else:
                next_level = torch.full_like(next_id, fill_value=self.signature_level_to_id["char"])
            if token_relation_lookup is not None:
                next_relation = _lookup_next_tokens(
                    token_relation_lookup_tensor,
                    next_id,
                    self.signature_relation_to_id["continuation"],
                )
            else:
                next_relation = torch.full_like(next_id, fill_value=self.signature_relation_to_id["continuation"])
            next_parent = next_sig.clone()
            generated = torch.cat([generated, next_id], dim=-1)
            generated_families = torch.cat([generated_families, next_family], dim=-1)
            generated_signatures = torch.cat([generated_signatures, next_sig], dim=-1)
            generated_levels = torch.cat([generated_levels, next_level], dim=-1)
            generated_relations = torch.cat([generated_relations, next_relation], dim=-1)
            generated_parents = torch.cat([generated_parents, next_parent], dim=-1)
            if step_idx + 1 >= min_new_tokens and torch.all(next_id.squeeze(-1) == self.cfg.eos_id):
                break
        return generated
