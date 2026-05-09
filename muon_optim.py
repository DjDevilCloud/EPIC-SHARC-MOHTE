# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

"""Local Muon-style optimizer helpers.

This implementation follows the practical setup used by Muon references:
- Muon for matrix-shaped weights
- AdamW for embeddings, norms, biases, and other non-matrix parameters

The matrix update uses a small Newton-Schulz orthogonalization loop to
approximate the polar factor of the momentum tensor.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

try:
    from .hierarchical_precision import HierarchicalPrecisionPolicy, dtype_name
except ImportError:  # pragma: no cover - supports direct script launching.
    from hierarchical_precision import HierarchicalPrecisionPolicy, dtype_name


def _is_matrix_parameter(name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad:
        return False
    if param.ndim != 2:
        return False
    # Very wide/tall matrices can make the Newton-Schulz orthogonalization step
    # allocate a huge temporary `n x n` workspace. Keep those on the scalar
    # optimizer path instead of forcing Muon to touch them.
    if max(int(dim) for dim in param.shape) > 4096:
        return False
    lowered = name.lower()
    if any(token in lowered for token in ("embed", "embedding", "norm", "bias", "layernorm", "rmsnorm")):
        return False
    return True


def split_muon_parameter_groups(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    muon_lr: float,
    scalar_lr: float,
    muon_weight_decay: float,
    scalar_weight_decay: float = 0.01,
    momentum_beta: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    extra_scale_factor: float = 1.0,
    scalar_optimizer: str = "adamw",
) -> list[dict]:
    muon_params: list[torch.nn.Parameter] = []
    scalar_params: list[torch.nn.Parameter] = []
    for name, param in named_parameters:
        if _is_matrix_parameter(name, param):
            muon_params.append(param)
        elif param.requires_grad:
            scalar_params.append(param)

    groups: list[dict] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": float(muon_lr),
                "weight_decay": float(muon_weight_decay),
                "momentum_beta": float(momentum_beta),
                "nesterov": bool(nesterov),
                "ns_steps": int(ns_steps),
                "extra_scale_factor": float(extra_scale_factor),
            }
        )
    if scalar_params:
        groups.append(
            {
                "params": scalar_params,
                "use_muon": False,
                "scalar_optimizer": scalar_optimizer.lower(),
                "lr": float(scalar_lr),
                "weight_decay": float(scalar_weight_decay),
            }
        )
    return groups


def _muon_orthogonalize(momentum: torch.Tensor, *, ns_steps: int, extra_scale_factor: float) -> torch.Tensor:
    if momentum.ndim != 2 or momentum.numel() == 0:
        return momentum

    transposed = False
    x = momentum.float()
    if x.shape[0] > x.shape[1]:
        x = x.transpose(0, 1)
        transposed = True

    x = x / (x.norm() + 1e-8)
    for _ in range(max(1, ns_steps)):
        x = 1.5 * x - 0.5 * x @ (x.transpose(0, 1) @ x)
    if transposed:
        x = x.transpose(0, 1)
    return x * float(extra_scale_factor)


@dataclass(frozen=True)
class _PrecisionAdaptiveSpec:
    role: str
    tier: str
    update_rule: str
    lr: float
    weight_decay: float
    state_dtype: torch.dtype


def _hierarchy_tier(name: str, param: torch.nn.Parameter) -> str:
    lowered = name.lower()
    depth = name.count(".")
    if any(token in lowered for token in ("adapter", "child_nests", "family_specialists", "leaf", "recursive_", "relay_")):
        return "local"
    if any(token in lowered for token in ("embedding", "embed", "construction_head", "signature_head", "token_head", "registry")):
        return "global"
    if depth <= 2:
        return "global"
    if depth >= 5:
        return "local"
    return "mid"


def _parameter_role(name: str, param: torch.nn.Parameter) -> str:
    lowered = name.lower()
    if not param.requires_grad or param.numel() == 0:
        return "frozen"
    if param.ndim <= 1 or any(token in lowered for token in ("bias", "norm", "layernorm", "rmsnorm", "gate", "scale", "temperature", "threshold")):
        return "scalar"
    if "adapter" in lowered:
        return "adapter"
    if any(token in lowered for token in ("embed", "embedding")) and param.ndim == 2:
        return "table"
    if param.ndim == 2:
        return "matrix"
    return "tensor"


def _state_dtype_for(role: str, tier: str, param: torch.nn.Parameter, *, state_precision: str = "mixed") -> torch.dtype:
    if param.device.type != "cuda":
        return torch.float32
    precision = state_precision.strip().lower()
    if precision == "fp32":
        return torch.float32
    if role in {"scalar", "table", "tensor"}:
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if tier == "local" or param.numel() >= 4096:
        return torch.bfloat16
    return torch.float32


def _update_rule_for(role: str, param: torch.nn.Parameter) -> str:
    if role in {"adapter", "matrix"}:
        if param.ndim == 2:
            return "muon"
        return "adamw"
    if role == "table":
        return "rowwise"
    return "adamw"


def _learning_rate_for(
    role: str,
    tier: str,
    *,
    base_lr: float,
    muon_lr: float,
) -> float:
    tier_scale = {"global": 0.75, "mid": 1.0, "local": 1.15}.get(tier, 1.0)
    role_scale = {"adapter": 1.0, "matrix": 0.95, "table": 0.80, "scalar": 0.35, "tensor": 0.90}.get(role, 1.0)
    base = float(muon_lr) if role in {"adapter", "matrix"} else float(base_lr)
    return float(base * tier_scale * role_scale)


def _weight_decay_for(role: str, tier: str, *, muon_weight_decay: float) -> float:
    if role in {"scalar", "table"}:
        return 0.0
    if role == "tensor":
        return float(muon_weight_decay) * 0.25
    tier_scale = {"global": 1.0, "mid": 0.75, "local": 0.5}.get(tier, 0.75)
    return float(muon_weight_decay) * tier_scale


def _precision_adaptive_spec(
    name: str,
    param: torch.nn.Parameter,
    *,
    base_lr: float,
    muon_lr: float,
    muon_weight_decay: float,
    state_precision: str = "mixed",
) -> Optional[_PrecisionAdaptiveSpec]:
    if not param.requires_grad or param.numel() == 0:
        return None
    role = _parameter_role(name, param)
    if role == "frozen":
        return None
    tier = _hierarchy_tier(name, param)
    update_rule = _update_rule_for(role, param)
    lr = _learning_rate_for(role, tier, base_lr=base_lr, muon_lr=muon_lr)
    weight_decay = _weight_decay_for(role, tier, muon_weight_decay=muon_weight_decay)
    state_dtype = _state_dtype_for(role, tier, param, state_precision=state_precision)
    lowered = name.lower()
    if update_rule == "muon" and param.ndim == 2 and max(int(dim) for dim in param.shape) > 4096 and "adapter" not in lowered:
        update_rule = "adamw"
        role = "tensor"
        lr = _learning_rate_for(role, tier, base_lr=base_lr, muon_lr=muon_lr)
        weight_decay = _weight_decay_for(role, tier, muon_weight_decay=muon_weight_decay)
        state_dtype = _state_dtype_for(role, tier, param, state_precision=state_precision)
    return _PrecisionAdaptiveSpec(
        role=role,
        tier=tier,
        update_rule=update_rule,
        lr=lr,
        weight_decay=weight_decay,
        state_dtype=state_dtype,
    )


def split_precision_adaptive_parameter_groups(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    base_lr: float,
    muon_lr: float,
    muon_weight_decay: float,
    momentum_beta: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    extra_scale_factor: float = 1.0,
    scalar_optimizer: str = "adamw",
    state_precision: str = "mixed",
    precision_policy: Optional[HierarchicalPrecisionPolicy] = None,
) -> list[dict]:
    grouped: dict[tuple[str, str, str, torch.dtype], dict] = {}
    for name, param in named_parameters:
        spec = _precision_adaptive_spec(
            name,
            param,
            base_lr=base_lr,
            muon_lr=muon_lr,
            muon_weight_decay=muon_weight_decay,
            state_precision=state_precision,
        )
        if spec is None:
            continue
        key = (spec.update_rule, spec.role, spec.tier, spec.state_dtype)
        group = grouped.get(key)
        if group is None:
            group = {
                "params": [],
                "param_names": [],
                "update_rule": spec.update_rule,
                "parameter_role": spec.role,
                "hierarchy_tier": spec.tier,
                "state_dtype": spec.state_dtype,
                "lr": float(spec.lr),
                "weight_decay": float(spec.weight_decay),
                "scalar_optimizer": scalar_optimizer.lower(),
            }
            if precision_policy is not None:
                requested_dtype = (
                    precision_policy.root_compute_dtype
                    if spec.tier == "global"
                    else precision_policy.leaf_compute_dtype
                    if spec.tier == "local"
                    else precision_policy.mid_compute_dtype
                )
                group["precision_policy_state"] = precision_policy.to_state_dict()
                group["precision_requested_compute_dtype"] = dtype_name(requested_dtype)
                group["precision_fallback_dtype"] = dtype_name(precision_policy.fallback_compute_dtype)
                group["precision_accumulator_dtype"] = dtype_name(precision_policy.accumulator_dtype)
            if spec.update_rule == "muon":
                group.update(
                    {
                        "momentum_beta": float(momentum_beta),
                        "nesterov": bool(nesterov),
                        "ns_steps": int(ns_steps),
                        "extra_scale_factor": float(extra_scale_factor),
                    }
                )
            else:
                group.update({"betas": (0.9, 0.95), "eps": 1e-8})
            grouped[key] = group
        group["params"].append(param)
        group["param_names"].append(name)
    return list(grouped.values())


class MuonAdamW(torch.optim.Optimizer):
    """Muon for matrix weights with AdamW fallback for scalars.

    The class behaves like a single optimizer so schedulers and checkpoint code
    can treat it like any other `torch.optim.Optimizer`.
    """

    def __init__(self, params, defaults=None):
        if defaults is None:
            defaults = {}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                params = group["params"]
                if not params:
                    continue
                if group.get("use_muon", False):
                    self._step_muon_group(group, params)
                else:
                    self._step_adamw_group(group, params)
        return loss

    def _step_muon_group(self, group: dict, params: Sequence[torch.nn.Parameter]) -> None:
        lr = float(group.get("lr", 1e-3))
        weight_decay = float(group.get("weight_decay", 0.0))
        beta = float(group.get("momentum_beta", 0.95))
        nesterov = bool(group.get("nesterov", True))
        ns_steps = int(group.get("ns_steps", 5))
        extra_scale_factor = float(group.get("extra_scale_factor", 1.0))

        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients.")
            if p.ndim != 2:
                raise RuntimeError("MuonAdamW muon group expected matrix-shaped parameters.")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            state["step"] += 1

            buf.mul_(beta).add_(grad, alpha=1.0 - beta)
            update_source = buf
            if nesterov:
                update_source = buf * beta + grad * (1.0 - beta)

            update = _muon_orthogonalize(update_source, ns_steps=ns_steps, extra_scale_factor=extra_scale_factor)

            if weight_decay:
                p.mul_(1.0 - lr * weight_decay)
            p.add_(update, alpha=-lr)

    def _step_adamw_group(self, group: dict, params: Sequence[torch.nn.Parameter]) -> None:
        lr = float(group.get("lr", 1e-3))
        weight_decay = float(group.get("weight_decay", 0.0))
        betas = group.get("betas", (0.9, 0.95))
        beta1, beta2 = float(betas[0]), float(betas[1])
        eps = float(group.get("eps", 1e-8))

        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients.")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            if weight_decay:
                p.mul_(1.0 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            denom = exp_avg_sq.sqrt().add_(eps)
            step_size = lr * (bias_correction2 ** 0.5) / max(bias_correction1, 1e-8)
            p.addcdiv_(exp_avg, denom, value=-step_size)


class PrecisionAdaptiveHierarchicalOptimizer(torch.optim.Optimizer):
    """Role-aware optimizer for quantized bases, adapters, tables, and scalars."""

    def __init__(self, params, defaults=None):
        if defaults is None:
            defaults = {}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                params = group.get("params", [])
                if not params:
                    continue
                update_rule = str(group.get("update_rule", "adamw")).lower()
                if update_rule == "muon":
                    self._step_muon_group(group, params)
                elif update_rule == "rowwise":
                    self._step_rowwise_group(group, params)
                else:
                    self._step_adamw_group(group, params)
        return loss

    def _state_dtype(self, group: dict) -> torch.dtype:
        dtype = group.get("state_dtype")
        return dtype if isinstance(dtype, torch.dtype) else torch.float32

    def _step_muon_group(self, group: dict, params: Sequence[torch.nn.Parameter]) -> None:
        lr = float(group.get("lr", 1e-3))
        weight_decay = float(group.get("weight_decay", 0.0))
        beta = float(group.get("momentum_beta", 0.95))
        nesterov = bool(group.get("nesterov", True))
        ns_steps = int(group.get("ns_steps", 5))
        extra_scale_factor = float(group.get("extra_scale_factor", 1.0))
        state_dtype = self._state_dtype(group)

        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if grad.is_sparse:
                raise RuntimeError("PrecisionAdaptiveHierarchicalOptimizer does not support sparse gradients.")
            if p.ndim != 2:
                raise RuntimeError("Muon-tier parameters must be matrix-shaped.")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(p, dtype=state_dtype)
            buf = state["momentum_buffer"]
            state["step"] += 1

            grad_state = grad.to(buf.dtype)
            buf.mul_(beta).add_(grad_state, alpha=1.0 - beta)
            update_source = buf
            if nesterov:
                update_source = buf * beta + grad_state * (1.0 - beta)

            update = _muon_orthogonalize(update_source, ns_steps=ns_steps, extra_scale_factor=extra_scale_factor)

            if weight_decay:
                p.mul_(1.0 - lr * weight_decay)
            p.add_(update.to(p.dtype), alpha=-lr)

    def _step_rowwise_group(self, group: dict, params: Sequence[torch.nn.Parameter]) -> None:
        lr = float(group.get("lr", 1e-3))
        weight_decay = float(group.get("weight_decay", 0.0))
        betas = group.get("betas", (0.9, 0.95))
        beta1, beta2 = float(betas[0]), float(betas[1])
        eps = float(group.get("eps", 1e-8))
        state_dtype = self._state_dtype(group)

        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if grad.is_sparse:
                raise RuntimeError("PrecisionAdaptiveHierarchicalOptimizer does not support sparse gradients.")
            if p.ndim != 2:
                raise RuntimeError("Row-wise parameters must be matrix-shaped.")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                row_shape = (p.shape[0], 1)
                state["exp_avg_row"] = torch.zeros(row_shape, dtype=state_dtype, device=p.device)
                state["exp_avg_sq_row"] = torch.zeros(row_shape, dtype=state_dtype, device=p.device)
            exp_avg_row = state["exp_avg_row"]
            exp_avg_sq_row = state["exp_avg_sq_row"]
            state["step"] += 1

            grad_float = grad.float()
            row_grad = grad_float.mean(dim=1, keepdim=True)
            row_rms = grad_float.pow(2).mean(dim=1, keepdim=True)

            if weight_decay:
                p.mul_(1.0 - lr * weight_decay)

            exp_avg_row.mul_(beta1).add_(row_grad.to(exp_avg_row.dtype), alpha=1.0 - beta1)
            exp_avg_sq_row.mul_(beta2).add_(row_rms.to(exp_avg_sq_row.dtype), alpha=1.0 - beta2)

            denom = exp_avg_sq_row.float().sqrt().add_(eps)
            row_scale = exp_avg_row.float() / denom
            update = grad_float * row_scale
            p.add_(update.to(p.dtype), alpha=-lr)

    def _step_adamw_group(self, group: dict, params: Sequence[torch.nn.Parameter]) -> None:
        lr = float(group.get("lr", 1e-3))
        weight_decay = float(group.get("weight_decay", 0.0))
        betas = group.get("betas", (0.9, 0.95))
        beta1, beta2 = float(betas[0]), float(betas[1])
        eps = float(group.get("eps", 1e-8))
        state_dtype = self._state_dtype(group)

        for p in params:
            if p.grad is None:
                continue
            grad = p.grad.detach()
            if grad.is_sparse:
                raise RuntimeError("PrecisionAdaptiveHierarchicalOptimizer does not support sparse gradients.")

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=state_dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=state_dtype)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1

            grad_state = grad.to(exp_avg.dtype)
            if weight_decay:
                p.mul_(1.0 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad_state, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_state, grad_state, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            denom = exp_avg_sq.float().sqrt().add_(eps)
            step_size = lr * (bias_correction2 ** 0.5) / max(bias_correction1, 1e-8)
            p.addcdiv_(exp_avg.float(), denom, value=-step_size)
