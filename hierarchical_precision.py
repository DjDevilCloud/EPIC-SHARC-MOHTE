# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


_CURRENT_PRECISION_SPEC: ContextVar[Optional["HierarchicalPrecisionSpec"]] = ContextVar(
    "prismal_current_precision_spec",
    default=None,
)


def _normalize_dtype_name(value: Any) -> str:
    if isinstance(value, torch.dtype):
        text = str(value)
    else:
        text = str(value or "").strip()
    text = text.lower()
    if text.startswith("torch."):
        text = text.split(".", 1)[1]
    aliases = {
        "half": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "float8": "float8_e4m3fn",
        "f8": "float8_e4m3fn",
    }
    return aliases.get(text, text or "float32")


def dtype_from_name(value: Any) -> torch.dtype:
    name = _normalize_dtype_name(value)
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "float8_e5m2" and hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2
    if name == "float8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return torch.float32


def dtype_name(dtype: Any) -> str:
    if isinstance(dtype, str):
        return _normalize_dtype_name(dtype)
    if dtype is None:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    if hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2:
        return "float8_e5m2"
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    return _normalize_dtype_name(str(dtype))


def is_float8_dtype(dtype: Any) -> bool:
    if hasattr(torch, "float8_e4m3fn") and dtype == torch.float8_e4m3fn:
        return True
    if hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2:
        return True
    return _normalize_dtype_name(dtype).startswith("float8")


def supports_bfloat16(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return False


def supports_float8(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    if not torch.cuda.is_available():
        return False
    return True


def _resolve_autocast_dtype(requested: torch.dtype, device: torch.device, fallback: torch.dtype) -> torch.dtype:
    if device.type != "cuda":
        if requested == torch.bfloat16:
            return torch.bfloat16
        return torch.float32
    if is_float8_dtype(requested):
        return fallback if fallback != torch.float32 else torch.bfloat16 if supports_bfloat16(device) else torch.float16
    if requested == torch.bfloat16:
        return torch.bfloat16 if supports_bfloat16(device) else torch.float16
    if requested == torch.float16:
        return torch.float16
    return requested


def _resolve_state_dtype(requested: torch.dtype, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if requested == torch.bfloat16 and supports_bfloat16(device):
        return torch.bfloat16
    if requested == torch.float16:
        return torch.float16
    return torch.float32 if requested == torch.float32 else requested


@dataclass(frozen=True)
class HierarchicalPrecisionSpec:
    level: int
    hierarchical_depth: int
    tier: str
    module_path: str
    module_kind: str
    requested_compute_dtype: torch.dtype
    effective_compute_dtype: torch.dtype
    accumulator_dtype: torch.dtype
    fallback_dtype: torch.dtype
    float8_requested: bool
    float8_supported: bool
    mode: str

    @property
    def can_attempt_float8(self) -> bool:
        return bool(self.float8_requested and self.float8_supported)

    @property
    def autocast_dtype(self) -> torch.dtype:
        return self.effective_compute_dtype

    def to_state_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("requested_compute_dtype", "effective_compute_dtype", "accumulator_dtype", "fallback_dtype"):
            payload[key] = dtype_name(payload[key])
        return payload

    @classmethod
    def from_state_dict(cls, payload: Dict[str, Any]) -> "HierarchicalPrecisionSpec":
        return cls(
            level=int(payload.get("level", 0)),
            hierarchical_depth=int(payload.get("hierarchical_depth", 1)),
            tier=str(payload.get("tier", "root")),
            module_path=str(payload.get("module_path", "")),
            module_kind=str(payload.get("module_kind", "nest")),
            requested_compute_dtype=dtype_from_name(payload.get("requested_compute_dtype", "float32")),
            effective_compute_dtype=dtype_from_name(payload.get("effective_compute_dtype", "float32")),
            accumulator_dtype=dtype_from_name(payload.get("accumulator_dtype", "float32")),
            fallback_dtype=dtype_from_name(payload.get("fallback_dtype", "float32")),
            float8_requested=bool(payload.get("float8_requested", False)),
            float8_supported=bool(payload.get("float8_supported", False)),
            mode=str(payload.get("mode", "disabled")),
        )


@dataclass
class HierarchicalPrecisionPolicy:
    enabled: bool = True
    root_compute_dtype: str = "bf16"
    mid_compute_dtype: str = "fp16"
    leaf_compute_dtype: str = "float8_e4m3fn"
    fallback_compute_dtype: str = "bf16"
    accumulator_dtype: str = "bf16"
    allow_float8_leaf: bool = True

    @classmethod
    def from_config(cls, cfg: Any) -> "HierarchicalPrecisionPolicy":
        return cls(
            enabled=bool(getattr(cfg, "hierarchical_precision_enabled", True)),
            root_compute_dtype=str(getattr(cfg, "hierarchical_precision_root_dtype", "bf16")),
            mid_compute_dtype=str(getattr(cfg, "hierarchical_precision_mid_dtype", "fp16")),
            leaf_compute_dtype=str(getattr(cfg, "hierarchical_precision_leaf_dtype", "float8_e4m3fn")),
            fallback_compute_dtype=str(getattr(cfg, "hierarchical_precision_fallback_dtype", "bf16")),
            accumulator_dtype=str(getattr(cfg, "hierarchical_precision_accumulator_dtype", "bf16")),
            allow_float8_leaf=bool(getattr(cfg, "hierarchical_precision_allow_float8_leaf", True)),
        )

    @classmethod
    def from_state_dict(cls, payload: Dict[str, Any]) -> "HierarchicalPrecisionPolicy":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            root_compute_dtype=str(payload.get("root_compute_dtype", "bf16")),
            mid_compute_dtype=str(payload.get("mid_compute_dtype", "fp16")),
            leaf_compute_dtype=str(payload.get("leaf_compute_dtype", "float8_e4m3fn")),
            fallback_compute_dtype=str(payload.get("fallback_compute_dtype", "bf16")),
            accumulator_dtype=str(payload.get("accumulator_dtype", "bf16")),
            allow_float8_leaf=bool(payload.get("allow_float8_leaf", True)),
        )

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "root_compute_dtype": _normalize_dtype_name(self.root_compute_dtype),
            "mid_compute_dtype": _normalize_dtype_name(self.mid_compute_dtype),
            "leaf_compute_dtype": _normalize_dtype_name(self.leaf_compute_dtype),
            "fallback_compute_dtype": _normalize_dtype_name(self.fallback_compute_dtype),
            "accumulator_dtype": _normalize_dtype_name(self.accumulator_dtype),
            "allow_float8_leaf": bool(self.allow_float8_leaf),
        }

    def progressive_qat_policy(
        self,
        progress: float,
        *,
        start_fraction: float = 0.65,
        ramp_fraction: float = 0.20,
    ) -> "HierarchicalPrecisionPolicy":
        """Return a staged policy for quantization-aware training.

        The schedule is intentionally small:
        - before `start_fraction`, keep leaf and mid tiers on bf16 for warmup
        - during the ramp, restore the configured mid tier first
        - at the end of the ramp, restore the configured leaf tier
        """
        if not self.enabled:
            return self

        progress = max(0.0, min(1.0, float(progress)))
        start_fraction = max(0.0, min(1.0, float(start_fraction)))
        ramp_fraction = max(0.0, float(ramp_fraction))
        if ramp_fraction <= 0.0:
            return self if progress >= start_fraction else replace(self, mid_compute_dtype="bf16", leaf_compute_dtype="bf16")

        mid_restore = min(1.0, start_fraction + (ramp_fraction * 0.5))
        leaf_restore = min(1.0, start_fraction + ramp_fraction)

        if progress < start_fraction:
            return replace(self, mid_compute_dtype="bf16", leaf_compute_dtype="bf16", allow_float8_leaf=False)
        if progress < mid_restore:
            return replace(self, leaf_compute_dtype="bf16", allow_float8_leaf=False)
        if progress < leaf_restore:
            return replace(self, allow_float8_leaf=False)
        return self

    def resolve_for_level(
        self,
        level: int,
        hierarchical_depth: int,
        device: torch.device,
        *,
        is_leaf: bool = False,
        module_path: str = "",
        module_kind: str = "nest",
    ) -> HierarchicalPrecisionSpec:
        if not self.enabled:
            fp32 = torch.float32
            return HierarchicalPrecisionSpec(
                level=int(level),
                hierarchical_depth=max(1, int(hierarchical_depth)),
                tier="disabled",
                module_path=module_path,
                module_kind=module_kind,
                requested_compute_dtype=fp32,
                effective_compute_dtype=fp32,
                accumulator_dtype=fp32,
                fallback_dtype=fp32,
                float8_requested=False,
                float8_supported=False,
                mode="disabled",
            )

        tier = "root" if int(level) <= 0 else "leaf" if bool(is_leaf) else "mid"
        requested_name = self.root_compute_dtype if tier == "root" else self.leaf_compute_dtype if tier == "leaf" else self.mid_compute_dtype
        requested = dtype_from_name(requested_name)
        fallback = _resolve_autocast_dtype(dtype_from_name(self.fallback_compute_dtype), device, torch.float32)
        accumulator = _resolve_state_dtype(dtype_from_name(self.accumulator_dtype), device)

        float8_requested = is_float8_dtype(requested)
        float8_supported = bool(float8_requested and supports_float8(device))

        if module_kind in {"state", "recurrent_state"}:
            effective = accumulator
            mode = "state"
        elif tier == "leaf" and self.allow_float8_leaf:
            effective = _resolve_autocast_dtype(dtype_from_name(self.fallback_compute_dtype), device, torch.float32)
            mode = "leaf-float8" if float8_supported else "leaf-fallback"
        else:
            effective = _resolve_autocast_dtype(requested, device, fallback)
            mode = tier

        if effective == torch.float32 and device.type == "cuda" and requested == torch.bfloat16 and supports_bfloat16(device):
            effective = torch.bfloat16

        return HierarchicalPrecisionSpec(
            level=int(level),
            hierarchical_depth=max(1, int(hierarchical_depth)),
            tier=tier,
            module_path=module_path,
            module_kind=module_kind,
            requested_compute_dtype=requested,
            effective_compute_dtype=effective,
            accumulator_dtype=accumulator,
            fallback_dtype=fallback,
            float8_requested=float8_requested,
            float8_supported=float8_supported,
            mode=mode,
        )

    def state_dtype_for_device(self, device: torch.device) -> torch.dtype:
        return _resolve_state_dtype(dtype_from_name(self.accumulator_dtype), device)

    @contextmanager
    def scope(
        self,
        spec: Optional[HierarchicalPrecisionSpec],
        *,
        device: torch.device,
        enabled: bool = True,
    ):
        if not enabled or not self.enabled or spec is None:
            yield
            return

        token = _CURRENT_PRECISION_SPEC.set(spec)
        try:
            autocast_dtype = spec.autocast_dtype
            if device.type == "cuda" and autocast_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
                    yield
            elif device.type == "cpu" and autocast_dtype == torch.bfloat16:
                with torch.autocast("cpu", dtype=autocast_dtype, enabled=True):
                    yield
            else:
                yield
        finally:
            _CURRENT_PRECISION_SPEC.reset(token)

    @contextmanager
    def training_context(self, device: torch.device, *, enabled: bool = True):
        spec = self.resolve_for_level(0, 1, device, is_leaf=False, module_path="root", module_kind="root")
        with self.scope(spec, device=device, enabled=enabled):
            yield


def current_precision_spec() -> Optional[HierarchicalPrecisionSpec]:
    return _CURRENT_PRECISION_SPEC.get()


def attach_precision_policy(module: nn.Module, policy: HierarchicalPrecisionPolicy) -> None:
    for child in module.modules():
        setattr(child, "precision_policy", policy)
