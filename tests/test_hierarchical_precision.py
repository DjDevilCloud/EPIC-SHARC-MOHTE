import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PrismalWaveConfig
from hierarchical_precision import HierarchicalPrecisionPolicy
from train import _amp_scaler_enabled


class HierarchicalPrecisionTests(unittest.TestCase):
    def _make_policy(self, *, fallback_dtype: str = "bf16") -> HierarchicalPrecisionPolicy:
        cfg = SimpleNamespace(
            hierarchical_precision_enabled=True,
            hierarchical_precision_root_dtype="float8_e4m3fn",
            hierarchical_precision_mid_dtype="float8_e4m3fn",
            hierarchical_precision_leaf_dtype="float8_e4m3fn",
            hierarchical_precision_fallback_dtype=fallback_dtype,
            hierarchical_precision_accumulator_dtype="bf16",
            hierarchical_precision_allow_float8_leaf=True,
        )
        return HierarchicalPrecisionPolicy.from_config(cfg)

    def test_float8_request_stays_float8_when_supported(self) -> None:
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("float8 dtype not available in this torch build")

        policy = self._make_policy()
        device = torch.device("cuda")

        with mock.patch("hierarchical_precision.supports_float8", return_value=True), mock.patch(
            "hierarchical_precision.supports_bfloat16", return_value=True
        ):
            spec = policy.resolve_for_level(0, 1, device, is_leaf=False, module_path="root", module_kind="root")

        self.assertEqual(spec.requested_compute_dtype, torch.float8_e4m3fn)
        self.assertEqual(spec.effective_compute_dtype, torch.float8_e4m3fn)
        self.assertTrue(spec.can_attempt_float8)

    def test_float8_request_falls_back_when_unsupported_even_if_fallback_is_float8(self) -> None:
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("float8 dtype not available in this torch build")

        policy = self._make_policy(fallback_dtype="float8_e4m3fn")
        device = torch.device("cuda")

        with mock.patch("hierarchical_precision.supports_float8", return_value=False), mock.patch(
            "hierarchical_precision.supports_bfloat16", return_value=True
        ):
            spec = policy.resolve_for_level(0, 1, device, is_leaf=False, module_path="root", module_kind="root")

        self.assertEqual(spec.requested_compute_dtype, torch.float8_e4m3fn)
        self.assertEqual(spec.effective_compute_dtype, torch.bfloat16)
        self.assertFalse(spec.can_attempt_float8)

    def test_config_normalizes_supported_dtype_aliases(self) -> None:
        cfg = PrismalWaveConfig()
        cfg.bitsandbytes_leaf_compute_dtype = "torch.float32"
        cfg.hierarchical_precision_root_dtype = "torch.float8"
        cfg.hierarchical_precision_mid_dtype = "fp16"
        cfg.hierarchical_precision_leaf_dtype = "float8_e5m2"
        cfg.hierarchical_precision_fallback_dtype = "bf16"
        cfg.hierarchical_precision_accumulator_dtype = "torch.float32"
        cfg.__post_init__()

        self.assertEqual(cfg.bitsandbytes_leaf_compute_dtype, "float32")
        self.assertEqual(cfg.hierarchical_precision_root_dtype, "float8_e4m3fn")
        self.assertEqual(cfg.hierarchical_precision_mid_dtype, "float16")
        self.assertEqual(cfg.hierarchical_precision_leaf_dtype, "float8_e5m2")
        self.assertEqual(cfg.hierarchical_precision_fallback_dtype, "bfloat16")
        self.assertEqual(cfg.hierarchical_precision_accumulator_dtype, "float32")

    def test_bitsandbytes_compute_dtype_rejects_float8(self) -> None:
        cfg = PrismalWaveConfig()
        cfg.bitsandbytes_leaf_compute_dtype = "float8_e4m3fn"
        cfg.__post_init__()

        self.assertEqual(cfg.bitsandbytes_leaf_compute_dtype, "bfloat16")

    def test_transformer_engine_recipe_is_locked_to_nvfp4(self) -> None:
        cfg = PrismalWaveConfig()
        cfg.use_transformer_engine_leaf_precision = True
        cfg.transformer_engine_leaf_recipe = "fp4"
        cfg.transformer_engine_leaf_params_dtype = "torch.bfloat16"
        cfg.__post_init__()

        self.assertTrue(cfg.use_transformer_engine_leaf_precision)
        self.assertEqual(cfg.transformer_engine_leaf_recipe, "nvfp4")
        self.assertEqual(cfg.transformer_engine_leaf_params_dtype, "bfloat16")

    def test_amp_scaler_is_disabled_for_bf16_and_float8_precision(self) -> None:
        device = torch.device("cuda")
        bf16_policy = HierarchicalPrecisionPolicy(
            enabled=True,
            root_compute_dtype="bfloat16",
            mid_compute_dtype="bfloat16",
            leaf_compute_dtype="bfloat16",
            fallback_compute_dtype="bfloat16",
            accumulator_dtype="bfloat16",
            allow_float8_leaf=False,
        )
        float8_policy = HierarchicalPrecisionPolicy(
            enabled=True,
            root_compute_dtype="float8_e4m3fn",
            mid_compute_dtype="float8_e4m3fn",
            leaf_compute_dtype="float8_e4m3fn",
            fallback_compute_dtype="bfloat16",
            accumulator_dtype="bfloat16",
            allow_float8_leaf=True,
        )

        self.assertFalse(_amp_scaler_enabled(bf16_policy, device, use_amp=True))
        self.assertFalse(_amp_scaler_enabled(float8_policy, device, use_amp=True))


if __name__ == "__main__":
    unittest.main()
