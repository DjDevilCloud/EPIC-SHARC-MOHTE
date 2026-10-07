"""CUDA repair parity, fallback gradients, and stream/autograd safety."""
import importlib.util
import unittest
from unittest import mock

import torch
from fused_finite import repair_transition
from config import PrismalWaveConfig
from model import PrismalTorusCore, PrismalTorusState


class FiniteBackendConfigTests(unittest.TestCase):
    def test_config_roundtrip_and_legacy_default(self):
        config = PrismalWaveConfig(training_finite_guard_backend='cuda')
        self.assertEqual(PrismalWaveConfig.from_dict(config.to_dict()).training_finite_guard_backend, 'cuda')
        self.assertEqual(PrismalWaveConfig.from_dict({}).training_finite_guard_backend, 'sync')
        with self.assertRaises(ValueError):
            PrismalWaveConfig(training_finite_guard_backend='unknown')


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('cupy'), 'CUDA and CuPy required')
class FusedFiniteTests(unittest.TestCase):
    def test_step_repairs_before_next_transition(self):
        cfg = PrismalWaveConfig(d_model=8, n_paths=1, torus_depth=2, torus_height=2, torus_width=2)
        core = PrismalTorusCore(cfg).cuda().train()
        hidden = torch.randn(2, 8, device='cuda', requires_grad=True)
        state = core.init_state(2, hidden.device)
        raw_hidden = hidden.clone()
        raw_hidden[0, 0] = float('nan')
        raw_field = state.field.clone()
        raw_field.flatten()[0] = float('inf')
        raw_bus = state.bus.clone()
        raw_bus.flatten()[0] = -float('inf')
        raw_stat = torch.tensor(float('nan'), device='cuda', requires_grad=True)
        transition = (raw_hidden, PrismalTorusState(raw_field, raw_bus), {'test_aux': raw_stat}, {})
        with mock.patch.object(core, 'transition', return_value=transition):
            cfg.training_finite_guard_backend = 'sync'
            reference = core.step(hidden, state, path_index=0)
            cfg.training_finite_guard_backend = 'cuda'
            with mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('host sync')):
                actual = core.step(hidden, state, path_index=0)
        torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1].field, reference[1].field, rtol=0, atol=0)
        torch.testing.assert_close(actual[1].bus, reference[1].bus, rtol=0, atol=0)
        for key, value in reference[2].items():
            torch.testing.assert_close(actual[2][key], value, rtol=0, atol=0)
        self.assertEqual(actual[2]['stability_step_repair_count'].item(), 4)
        self.assertEqual(actual[2]['stability_nonfinite_repair_count'].item(), 1)
        # Feed the repaired state into a real transition and differentiate.
        next_hidden, _, stats = core.step(actual[0], actual[1], path_index=0, step_index=1)
        (next_hidden.sum() + stats['torus_entropy']).backward()
        self.assertTrue(torch.isfinite(hidden.grad).all().item())

    def test_repair_and_gradients_match_where(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([1., float('nan'), float('inf'), -float('inf')], device='cuda', dtype=dtype, requires_grad=True)
                fallback = torch.arange(4, device='cuda', dtype=dtype, requires_grad=True)
                field = torch.tensor([2., float('nan')], device='cuda', dtype=dtype, requires_grad=True)
                stat = torch.tensor(float('inf'), device='cuda', dtype=dtype, requires_grad=True)
                entries = [(x, fallback, False), (field, None, False), (stat, None, False)]
                # No tensor host read is allowed in the guard.
                with mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('host sync')):
                    repaired, counts = repair_transition(entries, 2)
                expected = [torch.where(torch.isfinite(x), x, fallback),
                            torch.where(torch.isfinite(field), field, 0.),
                            torch.where(torch.isfinite(stat), stat, 0.)]
                for actual, reference in zip(repaired, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                torch.testing.assert_close(counts, torch.tensor([5, 1], device='cuda'))
                inputs = (x, fallback, field, stat)
                grads = torch.autograd.grad(sum(t.sum() for t in repaired), inputs, retain_graph=True)
                reference_grads = torch.autograd.grad(sum(t.sum() for t in expected), inputs)
                for actual, reference in zip(grads, reference_grads):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                self.assertTrue(torch.isnan(x[1]).item())

    def test_mixed_dtypes_and_noncontiguous(self):
        hidden = torch.randn(3, 4, device='cuda', dtype=torch.bfloat16).T.requires_grad_()
        fallback = torch.randn_like(hidden, requires_grad=True)
        field = torch.tensor([float('inf'), 3.], device='cuda', requires_grad=True)
        stat = torch.tensor(float('nan'), device='cuda')
        values, counts = repair_transition([(hidden, fallback, False), (field, None, False), (stat, None, False)], 2)
        torch.testing.assert_close(values[0], hidden)
        torch.testing.assert_close(counts, torch.tensor([2, 1], device='cuda'))
        sum(v.sum() for v in values).backward()
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
        torch.testing.assert_close(fallback.grad, torch.zeros_like(fallback))

    def test_recurrent_graph_and_external_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            initial = torch.randn(32, device='cuda', requires_grad=True)
            value = initial
            for _ in range(20):
                raw = value.tanh() + .1
                value = repair_transition([(raw, value, False)], 1)[0][0]
            value.sum().backward()
        torch.cuda.current_stream().wait_stream(stream)
        reference = initial.detach().clone().requires_grad_()
        ref_value = reference
        for _ in range(20):
            ref_value = ref_value.tanh() + .1
        ref_value.sum().backward()
        torch.testing.assert_close(value, ref_value)
        torch.testing.assert_close(initial.grad, reference.grad)

    def test_gradcheck_and_double_backward(self):
        x = torch.randn(5, device='cuda', dtype=torch.float64, requires_grad=True)
        f = torch.randn_like(x, requires_grad=True)
        fn = lambda a, b: repair_transition([(a, b, False)], 1)[0][0]
        self.assertTrue(torch.autograd.gradcheck(fn, (x, f)))
        self.assertTrue(torch.autograd.gradgradcheck(fn, (x, f)))


if __name__ == '__main__':
    unittest.main()
