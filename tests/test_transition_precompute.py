"""Token-local preparation preserves recurrence and its gradients."""
import copy
import unittest

import torch
from config import PrismalWaveConfig
from model import PrismalTorusCore
from quantization import QuantizationConfig


class TransitionPrecomputeTests(unittest.TestCase):
    def test_scan_and_gradients_match(self):
        torch.manual_seed(27)
        torch.set_num_threads(1)
        cfg = PrismalWaveConfig(d_model=8, n_paths=1, torus_depth=2, torus_height=2, torus_width=2)
        baseline = PrismalTorusCore(cfg, QuantizationConfig(enabled=False)).double().train()
        optimized = copy.deepcopy(baseline)
        hidden = torch.randn(2, 5, 8, dtype=torch.float64)
        contexts = {name: torch.randn_like(hidden) for name in
                    ('family_context', 'level_context', 'relation_context', 'parent_context')}

        def scan(core, prepare):
            local = hidden.clone().requires_grad_()
            context = {k: v.clone().requires_grad_() for k, v in contexts.items()}
            prepared = core.prepare_transition_sequence(local, path_index=0, **context) if prepare else None
            state = core.init_state(2, local.device)
            outputs, terms = [], []
            for index in range(local.size(1)):
                selected = None if prepared is None else {
                    k: v if k in {'active_radius', 'active_offsets'} else v[:, index]
                    for k, v in prepared.items()}
                out, state, stats = core.step(local[:, index], state, path_index=0, step_index=index,
                                             prepared_inputs=selected, **{k: v[:, index] for k, v in context.items()})
                outputs.append(out)
                terms.append(stats['torus_entropy'] + stats['torus_coverage_loss'] + stats['emitter_cell_mixture_loss'])
            outputs = torch.stack(outputs, 1)
            loss = outputs.square().mean() + sum(terms)
            loss.backward()
            gradients = {k: p.grad for k, p in core.named_parameters() if p.grad is not None}
            gradients['input'] = local.grad
            gradients.update({k: v.grad for k, v in context.items()})
            return outputs, state, gradients

        left, left_state, left_grad = scan(baseline, False)
        right, right_state, right_grad = scan(optimized, True)
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(left_state.field, right_state.field, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(left_state.bus, right_state.bus, atol=1e-7, rtol=1e-6)
        self.assertEqual(left_grad.keys(), right_grad.keys())
        for key in left_grad:
            torch.testing.assert_close(left_grad[key], right_grad[key], atol=1e-6, rtol=1e-5, msg=key)

    def test_future_tokens_do_not_change_prepared_prefix(self):
        cfg = PrismalWaveConfig(d_model=8, n_paths=1, torus_depth=2, torus_height=2, torus_width=2)
        core = PrismalTorusCore(cfg, QuantizationConfig(enabled=False))
        hidden = torch.randn(2, 5, 8)
        changed = hidden.clone()
        changed[:, 3:] += 10
        left = core.prepare_transition_sequence(hidden, path_index=0)
        right = core.prepare_transition_sequence(changed, path_index=0)
        for key, value in left.items():
            if key not in {'active_radius', 'active_offsets'}:
                torch.testing.assert_close(value[:, :3], right[key][:, :3], atol=0, rtol=0)

    def test_config_roundtrip_and_disable(self):
        cfg = PrismalWaveConfig()
        self.assertTrue(cfg.training_precompute_torus_inputs)
        cfg.training_precompute_torus_inputs = False
        self.assertFalse(PrismalWaveConfig.from_dict(cfg.to_dict()).training_precompute_torus_inputs)
        self.assertTrue(cfg.training_precompute_torus_metadata)
        cfg.training_precompute_torus_metadata = False
        self.assertFalse(PrismalWaveConfig.from_dict(cfg.to_dict()).training_precompute_torus_metadata)

    def test_metadata_matches_previous_prepared_path(self):
        cfg = PrismalWaveConfig(d_model=8, n_paths=1, torus_depth=2, torus_height=3, torus_width=2)
        core = PrismalTorusCore(cfg, QuantizationConfig(enabled=False)).train()
        hidden = torch.randn(3, 4, 8)
        cfg.training_precompute_torus_metadata = False
        previous = core.prepare_transition_sequence(hidden, path_index=0)
        cfg.training_precompute_torus_metadata = True
        current = core.prepare_transition_sequence(hidden, path_index=0)

        def token(values, index):
            return {key: value if key in {'active_offsets', 'active_radius'} else value[:, index]
                    for key, value in values.items()}

        for index in range(4):
            state = core.init_state(3, hidden.device)
            left = core.step(hidden[:, index], state, path_index=0, step_index=index,
                             prepared_inputs=token(previous, index))
            right = core.step(hidden[:, index], state, path_index=0, step_index=index,
                              prepared_inputs=token(current, index))
            torch.testing.assert_close(left[0], right[0])
            torch.testing.assert_close(left[1].field, right[1].field)
            torch.testing.assert_close(left[1].bus, right[1].bus)
            for key in left[2]:
                torch.testing.assert_close(left[2][key], right[2][key], msg=key)

    def test_multitoken_context_summary_is_preserved(self):
        cfg = PrismalWaveConfig(d_model=8, n_paths=1, torus_depth=2, torus_height=2, torus_width=2)
        core = PrismalTorusCore(cfg, QuantizationConfig(enabled=False))
        hidden = torch.randn(2, 8)
        state = core.init_state(2, hidden.device)
        context = torch.randn(2, 3, 8)
        left = core.transition(hidden, state, path_index=0, family_context=context)
        right = core.transition(hidden, state, path_index=0,
                                family_context=core._summarize_chunk_context(context))
        torch.testing.assert_close(left[0], right[0], atol=0, rtol=0)

    def test_hooks_and_quantized_layers_disable_preparation(self):
        cfg = PrismalWaveConfig(d_model=8, torus_depth=2, torus_height=2, torus_width=2)
        core = PrismalTorusCore(cfg, QuantizationConfig(enabled=False))
        self.assertTrue(core.can_prepare_transition_sequence())
        hook = core.write_gate.register_forward_hook(lambda *args: None)
        self.assertFalse(core.can_prepare_transition_sequence())
        hook.remove()
        self.assertFalse(PrismalTorusCore(cfg, QuantizationConfig(enabled=True)).can_prepare_transition_sequence())


if __name__ == '__main__':
    unittest.main()
