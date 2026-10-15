import unittest
import torch
from native_span_readout import NativeSpanReadout


class NativeSpanMixturePrecisionTests(unittest.TestCase):
    def test_saturated_copy_gate_keeps_native_loss_and_gradient(self):
        decoder = NativeSpanReadout(4)
        torch.nn.init.constant_(decoder.copy_gate.bias, 80.)
        logits = torch.zeros(1, 1, 6, requires_grad=True)
        plan = dict(candidates=[dict(tokens=(2,))], probabilities=torch.tensor([1.]), log_probabilities=torch.tensor([0.]))
        out = decoder.mix(logits, [(0, 0, torch.zeros(4), torch.zeros(4), plan, ())])
        loss = -out[0, 0, 5]
        self.assertGreater(float(loss), 80.)
        loss.backward()
        self.assertLess(float(logits.grad[0, 0, 5]), -.8)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_sub_epsilon_unsupported_mass_is_not_lost(self):
        decoder = NativeSpanReadout(4)
        torch.nn.init.constant_(decoder.copy_gate.bias, 80.)
        log_weights = torch.tensor([0., -18.42068074]).log_softmax(0)
        plan = dict(candidates=[dict(tokens=(1, 2)), dict(tokens=(3, 4))], probabilities=log_weights.exp(), log_probabilities=log_weights)
        out = decoder.mix(torch.zeros(1, 1, 6), [(0, 0, torch.zeros(4), torch.zeros(4), plan, (1,))])
        torch.testing.assert_close(out[0, 0, 5], log_weights[1] - torch.tensor(6.).log(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out.exp().sum(), torch.tensor(1.), rtol=1e-6, atol=1e-6)
