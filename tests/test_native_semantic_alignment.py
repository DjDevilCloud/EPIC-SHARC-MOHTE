import unittest
import torch
from torch import nn
from native_semantic_alignment import NativeSemanticAlignment


class SemanticAlignmentTests(unittest.TestCase):
    def test_shared_learning_and_question_dependence(self):
        torch.manual_seed(91)
        layer = NativeSemanticAlignment(8)
        embedding = nn.Embedding(16, 8)
        inputs = layer.inputs([(1, 2), (3,), (4, 5)], [(6,), (1, 2)], embedding)
        torch.testing.assert_close(layer(inputs), torch.zeros(3))
        nn.init.normal_(layer.score[-1].weight, std=.2)
        logits = layer(inputs)
        loss = -logits.log_softmax(0)[1]
        loss.backward()
        self.assertGreater(float(layer.word.weight_ih_l0.grad.norm()), 0.)
        self.assertGreater(float(layer.context.weight_ih_l0.grad.norm()), 0.)
        self.assertGreater(float(layer.attention.weight.grad.norm()), 0.)
        changed = layer.inputs([(1, 2), (3,), (4, 5)], [(7,), (8,)], embedding)
        self.assertFalse(torch.allclose(logits - logits.mean(), layer(changed) - layer(changed).mean()))

    def test_padding_does_not_change_word_encoding(self):
        torch.manual_seed(92)
        layer = NativeSemanticAlignment(8)
        nn.init.normal_(layer.score[-1].weight, std=.2)
        embedding = nn.Embedding(16, 8)
        inputs = layer.inputs([(1, 2), (3,)], [(4,)], embedding)
        padded = dict(inputs, units=torch.nn.functional.pad(inputs['units'], (0, 0, 0, 3)))
        torch.testing.assert_close(layer(inputs), layer(padded))
        with self.assertRaises(ValueError):
            layer.inputs([], [(4,)], embedding)

    def test_prior_scaling_preserves_initial_behavior_and_learns(self):
        layer = NativeSemanticAlignment(8, prior_scaling=True)
        inputs = layer.inputs([(1,), (2,), (3,)], [(4,), (5,)], nn.Embedding(8, 8))
        prior = torch.tensor([50., -10., -20.])
        correction = layer(inputs, prior)
        torch.testing.assert_close(correction, torch.zeros_like(prior), atol=0., rtol=0.)
        (-(prior + correction).log_softmax(0)[2]).backward()
        self.assertGreater(float(layer.prior_scale.weight.grad.norm()), 0.)
        with torch.no_grad():
            layer.prior_scale.bias.fill_(-3.)
            scaled = prior + layer(inputs, prior)
            self.assertLess(float(scaled.max() - scaled.min()), float(prior.max() - prior.min()))
        with self.assertRaises(ValueError):
            layer(inputs)


if __name__ == '__main__':
    unittest.main()
