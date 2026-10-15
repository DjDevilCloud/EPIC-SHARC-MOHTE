import unittest
import torch
from native_relation_alignment import NativeRelationAlignment


class NativeRelationAlignmentTests(unittest.TestCase):
    def test_neutral_initialization_and_source_gradients(self):
        with torch.random.fork_rng():
            torch.manual_seed(17)
            head = NativeRelationAlignment(8)
            words, phrases, original = torch.randn(5, 8), torch.randn(9, 8), torch.randn(8)
            source = torch.randn(4, 8)
            request = head(words, phrases, original)
            torch.testing.assert_close(request, original, atol=0., rtol=0.)
            scores = source @ request
            torch.testing.assert_close(scores, source @ original)
            (-scores.log_softmax(0)[2]).backward()
            self.assertGreater(float(head.output[-1].weight.grad.abs().sum()), 0.)

    def test_pooling_preserves_evidence_without_using_final_position(self):
        with torch.random.fork_rng():
            torch.manual_seed(19)
            head = NativeRelationAlignment(8)
            words, phrases = torch.randn(5, 8), torch.randn(9, 8)
            original = torch.randn(8)
            torch.nn.init.normal_(head.output[-1].weight, std=.1)
            expected = head(words, phrases, original)
            # Reordering already-encoded evidence leaves its learned pool intact.
            torch.testing.assert_close(head(words.flip(0), phrases.flip(0), original), expected)
            # Ordered phrase content remains an independent semantic channel.
            changed = head(words, phrases + 1., original)
            self.assertFalse(torch.allclose(changed, expected))

    def test_state_reload_preserves_scores(self):
        with torch.random.fork_rng():
            a, b = NativeRelationAlignment(8), NativeRelationAlignment(8)
            torch.nn.init.normal_(a.output[-1].weight, std=.1)
            b.load_state_dict(a.state_dict())
            inputs = torch.randn(5, 8), torch.randn(9, 8), torch.randn(8)
            torch.testing.assert_close(a(*inputs), b(*inputs), atol=0., rtol=0.)
