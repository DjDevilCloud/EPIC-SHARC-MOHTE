import unittest
import torch
from native_owner_evidence import NativeOwnerEvidence


class NativeOwnerEvidenceTests(unittest.TestCase):
    def test_completed_declaration_does_not_encode_leading_entity_identity(self):
        seen = []
        def word_vector(word):
            seen.append(word)
            self.assertNotIn(word, ((1,), (2,), (90, 91), (92,)))
            return torch.tensor([float(word[0]), 1., 2., 3.])
        encode = lambda seq: torch.stack([x.mean(0) for x in seq])
        a = [(((1,), (2,), (3,), (4,), (5,)), (0.,) * 4)]
        b = [(((90, 91), (92,), (3,), (4,), (5,)), (1.,) * 4)]
        x = NativeOwnerEvidence.features(a, word_vector, encode, torch.zeros(4))
        y = NativeOwnerEvidence.features(b, word_vector, encode, torch.zeros(4))
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(set(seen), {(3,), (4,), (5,)})

    def test_noisy_or_preserves_legacy_and_causal_access(self):
        from tests.test_native_antecedent_memory import NativeAntecedentTests
        fixture = NativeAntecedentTests()
        with torch.random.fork_rng():
            t, memory, embedding = fixture.fixture()
            bank = memory.antecedent_memory
            bank.context_evidence = NativeOwnerEvidence(8)
            torch.nn.init.constant_(bank.role[-1].bias, -6.)
            torch.nn.init.constant_(bank.context_evidence.classifier[-1].bias, 6.)
            with torch.no_grad():
                a = fixture.run_memory(t, memory, embedding, 'Quasmerin is the subject. He was born in Oakfield. Lumavex is the subject.')
                b = fixture.run_memory(t, memory, embedding, 'Quasmerin is the subject. He was born in Oakfield. Tavindor is the subject.')
                access = a.cached_packet[3]['antecedent']
                expected = 1 - (1 - access['legacy_role_logits'].sigmoid()) * (1 - access['evidence_logits'].sigmoid())
                torch.testing.assert_close(access['role_logits'].sigmoid(), expected)
                self.assertTrue(torch.all(access['role_logits'] >= access['legacy_role_logits']))
                for x, y in zip(access['ownership'][:8], b.cached_packet[3]['antecedent']['ownership'][:8]):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            memory.train()
            state = fixture.run_memory(t, memory, embedding, 'Quasmerin is the subject. He was born in Oakfield.')
            def vector(word):
                return memory.word_encoder(embedding(torch.tensor(word))[None])[1][-1, 0]
            _, access = bank.access(state, state.question, vector, memory.context_encoder, memory.encode_sequences)
            access['role_logits'].sum().backward()
            grad = bank.context_evidence.classifier[-1].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.)
