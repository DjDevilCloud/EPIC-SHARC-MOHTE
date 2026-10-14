import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory


class NativeQuestionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.previous_rng = torch.random.get_rng_state()

    def tearDown(self):
        torch.random.set_rng_state(self.previous_rng)

    def fixture(self):
        torch.manual_seed(9)
        tokenizer = PrismalTokenizer()
        ids = torch.tensor([tokenizer.encode('<BOI>Context:\nThe apple is red. The pear is blue.\nQuestion:\nWhat color is the apple?<EOI><BOO>red<EOO>', add_special_tokens=True)])
        embedding = nn.Embedding(len(tokenizer.construction_units), 8)
        memory = NativeQuestionMemory(8, capacity=3)
        return tokenizer, ids, embedding, memory

    def test_zero_initialization_and_bounded_owned_occurrences(self):
        t, ids, embedding, memory = self.fixture()
        hidden = torch.randn(1, ids.shape[1], 8)
        actual, states = memory(ids, hidden, embedding, t)
        torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
        self.assertEqual(len(states[0].source), 3)
        decoded = lambda word: t.decode(list(word), clean_text=False, collapse_structure=False).casefold()
        value, clause = states[0].source[-1]
        self.assertEqual(decoded(value), 'blue')
        self.assertIn('pear', [decoded(word) for word in clause])
        self.assertIn('apple', [decoded(word) for word in states[0].question])

    def test_incremental_matches_full_and_does_not_mutate_predecessor(self):
        t, ids, embedding, memory = self.fixture()
        nn.init.normal_(memory.output[-1].weight, std=.1)
        hidden = torch.randn(1, ids.shape[1], 8)
        whole, _ = memory(ids, hidden, embedding, t)
        state = None
        pieces = []
        for index in range(ids.shape[1]):
            previous = repr(state)
            piece, next_state = memory(ids[:, index:index + 1], hidden[:, index:index + 1], embedding, t, state)
            self.assertEqual(repr(state), previous)
            pieces.append(piece)
            state = next_state
        torch.testing.assert_close(torch.cat(pieces, 1), whole, rtol=1e-5, atol=1e-6)

    def test_future_input_does_not_change_prefix_and_new_turn_clears_memory(self):
        t, ids, embedding, memory = self.fixture()
        nn.init.normal_(memory.output[-1].weight, std=.1)
        hidden = torch.randn(1, ids.shape[1], 8)
        end = ids[0].tolist().index(t.special_tokens['<BOO>']) + 1
        full, state = memory(ids, hidden, embedding, t)
        prefix, _ = memory(ids[:, :end], hidden[:, :end], embedding, t)
        torch.testing.assert_close(prefix, full[:, :end])
        _, reset = memory(torch.tensor([[t.special_tokens['<BOI>']]]), hidden[:, :1], embedding, t, state)
        self.assertFalse(reset[0].question)
        self.assertFalse(reset[0].source)

    def test_attention_and_question_encoders_receive_gradients(self):
        t, ids, embedding, memory = self.fixture()
        nn.init.normal_(memory.output[-1].weight, std=.1)
        actual, _ = memory(ids, torch.randn(1, ids.shape[1], 8), embedding, t)
        actual.square().mean().backward()
        for module in (memory.word_encoder, memory.context_encoder, memory.question_encoder, memory.query, memory.key):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_model_reload_and_carried_question_state(self):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        from train import save_checkpoint, load_bundle_from_checkpoint
        _, t, m = RuntimeSignatureTests().setup_model()
        m.cfg.freeze_registry_observations = True
        m.registry.observation_updates_enabled = False
        ids = torch.tensor([t.prepare_generation_hierarchy('Context:\nThe apple is red.\nQuestion:\nWhat color is the apple?').token_ids])
        with torch.no_grad():
            baseline = m(ids).logits
        m.cfg.use_native_question_memory = True
        m.native_question_memory = NativeQuestionMemory(m.cfg.d_model)
        with torch.no_grad():
            first = m(ids)
        torch.testing.assert_close(first.logits, baseline, rtol=0, atol=0)
        state = first.signature_lattice_state.native_question_state
        self.assertTrue(state[0].question)
        marker = torch.tensor([[t.encode('red', add_special_tokens=False)[0]]])
        with torch.no_grad():
            second = m(marker, slot_state=first.slot_state, signature_lattice_state=first.signature_lattice_state, runtime_signature_state=first.runtime_signature_state, position_index=ids.shape[1])
        self.assertEqual(second.signature_lattice_state.native_question_state[0].question, state[0].question)
        nn.init.normal_(m.native_question_memory.output[-1].weight, std=.01)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(m, directory, tokenizer=t)
            loaded, _, cfg = load_bundle_from_checkpoint(Path(directory) / 'model.pt', device='cpu', load_training_state=False)
            loaded.eval()
            self.assertTrue(cfg.use_native_question_memory)
            with torch.no_grad():
                torch.testing.assert_close(loaded(ids).logits, m(ids).logits, rtol=1e-5, atol=1e-6)

    def test_passing_identity_reader_bypasses_native_adapter(self):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        fixture = RuntimeSignatureTests()
        fixture.identity_readout = True
        _, t, m = fixture.setup_model()
        m.cfg.use_native_question_memory = True
        m.native_question_memory = NativeQuestionMemory(m.cfg.d_model)
        ids = torch.tensor([t.prepare_generation_hierarchy('Context:\nThe apple is red.\nQuestion:\nWhat color is the apple?').token_ids])
        with patch.object(m.native_question_memory, 'forward', side_effect=AssertionError('Passing reader must bypass native adapter')):
            m(ids)

    def test_inference_reuses_source_and_question_encodings(self):
        t, ids, embedding, memory = self.fixture()
        memory.eval()
        end = ids[0].tolist().index(t.special_tokens['<BOO>']) + 1
        with torch.no_grad():
            _, state = memory(ids[:, :end], torch.randn(1, end, 8), embedding, t)
            self.assertIsNotNone(state[0].cached_packet)
            with patch.object(memory.word_encoder, 'forward', side_effect=AssertionError('Source replay during incremental output')):
                _, next_state = memory(ids[:, end:end + 1], torch.randn(1, 1, 8), embedding, t, state)
            self.assertIs(next_state[0].cached_packet, state[0].cached_packet)

    def test_ordinary_header_words_are_retained_without_a_colon(self):
        t = PrismalTokenizer()
        ids = torch.tensor([t.encode('<BOI>Context:\nContext matters.\nQuestion wording matters.\nQuestion:\nWhat matters?<EOI><BOO>', add_special_tokens=False)])
        memory = NativeQuestionMemory(8)
        _, state = memory(ids, torch.randn(1, ids.shape[1], 8), nn.Embedding(len(t.construction_units), 8), t)
        words = [t.decode(list(word), clean_text=False, collapse_structure=False).casefold() for word, _ in state[0].source]
        self.assertEqual(words, ['context', 'matters', 'question', 'wording', 'matters'])


if __name__ == '__main__':
    unittest.main()
