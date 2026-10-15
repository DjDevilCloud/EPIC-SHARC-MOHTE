import unittest
import torch
from torch import nn
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory
from native_span_readout import NativeSpanReadout
from source_answer_span import CompleteAnswerSpanPredictor


class NativeSpanReadoutTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.random.get_rng_state()

    def tearDown(self):
        torch.random.set_rng_state(self.rng)

    def state(self, value):
        t = PrismalTokenizer()
        ids = torch.tensor([t.encode(f'<BOI>Context:\nThe apple value is {value}.\nQuestion:\nWhat is the apple value?<EOI><BOO>', add_special_tokens=False)])
        memory = NativeQuestionMemory(8, span_readout=True)
        memory.eval()
        with torch.no_grad():
            _, states = memory(ids, torch.randn(1, ids.shape[1], 8), nn.Embedding(len(t.construction_units), 8), t)
        return t, memory, states[0]

    def test_source_spans_preserve_names_initials_currency_and_decimal_markers(self):
        for value in ('John Reed', 'A. R. Doe', '$75.50', 'red and blue'):
            t, memory, state = self.state(value)
            candidates = memory.span_readout.candidates(state, t, list(range(len(state.source))))
            matches = [c for c in candidates if c['text'] == value]
            self.assertTrue(matches, value)
            self.assertTrue(all(c['tokens'] == CompleteAnswerSpanPredictor.payload(t, value) for c in matches))

    def test_verified_continuation_and_predicted_endpoint(self):
        t, memory, _ = self.state('red and blue')
        payload = CompleteAnswerSpanPredictor.payload(t, 'red and blue')
        decoder = memory.span_readout
        nn.init.constant_(decoder.copy_gate.bias, 20.)
        plan = dict(candidates=[dict(tokens=payload)], probabilities=torch.tensor([1.]))
        for progress, expected in enumerate(payload):
            record = (0, 0, torch.zeros(8), torch.zeros(8), plan, payload[:progress])
            actual = decoder.mix(torch.zeros(1, 1, len(t.construction_units)), [record])
            self.assertEqual(int(actual.argmax(-1)), expected)
        mismatched = (0, 0, torch.zeros(8), torch.zeros(8), plan, (t.special_tokens['<UPPER>'],))
        logits = torch.randn(1, 1, len(t.construction_units))
        torch.testing.assert_close(decoder.mix(logits, [mismatched]), logits)

    def test_low_probability_prefix_cannot_force_a_completion(self):
        decoder = NativeSpanReadout(4)
        nn.init.constant_(decoder.copy_gate.bias, 20.)
        plan = dict(candidates=[dict(tokens=(1, 2)), dict(tokens=(3, 4))], probabilities=torch.tensor([.99, .01]))
        logits = torch.zeros(1, 1, 6)
        logits[0, 0, 5] = 10.
        record = (0, 0, torch.zeros(4), torch.zeros(4), plan, (3,))
        self.assertEqual(int(decoder.mix(logits, [record]).argmax(-1)), 5)

    def test_incremental_marker_progress_is_causal(self):
        t, memory, state = self.state('red and blue')
        surface = list(state.surface)
        payload = CompleteAnswerSpanPredictor.payload(t, 'red and blue')
        embedding = nn.Embedding(len(t.construction_units), 8)
        for progress, token in enumerate(payload[:-2]):
            with torch.no_grad():
                _, states, records = memory(torch.tensor([[token]]), torch.zeros(1, 1, 8), embedding, t, [state], return_readout=True)
            self.assertEqual(states[0].emitted, payload[:progress + 1])
            self.assertEqual(states[0].surface, surface)
            self.assertEqual(state.emitted, payload[:progress])
            state = states[0]

    def test_training_line_control_preserves_copy_support_and_gradient(self):
        t, memory, state = self.state('John Reed')
        memory.train()
        line = t.special_tokens['<LINE>']
        embedding = nn.Embedding(len(t.construction_units), 8)
        _, states, records = memory(torch.tensor([[line]]), torch.zeros(1, 1, 8), embedding, t, [state], return_readout=True)
        self.assertEqual(states[0].emitted, ())
        self.assertTrue(records)
        logits = memory.span_readout.mix(torch.zeros(1, 1, len(t.construction_units)), records)
        target = CompleteAnswerSpanPredictor.payload(t, 'John Reed')[0]
        torch.nn.functional.cross_entropy(logits.reshape(1, -1), torch.tensor([target])).backward()
        self.assertIsNotNone(memory.span_readout.copy_gate.bias.grad)
        self.assertGreater(float(memory.span_readout.copy_gate.bias.grad.abs().sum()), 0.)

    def test_span_path_reload_and_incremental_equivalence(self):
        import tempfile
        from pathlib import Path
        from tests.test_runtime_signatures import RuntimeSignatureTests
        from train import save_checkpoint, load_bundle_from_checkpoint
        _, t, model = RuntimeSignatureTests().setup_model()
        model.cfg.use_native_question_memory = True
        model.cfg.native_question_memory_span_readout = True
        model.cfg.native_question_memory_conditional_endpoints = True
        model.cfg.native_question_memory_relation_alignment = True
        model.cfg.native_question_memory_source_roles = True
        model.cfg.native_question_memory_antecedent_memory = True
        model.cfg.native_question_memory_owner_evidence = True
        model.cfg.native_question_memory_marker_endpoints = True
        model.cfg.native_question_memory_semantic_alignment = True
        model.cfg.native_question_memory_semantic_prior_scaling = True
        model.cfg.native_question_memory_semantic_decisions = True
        model.cfg.native_question_memory_conditional_copy = True
        model.cfg.native_question_memory_stable_span_gate = True
        model.cfg.native_question_memory_lexical_cues = True
        model.cfg.native_question_memory_future_context = 3
        model.cfg.native_question_memory_anchor_words = 2
        model.cfg.freeze_registry_observations = True
        model.registry.observation_updates_enabled = False
        model.native_question_memory = NativeQuestionMemory(model.cfg.d_model, span_readout=True, lexical_cues=True, future_context=3, anchor_words=2, conditional_endpoints=True, relation_alignment=True, source_roles=True, antecedent_memory=True, owner_evidence=True, marker_endpoints=True, semantic_alignment=True, semantic_prior_scaling=True, semantic_decisions=True, conditional_copy=True, stable_span_gate=True)
        nn.init.normal_(model.native_question_memory.semantic_request.weight, std=.03)
        nn.init.normal_(model.native_question_memory.semantic_source.weight, std=.03)
        nn.init.normal_(model.native_question_memory.semantic_alignment.prior_scale.weight, std=.02)
        nn.init.normal_(model.native_question_memory.semantic_alignment.score[-1].weight, std=.05)
        nn.init.normal_(model.native_question_memory.relation_alignment.output[-1].weight, std=.05)
        nn.init.normal_(model.native_question_memory.source_roles.source[-1].weight, std=.05)
        nn.init.normal_(model.native_question_memory.span_readout.marker_endpoint[-1].weight, std=.05)
        nn.init.normal_(model.native_question_memory.antecedent_memory.context_evidence.classifier[-1].weight, std=.05)
        ids = torch.tensor([t.prepare_generation_hierarchy('Context:\nThe apple is red and blue.\nQuestion:\nWhat color is the apple?').token_ids])
        with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
            expected = model(ids).logits
            save_checkpoint(model, directory, tokenizer=t)
            loaded, _, _ = load_bundle_from_checkpoint(Path(directory) / 'model.pt', device='cpu', load_training_state=False)
            loaded.eval()
            self.assertTrue(loaded.native_question_memory.span_readout.conditional_endpoints)
            self.assertIsNotNone(loaded.native_question_memory.relation_alignment)
            self.assertIsNotNone(loaded.native_question_memory.source_roles)
            self.assertIsNotNone(loaded.native_question_memory.semantic_alignment)
            self.assertIsNotNone(loaded.native_question_memory.semantic_request)
            self.assertTrue(loaded.native_question_memory.span_readout.conditional_copy)
            self.assertIsNotNone(loaded.native_question_memory.span_readout.marker_endpoint)
            self.assertIsNotNone(loaded.native_question_memory.antecedent_memory.context_evidence)
            torch.testing.assert_close(loaded(ids).logits, expected)
        memory = model.native_question_memory
        hidden = torch.randn(1, ids.shape[1], model.cfg.d_model)
        with torch.no_grad():
            full, states = memory(ids, hidden, model.construction_embedding, t)
            previous = None
            pieces = []
            for i in range(ids.shape[1]):
                piece, previous = memory(ids[:, i:i+1], hidden[:, i:i+1], model.construction_embedding, t, previous)
                pieces.append(piece)
            torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=1e-5, atol=1e-6)
            self.assertEqual(previous[0].source_ranges, states[0].source_ranges)

    def test_overflow_cannot_resurrect_prefix_support(self):
        t, memory, state = self.state('red')
        state.emitted = (t.special_tokens['<SPACE>'],) * (memory.span_readout.max_words * 72 + 8)
        state, _ = memory._observe(state, t.special_tokens['<SPACE>'], t)
        self.assertTrue(state.emission_invalid)
        state, _ = memory._observe(state, t.special_tokens['<LINE>'], t)
        self.assertEqual(state.emitted, (-1,))

    def test_stable_gate_uses_request_and_source_not_local_stop_state(self):
        decoder = NativeSpanReadout(4, stable_gate=True)
        nn.init.ones_(decoder.copy_gate.weight)
        plan = dict(candidates=[dict(tokens=(1, 2))], probabilities=torch.tensor([1.]), copy_context=torch.ones(4))
        logits = torch.zeros(1, 1, 6)
        question = torch.ones(4)
        a = decoder.mix(logits, [(0, 0, question, torch.ones(4) * -100, plan, (1,))])
        b = decoder.mix(logits, [(0, 0, question, torch.ones(4) * 100, plan, (1,))])
        torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()


class ConditionalEndpointTests(unittest.TestCase):
    def test_start_mass_offsets_and_endpoint_gradients(self):
        decoder = NativeSpanReadout(4, conditional_endpoints=True)
        class Energies(nn.Module):
            def __init__(self):
                super().__init__()
                self.values = nn.Parameter(torch.tensor([2., 1., -3.]))
            def forward(self, features):
                return self.values[:, None]
        decoder.endpoint = Energies()
        candidates = [dict(start=a, end=a, properties=[0.]*12) for a in (0, 0, 1)]
        question, words = torch.zeros(4), torch.zeros(2, 4)
        prior = torch.tensor([.7, .3]).log()
        scores = decoder.score_candidates(candidates, question, words, prior)
        torch.testing.assert_close(scores.exp()[:2].sum(), prior[0].exp())
        torch.testing.assert_close(scores.exp()[2], prior[1].exp())
        (-scores[0]).backward()
        self.assertLess(float(decoder.endpoint.values.grad[0]), 0.)
        self.assertGreater(float(decoder.endpoint.values.grad[1]), 0.)
        with torch.no_grad():
            decoder.endpoint.values[:2] += 100.
        torch.testing.assert_close(decoder.score_candidates(candidates, question, words, prior), scores.detach(), atol=1e-5, rtol=1e-5)
        decoder.conditional_endpoints = False
        torch.testing.assert_close(decoder.score_candidates(candidates, question, words, prior), prior[torch.tensor([0,0,1])] + decoder.endpoint.values)
