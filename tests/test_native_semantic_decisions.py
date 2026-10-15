import unittest
import torch
from torch import nn
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory
from native_span_readout import NativeSpanReadout
from config import PrismalWaveConfig
from source_answer_span import CompleteAnswerSpanPredictor


class SemanticDecisionTests(unittest.TestCase):
    def test_nonzero_shared_decisions_and_copy_match_incremental_rollout(self):
        torch.manual_seed(218)
        t = PrismalTokenizer()
        prefix = t.prepare_generation_hierarchy('Context:\nTen people saw Blue birds.\nQuestion:\nHow many people?').token_ids
        payload = CompleteAnswerSpanPredictor.payload(t, 'Ten')
        ids = torch.tensor([prefix + list(payload)])
        embedding = nn.Embedding(len(t.construction_units), 8)
        memory = NativeQuestionMemory(8, span_readout=True, semantic_alignment=True,
                                      semantic_decisions=True, stable_span_gate=True, conditional_copy=True)
        for layer in (memory.semantic_request, memory.semantic_source, memory.semantic_alignment.score[-1],
                      memory.span_readout.endpoint[-1]):
            nn.init.normal_(layer.weight, std=.1)
        memory.eval()
        hidden = torch.randn(1, ids.shape[1], 8)
        native = torch.randn(1, ids.shape[1], len(t.construction_units))
        with torch.no_grad():
            full, _, records = memory(ids, hidden, embedding, t, return_readout=True)
            expected = memory.span_readout.mix(native, records)
            state, pieces, mixed = None, [], []
            for i in range(ids.shape[1]):
                output, state, step_records = memory(ids[:, i:i+1], hidden[:, i:i+1], embedding, t, state, return_readout=True)
                pieces.append(output)
                mixed.append(memory.span_readout.mix(native[:, i:i+1], step_records))
            torch.testing.assert_close(torch.cat(pieces, 1), full, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(torch.cat(mixed, 1), expected, rtol=1e-5, atol=1e-6)
            # Completed answer tokens cannot change the observed-input plan.
            torch.testing.assert_close(records[0][2], records[-1][2], rtol=0, atol=0)
            torch.testing.assert_close(records[0][4]['log_probabilities'], records[-1][4]['log_probabilities'], rtol=0, atol=0)

    def test_required_dependencies(self):
        for kwargs in (dict(semantic_decisions=True), dict(conditional_copy=True)):
            with self.assertRaises(ValueError):
                NativeQuestionMemory(8, **kwargs)
        for kwargs in (dict(native_question_memory_semantic_decisions=True),
                       dict(native_question_memory_conditional_copy=True)):
            with self.assertRaises(ValueError):
                PrismalWaveConfig(**kwargs)

    def test_zero_residual_parity_then_extent_and_emission_gradients(self):
        torch.manual_seed(217)
        t = PrismalTokenizer()
        ids = torch.tensor([t.prepare_generation_hierarchy('Context:\nTen people saw Blue birds.\nQuestion:\nHow many people?').token_ids])
        embedding = nn.Embedding(len(t.construction_units), 8)
        base = NativeQuestionMemory(8, span_readout=True, semantic_alignment=True)
        shared = NativeQuestionMemory(8, span_readout=True, semantic_alignment=True, semantic_decisions=True)
        shared.load_state_dict(base.state_dict(), strict=False)
        hidden = torch.randn(1, ids.shape[1], 8)
        a, _, ar = base(ids, hidden, embedding, t, return_readout=True)
        b, _, br = shared(ids, hidden, embedding, t, return_readout=True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(ar[-1][4]['probabilities'], br[-1][4]['probabilities'], rtol=0, atol=0)
        nn.init.normal_(shared.semantic_request.weight, std=.1)
        nn.init.normal_(shared.semantic_source.weight, std=.1)
        nn.init.normal_(shared.output[-1].weight, std=.1)
        nn.init.normal_(shared.span_readout.endpoint[-1].weight, std=.1)
        nn.init.normal_(shared.span_readout.copy_gate.weight, std=.1)
        # Isolate each decision loss from retrieval's semantic scoring path.
        for path in ('extent', 'emission', 'task_gate'):
            shared.zero_grad()
            output, _, records = shared(ids, hidden, embedding, t, return_readout=True)
            plan = records[-1][4]
            if path == 'extent':
                loss = -plan['log_probabilities'][0]
            elif path == 'emission':
                loss = output[:, -1].square().sum()
            else:
                loss = shared.span_readout.copy_gate(torch.cat((plan['copy_context'], records[-1][2]))).sum()
            loss.backward()
            self.assertGreater(float(shared.semantic_request.weight.grad.norm()), 0.)
            self.assertGreater(float(shared.semantic_alignment.attention.weight.grad.norm()), 0.)
            self.assertGreater(float(shared.semantic_alignment.word.weight_ih_l0.grad.norm()), 0.)
            if path == 'extent':
                self.assertGreater(float(shared.semantic_source.weight.grad.norm()), 0.)


class ConditionalCopyTests(unittest.TestCase):
    def mixture(self, gate, weights=(.66, .34), emitted=(1,)):
        decoder = NativeSpanReadout(4, stable_gate=True, conditional_copy=True)
        nn.init.constant_(decoder.copy_gate.bias, gate)
        scores = torch.tensor(weights).log().requires_grad_()
        plan = dict(candidates=[dict(tokens=(3, 4)), dict(tokens=(1, 2))],
                    log_probabilities=scores.log_softmax(0), copy_context=torch.zeros(4))
        logits = torch.zeros(1, 1, 6, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 5] = 10.
        output = decoder.mix(logits, [(0, 0, torch.zeros(4), torch.zeros(4), plan, emitted)])
        return output, logits, scores

    def test_copy_confidence_is_independent_of_surviving_mass(self):
        for weights in ((.66, .34), (.99999999, .00000001)):
            output, _, _ = self.mixture(8., weights)
            self.assertEqual(int(output.argmax(-1)), 2)
            torch.testing.assert_close(output.exp().sum(), torch.tensor(1.))
        output, _, _ = self.mixture(-8.)
        self.assertEqual(int(output.argmax(-1)), 5)

    def test_no_verified_prefix_preserves_native_logits(self):
        output, logits, _ = self.mixture(80., emitted=(0,))
        torch.testing.assert_close(output, logits, rtol=0, atol=0)

    def test_saturated_gate_preserves_native_gradients(self):
        output, logits, scores = self.mixture(80.)
        (-output[0, 0, 0]).backward()
        self.assertLess(float(logits.grad[0, 0, 0]), -.9)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_shared_next_token_mass_and_continuation_gradients(self):
        decoder = NativeSpanReadout(4, stable_gate=True, conditional_copy=True)
        nn.init.constant_(decoder.copy_gate.bias, 10.)
        scores = torch.tensor([1., 2., 3.], requires_grad=True)
        plan = dict(candidates=[dict(tokens=(1, 2)), dict(tokens=(1, 2)), dict(tokens=(1, 4))],
                    log_probabilities=scores.log_softmax(0), copy_context=torch.zeros(4))
        output = decoder.mix(torch.zeros(1, 1, 6), [(0, 0, torch.zeros(4), torch.zeros(4), plan, (1,))])
        (-output[0, 0, 2]).backward()
        self.assertLess(float(scores.grad[0]), 0.)
        self.assertLess(float(scores.grad[1]), 0.)
        self.assertGreater(float(scores.grad[2]), 0.)
        torch.testing.assert_close(output.exp().sum(), torch.tensor(1.))
