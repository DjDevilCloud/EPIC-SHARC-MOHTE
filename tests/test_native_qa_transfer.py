import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from config import PrismalWaveConfig
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory
from native_semantic_alignment import NativeSemanticAlignment
from train import load_bundle_from_checkpoint, save_checkpoint


class NativeTransferTests(unittest.TestCase):
    def fixture(self, **kwargs):
        torch.manual_seed(219)
        t = PrismalTokenizer()
        ids = torch.tensor([t.prepare_generation_hierarchy('Context:\n"Ten" people paid $25.\n\nBlue birds saw 20%.\nQuestion:\nHow many people paid?').token_ids])
        emb = nn.Embedding(len(t.construction_units), 8)
        memory = NativeQuestionMemory(8, span_readout=True, semantic_alignment=True,
                                      semantic_decisions=True, **kwargs).eval()
        return t, ids, emb, memory

    def test_shared_live_cached_packet_and_diagnostics(self):
        t, ids, emb, memory = self.fixture(semantic_structure=True, question_attention=True)
        nn.init.normal_(memory.semantic_alignment.score[-1].weight, std=.1)
        nn.init.normal_(memory.semantic_request.weight, std=.1)
        nn.init.normal_(memory.semantic_source.weight, std=.1)
        with torch.no_grad():
            _, states, records = memory(ids, torch.zeros(1, ids.shape[1], 8), emb, t, return_readout=True, diagnostics=True)
            q, rec, prob, plan = states[0].cached_packet
            candidate_bank = memory.span_readout.candidates(states[0], t, list(range(len(states[0].source))))
            cached = memory.decision_packet(plan['legacy_question'], plan['base_word_vectors'],
                memory.value(plan['base_word_vectors']), plan['legacy_start_logits'], plan['marker_words'],
                plan['semantic_inputs'], lambda selected: [c for i in selected for c in candidate_bank if c['start'] == i],
                plan['source_words'], diagnostics=True)
            for a, b in zip((q, rec, prob), cached[:3]):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-5)
            self.assertEqual(plan['candidates'], cached[3]['candidates'])
            torch.testing.assert_close(plan['log_probabilities'], cached[3]['log_probabilities'], rtol=0, atol=0)
            logits = torch.randn(1, ids.shape[1], len(t.construction_units))
            expected = memory.span_readout.mix(logits, records)
            diagnostics = []
            actual = memory.span_readout.mix(logits, records, diagnostics)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(diagnostics)
            self.assertIn('shortlist', diagnostics[0]['source'])
            self.assertNotIn('answer', diagnostics[0])
            flags = plan['semantic_inputs']['structure']
            self.assertEqual(flags.shape, (len(states[0].source), 8))
            self.assertTrue(all(bool(flags[:, i].any()) for i in range(8)))

    def test_new_encoders_are_zero_residual_then_learn(self):
        torch.manual_seed(220)
        embedding = nn.Embedding(12, 8).requires_grad_(False)
        base = NativeSemanticAlignment(8)
        inputs = base.inputs([(1,), (2,), (3,)], [(4,), (5,)], embedding)
        inputs['structure'] = torch.eye(8)[:3]
        nn.init.normal_(base.score[-1].weight, std=.2)
        for option, target in (('structure', 'structure.weight'), ('question_attention', 'question_residual.weight')):
            layer = NativeSemanticAlignment(8, **{option: True})
            layer.load_state_dict(base.state_dict(), strict=False)
            torch.testing.assert_close(layer(inputs), base(inputs), rtol=0, atol=0)
            (-layer(inputs).log_softmax(0)[0]).backward()
            self.assertGreater(float(dict(layer.named_parameters())[target].grad.norm()), 0.)

    def test_normalized_fusion_is_a_distribution_and_shift_invariant(self):
        torch.manual_seed(221)
        layer = NativeSemanticAlignment(8, prior_scaling=True, normalized_fusion=True)
        inputs = layer.inputs([(1,), (2,), (3,)], [(4,), (5,)], nn.Embedding(8, 8))
        prior, q = torch.tensor([15., -2., 3.]), torch.randn(8)
        first = layer.score_details(inputs, prior, q)
        second = layer.score_details(inputs, prior + 100., q)
        torch.testing.assert_close(first['fused'].exp().sum(), torch.tensor(1.))
        torch.testing.assert_close(first['fused'], second['fused'], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(torch.nn.functional.softplus(layer.temperatures), torch.ones(2))
        self.assertEqual(float(layer.fusion_gate[-1].bias.detach()), -8.)
        with self.assertRaises(ValueError):
            layer(inputs, prior)
        torch.testing.assert_close(prior + layer(inputs, prior, legacy_question=q), first['fused'])
        (-first['fused'][1]).backward()
        self.assertTrue(torch.isfinite(layer.temperatures.grad).all())

    def test_routes_override_legacy_flag_and_survive_reload(self):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        fixture = RuntimeSignatureTests(); fixture.identity_readout = True
        _, t, m = fixture.setup_model()
        m.cfg.use_native_question_memory = True
        m.native_question_memory = NativeQuestionMemory(m.cfg.d_model)
        ids = torch.tensor([t.prepare_generation_hierarchy('Context:\nTen people.\nQuestion:\nHow many people?').token_ids])
        m.cfg.qa_readout_route = 'native'
        with torch.no_grad(), patch.object(m.bounded_identity_readout, 'forward', side_effect=AssertionError('Wrong QA route')):
            m(ids)
            # Ordinary generation must use the saved route, with both heads present.
            m.generate(ids, max_new_tokens=2, min_new_tokens=0, temperature=0., top_k=1,
                       use_speculative_decoding=False, repetition_penalty=1., no_repeat_ngram_size=0)
        with torch.no_grad(), patch.object(m.native_question_memory, 'forward', side_effect=AssertionError('Backbone route entered native head')), patch.object(m.bounded_identity_readout, 'forward', side_effect=AssertionError('Backbone route entered identity head')):
            m(ids, qa_readout_route='backbone', apply_identity_readout=False)
        m.cfg.qa_readout_route = 'identity'
        with torch.no_grad(), patch.object(m.native_question_memory, 'forward', side_effect=AssertionError('Explicit identity route entered native head')):
            m(ids, apply_identity_readout=False)
        m.cfg.qa_readout_route = 'legacy'
        with torch.no_grad(), patch.object(m.bounded_identity_readout, 'forward', side_effect=AssertionError('Explicit incremental native route entered identity head')):
            first = m(ids, qa_readout_route='native')
            m.forward_incremental(ids[:, -1:], position_index=ids.shape[1],
                slot_state=first.slot_state, signature_lattice_state=first.signature_lattice_state,
                runtime_signature_state=first.runtime_signature_state, qa_readout_route='native')
        m.cfg.qa_readout_route = 'native'
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(m, folder, tokenizer=t)
            loaded, _, cfg = load_bundle_from_checkpoint(Path(folder) / 'model.pt', device='cpu', load_training_state=False)
            loaded.eval()
            self.assertEqual(cfg.qa_readout_route, 'native')
            self.assertIsNotNone(loaded.bounded_identity_readout)
            with torch.no_grad():
                torch.testing.assert_close(loaded(ids).logits, m(ids).logits, rtol=1e-5, atol=1e-6)

    def test_route_validation_and_legacy_default(self):
        self.assertEqual(PrismalWaveConfig().qa_readout_route, 'legacy')
        for kwargs in (dict(qa_readout_route='invalid'), dict(qa_readout_route='native'),
                       dict(qa_readout_route='identity'), dict(native_question_memory_semantic_structure=True)):
            with self.assertRaises(ValueError):
                PrismalWaveConfig(**kwargs)

    def test_single_line_is_not_a_blank_line(self):
        t, ids, emb, memory = self.fixture(semantic_structure=True)
        simple = torch.tensor([t.prepare_generation_hierarchy('Context:\nOne.\nTwo.\n\nThree.\nQuestion:\nWhat?').token_ids])
        with torch.no_grad():
            _, states = memory(simple, torch.zeros(1, simple.shape[1], 8), emb, t)
        flags = states[0].cached_packet[3]['semantic_inputs']['structure']
        self.assertEqual(flags[:, 4].tolist(), [0., 0., 1.])
        lower = torch.tensor([t.prepare_generation_hierarchy('Context:\nlower Upper.\nQuestion:\nWhat?').token_ids])
        with torch.no_grad():
            _, states = memory(lower, torch.zeros(1, lower.shape[1], 8), emb, t)
        self.assertEqual(states[0].cached_packet[3]['semantic_inputs']['structure'][:, 0].tolist(), [0., 1.])
