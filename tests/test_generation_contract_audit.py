"""Training serialization and live native emission must use one contract."""
import unittest
import torch
from torch import nn
from data import PrismalTokenizer, _build_window_samples_from_text
from native_question_memory import NativeQuestionMemory
from source_answer_span import CompleteAnswerSpanPredictor


class GenerationContractAuditTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.random.get_rng_state()

    def tearDown(self):
        torch.random.set_rng_state(self.rng)

    def test_empty_and_blank_leading_answer_prefixes_match_training(self):
        for answer in ('', '\n', '\nred', '\n\nred', 'red'):
            with self.subTest(answer=answer):
                t = PrismalTokenizer()
                p = t.prepare_generation_hierarchy('Which value?')
                s = _build_window_samples_from_text(t, '<BOI>Which value?<EOI><BOO>'+answer+'<EOO>', seq_len=256, max_samples=1, hierarchy_vector_dtype='float32')[0]
                n = len(p.token_ids)
                self.assertEqual(s.input_ids[:n].tolist(), p.token_ids)
                for track in ('signature_ids', 'signature_level_ids', 'signature_relation_ids', 'parent_signature_ids', 'signature_family_ids'):
                    self.assertEqual(getattr(s, track)[:n].tolist(), getattr(p, track))
                torch.testing.assert_close(s.hierarchy_vectors[:n], torch.tensor(p.hierarchy_vectors), rtol=0, atol=0)

    def test_span_payload_matches_training_separators_and_termination(self):
        for answer in ('red', 'red\nblue', 'red\n\nblue', 'if x:\n\treturn x', '', '\nred'):
            with self.subTest(answer=answer):
                t = PrismalTokenizer()
                s = _build_window_samples_from_text(t, '<BOI>Which value?<EOI><BOO>'+answer+'<EOO>', seq_len=256, max_samples=1, hierarchy_vector_dtype='float32')[0]
                sequence = s.input_ids.tolist()+[int(s.labels[-1])]
                start = sequence.index(t.special_tokens['<BOO>'])+1
                if sequence[start] == t.special_tokens['<LINE>']:
                    start += 1
                self.assertEqual(CompleteAnswerSpanPredictor.payload(t, answer), tuple(sequence[start:]))

    def native(self):
        t = PrismalTokenizer()
        memory = NativeQuestionMemory(8, span_readout=True)
        memory.eval()
        embedding = nn.Embedding(t.vocab_size, 8)
        ids = torch.tensor([t.prepare_generation_hierarchy('Context:\nThe value is red.\nQuestion:\nWhich value?').token_ids])
        with torch.no_grad():
            _, states = memory(ids, torch.zeros(1, ids.shape[1], 8), embedding, t)
        return t, memory, embedding, states[0]

    def test_padding_does_not_advance_native_emission_or_source(self):
        t, memory, _, state = self.native()
        before = (state.emitted, list(state.surface), state.position)
        state, active = memory._observe(state, t.pad_id, t)
        self.assertFalse(active)
        self.assertEqual((state.emitted, state.surface, state.position), before)

    def test_span_readout_reaches_supervised_eos_after_eoo(self):
        t, memory, embedding, state = self.native()
        payload = CompleteAnswerSpanPredictor.payload(t, 'red')
        state.emitted = payload[:-2]
        packet = state.cached_packet
        packet[3]['candidates'] = [dict(tokens=payload)]
        packet[3]['probabilities'] = torch.tensor([1.])
        # Preserve the fixed proposal for this boundary-gradient diagnostic.
        memory.eval()
        with torch.no_grad():
            hidden = torch.randn(1,1,8)
            actual, states, records = memory(torch.tensor([[t.special_tokens['<EOO>']]]), hidden, embedding, t, [state], return_readout=True)
        torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
        self.assertTrue(records)
        self.assertEqual(records[-1][-1], payload[:-1])
        logits = memory.span_readout.mix(torch.zeros(1,1,t.vocab_size), records)
        nn.functional.cross_entropy(logits.reshape(1,-1),torch.tensor([t.eos_id])).backward()
        self.assertGreater(float(memory.span_readout.copy_gate.bias.grad.abs().sum()),0.)
        import copy
        _, active = memory._observe(copy.copy(states[0]), t.special_tokens['<SPACE>'], t)
        self.assertFalse(active, 'A closed answer must not re-enable native continuation.')
        ended, active = memory._observe(states[0], t.eos_id, t)
        self.assertFalse(active)
        self.assertFalse(ended.emitting)

    def test_default_generation_allows_learned_short_answer_termination(self):
        from unittest.mock import patch
        from tests.test_runtime_signatures import RuntimeSignatureTests
        _, t, model = RuntimeSignatureTests().setup_model()
        p = t.prepare_generation_hierarchy('Which value?')
        ids = torch.tensor([p.token_ids])
        def eos_logits(hidden):
            logits = hidden.new_zeros(*hidden.shape[:-1], t.vocab_size)
            logits[..., t.eos_id] = 100.
            return logits
        kwargs = dict(max_new_tokens=8, temperature=0., repetition_penalty=1., no_repeat_ngram_size=0,
                      use_speculative_decoding=False, suppressed_token_ids=t.generation_suppressed_token_ids())
        with patch.object(model, '_construction_logits_from_guided', side_effect=eos_logits):
            result = model.generate(ids, **kwargs)
            self.assertEqual(result.shape[1] - ids.shape[1], 1)
            result = model.generate(ids, min_new_tokens=4, **kwargs)
            self.assertEqual(result.shape[1] - ids.shape[1], 4)

    def test_pretokenized_cache_cannot_silently_reuse_old_answer_layout(self):
        import tempfile, json
        from pathlib import Path
        from data import save_pretokenized_windows, MemmapTokenDataset
        t = PrismalTokenizer()
        samples = _build_window_samples_from_text(t, '<BOI>Which value?<EOI><BOO>\nred<EOO>', seq_len=256, max_samples=1)
        with tempfile.TemporaryDirectory() as directory:
            save_pretokenized_windows(samples, directory, seq_len=256)
            self.assertEqual(len(MemmapTokenDataset(directory)), 1)
            p = Path(directory)/'meta.json'
            meta = json.loads(p.read_text())
            meta.pop('output_seed_protocol_version', None)
            p.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'rebuild'):
                MemmapTokenDataset(directory)

    def test_output_blank_line_is_supervised_but_input_blank_line_is_not(self):
        t = PrismalTokenizer()
        text = '<BOI>Context:\nred\n\nblue\nQuestion:\nWhich values?<EOI><BOO>red\n\nblue<EOO>'
        s = _build_window_samples_from_text(t, text, seq_len=256, max_samples=1)[0]
        masks = s.loss_mask[s.labels == t.special_tokens['<BLO>']].tolist()
        self.assertEqual(masks, [0., 1.])
        supervised = set(s.labels[s.loss_mask > 0].tolist())
        self.assertFalse(supervised.intersection(t.generation_suppressed_token_ids()))

    def test_structured_templates_match_training_without_answer_dependent_prefixes(self):
        from data import _compose_structured_qa_record, format_structured_qa_prompt
        from train import _format_generation_prompt
        for style in ('instruction', 'question', 'json', 'mixed'):
            for answer in ('red', 'blue', 'green'):
                with self.subTest(style=style, answer=answer):
                    t = PrismalTokenizer()
                    text = _compose_structured_qa_record(instruction='What is a torus?', context='', response=answer, style=style)
                    p = t.prepare_generation_hierarchy(format_structured_qa_prompt('What is a torus?', style=style))
                    s = _build_window_samples_from_text(t,text,seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
                    self.assertEqual(s.input_ids[:len(p.token_ids)].tolist(),p.token_ids)
                    for track in ('signature_ids', 'signature_level_ids', 'signature_relation_ids', 'parent_signature_ids', 'signature_family_ids'):
                        self.assertEqual(getattr(s,track)[:len(p.token_ids)].tolist(),getattr(p,track))
                    torch.testing.assert_close(s.hierarchy_vectors[:len(p.token_ids)],torch.tensor(p.hierarchy_vectors),rtol=0,atol=0)
        self.assertEqual(_format_generation_prompt('What is a torus?',template_prompt=True),format_structured_qa_prompt('What is a torus?'))

    def test_explicit_generation_prefill_preserves_trailing_whitespace(self):
        from train import _format_generation_prompt
        for suffix in ('red ', 'red\t', 'red\n'):
            p = '<BOI>Which value?<EOI><BOO>'+suffix
            self.assertEqual(_format_generation_prompt(p),p)

    def test_textual_response_cue_belongs_to_output_prefill(self):
        from train import _format_generation_prompt
        t = PrismalTokenizer()
        p = t.prepare_generation_hierarchy(_format_generation_prompt('Instruction: Which value?\nResponse:'))
        s = _build_window_samples_from_text(t,'<BOI>Instruction: Which value?<EOI><BOO>Response: red<EOO>',seq_len=256,max_samples=1)[0]
        self.assertEqual(s.input_ids[:len(p.token_ids)].tolist(),p.token_ids)


if __name__ == '__main__':
    unittest.main()
