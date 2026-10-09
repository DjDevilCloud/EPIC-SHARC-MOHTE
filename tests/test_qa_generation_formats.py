"""Regression checks for real QA formats and matched generation prefixes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from config import PrismalWaveConfig
from data import (PrismalTokenizer, _build_window_samples_from_text,
                  iter_text_corpus, normalize_qa_text)
from train import generate_text

TRACKS = ('signature_ids', 'signature_level_ids', 'signature_relation_ids',
          'parent_signature_ids', 'signature_family_ids')


class QAGenerationFormatTests(unittest.TestCase):
    def assert_prefix(self, tokenizer, text, generation):
        sample = _build_window_samples_from_text(tokenizer, text, seq_len=512,
            max_samples=1, hierarchy_vector_dtype='float32')[0]
        length = len(generation.token_ids)
        self.assertEqual(sample.input_ids[:length].tolist(), generation.token_ids)
        for name in TRACKS:
            self.assertEqual(getattr(sample, name)[:length].tolist(), getattr(generation, name))
        torch.testing.assert_close(sample.hierarchy_vectors[:length],
            torch.tensor(generation.hierarchy_vectors), rtol=0, atol=0)
        return sample

    def test_textual_qa_masks_question_and_matches_all_prompt_fields(self):
        for text in ('input: Explain rain. output: Clouds release water.',
                     'INPUT: Explain rain.\nOUTPUT: Clouds release water.'):
            with self.subTest(text=text):
                tokenizer = PrismalTokenizer()
                tokenizer.learn_from_texts([text], min_frequency=1, max_new_tokens=32)
                candidates, _ = tokenizer._collect_construction_learning_counts([text])
                self.assertNotIn('input', candidates)
                self.assertNotIn('output', candidates)
                generation = tokenizer.prepare_generation_hierarchy(text[:text.lower().index('output:')+7])
                sample = self.assert_prefix(tokenizer, text, generation)
                length = len(generation.token_ids)
                self.assertEqual(float(sample.loss_mask[:length-1].sum()), 0.)
                self.assertEqual(float(sample.loss_mask[length-1]), 1.)
                self.assertTrue(tokenizer.decode(sample.labels[length-1:], clean_text=False).startswith('Clouds'))

    def test_corpus_paths_normalize_textual_fields(self):
        text = 'input: Explain rain. output: Clouds release water.'
        expected = '<BOI>Explain rain.<EOI><BOO>Clouds release water.<EOO>'
        with tempfile.TemporaryDirectory() as directory:
            for name, payload in [('qa.txt', text), ('qa.jsonl', json.dumps({'category': 'Math', 'text': text}))]:
                path = Path(directory) / name
                path.write_text(payload, encoding='utf-8')
                self.assertEqual(list(iter_text_corpus(path)), [expected])

    def test_prose_and_explicit_spans_are_preserved(self):
        for text in ('The input: a string; output: its reverse.',
                     'input: the word output appears without a delimiter',
                     'input: Missing answer output:',
                     '<BOI>input: Explain output: fields.<EOI><BOO>Keep this.<EOO>'):
            self.assertEqual(normalize_qa_text(text), text)

    def test_explicit_qa_does_not_duplicate_boundary_markers(self):
        text = '<BOI>Explain rain.<EOI><BOO>Clouds release water.<EOO>'
        tokenizer = PrismalTokenizer()
        tokenizer.learn_from_texts([text], min_frequency=1, max_new_tokens=32)
        for prompt in ('Explain rain.', '<BOI>Explain rain.<EOI>',
                       '<BOI>Explain rain.<EOI><BOO>'):
            with self.subTest(prompt=prompt):
                generation = tokenizer.prepare_generation_hierarchy(prompt)
                self.assert_prefix(tokenizer, text, generation)
                for marker in ('<BOI>', '<EOI>', '<BOO>'):
                    self.assertEqual(generation.token_ids.count(tokenizer.special_tokens[marker]), 1)

    def test_document_continuation_preserves_output_metadata(self):
        text = 'Redwood trails offer peaceful rides through ancient forests.'
        prefix = 'Redwood trails offer '
        tokenizer = PrismalTokenizer()
        tokenizer.learn_from_texts([text], min_frequency=1, max_new_tokens=32)
        generation = tokenizer.prepare_generation_hierarchy(prefix, mode='continuation')
        self.assert_prefix(tokenizer, text, generation)
        self.assertEqual(generation.token_ids[:5], [tokenizer.bos_id,
            tokenizer.special_tokens['<BOI>'], tokenizer.special_tokens['<EOI>'],
            tokenizer.special_tokens['<BOO>'], tokenizer.special_tokens['<LINE>']])
        self.assertNotEqual(generation.token_ids, tokenizer.prepare_generation_hierarchy(prefix).token_ids)
        self.assertNotEqual(generation.token_ids[-1], tokenizer.special_tokens['<EOL>'])

    def test_multiline_and_explicit_output_prefill_match_training(self):
        tokenizer = PrismalTokenizer()
        text = '<BOI>Explain rain.<EOI><BOO>Clouds release water.\nDrops fall.<EOO>'
        tokenizer.learn_from_texts([text], min_frequency=1, max_new_tokens=32)
        for prefix in ('<BOI>Explain rain.<EOI><BOO>Clouds release ',
                       '<BOI>Explain rain.<EOI><BOO>Clouds release water.\n'):
            self.assert_prefix(tokenizer, text, tokenizer.prepare_generation_hierarchy(prefix))

    def test_wrapper_forwards_continuation_without_stripping_whitespace(self):
        tokenizer = PrismalTokenizer()
        tokenizer.learn_from_texts(['Redwood trails offer peaceful rides.'], min_frequency=1, max_new_tokens=32)
        model = mock.Mock()
        model.cfg = PrismalWaveConfig(use_fst=False)
        model.generate.side_effect = lambda ids, **kwargs: ids
        prefix = 'Redwood trails '
        generate_text(model, tokenizer, prefix, torch.device('cpu'), prompt_mode='continuation')
        actual = model.generate.call_args.args[0][0].tolist()
        self.assertEqual(actual, tokenizer.prepare_generation_hierarchy(prefix, mode='continuation').token_ids)

    def test_invalid_generation_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            PrismalTokenizer().prepare_generation_hierarchy('Hello', mode='invalid')

    def test_cosmopedia_schema_preserves_prompt_and_response_and_masks_input(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        row = dict(prompt='Explain this code:\n\tprint("rain")',
                   text='Certainly!\n\tIt prints "rain".', seed_data='test', format='textbook', audience='general')
        expected = f"<BOI>{row['prompt']}<EOI><BOO>{row['text']}<EOO>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cosmo.parquet'
            pq.write_table(pa.Table.from_pylist([row]), path)
            self.assertEqual(list(iter_text_corpus(path)), [expected])
            json_path = Path(directory) / 'cosmo.jsonl'
            json_path.write_text(json.dumps(row), encoding='utf-8')
            self.assertEqual(list(iter_text_corpus(json_path)), [expected])
        t = PrismalTokenizer()
        t.learn_from_texts([expected], min_frequency=1, max_new_tokens=32)
        p = t.prepare_generation_hierarchy(row['prompt'])
        sample = self.assert_prefix(t, expected, p)
        self.assertEqual(float(sample.loss_mask[:len(p.token_ids)-1].sum()), 0.)
        self.assertGreater(float(sample.loss_mask[len(p.token_ids)-1:].sum()), 0.)

    def test_generic_prompt_text_is_not_assumed_to_be_a_response(self):
        from data import _compose_record_text
        self.assertNotIn('<BOO>', _compose_record_text(dict(prompt='Summarize this.', text='Context document.')))

    def test_diverseqa_adapter_retains_context_and_supervises_only_answer(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        raw = 'The expedition visited Lima.\n\nQuestion: Which city was visited?\nAnswer: Lima'
        expected = '<BOI>Context:\nThe expedition visited Lima.\nQuestion:\nWhich city was visited?<EOI><BOO>Lima<EOO>'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'DiverseQA' / 'part.parquet'
            path.parent.mkdir()
            pq.write_table(pa.Table.from_pylist([dict(id='one', text=raw)]), path)
            self.assertEqual(list(iter_text_corpus(path)), [expected])
            other = Path(directory) / 'ordinary.parquet'
            pq.write_table(pa.Table.from_pylist([dict(text=raw)]), other)
            self.assertEqual(list(iter_text_corpus(other)), [raw])
        t = PrismalTokenizer()
        t.learn_from_texts([expected], min_frequency=1, max_new_tokens=32)
        p = t.prepare_generation_hierarchy('Context:\nThe expedition visited Lima.\nQuestion:\nWhich city was visited?')
        s = self.assert_prefix(t, expected, p)
        self.assertEqual(float(s.loss_mask[:len(p.token_ids)-1].sum()), 0.)
        self.assertGreater(float(s.loss_mask[len(p.token_ids)-1:].sum()), 0.)

    def test_diverseqa_incomplete_record_retains_original_text(self):
        from data import _normalize_diverse_qa_record
        text = 'Document.\nQuestion: Missing response\nAnswer:'
        self.assertEqual(_normalize_diverse_qa_record(text), text)


if __name__ == '__main__':
    unittest.main()
