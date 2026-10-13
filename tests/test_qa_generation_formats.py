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

    def test_diverseqa_multiple_questions_share_only_the_document(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from data import parse_diverse_qa_record
        raw='Lima is in Peru.\nQuestion: Which city?\nAnswer: Lima\n\nQuestion: Which country?\nAnswer: Peru'
        self.assertEqual(parse_diverse_qa_record(raw),('Lima is in Peru.',[('Which city?','Lima'),('Which country?','Peru')]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'DiverseQA'/'part.parquet';path.parent.mkdir()
            pq.write_table(pa.Table.from_pylist([dict(id='one',text=raw)]),path)
            records=list(iter_text_corpus(path))
        self.assertEqual(records,['<BOI>Context:\nLima is in Peru.\nQuestion:\nWhich city?<EOI><BOO>Lima<EOO>',
                                  '<BOI>Context:\nLima is in Peru.\nQuestion:\nWhich country?<EOI><BOO>Peru<EOO>'])
        for record,answer in zip(records,('Lima','Peru')):
            self.assertEqual(record.split('<BOO>')[1],answer+'<EOO>')
            self.assertEqual(record.count('Question:'),1)

    def test_diverseqa_partial_trailing_pair_is_preserved(self):
        from data import parse_diverse_qa_record,_normalize_diverse_qa_records
        text='Document.\nQuestion: One?\nAnswer: Yes.\nQuestion: Two?\nAnswer:'
        self.assertIsNone(parse_diverse_qa_record(text))
        self.assertEqual(_normalize_diverse_qa_records(text),[text])

    def test_full_context_qa_keeps_source_question_and_complete_answer(self):
        t=PrismalTokenizer()
        source='Lima is in Peru. '+('Ordinary reference details. '*40)
        prompt='Context:\n'+source+'\nQuestion:\nWhich city is in Peru?'
        text='<BOI>'+prompt+'<EOI><BOO>Lima<EOO>'
        t.learn_from_texts([text],min_frequency=1,max_new_tokens=64)
        windows=_build_window_samples_from_text(t,text,seq_len=32,max_samples=1,hierarchy_vector_dtype='float32',preserve_grounded_qa_context=True)
        self.assertEqual(len(windows),1);sample=windows[0]
        self.assertGreater(len(sample.input_ids),32)
        for marker in ('<BOI>','<EOI>','<BOO>','<EOO>'):self.assertIn(t.special_tokens[marker],sample.input_ids.tolist())
        generation=t.prepare_generation_hierarchy(prompt)
        self.assertEqual(sample.input_ids[:len(generation.token_ids)].tolist(),generation.token_ids)
        for name in TRACKS:self.assertEqual(getattr(sample,name)[:len(generation.token_ids)].tolist(),getattr(generation,name))
        torch.testing.assert_close(sample.hierarchy_vectors[:len(generation.token_ids)],torch.tensor(generation.hierarchy_vectors),atol=1e-6,rtol=1e-6)
        boo=sample.input_ids.tolist().index(t.special_tokens['<BOO>'])
        self.assertEqual(float(sample.loss_mask[:boo].sum()),0.)
        self.assertIsNotNone(sample.source_group)

    def test_sibling_questions_stay_in_one_validation_partition(self):
        from data import TextWindowDataset,split_text_window_dataset,StreamingTextCorpusDataset
        texts=['<BOI>Context:\n'+c+'\nQuestion:\n'+q+'<EOI><BOO>'+a+'<EOO>'
               for c,pairs in [('Lima is in Peru.',[('Which city?','Lima'),('Which country?','Peru')]),
                               ('Oslo is in Norway.',[('Which city?','Oslo'),('Which country?','Norway')])] for q,a in pairs]
        t=PrismalTokenizer();t.learn_from_texts(texts,min_frequency=1,max_new_tokens=64)
        dataset=TextWindowDataset(texts,t,seq_len=64,preserve_grounded_qa_context=True)
        train,val=split_text_window_dataset(dataset,val_fraction=.5,seed=7)
        self.assertTrue({dataset[i].source_group for i in train.indices}.isdisjoint({dataset[i].source_group for i in val.indices}))
        stream=StreamingTextCorpusDataset('unused',t,seq_len=64,split='train',val_fraction=.5,seed=7,preserve_grounded_qa_context=True)
        self.assertEqual(stream._include_record(0,texts[0]),stream._include_record(1,texts[1]))
        self.assertEqual(stream._include_record(2,texts[2]),stream._include_record(3,texts[3]))

    def test_streaming_buffer_does_not_bypass_grounded_document_split(self):
        from data import StreamingTextCorpusDataset
        texts=['<BOI>Context:\nLima is in Peru.\nQuestion:\nWhich city?<EOI><BOO>Lima<EOO>',
               '<BOI>Context:\nOslo is in Norway.\nQuestion:\nWhich city?<EOI><BOO>Oslo<EOO>']
        t=PrismalTokenizer();t.learn_from_texts(texts,min_frequency=1,max_new_tokens=64)
        stream=StreamingTextCorpusDataset('unused',t,seq_len=64,max_samples=10,split='train',val_fraction=.5,seed=3,shuffle_buffer_size=10,preserve_grounded_qa_context=True)
        with mock.patch('data.iter_text_corpus',return_value=iter(texts)),mock.patch.object(stream,'_include_record',side_effect=lambda i,text:i==0):
            samples=list(stream)
        self.assertEqual(len(samples),1)
        self.assertIn('Lima',t.decode(samples[0].input_ids.tolist(),clean_text=False))
        with mock.patch('data.iter_text_corpus',return_value=iter(texts)),mock.patch.object(stream,'_include_record',return_value=False):
            self.assertEqual(list(stream),[])

    def test_diverseqa_incomplete_record_retains_original_text(self):
        from data import _normalize_diverse_qa_record
        text = 'Document.\nQuestion: Missing response\nAnswer:'
        self.assertEqual(_normalize_diverse_qa_record(text), text)


if __name__ == '__main__':
    unittest.main()
