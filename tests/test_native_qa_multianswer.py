import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory
import native_qa_multianswer as review
from native_qa_review import qa_fingerprint


class MultianswerTests(unittest.TestCase):
    def entry(self,row,support,options):
        return dict(fingerprint=qa_fingerprint(row),original_answer=row['answer'],
                    decision='retain',reason='Explicit manual relation review.',support=support,
                    answers=[dict(text=text,occurrences=positions,reason='Reviewed direct evidence.') for text,positions in options])

    def test_three_answers_are_accepted_without_changing_original(self):
        row=dict(context='Groups: Latino, Hmong, Somali.',question='Name one group.',answer='Somali')
        entry=self.entry(row,[[0,len(row['context'])]],[(s,review.intervals(row['context'],s)) for s in ('Somali','Latino','Hmong')])
        before=copy.deepcopy((row,entry));review.validate_entry(row,entry)
        for s in ('Somali','Latino','Hmong'):
            result=review.score_text(s,True,entry)
            self.assertTrue(result['reviewed_exact_and_terminated'])
            self.assertEqual(result['original_single_gold_exact'],s=='Somali')
        self.assertFalse(review.score_text('Hmong',False,entry)['reviewed_exact_and_terminated'])
        self.assertEqual(before,(row,entry))

    def test_case_and_punctuation_are_not_globally_normalized(self):
        row=dict(context='The rate is 20%.',question='What percentage?',answer='20%')
        entry=self.entry(row,[[0,len(row['context'])]],[('20%',[[12,15]])])
        review.validate_entry(row,entry)
        self.assertTrue(review.score_text(' 20%\n',True,entry)['reviewed_multianswer_exact'])
        self.assertFalse(review.score_text('20',True,entry)['reviewed_multianswer_exact'])

    def test_unsupported_reference_and_unreviewed_occurrence_are_rejected(self):
        row=dict(context='Alice paid $10. Bob paid $10.',question='What did Bob pay?',answer='$10')
        pos=row['context'].rindex('$10')
        entry=self.entry(row,[[15,len(row['context'])]],[('$10',[[pos,pos+3]])])
        review.validate_entry(row,entry)
        bad=copy.deepcopy(entry);bad['answers'][0]['occurrences']=[[11,14]]
        with self.assertRaises(ValueError):review.validate_entry(row,bad)
        bad=copy.deepcopy(entry);bad['answers'].append(dict(text='$20',occurrences=[[pos,pos+3]],reason='Untrue'))
        with self.assertRaises(ValueError):review.validate_entry(row,bad)

    def test_substring_amounts_are_not_mistaken_for_whole_numbers(self):
        self.assertEqual(review.intervals('In 2024, 24 states applied.','24'),[[9,11]])

    def test_repeated_anchor_requires_explicit_occurrence(self):
        text='A person arrived. A person arrived.'
        with self.assertRaises(ValueError):review.anchor_support(text,'A person arrived.')
        lo,hi=review.anchor_support(text,('A person arrived.',1))
        self.assertEqual(text[lo:hi],'A person arrived.')

    def test_evidence_layout_keeps_initials_and_decimal_values(self):
        text='J. R. Climer arrived. The amount was 7.2 percent. Next.'
        a,b=review.anchor_support(text,'J. R. Climer')
        self.assertEqual(text[a:b].strip(),'J. R. Climer arrived.')
        a,b=review.anchor_support(text,'7.2 percent')
        self.assertEqual(text[a:b].strip(),'The amount was 7.2 percent.')

    def test_locked_split_cannot_be_reviewed(self):
        with patch.object(review,'read',side_effect=AssertionError('Must not open locked questions')):
            with self.assertRaises(ValueError):review.rows('test')

    def test_original_primary_and_duplicate_options_are_guarded(self):
        row=dict(context='York and Bath.',question='Name one.',answer='York')
        entry=self.entry(row,[[0,len(row['context'])]],[('Bath',[[9,13]])])
        with self.assertRaises(ValueError):review.validate_entry(row,entry)
        entry=self.entry(row,[[0,len(row['context'])]],[('York',[[0,4]]),('York',[[0,4]])])
        with self.assertRaises(ValueError):review.validate_entry(row,entry)

    def test_native_binding_credits_only_reviewed_repeated_occurrence(self):
        tokenizer=PrismalTokenizer();memory=NativeQuestionMemory(8,capacity=512,span_readout=True)
        row=dict(context='Alice bought 10 shares. Bob sold 10 shares.',question='How many did Bob sell?',answer='10')
        pos=row['context'].rindex('10')
        entry=self.entry(row,[[24,len(row['context'])]],[('10',[[pos,pos+2]])])
        binding=review.bind_native_occurrences(row,entry,memory,tokenizer)

        self.assertEqual(len(binding['occurrences']),1)
        self.assertEqual(binding['occurrences'][0]['start'],6)
        self.assertEqual(binding['occurrences'][0]['character_offsets'],[pos,pos+2])

    def test_atomic_full_document_binding_is_explicit_and_old_cap_is_preserved(self):
        tokenizer=PrismalTokenizer();memory=NativeQuestionMemory(8,capacity=512,span_readout=True)
        text='x'*2200+' Red costs 12.'
        row=dict(context=text,question='What does Red cost?',answer='12');pos=text.rindex('12')
        entry=self.entry(row,[[2200,len(text)]],[('12',[[pos,pos+2]])])
        with self.assertRaisesRegex(ValueError,'limits'):
            review.bind_native_occurrences(row,entry,memory,tokenizer)
        binding=review.bind_native_occurrences(row,entry,memory,tokenizer,max_total_tokens=None)
        self.assertGreater(binding['prompt_tokens'],2048)
        self.assertLess(binding['source_words'],512)

    def test_contiguous_partial_word_is_not_silently_accepted_as_native_span(self):
        tokenizer=PrismalTokenizer();memory=NativeQuestionMemory(8,capacity=512,span_readout=True)
        row=dict(context='NewYork exists.',question='Which place?',answer='York')
        entry=self.entry(row,[[0,len(row['context'])]],[('York',[[3,7]])])
        with self.assertRaises(ValueError):review.bind_native_occurrences(row,entry,memory,tokenizer)

    def test_supported_uncopyable_mention_is_preserved_beside_copyable_reference(self):
        tokenizer=PrismalTokenizer();memory=NativeQuestionMemory(8,capacity=512,span_readout=True)
        row=dict(context='Bay Pines VA’s staff arrived. Bay Pines VA hosted the event.',question='Which facility?',answer='Bay Pines VA')
        entry=self.entry(row,[[0,len(row['context'])]],[('Bay Pines VA',review.intervals(row['context'],'Bay Pines VA'))])
        binding=review.bind_native_occurrences(row,entry,memory,tokenizer)
        self.assertEqual(len(binding['occurrences']),1)
        self.assertEqual(len(binding['unavailable_supported_occurrences']),1)
        self.assertEqual(len(entry['answers'][0]['occurrences']),2)

    def test_frozen_outputs_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'overlay.json';review.write_new(path,dict(answer='Somali'))
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):review.write_new(path,dict(answer='Hmong'))
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
