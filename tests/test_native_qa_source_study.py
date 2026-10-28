import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from native_semantic_alignment import NativeSemanticAlignment
from native_qa_source_study import schedule, objective, packet, evaluate
from native_question_memory import NativeQuestionMemory
from data import PrismalTokenizer


class NativeSourceStudyTests(unittest.TestCase):
    def test_discrete_evaluation_uses_ordinary_inference_scorer(self):
        scorer=Mock(return_value=dict(fused=torch.tensor([1.,0.])))
        model=SimpleNamespace(native_question_memory=SimpleNamespace(score_sources=scorer))
        row=dict(context='Alice visited York.',question='Who visited?',answer='Alice',source_group='family')
        x=dict(q=torch.zeros(8),prior=torch.zeros(2),inputs={},row=row,
               positive=torch.tensor([True,False]),teacher_correct=True)
        with patch('native_qa_source_study.batch_scores',side_effect=AssertionError('Batch loss may not select candidates')):
            summary,_=evaluate(model,dict(development=[x]))
        self.assertEqual(summary['development']['top1'],1)
        scorer.assert_called_once()

    def test_metric_ties_follow_inference_argmax_and_actual_topk(self):
        logits=torch.zeros(40)
        scorer=Mock(return_value=dict(fused=logits))
        model=SimpleNamespace(native_question_memory=SimpleNamespace(score_sources=scorer))
        row=dict(context='Alice visited York.',question='Who visited?',answer='Alice',source_group='family')
        positive=torch.zeros(40,dtype=torch.bool);positive[0]=True
        x=dict(q=torch.zeros(8),prior=logits,inputs={},row=row,positive=positive,teacher_correct=True)
        summary,ledger=evaluate(model,dict(development=[x]))
        self.assertEqual(summary['development']['top1'],1)
        expected=positive[logits.softmax(0).topk(16).indices].any()
        self.assertEqual(summary['development']['top16'],int(expected))
        self.assertEqual(ledger['development'][0]['shortlist'],logits.softmax(0).topk(16).indices.tolist())

    def test_gold_starts_require_exact_punctuation_payload(self):
        tokenizer=PrismalTokenizer()
        row=dict(context='20 people arrived. Later 20% left.',question='What percentage later left?',answer='20%')
        tokenizer.prepare_generation_hierarchy('Context:\n'+row['context']+'\nQuestion:\n'+row['question'])
        model=SimpleNamespace(cfg=SimpleNamespace(d_model=8),
            native_question_memory=NativeQuestionMemory(8,span_readout=True,semantic_alignment=True).eval(),
            construction_embedding=nn.Embedding(len(tokenizer.construction_units),8).requires_grad_(False))
        result=packet(model,tokenizer,row)
        positive=result['positive'].nonzero().flatten().tolist()
        self.assertTrue(positive)
        self.assertGreater(min(positive),0)
        for index in positive:self.assertTrue(any(c['text']=='20%' for c in result['by_start'][index]))

    def test_packed_batch_matches_independent_scores_and_gradients(self):
        torch.manual_seed(117)
        embedding=nn.Embedding(20,8).requires_grad_(False)
        for structure,attention in [(False,False),(True,False),(False,True)]:
            model=NativeSemanticAlignment(8,prior_scaling=True,structure=structure,question_attention=attention)
            nn.init.normal_(model.score[-1].weight,std=.1)
            if structure:nn.init.normal_(model.structure.weight,std=.1)
            if attention:nn.init.normal_(model.question_residual.weight,std=.1)
            inputs=[model.inputs([(1,2),(3,),(4,5,6)],[(7,),(8,9)],embedding),
                    model.inputs([(10,),(11,12)],[(13,14,15)],embedding)]
            for x in inputs:x['structure']=torch.randn(len(x['source']),8)
            priors=[torch.randn(len(x['source'])) for x in inputs];questions=[torch.randn(8) for x in inputs]
            singles=[model.score_details(x,p,q)['fused'] for x,p,q in zip(inputs,priors,questions)]
            reference=sum(s.square().sum() for s in singles);reference.backward()
            gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            encoded=model.encode_many(inputs)
            batch=[model.score_details(x,p,q,e)['fused'] for x,p,q,e in zip(inputs,priors,questions,encoded)]
            for a,b in zip(singles,batch):torch.testing.assert_close(a,b,rtol=2e-4,atol=3e-5)
            sum(s.square().sum() for s in batch).backward()
            for n,p in model.named_parameters():
                if n in gradients:torch.testing.assert_close(p.grad,gradients[n],rtol=2e-4,atol=3e-5)

    def test_schedule_is_matched_and_balanced_without_test_examples(self):
        lengths=dict(new=128,legacy=58,native=60,controlled=320)
        first=schedule(lengths,1600)
        self.assertEqual(first,schedule(lengths,1600))
        self.assertNotEqual(first[:800],schedule(lengths,800,20261017))
        coverage=[0]*128
        for step in first:
            self.assertEqual([sum(k==kind for k,_ in step) for kind in lengths],[8,2,2,2])
            for kind,index in step:
                self.assertTrue(0<=index<lengths[kind])
                if kind=='new':coverage[index]+=1
        self.assertEqual(set(coverage),{100})

    def test_incorrect_teacher_does_not_supply_positive_replay_target(self):
        logits=torch.tensor([1.,2.],requires_grad=True)
        x=dict(positive=torch.tensor([True,False]),teacher=torch.tensor([-10.,0.]),teacher_correct=False)
        source=objective(dict(fused=logits),x,False)
        replay=objective(dict(fused=logits),x,True)
        torch.testing.assert_close(source,replay,rtol=0,atol=0)
        replay.backward()
        self.assertLess(float(logits.grad[0]),0.)


if __name__=='__main__':unittest.main()
