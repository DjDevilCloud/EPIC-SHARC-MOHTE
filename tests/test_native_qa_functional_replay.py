import unittest
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from native_question_memory import NativeQuestionMemory
from native_semantic_alignment import NativeSemanticAlignment
from native_span_readout import NativeSpanReadout
from native_qa_functional_replay import (gold_mass_loss,source_supervision,protected_decision,
    protected_gate,protected_tokens,cached_emission)


class FunctionalReplayTests(unittest.TestCase):
    def test_incorrect_source_teacher_is_not_preserved(self):
        current=torch.tensor([1.,0.],requires_grad=True);teacher=torch.tensor([10.,-10.])
        positive=torch.tensor([False,True])
        loss=protected_decision(current,teacher,positive)
        torch.testing.assert_close(loss,gold_mass_loss(current,positive))
        loss.backward();self.assertLess(float(current.grad[1]),0.)

    def test_branch_consistent_supervision_trains_semantic_distribution(self):
        corrected=torch.tensor([2.,0.],requires_grad=True);semantic=torch.tensor([0.,2.],requires_grad=True)
        positive=torch.tensor([True,False]);details=dict(fused=corrected,semantic=semantic)
        source_supervision(details,positive,'branch_consistent').backward()
        self.assertLess(float(semantic.grad[0]),0.);self.assertLess(float(corrected.grad[0]),0.)
        with self.assertRaises(ValueError):source_supervision(details,positive,'paired_assignment')

    def test_gate_correctness_and_saturation_are_handled_stably(self):
        logit=torch.tensor(79.,requires_grad=True)
        loss=protected_gate(logit,torch.tensor(80.),True)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logit.grad))
        logit=torch.tensor(-3.,requires_grad=True)
        loss=protected_gate(logit,torch.tensor(-8.),True)
        torch.testing.assert_close(loss,F.binary_cross_entropy_with_logits(logit,torch.tensor(1.)))

    def test_conflicting_token_teacher_uses_gold_and_normalizes_by_tokens(self):
        logits=torch.zeros(2,2,requires_grad=True)
        teacher=torch.tensor([[10.,-10.],[-10.,10.]])
        labels=torch.tensor([1,1]);mask=torch.ones(2)
        loss=protected_tokens(logits,teacher,labels,mask)
        repeated=protected_tokens(logits.repeat(3,1),teacher.repeat(3,1),labels.repeat(3),mask.repeat(3))
        torch.testing.assert_close(loss,repeated)
        loss.backward();self.assertLess(float(logits.grad[0,1]),0.)

    def test_zero_probability_teacher_entries_do_not_create_nan(self):
        logits=torch.tensor([[0.,-float('inf')]],requires_grad=True)
        loss=protected_tokens(logits,logits.detach(),torch.tensor([0]),torch.ones(1))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_frozen_endpoint_parameters_do_not_imply_frozen_behavior(self):
        torch.manual_seed(811)
        head=NativeSpanReadout(4,conditional_endpoints=True)
        nn.init.normal_(head.endpoint[-1].weight,std=.3);head.requires_grad_(False)
        words=torch.randn(2,4,requires_grad=True);q=torch.randn(4)
        candidates=[dict(start=0,end=e,properties=(0.,)*12) for e in (0,1)]
        teacher=head.score_candidates(candidates,q,words.detach(),torch.zeros(2)).detach()
        changed=head.score_candidates(candidates,q,words+torch.tensor([[2.,0.,0.,0.],[0.,1.,0.,0.]]),torch.zeros(2))
        self.assertFalse(torch.allclose(teacher.softmax(0),changed.softmax(0)))
        positive=torch.zeros(2,dtype=torch.bool);positive[teacher.argmax()]=True
        protected_decision(changed,teacher,positive).backward()
        self.assertGreater(float(words.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_cached_emission_uses_shared_guidance_and_ignores_labels(self):
        torch.manual_seed(812);memory=NativeQuestionMemory(4,span_readout=True,stable_span_gate=True,conditional_copy=True)
        model=SimpleNamespace(native_question_memory=memory,_construction_logits_from_guided=nn.Linear(4,7))
        packet=dict(backbone_hidden=torch.randn(2,4),residual_mask=torch.tensor([True,False]),emitted={0:(),1:(2,)},labels=torch.tensor([2,3]))
        q=torch.randn(4);recalled=torch.randn(4)
        plan=dict(candidates=[dict(tokens=(2,3))],log_probabilities=torch.zeros(1),copy_context=recalled)
        result=cached_emission(model,packet,q,recalled,plan)
        packet['labels']=torch.tensor([6,6])
        torch.testing.assert_close(result,cached_emission(model,packet,q,recalled,plan),rtol=0,atol=0)
        guided=memory.guide_readout(packet['backbone_hidden'],q,recalled,packet['residual_mask'])
        torch.testing.assert_close(guided[1],packet['backbone_hidden'][1],rtol=0,atol=0)

    def test_corrected_fusion_uses_complete_corrected_distribution(self):
        torch.manual_seed(813)
        layer=NativeSemanticAlignment(4,prior_scaling=True,normalized_fusion=True,fusion_experts='corrected_legacy_v1')
        inputs=layer.inputs([(1,),(2,),(3,)],[(4,)],nn.Embedding(8,4))
        prior=torch.tensor([5.,0.,-5.]);q=torch.zeros(4)
        details=layer.score_details(inputs,prior,q)
        torch.testing.assert_close(details['corrected'],prior)
        torch.testing.assert_close(details['fused'].exp().sum(),torch.tensor(1.))
        with torch.no_grad():
            layer.prior_scale.bias.fill_(-8.)
            layer.fusion_gate[-1].bias.fill_(-80.)
        details=layer.score_details(inputs,prior,q)
        torch.testing.assert_close(details['fused'],details['corrected'].log_softmax(0),rtol=1e-5,atol=1e-5)
        self.assertFalse(torch.allclose(details['fused'],prior.log_softmax(0)))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
