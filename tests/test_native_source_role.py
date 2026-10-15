import unittest
import torch
from native_source_role import NativeSourceRole

class NativeSourceRoleTests(unittest.TestCase):
 def test_candidate_identity_and_subword_count_do_not_enter_its_role_features(self):
  calls=[]
  def word_vector(word):
   calls.append(word)
   if word in ((11,), (99,98,97)):raise AssertionError('candidate identity leaked')
   return torch.tensor([float(word[0]),1.,2.,3.])
  encode=lambda scopes:torch.stack([scope.mean(0) for scope in scopes])
  a=[((11,),((1,),(2,),(3,)))];b=[((99,98,97),((1,),(2,),(3,)))]
  x=NativeSourceRole.features(a,word_vector,encode,torch.zeros(4))
  y=NativeSourceRole.features(b,word_vector,encode,torch.zeros(4))
  torch.testing.assert_close(x,y,rtol=0,atol=0)
  self.assertEqual(set(calls),{(1,),(2,),(3,)})
 def test_clause_boundary_and_empty_context_are_respected(self):
  def fail(*args):raise AssertionError('crossed a clause boundary')
  source=[((11,),()),((12,),())]
  features=NativeSourceRole.features(source,fail,fail,torch.zeros(4))
  torch.testing.assert_close(features[:,:8],torch.zeros(2,8))
  torch.testing.assert_close(features[:,8:],torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]))
  self.assertEqual(tuple(NativeSourceRole.features([],fail,fail,torch.zeros(4)).shape),(0,12))
 def test_neutral_initialization_and_role_gradients(self):
  with torch.random.fork_rng():
   torch.manual_seed(31);head=NativeSourceRole(4);features=torch.randn(3,12);request=torch.randn(4)
   scores=head(request,features);torch.testing.assert_close(scores,torch.zeros(3),rtol=0,atol=0)
   (-scores.log_softmax(0)[1]).backward()
   self.assertGreater(float(head.source[-1].weight.grad.abs().sum()),0.)
