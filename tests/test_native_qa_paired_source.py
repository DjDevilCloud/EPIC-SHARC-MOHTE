import unittest

import torch
import torch.nn.functional as F

from native_qa_paired_source import assignment_margin, pairs


class PairedSourceTests(unittest.TestCase):
    def test_question_independent_salience_cancels_and_gradients_distinguish_questions(self):
        a=torch.tensor([8.,-2.,4.],requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        left=torch.tensor([True,False,False]);right=torch.tensor([False,True,False])
        margin=assignment_margin(a,b,left,right)
        self.assertEqual(float(margin.detach()),0.)
        F.softplus(-margin).backward()
        self.assertLess(float(a.grad[0]),0.)
        self.assertGreater(float(a.grad[1]),0.)
        self.assertGreater(float(b.grad[0]),0.)
        self.assertLess(float(b.grad[1]),0.)
        self.assertEqual(float(a.grad[2]),0.)

    def test_normalization_and_question_order_do_not_change_margin(self):
        a=torch.tensor([3.,1.,2.]);b=torch.tensor([0.,4.,2.])
        p=torch.tensor([True,False,False]);q=torch.tensor([False,True,False])
        expected=assignment_margin(a,b,p,q)
        torch.testing.assert_close(expected,assignment_margin(a.log_softmax(0),b.log_softmax(0),p,q))
        torch.testing.assert_close(expected,assignment_margin(b,a,q,p))
        self.assertGreater(float(expected),0.)
        self.assertLess(float(assignment_margin(b,a,p,q)),0.)

    def test_multiple_gold_occurrences_and_invalid_pairs(self):
        a=torch.tensor([1.,2.,3.]);b=torch.tensor([3.,2.,1.])
        p=torch.tensor([True,True,False]);q=torch.tensor([False,False,True])
        margin=assignment_margin(a,b,p,q)
        self.assertTrue(torch.isfinite(margin))
        with self.assertRaises(ValueError):assignment_margin(a,b,p,p)
        with self.assertRaises(ValueError):assignment_margin(a,b,p,torch.zeros(3,dtype=torch.bool))

    def test_pairing_uses_identical_context_and_distinct_targets(self):
        def row(context,positive):
            return dict(row=dict(context=context),words=['alice','bob'],positive=torch.tensor(positive))
        xs=[row('same',[True,False]),row('same',[False,True])]
        self.assertEqual(pairs(xs),{0:1,1:0})
        with self.assertRaises(ValueError):pairs(xs[:1])
        xs[1]['positive']=xs[0]['positive']
        with self.assertRaises(ValueError):pairs(xs)


if __name__=='__main__':unittest.main()
