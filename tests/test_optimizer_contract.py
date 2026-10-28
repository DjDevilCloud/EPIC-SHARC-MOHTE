import copy
import unittest

import torch

from muon_optim import PrecisionAdaptiveHierarchicalOptimizer
from train import _optimizer_state_is_shape_compatible


def group(parameter, rule='rowwise', nested=False, beta=.5):
    return dict(params=[parameter],update_rule=rule,parameter_role='table' if rule=='rowwise' else 'scalar',
        hierarchy_tier='mid',lr=.03,weight_decay=0.,state_dtype=torch.float32,betas=(.9,.95),eps=1e-8,
        nested_learning_enabled=nested,nested_update_interval=2,nested_update_ema_beta=beta)


class OptimizerContractTests(unittest.TestCase):
    def test_first_table_update_descends_for_both_signs_and_zero_mean_rows(self):
        for values in ([[-1.,-1.]],[[1.,-1.]],[[1.,1.]]):
            with self.subTest(gradient=values):
                p=torch.nn.Parameter(torch.zeros(1,2));gradient=torch.tensor(values)
                optimizer=PrecisionAdaptiveHierarchicalOptimizer([group(p)])
                p.grad=gradient;optimizer.step()
                self.assertLess(float((p.detach()*gradient).sum()),0.)
                self.assertTrue(torch.all(p.detach()*gradient<0))

    def test_table_learning_reduces_mixed_sign_quadratic(self):
        p=torch.nn.Parameter(torch.zeros(2,3));target=torch.tensor([[-1.,2.,-3.],[2.,-2.,1.]])
        optimizer=PrecisionAdaptiveHierarchicalOptimizer([group(p)])
        initial=float((p.detach()-target).square().mean())
        for _ in range(150):
            optimizer.zero_grad();(p-target).square().mean().backward();optimizer.step()
        self.assertLess(float((p.detach()-target).square().mean()),initial*.05)

    def test_nested_gradient_uses_debiased_window_ema(self):
        p=torch.nn.Parameter(torch.zeros(1));g=group(p,'adamw',True,.5)
        optimizer=PrecisionAdaptiveHierarchicalOptimizer([g]);g=optimizer.param_groups[0]
        self.assertIsNone(optimizer._nested_prepare_grad(g,p,torch.tensor([2.]),update_due=False))
        first=optimizer._nested_prepare_grad(g,p,torch.tensor([4.]),update_due=True)
        torch.testing.assert_close(first,torch.tensor([3.]))
        self.assertIsNone(optimizer._nested_prepare_grad(g,p,torch.tensor([-2.]),update_due=False))
        second=optimizer._nested_prepare_grad(g,p,torch.tensor([-4.]),update_due=True)
        torch.testing.assert_close(second,torch.tensor([-1.]))

    def test_nested_beta_changes_updates(self):
        results=[]
        for beta in (0.,.99):
            p=torch.nn.Parameter(torch.ones(2));optimizer=PrecisionAdaptiveHierarchicalOptimizer([group(p,'adamw',True,beta)])
            for values in ([1.,2.],[2.,1.],[-1.,4.],[3.,-2.]):
                p.grad=torch.tensor(values);optimizer.step()
            results.append(p.detach().clone())
        self.assertFalse(torch.allclose(*results,rtol=0,atol=1e-6))

    def test_pending_gradient_flushes_at_boundary_without_current_gradient(self):
        for rule in ('adamw','muon','rowwise'):
            p=torch.nn.Parameter(torch.zeros(2,2));optimizer=PrecisionAdaptiveHierarchicalOptimizer([group(p,rule,True)])
            p.grad=torch.ones_like(p);optimizer.step();self.assertTrue(torch.equal(p,torch.zeros_like(p)))
            p.grad=None;optimizer.step();self.assertGreater(float(p.detach().norm()),0.)
            after=p.detach().clone();optimizer.step();optimizer.step()
            torch.testing.assert_close(p,after,rtol=0,atol=0)

    def test_midwindow_resume_is_exact_for_all_update_rules(self):
        for rule in ('adamw','muon','rowwise'):
            with self.subTest(rule=rule):
                p=torch.nn.Parameter(torch.ones(2,2));optimizer=PrecisionAdaptiveHierarchicalOptimizer([group(p,rule,True)])
                p.grad=torch.tensor([[1.,-2.],[3.,-1.]]);optimizer.step()
                state=copy.deepcopy(optimizer.state_dict())
                restored_p=torch.nn.Parameter(p.detach().clone())
                restored=PrecisionAdaptiveHierarchicalOptimizer([group(restored_p,rule,True)])
                restored.load_state_dict(state)
                self.assertTrue(_optimizer_state_is_shape_compatible(restored))
                for gradient in (None,torch.tensor([[-1.,2.],[1.,-2.]]),torch.ones(2,2)):
                    p.grad=gradient;restored_p.grad=gradient
                    optimizer.step();restored.step()
                    torch.testing.assert_close(p,restored_p,rtol=0,atol=0)

    def test_legacy_state_rejected_before_mutation(self):
        p=torch.nn.Parameter(torch.ones(2,2));optimizer=PrecisionAdaptiveHierarchicalOptimizer([group(p)])
        legacy=optimizer.state_dict()
        legacy.pop('precision_adaptive_schema_version',None)
        legacy['param_groups'][0]['lr']=9.
        with self.assertRaisesRegex(ValueError,'version|schema|legacy'):
            optimizer.load_state_dict(legacy)
        self.assertEqual(optimizer.param_groups[0]['lr'],.03)

    def test_version_and_compressed_variance_shape_are_explicit(self):
        p=torch.nn.Parameter(torch.zeros(2,3));optimizer=PrecisionAdaptiveHierarchicalOptimizer([group(p)])
        p.grad=torch.ones_like(p);optimizer.step()
        self.assertGreaterEqual(optimizer.state_dict()['precision_adaptive_schema_version'],2)
        self.assertEqual(optimizer.state[p]['exp_avg'].shape,p.shape)
        self.assertEqual(optimizer.state[p]['exp_avg_sq_row'].shape,(2,1))
        self.assertTrue(_optimizer_state_is_shape_compatible(optimizer))
        optimizer.state[p]['exp_avg_sq_row']=torch.zeros(2,2)
        self.assertFalse(_optimizer_state_is_shape_compatible(optimizer))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
