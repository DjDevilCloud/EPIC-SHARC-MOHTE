import unittest
import torch
from native_training_objectives import causal_owner_supervision
from native_antecedent_memory import NativeAntecedentMemory


class CausalOwnerSupervisionTests(unittest.TestCase):
    def test_absolute_mass_not_conditional_assignment(self):
        logs=torch.tensor([.2,.1]).log().requires_grad_()
        loss=causal_owner_supervision(dict(ownership_log_probabilities=[logs]),[dict(source_index=0,owner_index=0)])
        torch.testing.assert_close(loss,-torch.tensor(.2).log());loss.backward()
        torch.testing.assert_close(logs.grad,torch.tensor([-1.,0.]))

    def test_production_logs_match_weights_and_keep_underflow_gradient(self):
        bank=NativeAntecedentMemory(4)
        with torch.no_grad():bank.role[-1].weight.zero_();bank.role[-1].bias.fill_(-1000.)
        _,details=bank._score_scopes([(1,)],torch.randn(1,4),[(((1,),),(0.,0.,0.,0.))],torch.randn(1,4),[[0]])
        self.assertEqual(float(details['ownership'][0][0]),0.)
        loss=causal_owner_supervision(details,[dict(source_index=0,owner_index=0)])
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(bank.role[-1].bias.grad).all());self.assertLess(float(bank.role[-1].bias.grad),-.9)

    def test_duplicate_and_invalid_addresses_rejected(self):
        details=dict(ownership_log_probabilities=[torch.tensor([-.1,-3.])])
        for targets in ([],[dict(source_index=1,owner_index=0)], [dict(source_index=0,owner_index=True)],
                        [dict(source_index=0,owner_index=0),dict(source_index=0,owner_index=1)]):
            with self.subTest(targets=targets),self.assertRaises(ValueError):causal_owner_supervision(details,targets)

if __name__=='__main__':unittest.main()
