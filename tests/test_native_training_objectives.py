import unittest
import torch
from native_training_objectives import decision_margin, preserve_decision_margin


class NativeTrainingObjectiveTests(unittest.TestCase):
    def test_correct_order_and_offsets_preserve_the_margin(self):
        positive = torch.tensor([True, False, True])
        teacher = torch.tensor([2., 0., 1.])
        torch.testing.assert_close(decision_margin(teacher, positive), teacher[[0, 2]].logsumexp(0))
        torch.testing.assert_close(preserve_decision_margin(teacher + 100., teacher, positive), torch.tensor(0.), atol=1e-5, rtol=0.)

    def test_regression_gradient_restores_gold_and_teacher_is_frozen(self):
        positive = torch.tensor([True, False, False])
        teacher = torch.tensor([5., 0., -1.], requires_grad=True)
        current = torch.tensor([0., 1., -1.], requires_grad=True)
        loss = preserve_decision_margin(current, teacher, positive)
        torch.testing.assert_close(loss, torch.tensor(4.))  # capped teacher 3, current -1
        loss.backward()
        self.assertLess(float(current.grad[0]), 0.)
        self.assertGreater(float(current.grad[1]), 0.)
        self.assertEqual(float(current.grad[2]), 0.)
        self.assertIsNone(teacher.grad)

    def test_no_alternative_and_invalid_annotations(self):
        logits = torch.tensor([1., 2.], requires_grad=True)
        positive = torch.ones(2, dtype=torch.bool)
        loss = preserve_decision_margin(logits, logits, positive)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros(2))
        for mask in (torch.zeros(2, dtype=torch.bool), torch.ones(3, dtype=torch.bool), torch.ones(2)):
            with self.assertRaises(ValueError):
                decision_margin(logits, mask)
        with self.assertRaises(ValueError):
            preserve_decision_margin(logits, logits, positive, max_margin=-1.)
