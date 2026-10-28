import tempfile
import unittest
from pathlib import Path

import torch

from native_semantic_alignment import NativeSemanticAlignment
from native_question_memory import NativeQuestionMemory
from train import save_checkpoint, load_bundle_from_checkpoint


class QuestionContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(431)
        self.embedding = torch.nn.Embedding(20, 8)
        self.inputs = [NativeSemanticAlignment.inputs([(1,), (2, 3), (4,)], [(5,), (6,)], self.embedding),
                       NativeSemanticAlignment.inputs([(2, 3), (7,)], [(8,)], self.embedding)]
        for inputs in self.inputs:
            inputs['units']=inputs['units'].detach()

    def layer(self, stage):
        layer = NativeSemanticAlignment(8, question_attention=True, question_attention_stage=stage)
        with torch.no_grad():
            layer.question_residual.weight.normal_(std=.1)
            layer.score[-1].weight.normal_(std=.1)
        return layer

    def test_placements_have_identical_parameters_and_zero_initialization_matches_parent(self):
        parent = NativeSemanticAlignment(8)
        layers = []
        for stage in ('pre_context', 'post_context'):
            layer = NativeSemanticAlignment(8, question_attention=True, question_attention_stage=stage)
            missing, extra = layer.load_state_dict(parent.state_dict(), strict=False)
            self.assertFalse(extra)
            self.assertTrue(all(k.startswith(('question_attention.', 'question_residual.')) for k in missing))
            layers.append(layer)
            for a, b in zip(layer.encode(self.inputs[0]), parent.encode(self.inputs[0])):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual({k: v.shape for k, v in layers[0].state_dict().items()},
                         {k: v.shape for k, v in layers[1].state_dict().items()})

    def test_single_and_packed_paths_match_with_active_conditioning(self):
        for stage in ('pre_context', 'post_context'):
            layer = self.layer(stage)
            many = layer.encode_many(self.inputs)
            for x, outputs in zip(self.inputs, many):
                for actual, expected in zip(outputs, layer.encode(x)):
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)

    def test_padding_and_other_questions_do_not_change_an_example(self):
        layer = self.layer('pre_context')
        original = layer.encode_many(self.inputs)[0]
        unrelated = NativeSemanticAlignment.inputs([(9,)] * 15, [(10,), (11,), (12,)] * 4, self.embedding)
        changed = layer.encode_many([self.inputs[0], unrelated])[0]
        for a, b in zip(original, changed):
            torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-5)

    def test_question_evidence_enters_source_context_only_in_early_arm(self):
        for stage in ('pre_context', 'post_context'):
            layer = self.layer(stage)
            observed = []
            hook = layer.context.register_forward_pre_hook(lambda m, args: observed.append(args[0].detach().clone()))
            layer.encode(self.inputs[0]); first = observed[-1]
            observed.clear()
            changed = dict(self.inputs[0], question=self.inputs[0]['source'][:2])
            layer.encode(changed); second = observed[-1]
            hook.remove()
            if stage == 'pre_context':
                self.assertGreater(float((first-second).abs().max()), 1e-5)
            else:
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_zero_residual_gets_gradient_and_can_open_attention_path(self):
        layer = NativeSemanticAlignment(8, question_attention=True, question_attention_stage='pre_context')
        with torch.no_grad():layer.score[-1].weight.normal_(std=.1)
        loss = layer.encode(self.inputs[0])[0].square().sum()
        loss.backward()
        self.assertGreater(float(layer.question_residual.weight.grad.norm()), 0.)
        with torch.no_grad():layer.question_residual.weight.add_(-.1*layer.question_residual.weight.grad)
        layer.zero_grad(); layer.encode(self.inputs[0])[0].sum().backward()
        self.assertGreater(float(layer.question_attention.in_proj_weight.grad.norm()), 0.)

    def test_future_answer_fields_are_not_encoder_inputs(self):
        layer = self.layer('pre_context'); x=self.inputs[0]
        for a,b in zip(layer.encode(dict(x, answer='one')),layer.encode(dict(x, answer='unrelated continuation'))):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_saved_stage_and_active_weights_survive_reload(self):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        _, t, model = RuntimeSignatureTests().setup_model()
        cfg=model.cfg;cfg.use_native_question_memory=True;cfg.native_question_memory_semantic_alignment=True
        cfg.native_question_memory_question_attention=True;cfg.native_question_memory_question_attention_stage='pre_context'
        model.native_question_memory=NativeQuestionMemory(cfg.d_model,semantic_alignment=True,
            question_attention=True,question_attention_stage='pre_context')
        layer=model.native_question_memory.semantic_alignment
        with torch.no_grad():layer.question_residual.weight.normal_(std=.1)
        inputs=layer.inputs([(1,), (2,)],[(3,)],model.construction_embedding)
        expected=layer.encode(inputs)
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(model,folder,tokenizer=t)
            loaded,_,cfg=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu',load_training_state=False)
            self.assertEqual(cfg.native_question_memory_question_attention_stage,'pre_context')
            self.assertEqual(loaded.native_question_memory.semantic_alignment.question_attention_stage,'pre_context')
            for a,b in zip(expected,loaded.native_question_memory.semantic_alignment.encode(inputs)):
                torch.testing.assert_close(a,b)

    def test_invalid_or_disabled_early_attention_is_rejected(self):
        for stage in ('unknown','pre_context'):
            with self.assertRaises(ValueError):NativeSemanticAlignment(8,question_attention_stage=stage)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
