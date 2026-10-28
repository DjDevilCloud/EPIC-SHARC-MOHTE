import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import native_qa_objective_study as study
from native_question_memory import NativeQuestionMemory
from native_semantic_alignment import NativeSemanticAlignment
from train import save_checkpoint,load_bundle_from_checkpoint


class ObjectiveStudyTests(unittest.TestCase):
    def test_measurement_uses_reload_equivalent_flags_and_restores_training_state(self):
        model=nn.Linear(2,2);model.bias.requires_grad_(False)
        def scoring(m,packets):
            self.assertTrue(all(not p.requires_grad for p in m.parameters()))
            return {},{}
        with patch.object(study.previous,'evaluate',side_effect=scoring):
            study.evaluate(model,{})
        self.assertTrue(model.weight.requires_grad);self.assertFalse(model.bias.requires_grad)
        with patch.object(study.previous,'evaluate',side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):study.evaluate(model,{})
        self.assertTrue(model.weight.requires_grad);self.assertFalse(model.bias.requires_grad)

    def test_corrected_fusion_expert_selection_survives_checkpoint_reload(self):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        _,t,model=RuntimeSignatureTests().setup_model()
        cfg=model.cfg;cfg.use_native_question_memory=True
        cfg.native_question_memory_semantic_alignment=True;cfg.native_question_memory_normalized_fusion=True
        cfg.native_question_memory_fusion_experts='corrected_legacy_v1'
        model.native_question_memory=NativeQuestionMemory(cfg.d_model,semantic_alignment=True,normalized_fusion=True,fusion_experts='corrected_legacy_v1')
        layer=model.native_question_memory.semantic_alignment
        inputs=layer.inputs([(1,),(2,)],[(3,)],model.construction_embedding)
        prior=torch.tensor([2.,-1.]);question=torch.zeros(cfg.d_model)
        expected=layer.score_details(inputs,prior,question)['fused'].detach()
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(model,folder,tokenizer=t)
            loaded,_,reloaded_cfg=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu',load_training_state=False)
            self.assertEqual(reloaded_cfg.native_question_memory_fusion_experts,'corrected_legacy_v1')
            self.assertEqual(loaded.native_question_memory.semantic_alignment.fusion_experts,'corrected_legacy_v1')
            actual=loaded.native_question_memory.semantic_alignment.score_details(inputs,prior,question)['fused']
            torch.testing.assert_close(expected,actual)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
