import copy
import unittest
from unittest.mock import patch

import torch

import native_qa_context_study as study
from native_question_memory import NativeQuestionMemory


class ContextStudyTests(unittest.TestCase):
    def test_common_initial_tensors_and_saved_arm_mode(self):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        _,t,parent=RuntimeSignatureTests().setup_model()
        parent.native_question_memory=NativeQuestionMemory(parent.cfg.d_model,semantic_alignment=True)
        state=torch.get_rng_state().clone()
        try:
            with patch.object(study,'BASE_LOAD',side_effect=lambda seed,path:(copy.deepcopy(parent),t)):
                with patch.object(study,'ACTIVE_ARM','late_context'):late,_=study.load(47)
                with patch.object(study,'ACTIVE_ARM','early_context'):early,_=study.load(47)
                self.assertTrue(all(torch.equal(v,early.state_dict()[k]) for k,v in late.state_dict().items()))
                self.assertEqual(late.cfg.native_question_memory_question_attention_stage,'post_context')
                self.assertEqual(early.cfg.native_question_memory_question_attention_stage,'pre_context')
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
            with patch.object(study,'BASE_LOAD',return_value=(early,t)),patch.object(study,'ACTIVE_ARM','late_context'):
                reloaded,_=study.load(47,study.OUT/'seed47/early_context/source/final/model.pt')
                self.assertEqual(reloaded.cfg.native_question_memory_question_attention_stage,'pre_context')
        finally:torch.set_rng_state(state)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
