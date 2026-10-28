import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import cli
from config import PrismalWaveConfig
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory
from qa_generation import resolve_generation_options, MODEL_LEGACY, API_LEGACY, CLI_LEGACY, NATIVE_QA_EVAL
from source_answer_span import CompleteAnswerSpanPredictor
from train import generate_text, save_checkpoint, load_bundle_from_checkpoint


class QAGenerationProfileTests(unittest.TestCase):
    def test_shared_profile_is_identical_across_entrypoint_defaults(self):
        for defaults in (MODEL_LEGACY,API_LEGACY,CLI_LEGACY):
            options,record=resolve_generation_options('native_qa_eval_v1',{},defaults)
            self.assertEqual(options,NATIVE_QA_EVAL)
            self.assertTrue(record['canonical'])
        with self.assertRaises(ValueError):resolve_generation_options('native_qa_eval_v1',{'answer':'A'})

    def test_explicit_override_is_honored_and_recorded(self):
        options,record=resolve_generation_options('native_qa_eval_v1',{'max_new_tokens':64,'temperature':.5})
        self.assertEqual(options['max_new_tokens'],64)
        self.assertEqual(record['overrides'],dict(max_new_tokens=64,temperature=.5))
        self.assertFalse(record['canonical'])

    def test_old_defaults_and_old_config_stay_legacy(self):
        self.assertEqual(PrismalWaveConfig.from_dict({}).qa_generation_profile,'legacy')
        for defaults in (MODEL_LEGACY,API_LEGACY,CLI_LEGACY):
            self.assertEqual(resolve_generation_options('legacy',{},defaults)[0],defaults)
        with self.assertRaises(ValueError):PrismalWaveConfig(qa_generation_profile='unknown')

    def mock_api(self):
        model=Mock();model.cfg=PrismalWaveConfig(qa_generation_profile='native_qa_eval_v1',use_fst=False)
        model.to.return_value=model
        tokenizer=PrismalTokenizer()
        answer='A sufficiently long answer.'
        payload=CompleteAnswerSpanPredictor.payload(tokenizer,answer)
        model.generate.side_effect=lambda ids,**kwargs:torch.cat((ids,torch.tensor([payload],dtype=torch.long)),1)
        return model,tokenizer,answer

    def test_python_api_returns_raw_answer_and_canonical_settings(self):
        model,t,answer=self.mock_api()
        result=generate_text(model,t,'Context:\nOne fact.\nQuestion:\nWhat?',torch.device('cpu'),return_generation_record=True)
        self.assertEqual(result['raw'].strip(),answer)
        self.assertTrue(result['ended']);self.assertTrue(result['canonical'])
        self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'],128)
        self.assertFalse(model.generate.call_args.kwargs['use_speculative_decoding'])

    def test_cli_uses_saved_profile_and_honors_explicit_noncanonical_override(self):
        for flags,canonical,cap in [([],True,128),(['--max-new-tokens','64'],False,64)]:
            model,t,answer=self.mock_api()
            stream=io.StringIO()
            with patch.object(cli,'load_bundle_from_checkpoint',return_value=(model,t,model.cfg)), \
                 patch.object(cli,'maybe_compile_model',side_effect=lambda m,**kw:m), \
                 contextlib.redirect_stdout(stream):
                result=cli.main(['--device','cpu','infer','--checkpoint','unused','--prompt','What?',
                                 '--generation-json','--no-torch-compile']+flags)
            self.assertEqual(result,0)
            record=json.loads(stream.getvalue());self.assertEqual(record['canonical'],canonical)
            self.assertEqual(record['options']['max_new_tokens'],cap)

    def test_saved_profile_can_complete_long_payload_and_reload(self):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        _,t,m=RuntimeSignatureTests().setup_model()
        m.cfg.use_native_question_memory=True;m.cfg.native_question_memory_span_readout=True
        m.cfg.native_question_memory_stable_span_gate=True;m.cfg.native_question_memory_conditional_copy=True
        m.cfg.qa_readout_route='native';m.cfg.qa_generation_profile='native_qa_eval_v1'
        m.native_question_memory=NativeQuestionMemory(m.cfg.d_model,span_readout=True,stable_span_gate=True,conditional_copy=True).eval()
        with torch.no_grad():m.native_question_memory.span_readout.copy_gate.bias.fill_(100.)
        answer=' '.join(['abcdefgh']*12)
        expected=CompleteAnswerSpanPredictor.payload(t,answer)
        self.assertGreater(len(expected),64);self.assertLess(len(expected),128)
        prompt='Context:\n'+answer+'\nQuestion:\nWhich words?'
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(m,folder,tokenizer=t)
            loaded,lt,cfg=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu',load_training_state=False)
            self.assertEqual(cfg.qa_generation_profile,'native_qa_eval_v1')
            loaded.eval();loaded._prismal_tokenizer=lt
            original=loaded.native_question_memory.span_readout.candidates
            # Evaluator-only exact-proposal intervention; the saved gate is unchanged.
            def exact_candidate(state,tokenizer,selected):
                return [c for c in original(state,tokenizer,selected) if c['text']==answer]
            with patch.object(loaded.native_question_memory.span_readout,'candidates',side_effect=exact_candidate):
                result=generate_text(loaded,lt,prompt,torch.device('cpu'),return_generation_record=True)
                self.assertEqual(result['tokens'],list(expected));self.assertTrue(result['ended'])
                short=generate_text(loaded,lt,prompt,torch.device('cpu'),max_new_tokens=64,return_generation_record=True)
                self.assertFalse(short['canonical']);self.assertTrue(short['cap_hit']);self.assertFalse(short['ended'])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
