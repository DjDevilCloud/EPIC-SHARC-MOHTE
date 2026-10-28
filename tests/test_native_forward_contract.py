import copy
import tempfile
import unittest
from pathlib import Path
import torch
from data import PrismalTokenizer,TextWindowDataset,build_collate_fn
from native_qa_format import format_native_qa
from native_question_memory import NativeQuestionMemory


class FramingTests(unittest.TestCase):
    def memory(self,**extra):
        return NativeQuestionMemory(8,span_readout=True,representation='backbone_context_v1',framing='blocks_v1',**extra)

    def observe(self,text,m=None):
        t=PrismalTokenizer();m=m or self.memory();ids=torch.tensor([t.encode(text,add_special_tokens=False)])
        with torch.no_grad():_,states,records=m(ids,torch.randn(1,ids.shape[1],8),None,t,return_readout=True)
        return t,m,ids,states[0],records

    def test_literal_headers_are_source_and_payload_survives(self):
        context='The label says:\nQuestion: printed text.\nContext: printed again.\nAda was born in Oakfield.'
        t,m,ids,state,_=self.observe(format_native_qa(context,'Where was Ada born?',framing='blocks_v1'))
        words=[t.decode(list(w),clean_text=False,collapse_structure=False) for w,_ in state.source]
        self.assertIn('question',words);self.assertIn('context',words);self.assertIn('oakfield',words)
        self.assertEqual([t.decode(list(w),clean_text=False,collapse_structure=False) for w in state.question],['where','was','ada','born'])
        candidates=m.span_readout.candidates(state,t,list(range(len(words))))
        self.assertTrue(any(c['text']=='Oakfield' for c in candidates))

    def test_training_prefix_and_generation_share_codec_and_atomic_group(self):
        t=PrismalTokenizer();context='Question: printed source.\nAda owns $75.50.';question='How much does Ada own?'
        prefix=format_native_qa(context,question,framing='blocks_v1')
        text=format_native_qa(context,question,'$75.50',framing='blocks_v1')
        ds=TextWindowDataset([text],t,seq_len=8,preserve_grounded_qa_context=True)
        self.assertEqual(len(ds),1);self.assertIsNotNone(ds[0].source_group)
        b=build_collate_fn(t.pad_id,t.signature_pad_id)([ds[0]])
        generation=t.prepare_generation_hierarchy(prefix)
        self.assertEqual(b[0][0,:len(generation.token_ids)].tolist(),generation.token_ids)

    def test_reserved_controls_and_malformed_blocks_fail(self):
        with self.assertRaisesRegex(ValueError,'reserved'):
            format_native_qa('Literal <EOI> is reserved','Which marker?',framing='blocks_v1')
        for text in ('<BOI><BOP>source<EOP><EOI><BOO>', '<BOI><BOP>source<BOP>question<EOP><EOI><BOO>',
                     '<BOI><BOP>source<EOP><BOP>question<EOP><BOP>third<EOP><EOI><BOO>'):
            with self.subTest(text=text),self.assertRaises(ValueError):self.observe(text)

    def test_new_input_resets_and_output_does_not_reopen_input(self):
        first=format_native_qa('Ada owns 12.','What does Ada own?','12',framing='blocks_v1')
        second=format_native_qa('Ben owns 35.','What does Ben own?',framing='blocks_v1')
        t,_,_,state,_=self.observe(first+second)
        self.assertEqual([t.decode(list(w),clean_text=False,collapse_structure=False) for w,_ in state.source],['ben','owns','35'])

    def test_framed_surface_addresses_match_observed_tokens(self):
        t=PrismalTokenizer();m=self.memory();text=format_native_qa('Question: printed. Ada owns $75.50.','What does Ada own?',framing='blocks_v1')
        ids=torch.tensor([t.encode(text,add_special_tokens=False)])
        hidden=torch.arange(ids.shape[1]*8,dtype=torch.float32).reshape(1,ids.shape[1],8)
        with torch.no_grad():_,states=m(ids,hidden,None,t)
        state=states[0]
        self.assertTrue(all(int(ids[0,pos])==token for pos,token in state.surface))
        for vector,(_,stop,_) in zip(state.source_vectors,state.source_ranges):
            torch.testing.assert_close(vector,hidden[0,stop-1],rtol=0,atol=0)

    def test_model_saved_policy_and_python_api_use_same_prompt(self):
        from tests.test_native_backbone_representation import NativeBackboneTests
        from train import save_checkpoint,load_bundle_from_checkpoint,generate_text
        m,t=NativeBackboneTests().model(all_heads=True)
        m.cfg.native_question_memory_framing='blocks_v1';m.native_question_memory.framing='blocks_v1';m.requires_grad_(False)
        context='Question: label only.\nRed costs 12.';question='What does Red cost?'
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(m,folder,tokenizer=t)
            loaded,rt,cfg=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu',load_training_state=False)
            self.assertEqual(cfg.native_question_memory_framing,'blocks_v1');self.assertEqual(loaded.native_question_memory.framing,'blocks_v1')
            record=generate_text(loaded,rt,question,torch.device('cpu'),qa_context=context,max_new_tokens=2,return_generation_record=True)
            expected=rt.prepare_generation_hierarchy(format_native_qa(context,question,framing='blocks_v1')).token_ids
            self.assertEqual(record['prompt_ids'],expected)


class EventTests(FramingTests):
    # Explicit inheritance shares observation helpers; framing tests remain
    # applicable to the combined policies as well.
    def memory(self,**extra):
        return NativeQuestionMemory(8,span_readout=True,representation='backbone_context_v1',framing='blocks_v1',
            events='causal_clause_v2',antecedent_memory=True,owner_evidence=True,**extra)

    def test_decimal_initials_semicolon_events_preserve_payloads(self):
        context='A. R. Doe owns $75.50; Ben owns $20.25.'
        t,m,_,state,_=self.observe(format_native_qa(context,'How much does A. R. Doe own?',framing='blocks_v1'))
        events=[[t.decode(list(w),clean_text=False,collapse_structure=False) for w in e.words] for e in state.antecedents]
        self.assertEqual(events,[['a','r','doe','owns','75','50'],['ben','owns','20','25']])
        candidates=m.span_readout.candidates(state,t,list(range(len(state.source))))
        self.assertTrue(any(c['text']=='$75.50' for c in candidates))

    def test_period_followed_by_space_and_number_is_boundary(self):
        t,_,_,state,_=self.observe(format_native_qa('Ada owns 75. 50 people arrived.','Who owns 75?',framing='blocks_v1'))
        self.assertEqual(len(state.antecedents),2)

    def test_recent_context_bounded_and_early_source_causal(self):
        t=PrismalTokenizer();m=self.memory();text=format_native_qa('Ada with the red coat near the gate is the subject. He was born in Oakfield.','Where was Ada born?',framing='blocks_v1')
        ids=torch.tensor([t.encode(text,add_special_tokens=False)]);hidden=torch.randn(1,ids.shape[1],8)
        with torch.no_grad():_,states=m(ids,hidden,None,t)
        state=states[0];event=state.antecedents[0]
        self.assertEqual(len(event.words),8);self.assertEqual(len(event.contextual),8);self.assertGreater(event.tail_ordinal,7)
        captured=[];original=m.antecedent_memory._score_scopes
        def capture(*args,**kwargs):
            captured.append((args[2],args[3],args[4]));return original(*args,**kwargs)
        m.antecedent_memory._score_scopes=capture
        with torch.no_grad():m.antecedent_memory.access_contextual(state,torch.stack(state.question_vectors))
        scopes,contexts,rows=captured[-1]
        torch.testing.assert_close(contexts[rows[event.tail_ordinal][0]],state.source_vectors[event.tail_ordinal],rtol=0,atol=0)
        # Mutate only later observed source states. Earlier contextual decisions
        # must still use their own source addresses, not the final event tail.
        changed=copy.copy(state);changed.source_vectors=list(state.source_vectors)
        changed.source_vectors=[v if i<=3 else v+100 for i,v in enumerate(changed.source_vectors)]
        with torch.no_grad():m.antecedent_memory.access_contextual(changed,torch.stack(state.question_vectors))
        later=captured[-1];torch.testing.assert_close(contexts[rows[3]],later[1][later[2][3]],rtol=0,atol=0)
        m.antecedent_memory._score_scopes=original

    def test_evicted_event_context_still_available_without_extra_vectors(self):
        m=self.memory(capacity=3,antecedent_capacity=2)
        t,_,_,state,_=self.observe(format_native_qa('Ada with the red coat near the gate is the subject. He was born in Oakfield.','Where was Ada born?',framing='blocks_v1'),m)
        self.assertEqual(len(state.source_vectors),3)
        self.assertTrue(all(len(e.contextual)<=8 and len(e.words)<=8 for e in state.antecedents))
        with torch.no_grad():scores,details=m.antecedent_memory.access_contextual(state,torch.stack(state.question_vectors))
        self.assertTrue(torch.isfinite(scores).all())


    def test_suffix_at_evicted_boundary_uses_only_retained_causal_states(self):
        m=self.memory(capacity=10,antecedent_capacity=2)
        context=' '.join(['Ada']+['word']*30)+'.'
        t,_,_,state,_=self.observe(format_native_qa(context,'Which owner?',framing='blocks_v1'),m)
        with torch.no_grad():scores,_=m.antecedent_memory.access_contextual(state,torch.stack(state.question_vectors))
        self.assertEqual(len(state.source),10);self.assertEqual(len(state.antecedents[0].contextual),8)
        self.assertTrue(torch.isfinite(scores).all())


class TelemetryTests(unittest.TestCase):
    def test_router_reports_actual_calls_and_preserves_core_stat_names(self):
        from tests.test_native_backbone_representation import NativeBackboneTests
        m,t=NativeBackboneTests().model(all_heads=True);m.requires_grad_(False)
        ids=torch.tensor([t.prepare_generation_hierarchy('Context:\nRed costs 12.\nQuestion:\nWhat does Red cost?').token_ids])
        calls=[];original=m.router.route
        def route(*args,**kwargs):
            result=original(*args,**kwargs);calls.append(result[2]);return result
        m.router.route=route
        with torch.no_grad():output=m(ids,collect_telemetry=True)
        self.assertTrue(calls);self.assertEqual(float(output.route_stats['sharc_router_enabled']),1.)
        self.assertIn('sharc_router_emitter_entropy',output.route_stats)
        self.assertIn('emitter_cell_coverage_loss',output.route_stats)
        m.use_torus_sharc_router=False
        with torch.no_grad():disabled=m(ids,collect_telemetry=True)
        self.assertEqual(float(disabled.route_stats['sharc_router_enabled']),0.)
        m.router.route=original

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
