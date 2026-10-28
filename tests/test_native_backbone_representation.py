import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from data import PrismalTokenizer
from model import PrismalWaveModel
from native_question_memory import NativeQuestionMemory
from native_qa_transfer_study import generation_fields,full_incremental_check
from train import save_checkpoint,load_bundle_from_checkpoint


class NativeBackboneTests(unittest.TestCase):
    row=dict(context='Red costs 12. Blue costs 35.',question='What does Red cost?',answer='12')

    def memory(self,capacity=128,**extra):
        options=dict(span_readout=True,stable_span_gate=True,conditional_endpoints=True,
            marker_endpoints=True,conditional_copy=True,representation='backbone_context_v1')
        options.update(extra)
        return NativeQuestionMemory(8,capacity=capacity,**options)

    def observed(self,text=None,memory=None,hidden=None):
        t=PrismalTokenizer();m=memory or self.memory()
        text=text or '<BOI>Context:\nRed costs 12. Blue costs 35.\nQuestion:\nWhat does Red cost?<EOI><BOO>'
        ids=torch.tensor([t.encode(text,add_special_tokens=False)])
        hidden=hidden if hidden is not None else torch.randn(1,ids.shape[1],8)
        output,states,records=m(ids,hidden,None,t,return_readout=True,diagnostics=True)
        return t,m,ids,hidden,output,states[0],records

    def model(self,seed=47,all_heads=False):
        from tests.test_runtime_signatures import RuntimeSignatureTests
        torch.manual_seed(seed);_,t,base=RuntimeSignatureTests().setup_model()
        cfg=copy.deepcopy(base.cfg)
        cfg.use_native_question_memory=True;cfg.native_question_memory_span_readout=True
        cfg.qa_readout_route='native';cfg.qa_generation_profile='native_qa_eval_v1'
        cfg.native_question_memory_representation='backbone_context_v1'
        cfg.native_question_memory_stable_span_gate=True;cfg.native_question_memory_conditional_endpoints=True
        cfg.native_question_memory_conditional_copy=True;cfg.native_question_memory_marker_endpoints=True
        cfg.freeze_registry_observations=True
        if all_heads:
            cfg.native_question_memory_relation_alignment=True;cfg.native_question_memory_source_roles=True
            cfg.native_question_memory_antecedent_memory=True;cfg.native_question_memory_owner_evidence=True
            cfg.native_question_memory_lexical_cues=True;cfg.native_question_memory_future_context=8
            cfg.native_question_memory_anchor_words=4;cfg.native_question_memory_document_context=True
        model=PrismalWaveModel(type(cfg).from_dict(cfg.to_dict()));model._prismal_tokenizer=t
        model.prepare_capacity_for_tokenizer(t);model.eval()
        return model,t

    def plan(self,model,t,row=None):
        row=row or self.row;ids,tracks=generation_fields(t,row)
        output=model(ids,**tracks,collect_telemetry=False,return_native_readouts=True)
        self.assertTrue(output.native_qa_readouts)
        return output,output.native_qa_readouts[-1][4]

    def test_no_independent_encoder_and_source_scores_depend_on_backbone_states(self):
        t,m,ids,h,_,s,_=self.observed()
        self.assertFalse(any(isinstance(module,nn.GRU) for module in m.modules()))
        m.eval()
        with torch.no_grad():
            _,a=m(ids,torch.zeros_like(h),None,t)
            _,b=m(ids,h,None,t)
        self.assertFalse(torch.allclose(a[0].cached_packet[2],b[0].cached_packet[2]))

    def test_source_loss_reaches_torus_router_lattice_and_shared_hierarchy(self):
        for seed in (47,83):
            with self.subTest(seed=seed):
                model,t=self.model(seed,all_heads=True);output,plan=self.plan(model,t)
                indices=[i for i,w in enumerate(plan['source_words']) if t.decode(list(w),clean_text=False,collapse_structure=False)=='12']
                self.assertTrue(indices)
                loss=-plan['start_log_probabilities'][indices].logsumexp(0)
                loss.backward()
                for name in ('torus_core','router','signature_lattice_attention','shared_signature_bank'):
                    module=getattr(model,name)
                    norm=sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)
                    self.assertGreater(norm,0.,name)
                self.assertFalse(any(isinstance(module,nn.GRU) for module in model.native_question_memory.modules()))

    def test_native_memory_incremental_scores_payloads_and_contexts_match(self):
        m=self.memory(antecedent_memory=True,owner_evidence=True,source_roles=True,
                      relation_alignment=True,future_context=3,lexical_cues=True)
        t,_,ids,h,_,_,_=self.observed(memory=m);m.eval()
        with torch.no_grad():
            _,full=m(ids,h,None,t);states=None
            for i in range(ids.shape[1]):_,states=m(ids[:,i:i+1],h[:,i:i+1],None,t,states)
        a,b=full[0],states[0]
        self.assertEqual(a.source,b.source);self.assertEqual(a.source_ranges,b.source_ranges)
        self.assertEqual(a.cached_packet[3]['candidates'],b.cached_packet[3]['candidates'])
        torch.testing.assert_close(a.cached_packet[2],b.cached_packet[2],rtol=0,atol=0)
        for x,y in zip(a.source_summaries,b.source_summaries):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_completed_line_retains_following_predicate_and_future_answer_is_excluded(self):
        text='<BOI>Context:\nA. Reed owns red tokens.\nQuestion:\nWho owns red tokens?<EOI><BOO>'
        t,m,ids,h,_,s,_=self.observed(text=text)
        self.assertTrue(torch.equal(s.source_summaries[0],s.source_summaries[-1]))
        m.eval()
        with torch.no_grad():
            _,initial=m(ids,h,None,t)
            saved=initial[0];surface=list(saved.surface);scores=saved.cached_packet[2].clone()
            token=t.encode('unrelated',add_special_tokens=False)[0]
            _,following=m(torch.tensor([[token]]),torch.randn(1,1,8)*100,None,t,[saved])
        self.assertEqual(following[0].surface,surface)
        torch.testing.assert_close(following[0].cached_packet[2],scores,rtol=0,atol=0)
        self.assertEqual(saved.emitted,())

    def test_eviction_keeps_observations_aligned_and_inference_storage_bounded(self):
        m=self.memory(capacity=3,antecedent_memory=True,antecedent_capacity=2)
        t=PrismalTokenizer();text='<BOI>Context:\nOne two three four five six.\nQuestion:\nWhich one?<EOI><BOO>'
        ids=torch.tensor([t.encode(text,add_special_tokens=False)]);h=torch.randn(1,ids.shape[1],8)
        with torch.no_grad():_,states=m(ids,h,None,t)
        s=states[0]
        for field in ('source','source_vectors','source_summaries','source_segments','source_lines','source_ordinals','source_ranges'):
            self.assertEqual(len(getattr(s,field)),3,field)
        for v in s.source_vectors+s.source_summaries+s.question_vectors+[s.request_vector]:
            self.assertEqual(v.untyped_storage().nbytes(),v.numel()*v.element_size())
        self.assertLessEqual(len(s.antecedents),2)

    def test_repeated_identical_blocks_do_not_reuse_another_contexts_plan(self):
        text='<BOI>Context:\nRed costs 12.\nQuestion:\nWhat does Red cost?<EOI><BOO>'
        t,m,ids,h,_,_,_=self.observed(text=text)
        _,_,records=m(torch.cat((ids,ids),1),torch.cat((h,h*5),1),None,t,return_readout=True)
        plans=list({id(record[4]):record[4] for record in records}.values())
        self.assertEqual(len(plans),2)
        self.assertFalse(torch.allclose(plans[0]['start_log_probabilities'],plans[1]['start_log_probabilities']))

    def test_owner_prefix_features_remain_causal(self):
        m=self.memory(antecedent_memory=True,owner_evidence=True)
        t,m,ids,h,_,state,_=self.observed(memory=m)
        query=torch.stack(state.question_vectors)
        before=m.antecedent_memory.access_contextual(state,query)[1]
        event=state.antecedents[-1]
        # A later event cannot enter an earlier source occurrence's scope list.
        first=before['ownership'][0]
        self.assertEqual(len(first),1)
        self.assertGreater(len(before['ownership'][-1]),len(first))

    def test_model_reload_and_prefix_incremental_execution(self):
        model,t=self.model(all_heads=True);model.requires_grad_(False)
        self.assertLess(full_incremental_check(model,t,self.row),3e-4)
        _,expected=self.plan(model,t)
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(model,folder,tokenizer=t)
            loaded,rt,cfg=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu',load_training_state=False)
            loaded.requires_grad_(False)
            self.assertEqual(cfg.native_question_memory_representation,'backbone_context_v1')
            _,actual=self.plan(loaded,rt)
            self.assertEqual(expected['candidates'],actual['candidates'])
            torch.testing.assert_close(expected['log_probabilities'],actual['log_probabilities'],rtol=0,atol=0)

    def test_incompatible_parallel_semantic_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'replaces'):
            self.memory(semantic_alignment=True)

    def annotated_batch(self,model,t,answer='12'):
        from data import TextWindowDataset,build_collate_fn
        text=f"<BOI>Context:\n{self.row['context']}\nQuestion:\n{self.row['question']}<EOI><BOO>{answer}<EOO>"
        ds=TextWindowDataset([text],t,seq_len=512,max_samples=0,preserve_grounded_qa_context=True)
        self.assertEqual(len(ds),1)
        tensors=build_collate_fn(t.pad_id,t.signature_pad_id)([ds[0]])
        position=int((tensors[0][0]==t.special_tokens['<BOO>']).nonzero()[0])
        # Reviewed fixture order: Red / costs / 12 / Blue / costs / 35.
        return dict(tensors=tensors,native_source_targets=[dict(row=0,position=position,starts=[2])])

    def compute(self,model,batch,targets=None):
        fields=('input_ids','labels','signature_ids','signature_level_ids','signature_relation_ids',
                'parent_signature_ids','signature_family_ids','hierarchy_vectors','loss_mask')
        return model.compute_loss(**dict(zip(fields,batch['tensors'])),
            native_source_targets=batch['native_source_targets'] if targets is None else targets,collect_telemetry=False)

    def test_training_annotations_do_not_enter_inference_and_invalid_targets_fail(self):
        model,t=self.model();batch=self.annotated_batch(model,t)
        other=copy.deepcopy(batch['native_source_targets']);other[0]['starts']=[5]
        with torch.no_grad():
            _,a=self.compute(model,batch);_,b=self.compute(model,batch,other)
        torch.testing.assert_close(a.logits,b.logits,rtol=0,atol=0)
        for starts in ([2,2],[999],[],[True]):
            target=copy.deepcopy(batch['native_source_targets']);target[0]['starts']=starts
            with self.subTest(starts=starts),self.assertRaises(ValueError):self.compute(model,batch,target)
        with self.assertRaisesRegex(ValueError,'One empty-prefix'):
            self.compute(model,batch,batch['native_source_targets']*2)

    def test_ordinary_trainer_uses_live_native_supervision_and_updates_core(self):
        from train import train_model
        model,t=self.model();model.cfg.use_fst=False
        batch=self.annotated_batch(model,t)
        initial={name:p.detach().clone() for name,p in model.torus_core.named_parameters()}
        train_model(model,[batch],torch.device('cpu'),cfg=model.cfg,optimizer_name='adamw',
                    steps=1,lr=.001,progress=False,use_amp=False)
        self.assertTrue(any(not torch.equal(initial[name],p) for name,p in model.torus_core.named_parameters()))
        contract=model._prismal_training_state['native_source_supervision']
        self.assertEqual(contract['examples_presented'],1)
        self.assertEqual(contract['weights_presented'],[1.])

    def test_future_answer_substitution_preserves_earlier_model_predictions(self):
        model,t=self.model(all_heads=True);left=self.annotated_batch(model,t,'12');right=self.annotated_batch(model,t,'different')
        a,b=left['tensors'][0][0],right['tensors'][0][0]
        n=next(i for i in range(min(len(a),len(b))) if a[i]!=b[i])
        with torch.no_grad():
            _,x=self.compute(model,left);_,y=self.compute(model,right)
        torch.testing.assert_close(x.logits[:,:n],y.logits[:,:n],rtol=2e-4,atol=3e-5)
        torch.testing.assert_close(x.native_qa_readouts[0][4]['start_log_probabilities'],
                                   y.native_qa_readouts[0][4]['start_log_probabilities'],rtol=2e-4,atol=3e-5)

    def test_normal_python_and_model_generation_use_same_saved_profile(self):
        from train import generate_text
        from native_qa_transfer_study import prompt
        model,t=self.model();model.requires_grad_(False)
        result=generate_text(model,t,prompt(self.row),torch.device('cpu'),return_generation_record=True)
        ids,tracks=generation_fields(t,self.row)
        direct=model.generate(ids,**tracks,suppressed_token_ids=t.generation_suppressed_token_ids(),
            token_signature_lookup=t.signature_lookup_by_token_id(),token_family_lookup=t.signature_family_lookup_by_token_id(),
            token_level_lookup=t.signature_level_lookup_by_token_id(),token_relation_lookup=t.signature_relation_lookup_by_token_id())
        self.assertTrue(result['canonical'])
        self.assertEqual(result['tokens'],direct[0,len(ids[0]):].tolist())

    def test_context_cache_rejects_a_different_model_or_parameter_revision(self):
        from train import _build_optimizer
        for change in ('optimizer','reload','other_model'):
            model,t=self.model();ids,tracks=generation_fields(t,self.row)
            optimizer=_build_optimizer(model,optimizer_name='adamw',base_lr=.001,cfg=model.cfg)
            with torch.no_grad():output=model(ids,**tracks,collect_telemetry=False)
            if change=='optimizer':
                p=next(model.torus_core.parameters());p.grad=torch.ones_like(p);optimizer.step()
            elif change=='reload':model.load_state_dict(model.state_dict())
            else:
                replacement,_=self.model();replacement.load_state_dict(model.state_dict());model=replacement
            with self.subTest(change=change),torch.no_grad(),self.assertRaisesRegex(ValueError,'context.*stale|context.*producer'):
                model.forward_incremental(torch.tensor([[t.special_tokens['<LINE>']]]),position_index=ids.shape[1],
                    slot_state=output.slot_state,signature_lattice_state=output.signature_lattice_state,
                    runtime_signature_state=output.runtime_signature_state)

    def test_conditional_payload_mechanics_with_explicit_high_confidence_fixture(self):
        from source_answer_span import CompleteAnswerSpanPredictor
        for value in ('A. R. Doe','$75.50','72%','red and blue'):
            text=f'<BOI>Context:\nThe apple value is {value}.\nQuestion:\nWhat is the apple value?<EOI><BOO>'
            t,m,ids,h,_,state,records=self.observed(text=text)
            payload=CompleteAnswerSpanPredictor.payload(t,value)
            matches=m.span_readout.candidates(state,t,list(range(len(state.source))))
            self.assertTrue(any(c['text']==value and c['tokens']==payload for c in matches))
            # Mechanical conditional oracle only: no claim that an untrained
            # model has learned copy confidence or selected the answer proposal.
            nn.init.zeros_(m.span_readout.copy_gate.weight);nn.init.constant_(m.span_readout.copy_gate.bias,100.)
            record=records[-1]
            for token in payload:
                _,_,q,current,plan,emitted=record
                injected=dict(plan,candidates=[dict(tokens=payload)],probabilities=q.new_ones(1),log_probabilities=q.new_zeros(1))
                native=torch.zeros(1,1,t.vocab_size);native[0,0,(token+1)%t.vocab_size]=1000.
                actual=m.span_readout.mix(native,[(0,0,q,current,injected,emitted)])
                self.assertEqual(int(actual.argmax(-1)),token)
                _,states,following=m(torch.tensor([[token]]),torch.zeros(1,1,8),None,t,[state],return_readout=True)
                state=states[0]
                if following:record=following[-1]
            self.assertFalse(state.emitting)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
