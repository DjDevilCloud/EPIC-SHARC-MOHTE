"""Complete words, distinct clausal contexts, frozen routing geometry and streaming state."""
import unittest
import torch
from signature_bank import WordSpanState
from signature_bank import BoundedIdentityReadout
from data import _build_window_samples_from_text
from tests import test_lexical_binding as binding_tests

class WordSpanBindingTests(binding_tests.LexicalBindingTests):
    binding_mode='word_span_v3'

    def test_complete_fragmented_words_and_shared_suffix_contexts(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True);head.capacity=512
        p=t.prepare_generation_hierarchy('The red apple is pink. The green apple is gold.\nWhat color is the zorvax apple?')
        values={k:torch.tensor([getattr(p,k)]) for k in ('signature_ids','signature_family_ids','signature_level_ids','signature_relation_ids','parent_signature_ids','hierarchy_vectors')}
        with torch.no_grad():out=m(torch.tensor([p.token_ids]),**values)
        state=out.bounded_identity_state[0]
        question=[t.decode(list(w),clean_text=False,collapse_structure=False) for w in state.words.question()]
        self.assertEqual(question,['what','color','is','the','zorvax','apple'])
        self.assertGreater(len(state.words.question()[-2]),1)
        contexts={tuple(t.decode(list(w),clean_text=False,collapse_structure=False) for w in e[2]) for e in state.entries}
        self.assertIn(('the','red','apple','is'),contexts)
        self.assertIn(('the','green','apple','is'),contexts)
        before=state.words.question();branch=state.words.clone();branch.pending.append(t.eos_id)
        self.assertEqual(before,state.words.question())
        with torch.no_grad():
            a=head._word_binding_memory(state.entries,state.words,t,m.shared_signature_bank,m.construction_embedding,torch.device('cpu'))[0]
            head.question_word_attention.weight.normal_(0,5)
            b=head._word_binding_memory(state.entries,state.words,t,m.shared_signature_bank,m.construction_embedding,torch.device('cpu'))[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_disabled_positions_do_not_grow_even_on_long_input(self):
        helper,t,m=self.setup_model();m.cfg.use_absolute_position_embeddings=False;m.cfg.max_seq_len=0
        before=m.position_embedding.embedding.weight.clone()
        m._ensure_position_embedding_capacity(10000)
        self.assertEqual(m._position_context(torch.tensor([[9999]])),0.)
        torch.testing.assert_close(before,m.position_embedding.embedding.weight,atol=0,rtol=0)

    def test_bounded_word_state(self):
        helper,t,m=self.setup_model();state=WordSpanState(8)
        for token in t.encode('unfamiliarword red apple green fruit',add_special_tokens=False):state.observe(token,t.construction_units[token])
        self.assertLessEqual(sum(map(len,state.clause)),8)
        self.assertLessEqual(len(state.pending),8)

    def test_incremental_inference_composes_input_memory_once(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True)
        s=_build_window_samples_from_text(t,'<BOI>red apple green fruit<EOI><BOO>red apple<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        count=[0];original=head._word_binding_memory
        def counted(*args):count[0]+=1;return original(*args)
        head._word_binding_memory=counted
        values=helper.inputs(s);runtime=identity=slots=lattice=None
        with torch.no_grad():
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],**{k:v[:,i:i+1] for k,v in values.items()},slot_state=slots,signature_lattice_state=lattice,runtime_signature_state=runtime,bounded_identity_state=identity,position_index=i)
                runtime=o.runtime_signature_state;identity=o.bounded_identity_state;lattice=o.signature_lattice_state
        self.assertEqual(count[0],1)

class ObservedWordSpanTests(WordSpanBindingTests):
    binding_mode='word_span_v4'

    def test_zero_right_context_and_continuation_preserve_v3_logits(self):
        helper,t,m=self.setup_model()
        self.binding_mode='word_span_v3';old=self.enable(m,t,active=True);self.binding_mode='word_span_v4'
        s=_build_window_samples_from_text(t,'<BOI>Freight means property in transit. red apple<EOI><BOO>red apple<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():before=m(s.input_ids[None],**helper.inputs(s)).logits
        m.cfg.identity_readout_binding='word_span_v4';head=BoundedIdentityReadout(m.cfg);head.load_state_dict(old.state_dict(),strict=False)
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t)
        with torch.no_grad():after=m(s.input_ids[None],**helper.inputs(s)).logits
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        m.train();loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**helper.inputs(s),loss_mask=s.loss_mask[None],collect_telemetry=False);loss.backward()
        for parameter in (head.right_context_projection.weight,head.continuation_scale):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()),0.)

class NativeRouteWordTests(WordSpanBindingTests):
    binding_mode='word_span_v5'

    def test_nonseparable_calibration_cannot_replace_accepted_router(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True);head.applicability.requires_grad_(False)
        before={k:v.clone() for k,v in head.applicability.state_dict().items()}
        features=torch.zeros(2,m.cfg.d_model)
        result=head.calibrate_task_route(features,features,steps=20,lr=.03)
        self.assertFalse(result['accepted'])
        for k,v in before.items():torch.testing.assert_close(v,head.applicability.state_dict()[k],atol=0,rtol=0)
        self.assertFalse(head.applicability.weight.requires_grad)

    def test_separable_route_calibration_commits_without_unfreezing_router(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True);head.applicability.requires_grad_(False)
        features=torch.ones(2,m.cfg.d_model)
        result=head.calibrate_task_route(-features,features,steps=80,lr=.1)
        self.assertTrue(result['accepted'],result)
        self.assertFalse(head.applicability.weight.requires_grad)
        self.assertLessEqual(result['negative_max'],1-head.binding_confidence)
        self.assertGreaterEqual(result['positive_min'],head.binding_confidence)

    def test_source_partition_excludes_question_words_and_preserves_empty_source(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True);head.capacity=512
        m.cfg.identity_readout_exclude_query_candidates=True
        p=t.prepare_generation_hierarchy('The red apple is green.\nTell me the color of the apple.')
        values={k:torch.tensor([getattr(p,k)]) for k in ('signature_ids','signature_family_ids','signature_level_ids','signature_relation_ids','parent_signature_ids','hierarchy_vectors')}
        with torch.no_grad():state=m(torch.tensor([p.token_ids]),**values).bounded_identity_state[0]
        mask=state.words.source_candidate_mask(state.entries)
        selected=[e[0] for e,keep in zip(state.entries,mask) if keep]
        excluded=[e[0] for e,keep in zip(state.entries,mask) if not keep]
        self.assertEqual(t.decode(selected,clean_text=False,collapse_structure=False),'theredappleisgreen')
        self.assertEqual(t.decode(excluded,clean_text=False,collapse_structure=False),'tellmethecoloroftheapple')
        branch=state.words.clone()
        self.assertEqual(mask,branch.source_candidate_mask(state.entries))
        lone=WordSpanState(512)
        for token in t.encode('red apple',add_special_tokens=False):lone.observe(token,t.construction_units[token])
        lone.boundary(clause=True)
        self.assertEqual(lone.source_candidate_mask([(r[2][0],(),(),r[0],()) for r in lone.records]),[True]*len(lone.records))

    def test_partitioned_native_full_and_incremental_logits_agree(self):
        helper,t,m=self.setup_model();self.enable(m,t,active=True);m.cfg.identity_readout_exclude_query_candidates=True
        s=_build_window_samples_from_text(t,'<BOI>The red apple is green.\nWhat color is the apple?<EOI><BOO>green<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        values=helper.inputs(s);runtime=identity=slots=lattice=None;pieces=[]
        with torch.no_grad():
            full=m(s.input_ids[None],**values).logits
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],**{k:v[:,i:i+1] for k,v in values.items()},slot_state=slots,signature_lattice_state=lattice,runtime_signature_state=runtime,bounded_identity_state=identity,position_index=i)
                pieces.append(o.logits);runtime=o.runtime_signature_state;identity=o.bounded_identity_state;lattice=o.signature_lattice_state
        torch.testing.assert_close(full,torch.cat(pieces,dim=1),atol=2e-5,rtol=1e-5)

    def test_partition_is_neutral_for_confident_retained_route_and_reloads(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True)
        with torch.no_grad():
            head.applicability.weight.zero_();head.applicability.bias.fill_(-20)
            head.native_applicability.classifier[-1].weight.zero_();head.native_applicability.classifier[-1].bias.fill_(-20)
        s=_build_window_samples_from_text(t,'<BOI>red apple.\nWhat color is the fruit?<EOI><BOO>red<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():before=m(s.input_ids[None],**helper.inputs(s)).logits
        m.cfg.identity_readout_exclude_query_candidates=True
        with torch.no_grad():after=m(s.input_ids[None],**helper.inputs(s)).logits
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        with binding_tests.tempfile.TemporaryDirectory() as directory:
            path=binding_tests.save_checkpoint(m,directory,tokenizer=t)
            restored,_,cfg=binding_tests.load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
            self.assertTrue(cfg.identity_readout_exclude_query_candidates)
            with torch.no_grad():reloaded=restored(s.input_ids[None],**helper.inputs(s)).logits
            torch.testing.assert_close(after,reloaded,atol=0,rtol=0)

    def test_native_route_is_neutral_and_retained_matching_parameters_freeze(self):
        helper,t,m=self.setup_model();self.binding_mode='word_span_v3';old=self.enable(m,t,active=True);self.binding_mode='word_span_v5'
        s=_build_window_samples_from_text(t,'<BOI>red apple green fruit<EOI><BOO>red apple<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():before=m(s.input_ids[None],**helper.inputs(s)).logits
        m.cfg.identity_readout_binding=self.binding_mode;head=BoundedIdentityReadout(m.cfg);head.load_state_dict(old.state_dict(),strict=False)
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t);m.freeze_binding_backbone()
        with torch.no_grad():after=m(s.input_ids[None],**helper.inputs(s)).logits
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        self.assertFalse(head.span_projection.weight.requires_grad)
        self.assertFalse(head.context_word_attention.weight.requires_grad)
        self.assertFalse(head.context_query.weight.requires_grad)
        self.assertTrue(head.native_query.weight.requires_grad)
        self.assertTrue(head.native_surface_projection.weight.requires_grad)
        self.assertFalse(head.native_applicability.classifier[-1].weight.requires_grad)

if __name__=='__main__':unittest.main()
