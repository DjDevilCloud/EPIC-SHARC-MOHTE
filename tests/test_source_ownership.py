"""Neutral ownership migration, occurrence transitions and request-local state."""
import unittest,tempfile
import torch
from tests import test_native_span_boundaries as fixtures
from signature_bank import BoundedIdentityReadout,IdentityReadoutState,WordSpanState
from data import _build_window_samples_from_text
from train import save_checkpoint,load_bundle_from_checkpoint

class SourceOwnershipTests(unittest.TestCase):
    def fixture(self):
        f=fixtures.NativeSpanBoundaryTests();helper,t,m=f.fixture();f.enable(m,t,categorical=True)
        m.cfg.identity_readout_exclude_query_candidates=True
        m.bounded_identity_readout.native_span_ready.fill_(True)
        return helper,t,m

    def enable(self,m,t):
        m.cfg.identity_readout_source_ownership=True;head=BoundedIdentityReadout(m.cfg)
        missing=head.load_state_dict(m.bounded_identity_readout.state_dict(),strict=False)
        self.assertEqual(set(missing.missing_keys),{'native_span_start.weight','native_source_cursor_scale'})
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t);return head

    def test_cursor_preserves_occurrence_and_verifies_separators(self):
        words=WordSpanState();words.records=[(0,0,(21,)),(1,0,(22,)),(2,1,(21,)),(3,2,(23,))]
        words.following={0:[12,14],1:[56],2:[12],3:[]}
        entries=[(token,(),(),serial,()) for serial,clause,(token,) in words.records]
        prior=torch.tensor([1.,0.,0.,0.]);state=IdentityReadoutState(entries,words=words,source_cursor=prior)
        ids=torch.tensor([21,22,21,23])
        result=BoundedIdentityReadout.source_cursor_targets(state,[12,14],ids)
        torch.testing.assert_close(result,torch.tensor([0.,1.,0.,0.]))
        torch.testing.assert_close(prior,torch.tensor([1.,0.,0.,0.]))
        self.assertEqual(float(BoundedIdentityReadout.source_cursor_targets(state,[12],ids).sum()),0.)
        state.source_cursor=torch.tensor([0.,0.,1.,0.])
        # The next lexical record belongs to the query, so it is never proposed.
        self.assertEqual(float(BoundedIdentityReadout.source_cursor_targets(state,[12],ids).sum()),0.)

    def test_word_posterior_preserves_probability_without_fragment_frequency_bias(self):
        entries=[(21,(),(),0,()),(22,(),(),0,()),(23,(),(),0,()),(24,(),(),1,())]
        prior=torch.tensor([.8,0.,0.,.2])
        root=BoundedIdentityReadout.word_path_posterior(prior,{'next':[0,3]},entries,[0,3])
        torch.testing.assert_close(root,prior)
        child=BoundedIdentityReadout.word_path_posterior(prior,{'next':[1]},entries,[0,3])
        torch.testing.assert_close(child,torch.tensor([0.,1.,0.,0.]))
        self.assertEqual(float(BoundedIdentityReadout.word_path_posterior(prior,{'next':[]},entries,[0,3]).sum()),0.)

    def test_boundary_evidence_keeps_selected_repeated_occurrence(self):
        _,t,m=self.fixture()
        token=next(i for i,u in enumerate(t.construction_units) if u.is_lexical)
        space=next(i for i,u in enumerate(t.construction_units) if u.text=='<SPACE>')
        eol=next(i for i,u in enumerate(t.construction_units) if u.text=='<EOL>')
        words=WordSpanState();words.records=[(0,0,(token,)),(1,1,(token,)),(2,2,(token,))]
        words.following={0:[space],1:[eol]}
        entries=[(token,(),(),i,()) for i in range(3)]
        state=IdentityReadoutState(entries,words=words,output_units=[token],output_separators=[],source_mask=torch.tensor([True,True,False]),source_cursor=torch.tensor([1.,0.,0.]))
        vector=BoundedIdentityReadout.source_boundary_vector(m.shared_signature_bank,state,t,torch.device('cpu'))
        state.entries=entries[:1];state.source_cursor=torch.tensor([1.]);state.source_mask=torch.tensor([True])
        expected=BoundedIdentityReadout.source_boundary_vector(m.shared_signature_bank,state,t,torch.device('cpu'))
        torch.testing.assert_close(vector,expected,atol=0,rtol=0)
        posterior=torch.tensor([.6,.4,0.],requires_grad=True)
        state.entries=entries;state.source_mask=torch.tensor([True,True,False]);state.source_cursor=posterior
        mixed=BoundedIdentityReadout.source_boundary_vector(m.shared_signature_bank,state,t,torch.device('cpu'))
        mixed.sum().backward()
        self.assertTrue(torch.isfinite(posterior.grad).all())
        self.assertGreater(float(posterior.grad.abs().sum()),0.)

    def test_zero_migration_active_streaming_reload_and_gradients(self):
        self.active_path('word_neighbors_v1')

    def test_parent_span_features_stream_reload_and_keep_zero_migration(self):
        self.active_path('span_roles_v2')

    def test_context_roles_stream_reload_and_keep_zero_migration(self):
        self.active_path('context_roles_v3')

    def test_posterior_path_streams_reloads_and_has_finite_gradients(self):
        self.active_path('context_roles_v3',posterior=True)

    def test_context_roles_ignore_candidate_spelling_and_fragment_count(self):
        helper,t,m=self.fixture();m.cfg.identity_readout_source_start_features='context_roles_v3';head=self.enable(m,t)
        features=[]
        for answer in ('zorvax','zorvaxquethorin'):
            sample=_build_window_samples_from_text(t,f'<BOI>The apple is {answer}.\nWhat color is the apple?<EOI><BOO>{answer}<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
            captured=[];hook=head.native_span_start.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
            with torch.no_grad():out=m(sample.input_ids[None],**helper.inputs(sample))
            hook.remove();state=out.bounded_identity_state[0]
            source=[entry[0] for entry in state.entries];target=[i for i in t.encode(answer) if t.construction_units[i].is_lexical]
            start=next(i for i in range(len(source)) if source[i:i+len(target)]==target)
            features.append(captured[0][start])
        torch.testing.assert_close(features[0],features[1],atol=0,rtol=0)

    def active_path(self,feature_version,posterior=False):
        helper,t,m=self.fixture()
        s=_build_window_samples_from_text(t,'<BOI>The zorvax apple is gold.\nWhat color is the apple?<EOI><BOO>zorvax apple<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        vals=helper.inputs(s)
        if posterior:m.cfg.identity_readout_word_path_readout='posterior_v2'
        with torch.no_grad():before=m(s.input_ids[None],**vals).logits
        m.cfg.identity_readout_source_start_features=feature_version
        head=self.enable(m,t)
        with torch.no_grad():torch.testing.assert_close(before,m(s.input_ids[None],**vals).logits,atol=0,rtol=0)
        with torch.no_grad():head.native_span_start.weight.normal_(0,.2);head.native_source_cursor_scale.fill_(.8)
        runtime=identity=slots=lattice=None;parts=[]
        with torch.no_grad():
            full=m(s.input_ids[None],**vals).logits
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],**{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,signature_lattice_state=lattice,runtime_signature_state=runtime,bounded_identity_state=identity,position_index=i)
                parts.append(o.logits);runtime=o.runtime_signature_state;identity=o.bounded_identity_state;lattice=o.signature_lattice_state
            torch.testing.assert_close(full,torch.cat(parts,1),atol=2e-5,rtol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                path=save_checkpoint(m,directory,tokenizer=t);loaded,_,cfg=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
                self.assertTrue(cfg.identity_readout_source_ownership)
                self.assertEqual(cfg.identity_readout_source_start_features,feature_version)
                self.assertEqual(cfg.identity_readout_word_path_readout,'posterior_v2' if posterior else 'bias_v1')
                torch.testing.assert_close(full,loaded(s.input_ids[None],**vals).logits,atol=0,rtol=0)
        loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**vals,loss_mask=s.loss_mask[None],collect_telemetry=False);loss.backward()
        for p in (head.native_span_start.weight,head.native_source_cursor_scale):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())

if __name__=='__main__':unittest.main()
