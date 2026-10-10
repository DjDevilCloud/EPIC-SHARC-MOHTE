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

    def test_zero_migration_active_streaming_reload_and_gradients(self):
        self.active_path('word_neighbors_v1')

    def test_parent_span_features_stream_reload_and_keep_zero_migration(self):
        self.active_path('span_roles_v2')

    def active_path(self,feature_version):
        helper,t,m=self.fixture()
        s=_build_window_samples_from_text(t,'<BOI>The zorvax apple is gold.\nWhat color is the apple?<EOI><BOO>zorvax apple<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        vals=helper.inputs(s)
        with torch.no_grad():before=m(s.input_ids[None],**vals).logits
        m.cfg.identity_readout_source_start_features=feature_version;head=self.enable(m,t)
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
                torch.testing.assert_close(full,loaded(s.input_ids[None],**vals).logits,atol=0,rtol=0)
        loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**vals,loss_mask=s.loss_mask[None],collect_telemetry=False);loss.backward()
        for p in (head.native_span_start.weight,head.native_source_cursor_scale):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())

if __name__=='__main__':unittest.main()
