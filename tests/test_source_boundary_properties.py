"""Causal property separation, neutral migration, streaming and reload."""
import unittest,tempfile
import torch
from tests import test_source_ownership as ownership_fixtures
from signature_bank import BoundedIdentityReadout,WordSpanState,IdentityReadoutState
from data import _build_window_samples_from_text
from train import load_bundle_from_checkpoint,save_checkpoint

class SourceBoundaryPropertyTests(unittest.TestCase):
    def fixture(self):
        f=ownership_fixtures.SourceOwnershipTests();helper,t,m=f.fixture();f.enable(m,t)
        m.cfg.identity_readout_source_boundaries=True;m.cfg.identity_readout_source_boundary_readout='categorical_v1'
        head=BoundedIdentityReadout(m.cfg);missing=head.load_state_dict(m.bounded_identity_readout.state_dict(),strict=False)
        self.assertEqual(missing.missing_keys,['native_source_boundary.weight'])
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t);return helper,t,m

    def test_properties_distinguish_complete_numeric_and_alphabetic_words(self):
        _,t,m=self.fixture();alpha=next(i for i,u in enumerate(t.construction_units) if u.is_lexical and u.render.isalpha());digit=next(i for i,u in enumerate(t.construction_units) if u.kind=='digit');space=next(i for i,u in enumerate(t.construction_units) if u.text=='<SPACE>')
        words=WordSpanState();words.records=[(0,0,(alpha,)),(1,0,(digit,)),(2,1,(alpha,))];words.following={0:[space],1:[space]}
        entries=[(token,(),(),serial,()) for serial,clause,(token,) in words.records];vectors=[];base=[]
        for index in (0,1):
            cursor=torch.zeros(3);cursor[index]=1.;state=IdentityReadoutState(entries,words=words,output_units=[entries[index][0]],output_separators=[],source_mask=torch.tensor([True,True,False]),source_cursor=cursor)
            base.append(BoundedIdentityReadout.source_boundary_vector(m.shared_signature_bank,state,t,torch.device('cpu')))
            vectors.append(BoundedIdentityReadout.source_boundary_property_vector(m.shared_signature_bank,state,t,torch.device('cpu')))
        torch.testing.assert_close(base[0],base[1],atol=0,rtol=0)
        self.assertFalse(torch.equal(vectors[0],vectors[1]))
        state.source_cursor.zero_()
        self.assertEqual(float(BoundedIdentityReadout.source_boundary_property_vector(m.shared_signature_bank,state,t,torch.device('cpu')).abs().sum()),0.)

    def test_zero_migration_active_streaming_reload_and_gradients(self):
        helper,t,m=self.fixture();sample=_build_window_samples_from_text(t,'<BOI>The apple has 100 seeds.\nHow many seeds does the apple have?<EOI><BOO>100<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0];vals=helper.inputs(sample)
        with torch.no_grad():before=m(sample.input_ids[None],**vals).logits
        m.cfg.identity_readout_source_boundary_properties=True;head=BoundedIdentityReadout(m.cfg);missing=head.load_state_dict(m.bounded_identity_readout.state_dict(),strict=False)
        self.assertEqual(set(missing.missing_keys),{'native_source_boundary_properties.weight','native_source_boundary_property_gate.weight'});m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t)
        with torch.no_grad():torch.testing.assert_close(before,m(sample.input_ids[None],**vals).logits,atol=0,rtol=0)
        with torch.no_grad():head.native_source_boundary_properties.weight.normal_(0,3.);head.native_source_boundary_property_gate.weight.normal_(0,.2)
        runtime=identity=slots=lattice=None;parts=[]
        with torch.no_grad():
            full=m(sample.input_ids[None],**vals).logits
            for i in range(sample.input_ids.numel()):
                _,slots,out=m.forward_incremental(sample.input_ids[i:i+1][None],**{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,signature_lattice_state=lattice,runtime_signature_state=runtime,bounded_identity_state=identity,position_index=i)
                parts.append(out.logits);runtime=out.runtime_signature_state;identity=out.bounded_identity_state;lattice=out.signature_lattice_state
            torch.testing.assert_close(full,torch.cat(parts,1),atol=2e-5,rtol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                path=save_checkpoint(m,directory,tokenizer=t);loaded,_,cfg=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False);self.assertTrue(cfg.identity_readout_source_boundary_properties);torch.testing.assert_close(full,loaded(sample.input_ids[None],**vals).logits,atol=0,rtol=0)
        loss,_=m.compute_loss(sample.input_ids[None],sample.labels[None],**vals,loss_mask=sample.loss_mask[None],collect_telemetry=False);loss.backward()
        for parameter in (head.native_source_boundary_properties.weight,head.native_source_boundary_property_gate.weight):
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())

if __name__=='__main__':unittest.main()
