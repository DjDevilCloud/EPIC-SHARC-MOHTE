"""Neutral migration, word-end evidence, source signatures and causal branch state."""
import unittest,tempfile
import torch
from tests import test_word_span_binding as fixtures
from signature_bank import BoundedIdentityReadout
from data import _build_window_samples_from_text
from train import save_checkpoint,load_bundle_from_checkpoint

class NativeWordPathTests(unittest.TestCase):
    def setup_model(self):
        f=fixtures.NativeRouteWordTests();helper,t,m=f.setup_model();f.enable(m,t,active=True)
        m.cfg.identity_readout_capacity=512;m.bounded_identity_readout.capacity=512
        return helper,t,m

    def enable(self,m,t):
        old=m.bounded_identity_readout;m.cfg.identity_readout_native_word_paths=True
        m.cfg.identity_readout_capacity=512
        head=BoundedIdentityReadout(m.cfg);head.load_state_dict(old.state_dict(),strict=False)
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t)
        return head

    def test_zero_migration_and_owned_prefix_status(self):
        helper,t,m=self.setup_model()
        s=_build_window_samples_from_text(t,'<BOI>The zorvax apple is gold.\nWhat color is the apple?<EOI><BOO>zorvax<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():before=m(s.input_ids[None],**helper.inputs(s)).logits
        head=self.enable(m,t)
        with torch.no_grad():after=m(s.input_ids[None],**helper.inputs(s))
        torch.testing.assert_close(before,after.logits,atol=0,rtol=0)
        state=after.bounded_identity_state[0]
        mem=head._word_binding_memory(state.entries,state.words,t,m.shared_signature_bank,m.construction_embedding,torch.device('cpu'))
        starts,nodes=mem[5];word=next(w for _,_,w in state.words.records if t.decode(list(w),clean_text=False,collapse_structure=False)=='zorvax')
        self.assertGreater(len(word),1)
        node=0
        for j,token in enumerate(word):
            self.assertTrue(nodes[node]['next'])
            node=nodes[node]['children'][token]
            if j<len(word)-1:self.assertTrue(nodes[node]['partial'])
        self.assertTrue(nodes[node]['complete'])

    def test_word_path_is_causal_streams_reloads_and_keeps_branch_state_independent(self):
        helper,t,m=self.setup_model();head=self.enable(m,t);m.cfg.identity_readout_exclude_query_candidates=True
        m.cfg.identity_readout_native_conditional_prefix=True
        with torch.no_grad():
            head.native_source_signature.weight.normal_(0,.01);head.native_word_surface.weight.normal_(0,.01)
            head.native_word_gate.weight.normal_(0,.01);head.native_word_path_scale.fill_(.3)
            head.native_word_start_scale.fill_(.2)
        text='<BOI>The zorvax apple is gold.\nWhat color is the apple?<EOI><BOO>zorvax gold<EOO>'
        s=_build_window_samples_from_text(t,text,seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        alt=_build_window_samples_from_text(t,text.replace('<BOO>zorvax','<BOO>zerqux'),seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        vals=helper.inputs(s);runtime=identity=slots=lattice=None;parts=[]
        with torch.no_grad():
            full=m(s.input_ids[None],**vals).logits
            other=m(alt.input_ids[None],**helper.inputs(alt)).logits
            first=next(i for i,(a,b) in enumerate(zip(s.input_ids,alt.input_ids)) if a!=b)
            torch.testing.assert_close(full[:,:first],other[:,:first],atol=2e-6,rtol=1e-5)
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],**{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,signature_lattice_state=lattice,runtime_signature_state=runtime,bounded_identity_state=identity,position_index=i)
                parts.append(o.logits);runtime=o.runtime_signature_state;identity=o.bounded_identity_state;lattice=o.signature_lattice_state
            torch.testing.assert_close(full,torch.cat(parts,1),atol=2e-5,rtol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                p=save_checkpoint(m,directory,tokenizer=t);loaded,_,cfg=load_bundle_from_checkpoint(p,device='cpu',load_training_state=False)
                self.assertTrue(cfg.identity_readout_native_word_paths)
                self.assertTrue(cfg.identity_readout_native_conditional_prefix)
                torch.testing.assert_close(full,loaded(s.input_ids[None],**vals).logits,atol=0,rtol=0)
        m.train();loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**vals,loss_mask=s.loss_mask[None],collect_telemetry=False);loss.backward()
        for p in (head.native_source_signature.weight,head.native_word_surface.weight,head.native_word_gate.weight,head.native_word_path_scale,head.native_word_start_scale):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())

    def test_verified_prefix_conditions_support_without_mutating_prior(self):
        prior=torch.tensor([.001,.009,.99],requires_grad=True)
        posterior=BoundedIdentityReadout.condition_word_prior(prior,dict(complete=[0],partial=[1]))
        torch.testing.assert_close(posterior,torch.tensor([.1,.9,0.]))
        torch.testing.assert_close(prior.detach(),torch.tensor([.001,.009,.99]))
        posterior[0].backward();self.assertTrue(torch.isfinite(prior.grad).all())
        empty=BoundedIdentityReadout.condition_word_prior(prior.detach(),dict(complete=[],partial=[]))
        self.assertEqual(float(empty.sum()),0.)

    def test_earlier_word_checkpoint_adds_only_neutral_start_scalar(self):
        helper,t,m=self.setup_model();self.enable(m,t)
        s=_build_window_samples_from_text(t,'<BOI>red apple<EOI><BOO>red<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():before=m(s.input_ids[None],**helper.inputs(s)).logits
        with tempfile.TemporaryDirectory() as directory:
            p=save_checkpoint(m,directory,tokenizer=t);payload=torch.load(p,map_location='cpu')
            payload['model_state'].pop('bounded_identity_readout.native_word_start_scale');torch.save(payload,p)
            loaded,_,_=load_bundle_from_checkpoint(p,device='cpu',load_training_state=False)
            self.assertEqual(float(loaded.bounded_identity_readout.native_word_start_scale.detach()),0.)
            with torch.no_grad():torch.testing.assert_close(before,loaded(s.input_ids[None],**helper.inputs(s)).logits,atol=0,rtol=0)
            payload['model_state'].pop('bounded_identity_readout.native_source_signature.weight');torch.save(payload,p)
            with self.assertRaises(ValueError):load_bundle_from_checkpoint(p,device='cpu',load_training_state=False)

if __name__=='__main__':unittest.main()
