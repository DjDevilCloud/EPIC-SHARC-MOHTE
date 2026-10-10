"""Neutral migration, canonical span features, causal state, and persisted adapters."""
import unittest,tempfile
import torch
from tests import test_native_word_paths as word_fixtures
from signature_bank import BoundedIdentityReadout,IdentityReadoutState,WordSpanState
from data import _build_window_samples_from_text
from train import save_checkpoint,load_bundle_from_checkpoint

class NativeSpanBoundaryTests(unittest.TestCase):
    def fixture(self):
        f=word_fixtures.NativeWordPathTests();helper,t,m=f.setup_model();f.enable(m,t)
        m.cfg.identity_readout_native_conditional_prefix=True
        return helper,t,m

    def enable(self,m,t,categorical=False):
        old=m.bounded_identity_readout;m.cfg.identity_readout_native_span_boundaries=True
        if categorical:m.cfg.identity_readout_native_span_readout='categorical_v2'
        head=BoundedIdentityReadout(m.cfg);missing=head.load_state_dict(old.state_dict(),strict=False)
        expected={'native_span_surface.weight','native_span_gate.weight'}
        if categorical:expected.add('native_span_ready')
        self.assertEqual(set(missing.missing_keys),expected)
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t);return head

    def test_zero_migration_and_fragment_independent_word_roles(self):
        helper,t,m=self.fixture()
        s=_build_window_samples_from_text(t,'<BOI>The zorvax apple is gold.\nWhat color is the apple?<EOI><BOO>zorvax gold<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():before=m(s.input_ids[None],**helper.inputs(s)).logits
        head=self.enable(m,t)
        with torch.no_grad():out=m(s.input_ids[None],**helper.inputs(s))
        torch.testing.assert_close(before,out.logits,atol=0,rtol=0)
        self.assertEqual(out.bounded_identity_state[0].output_word_count,2)
        lexical=[u for u in t.construction_units if u.is_lexical]
        state=IdentityReadoutState([],word_node=1,output_word_count=1,source_mask=torch.tensor([True,True,False]))
        attention=torch.tensor([.1,.8,.1]);one=torch.tensor(1.);zero=torch.tensor(0.);nodes=[{'next':[]},{'next':[]}]
        a=head.span_boundary_vector(m.shared_signature_bank,state,lexical[0],one,zero,attention,[0,1,1],nodes)
        b=head.span_boundary_vector(m.shared_signature_bank,state,lexical[-1],one,zero,attention,[0,1,1],nodes)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        state.output_word_count=2
        c=head.span_boundary_vector(m.shared_signature_bank,state,lexical[0],one,zero,attention,[0,1,1],nodes)
        self.assertFalse(torch.equal(a,c))
        state.output_word_count=1
        d=head.span_boundary_vector(m.shared_signature_bank,state,lexical[0],one,zero,attention,[0,0,1],nodes)
        self.assertFalse(torch.equal(a,d))
        m.half()
        half=head.span_boundary_vector(m.shared_signature_bank,state,lexical[0],one,zero,attention,[0,1,1],nodes)
        self.assertEqual(half.dtype,head.native_span_surface.weight.dtype)
        self.assertTrue(torch.isfinite(half).all())

    def test_active_path_streams_reloads_and_has_finite_gradients(self):
        self.active_path(categorical=False)

    def test_categorical_path_is_neutral_until_ready_and_streams_reloads(self):
        self.active_path(categorical=True)

    def active_path(self,categorical):
        helper,t,m=self.fixture();head=self.enable(m,t,categorical=categorical)
        text='<BOI>The zorvax apple is gold.\nWhat color is the apple?<EOI><BOO>zorvax gold<EOO>'
        s=_build_window_samples_from_text(t,text,seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        alt=_build_window_samples_from_text(t,text.replace('<BOO>zorvax','<BOO>zerqux'),seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        vals=helper.inputs(s);runtime=identity=slots=lattice=None;parts=[]
        with torch.no_grad():before=m(s.input_ids[None],**vals).logits
        with torch.no_grad():head.native_span_surface.weight.normal_(0,3. if categorical else .2);head.native_span_gate.weight.normal_(0,.2)
        if categorical:
            with torch.no_grad():torch.testing.assert_close(before,m(s.input_ids[None],**vals).logits,atol=0,rtol=0)
            head.native_span_ready.fill_(True)
        with torch.no_grad():
            full=m(s.input_ids[None],**vals).logits;other=m(alt.input_ids[None],**helper.inputs(alt)).logits
            first=next(i for i,(a,b) in enumerate(zip(s.input_ids,alt.input_ids)) if a!=b)
            torch.testing.assert_close(full[:,:first],other[:,:first],atol=2e-6,rtol=1e-5)
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],**{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,signature_lattice_state=lattice,runtime_signature_state=runtime,bounded_identity_state=identity,position_index=i)
                parts.append(o.logits);runtime=o.runtime_signature_state;identity=o.bounded_identity_state;lattice=o.signature_lattice_state
            torch.testing.assert_close(full,torch.cat(parts,1),atol=2e-5,rtol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                p=save_checkpoint(m,directory,tokenizer=t);loaded,_,cfg=load_bundle_from_checkpoint(p,device='cpu',load_training_state=False)
                self.assertTrue(cfg.identity_readout_native_span_boundaries)
                if categorical:
                    self.assertEqual(cfg.identity_readout_native_span_readout,'categorical_v2')
                    self.assertTrue(bool(loaded.bounded_identity_readout.native_span_ready))
                torch.testing.assert_close(full,loaded(s.input_ids[None],**vals).logits,atol=0,rtol=0)
        loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**vals,loss_mask=s.loss_mask[None],collect_telemetry=False);loss.backward()
        for p in (head.native_span_surface.weight,head.native_span_gate.weight):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())

    def test_categorical_lexical_and_surface_mass_are_normalized(self):
        helper,t,m=self.fixture();head=self.enable(m,t,categorical=True)
        mask=torch.ones(t.vocab_size,dtype=torch.bool);mask[head.surface_ids]=False
        copy=torch.zeros(t.vocab_size);token=int(mask.nonzero()[0]);copy[token]=1.
        macro=torch.zeros(head.surface_ids.numel()+1);macro[-1]=1.
        probs=head.span_categorical_probabilities(torch.zeros(t.vocab_size),copy,torch.tensor(100.),macro)
        self.assertEqual(float(probs[token]),1.);self.assertEqual(float(probs.sum()),1.)
        macro.zero_();macro[0]=1.
        probs=head.span_categorical_probabilities(torch.zeros(t.vocab_size),copy,torch.tensor(0.),macro)
        self.assertEqual(float(probs[head.surface_ids[0]]),1.);self.assertEqual(float(probs.sum()),1.)

    def test_calibration_commits_only_correct_confident_positions(self):
        helper,t,m=self.fixture();head=self.enable(m,t,categorical=True)
        width=head.surface_ids.numel()+1;logits=torch.zeros(2,width);logits[0,0]=20.
        result=head.commit_span_calibration(logits,torch.tensor([1,2]))
        self.assertFalse(result['accepted']);self.assertFalse(bool(head.native_span_ready))
        result=head.commit_span_calibration(logits,torch.tensor([0,2]))
        self.assertTrue(result['accepted']);self.assertTrue(bool(head.native_span_ready))
        self.assertEqual(result['selected_positions'],1)

    def test_source_separators_distinguish_same_prefix_and_clone_without_future_output(self):
        helper,t,m=self.fixture();head=self.enable(m,t,categorical=True)
        def state(text,output):
            words=WordSpanState(256);entries=[]
            for token in t.encode(text):
                unit=t.construction_units[token]
                if unit.is_lexical:entries.append((token,(),(),words.serial,tuple(e[0] for e in entries[-8:])))
                words.observe(token,unit)
            return IdentityReadoutState(entries,words=words,output_units=[i for i in t.encode(output) if t.construction_units[i].is_lexical],output_separators=[])
        a=state('Gold with Blue!\n','Gold with Blue');b=state('Gold with Blue Red!\n','Gold with Blue')
        av=head.source_boundary_vector(m.shared_signature_bank,a,t,torch.device('cpu'))
        bv=head.source_boundary_vector(m.shared_signature_bank,b,t,torch.device('cpu'))
        self.assertFalse(torch.equal(av,bv))
        clone=a.words.clone();last=a.words.records[-1][0]
        self.assertEqual(clone.following,a.words.following)
        clone.following[last].append(t.eos_id)
        self.assertNotEqual(clone.following,a.words.following)
        a.output_units=[t.pad_id]
        self.assertEqual(float(head.source_boundary_vector(m.shared_signature_bank,a,t,torch.device('cpu')).abs().sum()),0.)

    def test_source_boundary_migration_streams_and_reloads(self):
        self.source_boundary_path('residual_v1')

    def test_source_categorical_migration_streams_reloads_and_differentiates(self):
        self.source_boundary_path('categorical_v1')

    def test_source_grammar_adapter_streams_reloads_and_is_neutral(self):
        self.source_boundary_path('categorical_v1',adapter=True)

    def test_uncertain_grammar_proposal_preserves_validated_source_logits(self):
        helper,t,m=self.fixture();self.enable(m,t,categorical=True)
        m.cfg.identity_readout_source_boundaries=True;m.cfg.identity_readout_source_boundary_readout='categorical_v1';m.cfg.identity_readout_source_boundary_adapter=True
        head=BoundedIdentityReadout(m.cfg);m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t)
        x=torch.ones(2*m.cfg.d_model)
        with torch.no_grad():
            head.native_source_boundary.weight.zero_();head.native_source_boundary.weight[0].fill_(20./x.numel())
            head.native_source_boundary_adapter.weight.zero_();head.native_source_boundary_adapter.weight[0].fill_(-20./x.numel())
            base=head.native_source_boundary(x)
            torch.testing.assert_close(head.source_boundary_logits(x),base,atol=0,rtol=0)
            head.native_source_boundary_adapter.weight[1].fill_(20./x.numel())
            self.assertEqual(int(head.source_boundary_logits(x).argmax()),1)

    def source_boundary_path(self,mode,adapter=False):
        helper,t,m=self.fixture();head=self.enable(m,t,categorical=True);head.native_span_ready.fill_(True)
        s=_build_window_samples_from_text(t,'<BOI>Gold with Blue Red!\nQuestion: What is the event?<EOI><BOO>Gold with Blue Red!<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        vals=helper.inputs(s)
        with torch.no_grad():before=m(s.input_ids[None],**vals).logits
        m.cfg.identity_readout_source_boundaries=True;m.cfg.identity_readout_source_boundary_readout=mode;m.cfg.identity_readout_source_boundary_adapter=adapter;replacement=BoundedIdentityReadout(m.cfg)
        missing=replacement.load_state_dict(head.state_dict(),strict=False)
        self.assertEqual(set(missing.missing_keys),{'native_source_boundary.weight'}|({'native_source_boundary_adapter.weight'} if adapter else set()))
        m.bounded_identity_readout=replacement;m.prepare_capacity_for_tokenizer(t)
        with torch.no_grad():torch.testing.assert_close(before,m(s.input_ids[None],**vals).logits,atol=0,rtol=0)
        with torch.no_grad():replacement.native_source_boundary.weight.normal_(0,10. if mode=='categorical_v1' else .3)
        if adapter:
            with torch.no_grad():replacement.native_source_boundary_adapter.weight.normal_(0,10.)
        runtime=identity=slots=lattice=None;parts=[]
        with torch.no_grad():
            full=m(s.input_ids[None],**vals).logits
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],**{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,signature_lattice_state=lattice,runtime_signature_state=runtime,bounded_identity_state=identity,position_index=i)
                parts.append(o.logits);runtime=o.runtime_signature_state;identity=o.bounded_identity_state;lattice=o.signature_lattice_state
            torch.testing.assert_close(full,torch.cat(parts,1),atol=2e-5,rtol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                p=save_checkpoint(m,directory,tokenizer=t);loaded,_,cfg=load_bundle_from_checkpoint(p,device='cpu',load_training_state=False)
                self.assertTrue(cfg.identity_readout_source_boundaries)
                self.assertEqual(cfg.identity_readout_source_boundary_readout,mode)
                self.assertEqual(cfg.identity_readout_source_boundary_adapter,adapter)
                torch.testing.assert_close(full,loaded(s.input_ids[None],**vals).logits,atol=0,rtol=0)
        if mode=='categorical_v1':
            loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**vals,loss_mask=s.loss_mask[None],collect_telemetry=False);loss.backward()
            self.assertIsNotNone(replacement.native_source_boundary.weight.grad)
            self.assertTrue(torch.isfinite(replacement.native_source_boundary.weight.grad).all())
            if adapter:self.assertTrue(torch.isfinite(replacement.native_source_boundary_adapter.weight.grad).all())

if __name__=='__main__':unittest.main()
