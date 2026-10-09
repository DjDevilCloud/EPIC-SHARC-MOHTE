"""Binding memory, structural answer-form decisions and compatible checkpoints."""
import tempfile
import unittest
import torch
from config import PrismalWaveConfig
from data import _build_window_samples_from_text
from signature_bank import BoundedIdentityReadout
from train import save_checkpoint, load_bundle_from_checkpoint
from tests import test_runtime_signatures as runtime_fixtures

class LexicalBindingTests(unittest.TestCase):
    def setup_model(self):
        fixture=runtime_fixtures.RuntimeSignatureTests();fixture.identity_readout=True
        helper,t,m=fixture.setup_model()
        return helper,t,m

    def enable(self,m,t,active=False):
        old=m.bounded_identity_readout
        m.cfg.identity_readout_binding='lexical_binding_v1'
        with torch.random.fork_rng():head=BoundedIdentityReadout(m.cfg)
        head.load_state_dict(old.state_dict(),strict=False)
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t)
        if active:
            with torch.no_grad():
                for layer in (head.context_query,head.context_gate,head.surface_projection,head.applicability):
                    layer.weight.normal_(0,.03)
                head.span_scale.fill_(.7)
        return head

    def test_zero_migration_and_parameters_bound_before_optimizer(self):
        helper,t,m=self.setup_model()
        old_payload=m.cfg.to_dict()
        for key in ('identity_readout_binding','identity_readout_context_units','identity_readout_query_units','identity_readout_surface_ids'):
            old_payload.pop(key)
        self.assertEqual(PrismalWaveConfig.from_dict(old_payload).identity_readout_binding,'independent_v1')
        s=_build_window_samples_from_text(t,'<BOI>red apple<EOI><BOO>red<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():before=m(s.input_ids[None],**helper.inputs(s)).logits
        head=self.enable(m,t);weight=head.surface_projection.weight
        opt=torch.optim.AdamW(m.parameters())
        with torch.no_grad():after=m(s.input_ids[None],**helper.inputs(s)).logits
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        self.assertIs(weight,head.surface_projection.weight)
        self.assertTrue(any(p is weight for group in opt.param_groups for p in group['params']))
        self.assertEqual(m.cfg.identity_readout_surface_ids,head.surface_ids.tolist())

    def test_active_path_is_causal_streams_reloads_and_receives_gradients(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True)
        text='<BOI>red apple café green fruit<EOI><BOO>red apple<EOO>'
        left=_build_window_samples_from_text(t,text,seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        right=_build_window_samples_from_text(t,text.replace('<BOO>red apple','<BOO>green'),seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        vals=helper.inputs(left)
        with torch.no_grad():
            full=m(left.input_ids[None],**vals)
            alternative=m(right.input_ids[None],**helper.inputs(right))
            first=next(i for i,(a,b) in enumerate(zip(left.input_ids,right.input_ids)) if a!=b)
            torch.testing.assert_close(full.logits[:,:first],alternative.logits[:,:first],atol=0,rtol=0)
            self.assertEqual(full.bounded_identity_state[0].entries,alternative.bounded_identity_state[0].entries)
            self.assertTrue(all(len(e)==3 and len(e[2])<=head.context_units for e in full.bounded_identity_state[0].entries))
            state=identity=slots=lattice=None;parts=[]
            for i in range(left.input_ids.numel()):
                _,slots,o=m.forward_incremental(left.input_ids[i:i+1][None],
                    **{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,signature_lattice_state=lattice,
                    runtime_signature_state=state,bounded_identity_state=identity,position_index=i)
                state=o.runtime_signature_state;identity=o.bounded_identity_state;lattice=o.signature_lattice_state;parts.append(o.logits)
            torch.testing.assert_close(full.logits,torch.cat(parts,1),rtol=1e-4,atol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                path=save_checkpoint(m,directory,tokenizer=t)
                restored,_,cfg=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
                self.assertEqual(cfg.identity_readout_binding,'lexical_binding_v1')
                torch.testing.assert_close(full.logits,restored(left.input_ids[None],**vals).logits,atol=0,rtol=0)
        m.train();loss,_=m.compute_loss(left.input_ids[None],left.labels[None],**vals,loss_mask=left.loss_mask[None],collect_telemetry=False);loss.backward()
        for layer in (head.context_query,head.context_gate,head.span_projection,head.surface_projection,head.applicability):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()),0)

    def test_memory_eviction_and_branch_isolation_with_neighbors(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True);head.capacity=3
        p=t.prepare_generation_hierarchy('red apple green fruit')
        vals={k:torch.tensor([getattr(p,k)]) for k in ('signature_ids','signature_family_ids','signature_level_ids','signature_relation_ids','parent_signature_ids','hierarchy_vectors')}
        with torch.no_grad():
            first=m(torch.tensor([p.token_ids]),**vals);entries=list(first.bounded_identity_state[0].entries)
            second=m(torch.tensor([p.token_ids]),**vals,bounded_identity_state=first.bounded_identity_state)
            self.assertEqual(entries,first.bounded_identity_state[0].entries)
            self.assertEqual(entries,second.bounded_identity_state[0].entries)
            self.assertEqual(len(entries),3)
            eligible=[i for i in p.token_ids if head.is_candidate(t.construction_units[i])]
            self.assertEqual([e[0] for e in entries],eligible[-3:])
            self.assertEqual(entries[-1][2],tuple(eligible[-3:-1]))

    def test_surface_structure_does_not_encode_lexical_value_identity(self):
        helper,t,m=self.setup_model()
        t.learn_from_texts(['gold gray gold gray']*4,min_frequency=2,max_new_tokens=16,max_word_tokens=16)
        m.resize_vocab(t.vocab_size);m.prepare_capacity_for_tokenizer(t);head=self.enable(m,t)
        def structure(word):
            p=t.prepare_generation_hierarchy('red apple');ids=p.token_ids+t.encode(word,add_special_tokens=False)
            features,_=m.shared_signature_bank.runtime_features(torch.tensor([ids]),t)
            token=ids[-2] if t.construction_units[ids[-1]].text=='<EOL>' else ids[-1]
            index=len(ids)-2 if token!=ids[-1] else len(ids)-1
            unit=t.construction_units[token]
            # Use the state just after the lexical unit rather than a trailing control.
            self.assertTrue(unit.is_lexical)
            def identity(x):return torch.nn.functional.layer_norm((m.construction_embedding(x)+m.shared_signature_bank.encode('token',x))/(2**.5),(m.cfg.d_model,))
            return token,head._binding_structure(token,unit,features[0,index],m.shared_signature_bank,identity,torch.zeros(m.cfg.d_model))
        a,x=structure('gold');b,y=structure('gray')
        self.assertNotEqual(a,b)
        torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_confident_retained_route_is_exactly_neutral_and_uncertainty_blends(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True)
        s=_build_window_samples_from_text(t,'<BOI>red apple<EOI><BOO>red apple<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():
            head.applicability.weight.zero_();head.applicability.bias.fill_(-20)
            head.binding_enabled=False;retained=m(s.input_ids[None],**helper.inputs(s)).logits
            head.binding_enabled=True;neutral=m(s.input_ids[None],**helper.inputs(s)).logits
            torch.testing.assert_close(retained,neutral,atol=0,rtol=0)
            question=torch.zeros(m.cfg.d_model)
            self.assertEqual(float(head.binding_activation(question)),0.)
            head.applicability.bias.zero_()
            self.assertEqual(float(head.binding_activation(question)),.5)
            head.applicability.bias.fill_(20)
            self.assertEqual(float(head.binding_activation(question)),1.)

    def test_adapter_training_freezes_registry_promotions_and_original_weights_on_reload(self):
        helper,t,m=self.setup_model();head=self.enable(m,t,active=True)
        m.freeze_binding_backbone()
        buffers={name:value.clone() for name,value in m.registry.named_buffers()}
        s=_build_window_samples_from_text(t,'<BOI>new red apple<EOI><BOO>red<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        m.train();loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**helper.inputs(s),loss_mask=s.loss_mask[None],collect_telemetry=False);loss.backward()
        for name,value in m.registry.named_buffers():torch.testing.assert_close(value,buffers[name],atol=0,rtol=0)
        self.assertFalse(m.shared_signature_bank.embedding.weight.requires_grad)
        self.assertIsNone(m.shared_signature_bank.embedding.weight.grad)
        self.assertGreater(float(head.context_query.weight.grad.abs().sum()),0.)
        with tempfile.TemporaryDirectory() as directory:
            restored,_,cfg=load_bundle_from_checkpoint(save_checkpoint(m,directory,tokenizer=t),device='cpu',load_training_state=False)
            self.assertTrue(cfg.binding_adapter_training)
            self.assertFalse(restored.registry.observation_updates_enabled)
            self.assertFalse(restored.shared_signature_bank.embedding.weight.requires_grad)
            self.assertTrue(restored.bounded_identity_readout.context_query.weight.requires_grad)

if __name__=='__main__':unittest.main()
