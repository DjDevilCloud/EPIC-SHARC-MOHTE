"""Entity modifiers, order, causality, neutral migration and native-branch separation."""
import unittest,tempfile
import torch
from signature_bank import BoundedIdentityReadout,WordSpanState
from data import _build_window_samples_from_text
from train import save_checkpoint,load_bundle_from_checkpoint
from tests import test_word_span_binding as fixtures

class OrderedSpanAgreementTests(unittest.TestCase):
    def setup_model(self):
        fixture=fixtures.NativeRouteWordTests();helper,t,m=fixture.setup_model();fixture.enable(m,t,active=True)
        return helper,t,m

    def enable_agreement(self,m,t):
        old=m.bounded_identity_readout;m.cfg.identity_readout_ordered_agreement=True
        head=BoundedIdentityReadout(m.cfg);head.load_state_dict(old.state_dict(),strict=False)
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t)
        return head

    def test_ordered_identity_agreement_retains_modifiers_and_word_boundaries(self):
        pink=(1,2);gray=(3,);melon=(4,5);is_word=(6,)
        question=[(7,),pink,melon]
        self.assertEqual(WordSpanState.ordered_agreement(question,[pink,melon,is_word]),2)
        self.assertEqual(WordSpanState.ordered_agreement(question,[gray,melon,is_word]),1)
        self.assertEqual(WordSpanState.ordered_agreement(question,[melon,pink,is_word]),1)
        self.assertEqual(WordSpanState.ordered_agreement(question,[(1,),(2,),melon]),1)
        self.assertEqual(WordSpanState.ordered_agreement(question,[]),0)

    def test_margin_excludes_question_answers_and_groups_duplicate_source_answers(self):
        scores=torch.tensor([.2,.7,.6,10.],requires_grad=True)
        positive=torch.tensor([True,True,False,True]);eligible=torch.tensor([True,True,True,False])
        loss=BoundedIdentityReadout.retrieval_margin_loss(scores,positive,eligible,margin=1.)
        loss.backward()
        self.assertLess(float(scores.grad[1]),0.)
        self.assertGreater(float(scores.grad[2]),0.)
        self.assertEqual(float(scores.grad[3]),0.)
        with self.assertRaises(ValueError):
            BoundedIdentityReadout.retrieval_margin_loss(scores,torch.tensor([False,False,False,True]),eligible)

    def test_neutral_migration_and_native_relation_independence(self):
        helper,t,m=self.setup_model()
        s=_build_window_samples_from_text(t,'<BOI>The red apple is gold. The green apple is blue.\nWhat color is the red apple?<EOI><BOO>gold<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        with torch.no_grad():before=m(s.input_ids[None],**helper.inputs(s)).logits
        head=self.enable_agreement(m,t)
        with torch.no_grad():after=m(s.input_ids[None],**helper.inputs(s))
        torch.testing.assert_close(before,after.logits,atol=0,rtol=0)
        state=after.bounded_identity_state[0]
        with torch.no_grad():
            a=head._word_binding_memory(state.entries,state.words,t,m.shared_signature_bank,m.construction_embedding,torch.device('cpu'))
            head.word_agreement_scale.fill_(.4)
            b=head._word_binding_memory(state.entries,state.words,t,m.shared_signature_bank,m.construction_embedding,torch.device('cpu'))
        torch.testing.assert_close(a[3],b[3],atol=0,rtol=0)
        expected=torch.tensor([WordSpanState.ordered_agreement(state.words.question(),e[2]) for e in state.entries])*.4
        torch.testing.assert_close(b[1]-a[1],expected,atol=1e-6,rtol=1e-5)

    def test_active_agreement_is_causal_reloads_streams_and_receives_gradient(self):
        helper,t,m=self.setup_model();head=self.enable_agreement(m,t);m.cfg.identity_readout_exclude_query_candidates=True
        with torch.no_grad():head.word_agreement_scale.fill_(.3)
        text='<BOI>The red apple is gold. The green apple is blue.\nWhat color is the red apple?<EOI><BOO>gold<EOO>'
        s=_build_window_samples_from_text(t,text,seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        alt=_build_window_samples_from_text(t,text.replace('<BOO>gold','<BOO>blue'),seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        vals=helper.inputs(s);runtime=identity=slots=lattice=None;pieces=[]
        with torch.no_grad():
            full=m(s.input_ids[None],**vals).logits
            other=m(alt.input_ids[None],**helper.inputs(alt)).logits
            first=next(i for i,(a,b) in enumerate(zip(s.input_ids,alt.input_ids)) if a!=b)
            torch.testing.assert_close(full[:,:first],other[:,:first],atol=0,rtol=0)
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],**{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,signature_lattice_state=lattice,runtime_signature_state=runtime,bounded_identity_state=identity,position_index=i)
                pieces.append(o.logits);runtime=o.runtime_signature_state;identity=o.bounded_identity_state;lattice=o.signature_lattice_state
            torch.testing.assert_close(full,torch.cat(pieces,dim=1),atol=2e-5,rtol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                path=save_checkpoint(m,directory,tokenizer=t);restored,_,cfg=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
                self.assertTrue(cfg.identity_readout_ordered_agreement)
                torch.testing.assert_close(full,restored(s.input_ids[None],**vals).logits,atol=0,rtol=0)
        m.requires_grad_(False);head.word_agreement_scale.requires_grad_(True)
        loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**vals,loss_mask=s.loss_mask[None],collect_telemetry=False);loss.backward()
        self.assertIsNotNone(head.word_agreement_scale.grad)
        self.assertTrue(torch.isfinite(head.word_agreement_scale.grad))
        self.assertGreater(float(head.word_agreement_scale.grad.abs()),0)

if __name__=='__main__':unittest.main()
