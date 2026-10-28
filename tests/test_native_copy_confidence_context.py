import copy,tempfile,unittest
from pathlib import Path
import torch
from config import PrismalWaveConfig
from native_question_memory import NativeQuestionMemory
from native_qa_transfer_study import full_incremental_check
from tests import test_native_backbone_representation as fixtures
from train import save_checkpoint,load_bundle_from_checkpoint

class CopyConfidenceContextTests(unittest.TestCase):
    def fixture(self):
        helper=fixtures.NativeBackboneTests();m,t=helper.model(all_heads=True)
        m.cfg.native_question_memory_copy_context='observed_mean_v1'
        m.native_question_memory.copy_context='observed_mean_v1'
        m.native_question_memory.span_readout.confidence_context='observed_mean_v1'
        m.invalidate_native_context()
        with torch.no_grad():m.native_question_memory.span_readout.copy_gate.weight.normal_(std=.1)
        return helper,m,t

    def test_default_and_invalid_modes(self):
        self.assertEqual(PrismalWaveConfig().native_question_memory_copy_context,'retrieved_v1')
        with self.assertRaisesRegex(ValueError,'Unknown native copy'):
            PrismalWaveConfig(native_question_memory_copy_context='wrong')
        with self.assertRaisesRegex(ValueError,'requires backbone'):
            PrismalWaveConfig(native_question_memory_copy_context='observed_mean_v1')
        with self.assertRaises(ValueError):NativeQuestionMemory(8,copy_context='observed_mean_v1')

    def test_retrieval_can_change_without_changing_confidence_or_its_source_gradient(self):
        helper,m,t=self.fixture();out,plan=helper.plan(m,t);memory=m.native_question_memory
        q=out.native_qa_readouts[-1][2];words=plan['base_word_vectors'];values=memory.value(words)
        scores=torch.linspace(-10,10,len(words),requires_grad=True)
        make=lambda x:memory.decision_packet(q,words,values,x,plan['marker_words'],None,lambda _:plan['candidates'],plan['source_words'])[3]
        a,b=make(scores),make(-scores)
        self.assertFalse(torch.equal(a['copy_context'],b['copy_context']))
        torch.testing.assert_close(a['copy_gate_context'],values.mean(0),rtol=0,atol=0)
        head=memory.span_readout;gate=head.gate_logit(q,q,a)
        torch.testing.assert_close(gate,head.gate_logit(q,q,b),rtol=0,atol=0)
        self.assertIsNone(torch.autograd.grad(gate,scores,allow_unused=True,retain_graph=True)[0])
        self.assertGreater(float(torch.autograd.grad(gate,values,retain_graph=True)[0].square().sum()),0.)
        head.confidence_context='retrieved_v1'
        self.assertGreater(float(torch.autograd.grad(head.gate_logit(q,q,a),scores)[0].square().sum()),0.)

    def test_source_extent_guidance_and_state_shapes_are_unchanged(self):
        helper,m,t=self.fixture();m.requires_grad_(False)
        with torch.no_grad():_,a=helper.plan(m,t)
        memory=m.native_question_memory;memory.copy_context='retrieved_v1';memory.span_readout.confidence_context='retrieved_v1';m.invalidate_native_context()
        with torch.no_grad():_,b=helper.plan(m,t)
        for key in ('selected_starts','candidates'):self.assertEqual(a[key],b[key])
        for key in ('copy_context','start_log_probabilities','log_probabilities','word_vectors','source_keys'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        current=torch.randn_like(a['copy_context']);q=torch.randn_like(current)
        torch.testing.assert_close(memory.guide_readout(current,q,a['copy_context']),memory.guide_readout(current,q,b['copy_context']),rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in memory.span_readout.copy_gate.parameters()),2*len(q)+1)

    def test_confidence_never_depends_on_proposal_or_prefix_interventions(self):
        helper,m,t=self.fixture();out,p=helper.plan(m,t);head=m.native_question_memory.span_readout
        q=out.native_qa_readouts[-1][2];before=head.gate_logit(q,q,p)
        changed=dict(p,candidates=[],copy_context=p['copy_context']*1000,probabilities=torch.ones(1),selected_starts=[])
        torch.testing.assert_close(before,head.gate_logit(q,q*100,changed),rtol=0,atol=0)
        del changed['copy_gate_context']
        with self.assertRaisesRegex(ValueError,'input-only'):head.gate_logit(q,q,changed)

    def test_carried_context_cannot_cross_confidence_policies(self):
        from native_qa_transfer_study import generation_fields
        helper,m,t=self.fixture();m.requires_grad_(False);ids,tracks=generation_fields(t,helper.row)
        with torch.no_grad():out=m(ids,**tracks,collect_telemetry=False)
        m.native_question_memory.copy_context='retrieved_v1'
        m.native_question_memory.span_readout.confidence_context='retrieved_v1'
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'context.*stale|context.*producer'):
            m.forward_incremental(torch.tensor([[t.special_tokens['<LINE>']]]),position_index=ids.shape[1],
                slot_state=out.slot_state,signature_lattice_state=out.signature_lattice_state,runtime_signature_state=out.runtime_signature_state)

    def test_live_functional_replay_uses_observed_context_and_remains_differentiable(self):
        from tests import test_native_live_replay as replay_tests
        helper,m,t,batch=replay_tests.fixture()
        m.cfg.native_question_memory_copy_context='observed_mean_v1'
        m.native_question_memory.copy_context='observed_mean_v1'
        m.native_question_memory.span_readout.confidence_context='observed_mean_v1';m.invalidate_native_context()
        loss,out=replay_tests.compute(m,batch)
        plan=out.native_qa_readouts[0][4]
        expected=m.native_question_memory.value(plan['base_word_vectors']).mean(0)
        torch.testing.assert_close(plan['copy_gate_context'],expected,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(out.route_stats['native_replay_gate']))
        loss.backward()
        self.assertGreater(float(m.native_question_memory.span_readout.copy_gate.weight.grad.norm()),0.)

    def test_reload_incremental_and_future_answer_causality(self):
        helper,m,t=self.fixture();m.requires_grad_(False)
        full_incremental_check(m,t,helper.row)
        left=helper.annotated_batch(m,t,'12');right=helper.annotated_batch(m,t,'different')
        a,b=left['tensors'][0][0],right['tensors'][0][0];n=next(i for i in range(min(len(a),len(b))) if a[i]!=b[i])
        with torch.no_grad():_,x=helper.compute(m,left);_,y=helper.compute(m,right)
        torch.testing.assert_close(x.logits[:,:n],y.logits[:,:n],rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(m,folder,tokenizer=t)
            loaded,rt,cfg=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu',load_training_state=False);loaded.eval();loaded.requires_grad_(False)
            self.assertEqual(cfg.native_question_memory_copy_context,'observed_mean_v1')
            with torch.no_grad():_,reloaded=helper.compute(loaded,left)
            torch.testing.assert_close(x.logits,reloaded.logits,rtol=0,atol=0)

if __name__=='__main__':torch.set_num_threads(1);unittest.main()
