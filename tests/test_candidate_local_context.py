import copy
import tempfile
import unittest
from pathlib import Path

import torch
from config import PrismalWaveConfig
from native_backbone_readout import candidate_local_context,clause_neighbors
from native_question_memory import NativeQuestionMemory
from native_qa_transfer_study import full_incremental_check,generation_fields
from tests import test_native_backbone_representation as backbone_fixtures
from train import save_checkpoint,load_bundle_from_checkpoint,generate_text


class CandidateLocalContextTests(unittest.TestCase):
    def fixture(self,all_heads=False):
        m,t=backbone_fixtures.NativeBackboneTests().model(all_heads=all_heads)
        m.cfg.native_question_memory_backbone_context='candidate_local_clause_v1'
        m.native_question_memory.backbone_context='candidate_local_clause_v1'
        m.invalidate_native_context()
        return m,t

    def test_saved_default_and_invalid_combinations(self):
        self.assertEqual(PrismalWaveConfig().native_question_memory_backbone_context,'completed_line_v1')
        with self.assertRaisesRegex(ValueError,'requires the backbone'):
            PrismalWaveConfig(native_question_memory_backbone_context='candidate_local_clause_v1')
        with self.assertRaisesRegex(ValueError,'Unknown native backbone'):
            PrismalWaveConfig(native_question_memory_backbone_context='other')
        with self.assertRaisesRegex(ValueError,'requires the backbone'):
            NativeQuestionMemory(8,backbone_context='candidate_local_clause_v1')

    def test_pool_excludes_candidate_and_other_clauses(self):
        words=torch.tensor([[1.,0.],[50.,50.],[0.,1.],[90.,90.]],requires_grad=True)
        ctx=candidate_local_context(words,[1,1,2,2])
        torch.testing.assert_close(ctx[1],words[0],rtol=0,atol=0)
        torch.testing.assert_close(ctx[3],words[2],rtol=0,atol=0)
        gradient=torch.autograd.grad(ctx[1].sum(),words)[0]
        self.assertTrue(torch.equal(gradient,torch.tensor([[1.,1.],[0.,0.],[0.,0.],[0.,0.]])))
        self.assertTrue(torch.equal(candidate_local_context(words,[0,1,2,3]),torch.zeros_like(words)))

    def test_window_is_bounded_and_empty_input_is_valid(self):
        self.assertEqual(clause_neighbors([0]*40,20),(list(range(19,11,-1)),list(range(21,29))))
        words=torch.arange(40.).view(-1,1)
        ctx=candidate_local_context(words,[0]*40)
        expected=words[list(range(12,20))+list(range(21,29))].mean(0)
        torch.testing.assert_close(ctx[20],expected)
        self.assertEqual(candidate_local_context(torch.zeros(0,8),[]).shape,(0,8))
        with self.assertRaises(ValueError):candidate_local_context(words,[0])

    def test_same_line_cancellation_and_local_context_carries_fact(self):
        words=torch.tensor([[1.,0.],[0.,0.],[0.,1.],[0.,0.]],requires_grad=True)
        query=torch.tensor([1.,0.]);projection=torch.eye(2,requires_grad=True)
        line=torch.tensor([.2,.4]).expand(4,2)
        constant=(line@projection.T)@query
        self.assertEqual(float(constant[1]-constant[3]),0.)
        contexts=candidate_local_context(words,[0,0,1,1])
        scores=(contexts@projection.T)@query
        self.assertEqual(int(scores.argmax()),1)
        loss=-scores.log_softmax(0)[1]
        self.assertGreater(float(torch.autograd.grad(loss,projection)[0].norm()),0.)

    def test_actual_policy_changes_no_parameters_or_proposals(self):
        fixture=backbone_fixtures.NativeBackboneTests();line=fixture.memory();local=fixture.memory(backbone_context='candidate_local_clause_v1')
        local.load_state_dict(line.state_dict())
        self.assertEqual(list(line.state_dict()),list(local.state_dict()))
        t,_,ids,h,_,state,records=fixture.observed(memory=line)
        _,_,lr=local(ids,h,None,t,return_readout=True,diagnostics=True)
        a=records[-1][4];b=lr[-1][4]
        self.assertEqual(sorted(a['candidates'],key=lambda c:(c['start'],c['end'],c['tokens'])),
                         sorted(b['candidates'],key=lambda c:(c['start'],c['end'],c['tokens'])))
        self.assertTrue(torch.equal(a['word_vectors'],b['word_vectors']))
        self.assertFalse(torch.equal(a['source_key_context'],b['source_key_context']))
        self.assertEqual(b['key_context_policy'],'candidate_local_clause_v1')

    def test_standalone_full_incremental_and_cached_plan(self):
        fixture=backbone_fixtures.NativeBackboneTests();m=fixture.memory(backbone_context='candidate_local_clause_v1',source_roles=True,
            relation_alignment=True,future_context=8);m.eval()
        t,_,ids,h,_,_,_=fixture.observed(memory=m)
        with torch.no_grad():
            _,full=m(ids,h,None,t);states=None
            for i in range(ids.shape[1]):_,states=m(ids[:,i:i+1],h[:,i:i+1],None,t,states)
        a,b=full[0].cached_packet[3],states[0].cached_packet[3]
        self.assertEqual(a['candidates'],b['candidates'])
        torch.testing.assert_close(a['start_log_probabilities'],b['start_log_probabilities'],rtol=0,atol=0)
        with torch.no_grad():
            _,carried=m(torch.tensor([[t.special_tokens['<LINE>']]]),torch.zeros(1,1,8),None,t,states)
        self.assertIs(carried[0].cached_packet,states[0].cached_packet)

    def test_policy_change_rejects_carried_state(self):
        m,t=self.fixture();ids,tracks=generation_fields(t,backbone_fixtures.NativeBackboneTests.row)
        with torch.no_grad():out=m(ids,**tracks,collect_telemetry=False)
        m.native_question_memory.backbone_context='completed_line_v1'
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'stale'):
            m.forward_incremental(torch.tensor([[t.special_tokens['<LINE>']]]),position_index=ids.shape[1],
                slot_state=out.slot_state,signature_lattice_state=out.signature_lattice_state,
                runtime_signature_state=out.runtime_signature_state)

    def test_reload_normal_generation_and_incremental(self):
        m,t=self.fixture(all_heads=True);m.requires_grad_(False)
        self.assertLess(full_incremental_check(m,t,backbone_fixtures.NativeBackboneTests.row),3e-4)
        from native_qa_transfer_study import prompt
        before=generate_text(m,t,prompt(backbone_fixtures.NativeBackboneTests.row),torch.device('cpu'),return_generation_record=True)
        with tempfile.TemporaryDirectory() as folder:
            save_checkpoint(m,folder,tokenizer=t)
            loaded,rt,cfg=load_bundle_from_checkpoint(Path(folder)/'model.pt',device='cpu',load_training_state=False)
            self.assertEqual(cfg.native_question_memory_backbone_context,'candidate_local_clause_v1')
            self.assertEqual(loaded.native_question_memory.backbone_context,'candidate_local_clause_v1')
            after=generate_text(loaded,rt,prompt(backbone_fixtures.NativeBackboneTests.row),torch.device('cpu'),return_generation_record=True)
            self.assertEqual(before,after)

    def test_gold_and_future_answer_do_not_change_observed_contexts(self):
        helper=backbone_fixtures.NativeBackboneTests();m,t=self.fixture(all_heads=True)
        a=helper.annotated_batch(m,t,'12');b=helper.annotated_batch(m,t,'unrelated')
        other=copy.deepcopy(a['native_source_targets']);other[0]['starts']=[5]
        with torch.no_grad():
            _,x=helper.compute(m,a);_,y=helper.compute(m,a,other);_,z=helper.compute(m,b)
        torch.testing.assert_close(x.logits,y.logits,rtol=0,atol=0)
        torch.testing.assert_close(x.native_qa_readouts[0][4]['start_log_probabilities'],
                                   z.native_qa_readouts[0][4]['start_log_probabilities'],rtol=2e-4,atol=3e-5)

    def test_real_source_context_gradient_and_shared_core_path(self):
        helper=backbone_fixtures.NativeBackboneTests();m,t=self.fixture(all_heads=True);batch=helper.annotated_batch(m,t)
        _,out=helper.compute(m,batch);loss=-out.native_qa_readouts[0][4]['start_log_probabilities'][2]
        loss.backward();width=m.cfg.d_model
        self.assertGreater(float(m.native_question_memory.key.weight.grad[:,width:].norm()),0.)
        for name in ('torus_core','router','signature_lattice_attention','shared_signature_bank'):
            norm=sum(float(p.grad.square().sum()) for p in getattr(m,name).parameters() if p.grad is not None)
            self.assertGreater(norm,0.,name)

    def test_oracle_payload_and_termination_with_local_context(self):
        class LocalFixture(backbone_fixtures.NativeBackboneTests):
            def memory(self,capacity=128,**extra):
                return super().memory(capacity,backbone_context='candidate_local_clause_v1',**extra)
        LocalFixture().test_conditional_payload_mechanics_with_explicit_high_confidence_fixture()

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
