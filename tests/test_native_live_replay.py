import copy
import unittest
import torch
import torch.nn.functional as F
from native_live_replay import balanced_output_ce,source_preservation,frozen_tensor,PACKET_KEYS
from native_qa_functional_replay import protected_gate,protected_tokens,protected_decision
from tests import test_native_backbone_representation as backbone_tests


def fixture():
    helper=backbone_tests.NativeBackboneTests();model,t=helper.model(all_heads=True);batch=helper.annotated_batch(model,t)
    _,out=helper.compute(model,batch);r,pos,q,current,plan,emitted=next(x for x in out.native_qa_readouts if not x[-1])
    tensors=batch['tensors'];positions=(tensors[-1][0]*tensors[1][0].ne(t.pad_id)).nonzero().flatten()
    payload=tuple(tensors[1][0,positions].tolist());assert payload
    groups=[]
    from native_question_memory import NativeQuestionState
    state=NativeQuestionState()
    for token in tensors[0][0,:pos+1].tolist():
        state,_=model.native_question_memory._observe(state,token,t,torch.zeros(model.cfg.d_model))
    for start in batch['native_source_targets'][0]['starts']:
        candidates=model.native_question_memory.span_readout.candidates(state,t,[start])
        assert any(tuple(c['tokens'])==payload for c in candidates)
        groups.append(dict(start=start,candidates=candidates,teacher_logits=None))
    packet=dict(version='native_live_replay_v1',input_ids=tensors[0][0].tolist(),position=pos,
        source_words=plan['source_words'],starts=[2],extent_groups=groups,positions=positions.tolist(),labels=list(payload),
        teacher_source=None,teacher_gate=None,teacher_tokens=out.logits[0,positions].detach().clone(),provenance={})
    assert set(packet)==PACKET_KEYS
    batch['native_replay_targets']=[dict(row=0,position=pos,packet=packet)]
    return helper,model,t,batch

def compute(model,batch):
    fields=('input_ids','labels','signature_ids','signature_level_ids','signature_relation_ids',
            'parent_signature_ids','signature_family_ids','hierarchy_vectors','loss_mask')
    return model.compute_loss(**dict(zip(fields,batch['tensors'])),native_source_targets=batch['native_source_targets'],
        native_replay_targets=batch['native_replay_targets'],collect_telemetry=False)


class LiveReplayTests(unittest.TestCase):
    def test_incorrect_teacher_decisions_get_gold_only(self):
        x=torch.tensor([0.,1.],requires_grad=True);gold=torch.tensor([True,False]);wrong=torch.tensor([-2.,2.])
        self.assertEqual(float(source_preservation(x,wrong,gold)),0.)
        torch.testing.assert_close(protected_decision(x,wrong,gold),-x.log_softmax(0)[0])
        gate=x[0];torch.testing.assert_close(protected_gate(gate,torch.tensor(-10.)),F.softplus(-gate))
        logits=x[None];labels=torch.tensor([0]);mask=torch.ones(1)
        torch.testing.assert_close(protected_tokens(logits,wrong[None],labels,mask),F.cross_entropy(logits,labels))

    def test_balancing_prevents_long_content_dominating_boundary(self):
        a=balanced_output_ce(torch.tensor([[2.,10.]]),torch.tensor([[4,2]]),torch.ones(1,2),0,[2])
        b=balanced_output_ce(torch.tensor([[2.,2.,2.,10.],[4.,6.,0.,0.]]),torch.tensor([[4,4,4,2],[4,2,0,0]]),torch.ones(2,4),0,[2])
        self.assertEqual(float(a),6.);self.assertEqual(float(b),5.5)

    def test_cached_learned_teacher_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'frozen'):frozen_tensor(torch.ones(2,requires_grad=True),torch.ones(2),(2,))

    def test_live_gradients_reach_core_and_all_decisions_without_changing_inference(self):
        helper,model,t,batch=fixture()
        with torch.no_grad():_,plain=helper.compute(model,batch)
        loss,out=compute(model,batch)
        torch.testing.assert_close(out.logits,plain.logits,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        for name in ('torus_core','router','shared_signature_bank'):
            self.assertGreater(sum(float(p.grad.square().sum()) for p in getattr(model,name).parameters() if p.grad is not None),0.)
        for name in ('endpoint','copy_gate'):
            self.assertGreater(sum(float(p.grad.square().sum()) for p in getattr(model.native_question_memory.span_readout,name).parameters() if p.grad is not None),0.)
        self.assertEqual(int(out.route_stats['native_replay_examples']),1)
        self.assertTrue(all(p.grad is None for p in [batch['native_replay_targets'][0]['packet']['teacher_tokens']]))

    def test_mismatched_packet_history_and_hidden_cache_fail(self):
        _,model,t,batch=fixture()
        for field,value in [('input_ids',[999]),('source_words',[]),('labels',[999]),('positions',[]),('hidden',torch.ones(2))]:
            packet=batch['native_replay_targets'][0]['packet'];old=packet.get(field);packet[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):compute(model,batch)
            if old is None:del packet[field]
            else:packet[field]=old

    def test_each_live_decision_loss_reaches_the_shared_core(self):
        from native_live_replay import live_replay
        from native_training_objectives import native_source_supervision
        helper,model,t,batch=fixture()
        # At construction the endpoint's zero output weights deliberately block
        # input gradients until that head learns. Test the active path separately.
        with torch.no_grad():
            model.native_question_memory.span_readout.endpoint[-1].weight.normal_(std=.01)
            model.native_question_memory.span_readout.copy_gate.weight.normal_(std=.01)
        _,out=helper.compute(model,batch)
        x,y,*_=batch['tensors'];mask=batch['tensors'][-1]
        _,parts=live_replay(model,out,x,y,mask,batch['native_source_targets'],batch['native_replay_targets'])
        terms={k:parts[k] for k in ('extent','gate','tokens')}
        terms['source']=native_source_supervision(out.native_qa_readouts,batch['native_source_targets'])
        parameters=tuple(model.torus_core.parameters())
        for name,term in terms.items():
            gradients=torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
            self.assertGreater(sum(float(g.square().sum()) for g in gradients if g is not None),0.,name)

    def test_zero_initialized_heads_block_input_gradients_but_can_learn(self):
        from native_live_replay import live_replay
        helper,model,t,batch=fixture();_,out=helper.compute(model,batch)
        x,y,*_=batch['tensors'];mask=batch['tensors'][-1]
        _,parts=live_replay(model,out,x,y,mask,batch['native_source_targets'],batch['native_replay_targets'])
        core=tuple(model.torus_core.parameters())
        for name,module in [('extent',model.native_question_memory.span_readout.endpoint),('gate',model.native_question_memory.span_readout.copy_gate)]:
            head=tuple(module.parameters())
            gradients=torch.autograd.grad(parts[name],core+head,allow_unused=True,retain_graph=True)
            self.assertEqual(sum(float(g.square().sum()) for g in gradients[:len(core)] if g is not None),0.,name)
            self.assertGreater(sum(float(g.square().sum()) for g in gradients[len(core):] if g is not None),0.,name)

    def test_duplicate_gold_payloads_are_one_acceptable_mass(self):
        logits=torch.tensor([0.,0.,0.],requires_grad=True);gold=torch.tensor([True,True,False])
        teacher=torch.tensor([2.,1.,0.])
        value=protected_decision(logits,teacher,gold);value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0]),0.)

    def test_missing_gold_bank_is_reported_and_auxiliary_extents_do_not_inject(self):
        _,model,t,batch=fixture()
        # Force shortlist to miss the reviewed start without changing labels or auxiliary descriptors.
        original=model.native_question_memory.span_readout.candidates
        def wrong(state,tokenizer,selected):return original(state,tokenizer,[5])
        model.native_question_memory.span_readout.candidates=wrong
        loss,out=compute(model,batch)
        self.assertEqual(int(out.route_stats['native_replay_gold_payload_absent']),1)
        payload=tuple(batch['native_replay_targets'][0]['packet']['labels'])
        self.assertFalse(any(tuple(c['tokens'])==payload for c in out.native_qa_readouts[0][4]['candidates']))
        self.assertTrue(torch.isfinite(loss))

    def test_normal_trainer_accepts_replay_and_records_contract(self):
        from train import train_model,_native_batch_kwargs
        _,model,t,batch=fixture();model.cfg.use_fst=False
        self.assertIn('native_replay_targets',_native_batch_kwargs(batch))
        train_model(model,[batch],torch.device('cpu'),cfg=model.cfg,optimizer_name='adamw',steps=1,lr=.001,progress=False,use_amp=False)
        self.assertEqual(model._prismal_training_state['native_functional_replay']['examples_presented'],1)
        with self.assertRaises(ValueError):_native_batch_kwargs(dict(tensors=batch['tensors'],native_replay_targets=[]))

if __name__=='__main__':unittest.main()
