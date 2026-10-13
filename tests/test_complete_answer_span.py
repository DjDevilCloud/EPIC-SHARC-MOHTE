"""Complete-span ownership, prefix visibility, neutral migration and verification."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from tests import test_source_ownership as ownership
from signature_bank import BoundedIdentityReadout,compose_word_identity,WordSpanState
from source_answer_span import AnswerSpanPlan,CompleteAnswerSpanPredictor
from train import save_checkpoint,load_bundle_from_checkpoint


class CompleteAnswerSpanTests(unittest.TestCase):
    def test_failed_name_router_allows_only_confident_digit_request(self):
        from unified_span_reader import ANSWER_ROLE_FACTS
        t,m,module=self.requested_role_fixture('relation_decisions_v10')
        module.ready.fill_(True);module.general_name_plan=True
        module.general_name_scope=torch.nn.Linear(m.cfg.d_model,1)
        roles=[0.]*len(ANSWER_ROLE_FACTS);roles[ANSWER_ROLE_FACTS.index('digits')]=.999
        proposed=AnswerSpanPlan(tuple(t.encode('December 2021.',add_special_tokens=False)),0,2,.999,requested_properties=tuple(roles))
        with patch.object(module,'unified_task_applicable',return_value=True),patch.object(module,'question_features',return_value=torch.zeros(1,m.cfg.d_model)),patch.object(module,'task_features',return_value=torch.zeros(1,m.cfg.d_model)),patch.object(module.general_scope,'forward',return_value=torch.tensor([20.])),patch.object(module.general_name_scope,'forward',return_value=torch.tensor([20.])),patch.object(module,'packet',return_value={}),patch.object(module,'scores',return_value={}),patch.object(module,'plan_from_scores',return_value=None),patch.object(module,'unified_proposal',return_value=proposed):
            from types import SimpleNamespace
            state=SimpleNamespace(words=SimpleNamespace(question=lambda:()))
            args=(state,t,m.shared_signature_bank,m.construction_embedding)
            self.assertIs(module.propose(*args),proposed)
            proposed.confidence=.98;self.assertIsNone(module.propose(*args))
            proposed.confidence=.999;proposed.requested_properties=tuple([0.]*len(roles));self.assertIsNone(module.propose(*args))
            proposed.requested_properties=tuple(roles);proposed.tokens=tuple(t.encode('John Reed',add_special_tokens=False));self.assertIsNone(module.propose(*args))
            protected_plan=AnswerSpanPlan((),0,1,.999)
            with patch.object(module,'plan_from_scores',return_value=protected_plan):self.assertIs(module.propose(*args),protected_plan)

    def requested_role_fixture(self,version='requested_roles_v6'):
        t,m,_=self.fixture(general=True)
        m.cfg.identity_readout_unified_span_reader=True
        m.cfg.identity_readout_unified_reader_version=version
        m.cfg.identity_readout_answer_span_word_encoder='shared_word_v2'
        head=BoundedIdentityReadout(m.cfg);head.load_state_dict(m.bounded_identity_readout.state_dict(),strict=False)
        m.bounded_identity_readout=head
        return t,m,head.complete_answer_span

    def test_requested_role_metadata_preserves_source_case_continuation(self):
        from unified_span_reader import ANSWER_ROLE_FACTS,ANSWER_EXTENT_FACTS
        t,m,module=self.requested_role_fixture()
        text='Context:\nJohn Reed wrote Blue River.\nQuestion:\nWho wrote Blue River?'
        ids,_,_,state=self.state(t,m,text)
        actual=module.packet(state,t,m.shared_signature_bank,m.construction_embedding,compact=True)
        rebuilt=module.packet(m.bounded_identity_readout.input_state_from_tokens(ids[0].tolist(),t),t,m.shared_signature_bank,m.construction_embedding,compact=True)
        torch.testing.assert_close(actual['reader_answer_facts'],rebuilt['reader_answer_facts'],rtol=0,atol=0)
        bit=len(ANSWER_ROLE_FACTS)+ANSWER_EXTENT_FACTS.index('capital_continuation')
        self.assertEqual(float(actual['reader_answer_facts'][actual['surfaces'].index('John'),bit]),1.)
        self.assertEqual(float(actual['reader_answer_facts'][actual['surfaces'].index('John Reed'),bit]),0.)

    def test_normal_compute_loss_supervises_requested_properties(self):
        for version in ('requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'):
            with self.subTest(version=version):self.check_normal_requested_properties(version)

    def check_normal_requested_properties(self,version):
        from data import _build_window_samples_from_text
        from model import PrismalWaveOutput
        t,m,module=self.requested_role_fixture(version);m.cfg.grounded_qa_span_loss_weight=1.;m.cfg.grounded_qa_full_context=True
        m.eval();m.requires_grad_(False);reader=module.unified_reader;reader.requires_grad_(True)
        sample=_build_window_samples_from_text(t,'<BOI>Context:\nThe budget is $75.\nQuestion:\nWhat is the budget?<EOI><BOO>$75<EOO>',seq_len=16,preserve_grounded_qa_context=True,hierarchy_vector_dtype='float32')[0]
        ids=sample.input_ids[None]
        output=PrismalWaveOutput(logits=torch.zeros(1,ids.size(1),t.vocab_size),input_signature=torch.zeros(1,m.cfg.d_model),output_signature=torch.zeros(1,m.cfg.d_model),path_logits=None,route_stats={})
        with patch.object(m,'forward',return_value=output):loss,result=m.compute_loss(ids,sample.labels[None],loss_mask=sample.loss_mask[None],collect_telemetry=False)
        stats=result.route_stats
        for key in ('grounded_qa_requested_role_loss','grounded_qa_requested_extent_loss'):self.assertGreater(float(stats[key]),0.)
        torch.testing.assert_close(stats['grounded_qa_span_loss'],stats['grounded_qa_start_loss']+stats['grounded_qa_endpoint_loss'])
        loss.backward();self.assertGreater(float(reader.requested_roles.weight.grad.abs().sum()),0.)
        if hasattr(reader,'requested_roles_nonlinear'):self.assertGreater(float(reader.requested_roles_nonlinear[-1].weight.grad.abs().sum()),0.)
        if hasattr(reader,'factored_matching'):
            self.assertGreater(float(reader.factored_matching.owner_gain.grad.abs()+reader.factored_matching.relation_gain.grad.abs()),0.)
        self.assertFalse(bool(reader.ready))
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(m,directory,tokenizer=t)
            loaded,_,cfg=load_bundle_from_checkpoint(Path(directory)/'model.pt',device='cpu',load_training_state=False)
            self.assertEqual(cfg.identity_readout_unified_reader_version,version)
            for key,value in reader.state_dict().items():torch.testing.assert_close(value,loaded.bounded_identity_readout.complete_answer_span.unified_reader.state_dict()[key],rtol=0,atol=0)

    def test_span_plan_carries_predicted_request_properties(self):
        t,m,module=self.requested_role_fixture()
        _,_,_,state=self.state(t,m,'Context:\nThe budget is $75.\nQuestion:\nWhat is the budget?')
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding,compact=True)
        joint=torch.full((1,len(packet['surfaces'])),-100.)
        joint[0,packet['surfaces'].index('$75')]=0.
        logits=torch.linspace(-2.,2.,29)[None]
        with patch.object(module.unified_reader,'forward',return_value=dict(joint=joint,requested_role_logits=logits)):
            plan=module.unified_proposal(state,t,m.shared_signature_bank,m.construction_embedding,.9)
        self.assertEqual(plan.tokens,module.payload(t,'$75'))
        torch.testing.assert_close(torch.tensor(plan.requested_properties),logits[0].sigmoid(),rtol=0,atol=0)

    def test_unified_reader_cannot_override_verified_causal_occurrence(self):
        t,m,_=self.fixture(general=True,endpoints=True,endpoint_router=True,causal=True,unified=True)
        module=m.bounded_identity_readout.complete_answer_span
        with torch.no_grad():
            module.ready.fill_(True);module.unified_reader.ready.fill_(True)
            module.general_scope.classifier[-1].bias.fill_(100.)
            module.endpoint_scope[-1].bias.fill_(100.)
        _,_,_,state=self.state(t,m,'Context:\nThe budget has notes with amount $75. Other costs have amount $123.\nQuestion:\nWhat is the budget?')
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        target=packet['surfaces'].index('$75');owner=packet['ends'][target][0]
        match=torch.zeros(len(packet['starts']));match[owner]=10.
        kind=torch.full_like(match,-100.);kind[owner]=100.
        endpoint=torch.zeros(len(packet['ends']));endpoint[target]=100.
        wrong=AnswerSpanPlan(module.payload(t,'$123'),0,1,.999,answer_kind='TEXT_SPAN',kind_confidence=1.)
        causal=dict(match=match,kind=kind,scope=torch.full_like(match,100.))
        with patch.object(module,'causal_owner_scores',return_value=causal),patch.object(module,'endpoint_property_scores',return_value=endpoint),patch.object(module,'unified_proposal',side_effect=AssertionError('generic reader overrode verified owner')):
            plan=module.propose(state,t,m.shared_signature_bank,m.construction_embedding)
            self.assertEqual(plan.tokens,module.payload(t,'$75'))
        # An uncalibrated owner route must leave generic native QA available.
        causal['scope']=torch.full_like(match,-100.)
        with patch.object(module,'causal_owner_scores',return_value=causal),patch.object(module,'unified_proposal',return_value=wrong):
            self.assertIs(module.propose(state,t,m.shared_signature_bank,m.construction_embedding),wrong)
        # Owner confidence alone cannot protect an uncertain endpoint.
        causal['scope']=torch.full_like(match,100.)
        with patch.object(module,'causal_owner_scores',return_value=causal),patch.object(module,'endpoint_property_scores',return_value=torch.zeros_like(endpoint)),patch.object(module,'unified_proposal',return_value=wrong):
            self.assertIs(module.propose(state,t,m.shared_signature_bank,m.construction_embedding),wrong)

    def test_document_context_reader_checkpoint_roundtrip(self):
        t,m,_=self.fixture(general=True)
        m.cfg.identity_readout_unified_span_reader=True
        m.cfg.identity_readout_unified_reader_version='semantic_context_v5'
        m.cfg.identity_readout_answer_span_word_encoder='shared_word_v2'
        head=BoundedIdentityReadout(m.cfg);head.load_state_dict(m.bounded_identity_readout.state_dict(),strict=False)
        m.bounded_identity_readout=head;reader=head.complete_answer_span.unified_reader
        with torch.no_grad():reader.document_context_gain.fill_(.25);reader.ready.fill_(True)
        from train import save_checkpoint,load_bundle_from_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(m,directory,tokenizer=t)
            loaded,_,cfg=load_bundle_from_checkpoint(Path(directory)/'model.pt',device='cpu',load_training_state=False)
            self.assertEqual(cfg.identity_readout_unified_reader_version,'semantic_context_v5')
            for name,value in reader.state_dict().items():
                torch.testing.assert_close(value,loaded.bounded_identity_readout.complete_answer_span.unified_reader.state_dict()[name],rtol=0,atol=0)

    def test_causal_anchor_survives_eviction_and_prunes_expired_clause(self):
        from types import SimpleNamespace
        lexical=SimpleNamespace(is_lexical=True)
        st=WordSpanState(16,anchor_words=8)
        for token in range(40):
            st.observe(token,lexical);st.boundary()
        self.assertEqual([w for _,w in st.anchors[0]],[(i,) for i in range(8)])
        self.assertEqual(st.records[0][0],24)
        cloned=st.clone();cloned.boundary(clause=True)
        for token in range(40,60):
            cloned.observe(token,lexical);cloned.boundary()
        self.assertNotIn(0,cloned.anchors)
        self.assertIn(0,st.anchors)
        self.assertLessEqual(sum(len(w) for prefix in cloned.anchors.values() for _,w in prefix),cloned.capacity+64)

    def test_causal_anchor_does_not_register_truncated_word(self):
        from types import SimpleNamespace
        st=WordSpanState(4,anchor_words=8)
        for token in range(8):st.observe(token,SimpleNamespace(is_lexical=True))
        st.boundary()
        self.assertIn(0,st.anchor_truncated)
        self.assertNotIn(0,st.anchors)

    def test_production_causal_owner_survives_lexical_eviction(self):
        t,m,_=self.fixture(general=True,endpoints=True,endpoint_router=True,causal=True)
        head=m.bounded_identity_readout;module=head.complete_answer_span
        text='Context:\nThe budget has notes '+('ordinary recorded detail '*30)+'with amount $75.\nQuestion:\nWhat is the budget?'
        packets=[];states=[]
        for capacity in (2048,128):
            head.capacity=capacity
            _,_,_,st=self.state(t,m,text);states.append(st)
            packets.append(module.packet(st,t,m.shared_signature_bank,m.construction_embedding))
        self.assertGreater(states[1].words.records[0][0],0)
        indices=[p['ends'][p['surfaces'].index('$75')][0] for p in packets]
        for key in ('causal_owner_contexts','causal_owner_properties'):
            torch.testing.assert_close(packets[0][key][indices[0]],packets[1][key][indices[1]],atol=0,rtol=0)
        self.assertTrue(bool(packets[1]['causal_owner_available'][indices[1]]))

    def test_causal_owner_neutral_and_value_independent(self):
        t,m,_=self.fixture(general=True,endpoints=True,endpoint_router=True,causal=True)
        module=m.bounded_identity_readout.complete_answer_span
        with torch.no_grad():
            module.general_scope.classifier[-1].bias.fill_(100.)
            module.endpoint_scope[-1].bias.fill_(100.)
        packets=[]
        for value in ('75','123'):
            _,_,_,state=self.state(t,m,f'Context:\nThe budget has notes ordinary detail with amount ${value}.\nQuestion:\nWhat is the budget?')
            p=module.packet(state,t,m.shared_signature_bank,m.construction_embedding);packets.append(p)
            baseline=module.scores(p)
            with patch.object(module,'causal_owner_scores',return_value=dict(match=torch.zeros(len(p['starts'])),kind=torch.zeros(len(p['starts'])),scope=torch.full((len(p['starts']),),-100.))):
                unchanged=module.scores(p)
            for key in ('start','end','joint'):torch.testing.assert_close(baseline[key],unchanged[key],atol=0,rtol=0)
            # First source word cannot see later source words.
            self.assertFalse(bool(p['causal_owner_available'][0]))
        for key in ('causal_owner_contexts','causal_owner_properties'):
            a=packets[0]['ends'][packets[0]['surfaces'].index('$75')][0]
            b=packets[1]['ends'][packets[1]['surfaces'].index('$123')][0]
            torch.testing.assert_close(packets[0][key][a],packets[1][key][b],atol=0,rtol=0)
        torch.testing.assert_close(packets[0]['start_features'][a,-m.cfg.d_model:],packets[1]['start_features'][b,-m.cfg.d_model:],atol=0,rtol=0)
        with torch.no_grad():
            for net in (module.causal_owner_match,module.causal_owner_kind,module.causal_owner_scope):net[-1].weight.fill_(.25)
        request=module.general_request(packets[0]['question'])
        scores=[module.causal_owner_scores(p,request) for p in packets]
        for key in ('match','kind','scope'):torch.testing.assert_close(scores[0][key][a],scores[1][key][b],atol=0,rtol=0)

    def test_confident_causal_owner_selects_supported_start_only(self):
        t,m,_=self.fixture(general=True,endpoints=True,endpoint_router=True,causal=True)
        module=m.bounded_identity_readout.complete_answer_span
        with torch.no_grad():
            module.general_scope.classifier[-1].bias.fill_(100.)
            module.endpoint_scope[-1].bias.fill_(100.)
        _,_,_,state=self.state(t,m,'Context:\nThe budget has notes with amount $75. Other costs have amount $123.\nQuestion:\nWhat is the budget?')
        p=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        a=p['ends'][p['surfaces'].index('$75')][0];b=p['ends'][p['surfaces'].index('$123')][0]
        kind=torch.full((len(p['starts']),),-100.);kind[[a,b]]=100.
        match=torch.zeros_like(kind);match[a]=10.
        before=module.scores(p)
        with patch.object(module,'causal_owner_scores',return_value=dict(match=match,kind=kind,scope=torch.full_like(kind,100.))):
            after=module.scores(p)
        with patch.object(module,'causal_owner_scores',return_value=dict(match=torch.full_like(kind,-100.),kind=kind,scope=torch.full_like(kind,100.))):
            unsupported=module.scores(p)
        torch.testing.assert_close(before['start'],unsupported['start'],atol=0,rtol=0)
        # Kind confidence cannot borrow ownership confidence from another candidate.
        biased=kind.clone();biased[b]=1000.;wrong=match.clone();wrong[b]=-10.
        with patch.object(module,'causal_owner_scores',return_value=dict(match=wrong,kind=biased,scope=torch.full_like(kind,100.))):
            declined=module.scores(p)
        torch.testing.assert_close(before['start'],declined['start'],atol=0,rtol=0)
        self.assertEqual(int(after['start'].argmax()),a)
        self.assertEqual(int(torch.isfinite(after['start']).sum()),2)
        torch.testing.assert_close(after['end'],module.endpoint_property_scores(p,module.general_request(p['question'])),atol=0,rtol=0)
        self.assertFalse(torch.isnan(after['joint']).any())

    def test_compact_reader_uses_explicit_full_question_and_removes_labels(self):
        t,m,_=self.fixture(general=True,endpoints=True,endpoint_router=True,causal=True)
        module=m.bounded_identity_readout.complete_answer_span
        packets=[]
        for label in ('Context:\n','CONTEXT:\t'):
            text=label+'The budget is $75.\nQuestion: Consider the budget. What is the amount?'
            _,_,_,state=self.state(t,m,text)
            p=module.packet(state,t,m.shared_signature_bank,m.construction_embedding,compact=True)
            source=[t.decode(list(ids),clean_text=False,collapse_structure=False).casefold() for ids in p['source_word_ids']]
            query=[t.decode(list(ids),clean_text=False,collapse_structure=False).casefold() for ids in p['question_word_ids']]
            self.assertNotIn('context',source);self.assertNotIn('question',source)
            self.assertNotIn('consider',source);self.assertEqual(query[0],'consider')
            self.assertNotIn('general_contexts',p)
            packets.append(p)
        torch.testing.assert_close(packets[0]['source_words'],packets[1]['source_words'],atol=0,rtol=0)
        torch.testing.assert_close(packets[0]['question'],packets[1]['question'],atol=0,rtol=0)

    def test_unified_schema_route_bypasses_question_family_rejection(self):
        t,m,_=self.fixture(general=True,unified=True)
        head=m.bounded_identity_readout;module=head.complete_answer_span
        ids,features,baseline,st=self.state(t,m,'Context:\nThe budget is $75.\nQuestion:\nWhich amount is the budget?')
        p=module.packet(st,t,m.shared_signature_bank,m.construction_embedding,compact=True)
        winner=p['surfaces'].index('$75');joint=torch.full((1,len(p['ends'])),-torch.inf);joint[0,winner]=0.
        module.ready.fill_(True);module.unified_reader.ready.fill_(True)
        with torch.no_grad():
            head.native_applicability.classifier[-1].bias.fill_(-100.)
            module.general_scope.classifier[-1].bias.fill_(-100.)
        with torch.no_grad(),patch.object(module.unified_reader,'forward',return_value={'joint':joint}):
            _,states=head(torch.zeros_like(baseline),torch.zeros(1,ids.size(1),m.cfg.d_model),ids,features,t,m.shared_signature_bank,lexical_embeddings=m.construction_embedding)
        self.assertEqual(states[0].answer_plan.tokens,module.payload(t,'$75'))
        self.assertTrue(module.unified_task_applicable(st,t))
        _,_,_,ordinary=self.state(t,m,'Which amount is the budget?')
        self.assertFalse(module.unified_task_applicable(ordinary,t))
        with tempfile.TemporaryDirectory() as directory:
            path=save_checkpoint(m,directory,tokenizer=t);loaded,_,cfg=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
            self.assertTrue(cfg.identity_readout_unified_span_reader)
            self.assertTrue(bool(loaded.bounded_identity_readout.complete_answer_span.unified_reader.ready))
            for name,value in module.unified_reader.state_dict().items():torch.testing.assert_close(value,loaded.bounded_identity_readout.complete_answer_span.unified_reader.state_dict()[name],atol=0,rtol=0)

    def test_compact_spans_tokenize_only_selected_answer(self):
        t,m,_=self.fixture(general=True,unified=True)
        module=m.bounded_identity_readout.complete_answer_span
        _,_,_,st=self.state(t,m,'Context:\nThe budget is $75.\nQuestion:\nWhat is the budget?')
        with patch.object(module,'payload',side_effect=AssertionError('candidate re-tokenized')):
            p=module.packet(st,t,m.shared_signature_bank,m.construction_embedding,compact=True)
        self.assertIn('$75',p['surfaces']);self.assertNotIn('tokens',p)

    def test_normal_compute_loss_backpropagates_explicit_span_supervision(self):
        from data import _build_window_samples_from_text
        from model import PrismalWaveOutput
        t,m,_=self.fixture(general=True,unified=True)
        m.cfg.grounded_qa_span_loss_weight=1.;m.cfg.grounded_qa_full_context=True
        m.eval();m.requires_grad_(False);reader=m.bounded_identity_readout.complete_answer_span.unified_reader;reader.requires_grad_(True)
        text='<BOI>Context:\nThe budget is $75.\nQuestion:\nWhat is the budget?<EOI><BOO>$75<EOO>'
        sample=_build_window_samples_from_text(t,text,seq_len=16,preserve_grounded_qa_context=True,hierarchy_vector_dtype='float32')[0]
        ids=sample.input_ids[None];labels=sample.labels[None]
        output=PrismalWaveOutput(logits=torch.zeros(1,ids.size(1),t.vocab_size),input_signature=torch.zeros(1,m.cfg.d_model),output_signature=torch.zeros(1,m.cfg.d_model),path_logits=None,route_stats={})
        with patch.object(m,'forward',return_value=output):loss,result=m.compute_loss(ids,labels,loss_mask=sample.loss_mask[None],collect_telemetry=False)
        self.assertEqual(int(result.route_stats['grounded_qa_records']),1)
        self.assertEqual(int(result.route_stats['grounded_qa_uncovered']),0)
        self.assertFalse(bool(reader.ready))
        loss.backward()
        self.assertGreater(float(reader.query.weight.grad.abs().sum()),0.)
        self.assertGreater(float(reader.end[-1].weight.grad.abs().sum()),0.)

    def test_training_source_metadata_matches_production_prefix(self):
        t,m,_=self.fixture(general=True,unified=True)
        ids,_,_,state=self.state(t,m,'Context:\nThe budget is $75.\nQuestion:\nWhat is the budget?')
        head=m.bounded_identity_readout;module=head.complete_answer_span
        actual=module.packet(state,t,m.shared_signature_bank,m.construction_embedding,compact=True)
        built=head.input_state_from_tokens(ids[0].tolist(),t)
        expected=module.packet(built,t,m.shared_signature_bank,m.construction_embedding,compact=True)
        for key,value in actual.items():
            if torch.is_tensor(value):torch.testing.assert_close(value,expected[key],atol=0,rtol=0)
            else:self.assertEqual(value,expected[key],key)

    def test_endpoint_request_router_preserves_other_tasks(self):
        t,m,_=self.fixture(general=True,endpoints=True,endpoint_router=True)
        module=m.bounded_identity_readout.complete_answer_span
        with torch.no_grad():
            module.general_scope.classifier[-1].bias.fill_(100.)
            module.endpoint_properties[-1].weight.fill_(.2)
        _,_,_,state=self.state(t,m,'Context:\nThe budget is $75.\nQuestion:\nWhat is the budget?')
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        with patch.object(module.endpoint_properties,'forward',side_effect=AssertionError('unfitted router applied correction')):
            baseline=module.scores(packet)
        with torch.no_grad():module.endpoint_scope[-1].bias.fill_(100.)
        updated=module.scores(packet)
        torch.testing.assert_close(baseline['start'],updated['start'],atol=0,rtol=0)
        self.assertFalse(torch.equal(baseline['end'],updated['end']))
    def test_endpoint_properties_change_boundaries_without_changing_starts(self):
        t,m,_=self.fixture(general=True,query_attention=True,endpoints=True)
        module=m.bounded_identity_readout.complete_answer_span
        with torch.no_grad():module.general_scope.classifier[-1].bias.fill_(100.)
        _,_,_,state=self.state(t,m,'Context:\nThe budget is $75.\nQuestion:\nWhat is the budget?')
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        before=module.scores(packet)
        with torch.no_grad():module.endpoint_properties[-1].weight.fill_(.2)
        after=module.scores(packet)
        torch.testing.assert_close(before['start'],after['start'],atol=0,rtol=0)
        self.assertFalse(torch.equal(before['end'],after['end']))
        variants=[packet['surfaces'].index(x) for x in ('$75','75','$75.')]
        properties=packet['general_endpoint_roles'][variants]
        self.assertFalse(torch.equal(properties[0],properties[1]))
        self.assertFalse(torch.equal(properties[0],properties[2]))
        request=module.general_request(packet['question'])
        (-module.endpoint_property_scores(packet,request)[variants[0]]).backward()
        self.assertTrue(torch.isfinite(module.endpoint_properties[-1].weight.grad).all())
        self.assertGreater(float(module.endpoint_properties[-1].weight.grad.abs().sum()),0.)

    def test_endpoint_properties_bypass_verified_person_planner(self):
        t,m,_=self.fixture(general=True,guide=True,endpoints=True)
        module=m.bounded_identity_readout.complete_answer_span
        with torch.no_grad():
            module.general_scope.classifier[-1].bias.fill_(100.)
            module.general_name_scope.classifier[-1].bias.fill_(100.)
        _,_,_,state=self.state(t,m,'Context:\nJ. Reed wrote Blue River.\nQuestion:\nWho wrote Blue River?')
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        with patch.object(module.endpoint_properties,'forward',side_effect=AssertionError('name path called endpoint adapter')):
            scored=module.scores(packet)
        self.assertTrue(scored['source_owned_person'])

    def test_relation_context_neutral_migration_and_remote_gradients(self):
        torch.manual_seed(23)
        old=CompleteAnswerSpanPredictor(8,general_context=True)
        new=CompleteAnswerSpanPredictor(8,general_context=True,general_relation_attention=True)
        new.load_state_dict(old.state_dict(),strict=False)
        source=torch.randn(96,8,requires_grad=True)
        packet={'relation_source':source,'question':torch.randn(9,8)}
        request=new.general_request(packet['question'])
        torch.testing.assert_close(new.relation_context(packet,request),torch.zeros_like(source),atol=0,rtol=0)
        with torch.no_grad():
            new.relation_output.weight.copy_(torch.eye(8))
            new.relation_distance.bias[1:9].fill_(2.)
        original=new.relation_context(packet,request)
        swapped=source.detach().clone();swapped[[2,50]]=swapped[[50,2]]
        changed=new.relation_context(dict(packet,relation_source=swapped),request)
        self.assertFalse(torch.equal(original[90],changed[90]))
        original[90].square().sum().backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(float(source.grad[2].abs().sum()),0.)
        self.assertGreater(float(new.relation_query.weight.grad.abs().sum()),0.)

    def test_relation_zero_residual_preserves_active_span_scores(self):
        t,m,_=self.fixture(general=True,query_attention=True)
        old=m.bounded_identity_readout.complete_answer_span
        with torch.no_grad():old.general_scope.classifier[-1].bias.fill_(100.)
        _,_,_,state=self.state(t,m,'Context:\nThe budget is $75 million.\nQuestion:\nWhat is the budget?')
        packet=old.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        new=CompleteAnswerSpanPredictor(m.cfg.d_model,kind_labels=old.kind_labels,
            general_context=True,general_endpoint=True,general_roles=True,
            general_query_attention=True,general_relation_attention=True,relation_eligibility=True,endpoint_properties=True)
        new.load_state_dict(old.state_dict(),strict=False)
        updated=new.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        a,b=old.scores(packet),new.scores(updated)
        for key in ('start','end','joint','kind'):
            torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)

    def test_relation_scope_uses_boundary_membership_and_supports_supervision(self):
        module=CompleteAnswerSpanPredictor(8,general_context=True,general_relation_attention=True)
        packet={'relation_source':torch.zeros(8,8),'question':torch.randn(5,8),
                'relation_scopes':torch.tensor([0,0,0,0,4,4,4,4])}
        request=module.general_request(packet['question'])
        with torch.no_grad():module.relation_scope.bias[1]=3.
        weights=module.relation_weights(packet,request)
        self.assertGreater(float(weights[2,:4].sum()),.95)
        self.assertGreater(float(weights[6,4:].sum()),.95)
        # Scope labels are categorical equality keys, never numeric distances.
        renumbered=dict(packet,relation_scopes=torch.tensor([91,91,91,91,7,7,7,7]))
        torch.testing.assert_close(weights,module.relation_weights(renumbered,request),atol=0,rtol=0)
        (-weights[2,0].log()-weights[6,4].log()).backward()
        self.assertTrue(torch.isfinite(module.relation_scope.weight.grad).all())
        self.assertGreater(float(module.relation_scope.weight.grad.abs().sum()),0.)

    def fixture(self,conditioned=False,typed=False,owner=False,general=False,guide=False,query_attention=False,relation_attention=False,eligibility=False,endpoints=False,endpoint_router=False,causal=False,unified=False):
        f=ownership.SourceOwnershipTests();helper,t,m=f.fixture();f.enable(m,t)
        old=m.bounded_identity_readout
        old._binding_cfg=copy.copy(m.cfg)
        m.cfg.identity_readout_complete_answer_spans=True
        m.cfg.identity_readout_answer_span_kind_conditioned_endpoints=conditioned
        m.cfg.identity_readout_answer_span_typed_extent_eligibility=typed
        m.cfg.identity_readout_answer_span_owner_context=owner
        m.cfg.identity_readout_answer_span_general_context=general
        m.cfg.identity_readout_answer_span_general_endpoint=general
        m.cfg.identity_readout_answer_span_general_roles=general
        m.cfg.identity_readout_answer_span_general_name_plan=guide
        m.cfg.identity_readout_answer_span_general_query_attention=query_attention
        m.cfg.identity_readout_answer_span_general_relation_attention=relation_attention
        m.cfg.identity_readout_answer_span_relation_eligibility=eligibility
        m.cfg.identity_readout_answer_span_endpoint_properties=endpoints
        m.cfg.identity_readout_answer_span_endpoint_task_router=endpoint_router
        m.cfg.identity_readout_answer_span_causal_owners=causal
        m.cfg.identity_readout_unified_span_reader=unified
        if unified:m.cfg.identity_readout_answer_span_word_encoder='shared_word_v2'
        if general:m.cfg.identity_readout_answer_span_kinds=list(m.cfg.identity_readout_answer_span_kinds)+['TEXT_SPAN']
        head=BoundedIdentityReadout(m.cfg);head.load_state_dict(old.state_dict(),strict=False)
        m.bounded_identity_readout=head;m.prepare_capacity_for_tokenizer(t)
        return t,m,old

    def state(self,t,m,text):
        ids=torch.tensor([t.prepare_generation_hierarchy(text).token_ids])
        features,_=m.shared_signature_bank.runtime_features(ids,t)
        with torch.no_grad():
            logits,state=m.bounded_identity_readout(torch.zeros(1,ids.size(1),t.vocab_size),torch.zeros(1,ids.size(1),m.cfg.d_model),ids,features,t,m.shared_signature_bank,lexical_embeddings=m.construction_embedding)
        return ids,features,logits,state[0]

    def test_general_request_legacy_contract_and_attention_gradients(self):
        torch.manual_seed(19)
        old=CompleteAnswerSpanPredictor(8,general_context=True)
        new=CompleteAnswerSpanPredictor(8,general_context=True,general_query_attention=True)
        new.load_state_dict(old.state_dict(),strict=False)
        question=torch.randn(12,8,requires_grad=True)
        _,last=old.general_encoder(question[None])
        torch.testing.assert_close(old.general_request(question),last[0,0],atol=0,rtol=0)
        states,_=new.general_encoder(question[None])
        expected=torch.nn.functional.layer_norm(states[0]+question,(8,)).mean(0)
        torch.testing.assert_close(new.general_request(question),expected)
        new.general_request(question).square().sum().backward()
        self.assertTrue(torch.isfinite(question.grad).all())
        self.assertGreater(float(question.grad[0].abs().sum()),0.)
        self.assertGreater(float(new.general_query_attention.weight.grad.abs().sum()),0.)

    def test_currency_prefix_and_complete_endpoints_reach_shared_features(self):
        t,m,_=self.fixture();packets=[];vectors=[]
        for prefix in ('','$'):
            _,_,_,state=self.state(t,m,f'Context:\nThe budget is {prefix}75 million.\nQuestion:\nWhat is the budget?')
            packet=m.bounded_identity_readout.complete_answer_span.packet(state,t,m.shared_signature_bank,m.construction_embedding)
            self.assertIn(prefix+'75 million',packet['surfaces']);self.assertIn(prefix+'75',packet['surfaces'])
            target=next(i for i,e in enumerate(state.entries) if t.construction_units[e[0]].render=='7')
            state.word_prior=torch.zeros(len(state.entries));state.word_prior[target]=1.
            vectors.append(BoundedIdentityReadout.source_boundary_property_vector(m.shared_signature_bank,state,t,torch.device('cpu'),include_source_prefix=True))
            packets.append(packet)
        self.assertFalse(torch.equal(vectors[0],vectors[1]))

    def test_generic_text_span_is_not_vetoed_by_unrelated_initial_owner(self):
        t,m,_=self.fixture(general=True);head=m.bounded_identity_readout
        head._binding_cfg.identity_readout_source_start_features='context_roles_v5'
        _,_,_,state=self.state(t,m,'Context:\nR. Finch visited. Sea otters live nearby.\nQuestion:\nWhich animals live nearby?')
        module=head.complete_answer_span;packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        index=packet['surfaces'].index('Sea otters');owner,_,begin,end=packet['ends'][index]
        old_index=packet['surfaces'].index('R. Finch');old_owner=packet['ends'][old_index][0]
        state.source_initial_prior=torch.zeros(len(state.entries));state.source_initial_prior[packet['starts'][old_owner]]=.999
        plan=AnswerSpanPlan(packet['tokens'][index],begin,end,.999,lexical_start=packet['starts'][owner],answer_kind='TEXT_SPAN',kind_confidence=.999)
        self.assertTrue(head.verify_answer_plan_start(state,plan,t))
        # An uncertain kind retains the previous ownership safeguard.
        plan.kind_confidence=.5
        self.assertFalse(head.verify_answer_plan_start(state,plan,t))
        plan.kind_confidence=.999;plan.answer_kind='PERSON'
        self.assertFalse(head.verify_answer_plan_start(state,plan,t))

    def test_generic_numeric_span_is_not_vetoed_by_unrelated_name_prior(self):
        t,m,_=self.fixture();head=m.bounded_identity_readout;module=head.complete_answer_span
        head._binding_cfg.identity_readout_source_start_features='context_roles_v5'
        _,_,_,state=self.state(t,m,'Context:\nA Reed wrote the report. The width is 75 inches.\nQuestion:\nWhat is the width?')
        p=module.packet(state,t,m.shared_signature_bank,m.construction_embedding);index=p['surfaces'].index('75 inches');owner=p['ends'][index][0]
        prior=next(i for i,e in enumerate(state.entries) if t.construction_units[e[0]].render.lower()=='a')
        state.source_initial_prior=torch.zeros(len(state.entries));state.source_initial_prior[prior]=1.
        plan=AnswerSpanPlan(p['tokens'][index],p['ends'][index][2],p['ends'][index][3],.999,lexical_start=p['starts'][owner],answer_kind='TEXT_SPAN',kind_confidence=.999)
        self.assertTrue(head.verify_answer_plan_start(state,plan,t))
        plan.answer_kind='PERSON'
        self.assertFalse(head.verify_answer_plan_start(state,plan,t))

    def test_neutral_migration_and_streamed_source_positions(self):
        t,m,old=self.fixture();ids,features,logits,state=self.state(t,m,'Context:\nAlice paid $75 million.\nQuestion:\nHow much did Alice pay?')
        with torch.no_grad():
            before,_=old(torch.zeros_like(logits),torch.zeros(1,ids.size(1),m.cfg.d_model),ids,features,t,m.shared_signature_bank,lexical_embeddings=m.construction_embedding)
            torch.testing.assert_close(before,logits,atol=0,rtol=0)
            parts=[];states=None
            for i in range(ids.size(1)):
                out,states=m.bounded_identity_readout(torch.zeros(1,1,t.vocab_size),torch.zeros(1,1,m.cfg.d_model),ids[:,i:i+1],features[:,i:i+1],t,m.shared_signature_bank,states,lexical_embeddings=m.construction_embedding)
                parts.append(out)
            torch.testing.assert_close(logits,torch.cat(parts,1),atol=0,rtol=0)
            self.assertEqual(state.source_surface,states[0].source_surface)
            self.assertEqual(state.source_positions,states[0].source_positions)
            packet=m.bounded_identity_readout.complete_answer_span.packet(state,t,m.shared_signature_bank,m.construction_embedding)
            scored=m.bounded_identity_readout.complete_answer_span.scores(packet)
            self.assertAlmostEqual(float(scored['joint'].exp().sum()),1.,places=5)

    def test_plan_requires_verified_progress_and_clone_does_not_mutate_parent(self):
        plan=AnswerSpanPlan((2,3,4),5,7,.99);cloned=copy.copy(plan)
        cloned.observe(2);self.assertEqual(cloned.next_token,3);self.assertEqual(plan.next_token,2)
        cloned.observe(8);self.assertIsNone(cloned.next_token)
        for token in (2,3,4):plan.observe(token)
        self.assertIsNone(plan.next_token)

    def test_payload_preserves_currency_case_and_punctuation(self):
        t,_,_=self.fixture()
        for text in ('$75 million',"5.5'",'A04673','Mary Ann Smith','Section (b)'):
            tokens=CompleteAnswerSpanPredictor.payload(t,text)
            self.assertEqual(t.decode(tokens,clean_text=False,collapse_structure=False).strip(),text)

    def test_sentence_capitalization_preserves_internal_proper_name_case(self):
        t,m,_=self.fixture()
        _,_,_,state=self.state(t,m,'Context:\nLocated on the South Polar cap.\nQuestion:\nWhat is the location?')
        packet=m.bounded_identity_readout.complete_answer_span.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        self.assertIn('The South Polar cap',packet['surfaces'])
        self.assertIn('the South Polar cap',packet['surfaces'])

    def test_general_context_observes_following_relation_but_excludes_question(self):
        t,m,_=self.fixture();legacy=m.bounded_identity_readout.complete_answer_span
        module=CompleteAnswerSpanPredictor(m.cfg.d_model,kind_labels=list(legacy.kind_labels)+['TEXT_SPAN'],general_context=True)
        packets=[]
        for relation,question in [('writer','Who is the writer?'),('editor','Who is the writer?'),('writer','Who is the editor?')]:
            _,_,_,state=self.state(t,m,f'Context:\nAlice is the {relation}.\nQuestion:\n{question}')
            packets.append(module.packet(state,t,m.shared_signature_bank,m.construction_embedding))
        index=packets[0]['ends'][packets[0]['surfaces'].index('Alice')][0]
        self.assertFalse(torch.equal(packets[0]['general_contexts'][index],packets[1]['general_contexts'][index]))
        torch.testing.assert_close(packets[0]['general_contexts'],packets[2]['general_contexts'],atol=0,rtol=0)

    def test_general_case_chain_roles_distinguish_full_start_suffix_and_sentence_boundary(self):
        t,m,_=self.fixture(general=True);module=m.bounded_identity_readout.complete_answer_span
        _,_,_,state=self.state(t,m,'Context:\nAlice Ann Reed is the writer. Bob Vale is the editor.\nQuestion:\nWho is the writer?')
        p=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        def role(text):return p['general_case_chains'][p['ends'][p['surfaces'].index(text)][0]]
        self.assertEqual(role('Alice'),(True,False,True))
        self.assertEqual(role('Ann'),(True,True,True))
        self.assertEqual(role('Reed'),(True,True,False))
        self.assertEqual(role('Bob'),(True,False,True))
        full=p['surfaces'].index('Alice Ann Reed');short=p['surfaces'].index('Ann Reed')
        self.assertFalse(torch.equal(p['general_endpoint_roles'][full],p['general_endpoint_roles'][short]))
        _,_,_,other=self.state(t,m,'Context:\nAlice Ann Reed is the writer. Bob Vale is the editor.\nQuestion:\nWho is the editor?')
        changed=module.packet(other,t,m.shared_signature_bank,m.construction_embedding)
        torch.testing.assert_close(p['general_endpoint_roles'],changed['general_endpoint_roles'],atol=0,rtol=0)

    def test_name_hypotheses_mask_values_and_preserve_first_source_case(self):
        t,m,_=self.fixture(general=True,guide=True);module=m.bounded_identity_readout.complete_answer_span
        with torch.no_grad():
            module.general_scope.classifier[-1].bias.fill_(8.)
            module.general_name_scope.classifier[-1].bias.fill_(8.)
        packets=[]
        for name in ('Alice Ann Reed','Bob Vale Gold'):
            _,_,_,state=self.state(t,m,f'{name} is the writer.\nQuestion:\nWho is the writer?')
            packets.append(module.packet(state,t,m.shared_signature_bank,m.construction_embedding))
            i=packets[-1]['surfaces'].index(name);owner=packets[-1]['ends'][i][0]
            self.assertTrue(packets[-1]['general_case_chains'][owner][0])
            self.assertTrue(bool(packets[-1]['name_eligibility'][i]))
        a=packets[0]['surfaces'].index('Alice Ann Reed');b=packets[1]['surfaces'].index('Bob Vale Gold')
        torch.testing.assert_close(packets[0]['name_contexts'][a],packets[1]['name_contexts'][b],atol=0,rtol=0)
        for text in ('Ann Reed','Alice Ann','writer'):
            i=packets[0]['surfaces'].index(text)
            if text!='writer':self.assertFalse(bool(packets[0]['name_eligibility'][i]),text)
        scores=module.scores(packets[0]);self.assertTrue(scores['source_owned_person'])
        self.assertAlmostEqual(float(scores['joint'].exp().sum()),1.,places=5)
        (-scores['joint'][a]).backward()
        for name,parameter in module.named_parameters():
            if name.startswith('general_name_') and parameter.grad is not None:self.assertTrue(torch.isfinite(parameter.grad).all(),name)
        _,_,_,tab_state=self.state(t,m,'Context:\tAlice Ann Reed is the writer.\nQuestion:\nWho is the writer?')
        tab=module.packet(tab_state,t,m.shared_signature_bank,m.construction_embedding)
        self.assertTrue(bool(tab['name_eligibility'][tab['surfaces'].index('Alice Ann Reed')]))
        self.assertFalse(bool(tab['name_eligibility'][tab['surfaces'].index('Context:\tAlice Ann Reed')]))

    def test_general_task_route_can_pass_an_untrained_old_genre_gate(self):
        t,m,_=self.fixture(general=True);head=m.bounded_identity_readout;module=head.complete_answer_span
        module.ready.fill_(True)
        with torch.no_grad():
            head.native_applicability.classifier[-1].weight.zero_();head.native_applicability.classifier[-1].bias.fill_(-100.)
            module.general_scope.classifier[-1].bias.fill_(8.)
        with patch.object(module,'propose',return_value=None) as propose:
            self.state(t,m,'Context:\nAlice is the writer.\nQuestion:\nWho is the writer?')
        propose.assert_called_once()

    def test_source_owned_person_retains_physical_continuation_verification(self):
        t,m,_=self.fixture();head=m.bounded_identity_readout;module=head.complete_answer_span
        head._binding_cfg.identity_readout_source_start_features='context_roles_v5'
        _,_,_,state=self.state(t,m,'Context:\nBelquorin J. Finch is the writer.\nQuestion:\nWho is the writer?')
        p=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        prior=next(i for i,e in enumerate(state.entries) if t.construction_units[e[0]].render.lower()=='j')
        state.source_initial_prior=torch.zeros(len(state.entries));state.source_initial_prior[prior]=1.
        def plan(text):
            index=p['surfaces'].index(text);owner=p['ends'][index][0]
            return AnswerSpanPlan(p['tokens'][index],p['ends'][index][2],p['ends'][index][3],.999,
                                  lexical_start=p['starts'][owner],answer_kind='PERSON',kind_confidence=.999,source_owned_person=True)
        self.assertTrue(head.verify_answer_plan_start(state,plan('Belquorin J. Finch'),t))
        self.assertFalse(head.verify_answer_plan_start(state,plan('Belquorin J.'),t))
        legacy=plan('Belquorin J. Finch');legacy.source_owned_person=False
        self.assertFalse(head.verify_answer_plan_start(state,legacy,t))

    def test_inactive_general_context_preserves_old_scores_and_active_branch_trains(self):
        t,m,_=self.fixture(conditioned=True);legacy=m.bounded_identity_readout.complete_answer_span
        module=CompleteAnswerSpanPredictor(m.cfg.d_model,kind_labels=list(legacy.kind_labels)+['TEXT_SPAN'],kind_conditioned_endpoints=True,general_context=True,general_endpoint=True,general_roles=True)
        module.load_state_dict({k:v for k,v in legacy.state_dict().items() if not k.startswith('answer_kind.')},strict=False)
        with torch.no_grad():
            module.answer_kind.weight[:-1].copy_(legacy.answer_kind.weight);module.answer_kind.bias[:-1].copy_(legacy.answer_kind.bias)
        _,_,_,state=self.state(t,m,'Context:\nAlice is the writer. Bob is the editor.\nQuestion:\nWho is the writer?')
        old=legacy.packet(state,t,m.shared_signature_bank,m.construction_embedding);new=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        a=legacy.scores(old);b=module.scores(new)
        for key in ('start','end','joint'):torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        with patch.object(module.general_scope,'forward',return_value=torch.tensor([6.])):
            scores=module.scores(new);self.assertAlmostEqual(float(scores['extent_confidence']),float(torch.tensor(6.).sigmoid()),places=6)
            (-scores['joint'][0]).backward()
        self.assertGreater(float(module.general_start_projection.grad.abs().sum()),0.)
        self.assertGreater(float(module.general_end_projection.grad.abs().sum()),0.)
        self.assertGreater(float(module.general_endpoint[-1].weight.grad.abs().sum()),0.)

    def test_query_occurrences_excluded_and_auxiliary_heads_have_gradients(self):
        t,m,_=self.fixture();_,_,_,state=self.state(t,m,'Context:\nThe budget is $75 million.\nQuestion:\nDoes the budget equal 123 million?')
        module=m.bounded_identity_readout.complete_answer_span;module.feature_version='surface_v2'
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        self.assertFalse(any('123' in text for text in packet['surfaces']))
        scores=module.scores(packet);gold=packet['surfaces'].index('$75 million')
        loss=-scores['joint'][gold]+scores['kind'].square().mean()+scores['applicability'].square()
        loss.backward()
        for name,parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad,name);self.assertTrue(torch.isfinite(parameter.grad).all(),name)

    def test_shared_word_composition_preserves_order_and_reuses_legacy_question_vectors(self):
        t,m,_=self.fixture();head=m.bounded_identity_readout;module=head.complete_answer_span
        _,_,_,state=self.state(t,m,'Context:\nThe budget is 75 million.\nQuestion:\nWhat is the budget?')
        words=[tuple(i for i in t.encode(text) if t.construction_units[i].is_lexical) for text in ('75','57')]
        means=[];ordered=[]
        with torch.no_grad():
            for word in words:
                ids=torch.tensor(word);means.append((m.construction_embedding(ids)+m.shared_signature_bank.encode('token',ids)).mean(0))
                ordered.append(compose_word_identity(word,t,m.shared_signature_bank,m.construction_embedding))
            torch.testing.assert_close(means[0],means[1],atol=0,rtol=0)
            self.assertFalse(torch.equal(ordered[0],ordered[1]))
            module.word_encoder='shared_word_v2'
            packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
            question=module.question_features(state.words.question(),m.shared_signature_bank,m.construction_embedding,t)
            torch.testing.assert_close(question,packet['question'],atol=0,rtol=0)
            pooled=torch.nn.functional.layer_norm(question.mean(0),(m.cfg.d_model,))
            torch.testing.assert_close(pooled,state.binding_cache[1][0],atol=0,rtol=0)

    def test_confident_initial_owner_rejects_other_words_and_shorter_aliases(self):
        t,m,_=self.fixture();head=m.bounded_identity_readout;head._binding_cfg.identity_readout_source_start_features='context_roles_v5'
        _,_,_,state=self.state(t,m,'Context:\nP. Belquorin Talia Red Blue Nera Reed Vale\nP. Belquorin Talia\nQuestion:\nWho is the author?')
        starts={}
        for i,entry in enumerate(state.entries):starts.setdefault(entry[3],i)
        records={serial:word for serial,_,word in state.words.records}
        initial=[starts[serial] for serial,word in records.items() if t.decode(word,clean_text=False,collapse_structure=False)=='p']
        self.assertEqual(len(initial),2)
        tail=next(starts[serial] for serial,word in records.items() if t.decode(word,clean_text=False,collapse_structure=False)=='vale')
        state.source_initial_prior=torch.zeros(len(state.entries));state.source_initial_prior[initial[0]]=1.
        def proposal(index):return AnswerSpanPlan((2,),0,1,.99,lexical_start=index)
        self.assertTrue(head.verify_answer_plan_start(state,proposal(initial[0]),t))
        self.assertFalse(head.verify_answer_plan_start(state,proposal(initial[1]),t))
        self.assertFalse(head.verify_answer_plan_start(state,proposal(tail),t))
        state.source_initial_prior.fill_(1./len(state.entries))
        self.assertTrue(head.verify_answer_plan_start(state,proposal(tail),t))
        state.source_initial_prior.zero_();state.source_initial_prior[initial[0]]=1.
        partial=AnswerSpanPlan(CompleteAnswerSpanPredictor.payload(t,'P. Belquorin Talia'),0,1,.99,
                               lexical_start=initial[0],answer_kind='PERSON',kind_confidence=1.)
        self.assertFalse(head.verify_answer_plan_start(state,partial,t))
        partial.tokens=CompleteAnswerSpanPredictor.payload(t,'P. Belquorin')
        self.assertFalse(head.verify_answer_plan_start(state,partial,t))
        partial.answer_kind='FIRST_NAME'
        self.assertTrue(head.verify_answer_plan_start(state,partial,t))

    def test_complete_value_endpoint_cannot_split_numeric_source_construct(self):
        t,m,_=self.fixture();head=m.bounded_identity_readout
        _,_,_,state=self.state(t,m,'Context:\nThe reference is MCL - Section 942.6381.\nQuestion:\nWhat legal section is cited?')
        surface=dict(state.source_surface)
        dot=next(pos for pos,token in state.source_surface if t.construction_units[token].render=='.' and
                 pos+1 in surface and t.construction_units[surface[pos+1]].kind=='digit')
        for endpoint in (dot,dot+1):
            plan=AnswerSpanPlan((2,),0,endpoint,.99,answer_kind='LEGAL_CODE',kind_confidence=1.)
            self.assertFalse(head.verify_answer_plan_start(state,plan,t))
            plan.answer_kind='YEAR';state.source_initial_prior=None
            self.assertTrue(head.verify_answer_plan_start(state,plan,t))

    def test_kind_conditioning_is_neutral_then_changes_only_endpoint_scores(self):
        t,m,_=self.fixture();module=m.bounded_identity_readout.complete_answer_span
        _,_,_,state=self.state(t,m,'Context:\nAlice paid $75 million.\nQuestion:\nWhat amount did Alice pay?')
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        conditioned=CompleteAnswerSpanPredictor(m.cfg.d_model,kind_conditioned_endpoints=True)
        conditioned.load_state_dict(module.state_dict(),strict=False)
        expanded=conditioned.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        before=module.scores(packet);neutral=conditioned.scores(expanded)
        for key in ('start','end','joint','kind','applicability'):
            torch.testing.assert_close(before[key],neutral[key],atol=0,rtol=0)
        with torch.no_grad():conditioned.kind_extent_projection.copy_(torch.eye(m.cfg.d_model))
        changed=conditioned.scores(expanded)
        torch.testing.assert_close(before['start'],changed['start'],atol=0,rtol=0)
        self.assertFalse(torch.equal(before['end'],changed['end']))
        conditioned.zero_grad();(-changed['joint'][0]).backward()
        self.assertGreater(float(conditioned.answer_kind.weight.grad.abs().sum()),0.)
        self.assertGreater(float(conditioned.kind_extent_projection.grad.abs().sum()),0.)

    def test_full_person_preserves_initial_chain_and_capitalized_continuation(self):
        t,m,_=self.fixture();head=m.bounded_identity_readout
        head._binding_cfg.identity_readout_source_start_features='context_roles_v5'
        _,_,_,state=self.state(t,m,'Context:\nV.Q. Orvenquess Blue Caldranivex, one author.\nQuestion:\nWho is the author?')
        module=head.complete_answer_span;packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        start=next(i for i,e in enumerate(state.entries) if t.construction_units[e[0]].render=='v')
        state.source_initial_prior=torch.zeros(len(state.entries));state.source_initial_prior[start]=.997
        def plan(text,kind='PERSON'):
            index=packet['surfaces'].index(text);owner,_,begin,end=packet['ends'][index]
            return AnswerSpanPlan(packet['tokens'][index],begin,end,.99,lexical_start=packet['starts'][owner],answer_kind=kind,kind_confidence=1.)
        for text in ('V','V.','V.Q.','V.Q. Orvenquess Blue'):
            self.assertFalse(head.verify_answer_plan_start(state,plan(text),t),text)
        self.assertTrue(head.verify_answer_plan_start(state,plan('V.Q. Orvenquess Blue Caldranivex'),t))
        self.assertTrue(head.verify_answer_plan_start(state,plan('V','FIRST_NAME'),t))

    def test_verified_noninitial_full_person_start_cannot_move_into_name(self):
        t,m,_=self.fixture();head=m.bounded_identity_readout
        head._binding_cfg.identity_readout_source_start_features='context_roles_v5'
        _,_,_,state=self.state(t,m,'Context:\nVelquessan Reed Blue Nera Talia Vale, one author.\nQuestion:\nWho is the author?')
        module=head.complete_answer_span;packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        index=packet['surfaces'].index('Velquessan Reed Blue Nera Talia Vale');owner=packet['ends'][index][0]
        state.source_initial_prior=torch.zeros(len(state.entries));state.source_initial_prior[packet['starts'][owner]]=.997
        index=packet['surfaces'].index('Talia Vale');owner,_,begin,end=packet['ends'][index]
        candidate=AnswerSpanPlan(packet['tokens'][index],begin,end,.99,lexical_start=packet['starts'][owner],answer_kind='PERSON',kind_confidence=1.)
        self.assertFalse(head.verify_answer_plan_start(state,candidate,t))

    def test_typed_year_eligibility_rejects_day_prefix_and_preserves_identity_choices(self):
        t,m,_=self.fixture(typed=True);module=m.bounded_identity_readout.complete_answer_span
        _,_,_,state=self.state(t,m,'Context:\nReleased November 19, 2034. Met June 18, 2023.\nQuestion:\nGive the release year.')
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        year=module.kind_labels.index('YEAR');logits=torch.full((len(module.kind_labels),),-100.);logits[year]=100.
        with patch.object(module.answer_kind,'forward',return_value=logits):scores=module.scores(packet)
        for i,text in enumerate(packet['surfaces']):
            if text in ('2034','2023'):self.assertTrue(torch.isfinite(scores['joint'][i]),text)
            else:self.assertTrue(torch.isneginf(scores['joint'][i]),text)
        self.assertAlmostEqual(float(scores['joint'].exp().sum()),1.,places=5)
        # An uncertain type must not prune candidates or alter their scores.
        with patch.object(module.answer_kind,'forward',return_value=torch.zeros_like(logits)):
            uncertain=module.scores(packet);module.typed_extent_eligibility=False;legacy=module.scores(packet)
        torch.testing.assert_close(uncertain['joint'],legacy['joint'],atol=0,rtol=0)

    def test_owner_prefix_sees_entity_swap_and_resets_after_numeric_sentence(self):
        t,m,_=self.fixture(owner=True);module=m.bounded_identity_readout.complete_answer_span
        packets=[]
        for main,pipe in [('6.75','7.25'),('7.25','6.75')]:
            _,_,_,state=self.state(t,m,f"Context:\nThe water main has a depth of {main}'. The service pipe has a depth of {pipe}'.\nQuestion:\nGive the integer part of the water-main depth.")
            packets.append(module.packet(state,t,m.shared_signature_bank,m.construction_embedding))
        owner_indices=[p['ends'][p['surfaces'].index('6')][0] for p in packets]
        torch.testing.assert_close(packets[0]['start_features'][owner_indices[0]],packets[1]['start_features'][owner_indices[1]],atol=0,rtol=0)
        self.assertFalse(torch.equal(packets[0]['owner_prefixes'][owner_indices[0]],packets[1]['owner_prefixes'][owner_indices[1]]))
        # Neither the later sentence nor the question can enter the first value's prefix.
        _,_,_,state=self.state(t,m,"Context:\nThe water main has a depth of 6.75'. Completely different words 7.25.\nQuestion:\nGive the integer part of the water-main depth.")
        changed=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        index=changed['ends'][changed['surfaces'].index('6')][0]
        torch.testing.assert_close(packets[0]['owner_prefixes'][owner_indices[0]],changed['owner_prefixes'][index],atol=0,rtol=0)
        decimal=packets[0]['ends'][packets[0]['surfaces'].index('75')][0]
        self.assertGreater(int(packets[0]['owner_lengths'][decimal]),int(packets[0]['owner_lengths'][owner_indices[0]]))
        self.assertLessEqual(int(packets[1]['owner_lengths'][owner_indices[1]]),8)

    def test_zero_owner_adapter_migration_preserves_scoring(self):
        t,m,_=self.fixture();legacy=m.bounded_identity_readout.complete_answer_span
        _,_,_,state=self.state(t,m,'Context:\nReport released November 19, 2034. Meeting June 18, 2023.\nQuestion:\nGive the report release year.')
        owner=CompleteAnswerSpanPredictor(m.cfg.d_model,owner_context=True);owner.load_state_dict(legacy.state_dict(),strict=False)
        packets=[x.packet(state,t,m.shared_signature_bank,m.construction_embedding) for x in (legacy,owner)]
        logits=torch.full((len(owner.kind_labels),),-100.);logits[owner.kind_labels.index('YEAR')]=100.
        with patch.object(legacy.answer_kind,'forward',return_value=logits),patch.object(owner.answer_kind,'forward',return_value=logits):
            old=legacy.scores(packets[0]);new=owner.scores(packets[1])
            for key in ('start','end','joint'):torch.testing.assert_close(old[key],new[key],atol=0,rtol=0)
            with torch.no_grad():owner.owner_projection.copy_(torch.eye(m.cfg.d_model))
            scores=owner.scores(packets[1]);(-scores['joint'][0]).backward()
        self.assertGreater(float(owner.owner_encoder.weight_ih_l0.grad.abs().sum()),0.)

    def test_typed_integer_part_excludes_fractional_suffix_and_declines_missing_support(self):
        t,m,_=self.fixture(typed=True);module=m.bounded_identity_readout.complete_answer_span
        module.kind_labels=tuple(list(module.kind_labels)+['INTEGER_PART'])
        module.answer_kind=torch.nn.Linear(m.cfg.d_model,len(module.kind_labels))
        _,_,_,state=self.state(t,m,"Context:\nDepth 6.75'. Alternative 3.2'.\nQuestion:\nGive the integer part.")
        packet=module.packet(state,t,m.shared_signature_bank,m.construction_embedding)
        validity=packet['extent_eligibility']['INTEGER_PART']
        for text in ('6','3'):self.assertTrue(any(bool(validity[i]) for i,v in enumerate(packet['surfaces']) if v==text),text)
        for text in ('75','2','6.75',"6.75'"):
            self.assertFalse(any(bool(validity[i]) for i,v in enumerate(packet['surfaces']) if v==text),text)
        _,_,_,signed_state=self.state(t,m,"Context:\nDepth -6.75'. Alternative +3.2'.\nQuestion:\nGive the integer part.")
        signed_packet=module.packet(signed_state,t,m.shared_signature_bank,m.construction_embedding)
        for i,text in enumerate(signed_packet['surfaces']):
            if text in ('-6','+3'):self.assertTrue(bool(signed_packet['extent_eligibility']['INTEGER_PART'][i]),text)
            if text in ('6','3'):self.assertFalse(bool(signed_packet['extent_eligibility']['INTEGER_PART'][i]),text)
        # A confident calendar-year request without a year must safely decline.
        year=module.kind_labels.index('YEAR');logits=torch.full((len(module.kind_labels),),-100.);logits[year]=100.
        module.ready.fill_(True)
        with patch.object(module.answer_kind,'forward',return_value=logits):
            scores=module.scores(packet);self.assertFalse(scores['extent_supported'])
            self.assertFalse(torch.isnan(scores['joint']).any())
            self.assertIsNone(module.propose(state,t,m.shared_signature_bank,m.construction_embedding))

    def test_active_plan_streams_reloads_and_task_rejection_is_neutral(self):
        t,m,old=self.fixture(conditioned=True,typed=True,owner=True,general=True,guide=True,query_attention=True,relation_attention=True,eligibility=True,endpoints=True,endpoint_router=True,causal=True);head=m.bounded_identity_readout;module=head.complete_answer_span
        ids,features,baseline,state=self.state(t,m,'Context:\nThe budget is $75 million.\nQuestion:\nWhat is the budget?')
        module.ready.fill_(True)
        with torch.no_grad():
            head.native_applicability.classifier[-1].weight.zero_()
            head.native_applicability.classifier[-1].bias.fill_(100.)
            module.general_scope.classifier[-1].bias.fill_(6.)
        # Exercise a confident plan independent of random fixture calibration.
        original=module.propose
        def active(state,tokenizer,bank,embeddings):
            result=original(state,tokenizer,bank,embeddings,threshold=0.)
            result.confidence=.99
            return result
        with torch.no_grad(),patch.object(module,'propose',side_effect=active):
            full,states=head(torch.zeros_like(baseline),torch.zeros(1,ids.size(1),m.cfg.d_model),ids,features,t,m.shared_signature_bank,lexical_embeddings=m.construction_embedding)
            self.assertIsNotNone(states[0].answer_plan)
            parts=[];stream=None
            for i in range(ids.size(1)):
                out,stream=head(torch.zeros(1,1,t.vocab_size),torch.zeros(1,1,m.cfg.d_model),ids[:,i:i+1],features[:,i:i+1],t,m.shared_signature_bank,stream,lexical_embeddings=m.construction_embedding)
                parts.append(out)
            torch.testing.assert_close(full,torch.cat(parts,1),atol=0,rtol=0)
            self.assertEqual(states[0].answer_plan,stream[0].answer_plan)
        with tempfile.TemporaryDirectory() as directory:
            path=save_checkpoint(m,directory,tokenizer=t);loaded,_,cfg=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
            self.assertTrue(cfg.identity_readout_complete_answer_spans)
            self.assertTrue(cfg.identity_readout_answer_span_kind_conditioned_endpoints)
            self.assertTrue(cfg.identity_readout_answer_span_typed_extent_eligibility)
            self.assertTrue(cfg.identity_readout_answer_span_owner_context)
            self.assertTrue(cfg.identity_readout_answer_span_general_context)
            self.assertTrue(cfg.identity_readout_answer_span_general_roles)
            self.assertTrue(cfg.identity_readout_answer_span_general_name_plan)
            self.assertTrue(cfg.identity_readout_answer_span_general_query_attention)
            self.assertTrue(cfg.identity_readout_answer_span_general_relation_attention)
            self.assertTrue(cfg.identity_readout_answer_span_relation_eligibility)
            self.assertTrue(cfg.identity_readout_answer_span_endpoint_properties)
            self.assertTrue(cfg.identity_readout_answer_span_endpoint_task_router)
            self.assertTrue(cfg.identity_readout_answer_span_causal_owners)
            self.assertTrue(bool(loaded.bounded_identity_readout.complete_answer_span.ready))
            for name,value in module.state_dict().items():torch.testing.assert_close(value,loaded.bounded_identity_readout.complete_answer_span.state_dict()[name],atol=0,rtol=0)
        # A rejected QA task cannot invoke even a very confident span predictor.
        with torch.no_grad():
            head.native_applicability.classifier[-1].bias.fill_(-100.)
            module.general_scope.classifier[-1].bias.fill_(-100.)
        with torch.no_grad(),patch.object(module,'propose') as propose:
            _,states=head(torch.zeros_like(baseline),torch.zeros(1,ids.size(1),m.cfg.d_model),ids,features,t,m.shared_signature_bank,lexical_embeddings=m.construction_embedding)
        propose.assert_not_called();self.assertIsNone(states[0].answer_plan)


if __name__=='__main__':unittest.main()
