"""Complete-span ownership, prefix visibility, neutral migration and verification."""
import copy
import unittest
import tempfile
from unittest.mock import patch
import torch
from tests import test_source_ownership as ownership
from signature_bank import BoundedIdentityReadout,compose_word_identity
from source_answer_span import AnswerSpanPlan,CompleteAnswerSpanPredictor
from train import save_checkpoint,load_bundle_from_checkpoint


class CompleteAnswerSpanTests(unittest.TestCase):
    def fixture(self,conditioned=False,typed=False,owner=False,general=False,guide=False):
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
        t,m,old=self.fixture(conditioned=True,typed=True,owner=True,general=True,guide=True);head=m.bounded_identity_readout;module=head.complete_answer_span
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
