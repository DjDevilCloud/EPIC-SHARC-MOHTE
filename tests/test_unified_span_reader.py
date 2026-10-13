import unittest
import torch
from unified_span_reader import UnifiedSpanReader,RelationalSpanReader,SemanticWordComposer,collate_reader_packets,answer_span_facts,ANSWER_ROLE_FACTS,ANSWER_EXTENT_FACTS

class UnifiedReaderTests(unittest.TestCase):
    def test_relation_decision_migration_preserves_scores_and_enables_heads(self):
        from unified_span_reader import create_unified_reader
        torch.manual_seed(41);old=create_unified_reader(8,'structural_owner_v9');new=create_unified_reader(8,'relation_decisions_v10')
        missing=new.load_state_dict(old.state_dict(),strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(all(k.startswith(('relation_properties','factored_matching.relation_applicability')) for k in missing.missing_keys))
        p=self.packet(5,3);p.update(scope_start=torch.zeros(5,dtype=torch.long),scope_end=torch.full((5,),5),query_properties=torch.rand(3,8),answer_facts=torch.rand(len(p['owners']),29))
        # Shared semantic units remain real ordered inputs, independent of labels.
        p['source_components']=[torch.tensor([[i+1,i+2]]) for i in range(5)]
        p['question_components']=[torch.tensor([[i+1,i+2]]) for i in range(3)]
        batch=collate_reader_packets([p]);a,b=old(batch),new(batch)
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        auxiliary=b['relation_requested_role_logits'].sum()+b['factored_applicability_logits'].sum()
        auxiliary.backward()
        self.assertGreater(float(new.relation_properties[-1].weight.grad.abs().sum()),0.)
        self.assertGreater(float(new.factored_matching.relation_applicability[-1].weight.grad.abs().sum()),0.)

    def test_query_properties_preserve_case_for_fragmented_unseen_words(self):
        from data import PrismalTokenizer
        from signature_bank import WordSpanState
        from unified_span_reader import query_word_properties
        t=PrismalTokenizer();tokens=t.encode('Where did Nora FINCH die? 2010',add_special_tokens=False)
        words=WordSpanState(1024);positions={}
        for pos,token in enumerate(tokens):
            unit=t.construction_units[token]
            if unit.is_lexical:positions[words.serial]=pos
            words.observe(token,unit)
        words.boundary();records=[(serial,ids) for serial,_,ids in words.records]
        props=query_word_properties(records,dict(enumerate(tokens)),positions,t)
        keys=[t.decode(list(ids),clean_text=False,collapse_structure=False).casefold() for _,ids in records]
        self.assertGreater(len(records[keys.index('nora')][1]),1)
        self.assertEqual(props[keys.index('nora'),0].item(),1.)
        self.assertEqual(props[keys.index('finch'),1].item(),1.)
        self.assertEqual(props[keys.index('die'),0].item(),0.)
        self.assertEqual(props[keys.index('2010'),3].item(),1.)
        self.assertEqual(props[keys.index('where'),7].item(),1.)

    def test_factored_matching_neutral_migration_and_value_exclusion(self):
        from unified_span_reader import FactoredMatching
        torch.manual_seed(31);reader=RelationalSpanReader(8,16);reader.scoped=True
        p=self.packet(8,4);p['scope_start']=torch.zeros(8,dtype=torch.long);p['scope_end']=torch.full((8,),8)
        p['exact'][0,1]=1.;p['exact'][4,2]=1.
        batch=collate_reader_packets([p]);old=reader(batch)
        reader.factored_matching=FactoredMatching(16);new=reader(batch)
        for key in old:torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        target=((p['owners']==2)&(p['finishes']==3)).nonzero()[0,0]
        changed=dict(p,source=p['source'].clone());changed['source'][2:4]+=20.
        altered=reader(collate_reader_packets([changed]))
        torch.testing.assert_close(new['factored_relation_logits'][0,target],altered['factored_relation_logits'][0,target],atol=1e-6,rtol=1e-6)
        (-new['joint'][0,target]).backward()
        self.assertGreater(float(reader.factored_matching.owner_gain.grad.abs()+reader.factored_matching.relation_gain.grad.abs()),0.)

    def test_factored_owner_matching_excludes_value_and_other_clause(self):
        from unified_span_reader import FactoredMatching
        torch.manual_seed(32);reader=RelationalSpanReader(8,16);reader.scoped=True;reader.factored_matching=FactoredMatching(16)
        p=self.packet(8,3);p['scope_start']=torch.tensor([0]*4+[4]*4);p['scope_end']=torch.tensor([4]*4+[8]*4)
        p['exact'][0,1]=1.
        a=reader(collate_reader_packets([p]))
        target=((p['owners']==6)&(p['finishes']==6)).nonzero()[0,0]
        self.assertLess(float(a['owner_match_logits'][0,target]),-1.)
        p['exact'][6,1]=1.;b=reader(collate_reader_packets([p]))
        torch.testing.assert_close(a['owner_match_logits'][0,target],b['owner_match_logits'][0,target],rtol=0,atol=0)

    def test_frozen_candidate_fit_matches_normal_scores_and_gradients(self):
        torch.manual_seed(20);reader=RelationalSpanReader(8,16)
        reader.requested_roles=torch.nn.Linear(16,29)
        reader.requested_roles_gain=torch.nn.Parameter(torch.tensor(.4))
        reader.requires_grad_(False);reader.score.requires_grad_(True);reader.requested_roles_gain.requires_grad_(True)
        p=self.packet(5,3);p['answer_facts']=torch.rand(len(p['owners']),29)
        batch=collate_reader_packets([p]);captured=[]
        hook=reader.score.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        normal=reader(batch);hook.remove()
        (-normal['joint'][0,0]).backward()
        gradients={name:parameter.grad.clone() for name,parameter in reader.named_parameters() if parameter.requires_grad}
        reader.zero_grad()
        cached=reader.score_candidates(captured[0],batch['owners'],batch['candidate_mask'],len(p['source']),normal['role_compatibility'].detach())
        for key in cached:torch.testing.assert_close(normal[key],cached[key],rtol=0,atol=0)
        (-cached['joint'][0,0]).backward()
        for name,parameter in reader.named_parameters():
            if parameter.requires_grad:torch.testing.assert_close(gradients[name],parameter.grad,rtol=0,atol=0)

    def test_nonlinear_request_decoder_neutral_migration_and_learning(self):
        from unified_span_reader import create_unified_reader
        torch.manual_seed(19)
        old=create_unified_reader(8,'requested_roles_v6')
        with torch.no_grad():
            old.requested_roles.weight.normal_(0.,.1)
            old.requested_roles.bias.normal_(0.,.1)
            old.requested_roles_gain.fill_(.8)
        new=create_unified_reader(8,'nonlinear_roles_v7')
        missing=new.load_state_dict(old.state_dict(),strict=False)
        self.assertEqual(set(missing.missing_keys),{'requested_roles_nonlinear.'+name for name in ('0.weight','0.bias','2.weight','2.bias')})
        self.assertEqual(missing.unexpected_keys,[])
        request=torch.randn(4,old.hidden)
        expected=old.requested_roles(request)
        actual=new.requested_roles(request)+new.requested_roles_nonlinear(request)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        target=torch.ones_like(actual)
        optimizer=torch.optim.SGD(new.requested_roles_nonlinear.parameters(),lr=.1)
        torch.nn.functional.binary_cross_entropy_with_logits(actual,target).backward()
        self.assertGreater(float(new.requested_roles_nonlinear[-1].weight.grad.abs().sum()),0.)
        optimizer.step();optimizer.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(new.requested_roles_nonlinear(request),target).backward()
        self.assertGreater(float(new.requested_roles_nonlinear[0].weight.grad.abs().sum()),0.)

    def test_role_facts_preserve_format_and_share_across_values(self):
        a=answer_span_facts('30-40%',2);b=answer_span_facts('50-60%',2)
        self.assertEqual(a,b)
        self.assertNotEqual(a,answer_span_facts('30-40',2))
        self.assertNotEqual(answer_span_facts('11.4',2),answer_span_facts('11.4)',2))
        self.assertNotEqual(answer_span_facts('roughly 12 inches',3),answer_span_facts('Roughly 12 inches',3))
        self.assertNotEqual(answer_span_facts('J. Reed',2),answer_span_facts('J. Reed',2,capital_continuation=True))
        self.assertEqual(len(a),len(ANSWER_ROLE_FACTS)+len(ANSWER_EXTENT_FACTS))

    def test_requested_roles_zero_gate_and_explicit_supervision(self):
        torch.manual_seed(14);old=RelationalSpanReader(8,16);new=RelationalSpanReader(8,16)
        new.load_state_dict(old.state_dict())
        size=len(ANSWER_ROLE_FACTS)+len(ANSWER_EXTENT_FACTS)
        new.requested_roles=torch.nn.Linear(16,size);torch.nn.init.zeros_(new.requested_roles.weight);torch.nn.init.zeros_(new.requested_roles.bias)
        new.requested_roles_gain=torch.nn.Parameter(torch.zeros(()))
        p=self.packet(5,3);p['answer_facts']=torch.zeros(len(p['owners']),size);p['answer_facts'][0,ANSWER_ROLE_FACTS.index('percent')]=1.
        batch=collate_reader_packets([p]);a,b=old(batch),new(batch)
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.nn.functional.binary_cross_entropy_with_logits(b['requested_role_logits'],p['answer_facts'][0][None]).backward()
        self.assertGreater(float(new.requested_roles.weight.grad.abs().sum()),0.)
        with torch.no_grad():
            new.requested_roles.bias[ANSWER_ROLE_FACTS.index('percent')]=5.;new.requested_roles_gain.fill_(1.)
        changed=new(batch)
        self.assertGreater(float(changed['role_compatibility'][0,0]-changed['role_compatibility'][0,1]),4.)
        self.assertGreater(float(changed['joint'][0,0]),float(b['joint'][0,0]))

    def test_document_context_separates_identical_cells_without_reading_value(self):
        torch.manual_seed(12);reader=RelationalSpanReader(8,16);reader.scoped=True
        reader.document_context_gain=torch.nn.Parameter(torch.zeros(()))
        p=self.packet(7,4);p['scope_start']=torch.arange(7);p['scope_end']=torch.arange(7)+1
        p['properties'].zero_();p['anchors'].zero_();p['end_properties'].zero_();p['exact'].zero_()
        first=((p['owners']==2)&(p['finishes']==2)).nonzero()[0,0]
        second=((p['owners']==5)&(p['finishes']==5)).nonzero()[0,0]
        batch=collate_reader_packets([p]);neutral=reader(batch)
        torch.testing.assert_close(neutral['relation_logits'][0,first],neutral['relation_logits'][0,second],rtol=0,atol=0)
        (-neutral['joint'][0,first]).backward()
        self.assertGreater(float(reader.document_context_gain.grad.abs()),0.)
        with torch.no_grad():reader.document_context_gain.fill_(1.)
        contextual=reader(batch)
        self.assertGreater(float((contextual['relation_logits'][0,first]-contextual['relation_logits'][0,second]).abs()),1e-6)
        changed=dict(p,source=p['source'].clone());changed['source'][2]+=100.
        altered=reader(collate_reader_packets([changed]))
        torch.testing.assert_close(contextual['relation_logits'][0,first],altered['relation_logits'][0,first],rtol=1e-6,atol=1e-6)

    def test_document_context_zero_gate_preserves_scoped_reader(self):
        torch.manual_seed(13);old=RelationalSpanReader(8,16);old.scoped=True
        new=RelationalSpanReader(8,16);new.scoped=True;new.document_context_gain=torch.nn.Parameter(torch.zeros(()));new.load_state_dict(old.state_dict(),strict=False)
        p=self.packet(7,4);other=self.packet(3,2)
        for item in (p,other):
            item['scope_start']=torch.zeros(len(item['source']),dtype=torch.long)
            item['scope_end']=torch.full((len(item['source']),),len(item['source']),dtype=torch.long)
        batch=collate_reader_packets([p,other])
        a,b=old(batch),new(batch)
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def packet(self,n,q,width=8):
        owners=torch.tensor([i for i in range(n) for j in range(i,min(n,i+3))])
        finishes=torch.tensor([j for i in range(n) for j in range(i,min(n,i+3))])
        return dict(source=torch.randn(n,width),question=torch.randn(q,width),properties=torch.randn(n,width),anchors=torch.randn(n,width),owners=owners,finishes=finishes,end_properties=torch.randn(len(owners),width),exact=torch.zeros(n,q))

    def test_batched_padding_normalization_and_gradients(self):
        torch.manual_seed(7);reader=UnifiedSpanReader(8,16)
        a,b=self.packet(7,4),self.packet(3,2)
        batch=collate_reader_packets([a,b]);out=reader(batch)
        torch.testing.assert_close(out['joint'].exp().sum(-1),torch.ones(2))
        single=reader(collate_reader_packets([b]))
        torch.testing.assert_close(single['joint'][0],out['joint'][1,:len(b['owners'])],atol=1e-6,rtol=1e-6)
        (-out['joint'][:,0].mean()).backward()
        for p in reader.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(reader.query.weight.grad.abs().sum()),0.)
        self.assertFalse(bool(reader.ready))

    def test_relational_selection_excludes_candidate_value_identity(self):
        torch.manual_seed(9);reader=RelationalSpanReader(8,16);p=self.packet(7,4)
        owner=2;finish=3;target=((p['owners']==owner)&(p['finishes']==finish)).nonzero()[0,0]
        a=reader(collate_reader_packets([p]));changed=dict(p,source=p['source'].clone(),exact=p['exact'].clone())
        changed['source'][owner:finish+1]+=100.
        changed['exact'][owner:finish+1]=1.
        b=reader(collate_reader_packets([changed]))
        torch.testing.assert_close(a['relation_logits'][0,target],b['relation_logits'][0,target],atol=1e-6,rtol=1e-6)
        (-a['joint'][0,target]).backward()
        for param in reader.parameters():
            if param.grad is not None:self.assertTrue(torch.isfinite(param.grad).all())
        paired=reader(collate_reader_packets([p,self.packet(3,2)]))
        torch.testing.assert_close(paired['joint'].exp().sum(-1),torch.ones(2))

    def test_scoped_relations_do_not_borrow_another_sentences_owner(self):
        torch.manual_seed(10);reader=RelationalSpanReader(8,16);reader.scoped=True
        p=self.packet(7,4);p['scope_start']=torch.tensor([0,0,0,3,3,3,3]);p['scope_end']=torch.tensor([3,3,3,7,7,7,7])
        target=((p['owners']==4)&(p['finishes']==5)).nonzero()[0,0]
        a=reader(collate_reader_packets([p]));changed=dict(p,source=p['source'].clone(),exact=p['exact'].clone())
        changed['source'][:3]+=100.;changed['exact'][:3]=1.
        b=reader(collate_reader_packets([changed]))
        torch.testing.assert_close(a['relation_logits'][0,target],b['relation_logits'][0,target],atol=1e-6,rtol=1e-6)
        cross=(p['owners']==2)&(p['finishes']==3)
        self.assertFalse(torch.isfinite(a['joint'][0,cross]).any())
        (-a['joint'][0,target]).backward()
        for param in reader.parameters():
            if param.grad is not None:self.assertTrue(torch.isfinite(param.grad).all())

    def test_semantic_word_bank_preserves_order_and_learns(self):
        torch.manual_seed(11);composer=SemanticWordComposer(8)
        units=torch.tensor([[[1,2],[3,4],[5,6]],[[5,6],[3,4],[1,2]]])
        out=composer(units,torch.tensor([3,3]))
        self.assertFalse(torch.equal(out[0],out[1]))
        (out[0,0]-out[1,0]).backward()
        self.assertTrue(torch.isfinite(composer.embedding.weight.grad).all())
        self.assertGreater(float(composer.embedding.weight.grad.abs().sum()),0.)

    def test_question_changes_source_and_extent_scores(self):
        torch.manual_seed(8);reader=UnifiedSpanReader(8,16);p=self.packet(5,3)
        a=reader(collate_reader_packets([p]));changed=dict(p,question=-p['question'])
        b=reader(collate_reader_packets([changed]))
        self.assertFalse(torch.equal(a['start'],b['start']))
        self.assertFalse(torch.equal(a['end'],b['end']))

if __name__=='__main__':unittest.main()
