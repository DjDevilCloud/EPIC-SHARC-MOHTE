import copy,unittest
import torch
from torch import nn
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory

class NativeRequestMatchingTests(unittest.TestCase):
    def setUp(self):self.rng=torch.random.get_rng_state()
    def tearDown(self):torch.random.set_rng_state(self.rng)
    def test_source_roles_neutral_incremental_and_trainable(self):
        from native_source_role import NativeSourceRole
        t,a,ids,embedding=self.fixture();b=copy.deepcopy(a)
        b.source_roles=NativeSourceRole(8)
        hidden=torch.randn(1,ids.shape[1],8);a.eval();b.eval()
        x,_,records=a(ids,hidden,embedding,t,return_attention=True)
        y,_,other=b(ids,hidden,embedding,t,return_attention=True)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(records[-1][-1],other[-1][-1],rtol=0,atol=0)
        (-other[-1][-1][2].log()).backward()
        self.assertGreater(float(b.source_roles.source[-1].weight.grad.abs().sum()),0.)
        nn.init.normal_(b.source_roles.source[-1].weight,std=.05)
        with torch.no_grad():
            _,states=b(ids,hidden,embedding,t);previous=None
            for i in range(ids.shape[1]):
                _,previous=b(ids[:,i:i+1],hidden[:,i:i+1],embedding,t,previous)
            torch.testing.assert_close(states[0].cached_packet[2],previous[0].cached_packet[2],rtol=1e-5,atol=1e-6)
    def test_request_alignment_neutral_incremental_and_trainable(self):
        from native_relation_alignment import NativeRelationAlignment
        t,a,ids,embedding=self.fixture();b=copy.deepcopy(a)
        b.relation_alignment=NativeRelationAlignment(8)
        hidden=torch.randn(1,ids.shape[1],8)
        a.eval();b.eval()
        x,_,records=a(ids,hidden,embedding,t,return_attention=True)
        y,_,other=b(ids,hidden,embedding,t,return_attention=True)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(records[-1][-1],other[-1][-1],rtol=0,atol=0)
        (-other[-1][-1][2].log()).backward()
        self.assertGreater(float(b.relation_alignment.output[-1].weight.grad.abs().sum()),0.)
        nn.init.normal_(b.relation_alignment.output[-1].weight,std=.05)
        with torch.no_grad():
            _,states=b(ids,hidden,embedding,t);previous=None
            for i in range(ids.shape[1]):
                _,previous=b(ids[:,i:i+1],hidden[:,i:i+1],embedding,t,previous)
            torch.testing.assert_close(states[0].cached_packet[2],previous[0].cached_packet[2],rtol=1e-5,atol=1e-6)
    def fixture(self,matcher='combined_v1'):
        t=PrismalTokenizer();memory=NativeQuestionMemory(8,span_readout=True,lexical_cues=True,future_context=3,anchor_words=2,matcher=matcher)
        ids=torch.tensor([t.prepare_generation_hierarchy('Context:\nThe zorquessin price is $75.50. The lumavex price is $28.25.\nQuestion:\nWhat is the price of zorquessin?').token_ids])
        return t,memory,ids,nn.Embedding(t.vocab_size,8)
    def test_batched_encoders_match_outputs_attention_and_gradients(self):
        t,a,ids,embedding=self.fixture();b=copy.deepcopy(a);b.batch_encoders=True
        nn.init.normal_(a.output[-1].weight,std=.05);b.load_state_dict(a.state_dict())
        hidden=torch.randn(1,ids.shape[1],8)
        x,_,records=a(ids,hidden,embedding,t,return_attention=True)
        y,_,other=b(ids,hidden,embedding,t,return_attention=True)
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(records[-1][-1],other[-1][-1],rtol=1e-5,atol=1e-6)
        (x.square().mean()-records[-1][-1][2].log()).backward()
        (y.square().mean()-other[-1][-1][2].log()).backward()
        for name,p in a.named_parameters():
            q=dict(b.named_parameters())[name]
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-5)
    def test_separate_matching_is_causal_incremental_and_receives_role_gradients(self):
        t,memory,ids,embedding=self.fixture('owner_relation_v1');memory.batch_encoders=True
        hidden=torch.randn(1,ids.shape[1],8)
        _,states,records=memory(ids,hidden,embedding,t,return_attention=True)
        (-records[-1][-1][2].log()).backward()
        self.assertGreater(float(memory.request_roles.weight.grad.abs().sum()),0.)
        memory.eval();previous=None
        with torch.no_grad():
            for i in range(ids.shape[1]):
                _,previous=memory(ids[:,i:i+1],hidden[:,i:i+1],embedding,t,previous)
            torch.testing.assert_close(previous[0].cached_packet[2],records[-1][-1],rtol=1e-5,atol=1e-6)
            self.assertEqual(previous[0].source,states[0].source)

    def test_document_context_neutral_incremental_eviction_and_owner_swap(self):
        t,a,_,embedding=self.fixture();b=copy.deepcopy(a)
        b.document_key=nn.Linear(8,8,bias=False);nn.init.zeros_(b.document_key.weight)
        def run(memory,owners,incremental=False):
            p=t.prepare_generation_hierarchy(f'Context:\n{owners[0]} is the subject. He was born in Oakfield. {owners[1]} is the subject. He was born in Stonehaven.\nQuestion:\nWhere was Quasmerin born?')
            ids=torch.tensor([p.token_ids]);hidden=torch.zeros(1,len(p.token_ids),8)
            if incremental:
                states=None
                for i in range(ids.shape[1]):_,states=memory(ids[:,i:i+1],hidden[:,i:i+1],embedding,t,states)
            else:_,states=memory(ids,hidden,embedding,t)
            state=states[0];indices={t.decode(list(w),clean_text=False,collapse_structure=False).casefold():i for i,(w,_) in enumerate(state.source)}
            scores=state.cached_packet[3]['start_log_probabilities']
            return state,scores[indices['oakfield']]-scores[indices['stonehaven']]
        a.eval();b.eval()
        with torch.no_grad():
            original,_=run(a,('Quasmerin','Lumavex'));neutral,_=run(b,('Quasmerin','Lumavex'))
            torch.testing.assert_close(original.cached_packet[2],neutral.cached_packet[2],rtol=0,atol=0)
            nn.init.eye_(b.document_key.weight)
            full,x=run(b,('Quasmerin','Lumavex'));incremental,y=run(b,('Quasmerin','Lumavex'),True)
            torch.testing.assert_close(full.cached_packet[2],incremental.cached_packet[2])
            swapped,z=run(b,('Lumavex','Quasmerin'))
            self.assertGreater(abs(float(x-z)),1e-7)
            self.assertEqual(len(full.source),len(full.document_contexts))
            # Eviction retains the causal summary for the retained tail.
            b.capacity=16
            bounded,_=run(b,('Quasmerin','Lumavex'))
            torch.testing.assert_close(bounded.document_hidden,full.document_hidden)
            torch.testing.assert_close(torch.stack(bounded.document_contexts),torch.stack(full.document_contexts[-16:]))
            self.assertLessEqual(len(bounded.document_contexts),16)
            self.assertEqual(full.document_pending,[])
            # Later owner replacement cannot change an earlier occurrence.
            b.capacity=128
            later,_=run(b,('Quasmerin','Tavindor'))
            oak=next(i for i,(w,_) in enumerate(full.source) if t.decode(list(w),clean_text=False,collapse_structure=False).casefold()=='oakfield')
            torch.testing.assert_close(full.document_contexts[oak],later.document_contexts[oak],rtol=0,atol=0)
            # Continuing a branch leaves the caller's state immutable.
            old_contexts=list(full.document_contexts);old_summary=full.document_hidden.clone()
            next_ids=torch.tensor([[t.eos_id]])
            b(next_ids,torch.zeros(1,1,8),embedding,t,[full])
            self.assertEqual(len(full.document_contexts),len(old_contexts))
            torch.testing.assert_close(full.document_hidden,old_summary,rtol=0,atol=0)
            reset_ids=torch.tensor([[t.special_tokens['<BOI>']]])
            _,reset=b(reset_ids,torch.zeros(1,1,8),embedding,t,[full])
            self.assertIsNone(reset[0].document_hidden)
            self.assertEqual(reset[0].document_contexts,[])

    def test_document_context_batching_and_projection_gradients(self):
        t,a,ids,embedding=self.fixture('owner_relation_v1')
        a.document_key=nn.Linear(8,8,bias=False);nn.init.zeros_(a.document_key.weight)
        b=copy.deepcopy(a);b.batch_encoders=True
        hidden=torch.zeros(1,ids.shape[1],8)
        _,_,left=a(ids,hidden,embedding,t,return_attention=True)
        _,_,right=b(ids,hidden,embedding,t,return_attention=True)
        torch.testing.assert_close(left[-1][-1],right[-1][-1],rtol=1e-5,atol=1e-6)
        (-left[-1][-1][2].log()).backward();(-right[-1][-1][2].log()).backward()
        self.assertGreater(float(a.document_key.weight.grad.abs().sum()),0.)
        torch.testing.assert_close(a.document_key.weight.grad,b.document_key.weight.grad,rtol=1e-4,atol=1e-5)

    def test_new_matcher_and_batched_encoder_reload(self):
        import tempfile
        from pathlib import Path
        from tests.test_runtime_signatures import RuntimeSignatureTests
        from train import save_checkpoint,load_bundle_from_checkpoint
        _,t,model=RuntimeSignatureTests().setup_model()
        model.cfg.use_native_question_memory=True;model.cfg.native_question_memory_span_readout=True
        model.cfg.native_question_memory_matcher='owner_relation_v1';model.cfg.native_question_memory_batch_encoders=True
        model.cfg.native_question_memory_document_context=True
        model.cfg.native_question_memory_antecedent_memory=True
        model.cfg.native_question_memory_antecedent_query='ordered_v1'
        model.cfg.freeze_registry_observations=True;model.registry.observation_updates_enabled=False
        model.native_question_memory=NativeQuestionMemory(model.cfg.d_model,span_readout=True,matcher='owner_relation_v1',batch_encoders=True,document_context=True,antecedent_memory=True,antecedent_query='ordered_v1')
        ids=torch.tensor([t.prepare_generation_hierarchy('Context:\nThe apple is red.\nQuestion:\nWhat color is the apple?').token_ids])
        with torch.no_grad(),tempfile.TemporaryDirectory() as directory:
            expected=model(ids).logits;save_checkpoint(model,directory,tokenizer=t)
            loaded,_,cfg=load_bundle_from_checkpoint(Path(directory)/'model.pt',device='cpu',load_training_state=False);loaded.eval()
            self.assertEqual(cfg.native_question_memory_matcher,'owner_relation_v1')
            self.assertTrue(loaded.native_question_memory.batch_encoders)
            self.assertEqual(loaded.native_question_memory.antecedent_memory.identity_query,'ordered_v1')
            torch.testing.assert_close(loaded(ids).logits,expected,rtol=1e-5,atol=1e-6)

if __name__=='__main__':unittest.main()
