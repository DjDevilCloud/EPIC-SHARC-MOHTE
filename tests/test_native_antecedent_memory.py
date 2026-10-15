import copy, unittest
import torch
from torch import nn
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory

class NativeAntecedentTests(unittest.TestCase):
 def setUp(self):self.rng=torch.random.get_rng_state();torch.manual_seed(47)
 def tearDown(self):torch.random.set_rng_state(self.rng)
 def fixture(self,capacity=128):
  t=PrismalTokenizer();m=NativeQuestionMemory(8,capacity=capacity,span_readout=True,antecedent_memory=True,batch_encoders=True);e=nn.Embedding(t.vocab_size,8);m.eval();return t,m,e
 def run_memory(self,t,m,e,context,incremental=False):
  p=t.prepare_generation_hierarchy('Context:\n'+context+'\nQuestion:\nWhere was Quasmerin born?');ids=torch.tensor([p.token_ids]);h=torch.zeros(1,len(p.token_ids),8)
  if incremental:
   states=None
   for i in range(ids.shape[1]):_,states=m(ids[:,i:i+1],h[:,i:i+1],e,t,states)
  else:_,states=m(ids,h,e,t)
  return states[0]
 def test_neutral_incremental_and_bounded_eviction(self):
  t,m,e=self.fixture();legacy=copy.deepcopy(m);legacy.antecedent_memory=None
  text='Quasmerin is the subject. He was born in Oakfield. Lumavex is the subject. He was born in Stonehaven.'
  with torch.no_grad():
   state=self.run_memory(t,m,e,text);old=self.run_memory(t,legacy,e,text);inc=self.run_memory(t,m,e,text,True)
   torch.testing.assert_close(state.cached_packet[2],old.cached_packet[2],rtol=0,atol=0)
   torch.testing.assert_close(state.cached_packet[2],inc.cached_packet[2])
   self.assertEqual(state.antecedents,inc.antecedents)
   m.capacity=8;small=self.run_memory(t,m,e,text)
   self.assertEqual(small.antecedents,state.antecedents)
   self.assertEqual(small.source_ordinals,state.source_ordinals[-8:])
   self.assertTrue(any(t.decode(list(event.words[0]),clean_text=False,collapse_structure=False).casefold()=='quasmerin' for event in small.antecedents))
   m.antecedent_memory.capacity=2;bounded=self.run_memory(t,m,e,text)
   self.assertEqual(len(bounded.antecedents),2)
 def test_access_uses_only_causal_prefixes_and_receives_gradients(self):
  t,m,e=self.fixture();m.train()
  p=t.prepare_generation_hierarchy('Context:\nQuasmerin is the subject. He was born in Oakfield. Lumavex is the subject. He was born in Stonehaven.\nQuestion:\nWhere was Quasmerin born?');ids=torch.tensor([p.token_ids])
  _,states,records=m(ids,torch.zeros(1,len(p.token_ids),8),e,t,return_readout=True)
  plan=records[0][4];access=plan['antecedent'];(-plan['start_log_probabilities'][8]+access['role_logits'].square().sum()+access['role_logits'].sum()).backward()
  self.assertIsNotNone(m.antecedent_memory.scale.grad)
  self.assertGreater(float(m.antecedent_memory.role[-1].weight.grad.abs().sum()),0.)
  # Later owner reassignment cannot affect earlier causal access values.
  m.eval()
  with torch.no_grad():
   first=self.run_memory(t,m,e,'Quasmerin is the subject. He was born in Oakfield. Lumavex is the subject.')
   other=self.run_memory(t,m,e,'Quasmerin is the subject. He was born in Oakfield. Tavindor is the subject.')
   a=first.cached_packet[3]['antecedent'];b=other.cached_packet[3]['antecedent']
   def compatibility(st,packet):
    def vector(w):return m.word_encoder(e(torch.tensor(w))[None])[1][-1,0]
    return m.antecedent_memory.access(st,st.question,vector,m.context_encoder,m.encode_sequences)[0]
   x,y=compatibility(first,a),compatibility(other,b)
   index=next(i for i,(w,_) in enumerate(first.source) if t.decode(list(w),clean_text=False,collapse_structure=False).casefold()=='oakfield')
   torch.testing.assert_close(x[:index+1],y[:index+1],rtol=0,atol=0)

 def test_ordered_identity_distinguishes_reassigned_word_order(self):
  t,m,e=self.fixture();m.antecedent_memory.identity_query='ordered_v1'
  with torch.no_grad():
   p=t.prepare_generation_hierarchy('Context:\nMara Vess is the subject. He was born in Oakfield. Vess Mara is the subject. He was born in Stonehaven.\nQuestion:\nWhere was Mara Vess born?')
   _,states=m(torch.tensor([p.token_ids]),torch.zeros(1,len(p.token_ids),8),e,t)
   packet=states[0].cached_packet[3]['antecedent'];scores={}
   for (words,props),value in zip(packet['scopes'],packet['identity_matches']):
    text=tuple(t.decode(list(w),clean_text=False,collapse_structure=False).casefold() for w in words)
    if len(text)==5 and text[-3:]==('is','the','subject'):scores[text[:2]]=float(value)
   self.assertGreater(scores[('mara','vess')],scores[('vess','mara')])

if __name__=='__main__':unittest.main()
