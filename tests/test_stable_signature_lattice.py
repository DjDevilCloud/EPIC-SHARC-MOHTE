"""Long cache writes must implement decay * cache + update, without scale collapse."""
import unittest
import torch
from config import PrismalWaveConfig
from model import SignatureLatticeAttention,SignatureLatticeState
from quantization import QuantizationConfig

class StableSignatureLatticeTests(unittest.TestCase):
 def layer(self,decay=.85,weight=0.):
  torch.manual_seed(19)
  cfg=PrismalWaveConfig(d_model=4,signature_lattice_dim=4,signature_lattice_buckets=8,signature_lattice_candidates=8,signature_lattice_chunk_len=1,signature_lattice_decay=decay,signature_lattice_weight=weight,hierarchical_precision_enabled=False)
  q=QuantizationConfig(enabled=False,use_torchao_weight_only=False,use_torchao_embedding_weight_only=False)
  return SignatureLatticeAttention(cfg,q)
 def test_long_constant_writes_match_geometric_series_in_reduced_precision(self):
  for decay in (.85,1.,0.,.01):
   for dtype in (torch.float32,torch.bfloat16,torch.float16):
    layer=self.layer(decay);layer.precision_state_dtype=dtype
    h=torch.ones(1,768,4)
    with torch.no_grad():
     _,state,_=layer(h,return_state=True,collect_telemetry=False)
     expected=layer.v_proj(h[:,0])*(768 if decay==1 else (1-decay**768)/(1-decay))
     if decay==1 and dtype!=torch.float32:
      # No-decay accumulation has ordinary low-precision rounding; compare its
      # eager arithmetic instead of requiring an exact real-valued sum.
      write=layer.v_proj(h[:,0]).to(dtype)
      expected=torch.zeros_like(write)
      for _ in range(768):expected=expected+write
      expected=expected.float()
     effective=(state.cache*state.cache_decay_scale).sum(1).float()
    self.assertTrue(torch.isfinite(state.cache).all());self.assertGreater(float(state.cache_decay_scale.min()),0.)
    torch.testing.assert_close(effective,expected,rtol=.04 if dtype!=torch.float32 else 1e-4,atol=.04 if dtype!=torch.float32 else 1e-5)
 def test_rebase_matches_eager_recurrence_outputs_and_gradients(self):
  layer=self.layer(weight=.05);h=torch.randn(1,384,4)*.05
  actual,_,_=layer(h,collect_telemetry=False)
  cache=torch.zeros(1,4);expected=[]
  for i in range(h.size(1)):
   out=h[:,i]+layer.weight*torch.sigmoid(layer.gate(h[:,i]))*layer.out_proj(cache)
   expected.append(out);cache=cache*layer.decay+layer.v_proj(out)
  reference=torch.stack(expected,1)
  torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-5)
  a=torch.autograd.grad(actual.square().mean(),layer.v_proj.weight)[0]
  b=torch.autograd.grad(reference.square().mean(),layer.v_proj.weight)[0]
  torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
 def test_phase_survives_irregular_stream_chunks_and_old_state_coercion(self):
  layer=self.layer(weight=.05);h=torch.randn(1,301,4)*.05
  with torch.no_grad():
   full,full_state,_=layer(h,return_state=True,collect_telemetry=False)
   parts=[];state=None;offset=0
   for length in (7,31,1,100,162):
    out,state,_=layer(h[:,offset:offset+length],state=state,return_state=True,collect_telemetry=False);parts.append(out);offset+=length
   torch.testing.assert_close(full,torch.cat(parts,1),rtol=1e-4,atol=1e-5)
   torch.testing.assert_close(full_state.cache*full_state.cache_decay_scale,state.cache*state.cache_decay_scale,rtol=1e-4,atol=1e-5)
   self.assertEqual(full_state.cache_scale_phase,state.cache_scale_phase)
   state.cache_scale_phase=None
   coerced=layer._coerce_state(state,1,torch.device('cpu'),torch.float32)
   torch.testing.assert_close(coerced.cache,state.cache*state.cache_decay_scale)
   self.assertEqual(float(coerced.cache_decay_scale.item()),1.)

if __name__=='__main__':
 torch.set_num_threads(1);unittest.main()
