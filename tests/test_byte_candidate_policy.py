"""Fallback symbols cannot masquerade as lexical copy candidates."""
import unittest
import torch
from config import PrismalWaveConfig
from data import ConstructionUnit,_build_window_samples_from_text
from train import evaluate_model
from tests import test_runtime_signatures as runtime_tests

class ByteCandidatePolicyTests(unittest.TestCase):
 def test_ascii_classification_preserves_utf8_fallbacks(self):
  for value,kind,lexical in ((0x7c,'punct',False),(0x5c,'punct',False),(0x20,'space',False),(0,'control',False),(0x41,'char',True),(0x39,'digit',True),(0xc3,'byte',True),(0xa9,'byte',True)):
   u=ConstructionUnit(f'<BYTE:{value:02x}>','byte',f'<BYTE:{value:02x}>')
   self.assertEqual(u.semantic_kind,kind);self.assertEqual(u.is_lexical,lexical)
   self.assertEqual(u.kind,'byte')
 def test_causal_candidate_filter_keeps_unicode_and_legacy_is_explicit(self):
  helper=runtime_tests.RuntimeSignatureTests();helper.identity_readout=True;_,t,m=helper.setup_model()
  before=t.to_state_dict();p=t.prepare_generation_hierarchy('café Ω | \\ red')
  values={k:torch.tensor([getattr(p,k)]) for k in ('signature_ids','signature_family_ids','signature_level_ids','signature_relation_ids','parent_signature_ids','hierarchy_vectors')}
  with torch.no_grad():o=m(torch.tensor([p.token_ids]),**values)
  lexical=[i for i in p.token_ids if t.construction_units[i].is_lexical]
  self.assertEqual([e[0] for e in o.bounded_identity_state[0].entries],lexical[-16:])
  symbols=[i for i in p.token_ids if t.construction_units[i].kind=='byte' and not t.construction_units[i].is_lexical]
  self.assertTrue(symbols);self.assertFalse(set(symbols)&{e[0] for e in o.bounded_identity_state[0].entries})
  unicode_bytes=[i for i in p.token_ids if t.construction_units[i].kind=='byte' and int(t.construction_units[i].text[6:-1],16)>=128]
  self.assertTrue(unicode_bytes);self.assertTrue(set(unicode_bytes)&{e[0] for e in o.bounded_identity_state[0].entries})
  m.bounded_identity_readout.candidate_policy='all_bytes_v1'
  with torch.no_grad():legacy=m(torch.tensor([p.token_ids]),**values)
  old=[i for i in p.token_ids if t.construction_units[i].kind in {'piece','word','phrase','char','digit','byte'}]
  self.assertEqual([e[0] for e in legacy.bounded_identity_state[0].entries],old[-16:])
  self.assertEqual(before,t.to_state_dict())
 def test_old_config_migration_and_new_roundtrip(self):
  cfg=PrismalWaveConfig();self.assertEqual(cfg.identity_readout_candidate_policy,'lexical_bytes_v2')
  payload=cfg.to_dict();self.assertEqual(PrismalWaveConfig.from_dict(payload).identity_readout_candidate_policy,'lexical_bytes_v2')
  payload.pop('identity_readout_candidate_policy')
  self.assertEqual(PrismalWaveConfig.from_dict(payload).identity_readout_candidate_policy,'all_bytes_v1')
 def test_byte_punctuation_is_not_reported_as_lexical_loss(self):
  helper=runtime_tests.RuntimeSignatureTests();_,t,m=helper.setup_model()
  s=_build_window_samples_from_text(t,'<BOI>q<EOI><BOO>|<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
  order=('input_ids','labels','signature_ids','signature_level_ids','signature_relation_ids','parent_signature_ids','signature_family_ids','hierarchy_vectors','loss_mask')
  metrics=evaluate_model(m,[tuple(getattr(s,k)[None] for k in order)],torch.device('cpu'),use_amp=False)
  self.assertNotIn('lexical_supervised_tokens',metrics)
  self.assertEqual(metrics['punctuation_supervised_tokens'],1)
  self.assertEqual(metrics['surface_supervised_tokens'],metrics['all_supervised_tokens'])

if __name__=='__main__':
 torch.set_num_threads(1);unittest.main()
