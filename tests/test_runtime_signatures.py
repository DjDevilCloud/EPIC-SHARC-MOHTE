"""Causal runtime composition, independent of catalog coverage and reload."""
import tempfile
import unittest
import torch
from config import PrismalWaveConfig
from data import PrismalTokenizer, _build_window_samples_from_text
from model import PrismalWaveModel
from train import resolve_runtime_config, save_checkpoint, load_bundle_from_checkpoint
from tests import test_causal_protocol as fixtures


class RuntimeSignatureTests(unittest.TestCase):
    def setup_model(self):
        helper=fixtures.CausalProtocolTests()
        t=helper.tokenizer()
        cfg=helper.model(t,lattice=True,chunk_len=1).cfg
        cfg.signature_representation='compositional_v2'
        cfg.signature_component_buckets=512
        model=PrismalWaveModel(cfg)
        model._prismal_tokenizer=t
        model.prepare_capacity_for_tokenizer(t)
        model.eval()
        return helper,t,model

    def test_unknown_words_have_dynamic_features_without_catalog_growth(self):
        _,t,m=self.setup_model()
        before=t.to_state_dict()
        a=torch.tensor([t.prepare_generation_hierarchy('quizzical').token_ids])
        b=torch.tensor([t.prepare_generation_hierarchy('xy').token_ids])
        self.assertEqual(t.signature_id_for_word('quizzical'),t.signature_special_ids['<OTHER>'])
        aw=[i for i in t.encode('quizzical',add_special_tokens=False) if t.construction_units[i].kind not in ('structure','special')]
        bw=[i for i in t.encode('xy',add_special_tokens=False) if t.construction_units[i].kind not in ('structure','special')]
        x,_=m.shared_signature_bank.runtime_features(torch.tensor([aw]),t)
        y,_=m.shared_signature_bank.runtime_features(torch.tensor([bw]),t)
        self.assertFalse(torch.equal(x[0,-1],y[0,-1]))
        self.assertEqual(before,t.to_state_dict())

    def test_partial_unicode_and_causal_feature_chunk_parity(self):
        _,t,m=self.setup_model()
        ids=torch.tensor([t.prepare_generation_hierarchy('A café\n\tUnicode Ω',mode='continuation').token_ids])
        full,_=m.shared_signature_bank.runtime_features(ids,t)
        state=None
        parts=[]
        for i in range(ids.size(1)):
            part,state=m.shared_signature_bank.runtime_features(ids[:,i:i+1],t,state)
            parts.append(part)
        self.assertTrue(torch.equal(full,torch.cat(parts,1)))
        prefix,_=m.shared_signature_bank.runtime_features(ids[:,:7],t)
        self.assertTrue(torch.equal(prefix,full[:,:7]))

    def test_state_branches_do_not_mutate_previous_request(self):
        _,t,m=self.setup_model()
        bank=m.shared_signature_bank
        prefix=torch.tensor([t.encode('qu',add_special_tokens=False)])
        _,state=bank.runtime_features(prefix,t)
        before=state[0].word_length
        tail=torch.tensor([t.encode('izzical',add_special_tokens=False)])
        first,_=bank.runtime_features(tail,t,state)
        second,_=bank.runtime_features(tail,t,state)
        self.assertTrue(torch.equal(first,second))
        self.assertEqual(before,state[0].word_length)
        with self.assertRaisesRegex(ValueError,'Carry output.runtime_signature_state'):
            m.forward_incremental(prefix[:,:1],position_index=1)

    def test_full_incremental_logits_gradients_and_reload(self):
        helper,t,m=self.setup_model()
        s=_build_window_samples_from_text(t,'<BOI>New quizzical café<EOI><BOO>red apple<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        vals=helper.inputs(s)
        with torch.no_grad():
            full=m(s.input_ids[None],**vals)
            state=slots=lattice=None
            parts=[]
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],
                    **{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,
                    signature_lattice_state=lattice,runtime_signature_state=state,position_index=i)
                state=o.runtime_signature_state
                lattice=o.signature_lattice_state
                parts.append(o.logits)
            torch.testing.assert_close(full.logits,torch.cat(parts,1),rtol=1e-4,atol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                path=save_checkpoint(m,directory,tokenizer=t)
                restored,_,cfg=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
                restored.set_capacity_growth_locked(True)
                self.assertEqual(cfg.signature_representation,'compositional_v2')
                self.assertEqual(restored.registry.family_vocab_size,m.registry.family_vocab_size)
                torch.testing.assert_close(full.logits,restored(s.input_ids[None],**vals).logits,rtol=0,atol=0)
        m.train()
        loss,_=m.compute_loss(s.input_ids[None],s.labels[None],**vals,loss_mask=s.loss_mask[None],collect_telemetry=False)
        loss.backward()
        features,_=m.shared_signature_bank.runtime_features(s.input_ids[None],t)
        grad=m.shared_signature_bank.embedding.weight.grad[features.unique()]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()),0)

    def test_generation_carries_runtime_state_and_matches_prefix_replay(self):
        from unittest import mock
        helper,t,m=self.setup_model()
        p=t.prepare_generation_hierarchy('New quizzical café')
        kw={k:torch.tensor([getattr(p,k)]) for k in ('signature_ids','signature_family_ids',
            'signature_level_ids','signature_relation_ids','parent_signature_ids','hierarchy_vectors')}
        options=dict(max_new_tokens=16,min_new_tokens=0,temperature=0.,top_k=1,top_p=1.,
            repetition_penalty=1.,no_repeat_ngram_size=0,use_speculative_decoding=False,
            suppressed_token_ids=t.generation_suppressed_token_ids(),
            token_signature_lookup=t.signature_lookup_by_token_id(),token_family_lookup=t.signature_family_lookup_by_token_id(),
            token_level_lookup=t.signature_level_lookup_by_token_id(),token_relation_lookup=t.signature_relation_lookup_by_token_id())
        # Family bias uses a sequence mean; isolate recurrent execution from that filter.
        with mock.patch.object(m,'_apply_signature_neighborhood_generation_bias',side_effect=lambda logits,*a,**k:logits):
            with mock.patch.object(m.shared_signature_bank,'runtime_features',wraps=m.shared_signature_bank.runtime_features) as trace:
                cached=m.generate(torch.tensor([p.token_ids]),**kw,**options)
                self.assertGreater(trace.call_count,1)
                self.assertIsNotNone(trace.call_args_list[-1].args[2])
                self.assertEqual(trace.call_args_list[-1].args[0].size(1),1)
            m.cfg.use_signature_lattice_generation_cache=False
            replay=m.generate(torch.tensor([p.token_ids]),**kw,**options)
        self.assertTrue(torch.equal(cached,replay))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_bfloat16_runtime_gradients(self):
        helper,t,m=self.setup_model()
        m=m.cuda().train()
        s=helper.sample(t)
        vals={k:v.cuda() for k,v in helper.inputs(s).items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,_=m.compute_loss(s.input_ids[None].cuda(),s.labels[None].cuda(),**vals,
                loss_mask=s.loss_mask[None].cuda(),collect_telemetry=False)
        loss.backward()
        self.assertTrue(torch.isfinite(m.shared_signature_bank.embedding.weight.grad).all())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
