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
    representation='compositional_v2'
    identity_readout=False
    readout_rule='mixture_v1'
    absolute_positions=True
    def setup_model(self):
        helper=fixtures.CausalProtocolTests()
        t=helper.tokenizer()
        cfg=helper.model(t,lattice=True,chunk_len=1).cfg
        cfg.signature_representation=self.representation
        cfg.signature_component_buckets=512
        cfg.use_bounded_identity_readout=self.identity_readout
        cfg.identity_readout_rule=self.readout_rule
        cfg.use_absolute_position_embeddings=self.absolute_positions
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
            state=identity=slots=lattice=None
            parts=[]
            for i in range(s.input_ids.numel()):
                _,slots,o=m.forward_incremental(s.input_ids[i:i+1][None],
                    **{k:v[:,i:i+1] for k,v in vals.items()},slot_state=slots,
                    signature_lattice_state=lattice,runtime_signature_state=state,bounded_identity_state=identity,position_index=i)
                state=o.runtime_signature_state
                identity=o.bounded_identity_state
                lattice=o.signature_lattice_state
                parts.append(o.logits)
            torch.testing.assert_close(full.logits,torch.cat(parts,1),rtol=1e-4,atol=1e-5)
            with tempfile.TemporaryDirectory() as directory:
                path=save_checkpoint(m,directory,tokenizer=t)
                restored,_,cfg=load_bundle_from_checkpoint(path,device='cpu',load_training_state=False)
                restored.set_capacity_growth_locked(True)
                self.assertEqual(cfg.signature_representation,self.representation)
                self.assertEqual(cfg.use_absolute_position_embeddings,self.absolute_positions)
                self.assertEqual(cfg.identity_readout_candidate_policy,m.cfg.identity_readout_candidate_policy)
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
        options=dict(max_new_tokens=16,min_new_tokens=4,temperature=0.,top_k=1,top_p=1.,
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


class RuntimeSignatureV3Tests(RuntimeSignatureTests):
    representation='compositional_v3'

    def test_distinct_short_positions_and_completed_boundaries(self):
        from signature_bank import RuntimeStructuralState
        from data import ConstructionUnit
        state=RuntimeStructuralState(3)
        unit=lambda text,kind='piece':ConstructionUnit(text=text,kind=kind,render=text)
        first=state.step(unit('red'))
        space=state.step(unit(' ','space'))
        second=state.step(unit('pear'))
        self.assertIn('runtime:word_pos=1',first)
        self.assertIn('runtime:word_pos=2',second)
        self.assertIn('runtime:completed_word:0=2',space)
        end=state.step(unit('<EOL>','structure'))
        self.assertIn('runtime:completed_line:0=3',end)
        self.assertIn('runtime:completed_word:0=2',end)
        output=state.step(unit('<BOO>','structure'))
        self.assertIn('runtime:completed_line:0=3',output)
        self.assertIn('runtime:role=output',output)
        self.assertIn('runtime:word_pos=0',output)
        # New requests reset summaries instead of inheriting another prompt.
        new=state.step(unit('<BOI>','structure'))
        self.assertIn('runtime:completed_line:0=0',new)


class BoundedIdentityReadoutTests(RuntimeSignatureTests):
    identity_readout=True

    def test_query_keys_receive_gradients_and_future_cannot_enter_memory(self):
        helper,t,m=self.setup_model()
        left=_build_window_samples_from_text(t,'<BOI>red apple fruit<EOI><BOO>apple<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        right=_build_window_samples_from_text(t,'<BOI>red apple fruit<EOI><BOO>green<EOO>',seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
        first=next(i for i,(a,b) in enumerate(zip(left.input_ids,right.input_ids)) if a!=b)
        with torch.no_grad():
            a=m(left.input_ids[None],**helper.inputs(left))
            b=m(right.input_ids[None],**helper.inputs(right))
            torch.testing.assert_close(a.logits[:,:first],b.logits[:,:first],rtol=0,atol=0)
            self.assertTrue(all(len(s.entries)<=m.cfg.identity_readout_capacity for s in a.bounded_identity_state))
            self.assertEqual(a.bounded_identity_state[0].entries,b.bounded_identity_state[0].entries)
        m.train()
        loss,_=m.compute_loss(left.input_ids[None],left.labels[None],**helper.inputs(left),loss_mask=left.loss_mask[None],collect_telemetry=False)
        loss.backward()
        for head in (m.bounded_identity_readout.query,m.bounded_identity_readout.key,m.bounded_identity_readout.gate):
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(float(head.weight.grad.abs().sum()),0.)

    def test_capacity_eviction_request_isolation_and_empty_memory_fallback(self):
        helper,t,m=self.setup_model()
        m.bounded_identity_readout.capacity=2
        p=t.prepare_generation_hierarchy('red apple fruit')
        vals={k:torch.tensor([getattr(p,k)]) for k in ('signature_ids','signature_family_ids',
            'signature_level_ids','signature_relation_ids','parent_signature_ids','hierarchy_vectors')}
        with torch.no_grad():
            first=m(torch.tensor([p.token_ids]),**vals)
            entries=list(first.bounded_identity_state[0].entries)
            lexical=[i for i in p.token_ids if t.construction_units[i].kind in {'piece','word','phrase','char','digit','byte'}]
            self.assertEqual([e[0] for e in entries],lexical[-2:])
            # New BOI clears a supplied earlier request's cache without mutating it.
            second=m(torch.tensor([p.token_ids]),**vals,bounded_identity_state=first.bounded_identity_state)
            self.assertEqual(first.bounded_identity_state[0].entries,entries)
            self.assertEqual(second.bounded_identity_state[0].entries,entries)
            empty=t.prepare_generation_hierarchy('')
            empty_vals={k:torch.tensor([getattr(empty,k)]) for k in vals}
            enabled=m(torch.tensor([empty.token_ids]),**empty_vals)
            head=m.bounded_identity_readout
            m.bounded_identity_readout=None
            disabled=m(torch.tensor([empty.token_ids]),**empty_vals)
            m.bounded_identity_readout=head
            torch.testing.assert_close(enabled.logits,disabled.logits,rtol=0,atol=0)


class AbsolutePositionAblationTests(unittest.TestCase):
    def test_disabled_positions_full_incremental_training_and_reload(self):
        helper=RuntimeSignatureTests()
        helper.identity_readout=True
        helper.absolute_positions=False
        helper.test_full_incremental_logits_gradients_and_reload()


class StructurePreservingReadoutTests(BoundedIdentityReadoutTests):
    readout_rule='preserve_structure_v1'

    def test_structural_mass_is_preserved_under_confident_copy(self):
        _,t,m=self.setup_model()
        head=m.bounded_identity_readout
        head.configure(t,t.vocab_size)
        logits=torch.randn(t.vocab_size,requires_grad=True)
        copy=torch.zeros(t.vocab_size)
        lexical=head.lexical_mask.nonzero().flatten()
        copy[lexical[0]]=1.
        base=logits.softmax(-1)
        mixed=head.mix_probabilities(logits,copy,torch.tensor(12.)).softmax(-1)
        torch.testing.assert_close(mixed[~head.lexical_mask],base[~head.lexical_mask],rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(mixed[head.lexical_mask].sum(),base[head.lexical_mask].sum(),rtol=1e-5,atol=1e-7)
        self.assertAlmostEqual(float(mixed.sum()),1.,places=6)
        cap=t.special_tokens['<CAP>']
        (-mixed[cap].log()).backward()
        self.assertGreater(float(logits.grad.abs().sum()),0.)
        self.assertFalse(head.lexical_mask[cap])
        self.assertFalse(head.lexical_mask[t.eos_id])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
