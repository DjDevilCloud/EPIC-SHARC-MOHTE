"""Signature coverage, independent capacities, and streaming metadata regressions."""
import tempfile
import unittest
from unittest import mock

import torch

from config import PrismalWaveConfig
from data import PrismalTokenizer
from model import PrismalWaveModel
from train import save_checkpoint, load_bundle_from_checkpoint
from tests import test_causal_protocol as causal_fixture


class SignatureFoundationTests(unittest.TestCase):
    def test_new_units_have_intrinsic_profiles_even_with_optional_profile_cap(self):
        t = PrismalTokenizer()
        t.learn_from_texts(['quasar quasar red green apple'], max_new_tokens=8,
                          min_frequency=1, max_signature_tokens=1)
        for index, unit in enumerate(t.construction_units):
            self.assertIn(unit.signature, t.signature_to_id, unit.text)
            self.assertEqual(t.token_signature_id_by_id[index], t.signature_to_id[unit.signature])
        self.assertNotEqual(t.token_signature_id_by_id[t.construction_text_to_id['quasar']],
                            t.signature_special_ids['<OTHER>'])
        original = dict(t.signature_to_id)
        normalization = t.hierarchy_normalization_capacities
        t.learn_from_texts(['nebula nebula'], max_new_tokens=1, max_signature_tokens=1)
        for code, index in original.items():
            self.assertEqual(t.signature_to_id[code], index)
        self.assertEqual(normalization, t.hierarchy_normalization_capacities)
        restored = PrismalTokenizer.from_state_dict(t.to_state_dict())
        self.assertEqual(t.token_signature_id_by_id, restored.token_signature_id_by_id)

    def test_parent_growth_does_not_expand_family_table(self):
        t = PrismalTokenizer()
        model = PrismalWaveModel(PrismalWaveConfig(signature_vocab_size=512,
            signature_bucket_vocab_size=16, hierarchical_precision_enabled=False))
        registry = model.registry
        self.assertEqual(registry.family_embedding.num_embeddings, 16)
        self.assertEqual(registry.parent_embedding.num_embeddings, 512)
        registry.parent_context(torch.tensor([700]))
        self.assertEqual(registry.family_embedding.num_embeddings, 16)
        self.assertEqual(registry.family_activity.numel(), 16)
        self.assertEqual(registry.parent_embedding.num_embeddings, 701)
        registry.set_capacity_growth_locked(True)
        with self.assertRaises(RuntimeError):
            registry._ensure_capacity(0, 0, 0, 900)

    def test_checkpoint_preserves_legacy_family_shape_and_logits(self):
        fixture = causal_fixture.CausalProtocolTests()
        t = fixture.tokenizer()
        cfg = fixture.model(t, lattice=True, chunk_len=1).cfg
        cfg.registry_family_capacity = t.signature_vocab_size
        model = PrismalWaveModel(cfg).eval()
        sample = fixture.sample(t)
        inputs = fixture.inputs(sample)
        with torch.no_grad():
            expected = model(sample.input_ids.unsqueeze(0), **inputs).logits
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(model, directory, tokenizer=t)
            payload = torch.load(checkpoint, weights_only=False)
            payload['config'].pop('registry_family_capacity', None)
            torch.save(payload, checkpoint)
            loaded, restored, _ = load_bundle_from_checkpoint(checkpoint, device='cpu')
            self.assertEqual(loaded.registry.family_embedding.num_embeddings,
                             model.registry.family_embedding.num_embeddings)
            self.assertEqual(restored.signature_to_id, t.signature_to_id)
            with torch.no_grad():
                actual = loaded(sample.input_ids.unsqueeze(0), **inputs).logits
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_streaming_metadata_matches_replay_through_utf8_and_lines(self):
        t = PrismalTokenizer()
        text = '<BOI>Question<EOI><BOO>A red apple.\nCafé 漢字 😀 and green.<EOO>'
        t.learn_from_texts([text], max_new_tokens=8, min_frequency=1)
        full = t.encode_hierarchy_bundle(text, add_special_tokens=True)
        first = full.token_ids.index(t.special_tokens['<BOO>']) + 2
        history = full.token_ids[:first]
        state = t.prepare_output_hierarchy_state(history)
        for token in full.token_ids[first:]:
            self.assertEqual(state.step(token), t.output_hierarchy_frame(history, token))
            history.append(token)

    def test_generation_uses_local_states_and_matches_replay(self):
        fixture = causal_fixture.CausalProtocolTests()
        t = fixture.tokenizer()
        model = fixture.model(t, lattice=True, chunk_len=1)
        model._prismal_tokenizer = t
        bundle = t.prepare_generation_hierarchy('Complete this.')
        inputs = {name: torch.tensor([getattr(bundle, name)]) for name in
            ('token_ids', 'signature_ids', 'signature_level_ids', 'signature_relation_ids',
             'parent_signature_ids', 'signature_family_ids')}
        inputs['input_ids'] = inputs.pop('token_ids')
        kwargs = dict(max_new_tokens=12, min_new_tokens=12, temperature=0.,
                      repetition_penalty=1., no_repeat_ngram_size=0, use_speculative_decoding=False)
        with mock.patch.object(t, 'output_hierarchy_frame', side_effect=AssertionError('history replay')):
            first = model.generate(**inputs, **kwargs)
            second = model.generate(**inputs, **kwargs)
        self.assertTrue(torch.equal(first, second))
        streaming = model._causal_generation_metadata
        def replay(history, next_ids, fallback, states=None):
            return streaming(history, next_ids, fallback, states=None)
        with mock.patch.object(model, '_causal_generation_metadata', side_effect=replay):
            expected = model.generate(**inputs, **kwargs)
        self.assertTrue(torch.equal(first, expected))


if __name__ == '__main__':
    unittest.main()
