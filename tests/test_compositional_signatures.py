"""Shared representation and conservative, state-safe span verification."""
import tempfile
import unittest

import torch

from config import PrismalWaveConfig
from data import PrismalTokenizer
from model import PrismalWaveModel
from signature_bank import SharedSignatureBank, SignatureSpanMemory, VerifiedSpanCursor, components
from train import resolve_runtime_config, save_checkpoint, load_bundle_from_checkpoint
from tests import test_causal_protocol as fixtures


class CompositionalSignatureTests(unittest.TestCase):
    def setup_model(self, spans=True):
        helper = fixtures.CausalProtocolTests()
        t = helper.tokenizer()
        cfg = helper.model(t, lattice=True, chunk_len=1).cfg
        cfg.signature_representation = 'compositional_v1'
        cfg.signature_component_buckets = 256
        cfg.use_verified_signature_spans = spans
        cfg.signature_span_context = 8
        cfg.signature_span_capacity = 16
        cfg.signature_span_tokens = 4
        model = PrismalWaveModel(cfg)
        model._prismal_tokenizer = t
        model.prepare_capacity_for_tokenizer(t)
        return helper, t, model

    def test_single_learned_table_and_gradients_from_model(self):
        helper, t, model = self.setup_model()
        for view in (model.registry.family_embedding, model.registry.parent_embedding,
                     model.router.signature_embedding, model.router.parent_embedding,
                     model.signature_neighborhood_embedding):
            self.assertIs(view.bank, model.shared_signature_bank)
            self.assertEqual(list(view.parameters()), [])
        self.assertIs(model.signature_lattice_attention.level_bias.bank, model.shared_signature_bank)
        self.assertFalse(any(key.endswith('signature_embedding.weight') or key.endswith('parent_embedding.weight')
                             for key in model.state_dict()))
        sample = helper.sample(t)
        model.train()
        loss, _ = model.compute_loss(sample.input_ids[None], sample.labels[None],
            **helper.inputs(sample), loss_mask=sample.loss_mask[None], collect_telemetry=False)
        loss.backward()
        self.assertGreater(float(model.shared_signature_bank.embedding.weight.grad.abs().sum()), 0.)
        self.assertTrue(torch.isfinite(model.shared_signature_bank.embedding.weight.grad).all())

    def test_numeric_id_vectors_are_not_used_and_incremental_matches(self):
        helper, t, model = self.setup_model(False)
        model.eval()
        sample = helper.sample(t)
        inputs = helper.inputs(sample)
        changed = dict(inputs, hierarchy_vectors=torch.randn_like(inputs['hierarchy_vectors']))
        with torch.no_grad():
            full = model(sample.input_ids[None], **inputs)
            other = model(sample.input_ids[None], **changed)
            torch.testing.assert_close(full.logits, other.logits, rtol=0, atol=0)
            slots = lattice = None
            pieces = []
            for i in range(sample.input_ids.numel()):
                token = {name: value[:, i:i+1] for name, value in inputs.items()}
                _, slots, output = model.forward_incremental(sample.input_ids[i:i+1][None], **token,
                    slot_state=slots, signature_lattice_state=lattice, position_index=i)
                lattice = output.signature_lattice_state
                pieces.append(output.logits)
            torch.testing.assert_close(full.logits, torch.cat(pieces, 1), rtol=1e-4, atol=1e-5)

    def test_overlapping_constituents_and_distinct_identity(self):
        t = PrismalTokenizer()
        t.learn_from_texts(['A red apple.', 'A green apple.'], min_frequency=1, max_new_tokens=16)
        cfg = resolve_runtime_config(PrismalWaveConfig(d_model=16, signature_component_buckets=128), t)
        bank = SharedSignatureBank(cfg)
        bank.configure(t)
        red, green = t.construction_text_to_id['red'], t.construction_text_to_id['green']
        red_components = set(bank.token_components[red].tolist()) - {0}
        green_components = set(bank.token_components[green].tolist()) - {0}
        self.assertTrue(red_components & green_components)
        self.assertNotEqual(int(bank.token_keys[red]), int(bank.token_keys[green]))
        self.assertEqual(components('line|wc=8|len=32')[1], components('line|wc=9|len=32')[1])

    def test_local_uncertainty_preserves_confident_suffix(self):
        _, _, model = self.setup_model()
        memory = model.signature_span_memory
        for _ in range(2):
            memory.add([1, 3], [10, 12, 15])
            memory.add([1, 3], [11, 12, 15])
        proposal = memory.propose([1, 3], model.shared_signature_bank)
        self.assertEqual(proposal.confidence, [.5, 1., 1.])
        cursor = VerifiedSpanCursor(memory, model.shared_signature_bank)
        self.assertEqual(cursor.verify([1, 3], 11), 11)
        self.assertEqual(cursor.verify([1, 3, 11], 12), 12)
        self.assertEqual(cursor.verify([1, 3, 11, 12], 15), 15)
        self.assertEqual(cursor.stats['uncertain_tokens'], 1)
        self.assertEqual(cursor.stats['verified_tokens'], 2)

    def test_wrong_proposal_never_changes_filtered_choice(self):
        _, _, model = self.setup_model()
        for _ in range(2):
            model.signature_span_memory.add([1, 3], [9, 10])
        cursor = VerifiedSpanCursor(model.signature_span_memory, model.shared_signature_bank)
        self.assertEqual(cursor.verify([1, 3], 11), 11)
        self.assertEqual(cursor.stats['rejected_tokens'], 1)
        self.assertIsNone(model.signature_span_memory.propose([400, 399], model.shared_signature_bank))

    def test_eviction_and_reload_rebuild_exact_tags(self):
        _, _, model = self.setup_model()
        memory = model.signature_span_memory
        for i in range(20):
            memory.add([i + 50], [i + 80])
        self.assertEqual(len(memory._entries), 16)
        self.assertNotIn(((50,), (80,)), memory._lookup)
        saved = {key: value.clone() for key, value in memory.state_dict().items()}
        memory.add([900], [901])
        memory.load_state_dict(saved)
        memory._ensure_index()
        self.assertNotIn(((900,), (901,)), memory._lookup)
        memory.add([69], [99])
        self.assertEqual(memory.propose([69], model.shared_signature_bank).tokens, [99])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_autocast_shared_bank_backward_is_finite(self):
        helper, t, model = self.setup_model(False)
        model = model.to('cuda').train()
        sample = helper.sample(t)
        inputs = {key: value.to('cuda') for key, value in helper.inputs(sample).items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model.compute_loss(sample.input_ids[None].cuda(), sample.labels[None].cuda(),
                **inputs, loss_mask=sample.loss_mask[None].cuda(), collect_telemetry=False)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.shared_signature_bank.embedding.weight.grad).all())

    def test_checkpoint_restores_bank_and_training_evidence(self):
        helper, t, model = self.setup_model()
        sample = helper.sample(t)
        model.signature_span_memory.add([1, 3], [9, 10])
        model.eval()
        with torch.no_grad():
            expected = model(sample.input_ids[None], **helper.inputs(sample)).logits
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(model, directory, tokenizer=t)
            loaded, restored, cfg = load_bundle_from_checkpoint(checkpoint, device='cpu')
            self.assertEqual(cfg.signature_representation, 'compositional_v1')
            self.assertTrue(torch.equal(model.signature_span_memory.support, loaded.signature_span_memory.support))
            with torch.no_grad():
                actual = loaded(sample.input_ids[None], **helper.inputs(sample)).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_proposals_preserve_generation_and_evaluation_does_not_learn(self):
        helper, t, model = self.setup_model()
        bundle = t.prepare_generation_hierarchy('Complete this.')
        inputs = {name: torch.tensor([getattr(bundle, name)]) for name in
            ('signature_ids', 'signature_level_ids', 'signature_relation_ids', 'parent_signature_ids', 'signature_family_ids')}
        kwargs = dict(max_new_tokens=8, min_new_tokens=8, temperature=0., repetition_penalty=1.,
                      no_repeat_ngram_size=0, use_speculative_decoding=False)
        model.eval()
        plain = model.generate(torch.tensor([bundle.token_ids]), **inputs, **kwargs)
        target = plain[0, len(bundle.token_ids):].tolist()
        for _ in range(2):
            model.signature_span_memory.add(bundle.token_ids, target)
        before = model.signature_span_memory.support.clone()
        proposed = model.generate(torch.tensor([bundle.token_ids]), **inputs, **kwargs)
        self.assertTrue(torch.equal(plain, proposed))
        self.assertGreater(model.last_signature_span_stats['verified_tokens'], 0)
        sample = helper.sample(t)
        model.compute_loss(sample.input_ids[None], sample.labels[None], **helper.inputs(sample), loss_mask=sample.loss_mask[None])
        self.assertTrue(torch.equal(before, model.signature_span_memory.support))


if __name__ == '__main__':
    unittest.main()
