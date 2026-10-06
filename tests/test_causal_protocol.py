"""Behavioral checks for the shared causal training/decoding contract."""
from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock

import torch

from config import PrismalWaveConfig
from data import CausalOutputHierarchy, MemmapTokenDataset, PrismalTokenizer, _build_window_samples_from_text
from model import PrismalWaveModel
from train import load_bundle_from_checkpoint, resolve_runtime_config, save_checkpoint


TRACKS = (
    "signature_ids", "signature_level_ids", "signature_relation_ids",
    "parent_signature_ids", "signature_family_ids",
)


class CausalProtocolTests(unittest.TestCase):
    def test_fitting_matches_structured_content_boundaries(self):
        t = PrismalTokenizer()
        text = "<BOI>  Quasar beacon\n\tSecond prompt<EOI><BOO>Answer text<EOO>"
        candidates, signatures = t._collect_construction_learning_counts([text])
        self.assertTrue({"boi", "eoi", "boo", "eoo"}.isdisjoint(candidates))
        for line in ("  Quasar beacon", "\tSecond prompt", "Answer text"):
            self.assertIn(t._line_signature_code(line), signatures)
        self.assertNotIn(t._line_signature_code(text), signatures)
        t.learn_from_texts([text], max_new_tokens=16, min_frequency=1)
        encoded = t.prepare_generation_hierarchy("  Quasar beacon\n\tSecond prompt")
        line_tokens = {t.special_tokens["<LINE>"], t.special_tokens["<EOL>"]}
        codes = {t.signature_id_for_code(t._line_signature_code(line))
                 for line in ("  Quasar beacon", "\tSecond prompt")}
        self.assertNotIn(t.signature_special_ids["<OTHER>"], codes)
        end = encoded.token_ids.index(t.special_tokens["<EOI>"])
        observed = [sig for token, sig in zip(encoded.token_ids[:end], encoded.signature_ids[:end])
                    if token in line_tokens]
        self.assertEqual(set(observed), codes)

    def test_unavailable_word_profiles_preserve_hierarchy_and_answer(self):
        from word_profile_control import unavailable_word_profiles
        t = PrismalTokenizer()
        text = "<BOI>Quasar beacon<EOI><BOO>Answer text<EOO>"
        t.learn_from_texts([text], max_new_tokens=16, min_frequency=1)
        sample = _build_window_samples_from_text(t, text, seq_len=256, max_samples=1)[0]
        values = {name: getattr(sample, name).unsqueeze(0)
                  for name in ("input_ids", "labels", "loss_mask", "hierarchy_vectors", *TRACKS)}
        dropped = unavailable_word_profiles(values, t, [True])
        unchanged = unavailable_word_profiles(values, t, [False])
        left = sample.input_ids.tolist().index(t.special_tokens["<BOI>"]) + 1
        right = sample.input_ids.tolist().index(t.special_tokens["<EOI>"])
        self.assertGreater(int(dropped["parent_signature_ids"].ne(values["parent_signature_ids"]).sum()), 0)
        for name in TRACKS:
            self.assertTrue(torch.equal(dropped[name][:, :left], values[name][:, :left]))
            self.assertTrue(torch.equal(dropped[name][:, right:], values[name][:, right:]))
            self.assertTrue(torch.equal(unchanged[name], values[name]))
        for position in range(left, right):
            if int(sample.input_ids[position]) in {t.special_tokens["<LINE>"], t.special_tokens["<EOL>"]}:
                for name in TRACKS:
                    self.assertEqual(int(dropped[name][0, position]), int(values[name][0, position]))
        self.assertNotIn("hierarchy_vectors", dropped)  # Reconstructed from changed IDs.

    def test_unavailable_line_profiles_preserve_word_edges_and_output(self):
        from word_profile_control import unavailable_prompt_profiles
        t = PrismalTokenizer()
        text = "<BOI>Quasar beacon<EOI><BOO>Answer text<EOO>"
        t.learn_from_texts([text], max_new_tokens=16, min_frequency=1)
        sample = _build_window_samples_from_text(t, text, seq_len=256, max_samples=1)[0]
        values = {name: getattr(sample, name).unsqueeze(0)
                  for name in ("input_ids", "labels", "loss_mask", "hierarchy_vectors", *TRACKS)}
        left = sample.input_ids.tolist().index(t.special_tokens["<BOI>"]) + 1
        right = sample.input_ids.tolist().index(t.special_tokens["<EOI>"])
        word_ids = {idx for idx, code in t._signature_id_to_code.items()
                    if code.startswith(("word|", "code|")) and "|st=" in code}
        other = t.signature_special_ids["<OTHER>"]
        for drop_word in (False, True):
            with self.subTest(drop_word=drop_word):
                changed = unavailable_prompt_profiles(values, t, [drop_word], [True])
                for name in ("input_ids", "labels", "loss_mask", "signature_level_ids", "signature_relation_ids"):
                    self.assertTrue(torch.equal(changed[name], values[name]))
                for name in TRACKS:
                    self.assertTrue(torch.equal(changed[name][:, :left], values[name][:, :left]))
                    self.assertTrue(torch.equal(changed[name][:, right:], values[name][:, right:]))
                for pos in range(left, right):
                    if int(sample.input_ids[pos]) in {t.special_tokens["<LINE>"], t.special_tokens["<EOL>"]}:
                        self.assertEqual(int(changed["signature_ids"][0, pos]), other)
                        self.assertEqual(int(changed["signature_family_ids"][0, pos]), t.signature_family_id_by_signature_id[other])
                    if int(sample.parent_signature_ids[pos]) in word_ids:
                        self.assertEqual(int(changed["parent_signature_ids"][0, pos]), other if drop_word else int(sample.parent_signature_ids[pos]))

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tokenizer(self):
        tokenizer = PrismalTokenizer()
        tokenizer.learn_from_texts(
            ["qzxalpha sleeps.\nAnother line.", "qzxbeta walks.\nDifferent ending."],
            max_new_tokens=1, min_frequency=1, max_word_tokens=1,
        )
        return tokenizer

    def model(self, tokenizer, *, lattice=True, chunk_len=1):
        cfg = PrismalWaveConfig()
        for name, value in dict(
            d_model=16, n_emitters=8, n_slots=8, top_k_emitters=2, top_k_slots=2,
            ff_mult=2, factorized_embedding_dim=8, torus_depth=2, torus_height=2,
            torus_width=2, torus_chunk_len=chunk_len, signature_lattice_chunk_len=chunk_len,
            signature_lattice_dim=8, hierarchical_precision_enabled=False,
            hierarchical_precision_accumulator_dtype="float32", use_turbo_quantization=False,
            use_torchao_weight_only=False, use_torchao_embedding_weight_only=False,
            use_gate=False, use_gatetrain=False, use_fullgatetrain=False,
            use_learned_residency_head=False, use_residency_with_reinforcement=False,
            use_contrastive_routing=False, use_signature_lattice_attention=lattice,
            use_token_memory_cross_attention=False, use_hmote=False,
            hierarchical_nest_depth=1, use_recursive_hmoe=False,
            use_fixed_point_solver=False, dropout=0.0, path_noise=0.0,
            disable_auxlosses=True, position_embedding_init_size=256,
        ).items():
            setattr(cfg, name, value)
        model = PrismalWaveModel(resolve_runtime_config(cfg, tokenizer))
        model._prismal_tokenizer = tokenizer
        model.eval()
        return model

    def sample(self, tokenizer, answer="qzxalpha sleeps.\nAnother line."):
        return _build_window_samples_from_text(
            tokenizer, f"<BOI>Complete this.<EOI><BOO>{answer}<EOO>",
            seq_len=256, hierarchy_vector_dtype="float32",
        )[0]

    def inputs(self, sample, start=0, end=None, *, vectors=True):
        values = {name: getattr(sample, name)[start:end].unsqueeze(0) for name in TRACKS}
        if vectors:
            values["hierarchy_vectors"] = sample.hierarchy_vectors[start:end].unsqueeze(0)
        return values

    def test_unseen_suffix_cannot_change_observed_output_features(self):
        tokenizer = self.tokenizer()
        left, right = self.sample(tokenizer), self.sample(tokenizer, "qzxbeta walks.\nDifferent ending.")
        prefix_len = next(i for i, (a, b) in enumerate(zip(left.input_ids, right.input_ids)) if a != b)
        self.assertGreater(prefix_len, left.input_ids.tolist().index(tokenizer.special_tokens["<BOO>"]) + 3)
        for name in (*TRACKS, "hierarchy_vectors"):
            torch.testing.assert_close(getattr(left, name)[:prefix_len], getattr(right, name)[:prefix_len], rtol=0, atol=0)

    def test_entire_answer_matches_incremental_hierarchy_including_unicode(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer, "qzxalpha sleeps.\nAnother line. 🔍")
        boo = sample.input_ids.tolist().index(tokenizer.special_tokens["<BOO>"])
        history = sample.input_ids[:boo].tolist()
        for index in range(boo, sample.input_ids.numel()):
            token_id = int(sample.input_ids[index])
            expected = tokenizer.output_hierarchy_frame(history, token_id)
            actual = tuple(int(getattr(sample, name)[index]) for name in TRACKS)
            self.assertEqual(actual, expected, msg=f"hierarchy differs at token {index}")
            history.append(token_id)

    def test_completed_summaries_are_exposed_after_boundaries(self):
        tokenizer = self.tokenizer()
        bundle = tokenizer.encode_hierarchy_bundle("<BOO>qzxalpha sleeps.<EOO>", add_special_tokens=False)
        state = CausalOutputHierarchy(tokenizer)
        observed_space = False
        for token_id in bundle.token_ids:
            state.step(token_id)
            if token_id == tokenizer.special_tokens["<SPACE>"]:
                self.assertEqual(state.partial_word, "")
                self.assertEqual(state.completed_word_signature, tokenizer.signature_id_for_word("qzxalpha"))
                observed_space = True
            if token_id == tokenizer.special_tokens["<EOL>"]:
                self.assertEqual(state.partial_line, "")
                self.assertEqual(state.completed_line_signature, tokenizer.signature_id_for_line("qzxalpha sleeps."))
        self.assertTrue(observed_space)

    def test_byte_targets_are_reachable_and_decode_as_utf8(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer, "café 🔍")
        blocked = set(tokenizer.generation_suppressed_token_ids())
        for token_id, mask in zip(sample.labels, sample.loss_mask):
            if float(mask) > 0:
                self.assertNotIn(int(token_id), blocked)
        decoded = tokenizer.decode(sample.input_ids.tolist(), clean_text=False)
        self.assertIn("🔍", decoded)
        self.assertNotIn("<BYTE:", decoded)

    def test_non_ascii_letters_are_not_silently_dropped(self):
        tokenizer = self.tokenizer()
        for text in ("café 🔍.", "naïve résumé.", "中文 русский.", "e\u0301."):
            with self.subTest(text=text):
                ids = tokenizer.encode(text, add_special_tokens=False)
                self.assertEqual(tokenizer.decode(ids).strip(), text)

    def test_case_controls_render_their_supervised_word(self):
        tokenizer = self.tokenizer()
        for text in ("Hello NASA ABC123.", "Café and résumé."):
            with self.subTest(text=text):
                ids = tokenizer.encode(text, add_special_tokens=False)
                self.assertEqual(tokenizer.decode(ids, clean_text=False).strip(), text)
                self.assertEqual(tokenizer._decode_construction(ids, clean_text=False).strip(), text)

    def test_internal_case_changes_have_a_reachable_literal_path(self):
        tokenizer = self.tokenizer()
        blocked = set(tokenizer.generation_suppressed_token_ids())
        for text in ("NASA's Voyager.", "iPhone McDonald eBay.", "CAN'T GPT4."):
            with self.subTest(text=text):
                ids = tokenizer.encode(text, add_special_tokens=False)
                self.assertEqual(tokenizer.decode(ids, clean_text=False).strip(), text)
                for token in ids:
                    self.assertNotIn(token, blocked)

    def test_normalization_is_batch_invariant_and_survives_checkpoint(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer)
        model = self.model(tokenizer)
        expected = model._hierarchy_embedding_context(sample.input_ids.unsqueeze(0), **self.inputs(sample))
        actual = torch.cat([
            model._hierarchy_embedding_context(sample.input_ids[i:i+1].unsqueeze(0), **self.inputs(sample, i, i+1, vectors=False))
            for i in range(sample.input_ids.numel())
        ], dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(model, directory, tokenizer=tokenizer, config=model.cfg)
            restored, restored_tokenizer, _ = load_bundle_from_checkpoint(checkpoint, device="cpu")
        self.assertEqual(restored.cfg.hierarchy_vector_normalization, model.cfg.hierarchy_vector_normalization)
        self.assertEqual(restored_tokenizer.hierarchy_normalization_capacities, tokenizer.hierarchy_normalization_capacities)

    def test_storage_rounding_is_the_same_for_provided_and_reconstructed_features(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer)
        model = self.model(tokenizer)
        for dtype_name in ("float32", "bfloat16", "float8_e4m3fn"):
            with self.subTest(dtype=dtype_name):
                model.cfg.hierarchy_vector_dtype = dtype_name
                explicit = model._hierarchy_embedding_context(sample.input_ids.unsqueeze(0), **self.inputs(sample))
                reconstructed = model._hierarchy_embedding_context(
                    sample.input_ids.unsqueeze(0), **self.inputs(sample, vectors=False)
                )
                torch.testing.assert_close(explicit, reconstructed, rtol=0, atol=0)

    def test_tokenizer_extension_does_not_rescale_existing_features(self):
        tokenizer = self.tokenizer()
        frozen = tokenizer.hierarchy_normalization_capacities
        families = dict(tokenizer.signature_family_to_id)
        tokenizer.learn_from_texts(["Additional unfamiliar words."], max_new_tokens=4, min_frequency=1)
        self.assertEqual(frozen, tokenizer.hierarchy_normalization_capacities)
        for family, family_id in families.items():
            self.assertEqual(tokenizer.signature_family_to_id[family], family_id)

    def test_decoder_refresh_preserves_checkpoint_hierarchy_contract(self):
        tokenizer = self.tokenizer()
        text = "<BOI>A prompt.<EOI><BOO>qzxalpha sleeps.\nAnother line.<EOO>"
        before = tokenizer.encode_hierarchy_bundle(text)
        restored = PrismalTokenizer.from_state_dict(tokenizer.to_state_dict())
        for _ in range(3):
            restored.refresh_construction_index()
            after = restored.encode_hierarchy_bundle(text)
            self.assertEqual(before.as_tuple(), after.as_tuple())
            self.assertEqual(before.hierarchy_vectors, after.hierarchy_vectors)

    def test_bootstrap_fingerprint_does_not_freeze_an_unlearned_vocabulary(self):
        tokenizer = PrismalTokenizer()
        state = tokenizer.to_state_dict()
        self.assertIsNone(state["hierarchy_vector_normalization"])
        tokenizer.learn_from_texts(["New vocabulary items."], max_new_tokens=8, min_frequency=1)
        self.assertEqual(tokenizer.hierarchy_normalization_capacities["token_vocab_size"], tokenizer.vocab_size)
        self.assertEqual(tokenizer.hierarchy_normalization_capacities["signature_vocab_size"], tokenizer.signature_vocab_size)

    def test_generation_metadata_helper_matches_teacher_forcing(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer)
        model = self.model(tokenizer)
        start = sample.input_ids.tolist().index(tokenizer.special_tokens["<BOO>"]) + 2
        expected = [getattr(sample, name)[start:].unsqueeze(0) for name in TRACKS]
        tokens = sample.input_ids[start:].unsqueeze(0)
        fallback = tuple(torch.zeros_like(tokens) for _ in range(5))
        signature, family, level, relation, parent = model._causal_generation_metadata(
            sample.input_ids[:start].unsqueeze(0), tokens, fallback,
        )
        for actual, wanted in zip((signature, level, relation, parent, family), expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)

    def test_legacy_pretokenized_hierarchy_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "meta.json").write_text(json.dumps({"format_version": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rebuild"):
                MemmapTokenDataset(directory)

    def test_inference_does_not_mutate_registry_observations(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer)
        model = self.model(tokenizer)
        before = {name: value.clone() for name, value in model.registry.named_buffers()}
        with torch.no_grad():
            model(sample.input_ids.unsqueeze(0), **self.inputs(sample))
        for name, value in model.registry.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_solver_and_nested_mixture_preserve_incremental_parity(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer, "qzxalpha.")
        for nested in (False, True):
            with self.subTest(nested=nested), torch.no_grad():
                base = self.model(tokenizer, lattice=True, chunk_len=7)
                cfg = PrismalWaveConfig.from_dict(base.cfg.to_dict())
                cfg.use_fixed_point_solver = True
                cfg.use_chunk_solver_training = True
                cfg.chunk_solver_training_audit_every = 3
                cfg.use_hmote = nested
                cfg.hierarchical_min_d_model = 8
                cfg.per_family_torus_enabled = False
                cfg.leaf_cell_enabled = False
                cfg.mot_num_experts = 2
                model = PrismalWaveModel(cfg).eval()
                full = model(sample.input_ids.unsqueeze(0), **self.inputs(sample), path_index=0, collect_telemetry=False)
                state, lattice_state, logits = None, None, []
                for index in range(sample.input_ids.numel()):
                    step = model(
                        sample.input_ids[index:index+1].unsqueeze(0),
                        **self.inputs(sample, index, index+1, vectors=False),
                        slot_state=state, signature_lattice_state=lattice_state,
                        position_index=index, path_index=0, collect_telemetry=False,
                    )
                    state, lattice_state = step.slot_state, step.signature_lattice_state
                    logits.append(step.logits)
                torch.testing.assert_close(torch.cat(logits, dim=1), full.logits, rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(state.field, full.slot_state.field, rtol=1e-4, atol=1e-5)

    def test_full_sequence_and_incremental_logits_and_states_agree(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer)
        for lattice in (False, True):
            for chunk_len in (1, 7):
                with self.subTest(lattice=lattice, chunk_len=chunk_len), torch.no_grad():
                    torch.manual_seed(42)
                    model = self.model(tokenizer, lattice=lattice, chunk_len=chunk_len)
                    full = model(sample.input_ids.unsqueeze(0), **self.inputs(sample), path_index=0, collect_telemetry=False)
                    state = None
                    lattice_state = None
                    steps = []
                    for index in range(sample.input_ids.numel()):
                        output = model(
                            sample.input_ids[index:index+1].unsqueeze(0),
                            **self.inputs(sample, index, index+1, vectors=False),
                            slot_state=state, signature_lattice_state=lattice_state,
                            position_index=index, path_index=0, collect_telemetry=False,
                        )
                        state, lattice_state = output.slot_state, output.signature_lattice_state
                        steps.append(output.logits)
                    torch.testing.assert_close(torch.cat(steps, dim=1), full.logits, rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(state.field, full.slot_state.field, rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(state.bus, full.slot_state.bus, rtol=1e-4, atol=1e-5)
                    if lattice:
                        torch.testing.assert_close(lattice_state.cache, full.signature_lattice_state.cache, rtol=1e-4, atol=1e-5)

    def test_lattice_causal_scan_supports_backward(self):
        tokenizer = self.tokenizer()
        sample = self.sample(tokenizer)
        model = self.model(tokenizer, lattice=True, chunk_len=7).train()
        loss, _ = model.compute_loss(
            sample.input_ids.unsqueeze(0), sample.labels.unsqueeze(0),
            **self.inputs(sample), loss_mask=sample.loss_mask.unsqueeze(0),
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), msg=name)
        self.assertGreater(float(model.torus_core.write_delta_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.signature_lattice_attention.v_proj.weight.grad.abs().sum()), 0.0)

    def test_greedy_is_rng_independent_and_cache_off_replays_same_model(self):
        tokenizer = self.tokenizer()
        model = self.model(tokenizer, lattice=True, chunk_len=7)
        prompt = tokenizer.prepare_generation_hierarchy("Complete this.")
        inputs = dict(zip(TRACKS, (torch.tensor([track]) for track in prompt.as_tuple()[1:])))
        kwargs = dict(inputs, max_new_tokens=4, min_new_tokens=0, temperature=0.0,
                      repetition_penalty=1.0, no_repeat_ngram_size=0, use_speculative_decoding=False)
        # Initialize lazy modules before observing RNG use during decoding.
        model.signature_lattice_attention
        with mock.patch("torch.multinomial", side_effect=AssertionError("greedy decode sampled")):
            before = torch.random.get_rng_state().clone()
            cached = model.generate(torch.tensor([prompt.token_ids]), **kwargs)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            model.cfg.use_signature_lattice_generation_cache = False
            replay = model.generate(torch.tensor([prompt.token_ids]), **kwargs)
        self.assertTrue(torch.equal(cached, replay))


if __name__ == "__main__":
    unittest.main()
