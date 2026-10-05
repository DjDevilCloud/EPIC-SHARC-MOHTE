# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training span / loss-mask / train-gen protocol boundary tests."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PrismalWaveConfig
from data import (
    PrismalTokenizer,
    StreamingTextCorpusDataset,
    _build_loss_mask,
    _build_window_samples_from_text,
    _compose_structured_qa_record,
    build_collate_fn,
)
from train import (
    _fast_context_bundle_from_text,
    _prepend_fast_context_prefix,
    _token_superposition_phase_active,
)


class TrainingBoundaryTests(unittest.TestCase):
    def _token_names(self, tokenizer: PrismalTokenizer, ids) -> list[str]:
        names: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if 0 <= token_id < len(tokenizer.construction_units):
                names.append(tokenizer.construction_units[token_id].text)
            else:
                names.append(f"<UNK:{token_id}>")
        return names

    def test_structured_qa_encodes_real_boundary_specials(self) -> None:
        tokenizer = PrismalTokenizer()
        text = _compose_structured_qa_record(
            instruction="What is a torus?",
            context="",
            response="A torus is a doughnut-shaped surface.",
            style="instruction",
        )
        encoded = tokenizer.encode_hierarchy(text, add_special_tokens=False)[0]
        for marker in ("<BOI>", "<EOI>", "<BOO>", "<EOO>"):
            special_id = tokenizer.special_tokens[marker]
            self.assertGreaterEqual(
                encoded.count(special_id),
                1,
                msg=f"{marker} missing as atomic special; stream={self._token_names(tokenizer, encoded[:40])}",
            )
        # Must not only be character-decomposed angle-bracket spellings of BOI.
        self.assertNotEqual(encoded.count(tokenizer.special_tokens["<BOI>"]), 0)

    def test_plain_window_has_eoi_boo_gen_parity(self) -> None:
        tokenizer = PrismalTokenizer()
        samples = _build_window_samples_from_text(tokenizer, "Hello world. Short line.", seq_len=0)
        self.assertGreaterEqual(len(samples), 1)
        names = self._token_names(tokenizer, samples[0].input_ids.tolist())
        self.assertIn("<BOI>", names)
        self.assertIn("<EOI>", names)
        self.assertIn("<BOO>", names)
        eoi = names.index("<EOI>")
        # Generation scaffold is EOI then BOO (possibly with BOS only at the very start).
        self.assertIn("<BOO>", names[eoi : eoi + 3])

    def test_structured_loss_mask_starts_at_response_content(self) -> None:
        tokenizer = PrismalTokenizer()
        text = _compose_structured_qa_record(
            instruction="What is a torus?",
            context="",
            response="A torus is a doughnut-shaped surface.",
            style="instruction",
        )
        samples = _build_window_samples_from_text(tokenizer, text, seq_len=0)
        self.assertGreaterEqual(len(samples), 1)
        sample = samples[0]
        mask = sample.loss_mask.tolist()
        labels = sample.labels.tolist()
        inputs = sample.input_ids.tolist()
        first_sup = next(i for i, value in enumerate(mask) if float(value) > 0.0)
        # Supervised target should be response content, not the instruction.
        label_name = self._token_names(tokenizer, [labels[first_sup]])[0]
        self.assertNotIn(label_name, {"<BOI>", "<EOI>", "<BOO>"})
        # Context around the flip should include BOO before supervised output.
        names = self._token_names(tokenizer, inputs)
        self.assertIn("<BOO>", names[: first_sup + 1])
        # Input-span specials must be unsupervised as targets. EOO is the
        # learned output terminator, so it is intentionally supervised.
        for index, token_id in enumerate(labels):
            name = self._token_names(tokenizer, [token_id])[0]
            if name in {"<BOI>", "<EOI>", "<BOO>", "<BLO>"}:
                self.assertEqual(float(mask[index]), 0.0, msg=f"structure target {name} supervised at {index}")
        eoo_targets = [i for i, token_id in enumerate(labels) if self._token_names(tokenizer, [token_id])[0] == "<EOO>"]
        self.assertTrue(eoo_targets)
        self.assertTrue(any(float(mask[i]) > 0.0 for i in eoo_targets))

    def test_output_line_and_end_markers_are_supervised(self) -> None:
        tokenizer = PrismalTokenizer()
        text = _compose_structured_qa_record(
            instruction="Give two lines.", context="", response="First line.\nSecond line.", style="instruction"
        )
        samples = _build_window_samples_from_text(tokenizer, text, seq_len=0)
        targets = {}
        for sample in samples:
            for token_id, mask in zip(sample.labels.tolist(), sample.loss_mask.tolist()):
                name = self._token_names(tokenizer, [token_id])[0]
                if name in {"<LINE>", "<EOL>", "<EOO>"}:
                    targets.setdefault(name, []).append(float(mask))
        for marker in ("<EOL>", "<EOO>"):
            self.assertTrue(targets.get(marker), msg=f"missing {marker} target")
            self.assertTrue(any(value > 0.0 for value in targets[marker]), msg=f"{marker} is not supervised")
        self.assertTrue(targets.get("<LINE>"))
        self.assertEqual(targets["<LINE>"][0], 0.0, "the seeded first line frame must remain contextual")
        self.assertTrue(any(value > 0.0 for value in targets["<LINE>"][1:]), "later line frames must be generated")

    def test_streaming_sample_order_is_reproducible_from_seed(self) -> None:
        tokenizer = PrismalTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "records.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps({"text": f"<BOI>Question {i}<EOI><BOO>Answer {i} with more context.<EOO>"})
                    for i in range(12)
                ),
                encoding="utf-8",
            )
            datasets = [
                StreamingTextCorpusDataset(source, tokenizer, seq_len=32, max_samples=8, seed=123)
                for _ in range(2)
            ]
            samples = [[sample.input_ids.tolist() for sample in dataset] for dataset in datasets]
        self.assertEqual(samples[0], samples[1])

    def test_loss_mask_label_shift_alignment(self) -> None:
        tokenizer = PrismalTokenizer()
        samples = _build_window_samples_from_text(tokenizer, "Alpha beta gamma.", seq_len=0)
        sample = samples[0]
        # Reconstruct window stream: input + final label == BOS + chunk + EOS
        full = sample.input_ids.tolist() + [int(sample.labels[-1])]
        self.assertEqual(full[0], tokenizer.bos_id)
        self.assertEqual(full[-1], tokenizer.eos_id)
        self.assertEqual(len(sample.input_ids), len(sample.labels))
        self.assertEqual(len(sample.loss_mask), len(sample.labels))
        for index in range(len(sample.labels)):
            self.assertEqual(int(sample.labels[index]), int(full[index + 1]))

    def test_prepare_generation_ends_with_eoi_boo(self) -> None:
        tokenizer = PrismalTokenizer()
        bundle = tokenizer.prepare_generation_hierarchy("What is a torus?")
        names = self._token_names(tokenizer, bundle.token_ids)
        self.assertEqual(names[:2], ["<BOS>", "<BOI>"])
        self.assertEqual(names[-2:], ["<EOI>", "<BOO>"])

    def test_encode_decode_preserves_single_newlines(self) -> None:
        tokenizer = PrismalTokenizer()
        text = "Line one.\nLine two."
        bundle = tokenizer.encode_hierarchy_bundle(text, add_special_tokens=True)
        decoded = tokenizer.decode(bundle.token_ids)
        # Single newline between lines (not doubled by LINE+EOL both rendering).
        self.assertIn("line one", decoded.lower())
        self.assertIn("line two", decoded.lower())
        self.assertNotIn("\n\n\n", decoded)
        between = decoded.lower().split("line one", 1)[1].split("line two", 1)[0]
        self.assertEqual(between.count("\n"), 1)

    def test_generation_allows_eol_newline(self) -> None:
        tokenizer = PrismalTokenizer()
        suppressed = set(tokenizer.generation_suppressed_token_ids())
        eol = tokenizer.special_tokens["<EOL>"]
        line = tokenizer.special_tokens["<LINE>"]
        self.assertNotIn(eol, suppressed)
        self.assertNotIn(line, suppressed)
        self.assertNotIn(tokenizer.special_tokens["<EOO>"], suppressed)

    def test_fst_training_prefix_default_off_and_content_only(self) -> None:
        cfg = PrismalWaveConfig()
        self.assertFalse(cfg.fst_use_training_prefix)
        tokenizer = PrismalTokenizer()
        bundle = _fast_context_bundle_from_text(tokenizer, "System: be helpful.")
        self.assertIsNotNone(bundle)
        assert bundle is not None
        names = self._token_names(tokenizer, bundle["input_ids"].tolist())
        self.assertNotIn("<BOS>", names)
        self.assertNotIn("<BOO>", names)
        self.assertNotIn("<EOI>", names)

    def test_fst_prepend_does_not_inject_second_bos_scaffold(self) -> None:
        tokenizer = PrismalTokenizer()
        sample = _build_window_samples_from_text(tokenizer, "Hello world.", seq_len=0)[0]
        batch = (
            sample.input_ids.unsqueeze(0),
            sample.labels.unsqueeze(0),
            sample.signature_ids.unsqueeze(0),
            sample.signature_level_ids.unsqueeze(0),
            sample.signature_relation_ids.unsqueeze(0),
            sample.parent_signature_ids.unsqueeze(0),
            sample.signature_family_ids.unsqueeze(0),
            sample.hierarchy_vectors.unsqueeze(0),
            sample.loss_mask.unsqueeze(0),
        )
        state = {
            "enabled": True,
            "use_training_prefix": True,
            "bundle": _fast_context_bundle_from_text(tokenizer, "System: be helpful."),
        }
        out = _prepend_fast_context_prefix(batch, state, pad_id=tokenizer.pad_id, signature_pad_id=0)
        names = self._token_names(tokenizer, out[0][0].tolist())
        bos_positions = [index for index, name in enumerate(names) if name == "<BOS>"]
        self.assertEqual(bos_positions, [names.index("<BOS>")])
        # No generation scaffold glued mid-stream.
        self.assertNotIn("EOI-BOO-BOS", "-".join(names))
        # Prefix region is unsupervised.
        prefix_len = int(state["bundle"]["input_ids"].numel())
        self.assertTrue(torch.all(out[8][0, :prefix_len] == 0))

    def test_span_based_mask_matches_encoded_boo_span(self) -> None:
        tokenizer = PrismalTokenizer()
        text = _compose_structured_qa_record(
            instruction="Name a shape.",
            context="",
            response="Circle.",
            style="instruction",
        )
        encoded = tokenizer.encode_hierarchy(text, add_special_tokens=False)[0]
        mask = _build_loss_mask(tokenizer, text, encoded)
        boo = tokenizer.special_tokens["<BOO>"]
        eoo = tokenizer.special_tokens["<EOO>"]
        boo_index = encoded.index(boo)
        eoo_index = encoded.index(eoo)
        for index, (token_id, value) in enumerate(zip(encoded, mask)):
            if index <= boo_index or index >= eoo_index or int(token_id) in {
                tokenizer.special_tokens["<LINE>"],
                tokenizer.special_tokens["<EOL>"],
                tokenizer.special_tokens["<BLO>"],
            }:
                # Structure zeros applied later in window builder; span mask itself is 0 outside output.
                if index <= boo_index or index >= eoo_index:
                    self.assertEqual(float(value), 0.0)
            else:
                self.assertEqual(float(value), 1.0)

    def test_collate_pads_loss_mask_and_labels(self) -> None:
        tokenizer = PrismalTokenizer()
        short = _build_window_samples_from_text(tokenizer, "Hi.", seq_len=0)[0]
        long = _build_window_samples_from_text(tokenizer, "This is a longer example sentence.", seq_len=0)[0]
        collate = build_collate_fn(tokenizer.pad_id, signature_pad_id=0)
        batch = collate([short, long])
        inputs, labels, *_rest, loss_masks = batch
        max_len = inputs.size(1)
        self.assertEqual(labels.size(1), max_len)
        self.assertEqual(loss_masks.size(1), max_len)
        short_len = short.input_ids.numel()
        self.assertTrue(torch.all(labels[0, short_len:] == tokenizer.pad_id))
        self.assertTrue(torch.all(loss_masks[0, short_len:] == 0))

    def test_token_superposition_phase_boundaries(self) -> None:
        cfg = PrismalWaveConfig()
        cfg.use_token_superposition_training = True
        cfg.token_superposition_phase_fraction = 0.3
        self.assertTrue(
            _token_superposition_phase_active(
                cfg,
                resume_global_step=0,
                optimizer_step=0,
                scheduler_total_steps=100,
            )
        )
        self.assertTrue(
            _token_superposition_phase_active(
                cfg,
                resume_global_step=0,
                optimizer_step=29,
                scheduler_total_steps=100,
            )
        )
        self.assertFalse(
            _token_superposition_phase_active(
                cfg,
                resume_global_step=0,
                optimizer_step=30,
                scheduler_total_steps=100,
            )
        )


if __name__ == "__main__":
    unittest.main()
