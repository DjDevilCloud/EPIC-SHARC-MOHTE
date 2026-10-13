"""Exercise optimizer continuation through the actual trainer."""
import sys
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_complete_answer_span as span_tests
from data import TextWindowDataset, build_collate_fn
from train import train_model


class TrainerResumeTests(unittest.TestCase):
    def test_cpu_resume_preserves_optimizer_moments_and_advances_step(self):
        torch.set_num_threads(1)
        t, m, _ = span_tests.CompleteAnswerSpanTests().fixture(general=True, unified=True)
        m.cfg.grounded_qa_span_loss_weight = 1.0
        m.cfg.grounded_qa_full_context = True
        m.requires_grad_(False)
        reader = m.bounded_identity_readout.complete_answer_span.unified_reader
        reader.requires_grad_(True)
        m._prismal_tokenizer = t
        ds = TextWindowDataset(
            ['<BOI>Context:\nThe budget is $75.\nQuestion:\nWhat is the budget?<EOI><BOO>$75<EOO>'],
            t, seq_len=512, max_samples=0, preserve_grounded_qa_context=True,
        )
        loader = DataLoader(ds, batch_size=1, collate_fn=build_collate_fn(t.pad_id, t.signature_pad_id))
        # Materialize the tiny fixture's lazy backbone before fixing optimizer addresses.
        batch = next(iter(loader))
        m.eval()
        with torch.no_grad():
            m.compute_loss(*batch[:2], **dict(zip(
                ('signature_ids', 'signature_level_ids', 'signature_relation_ids',
                 'parent_signature_ids', 'signature_family_ids', 'hierarchy_vectors', 'loss_mask'),
                batch[2:],
            )))
        m.requires_grad_(False)
        reader.requires_grad_(True)
        kwargs = dict(cfg=m.cfg, optimizer_name='adamw', steps=1, lr=.001,
                      progress=False, use_amp=False, diagnostic_interval=20)
        train_model(m, loader, torch.device('cpu'), **kwargs)
        first = m._prismal_training_state
        steps = [int(v['step']) for v in first['optimizer_state']['state'].values() if 'step' in v]
        self.assertTrue(steps)
        self.assertEqual(set(steps), {1})
        # Repeat the adapter trainability policy after lazy modules have materialized.
        m.requires_grad_(False)
        reader.requires_grad_(True)
        train_model(m, loader, torch.device('cpu'), **kwargs)
        second = m._prismal_training_state
        self.assertEqual(second['global_step'], 2)
        steps = [int(v['step']) for v in second['optimizer_state']['state'].values() if 'step' in v]
        self.assertEqual(set(steps), {2})
