"""Official evaluation must expose control errors hidden by lexical loss."""
import unittest
import torch
from data import _build_window_samples_from_text
from train import evaluate_model
from tests import test_causal_protocol as fixtures

class StructuralEvaluationTests(unittest.TestCase):
    def test_masked_class_losses_recompose_full_supervised_loss(self):
        helper=fixtures.CausalProtocolTests();t=helper.tokenizer();m=helper.model(t)
        m.train()
        rows=[_build_window_samples_from_text(t,text,seq_len=256,max_samples=1,hierarchy_vector_dtype='float32')[0]
              for text in ('<BOI>red apple<EOI><BOO>The color is red.<EOO>',
                           '<BOI>green pear<EOI><BOO>green<EOO>')]
        order=('input_ids','labels','signature_ids','signature_level_ids','signature_relation_ids',
               'parent_signature_ids','signature_family_ids','hierarchy_vectors','loss_mask')
        batches=[tuple(getattr(s,k)[None] for k in order) for s in rows]
        result=evaluate_model(m,batches,torch.device('cpu'),use_amp=False)
        lex=result['lexical_supervised_tokens'];surface=result['surface_supervised_tokens']
        self.assertEqual(lex+surface,result['all_supervised_tokens'])
        self.assertAlmostEqual((result['lexical_ce_loss']*lex+result['surface_ce_loss']*surface)/(lex+surface),
                               result['all_ce_loss'],places=5)
        self.assertEqual(result['all_supervised_tokens'],sum(float(s.loss_mask.sum()) for s in rows))
        for kind in ('case','space','ending','punctuation'):
            self.assertGreater(result[kind+'_supervised_tokens'],0)
            self.assertGreaterEqual(result[kind+'_ce_loss'],0)
        self.assertTrue(m.training)

if __name__=='__main__':unittest.main()
