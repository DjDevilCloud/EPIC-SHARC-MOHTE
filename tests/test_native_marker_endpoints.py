import unittest
import torch
from torch import nn
from data import PrismalTokenizer
from native_question_memory import NativeQuestionMemory


class NativeMarkerEndpointTests(unittest.TestCase):
    def test_semantic_markers_break_the_percent_comma_profile_collision(self):
        with torch.random.fork_rng():
            torch.manual_seed(37)
            tokenizer = PrismalTokenizer()
            memory = NativeQuestionMemory(8, span_readout=True, marker_endpoints=True)
            embedding = nn.Embedding(tokenizer.vocab_size, 8)
            memory.eval()
            rows = []
            for marker in ('%,', ','):
                prefix = tokenizer.prepare_generation_hierarchy(f'Context:\nThe recorded value is 30{marker} according to the report.\nQuestion:\nWhat value was recorded?')
                ids = torch.tensor([prefix.token_ids])
                _, states, records = memory(ids, torch.zeros(1, len(prefix.token_ids), 8), embedding, tokenizer, return_readout=True)
                state = states[0]
                plan = records[0][4]
                question = records[0][2]
                start = next(i for i, (word, _) in enumerate(state.source) if tokenizer.decode(list(word), clean_text=False, collapse_structure=False) == '30')
                candidates = memory.span_readout.candidates(state, tokenizer, [start])
                text = '30%' if marker == '%,' else '30,'
                candidate = next(c for c in candidates if c['text'] == text)
                rows.append(candidate)
                if marker == '%,':
                    bare = next(c for c in candidates if c['text'] == '30')
                    nn.init.normal_(memory.span_readout.marker_endpoint[-1].weight, std=.1)
                    scores = memory.span_readout.score_candidates([bare, candidate], question, plan['word_vectors'], plan['start_log_probabilities'], marker_words=plan['marker_words'])
                    (-scores.log_softmax(0)[1]).backward()
                    grad = memory.span_readout.marker_query.weight.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(float(grad.abs().sum()), 0.)
                    with self.assertRaisesRegex(ValueError, 'question-word'):
                        memory.span_readout.score_candidates([candidate], question, plan['word_vectors'], plan['start_log_probabilities'])
            self.assertEqual(rows[0]['properties'], rows[1]['properties'])
            self.assertNotEqual(rows[0]['marker_properties'], rows[1]['marker_properties'])
