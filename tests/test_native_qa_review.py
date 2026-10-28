import copy
import unittest
from native_qa_review import qa_fingerprint, reviewed_indices


class NativeQAReviewTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(context='A is 30%; B is 20%.', question='Which is highest?', answer=a)
                     for a in ('A', 'B')]
        self.decisions = [dict(index=i, fingerprint=qa_fingerprint(row),
                               decision=('retain' if i == 0 else 'quarantine'),
                               reason=('Supported maximum' if i == 0 else 'Contradicts the supplied figures'))
                          for i, row in enumerate(self.rows)]

    def test_quarantine_preserves_original_records(self):
        original = copy.deepcopy(self.rows)
        self.assertEqual(reviewed_indices(self.rows, self.decisions), [0])
        self.assertEqual(self.rows, original)

    def test_changed_source_question_or_answer_invalidates_review(self):
        for field in ('context', 'question', 'answer'):
            changed = copy.deepcopy(self.rows)
            changed[0][field] += ' changed'
            with self.assertRaises(ValueError):
                reviewed_indices(changed, self.decisions)

    def test_missing_reordered_and_invalid_decisions_are_rejected(self):
        for decisions in (self.decisions[:1], self.decisions[::-1]):
            with self.assertRaises(ValueError):
                reviewed_indices(self.rows, decisions)
        for update in (dict(decision='accept_silently'), dict(reason='')):
            decisions = copy.deepcopy(self.decisions)
            decisions[0].update(update)
            with self.assertRaises(ValueError):
                reviewed_indices(self.rows, decisions)
