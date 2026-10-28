import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import native_qa_objective_contract as study
from native_qa_review import qa_fingerprint


class ObjectivePrerequisiteTests(unittest.TestCase):
    def fixture(self):
        row = dict(context='Alice bought 10 shares. Bob sold 10 shares.',
                   question='How many shares did Alice buy?', answer='10')
        decision = dict(fingerprint=qa_fingerprint(row), decision='retain',
                        reason='The first sentence explicitly binds the amount to Alice buying.',
                        support=[[0, 23]], answer_occurrences=[[13, 15]])
        return row, decision

    def test_repeated_answer_must_be_inside_reviewed_evidence(self):
        row, decision = self.fixture()
        study.validate_support(row, decision)
        bad = dict(decision, answer_occurrences=[[32, 34]])
        with self.assertRaises(ValueError):
            study.validate_support(row, bad)

    def test_fingerprint_changes_if_question_or_context_changes(self):
        row, decision = self.fixture()
        for changed in (dict(row, question='How many did Bob sell?'), dict(row, context=row['context']+' Extra.')):
            with self.assertRaises(ValueError):
                study.validate_support(changed, decision)

    def test_ambiguous_question_cannot_receive_positive_starts(self):
        row, decision = self.fixture()
        ambiguous = dict(decision, decision='ambiguous', reason='Two answers satisfy the question.')
        with self.assertRaises(ValueError):
            study.validate_support(row, ambiguous)
        ambiguous['answer_occurrences'] = []
        study.validate_support(row, ambiguous)

    def test_failure_evidence_is_checked_against_original_passage(self):
        row, decision = self.fixture()
        bad = dict(decision, decision='unsupported', answer_occurrences=[], evidence_quote='Invented evidence')
        with self.assertRaises(ValueError):
            study.validate_support(row, bad)

    def test_diagnostic_alternatives_do_not_become_accepted_targets(self):
        row = dict(context='Groups: Latino, Hmong, Somali.', question='Name one group.', answer='Somali')
        decision = dict(fingerprint=qa_fingerprint(row), decision='ambiguous', reason='Three equally valid answers.',
                        support=[[0, len(row['context'])]], answer_occurrences=[],
                        diagnostic_valid_alternatives=[dict(text='Latino', offsets=[8,14]), dict(text='Hmong', offsets=[16,21])])
        before = copy.deepcopy((row, decision))
        study.validate_support(row, decision)
        self.assertEqual(before, (row, decision))
        decision['diagnostic_valid_alternatives'][0]['offsets'] = [0,6]
        with self.assertRaises(ValueError):
            study.validate_support(row, decision)

    def test_locked_test_is_rejected_before_manifest_is_opened(self):
        with patch.object(study, 'read', side_effect=AssertionError('Must not access locked questions')):
            with self.assertRaises(ValueError):
                study.review_rows('test')

    def test_writes_refuse_to_replace_frozen_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'result.json'
            study.write_new(target, dict(a=1))
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                study.write_new(target, dict(a=2))
            self.assertEqual(before, target.read_bytes())

    def full_fixture(self):
        splits = {}
        decisions = {}
        for split, n in [('train',128), ('development',64)]:
            splits[split] = []
            for i in range(n):
                row = dict(context=f'Alice visited York. {split} {i}', question='Where did Alice visit?', answer='York')
                splits[split].append(row)
                key = qa_fingerprint(row)
                decisions[key] = dict(fingerprint=key, decision='retain', reason='Explicit visit destination.',
                                      support=[[0,19]], answer_occurrences=[[14,18]])
        return splits, decisions

    def gate_fixture(self, folder, splits):
        return (patch.object(study,'OUT',Path(folder)),
                patch.object(study,'verify_archive',return_value=dict(passed=True)),
                patch.object(study,'review_rows',side_effect=lambda split:splits[split]),
                patch.object(study,'sha',return_value='frozen'))

    def test_review_verification_is_read_only_and_missing_review_blocks(self):
        splits, _ = self.full_fixture()
        with tempfile.TemporaryDirectory() as folder:
            a,b,c,d = self.gate_fixture(folder, splits)
            with a,b,c,d:
                before = list(Path(folder).rglob('*'))
                gate = study.review_gate()
                self.assertFalse(gate['passed'])
                self.assertEqual(gate['missing'],192)
                self.assertEqual(before,list(Path(folder).rglob('*')))

    def test_one_ambiguity_blocks_even_otherwise_complete_review(self):
        splits, decisions = self.full_fixture()
        one = next(iter(decisions))
        decisions[one] = dict(decisions[one], decision='ambiguous', answer_occurrences=[], reason='Unresolved binding.')
        with tempfile.TemporaryDirectory() as folder:
            study.write_new(Path(folder)/'support_overlay.json',dict(source_manifest_sha256='frozen',decisions=decisions))
            a,b,c,d = self.gate_fixture(folder, splits)
            with a,b,c,d:
                before=(Path(folder)/'support_overlay.json').read_bytes()
                gate=study.review_gate()
                self.assertFalse(gate['passed'])
                self.assertEqual(len(gate['failures']),1)
                self.assertEqual(gate['missing'],0)
                self.assertEqual(before,(Path(folder)/'support_overlay.json').read_bytes())

    def test_only_complete_supported_review_passes_data_gate(self):
        splits, decisions = self.full_fixture()
        with tempfile.TemporaryDirectory() as folder:
            study.write_new(Path(folder)/'support_overlay.json',dict(source_manifest_sha256='frozen',decisions=decisions))
            a,b,c,d = self.gate_fixture(folder, splits)
            with a,b,c,d:
                gate=study.review_gate()
                self.assertTrue(gate['passed'])
                self.assertEqual(gate['reviewed'],192)
                self.assertEqual(gate['optimization_updates'],0)

    def test_wrong_manifest_or_extra_locked_record_is_rejected(self):
        splits, decisions = self.full_fixture()
        decisions['not-a-train-or-development-question'] = dict(decision='retain')
        with tempfile.TemporaryDirectory() as folder:
            study.write_new(Path(folder)/'support_overlay.json',dict(source_manifest_sha256='frozen',decisions=decisions))
            a,b,c,d = self.gate_fixture(folder, splits)
            with a,b,c,d:
                with self.assertRaises(ValueError):
                    study.review_gate()

    def test_changed_denominator_cannot_pass_review(self):
        splits, _ = self.full_fixture()
        splits['train'].pop()
        with tempfile.TemporaryDirectory() as folder:
            a,b,c,d = self.gate_fixture(folder, splits)
            with a,b,c,d:
                with self.assertRaisesRegex(ValueError, 'denominator'):
                    study.review_gate()

    def test_cli_gate_signals_failure_to_automated_callers_without_writing(self):
        with patch.object(study,'review_gate',return_value=dict(passed=False)), \
             patch.object(study,'write_new',side_effect=AssertionError('Verification must not write')), \
             patch('builtins.print'):
            self.assertEqual(study.main(['review-gate']),2)
        with patch.object(study,'review_gate',return_value=dict(passed=True)), patch('builtins.print'):
            self.assertEqual(study.main(['review-gate']),0)

    def test_historical_verifier_allows_new_code_but_rejects_changed_evidence_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root/'v3'
            original = root/'review_artifacts/previous/result.json'
            original.parent.mkdir(parents=True)
            original.write_text('historical result', encoding='utf-8')
            archived_code = out/'history/files/model.py'
            archived_code.parent.mkdir(parents=True)
            archived_code.write_text('old code', encoding='utf-8')
            (root/'model.py').write_text('new code', encoding='utf-8')
            archived_result = out/'history/files/review_artifacts/previous/result.json'
            archived_result.parent.mkdir(parents=True)
            archived_result.write_bytes(original.read_bytes())
            relative = 'review_artifacts/previous/result.json'
            study.write_new(out/'history/archive.json',dict(
                snapshots={'model.py':study.sha(archived_code), relative:study.sha(original)},
                historical_bindings={relative:study.sha(original)}))
            with patch.object(study,'ROOT',root), patch.object(study,'OUT',out):
                before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
                self.assertTrue(study.verify_archive()['passed'])
                self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
                original.write_text('rewritten evidence', encoding='utf-8')
                with self.assertRaises(ValueError):
                    study.verify_archive()

    def test_native_windows_checkpoint_binding_is_revalidated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root/'v3'
            relative = Path('review_artifacts')/'baseline'/'model.pt'
            checkpoint = root/relative
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'protected checkpoint')
            study.write_new(out/'history/archive.json',dict(snapshots={},historical_bindings={str(relative):study.sha(checkpoint)}))
            with patch.object(study,'ROOT',root), patch.object(study,'OUT',out):
                self.assertTrue(study.verify_archive()['passed'])
                checkpoint.write_bytes(b'changed checkpoint')
                with self.assertRaisesRegex(ValueError,'Historical artifact'):
                    study.verify_archive()


if __name__=='__main__':
    unittest.main()
