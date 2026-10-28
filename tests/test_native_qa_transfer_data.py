import tempfile
import unittest
from pathlib import Path
from native_qa_review import qa_fingerprint
from native_qa_transfer_data import group_sources, validate_review, freeze_dataset, fixed_passages, regroup_passage_candidates
from review_native_qa_passages import materialized_reviews


class NativeTransferDataTests(unittest.TestCase):
    def test_old_passage_exposure_excludes_its_other_parent_passages(self):
        def record(context, answer, group):
            return dict(context=context, question='Which word?', answer=answer, source_group=group)
        old='alpha beta gamma delta epsilon zeta eta theta'
        fresh='one two three four five six seven eight'
        rows=[record(old,'alpha','a'),record(fresh,'one','a'),record(fresh,'two','a')]
        candidates=dict(documents=[dict(source_group='a',records=rows)],
                        parent_inventory={'parent':dict(group='a')},inventory=dict(paired_groups=1),partition={})
        result=regroup_passage_candidates(candidates,[old])
        self.assertEqual(result['documents'],[])
        self.assertEqual(candidates['documents'][0]['records'][0]['source_group'],'a')

    def test_manual_related_family_exposure_propagates_transitively(self):
        rows=[dict(context=f'Context {i}',question='Which?',answer=str(i),source_group=str(i)) for i in range(3)]
        candidates=dict(documents=[dict(source_group=str(i),records=[r]) for i,r in enumerate(rows)])
        bundle=dict(decisions={qa_fingerprint(r):dict(decision='retain',reason='Explicit') for r in rows},
                    family_links=[dict(groups=[0,1]),dict(groups=[1,2])],exposed_groups={'2':'Old family'})
        reviews=materialized_reviews(candidates,bundle)
        self.assertEqual({r['decision'] for r in reviews.values()},{'quarantine'})
        self.assertEqual(len({r['related_family'] for r in reviews.values()}),1)
        self.assertEqual(bundle['decisions'][qa_fingerprint(rows[0])]['decision'],'retain')

    def test_freeze_needs_same_passage_pairs_and_counts_family_alias_once(self):
        documents=[];reviews={}
        for i in range(128):
            records=[]
            for j,answer in enumerate(('Alice','York')):
                row=dict(context=f'Alice visited York. Entry {i}',question=f'Question {j}?',answer=answer,source_group=str(i))
                start=row['context'].index(answer)
                reviews[qa_fingerprint(row)]=dict(fingerprint=qa_fingerprint(row),decision='retain',reason='Explicit',
                    support=[0,len(row['context'])],answer_offsets=[start,start+len(answer)])
                records.append(row)
            documents.append(dict(source_group=str(i),records=records))
        candidates=dict(documents=documents,source_sha256='source',provenance_limitations='Synthetic unit-test fixtures')
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'manifest.json'
            # Two related source entries cannot count twice toward the 128.
            for d in documents[:2]:
                for row in d['records']:reviews[qa_fingerprint(row)]['related_family']='same'
            with self.assertRaisesRegex(ValueError,'127/128'):
                freeze_dataset(candidates,reviews,target)
            for d in documents[:2]:
                for row in d['records']:reviews[qa_fingerprint(row)].pop('related_family')
            # Different passages within one family do not form a contrast pair.
            documents[-1]['records'][1]['context']='Alice visited York. Different passage'
            with self.assertRaisesRegex(ValueError,'127/128'):
                freeze_dataset(candidates,reviews,target)
            documents[-1]['records'][1]['context']='Alice visited York. Entry 127'
            result=freeze_dataset(candidates,reviews,target)
            self.assertEqual([len(result['splits'][s]) for s in ('train','development','test')],[128,64,64])
            with self.assertRaises(FileExistsError):freeze_dataset(candidates,reviews,target)

    def test_passage_partition_has_exact_offsets_and_no_overlap(self):
        text = 'Heading\n\n' + ' '.join(f'word{i}' for i in range(420))
        parts = fixed_passages(text)
        self.assertEqual(' '.join(p['context'] for p in parts).split(), text.split())
        self.assertTrue(all(p['context'] == text[p['start']:p['stop']] for p in parts))
        self.assertTrue(all(len(p['context'].split()) <= 160 for p in parts))
        self.assertTrue(all(a['stop'] <= b['start'] for a,b in zip(parts, parts[1:])))
        self.assertEqual(parts, fixed_passages(text))
        with self.assertRaises(ValueError):
            fixed_passages(text, max_words=20, min_boundary_words=30)
    def test_containment_groups_fragments_and_exact_variants(self):
        base = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu'
        groups = group_sources([base, base.upper(), base + ' nu xi omicron', 'one two three four five six seven'])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])

    def test_offsets_and_explicit_support_are_required(self):
        record = dict(context='Alice visited York.', question='Where did Alice visit?', answer='York')
        review = dict(fingerprint=qa_fingerprint(record), decision='retain', reason='Explicit visit location',
                      support=[0, 19], answer_offsets=[14, 18])
        validate_review(record, review)
        for replacement in (dict(support=[0, 5]), dict(answer_offsets=[13, 18]), dict(reason=''), dict(fingerprint='changed')):
            with self.assertRaises(ValueError):
                validate_review(record, dict(review, **replacement))

    def test_provenance_unites_different_texts(self):
        contexts = ['first distinct passage about rivers', 'second separate text about stars']
        self.assertEqual(*group_sources(contexts, source_ids=['same-page', 'same-page']))
        with self.assertRaises(ValueError):
            group_sources(contexts, source_ids=['short'])

    def test_insufficient_reviews_cannot_freeze_a_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'manifest.json'
            with self.assertRaisesRegex(ValueError, 'Insufficient reviewed'):
                freeze_dataset(dict(documents=[]), {}, output)
            self.assertFalse(output.exists())
