"""Read-only experiment checks and paired source-group uncertainty report."""
import random

import native_qa_source_study as old
import native_qa_paired_source as pilot


def main():
    protocol=pilot.freeze()
    for name,expected in old.read(old.OUT/'artifact_hashes.json').items():
        assert old.sha(old.OUT.parent/name)==expected, ('Original artifact changed',name)
    decision=old.read(pilot.OUT/'acceptance.json')
    assert not decision['model_accepted'] and not decision['baseline_changed']
    assert decision['locked_test_predictions']==0
    uncertainty={};hashes={}
    for seed in (47,83):
        results={a:old.read(pilot.OUT/f'seed{seed}/{a}/result.json') for a in protocol['arms']}
        for arm,r in results.items():
            assert r['updates']==400 and r['locked_test_predictions']==0
            assert r['protocol_sha256']==old.sha(pilot.OUT/'protocol.json')
            assert [h['step'] for h in r['history']]==[100,200,300,400]
            assert r['protected_parameters_bitwise_equal'] and r['reload_state_bitwise_equal']
            assert r['live_cached_all_development_candidates_identical']
            assert all(k.startswith('native_question_memory.semantic_alignment.') for k in r['changed_keys'])
            assert r['checkpoint_sha256']==old.sha(pilot.OUT/f'seed{seed}/{arm}/final/model.pt')
            assert len(r['ledger']['new'])==128 and len(r['ledger']['development'])==64
        c,p=results['matched_control'],results['paired_assignment']
        grouped={}
        for a,b in zip(c['ledger']['development'],p['ledger']['development']):
            assert a['fingerprint']==b['fingerprint'] and a['source_group']==b['source_group']
            grouped.setdefault(a['source_group'],[]).append(int(b['start_top1'])-int(a['start_top1']))
        assert len(grouped)==32 and all(len(v)==2 for v in grouped.values())
        deltas=[sum(v)/2 for v in grouped.values()]
        rng=random.Random(20261017);samples=[]
        for _ in range(10000):
            samples.append(sum(rng.choice(deltas) for _ in deltas)/len(deltas))
        samples.sort()
        uncertainty[str(seed)]=dict(source_groups=32,bootstrap_samples=10000,
            development_top1_fraction_difference=sum(deltas)/32,
            paired_source_group_percentile_95_interval=[samples[249],samples[9749]],
            limitation='Previously inspected development groups; observed-family bootstrap does not resolve missing provenance or establish test transfer.')
    old.dump(pilot.OUT/'uncertainty.json',uncertainty)
    for path in sorted(pilot.OUT.rglob('*')):
        if path.is_file() and path.name!='artifact_audit.json':
            hashes[str(path.relative_to(pilot.OUT))]=old.sha(path)
    old.dump(pilot.OUT/'artifact_audit.json',dict(passed=True,updates_total=1600,
        locked_test_predictions=0,accepted_baselines_verified=True,original_study_unchanged=True,
        original_study_artifact_index_sha256=old.sha(old.OUT/'artifact_hashes.json'),artifact_hashes=hashes,
        supporting_file_hashes={name:old.sha(old.ROOT/name) for name in (
            'audit_native_qa_paired_source.py','diagnose_native_qa_paired_branches.py',
            'tests/test_native_qa_paired_source.py','tests/test_native_qa_source_study.py',
            'NATIVE_QA_PAIRED_SOURCE.md')},
        tests=dict(paired_loss_tests=4,source_scoring_regression_tests=6),
        note='The 10 tests were run during implementation; source-only checks do not certify raw-answer retention.'))
    print('PAIRED_ARTIFACT_AUDIT_PASS',uncertainty,flush=True)


if __name__=='__main__':main()
