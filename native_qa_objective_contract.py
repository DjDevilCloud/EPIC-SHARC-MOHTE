"""V3 prerequisite archive and support review; verification never writes files.

Training is deliberately unavailable until every training/development support
annotation passes. Historical studies and their locked split remain immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path

from native_qa_review import qa_fingerprint

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'review_artifacts/native_qa_transfer_passages_v2'
OUT = ROOT / 'review_artifacts/native_qa_objective_contract_v3'
BASELINES = {
    47: 'a0c3ecefe77c0a1346063c63990f442c264ce7e1f1f801df63f528073b3e0358',
    83: 'e1093251728cf100e32b75f2eb7718bc8e43a20a836005614affd3ca0206c35f',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to replace {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def historical_bindings():
    """Verify declared historical dependencies without invoking legacy writers."""
    paths = {}
    def bind(path, expected):
        path = Path(path)
        if sha(path) != expected:
            raise ValueError(f'Historical binding changed: {path}')
        paths[str(path.relative_to(ROOT))] = expected
    for seed, expected in BASELINES.items():
        bind(ROOT / f'review_artifacts/native_owner_markers/seed{seed}/candidate/model.pt', expected)
    study = OLD / 'source_study'
    protocol = read(study / 'frozen_experiment.json')
    for name, expected in protocol['code_hashes'].items():
        bind(ROOT / name, expected)
    for name, expected in protocol['file_hashes'].items():
        bind(ROOT / name, expected)
    for seed, expected in protocol['checkpoint_hashes'].items():
        bind(ROOT / f'review_artifacts/native_retained_decisions_counterfactual/seed{seed}/candidate/model.pt', expected)
    for name, expected in read(study / 'input_binding.json')['additional_dependencies'].items():
        bind(ROOT / name, expected)
    for name, expected in read(study / 'artifact_hashes.json').items():
        bind(OLD / name, expected)
    paired = OLD / 'paired_source_pilot'
    audit = read(paired / 'artifact_audit.json')
    for name, expected in audit['artifact_hashes'].items():
        bind(paired / name, expected)
    for name, expected in audit['supporting_file_hashes'].items():
        bind(ROOT / name, expected)
    # Bind every byte of the frozen dataset without loading locked questions.
    for name, expected in read(OLD / 'data_gate.json')['artifact_hashes'].items():
        bind(OLD / 'data' / name, expected)
    return paths


def archive():
    if (OUT / 'history/archive.json').exists():
        return verify_archive()
    paths = historical_bindings()
    selected = set(ROOT.glob('*.py')) | set(ROOT.glob('*.md')) | set((ROOT / 'tests').glob('*.py'))
    selected.update(ROOT / name for name in paths if not name.endswith('.pt'))
    for path in [OLD / 'source_study/frozen_experiment.json', OLD / 'source_study/input_binding.json',
                 OLD / 'source_study/execution_input_seal.json', OLD / 'source_study/artifact_hashes.json',
                 OLD / 'paired_source_pilot/protocol.json', OLD / 'paired_source_pilot/artifact_audit.json',
                 OLD / 'data/frozen_manifest.json', OLD / 'data/reviews.json', OLD / 'data_gate.json']:
        selected.add(path)
    snapshots = {}
    for source in sorted(selected):
        relative = source.relative_to(ROOT)
        target = OUT / 'history/files' / relative
        digest = sha(source)
        if target.exists():
            if sha(target) != digest:
                raise ValueError(f'Partial archive conflicts with source: {source}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        snapshots[str(relative)] = digest
    result = dict(version=3, purpose='Archive before any production changes',
                  historical_bindings=paths, snapshots=snapshots,
                  large_checkpoints='Original checkpoint files are hash-bound, not duplicated.',
                  locked_test_predictions=0)
    write_new(OUT / 'history/archive.json', result)
    return verify_archive()


def verify_archive():
    """Pure read-only verifier; safe after production code changes."""
    report = read(OUT / 'history/archive.json')
    for name, expected in report['snapshots'].items():
        if sha(OUT / 'history/files' / name) != expected:
            raise ValueError(f'Archived dependency changed: {name}')
        if name.replace('\\', '/').startswith('review_artifacts/') and sha(ROOT / name) != expected:
            raise ValueError(f'Original historical evidence changed: {name}')
    for name, expected in report['historical_bindings'].items():
        # Production dependencies can subsequently change; their archived
        # versions, historical artifacts and protected checkpoints cannot.
        original = ROOT / name
        if name.replace('\\', '/').startswith('review_artifacts/') and sha(original) != expected:
            raise ValueError(f'Historical artifact changed: {name}')
    return dict(passed=True, snapshots=len(report['snapshots']),
                historical_bindings=len(report['historical_bindings']), locked_test_predictions=0)


def review_rows(split):
    if split not in ('train', 'development'):
        raise ValueError('The locked test is not part of support review')
    return read(OLD / 'data/frozen_manifest.json')['splits'][split]


def validate_support(row, decision):
    if decision.get('fingerprint') != qa_fingerprint(row):
        raise ValueError('Support review fingerprint mismatch')
    if decision.get('decision') not in ('retain', 'unsupported', 'ambiguous') or not decision.get('reason'):
        raise ValueError('An explicit support decision and reason are required')
    supports = decision.get('support', [])
    for a, b in supports:
        if not 0 <= a < b <= len(row['context']):
            raise ValueError('Support interval is outside the original passage')
    if 'evidence_quote' in decision:
        if len(supports) != 1 or row['context'][supports[0][0]:supports[0][1]] != decision['evidence_quote']:
            raise ValueError('Evidence quotation does not match its recorded passage interval')
    alternatives = decision.get('diagnostic_valid_alternatives')
    if alternatives is not None:
        if len({r['text'] for r in alternatives}) < 2:
            raise ValueError('An alternatives diagnostic needs at least two distinct answers')
        for alternative in alternatives:
            a, b = alternative['offsets']
            if not 0 <= a < b <= len(row['context']) or row['context'][a:b] != alternative['text']:
                raise ValueError('Alternative-answer diagnostic does not match the passage')
            if not any(c <= a < b <= d for c, d in supports):
                raise ValueError('Alternative-answer diagnostic is outside the evidence')
    if decision['decision'] != 'retain':
        if decision.get('answer_occurrences'):
            raise ValueError('Invalid or ambiguous questions cannot receive positive source labels')
        return
    occurrences = decision.get('answer_occurrences', [])
    if not supports or not occurrences:
        raise ValueError('Retained questions require support and supported answer occurrences')
    for a, b in occurrences:
        if row['context'][a:b] != row['answer'] or not any(c <= a < b <= d for c, d in supports):
            raise ValueError('Supported answer occurrence must exactly match within evidence')
    if len({tuple(p) for p in occurrences}) != len(occurrences):
        raise ValueError('Duplicate answer occurrence')


def review_gate():
    """Read-only result; missing or invalid annotations block optimization."""
    verify_archive()
    path = OUT / 'support_overlay.json'
    overlay = read(path) if path.exists() else None
    if overlay is not None and overlay.get('source_manifest_sha256') != sha(OLD / 'data/frozen_manifest.json'):
        raise ValueError('Support overlay references a changed source manifest')
    decisions = overlay['decisions'] if overlay is not None else {}
    expected = {}
    for split, count in (('train', 128), ('development', 64)):
        rows = review_rows(split)
        if len(rows) != count:
            raise ValueError('Original review denominator changed')
        for row in rows:
            key = qa_fingerprint(row)
            if key in expected:
                raise ValueError('Duplicate review record')
            expected[key] = row
    if set(decisions) - set(expected):
        raise ValueError('Overlay includes unknown or locked questions')
    failures = []
    for key, decision in decisions.items():
        validate_support(expected[key], decision)
        if decision['decision'] != 'retain':
            failures.append(dict(decision, question=expected[key]['question']))
    missing = sorted(set(expected) - set(decisions))
    return dict(passed=not missing and not failures, required=192, reviewed=len(decisions),
                missing=len(missing), failures=failures, optimization_updates=0,
                locked_test_predictions=0, source_manifest_sha256=sha(OLD / 'data/frozen_manifest.json'))


def finalize_prerequisites():
    """Write a new study decision, never alter a previous gate or report."""
    gate = review_gate()
    if gate['failures']:
        status = 'STOPPED_SUPPORT_REVIEW_FAILED'
    elif gate['missing']:
        status = 'SUPPORT_REVIEW_INCOMPLETE'
    else:
        status = 'SUPPORT_REVIEW_PASSED_EXECUTION_NOT_RUN'
    result = dict(version=3, status=status, accepted=False, baseline_replaced=False,
                  training_authorized=False, optimization_updates=0, locked_test_predictions=0,
                  source_manifest_sha256=gate['source_manifest_sha256'],
                  support_overlay_sha256=sha(OUT / 'support_overlay.json') if (OUT / 'support_overlay.json').exists() else None,
                  historical_archive_sha256=sha(OUT / 'history/archive.json'),
                  annotation_gate=gate, original_denominators=dict(train=128, development=64, locked_test=64),
                  excluded_or_replaced_records=0, answers_or_questions_changed=False,
                  maximum_planned_updates=19200,
                  stages={name:'not_run_support_prerequisite_not_passed' for name in (
                      'generation_profile', 'functional_replay', 'execution_contract', 'source',
                      'fusion', 'extent', 'joint', 'locked_test')},
                  stop_instruction='If support review invalidates a question or leaves an ambiguity unresolved, stop before training rather than silently changing denominators or replacing records.')
    write_new(OUT / 'acceptance.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['archive', 'verify', 'review-gate', 'finalize-prerequisites'])
    args = parser.parse_args(argv)
    if args.phase == 'archive':
        result = archive()
    elif args.phase == 'verify':
        result = verify_archive()
    elif args.phase == 'review-gate':
        result = review_gate()
    else:
        result = finalize_prerequisites()
    print(json.dumps(result, indent=2))
    return 2 if args.phase == 'review-gate' and not result['passed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
