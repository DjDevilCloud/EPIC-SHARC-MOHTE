"""Apply explicit, fingerprinted QA annotation reviews before adapter fitting."""
import hashlib
import json


def qa_fingerprint(record):
    fields = {key: record[key] for key in ('context', 'question', 'answer')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def reviewed_indices(records, decisions):
    """Select retained rows without relabeling or changing the source manifest.

    This checks review identity and coverage, not semantic correctness. Reviews
    must assess the supplied context separately; literal answer occurrence is
    insufficient evidence that a question's answer is supported.
    """
    if len(records) != len(decisions):
        raise ValueError('Every QA record requires an explicit review decision')
    retained = []
    for index, (record, decision) in enumerate(zip(records, decisions)):
        if decision.get('index') != index or decision.get('fingerprint') != qa_fingerprint(record):
            raise ValueError('QA review does not match the current record')
        if decision.get('decision') not in {'retain', 'quarantine'}:
            raise ValueError('Unknown QA review decision')
        if not decision.get('reason'):
            raise ValueError('QA review requires a reason')
        if decision['decision'] == 'retain':
            retained.append(index)
    return retained
