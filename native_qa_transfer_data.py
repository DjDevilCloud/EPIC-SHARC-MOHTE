"""Conservative source grouping and explicit review for the native QA study."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from native_qa_review import qa_fingerprint


def normalized(text):
    return ' '.join(text.casefold().split())


def fixed_passages(context, max_words=160, min_boundary_words=64):
    """Contiguous, non-overlapping passages selected without questions/answers.

    Prefer paragraph or sentence boundaries after the minimum size. Otherwise
    cut at the word budget. Offsets always address the original unedited text;
    parent family assignment must happen before this function is applied.
    """
    if not 1 <= min_boundary_words <= max_words:
        raise ValueError('Invalid fixed passage word budgets')
    words = list(re.finditer(r'\S+', context))
    result = []
    cursor = 0
    while cursor < len(words):
        stop = min(len(words), cursor + max_words)
        if stop < len(words):
            paragraphs, sentences = [], []
            for j in range(cursor + min_boundary_words, stop + 1):
                gap = context[words[j-1].end():words[j].start()]
                if re.search(r'\n\s*\n', gap):
                    paragraphs.append(j)
                elif re.search(r'[.!?]["\'\)\]]*$', words[j-1].group()):
                    sentences.append(j)
            stop = (paragraphs or sentences or [stop])[-1]
        start_offset, stop_offset = words[cursor].start(), words[stop-1].end()
        result.append(dict(start=start_offset, stop=stop_offset, context=context[start_offset:stop_offset]))
        cursor = stop
    return result


def shingles(text):
    words = re.findall(r'\w+', normalized(text))
    return {tuple(words[i:i+5]) for i in range(len(words)-4)}


def group_sources(contexts, threshold=.8, source_ids=None):
    """Connected components of exact/provenance URLs and shingle containment.

    This supplies text-based groups, not a claim that missing parent provenance
    has been reconstructed. Reviewers must also record related asset groups.
    """
    parents = list(range(len(contexts)))
    def find(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    def union(i, j):
        parents[find(i)] = find(j)
    if source_ids is None:
        source_ids = [None] * len(contexts)
    if len(source_ids) != len(contexts):
        raise ValueError('Source identities must align with contexts')
    inverted, urls, exact, identities = defaultdict(list), {}, {}, {}
    sets = []
    for i, context in enumerate(contexts):
        identity = source_ids[i]
        if identity is not None:
            if identity in identities:
                union(i, identities[identity])
            identities[identity] = i
        key = normalized(context)
        if key in exact:
            union(i, exact[key])
        exact[key] = i
        tokens = shingles(context); sets.append(tokens)
        overlaps = defaultdict(int)
        for token in tokens:
            for j in inverted[token]:
                overlaps[j] += 1
        for j, count in overlaps.items():
            if count / max(1, min(len(tokens), len(sets[j]))) >= threshold:
                union(i, j)
        for token in tokens:
            inverted[token].append(i)
        for url in re.findall(r'https?://[^\s<>"\)]+', context.casefold()):
            url = url.rstrip('/.,;')
            if re.match(r'https?://[^/]+/.+', url):
                if url in urls:
                    union(i, urls[url])
                urls[url] = i
    members = defaultdict(list)
    for i in range(len(contexts)):
        members[find(i)].append(hashlib.sha256(normalized(contexts[i]).encode()).hexdigest())
    keys = {parent: hashlib.sha256('\n'.join(sorted(set(ids))).encode()).hexdigest() for parent, ids in members.items()}
    return [keys[find(i)] for i in range(len(contexts))]


def validate_review(record, review):
    if review.get('fingerprint') != qa_fingerprint(record):
        raise ValueError('Reviewed QA content changed')
    if review.get('decision') not in {'retain', 'quarantine'} or not review.get('reason'):
        raise ValueError('An explicit support decision and reason are required')
    if review['decision'] == 'retain':
        start, stop = review.get('support', (-1, -1))
        if not 0 <= start < stop <= len(record['context']):
            raise ValueError('Retained QA requires valid supporting character offsets')
        if record['answer'] not in record['context'][start:stop]:
            raise ValueError('Supporting excerpt must contain the exact answer')
        a, b = review.get('answer_offsets', (-1, -1))
        if not start <= a < b <= stop or record['context'][a:b] != record['answer']:
            raise ValueError('Answer offsets do not match the reviewed answer')


def regroup_passage_candidates(candidates, excluded_contexts):
    """Close parent families over passage similarities, including old exposure."""
    result = json.loads(json.dumps(candidates))
    records = [r for d in result['documents'] for r in d['records']]
    labels = group_sources([r['context'] for r in records] + excluded_contexts,
                          source_ids=[r['source_group'] for r in records] + [None] * len(excluded_contexts))
    exposed = set(labels[len(records):])
    groups = {}
    aliases = {}
    for record, family in zip(records, labels):
        original = record['source_group']
        aliases[original] = family
        if family in exposed:
            continue
        record['original_source_group'] = original
        record['source_group'] = family
        entry = groups.setdefault(family, dict(source_group=family, records=[]))
        if not any(qa_fingerprint(r) == qa_fingerprint(record) for r in entry['records']):
            entry['records'].append(record)
    for parent in result['parent_inventory'].values():
        original = parent['group']
        parent['original_group'] = original
        if original in aliases:
            parent['group'] = aliases[original]
    paired = []
    for document in groups.values():
        contexts = {}
        for record in document['records']:
            contexts.setdefault(record['context'], []).append(record)
        choices = [rs for rs in contexts.values() if len({r['answer'] for r in rs}) >= 2]
        if choices:
            choices.sort(key=lambda rs: hashlib.sha256(rs[0]['context'].encode()).hexdigest())
            first = choices[0]
            document['records'] = first + [r for r in document['records'] if r not in first]
            paired.append(document)
    paired.sort(key=lambda d: hashlib.sha256(('native-passages-v2-review:' + d['source_group']).encode()).hexdigest())
    result['documents'] = paired
    result['inventory']['paired_groups_before_passage_overlap_check'] = candidates['inventory']['paired_groups']
    result['inventory']['paired_groups'] = len(paired)
    result['inventory']['groups_removed_or_merged_by_passage_overlap'] = len(candidates['documents']) - len(paired)
    result['partition']['passage_overlap_closes_parent_families'] = True
    return result


def freeze_dataset(candidates, reviews, output):
    """Freeze only explicitly reviewed pairs. Never infer support from occurrence."""
    approved = []
    for document in candidates['documents']:
        retained = []
        for record in document['records']:
            key = qa_fingerprint(record)
            if key not in reviews:
                continue
            review = reviews[key]
            validate_review(record, review)
            if review['decision'] == 'retain':
                retained.append(dict(record, review=review))
        pair = next(([a, b] for i, a in enumerate(retained) for b in retained[i+1:]
                     if a['context'] == b['context'] and a['answer'] != b['answer']), None)
        if pair is not None:
            approved.append((document['source_group'], pair))
    # Related asset grouping supplied by review is applied before allocation.
    used = set(); unique = []
    for group, records in approved:
        family = records[0]['review'].get('related_family', group)
        if family in used:
            continue
        used.add(family)
        for record in records:
            record['passage_source_group'] = record['source_group']
            record['source_group'] = family
        unique.append(records)
    if len(unique) < 128:
        raise ValueError(f'Insufficient reviewed source groups: {len(unique)}/128; do not weaken review')
    salt = 'native-qa-transfer-passages-v2:' if candidates.get('version') == 2 else 'native-qa-transfer-v1:'
    unique.sort(key=lambda rs: hashlib.sha256((salt + rs[0]['source_group']).encode()).hexdigest())
    result = dict(version=candidates.get('version', 1), source_sha256=candidates['source_sha256'], provenance_limitations=candidates['provenance_limitations'],
                  grouping_threshold=.8, selection='Reviewed contextual support and representation only; no model scores', splits={})
    if candidates.get('version') == 2:
        result.update(task=candidates['task'], partition=candidates['partition'],
                      exclusion_sha256=candidates['exclusion_sha256'],
                      manual_family_links=candidates.get('manual_family_links', []),
                      candidate_fingerprint=hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest())
    for split, documents in (('train', unique[:64]), ('development', unique[64:96]), ('test', unique[96:128])):
        result['splits'][split] = [r for pair in documents for r in pair]
    target = Path(output)
    if target.exists():
        raise FileExistsError('Frozen data must not be replaced')
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
