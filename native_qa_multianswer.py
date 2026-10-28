"""V4 reviewed, evidence-grounded answer sets; never used as inference input."""
import hashlib
import json
import re
from pathlib import Path

from native_qa_review import qa_fingerprint
from native_qa_objective_contract import ROOT, OLD, verify_archive

OUT = ROOT / 'review_artifacts/native_qa_multianswer_contract_v4'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def rows(split):
    if split not in ('train', 'development'):
        raise ValueError('Locked-test questions cannot enter the answer overlay')
    return read(OLD / 'data/frozen_manifest.json')['splits'][split]


def intervals(text, needle):
    """Exact surface occurrences, excluding substrings of larger lexical words."""
    expression = (r'(?<!\w)' if needle[:1].isalnum() else '') + re.escape(needle)
    expression += r'(?!\w)' if needle[-1:].isalnum() else ''
    return [[m.start(), m.end()] for m in re.finditer(expression, text)]


def sentence_ranges(text):
    """Evidence layout only; explicit reviewer anchors choose the evidence.

    Preserve initials, common title/date abbreviations and decimal numbers.
    Blank lines also delimit table fields and headings. This is not an entailment
    classifier: an anchor by itself never approves a record.
    """
    boundaries = {0, len(text)}
    abbreviations = {'dr','mr','mrs','ms','st','jr','sr','gov','sgt','cfr','u.s','a.m','p.m',
                     'jan','feb','mar','apr','jun','jul','aug','sep','sept','oct','nov','dec','rel','calif'}
    for match in re.finditer(r'\n\s*\n|[.!?]["\u201d\u2019]*\s+', text):
        if match.group().startswith('\n'):
            boundaries.update((match.start(), match.end()))
            continue
        before = text[:match.start()]
        previous = re.search(r'([\w.]+)$', before)
        token = previous.group(1).casefold().strip('.') if previous else ''
        if text[match.start()] == '.' and (len(token) == 1 or token in abbreviations):
            continue
        boundaries.add(match.end())
    points = sorted(boundaries)
    return [[a,b] for a,b in zip(points, points[1:]) if text[a:b].strip()]


def anchor_support(context, anchor):
    ordinal = 0
    explicit_ordinal = isinstance(anchor, (list, tuple))
    if isinstance(anchor, (list, tuple)):
        anchor, ordinal = anchor
    matches = list(re.finditer(re.escape(anchor), context))
    if not matches or ordinal >= len(matches):
        raise ValueError(f'Reviewer anchor absent: {anchor!r}')
    if len(matches) != 1 and not explicit_ordinal:
        raise ValueError('An ambiguous anchor requires an ordinal')
    match = matches[ordinal]
    ranges = sentence_ranges(context)
    containing = [[a,b] for a,b in ranges if a < match.end() and match.start() < b]
    if not containing:
        raise ValueError('No evidence layout for anchor')
    return [min(a for a,b in containing), max(b for a,b in containing)]


def validate_entry(row, entry):
    if entry['fingerprint'] != qa_fingerprint(row) or entry['original_answer'] != row['answer']:
        raise ValueError('Original question/context/answer identity changed')
    if not entry.get('reason') or entry.get('decision') != 'retain':
        raise ValueError('Explicit contextual support review required')
    if not entry['support']:
        raise ValueError('Evidence required')
    for a,b in entry['support']:
        if not 0 <= a < b <= len(row['context']):
            raise ValueError('Evidence interval out of range')
    options = entry['answers']
    if not options or options[0]['text'] != row['answer']:
        raise ValueError('The original answer must remain the primary reference')
    if len({r['text'] for r in options}) != len(options):
        raise ValueError('Duplicate answer reference')
    for option in options:
        if not option.get('reason') or not option['occurrences']:
            raise ValueError('Each answer requires a reason and grounded occurrences')
        if len({tuple(p) for p in option['occurrences']}) != len(option['occurrences']):
            raise ValueError('Duplicate source occurrence')
        for a,b in option['occurrences']:
            if not 0 <= a < b <= len(row['context']) or row['context'][a:b] != option['text']:
                raise ValueError('Answer must be an unchanged contiguous source span')
            if not any(c <= a < b <= d for c,d in entry['support']):
                raise ValueError('Answer occurrence is outside reviewed evidence')
    return entry


def verify_history():
    verify_archive()
    v3 = ROOT / 'review_artifacts/native_qa_objective_contract_v3'
    audit = read(v3/'artifact_audit.json')
    for name, expected in audit['artifact_hashes'].items():
        if sha(v3/name) != expected:
            raise ValueError(f'Historical v3 artifact changed: {name}')
    for name, expected in audit['implementation_files'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Historical v3 implementation changed: {name}')
    return True


def verify_overlay(path=None):
    """Pure verification, including complete coverage and original split identity."""
    path = path or OUT/'reviewed_answers.json'
    overlay = read(path)
    if overlay['source_manifest_sha256'] != sha(OLD/'data/frozen_manifest.json'):
        raise ValueError('Original manifest changed')
    expected = {}
    for split, count in [('train',128),('development',64)]:
        source = rows(split)
        if len(source) != count:
            raise ValueError('Split denominator changed')
        for index, row in enumerate(source):
            key = qa_fingerprint(row)
            expected[key] = (split,index,row)
    if set(overlay['decisions']) != set(expected):
        raise ValueError('Overlay must cover exactly all training/development questions')
    for key, (split,index,row) in expected.items():
        entry = overlay['decisions'][key]
        if (entry['split'],entry['index'],entry['source_group']) != (split,index,row['source_group']):
            raise ValueError('Split, index or source family changed')
        validate_entry(row,entry)
    return overlay


def accepted_texts(entry):
    return tuple(option['text'] for option in entry['answers'])


def score_text(raw, ended, entry):
    """Evaluator-only exact surfaces; never normalize punctuation or case."""
    text = raw.strip()
    primary = text == entry['original_answer']
    reviewed = text in accepted_texts(entry)
    return dict(original_single_gold_exact=primary, reviewed_multianswer_exact=reviewed,
                ended=bool(ended), reviewed_exact_and_terminated=bool(reviewed and ended))


def bind_native_occurrences(row, entry, memory, tokenizer, *, max_total_tokens=2048):
    """Bind reviewed character locations to native addresses, without scoring.

    Parsing a prefix ending immediately before a reviewed span counts preceding
    native words. Candidate verification rejects partial-word matches and any
    span not actually representable at that exact occurrence. Nothing here
    receives model logits or selects annotations based on model correctness.
    """
    from native_question_memory import NativeQuestionState
    from source_answer_span import CompleteAnswerSpanPredictor
    validate_entry(row,entry)
    def observe(tokens):
        state = NativeQuestionState()
        for token in tokens:
            state,_ = memory._observe(state,token,tokenizer)
        return state
    prompt = f"Context:\n{row['context']}\nQuestion:\n{row['question']}"
    prompt_tokens = tokenizer.prepare_generation_hierarchy(prompt).token_ids
    state = observe(prompt_tokens)
    question = observe(tokenizer.encode('<BOI>Context:\n'+row['question']+'<EOI>',add_special_tokens=False))
    if len(question.source)>memory.question_capacity:
        raise ValueError('Question exceeds native question capacity')
    result = []
    unavailable = []
    for option in entry['answers']:
        option_count = len(result)
        payload = CompleteAnswerSpanPredictor.payload(tokenizer,option['text'])
        if len(payload)>128 or (max_total_tokens is not None and len(prompt_tokens)+len(payload)>max_total_tokens):
            raise ValueError('Reference exceeds the fixed generation/fitting limits')
        for a,b in option['occurrences']:
            prefix = observe(tokenizer.encode('<BOI>Context:\n'+row['context'][:a]+'<EOI>',add_special_tokens=False))
            start = len(prefix.source)
            if start>=len(state.source):
                raise ValueError('Reviewed occurrence has no retained native start')
            candidates = memory.span_readout.candidates(state,tokenizer,[start])
            exact = [c for c in candidates if c['text']==option['text'] and tuple(c['tokens'])==payload]
            if not exact:
                unavailable.append(dict(text=option['text'],character_offsets=[a,b],start=start,
                    reason='Supported character span is not a complete native candidate at this occurrence.'))
                continue
            result.append(dict(text=option['text'],character_offsets=[a,b],start=start,
                               ends=sorted({c['end'] for c in exact}),payload=list(payload)))
        if len(result)==option_count:
            raise ValueError(f'Accepted answer has no copyable supported occurrence: {option["text"]!r}')
    return dict(occurrences=result,source_words=len(state.source),question_words=len(question.source),
                unavailable_supported_occurrences=unavailable,
                prompt_tokens=len(prompt_tokens),max_answer_tokens=max(len(x['payload']) for x in result))
