"""One serializer for native document QA, using existing codec boundaries."""
import re


def format_native_qa(context, question, answer=None, *, framing='legacy'):
    if framing not in {'legacy', 'blocks_v1'}:
        raise ValueError('Unknown native QA framing policy')
    if not isinstance(context, str) or not isinstance(question, str) or not question.strip():
        raise ValueError('Native QA requires source text and a nonempty question')
    if answer is not None and not isinstance(answer, str):
        raise ValueError('Native QA answer must be text')
    if framing == 'blocks_v1':
        # Codec controls are reserved, rather than silently interpreted as
        # literal document text. Ordinary header-like words need no escaping.
        for value in (context, question, answer or ''):
            if re.search(r'<(?:BOI|EOI|BOO|EOO|BOP|EOP|BLO|LINE|EOL|CAP|UPPER|SIG:[^>]+)>', value):
                raise ValueError('Native QA fields contain reserved codec control text')
        prefix = f'<BOI><BOP>{context}<EOP><BOP>{question}<EOP><EOI><BOO>'
    else:
        prefix = f'<BOI>Context:\n{context}\nQuestion:\n{question}<EOI><BOO>'
    return prefix if answer is None else prefix + answer + '<EOO>'
