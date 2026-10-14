"""Learned native span extent and prefix-verified source continuation."""
import math
from collections import OrderedDict
import torch
from torch import nn
try:
    from .source_answer_span import CompleteAnswerSpanPredictor
except ImportError:
    from source_answer_span import CompleteAnswerSpanPredictor


class NativeSpanReadout(nn.Module):
    def __init__(self, width, max_words=16, starts=16, stable_gate=False):
        super().__init__()
        if max_words < 1 or starts < 1:
            raise ValueError('Native span budgets must be positive.')
        self.max_words = int(max_words)
        self.starts = int(starts)
        self.stable_gate = bool(stable_gate)
        self.endpoint = nn.Sequential(nn.Linear(4 * width + 12, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.endpoint[-1].weight)
        nn.init.zeros_(self.endpoint[-1].bias)
        self.copy_gate = nn.Linear(2 * width, 1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, -8.)
        self._candidates = OrderedDict()
        self._payloads = OrderedDict()

    def candidates(self, state, tokenizer, selected):
        last = max((stop for _, stop, _ in state.source_ranges), default=0)
        key = (id(tokenizer), tokenizer.vocab_size, tuple((p, i) for p, i in state.surface if p <= last + 4), tuple(state.source_ranges), tuple(selected))
        if key in self._candidates:
            self._candidates.move_to_end(key)
            return self._candidates[key]
        raw = dict(state.surface)
        def unit(position):
            return tokenizer.construction_units[raw[position]] if position in raw else None
        def boundary(u):
            return u is None or u.is_lexical or u.kind == 'signature' or u.text in {'<SPACE>', '<TAB>', '<EOL>', '<LINE>', '<BOI>', '<EOI>', '<BOP>', '<EOP>', '<BLO>'} or bool(u.render and u.render.isspace())
        result = []
        for start in selected:
            first, _, complete = state.source_ranges[start]
            if not complete or first not in raw:
                continue
            leading = first
            while first - leading < 4 and not boundary(unit(leading - 1)):
                leading -= 1
            for end in range(start, min(len(state.source), start + self.max_words)):
                _, stop, end_complete = state.source_ranges[end]
                if not end_complete or stop - 1 not in raw:
                    break
                if any(unit(p) is None or unit(p).text in {'<EOL>', '<BLO>', '<EOI>'} for p in range(first, stop)):
                    break
                tail = stop
                while tail - stop < 3 and not boundary(unit(tail)) and unit(tail).kind != 'case':
                    tail += 1
                for begin in range(leading, first + 1):
                    for finish in range(stop, tail + 1):
                        text = tokenizer.decode([raw[p] for p in range(begin, finish)], clean_text=False, collapse_structure=False).strip()
                        if not text:
                            continue
                        cache_key = (id(tokenizer), tokenizer.vocab_size, text)
                        if cache_key not in self._payloads:
                            self._payloads[cache_key] = CompleteAnswerSpanPredictor.payload(tokenizer, text)
                            if len(self._payloads) > 8192:
                                self._payloads.popitem(last=False)
                        next_unit = unit(stop)
                        prefix = [unit(p) for p in range(begin, first)]
                        last_text = tokenizer.decode(list(state.source[end][0]), clean_text=False, collapse_structure=False)
                        numeric_join = bool(next_unit is not None and next_unit.render in {'.', ',', '/', '-'} and unit(stop + 1) is not None and unit(stop + 1).kind == 'digit' and unit(stop - 1).kind == 'digit')
                        next_case = any(unit(p) is not None and unit(p).kind == 'case' for p in range(stop, min(stop + 8, state.source_ranges[end + 1][0] + 1))) if end + 1 < len(state.source_ranges) else False
                        start_join = bool(unit(first - 1) is not None and unit(first - 1).render in {'.', ',', '/', '-'} and unit(first - 2) is not None and unit(first - 2).kind == 'digit' and unit(first).kind == 'digit')
                        features = ((end - start + 1) / self.max_words, (first - begin) / 4, (finish - stop) / 3,
                                    float(next_unit is None or next_unit.text in {'<EOL>', '<BLO>'}),
                                    float(next_unit is not None and next_unit.render in '.!?'),
                                    float(any(u.kind == 'case' for u in prefix)),
                                    float(any(u.render in {'$', '+', '-'} for u in prefix)),
                                    float(any(u.render in {'"', "'", '('} for u in prefix)),
                                    float(numeric_join), float(len(last_text) == 1 and last_text.isalpha() and next_unit is not None and next_unit.render == '.'), float(next_case), float(start_join))
                        result.append(dict(start=start, end=end, text=text, tokens=self._payloads[cache_key], properties=features))
        self._candidates[key] = result
        if len(self._candidates) > 16:
            self._candidates.popitem(last=False)
        return result

    def plan(self, state, tokenizer, question, word_vectors, start_probabilities, start_log_probabilities):
        selected = start_probabilities.detach().topk(min(len(state.source), self.starts)).indices.cpu().tolist()
        candidates = self.candidates(state, tokenizer, selected)
        if not candidates:
            return None
        starts = torch.tensor([c['start'] for c in candidates], device=question.device)
        ends = torch.tensor([c['end'] for c in candidates], device=question.device)
        prefix = torch.cat((word_vectors.new_zeros(1, word_vectors.shape[1]), word_vectors.cumsum(0)))
        means = (prefix[ends + 1] - prefix[starts]) / (ends - starts + 1)[:, None]
        properties = question.new_tensor([c['properties'] for c in candidates])
        features = torch.cat((question.expand(len(candidates), -1), word_vectors[starts], word_vectors[ends], means, properties), -1)
        logits = start_log_probabilities[starts] + self.endpoint(features).squeeze(-1)
        return dict(candidates=candidates, probabilities=logits.softmax(0), log_probabilities=logits.log_softmax(0), source_words=[word for word, _ in state.source], start_log_probabilities=start_log_probabilities)

    def mix(self, logits, records):
        if not records:
            return logits
        # Keep non-output positions untouched. Copy support retains its original
        # plan probability: a low-probability surviving prefix cannot force a span.
        result = logits.clone()
        for row, position, question, current, plan, emitted in records:
            supported = [(i, c['tokens'][len(emitted)]) for i, c in enumerate(plan['candidates'])
                         if len(emitted) < len(c['tokens']) and c['tokens'][:len(emitted)] == emitted]
            if not supported:
                continue
            indices = torch.tensor([i for i, _ in supported], device=logits.device)
            targets = torch.tensor([token for _, token in supported], device=logits.device)
            mass = plan['probabilities'][indices]
            support = mass.sum()
            copy = logits.new_zeros(logits.shape[-1]).scatter_add(0, targets, mass) / support.clamp_min(1e-12)
            context = plan['copy_context'] if self.stable_gate else current
            gate = self.copy_gate(torch.cat((context, question))).sigmoid().squeeze() * support
            distribution = (1 - gate) * logits[row, position].softmax(-1) + gate * copy
            result[row, position] = distribution.clamp_min(1e-12).log()
        return result
