"""Learned native span extent and prefix-verified source continuation."""
import math
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
try:
    from .source_answer_span import CompleteAnswerSpanPredictor
except ImportError:
    from source_answer_span import CompleteAnswerSpanPredictor


class NativeSpanReadout(nn.Module):
    def __init__(self, width, max_words=16, starts=16, stable_gate=False, conditional_endpoints=False, marker_endpoints=False, conditional_copy=False, confidence_context='retrieved_v1'):
        super().__init__()
        if max_words < 1 or starts < 1:
            raise ValueError('Native span budgets must be positive.')
        self.max_words = int(max_words)
        self.starts = int(starts)
        self.stable_gate = bool(stable_gate)
        if confidence_context not in {'retrieved_v1','observed_mean_v1'}:
            raise ValueError('Unknown native copy confidence context policy')
        if confidence_context != 'retrieved_v1' and not (stable_gate and conditional_copy):
            raise ValueError('Observed copy confidence requires stable conditional copying')
        self.confidence_context = confidence_context
        if conditional_copy and not stable_gate:
            raise ValueError('Conditional copy requires a stable task gate')
        self.conditional_copy = bool(conditional_copy)
        self.conditional_endpoints = bool(conditional_endpoints)
        self.endpoint = nn.Sequential(nn.Linear(4 * width + 12, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.endpoint[-1].weight)
        nn.init.zeros_(self.endpoint[-1].bias)
        self.marker_query = nn.Linear(width, 1) if marker_endpoints else None
        self.marker_endpoint = nn.Sequential(nn.Linear(2 * width + 8, 2 * width), nn.GELU(), nn.Linear(2 * width, 1)) if marker_endpoints else None
        if self.marker_endpoint is not None:
            nn.init.zeros_(self.marker_query.weight)
            nn.init.zeros_(self.marker_query.bias)
            nn.init.zeros_(self.marker_endpoint[-1].weight)
            nn.init.zeros_(self.marker_endpoint[-1].bias)
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
                        suffix = [unit(p) for p in range(stop, finish)]
                        markers = (float(any(u.render in {'%', '‰'} for u in suffix)),
                                   float(next_unit is not None and next_unit.render in {'%', '‰'}),
                                   float(any(u.render in {'.', '!', '?'} for u in suffix)),
                                   float(any(u.render in {')', ']', '}', '"', "'"} for u in suffix)),
                                   float(any(u.render in {'$', '€', '£', '¥'} for u in prefix)),
                                   float(unit(stop - 1).kind == 'digit'), float(numeric_join), float(start_join))
                        result.append(dict(start=start, end=end, text=text, tokens=self._payloads[cache_key], properties=features, marker_properties=markers))
        self._candidates[key] = result
        if len(self._candidates) > 16:
            self._candidates.popitem(last=False)
        return result

    def plan(self, state, tokenizer, question, word_vectors, start_probabilities, start_log_probabilities, marker_words=None):
        selected = start_probabilities.detach().topk(min(len(state.source), self.starts)).indices.cpu().tolist()
        candidates = self.candidates(state, tokenizer, selected)
        if not candidates:
            return None
        logits = self.score_candidates(candidates, question, word_vectors, start_log_probabilities, marker_words=marker_words)
        return dict(candidates=candidates, probabilities=logits.softmax(0), log_probabilities=logits.log_softmax(0), source_words=[word for word, _ in state.source], start_log_probabilities=start_log_probabilities, word_vectors=word_vectors, marker_words=marker_words)

    def score_candidates(self, candidates, question, word_vectors, start_log_probabilities, marker_words=None):
        """One scorer for inferred spans and explicit training/diagnostic labels."""
        starts = torch.tensor([c['start'] for c in candidates], device=question.device)
        ends = torch.tensor([c['end'] for c in candidates], device=question.device)
        prefix = torch.cat((word_vectors.new_zeros(1, word_vectors.shape[1]), word_vectors.cumsum(0)))
        means = (prefix[ends + 1] - prefix[starts]) / (ends - starts + 1)[:, None]
        properties = question.new_tensor([c['properties'] for c in candidates])
        features = torch.cat((question.expand(len(candidates), -1), word_vectors[starts], word_vectors[ends], means, properties), -1)
        energies = self.endpoint(features).squeeze(-1)
        if self.marker_endpoint is not None:
            if marker_words is None:
                raise ValueError('Marker endpoint scoring requires complete question-word features')
            cue = self.marker_query(marker_words).squeeze(-1).softmax(0) @ marker_words
            markers = question.new_tensor([c.get('marker_properties', (0.,) * 8) for c in candidates])
            energies = energies + self.marker_endpoint(torch.cat((question.expand(len(candidates), -1), cue.expand(len(candidates), -1), markers), -1)).squeeze(-1)
        if self.conditional_endpoints:
            # Endpoint supervision is conditional on a source start. Its arbitrary
            # per-start energy offset must not override the learned start prior.
            normalizers = torch.zeros_like(energies)
            for start in starts.unique():
                mask = starts == start
                normalizers[mask] = energies[mask].logsumexp(0)
            energies = energies - normalizers
        logits = start_log_probabilities[starts] + energies
        return logits

    def gate_logit(self, question, current, plan):
        """The same task-confidence decision for live emission and replay."""
        if self.confidence_context == 'observed_mean_v1':
            if 'copy_gate_context' not in plan:
                raise ValueError('Observed copy confidence requires its input-only source summary')
            context = plan['copy_gate_context']
        else:
            context = plan['copy_context'] if self.stable_gate else current
        return self.copy_gate(torch.cat((context, question))).squeeze()

    def mix(self, logits, records, diagnostics=None):
        if not records:
            return logits
        # Keep non-output positions untouched. Legacy mode retains original
        # support mass; conditional mode separates task confidence from prefix
        # uncertainty, with the same stable task gate throughout the answer.
        result = logits.clone()
        for row, position, question, current, plan, emitted in records:
            diagnostic = None
            if diagnostics is not None:
                diagnostic = dict(row=row, position=position, emitted_prefix=emitted,
                                  source=plan.get('diagnostics'), surviving_prefix_mass=0., copy_gate=None)
                diagnostics.append(diagnostic)
            supported = [(i, c['tokens'][len(emitted)]) for i, c in enumerate(plan['candidates'])
                         if len(emitted) < len(c['tokens']) and c['tokens'][:len(emitted)] == emitted]
            if not supported:
                continue
            indices = torch.tensor([i for i, _ in supported], device=logits.device)
            targets = torch.tensor([token for _, token in supported], device=logits.device)
            log_weights = plan.get('log_probabilities')
            if log_weights is None:
                weights = plan['probabilities']
                positive = weights > 0
                log_weights = weights.new_full(weights.shape, -torch.inf)
                log_weights[positive] = weights[positive].log()
            selected = log_weights[indices]
            finite = torch.isfinite(selected)
            if not finite.any():
                continue
            selected, targets = selected[finite], targets[finite]
            if diagnostic is not None:
                diagnostic['surviving_prefix_mass'] = float(selected.logsumexp(0).exp().detach())
            if self.conditional_copy:
                # P(next | copying, verified emitted prefix). Rejected spans
                # express continuation uncertainty, not native-task evidence.
                selected = selected - selected.logsumexp(0)
            unique, inverse = targets.unique(return_inverse=True)
            maxima = selected.new_full((len(unique),), -torch.inf).scatter_reduce(0, inverse, selected, reduce='amax', include_self=True)
            totals = selected.new_zeros(len(unique)).scatter_add(0, inverse, (selected - maxima[inverse]).exp())
            log_copy = logits.new_full((logits.shape[-1],), -torch.inf).scatter(0, unique, maxima + totals.log())
            unsupported = torch.ones(len(log_weights), dtype=torch.bool, device=logits.device)
            unsupported[indices] = False
            log_unsupported = log_weights[unsupported].logsumexp(0) if unsupported.any() else logits.new_tensor(-torch.inf)
            gate_logit = self.gate_logit(question, current, plan)
            if diagnostic is not None:
                diagnostic['copy_gate'] = float(gate_logit.sigmoid().detach())
            log_gate = F.logsigmoid(gate_logit)
            # Legacy: 1 - gate * support = (1 - gate) + gate * unsupported.
            # Conditional mode keeps the independent task fallback 1 - gate.
            # Log space preserves native gradients even when sigmoid(gate)
            # rounds to one or the unsupported mass is below float epsilon.
            log_native = (F.logsigmoid(-gate_logit) if self.conditional_copy else
                          torch.logaddexp(F.logsigmoid(-gate_logit), log_gate + log_unsupported))
            result[row, position] = torch.logaddexp(log_native + logits[row, position].log_softmax(-1), log_gate + log_copy)
        return result
