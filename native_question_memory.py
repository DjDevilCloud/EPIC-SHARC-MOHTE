"""Bounded causal source occurrences queried by a persistent native question."""
from dataclasses import dataclass, field
import copy
import math

import torch
from torch import nn
try:
    from .native_span_readout import NativeSpanReadout
except ImportError:
    from native_span_readout import NativeSpanReadout


@dataclass
class NativeQuestionState:
    source: list = field(default_factory=list)
    question: list = field(default_factory=list)
    clause: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    mode: str = 'outside'
    line_start: bool = True
    header: str = ''
    header_word: tuple = ()
    has_question: bool = False
    emitting: bool = False
    cached_packet: object = None
    surface: list = field(default_factory=list)
    source_ranges: list = field(default_factory=list)
    pending_positions: list = field(default_factory=list)
    pending_count: int = 0
    header_positions: tuple = ()
    position: int = 0
    emitted: tuple = ()
    anchors: list = field(default_factory=list)
    clause_seen: int = 0
    emission_invalid: bool = False


class NativeQuestionMemory(nn.Module):
    def __init__(self, d_model, capacity=128, question_capacity=32, span_readout=False, span_words=16, span_starts=16, lexical_cues=False, future_context=0, anchor_words=0, stable_span_gate=False):
        super().__init__()
        if capacity < 1 or question_capacity < 1:
            raise ValueError('Native question memory capacities must be positive.')
        self.capacity = int(capacity)
        self.question_capacity = int(question_capacity)
        self.span_readout = NativeSpanReadout(d_model, span_words, span_starts, stable_span_gate) if span_readout else None
        self.future_context = int(future_context)
        self.anchor_words = int(anchor_words)
        self.future_key = nn.Linear(d_model, d_model, bias=False) if future_context else None
        if self.future_key is not None:
            nn.init.zeros_(self.future_key.weight)
        self.lexical_scale = nn.Parameter(torch.tensor(4.)) if lexical_cues else None
        self.previous_key = nn.Linear(d_model, d_model, bias=False) if lexical_cues else None
        if self.previous_key is not None:
            nn.init.zeros_(self.previous_key.weight)
        self.word_encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.context_encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.question_encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(2 * d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Sequential(nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.gate = nn.Linear(2 * d_model, 1)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _store_word(self, state, word, positions=(), complete=True):
        if state.mode == 'source':
            state.cached_packet = None
            before = tuple(state.anchors[:max(0, state.clause_seen - len(state.clause))] + state.clause)
            state.source.append((word, before))
            state.source = state.source[-self.capacity:]
            if self.span_readout is not None:
                state.source_ranges.append((positions[0], positions[-1] + 1, complete))
                state.source_ranges = state.source_ranges[-self.capacity:]
            state.clause = (state.clause + [word])[-8:]
            state.clause_seen += 1
            if len(state.anchors) < self.anchor_words:
                state.anchors.append(word)
        elif state.mode == 'question':
            state.cached_packet = None
            state.question = (state.question + [word])[-self.question_capacity:]

    def _restore_header_word(self, state):
        if state.header_word:
            self._store_word(state, state.header_word, state.header_positions)
        state.header = ''
        state.header_word = ()
        state.header_positions = ()

    def _flush(self, state, tokenizer):
        if not state.pending:
            return
        word = tuple(state.pending)
        positions = tuple(state.pending_positions)
        complete = state.pending_count <= 64
        state.pending = []
        state.pending_positions = []
        state.pending_count = 0
        text = tokenizer.decode(list(word), clean_text=False, collapse_structure=False).casefold()
        if state.line_start and text in {'question', 'context'}:
            state.header = text
            state.header_word = word
            state.header_positions = positions
        else:
            self._store_word(state, word, positions, complete)
        state.line_start = False

    def _observe(self, state, token, tokenizer):
        if token == tokenizer.pad_id:
            return state, False
        if state.emitted and state.emitted[-1] == tokenizer.special_tokens['<EOO>']:
            # Only the EOO position itself predicts the terminal EOS. Later
            # tokens cannot reopen a closed answer except an explicit BOO.
            state.emitting = False
        text = tokenizer.construction_units[token].text
        # Training and generation seed a leading LINE as prompt context. It
        # carries no answer surface; internal LINE controls remain significant.
        leading_line = text == '<LINE>' and not state.emitted and not state.emission_invalid
        if self.span_readout is not None and state.emitting and text != '<BOO>' and not leading_line:
            if len(state.emitted) >= self.span_readout.max_words * 72 + 8:
                state.emission_invalid = True
                state.emitted = (-1,)
            elif not state.emission_invalid:
                state.emitted = state.emitted + (token,)
        if token == tokenizer.bos_id or text == '<BOI>':
            state = NativeQuestionState(mode='source')
        position = state.position
        if self.span_readout is not None and state.mode in {'source', 'question'}:
            state.surface.append((position, token))
            if len(state.surface) > self.capacity * 80:
                del state.surface[:-self.capacity * 80]
            state.position += 1
        if token == tokenizer.bos_id or text == '<BOI>':
            pass  # Reset above, before recording this input boundary.
        elif text == '<EOI>':
            self._flush(state, tokenizer)
            self._restore_header_word(state)
            state.mode = 'outside'
        elif text == '<BOO>':
            self._flush(state, tokenizer)
            self._restore_header_word(state)
            state.mode = 'outside'
            state.emitting = state.has_question
            state.emitted = ()
            state.emission_invalid = False
        elif text == '<EOO>':
            # The supervised EOS is predicted *at* EOO. Keep the verified span
            # available for that final step, then disable it when EOS arrives.
            state.emitting = state.emitting and self.span_readout is not None
            state.mode = 'outside'
        elif token == tokenizer.eos_id:
            state.emitting = False
            state.mode = 'outside'
        elif state.mode in {'source', 'question'}:
            if tokenizer.construction_units[token].is_lexical:
                self._restore_header_word(state)
                state.pending = (state.pending + [token])[-64:]
                state.pending_positions = (state.pending_positions + [position])[-64:]
                state.pending_count += 1
            else:
                self._flush(state, tokenizer)
                surface = tokenizer.decode([token], clean_text=False, collapse_structure=False)
                if ':' in surface and state.header:
                    if state.header == 'question':
                        state.mode = 'question'
                        state.has_question = True
                        state.question = []
                        state.cached_packet = None
                    else:
                        state.mode = 'source'
                    state.header = ''
                    state.header_word = ()
                    state.header_positions = ()
                if any(char in surface for char in '.!?\n'):
                    self._restore_header_word(state)
                    state.clause = []
                    state.anchors = []
                    state.clause_seen = 0
                if '\n' in surface:
                    state.line_start = True
        return state, state.emitting and bool(state.source) and bool(state.question)

    def forward(self, input_ids, hidden, lexical_embeddings, tokenizer, states=None, return_attention=False, return_readout=False):
        if tokenizer is None:
            raise ValueError('Native question memory requires the checkpoint tokenizer.')
        batch, length, width = hidden.shape
        if states is not None and len(states) != batch:
            raise ValueError('Native question state batch size must match input tokens.')
        next_states = []
        outputs = []
        word_cache = {}
        packet_cache = {}
        attention_records = []
        readout_records = []

        def word_vector(word):
            if word not in word_cache:
                ids = torch.tensor(word, device=hidden.device, dtype=torch.long)
                embeddings = lexical_embeddings(ids).to(hidden.dtype)
                word_cache[word] = self.word_encoder(embeddings[None])[1][-1, 0]
            return word_cache[word]

        def packet(state):
            cacheable = not self.training and not torch.is_grad_enabled()
            if cacheable and state.cached_packet is not None and state.cached_packet[0].device == hidden.device and state.cached_packet[0].dtype == hidden.dtype:
                return state.cached_packet
            identity = (tuple(state.source), tuple(state.question), tuple(state.surface) if self.span_readout is not None else ())
            if identity not in packet_cache:
                values = []
                words = []
                keys = []
                context_words = []
                for index, (word, preceding) in enumerate(state.source):
                    value = word_vector(word)
                    words.append(value)
                    following = []
                    for next_index in range(index + 1, min(len(state.source), index + 1 + self.future_context)):
                        if not state.source[next_index][1]:
                            break
                        following.append(state.source[next_index][0])
                    context_words.append(preceding + tuple(following))
                    scope = torch.stack([word_vector(w) for w in preceding + (word,)])
                    context = self.context_encoder(scope[None])[1][-1, 0]
                    key = self.key(torch.cat((value, context)))
                    if preceding and self.previous_key is not None:
                        key = key + self.previous_key(word_vector(preceding[-1]))
                    if following:
                        future = self.context_encoder(torch.stack([word_vector(w) for w in (word,) + tuple(following)])[None])[1][-1, 0]
                        key = key + self.future_key(future)
                    keys.append(key)
                    values.append(self.value(value))
                question = self.question_encoder(torch.stack([word_vector(w) for w in state.question])[None])[1][-1, 0]
                keys = torch.stack(keys)
                source_logits = keys @ self.query(question) / math.sqrt(width)
                if self.lexical_scale is not None:
                    # Exact word tuples are retrieval identities, not numeric
                    # features. Relation semantics still use the learned path.
                    overlap = question.new_tensor([sum(word in context for word in state.question) for context in context_words])
                    source_logits = source_logits + self.lexical_scale * overlap
                probabilities = source_logits.softmax(0)
                recalled = probabilities @ torch.stack(values)
                plan = self.span_readout.plan(state, tokenizer, question, torch.stack(words), probabilities, source_logits.log_softmax(0)) if self.span_readout is not None else None
                if plan is not None:
                    plan['copy_context'] = recalled
                packet_cache[identity] = (question, recalled, probabilities, plan)
            result = packet_cache[identity]
            if cacheable:
                state.cached_packet = result
            return result

        for row, tokens in enumerate(input_ids.detach().cpu().tolist()):
            state = copy.copy(states[row]) if states is not None else NativeQuestionState()
            # Occurrence tuples and cached inference tensors are immutable.
            # Copy mutable containers without replaying or cloning the memory.
            state.source = list(state.source)
            state.question = list(state.question)
            state.clause = list(state.clause)
            state.pending = list(state.pending)
            state.pending_positions = list(state.pending_positions)
            state.surface = list(state.surface)
            state.source_ranges = list(state.source_ranges)
            state.anchors = list(state.anchors)
            sequence = []
            for position, token in enumerate(tokens):
                state, active = self._observe(state, token, tokenizer)
                current = hidden[row, position]
                if active:
                    question, recalled, probabilities, plan = packet(state)
                    if return_attention:
                        attention_records.append((row, position, [word for word, _ in state.source], probabilities))
                    # Legacy checkpoints never trained the residual at EOO.
                    # Expose verified EOS support without changing that base
                    # boundary prediction; the span gate receives EOS gradients.
                    if tokenizer.construction_units[token].text != '<EOO>':
                        gate = self.gate(torch.cat((current, question))).sigmoid()
                        current = current + gate * self.output(torch.cat((current, question, recalled)))
                    if return_readout and plan is not None:
                        readout_records.append((row, position, question, current, plan, state.emitted))
                sequence.append(current)
            outputs.append(torch.stack(sequence))
            next_states.append(state)
        result = (torch.stack(outputs), next_states)
        if return_attention:
            result += (attention_records,)
        if return_readout:
            result += (readout_records,)
        return result
