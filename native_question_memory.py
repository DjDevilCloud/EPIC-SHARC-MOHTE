"""Bounded causal source occurrences queried by a persistent native question."""
from dataclasses import dataclass, field
import copy
import math

import torch
from torch import nn


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


class NativeQuestionMemory(nn.Module):
    def __init__(self, d_model, capacity=128, question_capacity=32):
        super().__init__()
        if capacity < 1 or question_capacity < 1:
            raise ValueError('Native question memory capacities must be positive.')
        self.capacity = int(capacity)
        self.question_capacity = int(question_capacity)
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

    def _store_word(self, state, word):
        if state.mode == 'source':
            state.cached_packet = None
            state.source.append((word, tuple(state.clause)))
            state.source = state.source[-self.capacity:]
            state.clause = (state.clause + [word])[-8:]
        elif state.mode == 'question':
            state.cached_packet = None
            state.question = (state.question + [word])[-self.question_capacity:]

    def _restore_header_word(self, state):
        if state.header_word:
            self._store_word(state, state.header_word)
        state.header = ''
        state.header_word = ()

    def _flush(self, state, tokenizer):
        if not state.pending:
            return
        word = tuple(state.pending)
        state.pending = []
        text = tokenizer.decode(list(word), clean_text=False, collapse_structure=False).casefold()
        if state.line_start and text in {'question', 'context'}:
            state.header = text
            state.header_word = word
        else:
            self._store_word(state, word)
        state.line_start = False

    def _observe(self, state, token, tokenizer):
        text = tokenizer.construction_units[token].text
        if token == tokenizer.bos_id or text == '<BOI>':
            state = NativeQuestionState(mode='source')
        elif text == '<EOI>':
            self._flush(state, tokenizer)
            self._restore_header_word(state)
            state.mode = 'outside'
        elif text == '<BOO>':
            self._flush(state, tokenizer)
            self._restore_header_word(state)
            state.mode = 'outside'
            state.emitting = state.has_question
        elif text == '<EOO>' or token == tokenizer.eos_id:
            state.emitting = False
            state.mode = 'outside'
        elif token == tokenizer.pad_id:
            return state, False
        elif state.mode in {'source', 'question'}:
            if tokenizer.construction_units[token].is_lexical:
                self._restore_header_word(state)
                state.pending = (state.pending + [token])[-64:]
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
                if any(char in surface for char in '.!?\n'):
                    self._restore_header_word(state)
                    state.clause = []
                if '\n' in surface:
                    state.line_start = True
        return state, state.emitting and bool(state.source) and bool(state.question)

    def forward(self, input_ids, hidden, lexical_embeddings, tokenizer, states=None, return_attention=False):
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
            identity = (tuple(state.source), tuple(state.question))
            if identity not in packet_cache:
                values = []
                keys = []
                for word, preceding in state.source:
                    value = word_vector(word)
                    scope = torch.stack([word_vector(w) for w in preceding + (word,)])
                    context = self.context_encoder(scope[None])[1][-1, 0]
                    keys.append(self.key(torch.cat((value, context))))
                    values.append(self.value(value))
                question = self.question_encoder(torch.stack([word_vector(w) for w in state.question])[None])[1][-1, 0]
                keys = torch.stack(keys)
                probabilities = (keys @ self.query(question) / math.sqrt(width)).softmax(0)
                recalled = probabilities @ torch.stack(values)
                packet_cache[identity] = (question, recalled, probabilities)
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
            sequence = []
            for position, token in enumerate(tokens):
                state, active = self._observe(state, token, tokenizer)
                current = hidden[row, position]
                if active:
                    question, recalled, probabilities = packet(state)
                    if return_attention:
                        attention_records.append((row, position, [word for word, _ in state.source], probabilities))
                    gate = self.gate(torch.cat((current, question))).sigmoid()
                    current = current + gate * self.output(torch.cat((current, question, recalled)))
                sequence.append(current)
            outputs.append(torch.stack(sequence))
            next_states.append(state)
        result = (torch.stack(outputs), next_states)
        return result + (attention_records,) if return_attention else result
