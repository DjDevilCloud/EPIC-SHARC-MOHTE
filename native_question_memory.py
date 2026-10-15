"""Bounded causal source occurrences queried by a persistent native question."""
from dataclasses import dataclass, field
import copy
import math

import torch
from torch import nn
try:
    from .native_span_readout import NativeSpanReadout
    from .native_antecedent_memory import NativeAntecedentMemory
    from .native_relation_alignment import NativeRelationAlignment
    from .native_source_role import NativeSourceRole
    from .native_semantic_alignment import NativeSemanticAlignment
except ImportError:
    from native_span_readout import NativeSpanReadout
    from native_antecedent_memory import NativeAntecedentMemory
    from native_relation_alignment import NativeRelationAlignment
    from native_source_role import NativeSourceRole
    from native_semantic_alignment import NativeSemanticAlignment


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
    document_hidden: object = None
    document_contexts: list = field(default_factory=list)
    document_pending: list = field(default_factory=list)
    antecedents: list = field(default_factory=list)
    source_ordinals: list = field(default_factory=list)
    source_seen: int = 0
    next_word_case: bool = False


class NativeQuestionMemory(nn.Module):
    def __init__(self, d_model, capacity=128, question_capacity=32, span_readout=False, span_words=16, span_starts=16, lexical_cues=False, future_context=0, anchor_words=0, stable_span_gate=False, matcher='combined_v1', batch_encoders=False, document_context=False, antecedent_memory=False, antecedent_capacity=64, antecedent_query='learned_v1', conditional_endpoints=False, relation_alignment=False, source_roles=False, owner_evidence=False, marker_endpoints=False, semantic_alignment=False, semantic_prior_scaling=False, semantic_decisions=False, conditional_copy=False):
        super().__init__()
        if capacity < 1 or question_capacity < 1:
            raise ValueError('Native question memory capacities must be positive.')
        self.capacity = int(capacity)
        self.question_capacity = int(question_capacity)
        if matcher not in {'combined_v1', 'owner_relation_v1'}:
            raise ValueError('Unknown native question/source matcher')
        self.matcher = matcher
        self.batch_encoders = bool(batch_encoders)
        if owner_evidence and not antecedent_memory:
            raise ValueError('Native owner evidence requires the antecedent bank')
        if marker_endpoints and not span_readout:
            raise ValueError('Native marker endpoints require span readout')
        self.antecedent_memory = NativeAntecedentMemory(d_model, antecedent_capacity, identity_query=antecedent_query, context_evidence=owner_evidence) if antecedent_memory else None
        # Reuse the ordered context encoder across sentence boundaries. A
        # bounded recurrent summary survives occurrence eviction; exact source
        # identities and span surfaces stay in their existing bounded channels.
        self.document_key = nn.Linear(d_model, d_model, bias=False) if document_context else None
        if self.document_key is not None:
            nn.init.zeros_(self.document_key.weight)
        self.request_roles = nn.Linear(d_model, 2) if matcher == 'owner_relation_v1' else None
        if self.request_roles is not None:
            nn.init.zeros_(self.request_roles.weight)
            nn.init.zeros_(self.request_roles.bias)
        if semantic_decisions and not (semantic_alignment and span_readout):
            raise ValueError('Semantic decisions require semantic alignment and span readout')
        if conditional_copy and not (span_readout and stable_span_gate):
            raise ValueError('Conditional copy requires span readout and a stable task gate')
        self.span_readout = NativeSpanReadout(d_model, span_words, span_starts, stable_span_gate, conditional_endpoints, marker_endpoints, conditional_copy) if span_readout else None
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
        if relation_alignment and matcher != 'combined_v1':
            raise ValueError('Native relation alignment requires the combined base matcher')
        self.relation_alignment = NativeRelationAlignment(d_model) if relation_alignment else None
        if source_roles and matcher != 'combined_v1':
            raise ValueError('Native source roles require the combined base matcher')
        self.source_roles = NativeSourceRole(d_model) if source_roles else None
        if semantic_prior_scaling and not semantic_alignment:
            raise ValueError('Semantic prior scaling requires semantic alignment')
        self.semantic_alignment = NativeSemanticAlignment(d_model, semantic_prior_scaling) if semantic_alignment else None
        self.semantic_request = nn.Linear(2 * d_model, d_model, bias=False) if semantic_decisions else None
        self.semantic_source = nn.Linear(2 * d_model, d_model, bias=False) if semantic_decisions else None
        if semantic_decisions:
            nn.init.zeros_(self.semantic_request.weight)
            nn.init.zeros_(self.semantic_source.weight)
        self.key = nn.Linear(2 * d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Sequential(nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.gate = nn.Linear(2 * d_model, 1)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _store_word(self, state, word, positions=(), complete=True, properties=(0., 0., 0., 0.)):
        if state.mode == 'source':
            state.cached_packet = None
            before = tuple(state.anchors[:max(0, state.clause_seen - len(state.clause))] + state.clause)
            state.source.append((word, before))
            state.source = state.source[-self.capacity:]
            if self.antecedent_memory is not None:
                self.antecedent_memory.observe(state, word, properties)
            if self.document_key is not None:
                state.document_pending.append(word)
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
        surface = tokenizer.decode(list(word), clean_text=False, collapse_structure=False)
        text = surface.casefold()
        if state.line_start and text in {'question', 'context'}:
            state.header = text
            state.header_word = word
            state.header_positions = positions
        else:
            properties = (min(len(surface), 32) / 32., float(state.next_word_case or bool(surface[:1].isupper())), float(any(c.isdigit() for c in surface)), min(len(word), 16) / 16.)
            self._store_word(state, word, positions, complete, properties)
        state.next_word_case = False
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
                if tokenizer.construction_units[token].kind == 'case':
                    state.next_word_case = True
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

        def advance_document(state):
            if not state.document_pending:
                return
            if self.batch_encoders:
                missing = [w for w in dict.fromkeys(state.document_pending) if w not in word_cache]
                if missing:
                    sequences = [lexical_embeddings(torch.tensor(w, device=hidden.device)).to(hidden.dtype) for w in missing]
                    word_cache.update(zip(missing, self.encode_sequences(self.word_encoder, sequences)))
            sequence = torch.stack([word_vector(w) for w in state.document_pending])[None]
            summaries, final = self.context_encoder(sequence, state.document_hidden)
            state.document_hidden = final
            state.document_contexts = (state.document_contexts + list(summaries[0].unbind(0)))[-self.capacity:]
            state.document_pending = []

        def packet(state):
            cacheable = not self.training and not torch.is_grad_enabled()
            if cacheable and state.cached_packet is not None and state.cached_packet[0].device == hidden.device and state.cached_packet[0].dtype == hidden.dtype:
                return state.cached_packet
            identity = (tuple(state.source), tuple(state.question), tuple(state.surface) if self.span_readout is not None else (), tuple(state.document_contexts), tuple(state.antecedents), tuple(state.source_ordinals))
            if identity not in packet_cache:
                if self.batch_encoders:
                    missing = list(dict.fromkeys(w for word, preceding in state.source for w in preceding + (word,)))
                    missing += [w for w in state.question if w not in missing]
                    missing = [w for w in missing if w not in word_cache]
                    if missing:
                        sequences = [lexical_embeddings(torch.tensor(w,device=hidden.device)).to(hidden.dtype) for w in missing]
                        word_cache.update(zip(missing, self.encode_sequences(self.word_encoder, sequences)))
                    scopes = [torch.stack([word_vector(w) for w in preceding + (word,)]) for word, preceding in state.source]
                    contexts = self.encode_sequences(self.context_encoder, scopes)
                    future_scopes, future_indices = [], []
                    for index, (word, _) in enumerate(state.source):
                        following = []
                        for j in range(index+1,min(len(state.source),index+1+self.future_context)):
                            if not state.source[j][1]: break
                            following.append(state.source[j][0])
                        if following:
                            future_indices.append(index)
                            future_scopes.append(torch.stack([word_vector(w) for w in (word,) + tuple(following)]))
                    futures = dict(zip(future_indices,self.encode_sequences(self.context_encoder,future_scopes))) if future_scopes else {}
                values = []
                words = []
                keys = []
                owner_keys = []
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
                    context = contexts[index] if self.batch_encoders else self.context_encoder(scope[None])[1][-1, 0]
                    key = self.key(torch.cat((value, context)))
                    document_key = self.document_key(state.document_contexts[index]) if self.document_key is not None else None
                    if document_key is not None:
                        key = key + document_key
                    if self.request_roles is not None:
                        owner_scope = torch.stack([word_vector(w) for w in preceding + tuple(following) + (word,)])
                        owner_keys.append(self.key(torch.cat((value, owner_scope.mean(0))))[:width//2])
                        if document_key is not None:
                            owner_keys[-1] = owner_keys[-1] + document_key[:width//2]
                    if preceding and self.previous_key is not None:
                        key = key + self.previous_key(word_vector(preceding[-1]))
                    if following:
                        future = futures[index] if self.batch_encoders else self.context_encoder(torch.stack([word_vector(w) for w in (word,) + tuple(following)])[None])[1][-1, 0]
                        key = key + self.future_key(future)
                    keys.append(key)
                    values.append(self.value(value))
                query_words = torch.stack([word_vector(w) for w in state.question])
                query_states, query_final = self.question_encoder(query_words[None])
                question = query_final[-1, 0]
                keys = torch.stack(keys)
                if self.request_roles is None:
                    projected_request = self.query(question)
                    if self.relation_alignment is not None:
                        phrases = self.encode_sequences(self.context_encoder, [query_words[i:i+n] for n in (1, 2, 3) for i in range(len(query_words)-n+1)])
                        projected_request = self.relation_alignment(query_words, phrases, projected_request)
                    source_logits = keys @ projected_request / math.sqrt(width)
                else:
                    # Two learned request pools share the ordered word encoder.
                    # Owner matching keeps lexical identity in its context pool;
                    # relation matching uses ordered local source/query context.
                    roles = self.request_roles(query_states[0]).softmax(0)
                    owner_request = roles[:,0] @ query_words
                    relation_request = roles[:,1] @ query_states[0]
                    owner_score = torch.stack(owner_keys) @ self.query(owner_request)[:width//2]
                    relation_score = keys[:,width//2:] @ self.query(relation_request)[width//2:]
                    source_logits = (owner_score + relation_score) / math.sqrt(width)
                role_features = role_scores = None
                if self.source_roles is not None:
                    role_features = self.source_roles.features(state.source, word_vector,
                        lambda scopes: self.encode_sequences(self.context_encoder, scopes), question)
                    role_scores = self.source_roles(projected_request, role_features)
                    source_logits = source_logits + role_scores
                if self.lexical_scale is not None:
                    # Exact word tuples are retrieval identities, not numeric
                    # features. Relation semantics still use the learned path.
                    overlap = question.new_tensor([sum(word in context for word in state.question) for context in context_words])
                    source_logits = source_logits + self.lexical_scale * overlap
                antecedent = None
                if self.antecedent_memory is not None:
                    compatibility, antecedent = self.antecedent_memory.access(state, state.question, word_vector, self.context_encoder, self.encode_sequences)
                    source_logits = source_logits + self.antecedent_memory.scale * compatibility
                if self.semantic_alignment is not None:
                    semantic_inputs = self.semantic_alignment.inputs([w for w, _ in state.source], state.question, lexical_embeddings)
                    correction, request, source = self.semantic_alignment(semantic_inputs, prior=source_logits, return_features=True)
                    source_logits = source_logits + correction
                probabilities = source_logits.softmax(0)
                recalled = probabilities @ torch.stack(values)
                decision_words = torch.stack(words)
                if self.semantic_request is not None:
                    # Share the observed-input request with extent, task gating
                    # and native emission. Zero residuals preserve old features.
                    question = question + self.semantic_request(request)
                    decision_words = decision_words + self.semantic_source(source)
                plan = self.span_readout.plan(state, tokenizer, question, decision_words, probabilities, source_logits.log_softmax(0), marker_words=query_words) if self.span_readout is not None else None
                if plan is not None:
                    plan['base_word_vectors'] = torch.stack(words)
                    plan['source_keys'] = keys
                    plan['copy_context'] = recalled
                    if role_scores is not None:
                        plan['source_role_features'] = role_features
                        plan['source_role_scores'] = role_scores
                    if antecedent is not None:
                        plan['antecedent'] = antecedent
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
            state.document_contexts = list(state.document_contexts)
            state.document_pending = list(state.document_pending)
            state.antecedents = list(state.antecedents)
            state.source_ordinals = list(state.source_ordinals)
            sequence = []
            for position, token in enumerate(tokens):
                state, active = self._observe(state, token, tokenizer)
                # Only completed source words advance this causal state. Neither
                # the question nor generated answer can alter occurrence keys.
                # Batch a full prefix when available, while bounding pending
                # storage and preserving one-token incremental semantics.
                if active or len(state.document_pending) >= self.capacity:
                    advance_document(state)
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
            advance_document(state)
            next_states.append(state)
        result = (torch.stack(outputs), next_states)
        if return_attention:
            result += (attention_records,)
        if return_readout:
            result += (readout_records,)
        return result

    @staticmethod
    def encode_sequences(encoder, sequences):
        """Batch independent ordered spans without advancing through padding."""
        from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
        lengths = torch.tensor([len(s) for s in sequences],device='cpu')
        padded = pad_sequence(sequences,batch_first=True)
        packed = pack_padded_sequence(padded,lengths,batch_first=True,enforce_sorted=False)
        return encoder(packed)[1][-1]
