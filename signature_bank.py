# SPDX-License-Identifier: AGPL-3.0-or-later
"""Versioned, shared component embeddings and bounded continuation proposals.

Hashes address reusable feature parameters; exact prefix tokens tag retrieval
entries so an embedding/hash collision cannot authorize a copied continuation.
"""
from __future__ import annotations

import hashlib
import math
import weakref
import codecs
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def stable_key(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest(), 'little') & ((1 << 63) - 1)


def components(code: str) -> list[str]:
    """Compose properties independently; bucket counts rather than enumerate products."""
    parts = code.split('|')
    kind = parts[0]
    result = ['kind:' + kind]
    for part in parts[1:]:
        if '=' in part:
            name, value = part.split('=', 1)
            if name in {'len', 'wc', 'pc', 'ind'} and value.isdigit():
                value = str(int(math.log2(1 + int(value))))
            result.append(f'{kind}:{name}={value}')
        elif part:
            result.append('attribute:' + part)
    return list(dict.fromkeys(result))[:16]


class RuntimeStructuralState:
    """Bounded causal properties; never looks up a word or stores its full text."""
    def __init__(self, version=2):
        self.version = version
        self.role = 'outside'
        self.case = 'lower'
        self.word_length = self.vowels = self.digits = 0
        self.word_position = self.line_length = self.indent = 0
        self.in_indent = True
        self.previous_class = 'boundary'
        self.completed_word = (0, 0, 0)
        self.completed_line = (0, 0, 0)
        self.line_ended = False
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    def clone(self):
        result = copy.copy(self)
        result.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        result.decoder.setstate(self.decoder.getstate())
        return result

    @staticmethod
    def bucket(n):
        return int(math.log2(1 + n))

    def step(self, unit):
        text = unit.text
        if self.version == 3 and self.line_ended:
            self.word_length = self.vowels = self.digits = 0
            self.word_position = self.line_length = self.indent = 0
            self.in_indent = True
            self.case, self.previous_class = 'lower', 'boundary'
            self.line_ended = False
        reset_markers = ('<BOI>', '<BOO>', '<LINE>', '<BLO>', '<EOL>') if self.version == 2 else ('<BOI>', '<BOO>', '<LINE>', '<BLO>')
        if text in reset_markers:
            role = {'<BOI>':'input', '<BOO>':'output'}.get(text, self.role)
            completed_word, completed_line = self.completed_word, self.completed_line
            self.__init__(self.version)
            self.role = role
            if self.version == 3 and text != '<BOI>':
                self.completed_word, self.completed_line = completed_word, completed_line
        if text in ('<CAP>', '<UPPER>'):
            self.case = 'title' if text == '<CAP>' else 'upper'
        if unit.kind == 'byte':
            rendered = self.decoder.decode(bytes([int(text[6:-1], 16)]), final=False)
        elif unit.kind in ('structure', 'special', 'case', 'signature'):
            rendered = ''
        else:
            rendered = unit.render
        for ch in rendered:
            self.line_length += 1
            if self.in_indent and ch in ' \t':
                self.indent += 4 if ch == '\t' else 1
            else:
                self.in_indent = False
            if ch.isalnum() or ch in "_'":
                if not self.word_length:
                    self.word_position += 1
                self.word_length += 1
                self.vowels += int(ch.lower() in 'aeiou')
                self.digits += int(ch.isdigit())
                self.previous_class = 'digit' if ch.isdigit() else ('vowel' if ch.lower() in 'aeiou' else 'letter')
            else:
                if self.word_length:
                    self.completed_word = (self.word_length, self.vowels, self.digits)
                self.word_length = self.vowels = self.digits = 0
                self.case = 'lower'
                self.previous_class = ('space' if ch.isspace() else
                    'operator' if ch in '+-*/%=<>!&|' else
                    'bracket' if ch in '()[]{}' else 'punctuation')
        if self.version == 3 and text == '<EOL>':
            if self.word_length:
                self.completed_word = (self.word_length, self.vowels, self.digits)
            self.completed_line = (self.line_length, self.word_position, self.indent)
            self.line_ended = True
        # Exact short ordinals distinguish fields; long sequences have bounded tails.
        position = self.bucket(self.word_position) if self.version == 2 else (
            str(self.word_position) if self.word_position <= 16 else 'tail:' + str(self.bucket(self.word_position)))
        # Each property has its own address; no Cartesian profile identity.
        features = [f'runtime:role={self.role}', f'runtime:unit={unit.kind}',
                f'runtime:case={self.case}', f'runtime:word_len={self.bucket(self.word_length)}',
                f'runtime:vowels={self.bucket(self.vowels)}', f'runtime:digits={self.bucket(self.digits)}',
                f'runtime:word_pos={position}',
                f'runtime:line_len={self.bucket(self.line_length)}',
                f'runtime:indent={self.bucket(self.indent)}', f'runtime:edge={self.previous_class}',
                f'runtime:word_active={int(bool(self.word_length))}']
        if self.version == 3:
            for name, values in (('completed_word', self.completed_word), ('completed_line', self.completed_line)):
                for index, value in enumerate(values):
                    features.append(f'runtime:{name}:{index}={self.bucket(value)}')
        return features


class SharedSignatureBank(nn.Module):
    version = 1

    def __init__(self, cfg):
        super().__init__()
        self.width = cfg.d_model
        self.capacity = cfg.signature_component_buckets
        self.runtime_enabled = cfg.signature_representation in {'compositional_v2', 'compositional_v3'}
        self.runtime_version = 3 if cfg.signature_representation == 'compositional_v3' else 2
        self.embedding = nn.Embedding(self.capacity + 1, self.width, padding_idx=0)
        self._bound_contract = None
        self._configured = False
        for domain, size in [('signature', cfg.signature_vocab_size), ('family', cfg.signature_bucket_vocab_size),
                             ('level', cfg.signature_level_vocab_size), ('relation', cfg.signature_relation_vocab_size),
                             ('token', cfg.vocab_size or cfg.base_vocab_size)]:
            self.register_buffer(domain + '_components', torch.zeros(max(8, size), 32, dtype=torch.long))
            self.register_buffer(domain + '_keys', torch.zeros(max(8, size), dtype=torch.long))

    def configure(self, tokenizer):
        contract = (id(tokenizer), tokenizer.vocab_size, tokenizer.signature_vocab_size, tokenizer.signature_family_vocab_size)
        if self._bound_contract == contract:
            return
        domains = {
            'signature': tokenizer._signature_id_to_code,
            'family': tokenizer.signature_family_by_id,
            'level': {i: 'level:' + key for key, i in tokenizer.signature_level_to_id.items()},
            'relation': {i: 'relation:' + key for key, i in tokenizer.signature_relation_to_id.items()},
            'token': {i: 'lexical|' + 'text=' + unit.text + '|type=' + unit.kind
                      for i, unit in enumerate(tokenizer.construction_units)},
        }
        for domain, codes in domains.items():
            old = getattr(self, domain + '_components')
            size = max(old.size(0), max(codes, default=0) + 1)
            rows = torch.zeros(size, 32, dtype=torch.long)
            keys = torch.zeros(size, dtype=torch.long)
            for index, code in codes.items():
                feats = components(code)
                # Two independent addresses per property reduce accidental coupling.
                hashes = [1 + stable_key(salt + feature) % self.capacity for feature in feats for salt in ('a:', 'b:')]
                rows[index, :len(hashes)] = torch.tensor(hashes)
                keys[index] = stable_key(domain + ':' + code)
            setattr(self, domain + '_components', rows.to(old.device))
            setattr(self, domain + '_keys', keys.to(old.device))
        self._bound_contract = contract
        self._configured = True

    def encode(self, domain, ids):
        if not self._configured:
            raise RuntimeError('Bind a tokenizer with prepare_capacity_for_tokenizer before using compositional signatures.')
        table = getattr(self, domain + '_components')
        indices = table[ids.long().clamp_min(0)]
        values = self.embedding(indices)
        count = indices.ne(0).sum(-1, keepdim=True).clamp_min(1)
        return values.sum(-2) / count.sqrt()

    def runtime_features(self, input_ids, tokenizer, states=None):
        if tokenizer is None:
            raise RuntimeError('Runtime signatures require a bound tokenizer.')
        if states is not None and len(states) != input_ids.size(0):
            raise ValueError('Runtime signature state batch size differs from input.')
        if states is not None and any(s.version != self.runtime_version for s in states):
            raise ValueError('Runtime signature state representation version differs from this bank.')
        states = [s.clone() for s in states] if states is not None else [RuntimeStructuralState(self.runtime_version) for _ in range(input_ids.size(0))]
        rows = []
        for tokens, state in zip(input_ids.tolist(), states):
            frames = []
            for token in tokens:
                features = state.step(tokenizer.construction_units[token])
                frames.append([1 + stable_key(salt + f) % self.capacity for f in features for salt in ('a:', 'b:')])
            rows.append(frames)
        return torch.tensor(rows, device=input_ids.device, dtype=torch.long), states

    def encode_runtime(self, features):
        if features is None:
            raise RuntimeError('Runtime feature addresses are missing.')
        return self.embedding(features.long()).sum(-2) / math.sqrt(features.size(-1))

    def compose(self, input_ids, signature_ids, family_ids, level_ids, relation_ids, parent_ids, runtime_features=None):
        result = self.encode('token', input_ids)
        if self.runtime_enabled:
            # Runtime hierarchy and lexical identity cooperate, with no profile lookup.
            return (result + self.encode_runtime(runtime_features)) / math.sqrt(2.)
        for domain, ids in [('signature', signature_ids), ('family', family_ids), ('level', level_ids),
                            ('relation', relation_ids), ('signature', parent_ids)]:
            if ids is not None:
                result = result + self.encode(domain, ids)
        return result / math.sqrt(6.)

    def span_vectors(self, tokens, lengths):
        mask = torch.arange(tokens.size(1), device=tokens.device)[None] < lengths[:, None]
        values = self.encode('token', tokens.clamp_min(0))
        # Positional weights distinguish order while identical constituents share rows.
        positions = torch.arange(1, tokens.size(1) + 1, device=tokens.device).float()
        weights = positions[None] * mask
        return F.normalize((values * weights[..., None]).sum(1), dim=-1)

    def view(self, domain, size):
        return SignatureBankView(self, domain, size)

    def projected_view(self, domain, size, width):
        return ProjectedSignatureBankView(self, domain, size, width)

    def address(self, domain, ids, buckets):
        return getattr(self, domain + '_keys')[ids.long().clamp_min(0)].remainder(buckets)


class SignatureBankView(nn.Module):
    """Non-owning embedding interface; the bank owns the sole learned table."""
    def __init__(self, bank, domain, size):
        super().__init__()
        self._bank_ref = weakref.ref(bank)
        self.domain = domain
        self.num_embeddings = max(8, size)
        self.embedding_dim = bank.width

    @property
    def bank(self):
        return self._bank_ref()

    @property
    def weight(self):
        return self(torch.arange(self.num_embeddings, device=self.bank.embedding.weight.device))

    def resize(self, size):
        self.num_embeddings = max(self.num_embeddings, size)

    def forward(self, ids):
        return self.bank.encode(self.domain, ids)


@dataclass
class IdentityReadoutState:
    entries: list
    role: str = 'outside'


class BoundedIdentityReadout(nn.Module):
    """Trainable post-recurrence read of bounded, causally observed input units."""
    def __init__(self, cfg):
        super().__init__()
        self.capacity = cfg.identity_readout_capacity
        self.rule = cfg.identity_readout_rule
        self.candidate_policy = cfg.identity_readout_candidate_policy
        self._tokenizer_contract = None
        if self.rule == 'preserve_structure_v1':
            self.register_buffer('lexical_mask',torch.zeros(cfg.vocab_size or cfg.base_vocab_size,dtype=torch.bool))
        self.query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.key = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, 1)
        nn.init.constant_(self.gate.bias, -2.)

    def configure(self, tokenizer, vocab_size):
        if self.rule != 'preserve_structure_v1':
            return
        contract=(id(tokenizer),tokenizer.vocab_size,vocab_size,self.candidate_policy)
        if self._tokenizer_contract==contract:
            return
        mask=torch.zeros(vocab_size,dtype=torch.bool,device=self.lexical_mask.device)
        for index,unit in enumerate(tokenizer.construction_units):
            if index < vocab_size and self.is_candidate(unit):
                mask[index]=True
        self.lexical_mask=mask
        self._tokenizer_contract=contract

    def is_candidate(self, unit):
        if self.candidate_policy == 'lexical_bytes_v2':
            return unit.is_lexical
        return unit.kind in {'piece','word','phrase','char','digit','byte'}

    def mix_probabilities(self, logits, copied_probabilities, gate):
        base=F.log_softmax(logits.float(),-1)
        if self.rule == 'mixture_v1':
            return torch.logaddexp(base+F.logsigmoid(-gate),
                copied_probabilities.clamp_min(1e-30).log()+F.logsigmoid(gate)).to(logits.dtype)
        # Copying redistributes lexical probability only. Formatting and punctuation
        # retain their exact base probabilities, and total lexical mass is conserved.
        mask=self.lexical_mask
        lexical_mass=torch.logsumexp(base[mask],dim=-1)
        result=base.clone()
        result[mask]=torch.logaddexp(base[mask]+F.logsigmoid(-gate),
            copied_probabilities[mask].clamp_min(1e-30).log()+F.logsigmoid(gate)+lexical_mass)
        return result.to(logits.dtype)

    def forward(self, logits, hidden, input_ids, features, tokenizer, bank, states=None):
        self.configure(tokenizer,logits.size(-1))
        if states is not None and len(states) != input_ids.size(0):
            raise ValueError('Identity readout state batch size differs from input.')
        states = [IdentityReadoutState(list(s.entries), s.role) for s in states] if states is not None else [
            IdentityReadoutState([]) for _ in range(input_ids.size(0))]
        rows=[]
        feature_rows=features.tolist()
        for row,(tokens,state) in enumerate(zip(input_ids.tolist(),states)):
            steps=[]
            for position,token in enumerate(tokens):
                unit=tokenizer.construction_units[token]
                if token==tokenizer.bos_id or unit.text=='<BOI>':
                    state.entries=[]
                    state.role='input' if unit.text=='<BOI>' else 'outside'
                elif unit.text=='<EOI>':state.role='outside'
                elif unit.text=='<BOO>':state.role='output'
                elif unit.text=='<EOO>':state.role='outside'
                if state.role=='input' and self.is_candidate(unit):
                    state.entries.append((token, tuple(feature_rows[row][position])))
                    state.entries=state.entries[-self.capacity:]
                current=logits[row,position]
                if state.role=='output' and state.entries:
                    ids=torch.tensor([e[0] for e in state.entries],device=logits.device)
                    properties=torch.tensor([e[1] for e in state.entries],device=logits.device)
                    candidates=(bank.encode('token',ids)+bank.encode_runtime(properties))/math.sqrt(2.)
                    keys=self.key(candidates)
                    h=hidden[row,position]
                    # All bounded candidates participate in training, so q/k receive gradients.
                    scores=(keys*self.query(h)).sum(-1)/math.sqrt(keys.size(-1))
                    attention=F.softmax(scores.float(),-1)
                    copy=torch.zeros_like(current,dtype=torch.float32).scatter_add(0,ids,attention)
                    gate=self.gate(h).float().squeeze(-1)
                    current=self.mix_probabilities(current,copy,gate)
                steps.append(current)
            rows.append(torch.stack(steps))
        return torch.stack(rows),states


class ProjectedSignatureBankView(SignatureBankView):
    def __init__(self, bank, domain, size, width):
        super().__init__(bank, domain, size)
        self.projection = nn.Linear(bank.width, width, bias=False)
        self.embedding_dim = width

    def forward(self, ids):
        return self.projection(super().forward(ids))


@dataclass
class SpanProposal:
    tokens: list[int]
    confidence: list[float]
    support: int
    similarity: float


class VerifiedSpanCursor:
    """Verify local slots against filtered model choices; never force a token."""
    def __init__(self, memory, bank):
        self.memory, self.bank = memory, bank
        self.proposal = None
        self.position = 0
        self.stats = dict(proposals=0, verified_tokens=0, rejected_tokens=0, uncertain_tokens=0)

    def verify(self, history, chosen):
        if self.proposal is None or self.position >= len(self.proposal.tokens):
            self.proposal = self.memory.propose(history, self.bank)
            self.position = 0
            if self.proposal is not None:
                self.stats['proposals'] += 1
        if self.proposal is None:
            return chosen
        confidence = self.proposal.confidence[self.position]
        expected = self.proposal.tokens[self.position]
        if confidence < self.memory.confidence_threshold:
            self.stats['uncertain_tokens'] += 1
        elif expected == chosen:
            self.stats['verified_tokens'] += 1
            self.position += 1
            return expected
        else:
            self.stats['rejected_tokens'] += 1
        # Keep subsequent slots after a local mismatch; each is independently verified.
        self.position += 1
        return chosen


class SignatureSpanMemory(nn.Module):
    """Training-only, bounded evidence store; validation/generation never learn entries."""
    def __init__(self, cfg):
        super().__init__()
        self.context = cfg.signature_span_context
        self.length = cfg.signature_span_tokens
        self.minimum_support = cfg.signature_span_min_support
        self.confidence_threshold = cfg.signature_span_confidence
        self.similarity_threshold = cfg.signature_span_similarity
        self.margin = cfg.signature_span_margin
        capacity = cfg.signature_span_capacity
        self.register_buffer('prefixes', torch.zeros(capacity, self.context, dtype=torch.long))
        self.register_buffer('targets', torch.zeros(capacity, self.length, dtype=torch.long))
        self.register_buffer('prefix_lengths', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('target_lengths', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('support', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('cursor', torch.zeros((), dtype=torch.long))
        self._entries = None
        self._lookup = None
        self._next_index = 0

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._entries = self._lookup = None

    def _ensure_index(self):
        if self._entries is not None:
            return
        self._entries, self._lookup = {}, {}
        prefixes, targets = self.prefixes.cpu().tolist(), self.targets.cpu().tolist()
        counts = self.support.cpu().tolist()
        pl, tl = self.prefix_lengths.cpu().tolist(), self.target_lengths.cpu().tolist()
        for i, count in enumerate(counts):
            if count:
                key = (tuple(prefixes[i][:pl[i]]), tuple(targets[i][:tl[i]]))
                self._entries[i] = (*key, count)
                self._lookup[key] = i
        self._next_index = int(self.cursor)

    @torch.no_grad()
    def add(self, prefix, target):
        prefix, target = tuple(prefix[-self.context:]), tuple(target[:self.length])
        if not prefix or not target:
            return
        self._ensure_index()
        key = (prefix, target)
        existing = self._lookup.get(key)
        if existing is not None:
            previous = self._entries[existing][2]
            self._entries[existing] = (prefix, target, previous + 1)
            self.support[existing] += 1
            return
        i = self._next_index % self.support.numel()
        if i in self._entries:
            old = self._entries[i]
            self._lookup.pop((old[0], old[1]), None)
        self._entries[i] = (prefix, target, 1)
        self._lookup[key] = i
        self.prefixes[i].zero_()
        self.targets[i].zero_()
        self.prefixes[i, :len(prefix)] = torch.tensor(prefix, device=self.prefixes.device)
        self.targets[i, :len(target)] = torch.tensor(target, device=self.targets.device)
        self.prefix_lengths[i], self.target_lengths[i], self.support[i] = len(prefix), len(target), 1
        self._next_index += 1
        self.cursor.fill_(self._next_index)

    @torch.no_grad()
    def observe(self, inputs, labels, mask, pad_id):
        if not self.training or mask is None:
            return
        for ids, targets, valid in zip(inputs.tolist(), labels.tolist(), mask.tolist()):
            # Bound observation work per sequence; seed output and sample later spans.
            starts = [i for i, value in enumerate(valid) if value > 0 and targets[i] != pad_id]
            for i in starts[::max(1, self.length)][:4]:
                following = []
                for j in range(i, min(i + self.length, len(targets))):
                    if valid[j] <= 0 or targets[j] == pad_id:
                        break
                    following.append(targets[j])
                self.add(ids[:i+1], following)

    @torch.no_grad()
    def propose(self, history, bank):
        prefix = tuple(history[-self.context:])
        self._ensure_index()
        if not prefix or not self._entries:
            return None
        exact = [i for i, entry in self._entries.items() if entry[0] == prefix]
        similarity = 1.
        if exact:
            candidates = exact
        else:
            active = torch.tensor(list(self._entries), device=self.prefixes.device)
            query = torch.zeros(1, self.context, dtype=torch.long, device=self.prefixes.device)
            query[0, :len(prefix)] = torch.tensor(prefix, device=query.device)
            q = bank.span_vectors(query, torch.tensor([len(prefix)], device=query.device))
            keys = bank.span_vectors(self.prefixes[active], self.prefix_lengths[active])
            values, order = (keys @ q[0]).sort(descending=True)
            similarity = float(values[0])
            if similarity < self.similarity_threshold or (len(values) > 1 and float(values[0] - values[1]) < self.margin):
                return None
            candidates = [int(active[order[0]])]
        total = sum(self._entries[i][2] for i in candidates)
        if total < self.minimum_support:
            return None
        winner = max(candidates, key=lambda i: self._entries[i][2])
        tokens = list(self._entries[winner][1])
        confidence = [sum(self._entries[i][2] for i in candidates
                          if len(self._entries[i][1]) > p and self._entries[i][1][p] == token) / total
                      for p, token in enumerate(tokens)]
        return SpanProposal(tokens, confidence, total, similarity)
