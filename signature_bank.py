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
    words: object = None
    output_units: object = None
    binding_cache: object = None
    source_mask: object = None
    word_node: object = None
    word_prior: object = None
    output_word_count: int = 0
    output_in_word: bool = False
    output_separators: object = None
    source_initial_prior: object = None
    source_cursor: object = None


class WordSpanState:
    """Request-local causal word boundaries and bounded clause spans; no entity parser."""
    def __init__(self, capacity=512):
        self.capacity=capacity
        self.pending=[]
        self.clause=[]
        self.last_clause=[]
        self.records=[];self.serial=0;self.clause_index=0
        self.following={};self.separator_owner=None

    def clone(self):
        result=WordSpanState(self.capacity)
        result.pending=list(self.pending);result.clause=list(self.clause);result.last_clause=list(self.last_clause)
        result.records=list(self.records);result.serial=self.serial;result.clause_index=self.clause_index
        result.following={k:list(v) for k,v in self.following.items()};result.separator_owner=self.separator_owner
        return result

    def boundary(self, clause=False):
        if self.pending:
            self.records.append((self.serial,self.clause_index,tuple(self.pending)));self.serial+=1
            self.separator_owner=self.serial-1
            while sum(len(r[2]) for r in self.records)>self.capacity:
                removed=self.records.pop(0);self.following.pop(removed[0],None)
            self.clause.append(tuple(self.pending));self.pending=[]
            while sum(map(len,self.clause))>self.capacity:self.clause.pop(0)
        if clause and self.clause:
            self.last_clause=list(self.clause);self.clause=[]
            self.clause_index+=1

    def observe(self, token, unit):
        if unit.is_lexical:
            self.separator_owner=None
            self.pending.append(token)
            # Explicit bounded memory: an over-capacity word retains its causal tail.
            self.pending=self.pending[-self.capacity:]
        elif unit.kind not in {'case','signature'}:
            self.boundary(clause=unit.text in {'<EOL>','<EOI>'} or any(c in '.!?\n' for c in unit.render))
            if self.separator_owner is not None:
                self.following.setdefault(self.separator_owner,[]).append(token)
                self.following[self.separator_owner]=self.following[self.separator_owner][:self.capacity]
        elif unit.kind=='case' and not self.pending and self.separator_owner is not None:
            self.following.setdefault(self.separator_owner,[]).append(token)
            self.following[self.separator_owner]=self.following[self.separator_owner][:self.capacity]

    def question(self):
        current=self.clause+([tuple(self.pending)] if self.pending else [])
        return current or self.last_clause

    def source_candidate_mask(self, entries):
        """Final observed clause is the query, as in question(); retain earlier sources.

        A one-clause input has no separate source, so keep its original retrieval
        behavior. No source answer is inferred or required by this partition.
        """
        if not self.records:return [True]*len(entries)
        clauses={serial:clause for serial,clause,_ in self.records}
        query_clause=self.records[-1][1]
        mask=[clauses.get(entry[3],query_clause)!=query_clause for entry in entries]
        return mask if any(mask) else [True]*len(entries)

    @staticmethod
    def ordered_agreement(question, context):
        """Longest contiguous identity agreement, over complete words, not ID distances.

        IDs are used solely as exact retrieval keys. Reusable properties and
        learned semantic scores still handle rewording and nonliteral requests.
        """
        best=0;prior=[0]*(len(context)+1)
        for word in question:
            row=[0]*(len(context)+1)
            for j,other in enumerate(context):
                if word==other:
                    row[j+1]=prior[j]+1;best=max(best,row[j+1])
            prior=row
        return best


class WordTaskRouter(nn.Module):
    """Learned task-cue attention over complete observed question words."""
    def __init__(self, width, neutral_bias):
        super().__init__();self.width=width
        self.attention=nn.Linear(width,1,bias=False)
        self.classifier=nn.Sequential(nn.Linear(2*width,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.attention.weight)
        with torch.no_grad():
            self.classifier[0].weight.copy_(torch.cat((torch.eye(width),torch.eye(width)),dim=1)/2)
        nn.init.zeros_(self.classifier[0].bias);nn.init.zeros_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias,neutral_bias)

    def forward(self, features):
        values=features[...,:self.width];mask=features[...,self.width]>0
        logits=self.attention(values).squeeze(-1)
        weights=logits.masked_fill(~mask,torch.finfo(logits.dtype).min).softmax(-1)*mask
        attended=(values*weights.unsqueeze(-1)).sum(-2)
        mean=(values*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1,keepdim=True).clamp_min(1)
        return self.classifier(torch.cat((F.layer_norm(mean,(self.width,)),F.layer_norm(attended,(self.width,))),dim=-1))


class BoundedIdentityReadout(nn.Module):
    """Trainable post-recurrence read of bounded, causally observed input units."""
    def __init__(self, cfg):
        super().__init__()
        self.capacity = cfg.identity_readout_capacity
        self.rule = cfg.identity_readout_rule
        self.candidate_policy = cfg.identity_readout_candidate_policy
        self.binding_enabled = cfg.identity_readout_binding in {'lexical_binding_v1', 'lexical_binding_v2','word_span_v3','word_span_v4','word_span_v5'}
        self.word_binding = cfg.identity_readout_binding in {'word_span_v3','word_span_v4','word_span_v5'}
        self.observed_span_binding = cfg.identity_readout_binding in {'word_span_v4','word_span_v5'}
        self.native_route = cfg.identity_readout_binding == 'word_span_v5'
        self.positional_binding = cfg.identity_readout_binding in {'lexical_binding_v2','word_span_v3','word_span_v4','word_span_v5'}
        self.context_units = cfg.identity_readout_context_units
        self.query_units = cfg.identity_readout_query_units
        self.binding_confidence = cfg.identity_readout_binding_confidence
        self._binding_cfg = cfg
        self._tokenizer_contract = None
        if self.rule == 'preserve_structure_v1':
            self.register_buffer('lexical_mask',torch.zeros(cfg.vocab_size or cfg.base_vocab_size,dtype=torch.bool))
        self.query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.key = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, 1)
        nn.init.constant_(self.gate.bias, -2.)
        if self.binding_enabled:
            self.context_query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.context_gate = nn.Bilinear(cfg.d_model, cfg.d_model, 1, bias=False)
            self.span_projection = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.applicability = nn.Linear(cfg.d_model, 1)
            self.span_scale = nn.Parameter(torch.zeros(()))
            nn.init.zeros_(self.context_query.weight)
            nn.init.zeros_(self.context_gate.weight)
            nn.init.eye_(self.span_projection.weight)
            nn.init.zeros_(self.applicability.weight)
            nn.init.zeros_(self.applicability.bias)
            if self.positional_binding:
                self.context_slot_logits = nn.Parameter(torch.zeros(self.context_units))
                self.query_slot_logits = nn.Parameter(torch.zeros(self.query_units))
                # Migration is exactly neutral until the input router is calibrated.
                neutral_bias = min(-20., math.log1p(-self.binding_confidence) - 2.)
                nn.init.constant_(self.applicability.bias, neutral_bias)
            if self.word_binding:
                self.context_word_attention=nn.Linear(cfg.d_model,1,bias=False)
                self.question_word_attention=nn.Linear(cfg.d_model,1,bias=False)
                nn.init.zeros_(self.context_word_attention.weight)
                nn.init.zeros_(self.question_word_attention.weight)
                if cfg.identity_readout_ordered_agreement:
                    self.word_agreement_scale=nn.Parameter(torch.zeros(()))
            if self.observed_span_binding:
                self.right_context_projection=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
                nn.init.zeros_(self.right_context_projection.weight)
                self.continuation_scale=nn.Parameter(torch.zeros(()))
            if self.native_route:
                self.native_query=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
                self.native_gate=nn.Bilinear(cfg.d_model,cfg.d_model,1,bias=False)
                self.native_applicability=WordTaskRouter(cfg.d_model,neutral_bias)
                self.native_scale=nn.Parameter(torch.zeros(()))
                nn.init.zeros_(self.native_query.weight);nn.init.zeros_(self.native_gate.weight)
                self.native_surface_projection=None
                if cfg.identity_readout_native_word_paths:
                    self.native_source_signature=nn.Linear(cfg.d_model,cfg.d_model,bias=False)
                    self.native_word_surface=nn.Linear(2*cfg.d_model,len(cfg.identity_readout_surface_ids),bias=False)
                    self.native_word_gate=nn.Bilinear(cfg.d_model,cfg.d_model,1,bias=False)
                    self.native_word_path_scale=nn.Parameter(torch.zeros(()))
                    self.native_word_start_scale=nn.Parameter(torch.zeros(()))
                    nn.init.zeros_(self.native_source_signature.weight)
                    nn.init.zeros_(self.native_word_surface.weight)
                    nn.init.zeros_(self.native_word_gate.weight)
                    if cfg.identity_readout_source_ownership:
                        self.native_span_start=nn.Linear(4*cfg.d_model,cfg.d_model,bias=False)
                        nn.init.zeros_(self.native_span_start.weight)
                        self.native_source_cursor_scale=nn.Parameter(torch.zeros(()))
                    if cfg.identity_readout_native_span_boundaries:
                        extra=int(cfg.identity_readout_native_span_readout=='categorical_v2')
                        self.native_span_surface=nn.Linear(2*cfg.d_model,len(cfg.identity_readout_surface_ids)+extra,bias=False)
                        self.native_span_gate=nn.Linear(2*cfg.d_model,1,bias=False)
                        nn.init.zeros_(self.native_span_surface.weight);nn.init.zeros_(self.native_span_gate.weight)
                        if extra:self.register_buffer('native_span_ready',torch.tensor(False))
                        if cfg.identity_readout_source_boundaries:
                            self.native_source_boundary=nn.Linear(2*cfg.d_model,len(cfg.identity_readout_surface_ids)+extra,bias=False)
                            nn.init.zeros_(self.native_source_boundary.weight)
                            if cfg.identity_readout_source_boundary_adapter:
                                self.native_source_boundary_adapter=nn.Linear(2*cfg.d_model,len(cfg.identity_readout_surface_ids)+extra,bias=False)
                                nn.init.zeros_(self.native_source_boundary_adapter.weight)
            self.register_buffer('surface_ids', torch.tensor(cfg.identity_readout_surface_ids, dtype=torch.long))
            self.surface_projection = None
            if self.surface_ids.numel():
                self.surface_projection = nn.Linear(2 * cfg.d_model, self.surface_ids.numel(), bias=False)
                nn.init.zeros_(self.surface_projection.weight)
                if self.native_route:
                    self.native_surface_projection=nn.Linear(2*cfg.d_model,self.surface_ids.numel(),bias=False)
                    nn.init.zeros_(self.native_surface_projection.weight)

    def configure(self, tokenizer, vocab_size):
        if self.binding_enabled:
            self._configure_binding_surface(tokenizer, vocab_size)
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

    def _configure_binding_surface(self, tokenizer, vocab_size):
        contract = (id(tokenizer), tokenizer.vocab_size, vocab_size)
        if getattr(self, '_surface_contract', None) == contract:
            return
        ids = [i for i,u in enumerate(tokenizer.construction_units) if i < vocab_size and not u.is_lexical]
        old_ids = self.surface_ids.tolist()
        if self.surface_projection is None or ids != old_ids:
            projection = nn.Linear(2 * self.query.in_features, len(ids), bias=False).to(
                device=self.query.weight.device, dtype=self.query.weight.dtype)
            nn.init.zeros_(projection.weight)
            if self.surface_projection is not None:
                by_id = {token: i for i,token in enumerate(old_ids)}
                with torch.no_grad():
                    for row,token in enumerate(ids):
                        if token in by_id:
                            projection.weight[row].copy_(self.surface_projection.weight[by_id[token]])
                projection.weight.requires_grad_(self.surface_projection.weight.requires_grad)
            self.surface_projection = projection
            self.surface_ids = torch.tensor(ids, dtype=torch.long, device=self.query.weight.device)
        self._binding_cfg.identity_readout_surface_ids = ids
        if self.native_route and (self.native_surface_projection is None or ids!=old_ids):
            native=nn.Linear(2*self.query.in_features,len(ids),bias=False).to(device=self.query.weight.device,dtype=self.query.weight.dtype)
            nn.init.zeros_(native.weight)
            if self.native_surface_projection is not None:
                by_id={token:i for i,token in enumerate(old_ids)}
                with torch.no_grad():
                    for row,token in enumerate(ids):
                        if token in by_id:native.weight[row].copy_(self.native_surface_projection.weight[by_id[token]])
                native.weight.requires_grad_(self.native_surface_projection.weight.requires_grad)
            self.native_surface_projection=native
        self._surface_contract = contract
        if self.native_route and self._binding_cfg.identity_readout_native_word_paths and self.native_word_surface.out_features!=len(ids):
            prior=self.native_word_surface
            projection=nn.Linear(2*self.query.in_features,len(ids),bias=False).to(device=self.query.weight.device,dtype=self.query.weight.dtype)
            nn.init.zeros_(projection.weight)
            with torch.no_grad():
                for row,token in enumerate(ids):
                    if token in old_ids:projection.weight[row].copy_(prior.weight[old_ids.index(token)])
            projection.weight.requires_grad_(prior.weight.requires_grad);self.native_word_surface=projection
        extra=int(self._binding_cfg.identity_readout_native_span_readout=='categorical_v2')
        if self.native_route and self._binding_cfg.identity_readout_native_span_boundaries and self.native_span_surface.out_features!=len(ids)+extra:
            prior=self.native_span_surface
            projection=nn.Linear(2*self.query.in_features,len(ids)+extra,bias=False).to(device=self.query.weight.device,dtype=self.query.weight.dtype)
            nn.init.zeros_(projection.weight)
            with torch.no_grad():
                for row,token in enumerate(ids):
                    if token in old_ids:projection.weight[row].copy_(prior.weight[old_ids.index(token)])
                if extra:projection.weight[-1].copy_(prior.weight[-1])
            projection.weight.requires_grad_(prior.weight.requires_grad);self.native_span_surface=projection
        if self.native_route and self._binding_cfg.identity_readout_source_boundaries and self.native_source_boundary.out_features!=len(ids)+extra:
            prior=self.native_source_boundary
            projection=nn.Linear(2*self.query.in_features,len(ids)+extra,bias=False).to(device=prior.weight.device,dtype=prior.weight.dtype)
            nn.init.zeros_(projection.weight)
            with torch.no_grad():
                for row,token in enumerate(ids):
                    if token in old_ids:projection.weight[row].copy_(prior.weight[old_ids.index(token)])
                if extra:projection.weight[-1].copy_(prior.weight[-1])
            projection.weight.requires_grad_(prior.weight.requires_grad);self.native_source_boundary=projection
        if self.native_route and self._binding_cfg.identity_readout_source_boundary_adapter and self.native_source_boundary_adapter.out_features!=len(ids)+extra:
            prior=self.native_source_boundary_adapter
            projection=nn.Linear(2*self.query.in_features,len(ids)+extra,bias=False).to(device=prior.weight.device,dtype=prior.weight.dtype)
            nn.init.zeros_(projection.weight)
            with torch.no_grad():
                for row,token in enumerate(ids):
                    if token in old_ids:projection.weight[row].copy_(prior.weight[old_ids.index(token)])
                if extra:projection.weight[-1].copy_(prior.weight[-1])
            projection.weight.requires_grad_(prior.weight.requires_grad);self.native_source_boundary_adapter=projection

    def _binding_memory(self, entries, tokenizer, bank, lexical_embeddings, device):
        """Input-only identity composition; no output labels, hidden history or profile IDs."""
        if any(len(e) != 3 or len(e[2]) > self.context_units for e in entries):
            raise ValueError('Lexical binding needs its own request state; restart from the input prefix.')
        d = self.query.in_features
        def identity(ids):
            return F.layer_norm((lexical_embeddings(ids) + bank.encode('token', ids)) / math.sqrt(2.), (d,))
        ids = torch.tensor([e[0] for e in entries], device=device)
        lengths = torch.tensor([len(e[2]) for e in entries], device=device)
        question = F.layer_norm(identity(ids[-self.query_units:]).mean(0), (d,))
        if self.positional_binding:
            # Right-align bounded windows: slot meaning survives short prefixes.
            aligned = torch.tensor([[tokenizer.pad_id] * (self.context_units - len(e[2])) + list(e[2])
                                    for e in entries], dtype=torch.long, device=device)
            mask = torch.arange(self.context_units, device=device)[None,:] >= self.context_units - lengths[:,None]
            slot_logits = self.context_slot_logits.expand(len(entries), -1)
            weights = slot_logits.masked_fill(~mask, torch.finfo(slot_logits.dtype).min).softmax(-1) * mask
            relation = (identity(aligned) * weights.unsqueeze(-1)).sum(1)
            query_ids = ids[-self.query_units:]
            query_weights = self.query_slot_logits[-query_ids.numel():].softmax(-1)
            requested = (identity(query_ids) * query_weights.unsqueeze(-1)).sum(0)
            span_scores = F.cosine_similarity(self.span_projection(relation),
                                             (self.span_projection(requested) + self.context_query(question))[None,:], dim=-1) * math.sqrt(d)
        else:
            previous = torch.tensor([list(e[2]) + [tokenizer.pad_id] * (self.context_units - len(e[2]))
                                     for e in entries], dtype=torch.long, device=device)
            valid = torch.arange(self.context_units, device=device)[None,:] < lengths[:,None]
            context = (identity(previous) * valid.unsqueeze(-1)).sum(1) / valid.sum(-1,keepdim=True).clamp_min(1)
            context = F.layer_norm(context, (d,))
            span_scores = (self.span_projection(context) * self.span_projection(question)).sum(-1) / math.sqrt(d)
        return question, span_scores, identity

    def _word_binding_memory(self, entries, words, tokenizer, bank, lexical_embeddings, device):
        """Compositional words share lexical and character features in the existing bank."""
        d=self.query.in_features
        def identity(ids):
            return F.layer_norm((lexical_embeddings(ids)+bank.encode('token',ids))/math.sqrt(2.),(d,))
        question_words=words.question()
        contexts=[e[2] for e in entries]
        right_contexts=[[] for _ in entries]
        if self.observed_span_binding:
            if any(len(e)!=5 for e in entries):raise ValueError('Observed-span binding needs its own request state.')
            records={serial:(clause,index) for index,(serial,clause,_) in enumerate(words.records)}
            for index,e in enumerate(entries):
                if e[3] in records:
                    clause,position=records[e[3]]
                    for _,other,word in words.records[position+1:position+17]:
                        if other!=clause:break
                        right_contexts[index].append(word)
        unique=list(dict.fromkeys(w for context in contexts+right_contexts+[question_words] for w in context))
        if self._binding_cfg.identity_readout_source_ownership and self._binding_cfg.identity_readout_source_start_features=='span_roles_v2':
            unique=list(dict.fromkeys(unique+[word for _,_,word in words.records]))
        if not unique:
            zero=torch.zeros(d,device=device,dtype=self.query.weight.dtype)
            return zero,torch.zeros(len(entries),device=device),identity
        addresses={word:i for i,word in enumerate(unique)}
        vectors=[]
        for word in unique:
            ids=torch.tensor(word,device=device)
            text=tokenizer.decode(list(word),clean_text=False,collapse_structure=False)
            grams=['word-shape:length='+str(RuntimeStructuralState.bucket(len(text)))]
            padded='^'+text+'$'
            for n in (2,3):grams.extend('word-gram:'+padded[i:i+n] for i in range(max(0,len(padded)-n+1)))
            hashes=torch.tensor([1+stable_key(g)%bank.capacity for g in grams],device=device)
            vectors.append(F.layer_norm(identity(ids).mean(0)+bank.embedding(hashes).mean(0),(d,)))
        vectors=torch.stack(vectors)
        def compose(sequence,attention):
            if not sequence:return torch.zeros(d,device=device,dtype=vectors.dtype)
            value=vectors[torch.tensor([addresses[w] for w in sequence],device=device)]
            return (value*attention(value).squeeze(-1).softmax(-1)[:,None]).sum(0)
        # Question uses all complete words within the bounded input capacity, not a unit tail.
        # Keep router/format geometry fixed during adapter fitting; train attention only for matching.
        question=F.layer_norm(vectors[torch.tensor([addresses[w] for w in question_words],device=device)].mean(0),(d,))
        requested=compose(question_words,self.question_word_attention)
        context=torch.stack([compose(c,self.context_word_attention) for c in contexts])
        if self.observed_span_binding:
            right=torch.stack([compose(c,self.context_word_attention) for c in right_contexts])
            if not self.native_route:context=context+self.right_context_projection(right)
        scores=F.cosine_similarity(self.span_projection(context),
            (self.span_projection(requested)+self.context_query(question))[None,:],dim=-1)*math.sqrt(d)
        if self._binding_cfg.identity_readout_ordered_agreement:
            # Reuse one scalar across all identities and formats. Cache composition
            # once per input; do not search the source again at every output step.
            agreement={c:WordSpanState.ordered_agreement(question_words,c) for c in dict.fromkeys(contexts)}
            scores=scores+self.word_agreement_scale*torch.tensor([agreement[c] for c in contexts],device=device,dtype=scores.dtype)
        if self.native_route:
            native_context=context+self.right_context_projection(right)
            packet=None
            if self._binding_cfg.identity_readout_native_word_paths:
                groups={}
                for index,e in enumerate(entries):groups.setdefault(e[3],[]).append(index)
                ordinals={i:j+1 for group in groups.values() for j,i in enumerate(group)}
                sizes={i:len(group) for group in groups.values() for i in group}
                props=torch.tensor([e[1] for e in entries],device=device,dtype=torch.long)
                def count_code(n):return str(n) if n<=16 else 'tail:'+str(RuntimeStructuralState.bucket(n))
                extra=torch.tensor([[1+stable_key('source:unit_pos='+count_code(ordinals[i]))%bank.capacity,
                                     1+stable_key('source:word_units='+count_code(sizes[i]))%bank.capacity] for i in range(len(entries))],device=device)
                signatures=F.layer_norm(bank.encode_runtime(props)+bank.embedding(extra).mean(1),(d,))
                native_context=native_context+self.native_source_signature(signatures)
                starts=[group[0] for group in groups.values()]
                nodes=[dict(children={},next=[],complete=[],partial=[])]
                for group in groups.values():
                    node=0;start=group[0]
                    for index in group:
                        nodes[node]['next'].append(index);nodes[node]['partial'].append(start)
                        token=entries[index][0]
                        if token not in nodes[node]['children']:
                            nodes[node]['children'][token]=len(nodes);nodes.append(dict(children={},next=[],complete=[],partial=[]))
                        node=nodes[node]['children'][token]
                    nodes[node]['complete'].append(start)
                packet=(starts,nodes)
            native_scores=F.cosine_similarity(self.span_projection(native_context),
                (self.span_projection(requested)+self.context_query(question)+self.native_query(question))[None,:],dim=-1)*math.sqrt(d)
            question_vectors=vectors[torch.tensor([addresses[w] for w in question_words],device=device)]
            # Bounded router preserves both the start/task cue and end/requested span of long clauses.
            if len(question_vectors)>64:question_vectors=torch.cat((question_vectors[:32],question_vectors[-32:]))
            route_features=torch.zeros(64,d+1,device=device,dtype=vectors.dtype)
            route_features[:len(question_vectors),:d]=question_vectors
            route_features[:len(question_vectors),d]=1
            result=(question,scores,identity,native_scores,route_features)
            if self._binding_cfg.identity_readout_source_ownership:
                def neighbor(sequence,index):
                    return vectors[addresses[sequence[index]]] if len(sequence)>=abs(index) and sequence else torch.zeros(d,device=device,dtype=vectors.dtype)
                role_keys=[]
                owned_contexts=contexts
                if self._binding_cfg.identity_readout_source_start_features=='span_roles_v2':
                    record_positions={serial:i for i,(serial,_,_) in enumerate(words.records)}
                    owned_contexts=[c or [r[2] for r in words.records[max(0,record_positions.get(e[3],0)-2):record_positions.get(e[3],0)]] for e,c in zip(entries,contexts)]
                for e,c in zip(entries,contexts):
                    gap=words.following.get(e[3],[])
                    marker=tokenizer.construction_units[gap[0]].text if gap else 'unobserved'
                    keys=['next='+marker,'clause_start='+str(not c)]
                    if self._binding_cfg.identity_readout_source_start_features=='span_roles_v2':
                        index=record_positions.get(e[3]);clause=words.records[index][1] if index is not None else None;group=[r for r in words.records if clause is not None and r[1]==clause];ending=words.following.get(group[-1][0],[]) if group else []
                        end_marker=tokenizer.construction_units[ending[0]].text if ending else 'unobserved'
                        line_end=any(tokenizer.construction_units[x].text=='<EOL>' for x in ending)
                        keys+=['span_end='+end_marker,'span_line_end='+str(line_end),'span_words='+str(RuntimeStructuralState.bucket(len(group)))]
                    role_keys.append([1+stable_key('source_start:'+key)%bank.capacity for key in keys])
                roles=bank.embedding(torch.tensor(role_keys,device=device)).mean(1)
                start_features=torch.cat((torch.stack([neighbor(c,-1) for c in owned_contexts]),torch.stack([neighbor(c,-2) for c in owned_contexts]),torch.stack([neighbor(c,0) for c in right_contexts]),signatures+roles),dim=-1)
                correction=(self.native_span_start(start_features)*question).sum(-1)/math.sqrt(d)
                return result+(packet,correction)
            return result+(packet,) if packet is not None else result
        return question,scores,identity

    def _binding_structure(self, token, unit, runtime_features, bank, identity, hidden):
        # Preserve meaningful marker identity while keeping lexical value identity out of formatting.
        control = identity(torch.tensor(token,device=hidden.device)) if not unit.is_lexical else torch.zeros_like(hidden)
        return F.layer_norm(bank.encode_runtime(runtime_features) + control, (hidden.size(-1),))

    def binding_activation(self, question):
        probability = self.applicability(question).sigmoid().squeeze(-1)
        return self._confidence_activation(probability)

    @staticmethod
    def retrieval_margin_loss(scores, positive, eligible, margin=2.):
        """Training-only weak supervision over observed source candidates.

        Answer identity may mark several source occurrences as positive. Query
        candidates must be excluded by eligible; no output-prefix features enter
        the scores. This operates before learned attention temperature can hide
        a fragile source-selection margin behind low token loss.
        """
        if margin<=0:raise ValueError('Retrieval margin must be positive')
        if scores.shape!=positive.shape or scores.shape!=eligible.shape:
            raise ValueError('Retrieval scores and candidate masks must agree')
        positive=positive.bool()&eligible.bool();negative=eligible.bool()&~positive
        if not positive.any(-1).all() or not negative.any(-1).all():
            raise ValueError('Retrieval supervision needs both positive and competing source candidates')
        best_positive=scores.masked_fill(~positive,float('-inf')).max(-1).values
        best_negative=scores.masked_fill(~negative,float('-inf')).max(-1).values
        return F.relu(margin-best_positive+best_negative).square().mean()

    def calibrate_task_route(self, negative, positive, *, native=False, steps=2000, lr=.003):
        """Fit an isolated classifier; commit only confidently separated input routes.

        Failed recalibration leaves both the accepted weights and their training
        flags untouched. Features must come from observed inputs, never answers.
        """
        if steps<1 or lr<=0:raise ValueError('Calibration steps and learning rate must be positive')
        if not self.binding_enabled or (native and not self.native_route):
            raise ValueError('Requested task route is not available')
        if not len(negative) or not len(positive):raise ValueError('Both route classes need observed inputs')
        target=self.native_applicability if native else self.applicability
        candidate=copy.deepcopy(target).requires_grad_(True)
        parameter=next(candidate.parameters())
        negative=negative.detach().to(device=parameter.device,dtype=parameter.dtype)
        positive=positive.detach().to(device=parameter.device,dtype=parameter.dtype)
        if not torch.isfinite(negative).all() or not torch.isfinite(positive).all():
            raise ValueError('Route features must be finite')
        with torch.enable_grad():
            optimizer=torch.optim.Adam(candidate.parameters(),lr=lr)
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                neg=candidate(negative);pos=candidate(positive)
                loss=.5*F.binary_cross_entropy_with_logits(neg,torch.zeros_like(neg))+.5*F.binary_cross_entropy_with_logits(pos,torch.ones_like(pos))
                if not torch.isfinite(loss):break
                loss.backward();optimizer.step()
        with torch.no_grad():
            negative_max=float(candidate(negative).sigmoid().max())
            positive_min=float(candidate(positive).sigmoid().min())
        accepted=negative_max<=1.-self.binding_confidence and positive_min>=self.binding_confidence
        if accepted:target.load_state_dict(candidate.state_dict())
        return dict(accepted=accepted,negative_max=negative_max,positive_min=positive_min,steps=steps)

    def _confidence_activation(self, probability):
        # Confident retained-task routing is exactly neutral; uncertain requests stay blended.
        return torch.where(probability <= 1. - self.binding_confidence, torch.zeros_like(probability),
                           torch.where(probability >= self.binding_confidence, torch.ones_like(probability), probability))

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

    def forward(self, logits, hidden, input_ids, features, tokenizer, bank, states=None, *, lexical_embeddings=None):
        if self.binding_enabled and lexical_embeddings is None:
            raise ValueError('Lexical binding requires the model lexical embedding channel.')
        self.configure(tokenizer,logits.size(-1))
        if states is not None and len(states) != input_ids.size(0):
            raise ValueError('Identity readout state batch size differs from input.')
        states = [IdentityReadoutState(list(s.entries), s.role, s.words.clone() if s.words is not None else None,
                                      list(s.output_units) if s.output_units is not None else None,s.binding_cache,s.source_mask,s.word_node,s.word_prior,s.output_word_count,s.output_in_word,
                                      list(s.output_separators) if s.output_separators is not None else None,s.source_initial_prior,s.source_cursor) for s in states] if states is not None else [
            IdentityReadoutState([]) for _ in range(input_ids.size(0))]
        rows=[]
        feature_rows=features.tolist()
        for row,(tokens,state) in enumerate(zip(input_ids.tolist(),states)):
            steps=[]
            binding_memory = None
            cache_stamp=None
            if self.binding_enabled and self.word_binding and not torch.is_grad_enabled():
                cache_stamp=(id(self),id(bank),id(lexical_embeddings),
                    tuple(p._version for p in self.parameters()),tuple(p._version for p in lexical_embeddings.parameters()),bank.embedding.weight._version)
                if state.binding_cache is not None and state.binding_cache[0]==cache_stamp:
                    binding_memory=state.binding_cache[1]
            for position,token in enumerate(tokens):
                unit=tokenizer.construction_units[token]
                if token==tokenizer.bos_id or unit.text=='<BOI>':
                    state.entries=[]
                    if self.word_binding:state.words=WordSpanState(self.capacity)
                    if self.observed_span_binding:state.output_units=[]
                    binding_memory = None
                    state.binding_cache=None
                    state.source_mask=None
                    state.word_node=None;state.word_prior=None
                    state.output_word_count=0;state.output_in_word=False
                    state.output_separators=[]
                    state.source_initial_prior=None;state.source_cursor=None
                    state.role='input' if unit.text=='<BOI>' else 'outside'
                elif unit.text=='<EOI>':state.role='outside'
                elif unit.text=='<BOO>':
                    state.role='output'
                    state.output_word_count=0;state.output_in_word=False
                    state.output_separators=[]
                    state.source_initial_prior=None;state.source_cursor=None
                    if self.native_route and self._binding_cfg.identity_readout_native_word_paths:state.word_node=0;state.word_prior=None
                elif unit.text=='<EOO>':state.role='outside'
                if self.word_binding and (state.role=='input' or unit.text=='<EOI>'):
                    if state.words is None:raise ValueError('Word binding requires its own request state; restart from input.')
                    if self.is_candidate(unit):
                        # All fragments in one word receive the same complete preceding-word context.
                        context=tuple(state.words.clause[-16:])
                        entry=(token,tuple(feature_rows[row][position]),context)
                        if self.observed_span_binding:entry+=(state.words.serial,tuple(e[0] for e in state.entries[-8:]))
                        state.entries.append(entry)
                        state.entries=state.entries[-self.capacity:]
                    state.words.observe(token,unit)
                    binding_memory=None
                    state.binding_cache=None
                    state.source_mask=None
                    state.word_node=None;state.word_prior=None
                elif state.role=='input' and self.is_candidate(unit):
                    entry = (token, tuple(feature_rows[row][position]))
                    if self.binding_enabled:
                        entry += (tuple(e[0] for e in state.entries[-self.context_units:]),)
                    state.entries.append(entry)
                    state.entries=state.entries[-self.capacity:]
                    binding_memory = None
                current=logits[row,position]
                cursor_gap=state.output_separators or []
                if self.observed_span_binding and state.role=='output' and unit.is_lexical:
                    state.output_units=(state.output_units+[token])[-8:]
                    state.output_separators=[]
                elif self.observed_span_binding and state.role=='output' and state.output_units and unit.kind!='signature':
                    state.output_separators=(state.output_separators or [])+[token]
                    state.output_separators=state.output_separators[-self.capacity:]
                if state.role=='output':
                    if unit.is_lexical:
                        if not state.output_in_word:state.output_word_count+=1
                        state.output_in_word=True
                    elif unit.kind not in {'case','signature'}:state.output_in_word=False
                if state.role=='output' and state.entries:
                    ids=torch.tensor([e[0] for e in state.entries],device=logits.device)
                    properties=torch.tensor([e[1] for e in state.entries],device=logits.device)
                    candidates=(bank.encode('token',ids)+bank.encode_runtime(properties))/math.sqrt(2.)
                    keys=self.key(candidates)
                    h=hidden[row,position]
                    # All bounded candidates participate in training, so q/k receive gradients.
                    query = self.query(h)
                    gate=self.gate(h).float().squeeze(-1)
                    if self.binding_enabled:
                        if binding_memory is None:
                            binding_memory = (self._word_binding_memory(state.entries,state.words,tokenizer,bank,lexical_embeddings,logits.device)
                                if self.word_binding else self._binding_memory(state.entries, tokenizer, bank, lexical_embeddings, logits.device))
                            if cache_stamp is not None:state.binding_cache=(cache_stamp,binding_memory)
                        question, span_scores, identity = binding_memory[:3]
                        if self.native_route and self._binding_cfg.identity_readout_native_word_paths:
                            nodes=binding_memory[5][1]
                            if unit.is_lexical:
                                state.word_node=nodes[state.word_node]['children'].get(token) if state.word_node is not None else None
                            elif unit.kind not in {'case','signature'}:state.word_node=0;state.word_prior=None
                        activation = self.binding_activation(question)
                        # Formatting depends on causal properties, not a particular value's identity.
                        structure = self._binding_structure(token, unit, features[row,position], bank, identity, h)
                        query = query + activation * self.context_query(question)
                        gate = (self.gate(h) + activation * self.context_gate(structure,question)).float().squeeze(-1)
                        delta = activation * self.surface_projection(torch.cat((structure, structure * question)))
                        current = current.index_add(0, self.surface_ids, delta)
                        if self.native_route:
                            native_activation=self._confidence_activation(self.native_applicability(binding_memory[4]).sigmoid().squeeze(-1))
                            gate=gate+native_activation*self.native_gate(structure,question).squeeze(-1)
                            current=current.index_add(0,self.surface_ids,native_activation*self.native_surface_projection(torch.cat((structure,structure*question))))
                    scores=(keys*query).sum(-1)/math.sqrt(keys.size(-1))
                    if self.binding_enabled:
                        if self.positional_binding:
                            # A confident relation selects by its context, never the value's identity.
                            # Uncertain requests interpolate with the retained independent readout.
                            scores = (1. - activation) * scores + activation * self.span_scale.clamp(-4.,4.).exp() * span_scores
                        else:
                            scores = scores + activation * self.span_scale * span_scores
                        if self.native_route:
                            scores=(1.-native_activation)*scores+native_activation*(self.span_scale+self.native_scale).clamp(-4.,4.).exp()*binding_memory[3]
                            if self._binding_cfg.identity_readout_source_ownership and not state.output_units:
                                scores=scores+native_activation*binding_memory[6]
                        if self.observed_span_binding and state.output_units:
                            # Exact observed prefix equality verifies a possible source continuation;
                            # learned strength competes with ordinary attention, never forces a copy.
                            matches=[]
                            for entry in state.entries:
                                length=0
                                for n in range(1,min(len(entry[4]),len(state.output_units))+1):
                                    if tuple(state.output_units[-n:])==entry[4][-n:]:length=n
                                matches.append(length)
                            scores=scores+(native_activation if self.native_route else activation)*self.continuation_scale*torch.tensor(matches,device=logits.device,dtype=scores.dtype)
                    attention=F.softmax(scores.float(),-1)
                    if self.observed_span_binding and self._binding_cfg.identity_readout_exclude_query_candidates:
                        if state.source_mask is None or state.source_mask.device!=logits.device:
                            state.source_mask=torch.tensor(state.words.source_candidate_mask(state.entries),device=logits.device,dtype=torch.bool)
                        source_mask=state.source_mask
                        source_attention=F.softmax(scores.float().masked_fill(~source_mask,float('-inf')),-1)
                        # Independent legacy retrieval has its own input contract. Apply
                        # the source/query partition only through the learned QA routes.
                        partition_strength=torch.maximum(activation,native_activation) if self.native_route else activation
                        attention=(1.-partition_strength)*attention+partition_strength*source_attention
                    if self.native_route and self._binding_cfg.identity_readout_native_word_paths:
                        starts,nodes=binding_memory[5]
                        proposal=None;cursor_strength=None
                        if self._binding_cfg.identity_readout_source_ownership:
                            if unit.is_lexical:
                                prior=(state.source_initial_prior if state.source_cursor is None else self.source_cursor_targets(state,cursor_gap,ids))
                                if prior is None:prior=torch.zeros_like(scores,dtype=torch.float32)
                                posterior=prior*ids.eq(token)
                                state.source_cursor=posterior/posterior.sum().clamp_min(1e-30)
                            proposal=self.source_cursor_targets(state,state.output_separators or [],ids)
                            token_proposal=torch.zeros_like(current,dtype=torch.float32).scatter_add(0,ids,proposal)
                            cursor_strength=native_activation*self.native_source_cursor_scale.clamp(0.,1.)*token_proposal.max()
                        if state.word_node==0:
                            eligible=state.source_mask if state.source_mask is not None else torch.ones_like(ids,dtype=torch.bool)
                            start_scores=scores[starts].float().masked_fill(~eligible[starts],float('-inf'))
                            if not eligible[starts].any():start_scores=scores[starts].float()
                            prior=torch.zeros_like(scores,dtype=torch.float32).index_add(0,torch.tensor(starts,device=logits.device),start_scores.softmax(-1))
                            if proposal is not None:prior=(1.-cursor_strength)*prior+cursor_strength*proposal
                            state.word_prior=prior
                            if self._binding_cfg.identity_readout_source_ownership and not state.output_units:state.source_initial_prior=prior
                        if self._binding_cfg.identity_readout_native_conditional_prefix and state.word_node is not None and state.word_node!=0 and state.word_prior is not None:
                            state.word_prior=self.condition_word_prior(state.word_prior,nodes[state.word_node])
                        complete=partial=torch.zeros((),device=logits.device)
                        if state.word_node is not None and state.word_node!=0 and state.word_prior is not None:
                            node=nodes[state.word_node]
                            complete=state.word_prior[node['complete']].sum()
                            partial=state.word_prior[node['partial']].sum()
                        status=bank.embedding(torch.tensor([1+stable_key('source:word_complete')%bank.capacity,1+stable_key('source:word_partial')%bank.capacity],device=logits.device))
                        evidence=complete*status[0]+partial*status[1]
                        if state.word_node==0 and state.word_prior is not None:
                            # Expected next source-word properties survive a generated
                            # separator; a zero scalar keeps earlier checkpoints neutral.
                            source_start=bank.encode_runtime(properties[starts])
                            expected=(source_start*state.word_prior[starts,None]).sum(0)
                            evidence=evidence+self.native_word_start_scale*expected
                        word_structure=structure*evidence
                        current=current.index_add(0,self.surface_ids,native_activation*self.native_word_surface(torch.cat((word_structure,word_structure*question))))
                        gate=gate+native_activation*self.native_word_gate(word_structure,question).squeeze(-1)
                        if state.word_node is not None:
                            confidence=state.word_prior[starts].max() if state.word_node==0 else (complete+partial)
                            valid=torch.zeros_like(scores).index_fill(0,torch.tensor(nodes[state.word_node]['next'],device=logits.device,dtype=torch.long),1.)
                            word_scores=scores+native_activation*self.native_word_path_scale*confidence*valid
                            word_attention=F.softmax(word_scores.float(),-1)
                            if state.source_mask is not None:
                                source_attention=F.softmax(word_scores.float().masked_fill(~state.source_mask,float('-inf')),-1)
                                partition_strength=torch.maximum(activation,native_activation)
                                word_attention=(1.-partition_strength)*word_attention+partition_strength*source_attention
                            attention=word_attention
                        if proposal is not None and bool(cursor_strength>0):
                            attention=(1.-cursor_strength)*attention+cursor_strength*proposal
                        if self._binding_cfg.identity_readout_native_span_boundaries:
                            span_vector=self.span_boundary_vector(bank,state,unit,complete,partial,attention,
                                matches if state.output_units else [],nodes)
                            span_features=torch.cat((span_vector,span_vector*question))
                            span_logits=self.native_span_surface(span_features)
                            if self._binding_cfg.identity_readout_source_boundaries:
                                source_vector=self.source_boundary_vector(bank,state,tokenizer,logits.device)
                                source_logits=self.source_boundary_logits(torch.cat((source_vector,source_vector*question)))
                                if self._binding_cfg.identity_readout_source_boundary_readout=='categorical_v1':
                                    source_confidence=source_logits.float().softmax(-1).max()
                                    if bool(source_vector.abs().sum()>0) and bool(source_confidence>=self._binding_cfg.identity_readout_native_span_confidence):
                                        span_logits=source_logits
                                else:span_logits=span_logits+source_logits
                            span_gate=self.native_span_gate(span_features).squeeze(-1)
                            if self._binding_cfg.identity_readout_native_span_readout=='residual_v1':
                                current=current.index_add(0,self.surface_ids,native_activation*span_logits)
                                gate=gate+native_activation*span_gate
                    copy=torch.zeros_like(current,dtype=torch.float32).scatter_add(0,ids,attention)
                    span_base=current
                    current=self.mix_probabilities(current,copy,gate)
                    if self.native_route and self._binding_cfg.identity_readout_native_span_boundaries and self._binding_cfg.identity_readout_native_span_readout=='categorical_v2' and bool(self.native_span_ready):
                        supported=state.word_node is not None and state.word_prior is not None and bool(state.word_prior.sum()>0)
                        if supported:
                            macro=span_logits.float().softmax(-1)
                            confidence=macro.max()
                            strength=native_activation*torch.where(confidence>=self._binding_cfg.identity_readout_native_span_confidence,
                                self._confidence_activation(confidence),torch.zeros_like(confidence))
                            if bool(strength>0):
                                canonical=self.span_categorical_probabilities(span_base,copy,span_gate,macro)
                                current=((1.-strength)*current.float().exp()+strength*canonical).clamp_min(1e-30).log().to(current.dtype)
                steps.append(current)
            rows.append(torch.stack(steps))
        return torch.stack(rows),states


    def span_categorical_probabilities(self,base,copy,gate,macro):
        mask=torch.ones_like(base,dtype=torch.bool).index_fill(0,self.surface_ids,False)
        fallback=base[mask].float().softmax(-1)
        mass=copy[mask].sum()
        copied=torch.where(mass>0,copy[mask]/mass.clamp_min(1e-30),fallback)
        lexical=(1.-gate.float().sigmoid())*fallback+gate.float().sigmoid()*copied
        result=torch.zeros_like(base,dtype=torch.float32).index_copy(0,self.surface_ids,macro[:-1])
        result[mask]=macro[-1]*lexical
        return result

    def source_boundary_logits(self,features):
        base=self.native_source_boundary(features)
        if not self._binding_cfg.identity_readout_source_boundary_adapter:return base
        proposed=base+self.native_source_boundary_adapter(features)
        selected=proposed.float().softmax(-1).max(-1).values>=self._binding_cfg.identity_readout_native_span_confidence
        return torch.where(selected.unsqueeze(-1),proposed,base)

    def commit_span_calibration(self,macro_logits,targets):
        """Training-only guard; uncertain positions retain the previous readout."""
        if not self._binding_cfg.identity_readout_native_span_boundaries or self._binding_cfg.identity_readout_native_span_readout!='categorical_v2':
            raise ValueError('Categorical span calibration requires its readout')
        if macro_logits.ndim!=2 or macro_logits.size(1)!=self.surface_ids.numel()+1 or targets.shape!=macro_logits.shape[:1]:
            raise ValueError('Span calibration logits and targets differ')
        with torch.no_grad():
            p=macro_logits.float().softmax(-1);confidence,prediction=p.max(-1)
            selected=confidence>=self._binding_cfg.identity_readout_native_span_confidence
            count=int(selected.sum());correct=int((selected&prediction.eq(targets)).sum())
            accepted=count>0 and count==correct
            if accepted:self.native_span_ready.fill_(True)
        return dict(selected_positions=count,selected_correct=correct,accepted=accepted)

    @staticmethod
    def source_cursor_targets(state,emitted_separators,ids):
        """Advance only contiguous, observed source occurrences across verified gaps."""
        result=torch.zeros(ids.numel(),device=ids.device,dtype=torch.float32)
        if state.source_cursor is None or state.words is None:return result
        eligible=state.words.source_candidate_mask(state.entries)
        previous=[];following=[]
        for i in range(len(state.entries)-1):
            if not eligible[i] or not eligible[i+1]:continue
            serial=state.entries[i][3];other=state.entries[i+1][3]
            if other not in (serial,serial+1):continue
            gap=[] if other==serial else state.words.following.get(serial,[])
            if gap==list(emitted_separators):previous.append(i);following.append(i+1)
        if previous:
            positions=torch.tensor(previous,device=ids.device)
            result=result.index_add(0,torch.tensor(following,device=ids.device),state.source_cursor[positions])
        return result/result.sum().clamp_min(1e-30)

    @staticmethod
    def source_boundary_vector(bank,state,tokenizer,device):
        """Observed source occurrences, aligned to the emitted lexical/surface suffix.

        Matching selects evidence, not output tokens. Repeated occurrences share
        posterior mass; unsupported prefixes contribute zero. Following separators
        come only from the already observed input, never future output labels.
        """
        zero=bank.embedding.weight.new_zeros(bank.embedding.embedding_dim)
        if not state.output_units or state.words is None:return zero
        records={serial:(clause,word) for serial,clause,word in state.words.records}
        eligible=state.words.source_candidate_mask(state.entries) if state.source_mask is not None else [True]*len(state.entries)
        occurrences=[];best=0
        for i,entry in enumerate(state.entries):
            if not eligible[i]:continue
            history=entry[4]+(entry[0],)
            length=0
            for n in range(1,min(len(history),len(state.output_units))+1):
                if history[-n:]==tuple(state.output_units[-n:]):length=n
            if not length or length<best:continue
            serial=entry[3]
            if serial not in records:continue
            word=records[serial][1]
            # Locate this fragment within its source word using entry order.
            offset=sum(e[3]==serial for e in state.entries[:i+1])-1
            within=offset+1<len(word)
            following=[] if within else state.words.following.get(serial,[])
            emitted=state.output_separators or []
            if emitted and (within or following[:len(emitted)]!=emitted):continue
            if length>best:occurrences=[];best=length
            next_marker='LEX' if within or len(emitted)>=len(following) else tokenizer.construction_units[following[len(emitted)]].text
            clause_end=any(u.text=='<EOL>' or any(c in '.!?\n' for c in u.render) for u in (tokenizer.construction_units[x] for x in following))
            line_end=any(tokenizer.construction_units[x].text=='<EOL>' for x in following)
            occurrences.append((next_marker,clause_end,line_end,within))
        if not occurrences:return zero
        values={}
        for marker,end,line,within in occurrences:
            for key,value in [('next='+marker,1.),('clause_end',float(end)),('line_end',float(line)),('within_word',float(within)),('supported',1.)]:
                values[key]=values.get(key,0.)+value/len(occurrences)
        values['separator_offset='+str(min(len(state.output_separators or []),8))]=1.
        values['ambiguous']=float(len(set(occurrences))>1)
        keys=torch.tensor([[1+stable_key(salt+'source_boundary:'+key)%bank.capacity for salt in ('a:','b:')] for key in values],device=device)
        weights=bank.embedding.weight.new_tensor(list(values.values()))
        return (bank.embedding(keys).mean(1)*weights[:,None]).sum(0)/math.sqrt(8.)

    @staticmethod
    def span_boundary_vector(bank,state,unit,complete,partial,attention,matches,nodes):
        """Canonical word/span roles: independent of the last fragment's identity/shape."""
        continuation=torch.zeros_like(complete)
        if matches:
            verified=torch.tensor([n>0 for n in matches],device=attention.device,dtype=torch.bool)
            if state.source_mask is not None:verified=verified&state.source_mask
            continuation=attention[verified].sum()
        within=torch.zeros_like(complete)
        if state.word_node is not None:
            within=attention[nodes[state.word_node]['next']].sum()
        marker='lexical' if unit.is_lexical else unit.text
        position=str(state.output_word_count) if state.output_word_count<=8 else 'tail'
        values=[('word_complete',complete),('word_partial',partial),('verified_continuation',continuation),
                ('within_word',within),('root',float(state.word_node==0)),('unmatched',float(state.word_node is None)),
                ('word_position='+position,1.),('marker='+marker,1.)]
        keys=torch.tensor([[1+stable_key(salt+'span:'+key)%bank.capacity for salt in ('a:','b:')] for key,_ in values],device=attention.device)
        weights=torch.stack([v if isinstance(v,torch.Tensor) else complete.new_tensor(v) for _,v in values]).to(bank.embedding.weight.dtype)
        return (bank.embedding(keys).mean(1)*weights[:,None]).sum(0)/math.sqrt(len(values))

    @staticmethod
    def condition_word_prior(prior,node):
        """Verified lexical prefixes condition identity support, not output choices."""
        supported=torch.zeros_like(prior).index_fill(0,torch.tensor(node['complete']+node['partial'],device=prior.device,dtype=torch.long),1.)
        posterior=prior*supported
        return posterior/posterior.sum().clamp_min(torch.finfo(prior.dtype).tiny)


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
