"""Bounded antecedent identities with causal, learned ownership updates."""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
try:
    from .native_owner_evidence import NativeOwnerEvidence
except ImportError:
    from native_owner_evidence import NativeOwnerEvidence


@dataclass(frozen=True)
class AntecedentOccurrence:
    words: tuple
    ordinals: tuple
    properties: tuple


class NativeAntecedentMemory(nn.Module):
    def __init__(self, width, capacity=64, prefix_words=8, identity_query='learned_v1', context_evidence=False):
        super().__init__()
        if capacity < 1 or prefix_words < 1:
            raise ValueError('Antecedent capacities must be positive.')
        self.capacity = int(capacity)
        self.prefix_words = int(prefix_words)
        if identity_query not in {'learned_v1', 'ordered_v1'}:
            raise ValueError('Unknown antecedent identity query')
        self.identity_query = identity_query
        self.role = nn.Sequential(nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.role[-1].weight)
        nn.init.zeros_(self.role[-1].bias)
        self.request = nn.Linear(width, 1, bias=False)
        nn.init.zeros_(self.request.weight)
        self.scale = nn.Parameter(torch.zeros(()))
        self.context_evidence = NativeOwnerEvidence(width) if context_evidence else None

    def observe(self, state, word, properties):
        ordinal = state.source_seen
        state.source_seen += 1
        state.source_ordinals = (state.source_ordinals + [ordinal])[-len(state.source):]
        if state.clause_seen == 0:
            state.antecedents.append(AntecedentOccurrence((word,), (ordinal,), properties))
            state.antecedents = state.antecedents[-self.capacity:]
        elif state.antecedents and len(state.antecedents[-1].words) < self.prefix_words:
            last = state.antecedents[-1]
            state.antecedents[-1] = AntecedentOccurrence(last.words + (word,), last.ordinals + (ordinal,), last.properties)

    def access(self, state, query_words, word_vector, context_encoder, encode_sequences):
        # Occurrence numbers are only causal masks, never embedding features.
        # A prefix is truncated at the queried source word: later source words
        # cannot retroactively describe an earlier occurrence.
        scopes, lookup, rows = [], {}, []
        for ordinal in state.source_ordinals:
            indices = []
            for event in state.antecedents:
                words = tuple(w for w, p in zip(event.words, event.ordinals) if p <= ordinal)
                if not words:
                    continue
                identity = (words, event.properties)
                if identity not in lookup:
                    lookup[identity] = len(scopes)
                    scopes.append(identity)
                indices.append(lookup[identity])
            rows.append(indices)
        query = torch.stack([word_vector(w) for w in query_words])
        request = self.request(query).squeeze(-1).softmax(0)
        if not scopes:
            return request.new_zeros(len(rows)), dict(scopes=[], role_logits=request.new_empty(0))
        contexts = encode_sequences(context_encoder, [torch.stack([word_vector(w) for w in words]) for words, _ in scopes])
        properties = contexts.new_tensor([props for _, props in scopes])
        role_logits = self.role(torch.cat((contexts, properties), -1)).squeeze(-1)
        legacy_role_logits = role_logits
        evidence_logits = None
        if self.context_evidence is not None:
            features = self.context_evidence.features(scopes, word_vector,
                lambda sequences: encode_sequences(context_encoder, sequences), contexts, self.prefix_words)
            evidence_logits = self.context_evidence(features)
            # Either learned identity evidence or a causal declaration may
            # establish an owner; absence of a declaration does not erase it.
            log_no_update = F.logsigmoid(-role_logits) + F.logsigmoid(-evidence_logits)
            log_update = torch.log(-torch.expm1(log_no_update))
            role_logits = log_update - log_no_update
        if self.identity_query == 'ordered_v1':
            # Exact ordered identities must not depend on whether a learned
            # embedding pool recognizes a new name. Longer overlapping spans
            # add compositional evidence, without creating vocabulary rows.
            spans = [(tuple(query_words[i:i+n]), n) for n in range(1, min(4, len(query_words)) + 1) for i in range(len(query_words)-n+1)]
            matches = contexts.new_tensor([sum(n for span, n in spans if any(words[j:j+n] == span for j in range(len(words)-n+1))) / len(query_words) for words, _ in scopes])
        else:
            matches = contexts.new_tensor([[float(w in words) for w in query_words] for words, _ in scopes]) @ request
        compatibility, ownership = [], []
        for indices in rows:
            if not indices:
                compatibility.append(request.new_zeros(())); ownership.append(request.new_empty(0))
                continue
            logits = role_logits[indices]
            # Probability that each stored event is the latest owner update.
            # Its exact identity is retained, rather than repeatedly encoded
            # into a hidden vector. No generation-time role threshold is used.
            no_update = F.logsigmoid(-logits) if evidence_logits is None else log_no_update[indices]
            later = no_update.flip(0).cumsum(0).flip(0) - no_update
            updates = F.logsigmoid(logits) if evidence_logits is None else log_update[indices]
            weights = (updates + later).exp()
            compatibility.append(weights @ matches[indices])
            ownership.append(weights)
        return torch.stack(compatibility), dict(scopes=scopes, role_logits=role_logits, request_weights=request, identity_matches=matches, ownership=ownership,
            legacy_role_logits=legacy_role_logits, evidence_logits=evidence_logits)
