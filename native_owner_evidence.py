"""Causal declaration evidence shared across entity identities."""
import torch
from torch import nn


class NativeOwnerEvidence(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, -8.)

    def forward(self, features):
        return self.classifier(features).squeeze(-1)

    @staticmethod
    def features(scopes, word_vector, encode_contexts, reference, prefix_words=8):
        suffixes, lookup, indices = [], {}, []
        for words, _ in scopes:
            suffix = tuple(words[-3:])
            if suffix not in lookup:
                lookup[suffix] = len(suffixes)
                suffixes.append(torch.stack([word_vector(w) for w in suffix]))
            indices.append(lookup[suffix])
        contexts = encode_contexts(suffixes)
        props = reference.new_tensor([(len(words) / prefix_words, min(len(words), 3) / 3) for words, _ in scopes])
        return torch.cat((contexts[indices], props), -1)
