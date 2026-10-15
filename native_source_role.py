"""Candidate-independent local evidence for a requested source answer role."""
import math
import torch
from torch import nn


class NativeSourceRole(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.source = nn.Sequential(nn.Linear(2 * width + 4, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        nn.init.zeros_(self.source[-1].weight)
        nn.init.zeros_(self.source[-1].bias)

    def forward(self, request, features):
        return self.source(features) @ self.query(request) / math.sqrt(request.shape[-1])

    @staticmethod
    def features(source, word_vector, encode_contexts, reference, window=8):
        """Exclude the candidate word from both ordered context encodings.

        The right context is already observed in the input document, bounded by
        the stored clause boundary. No output tokens or answer labels are read.
        """
        scopes, lookup, positions, properties = [], {}, [], []
        for index, (_, before) in enumerate(source):
            left = tuple(before[-window:])
            right = []
            for j in range(index + 1, min(len(source), index + 1 + window)):
                if not source[j][1]:
                    break
                right.append(source[j][0])
            pair = []
            for words in (left, tuple(right)):
                if words and words not in lookup:
                    lookup[words] = len(scopes)
                    scopes.append(torch.stack([word_vector(w) for w in words]))
                pair.append(lookup[words] if words else None)
            positions.append(pair)
            properties.append((len(left) / window, len(right) / window, float(not left), float(not right)))
        encoded = encode_contexts(scopes) if scopes else []
        zero = reference.new_zeros(reference.shape[-1])
        rows = [torch.cat((encoded[a] if a is not None else zero, encoded[b] if b is not None else zero,
                           reference.new_tensor(prop))) for (a, b), prop in zip(positions, properties)]
        return torch.stack(rows) if rows else reference.new_empty((0, 2 * reference.shape[-1] + 4))
