"""Shared lexical and ordered local-phrase evidence for native source selection."""
import torch
from torch import nn


class NativeRelationAlignment(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.word_attention = nn.Linear(width, 1)
        self.phrase_attention = nn.Linear(width, 1)
        self.output = nn.Sequential(nn.Linear(3 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        for attention in (self.word_attention, self.phrase_attention):
            nn.init.zeros_(attention.weight)
            nn.init.zeros_(attention.bias)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, words, phrases, original):
        lexical = self.word_attention(words).squeeze(-1).softmax(0) @ words
        ordered = self.phrase_attention(phrases).squeeze(-1).softmax(0) @ phrases
        # Keep the trained source key space and native emission request intact.
        # Only retrieval's projected request receives this learned correction.
        return original + self.output(torch.cat((lexical, ordered, original)))
