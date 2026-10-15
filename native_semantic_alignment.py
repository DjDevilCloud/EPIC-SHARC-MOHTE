"""Trainable shared question/source encoding for residual native retrieval."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class NativeSemanticAlignment(nn.Module):
    def __init__(self, width, prior_scaling=False):
        super().__init__()
        self.word = nn.GRU(width, width, batch_first=True)
        self.context = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * width, 1)
        self.score = nn.Sequential(nn.Linear(8 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.prior_scale = nn.Linear(2 * width, 1) if prior_scaling else None
        if self.prior_scale is not None:
            nn.init.zeros_(self.prior_scale.weight)
            nn.init.zeros_(self.prior_scale.bias)

    @staticmethod
    def inputs(source_words, question_words, embeddings):
        """Compose exact lexical units without source labels or numeric ID features.

        Only completed source/question words observed in the input are supplied.
        The shared encoder may inspect both directions within that input, never
        any emitted answer. Frozen embeddings may be cached during fitting;
        learned word/context states must be recomputed on every update.
        """
        unique = list(dict.fromkeys(list(source_words) + list(question_words)))
        if not source_words or not question_words:
            raise ValueError('Semantic alignment requires source and question words')
        lookup = {word: i for i, word in enumerate(unique)}
        device = next(embeddings.parameters()).device
        sequences = [embeddings(torch.tensor(w, device=device, dtype=torch.long)) for w in unique]
        return dict(units=pad_sequence(sequences, batch_first=True),
                    lengths=torch.tensor([len(w) for w in unique], device=device),
                    source=torch.tensor([lookup[w] for w in source_words], device=device),
                    question=torch.tensor([lookup[w] for w in question_words], device=device))

    def forward(self, inputs, prior=None, return_features=False):
        """Score starts, optionally returning the same request and source states."""
        states, _ = self.word(inputs['units'])
        words = states[torch.arange(len(states), device=states.device), inputs['lengths'] - 1]
        source, _ = self.context(words[inputs['source']][None])
        question, _ = self.context(words[inputs['question']][None])
        source, question = source[0], question[0]
        request = self.attention(question).squeeze(-1).softmax(0) @ question
        expanded = request.expand_as(source)
        features = torch.cat((source, expanded, source * expanded, (source - expanded).abs()), -1)
        residual = self.score(features).squeeze(-1)
        if self.prior_scale is not None:
            if prior is None or prior.shape != residual.shape:
                raise ValueError('Semantic prior scaling requires one prior logit per source occurrence')
            # Initial scale is exactly one. Positive question-dependent scaling
            # changes the strength, not the ordering, of the existing evidence.
            scale = torch.nn.functional.softplus(self.prior_scale(request)).squeeze(-1) / math.log(2.)
            residual = residual + (scale - 1.) * prior
        if return_features:
            return residual, request, source
        return residual
