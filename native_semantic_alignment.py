"""Trainable shared question/source encoding for residual native retrieval."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NativeSemanticAlignment(nn.Module):
    def __init__(self, width, prior_scaling=False, structure=False, question_attention=False, normalized_fusion=False, fusion_experts='legacy_semantic_v1', question_attention_stage='post_context'):
        super().__init__()
        if fusion_experts not in {'legacy_semantic_v1','corrected_legacy_v1'}:
            raise ValueError('Unknown native source fusion experts')
        self.fusion_experts=fusion_experts
        if question_attention_stage not in {'post_context', 'pre_context'}:
            raise ValueError('Unknown question attention stage')
        if question_attention_stage == 'pre_context' and not question_attention:
            raise ValueError('Pre-context conditioning requires question attention')
        self.question_attention_stage=question_attention_stage
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
        self.structure = nn.Linear(8, width, bias=False) if structure else None
        if self.structure is not None:
            nn.init.zeros_(self.structure.weight)
        self.question_attention = nn.MultiheadAttention(2 * width, 1, batch_first=True) if question_attention else None
        self.question_residual = nn.Linear(2 * width, 2 * width, bias=False) if question_attention else None
        if self.question_residual is not None:
            nn.init.zeros_(self.question_residual.weight)
        self.fusion_gate = nn.Sequential(nn.Linear(3 * width + 5, 2 * width), nn.GELU(), nn.Linear(2 * width, 1)) if normalized_fusion else None
        if self.fusion_gate is not None:
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.constant_(self.fusion_gate[-1].bias, -8.)
        self.temperatures = nn.Parameter(torch.full((2,), math.log(math.expm1(1.)))) if normalized_fusion else None

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

    @staticmethod
    def observed_structure(state, tokenizer, reference):
        """Eight source-only flags, aligned with the retained lexical occurrences."""
        raw = dict(state.surface)
        rows = []
        for i, (first, stop, _) in enumerate(state.source_ranges):
            previous = state.source_ranges[i - 1][1] if i else min(raw, default=first)
            following = state.source_ranges[i + 1][0] if i + 1 < len(state.source_ranges) else stop + 4
            before = [tokenizer.construction_units[raw[p]] for p in range(max(previous, first - 80), first) if p in raw]
            after = [tokenizer.construction_units[raw[p]] for p in range(stop, min(following, stop + 80)) if p in raw]
            units = [tokenizer.construction_units[raw[p]] for p in range(first, stop) if p in raw]
            lines = sum(u.render.count('\n') for u in before)
            adjacent = before + after
            last_boundary = max((j for j, u in enumerate(before) if u.render.isspace() or u.text in {'<LINE>', '<BLO>', '<BOI>'}), default=-1)
            rows.append((float(any(u.kind == 'case' for u in before[last_boundary + 1:])),
                         float(any(u.kind == 'digit' for u in units)),
                         float(not state.source[i][1]), float(lines > 0),
                         float(lines > 1 or any(u.text == '<BLO>' for u in before)),
                         float(any(u.render in {'$', '€', '£', '¥'} for u in adjacent)),
                         float(any(u.render in {'%', '‰'} for u in adjacent)),
                         float(any(u.render in {'"', "'", '(', ')', '[', ']', '{', '}'} for u in adjacent))))
        return reference.new_tensor(rows)

    def condition_source(self, source, question, mask=None, before_context=False):
        """Parameter-matched placement of observed question evidence.

        Before context, duplicate the width-W lexical query into two channels,
        then average the two W residual channels back to W. This fixed mapping
        adds no parameters: both placements use the identical 2W attention and
        residual layers. Padding masks hide other examples' absent question words.
        """
        query = torch.cat((source, source), -1) if before_context else source
        attended, _ = self.question_attention(query, question, question,
                                              key_padding_mask=mask, need_weights=False)
        residual = self.question_residual(attended)
        if before_context:
            left, right = residual.chunk(2, -1)
            residual = (left + right) * .5
        return source + residual

    def encode(self, inputs):
        states, _ = self.word(inputs['units'])
        words = states[torch.arange(len(states), device=states.device), inputs['lengths'] - 1]
        source_words = words[inputs['source']]
        if self.structure is not None:
            flags = inputs.get('structure')
            if flags is None or flags.shape != (len(source_words), 8):
                raise ValueError('Structured semantic encoding requires eight flags per source word')
            source_words = source_words + self.structure(flags)
        question, _ = self.context(words[inputs['question']][None])
        source = source_words[None]
        if self.question_attention is not None and self.question_attention_stage == 'pre_context':
            source = self.condition_source(source, question, before_context=True)
        source, _ = self.context(source)
        if self.question_attention is not None and self.question_attention_stage == 'post_context':
            source = self.condition_source(source, question)
        source, question = source[0], question[0]
        request = self.attention(question).squeeze(-1).softmax(0) @ question
        expanded = request.expand_as(source)
        features = torch.cat((source, expanded, source * expanded, (source - expanded).abs()), -1)
        return self.score(features).squeeze(-1), request, source

    def encode_many(self, inputs_list):
        """Same encoder over independent variable-length inputs, without padding context.

        Packed recurrent sequences keep each backwards state inside its own
        observed question or source. No labels or answer states enter this path.
        """
        sequences=[]; offsets=[]; offset=0
        for inputs in inputs_list:
            offsets.append(offset)
            sequences.extend(inputs['units'][i,:int(n)] for i,n in enumerate(inputs['lengths']))
            offset+=len(inputs['lengths'])
        lengths=torch.tensor([len(s) for s in sequences],device=sequences[0].device)
        states,_=self.word(pad_sequence(sequences,batch_first=True))
        words=states[torch.arange(len(states),device=states.device),lengths-1]
        sources=[];questions=[]
        for inputs,offset in zip(inputs_list,offsets):
            source=words[inputs['source']+offset]
            if self.structure is not None:
                flags=inputs.get('structure')
                if flags is None or flags.shape!=(len(source),8):
                    raise ValueError('Structured semantic encoding requires eight flags per source word')
                source=source+self.structure(flags)
            sources.append(source);questions.append(words[inputs['question']+offset])
        def contextualize(sequences):
            sizes=torch.tensor([len(s) for s in sequences])
            packed=pack_padded_sequence(pad_sequence(sequences,batch_first=True),sizes,batch_first=True,enforce_sorted=False)
            result,_=self.context(packed)
            return pad_packed_sequence(result,batch_first=True)[0],sizes
        question,nq=contextualize(questions)
        mask=torch.arange(question.shape[1],device=question.device)[None]>=nq.to(question.device)[:,None]
        if self.question_attention is not None and self.question_attention_stage == 'pre_context':
            conditioned=self.condition_source(pad_sequence(sources,batch_first=True),question,mask,before_context=True)
            sources=[conditioned[i,:len(s)] for i,s in enumerate(sources)]
        source,ns=contextualize(sources)
        if self.question_attention is not None and self.question_attention_stage == 'post_context':
            source=self.condition_source(source,question,mask)
        weights=self.attention(question).squeeze(-1).masked_fill(mask,float('-inf')).softmax(-1)
        requests=(weights[:,:,None]*question).sum(1)
        result=[]
        for i,n in enumerate(ns):
            states=source[i,:int(n)];request=requests[i];expanded=request.expand_as(states)
            features=torch.cat((states,expanded,states*expanded,(states-expanded).abs()),-1)
            result.append((self.score(features).squeeze(-1),request,states))
        return result

    def score_details(self, inputs, prior, legacy_question, encoded=None):
        semantic, request, source = self.encode(inputs) if encoded is None else encoded
        if prior.shape != semantic.shape:
            raise ValueError('Source fusion requires one legacy score per source occurrence')
        scale = semantic.new_ones(())
        if self.prior_scale is not None:
            scale = torch.nn.functional.softplus(self.prior_scale(request)).squeeze(-1) / math.log(2.)
        fused = scale * prior + semantic
        corrected = fused
        if self.fusion_gate is not None:
            temperature = torch.nn.functional.softplus(self.temperatures).clamp_min(1e-4)
            left_scores,right_scores = (corrected,prior) if self.fusion_experts=='corrected_legacy_v1' else (prior,semantic)
            left, right = (left_scores / temperature[0]).log_softmax(0), (right_scores / temperature[1]).log_softmax(0)
            def properties(logp):
                p = logp.exp()
                entropy = -(p * logp).sum() / math.log(max(len(p), 2))
                top = p.topk(min(2, len(p))).values
                margin = top[0] - (top[1] if len(top) > 1 else 0.)
                return entropy, margin
            hl, ml = properties(left); hr, mr = properties(right)
            props = torch.stack((hl, hr, ml, mr, (left.argmax() == right.argmax()).to(left.dtype)))
            gate = self.fusion_gate(torch.cat((legacy_question, request, props))).squeeze()
            fused = torch.logaddexp(torch.nn.functional.logsigmoid(-gate) + left,
                                    torch.nn.functional.logsigmoid(gate) + right)
        return dict(legacy=prior, semantic=semantic, corrected=corrected, fused=fused, request=request, source=source, prior_scale=scale)

    def forward(self, inputs, prior=None, return_features=False, legacy_question=None):
        """Backward-compatible residual interface; production uses score_details."""
        if self.fusion_gate is not None:
            if prior is None or legacy_question is None:
                raise ValueError('Normalized fusion requires legacy scores and the legacy question')
            details = self.score_details(inputs, prior, legacy_question)
            result = details['fused'] - prior
            return (result, details['request'], details['source']) if return_features else result
        residual, request, source = self.encode(inputs)
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
