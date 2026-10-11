# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input-only, compositional complete answer spans with learned start/extent.

Occurrence addresses are request-local indexes, never embedding features. Gold
answers are used by the external trainer to label candidates, not by propose().
"""
from dataclasses import dataclass
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AnswerSpanPlan:
    tokens: tuple
    start: int
    endpoint: int
    confidence: float
    progress: int = 0
    valid: bool = True
    lexical_start: int = -1
    answer_kind: str = ''
    kind_confidence: float = 0.

    @property
    def next_token(self):
        return self.tokens[self.progress] if self.valid and self.progress < len(self.tokens) else None

    def observe(self, token):
        if self.next_token != token:
            self.valid = False
        else:
            self.progress += 1


class CompleteAnswerSpanPredictor(nn.Module):
    """Small shared request/start/endpoint heads; no observed-profile vocabulary."""
    def __init__(self, width, kinds=9, max_words=16,feature_version='surface_v1',task_router=False,word_encoder='unit_mean_v1',kind_labels=None,kind_conditioned_endpoints=False):
        super().__init__()
        self.max_words=max_words
        self.feature_version=feature_version
        self.word_encoder=word_encoder
        self.kind_conditioned_endpoints=kind_conditioned_endpoints
        self.kind_labels=tuple(kind_labels or ['AMOUNT','BILL_CODE','DATE','INTERVAL','LEGAL_CODE','MEASUREMENT','PERSON','QUALIFIED_COUNT','YEAR'])
        kinds=len(self.kind_labels)
        self.question_attention=nn.Linear(width,1,bias=False)
        self.request=nn.Linear(width,width)
        self.start_projection=nn.Linear(5*width,width)
        self.end_projection=nn.Linear(5*width,width)
        self.start_score=nn.Sequential(nn.Linear(3*width,64),nn.Tanh(),nn.Linear(64,1))
        self.end_score=nn.Sequential(nn.Linear(3*width,64),nn.Tanh(),nn.Linear(64,1))
        self.answer_kind=nn.Linear(width,kinds)
        self.applicability=nn.Linear(width,1)
        if task_router:
            from signature_bank import WordTaskRouter
            self.task_router=WordTaskRouter(width,-20.)
        self.register_buffer('ready',torch.tensor(False))
        self._payload_cache={}
        self._payload_contract=None
        nn.init.zeros_(self.question_attention.weight)
        if kind_conditioned_endpoints:
            # Neutral migration; reusable kind properties come from the shared
            # bank, not one learned row per observed answer or source profile.
            self.kind_extent_projection=nn.Parameter(torch.zeros(width,width))

    @staticmethod
    def payload(tokenizer,text):
        excluded={'<BOS>','<EOS>','<BOI>','<EOI>','<BOO>','<EOO>','<LINE>','<EOL>','<BOP>','<EOP>','<BLO>','<PAD>'}
        tokens=tuple(i for i in tokenizer.encode(text) if tokenizer.construction_units[i].text not in excluded and tokenizer.construction_units[i].kind!='signature')
        return tokens+(tokenizer.special_tokens['<EOL>'],tokenizer.special_tokens['<EOO>'],tokenizer.eos_id)

    @staticmethod
    def properties(bank,keys):
        def address(value):
            digest=hashlib.blake2b(value.encode(),digest_size=8).digest()
            return 1+(int.from_bytes(digest,'little')&((1<<63)-1))%bank.capacity
        ids=torch.tensor([[address(salt+'answer_span:'+key) for salt in ('a:','b:')] for key in keys],device=bank.embedding.weight.device)
        return bank.embedding(ids).mean((0,1))

    def packet(self,state,tokenizer,bank,lexical_embeddings):
        """Bounded candidates from input surface only, including unseen words."""
        if not state.source_surface or not state.source_positions or state.words is None:return None
        device=bank.embedding.weight.device;d=bank.embedding.embedding_dim
        contract=(id(tokenizer),tokenizer.vocab_size)
        if contract!=self._payload_contract:
            self._payload_cache={};self._payload_contract=contract
        properties={}
        def props(keys):
            key=tuple(keys)
            if key not in properties:properties[key]=self.properties(bank,keys)
            return properties[key]
        def payload(text):
            if text not in self._payload_cache:
                if len(self._payload_cache)>=32768:self._payload_cache.pop(next(iter(self._payload_cache)))
                self._payload_cache[text]=self.payload(tokenizer,text)
            return self._payload_cache[text]
        raw=dict(state.source_surface);eligible=state.words.source_candidate_mask(state.entries)
        groups={}
        for i,(entry,position,allowed) in enumerate(zip(state.entries,state.source_positions,eligible)):
            if allowed and position in raw:groups.setdefault(entry[3],[]).append((i,position,entry[0]))
        complete_words={serial:word for serial,_,word in state.words.records}
        # Eviction may cut the first retained word. A fragment is not a start.
        groups=[group for serial,group in groups.items()
                if tuple(item[2] for item in group)==complete_words.get(serial)]
        if not groups:return None
        def word(ids):
            if self.word_encoder=='shared_word_v2':
                from signature_bank import compose_word_identity
                return compose_word_identity(ids,tokenizer,bank,lexical_embeddings)
            ids=torch.tensor(ids,device=device)
            return F.layer_norm((lexical_embeddings(ids)+bank.encode('token',ids)).mean(0)/math.sqrt(2.),(d,))
        vectors=torch.stack([word([item[2] for item in group]) for group in groups])
        query=state.words.question()
        if not query:return None
        question=torch.stack([word(ids) for ids in query])
        zero=vectors.new_zeros(d)
        def neighbor(i):return vectors[i] if 0<=i<len(groups) else zero
        def unit(pos):return tokenizer.construction_units[raw[pos]] if pos in raw else None
        def boundary(u):return u is None or u.text in {'<SPACE>','<TAB>','<EOL>','<LINE>','<BOI>','<EOI>'} or u.is_lexical
        def category(group):
            units=[tokenizer.construction_units[item[2]] for item in group]
            return 'numeric' if all(u.kind=='digit' for u in units) else ('alpha' if all(u.render.isalpha() for u in units) else 'mixed')
        def numeric_join(pos):
            left,marker,right=unit(pos-1),unit(pos),unit(pos+1)
            return bool(left is not None and right is not None and marker is not None and left.kind=='digit' and right.kind=='digit' and marker.render in {'.',',','-','/'})
        start_features=[];starts=[];ends=[];end_features=[];tokens=[];surfaces=[]
        for i,group in enumerate(groups):
            first=group[0][1];leading=first
            while leading>0 and not boundary(unit(leading-1)):leading-=1
            case_start=first
            while case_start>leading and unit(case_start-1).kind=='case':case_start-=1
            keys=['role=start']+['leading='+unit(p).text for p in range(leading,first)]
            if self.feature_version=='surface_v2':
                keys+=['word_type='+category(group),'numeric_continuation='+str(numeric_join(first-1))]
            start_features.append(torch.cat([neighbor(i-2),neighbor(i-1),neighbor(i),neighbor(i+1),props(keys)]))
            starts.append(group[0][0])
            for j in range(i,min(len(groups),i+self.max_words)):
                last=groups[j][-1][1]
                if any(unit(p) is None or unit(p).text in {'<EOL>','<EOI>'} for p in range(first,last+1)):break
                tail=last+1
                while tail in raw and tail<last+5 and not boundary(unit(tail)) and unit(tail).kind!='case':tail+=1
                for begin in dict.fromkeys((case_start,leading)):
                    for finish in range(last+1,tail+1):
                        original=tokenizer.decode([raw[p] for p in range(begin,finish)],clean_text=False,collapse_structure=False).strip()
                        if not original:continue
                        for case_mode,text in enumerate(dict.fromkeys((original,original[:1].upper()+original[1:].lower()))):
                            keys=['role=endpoint','words='+str(min(j-i+1,8)), 'case_policy='+str(case_mode),
                                  'prefix='+str(begin!=case_start),'tail_count='+str(finish-last-1)]
                            keys+=['trailing='+unit(p).text for p in range(last+1,finish)]
                            # Actual markers and reusable roles, not serial/address arithmetic.
                            keys+=['leading='+unit(p).text for p in range(begin,first)]
                            if self.feature_version=='surface_v2':
                                following=unit(last+1)
                                keys+=['word_type='+category(groups[j]),'next_marker='+(following.text if following is not None else 'unobserved'),
                                       'numeric_continuation='+str(numeric_join(last+1))]
                            end_features.append(torch.cat([neighbor(i),neighbor(j),vectors[i:j+1].mean(0),neighbor(j+1),props(keys)]))
                            ends.append((i,j,begin,finish));tokens.append(payload(text));surfaces.append(text)
        if not ends:return None
        result=dict(question=question,start_features=torch.stack(start_features),end_features=torch.stack(end_features),
                    starts=starts,ends=ends,tokens=tokens,surfaces=surfaces)
        if self.kind_conditioned_endpoints:
            result['kind_features']=torch.stack([props(['role=requested_extent','answer_kind='+kind]) for kind in self.kind_labels])
        return result

    def scores(self,packet):
        q=self.request_vector(packet['question'])
        kind=self.answer_kind(q)
        end_request=q
        if self.kind_conditioned_endpoints:
            extent=kind.softmax(-1)@packet['kind_features']
            end_request=q+F.linear(extent,self.kind_extent_projection)
        def score(features,projection,head,request):
            x=torch.tanh(projection(features));request=request.expand_as(x)
            return head(torch.cat([request,x,request*x],-1)).squeeze(-1)
        start=score(packet['start_features'],self.start_projection,self.start_score,q)
        end=score(packet['end_features'],self.end_projection,self.end_score,end_request)
        owners=torch.tensor([e[0] for e in packet['ends']],device=end.device)
        # Separate distributions make start and endpoint losses interpretable.
        normalizer=torch.stack([torch.logsumexp(end[owners==i],0) for i in range(len(start))])
        joint=start.log_softmax(0)[owners]+end-normalizer[owners]
        applicability=self.task_router(self.task_features(packet['question'])).squeeze(-1) if hasattr(self,'task_router') else self.applicability(q).squeeze(-1)
        return dict(start=start,end=end,joint=joint,owners=owners,kind=kind,applicability=applicability)

    def request_vector(self,question):
        q=(question*self.question_attention(question).softmax(0)).sum(0)
        return torch.tanh(self.request(q))

    def question_features(self,words,bank,lexical_embeddings,tokenizer=None):
        d=bank.embedding.embedding_dim
        result=[]
        for word in words:
            if self.word_encoder=='shared_word_v2':
                if tokenizer is None:raise ValueError('Shared word composition requires the tokenizer')
                from signature_bank import compose_word_identity
                result.append(compose_word_identity(word,tokenizer,bank,lexical_embeddings));continue
            ids=torch.tensor(word,device=bank.embedding.weight.device)
            result.append(F.layer_norm((lexical_embeddings(ids)+bank.encode('token',ids)).mean(0)/math.sqrt(2.),(d,)))
        return torch.stack(result) if result else None

    def task_probability(self,words,bank,lexical_embeddings,tokenizer=None):
        question=self.question_features(words,bank,lexical_embeddings,tokenizer)
        if question is None:return bank.embedding.weight.new_zeros(())
        logits=self.task_router(self.task_features(question)) if hasattr(self,'task_router') else self.applicability(self.request_vector(question))
        return logits.sigmoid().squeeze(-1)

    @staticmethod
    def task_features(question):
        if len(question)>64:question=torch.cat((question[:32],question[-32:]))
        features=question.new_zeros(64,question.size(-1)+1)
        features[:len(question),:-1]=question;features[:len(question),-1]=1.
        return features

    def propose(self,state,tokenizer,bank,lexical_embeddings,threshold=.9):
        if not bool(self.ready):return None
        packet=self.packet(state,tokenizer,bank,lexical_embeddings)
        if packet is None:return None
        scored=self.scores(packet);prob=scored['joint'].exp()
        # Equivalent occurrences/surface variants support the same answer.
        totals={}
        for i,tokens in enumerate(packet['tokens']):totals[tokens]=totals.get(tokens,0.)+prob[i]
        tokens=max(totals,key=lambda key:float(totals[key]))
        confidence=min(1.,max(0.,float(totals[tokens])*float(scored['applicability'].sigmoid())))
        if confidence<threshold:return None
        matching=[i for i,item in enumerate(packet['tokens']) if item==tokens]
        best=max(matching,key=lambda i:float(prob[i]));start,_,begin,endpoint=packet['ends'][best]
        kind_prob=scored['kind'].float().softmax(-1);kind_index=int(kind_prob.argmax())
        return AnswerSpanPlan(tokens,begin,endpoint,confidence,lexical_start=packet['starts'][start],
                              answer_kind=self.kind_labels[kind_index],kind_confidence=float(kind_prob[kind_index]))
