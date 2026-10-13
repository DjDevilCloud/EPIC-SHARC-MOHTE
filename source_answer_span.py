# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input-only, compositional complete answer spans with learned start/extent.

Occurrence addresses are request-local indexes, never embedding features. Gold
answers label candidates in the training loss, never in propose().
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
    source_owned_person: bool = False
    requested_properties: tuple = ()

    @property
    def next_token(self):
        return self.tokens[self.progress] if self.valid and self.progress < len(self.tokens) else None

    def observe(self, token):
        if self.next_token != token:
            self.valid = False
        else:
            self.progress += 1


class OrderedQuestionScope(nn.Module):
    """Applicability from ordered question words, independent of span confidence."""
    def __init__(self,width):
        super().__init__()
        self.encoder=nn.GRU(width,width,batch_first=True)
        self.classifier=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias,-20.)

    def forward(self,features):
        single=features.ndim==2
        if single:features=features[None]
        encoded,_=self.encoder(features[...,:-1])
        lengths=(features[...,-1]>0).sum(-1).clamp_min(1)
        state=encoded[torch.arange(len(features),device=features.device),lengths-1]
        result=self.classifier(state)
        return result[0] if single else result


class CompleteAnswerSpanPredictor(nn.Module):
    """Small shared request/start/endpoint heads; no observed-profile vocabulary."""
    def __init__(self, width, kinds=9, max_words=16,feature_version='surface_v1',task_router=False,word_encoder='unit_mean_v1',kind_labels=None,kind_conditioned_endpoints=False,typed_extent_eligibility=False,kind_threshold=.9,owner_context=False,general_context=False,general_endpoint=False,general_roles=False,general_name_plan=False,general_query_attention=False,general_relation_attention=False,relation_eligibility=False,endpoint_properties=False,endpoint_task_router=False,causal_owners=False,unified_reader=False,unified_reader_version='aligned_v1'):
        super().__init__()
        if unified_reader:
            from unified_span_reader import create_unified_reader
            self.unified_reader=create_unified_reader(width,unified_reader_version)
        self.max_words=max_words
        self.feature_version=feature_version
        self.word_encoder=word_encoder
        self.kind_conditioned_endpoints=kind_conditioned_endpoints
        self.typed_extent_eligibility=typed_extent_eligibility
        self.kind_threshold=kind_threshold
        self.owner_context=owner_context
        self.general_context=general_context
        self.general_roles=general_roles
        self.general_name_plan=general_name_plan
        self.general_relation_attention=general_relation_attention
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
        if owner_context:
            # Shared ordered encoder for requests and bounded causal source
            # prefixes. Zero residual preserves all previous scoring at migration.
            self.owner_encoder=nn.GRU(width,width,batch_first=True)
            self.owner_projection=nn.Parameter(torch.zeros(width,width))
        if general_context:
            if causal_owners:
                self.causal_owner_match=nn.Sequential(nn.Linear(3*width,32),nn.Tanh(),nn.Linear(32,1))
                self.causal_owner_kind=nn.Sequential(nn.Linear(3*width,32),nn.Tanh(),nn.Linear(32,1))
                self.causal_owner_scope=nn.Sequential(nn.Linear(4*width,32),nn.Tanh(),nn.Linear(32,1))
                for module in (self.causal_owner_match,self.causal_owner_kind,self.causal_owner_scope):
                    nn.init.zeros_(module[-1].weight);nn.init.zeros_(module[-1].bias)
                nn.init.constant_(self.causal_owner_scope[-1].bias,-8.)
            if endpoint_properties:
                self.endpoint_properties=nn.Sequential(nn.Linear(3*width,64),nn.Tanh(),nn.Linear(64,1))
                nn.init.zeros_(self.endpoint_properties[-1].weight)
                nn.init.zeros_(self.endpoint_properties[-1].bias)
                if endpoint_task_router:
                    self.endpoint_scope=nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,1))
                    nn.init.zeros_(self.endpoint_scope[-1].weight)
                    nn.init.constant_(self.endpoint_scope[-1].bias,-8.)
            self.general_scope=OrderedQuestionScope(width)
            self.general_encoder=nn.GRU(width,width,batch_first=True)
            if general_relation_attention:
                self.relation_query=nn.Linear(width,width,bias=False)
                self.relation_key=nn.Linear(width,width,bias=False)
                nn.init.eye_(self.relation_query.weight);nn.init.eye_(self.relation_key.weight)
                self.relation_word_attention=nn.Linear(width,1,bias=False)
                nn.init.zeros_(self.relation_word_attention.weight)
                self.relation_distance=nn.Linear(width,17)
                nn.init.zeros_(self.relation_distance.weight);nn.init.zeros_(self.relation_distance.bias)
                self.relation_scope=nn.Linear(width,2)
                nn.init.zeros_(self.relation_scope.weight);nn.init.zeros_(self.relation_scope.bias)
                self.relation_output=nn.Linear(width,width,bias=False)
                nn.init.zeros_(self.relation_output.weight)
                if relation_eligibility:
                    self.relation_eligibility=nn.Sequential(nn.Linear(5*width,64),nn.Tanh(),nn.Linear(64,1))
                    nn.init.zeros_(self.relation_eligibility[-1].weight)
                    nn.init.zeros_(self.relation_eligibility[-1].bias)
            if general_query_attention:
                self.general_query_attention=nn.Linear(width,1,bias=False)
                nn.init.zeros_(self.general_query_attention.weight)
            self.general_start_projection=nn.Parameter(torch.zeros(width,width))
            self.general_end_projection=nn.Parameter(torch.zeros(width,width))
            if general_endpoint:
                self.general_endpoint=nn.Sequential(nn.Linear(7*width,64),nn.Tanh(),nn.Linear(64,1))
                nn.init.zeros_(self.general_endpoint[-1].weight)
                nn.init.zeros_(self.general_endpoint[-1].bias)
            if general_name_plan:
                self.general_name_scope=OrderedQuestionScope(width)
                self.general_name_encoder=nn.GRU(width,width,batch_first=True)
                self.general_name_attention=nn.Linear(width,1,bias=False)
                nn.init.zeros_(self.general_name_attention.weight)
                self.general_name_start=nn.Parameter(torch.zeros(width,width))
                self.general_name_end=nn.Sequential(nn.Linear(3*width,64),nn.Tanh(),nn.Linear(64,1))
                nn.init.zeros_(self.general_name_end[-1].weight);nn.init.zeros_(self.general_name_end[-1].bias)

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

    def packet(self,state,tokenizer,bank,lexical_embeddings,compact=False):
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
        compact_query=None
        if compact:
            fields=self.grounded_fields(state,tokenizer)
            if fields is None:return None
            question_label,labels=fields
            compact_query=[ids for serial,_,ids in state.words.records if serial>question_label]
            eligible=[entry[3]<question_label and labels.get(entry[3])!='context' for entry in state.entries]
        groups={}
        for i,(entry,position,allowed) in enumerate(zip(state.entries,state.source_positions,eligible)):
            if allowed and position in raw:groups.setdefault(entry[3],[]).append((i,position,entry[0]))
        complete_words={serial:word for serial,_,word in state.words.records}
        # Eviction may cut the first retained word. A fragment is not a start.
        groups=[group for serial,group in groups.items()
                if tuple(item[2] for item in group)==complete_words.get(serial)]
        if not groups:return None
        word_cache={}
        def word(ids):
            key=tuple(ids)
            if key in word_cache:return word_cache[key]
            if self.word_encoder=='shared_word_v2':
                from signature_bank import compose_word_identity
                value=compose_word_identity(ids,tokenizer,bank,lexical_embeddings)
            else:
                indices=torch.tensor(ids,device=device)
                value=F.layer_norm((lexical_embeddings(indices)+bank.encode('token',indices)).mean(0)/math.sqrt(2.),(d,))
            word_cache[key]=value
            return value
        vectors=torch.stack([word([item[2] for item in group]) for group in groups])
        if hasattr(self,'causal_owner_match') or hasattr(self,'unified_reader'):
            clauses={serial:clause for serial,clause,_ in state.words.records}
            summaries={};owner_contexts=[];owner_properties=[];owner_available=[];owner_evicted=[]
            earliest=state.words.records[0][0]
            for group in groups:
                serial=state.entries[group[0][0]][3];clause=clauses[serial]
                prefix=tuple((s,ids) for s,ids in state.words.anchors.get(clause,()) if s<serial)
                key=(clause,len(prefix))
                if key not in summaries:
                    if prefix:
                        values=torch.stack([word(ids) for _,ids in prefix])
                        encoded,_=self.general_encoder(values[None])
                        context=F.layer_norm(encoded[0]+values,(d,)).mean(0)
                    else:context=vectors.new_zeros(d)
                    available=bool(prefix) and clause not in state.words.anchor_truncated
                    owner_props=props(['role=causal_owner','prefix_words='+str(len(prefix)),
                                      'prefix_complete='+str(available)])
                    summaries[key]=(context,owner_props,available)
                context,owner_props,available=summaries[key]
                owner_contexts.append(context);owner_properties.append(owner_props);owner_available.append(available)
                owner_evicted.append(bool(prefix) and prefix[-1][0]<earliest and available)
        query=compact_query if compact else state.words.question()
        if not query:return None
        question=torch.stack([word(ids) for ids in query])
        name_enabled=not compact and self.general_name_plan and bool(self.general_scope(self.task_features(question)).sigmoid().squeeze(-1)>=.99) and bool(self.general_name_scope(self.task_features(question)).sigmoid().squeeze(-1)>=.99)
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
        owner_prefixes=[];owner_lengths=[];scope_start=0;scope_starts=[]
        if self.owner_context or self.general_context:
            for i,group in enumerate(groups):
                if i:
                    previous=groups[i-1];left=previous[-1][1];right=group[0][1]
                    previous_text=tokenizer.decode([item[2] for item in previous],clean_text=False,collapse_structure=False)
                    for pos in range(left+1,right):
                        marker=unit(pos)
                        if marker is None or marker.text in {'<EOL>','<EOI>','<LINE>'}:
                            scope_start=i;break
                        if marker.render in {'!','?',';'} or (marker.render=='.' and not numeric_join(pos) and not (len(previous_text)==1 and previous_text.isalpha())):
                            scope_start=i;break
                sequence=vectors[max(scope_start,i-32):i]
                prefix=vectors.new_zeros(32,d);prefix[:len(sequence)]=sequence
                owner_prefixes.append(prefix);owner_lengths.append(len(sequence))
                scope_starts.append(scope_start)
        case_chains=[]
        if self.general_context and self.general_roles:
            from signature_bank import BoundedIdentityReadout
            chain_roles=BoundedIdentityReadout.source_start_chain_roles(state.words,tokenizer)
            serials=[state.entries[group[0][0]][3] for group in groups]
            marked=[]
            for s,group in zip(serials,groups):
                position=group[0][1]-1;case=False
                while unit(position) is not None and unit(position).kind=='case':
                    case=True;position-=1
                marked.append(case or 'initial=True' in chain_roles.get(s,()))
            def continues(i,j):
                if not (marked[i] and marked[j]) or scope_starts[i]!=scope_starts[j]:return False
                initial='initial=True' in chain_roles.get(serials[i],())
                gap=[unit(p) for p in range(groups[i][-1][1]+1,groups[j][0][1])]
                return bool(gap) and all(u is not None and (u.kind=='case' or u.text in {'<SPACE>','<TAB>'} or (initial and u.render=='.')) for u in gap)
            case_chains=[(marked[i],i>0 and continues(i-1,i),i+1<len(groups) and continues(i,i+1)) for i in range(len(groups))]
            def case_keys(i):
                return ['case_marked='+str(case_chains[i][0]),'case_chain_left='+str(case_chains[i][1]),'case_chain_right='+str(case_chains[i][2])]
        start_features=[];starts=[];ends=[];end_features=[];tokens=[];surfaces=[]
        general_endpoint_roles=[]
        name_valid=[];name_properties=[];name_contexts=[];name_lengths=[]
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
                        for case_mode,text in enumerate(dict.fromkeys((original,original[:1].upper()+original[1:].lower(),original[:1].upper()+original[1:]))):
                            keys=['role=endpoint','words='+str(min(j-i+1,8)), 'case_policy='+str(case_mode),
                                  'prefix='+str(begin!=case_start),'tail_count='+str(finish-last-1)]
                            keys+=['trailing='+unit(p).text for p in range(last+1,finish)]
                            # Actual markers and reusable roles, not serial/address arithmetic.
                            keys+=['leading='+unit(p).text for p in range(begin,first)]
                            if self.feature_version=='surface_v2':
                                following=unit(last+1)
                                keys+=['word_type='+category(groups[j]),'next_marker='+(following.text if following is not None else 'unobserved'),
                                       'numeric_continuation='+str(numeric_join(last+1))]
                            end_features.append(props(keys) if compact else torch.cat([neighbor(i),neighbor(j),vectors[i:j+1].mean(0),neighbor(j+1),props(keys)]))
                            if case_chains:
                                general_endpoint_roles.append(props(keys+['start_'+key for key in case_keys(i)]+['end_'+key for key in case_keys(j)]))
                            if name_enabled:
                                initial='initial=True' in chain_roles.get(serials[j],())
                                expected_finish=last+2 if initial and unit(last+1) is not None and unit(last+1).render=='.' else last+1
                                name_valid.append(not case_chains[i][1] and not case_chains[j][2] and scope_starts[i]==scope_starts[j]
                                                  and begin==case_start and finish==expected_finish
                                                  and not (unit(last+1) is not None and unit(last+1).render==':')
                                                  and not any(unit(p) is not None and unit(p).render==':' for p in range(first,last+1)))
                                name_properties.append(props(keys+['answer_kind=PERSON']+['start_'+key for key in case_keys(i)]+['end_'+key for key in case_keys(j)]))
                                scope_end=next((k for k in range(j+1,len(groups)) if scope_starts[k]!=scope_starts[j]),len(groups))
                                before_words=vectors[max(scope_starts[i],i-8):i];after_words=vectors[j+1:min(scope_end,j+9)]
                                marker=props(['role=person_value','name_words='+str(min(j-i+1,8)),
                                              'before_words='+str(min(i-scope_starts[i],8)),
                                              'after_words='+str(min(scope_end-j-1,8))]+case_keys(i))
                                # Each hypothesis masks its entire proposed
                                # value before scoring the surrounding relation.
                                sequence=torch.cat((before_words,marker[None],after_words))
                                window=vectors.new_zeros(17,d);window[:len(sequence)]=sequence
                                name_contexts.append(window);name_lengths.append(len(sequence))
                            ends.append((i,j,begin,finish));
                            if not compact:tokens.append(payload(text))
                            surfaces.append(text)
        if not ends:return None
        result=dict(question=question,start_features=torch.stack(start_features),end_features=torch.stack(end_features),
                    starts=starts,ends=ends,tokens=tokens,surfaces=surfaces)
        if hasattr(self,'causal_owner_match') or hasattr(self,'unified_reader'):
            result['causal_owner_contexts']=torch.stack(owner_contexts)
            result['causal_owner_properties']=torch.stack(owner_properties)
            result['causal_owner_available']=torch.tensor(owner_available,device=device,dtype=torch.bool)
            result['causal_owner_evicted']=torch.tensor(owner_evicted,device=device,dtype=torch.bool)
        if self.general_relation_attention:
            result['relation_source']=vectors
            result['relation_scopes']=torch.tensor(scope_starts,device=device)
        if self.owner_context and not compact:
            result['owner_prefixes']=torch.stack(owner_prefixes)
            result['owner_lengths']=torch.tensor(owner_lengths,device=device,dtype=torch.long)
        if self.general_context and not compact:
            observed=[];lengths=[];marker=props(['role=source_occurrence'])
            scope_ends=[len(groups)]*len(groups)
            for i in range(len(groups)-2,-1,-1):
                scope_ends[i]=i+1 if scope_starts[i+1]!=scope_starts[i] else scope_ends[i+1]
            for i in range(len(groups)):
                finish=scope_ends[i]
                occurrence=props(['role=source_occurrence']+case_keys(i)) if case_chains else marker
                sequence=torch.cat((vectors[max(scope_starts[i],i-16):i],occurrence[None],vectors[i:min(finish,i+17)]))
                window=vectors.new_zeros(34,d);window[:len(sequence)]=sequence
                observed.append(window);lengths.append(len(sequence))
            result['general_contexts']=torch.stack(observed)
            result['general_lengths']=torch.tensor(lengths,device=device,dtype=torch.long)
            if case_chains:
                result['general_case_chains']=case_chains
                result['general_endpoint_roles']=torch.stack(general_endpoint_roles)
            if name_enabled:
                result['name_contexts']=torch.stack(name_contexts)
                result['name_lengths']=torch.tensor(name_lengths,device=device,dtype=torch.long)
                result['name_eligibility']=torch.tensor(name_valid,device=device,dtype=torch.bool)
                result['name_properties']=torch.stack(name_properties)
        if compact:
            result.pop('tokens')
            result['source_words']=vectors
            result['reader_endpoint_properties']=torch.stack(general_endpoint_roles) if case_chains else result['end_features'][:,-d:]
            result['source_word_ids']=[tuple(item[2] for item in group) for group in groups]
            result['reader_scope_start']=torch.tensor(scope_starts,device=device)
            scope_ends=[len(groups)]*len(groups)
            for i in range(len(groups)-2,-1,-1):scope_ends[i]=i+1 if scope_starts[i+1]!=scope_starts[i] else scope_ends[i+1]
            result['reader_scope_end']=torch.tensor(scope_ends,device=device)
            result['question_word_ids']=[tuple(ids) for ids in query]
            if hasattr(self,'unified_reader') and self.unified_reader.version in {'structural_owner_v9','relation_decisions_v10'}:
                from unified_span_reader import query_word_properties
                query_records=[(serial,ids) for serial,_,ids in state.words.records if serial>question_label]
                last_positions={entry[3]:position for entry,position in zip(state.entries,state.source_positions)}
                result['reader_query_properties']=query_word_properties(query_records,raw,last_positions,tokenizer).to(device)
            if hasattr(self,'unified_reader') and self.unified_reader.version in {'requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:
                from unified_span_reader import answer_span_facts
                facts=[];word_text={};fact_cache={}
                for text,(i,j,begin,finish) in zip(surfaces,ends):
                    closed=j+1==len(groups) or scope_starts[j+1]!=scope_starts[j]
                    if not closed and j+1 not in word_text:
                        first_next=groups[j+1][0][1];begin_next=first_next
                        while begin_next>0 and unit(begin_next-1) is not None and unit(begin_next-1).kind=='case':begin_next-=1
                        word_text[j+1]=tokenizer.decode([raw[pos] for pos in range(begin_next,groups[j+1][-1][1]+1)],clean_text=False,collapse_structure=False)
                    next_word=word_text.get(j+1,'') if not closed else ''
                    capital=bool(next_word) and next_word[0].isupper()
                    continuation=numeric_join(finish)
                    key=(text,j-i+1,closed,capital,continuation)
                    if key not in fact_cache:fact_cache[key]=answer_span_facts(text,j-i+1,scope_closed=closed,capital_continuation=capital,numeric_continuation=continuation)
                    facts.append(fact_cache[key])
                result['reader_answer_facts']=torch.tensor(facts,device=device,dtype=torch.float32)
            return result
        if self.kind_conditioned_endpoints:
            result['kind_features']=torch.stack([props(['role=requested_extent','answer_kind='+kind]) for kind in self.kind_labels])
        if self.typed_extent_eligibility:
            # Narrow structural contracts, not source identity or answer labels.
            # YEAR denotes a four-digit calendar year in this schema. INTEGER_PART
            # denotes one signed digit word, not its observed fractional suffix.
            years=[];integers=[]
            for text,(i,j,begin,finish) in zip(surfaces,ends):
                years.append(text.isdecimal() and len(text)==4 and i==j)
                first=groups[i][0][1];prefix=unit(first-1)
                fragment=numeric_join(first-1)
                signed=prefix is not None and prefix.render in {'+','-'} and not fragment
                integers.append(text.lstrip('+-').isdecimal() and i==j and not fragment and (not signed or begin<first))
            result['extent_eligibility']={
                'YEAR':torch.tensor(years,device=device,dtype=torch.bool),
                'INTEGER_PART':torch.tensor(integers,device=device,dtype=torch.bool)}
        return result

    def scores(self,packet):
        q=self.request_vector(packet['question'])
        kind=self.answer_kind(q)
        general_probability=kind.new_zeros(())
        general_active=False;general_query=general_states=None;name_active=False;name_probability=kind.new_zeros(())
        if self.general_context:
            general_probability=self.general_scope(self.task_features(packet['question'])).sigmoid().squeeze(-1)
            general_active=bool(general_probability>=.99)
            text_index=self.kind_labels.index('TEXT_SPAN')
            old_kind=kind.clone();old_kind[text_index]=-torch.inf
            if general_active:
                kind=old_kind.log_softmax(-1)+(1.-general_probability).clamp_min(1e-8).log()
                kind[text_index]=general_probability.clamp_min(1e-8).log()
                if self.general_name_plan:
                    name_probability=self.general_name_scope(self.task_features(packet['question'])).sigmoid().squeeze(-1)
                    name_active=bool(name_probability>=.99)
                    if name_active:
                        person_index=self.kind_labels.index('PERSON')
                        kind[person_index]=torch.logaddexp(kind[person_index],(general_probability*name_probability).clamp_min(1e-8).log())
                        kind[text_index]=(general_probability*(1.-name_probability)).clamp_min(1e-8).log()
                encoded,_=self.general_encoder(packet['general_contexts'])
                rows=torch.arange(len(packet['general_lengths']),device=q.device)
                general_states=encoded[rows,packet['general_lengths']-1]
                general_query=self.general_request(packet['question'])
                if self.general_relation_attention and not name_active:
                    general_states=general_states+self.relation_context(packet,general_query)
            else:kind=old_kind
        end_request=q
        if self.kind_conditioned_endpoints:
            extent=kind.softmax(-1)@packet['kind_features']
            end_request=q+F.linear(extent,self.kind_extent_projection)
        owners=torch.tensor([e[0] for e in packet['ends']],device=q.device)
        if general_active:
            end_request=end_request[None]+F.linear(general_states[owners]*general_query,self.general_end_projection)
        def score(features,projection,head,request):
            x=torch.tanh(projection(features));request=request.expand_as(x)
            return head(torch.cat([request,x,request*x],-1)).squeeze(-1)
        start=score(packet['start_features'],self.start_projection,self.start_score,q)
        if general_active:
            start=start+(F.linear(general_states,self.general_start_projection)*general_query).sum(-1)/math.sqrt(q.numel())
            if hasattr(self,'relation_eligibility') and not name_active:
                start=start+self.relation_eligibility_scores(packet,general_query)
        end=score(packet['end_features'],self.end_projection,self.end_score,end_request)
        if general_active and hasattr(self,'general_endpoint'):
            source_features=packet['end_features']
            if self.general_roles:
                source_features=torch.cat((source_features[:,:4*q.numel()],packet['general_endpoint_roles']),-1)
            endpoint_features=torch.cat((source_features,general_states[owners]*general_query,general_query.expand(len(owners),-1)),-1)
            end=end+self.general_endpoint(endpoint_features).squeeze(-1)
        if general_active and not name_active and hasattr(self,'endpoint_properties'):
            if not hasattr(self,'endpoint_scope') or bool(self.endpoint_scope(general_query).sigmoid().squeeze(-1)>=.99):
                end=end+self.endpoint_property_scores(packet,general_query)
        if self.owner_context:
            type_probs=kind.softmax(-1);type_index=int(type_probs.argmax())
            # This adapter is fitted for numeric/date ownership. Person-name
            # selection remains in its independently verified source channel.
            if self.kind_labels[type_index] in {'YEAR','INTEGER_PART','NUMBER','AMOUNT','MEASUREMENT','DATE','QUALIFIED_COUNT'} and bool(type_probs[type_index]>=self.kind_threshold):
                encoded,_=self.owner_encoder(packet['owner_prefixes'])
                lengths=packet['owner_lengths'];rows=torch.arange(len(lengths),device=start.device)
                context=encoded[rows,lengths.clamp_min(1)-1]*(lengths>0)[:,None]
                _,request_state=self.owner_encoder(packet['question'][-64:][None])
                owner_request=request_state[0,0]
                start=start+(F.linear(context,self.owner_projection)*owner_request).sum(-1)/math.sqrt(q.numel())
        causal_supported=None;causal_confidence=None
        if general_active and not name_active and hasattr(self,'causal_owner_match') and bool(self.endpoint_scope(general_query).sigmoid().squeeze(-1)>=.99):
            causal=self.causal_owner_scores(packet,general_query)
            supported=packet['causal_owner_available']&(causal['scope'].sigmoid()>=.99)&(causal['kind'].sigmoid()>=.99)
            if bool(supported.any()):
                proposed_start=(causal['match']+causal['kind']).masked_fill(~supported,-torch.inf)
                selected_confidence=causal['match'][proposed_start.argmax()].sigmoid()
                if bool(selected_confidence>=.99):
                    causal_confidence=selected_confidence
                    start=proposed_start;causal_supported=supported
                    # A verified structural plan must not inherit value/position
                    # bias from legacy endpoint features after source eviction.
                    end=self.endpoint_property_scores(packet,general_query)
        extent_confidence=general_probability if general_active else kind.new_ones(())
        if causal_confidence is not None:extent_confidence=extent_confidence*causal_confidence
        eligible_starts=torch.ones_like(start,dtype=torch.bool) if causal_supported is None else causal_supported
        if name_active:
            encoded,_=self.general_name_encoder(packet['name_contexts']);rows=torch.arange(len(packet['name_lengths']),device=q.device)
            contexts=encoded[rows,packet['name_lengths']-1]
            query_states,_=self.general_name_encoder(packet['question'][None])
            query_words=F.layer_norm(query_states[0]+packet['question'],(q.numel(),))
            request=(query_words*self.general_name_attention(query_words).softmax(0)).sum(0)
            span_relation=(F.linear(contexts,self.general_name_start)*request).sum(-1)/math.sqrt(q.numel())
            properties=packet['name_properties'];requests=request.expand_as(properties)
            end=span_relation+self.general_name_end(torch.cat((properties,requests,properties*requests),-1)).squeeze(-1)
            valid=packet['name_eligibility'];eligible_starts=torch.tensor([bool(valid[owners==i].any()) for i in range(len(start))],device=q.device)
            end=end.masked_fill(~valid,-torch.inf)
            span_posterior=end.log_softmax(0) if bool(valid.any()) else torch.full_like(end,-torch.inf)
            # Empty owner groups have no differentiable logsumexp: its all-inf
            # backward would yield NaNs even when the row is later masked.
            start=torch.stack([torch.logsumexp(span_posterior[owners==i],0) if bool(eligible_starts[i])
                               else end.new_full((),-torch.inf) for i in range(len(start))])
            start=start.masked_fill(~eligible_starts,-torch.inf);end=end.masked_fill(~valid,-torch.inf)
            extent_confidence=general_probability*name_probability
        elif self.typed_extent_eligibility:
            probabilities=kind.softmax(-1);index=int(probabilities.argmax())
            valid=packet['extent_eligibility'].get(self.kind_labels[index])
            if valid is not None and bool(probabilities[index]>=self.kind_threshold):
                extent_confidence=probabilities[index]
                eligible_starts=torch.tensor([bool(valid[owners==i].any()) for i in range(len(start))],device=start.device)
                start=start.masked_fill(~eligible_starts,-torch.inf)
                end=end.masked_fill(~valid,-torch.inf)
        # Separate distributions make start and endpoint losses interpretable.
        normalizer=torch.stack([torch.logsumexp(end[owners==i],0) if bool(eligible_starts[i]) else end.new_zeros(()) for i in range(len(start))])
        joint=start.log_softmax(0)[owners]+end-normalizer[owners] if bool(eligible_starts.any()) else torch.full_like(end,-torch.inf)
        applicability=self.task_router(self.task_features(packet['question'])).squeeze(-1) if hasattr(self,'task_router') else self.applicability(q).squeeze(-1)
        if general_active:applicability=torch.maximum(applicability,torch.logit(general_probability.clamp(1e-8,1.-1e-8)))
        return dict(start=start,end=end,joint=joint,owners=owners,kind=kind,applicability=applicability,extent_confidence=extent_confidence,extent_supported=bool(eligible_starts.any()),source_owned_person=name_active,causal_owner_verified=causal_confidence is not None)

    def causal_owner_scores(self,packet,request):
        context=packet['causal_owner_contexts'];query=request.expand_as(context)
        relation=torch.cat((query,context,query*context),-1)
        properties=packet['start_features'][:,-request.numel():]
        return dict(match=self.causal_owner_match(relation).squeeze(-1),
                    scope=self.causal_owner_scope(torch.cat((relation,packet['causal_owner_properties']),-1)).squeeze(-1),
                    kind=self.causal_owner_kind(torch.cat((query,properties,query*properties),-1)).squeeze(-1))

    def endpoint_property_scores(self,packet,request):
        properties=packet.get('general_endpoint_roles',packet['end_features'][:,-request.numel():])
        query=request.expand_as(properties)
        return self.endpoint_properties(torch.cat((query,properties,query*properties),-1)).squeeze(-1)

    def relation_eligibility_scores(self,packet,request):
        width=request.numel();features=packet['start_features']
        word=features[:,2*width:3*width];properties=features[:,4*width:5*width]
        query=request.expand_as(word)
        return self.relation_eligibility(torch.cat((query,word,properties,query*word,query*properties),-1)).squeeze(-1)

    def relation_weights(self,packet,request):
        """Query anchors over the entire bounded observed source, per occurrence.

        Relative offsets count actual source words; occurrence IDs are not
        linguistic features. No output tokens or answer labels enter here.
        """
        source=packet['relation_source'];width=source.size(-1)
        question=packet['question'][-64:]
        query=F.normalize(self.relation_query(question),dim=-1)
        keys=F.normalize(self.relation_key(source),dim=-1)
        word_weights=self.relation_word_attention(question).squeeze(-1).log_softmax(0)
        anchors=(4.*(keys@query.T)+word_weights[None]).logsumexp(-1)
        positions=torch.arange(len(source),device=source.device)
        offsets=positions[None,:]-positions[:,None]
        buckets=offsets.abs().clamp_min(1).float().log2().floor().long().clamp_max(7)
        roles=torch.where(offsets==0,0,torch.where(offsets<0,1+buckets,9+buckets))
        logits=anchors[None]+self.relation_distance(request)[roles]
        scopes=packet.get('relation_scopes')
        if scopes is not None:
            same_scope=(scopes[:,None]==scopes[None,:]).long()
            logits=logits+self.relation_scope(request)[same_scope]
        return logits.softmax(-1)

    def relation_context(self,packet,request):
        source=packet['relation_source'];width=source.size(-1)
        states,_=self.general_encoder(source[None])
        values=F.layer_norm(states[0]+source,(width,))
        return self.relation_output(self.relation_weights(packet,request)@values)

    def general_request(self,question):
        """Ordered words retain direct lexical support when attention is enabled."""
        question=question[-64:]
        states,last=self.general_encoder(question[None])
        if not hasattr(self,'general_query_attention'):return last[0,0]
        words=F.layer_norm(states[0]+question,(question.size(-1),))
        return (words*self.general_query_attention(words).softmax(0)).sum(0)

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
        probability=logits.sigmoid().squeeze(-1)
        if self.general_context:probability=torch.maximum(probability,self.general_scope(self.task_features(question)).sigmoid().squeeze(-1))
        return probability

    def general_task_probability(self,words,bank,lexical_embeddings,tokenizer=None):
        if not self.general_context:return bank.embedding.weight.new_zeros(())
        question=self.question_features(words,bank,lexical_embeddings,tokenizer)
        if question is None:return bank.embedding.weight.new_zeros(())
        return self.general_scope(self.task_features(question)).sigmoid().squeeze(-1)

    @staticmethod
    def task_features(question):
        if len(question)>64:question=torch.cat((question[:32],question[-32:]))
        features=question.new_zeros(64,question.size(-1)+1)
        features[:len(question),:-1]=question;features[:len(question),-1]=1.
        return features

    def supervised_qa_loss(self,input_ids,tokenizer,bank,lexical_embeddings,state_factory):
        """Differentiate source selection/extent using full training-only answers.

        Live proposals never use these targets. Incomplete QA frames are counted
        so clipped windows cannot masquerade as valid grounded training.
        """
        from unified_span_reader import collate_reader_packets,reader_packet_from_span,semantic_components
        zero=bank.embedding.weight.new_zeros(());items=[];records=missing=uncovered=0
        boi=tokenizer.special_tokens['<BOI>'];eoi=tokenizer.special_tokens['<EOI>'];boo=tokenizer.special_tokens['<BOO>'];eoo=tokenizer.special_tokens['<EOO>']
        for row in input_ids.detach().cpu().tolist():
            if boo not in row:continue
            output_begin=row.index(boo)
            if boi not in row[:output_begin] or eoi not in row[:output_begin]:missing+=1;continue
            if eoo not in row[output_begin+1:]:missing+=1;continue
            output_end=row.index(eoo,output_begin+1)
            state=state_factory(row[:output_begin],tokenizer)
            if self.grounded_fields(state,tokenizer) is None:continue
            records+=1
            p=self.packet(state,tokenizer,bank,lexical_embeddings,compact=True)
            if p is None:uncovered+=1;continue
            answer=tokenizer.decode(row[output_begin+1:output_end],clean_text=False,collapse_structure=False).strip()
            features=reader_packet_from_span(p,self.unified_reader.version)
            gold=torch.tensor([text==answer for text in p['surfaces']],device=bank.embedding.weight.device)
            if self.unified_reader.version in {'scoped_relations_v3','semantic_scoped_v4','semantic_context_v5','requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:
                gold=gold&(features['scope_start'][features['owners']]==features['scope_start'][features['finishes']])
            if not bool(gold.any()):uncovered+=1;continue
            features['gold']=gold
            starts=torch.zeros(len(features['source']),device=gold.device,dtype=torch.bool);starts[features['owners'][gold]]=True;features['gold_start']=starts
            if hasattr(self.unified_reader,'semantic_words'):
                features['source_components']=[semantic_components(ids,tokenizer) for ids in p['source_word_ids']]
                features['question_components']=[semantic_components(ids,tokenizer) for ids in p['question_word_ids']]
            items.append(features)
        stats={key:zero.new_tensor(value) for key,value in (('grounded_qa_records',records),('grounded_qa_missing_frames',missing),('grounded_qa_uncovered',uncovered))}
        if not items:
            stats.update(grounded_qa_start_loss=zero,grounded_qa_endpoint_loss=zero,grounded_qa_span_loss=zero)
            return zero,stats
        batch=collate_reader_packets(items,bank.embedding.weight.device);scored=self.unified_reader(batch)
        joint=scored['joint'].masked_fill(~batch['gold'],-torch.inf).logsumexp(-1)
        start=scored['start'].log_softmax(-1).masked_fill(~batch['gold_start'],-torch.inf).logsumexp(-1)
        start_loss=-start.mean();end_loss=(start-joint).mean();loss=start_loss+end_loss
        if 'requested_role_logits' in scored:
            from unified_span_reader import ANSWER_ROLE_FACTS
            weights=batch['gold'].float();target=(batch['answer_facts']*weights[:,:,None]).sum(1)/weights.sum(1)[:,None]
            split=len(ANSWER_ROLE_FACTS)
            role_loss=F.binary_cross_entropy_with_logits(scored['requested_role_logits'][:,:split],target[:,:split])
            extent_loss=F.binary_cross_entropy_with_logits(scored['requested_role_logits'][:,split:],target[:,split:])
            loss=loss+role_loss+extent_loss
            stats.update(grounded_qa_requested_role_loss=role_loss.detach(),grounded_qa_requested_extent_loss=extent_loss.detach())
        stats.update(grounded_qa_start_loss=start_loss.detach(),grounded_qa_endpoint_loss=end_loss.detach(),grounded_qa_span_loss=(start_loss+end_loss).detach(),grounded_qa_objective_loss=loss.detach())
        return loss,stats

    @staticmethod
    def grounded_fields(state,tokenizer):
        if state.words is None or not state.source_surface:return None
        raw=dict(state.source_surface)
        last_positions={entry[3]:position for entry,position in zip(state.entries,state.source_positions)}
        labels={}
        for serial,_,ids in state.words.records:
            after=raw.get(last_positions.get(serial,-2)+1)
            if after is not None and tokenizer.construction_units[after].render==':':
                text=tokenizer.decode(list(ids),clean_text=False,collapse_structure=False).casefold()
                if text in {'question','context'}:labels[serial]=text
        questions=[serial for serial,text in labels.items() if text=='question']
        if not questions:return None
        label=max(questions)
        if any(text=='context' and serial>label for serial,text in labels.items()):return None
        if not any(serial<label and labels.get(serial)!='context' for serial,_,_ in state.words.records):return None
        return label,labels

    def unified_task_applicable(self,state,tokenizer):
        return hasattr(self,'unified_reader') and bool(self.unified_reader.ready) and self.grounded_fields(state,tokenizer) is not None

    def unified_proposal(self,state,tokenizer,bank,lexical_embeddings,threshold):
        from unified_span_reader import collate_reader_packets,reader_packet_from_span
        p=self.packet(state,tokenizer,bank,lexical_embeddings,compact=True)
        if p is None:return None
        features=reader_packet_from_span(p,self.unified_reader.version)
        if hasattr(self.unified_reader,'semantic_words'):
            from unified_span_reader import semantic_components
            features['source_components']=[semantic_components(ids,tokenizer) for ids in p['source_word_ids']]
            features['question_components']=[semantic_components(ids,tokenizer) for ids in p['question_word_ids']]
        scored=self.unified_reader(collate_reader_packets([features],bank.embedding.weight.device))
        probabilities=scored['joint'][0].exp();totals={}
        for i,text in enumerate(p['surfaces']):totals[text]=totals.get(text,0.)+probabilities[i]
        text=max(totals,key=lambda key:float(totals[key]));confidence=float(totals[text])
        if confidence<threshold:return None
        matches=[i for i,key in enumerate(p['surfaces']) if key==text];best=max(matches,key=lambda i:float(probabilities[i]))
        owner,_,begin,endpoint=p['ends'][best]
        requested=tuple(scored['requested_role_logits'][0].sigmoid().detach().cpu().tolist()) if 'requested_role_logits' in scored else ()
        return AnswerSpanPlan(self.payload(tokenizer,text),begin,endpoint,confidence,lexical_start=p['starts'][owner],answer_kind='TEXT_SPAN',kind_confidence=1.,requested_properties=requested)

    def propose(self,state,tokenizer,bank,lexical_embeddings,threshold=.9):
        if not bool(self.ready):return None
        packet=scored=None
        if self.unified_task_applicable(state,tokenizer):
            # Preserve the separately verified complete-person planner during
            # migration; the generic reader has no per-question-family router.
            question=self.question_features(state.words.question(),bank,lexical_embeddings,tokenizer)
            protected=self.general_name_plan and question is not None and bool(self.general_scope(self.task_features(question)).sigmoid().squeeze(-1)>=.99) and bool(self.general_name_scope(self.task_features(question)).sigmoid().squeeze(-1)>=.99)
            if not protected and hasattr(self,'causal_owner_match') and question is not None:
                # Learned task, owner, kind and endpoint checks jointly protect
                # an already verified occurrence from generic-reader interference.
                if bool(self.general_scope(self.task_features(question)).sigmoid().squeeze(-1)>=.99):
                    request=self.general_request(question)
                    if bool(self.endpoint_scope(request).sigmoid().squeeze(-1)>=.99):
                        packet=self.packet(state,tokenizer,bank,lexical_embeddings)
                        if packet is not None:
                            scored=self.scores(packet)
                            if scored['causal_owner_verified']:
                                plan=self.plan_from_scores(packet,scored,threshold)
                                if plan is not None:return plan
            if not protected:
                proposed=self.unified_proposal(state,tokenizer,bank,lexical_embeddings,threshold)
                if proposed is not None:return proposed
        if packet is None:packet=self.packet(state,tokenizer,bank,lexical_embeddings)
        if packet is None:return None
        if scored is None:scored=self.scores(packet)
        plan=self.plan_from_scores(packet,scored,threshold)
        if plan is None and self.unified_task_applicable(state,tokenizer) and protected:
            # A name-routing false positive must not veto a verified numeric
            # request when the protected planner cannot supply any span.
            proposed=self.unified_proposal(state,tokenizer,bank,lexical_embeddings,threshold)
            if proposed is not None and proposed.confidence>=.99:
                from unified_span_reader import ANSWER_ROLE_FACTS
                digit_index=ANSWER_ROLE_FACTS.index('digits')
                if (len(proposed.requested_properties)>digit_index
                        and proposed.requested_properties[digit_index]>=.99
                        and any(c.isdecimal() for c in tokenizer.decode(proposed.tokens))):
                    return proposed
        return plan

    def plan_from_scores(self,packet,scored,threshold=.9):
        prob=scored['joint'].exp()
        if not scored['extent_supported']:return None
        # Equivalent occurrences/surface variants support the same answer.
        totals={}
        for i,tokens in enumerate(packet['tokens']):totals[tokens]=totals.get(tokens,0.)+prob[i]
        tokens=max(totals,key=lambda key:float(totals[key]))
        confidence=min(1.,max(0.,float(totals[tokens])*float(scored['applicability'].sigmoid())*float(scored['extent_confidence'])))
        if confidence<threshold:return None
        matching=[i for i,item in enumerate(packet['tokens']) if item==tokens]
        best=max(matching,key=lambda i:float(prob[i]));start,_,begin,endpoint=packet['ends'][best]
        kind_prob=scored['kind'].float().softmax(-1);kind_index=int(kind_prob.argmax())
        return AnswerSpanPlan(tokens,begin,endpoint,confidence,lexical_start=packet['starts'][start],
                              answer_kind=self.kind_labels[kind_index],kind_confidence=float(kind_prob[kind_index]),
                              source_owned_person=scored['source_owned_person'])
