# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared word-level question/source alignment and complete-span prediction.

Source words are encoded once. Exact word keys provide equality evidence only;
no occurrence ID, observed profile ID or evaluation answer enters the reader.
"""
import math
import re
import torch
from torch import nn
from torch.nn import functional as F

# Reusable, observed properties rather than an identity row for each answer.
ANSWER_ROLE_FACTS=('letters','digits','alphabetic_value','numeric_value','initial_upper','all_upper',
                   'title_words','percent','currency','hyphen','slash','decimal','comma',
                   'terminal_period','terminal_quote','terminal_bracket','balanced_parentheses',
                   'parentheses','leading_quote','leading_bracket')
ANSWER_EXTENT_FACTS=('words_ge_2','words_ge_3','words_ge_4','words_ge_8','words_ge_12','words_ge_16',
                     'scope_closed','capital_continuation','numeric_continuation')

def answer_span_facts(text,word_count,*,scope_closed=False,capital_continuation=False,numeric_continuation=False):
    text=text.strip();letters=[c for c in text if c.isalpha()];digits=any(c.isdecimal() for c in text)
    words=re.findall(r'[^\W\d_]+',text,flags=re.UNICODE)
    initial=next((c for c in text if c.isalpha()),'')
    roles=(bool(letters),digits,bool(letters) and not digits,digits and not letters,
           bool(initial) and initial.isupper(),bool(letters) and all(c.isupper() for c in letters),
           bool(words) and all(w[0].isupper() for w in words),'%' in text,
           any(c in text for c in '$€£¥₹'),'-' in text or '–' in text or '−' in text,'/' in text,
           bool(re.search(r'\d\.\d',text)),',' in text,text.endswith('.'),
           text.endswith(('"',"'",'”','’')),text.endswith((')',']','}')),
           text.count('(')==text.count(')'),'(' in text or ')' in text,
           text.startswith(('"',"'",'“','‘')),text.startswith(('(','[','{')))
    extent=tuple(word_count>=n for n in (2,3,4,8,12,16))+(scope_closed,capital_continuation,numeric_continuation)
    return tuple(float(v) for v in roles+extent)


def query_word_properties(word_ids,raw,last_positions,tokenizer):
    """Observed case/shape/position, retaining case controls before lexical words."""
    result=[]
    for index,(serial,ids) in enumerate(word_ids):
        end=last_positions.get(serial)
        if end is None:text=tokenizer.decode(list(ids),clean_text=False,collapse_structure=False)
        else:
            begin=end
            while begin-1 in raw and (tokenizer.construction_units[raw[begin-1]].is_lexical or tokenizer.construction_units[raw[begin-1]].kind=='case'):begin-=1
            text=tokenizer.decode([raw[p] for p in range(begin,end+1)],clean_text=False,collapse_structure=False)
        letters=[c for c in text if c.isalpha()];initial=letters[0] if letters else ''
        result.append([float(bool(initial) and initial.isupper()),float(bool(letters) and all(c.isupper() for c in letters)),
                       float(bool(letters)),float(any(c.isdecimal() for c in text)),min(len(text),32)/32.,
                       min(len(ids),16)/16.,min(index,16)/16.,float(index==0)])
    return torch.tensor(result,dtype=torch.float32)


class UnifiedSpanReader(nn.Module):
    def __init__(self,width,hidden=64):
        super().__init__()
        self.hidden=hidden
        self.word_projection=nn.Linear(3*width,hidden)
        self.role=nn.Embedding(2,hidden)
        self.encoder=nn.GRU(hidden,hidden//2,batch_first=True,bidirectional=True)
        self.query=nn.Linear(hidden,hidden,bias=False)
        self.key=nn.Linear(hidden,hidden,bias=False)
        self.exact_gain=nn.Parameter(torch.tensor(2.))
        self.question_pool=nn.Linear(hidden,1)
        self.fusion=nn.Linear(4*hidden,hidden)
        self.evidence=nn.GRU(hidden,hidden//2,batch_first=True,bidirectional=True)
        self.start=nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.length=nn.Embedding(17,8)
        self.end=nn.Sequential(nn.Linear(4*hidden+width+8,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.register_buffer('ready',torch.tensor(False))

    @staticmethod
    def _encode(encoder,values,mask):
        lengths=mask.sum(-1).cpu().clamp_min(1)
        packed=nn.utils.rnn.pack_padded_sequence(values,lengths,batch_first=True,enforce_sorted=False)
        encoded,_=encoder(packed)
        return nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=values.size(1))[0]

    def forward(self,batch):
        source=batch['source'];question=batch['question'];d=source.size(-1)
        zero=torch.zeros_like(question)
        s=F.layer_norm(self.word_projection(torch.cat((source,batch['properties'],batch['anchors']),-1))+self.role.weight[0],(self.hidden,))
        q=F.layer_norm(self.word_projection(torch.cat((question,zero,zero),-1))+self.role.weight[1],(self.hidden,))
        s=self._encode(self.encoder,s,batch['source_mask']);q=self._encode(self.encoder,q,batch['question_mask'])
        similarity=self.query(s)@self.key(q).transpose(-1,-2)/math.sqrt(self.hidden)
        similarity=similarity+self.exact_gain*batch['exact']
        alignment=similarity.masked_fill(~batch['question_mask'][:,None],-torch.inf).softmax(-1)@q
        pool=self.question_pool(q).squeeze(-1).masked_fill(~batch['question_mask'],-torch.inf).softmax(-1)
        request=(q*pool[:,:,None]).sum(1)
        fused=torch.tanh(self.fusion(torch.cat((s,alignment,s*alignment,request[:,None].expand_as(s)),-1)))
        evidence=self._encode(self.evidence,fused,batch['source_mask'])
        words=F.layer_norm(evidence+s,(self.hidden,));requests=request[:,None].expand_as(words)
        start=self.start(torch.cat((words,requests,words*requests),-1)).squeeze(-1).masked_fill(~batch['source_mask'],-torch.inf)
        owners=batch['owners'];finishes=batch['finishes'];rows=torch.arange(len(source),device=source.device)[:,None]
        a=words[rows,owners];b=words[rows,finishes]
        prefix=F.pad(words.cumsum(1),(0,0,1,0))
        count=(finishes-owners+1).clamp(1,16)
        mean=(prefix[rows,finishes+1]-prefix[rows,owners])/count[:,:,None]
        end=self.end(torch.cat((a,b,mean,request[:,None].expand_as(a),batch['end_properties'],self.length(count)),-1)).squeeze(-1)
        # Bounded candidates already encode source-owned markers and layout.
        # A segmented log-normalizer keeps endpoint decisions conditional on start.
        valid=batch['candidate_mask'];masked=end.masked_fill(~valid,-torch.inf)
        maximum=torch.full_like(start,-torch.inf).scatter_reduce(1,owners,masked,reduce='amax',include_self=True)
        stable=torch.where(valid,(end-maximum.gather(1,owners)).exp(),torch.zeros_like(end))
        normal=torch.zeros_like(start).scatter_add(1,owners,stable).clamp_min(1e-30).log()+maximum
        joint=start.log_softmax(-1).gather(1,owners)+end-normal.gather(1,owners)
        return dict(start=start,end=end,joint=joint.masked_fill(~valid,-torch.inf))


def collate_reader_packets(packets,device='cpu'):
    """Pad only word sequences and explicit candidate addresses."""
    def pad(key,value=0):return nn.utils.rnn.pad_sequence([p[key] for p in packets],batch_first=True,padding_value=value).to(device)
    result={key:pad(key) for key in ('source','question','properties','anchors','owners','finishes','end_properties')}
    result['source_mask']=torch.arange(result['source'].size(1),device=device)[None]<torch.tensor([len(p['source']) for p in packets],device=device)[:,None]
    result['question_mask']=torch.arange(result['question'].size(1),device=device)[None]<torch.tensor([len(p['question']) for p in packets],device=device)[:,None]
    result['candidate_mask']=torch.arange(result['owners'].size(1),device=device)[None]<torch.tensor([len(p['owners']) for p in packets],device=device)[:,None]
    exact=torch.zeros(len(packets),result['source'].size(1),result['question'].size(1),device=device)
    for i,p in enumerate(packets):exact[i,:len(p['source']),:len(p['question'])]=p['exact'].to(device)
    result['exact']=exact
    if 'scope_start' in packets[0]:
        result['scope_start']=pad('scope_start');result['scope_end']=pad('scope_end')
        segments=[]
        for row,p in enumerate(packets):
            for begin in sorted(set(p['scope_start'].detach().cpu().tolist())):
                segments.append((row,begin,int(p['scope_end'][begin])))
        longest=max(end-begin for _,begin,end in segments)
        segment_rows=torch.tensor([row for row,_,_ in segments],device=device)
        starts=torch.tensor([begin for _,begin,_ in segments],device=device)
        lengths=torch.tensor([end-begin for _,begin,end in segments],device=device)
        offsets=torch.arange(longest,device=device)[None]
        result['segment_rows']=segment_rows
        result['segment_positions']=(starts[:,None]+offsets).clamp_max(result['source'].size(1)-1)
        result['segment_mask']=offsets<lengths[:,None]

    if 'source_components' in packets[0]:
        words=[];indices={};source_rows=[];question_rows=[]
        def index(components):
            key=tuple(tuple(pair) for pair in components.tolist())
            if key not in indices:indices[key]=len(words);words.append(components)
            return indices[key]
        for p in packets:
            source_rows.append(torch.tensor([index(c) for c in p['source_components']]))
            question_rows.append(torch.tensor([index(c) for c in p['question_components']]))
        result['word_components']=nn.utils.rnn.pad_sequence(words,batch_first=True).to(device)
        result['word_lengths']=torch.tensor([len(c) for c in words])
        result['source_word_index']=nn.utils.rnn.pad_sequence(source_rows,batch_first=True).to(device)
        result['question_word_index']=nn.utils.rnn.pad_sequence(question_rows,batch_first=True).to(device)
    if 'gold' in packets[0]:result['gold']=pad('gold').bool()
    if 'gold_start' in packets[0]:result['gold_start']=pad('gold_start').bool()
    if 'answer_facts' in packets[0]:result['answer_facts']=pad('answer_facts')
    if 'query_properties' in packets[0]:result['query_properties']=pad('query_properties')
    return result


def reader_packet_from_span(packet,version='aligned_v1'):
    source_keys=packet['source_word_ids'];query_keys=packet['question_word_ids']
    exact=torch.tensor([[a==b for b in query_keys] for a in source_keys],device=packet['source_words'].device,dtype=torch.float32)
    d=packet['source_words'].size(-1)
    result=dict(source=packet['source_words'],question=packet['question'],properties=packet['start_features'][:,-d:],
                anchors=packet.get('causal_owner_contexts',torch.zeros_like(packet['source_words']))*(packet.get('causal_owner_evicted',torch.zeros(len(source_keys),device=exact.device,dtype=torch.bool))[:,None] if version in {'masked_relations_v2','scoped_relations_v3','semantic_scoped_v4','semantic_context_v5','requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'} else 1.),
                owners=torch.tensor([e[0] for e in packet['ends']],device=exact.device),
                finishes=torch.tensor([e[1] for e in packet['ends']],device=exact.device),
                end_properties=packet['reader_endpoint_properties'],exact=exact,scope_start=packet['reader_scope_start'],scope_end=packet['reader_scope_end'])
    if version in {'requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:result['answer_facts']=packet['reader_answer_facts']
    if version in {'structural_owner_v9','relation_decisions_v10'}:result['query_properties']=packet['reader_query_properties']
    return result


class RelationalSpanReader(nn.Module):
    """Select via context outside each span, with reusable marker/extent features."""
    def __init__(self,width,hidden=64):
        super().__init__();self.hidden=hidden;self.scoped=False
        self.word_projection=nn.Linear(2*width,hidden)
        self.role=nn.Embedding(2,hidden)
        self.encoder=nn.GRU(hidden,hidden//2,batch_first=True,bidirectional=True)
        self.question_pool=nn.Linear(hidden,1)
        self.length=nn.Embedding(17,8)
        self.score=nn.Sequential(nn.Linear(3*hidden+3*width+18,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.register_buffer('ready',torch.tensor(False))

    def encode_source(self,values,batch):
        if not self.scoped:return UnifiedSpanReader._encode(self.encoder,values,batch['source_mask'])
        rows=batch['segment_rows'][:,None];positions=batch['segment_positions'];mask=batch['segment_mask']
        encoded=UnifiedSpanReader._encode(self.encoder,values[rows,positions],mask)
        addresses=(rows*values.size(1)+positions)[mask]
        result=torch.zeros_like(values).flatten(0,1).index_copy(0,addresses,encoded[mask])
        return result.reshape_as(values)

    def forward(self,batch):
        source=batch['source'];question=batch['question'];h=self.hidden//2
        if hasattr(self,'semantic_words'):
            semantic=self.semantic_words(batch['word_components'],batch['word_lengths'])
            gain=self.semantic_gain.tanh()
            source=F.layer_norm(source+gain*semantic[batch['source_word_index']],(source.size(-1),))
            question=F.layer_norm(question+gain*semantic[batch['question_word_index']],(question.size(-1),))
        s=F.layer_norm(self.word_projection(torch.cat((source,batch['properties']),-1))+self.role.weight[0],(self.hidden,))
        q=F.layer_norm(self.word_projection(torch.cat((question,torch.zeros_like(question)),-1))+self.role.weight[1],(self.hidden,))
        source_input,question_input=s,q
        # The shared encoder also carries ordered document context across clauses.
        # Its forward-before / backward-after states still exclude the entire
        # candidate's value. No occurrence IDs or new profile embeddings enter.
        document=UnifiedSpanReader._encode(self.encoder,s,batch['source_mask']) if hasattr(self,'document_context_gain') else None
        s=self.encode_source(s,batch);q=UnifiedSpanReader._encode(self.encoder,q,batch['question_mask'])
        weights=self.question_pool(q).squeeze(-1).masked_fill(~batch['question_mask'],-torch.inf).softmax(-1)
        request=(q*weights[:,:,None]).sum(1)
        # Forward state before the start and backward state after the endpoint
        # cannot see candidate value tokens. Every source word is encoded once.
        before=F.pad(s[...,:h],(0,0,1,0))[:,:source.size(1)]
        after=F.pad(s[...,h:],(0,0,0,1))[:,1:]
        owners=batch['owners'];finishes=batch['finishes'];rows=torch.arange(len(source),device=source.device)[:,None]
        left_context=before[rows,owners];right_context=after[rows,finishes]
        if self.scoped:
            left_context=left_context*(owners>batch['scope_start'].gather(1,owners))[:,:,None]
            right_context=right_context*(finishes+1<batch['scope_end'].gather(1,finishes))[:,:,None]
        context=torch.cat((left_context,right_context),-1)
        if document is not None:
            document_before=F.pad(document[...,:h],(0,0,1,0))[:,:source.size(1)]
            document_after=F.pad(document[...,h:],(0,0,0,1))[:,1:]
            context=context+self.document_context_gain.tanh()*torch.cat((document_before[rows,owners],document_after[rows,finishes]),-1)
        # Exact lexical keys supply relational anchor evidence outside the span.
        # Repeated function words do not inflate question coverage.
        cumulative=F.pad(batch['exact'].cumsum(1),(0,0,1,0));positions=torch.arange(source.size(1),device=source.device)[None].expand(len(source),-1)
        left=[];right=[];r=torch.arange(len(source),device=source.device)[:,None]
        lengths=batch['source_mask'].sum(-1)[:,None]
        for window in (2,4,8,16,32):
            left_edge=(positions-window).clamp_min(0);right_edge=torch.minimum(positions+window+1,lengths)
            if self.scoped:
                left_edge=torch.maximum(left_edge,batch['scope_start']);right_edge=torch.minimum(right_edge,batch['scope_end'].clamp_min(positions+1))
            lp=(cumulative[r,positions]-cumulative[r,left_edge]).clamp_max(1.)
            rp=(cumulative[r,right_edge]-cumulative[r,positions+1]).clamp_max(1.)
            left.append((lp*weights[:,None]).sum(-1));right.append((rp*weights[:,None]).sum(-1))
        coverage=torch.cat((torch.stack(left,-1)[rows,owners],torch.stack(right,-1)[rows,finishes]),-1)
        props=batch['properties'][rows,owners];memory=batch['anchors'][rows,owners]
        count=(finishes-owners+1).clamp(1,16)
        req=request[:,None].expand_as(context)
        features=torch.cat((context,req,context*req,memory,props,batch['end_properties'],self.length(count),coverage),-1)
        factors=self.factored_matching(source_input,question_input,s,q,request,batch) if hasattr(self,'factored_matching') else None
        role_logits=role_compatibility=None
        if hasattr(self,'requested_roles'):
            role_logits=self.requested_roles(request)
            if hasattr(self,'requested_roles_nonlinear'):
                role_logits=role_logits+self.requested_roles_nonlinear(request)
            if hasattr(self,'relation_properties'):
                relation_logits=self.relation_properties(factors['relation_request'])
                mix=factors['factored_applicability_logits'].sigmoid()[:,None]*self.relation_properties_gain.tanh().clamp_min(0.)
                role_logits=role_logits*(1-mix)+relation_logits*mix
                factors['relation_requested_role_logits']=relation_logits
            facts=batch['answer_facts'];logits=role_logits[:,None].expand_as(facts)
            compatibility=-F.binary_cross_entropy_with_logits(logits,facts,reduction='none')
            split=len(ANSWER_ROLE_FACTS)
            role_compatibility=compatibility.sum(-1)
        valid=batch['candidate_mask']
        if self.scoped:valid=valid&(batch['scope_start'].gather(1,owners)==batch['scope_start'].gather(1,finishes))
        result=self.score_candidates(features,owners,valid,source.size(1),role_compatibility,
                                     factors['matching_offset'] if factors is not None else None)
        if factors is not None:result.update(factors)
        if role_logits is not None:result.update(requested_role_logits=role_logits,role_compatibility=role_compatibility)
        return result

    def score_candidates(self,features,owners,valid,source_length,role_compatibility=None,matching_offset=None):
        """Joint scorer shared by ordinary forwards and frozen-encoder fitting.

        Cached features are valid only while all their producing parameters stay
        frozen. Masks and source occurrence indices retain their original meaning.
        """
        score=self.score(features).squeeze(-1)
        if role_compatibility is not None:score=score+self.requested_roles_gain.tanh()*role_compatibility
        if matching_offset is not None:score=score+matching_offset
        joint=score.masked_fill(~valid,-torch.inf).log_softmax(-1)
        # Start and conditional endpoint probabilities remain separately visible.
        maximum=torch.full((len(features),source_length),-torch.inf,device=features.device,dtype=score.dtype).scatter_reduce(1,owners,joint,reduce='amax',include_self=True)
        mass=torch.where(valid,(joint-maximum.gather(1,owners)).exp(),torch.zeros_like(joint))
        start=torch.zeros_like(maximum).scatter_add(1,owners,mass).clamp_min(1e-30).log()+maximum
        result=dict(start=start,end=joint-start.gather(1,owners),joint=joint,relation_logits=score)
        return result


class FactoredMatching(nn.Module):
    """Learn query owner tags separately from owner-masked relation matching.

    Equality keys only establish ownership. Relation states before/after a
    candidate exclude its value; no gold addresses enter the forward pass.
    """
    def __init__(self,hidden,structural_owner=False):
        super().__init__();self.hidden=hidden;self.structural_owner=structural_owner
        self.owner_tagger=nn.Sequential(nn.Linear(8 if structural_owner else hidden,hidden//2),nn.Tanh(),nn.Linear(hidden//2,1))
        self.relation_encoder=nn.GRU(hidden,hidden//2,batch_first=True,bidirectional=True)
        self.relation_pool=nn.Linear(hidden,1)
        self.applicability=nn.Sequential(nn.Linear(2*hidden,hidden//2),nn.Tanh(),nn.Linear(hidden//2,1))
        self.owner_gain=nn.Parameter(torch.zeros(()));self.relation_gain=nn.Parameter(torch.zeros(()))

    def forward(self,source_input,question_input,source_states,question_states,request,batch):
        mask=batch['question_mask'];owner_logits=self.owner_tagger(batch['query_properties'] if self.structural_owner else question_states).squeeze(-1)
        owner=owner_logits.sigmoid()*mask
        source_owner=(batch['exact']@owner[:,:,None]).squeeze(-1).clamp(0,1)
        q=UnifiedSpanReader._encode(self.relation_encoder,question_input*(1-owner[:,:,None]),mask)
        pool=self.relation_pool(q).squeeze(-1).masked_fill(~mask,-torch.inf).softmax(-1)
        relation=(q*pool[:,:,None]).sum(1)
        rows=batch['segment_rows'][:,None];positions=batch['segment_positions'];segment_mask=batch['segment_mask']
        values=source_input*(1-source_owner[:,:,None])
        encoded=UnifiedSpanReader._encode(self.relation_encoder,values[rows,positions],segment_mask)
        addresses=(rows*values.size(1)+positions)[segment_mask]
        states=torch.zeros_like(values).flatten(0,1).index_copy(0,addresses,encoded[segment_mask]).reshape_as(values)
        h=self.hidden//2;owners=batch['owners'];finishes=batch['finishes'];r=torch.arange(len(values),device=values.device)[:,None]
        before=F.pad(states[...,:h],(0,0,1,0))[:,:values.size(1)]
        after=F.pad(states[...,h:],(0,0,0,1))[:,1:]
        left=before[r,owners]*(owners>batch['scope_start'].gather(1,owners))[:,:,None]
        right=after[r,finishes]*(finishes+1<batch['scope_end'].gather(1,finishes))[:,:,None]
        relation_context=torch.cat((left,right),-1)
        relation_logits=F.cosine_similarity(relation_context,relation[:,None],dim=-1)*10.
        cumulative=F.pad(batch['exact'].cumsum(1),(0,0,1,0))
        begins=batch['scope_start'].gather(1,owners);ends=batch['scope_end'].gather(1,finishes)
        covered=((cumulative[r,owners]-cumulative[r,begins])+(cumulative[r,ends]-cumulative[r,finishes+1])).clamp_max(1.)
        coverage=(covered*owner[:,None]).sum(-1)/owner.sum(-1).clamp_min(1e-6)[:,None]
        owner_logits_match=coverage.clamp_min(.001).log()*owner.max(-1).values[:,None]
        source_mean=(source_states*batch['source_mask'][:,:,None]).sum(1)/batch['source_mask'].sum(1)[:,None]
        applicable=self.applicability(torch.cat((request,source_mean),-1)).squeeze(-1)
        if hasattr(self,'relation_applicability'):
            applicable=applicable+self.relation_applicability(torch.cat((relation,source_mean),-1)).squeeze(-1)
        offset=applicable.sigmoid()[:,None]*(self.owner_gain.tanh()*10.*owner_logits_match+self.relation_gain.tanh()*10.*relation_logits)
        return dict(matching_offset=offset,query_owner_logits=owner_logits,owner_match_logits=owner_logits_match,relation_request=relation,source_summary=source_mean,
                    factored_relation_logits=relation_logits,factored_applicability_logits=applicable)


class SemanticWordComposer(nn.Module):
    """One bounded component bank and ordered unit encoder for source and query."""
    def __init__(self,width,capacity=512):
        super().__init__();self.embedding=nn.Embedding(capacity+1,width,padding_idx=0)
        self.encoder=nn.GRU(width,width,batch_first=True)
    def forward(self,components,lengths):
        values=self.embedding(components).mean(-2)
        packed=nn.utils.rnn.pack_padded_sequence(values,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,state=self.encoder(packed)
        return F.layer_norm(state[0],(values.size(-1),))


def semantic_components(word_ids,tokenizer,capacity=512):
    from signature_bank import stable_key
    pairs=[]
    for token in word_ids:
        unit=tokenizer.construction_units[token];key='semantic_unit:'+unit.kind+':'+unit.text
        pairs.append([1+stable_key(salt+key)%capacity for salt in ('a:','b:')])
    return torch.tensor(pairs,dtype=torch.long)


def create_unified_reader(width,version='aligned_v1'):
    if version=='aligned_v1':reader=UnifiedSpanReader(width)
    elif version in {'masked_relations_v2','scoped_relations_v3','semantic_scoped_v4','semantic_context_v5','requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:
        reader=RelationalSpanReader(width);reader.scoped=version!='masked_relations_v2'
        if version in {'semantic_scoped_v4','semantic_context_v5','requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:
            reader.semantic_words=SemanticWordComposer(width)
            reader.semantic_gain=nn.Parameter(torch.zeros(()))
        if version in {'semantic_context_v5','requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:reader.document_context_gain=nn.Parameter(torch.zeros(()))
        if version in {'requested_roles_v6','nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:
            reader.requested_roles=nn.Linear(reader.hidden,len(ANSWER_ROLE_FACTS)+len(ANSWER_EXTENT_FACTS))
            nn.init.zeros_(reader.requested_roles.weight);nn.init.zeros_(reader.requested_roles.bias)
            reader.requested_roles_gain=nn.Parameter(torch.zeros(()))
            if version in {'nonlinear_roles_v7','factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:
                reader.requested_roles_nonlinear=nn.Sequential(nn.Linear(reader.hidden,reader.hidden),nn.Tanh(),nn.Linear(reader.hidden,len(ANSWER_ROLE_FACTS)+len(ANSWER_EXTENT_FACTS)))
                # Preserve the learned linear head exactly at checkpoint migration.
                nn.init.zeros_(reader.requested_roles_nonlinear[-1].weight)
                nn.init.zeros_(reader.requested_roles_nonlinear[-1].bias)
        if version in {'factored_matching_v8','structural_owner_v9','relation_decisions_v10'}:reader.factored_matching=FactoredMatching(reader.hidden,version in {'structural_owner_v9','relation_decisions_v10'})
        if version=='relation_decisions_v10':
            reader.factored_matching.relation_applicability=nn.Sequential(nn.Linear(2*reader.hidden,reader.hidden//2),nn.Tanh(),nn.Linear(reader.hidden//2,1))
            reader.relation_properties=nn.Sequential(nn.Linear(reader.hidden,reader.hidden),nn.Tanh(),nn.Linear(reader.hidden,len(ANSWER_ROLE_FACTS)+len(ANSWER_EXTENT_FACTS)))
            reader.relation_properties_gain=nn.Parameter(torch.zeros(()))
            for head in (reader.factored_matching.relation_applicability,reader.relation_properties):
                nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    else:raise ValueError('Unknown unified span reader version: '+version)
    reader.version=version
    return reader
