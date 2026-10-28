"""Observed contextual features for the existing native QA heads.

No trainable encoder lives here. Every word, request and completed-line vector
is supplied by the model's ordinary torus/SHARC/hierarchy forward path.
"""
import math
import torch


def clause_neighbors(segments,index):
    """The existing eight-word-per-side observed clause scope."""
    left=[];right=[]
    for j in range(index-1,max(-1,index-9),-1):
        if segments[j]!=segments[index]:break
        left.append(j)
    for j in range(index+1,min(len(segments),index+9)):
        if segments[j]!=segments[index]:break
        right.append(j)
    return left,right


def candidate_local_context(words,segments):
    """Input-only context; no candidate word, request, answer or extra encoder."""
    if words.ndim!=2 or len(words)!=len(segments):
        raise ValueError('Candidate-local context requires aligned word states and clause IDs')
    if not len(words):return words.new_zeros(words.shape)
    contexts=[]
    for i in range(len(words)):
        left,right=clause_neighbors(segments,i);indices=left+right
        contexts.append(words[indices].mean(0) if indices else words.new_zeros(words.shape[-1]))
    return torch.stack(contexts)


def contextual_plan(memory,state,tokenizer,diagnostics=False):
    n=len(state.source)
    if not (n==len(state.source_vectors)==len(state.source_summaries)==len(state.source_segments)==len(state.source_lines)):
        raise ValueError('Source addresses and backbone observations are misaligned')
    if len(state.question)!=len(state.question_vectors):
        raise ValueError('Question addresses and backbone observations are misaligned')
    words=torch.stack(state.source_vectors)
    summaries=(candidate_local_context(words,state.source_segments)
               if memory.backbone_context=='candidate_local_clause_v1'
               else torch.stack(state.source_summaries))
    query_words=torch.stack(state.question_vectors)
    question=state.request_vector if state.request_vector is not None else query_words[-1]
    width=question.shape[-1]
    keys=memory.key(torch.cat((words,summaries),-1))
    if memory.document_key is not None:
        keys=keys+memory.document_key(words)
    context_words=[];role_features=[];corrections=[]
    for i,(_,before) in enumerate(state.source):
        # Preserve clause-local, already-observed right context. Completed-line
        # summaries additionally retain predicates that follow an answer name.
        correction=words.new_zeros(width)
        right=[]
        for j in range(i+1,min(n,i+1+memory.future_context)):
            if state.source_segments[j]!=state.source_segments[i]:
                break
            right.append(j)
        context_words.append(before+tuple(state.source[j][0] for j in right))
        if before and i and memory.previous_key is not None:
            correction=correction+memory.previous_key(words[i-1])
        if right:
            value=memory.future_key(words[[i]+right].mean(0))
            correction=correction+value
        corrections.append(correction)
        if memory.source_roles is not None:
            left_indices,right_indices=clause_neighbors(state.source_segments,i)
            zero=words.new_zeros(width)
            left=words[left_indices].mean(0) if left_indices else zero
            right=words[right_indices].mean(0) if right_indices else zero
            props=words.new_tensor([len(left_indices)/8,len(right_indices)/8,float(not left_indices),float(not right_indices)])
            role_features.append(torch.cat((left,right,props)))
    keys=keys+torch.stack(corrections)
    projected=memory.query(question)
    if memory.request_roles is None:
        if memory.relation_alignment is not None:
            phrases=torch.stack([query_words[i:i+k].mean(0) for k in (1,2,3)
                                 for i in range(len(query_words)-k+1)])
            projected=memory.relation_alignment(query_words,phrases,projected)
        scores=keys@projected/math.sqrt(width)
    else:
        roles=memory.request_roles(query_words).softmax(0)
        owner=memory.query(roles[:,0]@query_words)
        relation=memory.query(roles[:,1]@query_words)
        scores=(keys[:,:width//2]@owner[:width//2]+keys[:,width//2:]@relation[width//2:])/math.sqrt(width)
    if memory.source_roles is not None:
        scores=scores+memory.source_roles(projected,torch.stack(role_features))
    if memory.lexical_scale is not None:
        overlap=question.new_tensor([sum(word in context for word in state.question) for context in context_words])
        scores=scores+memory.lexical_scale*overlap
    antecedent=None
    if memory.antecedent_memory is not None:
        compatibility,antecedent=memory.antecedent_memory.access_contextual(state,query_words)
        scores=scores+memory.antecedent_memory.scale*compatibility
    packet=memory.decision_packet(question,words,memory.value(words),scores,query_words,None,
        lambda selected:memory.span_readout.candidates(state,tokenizer,selected),
        [word for word,_ in state.source],diagnostics)
    if packet[3] is not None:
        packet[3]['source_keys']=keys
        packet[3]['antecedent']=antecedent
        packet[3]['representation']='backbone_context_v1'
        packet[3]['key_context_policy']=memory.backbone_context
        if diagnostics:packet[3]['source_key_context']=summaries
    return packet
