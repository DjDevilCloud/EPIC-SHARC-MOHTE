"""Functional replay over a freshly computed native forward, never cached student states."""
import torch
import torch.nn.functional as F
try:
    from .native_training_objectives import gold_mass_loss,preserve_decision_margin
    from .native_qa_functional_replay import protected_decision,protected_gate,protected_tokens
except ImportError:
    from native_training_objectives import gold_mass_loss,preserve_decision_margin
    from native_qa_functional_replay import protected_decision,protected_gate,protected_tokens

PACKET_KEYS={'version','input_ids','position','source_words','starts','extent_groups','positions','labels',
             'teacher_source','teacher_gate','teacher_tokens','provenance'}

def boundary_ids(tokenizer):
    return sorted({tokenizer.eos_id}|{tokenizer.special_tokens[k] for k in ('<BOO>','<EOO>','<LINE>','<EOL>') if k in tokenizer.special_tokens})

def token_groups(labels,weights,terminal_ids):
    boundary=torch.zeros_like(labels,dtype=torch.bool)
    for token in terminal_ids:boundary|=labels.eq(token)
    groups=[weights*mask.to(weights.dtype) for mask in (~boundary,boundary)]
    return [mask for mask in groups if bool((mask>0).any())]

def balanced_output_ce(token_losses,labels,mask,pad_id,terminal_ids):
    """Mean over examples and content/boundary decisions, not answer lengths."""
    if token_losses.shape!=labels.shape or mask is None or mask.shape!=labels.shape:
        raise ValueError('Balanced native replay requires aligned token losses, labels and output mask')
    losses=[]
    for loss,row,weight in zip(token_losses,labels,mask):
        weight=weight.to(loss.dtype)*row.ne(pad_id)
        groups=token_groups(row,weight,terminal_ids)
        if not groups:raise ValueError('Each replay batch row requires supervised output tokens')
        losses.append(torch.stack([(loss*g).sum()/g.sum() for g in groups]).mean())
    return torch.stack(losses).mean()

def frozen_tensor(value,reference,shape):
    if torch.is_tensor(value) and value.requires_grad:
        raise ValueError('Replay teachers must be frozen; student learned states cannot be cached as teachers')
    value=torch.as_tensor(value,device=reference.device,dtype=reference.dtype)
    if value.shape!=shape or not bool(torch.isfinite(value).all()):
        raise ValueError('Replay teacher shape/nonfinite mismatch')
    return value

def source_preservation(logits,teacher,positive):
    """Gold source NLL is already applied once by ordinary native supervision."""
    if teacher is None:return logits.new_zeros(())
    old=frozen_tensor(teacher,logits,logits.shape).log_softmax(0)
    if not bool(positive[int(old.argmax())]):return logits.new_zeros(())
    weights=old.exp();support=weights>0
    kl=(weights[support]*(old[support]-logits.log_softmax(0)[support])).sum()
    return kl+preserve_decision_margin(logits,old,positive)

def live_replay(model,output,input_ids,labels,mask,source_targets,replay_targets):
    """Post-forward labels supervise live plans; auxiliary spans never replace inference proposals."""
    if not replay_targets or mask is None:
        raise ValueError('Live replay requires nonempty targets and explicit output masks')
    if mask.shape!=labels.shape or not bool(torch.isfinite(mask).all()) or bool((mask<0).any()):
        raise ValueError('Live replay output masks must be aligned, finite and nonnegative')
    if len(source_targets)!=labels.shape[0] or {x['row'] for x in source_targets}!=set(range(labels.shape[0])):
        raise ValueError('Live replay v1 requires exactly one annotated QA per batch row')
    head=model.native_question_memory.span_readout
    if not (head.stable_gate and head.conditional_copy):
        raise ValueError('Live replay requires independent stable conditional-copy confidence')
    by_position={(r,p):(q,current,plan,emitted) for r,p,q,current,plan,emitted in output.native_qa_readouts}
    source_by_position={(x['row'],x['position']):x for x in source_targets}
    components={k:[] for k in ('source','extent','gate','tokens')};seen=set();missing_payloads=0
    terminals=boundary_ids(model._prismal_tokenizer)
    for target in replay_targets:
        if not isinstance(target,dict) or set(target)!={'row','position','packet'}:
            raise ValueError('Live replay targets require row, position and frozen packet only')
        address=(target['row'],target['position'])
        if address in seen or address not in source_by_position or address not in by_position:
            raise ValueError('Duplicate or unaligned replay QA address')
        seen.add(address);row,position=address;p=target['packet']
        if not isinstance(p,dict) or set(p)!=PACKET_KEYS or p['version']!='native_live_replay_v1':
            raise ValueError('Unknown live replay packet schema; cached student features are not allowed')
        if p['position']!=position:raise ValueError('Replay prefix position mismatch')
        q,current,plan,emitted=by_position[address]
        if emitted:raise ValueError('Replay decisions require an empty output prefix')
        expected=torch.as_tensor(p['input_ids'],device=input_ids.device,dtype=input_ids.dtype)
        if expected.ndim!=1 or len(expected)>input_ids.shape[1] or not torch.equal(input_ids[row,:len(expected)],expected):
            raise ValueError('Replay input or teacher-forced history mismatch')
        if bool(input_ids[row,len(expected):].ne(model.cfg.pad_id).any()):
            raise ValueError('Replay input contains unrecorded nonpadding history')
        if [tuple(w) for w in p['source_words']]!=[tuple(w) for w in plan['source_words']]:
            raise ValueError('Replay source addresses changed')
        starts=p['starts']
        if sorted(starts)!=sorted(source_by_position[address]['starts']):raise ValueError('Replay support labels changed')
        source=plan['start_log_probabilities'];positive=torch.zeros_like(source,dtype=torch.bool);positive[starts]=True
        components['source'].append(source_preservation(source,p['teacher_source'],positive))
        expected_labels=torch.as_tensor(p['labels'],device=labels.device,dtype=labels.dtype)
        positions=torch.as_tensor(p['positions'],device=labels.device,dtype=torch.long)
        valid=(mask[row].to(torch.float32)*labels[row].ne(model.cfg.pad_id)).nonzero().flatten()
        if positions.ndim!=1 or not torch.equal(positions,valid) or not torch.equal(labels[row,positions],expected_labels):
            raise ValueError('Replay output positions or labels changed')
        payload=tuple(expected_labels.detach().cpu().tolist())
        missing_payloads+=int(not any(tuple(c['tokens'])==payload for c in plan['candidates']))
        # Candidate descriptors contain frozen observed surfaces only. These are
        # explicit gold-start auxiliary groups, even outside the inference bank.
        if {g['start'] for g in p['extent_groups']}!=set(starts) or len(p['extent_groups'])!=len(starts):
            raise ValueError('Conditional extent groups do not match reviewed source starts')
        extents=[]
        for group in p['extent_groups']:
            if set(group)!={'start','candidates','teacher_logits'}:raise ValueError('Invalid extent replay group')
            candidates=group['candidates']
            if not candidates or any(c['start']!=group['start'] or not c['start']<=c['end']<len(source) for c in candidates):
                raise ValueError('Invalid observed extent candidate addresses')
            scores=head.score_candidates(candidates,q,plan['word_vectors'],torch.zeros_like(source),plan['marker_words'])
            gold=torch.tensor([tuple(c['tokens'])==payload for c in candidates],device=scores.device,dtype=torch.bool)
            if group['teacher_logits'] is None:extent=gold_mass_loss(scores,gold)
            else:extent=protected_decision(scores,frozen_tensor(group['teacher_logits'],scores,scores.shape),gold)
            extents.append(extent)
        components['extent'].append(torch.stack(extents).mean())
        gate=head.gate_logit(q,current,plan)
        gate_loss=(F.binary_cross_entropy_with_logits(gate,gate.new_ones(())) if p['teacher_gate'] is None else
                   protected_gate(gate,frozen_tensor(p['teacher_gate'],gate,gate.shape),True))
        components['gate'].append(gate_loss)
        logits=output.logits[row,positions].float();teacher=frozen_tensor(p['teacher_tokens'],logits,logits.shape)
        weights=mask[row,positions].to(logits.dtype)
        terms=[protected_tokens(logits,teacher,expected_labels,g) for g in token_groups(expected_labels,weights,terminals)]
        components['tokens'].append(torch.stack(terms).mean())
    # Each example has equal decision-type weight; replay slots keep their
    # ordinary full-batch share, rather than gaining weight from answer length.
    totals={k:torch.stack(v).sum()/len(source_targets) for k,v in components.items()}
    loss=torch.stack(list(totals.values())).mean()
    return loss,dict(**totals,examples=len(replay_targets),gold_payload_absent=missing_payloads)
