"""Supervised decision preservation for native QA adapter fitting."""
import torch
import torch.nn.functional as F


def causal_owner_supervision(details, targets):
    """Absolute latest-owner NLL from production logs, without renormalization.

    Evaluator/training labels address an observed source row and its causal
    owner event. They do not alter retrieval or inference proposal construction.
    Using the production log mass preserves gradients when exp(mass) underflows.
    """
    logs = details.get('ownership_log_probabilities')
    if logs is None or not targets:
        raise ValueError('Causal owner supervision requires production log masses and explicit targets')
    losses=[];seen=set()
    for target in targets:
        if not isinstance(target,dict) or set(target)!={'source_index','owner_index'}:
            raise ValueError('Owner targets require source_index and owner_index only')
        source,owner=target['source_index'],target['owner_index']
        if any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in (source,owner)):
            raise ValueError('Owner target addresses must be nonnegative integers')
        if source in seen or source>=len(logs) or owner>=len(logs[source]):
            raise ValueError('Duplicate or out-of-range causal owner target')
        seen.add(source);losses.append(-logs[source][owner])
    return torch.stack(losses).mean()


def gold_mass_loss(logits, positive):
    if positive.dtype!=torch.bool or positive.shape!=logits.shape or not positive.any():
        raise ValueError('Nonempty aligned gold-decision mask required')
    return -logits.log_softmax(0)[positive].logsumexp(0)


def native_source_supervision(readouts,targets):
    """Labels select loss terms after ordinary inference constructed its plans.

    Targets specify row, input position of an empty-prefix readout, and all
    supported start indices. Explicit positions avoid guessing QA-block identity.
    """
    if not targets:
        raise ValueError('Native source supervision requires explicit nonempty targets')
    by_position={(row,position):(plan,emitted) for row,position,_,_,plan,emitted in (readouts or [])}
    losses=[];seen=set()
    for target in targets:
        if not isinstance(target,dict) or set(target)!={'row','position','starts'}:
            raise ValueError('Native source targets require row, position and starts only')
        if any(not isinstance(target[k],int) or isinstance(target[k],bool) or target[k]<0 for k in ('row','position')):
            raise ValueError('Native source target addresses must be nonnegative integers')
        address=(target['row'],target['position'])
        if address not in by_position:
            raise ValueError('Native source target has no active readout at its input address')
        plan,emitted=by_position[address]
        identity=(target['row'],id(plan))
        if emitted or identity in seen:
            raise ValueError('One empty-prefix native source target per QA block required')
        seen.add(identity)
        starts=target['starts'];n=len(plan['start_log_probabilities'])
        if not isinstance(starts,(list,tuple)) or not starts or any(not isinstance(i,int) or isinstance(i,bool) or not 0<=i<n for i in starts) or len(set(starts))!=len(starts):
            raise ValueError('Supported source starts must be unique in-range integer indices')
        positive=torch.zeros(n,dtype=torch.bool,device=plan['start_log_probabilities'].device)
        positive[starts]=True
        losses.append(gold_mass_loss(plan['start_log_probabilities'],positive))
    return torch.stack(losses).mean()


def decision_margin(logits, positive):
    """Log mass of acceptable decisions minus the strongest alternative.

    The positive mask is a training annotation; this function is never used
    to select a source or an endpoint during inference.
    """
    if logits.ndim != 1 or positive.shape != logits.shape or positive.dtype != torch.bool:
        raise ValueError('Decision margins require a vector and matching boolean labels')
    if not positive.any():
        raise ValueError('Decision margins require at least one acceptable decision')
    if positive.all():
        return logits.sum() * 0.
    return logits[positive].logsumexp(0) - logits[~positive].max()


def preserve_decision_margin(logits, teacher_logits, positive, max_margin=3.):
    """Preserve a frozen teacher's capped separation without raising its target.

    KL alone can stay small when a close correct/incorrect ordering flips.
    This hinge penalizes losing the teacher's positive decision margin. A
    nonpositive teacher margin supplies no positive separation target.
    """
    if max_margin < 0:
        raise ValueError('The preserved margin cap must be nonnegative')
    target = decision_margin(teacher_logits.detach(), positive).clamp(0., max_margin)
    return F.relu(target - decision_margin(logits, positive))
