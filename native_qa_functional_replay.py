"""Functional replay through the production decision, guidance and mixture paths."""
import torch
import torch.nn.functional as F
try:
    from .native_training_objectives import preserve_decision_margin,gold_mass_loss
except ImportError:
    from native_training_objectives import preserve_decision_margin,gold_mass_loss


def source_supervision(details, positive, mode):
    corrected = gold_mass_loss(details['fused'],positive)
    if mode=='control':
        return corrected
    if mode=='branch_consistent':
        if details.get('semantic') is None:
            raise ValueError('Semantic-only supervision requires a semantic scorer')
        return (corrected+gold_mass_loss(details['semantic'],positive))/2
    raise ValueError('Unknown source-objective arm')


def protected_decision(logits, teacher_logits, positive):
    """Gold NLL plus preservation only when the teacher's decision is correct."""
    loss=gold_mass_loss(logits,positive)
    teacher=teacher_logits.detach().log_softmax(0)
    if bool(positive[int(teacher.argmax())]):
        loss=loss+F.kl_div(logits.log_softmax(0),teacher.exp(),reduction='sum')
        loss=loss+preserve_decision_margin(logits,teacher,positive)
    return loss


def protected_gate(logit, teacher_logit, appropriate=True):
    teacher=teacher_logit.detach()
    correct=bool((teacher>0).item())==bool(appropriate)
    if not correct:
        return F.binary_cross_entropy_with_logits(logit,logit.new_tensor(float(appropriate)))
    log_yes,log_no=F.logsigmoid(teacher),F.logsigmoid(-teacher)
    return log_yes.exp()*(log_yes-F.logsigmoid(logit))+log_no.exp()*(log_no-F.logsigmoid(-logit))


def protected_tokens(logits, teacher_logits, labels, mask):
    """Mean per supervised token; incorrect teacher steps receive gold CE only."""
    if logits.shape!=teacher_logits.shape or labels.shape!=logits.shape[:-1] or mask.shape!=labels.shape:
        raise ValueError('Token replay arrays do not align')
    if not (mask>0).any():
        raise ValueError('Replay needs supervised output positions')
    logp=logits.log_softmax(-1);teacher=teacher_logits.detach().log_softmax(-1)
    correct=teacher.argmax(-1).eq(labels)
    support=teacher.exp()>0
    safe_teacher=torch.where(support,teacher,torch.zeros_like(teacher))
    safe_current=torch.where(support,logp,torch.zeros_like(logp))
    kl=(teacher.exp()*(safe_teacher-safe_current)).sum(-1)
    gold=F.nll_loss(logp.reshape(-1,logp.shape[-1]),labels.reshape(-1),reduction='none').reshape_as(labels)
    term=torch.where(correct,kl,gold)
    return (term*mask).sum()/mask.sum().clamp_min(1)


def cached_emission(model, packet, question, recalled, plan):
    """Frozen backbone states plus the same guidance and mixture as inference.

    Output labels are intentionally absent from this function. Prefix records
    contain only already emitted tokens. All learned decision states are passed
    from a freshly computed production decision_packet.
    """
    base=packet['backbone_hidden']
    if base.requires_grad:
        raise ValueError('Cached backbone states must be frozen')
    memory=model.native_question_memory
    guided=memory.guide_readout(base,question,recalled,packet['residual_mask'])
    logits=model._construction_logits_from_guided(guided[None])
    records=[(0,i,question,guided[i],plan,tuple(prefix)) for i,prefix in packet['emitted'].items()]
    return memory.span_readout.mix(logits,records)[0]
