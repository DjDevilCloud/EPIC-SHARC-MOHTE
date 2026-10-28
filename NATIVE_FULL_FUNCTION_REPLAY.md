# Native binding reconciliation and functional replay

This prerequisite is recorded in `review_artifacts/native_full_function_contract_v1`.
No scientific fitting or locked-test prediction is part of it. Accepted checkpoint
files and historical experiment directories remain unchanged.

## Model roles and saved execution

The accepted checkpoints contain both identity and native readers. Their saved
legacy route selects identity, whereas the historical native tests overrode that
selection. Explicit saved routes resolve the discrepancy without deleting modules.
The identity reference is a route diagnostic, never a student fallback or a
positive teacher for this contract.

Two frozen native references are available per seed: the retained-decisions
experimental parent and the accepted baseline. Reference clones save `native`
and `native_qa_eval_v1`; their parameter tensors are unchanged. The parent supplies
each correct teacher decision first, then the accepted native reference. Incorrect
teacher decisions receive reviewed gold supervision instead of preservation.

The restored student retains the candidate-local clause context and all previously
learned student tensors. It restores 12 missing existing native binding tensors
from the parent: antecedent memory, previous-word keys and lexical scale.
Antecedent/lexical/owner-evidence flags, ordered identity queries, capacity and
anchor settings match the parent. It keeps the shared backbone representation,
with no independent native GRU or semantic encoder. The identity reader is absent
from this student and the explicit native route supplies all QA outputs. Torus,
SHARC and shared hierarchy components remain upstream of its observed states.
This does not introduce hierarchical experts or change width or memory budgets.

## Reviewed replay inputs

438 replay examples comprise 288 controlled, 60 native, 58 reviewed legacy and
32 layout cases. The remaining 136 retention cases are evaluation only: fresh
values, unfamiliar owners and held grammar. Original denominators are retained,
including four repeated layout/native inputs. The 112 newer source controls
remain a separate evaluation ledger.

The fingerprinted support review was made from passages and questions, without
model predictions. The first binding review failed; its report is preserved.
Version two consistently credits same-rover coreference for Opportunity's two
questions. The possessive occurrence remains explicitly uncopyable; the separate
complete-name occurrence supports the same entity through the reviewed evidence.
Questions, answers, texts and splits are unchanged.

One full QA exceeds the historical passage-study binder's 2,048-token cap.
The existing atomic QA dataset already preserves this input and the model has no
sequence cap. The binder now allows an explicit `max_total_tokens=None`; its
historical default remains 2,048. Source capacity 512, question capacity 32,
span limit 16 and output limit 128 remain unchanged. Every replay binding passes.

## Live loss contract

`native_live_replay.py` consumes frozen observed candidate descriptors and frozen
teacher outputs. It never caches learned student states. The ordinary model
forward recomputes core observations, source scores, native guidance, endpoint
scores, confidence and mixed emission. Auxiliary gold-start candidate scoring
uses the production endpoint scorer; those candidates never enter the inference
bank. An absent gold payload is counted explicitly.

Each replay packet pins input history, source word addresses, supported starts,
output positions and payload labels. Unexpected fields, learned teacher tensors,
nonfinite teachers and misaligned histories are rejected. Teacher tensors and
observed packet files are fingerprinted. Labels select losses after inference.

Ordinary source supervision remains mean gold-start NLL over annotated examples.
Additional source preservation uses KL and capped margin only for a correct
teacher start. Conditional extent uses gold payload mass plus correct-teacher KL
and capped margin. Copy confidence uses correct-teacher Bernoulli KL, otherwise
gold copy-appropriate BCE. Emission uses correct-teacher token KL, otherwise gold
CE, on actual production mixed logits. Duplicate valid payloads share gold mass.

With replay enabled, ordinary CE is averaged per example and then across content
and boundary/EOS decisions. Replay emission has the same balance. Each of the four
additional replay components is summed over replay examples and divided by the
full annotated batch size, then the four components are averaged. This preserves
the replay-slot share without letting long answers dominate. `native_source_weight`
and `native_replay_weight` default to one and remain explicit recorded controls.
The no-replay loss path retains its existing behavior.

Annotated batches pass `native_source_targets` and `native_replay_targets` through
the ordinary trainer. The latter wraps each immutable packet with its batch row
and empty-output-prefix position. Exactly one annotated atomic QA per row is
required. The trainer records replay examples and weights in training state.
The common optimizer factory materializes modules and checks full parameter
coverage before constructing Muon; this preflight performs no optimizer step.

## Interpretation

Restoring modules alone is insufficient: both students retain 112/112 newer
controls, but raw historical retention is only 14/574 and 68/574. The native parent
has 574/574 and 570/574 respectively, with four inherited fresh-value failures in
seed 83. Accepted native references retain the controlled, native, owner and
layout gates but predate the reviewed legacy fit. Identity-route results are
reported separately, not mistaken for the historically evaluated native route.

Engineering verification and teacher coverage do not prove retention learning.
Any next fit needs a separately frozen budget, fixed sampling schedule and raw
retention gates. Accepted baselines are not replaced by these restored students.

The unchanged-gate oracle check fails: exact-proposal injection reproduces only
3/8 and 6/8 confidence-extreme diagnostic payloads respectively. This is a real
confidence failure, not permission to force the gate or insert training answers
into inference. The earlier oracle prerequisite therefore continues to block a
scientific fit. The selection of these eight cases per seed is explicitly a
diagnostic and does not estimate population accuracy.

The fresh-head tests also distinguish initialization from trained behavior:
zero endpoint output weights and zero copy-gate input weights initially block
those losses' core gradients while allowing their heads to learn. Separate
active-head tests verify all four paths to the core; the trained restored
checkpoints pass live finite-gradient preflight without changing their weights.
