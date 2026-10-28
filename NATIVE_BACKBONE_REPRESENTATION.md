# Native QA from shared backbone states

`native_question_memory_representation="backbone_context_v1"` is an experimental,
saved representation mode. It uses the existing native span bank and readout
heads with contextual states from the ordinary torus/SHARC/hierarchy forward.
It creates no independent word, context or question GRU and no semantic encoder.
`legacy_gru_v1` remains the default for existing checkpoints.

## Representation and causal state

Each completed lexical word stores its last observed backbone state, before QA
guidance. Source-word addresses, ranges and vectors evict together at the existing
source capacity. Question vectors use the existing question capacity. At a
completed source line, its observed boundary state supplies the line summary;
the last source word closes an unterminated line at the question header. This
retains predicates following names and initials. Clause boundaries still limit
local lookahead. Numeric segment/line addresses are masks, not neural features.

The request is the backbone state at EOI, or the last question-word state when
an EOI is absent. Answer tokens cannot update source identities, contextual
evidence or the request. Inference clones the small retained vectors so a view
cannot pin the entire prefill tensor. Training retains the live graph.

Existing projections form keys from word/line states and values from word states.
Existing relation, source-role, document-context, marker and endpoint heads
consume the contextual features. Antecedent events retain bounded causal prefix
states; the owner update/matching arithmetic is shared with the legacy path.
An event's features are truncated at the queried source occurrence, preserving
the ownership chronology. Exact lexical identity remains an observed matching
cue when explicitly enabled; it is not a substitute encoder.

Representation changes do change the distributions seen by trained heads.
Compatibility of shapes and head presence is not a raw-retention result. Any
checkpoint conversion must explicitly record copied, removed and new weights.

## Cache contract

Native request state is bound to its model producer and revision. The common
optimizer factory invalidates it after a step. Weight loading, mode changes and
root device/dtype conversion also invalidate it. A stale carried request raises
an error and requires a new prefill. Advanced callers that mutate weights outside
these APIs must call `model.invalidate_native_context()`.

Student training batches start from their observed inputs and recompute features.
No learned student state is cached across optimizer updates. Frozen teacher
outputs can be cached separately. Carried state is not interchangeable between
models, representation modes or inference routes.

## Live training interface

`forward(..., return_native_readouts=True)` exposes live plans separately from
detached diagnostics. Labels do not enter source scoring or proposal creation.
`compute_loss(..., native_source_targets=..., native_source_weight=1.0)` adds
per-question supported-start mass loss after ordinary inference has built plans.
Each target specifies `row`, `position` and unique supported `starts`; the
position must identify an empty-prefix readout. Duplicate QA-block targets,
invalid positions and empty/out-of-range labels fail explicitly.

The ordinary trainer and evaluation helpers accept an annotated batch:

```python
{
    "tensors": existing_nine_tensor_collated_batch,
    "native_source_targets": [{"row": 0, "position": boo_input_position,
                               "starts": reviewed_supported_start_indices}],
    "native_source_weight": 1.0,
}
```

Unannotated tuple batches retain their existing behavior. FST training-prefix
insertion and token superposition are rejected for explicitly addressed targets.
Saved training state records the supervision contract, presented example count
and weights. Candidate creation stays answer-blind; source labels do not inject
gold proposals. Conditional extent, proposal and token measurements remain
separate. Identity-span auxiliary supervision is rejected in the new mode so
training cannot silently supervise a bypassed reader.

New mode requires native memory, spans and a saved native route. Disable legacy
semantic-encoder flags explicitly; enabling both representations is rejected.
The Python API and saved-checkpoint inference path support the mode. A general
corpus annotation importer and GUI configuration are not claimed by this gate.

## Evidence and limits

Gate 1 is sealed under `review_artifacts/optimizer_initialization_repair_v1`.
Gate 2 evidence is under `review_artifacts/backbone_qa_representation_v1`.
The final CPU float32 suite ran 536 tests: 534 passed and two skipped. Checks
cover live source gradients into torus, router, signature lattice and shared
hierarchy; normal-trainer supervision; full-prefix/incremental execution;
reload; canonical generation entrypoints; future-answer substitution; cache
revision rejection; source eviction and causal owner scopes.

Both historical seeds retain their 646 recorded source decisions/shortlists and
four sampled ordinary generation results. All 211 reviewed train/development
references per tokenizer pass representability and conditional decoder mechanics.
That decoder check uses an explicitly high-confidence fixture and an injected
answer proposal; it is not learned selection, learned copy confidence or raw QA
accuracy. Gradient fixtures use width 16; historical inference checks use the
preserved width-32 checkpoints. No locked-test predictions or new QA fitting
belong to this engineering gate.

These results establish integration and execution, not document transfer.
The next gate is a separately frozen, small end-to-end learning control using
the common optimizer factory. Full raw retention and broader transfer still
require their declared checks. Optimizer ranking, larger geometry, real expert
nests, CUDA and reduced precision remain separate studies.
