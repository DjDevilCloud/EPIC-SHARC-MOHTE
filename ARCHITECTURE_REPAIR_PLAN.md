# Gated architecture repair

The objective is a QA learner that uses the shared EPIC-SHARC MOHTE
representations. Each gate has its own evidence and decision. Changes to
representation, optimizer, capacity and precision are separate comparisons.
The accepted controlled baselines and closed studies remain regression history.

## Gate 1: optimizer mechanics and initialization — implemented and verified

Production changes are limited to `muon_optim.py`, `model.py` and `train.py`.
All 105 pre-change source/test files were archived before editing.

The repaired hierarchical optimizer has version-2 state:

- Table updates preserve **elementwise first momentum** and use a nonnegative
  per-row second moment. Adam-style bias corrections apply. This retains the
  gradient's direction, including zero-mean rows. Its first-moment storage is
  now table-sized; only the second moment remains row-compressed. This is an
  explicit correctness/memory tradeoff.
- Nested cadence is measured in global optimizer calls. For each parameter,
  average gradients actually observed in its current window, update the
  window-gradient EMA and divide by `1 - product(beta)` before applying the
  group's update rule. A parameter without a current gradient still flushes a
  pending window at the boundary. An empty window performs no weight update.
  Accumulated windows, EMA state, decay products and group clocks resume exactly.
- Old unversioned hierarchical state is rejected before loading. Its signed
  row means cannot reconstruct elementwise momentum, and its stored EMA had a
  different meaning. Load old **model weights** with `--continue-checkpoint` for
  a fresh optimizer. Compatible version-2 state can use `--resume-checkpoint`.
  Existing model inference is unaffected by optimizer-state versioning.
  A requested resume with mismatched membership or malformed state also fails
  explicitly instead of advancing the run with silently reset moments.
- Enabled lazy attention modules are materialized before optimizer creation and
  preserve the owner's mode/device/dtype. Materialize before applying a partial
  trainability policy; the factory rejects ambiguous late allocation after
  partial freezing.
- Known dataset `seq_len` capacities and precision setup precede registration.
  Position-table resizing preserves its old dtype and frozen policy. Unknown
  later replacement is caught by the coverage guard rather than trained silently.
- The common factory checks missing, duplicate and stale parameter identities.
  The ordinary trainer checks again after forward and before stepping. Saved
  training state includes group membership, effective rates, roles, tiers and
  nested cadence. Applied nested settings are copied to the saved runtime config.
- Compressed table-variance state is recognized by the trainer's shape validator,
  allowing valid optimizer continuation instead of discarding its moments.

The numerical counterexamples were added as failing tests before the fix. The
first full-suite attempt caught a frozen-position-table regression introduced
by pre-sizing; it was corrected without relaxing the trainer-resume assertion.
The final full suite ran **521 tests: 519 passed, two skipped** on Python 3.12,
CPU float32, one thread. Focused checks include signed/zero-mean table updates,
quadratic learning, beta sensitivity, inactive boundaries, exact optimizer resume,
actual trainer checkpoint resume, all three optimizer factories, and lazy weight
updates. CUDA and mixed-precision behavior are not established by this gate.

Historical compatibility checks reproduced **646 source starts/shortlists per
seed and four sampled ordinary generation results per seed**. Protected V4/V5
artifacts and accepted checkpoint hashes remain unchanged. These checks are not
a new full raw-answer retention certificate.

Evidence: `review_artifacts/optimizer_initialization_repair_v1/`. The acceptance
record distinguishes optimizer-mechanics readiness from QA transfer. No new QA
study, optimizer ranking or capacity expansion belongs to Gate 1.

## Gate 2: a shared backbone representation contract

Before implementing another learner, specify and test these interfaces:

1. **Producer:** contextual token states after the selected torus/SHARC/hierarchy
   computation, before QA residual guidance and copying. Source supervision must
   have a live autograd path to that computation. Detached diagnostic tensors
   cannot be used as learning features.
2. **Addresses and summaries:** map the states to the existing observed source
   word ranges. Define completed phrase/line summaries and supported source
   lookahead explicitly. A state at a name may precede the predicate identifying
   that name; replacing all context with the name's causal state would silently
   lose evidence. Summaries may use later *observed source* tokens, never future
   answers. Source identity, contextual features and eviction must stay aligned.
3. **Request:** obtain persistent question features and question-word marker
   features from the same backbone/hierarchy stream. Fix their input-boundary
   lifecycle across full-prefix, chunked and incremental execution. Generated
   answer tokens cannot become source evidence or rewrite source identity.
4. **Consumers:** reuse the existing source projections, conditional endpoint,
   marker, copy-confidence and emission heads. Keep the bounded span/payload
   bank. Audit every ownership/relation head's input contract before conversion;
   no silent removal of the features that supported controlled tasks. The new
   path must not add another independent GRU semantic encoder.
5. **Training outputs:** expose live source/extent decisions through the ordinary
   model path. Define the correspondence between an annotated example and its
   input/output block; multiple QA blocks in one sequence must be supported
   explicitly or rejected at dataset preparation. Normalize losses by example
   and decision type, not by answer length. Keep actual inference proposals
   separate from gold-start auxiliary extent supervision.
6. **Cache rules:** cache input tokens/hierarchy tracks and genuinely frozen
   teacher outputs. Recompute student contextual states whenever their producer
   trains. Live and incremental inference caches must describe the same model
   state. Do not reuse old frozen-backbone student hidden caches.
7. **Compatibility:** save a versioned representation mode. Legacy checkpoints
   retain their old path. Conversion to an experimental backbone-driven path
   must record retained, discarded and newly initialized parameters explicitly.

Exit tests: supported source loss reaches the intended core/router/lattice
modules on representative examples; contextual interventions affect source
scores; the existing span machinery still represents reviewed payloads; future
answer substitution preserves earlier predictions; reload and ordinary entry
points agree; full-prefix and incremental candidate identities and greedy
outputs agree within the existing logit tolerances. No learning experiment
begins while these checks fail.

## Gate 3: demonstrate end-to-end learning

Freeze a small supported contrasting-question control and its budget before
training. Initially retain the tiny geometry to isolate the representation
wiring. Train through the core using the common factory. Measure each question
and both-question success; global assignment loss is not an accuracy metric.
Report source rank, shortlist, conditional extent, actual complete proposal,
raw answer and termination separately.

Require learning through the intended modules, ordinary greedy completion,
causal/reload parity, and functional replay on correct teacher decisions.
Gold supervises incorrect teacher decisions. A frozen parameter is not evidence
of retained behavior when its inputs change. Potentially eligible candidates
must run the actual raw retention suites: 58 reviewed legacy, 60 native replay,
288 controlled, 16 fresh values, both 36-case owner sets, all grammar sets and
32 layouts. Record inherited defects separately. The locked test stays closed.

Passing a tiny control establishes wiring and trainability, not broad transfer.
Only then freeze a new bounded passage-transfer study; do not extend or relabel
the closed V4/V5 adapter experiments.

## Gate 4: optimizers and capacity, separately

First declare which physical model modules own global/mid/local optimizer roles.
The current name-based tier heuristics remain a limitation; Gate 1 did not
replace them with a validated architecture ownership map.

Compare AdamW, Muon and repaired hierarchical optimization using identical
architecture, trainable membership, data, objectives and starting states.
Serialize actual group rates and clipping. Report presentations, global calls,
per-tier update counts and wall time: nested cadence changes actual update
counts. Test nested cadence separately from role-aware update rules.

Profile wider GPU models and real instantiated hierarchical/expert paths in a
separate capacity study. Verify module presence, usage and task gradients.
Parameter residency, sparsity and reduced precision need their own execution
checks; they are not automatically improvements in QA reasoning. Any additional
corpus training must exclude held-out source groups and related fragments.

Freeze final configurations and selection rules before comparing transfer.
Preserve the existing development, retention and one-time locked-test gates.
No optimizer improvement, raw-retention result or QA milestone is claimed by
passing only the initialization gate.
