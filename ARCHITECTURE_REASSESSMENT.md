# Architecture reassessment after the native-QA adapter studies

The completed studies establish failure of the tested native retrieval adapters.
They do not establish failure of EPIC-SHARC MOHTE trained as a unified model.
The immediate priority is to repair the training contract and connect QA
selection to the architecture's actual contextual states. Further standalone
GRU variants are not justified by the current evidence.

This reassessment used code inspection, both final V5 seeds, eight full-model
gradient probes, hidden-state interventions and isolated optimizer probes.
No model optimizer steps, production edits or locked-test predictions were made.
The optimizer probes updated toy tensors only. Accepted checkpoints and the
closed V3/V4/V5 studies remain unchanged.

## What we actually exercised

| Component | Recent QA studies |
|---|---|
| Torus core | Flat `PrismalTorusCore`, 2×2×2, width 32; frozen during fitting |
| SHARC emitter router | Instantiated; runs in ordinary full-model forward; frozen during fitting |
| Signature lattice | Instantiated; runs in ordinary full-model forward; frozen during fitting |
| Hierarchical parameter nest | Absent: `token_hierarchy=None` |
| Recursive expert tree | Absent: both `use_hmote` and `use_recursive_hmoe` false |
| Family specialists / hierarchical MoT experts | No instantiated hierarchical nest to host these paths |
| `n_layers=1` | Saved field; torus mode constructs an empty `blocks` list. It is not evidence of a one-layer dense backbone |
| QA source learner | Separate lexical GRUs, bidirectional context GRU, scoring/prior scaling and optional question attention |
| Source-training parameters | 56,387 semantic-alignment parameters, versus 524,857 total saved parameters |
| Optimizer | Study explicitly constructs AdamW; it bypasses `train._build_optimizer` |
| Architecture auxiliary losses | Disabled; the custom source loop does not call the ordinary joint loss |
| Learned-state caches | Valid for the frozen-backbone study; invalid as student caches once the backbone trains |

`use_mixture_of_torus=true`, `per_family_torus_enabled=true`, and leaf-related
settings in a saved file do not prove those specialists exist in the running
model. The constructor selects the flat core here. Within the hierarchical nest,
the MoT path is controlled by `use_hmote`. The feature manifest must report actual
modules and exercised paths, not just booleans.

The screenshot also selects `adamw` and leaves Nested learning unchecked.
Muon Nesterov being checked does not activate Muon. The current factory returns
plain AdamW immediately for that selection. Nested cadence is implemented only
by the hierarchical optimizer. Visible controls should identify when they are
inactive.

## Where the learning path disconnects

The ordinary forward executes the torus/SHARC path, then invokes native memory.
Native source scoring constructs its own source/question representations from
construction embeddings. Backbone hidden states enter later through residual
guidance and fallback token logits; they do not determine the source scores.

In both seeds, replacing the native reader's supplied hidden states with zeros
versus large random values left source log probabilities **bitwise identical**.
Live supervised source loss had **no autograd connection** to torus, router or
signature-lattice parameters on either sampled question type. It did connect to
the separate native reader and shared lexical embeddings.

For seed 47's sampled native replay question, token-loss gradient norms were:

| Route | Torus | SHARC router | Signature lattice |
|---|---:|---:|---:|
| Native copy readout | 4.60e-12 | 7.44e-13 | 1.11e-12 |
| Backbone readout | 5.81 | 0.836 | 1.57 |

These are diagnostic gradients with parameters temporarily enabled in memory,
not training results. The native token loss was approximately 0.000075; the
backbone token loss was 3.158. The copy branch largely solves this example while
providing almost no task gradient to the core. Suppression varies by example;
this is not a universal constant or a claim that the backbone never influences
generation. Full measurements for both seeds are in the audit artifacts.

Merely unfreezing the backbone or selecting a different optimizer cannot create
the missing source-loss dependency. The representation consumed by the source
and extent heads has to change.

## Optimizer and initialization defects to repair first

1. **Hierarchical table updates can ascend.** Its row-wise path multiplies the
   elementwise gradient by a signed row-mean momentum scale. At zero weights and
   gradient `[-1,-1]`, one step increases a linear loss by 0.008944. Gradient
   `[1,-1]` produces no update. This is a correctness defect in the existing
   implementation, not evidence against hierarchical optimization as an idea.
   It did not cause the completed AdamW study failures.
2. **Nested EMA does not influence updates in the probe.** The code updates
   `nested_grad_ema`, then returns the unsmoothed window gradient. Changing the
   beta from 0 to 0.99 changes stored EMA values but produces identical weights.
   Define the intended nested-gradient rule and either apply its EMA with a
   specified startup/bias convention, or stop advertising the beta as an active
   learning control. Existing cadence/resume smoke coverage does not check this.
3. **Lazy attention can miss the optimizer.** In the fresh-model normal-factory
   probe, lattice attention did not exist when the optimizer was created. The
   first forward added seven trainable tensors (657 parameters in the width-16
   fixture), none registered with the optimizer. Capacity locking did not stop
   this. Reloaded V5 models already have lattice tensors, so this finding does
   not establish an omission in those particular checkpoints. It is a fresh
   training initialization defect that needs a regression test.
4. **Hierarchy groups use naming heuristics.** Tier assignment uses substrings
   and dotted-name depth. All recent source-only trainable parameters land in
   the `mid` tier. Selecting hierarchical optimization for that same subset
   would still leave the actual torus hierarchy untrained. Group roles and
   physical hierarchy levels need explicit, inspectable ownership.

The effective rates also differ by role. In the source-only factory audit,
hierarchical matrix groups receive 0.0076, scalar groups 0.00035, and both update
every fourth global step with nested learning enabled. A single displayed base
LR cannot describe this configuration. Serialize effective group rates,
clipping, cadence, membership and optimizer-state version.

## Repair sequence and verification gates

1. **Make existing optimization trustworthy.** Repair the table rule with signed,
   mixed-sign and zero-mean gradient tests; specify and test nested EMA behavior;
   test inactive parameters and resume across a partially accumulated window.
   Version incompatible optimizer state explicitly. Materialize all enabled
   lazy modules and finalize shapes before optimizer construction. Check every
   trainable tensor's identity is registered exactly once before and after the
   first forward. Route QA and normal training through the same optimizer
   factory. Test Muon and hierarchical modes in float32 first; hierarchical
   optimization does not require low-precision execution.
2. **Define one backbone representation contract.** Retain bounded observed
   source locations/payloads and the existing span heads. Obtain source-word
   features and persistent question features from the actual torus/SHARC
   contextual outputs and shared signature hierarchy, before QA guidance. Use
   existing projections/heads to select and emit spans. Introduce a versioned
   representation mode so historical checkpoints retain their behavior. The
   new candidate should not need another independent semantic GRU encoder.
3. **Preserve observed context and causality.** A word's causal state alone may
   precede the fact that identifies it, such as a name before its predicate.
   Specify how completed phrase/line summaries and supported source lookahead
   contribute to its key using already observed backbone/hierarchy states.
   Do not silently discard the current ownership, boundary or lookahead
   information. Align token/word addresses under full-prefix and incremental
   execution; evict features and locations together. Questions and generated
   answers must not rewrite source identity or introduce future-answer features.
4. **Move supervision into the live model path.** Source, conditional extent,
   payload and token losses must consume the same representations used by
   ordinary inference. Cache input tokens and frozen teacher outputs; recompute
   student states whenever their producer trains. Require source-loss gradients
   to reach the intended torus/router/lattice modules across representative
   cases, rather than requiring every sparse expert to activate on every case.
   Test contextual-state interventions and both questions per contrasting
   passage. Keep oracle proposals evaluator-only and raw generation separate.
5. **Prove a small end-to-end learning control, then compare optimizers.** Use
   the existing tiny geometry initially to isolate wiring correctness. Require
   a supported contrasting task to learn through the core, causal/reload parity
   and correct raw termination. Then compare AdamW, Muon and the repaired
   hierarchical optimizer on identical parameters, data, losses and starting
   states. Record global steps, actual per-tier updates, token presentations,
   clipping, gradient conflicts and wall time. Nested cadence is a separate
   comparison; it changes actual update counts. Do not conflate optimizer,
   geometry, dataset and precision changes in one claimed improvement.
6. **Profile and validate the intended architecture.** Before a larger QA run,
   establish a GPU memory/time budget for a wider model and a real hierarchical
   nest/expert configuration. A width-128 candidate is a profiling target, not
   yet a validated final configuration. Verify that configured specialists
   exist, receive relevant gradients and execute in inference. Treat residency,
   sparsity and reduced precision as capacity/efficiency options whose parity
   must be checked, not automatic reasoning improvements. A larger fresh model
   needs an explicit training lineage; the 128-QA adapter study is not a
   substitute for backbone learning. Exclude held-out source groups and related
   documents from any new corpus training as well as QA fitting.

Only after these gates should a new bounded transfer study be frozen. Keep the
existing supported-start/extent/proposal/raw/termination ledger and all raw
retention requirements. Preserve accepted controlled baselines as regression
references; do not force a new architecture to reuse incompatible frozen hidden
states. The locked test remains reserved for one final qualified evaluation.

## Evidence and reproducibility

Artifacts live in `review_artifacts/architecture_reassessment_20261024/`:

- `audit.py`: both-seed module, optimizer-group, hidden intervention and gradient probes.
- `optimizer_probes.json`: reproducible table-direction and nested-beta counterexamples.
- `seed47.json` / `seed83.json`: actual module types and gradient measurements.
- `coverage_probe.py` and `lazy_optimizer_coverage.json`: fresh lattice coverage defect.
- `audit.json`: production source hashes, scope and zero model-update count.

Run with Python 3.12, `-X utf8 -B`, CPU float32 and one thread. Outputs use
exclusive creation. To repeat the full audit, copy `audit.py` into a new sibling
folder under `review_artifacts` and run that copy. The coverage probe accepts a
new explicit output path. Neither command trains a model. Historical V5 hashes
are checked before and after the main audit.
