# Bounded identity readout and boundary controls

The strongest tested improvement is `compositional_v2` plus
`--bounded-identity-readout`. It solves both color and fruit selection on the
reserved combinations in two initializations. V3's structural changes alone
do not show a reliable quality improvement, so they remain experimental.

## What changed

The optional readout retains at most 16 lexical construction units from the input
span, together with their observed runtime properties. Formatting controls are
not candidates. It never stores future answer tokens. New input spans reset the
cache; ordinary output transitions keep the observed input available. State is
request-local, bounded, cloned when advanced, and returned as
`PrismalWaveOutput.bounded_identity_state`.

A learned query uses the final hidden state **after the torus recurrence**, so
the question can guide which value to retrieve. Keys combine lexical identity
and runtime properties using the existing shared component bank, followed by a
small learned projection. All retained candidates participate in soft attention
during training. Their token probabilities are combined with the normal
vocabulary probabilities using a learned sigmoid gate. Empty input memory leaves
the original vocabulary logits unchanged. No color/fruit names, selector mapping,
correct-answer filter or handcrafted semantic classifier exists in this module.

At d_model 32 the two projections and gate add **2,081 parameters**:
83,491 baseline parameters become 85,572 (about 2.5% more). There is no additional
signature embedding table. The identity channel retrieves actual observed unit
IDs; the structural channel still supplies properties of unfamiliar and partial
words. Every generated token continues to update the torus normally.

This readout is distinct from the older `TokenMemoryCrossAttention` path. Code
review found that the older path detaches key/value projection outputs on append;
the default top-1 softmax also has no selection-weight gradient. In that mode its
retrieval projections are not trained by the standard differentiable read path.
The old module was disabled in these controls and was not changed. This finding
does not establish it as a cause of the earlier no-memory repetition failures.

## Controlled result

Six colors and six fruits form 36 pairs. Each pair has a color query and a fruit
query. There are 48 training prompts, 12 validation prompts and 12 test prompts.
Test pairs use a different split from the prior boundary experiment. Every word
occurs in training; held-out combinations do not. Both initializations (47 and 83)
use fresh weights, train-only tokenizer fitting, identical common initial
parameters and the same shuffled schedule. No old weights or stored proposal
answers are transferred. All four runs finish training before test evaluation.
Validation **answer-token CE** selects checkpoints. EOS and formatting tokens are
excluded from this score; normal training still supervises the required output
boundaries and termination.

Each run uses 720 updates, batch 4, 60 presentations per training record, AdamW
lr 0.003, d_model 32, 512 component buckets, FP32 CPU, and no proposals.

| Model / seed | Selected step | Training exact | Validation exact | Test exact | Test answer CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 baseline / 47 | 600 | 48/48 | 5/12 | 7/12 | 1.94517 |
| v2 + readout / 47 | 720 | 48/48 | 12/12 | 12/12 | 0.00050 |
| v2 baseline / 83 | 420 | 30/48 | 5/12 | 6/12 | 1.49377 |
| v2 + readout / 83 | 720 | 48/48 | 12/12 | 12/12 | 0.00080 |

Each readout model answers all six color and all six fruit test queries correctly,
and the paired selector changes give the correct different answers. For example,
`red plum. color?` returns `red`; `red plum. fruit?` returns `plum`.

The two seeds share the same 12 test prompts; they are two initializations,
not 24 independent reserved prompts. Validation selection chooses different
training steps for the variants. The baseline's final training accuracy is higher
than the selected early seed-83 checkpoint; its selected result is reported
without choosing a better test checkpoint afterward.

The saved readout models reproduce the same generated IDs after reload. Disabling
the readout on those same weights reduces exact test answers from **12/12 to 1/12**
in each seed. This confirms dependence on the learned retrieval path; it is not
an independently trained baseline. Full-sequence and incremental execution agree
in argmax at all positions on two trained-checkpoint probes per seed, with maximum
logit difference approximately 0.000004. Evaluation leaves registered buffers
unchanged. Regressions also verify nonzero query/key/gate gradients, bounded
eviction, empty-memory fallback, request isolation, causal future-answer exclusion,
checkpoint equality, cached/replayed generation equality and CUDA BF16 gradients.

## Structural v3 experiment

V2 logarithmic position buckets merge words one and two. It clears counters at
boundaries before showing completed summaries. V3 preserves exact word positions
through 16 (bucketed tail afterward), completed word length/vowel/digit summaries,
and completed line length/word-count/indentation summaries across boundaries.
It uses the same parameter table and no full history strings. V2 semantics are
preserved under their existing representation name.

A separate two-seed control with 48 training and 12 test prompts produced:

| Seed | v2 test exact | v3 test exact |
| --- | ---: | ---: |
| 47 | 6/12 | 7/12 |
| 83 | 8/12 | 6/12 |

These structural repairs retain useful information but are not a demonstrated
overall quality gain. The later readout control uses a different split and v2,
so its improvement should be judged against its own matched baselines above.
The original boundary-control selection metric excluded codec controls but still
included EOS. Post-selection `answer_content_scores.json` separately excludes
EOS; none of its checkpoint choices were changed afterward.

## Use and limits

Fresh training example:

```text
python cli.py train --data <corpus> --save-dir <new-directory> --signature-representation compositional_v2 --bounded-identity-readout --identity-readout-capacity 16 --no-token-superposition-training --no-speculative-decoding --hierarchy-vector-dtype float32 --no-hierarchical-precision-enabled
```

Use the saved checkpoint with ordinary decoding (beam size 1, speculative decoding
disabled). External incremental callers must carry both `runtime_signature_state`
and `bounded_identity_state`, along with the recurrent/cache states. Missing
identity state at a nonzero position raises an error. The flags are opt-in;
existing checkpoints load their existing architecture. New readout weights require
training rather than flipping an inference setting.

The initial control establishes retrieval and rule selection for one-word answers
on a fixed template with known lexical values. It did not establish novel-word
copying, longer answers, varied field order, broad QA, factual knowledge, or
recovery from arbitrary bad prefixes. Later answer-form controls are described below. The 16-slot cache can evict needed early units in longer
questions. The learned gate is not calibrated confidence. The implementation
adds attention/projection work and still computes vocabulary logits; CPU training
in this control took roughly 50–51 seconds with the readout versus 40–42 without.
No generation-speed gain is claimed, and optional quantization/compiled execution
remain unverified for this new head. The subsequent format controls vary field order, query placement and answer
length on separately reserved combinations.

Evidence and reproducible scripts:

- `review_artifacts/runtime_boundary_control_20261008/`: v2/v3 split, curves,
  checkpoints, raw outputs and post-selection answer scoring.
- `review_artifacts/identity_readout_control_20261008/`: matched split, curves,
  best-validation checkpoints, raw outputs, initialization hashes, implementation
  copies, `results.json` and same-weight `verification.json`.
- `review_artifacts/run_runtime_boundary_control.py`
- `review_artifacts/run_identity_readout_control.py`
- `review_artifacts/verify_identity_readout_control.py`
- `tests/test_runtime_signatures.py`

## Answer forms and formatting diagnostics

The later six-layout control trains four answer rules: color, fruit, pair, and a
canonical nine-word sentence. The original mixture reaches 45/48 and 47/48
reserved-value exact answers across two seeds, including 12/12 sentence answers
in each. However, a four-space stress layout reduces exactness to 12/48 and 8/48.
This is bounded rule learning, not general language understanding.

Official evaluation now returns `lexical_ce_loss` and `surface_ce_loss`, their
token accuracies and supervised counts, plus `case`, `space`, `punctuation` and
`ending` subgroups. Surface includes all nonlexical construction units. The
original CE remains batch-averaged; the new class metrics are token-weighted and
respect the existing supervision masks. Trainer progress prints lexical/surface
CE and its returned metrics retain these fields under `val_` prefixes. Missing
groups are omitted. Check raw generated IDs and uncleaned decoding separately;
teacher-forced scores alone do not guarantee the right answer form.

`--identity-readout-rule preserve_structure_v1` is an experimental alternative.
It redistributes lexical mass while preserving each nonlexical base probability.
It does not guarantee that a structural token remains the argmax. Its matched
control is weaker overall (35/48 and 37/48 versus 45/48 and 47/48), so retain the
default `mixture_v1`. It adds a tokenizer-derived Boolean buffer, no learned rows.

`--no-absolute-position-embeddings` is a separate fresh-training ablation. The
checkpoint records `use_absolute_position_embeddings=False`; ordinary inference
and resume preserve that setting. Old checkpoints default to `True`. Disabling
positions retains recurrent order and runtime hierarchy properties, and retains
the existing position table for compatible parameter shapes. It does not skip
whitespace tokens, recurrence, or cache updates, and is not a speed optimization.

Full manifests, checkpoint comparisons, reserved formatting results, first-error
probabilities and untrimmed text are documented in
[the answer-form and whitespace review](./review_artifacts/ANSWER_FORM_AND_WHITESPACE_REVIEW_20261008.md).

With the same six-layout training and a separately reserved split, disabling
absolute positions improves five-space exactness from 5/48 to 39/48 (seed 47)
and from 12/48 to 48/48 (seed 83). Both disabled-position models answer 48/48
held-out value combinations correctly. This is the recommended next fresh
control configuration with v2/readout and `mixture_v1`, not a default change or
an inference toggle for old weights. Pipe/multiline failures persist in seed 83,
including a repeating output; uppercase pair answers regress in that seed.
Keep those stress checks before scaling and do not claim arbitrary formatting
robustness or that repetition has been solved.

## Fallback-byte candidate repair

The subsequent pipe diagnostic isolated a correctness defect: `|` is encoded as
`<BYTE:7c>`, and the old readout admitted all bytes as lexical content. In the
failing seed, changing only the separator to a pipe reduced exactness from 48/48
to 12/48. Newline, indentation and query-tab changes alone remained 48/48.

Fresh models now default to `--identity-readout-candidate-policy lexical_bytes_v2`.
It excludes nonlexical ASCII byte fallbacks from copying; letters, digits and
partial non-ASCII UTF-8 bytes remain eligible. Stored kinds, IDs and reversible
encoding remain unchanged. Evaluation classifies ASCII byte punctuation as
surface/punctuation loss. Configurations missing the new field load with the
original `all_bytes_v1` policy for reproducibility. Use a corrected bundle or an
explicitly recorded new policy to apply the repair to existing weights.

On the same weights, pipe-only and combined pipe/multiline answers become 48/48
in both seeds, with no pipe tokens generated and all combined outputs reaching
EOS. Corrected bundles retain bitwise-identical learned parameters and reproduce
the filtered diagnostic after reload. Five-space and uppercase failures remain
unchanged. See [the pipe root-cause review](./review_artifacts/PIPE_REPETITION_ROOT_CAUSE_20261008.md)
for traces, corrected checkpoints and limitations.

## Layout coverage continuation

The remaining five-space and uppercase failures were investigated with frozen
weights and a budget-matched continuation. Editing only retrieval-key layout
features did not repair them. The incorrect first retrieval can make a pair end
as a one-word answer; teacher-forcing its correct first value restores the
required separator and second value. Original training had no UPPER input controls.

Ten additional presentations per original training record, with single-uppercase
field labels and space runs 1/2/4/6/8 across varied layouts, produce 48/48 exact
answers in both seeds on all seven tested layouts. These include untrained
5/7/9-space runs and the untrained combination of both uppercase labels. The
budget-matched old-layout continuation still fails longer spacing, and seed 83's
best validation checkpoint is its initial checkpoint with uppercase failures
unchanged. No vocabulary, model parameters, tokenizer rewriting or generation
penalties were added for this experiment.

The preferred follow-up checkpoints are the varied-coverage models in
`review_artifacts/layout_coverage_control_20261008/`. See
[the layout-selection review](./review_artifacts/LAYOUT_SELECTION_AND_COVERAGE_REVIEW_20261008.md)
for exact counts, validation selection, reload/parity checks and scope limits.
Semantics-preserving format coverage is supported for the controlled QA schema;
this is not evidence of arbitrary-format or general-language reliability.

## Native QA and long-cache follow-up

The one-seed native-data pilot uses complete targets from ReasoningOff, DiverseQA,
SFT-Code and short WikiHow-derived Cosmopedia records. It identified and repaired
Cosmopedia prompt/text target placement, DiverseQA context/question/answer
adaptation, and a lazy signature-cache scaling defect that attenuated writes
after roughly 171 updates at decay 0.85. Periodic rebasing now implements the
intended recurrence through long prefixes and streamed state without a per-update
CUDA scalar read. It changes no learned parameter shapes.

The real-data comparison still yields 0/16 exact held-out answers for both
canonical and format-varied training. Validation selects early checkpoints as
training loss falls and validation later worsens. Full/incremental execution
agrees on 571/719-step trained probes with finite cache scales, so this result
cannot be described as an unfixed cache-scale collapse. Both readout and
vocabulary paths can still repeat on these tasks.

See [the real-QA and cache review](./review_artifacts/REAL_QA_DATA_AND_CACHE_REVIEW_20261008.md)
for source audits, corrected ingestion, numerical proofs, full-target controls,
raw outputs and scope limits. The toy format-robustness results above do not
establish broader native-QA competence.

## Optional lexical binding and retained-state adaptation

`identity_readout_binding="lexical_binding_v1"` adds a bounded, input-only
context window to each candidate and an explicit recent-input query. It composes
existing lexical embeddings with the shared token bank. The default windows are
two preceding eligible units per candidate and four recent input units for the
query; these are **unit windows**, not a full syntactic or question parser.
The learned span score links a value to nearby evidence instead of relying only
on its position. All torus transitions and ordinary token verification remain.

A structural answer-form residual and gate delta use runtime properties, plus
exact control-marker identity. They exclude the current lexical value's identity.
This lets a learned answer boundary transfer to values with similar structural
properties. The residual addresses only nonlexical output units; it does not
force EOS or copy a complete answer outside normal generation.

A learned applicability gate multiplies all additions. At the default confidence
of .99, confidently retained requests receive exactly zero additions, confidently
adapted requests receive the complete path, and uncertain requests blend the
paths. The gate is trained from input features. In the control, applicability is
calibrated on old/new **training-input** route labels, then frozen; answer labels
are not passed to retrieval or routing at generation time.

New query/gate/format deltas and the span scale start at zero, preserving the old
function at migration. Old checkpoints default to `independent_v1`. Surface token
addresses are serialized with the configuration and must be bound through
`prepare_capacity_for_tokenizer` before creating the optimizer. Normal checkpoint
loading supports the new mode. Do not change modes/windows on an active request;
restart from its input prefix so the neighbor cache has the correct schema.

Use `model.freeze_binding_backbone()` for retained-checkpoint adaptation, or set
`binding_adapter_training=True` (`--binding-adapter-training`) with the new mode.
This freezes original parameters **and registry observation updates**. Merely
setting `requires_grad=False` does not freeze family activity, promotion or active
masks, which participate in forward features. The adapter-training flag survives
checkpoint reload. To retain the calibrated router during further fitting, also
freeze `model.bounded_identity_readout.applicability` after calibration/loading.

The production control adds 10,210 parameters to the migrated 100,482-parameter
base. Both seeds score 8/8 on reordered training bindings and 8/8 on held colors,
while preserving all 48/48 answers and their token IDs on every old layout.
A fresh 48-record test using four additional objects and two additional colors
scores 28/48 and 34/48; all terminate, and remaining failures select the wrong
value. This remains a controlled grammar/known-vocabulary result. Longer or
question-first prompts, unfamiliar multi-unit entities, and general native QA
are not established. No end-to-end speed improvement is claimed.

See [the binding and registry review](./review_artifacts/BINDING_AND_REGISTRY_RETENTION_20261009.md)
for checkpoints, comparisons, frozen-state evidence and limitations.
