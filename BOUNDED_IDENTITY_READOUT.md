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

### Context-only relational selection

`identity_readout_binding="lexical_binding_v2"` replaces the active selection
score with position-aware context/query similarity. Learned softmax weights over
the two preceding and four recent eligible units preserve order. A shared
projection and cosine similarity prevent vector magnitude from becoming a
similarity shortcut. The query has a learned residual; the candidate's own
identity is excluded from its relational score. A learned bounded log temperature
controls attention sharpness. Copying still emits the observed token ID through
the ordinary probability mixture and recurrent generation path.

Confident adapted requests use relational attention; uncertain requests blend
with the independent readout; confidently retained requests reproduce the old
scores exactly. v2 initializes applicability off, so migration is neutral even
though relational temperature starts at one. Calibrate applicability from
training inputs before answer adaptation. v2 confidence must remain below one
so a finite initial classifier can select the exactly neutral route.
v1 checkpoint behavior and the
`independent_v1` default are unchanged.

The two-seed control improves the diagnosed wider set from 28/48 and 34/48 to
48/48 in both seeds. Subsequent frozen-weight challenges reach 96/96 crossed
assignments and 72/72 three-fact questions in both seeds. All seven old layouts
retain 48/48 with identical output token IDs. This adds six position weights over
v1, for 110,698 total parameters in this control. See
[the selection review](./review_artifacts/RELATIONAL_SELECTION_FIX_20261009.md)
for raw results, checkpoints, verification and remaining limits.

### Complete-word and native task-cue paths

`word_span_v3` groups lexical fragments into causal words and uses bounded
clause contexts plus the final question clause. Existing lexical embeddings and
character ngram properties share the existing bank; whole-profile registration
is not required. `word_span_v5` freezes this branch and adds input-only learned
task-cue routing, observed right context, learned source-prefix continuation,
and separate native query/gate/format residuals. In adapter-only mode the native
router stays frozen on reload; its explicit calibration phase enables it
temporarily. Both modes retain ordinary recurrence and token generation.

The matched two-seed control reaches 24/24 on all three tested longer-name groups
while the equally trained unit-window path fails. Native curriculum reaches 3/3
one-word answers in both seeds, then 4/6 and 6/6 training answers, while retaining
the longer-name gains and all old-layout outputs. Disjoint native QA is still
0/22. Inference composes input memory once per request, and disabled absolute
position features no longer grow an unused table. See
[the word-span and native review](./review_artifacts/WORD_SPANS_AND_NATIVE_CURRICULUM_20261009.md)
for checkpoints, raw evidence, parameter/buffer retention and remaining limits.

## Calibration and source/query experiments

`calibrate_task_route` trains a private classifier and commits only after its
training inputs meet both confidence thresholds. Failed calibration retains
the accepted weights and gradient flags. Always replay retained raw outputs:
training-input separation alone does not establish held-out routing stability.

`identity_readout_exclude_query_candidates` is an opt-in v4/v5 experiment. It
uses the existing final-clause question assumption and a request-cached source
mask, blended through the learned QA route so independent legacy behavior stays
neutral. The partition-only experiment did not improve accuracy. It is enabled
alongside ordered entity agreement in the newer controlled baseline. See
[the boundary and grammar control](review_artifacts/ANSWER_BOUNDARIES_AND_GRAMMAR_20261009.md).

## Ordered entity ownership baseline

`identity_readout_ordered_agreement` adds one zero-initialized, shared scalar for
exact contiguous agreement of complete source/query words. Existing semantic
and structural bank features remain in use; IDs serve only as identity keys.
This preserves word order and modifiers that pooled attention can lose. The
source/query partition prevents the query prefix from authorizing source copying.
Both features default off for checkpoint compatibility.

Training-only `retrieval_margin_loss` separates positive source-answer identity
groups from competing eligible source candidates before attention temperature.
Question candidates are ineligible. Deliberately ambiguous training names,
including reversed word orders, give this feature useful supervision while all
preexisting learned state remains frozen. Native relation scores are unchanged.
The scalar stays frozen during v5 native-adapter fitting, protecting retained
word binding. See
[the stable controlled baseline](review_artifacts/FUNCTIONAL_BINDING_BASELINE_20261009.md)
for both-seed retention, fresh entity combinations, reload/streaming verification,
recommended checkpoints and remaining native-QA limitations.

## Native source signatures and word-prefix paths

`--identity-readout-native-word-paths` enables an opt-in `word_span_v5`
extension. It composes source runtime properties, unit position within a word,
and source word length through the existing shared signature bank. A learned
projection adds these properties to native retrieval keys. Exact lexical IDs
serve as bounded word-prefix identity keys; they are not numeric features.

A source-word trie is constructed once per input memory. Request state retains
the current prefix node and the posterior over candidate word starts. Complete
and partial prefix mass supplies shared-bank evidence to learned native copy
and formatting residuals. At word boundaries, a learned scalar introduces the
expected next-word signature. These are soft scores, not forced completions;
every generated token still updates recurrent state.

New weights initialize to zero, preserving migration behavior. The optional
zero-initialized `native_word_start_scale` can be absent in an earlier path
checkpoint; missing learned source projections remain load errors. The flag
defaults off. Adapter fitting freezes original word binding, core weights,
registry observations and calibrated task classifiers.

The two-seed transfer candidates retain prior raw token IDs and learn eight
native answers plus ten value substitutions in three layouts. Untrained
whole-word substitutions score 24/24 and 23/24; unfamiliar fragmented values
score 0/6 and 3/6. Broader native QA remains 0/22. This establishes a useful
transfer improvement, not general QA or faster generation. See
[the native transfer report](review_artifacts/NATIVE_QA_TRANSFER_20261010.md).

### Conditioning verified word prefixes

`--identity-readout-native-conditional-prefix` requires native word paths and
defaults off. At a verified lexical prefix, it renormalizes the source-word
posterior over exactly compatible complete and partial words. An incompatible
or unsupported source prefix supplies zero evidence. The prior tensor is not
mutated, preserving independent request branches and differentiable training.

This separates uncertainty about the initial source selection from uncertainty
about whether the current observed prefix is a complete word. Previously, a
small root probability could suppress partial-word evidence after the prefix
had already identified a source word. The conditional posterior changes soft
evidence and learned scores; it does not mask or force generated tokens. It adds
no parameters and retains per-token recurrent updates.

Training-only source-start margin supervision is then evaluated separately
from native boundary fitting. The diagnosed retrieval failures and isolation
procedure are in [the frozen-source review](review_artifacts/NATIVE_SOURCE_DIAGNOSIS_20261010.md).
Final source and boundary controls retain earlier outputs and repair the old
development substitutions, but unseen name boundaries still fail. See
[the final control report](review_artifacts/NATIVE_SELECTION_AND_BOUNDARIES_20261010.md)
for both-seed scores, frozen traces, checkpoint scope, and the remaining gate.

### Verified continuation and categorical boundary decisions

`--identity-readout-native-span-boundaries` adds a canonical span boundary
channel. It shares the component bank and uses complete/partial word evidence,
verified prefix continuation, within-word support, causal output word position,
and separator identity, conditioned on the question. Lexical fragments use a
common marker; the boundary vector does not encode their individual identities
or shapes. Selection weights need not change.

The default `residual_v1` readout initializes to zero for neutral migration.
`--identity-readout-native-span-readout categorical_v2` predicts structural
tokens plus lexical mass, preserving individual lexical selection through a
learned copy/base gate. It remains inactive until teacher-feature fitting and
`commit_span_calibration`. Confidence-selected training positions must all be
correct; inference below `--identity-readout-native-span-confidence` (default
0.9) retains the old readout. Unsupported prefixes also retain it. Known inactive
native routes remain exactly neutral. Teacher features can be cached only with
frozen upstream state; upstream changes require renewed calibration/replay.

Both seeds now pass the two old held-name cohorts at 72/72, separate fresh names
at 36/36, and fresh repeated values at 24/24 while retaining previous token IDs.
The accepted scope is controlled native values and boundaries. Broader native
QA stays 0/22; variable answer lengths and new question types are not established.
See [the categorical boundary review](review_artifacts/CATEGORICAL_SPAN_BOUNDARIES_20261010.md)
for controls, calibration, checkpoints, verification, and limits.

### Observed source separators and endpoints

`--identity-readout-source-boundaries` requires native span boundaries and adds
a zero-initialized source head. Word state retains observed separators/case
controls; request state retains emitted separators after the last lexical unit.
Exact lexical suffix and separator agreement select observed source evidence.
The existing shared bank composes next-marker, clause/line-ending, within-word,
support, ambiguity and separator-offset properties. No future output or label
is consulted at inference, and punctuation alone does not force termination.

The default source readout `residual_v1` adds evidence to prior span logits.
`--identity-readout-source-boundary-readout categorical_v1` instead uses the
source categorical prediction when supported and confident, retaining the
previous span head otherwise. It requires native `categorical_v2` spans. This
avoids requiring source evidence to overcome a fixed output-position prior at
untrained lengths. Source selection and recurrent token updates remain intact.
Checkpoint configurations default this channel off for backward compatibility.

The isolated source-head control freezes every prior weight and registry
buffer. Both seeds retain all earlier task token IDs and pass 24/24 independent
four-to-seven-word continuation cases **when supplied the correct first word**.
Their autonomous raw scores are 21/24 and 23/24; source-start and lexical
continuation failures remain, including repetition on a short title in seed 47.
Broader native QA is 0/22 and 2/22. New question/source selection has been traced
separately and remains incomplete. These are boundary-stage references, not a
complete raw-QA acceptance. See
[the source-boundary repair report](review_artifacts/SOURCE_BOUNDARY_REPAIR_20261010.md).

### Source occurrence ownership and safe grammar adaptation

`--identity-readout-source-ownership` adds a neutral source-start correction and
an occurrence posterior that conditions on emitted lexical units. Continuation
advances across contiguous observed entries only after verifying the emitted
separators. Query entries cannot be proposed; unsupported prefixes supply no
cursor evidence. Repeated source positions remain distinct while shared next
token identities aggregate confidence. Copy/base mixing and recurrent updates
continue normally. The cursor scalar initializes to zero.

`--identity-readout-source-start-features span_roles_v2` preserves neighboring
parent-span words when a clause starts, plus shared-bank ending/line/count
properties. It avoids discarding the previous line's role at a title's start.
No whole-profile catalog or categorical ID-distance feature is introduced.

`--identity-readout-source-boundary-adapter` requires categorical source
boundaries. Confident grammar proposals can replace source logits; uncertainty
retains the validated source formatter. Its zero initialization preserves old
outputs. Adapter calibration evaluates its own supported proposals, while raw
generation and retained-task replay are separate mandatory checks.

Both seeds now pass 54/54 controlled length outputs, 36/36 trained grammar
outputs and 24/24 earlier independent length cases, retaining all prior
controlled token IDs. Final untrained names/wordings score 15/24 and 21/24;
broader QA scores 0/22 and 1/22, including a held database-answer regression.
This is a controlled baseline, not general QA acceptance. See
[the ownership repair report](review_artifacts/SOURCE_OWNERSHIP_REPAIR_20261010.md).

## Verified source-word posterior repair (2026-10-10)

The earlier 15/24 and 21/24 confirmation scores above describe the parent checkpoint.
The next repair separates selecting a source word from reading its lexical units.
`--identity-readout-word-path-readout posterior_v2` carries word posterior mass
onto that occurrence's verified next fragment, rather than adding a bias to all
source units. It preserves uncertainty across words and falls back when unsupported.
`bias_v1` remains the default for compatibility and controlled comparison.

`--identity-readout-source-start-features context_roles_v3` omits candidate spelling,
fragment count, next value identity and span length from the start correction.
Preceding complete words and clause/line roles remain. Source-boundary features
also retain supported cursor ownership over repeated suffix occurrences.

After format variation and boundary calibration with the final selection path,
both seeds pass 24/24 previously failing development cases and 24/24 separately
reserved fragmented-name/length cases. This establishes controlled span transfer;
general native QA and generation speed require separate acceptance.
Final acceptance retains 492 prior native output sequences and 840 controlled
sequences per seed. Broader QA scores 1/22 and 2/22; the database regression is
recovered, but broader transfer remains a separate blocker to unrestricted scaling.
See [the verified continuation repair](review_artifacts/REMAINING_ARCHITECTURE_REPAIR_20261010.md)
and its executable retained-task gate for checkpoint provenance and final diagnostics.
