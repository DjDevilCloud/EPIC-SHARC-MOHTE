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

This establishes retrieval and rule selection for one-word answers on a fixed
template with known lexical values. It does not establish novel-word copying,
longer answers, varied field order, broad QA, factual knowledge, or recovery from
arbitrary bad prefixes. The 16-slot cache can evict needed early units in longer
questions. The learned gate is not calibrated confidence. The implementation
adds attention/projection work and still computes vocabulary logits; CPU training
in this control took roughly 50–51 seconds with the readout versus 40–42 without.
No generation-speed gain is claimed, and optional quantization/compiled execution
remain unverified for this new head. The next quality control should vary field
order, query placement and answer length on a newly reserved split.

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
