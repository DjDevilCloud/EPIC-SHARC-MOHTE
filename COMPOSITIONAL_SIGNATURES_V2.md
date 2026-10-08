# Causal runtime signatures v2

Fresh models can opt into `--signature-representation compositional_v2`.
Legacy and v1 checkpoint contracts remain unchanged. V2 needs fresh training;
changing a saved v1 configuration to v2 would change its learned conditioning.

```text
python cli.py train --data <corpus> --save-dir <new-directory> --signature-representation compositional_v2 --no-token-superposition-training --no-speculative-decoding --hierarchy-vector-dtype float32 --no-hierarchical-precision-enabled
```

For inference select the saved checkpoint and ordinary decoding
(`--no-speculative-decoding`, beam size 1). FP32 and CUDA BF16 gradients are
verified. Optional quantization and compiled execution need their own workload
checks; no speed improvement is claimed.

## What changes

V1 compiles constituent properties for registered profiles, but a new whole-word
or partial-word profile becomes OTHER before reaching the bank. V2 constructs
properties directly from the token sequence, without consulting the word/line
signature catalog. Its feature state keeps counters, case, span role and an
incremental UTF-8 decoder; it does not retain a growing word or history string.

Eleven independent features describe span role, construction-unit category,
case, word length, vowel/digit counts, word position, line length, indentation,
the most recently observed character class, and whether a word is active.
Counts and positions use logarithmic buckets. Properties use two salted bounded
hash addresses in the existing sole learned component table. All features are
derived from tokens already consumed: no next token or future completed word
is consulted. New words receive these features without new catalog rows.

The lexical bank view retains distinct unit identity. Root hierarchy conditioning
combines lexical and runtime-property embeddings. The torus parent-context channel
uses runtime properties directly rather than a whole-parent-profile lookup.
Existing catalog IDs remain available to auxiliary targets, registry/family
channels and lattice identity/bias consumers. V2 therefore removes the catalog
bottleneck in these two main structural-conditioning channels; it does not yet
remove every catalog-dependent path in the architecture. Lattice bucket collisions
and the learned rule-selection problem still need independent assessment.

`PrismalWaveOutput.runtime_signature_state` carries request-local runtime state.
Ordinary generation feeds it into the next one-token transition. External
incremental callers must pass it back with `runtime_signature_state=...` along
with torus/lattice/token-memory states; omitting it at a nonzero position raises
an error. State advancement clones the small input state so branching a prefix
does not mutate another request. Full-prefix replay and incremental generation
are tested for token equality. Training computes the same causal features once
per supplied sequence. A truncated training window only has that window's context.

V2 internally replaces supplied numeric hierarchy vectors with runtime feature
addresses. The tokenizer and dataset format stay unchanged, and the old numeric
ID vectors do not define v2 structural conditioning. Role/boundary behavior is
derived from actual construction units. Token superposition, beam decoding and
multi-token speculative decoding are rejected in v2 pending explicit state
branch/replay integration. Verified greedy proposals remain optional and use
the existing conservative verifier; they do not correct a wrong neural choice.

The checkpoint loader also now sizes family restoration using family vocabulary
size rather than signature vocabulary size. Already larger legacy tables retain
their saved shapes; fresh small tables avoid unnecessary expansion on reload.

## Recombination control, 2026-10-08

The saved control compares v1 and v2 using identical common initial parameters,
tokenizer, training schedule and AdamW settings. Four colors and four fruits form
16 pairs. Each pair has two explicit rules: return its color or return its fruit.
Ten pairs supply 20 training records; two pairs supply four validation records;
four diagonal pairs supply eight test records. Every word occurs in training.
The tokenizer is fitted only on training records. The test pairs are evaluated
after both runs finish, and validation content CE alone selects checkpoints.
Both runs use 1,200 updates, batch 4, 240 presentations per training record,
seed 47, d_model 32, 512 shared component buckets, FP32 CPU and no proposals.

| Selected checkpoint | Training exact | Validation exact | Test exact | Test content CE |
| --- | ---: | ---: | ---: | ---: |
| v1, step 100 | 8/20 | 0/4 | 2/8 | 1.0048 |
| v2, step 200 | 20/20 | 1/4 | 4/8 | 0.8013 |

V2 answers all four unseen color queries correctly, but none of the four unseen
fruit queries. Its incorrect answers are short, valid learned words rather than
character loops in this control. Both runs eventually reach perfect teacher-forced
training accuracy, while validation deteriorates; extra presentations still
encourage memorization. Different selected steps follow the predeclared validation
criterion, not test results. This single-seed, eight-record result is encouraging
for color recombination but does not establish robust rule selection, general
language quality, or a causal contribution of each changed channel. More seeds
and a separate untouched task are needed before promoting v2 to the default.

Evidence and reproduction:

- `review_artifacts/run_runtime_recombination_control.py`
- `review_artifacts/runtime_recombination_control_20261008/manifest.json`
- `review_artifacts/runtime_recombination_control_20261008/results.json`
- Best-validation checkpoints and implementation source copies inside that folder.
- `tests/test_runtime_signatures.py`: novel/partial-word coverage, causal UTF-8
  frame parity, request isolation, full/incremental logits, generation replay
  parity, missing-state rejection, gradients and checkpoint preservation.

The final missing-state guard was added after the control; archived sources retain
the exact implementation used during training. It does not alter valid calls or
checkpoint weights.
