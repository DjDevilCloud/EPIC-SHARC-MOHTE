# Shared compositional signatures v1

This opt-in representation shares one learned component table across registry,
SHARC router, parent, family, level/relation, hierarchy conditioning, signature
neighborhood guidance and lattice biases. Small consumer projections remain.
Legacy checkpoints keep their original representation; select the new mode for
fresh fitting/training.

```text
python cli.py train --data <corpus> --save-dir <new-directory> --signature-representation compositional_v1 --verified-signature-spans
```

The first integration control used float32 and a small model. For that precision
profile, add `--no-hierarchical-precision-enabled`. CUDA BF16 autocast has a
finite forward/backward regression; other optional quantization backends still
need workload-specific measurement.

## Representation

`signature_bank.py` parses constituent properties independently. Word, line and
unit properties reuse feature addresses; count/length/indentation attributes
use logarithmic buckets rather than a row for every compound combination.
Lexical token content is included in the composition. The default shared table
has 8,192 feature addresses plus a padding row, independent of the number of
word/line profiles. Two salted feature hashes reduce accidental coupling, but
hash collisions remain possible; these addresses are not unique identity IDs.

Role views do not own duplicate embedding parameters. Parent and own-signature
views consult the same component representation. The new hierarchy conditioning
uses these categorical/property features and ignores the old numerical ID
vectors. Token/signature mappings still index compact component-address buffers,
so the existing causal metadata contract remains compatible within this version.
This is property-level composition and weighted token-span composition, not a
semantic parse of adjectives, colors or syntax.

Lattice level/relation biases project the shared vectors. Its aggregate cache
addresses use stable typed identity keys rather than allocation-order IDs. The
lattice retains its configured bounded bucket budget and can aggregate colliding
entries. It does not authorize copying: the separate span store checks complete
prefix/target token tags and independently verifies every emitted token.

## Continuation evidence and verification

When enabled, `compute_loss()` records bounded spans from supervised training
positions after forward execution. Plain forward, evaluation and generation do
not add evidence. Token-superposition training currently skips evidence capture.
The default store retains 256 entries, with at most four observations per sample,
32-token prefix contexts and eight-token continuations. Buffers are serialized
with checkpoints; CPU identity indices are rebuilt from buffers after loading.

Retrieval prefers exact prefix tags. Otherwise, shared span vectors retrieve a
candidate only above a similarity threshold and separation margin. Evidence must
meet minimum support. Confidence is empirical per-position agreement among
stored exact-prefix alternatives, not a calibrated neural probability. A near
match still needs verification; similarity alone never emits a token.

Greedy, single-example ordinary generation proposes a span and verifies each
slot against the model's actual filtered token choice. A supported agreeing
token is accepted from the proposal. Uncertain or disagreeing slots use the
model choice; later slots remain eligible for independent verification. Token
positions are used for slot alignment, so unequal-length lexical alternatives
may cause additional rejections; this is not yet word-level alignment.

Verification occurs before hierarchy/state commit. Rejected proposal tokens
never enter the torus or cache state; accepted tokens take the normal causal
transitions. This conservative path verifies token by token and does not yet
batch verification or skip recurrent transitions. It adds proposal work, so no
end-to-end speedup is claimed. Sampled, beam and existing speculative generation
use their existing execution without these new span proposals.

Inspect `model.last_signature_span_stats` for request-local proposal, verified,
rejected and uncertain-token counts. Existing checkpoints cannot acquire this
representation by flipping an inference flag; the table contract differs.

## Validation

Regressions exercise shared parameter ownership and gradient flow, overlapping
constituents with distinct identity keys, independence from raw ID vectors,
full/incremental execution, local ambiguity with a confident suffix, incorrect
proposal rejection, eviction and checkpoint index reconstruction, evaluation
isolation, generation equality with proposals disabled, checkpoint equality and
CUDA BF16 gradients.

`review_artifacts/run_compositional_span_control.py` trains fresh weights on two
questions with the answers `A red apple.` and `A green apple.` for 200
presentations each. It checks raw exact answers, proposal-disabled token equality
and checkpoint token equality. It writes results to
`review_artifacts/compositional_span_control.json` and discards its temporary
checkpoint. This demonstrates integration/fitting only, with no held-out quality
or latency claim. A larger record-disjoint comparison should precede changes to
the default representation or claims of improved generalization.
