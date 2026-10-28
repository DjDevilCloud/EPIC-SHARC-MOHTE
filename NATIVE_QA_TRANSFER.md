# Native document-QA transfer: gated implementation

This study preserves the accepted checkpoints and runs consistency and data
prerequisites before any new learning. A failed prerequisite ends this study;
unrun learning stages must not be presented as completed experiments.

## Implemented interfaces

- `PrismalWaveConfig.qa_readout_route`: `legacy` preserves old dispatch;
  `native`, `identity`, and `backbone` select one route explicitly. Explicit
  routes take precedence over `apply_identity_readout`. The choice is saved
  with checkpoints and used by ordinary generation. CLI inference accepts
  `--qa-readout-route` as a run-local configuration override.
- `NativeQuestionMemory.score_sources` and `decision_packet` implement the
  scoring/plan path shared by inference and fitting. Cached inputs contain
  frozen observed features; learned semantic states are recomputed.
- `forward(..., return_qa_diagnostics=True)` exposes input-only source scores,
  shortlist, endpoint probabilities, copy confidence, and verified prefix
  mass through `output.qa_diagnostics`. It does not change inference scores.
- Semantic structure flags, question-word attention, and normalized fusion
  are optional configuration fields, disabled on old checkpoints. Structure
  and attention additions have zero-initialized residual projections; fusion
  uses positive temperatures initialized to one and a gate bias of -8.

## Execution gate and review

The evaluator checks frozen cached versus live packets, exact candidate
identity, full-prefix versus incremental logits, historical native outputs
through the normal generation route after reload, and exact payload/EOS
delivery when an evaluator injects the correct proposal. The oracle leaves
the learned copy gate intact and never enters normal inference.

The existing 104-record review retains 58 training, 13 validation, and 18 test
records. These are previously inspected diagnostics, not a new holdout. A
separate review of the old broader 16 retains 14: the Migrant Mother attribution
is absent from its context, and another question confuses regulatory agencies
with bodies granting their authority. Original denominators and labels remain
preserved in the review and ledger.

## Data stop condition

The fresh benchmark requires 128/64/64 reviewed questions from at least
64/32/32 disjoint source groups. Grouping conservatively merges exact texts,
observed URLs, and five-word shingle containment at 0.8. All prior manifest
contexts and related groups are excluded. Runtime budgets are unchanged:
512 source words, 32 question words, 16-word spans, and the existing 2,048-token
fitting protocol. No answers are cropped into the prompt to make them fit.

The annotated DiverseQA shard supplies only **108 fresh eligible source
groups** after those exclusions. Other inspected local shards contain only
four additional parseable document-QA rows combined, an upper bound of four
additional groups before exclusions, budgets, or review. Thus even the
optimistic upper bound is **112 groups, below 128**. Support review can only
reduce this count. Unannotated documents and generative SFT conversations
have not been silently converted into a new benchmark.

The original full-document data gate therefore fails. At that stop, no frozen training/development/test manifest,
new optimization updates, arm comparisons, or locked-test evaluations are
produced. The optional architecture changes are implemented and unit tested;
their learning or transfer benefit is unestablished. More independently
grouped annotated QA is required to resume the planned comparisons.

## Reproduce

Run with bytecode writing disabled. Data preparation requires `pyarrow`; the
local Python 3.12 installation has it. Execution checks use CPU PyTorch with
one thread, and require the existing experimental checkpoints/artifacts.

```powershell
python -B review_artifacts/review_native_transfer_broader.py
python -B review_artifacts/prepare_native_transfer_review.py
python -B native_qa_transfer_study.py contract --seed 47
python -B native_qa_transfer_study.py contract --seed 83
python -B native_qa_transfer_study.py status
```

Results are under `review_artifacts/native_qa_transfer_v1`: per-seed ledgers,
route-only checkpoint copies, source/exclusion inventory, fingerprinted review,
protocol/code hashes, `comparison.csv`, and `acceptance.json`. The original
accepted checkpoints are never written. Remaining training phases stay
unexecuted because their prerequisite failed; `comparison.csv` records that
status explicitly rather than inventing matched-budget results.

Observed checks: both seeds passed all 103 reviewed cases (206 checks total)
with exact oracle payload/termination and historical raw-answer parity. The
largest full/incremental logit difference was 0.0000191; cached/live differences
were below 0.000001. The full suite ran 437 tests: 435 passed and two skipped.

## Separately approved passage study (v2)

The user approved fixed, answer-blind passages from fresh long documents to
address the data shortage. This changes the task to **passage-grounded extractive
QA**. It does not satisfy the original full-document data gate, and passage
results must be reported separately.

`prepare_native_qa_passages.py` partitions original source text into contiguous,
nonoverlapping passages of at most 160 whitespace words, preferring existing
paragraph/sentence boundaries after 64 words. Partition offsets are determined
before reading questions or answers. Existing annotations are then matched to
these fixed passages and checked against the unchanged native limits. No new
questions or answers are generated. Original source identities are grouped
before partitioning, so extra passages never create extra independent groups.
Eligible passages are selected by text hash, without model predictions.

The initial passage inventory had 217 paired groups. Closing parent families
over passage overlap and prior exposure left 213. Explicit assistant review
recorded 438 annotation decisions, merged related curriculum, aircraft-accident,
publication, and event-calendar fragments, and excluded additional families
recognized as previously exposed. This left **140 reviewed usable families**.
The first 128 under the frozen group-hash allocation supply:

| Split | Questions | Source families |
|---|---:|---:|
| Training | 128 | 64 |
| Development | 64 | 32 |
| Locked test | 64 | 32 |

Each selected family contributes two questions about the same passage with
different answers. Review rejects unsupported premises, incorrect associations,
duplicate facts disguised by different answer lengths, and missing source
headings. `reviews.json` preserves decisions, reasons, supporting character
intervals, answer offsets, manual family joins, and exposure exclusions.
The manifest preserves parent identities, exact parent character offsets,
parent/partition hashes, review fingerprints, and split assignments. Neither
models' correctness nor their scores were used to choose records.

`validate_native_qa_passages.py` verifies original annotations and slices against
the local Parquet file, review fingerprints, split sizes, distinct-family pairs,
cross-family overlap, previous exposure, and exact native span representability
in **both seeds**. All 256 answers pass in each seed; maximum observed fitting
length is 1,062 tokens, native source count 178, and question count 29.
This validates representation only. It runs no neural prediction on the locked
test. Accepted baseline checkpoint hashes are unchanged.

Authoritative parent-page provenance is unavailable. Grouping combines local
identities, observed URLs, normalized text, five-word containment, and manual
family review; this remaining uncertainty must accompany reported results.
Review was performed by the assistant; no independent human annotation review
is claimed. The original broader 16 and previously inspected QA remain development
diagnostics and do not become fresh holdout data.

Artifacts are in `review_artifacts/native_qa_transfer_passages_v2`: frozen
`data/frozen_manifest.json`, fingerprinted candidates/reviews, validation log,
code snapshots in `repro_sources`, and machine-readable `data_gate.json`.
`acceptance.json` records `DATA_READY_LEARNING_NOT_RUN`; `protocol.json`
binds the passage manifest and starting checkpoints to the retained study
budgets and gates. `family_audit.json` records the lower-threshold overlap review.
An independent clean preparation rerun reproduced every inventory and QA record
field exactly; `reproduction.json` records that comparison. Code-hash metadata
records the later overwrite guard and output-directory interface separately.
The data gate passed for the passage study with **model acceptance false**.
At that data-freeze snapshot no training updates or locked-test model evaluations
had run. Its `data_gate.json` and data-only `acceptance.json` describe that stage;
the later source decision is recorded separately in `source_acceptance.json`.

Validate the frozen artifact with Python 3.12 and local PyTorch/pyarrow:

```powershell
python -X utf8 -B validate_native_qa_passages.py
```

Candidate preparation and freezing refuse to overwrite existing inventories or
manifests. To reproduce preparation, use a new directory:

```powershell
python -X utf8 -B prepare_native_qa_passages.py --output review_artifacts/native_passage_reproduction/data
```

Original artifact bytes and their hashes remain the authority for reviewed
inputs; regeneration does not automatically inherit semantic review approval.

Validation of the changes: 441 unit tests ran, 439 passed and two skipped.

## Bounded source-study outcome

The authorized follow-up bound the runner, complete schedules, replay/style
inputs, feature packets, both starting checkpoints, and code snapshots before
optimization. All three source arms ran exactly 1,600 updates per seed; the
selected structure-only encoder was then frozen for one 800-update normalized
fusion comparison per seed. No width, tokenizer, or memory expansion was used.

| Source arm | Seed 47 development top-1 /64 | Seed 83 development top-1 /64 |
|---|---:|---:|
| Control | 2 | 3 |
| Structure only | 3 | 6 |
| Question attention only | 3 | 5 |
| Selected encoder + normalized fusion | 4 | 3 |

The normalized mixture recovers 16/16 fresh-value starts in both seeds but
reaches only 27/64 and 37/64 development top-16 recall. Its training starts are
125/128 and 126/128. This fails the required 32/64 top-1 and 58/64 top-16 source
gate in both seeds. Seed 83 also regresses to 56/58 legacy and 57/60 native
replay source decisions. Source metrics are not raw-answer retention metrics.

Optimization stops at the failed source gate. Extent and joint fitting and
locked-test model evaluation remain unrun. Accepted baseline hashes are
unchanged. The result establishes that these encoder/fusion additions can fit
the reviewed training questions and recover the fresh-value start diagnostic,
but do not establish transfer to the reviewed unseen source families.

See `NATIVE_QA_SOURCE_STUDY.md` for the frozen protocol, execution repairs and
reproduction commands, and `source_study/REPORT.md` / `report.json` for the
comparison, source-family uncertainty, contrast results, and raw-answer
measurement denominators. No candidate is accepted as a general QA model.
