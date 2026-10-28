# V4 reviewed multi-answer QA and objective contract

The original passage manifest, its split assignments and all 192 primary
training/development answers remain unchanged. The stopped v3 decision remains
historical evidence. This separately authorized v4 contract adds reviewed
answer sets and evidence locations; it does not turn the locked test into
development data.

## Reviewed data

All **128 training and 64 development questions** received explicit review
anchors and evidence intervals, without model predictions. Eleven questions
have multiple accepted surfaces: **211 references** in total. The Minnesota
question accepts Somali, Latino and Hmong. The missing-hunter question accepts
the two named responders individually or together. Other enumerated alternatives
cover reviewed short-form or named-entity surfaces; there is no global case
folding, punctuation stripping or general paraphrase acceptance.

Every accepted surface has at least one supported, copyable occurrence in both
seeds. One additional supported mention, `Bay Pines VA’s`, cannot emit the shorter
name at that occurrence because `VA’s` is a single native word. The same event
paragraph contains a copyable `Bay Pines VA` reference. Both mentions remain in
the annotations; only actually copyable occurrences label inference-ready starts.
No tokenizer, memory or proposal budget was expanded to make this pass.

Review is by the assistant, with no independent human review claimed. Some
questions supply identifiers or qualifications absent from the fixed excerpt,
such as episode numbers. These are disclosed per record; the review grounds
the requested answer relation, not the truth of every question-provided label.
Reference sets enumerate approved literal spans and do not cover every valid
rephrasing. Missing authoritative parent-page provenance remains a limitation.

`reviewed_answers.json` is the approved overlay. The first draft and the initial
boundary diagnostic are retained separately. The clean review reproduction is
byte-identical. `native_bindings.json` connects character locations to native
starts/ends and records unavailable mentions. The maximum answer payload is
71 tokens, within the fixed 128-token generation profile.

## Shared execution

New QA candidates save `qa_readout_route=native` and
`qa_generation_profile=native_qa_eval_v1`. Ordinary model generation, the Python
API and CLI share the answer-independent profile: 128 tokens, greedy selection,
no repetition penalty, no repeated-ngram blocking and no speculative decoding.
Old checkpoints default to `legacy` and keep their prior behavior. Explicit
overrides are honored and recorded as noncanonical profile settings.

For structured generation results:

```python
record = generate_text(model, tokenizer, prompt, device,
                       return_generation_record=True)
```

```powershell
python cli.py --device cpu infer --checkpoint NEW_CANDIDATE/model.pt --prompt "Context: ... Question: ..." --generation-json --no-torch-compile
```

Profile canonicality concerns generation settings; acceptance also requires the
saved native route. Gold answers enter evaluation and explicitly named oracle
tests only, never normal prompt construction or generation limits.

The native memory's guidance function is shared by live and cached emission.
Functional replay protects source decisions, conditional extents, stable copy
confidence and teacher-forced token probabilities. Incorrect teacher decisions
receive gold supervision instead of positive preservation targets. Replay is
normalized by example/decision type, with output-token means within examples.

Both seeds passed 646 cached/live source comparisons, readout oracle rollouts
for all 211 reviewed references, sampled full-model emission and incremental
comparisons, future-answer substitution and checkpoint reload checks. The
readout oracle runs the actual native state machine and mixture against an
adversarial fallback; full-model ordinary generation checks are separately
identified. Historical stage-zero raw-generation references are also explicitly
separated from current sampled checks. Seed 83's parent inherits 12/16 fresh
answers; this is not recorded as a new regression.

Validation before optimization: the full suite ran 492 tests, with 490 passing
and two skipped. Two additional runner checks then passed, covering inference
measurement flags and corrected-expert serialization. Actual logs and test
records are stored with the v4 artifacts.

## Fixed source comparison

Both seeds and arms start from the same experimental parent per seed. The
architecture and all budgets remain unchanged. The control supervises corrected
source probabilities; the repair averages corrected-source and semantic-only
source supervision. Both use the same functional replay. The earlier global
paired-assignment loss is excluded.

Each source phase uses 1,600 updates; each frozen-encoder fusion phase uses 800.
Sampling is fixed at 8 new, 2 legacy, 2 native and 2 controlled presentations.
Four layout presentations repeat native inputs; they are not independent new
documents. Fusion compares legacy probabilities with the complete corrected
distribution, not an uncalibrated residual treated as an independent expert.

Source selection uses final checkpoints only. Both seeds must reach 122/128
training starts, 32/64 development starts, 58/64 development top-16 recall and
16/16 fresh starts. Failure ends this study without extra branches or updates.
Extent, joint fitting and locked-test evaluation require subsequent gates.

Run the guarded stages from the project root with Python 3.12 and bytecode
disabled:

```powershell
python -X utf8 -B native_qa_objective_study.py preflight --seed 47
python -X utf8 -B native_qa_objective_study.py preflight --seed 83
python -X utf8 -B native_qa_objective_study.py freeze
python -X utf8 -B native_qa_objective_study.py source --seed 47 --arm control
python -X utf8 -B native_qa_objective_study.py fusion --seed 47 --arm control
```

Repeat source/fusion for both arms and seeds, then run `decide`. Completed stages
refuse to rerun; interrupted fitting resumes only from its sealed 200-update
optimizer/RNG snapshots. Frozen final results are separate from working history.
Preflight revisions made zero optimizer updates and are recorded separately.

The authoritative outcome and measured comparison are in
`review_artifacts/native_qa_multianswer_contract_v4/study/acceptance.json` and
`study/REPORT.md`. Per-checkpoint ledgers distinguish source, shortlist,
gold-start extent, complete proposals, raw-answer measurement status and
termination. Cached teacher-forced replay is a proxy; it is not a raw-answer
retention certificate. Original-reference metrics remain separate from v4
reviewed-reference metrics. Accepted baseline files remain protected throughout.

## Completed outcome

The bounded comparison stopped at `STOPPED_SOURCE_GATE_FAILED` after 9,600
optimizer updates. Both arms fit all 128 training starts in both seeds.

| Selected arm | Seed 47 dev starts /64 | Seed 83 dev starts /64 | Dev top-16 recall, seeds 47/83 | Fresh starts, seeds 47/83 |
|---|---|---|---|---|
| Control, learned fusion | 3 | 8 | 27/64, 32/64 | 16/16, 16/16 |
| Branch-consistent repair, unchanged fusion | 4 | 7 | 27/64, 29/64 | 12/16, 9/16 |

The predeclared worst-seed selection chose unchanged fusion for the repair:
learned fusion retained its top-1 results but reduced worst-seed shortlist
recall. The repair failed development selection, shortlist recall and fresh
starts. Its seed-83 fresh result is below the parent's inherited 12/16; seed
47 also regressed from 16/16. Functional replay did not guarantee retention.
These are source decisions, not raw-answer retention measurements.

Training semantic scores as an independent branch did not produce the required
transfer at this budget. This result does not establish insufficient native
representation capacity. Extent and joint optimization were not run; the locked
test received zero predictions. The candidate was rejected and accepted baseline
hashes remain unchanged. A new architecture hypothesis requires a separate study;
no additional branch or training extension belongs to this completed experiment.
