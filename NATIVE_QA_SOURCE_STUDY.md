# Bounded native passage source-selection study

This is the user-authorized follow-up to the separately approved fixed-passage
data gate. It studies passage QA, with the existing tokenizer, width, memory,
and native span machinery. Accepted baseline checkpoints remain protected.

## Frozen experiment

`native_qa_source_study.py freeze` writes
`review_artifacts/native_qa_transfer_passages_v2/source_study/frozen_experiment.json`.
This binds the reviewed manifest, replay inputs, both starting checkpoints,
runtime, optimizer settings, code snapshots, and complete sampling schedules.
Every subsequent phase checks those hashes. Development and fresh-value cases
receive no fitting updates. The locked test receives no model predictions.

Both seeds (47 and 83) start all three encoder arms from their same experimental
checkpoint. The arms are control, eight observed structure flags through a
zero-initialized projection, and contextual question attention through a
zero-initialized residual. The additions are not combined. Only semantic
encoder/source-fusion parameters are optimized for 1,600 updates; endpoint,
emission, backbone, and other parameters remain frozen and are checked bitwise.

Each update draws eight new training, two reviewed legacy training, two native
replay, and two controlled/layout replay presentations. Schedules match across
arms and seeds. The existing layout pool contains four duplicate native inputs;
these remain replay presentations, not independent transfer examples.
New training questions each receive exactly 100 presentations per source arm.

The objective is all-source gold-start negative log mass. A positive start must
support the complete exact answer payload, including punctuation, rather than
merely match its lexical words. Correct teacher replay decisions additionally
receive source KL and a capped gold-decision margin. Incorrect teacher decisions
receive gold source supervision without a positive teacher preservation target.
Endpoint and emission losses are absent from this source-only stage.

The batched encoder uses packed variable-length source and question sequences
to prevent backward recurrent states from consuming padding. It shares the
inference source-scoring rule. Tests check scores and parameter gradients
against independent examples, including nonzero structure and attention
residuals. Production inference keeps its ordinary single-example path.

## Execution gate and selection

Before fitting, both seeds must pass frozen/live packet checks, independent/
batched source-score tolerance, identical live/cached candidates, saved route/module
reload, and ordinary greedy generation equality. New passage checks also cover
full-prefix/incremental execution and exact oracle payload/termination with
the learned copy gate intact. The original 103-case per-seed execution contract
remains required; oracle interventions remain evaluator-only.
The batched computation is used only for the continuous training loss. All
ranking, shortlist, metric, and generation decisions use the ordinary inference
scorer; score-tolerance agreement does not imply identical near-tie ordering.

After exactly 1,600 updates, encoder selection uses worst-seed development
top-1 count, then worst-seed top-16 count, then worst-seed source loss. No earlier
checkpoint is selected. One normalized-fusion comparison then runs for 800
updates, freezing the chosen encoder. It learns positive branch temperatures
and the prescribed request/distribution mixing gate, initialized with unit
temperatures and gate bias -8. The comparison is the unchanged selected encoder's
existing fusion.

Both seeds must reach at least 122/128 training starts, 32/64 development
starts, 58/64 development top-16 recall, and 16/16 fresh-value starts. Failure
ends the study without extra branches or updates. Passing only opens the next
extent stage; it does not accept a general QA model or replace the baseline.

## Reproduction

Use the recorded Python 3.12 environment with CPU PyTorch and one thread:

```powershell
python -X utf8 -B native_qa_source_study.py freeze
python -X utf8 -B native_qa_source_study.py preflight --seed 47
python -X utf8 -B native_qa_source_study.py preflight --seed 83
python -X utf8 -B run_native_qa_source_stage.py fit --seed 47 --arm control
```

Use the guarded `run_native_qa_source_stage.py` entry point for `fit` for each arm
and seed, then `select`, `fusion --seed 47`, `fusion --seed 83`, `decide`, and
`ledgers` for each seed. It verifies the full replay/style inputs, exact feature
packets, initial checkpoints, runner code, and schedules before and after each
stage against `execution_input_seal.json`. Fitting resumes from
200-update snapshots with optimizer and random state preserved. Run logs,
per-update evaluations, final checkpoints, source ledgers, comparison CSV,
and acceptance decisions live under `source_study`. Unmeasured raw QA fields
are explicitly marked; they are never counted as zero accuracy or success.

Two preflight revisions are preserved. The first exposed a batch-only near-tie
ordering change and made no optimization updates. The second passed execution
checks, then received a metric-only correction: top-1 follows inference's
`argmax`, and top-16 follows its actual probability shortlist. AST comparison
proved every nonmeasurement function identical, allowing the passed execution
contracts and frozen features to be reused. `metric_revision.json` records the
old/new packet hashes and the metadata-only rebinding. Neither correction
changed learning budgets or added an experiment arm.

The frozen fitting runner also had an overly strict post-save assertion on
exact floating-point averages. Control checkpoints reproduced all weights and
source selections, while two replay loss averages differed by about 1e-9.
`finalize_native_qa_source.py` repairs final validation without optimization or
checkpoint writes: it checks exact parameters, protected modules, candidates,
counts, and tested greedy answers, with the original logit tolerances. Its
results include its own code hash and measured differences. If this final
assertion fires after a full-budget run, invoke the read-only finalizer for that
seed and arm; do not restart optimization or add updates. Original failure logs
are preserved.
