# Question-conditioned source contextualization: V5

V4's branch-consistent objective fit all training starts but failed development
selection and fresh-value retention. Its accepted baselines, data and decisions
are preserved. V5 tests a separate architectural hypothesis using the same
reviewed passage-QA inputs: question evidence should reach source words before
their contextual states are computed.

## Fixed comparison

Both arms start from the original experimental parent for each seed, not from
the failed V4 fitted candidates. Add the same zero-residual question-attention
module to both. All parameter names, shapes and initial tensors match; only
the placement changes:

| Arm | Path |
|---|---|
| `late_context` | Source word encoding → source context GRU → question attention |
| `early_context` | Source word encoding → question attention → source context GRU |

Question words receive their own context pass first. In the early arm, duplicate
each W-wide lexical query into two channels for the existing 2W attention layer;
average the projected residual's two W-wide halves back to W before the source
GRU. This fixed mapping adds no parameters. It preserves model width, tokenizer,
memory, source shortlist and span budgets. This comparison isolates placement
within that mapping; it does not test every possible conditioning design.

Each arm receives 1,600 source updates in seeds 47 and 83: **6,400 total**. Both
use the V4 control's corrected-distribution source objective and the same
functional replay. No independent semantic-branch loss, paired-assignment loss,
fusion optimization or additional encoder is introduced. Only semantic
alignment parameters train. Endpoint, emission, legacy retrieval and backbone
parameters remain frozen; output preservation is measured separately.

Use the exact V4 source schedules: 8 new, 2 legacy, 2 native and 2 controlled
presentations per update; AdamW .001, weight decay .01, clipping 1. Runtime is
Python 3.12, CPU float32, one thread. Measurements every 200 updates are for
diagnostics; compare final checkpoints only. No extension or additional arm
follows an unfavorable result.

The early candidate must pass in both seeds: 122/128 training starts, 32/64
development starts, 58/64 development top-16 recall, and 16/16 fresh starts.
Failure ends V5. Passing source selection establishes eligibility for a specified
downstream extent/emission/retention study; it does not establish QA transfer or
accept a checkpoint. This bounded source experiment makes no locked-test
predictions.

## Inputs and execution contract

Reuse V4's frozen 128 training and 64 development questions, supporting
occurrences, 211 literal accepted references and source groups unchanged. The
64 locked-test questions remain unopened for predictions. Missing authoritative
provenance, assistant review and finite literal answer alternatives remain
limitations. Development data has been inspected; bootstrap intervals quantify
observed source-family variability rather than an unseen population guarantee.

Before editing production code, verify the complete V4 artifact index and archive
its 54 root code dependencies. V5 verifies historical code against these copies;
the old artifacts, hashes and decisions are never regenerated with new code.
V5's own code, schedules, data, parents and runtime are sealed before optimization.

Execution checks include both placements at zero initialization and early
conditioning with an active, discarded residual probe. Require cached/live
scores within existing tolerances, identical inference candidate identities,
full-prefix/incremental agreement, checkpoint reload, ordinary generation and
causal future-answer checks. All reviewed references receive the existing
readout-state-machine oracle check; sampled full-model checks remain separately
identified. No probe weights become a training parent.

Per-checkpoint ledgers retain source rank, shortlist, conditional gold-start
extent, complete proposal and raw-answer measurement status. Raw answers and
termination are unmeasured at this stage. Cached teacher-forced replay is a proxy,
not a free-running retention certificate. Contrasting questions are evaluated
individually and jointly; teacher errors receive gold supervision.

## Reproduction

Use the Python 3.12 executable and bytecode disabled. These commands create new
artifacts and refuse completed-stage overwrites:

```powershell
python -X utf8 -B native_qa_context_study.py preflight --seed 47
python -X utf8 -B native_qa_context_study.py preflight --seed 83
python -X utf8 -B native_qa_context_study.py freeze
python -X utf8 -B native_qa_context_study.py source --seed 47 --arm late_context
python -X utf8 -B native_qa_context_study.py source --seed 47 --arm early_context
python -X utf8 -B native_qa_context_study.py source --seed 83 --arm late_context
python -X utf8 -B native_qa_context_study.py source --seed 83 --arm early_context
python -X utf8 -B native_qa_context_study.py decide
python -X utf8 -B native_qa_context_study.py report
```

The authoritative result is
`review_artifacts/native_qa_question_context_v5/acceptance.json`, with tables,
ledgers and uncertainty beside it. V4's prior acceptance record stays unchanged.
