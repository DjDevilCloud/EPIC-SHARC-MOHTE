# V3 objective-contract study: prerequisite stop

The requested study stopped at its support-review prerequisite. No model,
generation defaults, tokenizer, frozen split, answer, or historical decision
was changed. Source, fusion, extent, joint fitting and locked-test inference
remain unrun. The later model features and training runner were not implemented
after the prerequisite failed.

## Recorded defect

Development index **20** asks:

> What is one of the three largest ethnic minority groups in Minnesota that this training focuses on?

The supplied passage explicitly lists **Latino, Hmong, and Somali**. The frozen
single answer is **Somali**. All three are supported answers to the question;
the passage cannot justify accepting only one. This is a single-gold evaluation
ambiguity, not evidence that the annotated answer itself is false.

The v3 support overlay records the question fingerprint, exact evidence interval
`[319, 433)`, and the three alternatives' character offsets. These alternatives
are diagnostic evidence only: none was installed as a new training target,
positive source start, or accepted answer. No record was excluded or replaced.
Original denominators remain 128 training / 64 development / 64 locked test.

The explicit stop instruction is:

> If support review invalidates a question or leaves an ambiguity unresolved, stop before training rather than silently changing denominators or replacing records.

Only the decisive failed review is formally recorded. The remaining 191 support
decisions are unapproved; an incomplete overlay cannot pass the gate. Resuming
requires a separately authorized, versioned annotation/answer-contract change.
The stopped v3 study must not silently restart with modified targets.

## Completed implementation

`native_qa_objective_contract.py` provides:

- `archive`: archive dependencies before production edits, pin historical
  artifacts and accepted/parent checkpoint hashes, and refuse conflicting
  snapshots.
- `verify`: read-only verification of archived dependencies and original
  historical evidence. Later production edits can coexist with their archived
  originals; historical artifacts and protected checkpoints cannot change.
- `review-gate`: read-only validation of training/development fingerprints,
  denominator preservation, evidence intervals, answer occurrences and review
  completeness. Locked-test and unknown records are rejected.
- `finalize-prerequisites`: write a new decision with zero optimization updates
  and explicit unrun stages; refuse to replace an existing decision.

The archive contains **168 snapshots** and **85 historical bindings**. Historical
large checkpoints are hash-bound at their existing locations, not duplicated.
All new artifacts are isolated under
`review_artifacts/native_qa_objective_contract_v3/`.

## Verify without rewriting evidence

Use the recorded Python 3.12 installation, UTF-8 and bytecode disabled:

```powershell
python -X utf8 -B native_qa_objective_contract.py verify
python -X utf8 -B native_qa_objective_contract.py review-gate
python -X utf8 -B -m unittest discover -s tests -p test_native_qa_objective_contract.py
```

`review-gate` returns a failed/incomplete review result without writing files
and exits with status 2, so automated callers cannot mistake a blocked gate for
a passing command.
Do not invoke historical validation/report writers against the new study.

The new tests cover repeated answer occurrences outside supporting evidence,
changed fingerprints and denominators, ambiguous questions receiving positive
labels, fabricated quotations, alternative-answer evidence, locked-record
exclusion, complete versus incomplete review, overwrite refusal, and read-only
historical verification after production changes.

`acceptance.json`, `failure_ledger.json`, `comparison.csv`, the intended
`protocol.json`, actual `test_run*.json`/`tests*.log`, and `artifact_audit.json`
record the stopped state. These artifacts must not be interpreted as completed
source-learning experiments or as a passing QA model.
