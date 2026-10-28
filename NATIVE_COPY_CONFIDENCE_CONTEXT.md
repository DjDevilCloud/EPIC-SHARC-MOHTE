# Copy confidence independent of source ranking

The repair and bounded calibration comparison are recorded under
`review_artifacts/native_copy_confidence_context_v1`. Accepted checkpoint files
are unchanged; old checkpoints retain their original inference behavior.

## Saved policies

`native_question_memory_copy_context` defaults to `retrieved_v1`. This preserves
the existing gate input: the source-distribution-weighted native value vector.

The opt-in `observed_mean_v1` policy supplies the unweighted mean of the same
observed source value vectors to the existing copy gate. Its other input remains
the shared backbone question request. This policy requires the backbone native
representation and stable conditional copying. It adds no learned parameters,
encoder, width, source slots, span proposals or decoder branch.

The retrieved vector remains unchanged for native readout guidance. Only the
confidence head receives the additional mean summary. Source scoring, shortlist
selection and conditional extent scoring retain their existing formulas.
The gate uses the same `gate_logit` function in live generation, functional replay
and frozen-producer head calibration. No labels or oracle candidates enter that
function. The policy is part of the carried-context producer identity so states
cannot silently cross policies.

At fixed observed word and question states, changing source scores can no longer
change this gate or send its loss directly into retrieval scoring. Source and
confidence still share backbone features; this does not eliminate shared-feature
gradient interference in a future end-to-end fit. Functional replay remains
necessary to protect behavior when those features train.

## Bounded calibration

The contract, source archive, initial checkpoint hashes and example schedules
were frozen before optimization. Two arms, `retrieved_v1` and `observed_mean_v1`,
start from identical restored-student tensors for seeds 47 and 83.

Each arm receives 200 updates through the common Muon optimizer factory. Only
the existing 65 gate parameters train. The backbone, source scoring, extent and
emission parameters remain frozen and are checked bitwise after calibration.
The previous Muon settings and group clipping caps are retained; there is no
scheduler or budget extension. The helper's trajectory field `clipped_groups`
counts groups processed by clipping, not the frequency of norms exceeding caps.

Each update uses eight newer training questions and two each of reviewed legacy,
native replay and controlled replay. The loss is the existing correct-teacher
Bernoulli KL on replay or gold copy-appropriate BCE on newer training cases.
Token and source losses do not optimize this confidence calibration. There is
no forced gate value, bias reset or proposal injection in training.

Gate inputs may be cached here because their entire producer is frozen. Live
versus cached gates are verified bitwise, and these caches must not be reused
for a fit that changes backbone, value or question representations. The ordinary
live functional replay implementation remains the path for such a fit.

## Interpretation limits

All supervised QA examples are reviewed extractive positives. This study tests
confidence recovery on those tasks; it does not establish no-answer detection,
general QA, transfer, or retained ordinary answers. Both the unchanged context
control and the mean context arm receive exactly the same calibration budget,
so improvement shared by both arms must be attributed to calibration rather
than solely to the new context input.

The experiment evaluates original and calibrated checkpoints with canonical
128-token ordinary generation, all 574 raw retention cases, 112 newer source
controls and all 438 reviewed exact-proposal oracle cases. Oracle injection
exists only in the evaluator and preserves the learned gate. A passing oracle
does not manufacture a correct source or extent during ordinary generation.
