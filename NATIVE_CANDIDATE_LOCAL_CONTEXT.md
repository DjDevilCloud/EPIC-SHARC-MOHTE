# Candidate-local native key context

`native_question_memory_backbone_context` is a saved configuration field:

- `completed_line_v1` is the default, including for checkpoints written before the field existed.
- `candidate_local_clause_v1` requires `backbone_context_v1` native QA. It replaces only the context half of the existing key projection input.

The local input is the mean of up to eight observed source-word core states before and eight after each candidate, within its recorded clause and excluding that candidate's own vector. Empty contexts receive zero. The scope uses the same clause-neighbor helper as the existing source-role feature extractor. No new encoder, projection width, parameter, source slot or proposal capacity is added.

This input is built by the common live native decision path, including teacher-forced source supervision. Learned states are recomputed during fitting. Answer labels never enter extraction. Right context is permitted because it belongs to the already-observed input document; generated answers cannot modify the frozen source state. Excluding a candidate's vector does not imply that neighboring contextual states are independent of that candidate's text.

The saved setting is applied when the model is constructed or reloaded. Native context producer identities include the context policy, so changing the policy invalidates carried state. Experiments changing a newly loaded initial configuration also set the corresponding module field and invalidate native context before use. There is no per-token policy override.

Conditional extent and emission heads retain their original feature inputs. Changing source scores can change proposal ordering and shortlist membership; it does not change the span-construction rules. Diagnostic plans record `key_context_policy`, and optionally expose the observed context matrix when diagnostics are requested.

The old source and tests were archived before editing in `review_artifacts/candidate_local_context_engineering_v1/history/source/`. The full regression suite ran 547 tests: 545 passed and two skipped. Completed-line mode reproduces 448 previously recorded source decisions, shortlists, proposal identities/probabilities and 16 sampled greedy answers bitwise.

The new option passes clause/window/exclusion checks, full-prefix/incremental execution, carried-state policy rejection, saved reload and ordinary canonical generation checks, source-gradient coverage and future-answer/label isolation. Explicit high-confidence decoder fixtures still reproduce exact payloads and termination for names, currency, percentages and text; these are mechanical checks, not learned gating or raw QA performance.

On the actual width-32 initial checkpoints, source-loss gradient norms into the key's context half change from approximately 5.1e-7 / 6.4e-7 under completed-line input to 2.516 / 4.486 under local input. The inherited diagnostic helper calls this column `completed_line_half_norm`; it denotes the same second half of the key weights under either policy. These measurements establish a nonconstant connected path, not improved learning.

Clause boundaries around initials, decimals and other punctuation remain a limitation. Local pooling can also lose useful context across clauses. Existing checkpoints retain completed-line behavior. The bounded learning comparison must pass separately, followed by retention before qualification. Accepted baselines and prior study artifacts remain unchanged; the locked test remains closed.
