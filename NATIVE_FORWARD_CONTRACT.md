# Native QA input and event contracts

New engineering candidates can save these options before constructing the model:

```json
{
  "native_question_memory_framing": "blocks_v1",
  "native_question_memory_events": "causal_clause_v2"
}
```

Both require the existing backbone native representation. Defaults remain `legacy` for older checkpoints. The changes add no parameters or vocabulary tokens. Source capacity, antecedent capacity, eight identity words/eight contextual vectors per event, span width and start shortlist stay at their existing limits. Temporary scoring work can change; this is not a throughput claim.

## One formatter for explicit fields

```python
from native_qa_format import format_native_qa

training_record = format_native_qa(context, question, answer, framing="blocks_v1")
generation_prompt = format_native_qa(context, question, framing="blocks_v1")
```

The existing codec encodes `<BOI><BOP>source<EOP><BOP>question<EOP><EOI><BOO>`. Paragraph blocks make the roles explicit without interpreting literal `Question:` or `Context:` lines as control instructions. Dataset grouping recognizes this form so atomic QA windows retain their complete source. Reserved codec control strings inside fields are rejected instead of silently changing their meaning. Ordinary document punctuation and header-like text are preserved.

The ordinary Python API uses the saved policy:

```python
record = generate_text(model, tokenizer, question, device,
                       qa_context=context, return_generation_record=True)
```

The CLI accepts `--qa-context-file document.txt --prompt "Your question"` with `infer`. Use a saved candidate with the new policy; no inference override silently changes an older checkpoint's parser. An explicitly serialized generation prompt is also supported. Raw combined header-style prompts retain their legacy ambiguity; new framed models require explicit fields or the block format.

## Causal event observations

`causal_clause_v2` delays an ambiguous period until its following word is observed. Adjacent numeric fragments stay in one event for decimal amounts; a capitalized single-letter initial followed by a capitalized word stays in the event; semicolons end events. This is a deterministic convention, not a complete natural-language sentence parser. Ambiguous sentence-final initials and general abbreviations still need reviewed treatment. Legacy punctuation behavior is unchanged by default.

Identity words remain the first eight words. The same eight contextual-vector slots now retain recent observations. At a candidate source ordinal, ownership scoring selects the latest causal contextual state and up to three retained causal suffix states using the existing source ring and event tail. Later words cannot enter an earlier candidate's features. At an evicted boundary, only retained causal suffix states are used; the model does not manufacture missing history. `antecedent['context_ordinals']` exposes the observation timestamp for source-label review without adding it as a learned feature.

## Reporting and verification

SHARC router scalar telemetry now propagates from each actual path under `sharc_router_*`. It does not overwrite core emitter statistics or losses. Instantiated architecture must be read alongside saved flags: the engineering parents use a flat torus core with shared hierarchy, lattice and SHARC augmentation; they do not instantiate hierarchical experts.

The engineering result, source snapshots, exact legacy comparisons, saved experimental candidates and machine-readable gate are under `review_artifacts/native_forward_contract_repair_v1`. These candidates have unchanged parameter values and are not accepted QA models. Input/core-state compatibility tests do not establish learned ownership or retention under the revised policies.

Before fitting, review causal labels against the new observation timestamps and rebuild functional-replay packets for the new serialized input. Old replay packets contain old token IDs and BOO positions and must not be reused as if those addresses were unchanged. Then freeze the ownership study's objectives, budget, schedules, update telemetry and retention gates. No training is part of this contract repair.
