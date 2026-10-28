# Shared-backbone QA: verified progress and next gate

The optimizer and representation repairs are implemented. The first bounded live-core control demonstrates learning, but fails its required two-seed generalization gate. Optimization stopped at exactly 200 updates per seed; no checkpoint replaces the accepted baseline.

## Verified implementation

The hierarchical table rule now preserves gradient direction, nested EMA affects actual updates with resumable state, and enabled modules materialize before optimizer registration. The common factory verifies parameter coverage. Its independent engineering gate passed 519 tests with two skips.

The new saved `backbone_context_v1` representation connects existing native source, endpoint and emission heads to observed torus/SHARC states and shared hierarchy features. It does not instantiate an independent native semantic GRU. Legacy checkpoints retain their existing representation. The representation gate passed 534 tests with two skips, including causal execution, reload, cache invalidation and gradients into the intended core components. Historical compatibility checks cover recorded decisions and sampled generation, not full raw retention.

## Frozen learning control

Python 3.12, CPU float32, one thread; inherited width 32 and flat 2x2x2 torus; existing tokenizer and proposal budgets. Common Muon factory: matrix LR 0.008, scale 0.5, NS 8, Nesterov, momentum 0.95; scalar AdamW LR 0.001; weight decay 0.01; nested learning disabled. Source NLL plus ordinary token CE. Eight training questions from four passages, four unseen-value questions from two passages. No lexical-overlap or exact-identity matching shortcuts. All model parameters are trainable; this is not an adapter-only study.

| Seed | Training source | Training complete raw answers | Development source | Development complete raw answers |
|---|---:|---:|---:|---:|
| 47 | 8/8 | 8/8 | 4/4 | 4/4 |
| 83 | 8/8 | 8/8 | 3/4 | 3/4 |

Both seeds started at zero correct sources and answers. Credited answers require exact token payload and proper termination under the shared answer-independent generation profile. Both contrasting questions succeed in all four training groups; development paired success is 2/2 groups for seed 47 and 1/2 for seed 83. Each seed changed 31 torus tensors, 26 router tensors, seven lattice tensors and one shared-signature-bank tensor. Final checkpoint reload reproduces every recorded metric and answer.

The remaining seed-83 case is `Bo has 29 tokens. Al has 46 tokens.` / `How many tokens does Bo have?` It selects 46. The supported start ranks second and the exact 29 payload survives the shortlist. This is a source-ranking failure; neither an absent proposal nor incorrect termination explains it.

The immutable run and machine decision are in `review_artifacts/backbone_core_control_v1r2/`. Two earlier conversion preflights failed before optimization and are preserved separately. Dormant identity-reader settings were cleared when excluding that module; the data, losses and scientific budget did not change.

## Read-only sensitivity audit

A separate post-stop diagnostic changes quantities, owner labels, or clause order in the same two development passages. It performs no optimization or checkpoint selection and adds no transfer credit. Seed 47 selects all four correct starts in each variant. Seed 83 scores 3/4 on the original, 2/4 after quantity swaps, 3/4 after owner swaps and 3/4 after clause reversal. This exposes instability under simple factual substitutions. It does not identify whether core encoding, source scoring or conflicting losses is the cause.

Evidence is sealed in `review_artifacts/backbone_core_control_analysis_v1/`. These inspected synthetic records are development diagnostics forever, not future held-out tests.

## Next steps, in order

1. Audit the selection failure without updates. Compare source-only and token-loss gradients by core component at the same initial checkpoints, including their directions and relative magnitudes. Inspect source keys and question requests under the paired questions and quantity substitutions. Separate a representation collapse from a scoring failure or objective conflict before choosing a repair.
2. Preregister one new bounded control for the identified cause. Preserve this failed control, its exact gate and all denominators. Do not extend its budget, tune against the inspected quantities, or add extra encoder branches. Use independent value combinations and document the diagnostic contamination.
3. Require both seeds to learn contrasting questions and complete answers, then evaluate functional replay and the full raw retention suites. Frozen heads alone cannot establish retention after their shared core inputs change. Record inherited defects separately.
4. Only after those gates pass, compare AdamW, Muon and the repaired hierarchical optimizer on matched data, starting states, trainable membership and update budgets. Declare physical hierarchy ownership before using role tiers. Wider models and instantiated hierarchical experts remain a separate capacity comparison.

Full raw retention has not been evaluated for these candidates. No optimizer winner or general QA transfer is established. No locked-test predictions were made. Accepted checkpoint hashes, historical studies, engineering snapshots and frozen root code were verified unchanged.
