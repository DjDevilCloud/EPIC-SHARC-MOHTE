# Paired-question source-objective pilot

This is a separate experiment after the 1,600-update source arms and 800-update
fusion comparison failed their source gate. Those runs, their selection rule,
and their rejection remain unchanged.

The diagnostic compares two contrasting questions about the same passage.
It measures the model's preference for assigning the two annotated answer
starts to their own questions rather than swapping them. This diagnostic
restricts choices using annotations; it is not ordinary source-selection
accuracy. All ordinary metrics still score every source position.

The previous structure encoder fit all 64 training pairs in each seed. On
development pairs, correct assignments changed from 23/32 to 19/32 in seed 47
and remained 18/32 in seed 83. Neither final encoder answered both starts
correctly in any development pair. Question sensitivity on training passages
therefore did not establish transferable relation learning.

The pilot tests one supervision change: a loss preferring the correct paired
assignment over its swap. Its margin subtracts question-independent source
salience. It uses only the existing training questions and their exact gold
starts; neither names nor answer patterns are added to the scorer.

Each arm starts from the same seed's original structure-arm initial checkpoint,
not its rejected final checkpoint. Both arms receive the same 400-update
schedule, paired forwards, source supervision, replay protection, optimizer,
and frozen endpoint/emission/backbone parameters. Each of the eight new-question
slots averages the source losses of that question and its partner. Six replay
slots preserve the original 2/2/2 balance. The experimental arm additionally
adds the mean paired-assignment softplus loss at weight one. No architecture,
tokenizer, memory, inference or accepted-checkpoint changes are made.

The protocol, code, inputs, packet hashes, starting checkpoints and runtime
are frozen before optimization. Compare final updates only. A useful pilot
signal requires both seeds to improve development start accuracy by at least
4/64 and paired assignment accuracy by at least 4/32 over matched control,
while preserving all 16 fresh starts. This is not model acceptance: the full
source, extent, raw-answer and retention gates still apply. No extension or
locked-test inference is authorized by the pilot protocol.

Run from this directory with a torch-enabled Python, UTF-8 and bytecode disabled:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -X utf8 -B native_qa_paired_source.py diagnose
python -X utf8 -B native_qa_paired_source.py freeze
python -X utf8 -B native_qa_paired_source.py train --seed 47 --arm matched_control
python -X utf8 -B native_qa_paired_source.py train --seed 47 --arm paired_assignment
python -X utf8 -B native_qa_paired_source.py train --seed 83 --arm matched_control
python -X utf8 -B native_qa_paired_source.py train --seed 83 --arm paired_assignment
python -X utf8 -B native_qa_paired_source.py report
```

Outputs are under
`review_artifacts/native_qa_transfer_passages_v2/paired_source_pilot/`.
Each result includes all-source failure ledgers, paired diagnostics, frozen
parameter checks, exact checkpoint reload checks, live/cached source and
candidate parity across all 64 development questions, and ordinary greedy
generation reload checks. Raw-answer accuracy and full retention are not
certified by a source-only pilot.

## Observed result

Both arms fit all 128 training starts in both seeds and retained 16/16 fresh
starts. Development top-1 was 3/64 versus 3/64 in seed 47, and 4/64 versus
11/64 in seed 83 (matched control versus paired assignment). Top-16 recall
was 26/64 versus 26/64, and 35/64 versus 36/64 respectively.

The restricted correct-versus-swapped development assignment diagnostic fell
from 19/32 to 18/32 in seed 47 and from 17/32 to 15/32 in seed 83. Thus the
single-seed source improvement does not establish the hypothesized transferable
question-pair discrimination. The predeclared pilot signal failed in both seeds.

Across 10,000 paired bootstrap resamples of the 32 observed development source
groups, the top-1 accuracy difference interval was [0, 0] percentage points in
seed 47 and [3.125, 20.3125] points in seed 83. These are descriptive intervals
on inspected development groups, not locked-test transfer estimates; missing
parent provenance remains a limitation. See `uncertainty.json` for the precise
procedure and machine-readable values.

All four checkpoint reloads, all 256 live/cached development comparisons,
protected-parameter checks and 16 ordinary greedy reload comparisons passed.
The ten implementation/scoring tests passed. The artifact audit reverified the
original study's sealed outputs and both accepted checkpoint hashes. No model
was accepted, no endpoint/emission fitting followed, and locked-test predictions
remain zero.

Post-hoc branch attribution found standalone semantic top-1 of 3/64 in both
seed-47 arms and 2/64 versus 4/64 in seed 83. Legacy top-1 remained 2/64 and
5/64 respectively. Seed 83's 11/64 fused result therefore exceeds either
standalone branch, but does not establish a broadly transferable semantic
selector. These branch diagnostics did not select a checkpoint or train a new
arm. Their complete input-only score decisions are in `branch_diagnostics.json`.
