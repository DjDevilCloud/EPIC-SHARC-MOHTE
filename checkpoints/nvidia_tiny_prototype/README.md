# NVIDIA tiny prototype

This folder contains a one-record probe extracted from `NVIDIAPT/General/part_000000.parquet`, plus checkpoints and metrics. It is a smoke-scale proof of training and output, not a useful general model.

## Results

- `run3/model.pt` is the usable tiny prototype: 300 steps on the short math record, 271,283 parameters, final loss 0.543 (best 0.520), final CE 0.338. The model emitted `\boxed{60}` for the matching prompt with deterministic six-token decoding.
- `run4_default_width/model.pt` is a 20-step architecture sizing pass with the configured defaults retained: width 128, 256 emitters, 128 slots, 6×6×6 torus, GATE/full GATE, lattice attention, mixture and leaf/family paths enabled. Loss fell from 8.70 on the first batch to 4.42 at step 20. It has 1,163,179 parameters.
- `run5_general_64/model.pt` is the broader 64-window probe using the same default-width architecture against the NVIDIA General parquet sample, with a 15% record holdout. After 120 steps, train loss was 3.35 (best 1.92) and held-out loss was 4.28 / CE 4.20. It has 2,944,257 parameters and is the more useful starting checkpoint for continued training on real samples.
- Recorded training-step allocated VRAM peaks were about 151 MB for `run3` and 183 MB for `run4_default_width`; the GPU was otherwise using roughly 1.5 GB for the desktop.
- `run5_general_64` recorded a 73 MB allocated training-step peak and about 130 MB reserved.
- `run3` is an exact-prompt overfit check and has no validation score (`val_fraction=0`). `run5_general_64` supplies the held-out validation result above.

## Reproduce the tiny run

From the repository root:

```powershell
python cli.py train --data checkpoints/nvidia_tiny_prototype/short_sample.jsonl --save-dir checkpoints/nvidia_tiny_prototype/repro --steps 300 --batch-size 1 --seq-len 512 --max-samples 0 --val-fraction 0 --lr 0.0008 --d-model 32 --n-layers 1 --n-emitters 16 --n-slots 16 --top-k-emitters 2 --top-k-slots 2 --ff-mult 2 --torus-depth 2 --torus-height 2 --torus-width 2 --position-embedding-init-size 512 --no-torch-compile
```

For this one-record probe, use the exact user prompt and `--max-new-tokens 6 --temperature 0 --no-torch-compile --no-speculative-decoding` to reproduce `\boxed{60}`. Longer decoding continues past the short answer, which is expected for this tiny overfit checkpoint.
