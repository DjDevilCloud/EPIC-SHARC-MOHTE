# NVIDIA tiny prototype

These checkpoints are small diagnostic runs, not general-purpose models. The corrected one-record control confirms that the model can learn an answer span and stop cleanly. The wider ReasoningOff probe still has weak held-out quality, so more data and optimization are needed before calling it a useful model.

## Current findings

- `run11_multiline_framing/model.pt` is the best boundary-control checkpoint. It trained 500 steps on the short NVIDIA math sample (32-wide model, 271,283 parameters; final CE 0.30; best total loss -0.06). With the default repetition settings and deterministic decoding, a 64-token budget returns `\\boxed{60}` and stops after 9 generated tokens. It no longer loops. It closes before reproducing the requested second line, `#### 60`, so this is a successful answer/termination smoke test, not exact instruction following.
- `run10_reasoningoff_aligned_512/model.pt` is the ReasoningOff chat-parser scale test on 512 streaming windows from `D:\AAISourceStart\SourceALLM\Datasets\NVIDIA\ReasoningOff\reasoning_offNVIDIA.jsonl`. It used the default-width architecture (128 model width, 256 emitters, 128 slots, 6×6×6 torus), 800 steps, batch 1, sequence length 128, and 15% record holdout. Training average CE was 3.35; held-out CE was 5.11. Step peak allocated VRAM was about 7.1 GB and reserved was about 7.6 GB, below the 12 GB GPU limit. A prompt from the sampled training portion terminated, but its answer was only a short fragment; broad quality is not solved.
- The ReasoningOff reader now converts its `messages` array to `<BOI>…<EOI>` input turns and `<BOO>…<EOO>` assistant targets. Earlier `run7` treated the array as Python dictionary text and is invalid. `run8` used the corrected reader but had the first `<LINE>` framing misaligned; those diagnostic checkpoints were removed.
- Output supervision learns `<EOL>` and `<EOO>`. The first `<LINE>` is seeded as contextual input during generation; later `<LINE>` frames are learned after newlines. Focused boundary and NVIDIA chat parsing tests pass.
- Checkpoints `run1`–`run5` preserve the earlier NVIDIAPT/General probes. The active checkpoint set is kept at eight model files; no copy of the 5.96 GB ReasoningOff source data is stored in this workspace.

## Reproduce the boundary control

From the repository root:

```powershell
python cli.py train --data checkpoints/nvidia_tiny_prototype/short_sample.jsonl --save-dir checkpoints/nvidia_tiny_prototype/repro --steps 500 --batch-size 1 --seq-len 512 --max-samples 0 --val-fraction 0 --lr 0.0008 --d-model 32 --n-layers 1 --n-emitters 16 --n-slots 16 --top-k-emitters 2 --top-k-slots 2 --ff-mult 2 --torus-depth 2 --torus-height 2 --torus-width 2 --position-embedding-init-size 512 --no-torch-compile
```

For the short sample, use its exact user text with `--max-new-tokens 64 --temperature 0 --no-torch-compile --no-speculative-decoding`. The default repetition penalty and no-repeat-ngram settings are retained in the verified run.
