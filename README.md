# EPIC-SHARC MOHTE v0.1.2 - Updated resuable emitter branch at https://github.com/DjDevilCloud/EPIC-SHARC-MOHTE/tree/ESM-v0.1.2

Emitter Prismal Instructional Core with Signature-Hierarchy Attention Routing Cache + Mixture of Hierarchical Toroidal Experts.

This repository contains the EPIC-SHARC MOHTE standalone implementation of the routing, memory, and toroidal expert components.
A tiny demo corpus is bundled at [`demo/corpus/tiny_example.txt`](./demo/corpus/tiny_example.txt) so the quickstart runs out of the box. For real work, choose your own text, JSONL, Parquet, or Markdown corpus in the UI or pass it on the command line.

## Licensing

**EPIC-SHARC MOHTE** is source-available under the **GNU AGPLv3** for non-commercial use. Non-commercial use includes research, personal projects, academics, and non-profits.

Commercial use requires a paid license. Companies, SaaS products, internal tools, or any revenue-generating deployment must obtain a commercial license from the author.

See [LICENSE](./LICENSE), [COMMERCIAL.md](./COMMERCIAL.md), and [LICENSES.md](./LICENSES.md) for full details.

## Install

```bash
python -m pip install -r requirements.txt
```

The core runtime depends on `numpy` and `torch`. Optional data-path helpers can also use `pandas`, `pyarrow`, or `bitsandbytes` if you install them.

## Overview

The architecture is built around a hierarchy-aware tokenizer, a SHARC-style routing cache, torus memory, and selective expert paths. The boundary markers carry span structure for input/output segments and paragraph-like blocks:

- `<BOI>` and `<EOI>` mark input spans
- `<BOO>` and `<EOO>` mark output spans
- `<BOP>` and `<EOP>` mark paragraph or block boundaries
- `<BLO>`, `<LINE>`, and `<EOL>` annotate lower-level structural flow

The full flow looks like this:

```mermaid
flowchart LR
    T["Token stream"] --> H["Hierarchy encoder<br/>(tokens + signature tracks)"]
    H --> B["Aligned hierarchy bundle<br/>(token, family, signature, level, relation, parent)"]
    B --> M["Boundary markers<br/>(BOI/EOI, BOO/EOO, BOP/EOP, BLO, LINE, EOL)"]
    B --> C["SHARC cache / signature lattice"]
    C --> G{"Family gate active?"}
    G -->|Yes| S["Family specialist bank"]
    G -->|No| R["Recursive HMOE / main routing"]
    S --> R
    R --> X["MoT / torus expert routing"]
    X --> F["Torus field core<br/>(local field + global bus)"]
    F --> O["Output heads<br/>(logits, signature level/relation, route stats)"]
    O --> P["Generation or training loss"]
```

For the detailed architecture writeup, see [`ARCHITECTUREOVERVIEW.md`](./ARCHITECTUREOVERVIEW.md).

## Default Configuration

The default runtime configuration lives in [`config.py`](./config.py) via `PrismalWaveConfig`.

Key defaults:

- `d_model = 1024`
- `n_layers = 1`
- `n_emitters = 4096`
- `n_slots = 2048`
- `n_paths = 1`
- `use_factorized_embedding = true`
- `use_turbo_quantization = false`
- `use_torus_core = true`
- `use_hmote = true`
- `use_recursive_hmoe = true`
- `use_signature_lattice_attention = true`
- `use_signature_lattice_generation_cache = true`
- `use_torus_race_lanes = true`
- `use_speculative_decoding = true`

## Core Modules

The main code paths are:

- `./data.py` for hierarchy encoding and loss-mask construction
- `./model.py` for torus routing, lattice attention, and decoding
- `./train.py` for training, checkpoint loading, and prompt generation
- `./quantization.py` for cached TurboQuant wrappers

## Data Alignment

The model expects these tensors to stay aligned at the boundary:

- `input_ids`
- `signature_ids`
- `signature_level_ids`
- `signature_relation_ids`
- `parent_signature_ids`
- `signature_family_ids`
- `loss_mask`

If one of those drifts, the model raises immediately in `forward()` or `generate()` rather than silently training on misaligned data.

## Run It

```bash
python cli.py train --data <your-data-path> --save-dir checkpoints/demo
python cli.py infer --checkpoint checkpoints/demo/model.pt --prompt "Explain torus routing"
python cli.py benchmark --data <your-data-path>
python gui.py
```

## Try It In 60 Seconds

```bash
python cli.py train --data demo/corpus --save-dir checkpoints/tiny
python cli.py infer --checkpoint checkpoints/tiny/model.pt --prompt "Explain the torus core."
```

Expected result: a short training log, a saved checkpoint under `checkpoints/tiny/`, and a brief generated response from the prompt.

## Input Format

- Training data can be JSONL, Parquet, Markdown, plain text, or a dataset folder.
- Each record is converted into a hierarchical text window.
- The tokenizer can emit `<BOO>`, `<EOO>`, `<BOP>`, `<EOP>`, `<BLO>`, `<LINE>`, `<EOL>`, and `<SIG:OTHER>` special tokens.
- These markers add structure for blocks, paragraphs, and line boundaries, with `<SIG:OTHER>` covering fallback structural cases.
- The hierarchy encoder also produces aligned signature-family, signature-level, relation, and parent-ID tracks for every token.
- For a tiny local demo workflow, see [`demo/pretokenizedemo.md`](./demo/pretokenizedemo.md).
- The shipped sample corpus lives in [`demo/corpus/tiny_example.txt`](./demo/corpus/tiny_example.txt); you can point `train`, `benchmark`, or `pretokenize.py` at `demo/corpus/` directly.

## Code Map

If you want to inspect the implementation, start here:

- `./data.py`
- `./model.py`
- `./train.py`

