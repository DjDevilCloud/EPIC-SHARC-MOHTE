# EPIC-SHARC MOHTE v0.1.1.

This project is my attempt at improving the current AI architecture paradigm of RNN, Transformer and Mamba stacks. It is an early research prototype but is fully trainable.

Emitter Prismal Instructional Core with Signature-Hierarchy Attention Routing Cache + Mixture of Hierarchical Toroidal Experts.

This repository contains the EPIC-SHARC MOHTE standalone implementation of the routing, memory, and toroidal expert components.
No dataset is bundled. Choose your own text, JSONL, Parquet, or Markdown corpus in the UI or pass it on the command line.

## Licensing

**EPIC-SHARC MOHTE** is source-available under the **GNU AGPLv3** for non-commercial use. Non-commercial use includes research, personal projects, academics, and non-profits.

Commercial use requires a paid license. Companies, SaaS products, internal tools, or any revenue-generating deployment must obtain a commercial license from the author.

See [LICENSE](./LICENSE), [COMMERCIAL.md](./COMMERCIAL.md), and [LICENSES.md](./LICENSES.md) for full details.

## Install

```bash
python -m pip install -e .
```

That gives you editable installs plus the `epic-sharc-mohte` and `epic-sharc-mohte-gui` entry points.

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
**NOTE** v0.1.1 added token cross-attention. This has not been updated in all of the documents yet.
For the detailed architecture writeup, see [`ARCHITECTUREOVERVIEW.md`](./ARCHITECTUREOVERVIEW.md).

## Default Configuration

The default runtime configuration lives in [`config.py`](./config.py) via `PrismalWaveConfig`.
NOTE: The default config is turned down on purpose to be lighter, you will want to turn up hmote_depth, hmote_branching, hierarchical_nest_depth, and recursive_hmoe_depth for the deeper levels. You may want to add more emitters and slots if you have the room. The architecture is highly configurable. On small datasets I have been able to hit almost 3B parameters full training on a single 12GB 40 series NVIDIA card.

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
python cli.py train --data demo/corpus/tiny_example.txt --save-dir checkpoints/tiny
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

## Code Map

If you want to inspect the implementation, start here:

- `./data.py`
- `./model.py`
- `./train.py`

