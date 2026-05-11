# Pretokenize Demo

This folder ships the tiny release-safe corpus used by the quickstart, so the repo works without any extra setup.

## What To Put Here

The shipped sample corpus lives here:

```text
demo/
  corpus/
    tiny_example.txt
```

If you want to swap in your own data, keep the same folder layout:

```text
demo/
  corpus/
    sample1.txt
    sample2.txt
```

The files can be plain text, Markdown, JSONL, or Parquet. For a first test, plain `.txt` files are the easiest.

## Quick Demo

Run training directly on the corpus folder:

```bash
python cli.py train --data demo/corpus --save-dir checkpoints/demo
```

Then try inference with the saved checkpoint:

```bash
python cli.py infer --checkpoint checkpoints/demo/model.pt --prompt "Explain the torus field"
```

## Optional Pretokenize Step

If you want a pretokenized memmap dataset for faster repeated runs, you can generate one from the same corpus:

```bash
python pretokenize.py --data demo/corpus --output-dir demo/pretokenized --seq-len 512 --max-samples 64
```

After that, point training at the pretokenized output if that fits your workflow better.

## Notes

- The shipped sample dataset is `demo/corpus/tiny_example.txt`.
- The UI and CLI both work with a folder of raw text files.
- Pretokenization is optional, not required.
