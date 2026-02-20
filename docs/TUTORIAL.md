# Tutorial: Build Your Own Small LLM

This tutorial explains exactly what you need, what tools are expected, where they come from, and how to train/use your model in this project.

## What you are building

You will train a small GPT-style decoder model from scratch on your own text corpus.

Pipeline:
1. Collect domain text data.
2. Merge into one corpus.
3. Train a BPE tokenizer.
4. Train a causal language model.
5. Generate text and iterate.

This project is for educational and practical prototyping. It is not intended to replicate large closed-source frontier models.

## Tooling you need

### Core tools

- Python 3.10+ from [python.org](https://www.python.org/downloads/)
- Git from [git-scm.com](https://git-scm.com/downloads)
- PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/)
- Hugging Face `tokenizers` from [pypi.org/project/tokenizers](https://pypi.org/project/tokenizers/)
- PyYAML from [pypi.org/project/PyYAML](https://pypi.org/project/PyYAML/)
- tqdm from [pypi.org/project/tqdm](https://pypi.org/project/tqdm/)

### Hardware expectations

- Minimum: modern CPU with 16GB RAM (small experiments).
- Recommended: NVIDIA GPU with 8GB+ VRAM for faster training.
- Apple Silicon works via MPS backend.

## Project layout

- `scripts/prepare_corpus.py`: merges `.txt` files from `data/raw`.
- `scripts/train_tokenizer.py`: trains BPE tokenizer.
- `scripts/train.py`: trains GPT-style decoder.
- `scripts/generate.py`: runs text generation.
- `configs/base.yaml`: training, model, and generation settings.
- `artifacts/`: saved tokenizer and checkpoints.

## Step 1: Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Prepare your dataset

### Data format expected

- Plain UTF-8 text files (`.txt`).
- Place them in `data/raw/`.
- More high-quality domain text gives better results than random mixed text.

### Example dataset sources (you choose)

- Your own documents and support articles.
- Public-domain text corpora.
- Internal docs (if you have rights to use them).

Important:
- Use only data you are legally allowed to train on.
- Remove sensitive/private information before training.

### Merge data

```bash
python scripts/prepare_corpus.py --input-dir data/raw --output-file data/processed/corpus.txt
```

## Step 3: Train tokenizer

```bash
python scripts/train_tokenizer.py \
  --input-file data/processed/corpus.txt \
  --output-dir artifacts/tokenizer \
  --vocab-size 8000 \
  --min-frequency 2
```

Guidance:
- `vocab-size 4000-8000`: smaller domain corpus.
- `vocab-size 16000+`: larger, diverse corpus.

## Step 4: Configure training

Edit `configs/base.yaml`.

Key fields:
- `model.n_layer`: number of transformer blocks.
- `model.n_embd`: embedding width.
- `model.n_head`: attention heads.
- `training.block_size`: max context window during training.
- `training.max_steps`: total optimization steps.
- `training.batch_size`: micro-batch size.
- `training.device`: `auto`, `cpu`, `cuda`, or `mps`.

## Step 5: Train the model

```bash
python scripts/train.py --config configs/base.yaml
```

During training you will see:
- `train_loss`
- `val_loss`
- `val_ppl` (perplexity estimate)

The script saves the best validation checkpoint to:
- `artifacts/checkpoints/model.pt`

## Step 6: Generate text

```bash
python scripts/generate.py \
  --checkpoint artifacts/checkpoints/model.pt \
  --prompt "Create a concise onboarding checklist for an ML platform engineer" \
  --max-new-tokens 200 \
  --temperature 0.7 \
  --top-k 40
```

Sampling controls:
- Lower `temperature` (0.6-0.8): more deterministic output.
- Higher `temperature` (0.9-1.1): more diverse output.
- Lower `top-k`: safer but narrower token choices.

## Step 7: Improve quality

Most improvements come from these, in order:
1. Better and larger training corpus.
2. More training steps.
3. Larger model (if hardware allows).
4. Better cleaning and deduplication.

## Typical next upgrades

- Add checkpoint resume support.
- Add gradient accumulation for bigger effective batch size.
- Add mixed precision training.
- Add instruction fine-tuning dataset format.
- Add evaluation scripts for benchmark tasks.

## Common errors

- `Not enough tokens for training/validation`:
  - Add more text data or reduce `training.block_size`.
- Out-of-memory on GPU:
  - Reduce `training.batch_size`, `model.n_embd`, or `model.n_layer`.
- Generated text is repetitive:
  - Increase data quality/diversity and adjust `temperature` / `top-k`.

## Reality check

Training a useful model from scratch requires significant data and compute. This project gives you a strong foundation and a transparent training pipeline you can evolve.
