# Getting Started

This guide gets you from zero to your first custom language model run.

## 1. Prerequisites

### Required tools (and where to get them)

- Python 3.10+ from [python.org](https://www.python.org/downloads/)
- `pip` (bundled with modern Python)
- Git from [git-scm.com](https://git-scm.com/downloads)

### Optional but recommended

- NVIDIA GPU + CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/)
- Apple Silicon users can use PyTorch MPS backend (also from [pytorch.org](https://pytorch.org/get-started/locally/))

## 2. Set up environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Add your training data

1. Place one or more `.txt` files in `data/raw/`.
2. Merge them into a single corpus file:

```bash
python scripts/prepare_corpus.py --input-dir data/raw --output-file data/processed/corpus.txt
```

## 4. Train tokenizer

```bash
python scripts/train_tokenizer.py \
  --input-file data/processed/corpus.txt \
  --output-dir artifacts/tokenizer \
  --vocab-size 8000
```

Output files:
- `artifacts/tokenizer/tokenizer.json`
- `artifacts/tokenizer/tokenizer_meta.json`

## 5. Train language model

```bash
python scripts/train.py --config configs/base.yaml
```

The best checkpoint is saved to:
- `artifacts/checkpoints/model.pt`

## 6. Generate text

```bash
python scripts/generate.py \
  --checkpoint artifacts/checkpoints/model.pt \
  --prompt "Write a short product announcement about AI agents" \
  --max-new-tokens 180 \
  --temperature 0.8 \
  --top-k 40
```

## 7. Tune for better results

Edit `configs/base.yaml`:
- Increase `model.n_layer`, `model.n_embd`, or `training.max_steps` for stronger capacity.
- Lower `training.block_size` if you have limited memory.
- Increase data volume and quality for the biggest improvement.

## Troubleshooting

- `Tokenizer not found`: run `scripts/train_tokenizer.py` first.
- `Corpus not found`: run `scripts/prepare_corpus.py` first.
- `Not enough tokens`: add more text files or reduce `training.block_size`.
- GPU not used: set `training.device: cuda` in `configs/base.yaml` and verify CUDA PyTorch install.
