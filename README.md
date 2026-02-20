# llm_builder

A practical starter project for building your own small decoder-only language model from scratch.

This repository includes:
- Data preparation for your text corpus.
- BPE tokenizer training.
- GPT-style decoder model training (PyTorch).
- Text generation from your trained checkpoint.
- Documentation for quick start and full tutorial workflow.

## Documentation

- Getting started: [GETTING_STARTED.md](GETTING_STARTED.md)
- Full tutorial: [docs/TUTORIAL.md](docs/TUTORIAL.md)
- Technical cheat sheet: [docs/LLM_CHEAT_SHEET.md](docs/LLM_CHEAT_SHEET.md)

## Quick command flow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_corpus.py --input-dir data/raw --output-file data/processed/corpus.txt
python scripts/train_tokenizer.py --input-file data/processed/corpus.txt --output-dir artifacts/tokenizer --vocab-size 8000
python scripts/train.py --config configs/base.yaml
python scripts/generate.py --checkpoint artifacts/checkpoints/model.pt --prompt "The future of AI is"
```

## Notes

- This is designed for learning and experimentation, not production-scale frontier model training.
- Model quality mostly depends on your dataset quality, size, and training budget.
