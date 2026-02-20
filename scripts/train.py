#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_builder.data import encode_text, get_batch, load_corpus, split_train_val
from llm_builder.model import GPTConfig, GPTModel
from llm_builder.utils import choose_device, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decoder-only language model.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Training config YAML path.")
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: GPTModel,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}

    for split_name, split_ids in (("train", train_ids), ("val", val_ids)):
        losses = torch.zeros(eval_batches)
        for i in range(eval_batches):
            x, y = get_batch(split_ids, batch_size, block_size, device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Loss should never be None during evaluation.")
            losses[i] = loss.item()
        out[split_name] = float(losses.mean().item())

    model.train()
    return out


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    tokenizer_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]

    device = choose_device(str(training_cfg.get("device", "auto")))
    print(f"Using device: {device}")

    tokenizer_path = Path(tokenizer_cfg["path"])
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. Run scripts/train_tokenizer.py first."
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    corpus_path = Path(data_cfg["corpus_path"])
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}. Run scripts/prepare_corpus.py first."
        )

    text = load_corpus(corpus_path)
    token_ids = encode_text(tokenizer, text)
    print(f"Total tokens in corpus: {len(token_ids):,}")

    train_split = float(data_cfg.get("train_split", 0.9))
    train_ids, val_ids = split_train_val(token_ids, train_split)

    block_size = int(training_cfg["block_size"])
    if len(train_ids) <= block_size + 1 or len(val_ids) <= block_size + 1:
        raise ValueError(
            "Not enough tokens for training/validation. Use more text data or reduce block_size."
        )

    gpt_cfg = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=block_size,
        n_embd=int(model_cfg["n_embd"]),
        n_head=int(model_cfg["n_head"]),
        n_layer=int(model_cfg["n_layer"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    model = GPTModel(gpt_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    learning_rate = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    max_steps = int(training_cfg["max_steps"])
    batch_size = int(training_cfg["batch_size"])
    eval_interval = int(training_cfg["eval_interval"])
    eval_batches = int(training_cfg.get("eval_batches", 20))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))

    checkpoint_path = Path(training_cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_path.with_suffix(".meta.json")

    best_val_loss = math.inf

    for step in trange(1, max_steps + 1, desc="Training"):
        x, y = get_batch(train_ids, batch_size, block_size, device)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Loss should never be None during training.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        should_eval = step == 1 or step == max_steps or step % eval_interval == 0
        if should_eval:
            losses = estimate_loss(
                model=model,
                train_ids=train_ids,
                val_ids=val_ids,
                batch_size=batch_size,
                block_size=block_size,
                eval_batches=eval_batches,
                device=device,
            )
            train_loss = losses["train"]
            val_loss = losses["val"]
            print(
                f"\nstep={step:6d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_ppl={math.exp(val_loss):.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                payload = {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(gpt_cfg),
                    "tokenizer_path": str(tokenizer_path),
                    "step": step,
                    "val_loss": val_loss,
                }
                torch.save(payload, checkpoint_path)

                metadata = {
                    "checkpoint_path": str(checkpoint_path),
                    "step": step,
                    "val_loss": val_loss,
                    "val_perplexity": math.exp(val_loss),
                    "tokenizer_path": str(tokenizer_path),
                    "model_config": asdict(gpt_cfg),
                }
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                print(f"Saved checkpoint: {checkpoint_path}")

    print("Training completed.")
    if checkpoint_path.exists():
        print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
