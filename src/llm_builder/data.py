from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer


def load_corpus(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as f:
        return f.read()


def encode_text(tokenizer: Tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text).ids
    return torch.tensor(ids, dtype=torch.long)


def split_train_val(token_ids: torch.Tensor, train_split: float) -> tuple[torch.Tensor, torch.Tensor]:
    split_idx = int(len(token_ids) * train_split)
    return token_ids[:split_idx], token_ids[split_idx:]


def get_batch(token_ids: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(token_ids) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            "Corpus is too small for the selected block_size. "
            "Use more data or lower training.block_size in configs/base.yaml."
        )

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([token_ids[int(i) : int(i) + block_size] for i in starts])
    y = torch.stack([token_ids[int(i) + 1 : int(i) + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
