#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_builder.model import GPTConfig, GPTModel
from llm_builder.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/model.pt"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt text.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=150, help="How many tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling limit.")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | mps | auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run scripts/train.py first."
        )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg = GPTConfig(**checkpoint["model_config"])

    tokenizer_path = Path(checkpoint["tokenizer_path"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    model = GPTModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        bos_id = tokenizer.token_to_id("[BOS]")
        prompt_ids = [bos_id if bos_id is not None else 0]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Generated ===")
    print(output_text)


if __name__ == "__main__":
    main()
