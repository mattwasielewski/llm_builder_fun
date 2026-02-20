#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text corpus.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/processed/corpus.txt"),
        help="Path to text corpus file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tokenizer"),
        help="Directory where tokenizer files are saved.",
    )
    parser.add_argument("--vocab-size", type=int, default=8000, help="Tokenizer vocabulary size.")
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency for BPE merges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Missing corpus file: {args.input_file}")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.train(files=[str(args.input_file)], trainer=trainer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "tokenizer.json"
    metadata_path = args.output_dir / "tokenizer_meta.json"

    tokenizer.save(str(tokenizer_path))

    metadata = {
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {
            token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Final vocab size: {metadata['vocab_size']}")


if __name__ == "__main__":
    main()
