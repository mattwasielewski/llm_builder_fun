#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge .txt files into a single training corpus.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Folder with source .txt files.")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/corpus.txt"),
        help="Path to merged corpus text file.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Skip files with fewer characters than this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    txt_files = sorted(path for path in args.input_dir.rglob("*.txt") if path.is_file())
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {args.input_dir}. Put text files there and rerun."
        )

    chunks: list[str] = []
    kept_files = 0
    skipped_files = 0
    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < args.min_chars:
            skipped_files += 1
            continue
        chunks.append(text)
        kept_files += 1

    if not chunks:
        raise RuntimeError(
            "All files were skipped. Lower --min-chars or add larger text files."
        )

    merged = "\n\n".join(chunks)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(merged, encoding="utf-8")

    print(f"Merged corpus written to: {args.output_file}")
    print(f"Files used: {kept_files} | files skipped: {skipped_files}")
    print(f"Total characters: {len(merged):,}")


if __name__ == "__main__":
    main()
