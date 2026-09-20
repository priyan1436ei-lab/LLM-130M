"""Script to clean, tokenize, chunk, and save binary datasets for Privan-130M."""

import sys
import json
import argparse
import random
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.tokenizer import ByteLevelBPETokenizer
from llm.dataset import load_documents_from_file


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare pre-tokenized binary datasets for training")
    parser.add_argument("--input", type=str, default="data/raw", help="Raw data directory or file")
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer", help="Trained tokenizer directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for binary files")
    parser.add_argument("--train-ratio", type=float, default=0.98, help="Train split fraction")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation split fraction")
    parser.add_argument("--test-ratio", type=float, default=0.01, help="Test split fraction")
    parser.add_argument("--text-field", type=str, default="text", help="JSONL/Parquet text field")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    return parser.parse_args()


def write_tokens_to_bin(token_list: List[int], file_path: Path, dtype=np.uint16):
    """Write list of integer token IDs directly to disk as binary buffer."""
    arr = np.array(token_list, dtype=dtype)
    with open(file_path, "wb") as f:
        f.write(arr.tobytes())


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("DATA PREPARATION & TOKENIZATION PIPELINE")
    print("=" * 50)
    print(f"Input:         {input_path}")
    print(f"Tokenizer dir: {tokenizer_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Split ratios:  Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")

    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer.load(tokenizer_dir)
    eos_token_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size()
    dtype = np.uint16 if vocab_size <= 65535 else np.int32
    print(f"Loaded tokenizer with vocab size {vocab_size}. Using dtype: {dtype.__name__}")

    # Gather files
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("**/*.txt")) + list(input_path.glob("**/*.jsonl")) + list(input_path.glob("**/*.parquet"))
    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    all_docs: List[str] = []
    print("Loading and cleaning documents...")
    for f in files:
        docs = load_documents_from_file(f, text_field=args.text_field)
        all_docs.extend(docs)

    num_docs = len(all_docs)
    if num_docs == 0:
        raise ValueError("No valid documents found after cleaning.")

    print(f"Total cleaned documents: {num_docs}")

    # Document-level shuffle and split to prevent cross-document contamination
    random.shuffle(all_docs)

    train_end = max(1, int(num_docs * args.train_ratio))
    if train_end >= num_docs and num_docs > 2:
        train_end = num_docs - 2

    val_end = train_end + max(1, int(num_docs * args.val_ratio))
    if val_end >= num_docs and num_docs > 1:
        val_end = num_docs - 1

    train_docs = all_docs[:train_end]
    val_docs = all_docs[train_end:val_end]
    test_docs = all_docs[val_end:]

    # Fallback if val or test is empty due to small dataset
    if not val_docs:
        val_docs = train_docs[-1:]
    if not test_docs:
        test_docs = train_docs[-1:]

    splits = {
        "train": (train_docs, output_dir / "train.bin"),
        "val": (val_docs, output_dir / "val.bin"),
        "test": (test_docs, output_dir / "test.bin"),
    }

    metadata = {
        "vocab_size": vocab_size,
        "dtype": dtype.__name__,
        "total_documents": num_docs,
        "splits": {},
    }

    for split_name, (docs, out_file) in splits.items():
        print(f"Tokenizing {split_name} split ({len(docs)} documents)...")
        token_ids: List[int] = []
        char_count = 0

        for doc in tqdm(docs, desc=f"Processing {split_name}"):
            char_count += len(doc)
            encoded = tokenizer.encode(doc)
            token_ids.extend(encoded)
            token_ids.append(eos_token_id)  # Separate documents with EOS

        write_tokens_to_bin(token_ids, out_file, dtype=dtype)
        metadata["splits"][split_name] = {
            "num_documents": len(docs),
            "num_tokens": len(token_ids),
            "char_count": char_count,
            "file_path": str(out_file),
            "file_size_bytes": out_file.stat().st_size,
        }
        print(f"  -> {split_name}: {len(token_ids):,} tokens saved to {out_file}")

    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 50)
    print("Dataset preparation complete! Metadata saved.")
    print("=" * 50)


if __name__ == "__main__":
    main()
