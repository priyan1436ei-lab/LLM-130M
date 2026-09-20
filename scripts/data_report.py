"""Script to generate a comprehensive Data Quality and Statistics Report."""

import sys
import json
import argparse
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.dataset import load_documents_from_file


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Data Quality and Statistics Report")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed binary directory")
    parser.add_argument("--text-field", type=str, default="text", help="Text column name for JSONL")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_path = Path(args.raw_dir)
    proc_path = Path(args.processed_dir)

    print("=" * 50)
    print("DATA QUALITY & STATISTICS REPORT")
    print("=" * 50)

    # 1. Inspect Raw Data
    total_docs = 0
    total_chars = 0
    min_len = float("inf")
    max_len = 0
    duplicates = 0
    seen_hashes = set()

    raw_files = []
    if raw_path.exists():
        if raw_path.is_file():
            raw_files = [raw_path]
        else:
            raw_files = list(raw_path.glob("**/*.txt")) + list(raw_path.glob("**/*.jsonl")) + list(raw_path.glob("**/*.parquet"))

    if raw_files:
        for f in raw_files:
            docs = load_documents_from_file(f, text_field=args.text_field, deduplicate=False)
            for d in docs:
                total_docs += 1
                length = len(d)
                total_chars += length
                if length < min_len:
                    min_len = length
                if length > max_len:
                    max_len = length

                h = hash(d)
                if h in seen_hashes:
                    duplicates += 1
                else:
                    seen_hashes.add(h)

        avg_len = (total_chars / total_docs) if total_docs > 0 else 0
        min_len_val = min_len if min_len != float("inf") else 0
    else:
        total_docs = 0
        total_chars = 0
        avg_len = 0
        min_len_val = 0
        max_len = 0
        duplicates = 0

    # 2. Inspect Processed Metadata
    meta_file = proc_path / "dataset_metadata.json"
    train_tokens = "NOT PREPARED"
    val_tokens = "NOT PREPARED"
    test_tokens = "NOT PREPARED"
    total_tokens = 0

    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        splits = meta.get("splits", {})
        if "train" in splits:
            cnt = splits["train"]["num_tokens"]
            train_tokens = f"{cnt:,}"
            total_tokens += cnt
        if "val" in splits:
            cnt = splits["val"]["num_tokens"]
            val_tokens = f"{cnt:,}"
            total_tokens += cnt
        if "test" in splits:
            cnt = splits["test"]["num_tokens"]
            test_tokens = f"{cnt:,}"
            total_tokens += cnt

    print(f"Total Documents:          {total_docs:,}")
    print(f"Total Characters:         {total_chars:,}")
    print(f"Average Document Length:  {avg_len:.1f} characters")
    print(f"Minimum Length:           {min_len_val:,} characters")
    print(f"Maximum Length:           {max_len:,} characters")
    print(f"Duplicate Documents:      {duplicates:,}")
    print("-" * 50)
    print(f"Train Tokens:             {train_tokens}")
    print(f"Validation Tokens:        {val_tokens}")
    print(f"Test Tokens:              {test_tokens}")
    if total_tokens > 0:
        print(f"Total Processed Tokens:   {total_tokens:,}")
    print("=" * 50)


if __name__ == "__main__":
    main()
