"""Script to train Byte-Level BPE Tokenizer for Privan-130M."""

import sys
import argparse
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.tokenizer import ByteLevelBPETokenizer
from llm.dataset import load_documents_from_file


def parse_args():
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE Tokenizer")
    parser.add_argument("--input", type=str, default="data/raw", help="Directory or file containing raw data")
    parser.add_argument("--output", type=str, default="data/tokenizer", help="Directory to save tokenizer")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Target vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--text-field", type=str, default="text", help="JSONL/Parquet text column name")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("PRIVAN-130M TOKENIZER TRAINING")
    print("=" * 50)
    print(f"Input source:    {input_path}")
    print(f"Output dir:      {output_path}")
    print(f"Vocab size:      {args.vocab_size}")
    print(f"Min frequency:   {args.min_frequency}")

    # Gather files
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("**/*.txt")) + list(input_path.glob("**/*.jsonl")) + list(input_path.glob("**/*.parquet"))
    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    if not files:
        raise ValueError(f"No valid .txt, .jsonl, or .parquet files found in {input_path}")

    print(f"Found {len(files)} data file(s). Extracting normalized documents...")

    # We write normalized documents to a temporary text file for the tokenizer trainer
    temp_corpus = output_path / "temp_training_corpus.txt"
    total_docs = 0
    with open(temp_corpus, "w", encoding="utf-8") as out_f:
        for file in files:
            docs = load_documents_from_file(file, text_field=args.text_field)
            total_docs += len(docs)
            for doc in docs:
                out_f.write(doc + "\n\n")

    print(f"Prepared {total_docs} cleaned documents for BPE training.")

    # Train tokenizer
    print("Training Byte-Level BPE tokenizer...")
    tokenizer = ByteLevelBPETokenizer.train_from_files(
        files=[temp_corpus],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Clean up temporary corpus
    if temp_corpus.exists():
        temp_corpus.unlink()

    # Save tokenizer
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")

    # Test encode / decode
    test_phrase = "Privan-130M: A foundation language model built from scratch with PyTorch!"
    token_ids = tokenizer.encode(test_phrase)
    decoded_phrase = tokenizer.decode(token_ids)

    print("-" * 50)
    print("VERIFICATION ENCODE / DECODE TEST:")
    print(f"Original:  {test_phrase}")
    print(f"Token IDs: {token_ids[:15]}... ({len(token_ids)} tokens)")
    print(f"Decoded:   {decoded_phrase}")
    print("-" * 50)
    print(f"FINAL VOCABULARY SIZE: {tokenizer.vocab_size()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
