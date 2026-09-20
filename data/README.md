# Privan-130M Dataset Guidelines & Storage

This directory manages datasets for training and evaluating Privan-130M.

## Directory Structure
- `data/raw/`: Place raw `.txt`, `.jsonl`, or `.parquet` files here.
- `data/processed/`: Binary memory-mapped tokenized chunks (`train.bin`, `val.bin`, `test.bin`).
- `data/tokenizer/`: Trained BPE tokenizer model files (`vocab.json`, `merges.txt`, `tokenizer.json`).

## Dataset License Compliance & Ethics
Privan-130M strictly adheres to ethical open-source AI principles:
- **No unauthorized scraping**: Do not download copyrighted text or proprietary corpora without explicit authorization.
- **Permissible Sources**:
  - Public domain corpora (e.g., Project Gutenberg, Open Science text).
  - Openly licensed datasets (e.g., CC-BY, CC0, Apache-2.0, MIT, FineWeb / C4 subsets).
  - Synthetic data created by authorized models or rule-based generators.
  - User-owned data.

## Data Preparation Pipeline
1. Place raw text files into `data/raw/` (e.g. `corpus.txt` or `corpus.jsonl`).
2. Train the BPE tokenizer:
   ```bash
   python scripts/train_tokenizer.py --input data/raw --output data/tokenizer --vocab-size 50257
   ```
3. Prepare and chunk the data:
   ```bash
   python scripts/prepare_data.py --input data/raw --tokenizer-dir data/tokenizer --output-dir data/processed
   ```
4. Verify dataset quality and statistics:
   ```bash
   python scripts/data_report.py --data-dir data/processed
   ```
