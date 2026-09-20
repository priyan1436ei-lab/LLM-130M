"""Dataset pipeline, cleaning, chunking, and memory-mapped loaders for Privan-130M."""

import os
import re
import json
import unicodedata
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


def clean_text(
    text: str,
    min_chars: int = 10,
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
) -> Optional[str]:
    """
    Configurable preprocessing for raw text.
    Removes invalid unicode, normalizes whitespace, preserves paragraph breaks.
    """
    if not text:
        return None

    # Normalize unicode
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
        # Remove non-printable / control characters (keep standard newlines and tabs)
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")

    # Normalize whitespace while preserving paragraphs
    if normalize_whitespace:
        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace 3 or more consecutive newlines with 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple horizontal spaces/tabs with a single space
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()

    if len(text) < min_chars:
        return None

    return text


def load_documents_from_file(
    file_path: str | Path,
    text_field: str = "text",
    min_chars: int = 10,
    deduplicate: bool = True,
) -> List[str]:
    """
    Load raw text documents from .txt, .jsonl, or .parquet.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    documents: List[str] = []
    seen_hashes = set()

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # If separated by double newlines or form feeds, split into document paragraphs
        raw_docs = [d.strip() for d in content.split("\n\n") if d.strip()] if "\n\n" in content else [content]
        for doc in raw_docs:
            cleaned = clean_text(doc, min_chars=min_chars)
            if cleaned:
                if deduplicate:
                    h = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                documents.append(cleaned)

    elif suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    raw_text = data.get(text_field, "")
                    cleaned = clean_text(raw_text, min_chars=min_chars)
                    if cleaned:
                        if deduplicate:
                            h = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                            if h in seen_hashes:
                                continue
                            seen_hashes.add(h)
                        documents.append(cleaned)
                except json.JSONDecodeError:
                    continue

    elif suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        if text_field in df.columns:
            for raw_text in df[text_field].dropna():
                cleaned = clean_text(str(raw_text), min_chars=min_chars)
                if cleaned:
                    if deduplicate:
                        h = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)
                    documents.append(cleaned)
        else:
            raise ValueError(f"Column '{text_field}' not found in {path}. Available: {list(df.columns)}")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .txt, .jsonl, .parquet")

    return documents


class CausalLanguageModelingDataset(Dataset):
    """
    Dataset for Causal Language Modeling supporting:
    - Memory-mapped binary token files (uint16 / int32)
    - In-memory PyTorch tensors or NumPy arrays
    - Sequence packing into fixed context_length chunks
    """

    def __init__(
        self,
        data_source: str | Path | np.ndarray | torch.Tensor,
        context_length: int = 1024,
        dtype: np.dtype = np.uint16,
    ):
        self.context_length = context_length
        self.chunk_size = context_length + 1  # input + 1 target token

        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"Binary token file not found at: {path}")

            # Determine size
            file_size_bytes = os.path.getsize(path)
            itemsize = np.dtype(dtype).itemsize
            num_tokens = file_size_bytes // itemsize

            # Open memory-mapped array in read-only mode
            self.tokens = np.memmap(path, dtype=dtype, mode="r", shape=(num_tokens,))
            self.total_tokens = int(num_tokens)
        elif isinstance(data_source, np.ndarray):
            self.tokens = data_source
            self.total_tokens = len(data_source)
        elif isinstance(data_source, torch.Tensor):
            self.tokens = data_source.cpu().numpy()
            self.total_tokens = len(self.tokens)
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")

        # Number of samples is total_tokens // chunk_size
        self.num_samples = self.total_tokens // self.chunk_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples")

        start = idx * self.chunk_size
        end = start + self.chunk_size

        chunk = self.tokens[start:end].astype(np.int64)

        # input tokens: [0 : context_length]
        # target tokens: [1 : context_length + 1]
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return {"input_ids": x, "targets": y}


class StreamingCausalLMDataset(torch.utils.data.IterableDataset):
    """
    Streaming IterableDataset for online or out-of-core token streaming.
    """

    def __init__(self, token_iterator_fn, context_length: int = 1024):
        super().__init__()
        self.token_iterator_fn = token_iterator_fn
        self.context_length = context_length
        self.chunk_size = context_length + 1

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        for token_id in self.token_iterator_fn():
            buffer.append(token_id)
            if len(buffer) >= self.chunk_size:
                chunk = np.array(buffer[:self.chunk_size], dtype=np.int64)
                x = torch.from_numpy(chunk[:-1])
                y = torch.from_numpy(chunk[1:])
                buffer = buffer[self.chunk_size:]
                yield {"input_ids": x, "targets": y}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = 2,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    """Create a DataLoader with production optimizations."""
    dataloader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    if sampler is not None:
        dataloader_kwargs["sampler"] = sampler
        dataloader_kwargs["shuffle"] = False
    else:
        dataloader_kwargs["shuffle"] = shuffle

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**dataloader_kwargs)
