"""Unit tests for Dataset loaders, sequence chunking, and text cleaning."""

import numpy as np
import torch
import pytest
from llm.dataset import (
    clean_text,
    CausalLanguageModelingDataset,
    create_dataloader,
)


def test_clean_text():
    raw = "  Hello   world! \r\n\r\n\r\n\r\nThis is   a test.  \n"
    cleaned = clean_text(raw, min_chars=5)
    assert cleaned is not None
    assert "\r" not in cleaned
    assert "  " not in cleaned  # no redundant spaces
    assert "Hello world!" in cleaned
    assert "This is a test." in cleaned

    # Too short
    short = "hi"
    assert clean_text(short, min_chars=10) is None


def test_causal_lm_dataset_memmap(tmp_path):
    # Create 500 integer tokens
    tokens = np.arange(500, dtype=np.uint16)
    bin_file = tmp_path / "test_tokens.bin"
    tokens.tofile(bin_file)

    context_length = 32
    # chunk_size = context_length + 1 = 33
    dataset = CausalLanguageModelingDataset(bin_file, context_length=context_length, dtype=np.uint16)

    expected_samples = 500 // 33
    assert len(dataset) == expected_samples

    sample0 = dataset[0]
    x = sample0["input_ids"]
    y = sample0["targets"]

    assert x.shape == (context_length,)
    assert y.shape == (context_length,)
    assert torch.equal(x, torch.arange(0, 32, dtype=torch.long))
    assert torch.equal(y, torch.arange(1, 33, dtype=torch.long))


def test_dataloader_creation(tmp_path):
    tokens = np.arange(1000, dtype=np.uint16)
    bin_file = tmp_path / "tokens.bin"
    tokens.tofile(bin_file)

    dataset = CausalLanguageModelingDataset(bin_file, context_length=16, dtype=np.uint16)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    assert batch["input_ids"].shape == (4, 16)
    assert batch["targets"].shape == (4, 16)
