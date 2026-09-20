"""Unit tests for ByteLevelBPETokenizer."""

import tempfile
from pathlib import Path
import pytest
from llm.tokenizer import ByteLevelBPETokenizer


@pytest.fixture
def sample_tokenizer(tmp_path):
    corpus_file = tmp_path / "sample.txt"
    corpus_file.write_text(
        "Privan is a modern transformer model.\n"
        "It is trained from scratch using PyTorch.\n"
        "Language models predict the next token given previous context.\n",
        encoding="utf-8"
    )

    tok = ByteLevelBPETokenizer.train_from_files(
        files=[corpus_file],
        vocab_size=300,
        min_frequency=1,
    )
    return tok, tmp_path


def test_encode_decode_roundtrip(sample_tokenizer):
    tok, _ = sample_tokenizer
    text = "Privan is a modern transformer model."
    ids = tok.encode(text)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)

    decoded = tok.decode(ids)
    assert decoded == text


def test_special_tokens(sample_tokenizer):
    tok, _ = sample_tokenizer
    assert tok.pad_token_id is not None
    assert tok.unk_token_id is not None
    assert tok.bos_token_id is not None
    assert tok.eos_token_id is not None
    assert tok.vocab_size() >= 256  # ByteLevel initial alphabet has 256 bytes + specials


def test_batch_encode(sample_tokenizer):
    tok, _ = sample_tokenizer
    texts = ["Hello world", "Privan transformer"]
    batch_ids = tok.batch_encode(texts)
    assert len(batch_ids) == 2
    assert len(batch_ids[0]) > 0
    assert len(batch_ids[1]) > 0


def test_save_and_load(sample_tokenizer):
    tok, tmp_path = sample_tokenizer
    save_dir = tmp_path / "saved_tok"
    tok.save(save_dir)

    assert (save_dir / "tokenizer.json").exists()
    assert (save_dir / "metadata.json").exists()

    loaded_tok = ByteLevelBPETokenizer.load(save_dir)
    assert loaded_tok.vocab_size() == tok.vocab_size()

    text = "Testing persistence."
    assert loaded_tok.encode(text) == tok.encode(text)
