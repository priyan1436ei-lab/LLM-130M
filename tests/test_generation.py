"""Unit tests for Text Generation and Sampling algorithms."""

import torch
import pytest
from llm.config import ModelConfig
from llm.model import CausalLM
from llm.tokenizer import ByteLevelBPETokenizer
from llm.generation.sampler import (
    sample_next_token,
    apply_repetition_penalty,
    top_k_top_p_filtering,
)
from llm.generation import generate_text, generate_stream


def test_greedy_sampling():
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])
    token = sample_next_token(logits, temperature=0.0)
    assert token == 1  # argmax is index 1


def test_top_k_filtering():
    logits = torch.tensor([[1.0, 5.0, 10.0, 2.0, 0.5]])
    filtered = top_k_top_p_filtering(logits, top_k=2)

    # Top 2 are indices 2 (10.0) and 1 (5.0); others must be -inf
    assert filtered[0, 2] == 10.0
    assert filtered[0, 1] == 5.0
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 3] == float("-inf")
    assert filtered[0, 4] == float("-inf")


def test_repetition_penalty():
    logits = torch.tensor([[2.0, 4.0, 6.0]])
    penalized = apply_repetition_penalty(logits.clone(), generated_tokens=[2], penalty=2.0)
    # Index 2 was 6.0, should now be 3.0
    assert penalized[0, 2].item() == 3.0
    assert penalized[0, 1].item() == 4.0


def test_generate_text_and_streaming(tmp_path):
    # Setup dummy tokenizer
    corpus = tmp_path / "c.txt"
    corpus.write_text("Hello world and language modeling test.\n", encoding="utf-8")
    tok = ByteLevelBPETokenizer.train_from_files([corpus], vocab_size=280)

    cfg = ModelConfig(
        vocab_size=tok.vocab_size(),
        context_length=32,
        embedding_dim=32,
        num_layers=2,
        num_heads=2,
        mlp_dim=64,
    )
    model = CausalLM(cfg)

    # Test standard generation
    res = generate_text(
        model=model,
        tokenizer=tok,
        prompt="Hello",
        max_new_tokens=10,
        temperature=0.8,
        use_kv_cache=True,
    )
    assert "tokens_generated" in res
    assert res["tokens_generated"] > 0
    assert "latency_ms" in res
    assert "text" in res

    # Test streaming generation
    stream = list(generate_stream(
        model=model,
        tokenizer=tok,
        prompt="Hello",
        max_new_tokens=5,
        temperature=0.0,
        use_kv_cache=True,
    ))
    assert len(stream) > 0
    assert isinstance(stream[0][0], str)
