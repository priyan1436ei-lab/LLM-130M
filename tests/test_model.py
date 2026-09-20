"""Unit tests for Privan-130M Model Architecture, Forward Pass, and Weight Tying."""

import torch
import pytest
from llm.config import ModelConfig
from llm.model import (
    TokenEmbedding,
    PositionEmbedding,
    MLP,
    TransformerBlock,
    Transformer,
    CausalLM,
)


def test_embedding_layers():
    vocab_size = 1000
    emb_dim = 64
    ctx_len = 32

    tok_emb = TokenEmbedding(vocab_size, emb_dim)
    pos_emb = PositionEmbedding(ctx_len, emb_dim)

    input_ids = torch.randint(0, vocab_size, (2, 10))
    t_out = tok_emb(input_ids)
    assert t_out.shape == (2, 10, emb_dim)

    p_out = pos_emb(10, device=torch.device("cpu"))
    assert p_out.shape == (1, 10, emb_dim)


def test_mlp_layer():
    emb_dim = 64
    mlp_dim = 256
    mlp = MLP(embedding_dim=emb_dim, mlp_dim=mlp_dim, dropout=0.0)

    x = torch.randn(2, 8, emb_dim)
    out = mlp(x)
    assert out.shape == (2, 8, emb_dim)


def test_transformer_block():
    emb_dim = 64
    num_heads = 4
    mlp_dim = 256
    ctx_len = 32

    block = TransformerBlock(
        embedding_dim=emb_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        context_length=ctx_len,
    )

    x = torch.randn(2, 12, emb_dim)
    out, cache = block(x)
    assert out.shape == (2, 12, emb_dim)
    assert cache is None


def test_causal_lm_forward_and_loss():
    config = ModelConfig(
        vocab_size=500,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        weight_tying=True,
    )
    model = CausalLM(config)

    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss, _ = model(input_ids, targets=targets)

    # Check shapes
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_weight_tying():
    config = ModelConfig(
        vocab_size=500,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        weight_tying=True,
    )
    model = CausalLM(config)

    # The lm_head weights MUST point to the same memory as token_embedding
    assert model.lm_head.weight is model.transformer.token_embedding.embedding.weight

    # Modifying one must reflect in the other
    with torch.no_grad():
        model.lm_head.weight[0, 0] += 1.0
    assert model.transformer.token_embedding.embedding.weight[0, 0] == model.lm_head.weight[0, 0]


def test_initialization_values():
    config = ModelConfig(
        vocab_size=1000,
        context_length=64,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        mlp_dim=256,
    )
    model = CausalLM(config)

    # Check linear layer bias initialized to 0
    assert torch.all(model.transformer.blocks[0].mlp.fc1.bias == 0)
    # Check LayerNorm initialized to weight 1, bias 0
    assert torch.all(model.transformer.blocks[0].ln_1.weight == 1)
    assert torch.all(model.transformer.blocks[0].ln_1.bias == 0)


def test_parameter_count_report():
    config = ModelConfig(
        vocab_size=1000,
        context_length=64,
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        mlp_dim=128,
        weight_tying=True,
    )
    model = CausalLM(config)
    stats = model.count_parameters()

    assert stats["total_trainable"] > 0
    assert stats["token_embedding"] == 1000 * 64
    assert stats["position_embedding"] == 64 * 64
    assert stats["total_unique"] == stats["total_trainable"]
