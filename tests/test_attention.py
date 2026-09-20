"""Unit tests for Multi-Head Causal Self-Attention and Causal Masking."""

import torch
import pytest
from llm.model.attention import CausalSelfAttention


def test_attention_shapes():
    B, T, D = 2, 16, 64
    H = 4
    attn = CausalSelfAttention(embedding_dim=D, num_heads=H, context_length=32, dropout=0.0)

    x = torch.randn(B, T, D)
    out, cache = attn(x)

    assert out.shape == (B, T, D)
    assert cache is None


def test_causal_mask_strictly_prevents_future_attention():
    """
    Mathematical proof test:
    Token at index i MUST NOT be influenced in any way by tokens at index j (j > i).
    We pass a sequence X1 and a sequence X2 where tokens after index i are modified.
    The outputs at positions 0..i must be exactly identical!
    """
    B, T, D = 1, 8, 32
    H = 4
    attn = CausalSelfAttention(embedding_dim=D, num_heads=H, context_length=16, dropout=0.0)
    attn.eval()

    x1 = torch.randn(B, T, D)
    x2 = x1.clone()

    # Modify tokens at positions 5, 6, 7 (future positions relative to 0..4)
    x2[:, 5:, :] = torch.randn(B, 3, D) * 100.0

    with torch.no_grad():
        out1, _ = attn(x1, use_manual_attn=True)
        out2, _ = attn(x2, use_manual_attn=True)

    # Positions 0 to 4 MUST be identical within floating point precision
    diff = (out1[:, :5, :] - out2[:, :5, :]).abs().max().item()
    assert diff < 1e-5, f"Causal mask violated! Future perturbation altered earlier tokens. Diff: {diff}"

    # Position 5 and later should differ because x2 was perturbed there
    future_diff = (out1[:, 5:, :] - out2[:, 5:, :]).abs().max().item()
    assert future_diff > 1e-3


def test_sdpa_and_manual_attention_parity():
    """Verify that PyTorch SDPA matches explicit manual attention."""
    B, T, D = 2, 8, 32
    H = 4
    attn = CausalSelfAttention(embedding_dim=D, num_heads=H, context_length=16, dropout=0.0)
    attn.eval()

    x = torch.randn(B, T, D)

    with torch.no_grad():
        out_manual, _ = attn(x, use_manual_attn=True)
        out_sdpa, _ = attn(x, use_manual_attn=False)

    max_diff = (out_manual - out_sdpa).abs().max().item()
    assert max_diff < 1e-4, f"Discrepancy between SDPA and manual attention: {max_diff}"


def test_kv_cache_equivalence():
    """
    Verify that generating step-by-step with KV cache produces identical outputs
    to computing the full sequence in a single forward pass.
    """
    B, T, D = 1, 6, 32
    H = 4
    attn = CausalSelfAttention(embedding_dim=D, num_heads=H, context_length=16, dropout=0.0)
    attn.eval()

    x = torch.randn(B, T, D)

    # 1. Full sequence pass
    with torch.no_grad():
        full_out, _ = attn(x)

    # 2. Step-by-step with KV cache
    step_outputs = []
    kv_cache = None
    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t:t+1, :]
            out_t, kv_cache = attn(x_t, kv_cache=kv_cache, use_kv_cache=True)
            step_outputs.append(out_t)

    cached_out = torch.cat(step_outputs, dim=1)

    diff = (full_out - cached_out).abs().max().item()
    assert diff < 1e-4, f"KV cache output diverged from full sequence pass! Max diff: {diff}"
