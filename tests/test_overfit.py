"""Overfit test to prove mathematically that Privan architecture learns."""

import torch
import pytest
from llm.config import ModelConfig
from llm.model import CausalLM


def test_tiny_model_overfit_single_batch():
    """
    CRITICAL ARCHITECTURE VERIFICATION TEST:
    Verifies that a tiny 2-layer model can memorize a small repetitive sequence.
    If the loss does not drop dramatically, backprop, causal masking,
    or target shifting is broken.
    """
    torch.manual_seed(42)

    # Tiny configuration as specified in Section 44:
    # layers = 2, embedding_dim = 128, heads = 4, mlp_dim = 512, context = 128
    config = ModelConfig(
        vocab_size=256,
        context_length=64,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        mlp_dim=512,
        dropout=0.0,
        weight_tying=True,
    )

    model = CausalLM(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # Fixed toy sequence repeated: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...]
    seq = [1, 2, 3, 4, 5, 6, 7, 8] * 4  # length 32
    input_ids = torch.tensor([seq], dtype=torch.long)
    targets = input_ids.clone()

    initial_loss = None
    final_loss = None

    # Overfit loop: 50 steps
    for step in range(50):
        optimizer.zero_grad()
        logits, loss, _ = model(input_ids, targets=targets)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"\n[Overfit Test] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")

    # Initial loss for uniform distribution over 256 tokens is ln(256) ~ 5.54
    # Final loss must drop dramatically (under 0.15)
    assert initial_loss > 4.0, f"Expected initial loss > 4.0, got {initial_loss}"
    assert final_loss < 0.15, (
        f"CRITICAL FAILURE: Overfit test failed! Final loss {final_loss} did not drop below 0.15. "
        f"Check attention masking, target shifting, or learning rate."
    )
    assert final_loss < initial_loss * 0.05, "Final loss must be < 5% of initial loss."
