"""Causal Multi-Head Self-Attention for Privan-130M."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Supports:
    - Fused QKV linear projection (D -> 3D)
    - PyTorch Scaled Dot-Product Attention (SDPA)
    - Fallback explicit mathematical attention for tests and debugging
    - Dynamic KV Caching for low-latency autoregressive generation
    - Output projection (D -> D) and residual dropout
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        context_length: int = 1024,
        dropout: float = 0.0,
        bias: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length
        self.dropout_p = dropout
        self.use_sdpa = use_sdpa

        # Fused Q, K, V projection: D -> 3D
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)

        # Output projection: D -> D
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer: lower triangular matrix [1, 1, context_length, context_length]
        # Registered as non-persistent buffer
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_kv_cache: bool = False,
        use_manual_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Causal Self-Attention.

        Args:
            x: Input tensor of shape [B, T, D]
            kv_cache: Optional tuple of (cached_k, cached_v) from prior generation steps
            use_kv_cache: If True, return updated KV cache
            use_manual_attn: Force explicit mathematical attention for debugging/testing

        Returns:
            output: [B, T, D]
            new_kv_cache: Optional (k, v) cache tuple
        """
        B, T, D = x.shape

        # 1. Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, T, 3 * D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, T, D]

        # 2. Reshape to multi-head format: [B, H, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. KV-Cache update for incremental autoregressive generation
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            new_kv_cache = (k, v)
        elif use_kv_cache:
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        total_k_len = k.size(2)

        # 4. Attention computation
        if use_manual_attn or (not self.use_sdpa):
            out = self._manual_attention(q, k, v, total_k_len=total_k_len)
        else:
            # When generating with KV-cache and single new query token (T=1), all past tokens are valid
            if (new_kv_cache is not None or kv_cache is not None) and T == 1 and total_k_len > 1:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )

        # 5. Merge heads: [B, H, T, head_dim] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 6. Final linear projection and dropout
        out = self.resid_dropout(self.out_proj(out))

        return out, new_kv_cache

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        total_k_len: int,
    ) -> torch.Tensor:
        """
        Explicit mathematical attention implementation:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        """
        T_q = q.size(2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if T_q == total_k_len:
            mask = self.causal_mask[:, :, :T_q, :total_k_len]
            scores = scores.masked_fill(~mask, float("-inf"))
        elif T_q > 1:
            offset = total_k_len - T_q
            mask = self.causal_mask[:, :, offset:offset + T_q, :total_k_len]
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        return out
