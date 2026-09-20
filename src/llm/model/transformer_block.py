"""Transformer Decoder Block with Pre-LayerNorm for Privan-130M."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import CausalSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    """
    Transformer Decoder Block.

    Architecture (Pre-LayerNorm):
    x -> LayerNorm -> Self-Attention -> Residual Add
    x -> LayerNorm -> MLP -> Residual Add
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        context_length: int = 1024,
        dropout: float = 0.0,
        bias: bool = True,
        norm_eps: float = 1e-5,
        activation: str = "gelu",
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim, eps=norm_eps)
        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
            bias=bias,
            use_sdpa=use_sdpa,
        )

        self.ln_2 = nn.LayerNorm(embedding_dim, eps=norm_eps)
        self.mlp = MLP(
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            bias=bias,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_kv_cache: bool = False,
        use_manual_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Pre-LayerNorm and residual connections.
        """
        # 1. Pre-LN Self-Attention
        attn_out, new_kv_cache = self.attention(
            self.ln_1(x),
            kv_cache=kv_cache,
            use_kv_cache=use_kv_cache,
            use_manual_attn=use_manual_attn,
        )
        x = x + attn_out

        # 2. Pre-LN MLP
        x = x + self.mlp(self.ln_2(x))

        return x, new_kv_cache
