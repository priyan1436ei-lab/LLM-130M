"""Transformer Decoder Backbone for Privan-130M."""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .embeddings import TokenEmbedding, PositionEmbedding
from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    """
    Transformer Decoder Backbone.

    Pipeline:
    input_ids -> TokenEmbedding + PositionEmbedding -> Dropout
              -> [TransformerBlock x N] -> final LayerNorm -> hidden_states
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 1024,
        embedding_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        bias: bool = True,
        norm_eps: float = 1e-5,
        activation: str = "gelu",
        use_sdpa: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionEmbedding(context_length, embedding_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                context_length=context_length,
                dropout=dropout,
                bias=bias,
                norm_eps=norm_eps,
                activation=activation,
                use_sdpa=use_sdpa,
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_kv_cache: bool = False,
        position_offset: int = 0,
        use_manual_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for Transformer backbone.

        Args:
            input_ids: [B, T] token IDs
            kv_caches: Optional list of (k, v) cache tuples for each layer
            use_kv_cache: If True, track and return KV caches for all layers
            position_offset: Start position offset for positional embeddings during KV generation
            use_manual_attn: Force explicit manual attention

        Returns:
            hidden_states: [B, T, D]
            new_kv_caches: List of updated (k, v) tuples if caching is used
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_embedding(input_ids)  # [B, T, D]
        pos_emb = self.position_embedding(T, device=device, offset=position_offset)  # [1, T, D]
        x = self.drop(tok_emb + pos_emb)

        wants_cache = use_kv_cache or (kv_caches is not None)
        new_kv_caches = [] if wants_cache else None

        # Pass through Transformer blocks
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None

            if self.gradient_checkpointing and self.training and layer_cache is None:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                x, cache_out = checkpoint(
                    create_custom_forward(block),
                    x,
                    None,
                    False,
                    use_manual_attn,
                    use_reentrant=False,
                )
            else:
                x, cache_out = block(
                    x,
                    kv_cache=layer_cache,
                    use_kv_cache=wants_cache,
                    use_manual_attn=use_manual_attn,
                )

            if new_kv_caches is not None:
                new_kv_caches.append(cache_out)

        # Final LayerNorm
        x = self.ln_f(x)

        return x, new_kv_caches
