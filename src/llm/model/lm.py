"""Causal Language Model (CausalLM) for Privan-130M."""

import math
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer
from ..config import ModelConfig


class CausalLM(nn.Module):
    """
    Causal Language Model consisting of:
    - Transformer Backbone
    - LM Head (with optional weight tying)
    - Autoregressive Cross-Entropy Loss computation
    - Explicit GPT-style weight initialization
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = Transformer(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            bias=config.bias,
            norm_eps=config.norm_eps,
            activation=config.activation,
            use_sdpa=True,
            gradient_checkpointing=False,
        )

        # LM Head projects hidden states [B, T, D] -> vocab logits [B, T, V]
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embeddings and output projection
        if config.weight_tying:
            self.lm_head.weight = self.transformer.token_embedding.embedding.weight

        # Apply explicit weight initialization
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-style weight initialization:
        - Linear: Normal(mean=0.0, std=0.02)
        - Embeddings: Normal(mean=0.0, std=0.02)
        - Biases: Constant(0.0)
        - LayerNorm: weight=1.0, bias=0.0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_kv_cache: bool = False,
        position_offset: int = 0,
        use_manual_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for Causal Language Model.

        Args:
            input_ids: [B, T] tensor of input token IDs
            targets: Optional [B, T] tensor of target token IDs (or shifted training labels)
            kv_caches: List of layer-wise KV caches for generation
            use_kv_cache: If True, return updated KV caches
            position_offset: Position offset for learned positional embeddings
            use_manual_attn: Use explicit mathematical attention

        Returns:
            logits: [B, T, V]
            loss: Optional scalar Cross-Entropy Loss
            new_kv_caches: Optional updated KV caches
        """
        # Pass through Transformer backbone
        hidden_states, new_kv_caches = self.transformer(
            input_ids=input_ids,
            kv_caches=kv_caches,
            use_kv_cache=use_kv_cache,
            position_offset=position_offset,
            use_manual_attn=use_manual_attn,
        )  # [B, T, D]

        # Compute logits via LM Head
        logits = self.lm_head(hidden_states)  # [B, T, V]

        loss = None
        if targets is not None:
            if targets.size(1) == input_ids.size(1):
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
            else:
                target_len = min(logits.size(1), targets.size(1))
                shift_logits = logits[:, :target_len, :].contiguous()
                shift_targets = targets[:, :target_len].contiguous()

            valid_tokens = (shift_targets != -100).sum()
            if valid_tokens == 0:
                loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )

        return logits, loss, new_kv_caches

    def count_parameters(self) -> Dict[str, int]:
        """Programmatically calculate parameter breakdown."""
        report = {
            "token_embedding": self.transformer.token_embedding.embedding.weight.numel(),
            "position_embedding": self.transformer.position_embedding.embedding.weight.numel(),
            "attention": 0,
            "mlp": 0,
            "layer_norm": 0,
            "lm_head": 0 if self.config.weight_tying else self.lm_head.weight.numel(),
            "total_trainable": 0,
            "total_non_trainable": 0,
            "total_unique": 0,
        }

        # Attention, MLP, LayerNorm
        for block in self.transformer.blocks:
            # Attention
            report["attention"] += block.attention.qkv_proj.weight.numel()
            if block.attention.qkv_proj.bias is not None:
                report["attention"] += block.attention.qkv_proj.bias.numel()
            report["attention"] += block.attention.out_proj.weight.numel()
            if block.attention.out_proj.bias is not None:
                report["attention"] += block.attention.out_proj.bias.numel()

            # MLP
            report["mlp"] += block.mlp.fc1.weight.numel()
            if block.mlp.fc1.bias is not None:
                report["mlp"] += block.mlp.fc1.bias.numel()
            report["mlp"] += block.mlp.fc2.weight.numel()
            if block.mlp.fc2.bias is not None:
                report["mlp"] += block.mlp.fc2.bias.numel()

            # LayerNorms in block
            report["layer_norm"] += block.ln_1.weight.numel()
            if block.ln_1.bias is not None:
                report["layer_norm"] += block.ln_1.bias.numel()
            report["layer_norm"] += block.ln_2.weight.numel()
            if block.ln_2.bias is not None:
                report["layer_norm"] += block.ln_2.bias.numel()

        # Final LayerNorm
        report["layer_norm"] += self.transformer.ln_f.weight.numel()
        if self.transformer.ln_f.bias is not None:
            report["layer_norm"] += self.transformer.ln_f.bias.numel()

        # Total unique trainable parameters (accounting for tied weights)
        trainable_params = set()
        total_trainable = 0
        total_non_trainable = 0

        for p in self.parameters():
            if p.requires_grad:
                if id(p) not in trainable_params:
                    trainable_params.add(id(p))
                    total_trainable += p.numel()
            else:
                total_non_trainable += p.numel()

        report["total_trainable"] = total_trainable
        report["total_non_trainable"] = total_non_trainable
        report["total_unique"] = total_trainable + total_non_trainable

        return report
