"""Low-Rank Adaptation (LoRA) implementation for parameter-efficient fine-tuning."""

import math
from typing import List, Optional
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA wrapper around an nn.Linear layer.

    Computes:
    output = base_linear(x) + (dropout(x) @ A^T @ B^T) * (alpha / rank)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Freeze base weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Low rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Weight initialization:
        # A is initialized with normal distribution; B is initialized with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> None:
    """Recursively replace targeted linear layers with LoRALinear modules."""
    if target_modules is None:
        target_modules = ["qkv_proj", "out_proj"]

    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child_name in target_modules and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)


def mark_only_lora_as_trainable(model: nn.Module) -> int:
    """Freeze all model parameters except LoRA parameters."""
    trainable_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
    return trainable_count
