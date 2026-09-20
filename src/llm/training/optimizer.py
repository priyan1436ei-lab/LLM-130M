"""Optimizer configuration with decoupled weight decay for Privan-130M."""

from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..config import OptimizerConfig


def create_optimizer(model: nn.Module, config: OptimizerConfig) -> AdamW:
    """
    Configure AdamW with decoupled weight decay:
    - Weight decay applied to 2D tensors (linear weights, embeddings)
    - No weight decay for 1D tensors (biases, LayerNorm scale and shift)
    """
    decay_params = []
    no_decay_params = []

    # Track parameter names for diagnostics
    decay_param_names = []
    no_decay_param_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1D parameters (biases, LayerNorm weights/biases) do not receive weight decay
        if param.dim() < 2 or "ln_" in name or "ln_f" in name or name.endswith(".bias"):
            no_decay_params.append(param)
            no_decay_param_names.append(name)
        else:
            decay_params.append(param)
            decay_param_names.append(name)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)

    optimizer = AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )

    optimizer.stats = {
        "num_decay_tensors": len(decay_params),
        "num_decay_params": num_decay,
        "num_no_decay_tensors": len(no_decay_params),
        "num_no_decay_params": num_no_decay,
    }

    return optimizer
