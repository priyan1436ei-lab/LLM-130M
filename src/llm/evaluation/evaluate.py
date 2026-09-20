"""Evaluation engine for Privan-130M."""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_loss_and_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_steps: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute average cross-entropy loss and perplexity on an evaluation dataset.

    Formula:
        Perplexity = exp(Loss)
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    device_type = "cuda" if device.type == "cuda" else "cpu"
    autocast_enabled = (device.type == "cuda") or (dtype in (torch.bfloat16, torch.float16))

    for step, batch in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
            _, loss, _ = model(input_ids, targets=targets)

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_batches
    try:
        perplexity = math.exp(min(avg_loss, 20.0))
    except OverflowError:
        perplexity = float("inf")

    return avg_loss, perplexity
