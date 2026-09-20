"""Learning rate scheduler with linear warmup and cosine decay."""

import math
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Learning Rate Scheduler with Linear Warmup and Cosine Annealing.

    Phase 1 (0 <= step < warmup_steps):
        lr = base_lr * (step + 1) / warmup_steps

    Phase 2 (warmup_steps <= step <= max_steps):
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(pi * decay_ratio))

    Phase 3 (step > max_steps):
        lr = min_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 3e-5,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            alpha = (step + 1) / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]

        if step > self.max_steps:
            # Past max steps: maintain min_lr
            return [self.min_lr for _ in self.base_lrs]

        # Cosine decay
        decay_ratio = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (base_lr - self.min_lr) for base_lr in self.base_lrs]
