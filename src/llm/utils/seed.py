"""Reproducibility utilities for Privan-130M."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python random, NumPy, PyTorch CPU, and PyTorch CUDA for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
