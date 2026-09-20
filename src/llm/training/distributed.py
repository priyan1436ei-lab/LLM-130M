"""Distributed Data Parallel (DDP) utilities for multi-GPU training."""

import os
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler


def setup_distributed() -> Tuple[bool, int, int, int]:
    """
    Initialize torch.distributed if running under torchrun / MPI.

    Returns:
        (is_distributed, rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


def is_main_process() -> bool:
    """Return True if this is the primary rank (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Return global rank of process."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return total number of distributed processes."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def cleanup_distributed() -> None:
    """Destroy process group on shutdown."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_distributed_sampler(
    dataset: Dataset,
    shuffle: bool = True,
    seed: int = 42,
) -> Optional[DistributedSampler]:
    """Create a DistributedSampler if distributed training is active."""
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
    return None
