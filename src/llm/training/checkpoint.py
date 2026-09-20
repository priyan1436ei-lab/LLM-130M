"""Checkpoint management system for Privan-130M."""

import os
import random
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_git_commit() -> str:
    """Safely obtain current git commit hash if in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


class CheckpointManager:
    """
    Manages saving, rotating, and restoring training checkpoints.
    """

    def __init__(self, output_dir: str | Path, keep_last: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.step_checkpoints = []

    def save(
        self,
        step: int,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        scaler: Optional[torch.amp.GradScaler],
        config: Dict[str, Any],
        is_best: bool = False,
        is_emergency: bool = False,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save a complete checkpoint containing all reproducible state."""
        # Unwrap DDP if model is wrapped
        raw_model = model.module if hasattr(model, "module") else model

        checkpoint_data = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "metrics": metrics or {},
            "random_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "git_commit": get_git_commit(),
        }

        if is_emergency:
            filename = f"emergency_checkpoint_step_{step}.pt"
            target_path = self.output_dir / filename
            torch.save(checkpoint_data, target_path)
            return target_path

        # Standard step checkpoint
        step_filename = f"step_{step}.pt"
        step_path = self.output_dir / step_filename
        torch.save(checkpoint_data, step_path)
        self.step_checkpoints.append(step_path)

        # Rotate older step checkpoints if exceeding keep_last
        if self.keep_last > 0 and len(self.step_checkpoints) > self.keep_last:
            oldest = self.step_checkpoints.pop(0)
            if oldest.exists() and oldest != step_path:
                try:
                    oldest.unlink()
                except OSError:
                    pass

        # Always update latest.pt
        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint_data, latest_path)

        # Update best.pt if specified
        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint_data, best_path)

        return step_path

    @staticmethod
    def load(
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Restore training state from a checkpoint file.

        Returns:
            (step, epoch, config)
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        raw_model = model.module if hasattr(model, "module") else model
        try:
            raw_model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Error loading checkpoint '{path}': model architecture mismatch.\n"
                f"Details: {e}\n"
                f"Hint: Use CheckpointManager.load_model(checkpoint_path) to automatically instantiate the matching architecture."
            ) from e

        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore RNG states if present
        rng_states = checkpoint.get("random_states", {})
        if "python" in rng_states and rng_states["python"] is not None:
            random.setstate(rng_states["python"])
        if "numpy" in rng_states and rng_states["numpy"] is not None:
            np.random.set_state(rng_states["numpy"])
        if "torch_cpu" in rng_states and rng_states["torch_cpu"] is not None:
            torch.set_rng_state(rng_states["torch_cpu"])
        if torch.cuda.is_available() and rng_states.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng_states["torch_cuda"])

        step = checkpoint.get("step", 0)
        epoch = checkpoint.get("epoch", 0)
        config = checkpoint.get("config", {})

        return step, epoch, config

    @staticmethod
    def load_model(
        checkpoint_path: str | Path,
        device: torch.device = torch.device("cpu"),
        fallback_config: Optional[Any] = None,
    ) -> Tuple[nn.Module, int, int, Dict[str, Any]]:
        """
        Dynamically load a CausalLM model from a checkpoint, matching its saved architecture.
        If config is saved in checkpoint, reconstructs the exact ModelConfig.
        Otherwise falls back to fallback_config or default ModelConfig.

        Returns:
            (model, step, epoch, config_dict)
        """
        from ..config import ModelConfig
        from ..model import CausalLM

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg_data = checkpoint.get("config", {})

        if isinstance(cfg_data, dict) and "model" in cfg_data:
            model_cfg = ModelConfig(**cfg_data["model"])
        elif fallback_config is not None:
            model_cfg = fallback_config.model if hasattr(fallback_config, "model") else fallback_config
        else:
            model_cfg = ModelConfig()

        model = CausalLM(model_cfg)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        step = checkpoint.get("step", 0)
        epoch = checkpoint.get("epoch", 0)
        return model, step, epoch, cfg_data
