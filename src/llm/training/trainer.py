"""Comprehensive Trainer for Privan-130M foundation language model."""

import time
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .optimizer import create_optimizer
from .scheduler import CosineWarmupScheduler
from .checkpoint import CheckpointManager
from .distributed import is_main_process, get_rank, get_world_size
from ..config import Config
from ..utils.logging import setup_logger
from ..utils.device import get_device_info


class Trainer:
    """
    Production-grade Trainer implementing:
    - Micro-batching and gradient accumulation
    - Mixed precision (BF16 / FP16 / FP32)
    - Gradient clipping and norm monitoring
    - Real-time token throughput calculation
    - NaN and Inf loss/grad detection with emergency checkpoints
    - TensorBoard and structured console logging
    - Validation loss and Perplexity evaluation
    - Fully resumable checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Config,
        device: torch.device,
        dtype: torch.dtype,
        use_scaler: bool = False,
        checkpoint_dir: Optional[str | Path] = None,
        log_dir: Optional[str | Path] = None,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.use_scaler = use_scaler

        self.model = model.to(device)

        # Wrap with DDP if distributed is active
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer and Scheduler
        self.optimizer = create_optimizer(self.model, config.optimizer)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.scheduler.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr=config.scheduler.min_lr,
        )

        # Mixed precision GradScaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if (use_scaler and device.type == "cuda") else None

        # Checkpoints and Logs
        ckpt_path = checkpoint_dir or config.checkpoint.output_dir
        self.checkpoint_manager = CheckpointManager(ckpt_path, keep_last=config.checkpoint.keep_last)

        log_path = log_dir or config.logging.log_dir
        self.logger = setup_logger("privan.trainer", log_file=Path(log_path) / "train.log")

        # TensorBoard writer
        self.tb_writer = None
        if config.logging.use_tensorboard and is_main_process():
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(Path(log_path) / "tensorboard"))
            except ImportError:
                self.logger.warning("Tensorboard not installed. Skipping tensorboard logging.")

        # State tracking
        self.step = 0
        self.epoch = 0
        self.total_tokens_processed = 0
        self.best_val_loss = float("inf")

        # Tokens per step calculation
        self.tokens_per_batch = config.training.batch_size * config.model.context_length
        self.tokens_per_opt_step = (
            self.tokens_per_batch
            * config.training.gradient_accumulation_steps
            * get_world_size()
        )

        if is_main_process():
            self.logger.info("=" * 60)
            self.logger.info("INITIALIZING PRIVAN-130M TRAINER")
            self.logger.info(f"Target Device:             {device}")
            self.logger.info(f"Precision:                 {dtype}")
            self.logger.info(f"GradScaler Enabled:        {use_scaler}")
            self.logger.info(f"Micro-Batch Size:          {config.training.batch_size}")
            self.logger.info(f"Gradient Accumulation:     {config.training.gradient_accumulation_steps}")
            self.logger.info(f"Effective Batch (Seqs):    {config.training.batch_size * config.training.gradient_accumulation_steps * get_world_size()}")
            self.logger.info(f"Tokens Per Optimizer Step: {self.tokens_per_opt_step:,}")
            self.logger.info(f"Max Steps:                 {config.training.max_steps:,}")
            self.logger.info("=" * 60)

    def train(self, resume_checkpoint: Optional[str | Path] = None) -> None:
        """Execute the main training loop."""
        if resume_checkpoint:
            self.logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            self.step, self.epoch, _ = CheckpointManager.load(
                resume_checkpoint,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                device=self.device,
            )
            self.total_tokens_processed = self.step * self.tokens_per_opt_step
            self.logger.info(f"Resumed at step {self.step:,} (Total tokens: {self.total_tokens_processed:,})")

        self.model.train()
        train_iter = iter(self.train_loader)

        step_start_time = time.perf_counter()
        accum_loss = 0.0

        while self.step < self.config.training.max_steps:
            self.optimizer.zero_grad(set_to_none=True)

            # Gradient accumulation loop
            for micro_step in range(self.config.training.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    self.epoch += 1
                    # Update DistributedSampler epoch if applicable
                    if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                        self.train_loader.sampler.set_epoch(self.epoch)
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                targets = batch["targets"].to(self.device, non_blocking=True)

                # Autocast forward pass
                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                autocast_enabled = (self.device.type == "cuda") or (self.dtype in (torch.bfloat16, torch.float16))
                with torch.autocast(device_type=device_type, dtype=self.dtype, enabled=autocast_enabled):
                    logits, loss, _ = self.model(input_ids, targets=targets)
                    loss = loss / self.config.training.gradient_accumulation_steps

                # NaN / Inf Loss Detection
                if torch.isnan(loss) or torch.isinf(loss):
                    self._handle_nan_inf("Loss is NaN or Inf during forward pass", loss.item())
                    return

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

            # Gradient clipping and norm computation
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.max_grad_norm,
            )

            # NaN / Inf Gradient Detection
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self._handle_nan_inf("Gradient norm is NaN or Inf after backward pass", float(grad_norm))
                return

            # Optimizer and Scheduler step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.step += 1
            self.total_tokens_processed += self.tokens_per_opt_step

            # Step metrics
            step_duration = time.perf_counter() - step_start_time
            tokens_per_sec = self.tokens_per_opt_step / max(1e-5, step_duration)
            examples_per_sec = (self.tokens_per_opt_step / self.config.model.context_length) / max(1e-5, step_duration)
            current_lr = self.scheduler.get_last_lr()[0]

            # Logging
            if self.step % self.config.training.log_interval == 0 and is_main_process():
                mem_str = self._get_memory_string()
                self.logger.info(
                    f"Step: {self.step:6d} | "
                    f"Loss: {accum_loss:6.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Grad Norm: {grad_norm:5.2f} | "
                    f"Tokens/sec: {tokens_per_sec:9,.0f} | "
                    f"Processed: {self.total_tokens_processed / 1e6:6.2f}M tok | "
                    f"{mem_str}"
                )

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("train/loss", accum_loss, self.step)
                    self.tb_writer.add_scalar("train/learning_rate", current_lr, self.step)
                    self.tb_writer.add_scalar("train/grad_norm", grad_norm, self.step)
                    self.tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.step)
                    self.tb_writer.add_scalar("train/total_tokens", self.total_tokens_processed, self.step)

            # Reset accumulators for next step
            accum_loss = 0.0
            step_start_time = time.perf_counter()

            # Evaluation
            if self.val_loader is not None and self.step % self.config.training.eval_interval == 0:
                val_loss, perplexity = self.evaluate()
                if is_main_process():
                    self.logger.info("-" * 60)
                    self.logger.info(
                        f"[EVALUATION @ Step {self.step:,}] Validation Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}"
                    )
                    self.logger.info("-" * 60)
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar("val/loss", val_loss, self.step)
                        self.tb_writer.add_scalar("val/perplexity", perplexity, self.step)

                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss

                    self.checkpoint_manager.save(
                        step=self.step,
                        epoch=self.epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        config=self.config.to_dict(),
                        is_best=is_best,
                        metrics={"val_loss": val_loss, "perplexity": perplexity},
                    )
                self.model.train()

            # Periodic checkpoint saving
            elif self.step % self.config.training.save_interval == 0 and is_main_process():
                saved_path = self.checkpoint_manager.save(
                    step=self.step,
                    epoch=self.epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    config=self.config.to_dict(),
                    is_best=False,
                )
                self.logger.info(f"Checkpoint saved: {saved_path}")

        # Final save at completion
        if is_main_process():
            self.checkpoint_manager.save(
                step=self.step,
                epoch=self.epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                config=self.config.to_dict(),
                is_best=False,
            )
            self.logger.info("=" * 60)
            self.logger.info(f"TRAINING COMPLETE! Processed {self.total_tokens_processed:,} total tokens.")
            self.logger.info("=" * 60)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Compute validation loss and perplexity on the validation dataset."""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        autocast_enabled = (self.device.type == "cuda") or (self.dtype in (torch.bfloat16, torch.float16))

        for i, batch in enumerate(self.val_loader):
            if i >= self.config.training.eval_steps:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)

            with torch.autocast(device_type=device_type, dtype=self.dtype, enabled=autocast_enabled):
                _, loss, _ = self.model(input_ids, targets=targets)

            total_loss += loss.item()
            total_steps += 1

        avg_loss = total_loss / max(1, total_steps)
        # Numerical protection for perplexity calculation
        try:
            perplexity = math.exp(min(avg_loss, 20.0))
        except OverflowError:
            perplexity = float("inf")

        return avg_loss, perplexity

    def _handle_nan_inf(self, message: str, val: float) -> None:
        """Handle numerical instability by logging and saving emergency checkpoint."""
        self.logger.error("!" * 60)
        self.logger.error(f"NUMERICAL INSTABILITY DETECTED @ STEP {self.step}: {message} (Value: {val})")
        if is_main_process():
            emergency_path = self.checkpoint_manager.save(
                step=self.step,
                epoch=self.epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                config=self.config.to_dict(),
                is_emergency=True,
            )
            self.logger.error(f"Emergency diagnostic checkpoint saved to: {emergency_path}")
        self.logger.error("Stopping training safely.")
        self.logger.error("!" * 60)

    def _get_memory_string(self) -> str:
        """Format current device memory usage."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            return f"GPU Mem: {allocated:.1f}/{reserved:.1f} GB"
        else:
            import psutil
            process = psutil.Process()
            ram_gb = process.memory_info().rss / (1024 ** 3)
            return f"RAM: {ram_gb:.1f} GB"
