"""Unit tests for Optimizer, Scheduler, Checkpointing, and NaN detection."""

import torch
import torch.nn as nn
import pytest
from llm.config import ModelConfig, OptimizerConfig
from llm.model import CausalLM
from llm.training.optimizer import create_optimizer
from llm.training.scheduler import CosineWarmupScheduler
from llm.training.checkpoint import CheckpointManager


def test_optimizer_weight_decay_decoupling():
    cfg = ModelConfig(
        vocab_size=200,
        context_length=32,
        embedding_dim=32,
        num_layers=2,
        num_heads=2,
        mlp_dim=64,
        bias=True,
    )
    model = CausalLM(cfg)
    optim_cfg = OptimizerConfig(weight_decay=0.1)

    optimizer = create_optimizer(model, optim_cfg)

    # Group 0: weight_decay = 0.1
    # Group 1: weight_decay = 0.0
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    # Ensure all parameters in group 1 are 1D (biases or LayerNorm scales)
    for p in optimizer.param_groups[1]["params"]:
        assert p.dim() < 2

    # Ensure all parameters in group 0 are 2D or higher
    for p in optimizer.param_groups[0]["params"]:
        assert p.dim() >= 2


def test_cosine_warmup_scheduler():
    linear = nn.Linear(10, 10)
    opt = torch.optim.Adam(linear.parameters(), lr=1e-3)
    warmup_steps = 10
    max_steps = 100
    min_lr = 1e-4

    sched = CosineWarmupScheduler(
        opt,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
    )

    # Step 0 (during warmup): lr should be base_lr * 1 / 10 = 1e-4
    opt.step()
    sched.step()
    lr_1 = sched.get_last_lr()[0]
    assert lr_1 > 0
    assert lr_1 < 1e-3

    # Step 10 (warmup peak): lr should be approx base_lr
    for _ in range(warmup_steps - 1):
        opt.step()
        sched.step()
    lr_peak = sched.get_last_lr()[0]
    assert abs(lr_peak - 1e-3) < 1e-5

    # Step max_steps: lr should be min_lr
    for _ in range(max_steps - warmup_steps):
        opt.step()
        sched.step()
    lr_final = sched.get_last_lr()[0]
    assert abs(lr_final - min_lr) < 1e-5


def test_checkpoint_save_and_load(tmp_path):
    cfg = ModelConfig(
        vocab_size=200,
        context_length=32,
        embedding_dim=32,
        num_layers=1,
        num_heads=1,
        mlp_dim=32,
    )
    model1 = CausalLM(cfg)
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    manager = CheckpointManager(tmp_path, keep_last=2)
    saved_path = manager.save(
        step=50,
        epoch=1,
        model=model1,
        optimizer=opt1,
        scheduler=None,
        scaler=None,
        config={"test": True},
    )

    assert saved_path.exists()
    assert (tmp_path / "latest.pt").exists()

    # Load into model2
    model2 = CausalLM(cfg)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    step, epoch, conf = CheckpointManager.load(saved_path, model2, optimizer=opt2)
    assert step == 50
    assert epoch == 1
    assert conf["test"] is True

    # Parameter values must be identical
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
