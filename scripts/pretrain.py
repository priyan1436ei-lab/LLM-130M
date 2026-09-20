"""Pretraining script for Privan-130M."""

import sys
import argparse
from pathlib import Path
import torch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config
from llm.model import CausalLM
from llm.dataset import CausalLanguageModelingDataset, create_dataloader
from llm.training import (
    Trainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)
from llm.training.distributed import create_distributed_sampler
from llm.utils import seed_everything, select_device_and_dtype, print_device_info


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Privan-130M Foundation Model")
    parser.add_argument("--config", type=str, default="configs/130m.yaml", help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (e.g. cuda, cpu)")
    parser.add_argument("--precision", type=str, default=None, help="Override precision (bf16, fp16, fp32)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Distributed setup
    is_dist, rank, local_rank, world_size = setup_distributed()

    # 2. Config load
    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.training.seed + rank)

    precision = args.precision or cfg.training.precision

    # 3. Device & Precision selection
    if args.device:
        device = torch.device(args.device)
        dtype = torch.float32
        use_scaler = False
    else:
        device, dtype, use_scaler = select_device_and_dtype(precision)

    if is_main_process():
        print_device_info(selected_precision=precision)
        if is_dist:
            print(f"Distributed Data Parallel active: World size = {world_size}")

    # 4. Initialize Model
    model = CausalLM(cfg.model)

    # 5. Datasets & DataLoaders
    train_path = Path(cfg.dataset.train_path)
    val_path = Path(cfg.dataset.val_path)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train dataset binary not found at '{train_path}'. "
            f"Please run 'python scripts/prepare_data.py' first."
        )

    train_dataset = CausalLanguageModelingDataset(
        train_path,
        context_length=cfg.model.context_length,
    )
    val_dataset = None
    if val_path.exists():
        val_dataset = CausalLanguageModelingDataset(
            val_path,
            context_length=cfg.model.context_length,
        )

    train_sampler = create_distributed_sampler(train_dataset, shuffle=True, seed=cfg.training.seed)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=cfg.dataset.drop_last,
        prefetch_factor=cfg.dataset.prefetch_factor,
        sampler=train_sampler,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=cfg.dataset.pin_memory,
            drop_last=False,
        )

    # 6. Initialize Trainer & Run Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        dtype=dtype,
        use_scaler=use_scaler,
    )

    try:
        trainer.train(resume_checkpoint=args.resume)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
