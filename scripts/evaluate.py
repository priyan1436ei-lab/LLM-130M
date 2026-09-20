"""Script to evaluate a Privan-130M checkpoint on test/validation data."""

import sys
import argparse
from pathlib import Path
import torch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config
from llm.model import CausalLM
from llm.dataset import CausalLanguageModelingDataset, create_dataloader
from llm.training.checkpoint import CheckpointManager
from llm.evaluation import evaluate_loss_and_perplexity
from llm.utils import select_device_and_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint loss and perplexity")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to checkpoint .pt")
    parser.add_argument("--config", type=str, default="configs/130m.yaml", help="Path to model config")
    parser.add_argument("--data", type=str, default="data/processed/val.bin", help="Path to binary validation/test set")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max-steps", type=int, default=100, help="Max evaluation steps")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.device:
        device = torch.device(args.device)
        dtype = torch.float32
    else:
        device, dtype, _ = select_device_and_dtype(cfg.training.precision)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint '{ckpt_path}' does not exist! Evaluating untrained model baseline.")
        model = CausalLM(cfg.model)
        step = 0
    else:
        model, step, epoch, _ = CheckpointManager.load_model(
            ckpt_path, device=device, fallback_config=cfg.model
        )
        print(f"Loaded checkpoint from step {step:,} (epoch {epoch})")

    model = model.to(device)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {data_path}")

    dataset = CausalLanguageModelingDataset(data_path, context_length=cfg.model.context_length)
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    print("=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Checkpoint:      {args.checkpoint} (Step {step:,})")
    print(f"Dataset:         {args.data} ({len(dataset)} chunks)")
    print(f"Device:          {device}")
    print(f"Precision:       {dtype}")
    print("-" * 50)

    val_loss, perplexity = evaluate_loss_and_perplexity(
        model=model,
        data_loader=dataloader,
        device=device,
        dtype=dtype,
        max_steps=args.max_steps,
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Perplexity:      {perplexity:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
