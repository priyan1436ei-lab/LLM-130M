"""Script to export Privan-130M models to PyTorch .pt, Safetensors, and Hugging Face format."""

import sys
import json
import argparse
from pathlib import Path
import torch

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config
from llm.model import CausalLM
from llm.training.checkpoint import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Export Privan-130M checkpoints to diverse formats")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to input checkpoint .pt")
    parser.add_argument("--config", type=str, default="configs/130m.yaml", help="Path to config")
    parser.add_argument("--output-dir", type=str, default="export", help="Output directory for exported model")
    parser.add_argument("--format", type=str, default="all", choices=["pt", "safetensors", "hf", "all"])
    return parser.parse_args()


def export_hf_format(model: CausalLM, cfg: Config, output_dir: Path):
    """Export configuration and weights into Hugging Face GPT2-compatible format."""
    hf_config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": cfg.model.vocab_size,
        "n_positions": cfg.model.context_length,
        "n_ctx": cfg.model.context_length,
        "n_embd": cfg.model.embedding_dim,
        "n_layer": cfg.model.num_layers,
        "n_head": cfg.model.num_heads,
        "n_inner": cfg.model.mlp_dim,
        "activation_function": "gelu_new",
        "resid_pdrop": cfg.model.dropout,
        "attn_pdrop": cfg.model.dropout,
        "embd_pdrop": cfg.model.dropout,
        "layer_norm_epsilon": cfg.model.norm_eps,
        "initializer_range": 0.02,
        "tie_word_embeddings": cfg.model.weight_tying,
        "torch_dtype": "float32",
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # Save state dict
    try:
        from safetensors.torch import save_model
        save_model(model, str(output_dir / "model.safetensors"))
        print(f"Saved Hugging Face config and safetensors to {output_dir}")
    except ImportError:
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
        print(f"Saved Hugging Face config and pytorch_model.bin to {output_dir}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_yaml(args.config)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        model, step, epoch, _ = CheckpointManager.load_model(
            ckpt_path, device=torch.device("cpu"), fallback_config=cfg.model
        )
        cfg.model = model.config
        print(f"Loaded checkpoint from: {ckpt_path} ({model.config.name})")
    else:
        print(f"Warning: Checkpoint '{ckpt_path}' not found. Exporting initialized base weights.")
        model = CausalLM(cfg.model)

    state_dict = model.state_dict()

    # 1. PyTorch .pt
    if args.format in ("pt", "all"):
        pt_path = out_dir / "privan_130m.pt"
        torch.save({"model_state_dict": state_dict, "config": cfg.to_dict()}, pt_path)
        print(f"Exported PyTorch weights to: {pt_path}")

    # 2. Safetensors
    if args.format in ("safetensors", "all"):
        try:
            from safetensors.torch import save_model
            st_path = out_dir / "privan_130m.safetensors"
            save_model(model, str(st_path))
            print(f"Exported Safetensors weights to: {st_path}")
        except ImportError:
            print("safetensors library not available, skipping safetensors export.")

    # 3. Hugging Face format
    if args.format in ("hf", "all"):
        hf_dir = out_dir / "hf_model"
        hf_dir.mkdir(parents=True, exist_ok=True)
        export_hf_format(model, cfg, hf_dir)

    print("=" * 50)
    print("Export complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
