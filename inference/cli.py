"""Interactive terminal CLI for chatting with Privan-130M."""

import sys
import argparse
from pathlib import Path
import torch

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config
from llm.model import CausalLM
from llm.tokenizer import ByteLevelBPETokenizer
from llm.generation import generate_stream
from llm.training.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Interactive Privan-130M Shell")
    parser.add_argument("--config", type=str, default="configs/130m.yaml", help="Path to config")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer", help="Tokenizer directory")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Max tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("\n[Notice: Interactive terminal input requires a TTY.]")
        print("To chat interactively, run directly in your terminal: python inference/cli.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok_dir = Path(args.tokenizer_dir)
    if not tok_dir.exists():
        print(f"Error: Tokenizer directory {tok_dir} not found. Please train the tokenizer first.")
        return

    tokenizer = ByteLevelBPETokenizer.load(tok_dir)

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        fallback_cfg = Config.from_yaml(args.config) if Path(args.config).exists() else None
        model, step, epoch, _ = CheckpointManager.load_model(
            ckpt, device=device, fallback_config=fallback_cfg
        )
        print(f"Loaded model checkpoint from step {step:,} ({model.config.name})")
    else:
        print(f"Warning: Checkpoint {ckpt} not found. Using initialized weights.")
        cfg = Config.from_yaml(args.config)
        model = CausalLM(cfg.model)

    model = model.to(device)
    model.eval()

    print("=" * 60)
    print("Welcome to Privan-130M Interactive Console!")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nUser > ").strip()
            if prompt.lower() in ("exit", "quit", "q"):
                print("Exiting Privan-130M console.")
                break
            if not prompt:
                continue

            print("Privan > ", end="", flush=True)

            stream = generate_stream(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                use_kv_cache=True,
                device=device,
            )

            for piece, _, is_eos in stream:
                print(piece, end="", flush=True)
                if is_eos:
                    break
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Privan-130M console.")
            break


if __name__ == "__main__":
    main()
