"""CLI script for text generation with Privan-130M."""

import sys
import argparse
from pathlib import Path
import torch

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config, ModelConfig
from llm.model import CausalLM
from llm.tokenizer import ByteLevelBPETokenizer
from llm.generation import generate_text, generate_stream
from llm.training.checkpoint import CheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using Privan-130M")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to checkpoint .pt")
    parser.add_argument("--config", type=str, default="configs/130m.yaml", help="Path to model config")
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer", help="Path to tokenizer directory")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence is", help="Input prompt")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-P nucleus sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--use-kv-cache", action="store_true", default=True, help="Enable KV cache")
    parser.add_argument("--stream", action="store_true", help="Stream tokens live to console")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tok_path = Path(args.tokenizer_dir)
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer directory not found at {tok_path}. Train tokenizer first.")
    tokenizer = ByteLevelBPETokenizer.load(tok_path)

    # Load config and model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint '{ckpt_path}' not found! Initializing untrained model for demonstration.")
        cfg = Config.from_yaml(args.config)
        model = CausalLM(cfg.model)
    else:
        fallback_cfg = Config.from_yaml(args.config) if Path(args.config).exists() else None
        model, step, epoch, _ = CheckpointManager.load_model(
            ckpt_path, device=device, fallback_config=fallback_cfg
        )
        print(f"Loaded checkpoint from step {step:,} (epoch {epoch})")

    model = model.to(device)

    print("=" * 60)
    print("PRIVAN-130M TEXT GENERATION")
    print("=" * 60)
    print(f"Device:             {device}")
    print(f"Prompt:             {args.prompt}")
    print(f"Max New Tokens:     {args.max_new_tokens}")
    print(f"Temperature:        {args.temperature}")
    print(f"Top-K:              {args.top_k}")
    print(f"Top-P:              {args.top_p}")
    print(f"Repetition Penalty: {args.repetition_penalty}")
    print(f"KV Cache:           {args.use_kv_cache}")
    print("-" * 60)

    if args.stream:
        print(args.prompt, end="", flush=True)
        stream = generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=args.use_kv_cache,
            device=device,
        )
        tokens_count = 0
        import time
        t0 = time.perf_counter()
        for piece, _, is_eos in stream:
            print(piece, end="", flush=True)
            tokens_count += 1
            if is_eos:
                break
        t1 = time.perf_counter()
        dur = t1 - t0
        print("\n" + "-" * 60)
        print(f"Generation tokens/sec: {tokens_count / max(1e-5, dur):.2f}")
        print(f"Latency:               {dur * 1000:.2f} ms")
    else:
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=args.use_kv_cache,
            device=device,
        )
        print("Generated text:")
        print(result["text"])
        print("-" * 60)
        print(f"Generation tokens/sec: {result['tokens_per_sec']}")
        print(f"Latency:               {result['latency_ms']} ms")

    print("=" * 60)


if __name__ == "__main__":
    main()
