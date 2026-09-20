"""Privan-130M top-level package entry point and CLI runner."""

import sys
import argparse
from pathlib import Path
import torch

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure src is in Python path
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from llm.config import Config, ModelConfig
from llm.model import (
    CausalLM,
    Transformer,
    TransformerBlock,
    CausalSelfAttention,
    MLP,
    TokenEmbedding,
    PositionEmbedding,
)
from llm.tokenizer import ByteLevelBPETokenizer
from llm.generation import generate_text, generate_stream
from llm.training import Trainer
from llm.training.checkpoint import CheckpointManager

__all__ = [
    "Config",
    "ModelConfig",
    "CausalLM",
    "Transformer",
    "TransformerBlock",
    "CausalSelfAttention",
    "MLP",
    "TokenEmbedding",
    "PositionEmbedding",
    "ByteLevelBPETokenizer",
    "generate_text",
    "generate_stream",
    "Trainer",
]


def run_llm(
    prompt: str = "Artificial intelligence and neural networks",
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    checkpoint_path: str = "checkpoints/latest.pt",
    tokenizer_dir: str = "data/tokenizer",
    interactive: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = Path(checkpoint_path)
    tok_file = Path(tokenizer_dir)

    print("=" * 60)
    print("PRIVAN-130M LLM RUNNER")
    print("=" * 60)
    print(f"Device: {device}")

    # 1. Load or train tokenizer first to ensure vocabulary alignment
    if not tok_file.exists():
        print(f"Notice: Tokenizer not found at {tok_file}. Training lightweight tokenizer...")
        temp_corpus = Path("data/raw/corpus.txt")
        if not temp_corpus.exists():
            temp_corpus.parent.mkdir(parents=True, exist_ok=True)
            temp_corpus.write_text("Artificial intelligence and neural networks.", encoding="utf-8")
        tokenizer = ByteLevelBPETokenizer.train_from_files([temp_corpus], vocab_size=1000)
        tokenizer.save(tok_file)
    else:
        tokenizer = ByteLevelBPETokenizer.load(tok_file)

    tok_vocab = tokenizer.vocab_size()

    # 2. Determine config and load model/checkpoint
    if ckpt_file.exists():
        ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        cfg_dict = ckpt_data.get("config", {})
        if "model" in cfg_dict:
            model_cfg = ModelConfig(**cfg_dict["model"])
        else:
            model_cfg = ModelConfig(vocab_size=tok_vocab)
        model = CausalLM(model_cfg)
        model.load_state_dict(ckpt_data["model_state_dict"])
        step = ckpt_data.get("step", 0)
        print(f"Loaded checkpoint from: {ckpt_file} (Step {step:,})")
    else:
        model_cfg = ModelConfig(vocab_size=tok_vocab)
        model = CausalLM(model_cfg)
        print(f"Running with initialized architecture ({model_cfg.name}, vocab={tok_vocab}):")

    stats = model.count_parameters()
    print(f"Model Architecture: {model_cfg.name}")
    print(f"Parameters:         {stats['total_trainable']:,} (~{stats['total_trainable']/1e6:.1f}M)")
    print(f"Layers:             {model_cfg.num_layers} layers, {model_cfg.num_heads} heads, {model_cfg.embedding_dim} hidden dim")
    print(f"Vocabulary Size:    {model_cfg.vocab_size:,}")
    print(f"Context Length:     {model_cfg.context_length}")
    print("-" * 60)

    model = model.to(device)
    model.eval()

    if interactive:
        if not sys.stdin.isatty():
            print("\n[Notice: Non-interactive terminal detected. Interactive prompt input requires a TTY.]")
            print("To chat interactively, run directly in your terminal: python llm.py --interactive")
            return

        print("\n[Interactive Mode Active - Type 'exit' to quit]")
        while True:
            try:
                user_input = input("\nPrompt > ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue

                print("Response > ", end="", flush=True)
                stream = generate_stream(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_kv_cache=True,
                    device=device,
                )
                for piece, _, is_eos in stream:
                    print(piece, end="", flush=True)
                    if is_eos:
                        break
                print()
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting interactive session.")
        return

    # Single prompt run with streaming
    print(f"Prompt: {prompt}")
    print("Generating completion:")
    print("-" * 60)
    print(prompt, end="", flush=True)

    stream = generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_kv_cache=True,
        device=device,
    )

    import time
    t0 = time.perf_counter()
    tokens_generated = 0
    for piece, _, is_eos in stream:
        print(piece, end="", flush=True)
        tokens_generated += 1
        if is_eos:
            break
    dur = time.perf_counter() - t0

    print("\n" + "-" * 60)
    print(f"Tokens Generated:      {tokens_generated}")
    print(f"Generation Speed:      {tokens_generated / max(1e-5, dur):.2f} tokens/sec")
    print(f"Generation Latency:    {dur * 1000:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Privan-130M LLM")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence and neural networks", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt loop")
    args = parser.parse_args()

    run_llm(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        checkpoint_path=args.checkpoint,
        interactive=args.interactive,
    )
