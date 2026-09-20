"""Script to programmatically count and report parameters for Privan-130M."""

import sys
import argparse
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config
from llm.model import CausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Programmatically count model parameters")
    parser.add_argument("--config", type=str, default="configs/130m.yaml", help="Path to model config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    model = CausalLM(cfg.model)
    stats = model.count_parameters()

    print("==================================================")
    print("MODEL PARAMETER REPORT")
    print("======================")
    print("")
    print("Vocabulary:")
    print(f"{cfg.model.vocab_size:,}")
    print("")
    print("Embedding dimension:")
    print(f"{cfg.model.embedding_dim}")
    print("")
    print("Layers:")
    print(f"{cfg.model.num_layers}")
    print("")
    print("Heads:")
    print(f"{cfg.model.num_heads}")
    print("")
    print("Context:")
    print(f"{cfg.model.context_length}")
    print("")
    print("---")
    print("")
    print("Token Embedding:")
    print(f"{stats['token_embedding']:,}")
    print("")
    print("Position Embedding:")
    print(f"{stats['position_embedding']:,}")
    print("")
    print("Attention:")
    print(f"{stats['attention']:,}")
    print("")
    print("MLP:")
    print(f"{stats['mlp']:,}")
    print("")
    print("LayerNorm:")
    print(f"{stats['layer_norm']:,}")
    print("")
    print("LM Head:")
    if cfg.model.weight_tying:
        print(f"{stats['token_embedding']:,} (Tied to Token Embedding)")
    else:
        print(f"{stats['lm_head']:,}")
    print("")
    print("---")
    print("")
    print("TOTAL PARAMETERS:")
    print(f"{stats['total_unique']:,}")
    print("")
    print("TRAINABLE PARAMETERS:")
    print(f"{stats['total_trainable']:,}")
    print("")
    print("NON-TRAINABLE PARAMETERS:")
    print(f"{stats['total_non_trainable']:,}")
    print("")
    print("TOTAL:")
    approx_m = stats["total_trainable"] / 1e6
    print(f"~{approx_m:.1f}M")
    print("")
    print("==================================================")
    print(f"Weight Tying: {cfg.model.weight_tying}")
    if cfg.model.weight_tying:
        untied_total = stats['total_unique'] + stats['token_embedding']
        print(f"Note: With weight tying, lm_head shares weights with token_embedding, yielding {stats['total_trainable']:,} unique trainable weights.")
        print(f"      Without weight tying, total unique parameters would be {untied_total:,} (~{untied_total/1e6:.1f}M).")


if __name__ == "__main__":
    main()
