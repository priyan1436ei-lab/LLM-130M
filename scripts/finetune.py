"""Instruction fine-tuning script with response-only loss masking and optional LoRA."""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config
from llm.model import CausalLM
from llm.model.lora import apply_lora_to_model, mark_only_lora_as_trainable
from llm.tokenizer import ByteLevelBPETokenizer
from llm.training.checkpoint import CheckpointManager
from llm.training import Trainer
from llm.utils import seed_everything, select_device_and_dtype


DEFAULT_CHAT_TEMPLATE = (
    "<|system|>\n{system}\n\n"
    "<|user|>\n{user}\n\n"
    "<|assistant|>\n{assistant}"
)


class InstructionDataset(Dataset):
    """
    Instruction fine-tuning dataset with response-only loss masking.

    Prompt tokens are masked with label -100 so the cross-entropy loss
    is only calculated over the assistant response tokens.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: ByteLevelBPETokenizer,
        context_length: int = 1024,
        chat_template: str = DEFAULT_CHAT_TEMPLATE,
        default_system: str = "You are Privan, a helpful and precise AI assistant.",
        mask_prompt_loss: bool = True,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.chat_template = chat_template
        self.default_system = default_system
        self.mask_prompt_loss = mask_prompt_loss
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Instruction dataset not found at {path}")

        eos_id = tokenizer.eos_token_id

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                output = item.get("output", "")

                user_prompt = f"{instruction}\n{inp}".strip() if inp else instruction

                # Format prompt prefix and assistant completion
                prompt_text = (
                    f"<|system|>\n{default_system}\n\n"
                    f"<|user|>\n{user_prompt}\n\n"
                    f"<|assistant|>\n"
                )
                response_text = f"{output}"

                prompt_tokens = tokenizer.encode(prompt_text)
                response_tokens = tokenizer.encode(response_text) + [eos_id]

                full_tokens = (prompt_tokens + response_tokens)[:context_length]
                pad_id = tokenizer.pad_token_id
                pad_len = context_length - len(full_tokens)

                if pad_len > 0:
                    input_tokens = full_tokens + [pad_id] * pad_len
                else:
                    input_tokens = full_tokens

                target_tokens = list(input_tokens)
                if self.mask_prompt_loss:
                    # Mask prompt tokens
                    prompt_len = min(len(prompt_tokens), len(target_tokens))
                    for i in range(prompt_len):
                        target_tokens[i] = -100

                # Mask padding tokens in targets
                if pad_len > 0:
                    for i in range(len(full_tokens), context_length):
                        target_tokens[i] = -100

                x = torch.tensor(input_tokens, dtype=torch.long)
                y = torch.tensor(target_tokens, dtype=torch.long)
                self.examples.append((x, y))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x, y = self.examples[idx]
        return {"input_ids": x, "targets": y}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Privan-130M on instruction data")
    parser.add_argument("--config", type=str, default="configs/130m.yaml", help="Model config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base pretrained checkpoint .pt")
    parser.add_argument("--data", type=str, required=True, help="JSONL instruction data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/finetuned", help="Output directory")
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer", help="Tokenizer directory")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--response-only-loss", action="store_true", default=True, help="Mask prompt loss")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.training.seed)

    device, dtype, use_scaler = select_device_and_dtype(cfg.training.precision)

    # 1. Load tokenizer
    tokenizer = ByteLevelBPETokenizer.load(args.tokenizer_dir)

    # 2. Load Base Model
    if Path(args.checkpoint).exists():
        model, step, epoch, _ = CheckpointManager.load_model(
            args.checkpoint, device=device, fallback_config=cfg.model
        )
        cfg.model = model.config
        print(f"Loaded pretrained checkpoint from: {args.checkpoint} ({model.config.name})")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Training from initialized weights.")
        model = CausalLM(cfg.model).to(device)

    # 3. Apply LoRA if requested
    if args.use_lora or cfg.lora.enabled:
        rank = args.lora_rank or cfg.lora.rank
        alpha = args.lora_alpha or cfg.lora.alpha
        print(f"Applying LoRA (rank={rank}, alpha={alpha})...")
        apply_lora_to_model(model, rank=rank, alpha=alpha, dropout=cfg.lora.dropout)
        trainable_params = mark_only_lora_as_trainable(model)
        print(f"LoRA enabled! Trainable parameters: {trainable_params:,}")

    # 4. Prepare Dataset
    dataset = InstructionDataset(
        jsonl_path=args.data,
        tokenizer=tokenizer,
        context_length=cfg.model.context_length,
        mask_prompt_loss=args.response_only_loss,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # 5. Fine-tuning Trainer
    cfg.checkpoint.output_dir = args.output_dir
    trainer = Trainer(
        model=model,
        train_loader=dataloader,
        val_loader=None,
        config=cfg,
        device=device,
        dtype=dtype,
        use_scaler=use_scaler,
    )

    print("Starting instruction fine-tuning...")
    trainer.train()


if __name__ == "__main__":
    main()
