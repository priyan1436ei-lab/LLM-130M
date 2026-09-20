"""Configuration schema and loader for Privan-130M."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "Privan-130M"
    vocab_size: int = 50257
    context_length: int = 1024
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.0
    bias: bool = True
    weight_tying: bool = True
    activation: str = "gelu"
    norm_eps: float = 1e-5
    use_kv_cache: bool = False

    @property
    def head_dim(self) -> int:
        assert self.embedding_dim % self.num_heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        return self.embedding_dim // self.num_heads


@dataclass
class TrainingConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    max_steps: int = 100000
    eval_interval: int = 1000
    eval_steps: int = 100
    log_interval: int = 10
    save_interval: int = 1000
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    precision: str = "auto"  # 'bf16', 'fp16', 'fp32', 'auto'
    seed: int = 42


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    learning_rate: float = 0.0003
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    warmup_steps: int = 2000
    min_lr: float = 0.00003


@dataclass
class DatasetConfig:
    train_path: str = "data/processed/train.bin"
    val_path: str = "data/processed/val.bin"
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: Optional[int] = 2
    drop_last: bool = True


@dataclass
class CheckpointConfig:
    output_dir: str = "checkpoints"
    save_interval: int = 1000
    keep_last: int = 5


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = "privan-130m"
    wandb_entity: Optional[str] = None


@dataclass
class LoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["qkv_proj", "out_proj"])


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            scheduler=SchedulerConfig(**data.get("scheduler", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            lora=LoRAConfig(**data.get("lora", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
