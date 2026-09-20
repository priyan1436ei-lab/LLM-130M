"""FastAPI REST API server for Privan-130M inference."""

import os
import sys
from pathlib import Path
from typing import List, Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm.config import Config, ModelConfig
from llm.model import CausalLM
from llm.tokenizer import ByteLevelBPETokenizer
from llm.generation import generate_text
from llm.training.checkpoint import CheckpointManager
from llm.utils import select_device_and_dtype


# Pydantic Schemas
class GenerateRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "Explain neural networks"})
    max_new_tokens: int = Field(150, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    use_kv_cache: bool = True


class GenerateResponse(BaseModel):
    text: str
    completion: str
    tokens_generated: int
    latency_ms: float
    tokens_per_sec: float


class TokenizeRequest(BaseModel):
    text: str


class TokenizeResponse(BaseModel):
    token_ids: List[int]
    num_tokens: int


class DecodeRequest(BaseModel):
    token_ids: List[int]


class DecodeResponse(BaseModel):
    text: str


class ModelInfoResponse(BaseModel):
    model_name: str
    parameter_count: int
    trainable_parameters: int
    layers: int
    heads: int
    embedding_dimension: int
    context_length: int
    vocabulary_size: int
    checkpoint: str
    precision: str
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="Privan-130M Inference API",
    description="High-performance inference API for Privan-130M Transformer Language Model",
    version="0.1.0",
)

# Global model state
CONFIG_PATH = os.environ.get("PRIVAN_CONFIG", "configs/130m.yaml")
CHECKPOINT_PATH = os.environ.get("PRIVAN_CHECKPOINT", "checkpoints/latest.pt")
TOKENIZER_DIR = os.environ.get("PRIVAN_TOKENIZER", "data/tokenizer")

model: Optional[CausalLM] = None
tokenizer: Optional[ByteLevelBPETokenizer] = None
device: Optional[torch.device] = None
dtype: Optional[torch.dtype] = None
cfg: Optional[Config] = None


@app.on_event("startup")
def load_model_and_tokenizer():
    global model, tokenizer, device, dtype, cfg

    tok_path = Path(TOKENIZER_DIR)
    if not tok_path.exists():
        print(f"Warning: Tokenizer not found at {tok_path}.")
    else:
        tokenizer = ByteLevelBPETokenizer.load(tok_path)

    ckpt = Path(CHECKPOINT_PATH)
    if ckpt.exists():
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt_data.get("config", {})
        if "model" in ckpt_cfg:
            model_cfg = ModelConfig(**ckpt_cfg["model"])
            cfg = Config(model=model_cfg)
        else:
            cfg = Config.from_yaml(CONFIG_PATH)
        device, dtype, _ = select_device_and_dtype(cfg.training.precision)
        model = CausalLM(cfg.model)
        model.load_state_dict(ckpt_data["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt} ({cfg.model.name})")
    else:
        cfg = Config.from_yaml(CONFIG_PATH)
        device, dtype, _ = select_device_and_dtype(cfg.training.precision)
        model = CausalLM(cfg.model)
        print(f"Checkpoint not found at {ckpt}. Running with initialized weights.")

    model = model.to(device)
    model.eval()


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "privan-130m"}


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    if model is None or cfg is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    stats = model.count_parameters()
    return ModelInfoResponse(
        model_name=cfg.model.name,
        parameter_count=stats["total_unique"],
        trainable_parameters=stats["total_trainable"],
        layers=cfg.model.num_layers,
        heads=cfg.model.num_heads,
        embedding_dimension=cfg.model.embedding_dim,
        context_length=cfg.model.context_length,
        vocabulary_size=cfg.model.vocab_size,
        checkpoint=CHECKPOINT_PATH,
        precision=str(dtype),
        device=str(device),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded.")

    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        use_kv_cache=req.use_kv_cache,
        device=device,
    )

    return GenerateResponse(
        text=result["text"],
        completion=result["completion"],
        tokens_generated=result["tokens_generated"],
        latency_ms=result["latency_ms"],
        tokens_per_sec=result["tokens_per_sec"],
    )


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize_endpoint(req: TokenizeRequest):
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded.")

    ids = tokenizer.encode(req.text)
    return TokenizeResponse(token_ids=ids, num_tokens=len(ids))


@app.post("/decode", response_model=DecodeResponse)
def decode_endpoint(req: DecodeRequest):
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded.")

    text = tokenizer.decode(req.token_ids)
    return DecodeResponse(text=text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
