"""Autoregressive text generation with KV-cache and streaming support."""

import time
from typing import Optional, List, Dict, Any, Iterator, Tuple
import torch
import torch.nn as nn

from .sampler import sample_next_token
from ..tokenizer import ByteLevelBPETokenizer


@torch.no_grad()
def generate_stream(
    model: nn.Module,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    use_kv_cache: bool = True,
    device: Optional[torch.device] = None,
) -> Iterator[Tuple[str, int, bool]]:
    """
    Stream tokens autoregressively from the model as they are generated.

    Yields:
        (token_text, token_id, is_eos)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        # Fallback if empty prompt
        prompt_tokens = [tokenizer.bos_token_id]

    # Context window guard
    context_limit = getattr(model.config, "context_length", 1024) if hasattr(model, "config") else 1024
    if len(prompt_tokens) >= context_limit:
        prompt_tokens = prompt_tokens[-(context_limit - 1):]

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    eos_id = tokenizer.eos_token_id
    generated_tokens: List[int] = []

    kv_caches = None
    curr_input_ids = input_ids
    pos_offset = 0

    for step in range(max_new_tokens):
        # Prevent position embedding out of bounds
        cur_total_len = (pos_offset + curr_input_ids.size(1)) if use_kv_cache else (len(prompt_tokens) + len(generated_tokens))
        if cur_total_len > context_limit:
            break

        # Forward pass
        logits, _, new_kv_caches = model(
            input_ids=curr_input_ids,
            kv_caches=kv_caches if use_kv_cache else None,
            use_kv_cache=use_kv_cache,
            position_offset=pos_offset if use_kv_cache else 0,
        )

        # Take logits at last token position
        next_token_logits = logits[:, -1, :]

        # Sample next token
        next_token_id = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=prompt_tokens + generated_tokens,
        )

        generated_tokens.append(next_token_id)
        token_str = tokenizer.decode([next_token_id], skip_special_tokens=False)

        is_eos = (next_token_id == eos_id)

        yield token_str, next_token_id, is_eos

        if is_eos:
            break

        # Prepare for next token
        if use_kv_cache:
            curr_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            pos_offset += 1 if step > 0 else input_ids.size(1)
            kv_caches = new_kv_caches
        else:
            curr_input_ids = torch.tensor([prompt_tokens + generated_tokens], dtype=torch.long, device=device)


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    use_kv_cache: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Generate text from a prompt and measure latency and tokens/sec.
    """
    start_time = time.perf_counter()

    generated_pieces = []
    generated_ids = []

    stream = generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_kv_cache=use_kv_cache,
        device=device,
    )

    for piece, token_id, is_eos in stream:
        generated_pieces.append(piece)
        generated_ids.append(token_id)
        if is_eos:
            break

    elapsed_time = time.perf_counter() - start_time
    total_tokens = len(generated_ids)
    tokens_per_sec = total_tokens / max(1e-5, elapsed_time)
    generated_text = "".join(generated_pieces)

    return {
        "prompt": prompt,
        "text": prompt + generated_text,
        "completion": generated_text,
        "tokens_generated": total_tokens,
        "latency_ms": round(elapsed_time * 1000, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }
