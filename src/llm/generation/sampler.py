"""Sampling algorithms: greedy, temperature, top-k, top-p, and repetition penalty."""

from typing import List, Optional
import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: List[int],
    penalty: float = 1.0,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for tokens that have already been generated.

    For positive logits: logit = logit / penalty
    For negative logits: logit = logit * penalty
    """
    if penalty == 1.0 or not generated_tokens:
        return logits

    for token_id in set(generated_tokens):
        if token_id < logits.size(-1):
            if logits[0, token_id] < 0:
                logits[0, token_id] = logits[0, token_id] * penalty
            else:
                logits[0, token_id] = logits[0, token_id] / penalty

    return logits


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    """
    logits = logits.clone()

    # Top-K filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[List[int]] = None,
) -> int:
    """
    Sample a single token index from unnormalized logits [1, vocab_size].
    """
    # 1. Apply repetition penalty if history is provided
    if generated_tokens and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_tokens, penalty=repetition_penalty)

    # 2. Greedy decoding if temperature is 0
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    # 3. Temperature scaling
    logits = logits / temperature

    # 4. Top-K and Top-P filtering
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # 5. Softmax to obtain categorical probability distribution
    probs = F.softmax(filtered_logits, dim=-1)

    # 6. Sample token index
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())
