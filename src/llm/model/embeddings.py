"""Token and learned positional embeddings for Privan-130M."""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Lookup table mapping discrete token IDs to dense embedding vectors.

    Input: [B, T] integer tensor of token IDs
    Output: [B, T, D] dense embeddings
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight


class PositionEmbedding(nn.Module):
    """
    Learned positional embeddings for sequence positions.

    Table of shape [context_length, embedding_dim].
    Generates positions 0, 1, ..., T-1 and indexes into the learned table.
    """

    def __init__(self, context_length: int, embedding_dim: int):
        super().__init__()
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, seq_len: int, device: torch.device, offset: int = 0) -> torch.Tensor:
        """
        Generate positional embeddings for sequence length seq_len.

        Args:
            seq_len: Current length of input sequence
            device: Target torch device
            offset: Position index offset (useful during incremental KV-cache generation)

        Returns:
            Positional embedding tensor of shape [1, seq_len, embedding_dim]
        """
        if offset + seq_len > self.context_length:
            raise ValueError(
                f"Requested position range [{offset}, {offset + seq_len}) exceeds context_length {self.context_length}"
            )
        positions = torch.arange(offset, offset + seq_len, dtype=torch.long, device=device)
        return self.embedding(positions).unsqueeze(0)  # [1, seq_len, D]

    @property
    def weight(self) -> nn.Parameter:
        return self.embedding.weight
