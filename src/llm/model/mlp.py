"""Multi-Layer Perceptron (Feed-Forward Network) for Privan-130M."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Architecture:
    Linear(embedding_dim -> mlp_dim)
    GELU activation (approximate='tanh')
    Linear(mlp_dim -> embedding_dim)
    Dropout
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim

        self.fc1 = nn.Linear(embedding_dim, mlp_dim, bias=bias)

        if activation.lower() == "gelu":
            self.act = nn.GELU(approximate="tanh")
        elif activation.lower() == "exact_gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.fc2 = nn.Linear(mlp_dim, embedding_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = GELU(x W1 + b1) W2 + b2
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
