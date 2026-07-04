from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.config.schema import GPTConfig


class MultiHeadAttention(nn.Module):
    """
    Production-style Multi-Head Self-Attention.

    Input Shape:
        (B, T, C)

    Output Shape:
        (B, T, C)

    Where:
        B = Batch Size
        T = Sequence Length (Context Length)
        C = Embedding Dimension
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Model configuration
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Ensure the embedding dimension can be evenly divided
        # across all attention heads.
        assert (
            self.embedding_dim % self.num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        # Dimension processed by each attention head
        self.head_dim = self.embedding_dim // self.num_heads

        # Single projection to generate Query, Key and Value.
        # Output dimension = 3 * embedding_dim.
        self.qkv = nn.Linear(
            self.embedding_dim,
            self.embedding_dim * 3,
            bias=False,
        )

        # Final output projection after merging all heads.
        self.projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
        )

        # Dropout applied to attention weights.
        self.dropout = nn.Dropout(
            dropout,
        )

        self.mask: torch.Tensor

        # Lower triangular causal mask.
        mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )

        # Register as a buffer so it moves automatically
        # between CPU and GPU with the model.
        self.register_buffer(
            "mask",
            mask,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C)

        Returns:
            Tensor of shape (B, T, C)
        """

        # Batch Size
        # Sequence Length
        # Embedding Dimension
        B, T, C = x.shape

        # -----------------------------------------------------
        # Generate Query, Key and Value
        # Shape:
        # (B, T, 3C)
        # -----------------------------------------------------
        qkv = self.qkv(x)

        # Split into three tensors
        # Each becomes:
        # (B, T, C)
        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        # -----------------------------------------------------
        # Split embedding dimension into multiple heads
        #
        # Before:
        # (B, T, C)
        #
        # After view:
        # (B, T, H, D)
        #
        # After transpose:
        # (B, H, T, D)
        # -----------------------------------------------------
        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # -----------------------------------------------------
        # Compute attention scores
        #
        # q : (B, H, T, D)
        # k : (B, H, T, D)
        #
        # k transpose:
        # (B, H, D, T)
        #
        # scores:
        # (B, H, T, T)
        # -----------------------------------------------------
        # scores = q @ k.transpose(-2, -1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Scale scores for numerical stability
        scores = scores / math.sqrt(self.head_dim)

        # -----------------------------------------------------
        # Apply causal mask
        #
        # Future tokens receive -inf.
        # -----------------------------------------------------
        scores = scores.masked_fill(
            ~self.mask[:T, :T],
            float("-inf"),
        )

        # -----------------------------------------------------
        # Convert scores into probabilities
        # -----------------------------------------------------
        attention = torch.softmax(
            scores,
            dim=-1,
        )

        if torch.isnan(attention).any():
            raise RuntimeError("NaN detected in attention weights.")

        attention = self.dropout(
            attention,
        )

        # -----------------------------------------------------
        # Weighted sum of Value vectors
        #
        # Result:
        # (B, H, T, D)
        # -----------------------------------------------------
        output = attention @ v

        # -----------------------------------------------------
        # Merge attention heads
        #
        # (B, H, T, D)
        #
        # -> (B, T, H, D)
        #
        # -> (B, T, C)
        # -----------------------------------------------------
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        # -----------------------------------------------------
        # Final projection
        # -----------------------------------------------------
        output = self.projection(
            output,
        )

        return output