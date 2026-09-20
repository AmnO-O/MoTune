"""Two-Stream Prototype & Semantic Shift Representation Learning.

This module implements the components for the Two-Stream Bi-Encoder architecture:
    Stream 1 (Prototype): [CLS] target_word [SEP] -> h_proto
    Stream 2 (Context):   [CLS] context_sentence [SEP] -> h_ctx

Features:
    1. pool_prototype: Pools lexical subwords of the isolated target word,
       excluding special tokens [CLS] and [SEP] to extract the pure prototype.
    2. SemanticShiftFusion: Disentangled feature interaction between h_ctx and h_proto
       capturing semantic displacement (h_ctx - h_proto), Hadamard alignment (h_ctx * h_proto),
       and cosine similarity, combined via residual MLP.
    3. prototype_rank_loss: Optional auxiliary margin ranking loss aligning
       cosine similarities with human compositionality ratings.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_prototype(hidden: torch.Tensor, proto_mask: torch.Tensor) -> torch.Tensor:
    """Pool lexical subword representations from Stream 1.

    For an isolated target word sequence [CLS] w_1 ... w_K [SEP], this excludes
    position 0 ([CLS]) and the last active position ([SEP]) whenever K >= 1
    so that only the true lexical subword tokens are pooled.
    Degrades gracefully to masked mean over all active tokens if sequence length < 3.

    Args:
        hidden: Tensor of shape (B, L, H) from the prototype forward pass.
        proto_mask: Attention mask of shape (B, L) where 1 indicates active tokens.

    Returns:
        Tensor of shape (B, H) containing the unpolluted prototype vector.
    """
    B, L, H = hidden.shape
    word_mask = proto_mask.clone().bool()
    lengths = proto_mask.sum(dim=-1).long()

    for i, length in enumerate(lengths):
        l_int = int(length.item())
        if l_int >= 3:
            word_mask[i, 0] = False           # Exclude [CLS]
            word_mask[i, l_int - 1] = False   # Exclude [SEP]

    mask_float = word_mask.unsqueeze(-1).float()
    denom = mask_float.sum(dim=1).clamp(min=1.0)
    return (hidden * mask_float).sum(dim=1) / denom


class SemanticShiftFusion(nn.Module):
    """Disentangled feature interaction between Context and Prototype.

    Given:
        h_ctx: Contextual representation of the target word in context (B, H)
        h_proto: Isolated lexical prototype representation (B, H)

    Computes:
        - Displacement vector: diff = h_ctx - h_proto (direction and scale of semantic shift)
        - Hadamard alignment:  prod = h_ctx * h_proto (dimension-wise agreement)
        - Cosine similarity:   cos  = CosineSimilarity(h_ctx, h_proto)
        - Composite feature:   [h_ctx, h_proto, diff, prod, cos] in R^(4H + 1)

    Uses a residual projection around h_ctx so that at initialization the model
    smoothly inherits the strong in-context baseline while learning to adjust
    based on lexical deviation.
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        in_dim = hidden_size * 4 + 1

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.last_cos: Optional[torch.Tensor] = None

    def forward(self, h_ctx: torch.Tensor, h_proto: torch.Tensor) -> torch.Tensor:
        """Fuse contextual and prototype vectors into a shifted representation.

        Args:
            h_ctx: Contextual vector of shape (B, H).
            h_proto: Prototype vector of shape (B, H).

        Returns:
            Fused vector of shape (B, H).
        """
        diff = h_ctx - h_proto
        prod = h_ctx * h_proto
        cos = F.cosine_similarity(h_ctx, h_proto, dim=-1, eps=1e-8).unsqueeze(-1)
        self.last_cos = cos.squeeze(-1)

        feat = torch.cat([h_ctx, h_proto, diff, prod, cos], dim=-1)
        shift = self.proj(feat)
        return self.out_norm(h_ctx + shift)


def prototype_rank_loss(
    cos_sim: torch.Tensor,
    ratings: torch.Tensor,
    margin: float = 0.2,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise margin ranking loss on prototype-context cosine similarity.

    Encourages cosine similarity to correlate positively with compositionality:
    samples with higher ratings should have higher prototype-context similarity.

    Args:
        cos_sim: Cosine similarities of shape (B,).
        ratings: Ground truth ratings of shape (B,).
        margin: Hinge loss margin.
        mask: Valid sample mask of shape (B,).

    Returns:
        Scalar loss tensor.
    """
    if mask is not None:
        cos_sim = cos_sim[mask]
        ratings = ratings[mask]

    n = cos_sim.size(0)
    if n < 2:
        return cos_sim.sum() * 0.0

    target_diff = ratings[:, None] - ratings[None, :]
    cos_diff = cos_sim[:, None] - cos_sim[None, :]

    pos_mask = target_diff > 0.5  # Only compare pairs with meaningful score gap
    if not pos_mask.any():
        return cos_sim.sum() * 0.0

    return F.relu(margin - cos_diff[pos_mask]).mean()
