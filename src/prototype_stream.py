"""Two-Stream Prototype & Semantic Shift Representation Learning.

This module implements the components for the Two-Stream Bi-Encoder architecture:
    Stream 1 (Prototype): [CLS] target_word [SEP] -> h_proto
    Stream 2 (Context):   [CLS] context_sentence [SEP] -> h_ctx

Features:
    1. pool_prototype: Pools lexical subwords of the isolated target word,
       excluding special tokens [CLS] and [SEP] to extract the pure prototype.
    2. SemanticShiftFusion: Attention-based Cross-Fusion Transformer between h_ctx,
       h_proto, and their directional displacement (h_ctx - h_proto) using FusionBlock.
    3. prototype_rank_loss: Optional auxiliary margin ranking loss aligning
       cosine similarities with human compositionality ratings.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import FusionBlock


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
    """Attention-based Cross-Fusion between Context and Prototype.

    Instead of a simple flat concatenation or linear projection, treats
    contextual pool, lexical prototype, and semantic displacement as a
    3-token sequence with learnable role embeddings:
        Token 0: h_ctx (in-context meaning)
        Token 1: h_proto (out-of-context prototype meaning)
        Token 2: h_ctx - h_proto (directional semantic displacement)

    A Multi-Head Cross-Attention Transformer (FusionBlock) lets h_ctx dynamically
    attend across all three tokens, discovering which subspace dimensions shift
    or stay literal.

    Residual connection with h_ctx guarantees smooth training stability from step 0.
    """

    def __init__(self, hidden_size: int = 768, num_layers: int = 1,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        heads = num_heads if (hidden_size % num_heads == 0) else 1

        self.type_emb = nn.Parameter(torch.empty(3, hidden_size))
        nn.init.normal_(self.type_emb, std=0.02)

        self.layers = nn.ModuleList([
            FusionBlock(hidden_size, num_heads=heads, ffn_expansion=2, dropout=dropout)
            for _ in range(max(1, num_layers))
        ])
        self.out_norm = nn.LayerNorm(hidden_size)
        self.last_cos: Optional[torch.Tensor] = None

    def forward(self, h_ctx: torch.Tensor, h_proto: torch.Tensor) -> torch.Tensor:
        """Fuse contextual and prototype vectors via multi-head cross-attention.

        Args:
            h_ctx: Contextual vector of shape (B, H).
            h_proto: Prototype vector of shape (B, H).

        Returns:
            Fused vector of shape (B, H).
        """
        # Record cosine similarity for ranking loss and diagnostic tracking
        self.last_cos = F.cosine_similarity(h_ctx, h_proto, dim=-1, eps=1e-8)

        # Token sequence: [0: Context, 1: Prototype, 2: Displacement]
        diff = h_ctx - h_proto
        kv = torch.stack([h_ctx, h_proto, diff], dim=1)  # (B, 3, H)
        type_ids = torch.tensor([0, 1, 2], device=h_ctx.device, dtype=torch.long)
        kv = kv + self.type_emb[type_ids]

        # h_ctx acts as Query attending across all three semantic roles
        q = h_ctx.unsqueeze(1)  # (B, 1, H)
        for layer in self.layers:
            q = layer(q, kv)

        # Residual connection with input contextual vector
        return self.out_norm(h_ctx + q.squeeze(1))


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
