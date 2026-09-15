"""Uncertainty-aware Gaussian loss: ``KL(N(mu_p, sigma_p^2) || N(y, sigma_t^2))``.

This is the continuous counterpart of the softmax bin-CE used with
``head_mode='softmax'``: the model predicts a value (``mu``) and an uncertainty
(``sigma``); the target is a Gaussian centred on the gold mean whose width comes
from the crowd label std (fallback ``bin_sigma``). Using a KL between two
Gaussians instead of pure NLL avoids the sigma "cheat" (inflating sigma_p to
cheaply reduce the loss) because of the ``+ sigma_p^2 / (2 sigma_t^2)`` term.

Kept out of ``losses.py`` so nothing else changes; the shared ranking term is
reused from there.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .losses import margin_rank_loss


def gauss_kl(mu_p: torch.Tensor, sigma_p: torch.Tensor,
             target: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu_p, sigma_p^2) || N(target, sigma_t^2)), element-wise mean."""
    mu_p = mu_p.float()
    sigma_p = sigma_p.float().clamp(min=1e-3)
    target = target.float()
    sigma_t = sigma_t.float().clamp(min=1e-3)
    expect = (sigma_p ** 2 + (mu_p - target) ** 2) / (2 * sigma_t ** 2)
    return ((sigma_t / sigma_p).log() + expect - 0.5).mean()


def _target_sigma(std: Optional[torch.Tensor], bin_sigma: float) -> torch.Tensor:
    """Width of the target Gaussian from the annotated crowd std."""
    sigma = torch.nan_to_num(std.float(), nan=bin_sigma, posinf=bin_sigma, neginf=bin_sigma)
    return torch.clamp(sigma, min=max(float(bin_sigma) * 0.5, 0.25), max=5.0)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor,
             w: Optional[torch.Tensor] = None, var_floor: float = 0.05) -> torch.Tensor:
    """Lin's concordance correlation coefficient, batch-level, as a loss (1 - CCC)."""
    pred = pred.float()
    target = target.float()
    if w is None:
        w = torch.ones_like(pred)
    ws = w.sum(dim=0)
    pm = (pred * w).sum(0) / ws
    tm = (target * w).sum(0) / ws
    pv = ((pred - pm) ** 2 * w).sum(0) / ws
    tv = ((target - tm) ** 2 * w).sum(0) / ws
    cov = ((pred - pm) * (target - tm) * w).sum(0) / ws
    denom = pv + tv + (pm - tm) ** 2 + var_floor
    return (1.0 - 2 * cov / denom).mean()


class GaussLoss(nn.Module):
    """KL-Gaussian distribution loss + CCC + pairwise ranking for head_mode='gauss'.

    ``lambda_dist`` weights the Gaussian KL term (``cfg.ce_weight`` is reused for
    it). The predicted ``sigma`` travels through the ``logits`` channel, which is
    why ``requires_logits`` is True (the training loop needs no changes).
    """

    def __init__(self, lambda_dist: float = 0.0, ccc_weight: float = 0.7,
                 lambda_rank: float = 0.5, rank_margin: float = 0.5,
                 rank_margin_mode: str = 'dynamic', ccc_var_floor: float = 0.05,
                 bin_sigma: float = 0.5, use_label_std: bool = True,
                 std_alpha: float = 0.0):
        super().__init__()
        self.lambda_dist = lambda_dist
        self.ccc_weight = ccc_weight
        self.lambda_rank = lambda_rank
        self.rank_margin = rank_margin
        self.rank_margin_mode = rank_margin_mode
        self.ccc_var_floor = ccc_var_floor
        self.bin_sigma = bin_sigma
        self.use_label_std = use_label_std
        self.std_alpha = std_alpha
        self.requires_logits = True

    def _weights(self, pred: torch.Tensor, std: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.std_alpha <= 0 or std is None:
            return torch.ones_like(pred)
        return 1.0 / (1.0 + self.std_alpha * torch.nan_to_num(std.float(), nan=0.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                logits: Optional[torch.Tensor] = None,
                std: Optional[torch.Tensor] = None,
                compound_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        mu = pred.float()
        sigma_p = logits.float() if logits is not None else torch.full_like(mu, self.bin_sigma)
        if self.use_label_std and std is not None:
            sigma_t = _target_sigma(std, self.bin_sigma)
        else:
            sigma_t = torch.full_like(mu, float(self.bin_sigma))

        loss = torch.zeros((), dtype=mu.dtype, device=mu.device)
        if self.lambda_dist > 0:
            loss = loss + self.lambda_dist * gauss_kl(mu, sigma_p, target, sigma_t)
        if self.ccc_weight > 0:
            w = self._weights(mu, std)
            loss = loss + self.ccc_weight * ccc_loss(mu, target, w, self.ccc_var_floor)
        if self.lambda_rank > 0:
            loss = loss + self.lambda_rank * margin_rank_loss(
                mu, target, margin=self.rank_margin,
                compound_ids=compound_ids, mode=self.rank_margin_mode)
        return loss