"""Gauss-only loss terms for the compositionality pipeline.

Merged from the old ``losses.py`` + ``losses_gauss.py`` (the reg/softmax
``CombinedLoss`` and its soft-target bin machinery are gone): the Gaussian
distribution loss ``KL(N(mu_p, sigma_p^2) || N(y, sigma_t^2))`` plus the shared
pairwise-ranking and compound-center terms.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# floor of predicted sigma in gauss_kl matches the GaussHead floor (heads.py)
_SIGMA_FLOOR = 0.05


# --------------------------------------------------------------------------- #
# pairwise ranking (within same compound)
# --------------------------------------------------------------------------- #
def margin_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.5,
    compound_ids: Optional[torch.Tensor] = None,
    mode: str = 'dynamic',
) -> torch.Tensor:
    """Hinge loss on PAIRS INSIDE the same compound.

    ``mode``: 'dynamic' = relu(target_gap - pred_gap) unbounded margin;
    'clamp' = cap the margin at ``margin`` (gradient capped on extreme pairs).
    """
    if pred.ndim > 1:
        return torch.stack([
            margin_rank_loss(pred[:, i], target[:, i], margin, compound_ids, mode)
            for i in range(pred.shape[1])
        ]).mean()

    n = pred.shape[0]
    if n < 2:
        return pred.sum() * 0.0   # graph-connected zero

    target_diff = target[:, None] - target[None, :]
    pred_diff = pred[:, None] - pred[None, :]

    mask = target_diff > 0
    if compound_ids is not None:
        mask = mask & (compound_ids[:, None] == compound_ids[None, :]) & (compound_ids[:, None] >= 0)

    if not mask.any():
        return pred.sum() * 0.0   # graph-connected zero

    dynamic_margin = torch.clamp(target_diff[mask], max=margin) if mode == 'clamp' \
        else target_diff[mask]
    return F.relu(dynamic_margin - pred_diff[mask]).mean()


# --------------------------------------------------------------------------- #
# compound-center calibration (between-compound ranking)
# --------------------------------------------------------------------------- #
def compound_center_loss(pred: torch.Tensor, target: torch.Tensor,
                         compound_ids: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and gold compound centroids, per batch.

    Teaches the model the between-compound ordering directly: rows of the
    same compound are averaged, and the predicted mean is pulled toward the
    gold mean. Ignored labels (has_label=False rows have NaN gold) are left
    out via NaN masking of ``target``.
    """
    if pred.ndim > 1:
        return sum(
            compound_center_loss(pred[:, i], target[:, i], compound_ids)
            for i in range(pred.shape[1])
        ) / pred.shape[1]

    ids = compound_ids.to(pred.device)
    target = target.float()
    pred = pred.float()
    labeled = (~torch.isnan(target)) & (ids >= 0)
    if not labeled.any():
        return pred.sum() * 0.0   # graph-connected zero; .backward() works in frozen phase

    p, t = pred[labeled], target[labeled]
    g = ids[labeled]
    uniq, inv = torch.unique(g, return_inverse=True)
    ncomp = uniq.shape[0]
    onehot = F.one_hot(inv, ncomp).float()
    counts = onehot.sum(0).clamp(min=1.0)
    p_center = (onehot.t() @ p) / counts
    t_center = (onehot.t() @ t) / counts
    return F.mse_loss(p_center, t_center)


# --------------------------------------------------------------------------- #
# gaussian distribution loss
# --------------------------------------------------------------------------- #
def gauss_kl(mu_p: torch.Tensor, sigma_p: torch.Tensor,
             target: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu_p, sigma_p^2) || N(target, sigma_t^2)), element-wise mean."""
    mu_p = mu_p.float()
    sigma_p = sigma_p.float().clamp(min=_SIGMA_FLOOR)
    target = target.float()
    sigma_t = sigma_t.float().clamp(min=_SIGMA_FLOOR)
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
    ws = w.sum(dim=0).clamp(min=1e-8)
    pm = (pred * w).sum(0) / ws
    tm = (target * w).sum(0) / ws
    pv = ((pred - pm) ** 2 * w).sum(0) / ws
    tv = ((target - tm) ** 2 * w).sum(0) / ws
    cov = ((pred - pm) * (target - tm) * w).sum(0) / ws
    denom = pv + tv + (pm - tm) ** 2 + var_floor
    return (1.0 - 2 * cov / denom).mean()


class GaussLoss(nn.Module):
    """KL-Gaussian distribution loss + CCC + pairwise ranking.

    ``lambda_dist`` weights the Gaussian KL term (was ``cfg.ce_weight`` in the
    shared package). The predicted ``sigma`` travels through the ``logits``
    channel, which is why ``requires_logits`` is True (the training loop needs
    no changes).
    """

    def __init__(self, lambda_dist: float = 1.0, ccc_weight: float = 0.7,
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
                compound_ids: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(pred.device)
            if not mask.any():
                anchor = pred.sum() * 0.0
                if logits is not None:
                    anchor = anchor + logits.sum() * 0.0
                return anchor
            pred = pred[mask]
            target = target[mask]
            logits = logits[mask] if logits is not None else None
            std = std[mask] if std is not None else None
            compound_ids = compound_ids[mask] if compound_ids is not None else None

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