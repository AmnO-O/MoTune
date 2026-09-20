"""Gauss-only loss terms for the compositionality pipeline.

The objective combines an always-on Gaussian distribution loss
``KL(N(mu_p, sigma_p^2) || N(y, sigma_t^2))`` and CCC.  The KL term is
not weighted: predicting uncertainty without supervising its ``sigma``
output would leave that branch untrained.  (The prototype ranking term
``prototype_rank_loss`` lives in src/prototype_stream and is applied by
the two-stream trainer, not here.)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# floor of predicted sigma in gauss_kl matches the GaussHead floor (heads.py)
_SIGMA_FLOOR = 0.05


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
    """Gaussian KL + CCC.

    The predicted ``sigma`` travels through the ``logits`` channel, which is
    why ``requires_logits`` is True.
    """

    def __init__(self, ccc_weight: float = 0.7,
                 ccc_var_floor: float = 0.05,
                 bin_sigma: float = 0.5, use_label_std: bool = True):
        super().__init__()
        self.ccc_weight = ccc_weight
        self.ccc_var_floor = ccc_var_floor
        self.bin_sigma = bin_sigma
        self.use_label_std = use_label_std
        self.requires_logits = True

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                logits: Optional[torch.Tensor] = None,
                std: Optional[torch.Tensor] = None,
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

        mu = pred.float()
        sigma_p = logits.float() if logits is not None else torch.full_like(mu, self.bin_sigma)
        if self.use_label_std and std is not None:
            sigma_t = _target_sigma(std, self.bin_sigma)
        else:
            sigma_t = torch.full_like(mu, float(self.bin_sigma))

        loss = gauss_kl(mu, sigma_p, target, sigma_t)
        if self.ccc_weight > 0:
            loss = loss + self.ccc_weight * ccc_loss(mu, target, var_floor=self.ccc_var_floor)
        return loss
