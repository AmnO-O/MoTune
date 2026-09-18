"""Deviated Gaussian head: predicts ``N(mu, sigma^2)`` per span.

Kept in its own module so the ordinal-softmax and plain-regression heads stay
untouched. ``mu`` is the point value (bounded at inference via clamp), ``sigma``
is the model's claimed uncertainty, guaranteed > 0 through ``softplus``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_FLOOR = 0.05


class GaussHead(nn.Module):
    """Two-branch head predicting mean (mu) and uncertainty (sigma)."""

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.2,
                 floor: float = SIGMA_FLOOR, separate_trunks: bool = False):
        super().__init__()
        self.floor = floor
        self.separate_trunks = separate_trunks

        if separate_trunks:
            # Tách riêng 2 trunk để gradient của sigma không làm nhiễu mu
            self.mu_trunk = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.sigma_trunk = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )

        self.mu_proj = nn.Linear(hidden, 1)
        self.sigma_proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        if self.separate_trunks:
            h_mu = self.mu_trunk(x)
            h_sigma = self.sigma_trunk(x)
        else:
            h_mu = h_sigma = self.net(x)

        mu = self.mu_proj(h_mu).squeeze(-1)
        sigma_logits = self.sigma_proj(h_sigma).squeeze(-1)
        sigma = F.softplus(sigma_logits) + self.floor
        
        return mu, sigma