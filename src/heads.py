"""Deviated Gaussian head: predicts ``N(mu, sigma^2)`` per span.

Kept in its own module so the ordinal-softmax and plain-regression heads stay
untouched. ``mu`` is the point value (bounded at inference via clamp), ``sigma``
is the model's claimed uncertainty, guaranteed > 0 through ``softplus``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_FLOOR = 0.04


class GaussHead(nn.Module):
    """Two-branch head: shared MLP trunk -> one head for ``mu``, one for ``sigma``."""

    def __init__(self, in_features: int, hidden: int = 128, dropout: float = 0.2,
                 floor: float = SIGMA_FLOOR):
        super().__init__()
        self.floor = floor
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.mu = nn.Linear(hidden, 1)
        self.logvar = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        """Returns ``(mu (B,), sigma (B,))``; sigma > ``floor`` always."""
        h = self.net(x)
        mu = self.mu(h).squeeze(-1)
        logvar = self.logvar(h).squeeze(-1)
        sigma = F.softplus(logvar) + self.floor
        return mu, sigma