"""Gaussian head: predicts N(mu, sigma^2) per span, with mu and sigma
architecturally decoupled instead of sharing a hidden trunk.

Why: Stirn et al. (2023, AISTATS, "Faithful Heteroscedastic Regression with
Neural Networks") show a heteroscedastic model's mean subnetwork trains
identically to an equivalent mean-only baseline once two conditions hold:

  Proposal 1 - the mean's gradient must not depend on the model's OWN
  predicted variance.
  Proposal 2 - the variance branch's gradient must not reach any parameter
  shared with the mean branch.

Proposal 1 already holds for this pipeline: in losses.gauss_kl, the mean
term is (mu_p - target)^2 / (2 * sigma_t^2) -- divided by the *target's*
sigma, not sigma_p. Differentiate it: d(loss)/d(mu_p) = (mu_p - target) /
sigma_t^2, which never involves sigma_p at all. That's the classic Nix &
Weigend (1994) failure mode Beta-NLL was built to patch, and this loss
doesn't have it -- nothing to change here.

Proposal 2 is a real gap in the previous head: mu and sigma shared one
hidden layer (`self.net`), so sigma's gradient reached the same trunk
weights mu's prediction depends on, and reached them through a shared
`h_mu == h_sigma` activation with no name distinguishing the two uses. This
version gives them separate trunks, so sigma cannot reshape the features mu
reads. ``sigma_input='detached'`` goes one step further and protects
*everything upstream of this head* (the shared encoder, LoRA adapters, span
pooling -- shared across all three Gaussian heads) from the sigma branch's
gradient entirely, at the cost of sigma only ever seeing features shaped by
someone else's objective. Try both; this is a real trade-off, not a strict
improvement (see the module-level ablation note in the training config).

The previous version's `logvar` name was misleading -- it was never
exponentiated or treated as a log-variance anywhere; it went straight into
softplus. Renamed to `raw_sigma` here to match what it actually is.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Single source of truth for the sigma floor. losses.py imports this same
# constant instead of hard-coding its own -- the previous two files disagreed
# (0.04 here vs. 0.05 there), which meant the head's own floor was silently
# inactive for the (0.04, 0.05) range GaussLoss's clamp overrode anyway.
SIGMA_FLOOR = 0.05


class GaussHead(nn.Module):
    """Independent (mu, sigma) branches off a shared *input*, not a shared hidden layer.

    Parameters
    ----------
    in_features : dimensionality of the pooled span/context feature vector
        this head receives (unchanged from the previous GaussHead).
    hidden : mu branch's hidden width. Kept at the previous default (128) --
        see the module docstring in the accompanying loss/architecture note
        for why capacity is probably not the lever to pull here.
    sigma_hidden : sigma branch's hidden width. Deliberately smaller than
        `hidden` by default: sigma only needs to rank "how noisy is this
        context", a coarser signal than mu's fine-grained score, and giving
        it a separate but *smaller* trunk keeps total head parameters close
        to the original shared-trunk design instead of roughly doubling them.
    sigma_input : 'shared' (sigma sees the same, gradient-carrying input as
        mu -- still protects mu from sigma via separate trunks, i.e.
        Proposal 2 at the head level only) or 'detached' (sigma sees
        `x.detach()` -- Proposal 2 extended to everything upstream of this
        head, including the shared encoder/LoRA). Start with 'shared'; try
        'detached' as an ablation arm, not a default, since severing the
        gradient path can starve sigma of calibration-relevant signal it
        would otherwise help shape upstream (this is a documented failure
        mode of the fully-severed approach on some tasks, not a hypothetical).
    """

    def __init__(self, in_features: int, hidden: int = 128,
                 sigma_hidden: Optional[int] = None, dropout: float = 0.2,
                 floor: float = SIGMA_FLOOR, sigma_input: str = 'shared'):
        super().__init__()
        if sigma_input not in ('shared', 'detached'):
            raise ValueError(f"sigma_input must be 'shared' or 'detached', got {sigma_input!r}")
        self.sigma_input = sigma_input
        self.floor = floor
        sigma_hidden = sigma_hidden or max(32, hidden // 2)

        self.mu_trunk = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.mu_out = nn.Linear(hidden, 1)
        # cheap residual path around the trunk -- eases optimization, doesn't
        # meaningfully add capacity (it's a single linear map of the input).
        self.mu_skip = nn.Linear(in_features, 1)

        self.sigma_trunk = nn.Sequential(
            nn.Linear(in_features, sigma_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.sigma_out = nn.Linear(sigma_hidden, 1)

    def forward(self, x: torch.Tensor):
        """Returns ``(mu (B,), sigma (B,))``; sigma > ``floor`` always."""
        h_mu = self.mu_trunk(x)
        mu = self.mu_out(h_mu).squeeze(-1) + self.mu_skip(x).squeeze(-1)

        sigma_in = x.detach() if self.sigma_input == 'detached' else x
        h_sigma = self.sigma_trunk(sigma_in)
        raw_sigma = self.sigma_out(h_sigma).squeeze(-1)
        sigma = F.softplus(raw_sigma) + self.floor
        return mu, sigma