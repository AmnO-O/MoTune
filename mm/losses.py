"""Loss terms for scoring + warmup (ported from the old src/loss.py).

Ported unchanged: ``margin_rank_loss`` (within-compound pairwise hinge),
``compound_consistency_loss`` (label-free rep tying), ``CombinedLoss``
(MSE + CCC + rank + optional Gaussian soft-target CE). Added for the
mmBERT rebuild: ``compound_center_loss`` (per-batch alignment of predicted
and gold compound centroids -> teaches between-compound ranking).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import SCORE_MAX, SCORE_MIN


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
# label-free consistency
# --------------------------------------------------------------------------- #
def compound_consistency_loss(
    mod_emb: torch.Tensor,
    head_emb: torch.Tensor,
    compound_ids: torch.Tensor,
    mode: str = 'pull',
    temp: float = 0.1,
) -> torch.Tensor:
    """Self-supervised loss pulling reps of the SAME compound together.

    'pull': distance to the compound's own centroid (gradient detached, so the
    loss cannot cheat by shrinking norms). 'infonce': InfoNCE positives within
    the batch. Rows with compound_id == -1 are excluded.
    """
    rep = torch.cat([mod_emb, head_emb], dim=-1).float()
    ids = compound_ids.to(rep.device)
    keep = ids >= 0
    if not keep.any():
        return rep.sum() * 0.0   # graph-connected zero; .backward() works in frozen phase
    ids, rep = ids[keep], rep[keep]

    if mode == 'pull':
        uniq, inv = torch.unique(ids, return_inverse=True)
        ncomp = uniq.shape[0]
        onehot = F.one_hot(inv, ncomp).float()
        counts = onehot.sum(0).clamp(min=1.0)
        centers = (onehot.t() @ rep) / counts.unsqueeze(1)
        var = ((rep - centers[inv].detach()) ** 2).sum(-1)
        return var.mean()

    n = rep.shape[0]
    if n < 2:
        return rep.sum() * 0.0   # graph-connected zero
    rep = F.normalize(rep, dim=-1)
    logits = rep @ rep.t() / temp
    same = (ids[:, None] == ids[None, :]).float()
    pos = same - torch.eye(n, device=rep.device, dtype=rep.dtype)
    has_pos = pos.sum(-1) > 0
    if not has_pos.any():
        return rep.sum() * 0.0   # graph-connected zero
    num_pos = pos.sum(-1).clamp(min=1.0)
    log_p = F.log_softmax(logits, dim=-1)
    loss_per_row = (pos * log_p).sum(-1) / num_pos
    return -loss_per_row[has_pos].mean()


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
# supervised composite loss
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    """MSE + CCC + pairwise-ranking + optional Gaussian soft-target CE.

    Optional per-sample weighting by annotator disagreement (``std_alpha``):
    rows with high ModStd/HeadStd are ambiguous -> less weight on MSE/CCC.
    """

    def __init__(self, ccc_weight: float = 0.5, lambda_rank: float = 0.1,
                 rank_margin: float = 0.1, ce_weight: float = 0.0,
                 num_bins: int = 6, bin_sigma: float = 0.5,
                 use_label_std: bool = True, ccc_var_floor: float = 0.05,
                 rank_margin_mode: str = 'dynamic', std_alpha: float = 0.0):
        super().__init__()
        self.ccc_weight = ccc_weight
        self.lambda_rank = lambda_rank
        self.rank_margin = rank_margin
        self.rank_margin_mode = rank_margin_mode
        self.ccc_var_floor = ccc_var_floor
        self.std_alpha = std_alpha
        self.ce_weight = ce_weight
        self.bin_sigma = bin_sigma
        self.use_label_std = use_label_std
        self.mse = nn.MSELoss()
        self.centers = torch.linspace(SCORE_MIN, SCORE_MAX, max(num_bins, 2)).tolist()
        self.requires_logits = ce_weight > 0

    def _soft_target(self, target: torch.Tensor, std: Optional[torch.Tensor] = None):
        centers = torch.as_tensor(self.centers, dtype=target.dtype,
                                  device=target.device).unsqueeze(0)
        if self.use_label_std and std is not None:
            sigma = torch.nan_to_num(std.float(), nan=self.bin_sigma,
                                     posinf=self.bin_sigma, neginf=self.bin_sigma)
            sigma = torch.clamp(sigma.unsqueeze(-1),
                                min=max(float(self.bin_sigma) * 0.5, 0.25), max=5.0)
        else:
            sigma = self.bin_sigma
        d = (centers - target.unsqueeze(-1)) / sigma
        w = torch.exp(-0.5 * d * d)
        denom = w.sum(-1, keepdim=True)
        zero_mask = denom < 1e-7
        if zero_mask.any():
            closest = (centers - target.unsqueeze(-1)).abs().argmin(-1, keepdim=True)
            w = torch.where(zero_mask, torch.zeros_like(w).scatter_(-1, closest, 1.0), w)
            denom = w.sum(-1, keepdim=True)
        return w / denom

    # def _ce(self, logits: torch.Tensor, target: torch.Tensor,
    #         std: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     log_p = torch.log_softmax(torch.clamp(logits.float(), -50.0, 50.0), dim=-1)
    #     soft = self._soft_target(target.float(), std)
    #     return -(soft * log_p).sum(dim=-1).mean()

    def _ce(self, logits: torch.Tensor, target: torch.Tensor,
            std: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Đầu ra mô hình Q (cần dùng log_softmax)
        log_q = F.log_softmax(torch.clamp(logits.float(), -50.0, 50.0), dim=-1)
        
        # 2. Nhãn mềm P (Phân phối xác suất từ avg & std)
        p_target = self._soft_target(target.float(), std)
        
        # 3. Tính KL-Divergence chuẩn PyTorch
        return F.kl_div(log_q, p_target, reduction='batchmean')

    def _weights(self, pred: torch.Tensor, std: Optional[torch.Tensor] = None):
        if self.std_alpha <= 0 or std is None:
            return torch.ones_like(pred)
        return 1.0 / (1.0 + self.std_alpha * torch.nan_to_num(std.float(), nan=0.0))

    def _compute_ccc(self, pred: torch.Tensor, target: torch.Tensor,
                     w: Optional[torch.Tensor] = None) -> torch.Tensor:
        if w is None:
            w = torch.ones_like(pred)
        ws = w.sum(dim=0)
        pm = (pred * w).sum(0) / ws
        tm = (target * w).sum(0) / ws
        pv = ((pred - pm) ** 2 * w).sum(0) / ws
        tv = ((target - tm) ** 2 * w).sum(0) / ws
        cov = ((pred - pm) * (target - tm) * w).sum(0) / ws
        denom = pv + tv + (pm - tm) ** 2 + self.ccc_var_floor
        return (1.0 - 2 * cov / denom).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                logits: Optional[torch.Tensor] = None,
                std: Optional[torch.Tensor] = None,
                compound_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        w = self._weights(pred, std)

        mse_loss = (((w * (pred - target) ** 2).sum() / w.sum().clamp(min=1.0))
                    if self.std_alpha > 0 and std is not None else self.mse(pred, target))
        loss = mse_loss
        if self.ccc_weight > 0:
            loss = loss + self.ccc_weight * self._compute_ccc(pred, target, w)
        if self.lambda_rank > 0:
            loss = loss + self.lambda_rank * margin_rank_loss(
                pred, target, margin=self.rank_margin,
                compound_ids=compound_ids, mode=self.rank_margin_mode)
        if self.ce_weight > 0 and logits is not None:
            loss = loss + self.ce_weight * self._ce(logits, target, std)
        return loss