import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import SCORE_MAX, SCORE_MIN

def margin_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.5,
    compound_ids: torch.Tensor = None,
    mode: str = 'dynamic',
) -> torch.Tensor:
    """Pairwise hinge loss, computed WITHIN same compound.

    ``mode``:
      - 'dynamic': hinge = relu(target_gap - pred_gap), margin UNBOUNDED so the
        most extreme pairs are pulled hardest (a fully frozen encoder can then
        still make large-scale corrections through the head alone).
      - 'clamp':   same hinge but the margin is capped at ``margin`` (old
        behaviour -- caps the gradient on extreme pairs).
    """

    # Xử lý đa đầu ra (Mod & Head heads)
    if pred.ndim > 1:
        losses = [
            margin_rank_loss(pred[:, i], target[:, i], margin, compound_ids, mode)
            for i in range(pred.shape[1])
        ]
        return torch.stack(losses).mean()

    n = pred.shape[0]
    if n < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    # 1. Ma trận chênh lệch
    target_diff = target[:, None] - target[None, :]   # y_i - y_j
    pred_diff = pred[:, None] - pred[None, :]         # p_i - p_j

    # 2. Lọc các cặp y_i > y_j
    mask = target_diff > 0

    # CRITICAL FIX: Chỉ giữ lại các cặp thuộc CÙNG một từ ghép (Compound)
    if compound_ids is not None:
        same_compound = (compound_ids[:, None] == compound_ids[None, :])
        mask = mask & same_compound

    if not mask.any():
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    # 3. Dynamic Margin & Hinge Loss
    if mode == 'clamp':
        dynamic_margin = torch.clamp(target_diff[mask], max=margin)
    else:
        dynamic_margin = target_diff[mask]
    hinge = F.relu(dynamic_margin - pred_diff[mask])

    return hinge.mean()


def compound_consistency_loss(
    mod_emb: torch.Tensor,
    head_emb: torch.Tensor,
    compound_ids: torch.Tensor,
    mode: str = 'pull',
    temp: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Self-supervised loss tying together span reps of the SAME compound.

    With grouped sampling (3-5 compounds per batch) most batches hold multiple
    rows of the same compound in different contexts; this loss encourages the
    (mod, head) span representation to converge for each compound ("same MWE =
    same concept"), the label-free component of data enrichment.

    ``mode``:
      - 'pull':  centroid-variance -- distance to the compound's own centroid,
        with the centroid gradient detached (K-means style so the loss cannot
        cheat by shrinking norms). Recommended: it needs no negatives and does
        not conflict with the ranking task (which pushes reps apart).
      - 'infonce': contrastive -- same-compound positives within the batch,
        InfoNCE over normalized reps. Too few negatives per tiny grouped batch
        makes this weak; provided as an A/B alternative.
    """
    rep = torch.cat([mod_emb, head_emb], dim=-1)     # (B, 2H)
    ids = compound_ids.to(rep.device)
    if mode == 'pull':
        uniq, inv = torch.unique(ids, return_inverse=True)
        ncomp = uniq.shape[0]
        onehot = F.one_hot(inv, ncomp).float()       # (B, ncomp)
        counts = onehot.sum(0).clamp(min=1.0)        # (ncomp,)
        centers = (onehot.t() @ rep) / counts.unsqueeze(1)
        rep_c = centers[inv]
        var = ((rep - rep_c.detach()) ** 2).sum(-1)  # (B,)
        return var.mean()

    # InfoNCE fallback
    n = rep.shape[0]
    if n < 2:
        return torch.zeros((), device=rep.device, dtype=rep.dtype)
    rep = F.normalize(rep, dim=-1)
    logits = rep @ rep.t() / temp                    # (B, B)
    same = (ids[:, None] == ids[None, :]).float()
    pos = same - torch.eye(n, device=rep.device, dtype=rep.dtype)
    num_pos = pos.sum(-1).clamp(min=1.0)
    log_p = F.log_softmax(logits, dim=-1)
    return -((pos * log_p).sum(-1) / num_pos).mean()


class CombinedLoss(nn.Module):
    """MSE + CCC + Pairwise Ranking Loss + optional Gaussian soft-target CE.

    CE is only active when `ce_weight > 0` AND the caller passes `logits`
    (ordinal 'softmax' head mode). Soft targets are a Gaussian centred on the
    (continuous) target so a float label like 2.34 spreads mass over the bins.

    Optional per-sample weighting via ``loss_std_alpha``: rows whose annotators
    disagreed a lot (high ModStd/HeadStd) are ambiguous, so give them less say
    in MSE/CCC through weights ``1/(1 + alpha*std)``. The ranking and CE losses
    are untouched (order information is precisely what such rows still carry).
    """

    def __init__(self, ccc_weight: float = 0.5, lambda_rank: float = 0.1, rank_margin: float = 0.1,
                 ce_weight: float = 0.0, num_bins: int = 6, bin_sigma: float = 0.5,
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
        # Plain Python list -- NOT a module buffer. The criterion may be built on
        # CPU while targets live on cuda (notebook constructs by hand); building
        # the center tensor directly on target.device/dtype each call is
        # immune to any device mismatch (a buffer would stay on CPU until .to()).
        self.centers = torch.linspace(SCORE_MIN, SCORE_MAX, max(num_bins, 2)).tolist()
        # train_epoch uses this to decide whether to request logits from the model
        self.requires_logits = ce_weight > 0

    def _soft_target(self, target: torch.Tensor, std: torch.Tensor = None) -> torch.Tensor:
        """Gaussian mass on each bin for continuous target(s).

        With per-sample ``std`` (ModStd/HeadStd) given and ``use_label_std`` on,
        each sample gets its own Gaussian width so high-disagreement (ambiguous)
        items spread mass over more bins and low-disagreement items concentrate
        on the nearest bin. Otherwise a single ``bin_sigma`` is used for all.
        """
        centers = torch.as_tensor(
            self.centers, dtype=target.dtype, device=target.device
        ).unsqueeze(0)
        if self.use_label_std and std is not None:
            sigma = torch.nan_to_num(
                std.float(), nan=self.bin_sigma, posinf=self.bin_sigma, neginf=self.bin_sigma
            ).unsqueeze(-1)
            # Annotator std=0.0 means full agreement, NOT a zero-width delta function.
            # Floor to a safe minimum width so the Gaussian always has mass over nearest bins.
            min_sigma = max(float(self.bin_sigma) * 0.5, 0.25)
            sigma = torch.clamp(sigma, min=min_sigma, max=5.0)
        else:
            sigma = self.bin_sigma

        d = (centers - target.unsqueeze(-1)) / sigma
        w = torch.exp(-0.5 * d * d)
        denom = w.sum(dim=-1, keepdim=True)

        # Fallback guarantee: if denom underflows for any reason, place one-hot on closest bin
        zero_mask = (denom < 1e-7)
        if zero_mask.any():
            closest = (centers - target.unsqueeze(-1)).abs().argmin(dim=-1, keepdim=True)
            w = torch.where(zero_mask, torch.zeros_like(w).scatter_(-1, closest, 1.0), w)
            denom = w.sum(dim=-1, keepdim=True)

        return w / denom

    def _ce(self, logits: torch.Tensor, target: torch.Tensor,
            std: torch.Tensor = None) -> torch.Tensor:
        # 1. Ép kiểu float32 để đảm bảo độ chính xác số học (nhất là dưới AMP)
        # Clamp chống inf từ fp16 head: log_softmax([.., inf, ..]) = NaN.
        log_p = torch.log_softmax(torch.clamp(logits.float(), -50.0, 50.0), dim=-1)
        soft = self._soft_target(target.float(), std)

        # 2. -(soft * log_p).sum(dim=-1) tính CE theo từng sample
        # .mean() sẽ tự động lấy trung bình trên toàn bộ chiều Batch (và chiều Heads nếu có)
        return -(soft * log_p).sum(dim=-1).mean()

    def _weights(self, pred: torch.Tensor, std: torch.Tensor = None) -> torch.Tensor:
        """Per-sample/column weights for MSE & CCC; ones when std weighting is off."""
        if self.std_alpha <= 0 or std is None:
            return torch.ones_like(pred)
        return 1.0 / (1.0 + self.std_alpha * torch.nan_to_num(std.float(), nan=0.0))

    def _compute_ccc(self, pred: torch.Tensor, target: torch.Tensor,
                     w: torch.Tensor = None) -> torch.Tensor:
        """Tính CCC theo từng column (dim 0), optionally weighted.

        A variance floor keeps the denominator controlled on the small
        grouped batches (var + tiny variance ~ near-0 -> CCC saturates at 1
        with useless gradients).
        """
        if w is None:
            w = torch.ones_like(pred)
        ws = w.sum(dim=0)                            # (ncol,)

        pred_mean = (pred * w).sum(dim=0) / ws
        target_mean = (target * w).sum(dim=0) / ws

        pred_var = ((pred - pred_mean) ** 2 * w).sum(dim=0) / ws
        target_var = ((target - target_mean) ** 2 * w).sum(dim=0) / ws
        cov = ((pred - pred_mean) * (target - target_mean) * w).sum(dim=0) / ws

        denom = pred_var + target_var + (pred_mean - target_mean) ** 2 + self.ccc_var_floor
        ccc = 2 * cov / denom
        return (1.0 - ccc).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, logits: torch.Tensor = None,
                std: torch.Tensor = None, compound_ids: torch.Tensor = None) -> torch.Tensor:
        # 1. Ép kiểu float32 NGAY TỪ ĐẦU cho toàn bộ phép tính để ổn định AMP
        pred = pred.float()
        target = target.float()
        w = self._weights(pred, std)

        # 2. MSE Loss (weighted by 1/(1+alpha*std) if enabled)
        if self.std_alpha > 0 and std is not None:
            mse_loss = ((w * (pred - target) ** 2).sum()) / w.sum().clamp(min=1.0)
        else:
            mse_loss = self.mse(pred, target)
        loss = mse_loss

        # 3. CCC Loss
        if self.ccc_weight > 0:
            ccc_loss = self._compute_ccc(pred, target, w)
            loss = loss + self.ccc_weight * ccc_loss

        # 4. Pairwise Ranking Loss (chỉ giữ cặp cùng compound khi compound_ids được cấp)
        if self.lambda_rank > 0:
            rank_loss = margin_rank_loss(
                pred, target, margin=self.rank_margin, compound_ids=compound_ids,
                mode=self.rank_margin_mode,
            )
            loss = loss + self.lambda_rank * rank_loss

        # 5. Ordinal: Gaussian soft-target CE on the bin logits
        if self.ce_weight > 0 and logits is not None:
            loss = loss + self.ce_weight * self._ce(logits, target, std)

        return loss