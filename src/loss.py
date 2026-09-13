import torch
import torch.nn as nn

from src.constants import SCORE_MAX, SCORE_MIN


def margin_rank_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Pairwise hinge loss scaled by target difference to penalize order violations."""
    if pred.ndim > 1:
        # Xử lý cho từng output head riêng biệt nếu input là 2D
        losses = [margin_rank_loss(pred[:, i], target[:, i], margin) for i in range(pred.shape[1])]
        return torch.stack(losses).mean()

    n = pred.shape[0]
    if n < 2:
        return pred.sum() * 0.0

    target_diff = target[:, None] - target[None, :]   # y_i - y_j
    pred_diff = pred[:, None] - pred[None, :]         # p_i - p_j
    
    # Chỉ xét các cặp y_i > y_j
    mask = target_diff > 0
    if not mask.any():
        return pred.sum() * 0.0

    # Margin linh hoạt dựa trên khoảng cách target thực tế, tránh phạt vô lý các cặp gần nhau
    dynamic_margin = torch.clamp(target_diff[mask], max=margin)
    hinge = torch.relu(dynamic_margin - pred_diff[mask])
    
    return hinge.mean()


class CombinedLoss(nn.Module):
    """MSE + CCC + Pairwise Ranking Loss + optional Gaussian soft-target CE.

    CE is only active when `ce_weight > 0` AND the caller passes `logits`
    (ordinal 'softmax' head mode). Soft targets are a Gaussian centred on the
    (continuous) target so a float label like 2.34 spreads mass over the bins.
    """

    def __init__(self, ccc_weight: float = 0.5, lambda_rank: float = 0.1, rank_margin: float = 0.1,
                 ce_weight: float = 0.0, num_bins: int = 6, bin_sigma: float = 0.5,
                 use_label_std: bool = True):
        super().__init__()
        self.ccc_weight = ccc_weight
        self.lambda_rank = lambda_rank
        self.rank_margin = rank_margin
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

    def _compute_ccc(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Tính CCC theo từng column (dim 0)."""
        pred_mean = pred.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)
        
        pred_var = pred.var(dim=0, unbiased=False, keepdim=True)
        target_var = target.var(dim=0, unbiased=False, keepdim=True)
        
        cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0, keepdim=True)
        denom = pred_var + target_var + (pred_mean - target_mean) ** 2
        
        ccc = (2 * cov) / (denom + 1e-8)
        return (1.0 - ccc).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, logits: torch.Tensor = None,
                std: torch.Tensor = None) -> torch.Tensor:
        # 1. Ép kiểu float32 NGAY TỪ ĐẦU cho toàn bộ phép tính để ổn định AMP
        pred = pred.float()
        target = target.float()

        # 2. MSE Loss
        mse_loss = self.mse(pred, target)
        loss = mse_loss

        # 3. CCC Loss
        if self.ccc_weight > 0:
            ccc_loss = self._compute_ccc(pred, target)
            loss = loss + self.ccc_weight * ccc_loss

        # 4. Pairwise Ranking Loss
        if self.lambda_rank > 0:
            rank_loss = margin_rank_loss(pred, target, margin=self.rank_margin)
            loss = loss + self.lambda_rank * rank_loss

        # 5. Ordinal: Gaussian soft-target CE on the bin logits
        if self.ce_weight > 0 and logits is not None:
            loss = loss + self.ce_weight * self._ce(logits, target, std)

        return loss