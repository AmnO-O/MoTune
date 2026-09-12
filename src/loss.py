import torch
import torch.nn as nn


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
    """MSE + CCC + Pairwise Ranking Loss (Tối ưu cho Spearman's Rho & AMP)."""

    def __init__(self, ccc_weight: float = 0.5, lambda_rank: float = 0.1, rank_margin: float = 0.1):
        super().__init__()
        self.ccc_weight = ccc_weight
        self.lambda_rank = lambda_rank
        self.rank_margin = rank_margin
        self.mse = nn.MSELoss()

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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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

        return loss