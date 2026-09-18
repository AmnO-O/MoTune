from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_FLOOR = 0.04


class GaussHead(nn.Module):
    """Deviated Gaussian head chuẩn hóa: 
    - Trunk MLP dày hơn với LayerNorm + GELU
    - Tách riêng 2 nhánh chuyên biệt cho Mu và Sigma
    - Khởi tạo Bias thông minh giúp NLL Loss ổn định ngay từ Epoch 0
    """

    def __init__(
        self, 
        in_features: int, 
        hidden: int = 256, 
        dropout: float = 0.1,
        floor: float = SIGMA_FLOOR,
        init_sigma: float = 0.5,
    ):
        super().__init__()
        self.floor = floor

        # Trunk chung: Chiếu vector Transformer về không gian ẩn sạch sẽ
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Nhánh 1: Dự đoán Điểm trung bình (Mu)
        self.mu_branch = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

        # Nhánh 2: Dự đoán Độ bất định (Sigma)
        self.sigma_branch = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

        self._init_weights(init_sigma)

    def _init_weights(self, init_sigma: float):
        # Khởi tạo Linear thông thường
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # CẤU HÌNH ĐẶC BIỆT CHO SIGMA:
        # Giúp Softplus(bias) + floor ≈ init_sigma ngay khi mới bắt đầu train
        target_raw = max(init_sigma - self.floor, 1e-4)
        # Nghịch đảo của softplus: inv_softplus(y) = log(exp(y) - 1)
        init_bias = math.log(math.expm1(target_raw))
        
        last_sigma_layer = self.sigma_branch[-1]
        nn.init.zeros_(last_sigma_layer.weight)  # Trọng số = 0 để ban đầu chưa bị nhiễu bởi input
        nn.init.constant_(last_sigma_layer.bias, init_bias)

    def forward(self, x: torch.Tensor):
        """Returns (mu (B,), sigma (B,)); sigma > floor luôn được đảm bảo."""
        h = self.trunk(x)
        
        mu = self.mu_branch(h).squeeze(-1)
        
        raw_sigma = self.sigma_branch(h).squeeze(-1)
        sigma = F.softplus(raw_sigma) + self.floor
        
        return mu, sigma