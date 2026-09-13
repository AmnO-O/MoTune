import torch
import torch.nn as nn
from transformers import AutoModel

from src.constants import SCORE_MAX, SCORE_MIN


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states over the (boolean) span mask.
    
    Guarded against zero-mask division.
    """
    mask = mask.float().unsqueeze(-1)          # (B, L, 1)
    summed = (hidden * mask).sum(dim=1)        # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)    # (B, 1)
    return summed / counts


def _safe_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Cosine similarity ổn định số học, tránh lỗi NaN under FP16/AMP cả chiều forward lẫn backward."""
    norm1 = torch.sqrt((x1 ** 2).sum(dim=1, keepdim=True) + eps)
    norm2 = torch.sqrt((x2 ** 2).sum(dim=1, keepdim=True) + eps)
    return (x1 * x2).sum(dim=1, keepdim=True) / (norm1 * norm2)


class ModernBERTRegressor(nn.Module):
    """Encodes ONE span-marked sentence and scores modifier / head compositionality.

    Concatenates pooled embeddings (mod/head/mwe/context) with explicit cosine similarity 
    features to predict modifier and head target scores.

    Two head modes:
      - 'reg': two MLPs predict scalars directly.
      - 'softmax': two MLPs predict logits over `num_bins` ordinal bins whose centers
        are uniformly spaced over [SCORE_MIN, SCORE_MAX]; the output is the expected value E[Y].
        Both paths return scalar predictions, so evaluation / prediction / submission
        code is identical. CE training uses `forward(batch, with_logits=True)`.
    """

    def __init__(self, model_name: str, hidden_size: int = 768, dropout: float = 0.2,
                 freeze_bert: bool = False, head_mode: str = 'reg', num_bins: int = 6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.head_mode = head_mode
        self.num_bins = num_bins

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 4 cosine features + 3 pooled embeddings (hidden_size * 3)
        self.head_in = hidden_size * 3 + 4

        if head_mode == 'softmax':
            self.register_buffer(
                'bin_centers', torch.linspace(SCORE_MIN, SCORE_MAX, num_bins)   # uniform centers
            )
            self.out_features = num_bins
        else:
            self.out_features = 1

        # Separate head for Modifier and Head scores
        self.mod_regressor = self._build_head(dropout)
        self.head_regressor = self._build_head(dropout)

    def _build_head(self, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.head_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.out_features),
        )

    def forward(self, batch, with_logits: bool = False):
        outputs = self.bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        hidden = outputs.last_hidden_state

        mod_emb = _masked_mean(hidden, batch['mod_span_mask'])      # (B, H)
        head_emb = _masked_mean(hidden, batch['head_span_mask'])    # (B, H)
        mwe_emb = _masked_mean(hidden, batch['mwe_span_mask'])      # (B, H)
        context_emb = _masked_mean(hidden, batch['attention_mask'])  # (B, H)

        # Tránh NaN bằng Safe Cosine Similarity
        cos_sim_mod = _safe_cosine_similarity(mod_emb, mwe_emb)
        cos_sim_head = _safe_cosine_similarity(head_emb, mwe_emb)
        cos_sim_mod_head = _safe_cosine_similarity(mod_emb, head_emb)
        cos_sim_mwe_cont = _safe_cosine_similarity(mwe_emb, context_emb)

        cos_feats = torch.cat(
            [cos_sim_mod, cos_sim_head, cos_sim_mod_head, cos_sim_mwe_cont], dim=1
        )

        mod_features = torch.cat([cos_feats, mod_emb, mwe_emb, context_emb], dim=1)
        head_features = torch.cat([cos_feats, head_emb, mwe_emb, context_emb], dim=1)

        mod_out = self.mod_regressor(mod_features)
        head_out = self.head_regressor(head_features)

        if self.head_mode == 'softmax':
            # Chống inf fp16: logits tràn → softmax nan → E[Y] nan → toàn bộ
            # loss nan. Clamp rồi mới softmax (tính ở fp32 cho ổn định).
            mod_out = torch.clamp(mod_out, -50.0, 50.0)
            head_out = torch.clamp(head_out, -50.0, 50.0)
            mod_probs = mod_out.float().softmax(dim=-1)
            head_probs = head_out.float().softmax(dim=-1)
            mod_pred = (mod_probs * self.bin_centers).sum(-1)
            head_pred = (head_probs * self.bin_centers).sum(-1)
            if with_logits:
                return mod_pred, head_pred, mod_out, head_out
            return mod_pred, head_pred

        mod_pred = mod_out.squeeze(-1)
        head_pred = head_out.squeeze(-1)
        return mod_pred, head_pred


def embedding_table(model: ModernBERTRegressor) -> nn.Module:
    """Trả về module embedding chuẩn hóa qua API của HuggingFace."""
    return model.bert.get_input_embeddings()


def build_model(cfg, tokenizer, device, dropout=None) -> ModernBERTRegressor:
    """Tạo mô hình và resize embedding tương thích với tokenizer (chứa các marker token mới)."""
    if dropout is None:
        dropout = cfg.dropout

    model = ModernBERTRegressor(
        cfg.model_name,
        hidden_size=cfg.hidden_size,
        dropout=dropout,
        freeze_bert=False,  # Để Trainer làm nhiệm vụ freeze/unfreeze linh hoạt
        head_mode=cfg.head_mode,
        num_bins=cfg.num_bins,
    )
    
    # Resize embedding khi thêm special marker tokens
    model.bert.resize_token_embeddings(len(tokenizer))
    
    return model.to(device)