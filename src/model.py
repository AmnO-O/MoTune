import torch
import torch.nn as nn
from transformers import AutoModel


def _masked_mean(hidden, mask):
    """Mean-pool hidden states over the (boolean) span mask. All-zero rows are
    guarded against division by zero."""
    mask = mask.float().unsqueeze(-1)          # (B, L, 1)
    summed = (hidden * mask).sum(dim=1)        # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)    # (B, 1)
    return summed / counts


class ModernBERTRegressor(nn.Module):
    """Encodes ONE span-marked sentence and scores modifier / head compositionality.

    The modifier, head and compound (MWE) representations are the mean-pooled
    hidden states over their respective marked spans. A whole-sequence (context)
    embedding plus four token-level cosine signals are concatenated into the head
    input: cos(mod, mwe), cos(head, mwe), cos(mod, head) and cos(mwe, context) --
    the last one is the strongest single unsupervised compositionality predictor.
    """

    def __init__(self, model_name, hidden_size=768, dropout=0.2, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 4 cosine features + 3 pooled embeddings + context pooling
        self.head_in = hidden_size * 3 + 4

        # Separate head for Modifier score
        self.mod_regressor = nn.Sequential(
            nn.Linear(self.head_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Separate head for Head score
        self.head_regressor = nn.Sequential(
            nn.Linear(self.head_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch):
        outputs = self.bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        hidden = outputs.last_hidden_state

        mod_emb = _masked_mean(hidden, batch['mod_span_mask'])      # (B, H)
        head_emb = _masked_mean(hidden, batch['head_span_mask'])    # (B, H)
        mwe_emb = _masked_mean(hidden, batch['mwe_span_mask'])      # (B, H)
        context_emb = _masked_mean(hidden, batch['attention_mask'])  # (B, H)

        cos_sim_mod = nn.functional.cosine_similarity(
            mod_emb, mwe_emb, dim=1
        ).unsqueeze(1)
        cos_sim_head = nn.functional.cosine_similarity(
            head_emb, mwe_emb, dim=1
        ).unsqueeze(1)
        cos_sim_mod_head = nn.functional.cosine_similarity(
            mod_emb, head_emb, dim=1
        ).unsqueeze(1)
        cos_sim_mwe_cont = nn.functional.cosine_similarity(
            mwe_emb, context_emb, dim=1
        ).unsqueeze(1)
        cos_feats = torch.cat(
            [cos_sim_mod, cos_sim_head, cos_sim_mod_head, cos_sim_mwe_cont], dim=1
        )

        mod_features = torch.cat([cos_feats, mod_emb, mwe_emb, context_emb], dim=1)
        head_features = torch.cat([cos_feats, head_emb, mwe_emb, context_emb], dim=1)

        mod_pred = self.mod_regressor(mod_features).squeeze(-1)
        head_pred = self.head_regressor(head_features).squeeze(-1)

        return mod_pred, head_pred


def build_model(cfg, tokenizer, device, dropout=None):
    """Single source for constructing a regressor whose embedding table matches
    the tokenizer (including the added marker tokens).

    MUST be used by both training and prediction so checkpoint state_dicts are
    shape-identical.
    """
    if dropout is None:
        dropout = cfg.dropout

    model = ModernBERTRegressor(
        cfg.model_name,
        hidden_size=cfg.hidden_size,
        dropout=dropout,
        freeze_bert=True,
    )
    model.bert.resize_token_embeddings(len(tokenizer))
    return model.to(device)