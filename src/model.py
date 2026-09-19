"""mmBERT gauss scoring model: marker-free span pooling + LoRA + gauss heads only.

Loads the backbone as a plain ``AutoModel`` (encoder only — the MLM head of
the old MLM-wrapper design is gone, together with the LM-predictability stats
that were its only consumer). No marker tokens and therefore no embedding
resize. No ordinal/regression heads exist in this package; both modifier and
head noun are scored by ``GaussHead`` instances predicting ``(mu, sigma)``.

Head input: per-exit cross-attended mod/head span embeddings (learned pooling)
+ mean-and-CLS context + a per-word literality ``cos(use, prototype)`` scalar,
fused by ``SpanFusion`` (no torch.cat) into one ``[B, H]`` vector.

``CrossSpanAttentionBlock`` and ``SpanFusion`` are new, freshly-
initialized modules with nothing to fall back on but this task's own small
training set. Both use learnable, **ones-initialized** gates (``alpha_attn`` /
``alpha_ffn``) on their attention/FFN contributions: at step 0 they are
exactly the same ``query + attn_out → LayerNorm`` residual that worked in the
previous ungated version, so the feature extractor produces meaningful,
input-dependent features from the first step. As training progresses the gates
are free to move in either direction -- shrink toward zero if the network
finds the attention or FFN residual unnecessary for a given layer, or grow
beyond 1.0 if it needs more emphasis -- giving the model a continuously
learnable knob without sacrificing the known-good initialization point.

LoRA lives in :mod:`src.lora` (no ``peft`` dependency) and is re-exported
here (``apply_lora`` / ``lora_parameters`` / ``merge_lora``) for the trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .constants import SCORE_MAX, SCORE_MIN
from .heads import GaussHead
from .lora import apply_lora, lora_parameters, merge_lora


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(-1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / counts


class AttentionPool(nn.Module):
    """Learned attention pooling over the (1-4 token) span."""

    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.query = nn.Parameter(torch.zeros(hidden))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = (self.linear(hidden) * self.query).sum(-1)        # (B, L)
        # Use -1e4 instead of -1e9 to prevent FP16 underflow to -inf (which produces NaN in softmax)
        scores = scores.masked_fill(~mask, -1e4)
        attn = F.softmax(scores, dim=-1)
        # If a row has no span tokens at all, zero out attention instead of NaN / uniform padding
        has_span = mask.any(dim=-1, keepdim=True)
        attn = torch.where(has_span, attn, torch.zeros_like(attn)).unsqueeze(-1)
        return (hidden * attn).sum(dim=1)


class SpanPool(nn.Module):
    """Learned attention pooling for an addressed constituent span."""

    def __init__(self, hidden: int):
        super().__init__()
        self.attn = AttentionPool(hidden)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden, mask)


class HybridSpanPool(nn.Module):
    """Combines Attention, Mean, and Max pooling for short constituent spans.

    NOT currently wired in anywhere -- ``mod_pool``/``head_role_pool`` below
    still use plain ``SpanPool``. Left in place since it may be worth its own
    ablation arm, but as-is it's dead code; either wire it in behind a flag
    and test it on its own, or remove it so the next reader doesn't wonder
    whether it's silently active.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.attn_pool = AttentionPool(hidden)
        self.fuse = nn.Linear(hidden * 3, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1. Attention Pool
        v_attn = self.attn_pool(hidden, mask)

        # 2. Masked Mean Pool
        mask_f = mask.float().unsqueeze(-1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        v_mean = (hidden * mask_f).sum(dim=1) / counts

        # 3. Masked Max Pool
        hidden_masked = hidden.masked_fill(~mask.unsqueeze(-1), -1e4)
        v_max = hidden_masked.max(dim=1).values
        v_max = torch.where(mask.any(dim=-1, keepdim=True), v_max, torch.zeros_like(v_max))

        # 4. Concatenate & Project back to H
        fused = torch.cat([v_attn, v_mean, v_max], dim=-1)
        return self.norm(self.fuse(fused))


class CrossSpanAttentionBlock(nn.Module):
    """Full Transformer Cross-Attention Block with Multi-Head, FFN, and Residuals.

    Flow:
    query_hidden --+--> Cross-MHA (Key/Val: kv_hidden) --> (*alpha_attn) --> (+) --> LayerNorm --+--> FFN --> (*alpha_ffn) --> (+) --> LayerNorm --> Masking
                   +-- (Residual 1) -----------------------------------------+                   +-- (Residual 2) -------------------+

    ``alpha_attn``/``alpha_ffn`` are learnable scalars, **ones-initialized**:
    at step 0 the residuals are exactly the ungated ``+ attn_out`` /
    ``+ ffn_out`` form that previously worked, so attention gets real
    gradients from the first step and the features are input-dependent; the
    gates can then shrink or grow through training (the reversible
    generalization of a ReZero-style gate that keeps LayerNorm and the
    known-good starting point).

    ``ffn_expansion`` defaults to 2 rather than the more common 4: this is a
    freshly-initialized block with no pretraining to fall back on, and the
    smaller default roughly halves its parameter count (~4.7M vs ~7.1M at
    hidden=768) for a first test. Widen it later if an ablation shows it's
    worth the extra capacity.

    NOTE (efficiency, not correctness): ``query_hidden``/``kv_hidden`` are
    passed in as the FULL sequence, and only ``query_mask`` positions are
    kept at the end (``x * query_mask``) -- so this computes attention
    outputs at every sequence position even though only the ~1-4 span
    positions survive. Correct, but O(seq_len) more compute than necessary;
    gathering the span positions first would be the efficient version if
    this ever becomes a bottleneck.
    """
    def __init__(self, hidden: int = 768, num_heads: int = 2, ffn_expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.alpha_attn = nn.Parameter(torch.ones(1))

        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * ffn_expansion, hidden),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.alpha_ffn = nn.Parameter(torch.ones(1))

    def forward(self, query_hidden: torch.Tensor, kv_hidden: torch.Tensor,
                query_mask: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        
        # --- STAGE 1: Cross-Attention ---
        # Key/value positions outside kv_mask get a FINITE additive penalty
        # (-1e4), never -inf: a fully-masked row then softmaxes to near-uniform
        # weights (harmless mean-fallback) instead of NaN in fp32 or fp16.
        # MHA accepts 2-D/3-D masks only, so build the per-head form
        # (B * num_heads, Tq, Tk) by expanding the per-batch penalty over heads.
        q_len = query_hidden.size(1)
        penalty = (1.0 - kv_mask.unsqueeze(1).float()) * -1e4      # [B, 1, Tk]
        attn_mask = penalty.unsqueeze(0).expand(
            self.mha.num_heads, -1, q_len, -1
        ).reshape(query_hidden.size(0) * self.mha.num_heads, q_len, -1)

        attn_out, _ = self.mha(
            query=query_hidden,
            key=kv_hidden,
            value=kv_hidden,
            attn_mask=attn_mask,
            need_weights=False,
        )

        # Add & Norm (Residual 1)
        x = self.norm1(query_hidden + self.alpha_attn * attn_out)

        # --- STAGE 2: Feed-Forward Network (FFN) ---
        ffn_out = self.ffn(x)

        # Add & Norm (Residual 2)
        x = self.norm2(x + self.alpha_ffn * ffn_out)

        # Masking nhẹ nhàng ở cuối
        return x * query_mask.unsqueeze(-1)

import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    """Khối Transformer Fusion tối ưu (Pre-LN Cross-Attn + FFN với ReZero Init)"""
    def __init__(self, hidden: int, num_heads: int = 2, ffn_expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        # Pre-LN riêng biệt cho Query và Key/Value
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        
        # ReZero Trick: khởi tạo alpha = 0.0 để giữ nguyên thông tin ở step 0
        self.alpha_attn = nn.Parameter(torch.zeros(1))

        # Khối Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * ffn_expansion, hidden),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(hidden)
        self.alpha_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, q: torch.Tensor, kv_toks: torch.Tensor) -> torch.Tensor:
        # 1. Pre-LN Cross-Attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv_toks)
        attn_out, _ = self.attn(query=q_norm, key=kv_norm, value=kv_norm)
        x = q + self.alpha_attn * attn_out

        # 2. Pre-LN FFN
        out = x + self.alpha_ffn * self.ffn(self.norm_ffn(x))
        return out

class SpanFusion(nn.Module):
    """Module Fusion nâng cấp hỗ trợ xếp chồng N Layer (Iterative Refinement)"""
    def __init__(self, hidden: int = 768, num_heads: int = 4, ffn_expansion: int = 2, 
                 dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        
        # 1. Learned Query
        self.fuse_q = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.fuse_q, std=0.02)

        # 2. Xếp chồng N Layer Fusion (Mặc định 2 layers)
        self.layers = nn.ModuleList([
            FusionBlock(hidden, num_heads, ffn_expansion, dropout)
            for _ in range(num_layers)
        ])

        # Scalar & Role embeddings
        self.scalar_embed = nn.Linear(1, hidden)
        self.type_emb = nn.Parameter(torch.empty(5, hidden))
        nn.init.normal_(self.type_emb, std=0.02)

    def forward(self, mod: torch.Tensor, head: torch.Tensor, context: torch.Tensor, cos_: torch.Tensor = None) -> torch.Tensor:
        # --- Gather tokens ---
        ctx_mean, ctx_cls = context.chunk(2, dim=1)
        toks = [mod, head, ctx_mean, ctx_cls]
        type_ids = [0, 1, 2, 3]
        
        if cos_ is not None:
            toks.append(self.scalar_embed(cos_))
            type_ids.append(4)

        type_tensor = torch.tensor(type_ids, device=mod.device, dtype=torch.long)
        kv_toks = torch.stack(toks, dim=1) + self.type_emb[type_tensor]   # [B, T, 768]
        
        # Bắt đầu với learned query khởi tạo
        q = self.fuse_q.expand(kv_toks.size(0), -1, -1)                   # [B, 1, 768]

        # --- Pass qua N Fusion Layers ---
        for layer in self.layers:
            q = layer(q, kv_toks)                                         # q được "tinh lọc" qua từng layer

        return q.squeeze(1)                                               # [B, 768]
    
# --------------------------------------------------------------------------- #
# the model (gauss only)
# --------------------------------------------------------------------------- #
class MMBertModel(nn.Module):
    """Encodes one sentence verbatim and scores modifier / head pair as
    Gaussians ``N(mu, sigma^2)``.

    ``forward(batch, with_logits)``: predictions (B,), optional
    sigma via the "logits" channel (B, 2).
    """

    def __init__(self, backbone: str, hidden_size: int = 768, dropout: float = 0.2,
                 head_hidden: int = 128,
                 gauss_ctx_mod: Optional[Sequence[int]] = None,
                 gauss_ctx_head: Optional[Sequence[int]] = None,
                 gauss_ctx_pv: Optional[Sequence[int]] = None):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        # Hidden-state index 0 is the embedding output, so indices 19/20/21,22
        # correspond to transformer blocks 18/19/20,21.
        self.gauss_ctx_mod = tuple(int(i) for i in (gauss_ctx_mod or (19,)))
        self.gauss_ctx_head = tuple(int(i) for i in (gauss_ctx_head or (20,)))
        self.gauss_ctx_pv = tuple(int(i) for i in (gauss_ctx_pv or (21, 22)))

        self.lm = AutoModel.from_pretrained(backbone)
        # Alias to the encoder (there is no wrapper head anymore), kept so the
        # feature/pooling code reads uniformly. Deliberately NOT registered as
        # a child module: registering the same instance under a second name
        # would duplicate every state_dict key/parameter (double optimizer
        # updates, 2x ckpt size, strict-load failures, and LoRA applied twice).
        object.__setattr__(self, 'base_model', self.lm)

        # No concat feature bundle: the fused pieces (cross-attended span pair,
        # context mean/CLS, literalness scalar) collapse to ONE H-vector per
        # exit via the learned-query SpanFusion attention.
        self.head_in = hidden_size
        self.fusion = SpanFusion(hidden_size, dropout=dropout)

        # Token-level cross-attention between spans, shared across role exits.
        self.cross_attn = CrossSpanAttentionBlock(hidden_size, dropout=dropout)

        self.mod_pool = SpanPool(hidden_size)
        self.head_role_pool = SpanPool(hidden_size)
        self.mod_gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)
        self.head_gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)
        self.pv_gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)

    # ------------------------------------------------------------------ #
    def _prototype_cos(self, input_ids: torch.Tensor, span_mask: torch.Tensor,
                       use_emb: torch.Tensor) -> torch.Tensor:
        """Literality: cosine between a word's contextual USE embedding and the
        static prototype (base/lemma) embedding rows of the span's own tokens.

        High = the word keeps its literal meaning in this context (e.g.
        "market" in "flea market"); low = drift/lexicalised use (e.g. "tower"
        in "ivory tower"). ``use_emb`` must be the span's OWN contextual
        vector (span-mean of the raw role-exit hidden states), not the pooled
        cross-attended output -- cross-attention contaminates a token with its
        counterpart span. Rows with no span contribute 0.
        """
        weight = self.lm.get_input_embeddings().weight
        tok = F.embedding(input_ids, weight)
        n = span_mask.long().sum(-1).clamp(min=1).unsqueeze(-1)
        proto = (tok * span_mask.float().unsqueeze(-1)).sum(1) / n
        # Rows with no span produce a zero vector; cosine_similarity would
        # then divide by norm 0 -> NaN. torch.where fixes the FORWARD value
        # but NaN still flows into the backward graph (0 * NaN = NaN), so
        # substitute ones BEFORE the cosine and select afterwards.
        has_span = span_mask.any(-1, keepdim=True)
        use_safe = torch.where(has_span, use_emb, torch.ones_like(use_emb))
        proto_safe = torch.where(has_span, proto, torch.ones_like(proto))
        cos = F.cosine_similarity(use_safe.float(), proto_safe.float(), dim=-1)
        cos = torch.where(has_span.squeeze(-1), cos, torch.zeros_like(cos))
        return cos.type_as(use_emb).unsqueeze(-1)

    def _context_emb(self, hidden: torch.Tensor, batch) -> torch.Tensor:
        """Mean-and-CLS whole-sentence context for a dedicated exit."""
        mean_emb = _masked_mean(hidden, batch['attention_mask'])
        return torch.cat([mean_emb, hidden[:, 0]], dim=1)

    def _role_hidden(self, layers, hid_all) -> torch.Tensor:
        """Mean selected hidden states and put them on the final-norm scale."""
        hidden = torch.mean(torch.stack([hid_all[i] for i in layers]), dim=0)
        norm = getattr(self.base_model, 'final_norm', None) \
            or getattr(getattr(self.base_model, 'encoder', None), 'final_norm', None)
        return norm(hidden) if norm is not None else hidden

    def _span_pair(self, hidden: torch.Tensor, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pooled (mod, head) span embeddings at one exit.

        Each span's tokens first attend over the OTHER span (token-level,
        before pooling), so the constituents interact before being collapsed.
        """
        mod_mask = batch['mod_span_mask']
        head_mask = batch['head_span_mask']
        mod_tok = self.cross_attn(hidden, hidden, mod_mask, head_mask)
        head_tok = self.cross_attn(hidden, hidden, head_mask, mod_mask)
        return (self.mod_pool(mod_tok, mod_mask),
                self.head_role_pool(head_tok, head_mask))

    def _compose_gauss_feat(self, mod_emb, head_emb, context_emb,
                            cos_=None) -> torch.Tensor:
        """Attention-fused feature vector (no torch.cat): the cross-attended
        span pair, the context mean/CLS, and the literalness scalar become
        role-tagged tokens of one learned-query SpanFusion attention,
        collapsing to a single ``[B, H]`` vector."""
        return self.fusion(mod_emb, head_emb, context_emb, cos_)

    def _features(self, batch):
        """Build one feature bundle per dedicated Gaussian exit."""
        outputs = self.lm(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True,
        )
        hid_all = outputs.hidden_states
        n_layers = len(hid_all)

        mod_hidden = self._role_hidden(
            tuple(i % n_layers for i in self.gauss_ctx_mod), hid_all)
        head_hidden = self._role_hidden(
            tuple(i % n_layers for i in self.gauss_ctx_head), hid_all)
        pv_hidden = self._role_hidden(
            tuple(i % n_layers for i in self.gauss_ctx_pv), hid_all)

        mod_u, mod_v = self._span_pair(mod_hidden, batch)
        head_u, head_v = self._span_pair(head_hidden, batch)
        pv_u, pv_v = self._span_pair(pv_hidden, batch)

        mod_exit_emb, head_exit_emb = mod_u, head_v

        # Literality is a property of the WORD's own contextual vector: compare
        # the span-mean of the RAW role-exit hidden states against the static
        # prototype, NOT the pooled output of cross-attention. The cross-attn
        # mixes in the counterpart span ("flea" attends over "market"), so its
        # pooled vector would be contaminated as a literalness measurement.
        # (Restored -- this was computed and then discarded in the previous
        # version: mod_use/head_use were built but cos_mod/cos_head/cos_pv
        # were hardcoded to None and the three lines below were commented out,
        # so the feature never reached the fusion module or the heads.)
        mod_use = _masked_mean(mod_hidden, batch['mod_span_mask'])
        head_use = _masked_mean(head_hidden, batch['head_span_mask'])

        cos_mod = self._prototype_cos(batch['input_ids'], batch['mod_span_mask'], mod_use)
        cos_head = self._prototype_cos(batch['input_ids'], batch['head_span_mask'], head_use)
        cos_pv = 0.5 * (
            self._prototype_cos(batch['input_ids'], batch['mod_span_mask'],
                                _masked_mean(pv_hidden, batch['mod_span_mask']))
            + self._prototype_cos(batch['input_ids'], batch['head_span_mask'],
                                  _masked_mean(pv_hidden, batch['head_span_mask'])))

        mod_feat = self._compose_gauss_feat(
            mod_u, mod_v, self._context_emb(mod_hidden, batch), cos_mod)
        head_feat = self._compose_gauss_feat(
            head_u, head_v, self._context_emb(head_hidden, batch), cos_head)
        pv_feat = self._compose_gauss_feat(
            pv_u, pv_v, self._context_emb(pv_hidden, batch), cos_pv)

        return mod_feat, head_feat, pv_feat, mod_exit_emb, head_exit_emb, pv_u, pv_v

    def _forward_gauss(self, batch, with_logits: bool = False, with_pv: bool = False):
        """Predict N(mu, sigma^2) per span; sigma travels the "logits" channel."""
        (mod_feat, head_feat, pv_feat, mod_exit_emb, head_exit_emb,
         pv_mod_exit_emb, pv_head_exit_emb) = self._features(batch)
        mod_mu, mod_sigma = self.mod_gauss(mod_feat)
        head_mu, head_sigma = self.head_gauss(head_feat)
        pv_mu, pv_sigma = self.pv_gauss(pv_feat)
        # Clamp only where a bounded score is reported (eval/inference). During
        # training the raw mu flows into the losses: clamp has zero gradient
        # outside [SCORE_MIN, SCORE_MAX], so an out-of-range output would be
        # pinned at the boundary with no (mu - target) pull-back. The KL/CCC
        # terms are the correct regulator for range drift while training.
        if self.training:
            mod_pred, head_pred, pv_pred = mod_mu, head_mu, pv_mu
        else:
            mod_pred = mod_mu.clamp(SCORE_MIN, SCORE_MAX)
            head_pred = head_mu.clamp(SCORE_MIN, SCORE_MAX)
            pv_pred = pv_mu.clamp(SCORE_MIN, SCORE_MAX)
        if with_logits:
            return (mod_pred, head_pred, pv_pred, mod_sigma, head_sigma, pv_sigma) if with_pv \
                else (mod_pred, head_pred, mod_sigma, head_sigma)
        if with_pv:
            return mod_pred, head_pred, pv_pred
        return mod_pred, head_pred

    def forward(self, batch, with_logits: bool = False, with_pv: bool = False):
        return self._forward_gauss(batch, with_logits, with_pv)


def build_model(cfg, device, load_from: Optional[str | Path] = None) -> MMBertModel:
    """Construct the gauss scoring model, optionally loading a backbone state dict.

    ``load_from`` is a torch state_dict with the LM backbone keys only
    (``model.lm.state_dict()``); scorer heads always get fresh init.

    Dispatches to :mod:`src.model_combined` when ``cfg.model_backend == 'combined'``.
    """
    if getattr(cfg, 'model_backend', 'exits') == 'combined':
        from .model_combined import build_combined_model
        return build_combined_model(cfg, device, load_from=load_from)
    model = MMBertModel(
        cfg.backbone, hidden_size=cfg.hidden_size, dropout=cfg.dropout,
        head_hidden=cfg.head_hidden,
        gauss_ctx_mod=cfg.gauss_ctx_mod,
        gauss_ctx_head=cfg.gauss_ctx_head, gauss_ctx_pv=cfg.gauss_ctx_pv,
    )
    if load_from is not None:
        load_from = Path(load_from)
        if not load_from.is_file():
            raise FileNotFoundError(f'state dict not found: {load_from}')
        state = torch.load(load_from, map_location='cpu', weights_only=True)
        model.lm.load_state_dict(state)
    return model.to(device)