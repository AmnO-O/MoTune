"""Single-target scoring model: one Gaussian per sample.

One sample = one sentence, one active target. The sentence runs through ALL
22 layers of the pretrained mmBERT backbone (LoRA adapters tune it); the
FINAL layer's hidden states are pooled over the span(s) that belong to the
active target (see :mod:`src.targets`) and fed to that target's ``GaussHead``:

    target  pooled span               head        label
    "mod"   modifier span             mod_gauss   ModAvg
    "head"  head-noun span            head_gauss  HeadAvg
    "pv"    mod + head (compound)     pv_gauss    Avg

Rows carry an explicit ``batch['target']``, so a single batch may mix
targets -- that is what enables the 3x data augmentation (each sample is
prompted once per target, giving 3N independent training rows). Only the
active target's prediction receives a gradient: inactive heads output zeros,
and their supervised loss is masked out via label masks.

No concatenation of spans, no learned query, no stacked fusion, no
literalness cosine, no intermediate exits. The backbone does the composition;
the readout just pools the marked span.

Interface is identical to ``MMBertModel`` so the trainer, LoRA, and
freeze/unfreeze wiring work unchanged:

  * ``.lm`` backbone named exactly ``lm`` (``apply_lora`` / ``unfreeze_top_layers``
    both walk ``model.lm``),
  * ``.base_model`` alias (not registered as a child, as in ``MMBertModel``),
  * ``.mod_gauss`` / ``.head_gauss`` / ``.pv_gauss`` (the trainer's heads),
  * the same ``forward(batch, with_logits, with_pv)`` return contract.

All fresh-initialized parameters live inside the three ``GaussHead`` modules,
so the frozen-phase training budget (heads only, ``head_lr``) covers the readout
as well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .constants import SCORE_MIN, SCORE_MAX
from .heads import GaussHead
from .model import FusionBlock
from .targets import TARGETS, pool_active, pool_span, row_targets, target_selector


class StaticFusion(nn.Module):
    """Cross-attention fusion of the contextual pool and the static pool.

    Instead of ``torch.cat([ctx, static])`` (which doubles the head input and
    forces a Linear to re-project on every forward), treat the two ``(B, H)``
    vectors as a 2-token sequence ``[ctx, static]`` with distinct type
    embeddings, and let a learned query cross-attend over them through
    ``FusionBlock`` layers (the same gated cross-attn + FFN transformer used
    by ``SpanFusion`` in :mod:`src.model`). Output is one ``(B, H)`` vector:
    the transformer itself learns how to weight the contextual vs. the
    general meaning, so the head input dim stays ``H``.

    Freshly initialized and trained in the head group (``head_lr``) like the
    rest of the readout.
    """

    def __init__(self, hidden: int, num_layers: int = 1, num_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FusionBlock(hidden, num_heads, ffn_expansion=2, dropout=dropout)
            for _ in range(max(1, num_layers))
        ])
        self.type_emb = nn.Parameter(torch.empty(2, hidden))
        nn.init.normal_(self.type_emb, std=0.02)
        self.fuse_q = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.fuse_q, std=0.02)

    def forward(self, ctx: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        kv = torch.stack([ctx, static], dim=1)              # (B, 2, H)
        type_ids = torch.tensor([0, 1], device=ctx.device, dtype=torch.long)
        kv = kv + self.type_emb[type_ids]                   # role-aware tokens
        q = self.fuse_q.expand(ctx.size(0), -1, -1)         # (B, 1, H)
        for layer in self.layers:
            q = layer(q, kv)
        return q.squeeze(1)


class CombinedBackboneModel(nn.Module):
    """One Gaussian per row: final-layer pool of the active target's span."""

    def __init__(self, backbone: str, hidden_size: int = 768, dropout: float = 0.2,
                 head_hidden: int = 128, target_prefix: bool = False,
                 static_span: bool = False,
                 static_fuse_layers: int = 3, static_fuse_heads: int = 2):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.target_prefix = target_prefix
        self.static_span = static_span

        self.lm = AutoModel.from_pretrained(backbone)
        # Alias kept for uniform reader code; NOT registered as a child so the
        # state_dict keys/parameters are not duplicated (see src/model.py).
        object.__setattr__(self, 'base_model', self.lm)

        # Learned per-target marker (3 vectors: mod/head/pv). The batch's
        # input_ids carry one of mmBERT's unused vocab ids (7/8/9) right after
        # <bos>; in forward we OVERWRITE that position's base embedding with
        # this module's vector, so the backbone sees a learnable task token
        # from layer 0 WITHOUT touching the frozen 256k embedding matrix.
        self.marker_emb = nn.Embedding(3, hidden_size) if target_prefix else None

        # Readout feature: always the final-layer span pool. With static_span
        # the pool is fused with the span's frozen embedding-table mean (its
        # general, context-free meaning) by a tiny cross-attention transformer
        # (reusing FusionBlock) instead of a raw concat -- the head never sees
        # a doubled input dim, and the transformer learns how to blend
        # "what the word means" with "what it means here".
        self.head_in = hidden_size
        self.static_fuse = None
        if static_span:
            self.static_fuse = StaticFusion(
                hidden_size, static_fuse_layers, static_fuse_heads, dropout=dropout,
            )
        # ONE shared readout head. Every row answers exactly one target and
        # routing picks its prediction, so three per-target heads would just
        # split the (already small) training signal; a single head sees 3x
        # more rows per parameter. The names mod/head/pv_gauss are kept as
        # aliases of the same module so the trainer/run.py wiring is unchanged.
        self.gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)
        object.__setattr__(self, 'mod_gauss', self.gauss)
        object.__setattr__(self, 'head_gauss', self.gauss)
        object.__setattr__(self, 'pv_gauss', self.gauss)

    def _heads(self) -> dict:
        return {'mod': self.gauss, 'head': self.gauss, 'pv': self.gauss}

    # ------------------------------------------------------------------ #
    def _predict(self, hidden: torch.Tensor, batch) -> dict:
        """Per-target (mu, sigma) on the pooled span of that target.

        Single-target mode (``batch['target']`` present): every row has
        exactly ONE active target, so we pool each row's OWN span in a single
        pass (``pool_active``) and run the shared head exactly once for the
        whole batch -- instead of 3 heads-pass (one per target) followed by
        ``torch.where`` cleanup. The head sees each row once, gradients are
        identical, and forward cost is ~3x cheaper for the readout. Joint
        mode (no target) still scores every row on every target.
        """
        static = None
        if self.static_span:
            static = self.lm.get_input_embeddings()(batch['input_ids'])

        targets = row_targets(batch)
        out = {}
        if targets is None:
            for t in TARGETS:
                pool = pool_span(hidden, batch, t)
                if self.static_span:
                    # Transformer fusion of [contextual pool | static pool],
                    # not a raw concat: the two vectors are stacked as tokens
                    # with type embeddings and cross-attended by a learned
                    # query (n FusionBlock layers). Output stays (B, H).
                    pool = self.static_fuse(pool, pool_span(static, batch, t))
                mu, sigma = self.gauss(pool)
                out[t] = (mu, sigma)
            return out

        # --- single-target fast path: one pool + one head pass per batch ---
        pool = pool_active(hidden, batch, targets)
        if self.static_span:
            pool = self.static_fuse(pool, pool_active(static, batch, targets))
        mu, sigma = self.gauss(pool)                       # (B,) routed already
        dev = mu.device
        zero = torch.zeros_like(mu)
        zero_s = torch.zeros_like(sigma)
        for t in TARGETS:
            sel = target_selector(targets, t)
            if sel is None:
                out[t] = (mu, sigma)
            else:
                sel = sel.to(dev)
                out[t] = (torch.where(sel, mu, zero), torch.where(sel, sigma, zero_s))
        return out

    def _route(self, preds: dict, batch) -> tuple:
        """Route each row to the prediction of its own active target."""
        mus, sigmas = {}, {}
        for t in TARGETS:
            mus[t], sigmas[t] = preds[t]
        targets = row_targets(batch)
        if targets is None:
            return (mus['mod'], mus['head'], mus['pv'],
                    sigmas['mod'], sigmas['head'], sigmas['pv'])

        dev = mus['mod'].device
        zero = torch.zeros_like(mus['mod'])
        def select(t: str) -> torch.Tensor:
            return target_selector(targets, t).to(dev)
        mod_pred = torch.where(select('mod'), mus['mod'], zero)
        head_pred = torch.where(select('head'), mus['head'], zero)
        pv_pred = torch.where(select('pv'), mus['pv'], zero)
        mod_sigma = torch.where(select('mod'), sigmas['mod'], zero)
        head_sigma = torch.where(select('head'), sigmas['head'], zero)
        pv_sigma = torch.where(select('pv'), sigmas['pv'], zero)
        return (mod_pred, head_pred, pv_pred,
                mod_sigma, head_sigma, pv_sigma)

    def _forward_gauss(self, batch, with_logits: bool = False, with_pv: bool = False):
        # Marker injection requires per-row targets in the batch. Predict/joint
        # batches (no ``target`` key) fall back to the plain token path even
        # when ``target_prefix`` is on -- the per-target marker is only ever
        # inserted into the input by the dataset for single-target rows.
        if self.target_prefix and 'target' in batch:
            emb = self.lm.get_input_embeddings()
            hidden_states = emb(batch['input_ids'])
            hidden_states = hidden_states.clone()
            codes = batch['target']
            marker = self.marker_emb(codes.to(hidden_states.device))
            hidden_states[:, 1] = marker
            outputs = self.lm(
                inputs_embeds=hidden_states,
                attention_mask=batch['attention_mask'],
                output_hidden_states=False,
            )
        else:
            outputs = self.lm(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=False,
            )
        preds = self._predict(outputs.last_hidden_state, batch)
        (mod_pred, head_pred, pv_pred,
         mod_sigma, head_sigma, pv_sigma) = self._route(preds, batch)

        if not self.training:
            mod_pred = mod_pred.clamp(SCORE_MIN, SCORE_MAX)
            head_pred = head_pred.clamp(SCORE_MIN, SCORE_MAX)
            pv_pred = pv_pred.clamp(SCORE_MIN, SCORE_MAX)

        if with_logits:
            return (mod_pred, head_pred, pv_pred, mod_sigma, head_sigma, pv_sigma) if with_pv \
                else (mod_pred, head_pred, mod_sigma, head_sigma)
        if with_pv:
            return mod_pred, head_pred, pv_pred
        return mod_pred, head_pred

    def forward(self, batch, with_logits: bool = False, with_pv: bool = False):
        return self._forward_gauss(batch, with_logits, with_pv)


def build_combined_model(cfg, device, load_from: Optional[str | Path] = None) -> CombinedBackboneModel:
    """Construct the single-target model, optionally loading LM backbone weights.

    ``load_from`` follows the same contract as ``src.model.build_model``: a
    state dict of the LM backbone only (``model.lm.state_dict()``); the
    GaussHeads always get fresh init.
    """
    model = CombinedBackboneModel(
        cfg.backbone, hidden_size=cfg.hidden_size, dropout=cfg.dropout,
        head_hidden=cfg.head_hidden,
        target_prefix=bool(getattr(cfg, 'target_prefix', False)),
        static_span=bool(getattr(cfg, 'static_span', False)),
        static_fuse_layers=int(getattr(cfg, 'static_fuse_layers', 1)),
        static_fuse_heads=int(getattr(cfg, 'static_fuse_heads', 2)),
    )
    if load_from is not None:
        load_from = Path(load_from)
        if not load_from.is_file():
            raise FileNotFoundError(f'state dict not found: {load_from}')
        state = torch.load(load_from, map_location='cpu', weights_only=True)
        model.lm.load_state_dict(state)
    return model.to(device)