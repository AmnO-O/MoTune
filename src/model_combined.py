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
from .targets import TARGETS, pool_span, row_targets, target_selector


class CombinedBackboneModel(nn.Module):
    """One Gaussian per row: final-layer pool of the active target's span."""

    def __init__(self, backbone: str, hidden_size: int = 768, dropout: float = 0.2,
                 head_hidden: int = 128):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size

        self.lm = AutoModel.from_pretrained(backbone)
        # Alias kept for uniform reader code; NOT registered as a child so the
        # state_dict keys/parameters are not duplicated (see src/model.py).
        object.__setattr__(self, 'base_model', self.lm)

        self.head_in = hidden_size          # single span pool per target
        self.mod_gauss = GaussHead(hidden_size, head_hidden, dropout=dropout)
        self.head_gauss = GaussHead(hidden_size, head_hidden, dropout=dropout)
        self.pv_gauss = GaussHead(hidden_size, head_hidden, dropout=dropout)

    def _heads(self) -> dict:
        return {'mod': self.mod_gauss, 'head': self.head_gauss, 'pv': self.pv_gauss}

    # ------------------------------------------------------------------ #
    def _predict(self, hidden: torch.Tensor, batch) -> dict:
        """Per-target (mu, sigma) on the pooled span of that target."""
        out = {}
        for t in TARGETS:
            pool = pool_span(hidden, batch, t)
            mu, sigma = self._heads()[t](pool)
            sel = target_selector(row_targets(batch), t)
            if sel is not None and not sel.any():
                mu = torch.zeros_like(mu)
                sigma = torch.zeros_like(sigma)
            out[t] = (mu, sigma)
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

        zero = torch.zeros_like(mus['mod'])
        mod_pred = torch.where(target_selector(targets, 'mod'), mus['mod'], zero)
        head_pred = torch.where(target_selector(targets, 'head'), mus['head'], zero)
        pv_pred = torch.where(target_selector(targets, 'pv'), mus['pv'], zero)
        mod_sigma = torch.where(target_selector(targets, 'mod'), sigmas['mod'], zero)
        head_sigma = torch.where(target_selector(targets, 'head'), sigmas['head'], zero)
        pv_sigma = torch.where(target_selector(targets, 'pv'), sigmas['pv'], zero)
        return (mod_pred, head_pred, pv_pred,
                mod_sigma, head_sigma, pv_sigma)

    def _forward_gauss(self, batch, with_logits: bool = False, with_pv: bool = False):
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
    )
    if load_from is not None:
        load_from = Path(load_from)
        if not load_from.is_file():
            raise FileNotFoundError(f'state dict not found: {load_from}')
        state = torch.load(load_from, map_location='cpu', weights_only=True)
        model.lm.load_state_dict(state)
    return model.to(device)