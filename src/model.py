"""mmBERT gauss scoring model: marker-free span pooling + LoRA + gauss heads only.

Loads the backbone as a plain ``AutoModel`` (encoder only — the MLM head of
the old MLM-wrapper design is gone, together with the LM-predictability stats
that were its only consumer). No marker tokens and therefore no embedding
resize. No ordinal/regression heads exist in this package; both modifier and
head noun are scored by ``GaussHead`` instances predicting ``(mu, sigma)``.

Head input: mod/head span embeddings (learned attention pooling) + span-length
fractions + mean-and-CLS context, plus a per-word
literality ``cos(use, prototype)`` scalar (no cross-talk between the two
words).

LoRA is implemented inline (no ``peft`` dependency): target ``nn.Linear``
modules are wrapped in ``LoRAAdapter`` (keeps the frozen base path), trained,
then ``merge_lora`` folds ``scaling * B @ A`` into the base weight and unwraps
them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from .constants import SCORE_MAX, SCORE_MIN
from .heads import GaussHead


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


# --------------------------------------------------------------------------- #
# LoRA (inline, peft-free)
# --------------------------------------------------------------------------- #
class LoRAAdapter(nn.Module):
    """Freeze-base + low-rank adapter. ``merge()`` folds B@A into base."""

    def __init__(self, linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.linear = linear
        # Hard invariant: the base weight is NEVER trainable once wrapped
        # (docstring: "Freeze-base"). Some callers (e.g. unfreeze_top_layers)
        # walk the layer tree and flip requires_grad=True on everything, which
        # would double-train base+adapter — the wrap points re-freeze below.
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.a = nn.Linear(linear.in_features, r, bias=False)
        self.b = nn.Linear(r, linear.out_features, bias=False)
        # adapters may be created after the base already sits on GPU/device
        dev = linear.weight.device
        if dev.type not in ('cpu', 'meta'):
            self.a.to(dev)
            self.b.to(dev)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scaling * self.b(self.lora_dropout(self.a(x)))

    def merge(self) -> None:
        with torch.no_grad():
            self.linear.weight += self.scaling * (self.b.weight @ self.a.weight)


def _linear_is(module: nn.Module) -> bool:
    """True for nn.Linear and any duck-typed linear (custom proj classes)."""
    return (
        isinstance(module, nn.Linear)
        or (hasattr(module, 'weight') and hasattr(module, 'in_features')
            and hasattr(module, 'out_features') and hasattr(module, 'bias'))
    )


def _layer_idx(full: str) -> Optional[int]:
    """Index of the transformer layer in a dotted path, e.g. ``layers.14`` -> 14."""
    segs = full.split('.')
    for i, s in enumerate(segs[:-1]):
        if s == 'layers' and segs[i + 1].isdigit():
            return int(segs[i + 1])
    return None


def _derive_attn_targets(model: nn.Module, from_layer: int = 0) -> List[str]:
    """Path-suffix targets for linear modules living under an attention block.

    Returns one candidate per projection (layers >= ``from_layer`` only),
    ordered so common names come first. Candidates are full-ish paths (digits
    stripped, last two segments), e.g. 'attn.Wqkv' / 'attn.Wo' /
    'self_attn.q_proj', so a leaf like 'Wo' that is also used by the MLP
    ('mlp.Wo') is NOT captured. Singly-named projections (q/k/v/o, *_proj,
    query/key/value) are returned as bare names.
    """
    tails: List[str] = []
    names: List[str] = []

    def _walk(m: nn.Module, path: str) -> None:
        for name, child in list(m.named_children()):
            full = f'{path}.{name}' if path else name
            layer = _layer_idx(full)
            if layer is not None and layer < from_layer:
                if len(list(child.children())) > 0:
                    _walk(child, full)
                continue
            if _linear_is(child) and any(
                    s in path for s in ('self_attn', 'attention', 'mha',
                                        'mhsa', 'attn')):
                segs = [s for s in full.split('.') if not s.isdigit()]
                tails.append('.'.join(segs[-2:]))
                names.append(name)
            elif len(list(child.children())) > 0:
                _walk(child, full)

    _walk(model, '')
    if not tails:
        return []
    preferred = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'Wqkv', 'Wo',
                 'q', 'k', 'v', 'o', 'query', 'key', 'value',
                 'query_proj', 'key_proj', 'value_proj']
    pri = {n: i for i, n in enumerate(preferred)}

    best: Dict[str, str] = {}
    for t, n in zip(tails, names):
        if n not in best or len(t) < len(best[n]):
            best[n] = t

    ordered = sorted(best.items(), key=lambda kv: pri.get(kv[0], len(preferred)))
    safe_bare = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'q', 'k', 'v', 'o',
                 'query', 'key', 'value', 'query_proj', 'key_proj',
                 'value_proj'}
    out: List[str] = []
    for n, t in ordered:
        out.append(n if n in safe_bare else t)
    return out


def apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16,
               dropout: float = 0.1, targets: Optional[List[str]] = None,
               from_layer: int = 0) -> List[LoRAAdapter]:
    """Wrap every target Linear in-place (recursive) and return the adapters.

    Targets may be bare suffixes ('q_proj') or full dotted paths
    ('self_attn.q_proj'): a module matches when the full path equals the target
    or ends with '<target>'. ``from_layer`` restricts the window to transformer
    layers with index >= ``from_layer`` (0 = all layers). If the configured
    targets match nothing, the attention-block leaf names are derived from the
    real module tree and used as a fallback (logs via ``model._lora_targets_used``);
    if that still finds nothing the error lists the Linear layers so
    lora_targets can be fixed.
    """
    if targets is None:
        targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    targets = [str(t).strip() for t in targets]
    adapters: List[LoRAAdapter] = []
    paths: List[str] = []

    def _run(tgt: List[str]) -> None:
        def _matches(full: str) -> bool:
            return any(full == t or full.endswith('.' + t) for t in tgt)

        def _walk(module: nn.Module, path: str) -> None:
            for name, child in list(module.named_children()):
                full = f'{path}.{name}' if path else name
                layer = _layer_idx(full)
                if layer is not None and layer < from_layer:
                    # outside the LoRA window; keep descending to reach deeper layers
                    if len(list(child.children())) > 0:
                        _walk(child, full)
                    continue
                if _linear_is(child) and _matches(full):
                    child.weight.requires_grad = False
                    if child.bias is not None:
                        child.bias.requires_grad = False
                    setattr(module, name, LoRAAdapter(child, rank, alpha, dropout))
                    adapters.append(getattr(module, name))
                    paths.append(full)
                elif len(list(child.children())) > 0:
                    _walk(child, full)

        _walk(model, '')

    _run(targets)
    if not adapters:
        derived = _derive_attn_targets(model, from_layer=from_layer)
        if derived:
            merged = list(dict.fromkeys(targets + derived))
            _run(merged)
            setattr(model, '_lora_targets_used', merged)
    # diagnostics for callers / logs
    setattr(model, '_lora_paths', list(dict.fromkeys(paths)))
    if not adapters:
        # surface the actual projection names so lora_targets can be fixed
        lin = []

        def _collect(m: nn.Module, path: str) -> None:
            for name, child in list(m.named_children()):
                full = f'{path}.{name}' if path else name
                layer = _layer_idx(full)
                if layer is not None and layer < from_layer:
                    continue
                if isinstance(child, nn.Linear):
                    lin.append(full)
                elif len(list(child.children())) > 0:
                    _collect(child, full)

        _collect(model, '')
        raise RuntimeError(
            f'apply_lora matched zero {targets} (layers >= {from_layer}). '
            f'Backbone linear layers include, e.g.: {lin[:30]} - '
            f'update lora_targets to match these names.'
        )
    return adapters


def lora_parameters(adapters: List[LoRAAdapter]) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for a in adapters:
        params += list(a.a.parameters()) + list(a.b.parameters())
    return params


def merge_lora(model: nn.Module, adapters: List[LoRAAdapter]) -> None:
    """Fold adapter deltas into the base weights, then unwrap the wrappers.

    The returned module has the same parameter layout as a freshly loaded
    backbone (the adapters are gone), so it can be re-adaptered cleanly.
    """
    for a in adapters:
        a.merge()

    def _unwrap(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, LoRAAdapter):
                setattr(module, name, child.linear)
            else:
                _unwrap(child)

    _unwrap(model)


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

        # 2 spans + 2 length fractions + mean-and-CLS context + literalness.
        self.head_in = hidden_size * 4 + 2
        # Each output branch gets ONLY its own word's cos(use, prototype)
        # tacked on (torch.cat, last axis), so mod_out never sees head's
        # literalness and vice-versa. That extra column IS part of head_in.
        self.head_in += 1

        self.mod_pool = SpanPool(hidden_size)
        self.head_role_pool = SpanPool(hidden_size)
        self.mod_gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)
        self.head_gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)
        self.pv_gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)

    # ------------------------------------------------------------------ #
    def _prototype_cos(self, input_ids: torch.Tensor, span_mask: torch.Tensor,
                       use_emb: torch.Tensor) -> torch.Tensor:
        """Literality: cosine between the contextualised USE embedding and the
        static prototype (base/lemma) embedding rows of the span's own tokens.

        High = the word keeps its literal meaning in this context (e.g.
        "market" in "flea market"); low = drift/lexicalised use (e.g. "tower"
        in "ivory tower"). Rows with no span contribute 0.
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

    def _compose_gauss_feat(self, mod_emb, head_emb, mod_len, head_len,
                            context_emb) -> torch.Tensor:
        """Feature bundle: two spans, two lengths, and role-specific context."""
        return torch.cat([mod_emb, head_emb, mod_len, head_len, context_emb], dim=1)

    def _features(self, batch):
        """Build one feature bundle per dedicated Gaussian exit."""
        outputs = self.lm(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True,
        )
        hid_all = outputs.hidden_states
        n_layers = len(hid_all)

        seq_len = batch['attention_mask'].sum(dim=1).clamp(min=1.0).float()
        mod_len = (batch['mod_span_mask'].float().sum(dim=1) / seq_len).unsqueeze(-1)
        head_len = (batch['head_span_mask'].float().sum(dim=1) / seq_len).unsqueeze(-1)

        mod_hidden = self._role_hidden(
            tuple(i % n_layers for i in self.gauss_ctx_mod), hid_all)
        head_hidden = self._role_hidden(
            tuple(i % n_layers for i in self.gauss_ctx_head), hid_all)
        pv_hidden = self._role_hidden(
            tuple(i % n_layers for i in self.gauss_ctx_pv), hid_all)
        
        mod_exit_emb = self.mod_pool(mod_hidden, batch['mod_span_mask'])
        head_exit_emb = self.head_role_pool(head_hidden, batch['head_span_mask'])
        pv_mod_emb = self.mod_pool(pv_hidden, batch['mod_span_mask'])
        pv_head_emb = self.head_role_pool(pv_hidden, batch['head_span_mask'])

        mod_feat = self._compose_gauss_feat(
            mod_exit_emb, self.head_role_pool(mod_hidden, batch['head_span_mask']),
            mod_len, head_len, self._context_emb(mod_hidden, batch))
        
        head_feat = self._compose_gauss_feat(
            self.mod_pool(head_hidden, batch['mod_span_mask']), head_exit_emb,
            mod_len, head_len, self._context_emb(head_hidden, batch))
        pv_feat = self._compose_gauss_feat(
            pv_mod_emb, pv_head_emb, mod_len, head_len,
            self._context_emb(pv_hidden, batch))
        
        return mod_feat, head_feat, pv_feat, mod_exit_emb, head_exit_emb, pv_mod_emb, pv_head_emb

    def _forward_gauss(self, batch, with_logits: bool = False, with_pv: bool = False):
        """Predict N(mu, sigma^2) per span; sigma travels the "logits" channel."""
        (mod_feat, head_feat, pv_feat, mod_exit_emb, head_exit_emb,
         pv_mod_exit_emb, pv_head_exit_emb) = self._features(batch)
        cos_mod = self._prototype_cos(
            batch['input_ids'], batch['mod_span_mask'], mod_exit_emb)
        cos_head = self._prototype_cos(
            batch['input_ids'], batch['head_span_mask'], head_exit_emb)
        cos_pv = 0.5 * (self._prototype_cos(
            batch['input_ids'], batch['mod_span_mask'], pv_mod_exit_emb) + self._prototype_cos(
            batch['input_ids'], batch['head_span_mask'], pv_head_exit_emb))
        mod_mu, mod_sigma = self.mod_gauss(torch.cat([mod_feat, cos_mod], dim=1))
        head_mu, head_sigma = self.head_gauss(torch.cat([head_feat, cos_head], dim=1))
        pv_mu, pv_sigma = self.pv_gauss(torch.cat([pv_feat, cos_pv], dim=1))
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
    """
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
