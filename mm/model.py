"""mmBERT scoring model: marker-free span pooling + LoRA + pretrained MLM head.

Loads the backbone as an ``AutoModelForMaskedLM`` so the SAME forward pass
yields both encoder hidden states (for span pooling) and full-vocabulary
logits (for MLM-warmup / LM-predictability features), with no marker tokens
and therefore no embedding resize.

Head input (compact, ported): ``cos(mod, head)`` + mod/head span embeddings
(+ optional learned attention pooling) + span-length fractions + context
(mean and/or CLS). Two regressors score the modifier and the head.

LoRA is implemented inline (no ``peft`` dependency): target ``nn.Linear``
modules are wrapped in ``LoraAdapter`` (keeps the frozen base path), trained,
then ``merge_lora`` folds ``scaling * B @ A`` into the base weight and unwraps
them, leaving a plain backbone for downstream phases.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from .constants import SCORE_MAX, SCORE_MIN


def _backbone(model: nn.Module) -> nn.Module:
    """The base transformer (BERT / ModernBERT / Llama-style) inside the wrapper."""
    lm = model.lm
    for attr in ('model', 'bert', 'base_model', 'transformer'):
        if hasattr(lm, attr):
            return getattr(lm, attr)
    raise AttributeError("Cannot locate the base transformer in the MLM model")


def _backbone_embeddings(model: nn.Module) -> nn.Module:
    base = _backbone(model)
    for attr in ('embed_tokens', 'embeddings', 'word_embeddings', 'wte'):
        m = getattr(base, attr, None)
        if m is not None:
            return m.word_embeddings if hasattr(m, 'word_embeddings') else m
    raise AttributeError("Cannot locate the embedding module")


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(-1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / counts


def _safe_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor,
                            eps: float = 1e-7) -> torch.Tensor:
    n1 = torch.sqrt((x1 ** 2).sum(dim=1, keepdim=True) + eps)
    n2 = torch.sqrt((x2 ** 2).sum(dim=1, keepdim=True) + eps)
    return (x1 * x2).sum(dim=1, keepdim=True) / (n1 * n2)


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
    """Pools a span; 'mean' or 'attn'. Un-addressable spans fall back to mean
    over the whole sentence so the head never sees a zero vector."""

    def __init__(self, hidden: int, mode: str = 'attn'):
        super().__init__()
        self.mode = mode
        self.attn = AttentionPool(hidden) if mode == 'attn' else None

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.mode == 'attn':
            return self.attn(hidden, mask)
        return _masked_mean(hidden, mask)


# --------------------------------------------------------------------------- #
# LoRA (inline, peft-free)
# --------------------------------------------------------------------------- #
class LoRAAdapter(nn.Module):
    """Freeze-base + low-rank adapter. ``merge()`` folds B@A into base."""

    def __init__(self, linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.linear = linear
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


def _derive_attn_targets(model: nn.Module) -> List[str]:
    """Path-suffix targets for linear modules living under an attention block.

    Returns one candidate per projection, ordered so common names come first.
    Candidates are full-ish paths (digits stripped, last two segments), e.g.
    'attn.Wqkv' / 'attn.Wo' / 'self_attn.q_proj', so a leaf like 'Wo' that is
    also used by the MLP ('mlp.Wo') is NOT captured. Singly-named projections
    (q/k/v/o, *_proj, query/key/value) are returned as bare names.
    """
    tails: List[str] = []
    names: List[str] = []

    def _walk(m: nn.Module, path: str) -> None:
        for name, child in list(m.named_children()):
            full = f'{path}.{name}' if path else name
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
               dropout: float = 0.1, targets: Optional[List[str]] = None) -> List[LoRAAdapter]:
    """Wrap every target Linear in-place (recursive) and return the adapters.

    Targets may be bare suffixes ('q_proj') or full dotted paths
    ('self_attn.q_proj'): a module matches when the full path equals the target
    or ends with '<target>'. If the configured targets match nothing, the
    attention-block leaf names are derived from the real module tree and used
    as a fallback (logs via ``model._lora_targets_used``); if that still finds
    nothing the error lists the Linear layers so lora_targets can be fixed.
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
                if _linear_is(child) and _matches(full):
                    setattr(module, name, LoRAAdapter(child, rank, alpha, dropout))
                    adapters.append(getattr(module, name))
                    paths.append(full)
                elif len(list(child.children())) > 0:
                    _walk(child, full)

        _walk(model, '')

    _run(targets)
    if not adapters:
        derived = _derive_attn_targets(model)
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
                if isinstance(child, nn.Linear):
                    lin.append(full)
                elif len(list(child.children())) > 0:
                    _collect(child, full)

        _collect(model, '')
        raise RuntimeError(
            f'apply_lora matched zero {targets}. Backbone linear layers include, '
            f'e.g.: {lin[:30]} - update lora_targets to match these names.'
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
# prediction head
# --------------------------------------------------------------------------- #
def _build_head(head_in: int, head_hidden: int, out_features: int,
                dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(head_in),
        nn.Linear(head_in, head_hidden),
        nn.GELU(),
        nn.LayerNorm(head_hidden),
        nn.Dropout(dropout),
        nn.Linear(head_hidden, 64),
        nn.GELU(),
        nn.LayerNorm(64),
        nn.Dropout(dropout),
        nn.Linear(64, out_features),
    )


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class MMBertRegressor(nn.Module):
    """Encodes one sentence verbatim and scores modifier / head pair.

    ``forward(batch, with_logits, with_reps, with_lm)`` mirrors the old
    interface: predictions (B,), optional ordinal logits (B, num_bins) and
    optional span reps (B, H) for the consistency loss.
    """

    def __init__(self, backbone: str, hidden_size: int = 768, dropout: float = 0.2,
                 head_mode: str = 'reg', num_bins: int = 6,
                 context_pool: str = 'mean+cls', head_pool: str = 'attn',
                 head_hidden: int = 128, use_lm_features: bool = False):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.head_mode = head_mode
        self.num_bins = num_bins
        self.context_pool = context_pool
        self.head_pool = head_pool
        self.use_lm_features = use_lm_features

        self.lm = AutoModelForMaskedLM.from_pretrained(backbone)
        # Cached reference to the base transformer for the no-logits forward
        # path. Deliberately NOT registered as a child module: registering the
        # same instance under a second name would duplicate every state_dict
        # key/parameter (double optimizer updates, 2x ckpt size, strict-load
        # failures on pre-existing snapshots, and LoRA applied twice).
        object.__setattr__(self, 'base_model', _backbone(self))

        context_dim = hidden_size if context_pool in ('mean', 'cls') else 2 * hidden_size
        self.head_in = hidden_size * 2 + context_dim + 3      # cos + 2 spans + 2 lens + context
        if use_lm_features:
            self.head_in += 4                                  # avg_logp + entropy x mod/head

        self.mod_pool = SpanPool(hidden_size, head_pool)
        self.head_role_pool = SpanPool(hidden_size, head_pool)
        self.out_features = num_bins if head_mode == 'softmax' else 1
        self.register_buffer('bin_centers', torch.linspace(SCORE_MIN, SCORE_MAX, num_bins))
        self.mod_regressor = _build_head(self.head_in, head_hidden, self.out_features, dropout)
        self.head_regressor = _build_head(self.head_in, head_hidden, self.out_features, dropout)

    # ------------------------------------------------------------------ #
    def mask_token_id(self):
        tok = getattr(self.lm.config, 'mask_token_id', None)
        return tok

    def _lm_span_stats(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       mod_mask: torch.Tensor, head_mask: torch.Tensor) -> torch.Tensor:
        """avg logP + entropy of the pretrained MLM head over the two spans.

        Runs one extra forward with the span tokens replaced by [MASK], and
        reads the softmax distribution at those positions (cheap: only the
        span logits are ever materialized). Returns (B, 4).
        """
        mask_id = self.mask_token_id()
        if mask_id is None:
            return torch.zeros(input_ids.size(0), 4, device=input_ids.device)
        ids = input_ids.clone()
        span = (mod_mask | head_mask)
        if not span.any():
            return torch.zeros(input_ids.size(0), 4, device=input_ids.device)
        ids = ids.masked_fill(span, mask_id)
        logp = self.lm(input_ids=ids, attention_mask=attention_mask).logits.float()
        logp = F.log_softmax(logp, dim=-1)                        # (B, L, V)
        p = logp.exp()

        def _stats(mask: torch.Tensor) -> torch.Tensor:
            n = mask.long().sum(-1).clamp(min=1)
            idx = input_ids[:, None, :].expand_as(ids)
            tok_logp = torch.gather(logp, -1, idx.unsqueeze(-1)).squeeze(-1)
            avg = (tok_logp * mask.float()).sum(-1) / n           # (B,)
            ent = -(p * logp * mask.float().unsqueeze(-1)).sum(-1).sum(-1) / n
            return torch.stack([avg, ent], dim=-1)                # (B, 2)

        return torch.cat([_stats(mod_mask), _stats(head_mask)], dim=-1)

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor,
              pool: SpanPool) -> torch.Tensor:
        return pool(hidden, mask)

    def forward(self, batch, with_logits: bool = False, with_reps: bool = False,
                with_lm: bool = False):
        if with_logits or self.use_lm_features:
            outputs = self.lm(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            logits = outputs.logits if with_logits else None
        else:
            outputs = self.base_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            logits = None

        mod_emb = self.mod_pool(hidden, batch['mod_span_mask'])
        head_emb = self.head_role_pool(hidden, batch['head_span_mask'])
        mean_emb = _masked_mean(hidden, batch['attention_mask'])

        if self.context_pool == 'cls':
            context_emb = hidden[:, 0]
        elif self.context_pool == 'mean':
            context_emb = mean_emb
        else:
            context_emb = torch.cat([mean_emb, hidden[:, 0]], dim=1)

        cos = _safe_cosine_similarity(mod_emb, head_emb)
        seq_len = batch['attention_mask'].sum(dim=1).clamp(min=1.0).float()
        mod_len = (batch['mod_span_mask'].float().sum(dim=1) / seq_len).unsqueeze(-1)
        head_len = (batch['head_span_mask'].float().sum(dim=1) / seq_len).unsqueeze(-1)

        feats = [cos, mod_emb, head_emb, mod_len, head_len, context_emb]
        if self.use_lm_features:
            feats.append(self._lm_span_stats(
                batch['input_ids'], batch['attention_mask'],
                batch['mod_span_mask'], batch['head_span_mask']))
        features = torch.cat(feats, dim=1)

        mod_out = self.mod_regressor(features)
        head_out = self.head_regressor(features)

        if self.head_mode == 'softmax':
            mod_out = torch.clamp(mod_out, -50.0, 50.0)
            head_out = torch.clamp(head_out, -50.0, 50.0)
            mod_pred = (mod_out.float().softmax(-1) * self.bin_centers).sum(-1)
            head_pred = (head_out.float().softmax(-1) * self.bin_centers).sum(-1)
            mod_logits, head_logits = mod_out, head_out
        else:
            mod_pred, head_pred = mod_out.squeeze(-1), head_out.squeeze(-1)
            mod_logits = head_logits = logits

        if with_reps and with_logits:
            return mod_pred, head_pred, mod_emb, head_emb, mod_logits, head_logits
        if with_reps:
            return mod_pred, head_pred, mod_emb, head_emb
        if with_logits:
            return mod_pred, head_pred, mod_logits, head_logits
        return mod_pred, head_pred


def build_model(cfg, device, load_from: Optional[str | Path] = None) -> MMBertRegressor:
    """Construct the scoring model, optionally loading a warmup-merged state dict.

    ``load_from`` is a torch state_dict SAVED BY US (e.g.
    ``output_dir/models/warmup_merged.pt``), NOT a Hugging Face directory.
    """
    model = MMBertRegressor(
        cfg.backbone, hidden_size=cfg.hidden_size, dropout=cfg.dropout,
        head_mode=cfg.head_mode, num_bins=cfg.num_bins,
        context_pool=cfg.context_pool, head_pool=cfg.head_pool,
        head_hidden=cfg.head_hidden, use_lm_features=cfg.use_lm_features,
    )
    if load_from is not None:
        load_from = Path(load_from)
        if not load_from.is_file():
            raise FileNotFoundError(f'state dict not found: {load_from}')
        state = torch.load(load_from, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    return model.to(device)