"""mmBERT gauss scoring model: marker-free span pooling + LoRA + gauss heads only.

Loads the backbone as an ``AutoModelForMaskedLM`` so the SAME forward pass
yields both encoder hidden states (for span pooling) and, when requested, the
LM-predictability / prototype signals — with no marker tokens and therefore
no embedding resize. No ordinal/regression heads exist in this package; both
modifier and head noun are scored by ``GaussHead`` instances predicting
``(mu, sigma)``.

Head input: mod/head span embeddings (+ optional learned attention pooling)
+ span-length fractions + context (mean and/or CLS), plus optional
LM-predictability stats and a per-word literality ``cos(use, prototype)``
scalar (no cross-talk between the two words).

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
from transformers import AutoModelForMaskedLM

from .constants import SCORE_MAX, SCORE_MIN
from .heads import GaussHead


def _backbone(model: nn.Module) -> nn.Module:
    """The base transformer (BERT / ModernBERT / Llama-style) inside the wrapper."""
    lm = model.lm
    for attr in ('model', 'bert', 'base_model', 'transformer'):
        if hasattr(lm, attr):
            return getattr(lm, attr)
    raise AttributeError("Cannot locate the base transformer in the MLM model")


def _backbone_embeddings(model: nn.Module) -> nn.Module:
    lm = getattr(model, 'lm', model)
    if hasattr(lm, 'get_input_embeddings') and lm.get_input_embeddings() is not None:
        return lm.get_input_embeddings()
    base = _backbone(model)
    if hasattr(base, 'get_input_embeddings') and base.get_input_embeddings() is not None:
        return base.get_input_embeddings()
    for attr in ('embed_tokens', 'embeddings', 'word_embeddings', 'wte', 'tok_embeddings'):
        m = getattr(base, attr, None)
        if m is not None:
            for sub in ('tok_embeddings', 'word_embeddings'):
                if hasattr(m, sub):
                    return getattr(m, sub)
            return m
    raise AttributeError("Cannot locate the embedding module")


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

    ``forward(batch, with_logits, with_reps, with_lm)`` mirrors the old
    interface: predictions (B,), optional sigma via the "logits" channel
    (B, 2), and optional span reps (B, H) for the consistency loss.
    """

    def __init__(self, backbone: str, hidden_size: int = 768, dropout: float = 0.2,
                 context_pool: str = 'mean+cls', head_pool: str = 'attn',
                 head_hidden: int = 128, use_lm_features: bool = False,
                 use_proto_cos: bool = False,
                 span_layers: Optional[Sequence[int]] = None,
                 context_layers: Optional[Sequence[int]] = None,
                 gauss_dedicated: bool = False,
                 gauss_ctx_mod: Optional[Sequence[int]] = None,
                 gauss_ctx_head: Optional[Sequence[int]] = None,
                 gauss_ctx_pv: Optional[Sequence[int]] = None):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.context_pool = context_pool
        self.head_pool = head_pool
        self.use_lm_features = use_lm_features
        self.use_proto_cos = use_proto_cos
        # MID-layer span pooling. Empty/None = auto mid-5; a concrete tuple
        # (e.g. (-1,)) pins exact layers. Resolved lazily in _features on the
        # first forward, so __init__ only needs the raw knob.
        self.span_layers = tuple(int(i) for i in (span_layers or ()))
        self._span_hidden = None

        # Whole-sentence context pooling. Empty/None = LAST layer (deepest +
        # global for mmBERT/ModernBERT; the old behaviour). A concrete tuple
        # (hidden_states indices, `-1` = last) mean-pools those layers instead
        # (e.g. (10, 16, 22) = upper global layers of mmBERT-base). Resolved
        # lazily in _features so __init__ only needs the raw knob.
        self.context_layers = tuple(int(i) for i in (context_layers or ()))
        self._context_hidden = None

        # Per-role dedicated context: each gauss head reads the whole-sentence
        # context at its own hidden-state layer(s) instead of the last one.
        # Word spans stay at mid-5; feature width unchanged.
        self.gauss_dedicated = gauss_dedicated
        self.gauss_ctx_mod  = tuple(int(i) for i in (gauss_ctx_mod  or ()))
        self.gauss_ctx_head = tuple(int(i) for i in (gauss_ctx_head or ()))
        self.gauss_ctx_pv   = tuple(int(i) for i in (gauss_ctx_pv   or ()))
        self._ctx_mod_hidden  = None   # resolved indices, cached per epoch
        self._ctx_head_hidden = None
        self._ctx_pv_hidden   = None

        self.lm = AutoModelForMaskedLM.from_pretrained(backbone, tie_word_embeddings=False)
        # Cached reference to the base transformer for the no-logits forward
        # path. Deliberately NOT registered as a child module: registering the
        # same instance under a second name would duplicate every state_dict
        # key/parameter (double optimizer updates, 2x ckpt size, strict-load
        # failures on pre-existing snapshots, and LoRA applied twice).
        object.__setattr__(self, 'base_model', _backbone(self))

        context_dim = hidden_size if context_pool in ('mean', 'cls') else 2 * hidden_size
        # 2 spans + 2 lens + context (+ optional LM stats, + optional proto cos)
        self.head_in = hidden_size * 2 + context_dim + 2
        if use_lm_features:
            self.head_in += 4                                  # avg_logp + entropy x mod/head
        # Each output branch gets ONLY its own word's cos(use, prototype)
        # tacked on (torch.cat, last axis), so mod_out never sees head's
        # literalness and vice-versa. That extra column IS part of head_in.
        self.head_in += 1 if use_proto_cos else 0

        self.mod_pool = SpanPool(hidden_size, head_pool)
        self.head_role_pool = SpanPool(hidden_size, head_pool)
        self.mod_gauss = GaussHead(self.head_in, head_hidden, dropout)
        self.head_gauss = GaussHead(self.head_in, head_hidden, dropout)

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
        logits = self.lm(input_ids=ids, attention_mask=attention_mask).logits  # (B, L, V)

        def _stats(mask: torch.Tensor) -> torch.Tensor:
            B = input_ids.size(0)
            rows = torch.nonzero(mask)                    # (N, 2): (batch, seq)
            if rows.numel() == 0:
                return torch.zeros(B, 2, device=input_ids.device)
            b, l = rows[:, 0], rows[:, 1]
            # log_softmax ONLY at the ~few span positions: a full (B, L, V)
            # log-prob + exp table is tens of GB with a large-Vocab backbone.
            pos_logp = F.log_softmax(logits[b, l].float(), dim=-1)            # (N, V)
            tok_logp = torch.gather(
                pos_logp, -1, input_ids[b, l].unsqueeze(-1)).squeeze(-1)      # (N,)
            ent = -(pos_logp.exp() * pos_logp).sum(-1)                        # (N,)
            avg = torch.zeros(B, device=input_ids.device).index_add_(0, b, tok_logp)
            ent_sum = torch.zeros(B, device=input_ids.device).index_add_(0, b, ent)
            n = mask.long().sum(-1).clamp(min=1)
            return torch.stack([avg / n, ent_sum / n], dim=-1)                # (B, 2)

        return torch.cat([_stats(mod_mask), _stats(head_mask)], dim=-1)

    def reset_span_cache(self) -> None:
        """Drop the lazily-cached layer indices.

        Called once per training epoch so LoRA-updated backbone weights are
        re-pooled on the first forward of the epoch.
        """
        self._span_hidden = None
        self._context_hidden = None
        self._ctx_mod_hidden = None
        self._ctx_head_hidden = None
        self._ctx_pv_hidden = None

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
        proto_safe = torch.where(has_span, proto, torch.ones_like(proto))
        cos = F.cosine_similarity(use_emb.float(), proto_safe.float(), dim=-1)
        cos = torch.where(has_span.squeeze(-1), cos, torch.zeros_like(cos))
        return cos.type_as(use_emb).unsqueeze(-1)

    def _context_emb(self, hidden: torch.Tensor, batch) -> torch.Tensor:
        """Pool a [B, S, H] hidden tensor into the whole-sentence context
        embedding (mean and/or [CLS]) and mask at the pool mode."""
        mean_emb = _masked_mean(hidden, batch['attention_mask'])
        if self.context_pool == 'cls':
            return hidden[:, 0]
        if self.context_pool == 'mean':
            return mean_emb
        return torch.cat([mean_emb, hidden[:, 0]], dim=1)

    def _role_context(self, layers, hid_all, batch) -> torch.Tensor:
        """Whole-sentence context pooled at explicit hidden-state indices.
        Raw pre-norm hidden states are larger in magnitude than the post-norm
        last layer the heads expect, so the mean is re-normalised by the
        backbone final LayerNorm (same fix as the context_layers path)."""
        hidden = torch.mean(torch.stack([hid_all[i] for i in layers]), dim=0)
        norm = getattr(self.base_model, 'final_norm', None) \
            or getattr(getattr(self.base_model, 'encoder', None), 'final_norm', None)
        if norm is not None:
            hidden = norm(hidden)
        return self._context_emb(hidden, batch)

    def _compose_gauss_feat(self, mod_emb, head_emb, mod_len, head_len,
                            context_emb, batch, lm_stats=None) -> torch.Tensor:
        """Gauss-head feature bundle: mid-5 spans + 2 lens + (role-specific)
        context + optional LM stats. Same width as the default shared soup."""
        feats = [mod_emb, head_emb, mod_len, head_len, context_emb]
        if lm_stats is not None:
            feats.append(lm_stats)
        return torch.cat(feats, dim=1)

    def _features(self, batch, need_lm: bool, with_mlm_logits: bool = False):
        """Encoder forward + shared span/context feature building.

        Returns ``(mod_emb, head_emb, features, logits, mod_feat, head_feat)``;
        ``logits`` are the MLM logits only when the caller requests them,
        ``mod_feat``/``head_feat`` are the per-role bundles when
        ``gauss_dedicated`` is on (``None`` otherwise).
        """
        if need_lm:
            outputs = self.lm(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            logits = outputs.logits if with_mlm_logits else None
        else:
            outputs = self.base_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True,
            )
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            logits = None

        # Span embeddings come from a MID-pool of layers. Early/mid layers keep
        # the span's literal (word-identity) signal; late layers over-contextualize
        # and blur it (BERT mixing layer). `span_layers=None` auto-selects 5 layers
        # around half-depth; `(-1,)` = last layer (the old behaviour / A-B base).
        # Context + LM signals always stay on the LAST layer. Cache only the
        # resolved layer INDICES (not the tensor itself — tensors are shaped
        # [B, seq_len, H] and seq_len varies between batches, so caching the
        # tensor causes a shape mismatch on the second batch).
        if self._span_hidden is None:
            hid_all = outputs.hidden_states
            layers = self.span_layers
            n_layers = len(hid_all)
            if not layers:
                if n_layers > 8:
                    mid = n_layers // 2
                    layers = tuple(range(mid - 2, mid + 3))
                else:
                    layers = (-1,)
            # Resolve negative indices once and store as a plain tuple of ints
            self._span_hidden = tuple(int(i) % n_layers for i in layers)
        # Always recompute the mean hidden tensor from the CURRENT batch
        hid_all = outputs.hidden_states
        n_layers = len(hid_all)
        span_hidden = torch.mean(
            torch.stack([hid_all[i] for i in self._span_hidden]), dim=0)

        mod_emb = self.mod_pool(span_hidden, batch['mod_span_mask'])
        head_emb = self.head_role_pool(span_hidden, batch['head_span_mask'])
        # Context embeddings: `context_layers=None` keeps the LAST layer
        # exactly as before (`hidden` may carry the final LayerNorm on the
        # no-LM path). A concrete tuple mean-pools the resolved hidden-state
        # indices instead (e.g. multiple GLOBAL attention layers of mmBERT).
        if self.context_layers:
            if self._context_hidden is None:
                hid_all = outputs.hidden_states
                n_layers = len(hid_all)
                self._context_hidden = tuple(
                    int(i) % n_layers for i in self.context_layers)
            hid_all = outputs.hidden_states
            context_hidden = torch.mean(
                torch.stack([hid_all[i] for i in self._context_hidden]), dim=0)
            # Raw pre-final-norm hidden states are several times LARGER per
            # token than the post-norm last layer the Gauss head expects; the
            # mean then saturates the head's (un-normalised) Tanh bottleneck,
            # killing gradients. Re-run the backbone final LayerNorm so the
            # pooled context stays at unit scale, like the default path.
            norm = getattr(self.base_model, 'final_norm', None) \
                or getattr(getattr(self.base_model, 'encoder', None), 'final_norm', None)
            if norm is not None:
                context_hidden = norm(context_hidden)
        else:
            context_hidden = hidden
        context_emb = self._context_emb(context_hidden, batch)

        seq_len = batch['attention_mask'].sum(dim=1).clamp(min=1.0).float()
        mod_len = (batch['mod_span_mask'].float().sum(dim=1) / seq_len).unsqueeze(-1)
        head_len = (batch['head_span_mask'].float().sum(dim=1) / seq_len).unsqueeze(-1)

        feats = [mod_emb, head_emb, mod_len, head_len, context_emb]
        lm_stats = None
        if self.use_lm_features:
            lm_stats = self._lm_span_stats(
                batch['input_ids'], batch['attention_mask'],
                batch['mod_span_mask'], batch['head_span_mask'])
            feats.append(lm_stats)
        features = torch.cat(feats, dim=1)
        mod_feat = head_feat = None
        if self.gauss_dedicated:
            hid_all = outputs.hidden_states
            n_layers = len(hid_all)
            if self._ctx_mod_hidden is None:
                self._ctx_mod_hidden = tuple(
                    int(i) % n_layers for i in (self.gauss_ctx_mod or (-1,)))
            if self._ctx_head_hidden is None:
                self._ctx_head_hidden = tuple(
                    int(i) % n_layers for i in (self.gauss_ctx_head or (-1,)))
            ctx_mod = self._role_context(self._ctx_mod_hidden, hid_all, batch)
            ctx_head = self._role_context(self._ctx_head_hidden, hid_all, batch)
            pv = batch.get('is_pv')
            has_pv = bool(self.gauss_ctx_pv) and pv is not None
            if has_pv:
                if self._ctx_pv_hidden is None:
                    self._ctx_pv_hidden = tuple(
                        int(i) % n_layers for i in self.gauss_ctx_pv)
                ctx_pv = self._role_context(self._ctx_pv_hidden, hid_all, batch)
                use_pv = pv.bool().unsqueeze(-1)
                ctx_mod = torch.where(use_pv, ctx_pv, ctx_mod)
                ctx_head = torch.where(use_pv, ctx_pv, ctx_head)
            mod_feat = self._compose_gauss_feat(
                mod_emb, head_emb, mod_len, head_len, ctx_mod, batch,
                lm_stats=lm_stats)
            head_feat = self._compose_gauss_feat(
                mod_emb, head_emb, mod_len, head_len, ctx_head, batch,
                lm_stats=lm_stats)
        return mod_emb, head_emb, features, logits, mod_feat, head_feat

    def _forward_gauss(self, batch, with_logits: bool = False, with_reps: bool = False,
                       with_lm: bool = False):
        """Predict N(mu, sigma^2) per span; sigma travels the "logits" channel."""
        need_lm = self.use_lm_features or with_lm
        mod_emb, head_emb, features, _, mod_feat, head_feat = self._features(
            batch, need_lm, with_mlm_logits=False)
        if mod_feat is None:
            mod_feat = head_feat = features
        if self.use_proto_cos:
            cos_mod = self._prototype_cos(
                batch['input_ids'], batch['mod_span_mask'], mod_emb)
            cos_head = self._prototype_cos(
                batch['input_ids'], batch['head_span_mask'], head_emb)
            mod_mu, mod_sigma = self.mod_gauss(torch.cat([mod_feat, cos_mod], dim=1))
            head_mu, head_sigma = self.head_gauss(torch.cat([head_feat, cos_head], dim=1))
        else:
            mod_mu, mod_sigma = self.mod_gauss(mod_feat)
            head_mu, head_sigma = self.head_gauss(head_feat)
        mod_pred = mod_mu.clamp(SCORE_MIN, SCORE_MAX)
        head_pred = head_mu.clamp(SCORE_MIN, SCORE_MAX)
        if with_reps and with_logits:
            return mod_pred, head_pred, mod_emb, head_emb, mod_sigma, head_sigma
        if with_reps:
            return mod_pred, head_pred, mod_emb, head_emb
        if with_logits:
            return mod_pred, head_pred, mod_sigma, head_sigma
        return mod_pred, head_pred

    def forward(self, batch, with_logits: bool = False, with_reps: bool = False,
                with_lm: bool = False):
        return self._forward_gauss(batch, with_logits, with_reps, with_lm)


def build_model(cfg, device, load_from: Optional[str | Path] = None) -> MMBertModel:
    """Construct the gauss scoring model, optionally loading a backbone state dict.

    ``load_from`` is a torch state_dict with the LM backbone keys only
    (``model.lm.state_dict()``); scorer heads always get fresh init.
    """
    model = MMBertModel(
        cfg.backbone, hidden_size=cfg.hidden_size, dropout=cfg.dropout,
        context_pool=cfg.context_pool, head_pool=cfg.head_pool,
        head_hidden=cfg.head_hidden, use_lm_features=cfg.use_lm_features,
        use_proto_cos=cfg.use_proto_cos, span_layers=cfg.span_layers,
        context_layers=cfg.context_layers,
        gauss_dedicated=cfg.gauss_dedicated, gauss_ctx_mod=cfg.gauss_ctx_mod,
        gauss_ctx_head=cfg.gauss_ctx_head, gauss_ctx_pv=cfg.gauss_ctx_pv,
    )
    if load_from is not None:
        load_from = Path(load_from)
        if not load_from.is_file():
            raise FileNotFoundError(f'state dict not found: {load_from}')
        state = torch.load(load_from, map_location='cpu', weights_only=True)
        model.lm.load_state_dict(state)
    return model.to(device)