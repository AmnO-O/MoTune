"""LoRA, peft-free: wraps ``nn.Linear`` targets in low-rank adapters.

LoRA is for the pretrained backbone only: target ``nn.Linear`` modules under
``model.lm`` are wrapped in ``LoRAAdapter`` (keeps the frozen base path),
trained, then ``merge_lora`` folds ``scaling * B @ A`` into the base weight and
unwraps them. New random-init scorer blocks (attention/fusion/MHA) are never
wrapped — their ``nn.MultiheadAttention`` internals are not attribute-safe to
wrap and were not pretrained.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


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
            # LoRA is for the pretrained backbone (``model.lm``) only: the
            # scorer-side attention/fusion blocks are random-init, stay fully
            # trainable, and expose nn.MultiheadAttention internals
            # (``self.out_proj.weight``) that break when wrapped in an
            # adapter. Skip any subtree whose root is not ``lm``.
            if full and full.split('.', 1)[0] not in ('lm', 'model'):
                continue
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
                # LoRA is for the pretrained backbone only: the scorer-side
                # attention/fusion blocks are random-init and stay fully
                # trainable (their MHA params are also not attribute-safe to
                # wrap). The backbone module root is ``lm`` in the downstream
                # combined model but ``model`` in an AutoModelForMaskedLM
                # (ModernBERT); both are pretrained and wrappable.
                if full and full.split('.', 1)[0] not in ('lm', 'model'):
                    continue
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