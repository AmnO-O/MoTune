"""Active-target definitions shared by the dataset and the single-target model.

The single-target design makes every sample answer exactly one question: is
this row graded on the modifier, the head noun, or the whole compound?
Everything that follows from the choice -- which span(s) to pool, which
label to supervise, which head to route -- derives from the target name.

``pool_span`` is pure geometry over the final-layer hidden states: masked
mean over the modifier span, the head span, or the whole compound
(mod | head), depending on the active target.
"""

from __future__ import annotations

from typing import Optional, Sequence

try:
    import torch
except ImportError:
    torch = None

TARGETS = ('mod', 'head', 'pv')

# One of mmBERT's reserved-but-unused vocab ids, reused as a learned task
# marker (verified: id 7/8/9 = '<unused0>'/'<unused1>'/'<unused2>'). No embed
# resize needed; the marker embedding is a tiny separate module that trains at
# head_lr. (jhu-clsp/mmBERT-base, Gemma-2-style 256k vocab.)
MARKER_CODE = {'mod': 7, 'head': 8, 'pv': 9}
TARGET_PREFIX_TOKENS = {'mod': '<unused0>', 'head': '<unused1>', 'pv': '<unused2>'}


def target_code(t):
    """Stable integer code for a target (0=mod, 1=head, 2=pv).

    Accepts the name (``'mod'``) or the code itself (``0``) so per-row
    ``batch['target']`` tensors and config strings both route identically.
    """
    if isinstance(t, int):
        if not 0 <= t < len(TARGETS):
            raise ValueError(f'unknown target code {t!r}, expected 0..{len(TARGETS) - 1}')
        return t
    if t not in TARGETS:
        raise ValueError(f'unknown target {t!r}, expected one of {TARGETS}')
    return TARGETS.index(t)


def row_targets(batch) -> Optional[Sequence[str]]:
    """Per-row active targets as strings; None means 'predict all jointly'.

    A batch may mix targets (the 3N augmentation relies on it). When no
    ``target`` field is present the caller falls back to joint inference,
    scoring every row on every head.
    """
    tgt = batch.get('target')
    if tgt is None:
        return None
    if isinstance(tgt, torch.Tensor):
        tgt = tgt.cpu().tolist()
    return list(tgt)


def target_selector(targets, t: str) -> Optional[torch.Tensor]:
    """Bool row mask selecting rows whose active target is ``t``.

    Returns None when ``targets`` is None (no restriction), so callers can
    pass the result straight to ``torch.where`` / indexing without branching.
    """
    if targets is None:
        return None
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().tolist()
    codes = [target_code(x) for x in targets]
    return torch.as_tensor(codes, dtype=torch.long).eq(target_code(t))


def pool_span(hidden: torch.Tensor, batch, t: str) -> torch.Tensor:
    """Final-layer masked-mean over the span(s) belonging to target ``t``.

    ''mod'' -> modifier span, ''head'' -> head-noun span, ''pv'' -> the whole
    compound (mod | head).
    """
    mod_mask = batch['mod_span_mask']
    head_mask = batch['head_span_mask']
    if t == 'mod':
        mask = mod_mask
    elif t == 'head':
        mask = head_mask
    else:
        mask = mod_mask | head_mask

    mask = mask.float().unsqueeze(-1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / counts


def pool_active(hidden: torch.Tensor, batch, targets: Sequence[str]) -> torch.Tensor:
    """Masked-mean over each row's OWN active-target span, in one pass.

    Rows are heterogeneously routed: a row with target 'mod' pools the
    modifier span, 'head' pools the head-noun span, 'pv' pools the whole
    compound (mod | head). Returns one ``(B, H)`` vector per row, so a shared
    head runs ONCE for the whole batch instead of once per target with the
    inactive outputs zeroed by ``torch.where`` afterwards.
    """
    mod_mask = batch['mod_span_mask']
    head_mask = batch['head_span_mask']
    pv_mask = mod_mask | head_mask
    codes = [target_code(x) for x in targets]
    choices = torch.stack([mod_mask, head_mask, pv_mask], dim=0)   # (3, B, T)
    rows = torch.arange(hidden.size(0), device=choices.device)
    mask = choices[
        torch.as_tensor(codes, device=choices.device, dtype=torch.long), rows
    ]    # (B, T): row r gets its own target's span mask

    mask = mask.float().unsqueeze(-1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / counts