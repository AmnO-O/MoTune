"""Locate the modifier / head spans inside the RAW context via offsets (no markers).

The context is tokenized untouched; given the tokenizer's ``offset_mapping``
we map the surface forms of Mod and Head back onto token spans by string
matching inside the original text. Works without a model and is unit-tested
on synthetic offsets in ``tests/smoke_mm.py``.

Matching strategies, tried in order:
  1. FUSED closed compounds written without a space ("Abiturzeugnis"): the
     modifier continues straight into the head with no word boundary.
  2. SPACED compounds ("night watch"): modifier, whitespace, head (+ inflection).
  3. INDEPENDENT: modifier and head anywhere in the sentence, head after mod.

Handled cases:
  - English plural / 3rd-person / possessive head inflection
    ("watch" -> "watches" / "watch's").
  - German closed compounds where mod+head collapse onto the SAME token
    (``degenerate=True``): the caller skips span-supervised losses for that
    row instead of letting cos(mod, head) collapse to 1.

Matching is deliberately conservative: an unaligned row returns
``Span(None, None)`` and the dataset skips the affected supervision rather
than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Head-inflection suffixes allowed AFTER a base form, longest-first so
# "watch's" matches before plain "s" could shorten it. Lowercased matching.
_INFLECTION = (
    "\u2019s", "'s",           # possessive  -> watch's
    "es",                      # plural/3rd  -> watches
    "en", "ern", "er",         # German      -> kindern, kindern, kinder
    "s", "e", "n",             # German      -> hands, knechte, garten
)
_INFL_RE = "(?:" + "|".join(re.escape(s) for s in _INFLECTION) + ")?"
_WORD = "A-Za-z\u00C0-\u024F"
_START_BOUNDARY = f"(?<![{_WORD}])"
_END_BOUNDARY = f"(?![{_WORD}])"


def normalize(s: str) -> str:
    # `.lower()` (NOT casefold): casefold expands German 'ß' -> 'ss' (1->2
    # chars), shifting character indices so they no longer line up with the
    # tokenizer's offset_mapping (computed on the ORIGINAL string). lower()
    # keeps 'ß' as 'ß', so every codepoint keeps its length for our DE/EN data.
    return s.lower()


@dataclass(frozen=True)
class Span:
    """Token span (half-open) of one role, or None when not aligned."""

    start: Optional[int]
    end: Optional[int]          # exclusive
    degenerate: bool = False    # mod and head collapsed onto the same tokens


@dataclass(frozen=True)
class SpanResult:
    mod: Span
    head: Span
    adjacent: bool = False      # head starts where mod ends (or same token)
    found: bool = False         # both roles aligned

    @property
    def degenerate(self) -> bool:
        return self.mod.degenerate or self.head.degenerate


def _char_to_token(offsets: List[Tuple[int, int]], c_end: int) -> Optional[int]:
    """Token containing char ``c_end-1`` (tokens are contiguous; specials (0,0) skipped)."""
    best: Optional[int] = None
    for i, (s, e) in enumerate(offsets):
        if (s, e) == (0, 0):
            continue
        if s < c_end:
            best = i
        else:
            break
    return best


def _match_fused(text_n: str, mod: str, head: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Closed compound: modifier directly fused into head, no word boundary."""
    t_mod, t_head = normalize(mod), normalize(head)
    if not t_mod or not t_head:
        return None
    pattern = re.escape(t_mod + t_head) + _INFL_RE + _END_BOUNDARY
    m = re.search(_START_BOUNDARY + pattern, text_n)
    if not m:
        return None
    s0 = m.start()
    mid = s0 + len(t_mod)
    return ((s0, mid), (mid, m.end()))


def _match_spaced(text_n: str, mod: str, head: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Spaced compound: ``mod WHITESPACE head`` (+ head inflection)."""
    t_mod, t_head = normalize(mod), normalize(head)
    if not t_mod or not t_head:
        return None
    pattern = re.escape(t_mod) + r"\s+" + re.escape(t_head) + _INFL_RE + _END_BOUNDARY
    m = re.search(_START_BOUNDARY + pattern, text_n)
    if not m:
        return None
    s0 = m.start()
    mid = s0 + len(t_mod)
    ws = re.match(r"\s+", text_n[mid:])
    head_start = mid + (ws.end() if ws else 0)
    return ((s0, mid), (head_start, m.end()))


def _match_independent(text_n: str, mod: str, head: str,
                       ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Modifier and head anywhere in the sentence, head after modifier."""
    m_mod = re.search(
        _START_BOUNDARY + re.escape(normalize(mod)) + _END_BOUNDARY, text_n
    )
    if not m_mod:
        return None
    m_head = re.search(
        _START_BOUNDARY + re.escape(normalize(head)) + _INFL_RE + _END_BOUNDARY,
        text_n[m_mod.end():],
    )
    if not m_head:
        return None
    s = m_head.start() + m_mod.end()
    e = m_head.end() + m_mod.end()
    return ((m_mod.start(), m_mod.end()), (s, e))


def find_spans(text: str, offsets: Iterable[Tuple[int, int]],
               mod: str, head: str) -> SpanResult:
    """Character-match Mod/Head inside ``text`` and map them to token spans.

    ``offsets`` is the tokenizer offset_mapping for ``text`` (special tokens
    with (0,0) offsets are skipped). See module docstring for the strategy
    order (fused -> spaced -> independent).
    """
    offsets = list(offsets)
    text_n = normalize(text)

    pair = (
        _match_fused(text_n, mod, head)
        or _match_spaced(text_n, mod, head)
        or _match_independent(text_n, mod, head)
    )
    if pair is None:
        return SpanResult(Span(None, None), Span(None, None), found=False)

    m_mod, m_head = pair

    def tok_span(cs: Tuple[int, int]) -> Optional[Span]:
        s, e = cs
        ts = _char_to_token(offsets, s + 1)   # token containing char s
        te = _char_to_token(offsets, e)       # token containing char e-1
        if ts is None or te is None or te < ts:
            return None
        return Span(ts, te + 1)

    mod_sp = tok_span(m_mod)
    head_sp = tok_span(m_head)
    if mod_sp is None or head_sp is None:
        return SpanResult(Span(None, None), Span(None, None), found=False)

    same = mod_sp.start == head_sp.start
    adjacent = (mod_sp.end == head_sp.start) or (same and mod_sp.end == head_sp.end)
    return SpanResult(
        mod=Span(mod_sp.start, mod_sp.end, degenerate=same),
        head=Span(head_sp.start, head_sp.end, degenerate=same),
        adjacent=adjacent,
        found=True,
    )