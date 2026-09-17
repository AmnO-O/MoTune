"""Locate the modifier / head spans inside the RAW context via offsets (no markers).

The context is tokenized untouched; given the tokenizer's ``offset_mapping``
we map the surface forms of Mod and Head back onto token spans by string
matching inside the original text. Works without a model and is unit-tested
on synthetic offsets in ``tests/smoke_src.py``.

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
    row instead of feeding duplicate mod==head span embeddings.

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
    "ses", "sen", "se",        # German -nis -> -nisses, -nissen, -nisse
    "es",                      # plural/3rd  -> watches
    "ed", "ing",               # English participles
    "en", "ern", "er",         # German      -> kindern, kindern, kinder
    "s", "e", "n",             # German      -> hands, knechte, garten
)
_INFL_RE = "(?:" + "|".join(re.escape(s) for s in _INFLECTION) + ")?"
_FUGEN = r"(?:s|es|en|er|n|e|ens|-)?"
# Any Unicode letter or combining mark, minus digits/underscore (covers umlauts,
# ß, accented/é, Vietnamese letters), so word-boundary lookarounds are portable.
_LETTER = r"[^\W\d_]"
_START_BOUNDARY = f"(?<!{_LETTER})"
_END_BOUNDARY = f"(?!{_LETTER})"

# Suppletive English verb surfaces for PV rows (walk/drop/step etc. are covered
# by the regular e-drop / consonant-doubling alternates below, so only the truly
# irregular bases live here). Base first so exact matches win.
_IRREGULAR = {
    "be": ("is", "are", "was", "were", "been", "being", "am"),
    "blow": ("blew", "blown", "blowing"),
    "break": ("broke", "broken", "breaking"),
    "bring": ("brought", "bringing"),
    "build": ("built", "building"),
    "burn": ("burnt", "burned", "burning"),
    "catch": ("caught", "catching"),
    "come": ("came", "coming"),
    "cut": ("cut", "cutting"),
    "dig": ("dug", "digging"),
    "draw": ("drew", "drawn", "drawing"),
    "drive": ("drove", "driven", "driving"),
    "fall": ("fell", "fallen", "falling"),
    "fight": ("fought", "fighting"),
    "find": ("found", "finding"),
    "fly": ("flew", "flown", "flying"),
    "get": ("got", "gotten", "getting"),
    "give": ("gave", "given", "giving"),
    "go": ("went", "gone", "going"),
    "hang": ("hung", "hanging"),
    "hold": ("held", "holding"),
    "keep": ("kept", "keeping"),
    "lay": ("laid", "laid", "laying"),
    "lead": ("led", "leading"),
    "leave": ("left", "leaving"),
    "let": ("let", "letting"),
    "lie": ("lay", "lain", "lying"),
    "make": ("made", "making"),
    "meet": ("met", "meeting"),
    "pay": ("paid", "paying"),
    "run": ("ran", "running"),
    "seek": ("sought", "seeking"),
    "sell": ("sold", "selling"),
    "send": ("sent", "sending"),
    "shake": ("shook", "shaken", "shaking"),
    "shoot": ("shot", "shooting"),
    "sit": ("sat", "sitting"),
    "speak": ("spoke", "spoken", "speaking"),
    "strike": ("struck", "striking"),
    "take": ("took", "taken", "taking"),
    "tear": ("tore", "torn", "tearing"),
    "throw": ("threw", "thrown", "throwing"),
    "wake": ("woke", "woken", "waking"),
    "wear": ("wore", "worn", "wearing"),
    "wind": ("wound", "winding"),
    "write": ("wrote", "written", "writing"),
}


def _alternate_forms(base: str) -> Tuple[str, ...]:
    """Surface candidates for ``base`` (base first, then inflections).

    Covers the regular English verb endings -- ``-s/-es``, ``-ed/-d``,
    ``-ing`` with the ``e``-drop ("move" -> "moving"), consonant doubling
    ("step" -> "stepping") and ``y -> -ies/-ied`` rules -- plus the suppletive
    ``_IRREGULAR`` set. Used only by the independent (head-anywhere) fallback,
    so NN compounds that already matched via compound/fused/spaced are
    unaffected.
    """
    t = normalize(base)
    if not t:
        return ()
    out: List[str] = [t]
    for ir in _IRREGULAR.get(t, ()):
        out.append(ir)

    last = t[-1]
    def doubles() -> bool:
        # w and x never double in English (draw -> drew, fix -> fixed).
        return (len(t) >= 3 and last not in "aeiouwyx"
                and t[-2] in "aeiou" and t[-3] not in "aeiou")

    if last in "sxz" or t.endswith(("ch", "sh")):
        out.append(t + "es")                   # watch -> watches
    else:
        out.append(t + "s")                    # line -> lines
    if last == "e" and t.endswith("ie"):
        out.append(t[:-2] + "ying")            # tie -> tying (not "tiing")
    elif last == "e":
        out.append(t[:-1] + "ing")             # move -> moving
    else:
        out.append(t + "ing")
    if last == "e":
        out.append(t + "d")                    # tie -> tied, line -> lined
    elif doubles():
        out.append(t + last + "ed")            # step -> stepped
        out.append(t + last + "ing")           # step -> stepping
        if t.endswith(("p", "t")):
            out.append(t + last + "s")         # drops -> not used; keep simple
    else:
        out.append(t + "ed")
        if last == "y" and len(t) >= 2 and t[-2] not in "aeiou":
            stem = t[:-1]                      # try -> tries / tied
            out.append(stem + "ies")
            out.append(stem + "ied")
    dedup: List[str] = []
    for s in out:
        if s and s not in dedup:
            dedup.append(s)
    return tuple(dedup)


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


def _char_to_token(offsets: List[Tuple[int, int]], c_pos: int) -> Optional[int]:
    """Token whose character range contains char ``c_pos`` (specials (0,0) skipped).

    Falls back to the first token that starts after ``c_pos`` (whitespace gap),
    then to the last real token when ``c_pos`` is past the end. Ranges from a
    wordpiece ``offset_mapping`` are contiguous, so containment is exact.
    """
    best: Optional[int] = None
    for i, (s, e) in enumerate(offsets):
        if (s, e) == (0, 0):
            continue
        if s <= c_pos < e:
            return i
        if best is None and s > c_pos:
            best = i
    if best is None:
        for i, (s, e) in enumerate(offsets):
            if (s, e) != (0, 0):
                best = i
    return best


def _match_from_compound(text_n: str, compound: Optional[str], mod: str, head: str
                         ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Direct match using the dataset's ground-truth compound surface form."""
    if not compound:
        return None
    t_comp, t_head = normalize(compound), normalize(head)
    # Search for compound (+ optional head inflection) in text
    pat = f"({re.escape(t_comp)}{_INFL_RE}){_END_BOUNDARY}"
    m = re.search(_START_BOUNDARY + pat, text_n)
    if not m:
        return None
    c_start, c_end = m.start(1), m.end(1)
    matched_comp = text_n[c_start:c_end]
    # Find head inside matched_comp (usually at the end, possibly inflected)
    m_head = re.search(f"({re.escape(t_head)}{_INFL_RE})$", matched_comp)
    if m_head:
        h_start = c_start + m_head.start(1)
        h_end = c_end
        m_start = c_start
        m_end = h_start
        if m_end > m_start:
            return ((m_start, m_end), (h_start, h_end))
    return None


def _match_fused(text_n: str, mod: str, head: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Closed compound: modifier directly fused into head, with optional Fugen or hyphen."""
    t_mod, t_head = normalize(mod), normalize(head)
    if not t_mod or not t_head:
        return None
    pattern = f"({re.escape(t_mod)})({_FUGEN})({re.escape(t_head)}{_INFL_RE}){_END_BOUNDARY}"
    m = re.search(_START_BOUNDARY + pattern, text_n)
    if not m:
        return None
    return ((m.start(1), m.end(1)), (m.start(3), m.end(3)))


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
    """Modifier (any surface alternate) and head anywhere in the sentence.

    Tries each surface form of ``mod`` (base, regular inflections, irregular
    verbs) and for each *occurrence* searches for the head strictly after it,
    so split phrasal verbs ("struck ... down", "stepped down") align even when
    the base form never appears verbatim.
    """
    t_head = re.escape(normalize(head))
    head_pat = re.compile(_START_BOUNDARY + t_head + _INFL_RE + _END_BOUNDARY)
    for alt in _alternate_forms(mod):
        pat = re.compile(_START_BOUNDARY + re.escape(alt) + _END_BOUNDARY)
        pos = 0
        while True:
            # .search(text_n, pos) (NOT text_n[pos:]) keeps the lookbehind able
            # to inspect characters before ``pos``, so a candidate alternate is
            # not falsely matched mid-word directly after a previous occurrence.
            m_mod = pat.search(text_n, pos)
            if not m_mod:
                break
            e0 = m_mod.end()
            m_head = head_pat.search(text_n, e0)
            if m_head:
                return ((m_mod.start(), m_mod.end()),
                        (m_head.start(), m_head.end()))
            pos = e0
    return None


def find_spans(text: str, offsets: Iterable[Tuple[int, int]],
               mod: str, head: str, compound: Optional[str] = None) -> SpanResult:
    """Character-match Mod/Head inside ``text`` and map them to token spans.

    ``offsets`` is the tokenizer offset_mapping for ``text`` (special tokens
    with (0,0) offsets are skipped). See module docstring for the strategy
    order (compound surface -> fused -> spaced -> independent).
    """
    offsets = list(offsets)
    text_n = normalize(text)

    pair = (
        _match_from_compound(text_n, compound, mod, head)
        or _match_fused(text_n, mod, head)
        or _match_spaced(text_n, mod, head)
        or _match_independent(text_n, mod, head)
    )
    if pair is None:
        return SpanResult(Span(None, None), Span(None, None), found=False)

    m_mod, m_head = pair

    def tok_span(cs: Tuple[int, int]) -> Optional[Span]:
        s, e = cs
        ts = _char_to_token(offsets, s)       # token containing char s (start)
        te = _char_to_token(offsets, e - 1)   # token containing char e-1 (end)
        if s >= e or ts is None or te is None or te < ts:
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