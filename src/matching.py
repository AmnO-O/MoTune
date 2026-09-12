"""Locate a compound inside its real context sentence and wrap the spans in
``<mod>`` / ``<head>`` / ``<mwe>`` markers.

Stays dependency-free (regex only) so matching can be unit-tested and
debugged without a model or tokenizer.
"""

from __future__ import annotations

import re
from typing import Optional

_MWE_OPEN, _MWE_CLOSE = '<mwe>', '</mwe>'
_MOD_OPEN, _MOD_CLOSE = '<mod>', '</mod>'
_HEAD_OPEN, _HEAD_CLOSE = '<head>', '</head>'

# Inflection allowed on the head word: plural/possessive/3rd person.
_HEAD_INFL = r"(?:es|s|'s|\u2019s)?"


def _escape(word: str) -> str:
    return re.escape(word)


def mark_compound(context: str, mod: str, head: str,
                  mwe: Optional[str] = None) -> Optional[str]:
    """Return ``context`` with marker tags around the matched spans.

    The compound is matched as a word sequence: ``mod`` followed by ``head``
    (head may carry a plural/possessive suffix, e.g. "night watch" matches
    "night watches"). ``mwe`` defaults to ``mod + head`` when not given.

    Returns None when no match is found (caller decides the fallback).
    """
    if not mod or not head:
        return None

    mwe = (mwe or f'{mod} {head}').strip()
    mod_re = _escape(mod)
    head_re = _escape(head) + _HEAD_INFL

    pattern = re.compile(
        rf'\b({mod_re})\b\s+({head_re})\b',
        re.IGNORECASE,
    )
    match = pattern.search(context)
    if not match:
        return None

    mod_txt, head_txt = match.group(1), match.group(2)
    mwe_start, mwe_end = match.start(), match.end()

    marked_compound = (
        f'{_MWE_OPEN}{_MOD_OPEN}{mod_txt}{_MOD_CLOSE}'
        f'{_HEAD_OPEN}{head_txt}{_HEAD_CLOSE}{_MWE_CLOSE}'
    )
    return context[:mwe_start] + marked_compound + context[mwe_end:]


def fallback_marked_text(mod: str, head: str, context: str,
                         mwe: Optional[str] = None) -> str:
    """Deterministic marked input when the compound is not found in context."""
    mwe = (mwe or f'{mod} {head}').strip()
    marked = (
        f'{_MWE_OPEN}{_MOD_OPEN}{mod}{_MOD_CLOSE}'
        f'{_HEAD_OPEN}{head}{_HEAD_CLOSE}{_MWE_CLOSE}'
    )
    context = ' '.join(context.strip().split())
    return f'{marked} {context}'.strip()


def span_text_offsets(marked: str, tag: str) -> Optional[tuple[int, int]]:
    """Character offsets of the text between an opening and closing marker.

    ``tag`` is one of 'mod', 'head', 'mwe'. Returns None if either boundary
    of that tag pair is missing (i.e. the marked text could not carry it).
    """
    try:
        open_tag, close_tag = TAG_LOOKUP[tag]
    except KeyError as exc:  # pragma: no cover - programmer error
        raise ValueError(f'unknown tag {tag!r}') from exc

    start = marked.find(open_tag)
    if start == -1:
        return None
    start += len(open_tag)

    end = marked.find(close_tag, start)
    if end == -1:
        return None
    return start, end


TAG_LOOKUP = {
    'mod': ('<mod>', '</mod>'),
    'head': ('<head>', '</head>'),
    'mwe': ('<mwe>', '</mwe>'),
}