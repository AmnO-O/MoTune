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

_GERMAN_PARTICLES = (
    "ab", "an", "auf", "aus", "bei", "ein", "fest", "fort", "her", "hin",
    "los", "mit", "nach", "vor", "weg", "zu", "zurück", "zusammen",
)

# Irregular German verbs (Ablaut / strong verbs / dental stems) for German PV rows
_IRREGULAR_DE = {
    "biegen": ("bog", "böge", "gebogen", "biegt", "biegst"),
    "bieten": ("bot", "böte", "geboten", "bietet", "bietest"),
    "bleiben": ("blieb", "geblieben", "bleibt", "bleibst", "bliebe"),
    "brennen": ("brannt", "brannte", "brannten", "gebrannt", "brennt", "brennst"),
    "bringen": ("bracht", "brachte", "brächte", "brachten", "brächten", "gebracht", "bringt", "bringst"),
    "denken": ("dacht", "dachte", "dächte", "dachten", "dächten", "gedacht", "denkt", "denkst"),
    "fahren": ("fuhr", "führe", "gefahren", "fährt", "fährst"),
    "fangen": ("fing", "finge", "gefangen", "fängt", "fängst"),
    "geben": ("gab", "gäbe", "gegeben", "gibt", "gibst"),
    "gehen": ("ging", "ginge", "gegangen", "geht", "gehst"),
    "graben": ("grub", "grübe", "gegraben", "gräbt", "gräbst"),
    "hauen": ("hieb", "gehauen", "haut", "haust"),
    "heben": ("hob", "höbe", "gehoben", "hebt", "hebst"),
    "hängen": ("hing", "hinge", "gehangen", "hängt", "hängst"),
    "klingen": ("klang", "klänge", "geklungen", "klingt", "klingst"),
    "kommen": ("kam", "käme", "gekommen", "kommt", "kommst"),
    "lassen": ("ließ", "ließe", "gelassen", "lässt", "läßt", "lies"),
    "leihen": ("lieh", "liehe", "geliehen", "leiht", "leihst"),
    "lesen": ("las", "läse", "gelesen", "liest"),
    "nehmen": ("nahm", "nähme", "genommen", "nimmt", "nimmst"),
    "passen": ("gepasst", "gepaßt", "passte", "paßte", "passt", "paßt"),
    "reißen": ("riss", "risse", "gerissen", "reißt", "reissen", "reisst"),
    "rufen": ("rief", "riefe", "gerufen", "ruft", "rufst"),
    "saugen": ("sog", "gesogen", "saugt", "saugst"),
    "schaffen": ("schuf", "schüfe", "geschaffen", "schafft", "schaffst"),
    "schieben": ("schob", "schöbe", "geschoben", "schiebt", "schiebst"),
    "schlagen": ("schlug", "schlüge", "geschlagen", "schlägt", "schlägst"),
    "schließen": ("schloss", "schloß", "schlösse", "geschlossen", "schließt", "schliessen", "schliess"),
    "schneiden": ("schnitt", "schnitte", "geschnitten", "schneidet", "schneidest"),
    "schreiben": ("schrieb", "schriebe", "geschrieben", "schreibt", "schreibst"),
    "schreien": ("schrie", "geschrien", "schreit", "schreist"),
    "sehen": ("sah", "sähe", "gesehen", "sieht", "siehst"),
    "sprechen": ("sprach", "spräche", "gesprochen", "spricht", "sprichst"),
    "stoßen": ("stieß", "stieße", "gestoßen", "stößt"),
    "tragen": ("trug", "trüge", "getragen", "trägt", "trägst"),
    "treffen": ("traf", "träfe", "getroffen", "trifft", "triffst"),
    "treiben": ("trieb", "triebe", "getrieben", "treibt", "treibst"),
    "treten": ("trat", "träte", "getreten", "tritt", "trittst"),
    "wachsen": ("wuchs", "wüchse", "gewachsen", "wächst"),
    "weichen": ("wich", "wiche", "gewichen", "weicht", "weichst"),
    "weisen": ("wies", "wiese", "gewiesen", "weist"),
    "wenden": ("wandt", "wandte", "gewandt", "wendet", "wendest"),
    "ziehen": ("zog", "zöge", "gezogen", "zieht", "ziehst"),
    "zwingen": ("zwang", "zwänge", "gezwungen", "zwingt", "zwingst"),
}

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

    irreg = _IRREGULAR.get(t)
    if irreg:
        # Known irregulars: keep only their actual surfaces plus the regular
        # third-person -s/-es (always valid: takes/writes/breaks/goes). The
        # -ed/-ing/-d/doubling/y rules only produce non-words here (be -> bing,
        # bed), so they are skipped to keep the candidate set lean and avoid a
        # "bed ... out" false positive for mod=be.
        out.extend(irreg)
        if t.endswith(("s", "x", "z", "ch", "sh", "o")):
            out.append(t + "es")             # go -> goes, echo -> echoes
        else:
            out.append(t + "s")
        return tuple(dict.fromkeys(out))

    last = t[-1]
    def doubles() -> bool:
        # w and x never double in English (draw -> drew, fix -> fixed).
        return (len(t) >= 3 and last not in "aeiouwyx"
                and t[-2] in "aeiou" and t[-3] not in "aeiou")

    if last in "sxz" or t.endswith(("ch", "sh", "o")):
        out.append(t + "es")                   # watch -> watches, echo -> echoes
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
    """Spaced compound: ``mod SEP head`` (+ head inflection), where SEP is
    whitespace or a hyphen ("night watch" / "night-watch")."""
    t_mod, t_head = normalize(mod), normalize(head)
    if not t_mod or not t_head:
        return None
    pattern = re.escape(t_mod) + r"[ \t-]+" + re.escape(t_head) + _INFL_RE + _END_BOUNDARY
    m = re.search(_START_BOUNDARY + pattern, text_n)
    if not m:
        return None
    s0 = m.start()
    mid = s0 + len(t_mod)
    sep = re.match(r"[ \t-]+", text_n[mid:])
    head_start = mid + (sep.end() if sep else 0)
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


def _german_base_forms(base: str) -> Tuple[str, ...]:
    """Surface candidates for a German base verb (infinitive/stem -> inflections)."""
    b = normalize(base)
    if not b:
        return ()
    forms: List[str] = [b]

    # Stem extraction
    if b.endswith(("eln", "ern")) and len(b) > 4:
        stem = b[:-1]
    elif b.endswith("en") and len(b) > 3:
        stem = b[:-2]
    elif b.endswith("n") and len(b) > 2:
        stem = b[:-1]
    else:
        stem = b

    # Regular weak endings (present, past weak, dental/nasal stems, participles)
    endings = (
        "e", "st", "t", "en",
        "te", "test", "ten", "tet",
        "ete", "etest", "eten", "etet", "et", "est",
        "end", "ende", "enden", "nd",
    )
    for end in endings:
        forms.append(stem + end)

    # Weak participle forms
    forms.append("ge" + stem + "t")
    forms.append("ge" + stem + "et")
    forms.append("ge" + stem + "te")
    forms.append("ge" + stem + "ten")
    forms.append("ge" + stem + "ter")
    forms.append("ge" + stem + "tes")

    # -eln / -ern verbs (e.g. wickeln -> wickle, wickelst, wickelt)
    if b.endswith("eln") and len(b) > 4:
        s = b[:-3] + "l"
        forms.extend([s + "e", s + "st", s + "t", s + "te", s + "ten"])
        forms.append(b[:-1])
    elif b.endswith("ern") and len(b) > 4:
        s = b[:-3] + "r"
        forms.extend([s + "e", s + "st", s + "t", s + "te", s + "ten"])
        forms.append(b[:-1])

    # Irregular verbs
    irreg = _IRREGULAR_DE.get(b)
    if irreg:
        for v in irreg:
            forms.append(v)
            for end in ("st", "en", "t", "e", "er", "es", "em", "te", "ten"):
                forms.append(v + end)

    # Orthographic tolerance: ß <-> ss
    expanded: List[str] = []
    for f in forms:
        expanded.append(f)
        if "ß" in f:
            expanded.append(f.replace("ß", "ss"))
        if "ss" in f:
            expanded.append(f.replace("ss", "ß"))

    return tuple(sorted(dict.fromkeys(expanded), key=len, reverse=True))


def _match_german_pv(text_n: str, mod: str, head: str
                     ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """German particle verb matching (trennbare Verben).

    Here mod = Base (verb stem/infinitive), head = Particle (ab, an, auf, aus, etc.).
    Covers:
      1. Fused direct: particle + base_form (e.g. abhauen, abgehauen, abgeschlossen, ablehnte)
      2. Fused zu-infinitive: particle + 'zu' + base_form (e.g. abzuhauen, abzuwickeln)
      3. Fused hyphenated: particle + '-' + base_form
      4. Separated main clause (V2): base_form appears first, particle appears later
      5. Separated inverted: particle appears first, base_form appears later
    """
    t_part = normalize(head)
    t_base = normalize(mod)
    forms = _german_base_forms(t_base)
    if not forms or not t_part:
        return None

    # 1. Fused direct: particle directly attached to base form (incl. ge- participles)
    for f in forms:
        f_joined = t_part + f
        pat = re.compile(_START_BOUNDARY + re.escape(f_joined) + _END_BOUNDARY)
        m = pat.search(text_n)
        if m:
            h_span = (m.start(), m.start() + len(t_part))
            m_span = (m.start() + len(t_part), m.end())
            return (m_span, h_span)

    # 2. Fused zu-infinitive: particle + 'zu' + infinitive
    zu_candidates = [b for b in forms if b.endswith(("en", "eln", "ern", "n"))]
    for f in zu_candidates:
        f_zu = t_part + "zu" + f
        pat = re.compile(_START_BOUNDARY + re.escape(f_zu) + _END_BOUNDARY)
        m = pat.search(text_n)
        if m:
            h_span = (m.start(), m.start() + len(t_part))
            m_span = (m.start() + len(t_part) + 2, m.end())
            return (m_span, h_span)

    # 3. Fused hyphenated
    for f in forms:
        f_hyphen = t_part + "-" + f
        pat = re.compile(_START_BOUNDARY + re.escape(f_hyphen) + _END_BOUNDARY)
        m = pat.search(text_n)
        if m:
            h_span = (m.start(), m.start() + len(t_part))
            m_span = (m.start() + len(t_part) + 1, m.end())
            return (m_span, h_span)

    # 4. Separated (V2): base form appears first, particle appears later in the clause
    part_pat = re.compile(_START_BOUNDARY + re.escape(t_part) + _END_BOUNDARY)
    for f in forms:
        base_pat = re.compile(_START_BOUNDARY + re.escape(f) + _END_BOUNDARY)
        for m_base in base_pat.finditer(text_n):
            m_part = part_pat.search(text_n, m_base.end())
            if m_part:
                return ((m_base.start(), m_base.end()), (m_part.start(), m_part.end()))

    # 5. Separated inverted: particle appears before base
    for f in forms:
        base_pat = re.compile(_START_BOUNDARY + re.escape(f) + _END_BOUNDARY)
        for m_part in part_pat.finditer(text_n):
            m_base = base_pat.search(text_n, m_part.end())
            if m_base:
                return ((m_base.start(), m_base.end()), (m_part.start(), m_part.end()))

    return None


def find_spans(text: str, offsets: Iterable[Tuple[int, int]],
               mod: str, head: str, compound: Optional[str] = None) -> SpanResult:
    """Character-match Mod/Head inside ``text`` and map them to token spans.

    ``offsets`` is the tokenizer offset_mapping for ``text`` (special tokens
    with (0,0) offsets are skipped). See module docstring for the strategy
    order (German PV -> compound surface -> fused -> spaced -> independent).
    """
    offsets = list(offsets)
    text_n = normalize(text)

    # Detect German separable particle verbs (trennbare Verben) where head is the particle
    t_head_n = normalize(head)
    is_de_pv = t_head_n in _GERMAN_PARTICLES or bool(
        compound and normalize(compound).startswith(t_head_n)
    )

    pair = None
    if is_de_pv:
        pair = _match_german_pv(text_n, mod, head)

    if pair is None:
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
    adjacent = (mod_sp.end == head_sp.start) or (head_sp.end == mod_sp.start) or (same and mod_sp.end == head_sp.end)
    return SpanResult(
        mod=Span(mod_sp.start, mod_sp.end, degenerate=same),
        head=Span(head_sp.start, head_sp.end, degenerate=same),
        adjacent=adjacent,
        found=True,
    )