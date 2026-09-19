"""Build a single, filtered fastText .vec for the external static anchors.

Streams the big cc crawl word vectors (EN + DE, ~3 GB gz in total) and keeps
ONLY the tokens this dataset's constituent surface forms need, writing them as
one small .vec that the trainer's ``StaticVec`` reads (see ``static_ext_path`` /
``static_ext_dim`` config knobs). Reruns re-download unless ``--cache`` is given.

Usage (repo root)::

    python scripts/build_static_vec.py --out vectors/cc_en_de_300.vec

Then point a run at it::

    --set static_ext_path=vectors/cc_en_de_300.vec --set static_ext_dim=300

``--limit N`` processes at most N lines per source and skips the coverage
report -- a quick connectivity/parse probe without the full 3 GB transfer.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data import load_labeled
from src.marks import normalize

EN_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz'
DE_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz'

# Which surface fields decide "anchor available" per row type (NN vs PV).
_NEEDED = {'mod', 'head', 'compound'}


def collect_wanted_words() -> Set[str]:
    """Set of normalized surface-form tokens the training rows will look up.

    Mirror of the trainer: 'mod'/'head' for every row, 'compound' (the PV
    anchor) as well -- both NN (`Compound`) and PV (`ParticleVerb`) land there.
    """
    rows = load_labeled(Config.defaults())
    wanted: Set[str] = set()
    for r in rows:
        for k in _NEEDED:
            w = r.get(k)
            if w:
                wanted.update(normalize(p) for p in w.strip().lower().split() if p)
    print(f'wanted: {len(rows)} rows -> {len(wanted)} unique surface tokens')
    return wanted


def download(url: str, cache: Optional[str]) -> Path:
    """Stream ``url`` to disk (or reuse ``cache`` if present)."""
    if cache and Path(cache).exists() and Path(cache).stat().st_size > 0:
        print(f'  reusing cached {cache}')
        return Path(cache)
    dest = Path(cache or Path(url).name)
    part = Path(str(dest) + '.part')
    print(f'  downloading {url} -> {dest}', flush=True)
    import urllib.request
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                print(f'    {done / 1e6:.0f}/{total / 1e6:.0f} MB', end='\r', flush=True)
    print()
    part.rename(dest)
    return dest


def keep_wanted(src: Path, wanted: Set[str], found: Set[str],
                out: Path, tag: str, limit: int) -> None:
    """One pass over a (gz) file, appending wanted lines to ``out``."""
    n, kept = 0, 0
    with gzip.open(src, 'rb') as fin, open(out, 'a', encoding='utf-8') as fout:
        for raw in fin:
            n += 1
            if limit and n > limit:
                break
            try:
                line = raw.decode('utf-8')
            except UnicodeDecodeError:
                continue
            parts = line.rstrip().split(' ')
            if len(parts) < 2:
                continue                       # header "N D" / blank lines
            w = parts[0]
            if w in wanted and w not in found:
                found.add(w)
                kept += 1
                fout.write(line.rstrip('\r\n') + '\n')
            if n % 500_000 == 0:
                print(f'  [{tag}] {n} lines, {len(found)} kept so far', flush=True)
    print(f'  [{tag}] done: {n} lines, {kept} new kept')


def report(out: Path, dim: int, wanted: Set[str], limit: int) -> None:
    """Coverage of the built .vec against the real rows (mirrors the trainer)."""
    from src.static_vec import StaticVec

    sv = StaticVec(out, dim, wanted)
    print(f'\nStaticVec: {len(sv)}/{len(wanted)} wanted tokens found '
          f'({100 * sv.coverage:.1f}%)')

    def parts(word: str) -> list:
        return [normalize(p) for p in str(word).strip().lower().split() if p]

    def present(word: str) -> bool:
        return all(p in sv for p in parts(word)) if word else False

    all_rows = load_labeled(Config.defaults())

    def usable(r) -> bool:
        # NN rows: mod AND head anchors must both exist. PV rows are anchored
        # on the WHOLE compound surface; its parts (base+particle) are the
        # backstop when the compound is OOV.
        if r.get('is_pv'):
            return present(r.get('compound', '')) or (
                present(r.get('mod', '')) and present(r.get('head', '')))
        return present(r.get('mod', '')) and present(r.get('head', ''))

    n_ok = sum(1 for r in all_rows if usable(r))
    print(f'rows with a usable static anchor: '
          f'{n_ok}/{len(all_rows)} ({100 * n_ok / len(all_rows):.1f}%)')

    pv_rows = [r for r in all_rows if r.get('is_pv')]
    pv_ok = sum(1 for r in pv_rows if usable(r))
    print(f'  of which PV rows: {pv_ok}/{len(pv_rows)} '
          f'({100 * pv_ok / len(pv_rows):.1f}%) anchored on the compound')

    de_rows = [r for r in all_rows if r.get('lang') == 'de']
    de_ok = sum(1 for r in de_rows if usable(r))
    print(f'  of which German rows: {de_ok}/{len(de_rows)} '
          f'({100 * de_ok / len(de_rows):.1f}%)')

    probes = ['abgehauen', 'Abiturzeugnis', 'abhauen']
    print('\nSpot checks (fused German forms, if present in this dataset):')
    in_vocab = {normalize(p): p in wanted for p in probes}
    for probe in probes:
        hit = bool(probe in sv)
        print(f'  {probe}: {"FOUND" if hit else "missing"}'
              + ('' if in_vocab.get(probe) else ' (not a dataset surface form)'))


def build(out: Path, dim: int, cache: Optional[str], limit: int) -> None:
    wanted = collect_wanted_words()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    found: Set[str] = set()
    for tag, url in (('en', EN_URL), ('de', DE_URL)):
        src = download(url, cache)
        keep_wanted(src, wanted, found, out, tag, limit)
    if not limit:
        report(out, dim, wanted, limit)
    print('\nbuilt:', out, f'({out.stat().st_size / 1e6:.2f} MB)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='vectors/cc_en_de_300.vec')
    ap.add_argument('--dim', type=int, default=300)
    ap.add_argument('--cache', default=None, help='dir to keep the .gz downloads')
    ap.add_argument('--limit', type=int, default=0,
                    help='process at most N lines per source (quick probe)')
    args = ap.parse_args()
    build(Path(args.out), args.dim, args.cache, args.limit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())