#!/usr/bin/env python3
"""Build dataset/nctti_en.tsv + dataset/nctti_en_scored.tsv from the NCTTI
annotation + retrieved sentences shipped in nctti/data/.

Only sentences where the noun compound appears VERBATIM (adjacent words, the
same criterion mark_compound uses) are kept, so every row carries the real
compound in its real context. Mod/Head are derived by splitting the two-word
compound.

Outputs:
  dataset/nctti_en.tsv        label-free (ContextID, Compound, Mod, Head,
                              Context): feeds the self-supervised
                              compound-consistency / MLM term only.
  dataset/nctti_en_scored.tsv PV-schema (ContextID, ParticleVerb, Base,
                              Particle, Avg, Std, Context): each row carries the
                              per-sentence token-level compositionality score
                              MeanS{1..3}, so NCTTI rows train the OVERALL
                              gauss head exactly like particle-verb data
                              (``is_pv``). Std is a singleton: NCTTI reports
                              only means, so it borrows the mean annotator Std
                              of the real PV train files (or 1.51 fallback).

Run from the repo root (requires the upstream nctti/ clone for its data/):
    python tools/build_nctti_aux.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SENT_COLS = ('sentence1', 'sentence2', 'sentence3')
STD_FALLBACK = 1.51


def _pv_std() -> float:
    """Mean annotator Std of the real PV train files, or a documented fallback."""
    stds = []
    for name in ('en-pv-train.tsv', 'de-pv-train.tsv'):
        path = ROOT / 'dataset' / name
        if not path.is_file():
            continue
        df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
        if 'Std' in df.columns:
            v = pd.to_numeric(df['Std'], errors='coerce')
            stds.append(v.dropna())
    if stds:
        return float(np.mean(np.concatenate(stds)))
    return STD_FALLBACK


def main() -> int:
    sentids = pd.read_csv(ROOT / 'nctti' / 'data' / 'sentids_en.csv',
                          keep_default_na=False)
    means = pd.read_csv(ROOT / 'nctti' / 'data' / 'data_en.tsv',
                        sep='\t', keep_default_na=False)
    score_by_compound = {
        str(r['compound']).strip().lower(): r
        for _, r in means.iterrows()
    }

    pv_std = _pv_std()

    rows = []
    scored = []
    skip_reasons = {'missing': 0, 'placeholder': 0, 'not_2word': 0, 'no_match': 0}
    for _, r in sentids.iterrows():
        compound = str(r.get('compound', '')).strip()
        parts = compound.lower().split()
        if len(parts) != 2:
            skip_reasons['not_2word'] += 1
            continue
        # mark_compound matches the compound as a contiguous word pair (head may
        # carry a plural/possessive suffix); mirror that without the suffix here.
        pattern = re.compile(
            rf'\b({re.escape(parts[0])})\b\s+({re.escape(parts[1])}(?:es|s|\'s|\u2019s)?)\b',
            re.IGNORECASE,
        )
        mrow = score_by_compound.get(compound.lower())
        per_compound = 0
        for n, col in enumerate(SENT_COLS, start=1):
            text = str(r.get(col, '')).strip()
            if not text or text in {'nan'}:
                skip_reasons['missing'] += 1
                continue
            if text.startswith('sent'):
                skip_reasons['placeholder'] += 1
                continue
            if pattern.search(text) is None:
                skip_reasons['no_match'] += 1
                continue
            rows.append({
                'ContextID': f'NCTTI-{len(rows):04d}',
                'Compound': compound,
                'Mod': parts[0],
                'Head': parts[1],
                'Context': text,
            })
            avg = np.nan
            if mrow is not None:
                avg = pd.to_numeric(mrow.get(f'MeanS{n}'), errors='coerce')
            avg = float(avg) if avg is not None and np.isfinite(avg) else np.nan
            if np.isfinite(avg):
                scored.append({
                    'ContextID': f'NCTTI-{len(scored):04d}',
                    'ParticleVerb': compound,
                    'Base': parts[0],
                    'Particle': parts[1],
                    'Avg': f'{avg:.4f}',
                    'Std': f'{pv_std:.4f}',
                    'Context': text,
                })
            per_compound += 1
        if per_compound == 0:
            # not counted separately; derive per-compound stats elsewhere
            pass

    out = pd.DataFrame(rows, columns=['ContextID', 'Compound', 'Mod', 'Head', 'Context'])
    out.to_csv(ROOT / 'dataset' / 'nctti_en.tsv', sep='\t', index=False)

    scored_out = pd.DataFrame(scored, columns=[
        'ContextID', 'ParticleVerb', 'Base', 'Particle', 'Avg', 'Std', 'Context'])
    scored_out.to_csv(ROOT / 'dataset' / 'nctti_en_scored.tsv', sep='\t', index=False)

    def one_or_more() -> int:
        return int(sentids.apply(
            lambda r: (
                len(str(r.get('compound', '')).split()) == 2
                and any(
                    not str(r.get(c, '')).strip() in {'', 'nan'}
                    and not str(r.get(c, '')).strip().startswith('sent')
                    and re.search(
                        rf'\b({re.escape(str(r["compound"]).split()[0])})\b\s+'
                        rf'({re.escape(str(r["compound"]).split()[1])}(?:es|s|\'s|\u2019s)?)\b',
                        str(r.get(c, '')), re.IGNORECASE,
                    ) is not None
                    for c in ('sentence1', 'sentence2', 'sentence3')
                )
            ),
            axis=1,
        ).sum())

    print(f'wrote {len(out)} rows -> dataset/nctti_en.tsv')
    print(f'wrote {len(scored_out)} rows -> dataset/nctti_en_scored.tsv (PV schema, std={pv_std:.3f})')
    print('aux compounds:', out['Compound'].nunique())
    if len(scored_out):
        print('scored compounds:', scored_out['ParticleVerb'].nunique())
    print(f'(skips: {skip_reasons}) (compounds with >=1 row: {one_or_more()})')
    return 0


if __name__ == '__main__':
    sys.exit(main())