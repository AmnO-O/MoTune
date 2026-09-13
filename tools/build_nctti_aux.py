#!/usr/bin/env python3
"""Build dataset/nctti_en.tsv (label-free auxiliary consistency rows) from the
NCTTI annotation + retrieved sentences shipped in nctti/data/.

Only sentences where the noun compound appears VERBATIM (adjacent words, the
same criterion mark_compound uses) are kept, so every aux row carries the real
compound in its real context. Mod/Head are derived by splitting the two-word
compound. Output has the same core schema as the training files
(ContextID, Compound, Mod, Head, Context) with NO label columns: these rows
feed the self-supervised compound-consistency term only.

Run from the repo root (requires the upstream nctti/ clone for its data/):
    python tools/build_nctti_aux.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sentids = pd.read_csv(ROOT / 'nctti' / 'data' / 'sentids_en.csv',
                          keep_default_na=False)

    rows = []
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
        per_compound = 0
        for col in ('sentence1', 'sentence2', 'sentence3'):
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
            per_compound += 1
        if per_compound == 0:
            # not counted separately; derive per-compound stats elsewhere
            pass

    out = pd.DataFrame(rows, columns=['ContextID', 'Compound', 'Mod', 'Head', 'Context'])
    out.to_csv(ROOT / 'dataset' / 'nctti_en.tsv', sep='\t', index=False)

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
    print('aux compounds:', out['Compound'].nunique())
    print(f'(skips: {skip_reasons}) (compounds with >=1 row: {one_or_more()})')
    return 0


if __name__ == '__main__':
    sys.exit(main())