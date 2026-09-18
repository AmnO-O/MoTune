"""Estimate the label-noise ceiling for the transparency scoring task.

For every labeled row we know the annotator dispersion (NN: ModStd/HeadStd,
PV: Std). The label is the mean annotation, so the best conceivable regressor
predicts the latent transparency exactly and its achievable Pearson vs the
noisy mean label is capped by

    rho* = sqrt(1 - E[ var(noise) ] / var(label))

`var(noise)` is the crowd std averaged over rows, using the row std directly
if the TSV reports per-annotator spread over all annotators, or the shrunk
std/sqrt(n) if we assume the reported std is of the mean. NN rows report
ModStd/HeadStd (per-role); PV rows report a single Std.

Because we don't know the annotator count n, we report a range:
  low  : treat Std as pure noise on every prediction (pessimistic)
  high : assume ~3 annotators, noise = Std/3 (optimistic)
The truth for score-averaged labels is inside this band. If the model's val
rho sits at the ceiling, further loss/architecture work is unlikely to help.

Outputs, per lineage (en-nn / de-nn / en-pv / de-pv) and overall:
  - labeled rows, label mean/std
  - mean crowd std (of mean label), noise variance
  - rho* ceiling (low/high)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset'

FILES = (
    'en-nn-train.tsv', 'de-nn-train.tsv',
    'en-pv-train.tsv', 'de-pv-train.tsv',
)


def _load(name: str) -> tuple:
    df = pd.read_csv(DATA / name, sep='\t', dtype=str, keep_default_na=False)
    if 'Mod' in df.columns and 'Head' in df.columns:
        tag, y1, s1, y2, s2 = 'nn', 'ModAvg', 'ModStd', 'HeadAvg', 'HeadStd'
    else:
        tag, y1, s1, y2, s2 = 'pv', 'Avg', 'Std', None, None
    y = pd.to_numeric(df[y1], errors='coerce')
    s = pd.to_numeric(df[s1], errors='coerce')
    if y2 is not None:
        y2v = pd.to_numeric(df[y2], errors='coerce')
        s2v = pd.to_numeric(df[s2], errors='coerce')
    else:
        y2v, s2v = y, s
    lab = (y.notna() & s.notna())
    if y2v is not None:
        lab &= y2v.notna() & s2v.notna()
    return tag, lab, y[lab], y2v[lab], s[lab], s2v[lab]


def _row_std(smod, shead):
    sm = smod.to_numpy(dtype=float)
    sh = shead.to_numpy(dtype=float)
    return np.sqrt((sm ** 2 + sh ** 2) / 2.0)


def profile() -> None:
    grand_y, grand_noise_low, grand_noise_high = [], [], []
    for name in FILES:
        tag, lab, y1, y2, s1, s2 = _load(name)
        if not lab.any():
            print(f'{name}: no labeled rows, skipped')
            continue
        yy = np.concatenate([y1.to_numpy(), y2.to_numpy()]) if tag == 'nn' \
            else y1.to_numpy()
        srow = _row_std(s1, s2) if tag == 'nn' else s1.to_numpy(dtype=float)
        var_y = float(np.var(yy))
        # pessimistic: the mean label itself is contaminated by std_i
        low_noise = float(np.mean(srow ** 2))
        # optimistic: assume 3 annotators averaged -> std of mean = std_i/sqrt(3)
        high_noise = float(np.mean(srow ** 2)) / 3.0
        ceil_low = float(np.sqrt(max(0.0, 1.0 - low_noise / var_y)))
        ceil_high = float(np.sqrt(max(0.0, 1.0 - high_noise / var_y)))
        grand_y.append(yy)
        grand_noise_low.append(low_noise)
        grand_noise_high.append(high_noise)
        print(f'[{name}] ({tag}) n_labeled={len(yy)} | label mean={np.mean(yy):.3f} '
              f'std={np.sqrt(var_y):.3f} | mean crowd std={np.sqrt(low_noise):.3f}')
        print(f'    rho* ceiling: {ceil_low:.3f} (pessimistic) ~ {ceil_high:.3f} (optimistic, ~3 annotators)')

    if grand_y:
        yy = np.concatenate(grand_y)
        var_y = float(np.var(yy))
        l = float(np.mean(grand_noise_low))
        h = float(np.mean(grand_noise_high))
        print(f'\nALL labeled: n={len(yy)} var(y)={var_y:.3f} '
              f'mean crowd std={np.sqrt(l):.3f}')
        print(f'OVERALL rho* ceiling: {np.sqrt(max(0., 1 - l/var_y)):.3f} '
              f'~ {np.sqrt(max(0., 1 - h/var_y)):.3f}')


if __name__ == '__main__':
    profile()