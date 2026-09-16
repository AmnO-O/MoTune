"""Compound-grouped stratified folds + batch sampler (ported from the old stack).

Fold split is done at the COMPOUND level (a compound never appears in both
train and validation) and stratified by the compound's mean human score, so
the validation set is representative of the whole score range. Returns a row
dict with a ``fold`` key, not a DataFrame, so it plugs straight into
``load_labeled`` rows.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - sklearn is present on Kaggle
    StratifiedGroupKFold = None


def assign_folds(rows: List[Dict], n_splits: int = 5, seed: int = 42) -> List[Dict]:
    """Group rows by compound and assign a fold per compound (stratified).

    Mutates nothing; returns ``rows`` with a new ``fold`` integer per row.
    Compounds whose rows are all unlabeled are bucketed with the labeled ones'
    bins left as NaN and fall back to the last fold for stratification.
    """
    if StratifiedGroupKFold is None:
        raise RuntimeError('scikit-learn is required for fold assignment')

    out: List[Dict] = []
    for r in rows:
        out.append(dict(r))

    # stable compound ordering -> group ids are input-order invariant, so the
    # fold assignment does not depend on the order rows arrive in
    compounds = sorted({str(r['compound']) for r in rows})
    rows_by = {c: [r for r in rows if str(r['compound']) == c] for c in compounds}
    scores = [np.round(np.nanmean([(r['mod_avg'] + r['head_avg']) / 2.0
                                   for r in rows_by[c] if r['has_label']]), 6)
              if any(r['has_label'] for r in rows_by[c]) else float('nan')
              for c in compounds]

    cdf = pd.DataFrame({'compound': compounds, 'score': scores})
    graded = ~cdf['score'].isna()
    bins = cdf.loc[graded, 'score'].rank(method='dense')
    try:
        bins = pd.qcut(bins, q=n_splits, labels=False, duplicates='drop')
    except ValueError:
        bins = pd.cut(bins, bins=n_splits, labels=False)
    cdf.loc[graded, 'bin'] = bins.values

    graded_df = cdf[graded].reset_index(drop=True)
    group_ids = np.arange(len(graded_df))       # stable ids, order matches cdf

    fold_of: Dict[str, int] = {}
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
            sgkf.split(X=graded_df, y=graded_df['bin'], groups=group_ids)):
        for idx in val_idx:
            fold_of[graded_df['compound'].iloc[idx]] = int(fold)

    ungraded = cdf[~graded].index
    for i, pos in enumerate(ungraded):
        fold_of[cdf['compound'].iloc[pos]] = i % n_splits

    for r in out:
        r['fold'] = fold_of[str(r['compound'])]
    return out


# --------------------------------------------------------------------------- #
# batch sampler (ported unchanged from src/sampler.py)
# --------------------------------------------------------------------------- #
try:
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover - torch not installed locally
    class Sampler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def __iter__(self):
            raise ImportError('torch is required to use CompoundGroupSampler')

        def __len__(self):
            raise ImportError('torch is required to use CompoundGroupSampler')


class CompoundGroupSampler(Sampler):
    """Yields dataset indices grouped by compound so each batch holds many
    intra-compound sentence pairs for the pairwise ranking loss.

    Uses the ``set_epoch`` pattern (like ``DistributedSampler``): each epoch
    gets a reproducible but different shuffle without hidden auto-increment.
    """

    def __init__(self, compound_ids: Sequence[int], batch_size: int,
                 seed: int = 42):
        super().__init__()
        self.compound_ids = list(compound_ids)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.epoch = 0

        self.groups: dict = {}
        for i, c in enumerate(self.compound_ids):
            self.groups.setdefault(c, []).append(i)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.compound_ids)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 7919)
        compounds = list(self.groups)
        rng.shuffle(compounds)
        indices: List[int] = []
        for c in compounds:
            rows = self.groups[c][:]
            rng.shuffle(rows)
            indices.extend(rows)
        return iter(indices)