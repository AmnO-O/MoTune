"""Compound-grouped stratified folds + batch sampler (ported from the old stack).

Fold split is done at the COMPOUND level (a compound never appears in both
train and validation) and stratified by the compound's mean human score, so
the validation set is representative of the whole score range. Returns a row
dict with a ``fold`` key, not a DataFrame, so it plugs straight into
``load_labeled`` rows.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:  # pragma: no cover - sklearn is present on Kaggle
    StratifiedKFold = None


def assign_folds(rows: List[Dict], n_splits: int = 5, seed: int = 42) -> List[Dict]:
    """Group rows by compound and assign a fold per compound (stratified).

    Mutates nothing; returns ``rows`` with a new ``fold`` integer per row.
    Compounds whose rows are all unlabeled are bucketed with the labeled ones'
    bins left as NaN and fall back to the last folds for stratification.
    """
    if StratifiedKFold is None:
        raise RuntimeError('scikit-learn is required for fold assignment')

    out: List[Dict] = [dict(r) for r in rows]

    # stable compound ordering -> the fold assignment does not depend on the
    # order rows arrive in
    rows_by: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        rows_by[str(r['compound'])].append(r)
    compounds = sorted(rows_by.keys())

    scores: List[float] = []
    for c in compounds:
        labeled = [(r['mod_avg'] + r['head_avg']) / 2.0
                   for r in rows_by[c] if r['has_label']]
        scores.append(np.round(float(np.mean(labeled)), 6) if labeled else float('nan'))

    cdf = pd.DataFrame({'compound': compounds, 'score': scores})
    graded = ~cdf['score'].isna()

    graded_has_bin = graded.any()
    if graded_has_bin:
        bins = cdf.loc[graded, 'score'].rank(method='dense')
        try:
            bins = pd.qcut(bins, q=n_splits, labels=False, duplicates='drop')
        except ValueError:
            bins = pd.cut(bins, bins=n_splits, labels=False)
        cdf.loc[graded, 'bin'] = bins.values

    # cdf holds ONE row per compound, so the compound-disjoint constraint is
    # already satisfied by construction: plain StratifiedKFold on the compound
    # frame == StratifiedGroupKFold with per-row-unique groups (which would
    # only add sklearn-version-dependent behavior).
    fold_of: Dict[str, int] = {}
    graded_df = cdf[graded].reset_index(drop=True)
    if len(graded_df):
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            fold_splits = list(skf.split(graded_df, graded_df['bin']))
        except ValueError:
            from sklearn.model_selection import KFold
            fold_splits = list(KFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed).split(graded_df))
        for fold, (_, val_idx) in enumerate(fold_splits):
            for idx in val_idx:
                fold_of[graded_df['compound'].iloc[idx]] = int(fold)

    for i, comp in enumerate(cdf.loc[~graded, 'compound']):
        fold_of[comp] = i % n_splits

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