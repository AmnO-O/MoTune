"""Batch sampler that groups rows by compound so each batch contains many
intra-compound sentence pairs for the pairwise ranking loss.

Random batches almost never contain two rows of the same compound (~1.1
same-compound pairs per batch), which starves ``margin_rank_loss`` of signal.
Grouping consecutive rows of the same compound yields dense pairs while
guaranteeing every row is seen exactly once per epoch.
"""

from __future__ import annotations

import random
from typing import List, Sequence

try:
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover - torch not installed locally
    class Sampler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError('torch is required to use CompoundGroupSampler')

        def __iter__(self):
            raise ImportError('torch is required to use CompoundGroupSampler')

        def __len__(self):
            raise ImportError('torch is required to use CompoundGroupSampler')


class CompoundGroupSampler(Sampler):
    """Yields dataset indices grouped into ``batch_size`` blocks by compound.

    All rows of each compound are emitted together so every batch contains
    rows from only 3–5 compounds, producing many intra-compound pairs for the
    ranking loss. Every row is seen exactly once per epoch (no padding, no
    dropped rows).
    """

    def __init__(self, compound_ids: Sequence[int], batch_size: int,
                 s_per_compound: int = 4, seed: int = 42):
        super().__init__()
        self.compound_ids = list(compound_ids)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)

        self.groups: dict = {}
        for i, c in enumerate(self.compound_ids):
            self.groups.setdefault(c, []).append(i)
        self._call = 0

    def __len__(self) -> int:
        return len(self.compound_ids)

    def __iter__(self):
        self._call += 1
        rng = random.Random(self.seed + self._call * 7919)

        # Shuffle compound order, then emit all rows of each compound together.
        # DataLoader batches these consecutively, so every batch holds only 3-5
        # compounds and the (possibly partial) last batch is handled normally.
        compounds = list(self.groups)
        rng.shuffle(compounds)

        indices: List[int] = []
        for c in compounds:
            rows = self.groups[c][:]
            rng.shuffle(rows)
            indices.extend(rows)

        return iter(indices)