"""Batch sampler that groups rows by compound so each batch contains many
intra-compound sentence pairs for the pairwise ranking loss.

Random batches almost never contain two rows of the same compound (~1.1
same-compound pairs per batch), which starves ``margin_rank_loss`` of signal.
Grouping K compounds x S rows per compound yields ~K*C(S,2) real ranking pairs
per batch (8x4 -> 48 pairs).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence

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
    """Yields dataset indices already grouped into full ``batch_size`` blocks.

    Every block is built by sampling ``K = batch_size // s_per_compound``
    compounds and drawing ``s_per_compound`` rows from each (rows are drawn
    without replacement when the compound has enough rows). Blocks are shuffled
    and padded up to ``batch_size`` with random rows when short.
    """

    def __init__(self, compound_ids: Sequence[int], batch_size: int,
                 s_per_compound: int = 4, seed: int = 42):
        super().__init__()
        self.compound_ids = list(compound_ids)
        self.batch_size = max(1, int(batch_size))
        self.s_per_compound = max(2, int(s_per_compound))
        self.seed = int(seed)

        self.groups: dict = {}
        for i, c in enumerate(self.compound_ids):
            self.groups.setdefault(c, []).append(i)
        self.compounds = sorted(self.groups)
        self._call = 0

    def __len__(self) -> int:
        return len(self.compound_ids)

    def __iter__(self):
        self._call += 1
        rng = random.Random(self.seed + self._call * 7919)
        k = max(1, self.batch_size // self.s_per_compound)

        n_batches = math.ceil(len(self.compound_ids) / self.batch_size)
        indices: List[int] = []
        rng.shuffle(self.compounds)

        for _ in range(n_batches):
            block: List[int] = []
            for c in rng.choices(self.compounds, k=k):
                idxs = self.groups[c]
                if len(idxs) <= self.s_per_compound:
                    block.extend(idxs)
                else:
                    block.extend(rng.sample(idxs, self.s_per_compound))
            rng.shuffle(block)
            while len(block) > self.batch_size:
                block.pop()
            while len(block) < self.batch_size:
                block.append(rng.randrange(len(self.compound_ids)))
            indices.extend(block)

        return iter(indices[: n_batches * self.batch_size])