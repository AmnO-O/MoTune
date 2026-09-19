"""External static embeddings (fastText/word2vec .vec) for target constituents.

The combined backend fuses the span's CONTEXTUAL pool (final-layer hidden
states) with a STATIC, context-free representation of the constituent -- the
word's general meaning. Two sources exist:

* default (``config.static_span`` only): the frozen mmBERT embedding-table
  mean over the span's own tokens (same distribution as the backbone);
* external (``config.static_ext_path`` set): a word-vector ``.vec`` file
  (whitespace format ``<word> <float> ...``, optionally an ``N D`` header),
  looked up by the constituent's SURFACE form ("acid", "solution"). This is
  a truly different distribution (fastText captures German subword
  morphology that the BPE tokenizer folds into one token), so it anchors the
  readout against the lexical drift that fine-tuning causes in the top
  layers.

``StaticVec`` encapsulates the load, the word->vector lookup (unit-length,
mean over the surface form's whitespace parts), OOV fallback (zero vector,
which the fusion transformer learns to discard) and a coverage report for
debugging the del file.

Only the words actually present in the training/validation rows are kept in
memory -- the file is scanned once, every other line is thrown away -- so a
2M-word giant stays cheap for a small dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .marks import normalize

try:  # pragmatic: static_vec is a dataset/model helper, torch is always available there
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from .utils import get_logger
except ImportError:  # pragma: no cover
    import logging
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger('src.static_vec')


class StaticVec:
    """Word-level static vectors for the constituent surface forms of a dataset."""

    def __init__(self, path: str | Path, dim: int, words: Iterable[str]):
        self.path = Path(path)
        self.dim = int(dim)
        self._map: Dict[str, np.ndarray] = {}

        # Wanted set of normalized tokens taken from every surface form, so
        # the single pass over the .vec file only keeps rows we will use.
        self._wanted: Dict[str, None] = {}
        for w in words:
            for part in str(w).strip().lower().split():
                if part:
                    self._wanted.setdefault(normalize(part), None)

        self._load()
        self.coverage = len(self._map) / len(self._wanted) if self._wanted else 0.0
        logger.info(
            'StaticVec: %d/%d wanted words found in %s (dim=%d, coverage=%.1f%%)',
            len(self._map), len(self._wanted), self.path, self.dim,
            100.0 * self.coverage,
        )

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f'static_ext_path not found: {self.path}')
        logger.info('Scanning %s for %d wanted words...', self.path, len(self._wanted))
        line_no = 0
        with open(self.path, encoding='utf-8', errors='replace') as f:
            for line in f:
                line_no += 1
                parts = line.rstrip().split(' ')
                if len(parts) < 2:
                    continue
                w = parts[0].strip()
                if w not in self._wanted:
                    continue
                try:
                    vec = np.asarray(parts[1:1 + self.dim], dtype=np.float32)
                except ValueError:
                    continue                     # malformed line -> skip word
                if vec.size < self.dim:
                    continue                     # truncated vector -> skip word
                n = float(np.linalg.norm(vec))
                if n > 0:
                    vec = vec / n                # unit length before fusion
                self._map[w] = vec
                if not line_no % 250_000:
                    logger.info('  scanned %d lines, %d wanted found so far',
                                line_no, len(self._map))
        logger.info('finished: %d lines scanned, %d kept', line_no, len(self._map))

    def __contains__(self, word: str) -> bool:
        return any(p in self._map for p in str(word).strip().lower().split() if p)

    def __len__(self) -> int:
        return len(self._map)

    def vector(self, word: str) -> Optional[np.ndarray]:
        """Unit vector for ``word``, or None when every part is OOV."""
        if not word:
            return None
        vecs = [self._map[p] for p in str(word).strip().lower().split() if p in self._map]
        if not vecs:
            return None
        arr = np.mean(vecs, axis=0).astype(np.float32)
        n = float(np.linalg.norm(arr))
        return arr / n if n > 0 else None

    def tensor(self, word: str) -> 'torch.Tensor':
        """(dim,) float tensor for ``word``; zero-vector fallback on OOV."""
        v = self.vector(word)
        if v is None:
            return torch.zeros(self.dim, dtype=torch.float32)
        return torch.as_tensor(v, dtype=torch.float32)