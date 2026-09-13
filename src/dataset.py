import logging
import random
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.matching import fallback_marked_text, mark_compound, span_text_offsets

logger = logging.getLogger('compartment')


def _report_nonfinite(df: pd.DataFrame, tag: str, columns) -> None:
    """Warn when a label column carries NaN/inf, listing the offending rows.

    DataFrame columns that should never be non-finite (human-judgment
    stats) occasionally leak in from hand-made TSVs; a NaN ModStd becomes an
    all-NaN Gaussian soft target and silently NaN-poisons the CE loss.
    """
    present = [c for c in columns if c in df.columns]
    if not present:
        return
    finite = pd.DataFrame(index=df.index)
    for c in present:
        finite[c] = df[c].apply(lambda v: _is_finite(v))
    bad_rows = ~finite[present].all(axis=1)
    n = int(bad_rows.sum())
    if n == 0:
        return
    show = ['Context', 'Mod', 'Head', 'Compound'] + present
    show = [c for c in show if c in df.columns]
    logger.warning(
        '%s: %d row(s) have non-finite %s; these fall back to fixed bin_sigma. Examples:\n%s',
        tag, n, '/'.join(present),
        df.loc[bad_rows, show].head(5).to_string(index=True),
    )


def _is_finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


_MARKER_SPLIT = re.compile(r'(<mod>|</mod>|<head>|</head>)')


def _augment_marked_text(marked: str, rng: random.Random) -> str:
    """Label-preserving context augmentation.

    Random token-dropout of ~15% of NON-marker words ONLY. No tail-cropping:
    trimming removes the words nearest to the MWE (the most context a
    compositionality judgment depends on) and breaks grammar harder, so it is
    dropped. Token-dropout is the perturbation BERT-style encoders tolerate
    best and teaches span invariance to distant context. The marked spans are
    always kept intact. Returns the input unchanged if the protected spans
    could not be preserved.
    """
    out = []
    for part in _MARKER_SPLIT.split(marked):
        if not part:
            continue
        if part.startswith('<'):
            out.append(part)
            continue
        words = part.split()
        if not words:
            continue
        keep = [w for w in words if rng.random() > 0.15]
        if keep:
            out.append(' '.join(keep))
    augmented = ' '.join(out)
    if '<mod>' not in augmented or '<head>' not in augmented:
        return marked
    return augmented


class NNDataset(Dataset):
    """Tokenizer wrapper producing ONE marked sentence per row.

    The compound's modifier / head spans are wrapped in ``<mod>`` / ``<head>``
    markers inside the real context sentence. Each item carries ``input_ids``,
    ``attention_mask`` and one boolean span mask per role, so the model pools
    hidden states over the marked words only.

    ``max_context_length`` caps the marked sentence (the context sentences are
    longer than the old standalone Mod/Head/Compound items).

    Tokenization is cached per row (``augment_prob == 0``), so epoch switches
    are cheap. With ``augment_prob > 0`` caching is disabled and each draw
    rounds up to 256 tokens anyway.
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_context_length: int = 256,
        is_test: bool = False,
        augment_prob: float = 0.0,
        seed: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.is_test = is_test
        self.augment_prob = augment_prob
        self.rng = random.Random(seed)
        self._cache: dict[int, dict[str, torch.Tensor]] = {}
        self._cache_enabled = augment_prob == 0.0
        if not is_test:
            _report_nonfinite(self.df, 'train split', ('ModStd', 'HeadStd'))

        # Numeric compound id (per row, stable) for the within-compound
        # pairwise ranking loss. Test/unlabeled frames without a Compound
        # column fall back to all-zeros (no ranking loss is computed there).
        if 'Compound' in self.df.columns:
            self.compound_ids = pd.factorize(self.df['Compound'])[0]
        else:
            self.compound_ids = np.zeros(len(self.df), dtype=np.int64)
        self.compound_ids = self.compound_ids.astype(np.int64)

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self._cache_enabled and idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]

        marked = mark_compound(
            str(row['Context']), str(row['Mod']), str(row['Head'])
        )
        if marked is None:
            # Rare: compound not found verbatim in the sentence. Fall back to a
            # deterministic marked compound + the (unmarked) context.
            marked = fallback_marked_text(
                str(row['Mod']), str(row['Head']), str(row['Context'])
            )

        if self.augment_prob > 0 and self.rng.random() < self.augment_prob:
            marked = _augment_marked_text(marked, self.rng)

        encoded = self.tokenizer(
            marked,
            max_length=self.max_context_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        offsets = encoded['offset_mapping'].squeeze(0).tolist()

        mod_span_mask = self._span_mask(marked, 'mod', offsets)
        head_span_mask = self._span_mask(marked, 'head', offsets)
        # Compound span is ALWAYS the contiguous mod+head pair (markers
        # guarantee adjacency), so no separate <mwe> fallback is needed.
        mwe_span_mask = mod_span_mask | head_span_mask

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mod_span_mask': mod_span_mask,
            'head_span_mask': head_span_mask,
            'mwe_span_mask': mwe_span_mask,
            'compound_id': torch.tensor(self.compound_ids[idx], dtype=torch.long),
        }

        if not self.is_test and 'ModAvg' in row and 'HeadAvg' in row:
            item['mod_avg'] = torch.tensor(float(row['ModAvg']), dtype=torch.float)
            item['head_avg'] = torch.tensor(float(row['HeadAvg']), dtype=torch.float)
            if 'ModStd' in row:
                item['mod_std'] = torch.tensor(float(row['ModStd']), dtype=torch.float)
            if 'HeadStd' in row:
                item['head_std'] = torch.tensor(float(row['HeadStd']), dtype=torch.float)

        if self._cache_enabled:
            self._cache[idx] = item
        return item

    def _span_mask(self, marked, tag, offsets):
        """Boolean mask over tokens whose character window lies inside the span."""
        span = span_text_offsets(marked, tag)
        mask = torch.zeros(len(offsets), dtype=torch.bool)
        if span is None:
            return mask
        start, end = span
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end > start and tok_start < end:
                mask[i] = True
        return mask