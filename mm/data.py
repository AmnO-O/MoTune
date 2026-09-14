"""Data loading, tokenization and span alignment for the mmBERT pipeline.

Marker-free design (phase M2): the raw context is tokenized verbatim with
``return_offsets_mapping=True``; the compound's Mod/Head surface forms are
matched inside the original text by ``marks.find_spans`` and mapped onto
token spans. No ``<mod>`` / ``<head>`` markers are inserted, so the pretrained
mmBERT tokenizer/embeddings never see out-of-vocabulary artifacts.

Two dataset types:

  * ``CompDataset``  -- one row per labeled / aux sentence, for phase-1
    scoring. Yields span masks plus (possibly NaN) soft labels; aux and
    unaligned rows keep the row but mark ``has_label`` / ``has_mod`` /
    ``has_head`` False so losses can mask them.
  * ``MlmDataset``   -- compound-aware MLM warmup: masks the modifier span,
    the head span, or the whole compound (``mlm_mask_span`` = one|both) with
    BERT's 80/10/10 recipe; rows without an aligned compound fall back to
    standard 15% random masking so every sentence stays useful.

Both datasets pre-tokenize and pre-align in the constructor (deterministic,
single pass), which also surfaces a per-source alignment report for the
Kaggle M2 verification step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .marks import Span, SpanResult, find_spans
from .utils import get_logger

logger = get_logger('mm.data')

try:
    import torch
except ImportError:  # pragma: no cover - datasets need torch; loaders don't
    torch = None

_PAD_ID = 0
_SPECIAL_OFFSET = (0, 0)


def _need_torch():
    if torch is None:
        raise RuntimeError('torch is required to build torch Datasets')


_DatasetBase = torch.utils.data.Dataset if torch is not None else object


def _is_nn(df: pd.DataFrame) -> bool:
    return 'Mod' in df.columns and 'Head' in df.columns


def _is_pv(df: pd.DataFrame) -> bool:
    return 'Base' in df.columns and 'Particle' in df.columns


def _auto_lang(path: str) -> str:
    name = Path(path).name.lower()
    return 'de' if name.startswith('de-') else 'en'


def _f(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a dataset TSV as raw strings (numeric coercion happens per row)."""
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    return df


def _df_to_rows(df: pd.DataFrame, tag: str, lang: str) -> List[Dict]:
    """Normalize an NN or PV dataframe to row dicts (see module docstring)."""
    nn = _is_nn(df)
    if not nn and not _is_pv(df):
        raise ValueError(
            f'{tag}: expected NN columns (Compound/Mod/Head) or PV columns '
            f'(ParticleVerb/Base/Particle), got {list(df.columns)}'
        )

    rows: List[Dict] = []
    for _, r in df.iterrows():
        if nn:
            mod, head, compound = r['Mod'], r['Head'], r.get('Compound', '')
        else:
            mod, head, compound = r['Base'], r['Particle'], r.get('ParticleVerb', '')

        def val(col):
            return _f(r[col]) if col in r else float('nan')

        mod_avg = val('ModAvg' if nn else 'Avg')
        head_avg = val('HeadAvg' if nn else 'Avg')
        mod_std = val('ModStd' if nn else 'Std')
        head_std = val('HeadStd' if nn else 'Std')

        rows.append({
            'context': str(r['Context']),
            'mod': str(mod),
            'head': str(head),
            'compound': str(compound),
            'lang': lang,
            'has_label': bool(np.isfinite(mod_avg)),
            'mod_avg': mod_avg,
            'head_avg': head_avg,
            'mod_std': mod_std,
            'head_std': head_std,
            'compound_id': -1,
            'context_id': str(r.get('ContextID', '')),
        })
    return rows


# --------------------------------------------------------------------------- #
# top-level loaders
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# top-level loaders
# --------------------------------------------------------------------------- #
def _load_files(cfg, file_attrs: List[str], fallback_attr: str) -> List[Dict]:
    """Helper nạp và gộp nhiều file TSV (EN/DE, NN/PV) theo cấu hình Config."""
    data_dir, _ = _resolve(cfg)
    
    files_to_load: List[str] = []
    for attr in file_attrs:
        fname = getattr(cfg, attr, None)
        if fname and fname.strip() and fname not in files_to_load:
            files_to_load.append(fname)

    # Fallback về thuộc tính đơn lẻ cũ nếu không chỉ định multi-task
    if not files_to_load:
        fallback = getattr(cfg, fallback_attr, None)
        if fallback and fallback.strip():
            files_to_load.append(fallback)

    all_rows: List[Dict] = []
    for fname in files_to_load:
        path = data_dir / fname
        if not path.exists():
            logger.warning('Dataset file missing, skipping: %s', path)
            continue
        df = read_tsv(path)
        rows = _df_to_rows(df, fname, _auto_lang(fname))
        all_rows.extend(rows)
        logger.info('%s: %d rows', fname, len(rows))

    if not all_rows:
        raise FileNotFoundError(f'No valid dataset files found in {data_dir}')

    # Định danh compound_id duy nhất kèm ngôn ngữ (vd: en_blackboard vs de_apfelbaum)
    compound_keys = [f"{r['lang']}_{r['compound']}" for r in all_rows]
    codes, _ = pd.factorize(pd.Series(compound_keys))
    for r, c in zip(all_rows, codes):
        r['compound_id'] = int(c)

    logger.info('Loaded %d total rows across %d unique compounds', len(all_rows), int(codes.max()) + 1)
    return all_rows


def load_labeled(cfg) -> List[Dict]:
    """Load dữ liệu Train đa ngữ/đa dạng (EN/DE, NN/PV)."""
    train_attrs = ['en_nn_train', 'de_nn_train', 'en_pv_train', 'de_pv_train']
    return _load_files(cfg, train_attrs, 'train_file')


def load_trial(cfg) -> List[Dict]:
    """Load dữ liệu Trial/Validation đa ngữ/đa dạng (EN/DE, NN/PV)."""
    trial_attrs = ['en_nn_trial', 'de_nn_trial', 'en_pv_trial', 'de_pv_trial']
    return _load_files(cfg, trial_attrs, 'trial_file')


def load_aux(cfg) -> List[Dict]:
    """Label-free consistency rows (aux_data_paths); used for representation only."""
    data_dir, _ = _resolve(cfg)
    rows: List[Dict] = []
    for name in cfg.aux_data_paths:
        path = data_dir / name
        if not path.exists():
            logger.warning('aux_data_path missing, skipping: %s', path)
            continue
        aux = _df_to_rows(read_tsv(path), name, _auto_lang(name))
        for r in aux:
            r.update({
                'has_label': False,
                'mod_avg': float('nan'), 'head_avg': float('nan'),
                'mod_std': float('nan'), 'head_std': float('nan'),
            })
        rows += aux
        logger.info('%s: %d aux rows', name, len(aux))
    return rows


def load_mlm_rows(cfg) -> List[Dict]:
    """Unsupervised sentences for the compound-aware MLM warmup.

    Accepts any mix of NN/PV/aux files (any language); only the context and
    the (optional) compound are used.
    """
    data_dir, _ = _resolve(cfg)
    rows: List[Dict] = []
    for name in cfg.mlm_data_paths:
        path = data_dir / name
        if not path.exists():
            logger.warning('mlm_data_path missing, skipping: %s', path)
            continue
        part = _df_to_rows(read_tsv(path), name, _auto_lang(name))
        for r in part:
            r.update({'has_label': False, 'compound_id': -1})
        rows += part
        logger.info('%s: %d warmup rows', name, len(part))
    if not rows:
        raise ValueError('mlm_data_paths resolved to zero rows (all files missing?)')
    logger.info('total warmup rows: %d', len(rows))
    return rows


def _resolve(cfg):
    from .utils import resolve_paths
    return resolve_paths(cfg)


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def _span_mask(span: Optional[Span], length: int) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    if span is not None and span.start is not None:
        mask[span.start:span.end] = True
    return mask


class CompDataset(_DatasetBase):
    """One sentence (tokenized verbatim) per row, for scoring."""

    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 256,
                 is_test: bool = False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.items = [self._encode(r) for r in rows]
        self._report()

    def _encode(self, r: Dict) -> Dict:
        enc = self.tokenizer(
            r['context'], max_length=self.max_len, truncation=True,
            return_tensors='pt', return_offsets_mapping=True,
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        offsets = enc['offset_mapping'].squeeze(0).tolist()
        length = input_ids.size(0)

        result: SpanResult = find_spans(
            r['context'], offsets, r['mod'], r['head'], r.get('compound', '')
        ) if (r['mod'] and r['head']) \
            else SpanResult(Span(None, None), Span(None, None), found=False)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mod_span_mask': _span_mask(result.mod, length),
            'head_span_mask': _span_mask(result.head, length),
            'has_mod': torch.tensor(result.mod.start is not None, dtype=torch.bool),
            'has_head': torch.tensor(result.head.start is not None, dtype=torch.bool),
            'degenerate': torch.tensor(result.degenerate, dtype=torch.bool),
            'compound_id': torch.tensor(int(r['compound_id']), dtype=torch.long),
            'has_label': torch.tensor(bool(r['has_label']), dtype=torch.bool),
            'mod_avg': torch.tensor(float(r['mod_avg']), dtype=torch.float),
            'head_avg': torch.tensor(float(r['head_avg']), dtype=torch.float),
            'mod_std': torch.tensor(float(r['mod_std']), dtype=torch.float),
            'head_std': torch.tensor(float(r['head_std']), dtype=torch.float),
            'row_id': torch.tensor(int(r.get('row_id', 0)), dtype=torch.long),
        }

    def _report(self) -> None:
        if self.is_test:
            return
        n = len(self.items) or 1
        found = sum(1 for it in self.items if bool(it['has_mod'] and it['has_head']))
        deg = sum(1 for it in self.items if bool(it['degenerate']))
        lbl = sum(1 for it in self.items if bool(it['has_label']))
        logger.info(
            'CompDataset: %d rows, %d%% aligned, %d degenerate, %d labeled',
            len(self.items), int(100 * found / n), deg, lbl,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


class MlmDataset(_DatasetBase):
    """Compound-aware MLM warmup: mask span tokens (80/10/10) per BERT recipe.

    Rows without an aligned compound (or with a degenerate fused token) fall
    back to standard 15% random masking over non-special tokens, so every
    sentence still contributes a masking signal.
    """

    RANDOM_RATIO = 0.15

    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 128,
                 mask_span: str = 'both', mask_prob: float = 0.8,
                 random_prob: float = 0.1, seed: int = 0, vocab_size: Optional[int] = None):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_span = mask_span          # 'one' | 'both'
        self.mask_prob = mask_prob          # P([MASK]) over span tokens
        self.random_prob = random_prob      # P(random token) over span tokens
        self.seed = seed
        # random-replacement tokens must be EMBEDDABLE: never sample above the
        # model's real vocabulary (which can differ from tokenizer.vocab_size,
        # e.g. mmBERT: model 256k vs tokenizer 256k+added specials). The caller
        # may pass the model's embedding-table width via `vocab_size`; clamp to
        # the tokenizer's size too so ids are in range for BOTH.
        tok_vocab = int(getattr(tokenizer, 'vocab_size', 0) or 0)
        requested = int(vocab_size) if vocab_size else 0
        self.vocab_size = requested or tok_vocab or 32000
        if tok_vocab:
            self.vocab_size = min(self.vocab_size, tok_vocab)
        if requested and tok_vocab and requested != tok_vocab:
            logger.warning(
                'MlmDataset: tokenizer vocab=%d != model vocab=%d; '
                'random-replacement ids clamped to [0, %d)',
                tok_vocab, requested, self.vocab_size,
            )
        self.rng = np.random.RandomState(seed)

        # pre-tokenize once; store (ids, mask, label) templates
        self._base: List[Dict] = []
        self._plan: List[Dict] = []         # span token positions per row
        for r in rows:
            enc = self.tokenizer(
                r['context'], max_length=self.max_len, truncation=True,
                return_tensors='pt', return_offsets_mapping=True,
            )
            input_ids = enc['input_ids'].squeeze(0)
            attention_mask = enc['attention_mask'].squeeze(0)
            offsets = enc['offset_mapping'].squeeze(0).tolist()
            length = input_ids.size(0)

            non_special = [i for i, (s, e) in enumerate(offsets)
                           if (s, e) != _SPECIAL_OFFSET and bool(attention_mask[i])]

            r_sp = None
            if r['mod'] and r['head']:
                res = find_spans(
                    r['context'], offsets, r['mod'], r['head'], r.get('compound', '')
                )
                if res.found and not res.degenerate:
                    r_sp = {
                        'mod': list(range(res.mod.start, res.mod.end)),
                        'head': list(range(res.head.start, res.head.end)),
                    }
            self._plan.append({'spans': r_sp, 'non_special': non_special})
            self._base.append({'input_ids': input_ids, 'attention_mask': attention_mask})

    # ------------------------------------------------------------------ #
    def _rand_token_id(self) -> int:
        """Random replacement id that is guaranteed to be embeddable."""
        return int(self.rng.randint(max(1, self.vocab_size)))

    def _span_positions(self, plan: Dict) -> Optional[List[int]]:
        spans = plan['spans']
        if spans is None:
            return None
        if self.mask_span == 'one':
            pool = [p for p in (spans['mod'], spans['head']) if p]
            if not pool:
                return None
            chosen = pool[int(self.rng.randint(len(pool)))]
            return list(chosen) if chosen else None
        joined = spans['mod'] + spans['head']
        return list(dict.fromkeys(joined)) if joined else None

    def __getitem__(self, idx: int) -> Dict:
        base = self._base[idx]
        plan = self._plan[idx]
        input_ids = base['input_ids'].clone()
        labels = torch.full_like(input_ids, -100)

        positions = self._span_positions(plan)
        if positions:
            # compound-aware: mask (nearly) all span tokens, 80/10/10
            ids = input_ids.tolist()
            for p in positions:
                r = self.rng.rand()
                if r < self.mask_prob:
                    ids[p] = self.tokenizer.mask_token_id
                elif r < self.mask_prob + self.random_prob:
                    ids[p] = self._rand_token_id()
            labels[positions] = base['input_ids'][positions]
            return {
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': base['attention_mask'].clone(),
                'labels': labels,
            }

        # fallback: standard 15% random masking over non-special tokens
        cand = plan['non_special']
        n = max(1, int(round(self.RANDOM_RATIO * len(cand))))
        chosen = cand if len(cand) <= n else list(
            self.rng.choice(cand, size=n, replace=False))
        ids = input_ids.tolist()
        for p in chosen:
            r = self.rng.rand()
            if r < self.mask_prob:
                ids[p] = self.tokenizer.mask_token_id
            elif r < self.mask_prob + self.random_prob:
                ids[p] = self._rand_token_id()
        labels[chosen] = base['input_ids'][chosen]
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': base['attention_mask'].clone(),
            'labels': labels,
        }

    def __len__(self) -> int:
        return len(self._base)


def _mlm_worker_init_fn(_worker_id: int) -> None:
    """Re-seed MlmDataset's RNG per worker so multi-worker DataLoaders do not
    replicate the same mask positions across processes.

    ``get_worker_info()`` hands each worker a unique seed (derived from the
    torch seed + epoch + worker id), reproduced per epoch, so masking stays
    evidence-free and reproducible given a fixed global seed.
    """
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = getattr(info, 'dataset', None)
    if ds is not None and hasattr(ds, 'rng'):
        # torch worker seeds are 64-bit; numpy RandomState caps at 2**32-1
        ds.rng = np.random.RandomState(int(info.seed) & 0xFFFFFFFF)


# --------------------------------------------------------------------------- #
# collate helpers (pad to max length within the batch)
# --------------------------------------------------------------------------- #
def collate_comp(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    seq_keys = ('input_ids', 'attention_mask', 'mod_span_mask', 'head_span_mask')
    for key in batch[0]:
        if key in seq_keys:
            length = max(int(b[key].size(0)) for b in batch)
            out[key] = torch.zeros(len(batch), length, dtype=batch[0][key].dtype)
            for i, b in enumerate(batch):
                n = int(b[key].size(0))
                out[key][i, :n] = b[key]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out


def collate_mlm(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ('input_ids', 'attention_mask', 'labels'):
        length = max(int(b[key].size(0)) for b in batch)
        # labels: padded positions must be -100 so CrossEntropyLoss skips them
        # (in-sequence unmasked positions are already -100 from MlmDataset);
        # anything else pads with _PAD_ID.
        pad_val = -100 if key == 'labels' else _PAD_ID
        t = torch.full((len(batch), length), pad_val, dtype=batch[0][key].dtype)
        for i, b in enumerate(batch):
            n = int(b[key].size(0))
            t[i, :n] = b[key]
        out[key] = t
    return out