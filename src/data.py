"""Data loading, tokenization and span alignment for the gauss pipeline.

Marker-free design (phase M2): the raw context is tokenized verbatim with
``return_offsets_mapping=True``; the compound's Mod/Head surface forms are
matched inside the original text by ``marks.find_spans`` and mapped onto
token spans. No ``<mod>`` / ``<head>`` markers are inserted, so the pretrained
mmBERT tokenizer/embeddings never see out-of-vocabulary artifacts.

``span_markers=True`` is the marked alternative: the row's own target span is
wrapped with a single unused id that opens AND closes it (ids 7/8/9 for
mod/head/pv, spliced post-tokenization at the span's token boundaries) so the
encoder sees exactly which instance and which role each row answers. Mutual
with ``target_prefix`` and combined-backend only.

``CompDataset`` -- one row per labeled / aux sentence, for scoring. Yields
span masks plus (possibly NaN) soft labels; aux and unaligned rows keep the
row but mark ``has_label`` / ``has_mod`` / ``has_head`` False so losses can
mask them. Rows pre-tokenize and pre-align in the constructor (deterministic,
single pass), which also surfaces a per-source alignment report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .marks import Span, SpanResult, find_spans
from .targets import MARKER_CODE, TARGETS, target_code
from .utils import get_logger

try:  # pragma: no cover - optional feature, torch is required for datasets anyway
    from .static_vec import StaticVec
except ImportError:
    StaticVec = None

_TARGET_CODE = {t: target_code(t) for t in TARGETS}

logger = get_logger('src.data')

try:
    import torch
except ImportError:  # pragma: no cover - datasets need torch; loaders don't
    torch = None


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
    pv = _is_pv(df)
    if not nn and not pv:
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
            'is_pv': pv,
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
def _load_files(cfg, file_attrs: List[str]) -> List[Dict]:
    """Helper nạp và gộp nhiều file TSV (EN/DE, NN/PV) theo cấu hình Config."""
    data_dir, _ = _resolve(cfg)
    
    files_to_load: List[str] = []
    for attr in file_attrs:
        fname = getattr(cfg, attr, None)
        if fname and fname.strip() and fname not in files_to_load:
            files_to_load.append(fname)

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
    """Load all configured NN and PV training datasets (+ train_aux extras)."""
    train_attrs = ['en_nn_train', 'de_nn_train', 'en_pv_train', 'de_pv_train']
    all_rows = _load_files(cfg, train_attrs)
    aux_names = [a.strip() for a in (getattr(cfg, 'train_aux', None) or [])
                 if a and a.strip()] or []
    if aux_names:
        data_dir, _ = _resolve(cfg)
        n_aux = 0
        for fname in aux_names:
            path = _resolve_aux_path(fname, data_dir)
            if path is None:
                logger.warning('Aux dataset file not found (data_dir + repo dataset/): %s', fname)
                continue
            df = read_tsv(path)
            rows = _df_to_rows(df, fname, _auto_lang(fname))
            for r in rows:
                r['is_aux'] = True
            all_rows.extend(rows)
            n_aux += len(rows)
            logger.info('%s: %d aux rows', fname, len(rows))
        if n_aux > 0:
            # aux rows are new compounds: key them in a dedicated id range ABOVE all
            # core ids so the batch/ranking compound grouping can never mix aux and
            # core compounds together under one compound_id.
            core_ids = [r['compound_id'] for r in all_rows[: len(all_rows) - n_aux]]
            base = (max(core_ids) + 1) if core_ids else 0
            aux_keys = [f"{r['lang']}_{r['compound']}" for r in all_rows[len(all_rows) - n_aux:]]
            aux_codes, _ = pd.factorize(pd.Series(aux_keys))
            for r, c in zip(all_rows[len(all_rows) - n_aux:], aux_codes):
                r['compound_id'] = base + int(c)
            logger.info('Loaded %d rows (incl %d aux) across %d core + %d aux compounds',
                        len(all_rows), n_aux, len(core_ids), int(aux_codes.max()) + 1)
        else:
            logger.warning('train_aux configured but NO aux rows loaded (%s); continuing core-only',
                           ', '.join(aux_names))
    return all_rows


def _resolve_aux_path(fname: str, data_dir: Path) -> Optional[Path]:
    """Locate a train_aux file: data_dir first, then the repo/working dataset/ dir.

    On Kaggle the core TSVs are mounted under the dataset input, while built aux
    files live in the repo clone, so a plain ``data_dir / fname`` misses them.
    """
    candidates = [
        data_dir / fname,
        data_dir / 'dataset' / fname,
        data_dir.parent / 'dataset' / fname,
        Path('dataset') / fname,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _find_trial_path(fname: str, data_dir: Path) -> Optional[Path]:
    """Find a trial file in trial/ (sibling of dataset/ on Kaggle or root) or data_dir."""
    candidates = [
        data_dir / fname,
        data_dir / 'trial' / fname,
        data_dir.parent / 'trial' / fname,
        data_dir.parent / fname,
        Path('trial') / fname,
        Path('dataset') / fname,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_trial(cfg) -> Dict[str, List[Dict]]:
    """Load the per-lineage TRIAL (or Test) files -> {key: rows}.

    ``key`` is one of ``en-nn`` / ``en-pv`` / ``de-nn`` / ``de-pv``, read from the
    four ``*_trial`` config attrs. Rows keep the file's row order and get a
    per-lineage ``compound_id`` (paths don't participate in train-time splits).
    """
    data_dir, _ = _resolve(cfg)
    lineage_attrs = (
        ('en-nn', 'en_nn_trial'), ('en-pv', 'en_pv_trial'),
        ('de-nn', 'de_nn_trial'), ('de-pv', 'de_pv_trial'),
    )
    out: Dict[str, List[Dict]] = {}
    for key, attr in lineage_attrs:
        fname = (getattr(cfg, attr, '') or '').strip()
        if not fname:
            continue
        path = _find_trial_path(fname, data_dir)
        if path is None:
            logger.warning('%s file missing, skipping: %s', key, fname)
            continue
        rows = _df_to_rows(read_tsv(path), fname, _auto_lang(fname))
        codes, _ = pd.factorize(pd.Series(
            [f"{r['lang']}_{r['compound']}" for r in rows]))
        for r, c in zip(rows, codes):
            r['compound_id'] = int(c)
        out[key] = rows
        logger.info('%s trial: %d rows from %s', key, len(rows), path)

    return out


def _resolve(cfg):
    from .utils import resolve_paths
    return resolve_paths(cfg)


# --------------------------------------------------------------------------- #
# single-target expansion
# --------------------------------------------------------------------------- #
def expand_targets(rows: List[Dict], targets: List[str]) -> List[Dict]:
    """Expand rows into one raw per active target (single-target design).

    A row is a (source sentence + spans + labels) candidate for the targets
    that actually make sense for its type -- NN rows grade the modifier and
    the head noun separately, PV rows grade only the whole phrasal verb:

        NN row  + 'mod'  => kept   (label ModAvg)
        NN row  + 'head' => kept   (label HeadAvg)
        NN row  + 'pv'   => DROPPED (no overall Avg gold for NN compounds)
        PV row  + 'pv'   => kept   (label Avg)
        PV row  + 'mod'/'head' => DROPPED (PV rows have no per-role gold)

    Each kept row carries a ``'target'`` key; ``has_label`` reflects whether
    the gold for that target is present. An empty ``targets`` list returns the
    rows untouched (joint mode: one row scores all targets at once).
    """
    if not targets:
        return rows

    out: List[Dict] = []
    for r in rows:
        nn = not r.get('is_pv', False)
        for t in targets:
            if (nn and t == 'mod') or (nn and t == 'head'):
                has_label = bool(np.isfinite(r['mod_avg' if t == 'mod' else 'head_avg']))
            elif (not nn) and t == 'pv':
                has_label = bool(np.isfinite(r['mod_avg']))   # Avg lives in mod_avg
            else:
                continue                                       # drop entirely
            c = dict(r)
            c['target'] = t
            c['has_label'] = has_label
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def _span_mask(span: Optional[Span], length: int) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    if span is not None and span.start is not None:
        mask[span.start:span.end] = True
    return mask


def _target_span(res: SpanResult, target: str) -> Optional[Span]:
    """Token span of the target role for ``span_markers`` wrapping.

    mod/head -> their own span; pv -> the whole compound, taken as the union of
    the mod and head spans (mod here for a fused one-token German compound via
    the fallback to whichever role aligned). ``None`` when the target could not
    be aligned, in which case the caller leaves the row unmarked (it is
    representation-only anyway).
    """
    def _ok(s: Optional[Span]) -> bool:
        return s is not None and s.start is not None

    if target == 'mod':
        return res.mod if _ok(res.mod) else None
    if target == 'head':
        return res.head if _ok(res.head) else None
    if target == 'pv':
        m, h = res.mod, res.head
        if _ok(m) and _ok(h):
            return Span(min(m.start, h.start), max(m.end, h.end))
        if _ok(m):
            return m
        return h if _ok(h) else None
    return None


class CompDataset(_DatasetBase):
    """One sentence (tokenized verbatim) per row, for scoring."""

    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 256,
                 is_test: bool = False, target_prefix: bool = False,
                 static_vec: Optional[StaticVec] = None,
                 span_markers: bool = False,
                 proto_stream: bool = False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.target_prefix = target_prefix
        self.static_vec = static_vec
        self.span_markers = span_markers
        self.proto_stream = proto_stream
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

        mod_span_mask = _span_mask(result.mod, length)
        head_span_mask = _span_mask(result.head, length)
        prefix_mask = torch.zeros(length, dtype=torch.bool)

        if self.target_prefix and r.get('target') is not None and r['target'] in MARKER_CODE:
            t = r['target']
            marker_id = MARKER_CODE[t]
            if t == 'mod':
                word = r.get('mod', '')
            elif t == 'head':
                word = r.get('head', '')
            else:
                word = r.get('compound', '')

            word_ids = self.tokenizer.encode(word, add_special_tokens=False) if (
                word and hasattr(self.tokenizer, 'encode')
            ) else []

            prefix_ids = [marker_id] + list(word_ids) + [marker_id]
            k = len(prefix_ids)
            prefix_tok = torch.tensor(prefix_ids, dtype=input_ids.dtype)
            prefix_attn = torch.ones(k, dtype=attention_mask.dtype)
            prefix_span = torch.zeros(k, dtype=torch.bool)
            prefix_word = torch.zeros(k, dtype=torch.bool)
            if word_ids:
                prefix_word[1:1 + len(word_ids)] = True

            input_ids = torch.cat([input_ids[:1], prefix_tok, input_ids[1:]])[:self.max_len]
            attention_mask = torch.cat([attention_mask[:1], prefix_attn, attention_mask[1:]])[:self.max_len]
            mod_span_mask = torch.cat([mod_span_mask[:1], prefix_span, mod_span_mask[1:]])[:self.max_len]
            head_span_mask = torch.cat([head_span_mask[:1], prefix_span, head_span_mask[1:]])[:self.max_len]
            prefix_mask = torch.cat([prefix_mask[:1], prefix_word, prefix_mask[1:]])[:self.max_len]
        elif self.span_markers and r.get('target') is not None and r['target'] in MARKER_CODE:
            # Border markers (ids spliced, NOT strings): wrap the row's own
            # target span with a single unused id that both opens and closes --
            #   mod -> <unused0>..<unused0>, head -> <unused1>..<unused1>,
            #   pv -> <unused2>..<unused2> around the whole compound (the union
            #   of the mod/head spans, or just the found role when fused).
            # The id itself is the role, so target_prefix stays off. mmBERT's
            # tokenizer cannot produce these ids from '<unusedN>' strings, so
            # we splice the numeric ids post-tokenization (same mechanism as
            # the target_prefix insert, generalized to any span boundary).
            wrap = _target_span(result, r['target'])
            if wrap is not None and wrap.start is not None:
                marker_id = MARKER_CODE[r['target']]
                def _splice(t, pos, fill):
                    return torch.cat([t[:pos], torch.tensor([fill], dtype=t.dtype), t[pos:]])
                # close FIRST (higher index), then open, so the open insert
                # does not move the close boundary
                for pos in (wrap.end, wrap.start):
                    input_ids = _splice(input_ids, pos, marker_id)
                    attention_mask = _splice(attention_mask, pos, 1)
                    mod_span_mask = _splice(mod_span_mask, pos, False)
                    head_span_mask = _splice(head_span_mask, pos, False)
                    prefix_mask = _splice(prefix_mask, pos, False)
                input_ids = input_ids[:self.max_len]
                attention_mask = attention_mask[:self.max_len]
                mod_span_mask = mod_span_mask[:self.max_len]
                head_span_mask = head_span_mask[:self.max_len]
                prefix_mask = prefix_mask[:self.max_len]

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'mod_span_mask': mod_span_mask,
            'head_span_mask': head_span_mask,
            'prefix_mask': prefix_mask,
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
            'is_pv': torch.tensor(bool(r.get('is_pv', False)), dtype=torch.bool),
        }
        if self.static_vec is not None:
            # External static anchors for the model_combined readout: the
            # modifier/head/whole-compound SURFACE forms as (dim,) vectors
            # (zero on OOV). PV rows are anchored on the WHOLE compound's own
            # vector -- e.g. the fastText subword vector of a fused German
            # "abgehauen" -- and only fall back to the mean of base+particle
            # when the compound itself is OOV. (NN rows never route to the pv
            # target, so their pv_static is inert.)
            mod_v = self.static_vec.tensor(r.get('mod', ''))
            head_v = self.static_vec.tensor(r.get('head', ''))
            item['mod_static'] = mod_v
            item['head_static'] = head_v
            compound_v = self.static_vec.tensor(r.get('compound', ''))
            if compound_v.norm() > 0:
                pv_static = compound_v
            else:
                pv_static = 0.5 * (mod_v + head_v)
                n = float(pv_static.norm())
                if n > 0:                  # keep the anchor unit-length
                    pv_static = pv_static / n
            item['pv_static'] = pv_static
        if self.proto_stream:
            # Stream 1: Tokenize the isolated target word for dynamic prototype representation
            t = r.get('target')
            if t == 'mod':
                word = r.get('mod', '')
            elif t == 'head':
                word = r.get('head', '')
            elif t == 'pv':
                word = r.get('compound', '')
            else:
                word = r.get('mod', '') or r.get('compound', '')
            p_enc = self.tokenizer(
                word, max_length=16, truncation=True, return_tensors='pt'
            ) if (word and hasattr(self.tokenizer, '__call__')) else None
            if p_enc is not None:
                item['proto_ids'] = p_enc['input_ids'].squeeze(0)
                item['proto_mask'] = p_enc['attention_mask'].squeeze(0)
            else:
                item['proto_ids'] = torch.zeros(1, dtype=input_ids.dtype)
                item['proto_mask'] = torch.zeros(1, dtype=attention_mask.dtype)
        if r.get('target') is not None:
            item['target'] = torch.tensor(_TARGET_CODE[r['target']], dtype=torch.long)
        return item

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


# --------------------------------------------------------------------------- #
# collate helpers (pad to max length within the batch)
# --------------------------------------------------------------------------- #
def collate_comp(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    seq_keys = ('input_ids', 'attention_mask', 'mod_span_mask', 'head_span_mask',
                'prefix_mask', 'proto_ids', 'proto_mask')
    for key in batch[0]:
        if key in seq_keys:
            length = max(int(b[key].size(0)) for b in batch)
            fill = pad_token_id if key in ('input_ids', 'proto_ids') else 0
            out[key] = torch.full((len(batch), length), fill_value=fill,
                                  dtype=batch[0][key].dtype)
            for i, b in enumerate(batch):
                n = int(b[key].size(0))
                out[key][i, :n] = b[key]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out
