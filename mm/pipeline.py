"""End-to-end pipelines orchestrated by `run.py`.

Build status:
  - M2: data layer (marker-free span alignment on real tokenizers) +
        ``run_probe`` DONE. Probe verifies we align on the real mmBERT
        tokenizer and that MLM span-masking is sane, on Kaggle.
  - M3: model + losses + trainer (attention pool, LoRA, MLM warmup)
  - M4: probes + whether the A/B grid produces the submission
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from mm.config import Config


def _not_built(name: str) -> Dict[str, float]:
    raise NotImplementedError(
        f'{name} is not wired yet (milestone M3+). Build order: '
        'M2 = mm/data.py + probe, M3 = model/losses/trainer.'
    )


def run_warmup(cfg: Config, logger: logging.Logger, device,
               data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Phase 0: compound-aware MLM warmup on mmBERT with LoRA, then merge."""
    logger.info('=== warmup: compound-aware MLM (epochs=%d) ===', cfg.warmup_mlm_epochs)
    return _not_built('run_warmup')


def run_train80(cfg: Config, logger: logging.Logger, device,
                data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Compound-level 80/20 split, single checkpoint."""
    logger.info('=== train80 (compound-level 80/20 split) ===')
    return _not_built('run_train80')


def run_train5(cfg: Config, logger: logging.Logger, device,
               data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Stratified compound-grouped 5-fold CV + OOF evaluation."""
    logger.info('=== train5 (stratified %d-fold CV) ===', cfg.n_splits)
    return _not_built('run_train5')


def run_predict(cfg: Config, logger: logging.Logger, device,
                data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Predict the trial split from checkpoint(s) + write the submission."""
    logger.info('=== predict (trial + submission) ===')
    return _not_built('run_predict')


# --------------------------------------------------------------------------- #
# M2 probe: marker-free alignment + MLM masking on the REAL tokenizer.
# Runs only on Kaggle (needs torch + transformers). Exits clean even when the
# alignment stats look rough; it reports instead of asserting.
# --------------------------------------------------------------------------- #
def run_probe(cfg: Config, logger: logging.Logger, device,
              data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Verify data layer on the real tokenizer (no model needed for M2)."""
    logger.info('=== probe: span alignment + MLM masking on the real tokenizer ===')

    from transformers import AutoTokenizer

    from mm.data import CompDataset, MlmDataset, load_aux, load_labeled, load_mlm_rows

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    if tokenizer.mask_token_id is None:
        raise ValueError(f'{cfg.backbone}: tokenizer has no [MASK] token')

    rows = load_labeled(cfg)
    if cfg.aux_data_paths:
        rows += load_aux(cfg)
    align = _probe_alignment(CompDataset(rows, tokenizer, max_len=cfg.max_context_length),
                             rows, tokenizer, logger)

    if cfg.mlm_data_paths:
        mstats = _probe_mlm(
            MlmDataset(
                load_mlm_rows(cfg), tokenizer,
                max_len=cfg.max_mlm_length,
                mask_span=cfg.mlm_mask_span,
                mask_prob=cfg.mlm_mask_prob,
                random_prob=cfg.mlm_random_prob,
                seed=cfg.seed,
                vocab_size=getattr(tokenizer, 'vocab_size', None),
            ),
            tokenizer, logger,
        )
    else:
        mstats = {}

    logger.info('probe summary: aligned=%d%% (mod-only=%d head-only=%d degenerate=%d)',
                align['aligned_pct'], align['mod_only'], align['head_only'],
                align['degenerate'])
    return {**align, **mstats}


def _probe_alignment(ds, rows, tokenizer, logger) -> Dict[str, int]:
    per_lang: Dict[str, Counter] = defaultdict(Counter)
    for row, item in zip(rows, ds.items):
        hm = bool(item['has_mod'] and item['has_head'])
        c = per_lang[row['lang']]
        c['total'] += 1
        if hm:
            c['aligned'] += 1
            if bool(item['degenerate']):
                c['degenerate'] += 1
        if bool(item['has_mod']):
            c['mod_only'] += 1
        if bool(item['has_head']):
            c['head_only'] += 1

    totals = Counter()
    for c in per_lang.values():
        totals.update(c)
    per_lang['all'] = totals

    for name, c in per_lang.items():
        t = max(1, c['total'])
        logger.info(
            'alignment %-4s: %3d%% aligned (%d/%d) | mod-only %d | head-only %d | degenerate %d',
            name, int(100 * c['aligned'] / t), c['aligned'], c['total'],
            c['mod_only'], c['head_only'], c['degenerate'])
    logger.info(
        'alignment all : aligned=%d (%.0f%%) degenerate=%d',
        totals['aligned'], 100 * totals['aligned'] / max(1, totals['total']),
        totals['degenerate'])

    # show a few raw tokenized examples so the mask semantics are visible
    for i in range(min(3, len(ds.items))):
        item = ds.items[i]
        toks = tokenizer.convert_ids_to_tokens(item['input_ids'].tolist())
        _show_example(i, toks, item, logger)

    return {
        'aligned_pct': int(100 * totals['aligned'] / max(1, totals['total'])),
        'mod_only': totals['mod_only'],
        'head_only': totals['head_only'],
        'degenerate': totals['degenerate'],
    }


def _show_example(i: int, toks: List[str], item, logger) -> None:
    for role, key in (('mod', 'mod_span_mask'), ('head', 'head_span_mask')):
        ids = [j for j, on in enumerate(item[key].tolist()) if on]
        pieces = [f'[{toks[j]}]' if on else toks[j]
                  for j, on in enumerate(item[key].tolist())]
        logger.info('probe row %d %-4s toks=%s  text=%s', i, role, ids, ' '.join(pieces))


def _probe_mlm(mds, tokenizer, logger) -> Dict[str, int]:
    n = min(200, len(mds))
    span_masked = random_masked = 0
    total_masks = 0
    examples = 0
    for i in range(n):
        item = mds[i]
        masked = (item['labels'] != -100).sum().item()
        total_masks += masked
        is_span = _masks_match_span(item, mds, i)
        if is_span:
            span_masked += 1
        else:
            random_masked += 1
        if examples < 2 and masked > 0:
            toks = tokenizer.convert_ids_to_tokens(item['input_ids'].tolist())
            pos = [j for j, l in enumerate(item['labels'].tolist()) if l != -100]
            logger.info('probe mlm row %d: %d masked @ %s -> %s',
                        i, masked, pos, ' '.join(toks))
            examples += 1
    logger.info('mlm masking: %d span-masked rows, %d random-fallback rows, avg %.1f masks/row',
                span_masked, random_masked, total_masks / max(1, n))
    return {'mlm_span_rows': span_masked, 'mlm_random_rows': random_masked}


def _masks_match_span(item, mds, idx) -> bool:
    """True when masked positions are within the row's span tokens (MlmDataset).

    Covers both 'both' (masked == union) and 'one' (masked == one span, a
    subset of the union). Random-fallback rows have no span plan and return
    False.
    """
    plan = mds._plan[idx]['spans']
    if plan is None:
        return False
    span_pos = set(plan['mod']) | set(plan['head'])
    masked = (item['labels'] != -100).nonzero().flatten().tolist()
    return bool(masked) and set(masked).issubset(span_pos)