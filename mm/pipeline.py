"""End-to-end pipelines orchestrated by `run.py`.

Build status:
  - M2 DONE: data layer (marker-free span alignment) + ``run_probe``.
  - M3: model + losses + trainer + warmup/train80/train5/predict wired.
  - M4: probes + A/B grid + submission (predict already produces the TSV).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from mm.config import Config


def _not_built(name: str) -> Dict[str, float]:
    raise NotImplementedError(f'{name} is not wired yet (M4).')


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _load_all(cfg: Config, logger: logging.Logger) -> List[Dict]:
    from mm.data import load_aux, load_labeled
    rows = load_labeled(cfg)
    if cfg.aux_data_paths:
        rows += load_aux(cfg)
    return rows


def _resolve_trial_path(cfg: Config, data_dir: Path) -> Path:
    """Find the trial file wherever it lives relative to the data dir.

    Kaggle layout mounts the repo with ``dataset/`` (data) and ``trial/``
    (trial rows) as siblings; locally the trial may sit directly in data_dir.
    """
    cands = [
        data_dir / 'trial' / cfg.trial_file,
        data_dir.parent / 'trial' / cfg.trial_file,
        data_dir / cfg.trial_file,
    ]
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f'Trial data not found (looked in {[str(c) for c in cands]})')


def _load_trial(cfg: Config, data_dir: Path) -> List[Dict]:
    from mm.data import _df_to_rows, read_tsv
    path = _resolve_trial_path(cfg, data_dir)
    rows = _df_to_rows(read_tsv(path.as_posix()), path.name, 'en')
    return rows


def _tokenizer(cfg: Config, logger: logging.Logger):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    return tokenizer


def _warmup_snapshot(output_dir: Path) -> Optional[Path]:
    p = Path(output_dir) / 'models' / 'warmup_merged.pt'
    return p if p.is_file() else None


# --------------------------------------------------------------------------- #
# phase 0: compound-aware MLM warmup + LoRA merge
# --------------------------------------------------------------------------- #
def run_warmup(cfg: Config, logger: logging.Logger, device,
               data_dir: Path, output_dir: Path) -> Dict[str, float]:
    if cfg.warmup_mlm_epochs <= 0:
        logger.info('warmup disabled (warmup_mlm_epochs=0)')
        return {'warmup_epochs': 0}

    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup as _linearsched

    from mm.data import MlmDataset, _mlm_worker_init_fn, collate_mlm, load_mlm_rows
    from mm.model import _backbone, apply_lora, build_model, lora_parameters, merge_lora

    logger.info('=== warmup: compound-aware MLM (epochs=%d, mask_span=%s) ===',
                cfg.warmup_mlm_epochs, cfg.mlm_mask_span)

    tokenizer = _tokenizer(cfg, logger)
    rows = load_mlm_rows(cfg)
    mds = MlmDataset(
        rows, tokenizer, max_len=cfg.max_mlm_length, mask_span=cfg.mlm_mask_span,
        mask_prob=cfg.mlm_mask_prob, random_prob=cfg.mlm_random_prob,
        seed=cfg.seed, vocab_size=getattr(tokenizer, 'vocab_size', None),
    )
    loader = DataLoader(mds, batch_size=cfg.warmup_batch_size, shuffle=True,
                        num_workers=cfg.num_workers, collate_fn=collate_mlm,
                        worker_init_fn=_mlm_worker_init_fn)
    logger.info('warmup rows: %d (%d batches/epoch)', len(rows), len(loader))

    model = build_model(cfg, device)
    adapters = apply_lora(model, rank=cfg.warmup_lora_rank,
                          alpha=cfg.warmup_lora_alpha,
                          dropout=cfg.warmup_lora_dropout,
                          targets=cfg.lora_targets)
    wpaths = getattr(model, '_lora_paths', [])
    logger.info('warmup LoRA matched %d modules, e.g. %s',
                len(wpaths), wpaths[:3])

    # freeze everything except LoRA + the pretrained MLM head
    base_ids = {id(p) for p in _backbone(model).parameters()}
    mlm_head = [p for n, p in model.lm.named_parameters() if id(p) not in base_ids]
    for p in model.parameters():
        p.requires_grad = False
    for p in lora_parameters(adapters):
        p.requires_grad = True
    for p in mlm_head:
        p.requires_grad = True
    logger.info('warmup trainable: %d LoRA + %d MLM-head params',
                len(lora_parameters(adapters)), len(mlm_head))

    from torch.optim import AdamW
    optimizer = AdamW([
        {'params': lora_parameters(adapters) + mlm_head,
         'lr': cfg.warmup_lr, 'weight_decay': 0.01},
    ])
    from mm.train import track_optimizer_steps
    track_optimizer_steps(optimizer)
    steps = len(loader) * cfg.warmup_mlm_epochs
    scheduler = _linearsched(optimizer, num_warmup_steps=int(steps * cfg.warmup_warmup_ratio),
                             num_training_steps=steps)

    from torch.amp import GradScaler
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'),
                        init_scale=cfg.amp_init_scale,
                        growth_interval=cfg.amp_growth_interval)
    criterion = nn.CrossEntropyLoss()

    from mm.train import warmup_epoch
    for epoch in range(1, cfg.warmup_mlm_epochs + 1):
        loss = warmup_epoch(model, loader, optimizer, scheduler, criterion, scaler,
                            device, grad_clip=cfg.grad_clip,
                            accum_steps=cfg.accum_steps)
        logger.info('warmup epoch %d/%d | MLM loss %.4f', epoch,
                    cfg.warmup_mlm_epochs, loss)

    merge_lora(model, adapters)
    out = Path(output_dir) / 'models' / 'warmup_merged.pt'
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    logger.info('warmup LoRA merged + snapshot saved: %s', out)
    return {'warmup_epochs': cfg.warmup_mlm_epochs}


# --------------------------------------------------------------------------- #
# train80 - single compound-level 80/20 split
# --------------------------------------------------------------------------- #
def run_train80(cfg: Config, logger: logging.Logger, device,
                data_dir: Path, output_dir: Path) -> Dict[str, float]:
    logger.info('=== MODE: train80 (compound-level 80/20 split) ===')
    rows = _load_all(cfg, logger)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    compounds = [r['compound'] for r in rows]
    tr, va = next(gss.split(rows, groups=compounds))
    train_rows = [rows[i] for i in tr]
    val_rows = [rows[i] for i in va]
    overlap = set(r['compound'] for r in train_rows) & set(r['compound'] for r in val_rows)
    logger.info('Train %d rows | Val %d rows | compound overlap %d',
                len(train_rows), len(val_rows), len(overlap))

    tokenizer = _tokenizer(cfg, logger)
    from mm.trainer import Trainer
    trainer = Trainer(cfg, device, logger, output_dir)
    result = trainer.fit(train_rows, val_rows, tokenizer, fold=None,
                         ckpt_name='best.pt', load_from=_warmup_snapshot(output_dir))

    metrics = {
        'mode': cfg.mode,
        'val_rho_mod': round(result.rho_mod, 5),
        'val_rho_head': round(result.rho_head, 5),
        'val_rho_mean': round(result.rho_mean, 5),
        'val_rmse_mod': round(_rmse(result.best_mod_label, result.best_mod_pred), 5),
        'val_rmse_head': round(_rmse(result.best_head_label, result.best_head_pred), 5),
        'best_epoch': result.best_epoch,
        'checkpoint': result.ckpt_path,
    }
    _write_json(metrics, output_dir / 'metrics.json')
    _write_json(result.history, output_dir / 'history.json')
    logger.info('=== RESULTS (best epoch %s) ===', metrics['best_epoch'])
    logger.info('Val Mod  ρ %.4f / RMSE %.4f', metrics['val_rho_mod'], metrics['val_rmse_mod'])
    logger.info('Val Head ρ %.4f / RMSE %.4f', metrics['val_rho_head'], metrics['val_rmse_head'])
    logger.info('Val Mean ρ %.4f', metrics['val_rho_mean'])
    return metrics


# --------------------------------------------------------------------------- #
# train5 - stratified compound-grouped K-fold CV
# --------------------------------------------------------------------------- #
def run_train5(cfg: Config, logger: logging.Logger, device,
               data_dir: Path, output_dir: Path) -> Dict[str, float]:
    logger.info('=== MODE: train5 (stratified %d-fold CV, compounds never leaked) ===',
                cfg.n_splits)
    rows = _load_all(cfg, logger)
    from mm.folds import assign_folds
    rows = assign_folds(rows, n_splits=cfg.n_splits, seed=cfg.seed)

    tokenizer = _tokenizer(cfg, logger)
    from mm.trainer import Trainer
    trainer = Trainer(cfg, device, logger, output_dir)
    load_from = _warmup_snapshot(output_dir)

    n_rows = len(rows)
    oof_mod = np.zeros(n_rows)
    oof_head = np.zeros(n_rows)
    completed: List[int] = []
    fold_results: List[Dict] = []
    all_history: List[Dict] = []

    for fold in range(cfg.n_splits):
        logger.info('=' * 60)
        logger.info('FOLD %d', fold)
        logger.info('=' * 60)
        val_rows = [r for r in rows if r['fold'] == fold]
        train_rows = [r for r in rows if r['fold'] != fold]

        result = trainer.fit(train_rows, val_rows, tokenizer, fold=fold,
                             ckpt_name=f'fold{fold}_best.pt', load_from=load_from)
        all_history.append({'fold': fold, 'history': result.history})

        positions = [i for i, r in enumerate(rows) if r['fold'] == fold]
        oof_mod[positions] = result.best_mod_pred
        oof_head[positions] = result.best_head_pred
        completed.append(fold)

        fold_results.append({
            'fold': fold, 'rho_mod': round(result.rho_mod, 5),
            'rho_head': round(result.rho_head, 5),
            'rho_mean': round(result.rho_mean, 5),
            'best_epoch': result.best_epoch,
        })
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    labeled = [i for i, r in enumerate(rows) if r['has_label']]
    mod_y = np.array([rows[i]['mod_avg'] for i in labeled])
    head_y = np.array([rows[i]['head_avg'] for i in labeled])
    oof_rho_mod = float(spearmanr(mod_y, oof_mod[labeled]).statistic)
    oof_rho_head = float(spearmanr(head_y, oof_head[labeled]).statistic)
    oof_rho_mean = (oof_rho_mod + oof_rho_head) / 2.0

    metrics = {
        'mode': cfg.mode, 'n_splits': cfg.n_splits,
        'oof_rho_mod': round(oof_rho_mod, 5),
        'oof_rho_head': round(oof_rho_head, 5),
        'oof_rho_mean': round(oof_rho_mean, 5),
        'oof_rmse_mod': round(_rmse(mod_y, oof_mod[labeled]), 5),
        'oof_rmse_head': round(_rmse(head_y, oof_head[labeled]), 5),
        'fold_results': fold_results,
    }
    _write_json(metrics, output_dir / 'metrics.json')
    _write_json(all_history, output_dir / 'history.json')
    np.savez(output_dir / 'oof_predictions.npz',
             oof_mod=oof_mod, oof_head=oof_head,
             labels_mod=mod_y, labels_head=head_y)

    logger.info('=== PER-FOLD RESULTS ===')
    for row in fold_results:
        logger.info('fold %d | Mod ρ %.4f | Head ρ %.4f | Mean ρ %.4f',
                    row['fold'], row['rho_mod'], row['rho_head'], row['rho_mean'])
    logger.info('=== OVERALL OOF ===')
    logger.info('OOF Mod ρ %.4f | OOF Head ρ %.4f | OOF Mean ρ %.4f',
                oof_rho_mod, oof_rho_head, oof_rho_mean)
    return {k: metrics[k] for k in ('oof_rho_mean', 'oof_rho_mod', 'oof_rho_head')}


# --------------------------------------------------------------------------- #
# predict - trial predictions + submission file
# --------------------------------------------------------------------------- #
def run_predict(cfg: Config, logger: logging.Logger, device,
                data_dir: Path, output_dir: Path) -> Dict[str, float]:
    logger.info('=== MODE: predict (trial + submission) ===')
    from torch.utils.data import DataLoader

    from mm.constants import SCORE_MAX, SCORE_MIN
    from mm.data import CompDataset, collate_comp
    from mm.model import apply_lora, build_model
    from mm.train import evaluate

    trial_rows = _load_trial(cfg, data_dir)
    tokenizer = _tokenizer(cfg, logger)
    trial_ds = CompDataset(trial_rows, tokenizer,
                           max_len=cfg.max_context_length, is_test=True)
    trial_loader = DataLoader(trial_ds, batch_size=cfg.batch_size * 2,
                              shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=True, collate_fn=collate_comp)

    load_from = _warmup_snapshot(output_dir)

    def _predict(ckpt: str):
        model = build_model(cfg, device, load_from=load_from)
        adapters = apply_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
                              dropout=cfg.lora_dropout, targets=cfg.lora_targets)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        mod_pred, head_pred = evaluate(model, trial_loader, device)
        return mod_pred, head_pred

    if cfg.predict_mode == '5fold':
        mod_preds, head_preds = [], []
        for fold in range(cfg.n_splits):
            ckpt = output_dir / 'models' / f'fold{fold}_best.pt'
            if not ckpt.is_file():
                raise FileNotFoundError(f'Missing {ckpt} - run train5 first')
            pm, ph = _predict(str(ckpt))
            mod_preds.append(pm)
            head_preds.append(ph)
            logger.info('  Ensembled fold %d', fold)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        trial_pred_mod = np.mean(mod_preds, axis=0)
        trial_pred_head = np.mean(head_preds, axis=0)
    else:
        ckpt = output_dir / 'models' / 'best.pt'
        if not ckpt.is_file():
            raise FileNotFoundError(f'Missing {ckpt} - run train80 first')
        trial_pred_mod, trial_pred_head = _predict(str(ckpt))

    trial_pred_mod = np.clip(trial_pred_mod, SCORE_MIN, SCORE_MAX)
    trial_pred_head = np.clip(trial_pred_head, SCORE_MIN, SCORE_MAX)

    labeled_idx = [i for i, r in enumerate(trial_rows) if r['has_label']]
    metrics = {'mode': cfg.mode, 'predict_mode': cfg.predict_mode}
    if labeled_idx:
        mod_y = np.array([trial_rows[i]['mod_avg'] for i in labeled_idx])
        head_y = np.array([trial_rows[i]['head_avg'] for i in labeled_idx])
        metrics['trial_rho_mod'] = round(float(spearmanr(
            mod_y, trial_pred_mod[labeled_idx]).statistic), 5)
        metrics['trial_rho_head'] = round(float(spearmanr(
            head_y, trial_pred_head[labeled_idx]).statistic), 5)
        metrics['trial_rho_mean'] = round((metrics['trial_rho_mod'] + metrics['trial_rho_head']) / 2, 5)
        logger.info('=== TRIAL METRICS ===')
        logger.info('Mod ρ %.4f | Head ρ %.4f | Mean ρ %.4f',
                    metrics['trial_rho_mod'], metrics['trial_rho_head'],
                    metrics['trial_rho_mean'])
    _write_json(metrics, output_dir / 'trial_metrics.json')

    tids = [r['context_id'] for r in trial_rows]
    submission = {
        'tID': tids, 'Modifier': trial_pred_mod, 'Head': trial_pred_head,
    }
    import pandas as pd
    sub_df = pd.DataFrame(submission)
    sub_path = output_dir / 'submission' / 'en-nn-trial-pred.tsv'
    sub_df.to_csv(sub_path, sep='\t', index=False, header=False)
    logger.info('Submission written: %s (%d rows, no header)', sub_path, len(sub_df))
    return metrics


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