"""End-to-end pipelines orchestrated by the live config.

Three entry points, one per phase of the workflow:

- `run_train80`  : quick compound-level 80/20 split + single checkpoint.
- `run_train5`   : stratified 5-fold CV + OOF evaluation + per-fold checkpoints.
- `run_predict`  : build a submission from trained checkpoint(s).

Each pipeline writes reproducible artifacts (config, metrics, history) into
the output directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config
from src.constants import MARKER_TOKENS, SCORE_MAX, SCORE_MIN
from src.dataset import NNDataset
from src.folds import prepare_stratified_folds
from src.model import build_model
from src.train import evaluate
from src.trainer import Trainer

_REQUIRED_COLUMNS = ['Mod', 'Head', 'Compound', 'Context', 'ModAvg', 'HeadAvg']
_ID_CANDIDATES = ('tID', 'ContextID')

TRAIN_FILE = 'dataset/en-nn-train.tsv'
TRIAL_FILE = 'trial/en-nn-trial.tsv'


# --------------------------------------------------------------------------- #
# shared I/O helpers
# --------------------------------------------------------------------------- #
def load_training_frame(data_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    path = Path(data_dir) / TRAIN_FILE
    if not path.is_file():
        raise FileNotFoundError(f'Training data not found at {path}')
    df = pd.read_csv(path, sep='\t')
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in {path}: {missing}')
    logger.info('Loaded %s: %d rows, %d compounds', path, len(df), df['Compound'].nunique())
    return df


def load_trial_frame(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / TRIAL_FILE
    if not path.is_file():
        raise FileNotFoundError(f'Trial data not found at {path}')
    return pd.read_csv(path, sep='\t')


def load_tokenizer(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenizer.add_special_tokens({'additional_special_tokens': MARKER_TOKENS})
    return tokenizer


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --------------------------------------------------------------------------- #
# train80 - single compound-level 80/20 split
# --------------------------------------------------------------------------- #
def run_train80(cfg: Config, logger: logging.Logger, device, data_dir: Path,
                output_dir: Path) -> Dict[str, float]:
    logger.info('=== MODE: train80 (compound-level 80/20 split) ===')

    df = load_training_frame(data_dir, logger)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    train_idx, val_idx = next(gss.split(df, groups=df['Compound']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df['Compound']) & set(val_df['Compound'])
    logger.info('Train %d rows | Val %d rows | compound overlap %d',
                len(train_df), len(val_df), len(overlap))

    tokenizer = load_tokenizer(cfg)
    trainer = Trainer(cfg, device, logger, output_dir)
    result = trainer.fit(train_df, val_df, tokenizer, fold=None, ckpt_name='best.pt')

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

    logger.info('=== RESULTS (best %s) ===', metrics['best_epoch'])
    logger.info('Val Mod  ρ %.4f / RMSE %.4f', metrics['val_rho_mod'], metrics['val_rmse_mod'])
    logger.info('Val Head ρ %.4f / RMSE %.4f', metrics['val_rho_head'], metrics['val_rmse_head'])
    logger.info('Val Mean ρ %.4f', metrics['val_rho_mean'])
    return metrics


# --------------------------------------------------------------------------- #
# train5 - stratified compound-grouped K-fold CV
# --------------------------------------------------------------------------- #
def run_train5(cfg: Config, logger: logging.Logger, device, data_dir: Path,
               output_dir: Path) -> Dict[str, float]:
    logger.info('=== MODE: train5 (stratified %d-fold CV, compounds never leaked) ===',
                cfg.n_splits)

    df = load_training_frame(data_dir, logger)
    df = prepare_stratified_folds(df, target_col='Compound',
                                  n_splits=cfg.n_splits, seed=cfg.seed)

    tokenizer = load_tokenizer(cfg)
    trainer = Trainer(cfg, device, logger, output_dir)

    oof_mod = np.zeros(len(df))
    oof_head = np.zeros(len(df))
    completed: List[int] = []
    fold_results: List[Dict] = []
    all_history: List[Dict] = []

    for fold in range(cfg.n_splits):
        logger.info('=' * 60)
        logger.info('FOLD %d', fold)
        logger.info('=' * 60)

        train_df = df[df['fold'] != fold].reset_index(drop=True)
        val_df = df[df['fold'] == fold].reset_index(drop=True)

        result = trainer.fit(train_df, val_df, tokenizer, fold=fold,
                             ckpt_name=f'fold{fold}_best.pt')
        all_history.append({'fold': fold, 'history': result.history})

        val_indices = df[df['fold'] == fold].index.values
        oof_mod[val_indices] = result.best_mod_pred
        oof_head[val_indices] = result.best_head_pred
        completed.append(fold)

        fold_results.append({
            'fold': fold,
            'rho_mod': round(result.rho_mod, 5),
            'rho_head': round(result.rho_head, 5),
            'rho_mean': round(result.rho_mean, 5),
            'best_epoch': result.best_epoch,
        })

        active = df['fold'].isin(completed)
        real_oof_mod = float(spearmanr(df.loc[active, 'ModAvg'], oof_mod[active]).statistic)
        real_oof_head = float(spearmanr(df.loc[active, 'HeadAvg'], oof_head[active]).statistic)
        logger.info('>>> Fold %d done | best Mean ρ %.4f (epoch %d)',
                    fold, result.rho_mean, result.best_epoch)
        logger.info('>>> REAL OOF (folds %s) Mean ρ %.4f',
                    completed, (real_oof_mod + real_oof_head) / 2)

        torch.cuda.empty_cache()

    oof_rho_mod = float(spearmanr(df['ModAvg'], oof_mod).statistic)
    oof_rho_head = float(spearmanr(df['HeadAvg'], oof_head).statistic)
    oof_rho_mean = (oof_rho_mod + oof_rho_head) / 2.0

    metrics = {
        'mode': cfg.mode,
        'n_splits': cfg.n_splits,
        'oof_rho_mod': round(oof_rho_mod, 5),
        'oof_rho_head': round(oof_rho_head, 5),
        'oof_rho_mean': round(oof_rho_mean, 5),
        'oof_rmse_mod': round(_rmse(df['ModAvg'], oof_mod), 5),
        'oof_rmse_head': round(_rmse(df['HeadAvg'], oof_head), 5),
        'fold_results': fold_results,
    }
    _write_json(metrics, output_dir / 'metrics.json')
    _write_json(all_history, output_dir / 'history.json')
    np.savez(output_dir / 'oof_predictions.npz',
             oof_mod=oof_mod, oof_head=oof_head,
             labels_mod=df['ModAvg'].to_numpy(), labels_head=df['HeadAvg'].to_numpy())

    logger.info('=== PER-FOLD RESULTS ===')
    for row in fold_results:
        logger.info('fold %d | Mod ρ %.4f | Head ρ %.4f | Mean ρ %.4f',
                    row['fold'], row['rho_mod'], row['rho_head'], row['rho_mean'])
    logger.info('=== OVERALL OOF ===')
    logger.info('OOF Mod ρ %.4f | OOF Head ρ %.4f | OOF Mean ρ %.4f',
                oof_rho_mod, oof_rho_head, oof_rho_mean)
    return metrics


# --------------------------------------------------------------------------- #
# predict - trial predictions + submission file
# --------------------------------------------------------------------------- #
def run_predict(cfg: Config, logger: logging.Logger, device, data_dir: Path,
                output_dir: Path) -> Dict[str, float]:
    logger.info('=== MODE: predict (trial + submission) ===')

    df_trial = load_trial_frame(data_dir)
    tokenizer = load_tokenizer(cfg)

    trial_dataset = NNDataset(df_trial, tokenizer, cfg.max_length,
                              cfg.max_context_length, is_test=True)
    trial_loader = DataLoader(trial_dataset, batch_size=cfg.batch_size * 2,
                              shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=True)

    def _predict(ckpt: str):
        model = build_model(cfg, tokenizer, device, dropout=0.0)
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
                raise FileNotFoundError(f'Missing {ckpt} - run --mode train5 first')
            pm, ph = _predict(str(ckpt))
            mod_preds.append(pm)
            head_preds.append(ph)
            logger.info('  Ensembled fold %d', fold)
            if device.type == 'cuda':
                torch.cuda.empty_cache()   # release the fold model before the next load
        trial_pred_mod = np.mean(mod_preds, axis=0)
        trial_pred_head = np.mean(head_preds, axis=0)
    else:
        ckpt = output_dir / 'models' / 'best.pt'
        if not ckpt.is_file():
            raise FileNotFoundError(f'Missing {ckpt} - run --mode train80 first')
        trial_pred_mod, trial_pred_head = _predict(str(ckpt))

    trial_pred_mod = np.clip(trial_pred_mod, SCORE_MIN, SCORE_MAX)
    trial_pred_head = np.clip(trial_pred_head, SCORE_MIN, SCORE_MAX)

    trial_rho_mod = float(spearmanr(df_trial['ModAvg'], trial_pred_mod).statistic)
    trial_rho_head = float(spearmanr(df_trial['HeadAvg'], trial_pred_head).statistic)
    metrics = {
        'mode': cfg.mode,
        'predict_mode': cfg.predict_mode,
        'trial_rho_mod': round(trial_rho_mod, 5),
        'trial_rho_head': round(trial_rho_head, 5),
        'trial_rho_mean': round((trial_rho_mod + trial_rho_head) / 2, 5),
        'trial_rmse_mod': round(_rmse(df_trial['ModAvg'], trial_pred_mod), 5),
        'trial_rmse_head': round(_rmse(df_trial['HeadAvg'], trial_pred_head), 5),
    }
    _write_json(metrics, output_dir / 'trial_metrics.json')

    logger.info('=== TRIAL RESULTS ===')
    logger.info('Mod ρ %.4f / RMSE %.4f', metrics['trial_rho_mod'], metrics['trial_rmse_mod'])
    logger.info('Head ρ %.4f / RMSE %.4f', metrics['trial_rho_head'], metrics['trial_rmse_head'])
    logger.info('Mean ρ %.4f', metrics['trial_rho_mean'])

    id_col = next((c for c in _ID_CANDIDATES if c in df_trial.columns), df_trial.columns[0])
    submission = pd.DataFrame({
        'tID': df_trial[id_col],
        'Modifier': trial_pred_mod,
        'Head': trial_pred_head,
    })
    sub_path = output_dir / 'submission' / 'en-nn-trial-pred.tsv'
    submission.to_csv(sub_path, sep='\t', index=False, header=False)
    logger.info('Submission written: %s (%d rows, no header)', sub_path, len(submission))
    return metrics