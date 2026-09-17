"""End-to-end pipelines orchestrated by ``src.run``.

train80: single compound-level 80/20 split, trained with the gauss-only
architecture (no reg/softmax heads, no MLM warmup phase).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from src.config import Config


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _load_all(cfg: Config, logger: logging.Logger) -> List[Dict]:
    from src.data import load_aux, load_labeled
    rows = load_labeled(cfg)
    if cfg.aux_data_paths:
        aux_rows = load_aux(cfg)
        rows.extend(aux_rows)
    return rows


def _tokenizer(cfg: Config, logger: logging.Logger):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.backbone)


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
    from src.trainer import Trainer
    trainer = Trainer(cfg, device, logger, output_dir)
    result = trainer.fit(train_rows, val_rows, tokenizer, fold=None,
                         ckpt_name='best.pt')

    val_my = result.best_mod_label
    val_hy = result.best_head_label
    val_mp = result.best_mod_pred
    val_hp = result.best_head_pred

    metrics = {
        'mode': cfg.mode,
        'val_rho_mod': round(result.rho_mod, 5),
        'val_rho_head': round(result.rho_head, 5),
        'val_rho_pv': round(result.rho_pv, 5),
        'val_rho_mean': round(result.rho_mean, 5),
        'val_rmse_mod': round(_rmse(val_my, val_mp), 5) if len(val_my) > 0 else float('nan'),
        'val_rmse_head': round(_rmse(val_hy, val_hp), 5) if len(val_hy) > 0 else float('nan'),
        'best_epoch': result.best_epoch,
        'checkpoint': result.ckpt_path,
    }
    _write_json(metrics, output_dir / 'metrics.json')
    _write_json(result.history, output_dir / 'history.json')
    logger.info('=== RESULTS (best epoch %s) ===', metrics['best_epoch'])
    logger.info('Val Mod  ρ %.4f / RMSE %.4f', metrics['val_rho_mod'], metrics['val_rmse_mod'])
    logger.info('Val Head ρ %.4f / RMSE %.4f', metrics['val_rho_head'], metrics['val_rmse_head'])
    if result.rho_pv > 0:
        logger.info('Val PV   ρ %.4f', metrics['val_rho_pv'])
    logger.info('Val Mean ρ %.4f', metrics['val_rho_mean'])
    return metrics