"""Trainer: owns the complete fit loop for a single split or fold.

Implements the two-phase schedule (frozen encoder -> gradual unfreeze of the
top layers) with optional early stopping, mixed-precision training, and
faithful per-epoch history so runs are auditable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from transformers import get_linear_schedule_with_warmup

from config import Config
from src.dataset import NNDataset
from src.loss import CombinedLoss
from src.model import build_model, embedding_table
from src.train import evaluate, track_optimizer_steps, train_epoch, unfreeze_top_layers


def _safe_rho(y: np.ndarray, p: np.ndarray) -> float:
    """Spearman rho with a 0.0 fallback for a constant input (scipy returns NaN)."""
    r = float(spearmanr(y, p).statistic) if len(y) > 1 else 0.0
    return 0.0 if r != r else r   # NaN check


@dataclass
class FoldResult:
    """Outcome of a single fit() call (one fold or the 80/20 split)."""

    fold: Optional[int]
    rho_mod: float
    rho_head: float
    rho_mean: float
    best_epoch: int
    ckpt_path: str
    history: List[Dict]
    best_mod_pred: np.ndarray
    best_head_pred: np.ndarray
    best_mod_label: np.ndarray
    best_head_label: np.ndarray


class Trainer:
    """Fits a ModernBERTRegressor on a given train/validation split."""

    def __init__(self, cfg: Config, device, logger: logging.Logger, output_dir: Path):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _build_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer):
        train_ds = NNDataset(
            train_df, tokenizer, self.cfg.max_length, self.cfg.max_context_length
        )
        val_ds = NNDataset(
            val_df, tokenizer, self.cfg.max_length, self.cfg.max_context_length
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size * 2, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
        )
        return train_loader, val_loader

    def _phase1(self, model, steps: int):
        """Only the two regression heads and the word embeddings train; the
        encoder layers stay frozen. The embeddings must be trainable so the new
        random marker tokens start learning immediately."""
        emb_table = embedding_table(model).weight
        emb_table.requires_grad = True

        # build_model starts fully trainable (freeze_bert=False); freeze every
        # bert param except the embedding table so Phase 1 is truly frozen and
        # Phase 2's unfreeze selects the top layers only.
        for name, param in model.named_parameters():
            if param is not emb_table and not name.startswith(('mod_regressor', 'head_regressor')):
                param.requires_grad = False

        head_params = list(model.mod_regressor.parameters()) + list(model.head_regressor.parameters())
        optimizer = AdamW(
            [
                {'params': [emb_table], 'lr': self.cfg.embedding_lr},
                {'params': head_params, 'lr': self.cfg.head_lr},
            ],
            weight_decay=self.cfg.weight_decay,
        )
        track_optimizer_steps(optimizer)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(steps * self.cfg.warmup_ratio),
            num_training_steps=steps,
        )
        return optimizer, scheduler

    def _phase2(self, model, steps: int):
        """Unfrozen top encoder layers at encoder_lr, embeddings at their own
        (lower) LR, heads at head_lr."""
        emb_table = embedding_table(model).weight
        emb_table.requires_grad = True

        encoder_params = [
            p for n, p in model.named_parameters()
            if p is not emb_table
            and 'mod_regressor' not in n and 'head_regressor' not in n
            and p.requires_grad
        ]
        head_params = list(model.mod_regressor.parameters()) + list(model.head_regressor.parameters())
        optimizer = AdamW(
            [
                {'params': [emb_table], 'lr': self.cfg.embedding_lr},
                {'params': encoder_params, 'lr': self.cfg.encoder_lr},
                {'params': head_params, 'lr': self.cfg.head_lr},
            ],
            weight_decay=self.cfg.weight_decay,
        )
        track_optimizer_steps(optimizer)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(steps * self.cfg.warmup_ratio),
            num_training_steps=steps,
        )
        return optimizer, scheduler

    # ------------------------------------------------------------------ #
    # fit loop
    # ------------------------------------------------------------------ #
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer,
            fold: Optional[int] = None, ckpt_name: str = 'best.pt') -> FoldResult:
        train_loader, val_loader = self._build_loaders(train_df, val_df, tokenizer)

        model = build_model(self.cfg, tokenizer, self.device)
        scaler = GradScaler(
            'cuda',
            enabled=(self.device.type == 'cuda'),
            init_scale=self.cfg.amp_init_scale,
            growth_interval=self.cfg.amp_growth_interval,
        )
        criterion = CombinedLoss(
            ccc_weight=self.cfg.ccc_weight,
            lambda_rank=self.cfg.lambda_rank,
            rank_margin=self.cfg.rank_margin,
            ce_weight=self.cfg.ce_weight,
            num_bins=self.cfg.num_bins,
            bin_sigma=self.cfg.bin_sigma,
            use_label_std=self.cfg.use_label_std,
        ).to(self.device)

        # Number of optimizer updates per epoch (fewer than micro-batches when
        # gradient accumulation is on) -- drives the LR schedule length.
        steps_per_epoch = math.ceil(len(train_loader) / self.cfg.accum_steps)

        optimizer, scheduler = self._phase1(
            model, steps_per_epoch * self.cfg.freeze_epochs
        )

        best_rho, best_epoch, no_improve_epochs = -1.0, -1, 0
        best: Optional[Dict[str, np.ndarray]] = None
        history: List[Dict] = []

        for epoch in range(self.cfg.num_epochs):
            if epoch == self.cfg.freeze_epochs:
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()   # release Phase-1 graph buffers before rebuild
                self.logger.info(
                    '>>> Unfreezing top layers (from layer %d) at epoch %d <<<',
                    self.cfg.unfreeze_from_layer, epoch + 1,
                )
                unfreeze_top_layers(model, self.cfg.unfreeze_from_layer)
                optimizer, scheduler = self._phase2(
                    model, steps_per_epoch * (self.cfg.num_epochs - self.cfg.freeze_epochs)
                )

            phase = 'FROZEN' if epoch < self.cfg.freeze_epochs else 'UNFROZEN-TOP'

            diag: Dict[str, float] = {}
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler,
                self.device, grad_clip=self.cfg.grad_clip,
                accum_steps=self.cfg.accum_steps, report=diag,
            )
            mod_pred, head_pred, mod_label, head_label = evaluate(model, val_loader, self.device)

            rho_mod = _safe_rho(mod_label, mod_pred)
            rho_head = _safe_rho(head_label, head_pred)
            rho_mean = (rho_mod + rho_head) / 2.0

            history.append({
                'epoch': epoch + 1,
                'phase': phase,
                'loss': round(float(train_loss), 5),
                'rho_mod': round(rho_mod, 5),
                'rho_head': round(rho_head, 5),
                'rho_mean': round(rho_mean, 5),
                'opt_steps': diag['opt_steps'],
                'skipped': diag['skipped'],
                'grad_norm': round(diag['grad_norm'], 4),
                'group_grads': {k: round(v, 4) for k, v in diag.get('group_grads', {}).items()},
                'scale': round(diag['scale'], 1),
                'lr': diag['lr'],
                'pred_mod_min': round(float(mod_pred.min()), 4),
                'pred_mod_max': round(float(mod_pred.max()), 4),
                'pred_head_min': round(float(head_pred.min()), 4),
                'pred_head_max': round(float(head_pred.max()), 4),
            })
            gg = diag.get('group_grads', {})
            gg_str = ' '.join(f'{k}={v:.1e}' for k, v in sorted(gg.items())) or '(none)'
            ovf = diag.get('overflow', {})
            ovf_str = ' '.join(f'ovf-{k}={v}' for k, v in sorted(ovf.items()))
            if ovf_str:
                gg_str = f'{gg_str} [{ovf_str}]'
            self.logger.info(
                'Epoch %d/%d [%s] | Loss %.4f | Mod ρ %.4f | Head ρ %.4f | Mean ρ %.4f'
                ' | steps %d (skip %d) | grads %s | scale %.1f | lr %.2e'
                ' | pred mod[%.2f,%.2f] head[%.2f,%.2f]',
                epoch + 1, self.cfg.num_epochs, phase, train_loss,
                rho_mod, rho_head, rho_mean,
                diag['opt_steps'], diag['skipped'], gg_str,
                diag['scale'], diag['lr'],
                mod_pred.min(), mod_pred.max(), head_pred.min(), head_pred.max(),
            )

            if rho_mean > best_rho:
                best_rho, best_epoch, no_improve_epochs = rho_mean, epoch + 1, 0
                best = {
                    'mod': mod_pred.copy(),
                    'head': head_pred.copy(),
                    'mod_y': mod_label.copy(),
                    'head_y': head_label.copy(),
                }
                ckpt_dir = self.output_dir / 'models'
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / ckpt_name
                torch.save(model.state_dict(), ckpt_path)
            else:
                if epoch >= self.cfg.freeze_epochs:
                    no_improve_epochs += 1
                    if no_improve_epochs >= self.cfg.patience:
                        self.logger.info(
                            'Early stop at epoch %d (best epoch %d, Mean ρ %.4f)',
                            epoch + 1, best_epoch, best_rho,
                        )
                        break

        if best is None:
            raise RuntimeError(
                'No improvement over any epoch - check the hyperparameters.'
            )

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()   # release last epoch's activations

        best_rho_mod = _safe_rho(best['mod_y'], best['mod'])
        best_rho_head = _safe_rho(best['head_y'], best['head'])
        best_rho_mean = (best_rho_mod + best_rho_head) / 2.0

        self.logger.info(
            'Split %s finished | best Mean ρ %.4f at epoch %d',
            'n/a' if fold is None else fold, best_rho_mean, best_epoch,
        )

        return FoldResult(
            fold=fold,
            rho_mod=best_rho_mod,
            rho_head=best_rho_head,
            rho_mean=best_rho_mean,
            best_epoch=best_epoch,
            ckpt_path=str(ckpt_path),
            history=history,
            best_mod_pred=best['mod'],
            best_head_pred=best['head'],
            best_mod_label=best['mod_y'],
            best_head_label=best['head_y'],
        )