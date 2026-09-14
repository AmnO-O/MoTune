"""Owns the full scoring fit loop for one split/fold (ported from src/trainer.py).

Adapted for the mmBERT rebuild:
  * LoRA adapter is applied to the backbone ONCE per fit; the adapter weights
    live in the same state_dict (so fold checkpoints are self-contained).
  * Phase 1 = frozen encoder + LoRA + heads; Phase 2 additionally unfreezes
    top ``unfreeze_from_layer..last``. Embeddings train only if
    ``embedding_lr > 0`` (mmBERT table ~197M, default frozen).
  * Rows that did not align (has_mod/has_head False) or German-closed
    degenerate rows are excluded from span supervision inside train_epoch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from mm.config import Config
from mm.data import CompDataset, collate_comp
from mm.folds import CompoundGroupSampler
from mm.losses import CombinedLoss
from mm.model import apply_lora, build_model, lora_parameters, _backbone_embeddings
from mm.train import evaluate, track_optimizer_steps, train_epoch, unfreeze_top_layers


def _safe_rho(y: np.ndarray, p: np.ndarray) -> float:
    r = float(spearmanr(y, p).statistic) if len(y) > 1 else 0.0
    return 0.0 if r != r else r


@dataclass
class FoldResult:
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
    def __init__(self, cfg: Config, device, logger: logging.Logger, output_dir: Path):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------ #
    def _build_loaders(self, train_rows, val_rows, tokenizer):
        train_ds = CompDataset(train_rows, tokenizer,
                               max_len=self.cfg.max_context_length)
        val_ds = CompDataset(val_rows, tokenizer,
                             max_len=self.cfg.max_context_length)
        if self.cfg.lambda_rank > 0:
            cids = [r['compound_id'] for r in train_rows]
            self._train_sampler = CompoundGroupSampler(
                cids, batch_size=self.cfg.batch_size, seed=self.cfg.seed)
            train_loader = DataLoader(
                train_ds, batch_size=self.cfg.batch_size, shuffle=False,
                sampler=self._train_sampler, num_workers=self.cfg.num_workers,
                pin_memory=True, collate_fn=collate_comp)
        else:
            self._train_sampler = None
            train_loader = DataLoader(
                train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                num_workers=self.cfg.num_workers, pin_memory=True,
                collate_fn=collate_comp)
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size * 2, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=collate_comp)
        return train_loader, val_loader

    def _apply_lora(self, model):
        if not self.cfg.lora_targets:
            return []
        adapters = apply_lora(
            model, rank=self.cfg.lora_rank, alpha=self.cfg.lora_alpha,
            dropout=self.cfg.lora_dropout, targets=self.cfg.lora_targets)
        self.logger.info('Applied LoRA: %d adapters (r=%d, alpha=%d)',
                         len(adapters), self.cfg.lora_rank, self.cfg.lora_alpha)
        return adapters

    # ------------------------------------------------------------------ #
    def _param_groups(self, model, adapters, phase: int):
        emb = list(_backbone_embeddings(model).parameters())
        head = list(model.mod_regressor.parameters()) + list(model.head_regressor.parameters())
        head_ids = {id(p) for p in head}
        emb_ids = {id(p) for p in emb}
        others = [p for n, p in model.named_parameters()
                  if p.requires_grad and id(p) not in head_ids and id(p) not in emb_ids]

        groups = []
        if self.cfg.embedding_lr > 0 and emb:
            groups.append({'params': emb, 'lr': self.cfg.embedding_lr, 'weight_decay': 0.0})
        groups.append({'params': head, 'lr': self.cfg.head_lr,
                       'weight_decay': self.cfg.weight_decay})
        if others:
            groups.append({'params': others, 'lr': self.cfg.encoder_lr,
                           'weight_decay': self.cfg.weight_decay})
        self.logger.info('phase %d param groups: %s',
                         phase, [len(g['params']) for g in groups])
        return groups

    def _freeze_with_lora(self, model, adapters):
        """All frozen except heads, the LoRA adapters and (optionally) embeddings."""
        for p in model.parameters():
            p.requires_grad = False
        for p in model.mod_regressor.parameters():
            p.requires_grad = True
        for p in model.head_regressor.parameters():
            p.requires_grad = True
        if self.cfg.embedding_lr > 0:
            for p in _backbone_embeddings(model).parameters():
                p.requires_grad = True
        for p in lora_parameters(adapters):
            p.requires_grad = True

    def _optimizer(self, model, adapters, phase, steps):
        groups = self._param_groups(model, adapters, phase)
        optimizer = AdamW(groups)
        track_optimizer_steps(optimizer)
        if phase == 1 and self.cfg.phase1_schedule == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=int(steps * self.cfg.warmup_ratio))
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(steps * self.cfg.warmup_ratio),
                num_training_steps=steps)
        return optimizer, scheduler

    # ------------------------------------------------------------------ #
    def fit(self, train_rows, val_rows, tokenizer, fold: Optional[int] = None,
            ckpt_name: str = 'best.pt', load_from: Optional[str | Path] = None) -> FoldResult:
        train_loader, val_loader = self._build_loaders(train_rows, val_rows, tokenizer)

        model = build_model(self.cfg, self.device, load_from=load_from)
        adapters = self._apply_lora(model)
        self._freeze_with_lora(model, adapters)

        scaler = GradScaler(
            'cuda',
            enabled=(self.device.type == 'cuda'),
            init_scale=self.cfg.amp_init_scale,
            growth_interval=self.cfg.amp_growth_interval,
        )
        criterion = CombinedLoss(
            ccc_weight=self.cfg.ccc_weight, lambda_rank=self.cfg.lambda_rank,
            rank_margin=self.cfg.rank_margin,
            rank_margin_mode=self.cfg.rank_margin_mode,
            ccc_var_floor=self.cfg.ccc_var_floor,
            std_alpha=self.cfg.loss_std_alpha,
            ce_weight=self.cfg.ce_weight, num_bins=self.cfg.num_bins,
            bin_sigma=self.cfg.bin_sigma, use_label_std=self.cfg.use_label_std,
        ).to(self.device)

        steps_per_epoch = math.ceil(len(train_loader) / self.cfg.accum_steps)
        optimizer, scheduler = self._optimizer(
            model, adapters, phase=1, steps=steps_per_epoch * self.cfg.freeze_epochs)

        best_rho, best_epoch, no_improve_epochs = -1.0, -1, 0
        best: Optional[Dict[str, np.ndarray]] = None
        history: List[Dict] = []

        for epoch in range(self.cfg.total_epochs):
            if self._train_sampler is not None:
                self._train_sampler.set_epoch(epoch)
            if epoch == self.cfg.freeze_epochs:
                if self.cfg.unfreeze_from_layer > 0:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    self.logger.info(
                        '>>> Unfreezing top layers (from layer %d) at epoch %d <<<',
                        self.cfg.unfreeze_from_layer, epoch + 1)
                    unfreeze_top_layers(model, self.cfg.unfreeze_from_layer)
                optimizer, scheduler = self._optimizer(
                    model, adapters, phase=2,
                    steps=steps_per_epoch * (self.cfg.total_epochs - self.cfg.freeze_epochs))

            phase = 'FROZEN' if epoch < self.cfg.freeze_epochs else 'UNFROZEN-TOP'
            diag: Dict[str, float] = {}
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler,
                self.device, grad_clip=self.cfg.grad_clip,
                accum_steps=self.cfg.accum_steps, report=diag,
                consist_weight=self.cfg.lambda_consist,
                consist_mode=self.cfg.consist_mode,
                consist_temp=self.cfg.consist_temp,
                compound_weight=self.cfg.lambda_compound,
            )
            tr_mod_p, tr_head_p, tr_mod_y, tr_head_y = evaluate(model, train_loader, self.device)
            tr_rho_m = _safe_rho(tr_mod_y, tr_mod_p)
            tr_rho_h = _safe_rho(tr_head_y, tr_head_p)

            mod_pred, head_pred, mod_label, head_label = evaluate(model, val_loader, self.device)
            rho_mod = _safe_rho(mod_label, mod_pred)
            rho_head = _safe_rho(head_label, head_pred)
            rho_mean = (rho_mod + rho_head) / 2.0

            gg = diag.get('group_grads', {})
            ovf = diag.get('overflow', {})
            gg_str = ' '.join(f'{k}={v:.1e}' for k, v in sorted(gg.items())) or '(none)'
            if ovf:
                gg_str += ' [' + ' '.join(f'ovf-{k}={v}' for k, v in sorted(ovf.items())) + ']'
            self.logger.info(
                'Epoch %d/%d [%s] | Loss %.4f | Train ρ %.4f | Val Mod ρ %.4f | Val Head ρ %.4f'
                ' | Val Mean ρ %.4f | steps %d (skip %d) | grads %s | scale %.1f | lr %.2e',
                epoch + 1, self.cfg.total_epochs, phase, train_loss,
                (tr_rho_m + tr_rho_h) / 2, rho_mod, rho_head, rho_mean,
                diag['opt_steps'], diag['skipped'], gg_str, diag['scale'], diag['lr'])

            history.append({
                'epoch': epoch + 1, 'phase': phase,
                'loss': round(float(train_loss), 5),
                'train_rho_mean': round((tr_rho_m + tr_rho_h) / 2, 5),
                'rho_mod': round(rho_mod, 5), 'rho_head': round(rho_head, 5),
                'rho_mean': round(rho_mean, 5),
                'opt_steps': diag['opt_steps'], 'skipped': diag['skipped'],
                'grad_norm': round(diag['grad_norm'], 4),
                'group_grads': {k: round(v, 4) for k, v in gg.items()},
                'scale': round(diag['scale'], 1), 'lr': diag['lr'],
            })

            if rho_mean > best_rho:
                best_rho, best_epoch, no_improve_epochs = rho_mean, epoch + 1, 0
                best = {
                    'mod': mod_pred.copy(), 'head': head_pred.copy(),
                    'mod_y': mod_label.copy(), 'head_y': head_label.copy(),
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
                            epoch + 1, best_epoch, best_rho)
                        break

        if best is None:
            raise RuntimeError('No improvement over any epoch - check the hyperparameters.')
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        best_rho_mod = _safe_rho(best['mod_y'], best['mod'])
        best_rho_head = _safe_rho(best['head_y'], best['head'])
        best_rho_mean = (best_rho_mod + best_rho_head) / 2.0
        self.logger.info('Split %s finished | best Mean ρ %.4f at epoch %d',
                         'n/a' if fold is None else fold, best_rho_mean, best_epoch)

        return FoldResult(
            fold=fold, rho_mod=best_rho_mod, rho_head=best_rho_head,
            rho_mean=best_rho_mean, best_epoch=best_epoch, ckpt_path=str(ckpt_path),
            history=history, best_mod_pred=best['mod'], best_head_pred=best['head'],
            best_mod_label=best['mod_y'], best_head_label=best['head_y'],
        )