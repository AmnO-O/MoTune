"""Owns the full gauss scoring fit loop for one split/fold."""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from transformers import get_constant_schedule, get_linear_schedule_with_warmup

from src.config import Config
from src.data import CompDataset, collate_comp
from src.folds import CompoundGroupSampler
from src.losses import GaussLoss
from src.model import apply_lora, build_model, lora_parameters
from src.train import evaluate, track_optimizer_steps, train_epoch, unfreeze_top_layers


def _safe_rho(y: np.ndarray, p: np.ndarray) -> float:
    r = float(spearmanr(y, p).statistic) if len(y) > 1 else 0.0
    return 0.0 if r != r else r


def _embeddings(model) -> nn.Module:
    """Input-embedding module of the plain AutoModel backbone."""
    return model.lm.get_input_embeddings()


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
    rho_pv: float = 0.0


class Trainer:
    def __init__(self, cfg: Config, device, logger: logging.Logger, output_dir: Path):
        self.cfg = cfg
        self.device = torch.device(device)
        self.logger = logger
        self.output_dir = Path(output_dir)
        self._val_rows: List[Dict] = []

    # ------------------------------------------------------------------ #
    def _build_loaders(self, train_rows, val_rows, tokenizer):
        self.logger.info("Building Datasets & Tokenizing %d train / %d val rows...", len(train_rows), len(val_rows))
        self._val_rows = list(val_rows)
        train_ds = CompDataset(train_rows, tokenizer, max_len=self.cfg.max_context_length)
        val_ds = CompDataset(val_rows, tokenizer, max_len=self.cfg.max_context_length)
        
        # Chỉ bật persistent_workers khi num_workers > 0 để tránh deadlock
        num_workers = max(0, self.cfg.num_workers)
        use_workers = num_workers > 0
        is_cuda = (getattr(self.device, 'type', '') == 'cuda')
        
        if self.cfg.lambda_rank > 0:
            cids = [r['compound_id'] for r in train_rows]
            self._train_sampler = CompoundGroupSampler(
                cids, batch_size=self.cfg.batch_size, seed=self.cfg.seed)
            train_loader = DataLoader(
                train_ds, batch_size=self.cfg.batch_size, shuffle=False,
                sampler=self._train_sampler, num_workers=num_workers,
                pin_memory=is_cuda, persistent_workers=use_workers,
                collate_fn=collate_comp)
        else:
            self._train_sampler = None
            train_loader = DataLoader(
                train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=is_cuda,
                persistent_workers=use_workers, collate_fn=collate_comp)
                
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=is_cuda,
            persistent_workers=use_workers, collate_fn=collate_comp)
            
        self.logger.info("DataLoaders ready (num_workers=%d, pin_memory=%s)", num_workers, is_cuda)
        return train_loader, val_loader

    def _apply_lora(self, model):
        if not self.cfg.lora_targets:
            return []
        adapters = apply_lora(
            model, rank=self.cfg.lora_rank, alpha=self.cfg.lora_alpha,
            dropout=self.cfg.lora_dropout, targets=self.cfg.lora_targets,
            from_layer=self.cfg.lora_from_layer)
        paths = getattr(model, '_lora_paths', [])
        wtgt = getattr(model, '_lora_targets_used', None)
        self.logger.info('Applied LoRA: %d adapters (r=%d, alpha=%d, layers >= %d) e.g. %s%s',
                         len(adapters), self.cfg.lora_rank, self.cfg.lora_alpha,
                         self.cfg.lora_from_layer, paths[:3],
                         f' (auto-fell back to targets {wtgt})' if wtgt else '')
        return adapters

    # ------------------------------------------------------------------ #
    def _pred_heads(self, model) -> List[nn.Module]:
        """The NN modifier/head exits and the overall PV composition exit."""
        return [model.mod_gauss, model.head_gauss, model.pv_gauss]

    # ------------------------------------------------------------------ #
    def _param_groups(self, model, adapters, phase: int):
        emb = list(_embeddings(model).parameters())
        heads = self._pred_heads(model)
        head = []
        for m in heads:
            head += list(m.parameters())
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
        self.logger.info('Phase %d param groups: %s',
                         phase, [len(g['params']) for g in groups])
        return groups

    def _freeze_phase1(self, model, adapters):
        """Phase 1: freeze backbone and LoRA, train only prediction heads."""
        for p in model.parameters():
            p.requires_grad = False
        for m in self._pred_heads(model):
            for p in m.parameters():
                p.requires_grad = True
        if self.cfg.embedding_lr > 0:
            for p in _embeddings(model).parameters():
                p.requires_grad = True
        for p in lora_parameters(adapters):
            p.requires_grad = False

    def _unfreeze_phase2(self, model, adapters):
        """Phase 2: unfreeze LoRA adapters and optionally top encoder layers."""
        for p in lora_parameters(adapters):
            p.requires_grad = True
        if self.cfg.unfreeze_from_layer > 0:
            unfreeze_top_layers(model, self.cfg.unfreeze_from_layer)

    def _optimizer(self, model, adapters, phase, steps):
        groups = self._param_groups(model, adapters, phase)
        optimizer = AdamW(groups)
        track_optimizer_steps(optimizer)
        if phase == 1:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=steps)
        return optimizer, scheduler

    # ------------------------------------------------------------------ #
    def fit(self, train_rows, val_rows, tokenizer, fold: Optional[int] = None,
            ckpt_name: str = 'best.pt', load_from: Optional[str | Path] = None) -> FoldResult:
        
        train_loader, val_loader = self._build_loaders(train_rows, val_rows, tokenizer)

        self.logger.info("Building model architecture (load_from=%s)...", load_from)
        model = build_model(self.cfg, self.device, load_from=load_from)
        adapters = self._apply_lora(model)

        steps_per_epoch = math.ceil(len(train_loader) / self.cfg.accum_steps)
        if self.cfg.freeze_epochs > 0:
            self._freeze_phase1(model, adapters)
            phase_init = 1
            steps_init = steps_per_epoch * self.cfg.freeze_epochs
        else:
            self._freeze_phase1(model, adapters)
            self._unfreeze_phase2(model, adapters)
            phase_init = 2
            steps_init = steps_per_epoch * self.cfg.total_epochs

        optimizer, scheduler = self._optimizer(
            model, adapters, phase=phase_init, steps=steps_init)

        scaler = GradScaler(
            'cuda',
            enabled=(self.device.type == 'cuda'),
            init_scale=self.cfg.amp_init_scale,
            growth_interval=self.cfg.amp_growth_interval,
        )
        criterion = GaussLoss(
            lambda_dist=self.cfg.lambda_dist, ccc_weight=self.cfg.ccc_weight,
            lambda_rank=self.cfg.lambda_rank,
            rank_margin=self.cfg.rank_margin,
            rank_margin_mode=self.cfg.rank_margin_mode,
            ccc_var_floor=self.cfg.ccc_var_floor,
            bin_sigma=self.cfg.bin_sigma, use_label_std=self.cfg.use_label_std,
            std_alpha=self.cfg.loss_std_alpha,
        ).to(self.device)

        best_rho, best_epoch, no_improve_epochs = -float('inf'), -1, 0
        best: Optional[Dict[str, np.ndarray]] = None
        history: List[Dict] = []
        ckpt_dir = self.output_dir / 'models'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / ckpt_name

        self.logger.info("Starting training loop for %d epochs...", self.cfg.total_epochs)

        for epoch in range(self.cfg.total_epochs):
            if self._train_sampler is not None:
                self._train_sampler.set_epoch(epoch)

            # Re-pool the mid-5 span cache whenever the backbone can have
            # moved since the previous epoch (LoRA phase-1 unfreezing steps
            # the LM every batch). Frozen-backbone runs pay a trivial one-off
            # recompute and stay correct either way.
            model.reset_span_cache()
                
            if epoch == self.cfg.freeze_epochs and self.cfg.freeze_epochs > 0:
                self.logger.info(
                    '>>> Entering Phase 2 (unfreezing LoRA%s) at epoch %d <<<',
                    f', top layers from {self.cfg.unfreeze_from_layer}' if self.cfg.unfreeze_from_layer > 0 else '',
                    epoch + 1)
                
                # Giải phóng optimizer & scheduler cũ để tránh đọng VRAM
                del optimizer, scheduler
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                self._unfreeze_phase2(model, adapters)
                optimizer, scheduler = self._optimizer(
                    model, adapters, phase=2,
                    steps=steps_per_epoch * self.cfg.lora_epochs)

            phase = 'FROZEN' if epoch < self.cfg.freeze_epochs else 'UNFROZEN-TOP'
            diag: Dict[str, float] = {}
            
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler,
                self.device, grad_clip=self.cfg.grad_clip,
                accum_steps=self.cfg.accum_steps, report=diag,
                compound_weight=self.cfg.lambda_compound,
            )

            # Compute train rho directly from in-epoch predictions
            if 'train_preds' in diag:
                tr_m, tr_h, tr_my, tr_hy, tr_al = diag['train_preds']
                tr_rho_m = _safe_rho(tr_my[tr_al], tr_m[tr_al]) if tr_al.any() else 0.0
                tr_rho_h = _safe_rho(tr_hy[tr_al], tr_h[tr_al]) if tr_al.any() else 0.0
            else:
                tr_rho_m = tr_rho_h = 0.0

            val_mod, val_head, val_pv, val_mod_y, val_head_y, val_mask = evaluate(
                model, val_loader, self.device, return_all=True, return_pv=True)

            is_pv_mask = np.array([bool(r.get('is_pv', False)) for r in self._val_rows])
            nn_mask = val_mask & (~is_pv_mask)
            pv_mask = val_mask & is_pv_mask

            rho_mod = _safe_rho(val_mod_y[nn_mask], val_mod[nn_mask]) if nn_mask.any() else 0.0
            rho_head = _safe_rho(val_head_y[nn_mask], val_head[nn_mask]) if nn_mask.any() else 0.0

            rho_pv = _safe_rho(val_mod_y[pv_mask], val_pv[pv_mask]) if pv_mask.any() else 0.0

            if pv_mask.any() and nn_mask.any():
                rho_mean = (rho_mod + rho_head + rho_pv) / 3.0
            elif pv_mask.any():
                rho_mean = rho_pv
            else:
                rho_mean = (rho_mod + rho_head) / 2.0

            ovf_str = ''
            if pv_mask.any():
                self.logger.info(
                    'Epoch %d/%d [%s] | Loss %.4f (nn_m %.4f / nn_h %.4f / pv %.4f) | Train ρ %.4f | '
                    'Val Mod ρ %.4f | Val Head ρ %.4f | Val PV ρ %.4f | Val Mean ρ %.4f | '
                    'steps %d (skip %d) | scale %.1f | lr %.2e',
                    epoch + 1, self.cfg.total_epochs, phase, train_loss,
                    diag.get('nn_mod_loss', diag['mod_loss']),
                    diag.get('nn_head_loss', diag['head_loss']),
                    diag.get('pv_loss', 0.0),
                    (tr_rho_m + tr_rho_h) / 2, rho_mod, rho_head, rho_pv, rho_mean,
                    diag['opt_steps'], diag['skipped'], diag['scale'], diag['lr'])
            else:
                self.logger.info(
                    'Epoch %d/%d [%s] | Loss %.4f (mod %.4f / head %.4f) | Train ρ %.4f | '
                    'Val Mod ρ %.4f | Val Head ρ %.4f | Val Mean ρ %.4f | '
                    'steps %d (skip %d) | scale %.1f | lr %.2e',
                    epoch + 1, self.cfg.total_epochs, phase, train_loss,
                    diag['mod_loss'], diag['head_loss'],
                    (tr_rho_m + tr_rho_h) / 2, rho_mod, rho_head, rho_mean,
                    diag['opt_steps'], diag['skipped'], diag['scale'], diag['lr'])

            history.append({
                'epoch': epoch + 1, 'phase': phase,
                'loss': round(float(train_loss), 5),
                'loss_mod': round(float(diag.get('nn_mod_loss', diag['mod_loss'])), 5),
                'loss_head': round(float(diag.get('nn_head_loss', diag['head_loss'])), 5),
                'loss_pv': round(float(diag.get('pv_loss', 0.0)), 5),
                'train_rho_mean': round((tr_rho_m + tr_rho_h) / 2, 5),
                'rho_mod': round(rho_mod, 5), 'rho_head': round(rho_head, 5),
                'rho_pv': round(rho_pv, 5),
                'rho_mean': round(rho_mean, 5),
                'opt_steps': diag['opt_steps'], 'skipped': diag['skipped'],
                'grad_norm': round(diag['grad_norm'], 4),
                'scale': round(diag['scale'], 1), 'lr': diag['lr'],
            })

            if rho_mean > best_rho:
                best_rho, best_epoch, no_improve_epochs = rho_mean, epoch + 1, 0
                best = {
                    'mod': val_mod[val_mask].copy() if val_mask.any() else np.array([]),
                    'head': val_head[val_mask].copy() if val_mask.any() else np.array([]),
                    'mod_y': val_mod_y[val_mask].copy() if val_mask.any() else np.array([]),
                    'head_y': val_head_y[val_mask].copy() if val_mask.any() else np.array([]),
                    'rho_mod': rho_mod, 'rho_head': rho_head, 'rho_pv': rho_pv,
                }
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
            
        # Dọn dẹp GPU sạch đống rác sau khi fit xong 1 fold
        del model, optimizer, scheduler, criterion
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        best_rho_mod = best['rho_mod']
        best_rho_head = best['rho_head']
        best_rho_pv = best.get('rho_pv', 0.0)
        best_rho_mean = best_rho
        self.logger.info('Split %s finished | best Mean ρ %.4f (Mod %.4f / Head %.4f / PV %.4f) at epoch %d',
                         'n/a' if fold is None else fold, best_rho_mean, best_rho_mod, best_rho_head, best_rho_pv, best_epoch)

        return FoldResult(
            fold=fold, rho_mod=best_rho_mod, rho_head=best_rho_head,
            rho_mean=best_rho_mean, best_epoch=best_epoch, ckpt_path=str(ckpt_path),
            history=history, best_mod_pred=best['mod'], best_head_pred=best['head'],
            best_mod_label=best['mod_y'], best_head_label=best['head_y'],
            rho_pv=best_rho_pv,
        )
