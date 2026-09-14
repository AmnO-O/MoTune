"""Training primitives (ported from the old src/train.py, adapted to mmBERT).

Key adaptations for the rebuild:
  * ``allowed`` row mask = has_label & has_mod & has_head & ~degenerate, so
    unaligned (missing span) and German-collapsed (mod==head token) rows are
    never fed to span-based supervised / consistency / center terms.
  * per-batch compound-centroid MSE (``compound_center_loss``) added with
    ``lambda_compound`` weight.
  * ``unfreeze_top_layers`` walks the MLM wrapper's base model generically.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.amp import autocast

from .losses import compound_center_loss, compound_consistency_loss


def track_optimizer_steps(optimizer) -> None:
    """Version-independent 'did the optimizer really step' detector.

    Counts actual ``optimizer.step()`` calls via ``register_step_post_hook``
    (AMP's GradScaler skips the call entirely on overflow, so a real call ==
    a real parameter update). Do NOT rewrite ``optimizer.step``: torch >= 2.6
    rebuilds it when a scheduler is constructed (`patch_track_step_called`).
    """
    if getattr(optimizer, '_cmp_step_counter', None) is not None:
        return
    counter = [0]
    optimizer._cmp_step_counter = counter
    try:
        optimizer.register_step_post_hook(lambda opt, args, kwargs: counter.__setitem__(0, counter[0] + 1))
    except AttributeError:
        optimizer._cmp_step_counter = None


def _supervised_term(criterion, pred, target, logits, std, cid, allowed):
    if allowed is not None:
        pred = pred[allowed]
        target = target[allowed]
        logits = logits[allowed] if logits is not None else None
        std = std[allowed] if std is not None else None
        cid = cid[allowed] if cid is not None else None
        if pred.numel() == 0:
            return torch.zeros((), device=pred.device, dtype=torch.float)
    return criterion(pred, target, logits, std, compound_ids=cid)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device,
                grad_clip=1.0, accum_steps=1, report=None,
                consist_weight=0.0, consist_mode='pull', consist_temp=0.1,
                compound_weight=0.0):
    """One scoring epoch with AMP + gradient accumulation + per-group clipping.

    ``compound_weight > 0`` adds the gold-centroid MSE on labeled rows.
    Returns the mean supervised loss of the epoch.
    """
    if not hasattr(optimizer, '_cmp_step_counter'):
        track_optimizer_steps(optimizer)
    step_counter = getattr(optimizer, '_cmp_step_counter', None)

    model.train()
    total_loss = 0.0
    consist_sum, n_consist = 0.0, 0
    optimizer.zero_grad()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    opt_steps = skipped = 0
    last_grad_norm = last_scale = last_lr = float('nan')
    group_grads: Dict[str, float] = {}
    overflow: Dict[str, int] = {}
    n_micro = len(dataloader)

    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        allowed = (batch['has_label'] & batch['has_mod'] & batch['has_head']
                   & ~batch['degenerate'])

        with autocast(device_type):
            requires_logits = getattr(criterion, 'requires_logits', False)
            use_reps = consist_weight > 0
            if requires_logits and use_reps:
                (mod_pred, head_pred, mod_emb, head_emb,
                 mod_logits, head_logits) = model(batch, with_logits=True, with_reps=True)
            elif requires_logits:
                mod_pred, head_pred, mod_logits, head_logits = model(batch, with_logits=True)
            elif use_reps:
                mod_pred, head_pred, mod_emb, head_emb = model(batch, with_reps=True)
            else:
                mod_pred, head_pred = model(batch)

            loss = _supervised_term(
                criterion, mod_pred, batch['mod_avg'], mod_logits,
                batch['mod_std'], batch['compound_id'], allowed,
            ) + _supervised_term(
                criterion, head_pred, batch['head_avg'], head_logits,
                batch['head_std'], batch['compound_id'], allowed,
            )

            consist_val = None
            if use_reps:
                cid = batch['compound_id'].clone()
                cid[~allowed] = -1
                consist = compound_consistency_loss(
                    mod_emb, head_emb, cid, mode=consist_mode, temp=consist_temp)
                loss = loss + consist_weight * consist
                consist_val = consist.item() if torch.isfinite(consist) else float('nan')

            if compound_weight > 0:
                center_ids = batch['compound_id'].clone()
                center_ids[~allowed] = -1          # aux/unaligned/degenerate out
                center = compound_center_loss(
                    mod_pred, batch['mod_avg'], center_ids,
                ) + compound_center_loss(
                    head_pred, batch['head_avg'], center_ids,
                )
                loss = loss + compound_weight * center

            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if step_idx % accum_steps == 0 or step_idx == n_micro:
            scaler.unscale_(optimizer)
            param_to_name = {id(p): n for n, p in model.named_parameters()}
            total_norm_sq = 0.0
            trainable: List[torch.Tensor] = []
            for group in optimizer.param_groups:
                params = [p for p in group['params'] if p.grad is not None]
                if not params:
                    continue
                trainable += params
                raw = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in params))
                raw = raw.item() if torch.isfinite(raw) else float('nan')
                label = 'enc'
                for p in params:
                    name = param_to_name.get(id(p), '')
                    if 'embed' in name:
                        label = 'emb'
                        break
                    if 'regressor' in name:
                        label = 'head'
                        break
                    if 'lora' in name or 'adapter' in name:
                        label = 'lora'
                if math.isnan(raw):
                    overflow[label] = overflow.get(label, 0) + 1
                    continue
                if group_grads.get(label, -1.0) < 0 or raw > group_grads.get(label, 0.0):
                    group_grads[label] = raw
                total_norm_sq += raw ** 2
            # ONE global clip over all trainable params preserves the relative
            # gradient magnitudes between backbone/LoRA/head/embeddings;
            # per-group norms are kept as diagnostics only.
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
            last_grad_norm = total_norm_sq ** 0.5

            if step_counter is not None:
                prev = step_counter[0]
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if step_counter[0] > prev:
                    scheduler.step()
                    opt_steps += 1
                else:
                    skipped += 1
            else:
                prev = getattr(optimizer, '_step_count', None)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if getattr(optimizer, '_step_count', None) != prev:
                    scheduler.step()
                    opt_steps += 1
                else:
                    skipped += 1

            last_scale = float(scaler.get_scale())
            last_lr = float(scheduler.get_last_lr()[0])

        if consist_val is not None:
            consist_sum += consist_val
            n_consist += 1

        v = loss.item() * accum_steps
        total_loss += v if math.isfinite(v) else 0.0

    if report is not None:
        report.update({
            'opt_steps': opt_steps, 'skipped': skipped,
            'grad_norm': last_grad_norm, 'group_grads': group_grads,
            'overflow': overflow, 'scale': last_scale, 'lr': last_lr,
        })
        if n_consist:
            report['consist'] = consist_sum / n_consist
    return total_loss / n_micro


def warmup_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device,
                 grad_clip=1.0, accum_steps=1):
    """One compound-aware MLM warmup epoch (CrossEntropy on -100-masked labels)."""
    if not hasattr(optimizer, '_cmp_step_counter'):
        track_optimizer_steps(optimizer)
    step_counter = getattr(optimizer, '_cmp_step_counter', None)
    model.train()
    total, n_micro = 0.0, len(dataloader)
    optimizer.zero_grad()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast(device_type):
            outputs = model.lm(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'])
            loss = criterion(outputs.logits.float(), batch['labels']) / accum_steps
        scaler.scale(loss).backward()
        if step_idx % accum_steps == 0 or step_idx == n_micro:
            scaler.unscale_(optimizer)
            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if params:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            if step_counter is not None:
                prev = step_counter[0]
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if step_counter[0] > prev:
                    scheduler.step()
            else:
                prev = getattr(optimizer, '_step_count', None)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                cur = getattr(optimizer, '_step_count', None)
                if cur is not None and cur != prev:
                    scheduler.step()
        v = loss.item() * accum_steps
        total += v if math.isfinite(v) else 0.0
    return total / n_micro


def evaluate(model, dataloader, device):
    """Predictions + gold labels masked by has_label (like the old evaluate).

    Returns (mod_preds, head_preds[, mod_labels, head_labels]); the label
    pair is returned only when at least one row is labeled.
    """
    model.eval()
    all_mod, all_head = [], []
    all_mod_y, all_head_y = [], []
    masks, has_mask = [], False
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast(device_type, enabled=(device_type == 'cuda')):
                mod_pred, head_pred = model(batch)
            all_mod.append(mod_pred.detach().cpu().numpy().reshape(-1))
            all_head.append(head_pred.detach().cpu().numpy().reshape(-1))
            lab = batch.get('has_label')
            if lab is not None:
                has_mask = True
                masks.append(lab.cpu().numpy().reshape(-1))
            all_mod_y.append(batch['mod_avg'].cpu().numpy().reshape(-1))
            all_head_y.append(batch['head_avg'].cpu().numpy().reshape(-1))

    mod, head = np.concatenate(all_mod), np.concatenate(all_head)
    if has_mask:
        mask = np.concatenate(masks)
        if mask.any():
            return (mod[mask], head[mask],
                    np.concatenate(all_mod_y)[mask], np.concatenate(all_head_y)[mask])
    return mod, head


def unfreeze_top_layers(model, from_layer: int) -> None:
    """Unfreeze top encoder layers (from_layer..last) + final norm (if any)."""
    base = model.lm
    for attr in ('model', 'bert', 'base_model', 'transformer'):
        if hasattr(base, attr):
            base = getattr(base, attr)
            break

    layers = None
    if hasattr(base, 'layers'):
        layers = base.layers
    elif hasattr(base, 'encoder'):
        enc = base.encoder
        if hasattr(enc, 'layers'):
            layers = enc.layers
        elif hasattr(enc, 'layer'):
            layers = enc.layer
    if layers is None:
        raise AttributeError('Cannot locate the transformer encoder layers')

    for idx, layer in enumerate(layers):
        if idx >= from_layer:
            for param in layer.parameters():
                param.requires_grad = True

    norm = getattr(base, 'final_norm', None)
    if norm is None and hasattr(base, 'encoder'):
        norm = getattr(base.encoder, 'final_norm', None)
    if norm is not None:
        for param in norm.parameters():
            param.requires_grad = True