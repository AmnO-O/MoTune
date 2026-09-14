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
from .utils import get_logger
import torch.nn.functional as F

logger = get_logger('mm.train')


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
    # Compute anchor BEFORE any indexing so requires_grad is preserved.
    # Boolean indexing on an empty result drops requires_grad in PyTorch;
    # anchoring to the full pred.sum()*0 ensures .backward() works even
    # when all rows are filtered (backbone frozen, only heads trainable).
    _zero = pred.sum() * 0.0
    if allowed is not None:
        pred = pred[allowed]
        target = target[allowed]
        logits = logits[allowed] if logits is not None else None
        std = std[allowed] if std is not None else None
        cid = cid[allowed] if cid is not None else None
        if pred.numel() == 0:
            return _zero
    return criterion(pred, target, logits, std, compound_ids=cid)
def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device,
                grad_clip=1.0, accum_steps=1, report=None,
                consist_weight=0.0, consist_mode='pull', consist_temp=0.1,
                compound_weight=0.0):
    """One scoring epoch with AMP + gradient accumulation + clipping.

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
    overflow: Dict[str, int] = {}
    n_micro = len(dataloader)

    tr_mod_preds, tr_head_preds = [], []
    tr_mod_targets, tr_head_targets = [], []
    tr_allowed = []

    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        allowed = (batch['has_label'] & batch['has_mod'] & batch['has_head']
                   & ~batch['degenerate'])

        with torch.amp.autocast(device_type):
            requires_logits = getattr(criterion, 'requires_logits', False)
            use_reps = consist_weight > 0
            
            mod_logits = head_logits = None
            mod_emb = head_emb = None

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
            if use_reps and mod_emb is not None:
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

            if not loss.requires_grad:
                loss = loss + (mod_pred.sum() + head_pred.sum()) * 0.0

            loss = loss / accum_steps

        scaler.scale(loss).backward()

        # Collect train predictions directly to avoid re-evaluating on train_loader
        tr_mod_preds.append(mod_pred.detach().cpu())
        tr_head_preds.append(head_pred.detach().cpu())
        tr_mod_targets.append(batch['mod_avg'].cpu())
        tr_head_targets.append(batch['head_avg'].cpu())
        tr_allowed.append(allowed.cpu())

        if step_idx % accum_steps == 0 or step_idx == n_micro:
            scaler.unscale_(optimizer)
            trainable = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            if trainable:
                last_grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip))
            else:
                last_grad_norm = 0.0

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
            'grad_norm': last_grad_norm,
            'overflow': overflow, 'scale': last_scale, 'lr': last_lr,
        })
        if n_consist:
            report['consist'] = consist_sum / n_consist
        if tr_mod_preds:
            m_p = torch.cat(tr_mod_preds).numpy()
            h_p = torch.cat(tr_head_preds).numpy()
            m_y = torch.cat(tr_mod_targets).numpy()
            h_y = torch.cat(tr_head_targets).numpy()
            al = torch.cat(tr_allowed).numpy()
            report['train_preds'] = (m_p, h_p, m_y, h_y, al)
    return total_loss / n_micro


def warmup_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device,
                 grad_clip=1.0, accum_steps=1):
    """One compound-aware MLM warmup epoch (CrossEntropy on -100-masked labels).

    Runs the base transformer WITHOUT the 256k-vocab prediction head, then
    applies the MLM head only to the masked positions (``labels != -100``).
    This sidesteps ModernBERT's fully-materialised dense logits
    ``[batch, seq, vocab]``, which alone (~1.2 GiB fp16 + 2.4 GiB fp32 copy)
    would OOM a 15 GiB T4 on top of the frozen encoder.
    """
    from .model import _backbone, _mlm_projection

    if not hasattr(optimizer, '_cmp_step_counter'):
        track_optimizer_steps(optimizer)
    step_counter = getattr(optimizer, '_cmp_step_counter', None)
    model.train()
    total, n_micro = 0.0, len(dataloader)
    optimizer.zero_grad()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    encoder = getattr(model, 'base_model', None) or _backbone(model)
    head, decoder, head_name, vocab = _mlm_projection(model)
    if not getattr(model, '_warmup_head_logged', False):
        model._warmup_head_logged = True
        from .utils import logger
        logger.info('MLM head chosen: %s (%s), %s, vocab=%d',
                    type(head).__name__, head_name,
                    f'decoder={decoder[0]}' if decoder else 'vocab-decoder',
                    vocab)

    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast(device_type):
            outputs = encoder(input_ids=batch['input_ids'],
                              attention_mask=batch['attention_mask'])
            hidden = (outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state')
                      else outputs[0])
            labels = batch['labels']
            mask = labels != -100
            if mask.any():
                # run the (transform + decoder) head ONLY on the masked rows:
                # hidden[B,T,H] -> rows[M,H] -> head -> [M,H] -> decoder -> [M,V]
                rows = hidden[mask]
                logits = head(rows)
                if logits.size(-1) != vocab:
                    if decoder is None:
                        raise ValueError(
                            f"MLM head '{head_name}' outputs {logits.size(-1)} "
                            f"classes and no {vocab}-output decoder was found "
                            "in the LM module tree")
                    logits = F.linear(logits, decoder[1].weight, decoder[1].bias)
                targets = labels[mask]
                if logits.size(-1) <= int(targets.max()):
                    raise ValueError(
                        f"MLM head outputs {logits.size(-1)} classes but labels "
                        f"reach {int(targets.max())} — wrong head picked or "
                        "tokenizer/model vocab mismatch")
                loss = criterion(logits.float(), targets) / accum_steps
            else:
                loss = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
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

import numpy as np
import torch

def evaluate(model, dataloader, device, return_all: bool = False):
    """Predictions + gold labels.

    If return_all=True: returns (all_mod, all_head, all_mod_y, all_head_y, label_mask)
    for all rows regardless of whether has_label is True or False.
    If return_all=False: returns (mod[mask], head[mask], mod_y[mask], head_y[mask]) if labeled,
    or (mod, head).
    """
    model.eval()
    all_mod, all_head = [], []
    all_mod_y, all_head_y = [], []
    masks, has_mask = [], False
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                mod_pred, head_pred = model(batch)

            all_mod.append(mod_pred.detach().cpu().numpy().reshape(-1))
            all_head.append(head_pred.detach().cpu().numpy().reshape(-1))

            lab = batch.get('has_label')
            if lab is not None:
                has_mask = True
                masks.append(lab.cpu().numpy().astype(bool).reshape(-1))

            # Xử lý an toàn nếu batch không chứa ground truth targets (ví dụ tập Test)
            if 'mod_avg' in batch and 'head_avg' in batch:
                all_mod_y.append(batch['mod_avg'].cpu().numpy().reshape(-1))
                all_head_y.append(batch['head_avg'].cpu().numpy().reshape(-1))
            else:
                all_mod_y.append(np.zeros(len(mod_pred)))
                all_head_y.append(np.zeros(len(head_pred)))

    mod = np.concatenate(all_mod) if all_mod else np.array([])
    head = np.concatenate(all_head) if all_head else np.array([])
    mod_y = np.concatenate(all_mod_y) if all_mod_y else np.array([])
    head_y = np.concatenate(all_head_y) if all_head_y else np.array([])
    mask = np.concatenate(masks) if has_mask else np.zeros(len(mod), dtype=bool)

    if return_all:
        return mod, head, mod_y, head_y, mask

    if has_mask and mask.any():
        return mod[mask], head[mask], mod_y[mask], head_y[mask]
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