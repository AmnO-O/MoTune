"""Training primitives for the gauss-only pipeline.

  * ``allowed`` row mask = has_label & has_mod & has_head & ~degenerate, so
    unaligned (missing span) and German-collapsed (mod==head token) rows are
    never fed to span-based supervised / center terms.
  * per-batch compound-centroid MSE (``compound_center_loss``) added with
    ``lambda_compound`` weight.
  * ``unfreeze_top_layers`` walks the encoder generically and skips
    ``.linear.`` paths so LoRA base weights stay frozen.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .losses import compound_center_loss
from .utils import get_logger

logger = get_logger('src.train')


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


def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device,
                grad_clip=1.0, accum_steps=1, report=None,
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
    mod_loss_sum = head_loss_sum = pv_loss_sum = 0.0
    optimizer.zero_grad()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    opt_steps = skipped = 0
    last_grad_norm = last_scale = last_lr = float('nan')
    n_micro = len(dataloader)

    tr_mod_preds, tr_head_preds = [], []
    tr_mod_targets, tr_head_targets = [], []
    tr_allowed = []

    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        allowed = (batch['has_label'] & batch['has_mod'] & batch['has_head']
                   & ~batch['degenerate'])
        is_pv = batch.get('is_pv', torch.zeros_like(allowed))
        allowed_nn = allowed & (~is_pv)
        allowed_pv = allowed & is_pv

        with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
            requires_logits = getattr(criterion, 'requires_logits', False)

            mod_logits = head_logits = pv_logits = None

            if requires_logits:
                (mod_pred, head_pred, pv_pred, mod_logits, head_logits, pv_logits) = model(
                    batch, with_logits=True, with_pv=True)
            else:
                mod_pred, head_pred, pv_pred = model(batch, with_pv=True)

            # NN loss (mask=allowed on NN rows)
            mod_loss = criterion(
                mod_pred, batch['mod_avg'], mod_logits, batch['mod_std'],
                compound_ids=batch['compound_id'], mask=allowed_nn,
            )
            head_loss = criterion(
                head_pred, batch['head_avg'], head_logits, batch['head_std'],
                compound_ids=batch['compound_id'], mask=allowed_nn,
            )

            # PV has only an overall Avg/Std label.  Its dedicated composition
            # exit consumes both Base and Particle spans; mod/head exits do not
            # receive PV supervision.
            pv_loss = criterion(
                pv_pred, batch['mod_avg'], pv_logits, batch['mod_std'],
                compound_ids=batch['compound_id'], mask=allowed_pv,
            )

            loss = mod_loss + head_loss + pv_loss

            if compound_weight > 0:
                center_ids_nn = batch['compound_id'].clone()
                center_ids_nn[~allowed_nn] = -1
                center_ids_pv = batch['compound_id'].clone()
                center_ids_pv[~allowed_pv] = -1
                center = compound_center_loss(
                    mod_pred, batch['mod_avg'], center_ids_nn,
                ) + compound_center_loss(
                    head_pred, batch['head_avg'], center_ids_nn,
                ) + compound_center_loss(
                    pv_pred, batch['mod_avg'], center_ids_pv,
                )
                loss = loss + compound_weight * center

            if not loss.requires_grad:
                loss = loss + (mod_pred.sum() + head_pred.sum() + pv_pred.sum()) * 0.0

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

        v = loss.item() * accum_steps
        total_loss += v if math.isfinite(v) else 0.0
        mv = mod_loss.item() * accum_steps
        hv = head_loss.item() * accum_steps
        pvv = pv_loss.item() * accum_steps
        mod_loss_sum += mv if math.isfinite(mv) else 0.0
        head_loss_sum += hv if math.isfinite(hv) else 0.0
        pv_loss_sum += pvv if math.isfinite(pvv) else 0.0

    if report is not None:
        report.update({
            'opt_steps': opt_steps, 'skipped': skipped,
            'grad_norm': last_grad_norm,
            'scale': last_scale, 'lr': last_lr,
            'nn_mod_loss': mod_loss_sum / n_micro,
            'nn_head_loss': head_loss_sum / n_micro,
            'pv_loss': pv_loss_sum / n_micro,
            'mod_loss': mod_loss_sum / n_micro,
            'head_loss': head_loss_sum / n_micro,
        })
        if tr_mod_preds:
            m_p = torch.cat(tr_mod_preds).numpy()
            h_p = torch.cat(tr_head_preds).numpy()
            m_y = torch.cat(tr_mod_targets).numpy()
            h_y = torch.cat(tr_head_targets).numpy()
            al = torch.cat(tr_allowed).numpy()
            report['train_preds'] = (m_p, h_p, m_y, h_y, al)
    return total_loss / n_micro



def evaluate(model, dataloader, device, return_all: bool = False,
             return_pv: bool = False):
    """Predictions + gold labels.

    If return_all=True: returns (all_mod, all_head, all_mod_y, all_head_y, label_mask)
    for all rows regardless of whether has_label is True or False.
    If return_all=False: returns (mod[mask], head[mask], mod_y[mask], head_y[mask]) if labeled,
    or (mod, head).
    """
    model.eval()
    all_mod, all_head, all_pv = [], [], []
    all_mod_y, all_head_y = [], []
    masks, has_mask = [], False
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                if return_pv:
                    mod_pred, head_pred, pv_pred = model(batch, with_pv=True)
                else:
                    mod_pred, head_pred = model(batch)

            all_mod.append(mod_pred.detach().cpu().numpy().reshape(-1))
            all_head.append(head_pred.detach().cpu().numpy().reshape(-1))
            if return_pv:
                all_pv.append(pv_pred.detach().cpu().numpy().reshape(-1))

            lab = batch.get('has_label')
            if lab is not None:
                has_mask = True
                masks.append(lab.cpu().numpy().astype(bool).reshape(-1))
            else:
                masks.append(np.zeros(int(mod_pred.numel()), dtype=bool))

            # Xử lý an toàn nếu batch không chứa ground truth targets (ví dụ tập Test)
            if 'mod_avg' in batch and 'head_avg' in batch:
                all_mod_y.append(batch['mod_avg'].cpu().numpy().reshape(-1))
                all_head_y.append(batch['head_avg'].cpu().numpy().reshape(-1))
            else:
                all_mod_y.append(np.zeros(len(mod_pred)))
                all_head_y.append(np.zeros(len(head_pred)))

    mod = np.concatenate(all_mod) if all_mod else np.array([])
    head = np.concatenate(all_head) if all_head else np.array([])
    pv = np.concatenate(all_pv) if all_pv else np.array([])
    mod_y = np.concatenate(all_mod_y) if all_mod_y else np.array([])
    head_y = np.concatenate(all_head_y) if all_head_y else np.array([])
    mask = np.concatenate(masks) if has_mask else np.zeros(len(mod), dtype=bool)

    if return_all:
        if return_pv:
            return mod, head, pv, mod_y, head_y, mask
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
            for name, param in layer.named_parameters():
                if '.linear.' in name:
                    continue   # base weight of a LoRAAdapter stays frozen
                param.requires_grad = True

    norm = getattr(base, 'final_norm', None)
    if norm is None and hasattr(base, 'encoder'):
        norm = getattr(base.encoder, 'final_norm', None)
    if norm is not None:
        for param in norm.parameters():
            param.requires_grad = True
