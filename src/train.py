import math

import numpy as np
import torch
from torch.amp import autocast


def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device,
                grad_clip=1.0, accum_steps=1):
    """One epoch with optional gradient accumulation.

    `accum_steps > 1` keeps micro-batches small (large backbones on a T4) while
    the effective batch (and therefore LR schedule steps / gradient quality)
    matches `accum_steps * batch_size`. Scheduler steps once per optimizer step.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    n_micro = len(dataloader)
    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        
        with autocast(device_type):
            if getattr(criterion, 'requires_logits', False):
                mod_pred, head_pred, mod_logits, head_logits = model(batch, with_logits=True)
                loss = criterion(mod_pred, batch['mod_avg'], mod_logits) \
                    + criterion(head_pred, batch['head_avg'], head_logits)
            else:
                mod_pred, head_pred = model(batch)
                loss = criterion(mod_pred, batch['mod_avg']) + criterion(head_pred, batch['head_avg'])
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if step_idx % accum_steps == 0 or step_idx == n_micro:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            prev_steps = getattr(optimizer, '_step_count', None)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # Advance the LR schedule only when the optimizer really stepped.
            # _step_count stays `None` until the first completed step, so a
            # first-batch AMP overflow (cur == None) must NOT trigger the
            # scheduler -- that is exactly when torch would emit
            # "lr_scheduler.step() before optimizer.step()".
            cur_steps = getattr(optimizer, '_step_count', None)
            if cur_steps is not None and cur_steps != prev_steps:
                scheduler.step()

        total_loss += loss.item() * accum_steps

    return total_loss / n_micro


def evaluate(model, dataloader, device):
    model.eval()
    all_mod_preds, all_head_preds = [], []
    all_mod_labels, all_head_labels = [], []
    has_labels = False

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast('cuda'):
                mod_pred, head_pred = model(batch)

            all_mod_preds.append(mod_pred.detach().cpu().numpy().reshape(-1))
            all_head_preds.append(head_pred.detach().cpu().numpy().reshape(-1))

            # Safe for both validation (labeled) and test (unlabeled) sets
            if 'mod_avg' in batch and 'head_avg' in batch:
                has_labels = True
                all_mod_labels.append(batch['mod_avg'].cpu().numpy().reshape(-1))
                all_head_labels.append(batch['head_avg'].cpu().numpy().reshape(-1))

    mod_preds = np.concatenate(all_mod_preds)
    head_preds = np.concatenate(all_head_preds)

    if has_labels:
        mod_labels = np.concatenate(all_mod_labels)
        head_labels = np.concatenate(all_head_labels)
        return mod_preds, head_preds, mod_labels, head_labels

    return mod_preds, head_preds


def unfreeze_top_layers(model, from_layer):
    """Unfreeze top layers (from_layer..last). Keeps embeddings + lower layers frozen."""
    layers = model.bert.layers
    total_layers = len(layers)

    for idx, layer in enumerate(layers):
        if idx >= from_layer:
            for param in layer.parameters():
                param.requires_grad = True

    print(f'Unfroze layers {from_layer}-{total_layers-1} ({total_layers - from_layer} of {total_layers} layers)')