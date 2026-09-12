import math

import numpy as np
import torch
from torch.amp import autocast


def track_optimizer_steps(optimizer) -> None:
    """Version-independent "did the optimizer really step" detector.

    `_step_count` / `_optimizer_step_count` semantics vary across torch
    releases (and silently break the LR schedule, locking LR at 0 within
    warmup). Instead we wrap `optimizer.step` and count actual calls: AMP's
    GradScaler skips the call entirely on gradient overflow, so a real call
    == a real parameter update.
    """
    counter = [0]
    real_step = optimizer.step

    def step_wrapper(*args, **kwargs):
        result = real_step(*args, **kwargs)
        counter[0] += 1
        return result

    optimizer.step = step_wrapper           # scaler.step() -> optimizer.step() -> wrapper
    optimizer._cmp_step_counter = counter    # read by train_epoch


def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device,
                grad_clip=1.0, accum_steps=1, report=None):
    """One epoch with optional gradient accumulation.

    `accum_steps > 1` keeps micro-batches small (large backbones on a T4) while
    the effective batch (and therefore LR schedule steps / gradient quality)
    matches `accum_steps * batch_size`. Scheduler steps once per optimizer step.

    If `report` (a dict) is given, it is filled with per-epoch diagnostics that
    distinguish a genuinely stuck model (zero grads) from AMP overflow skips
    (gradient inf/nan on every step) from normal learning:
    opt_steps / skipped, last grad norm (before clip), scale, current LR.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    opt_steps = skipped = 0
    last_grad_norm = float('nan')
    last_scale = float('nan')
    last_lr = float('nan')
    step_counter = getattr(optimizer, '_cmp_step_counter', None)

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
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            last_grad_norm = float(gnorm)
            optimizer.zero_grad()

            if step_counter is not None:
                prev_opt_calls = step_counter[0]
                opt_step_result = scaler.step(optimizer)
                scaler.update()
                if step_counter[0] > prev_opt_calls:
                    scheduler.step()
                    opt_steps += 1
                else:
                    skipped += 1
            else:
                # No wrapper (e.g. notebook-built optimizers): fall back to the
                # _step_count heuristic. Only step the scheduler when the count
                # demonstrably advanced, so a first-batch AMP overflow (count
                # still None) must not emit 'scheduler.before optimizer.step'.
                prev_steps = getattr(optimizer, '_step_count', None)
                scaler.step(optimizer)
                scaler.update()
                cur_steps = getattr(optimizer, '_step_count', None)
                if cur_steps is not None and cur_steps != prev_steps:
                    scheduler.step()
                    opt_steps += 1
                else:
                    skipped += 1

            last_scale = float(scaler.get_scale())
            last_lr = float(scheduler.get_last_lr()[0])

        total_loss += loss.item() * accum_steps

    if report is not None:
        report.update({
            'opt_steps': opt_steps,
            'skipped': skipped,
            'grad_norm': last_grad_norm,
            'scale': last_scale,
            'lr': last_lr,
        })

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