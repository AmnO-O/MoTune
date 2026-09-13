import math
from typing import Dict

import numpy as np
import torch
from torch.amp import autocast


def track_optimizer_steps(optimizer) -> None:
    """Version-independent "did the optimizer really step" detector.

    `_step_count` / `_optimizer_step_count` semantics vary across torch
    releases (and silently break the LR schedule, locking LR at 0 within
    warmup). Instead we count actual `optimizer.step()` calls: AMP's
    GradScaler skips the call entirely on gradient overflow, so a real call
    == a real parameter update.

    Counting uses `register_step_post_hook`, which torch invokes after every
    real step through its `profile_hook_step` wrapper. This deliberately does
    NOT rewrite `optimizer.step`: torch >= 2.6 rebuilds `optimizer.step` when
    the LR scheduler is constructed (`patch_track_step_called` -> `wrap_step`
    reads `step_fn.__func__`) and a hand-rolled closure there raises
    "AttributeError: 'function' object has no attribute '__func__'".
    """
    if hasattr(optimizer, '_cmp_step_counter'):
        return
    counter = [0]
    optimizer._cmp_step_counter = counter     # read by train_epoch

    def _count_step(opt, args, kwargs):
        counter[0] += 1

    try:
        optimizer.register_step_post_hook(_count_step)
    except AttributeError:
        # Very old torch without step hooks: degrade to the _step_count
        # heuristic used as a fallback inside train_epoch.
        optimizer._cmp_step_counter = None
        del optimizer._cmp_step_counter


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
    group_grads: Dict[str, float] = {}
    overflow: Dict[str, int] = {}
    # Ensure step counting is active. This is a no-op if _phase1/_phase2
    # already installed tracking; in legacy notebook flows it installs the
    # hook lazily here (post-hooks work regardless of scheduler wrapping).
    if not hasattr(optimizer, '_cmp_step_counter'):
        track_optimizer_steps(optimizer)
    step_counter = getattr(optimizer, '_cmp_step_counter', None)

    n_micro = len(dataloader)
    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        
        with autocast(device_type):
            if getattr(criterion, 'requires_logits', False):
                mod_pred, head_pred, mod_logits, head_logits = model(batch, with_logits=True)
                loss = criterion(mod_pred, batch['mod_avg'], mod_logits, batch.get('mod_std')) \
                    + criterion(head_pred, batch['head_avg'], head_logits, batch.get('head_std'))
            else:
                mod_pred, head_pred = model(batch)
                loss = criterion(mod_pred, batch['mod_avg']) + criterion(head_pred, batch['head_avg'])
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if step_idx % accum_steps == 0 or step_idx == n_micro:
            scaler.unscale_(optimizer)
            # Clip per optimizer param-group, NOT globally. The embedding table
            # (~38M freshly-resized marker rows) has gradient norms hundreds of
            # times larger than the small heads; a single global clip to 1.0
            # scales everything by ~1/400 and starves head updates to ~0,
            # freezing the predictions while "training" runs.
            param_to_name = {id(p): n for n, p in model.named_parameters()}
            total_norm_sq = 0.0
            for group in optimizer.param_groups:
                params = [p for p in group['params'] if p.grad is not None]
                if not params:
                    continue
                raw_norm = torch.sqrt(sum(
                    (p.grad.detach().float() ** 2).sum() for p in params
                ))
                raw_norm = raw_norm.item() if torch.isfinite(raw_norm) else float('nan')
                # label the group by where its first named param lives
                label = 'enc'
                for p in params:
                    name = param_to_name.get(id(p), '')
                    if 'embedding' in name:
                        label = 'emb'
                        break
                    if 'regressor' in name:
                        label = 'head'
                if math.isnan(raw_norm):
                    # AMP overflow / inf or nan grad in this group: scaler.step
                    # will skip the update, so do not clip (clipping would zero
                    # other groups' grads) and only record the event.
                    overflow[label] = overflow.get(label, 0) + 1
                    continue
                if group_grads.get(label, -1.0) < 0 or raw_norm > group_grads.get(label, 0.0):
                    group_grads[label] = raw_norm
                total_norm_sq += raw_norm ** 2
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            last_grad_norm = total_norm_sq ** 0.5

            if step_counter is not None:
                prev_opt_calls = step_counter[0]
                opt_step_result = scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
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
                optimizer.zero_grad()
                cur_steps = getattr(optimizer, '_step_count', None)
                if cur_steps is not None and cur_steps != prev_steps:
                    scheduler.step()
                    opt_steps += 1
                else:
                    skipped += 1

            last_scale = float(scaler.get_scale())
            last_lr = float(scheduler.get_last_lr()[0])

        v = loss.item() * accum_steps
        total_loss += v if math.isfinite(v) else 0.0

    if report is not None:
        report.update({
            'opt_steps': opt_steps,
            'skipped': skipped,
            'grad_norm': last_grad_norm,
            'group_grads': group_grads,
            'overflow': overflow,
            'scale': last_scale,
            'lr': last_lr,
        })

    return total_loss / n_micro


def evaluate(model, dataloader, device):
    model.eval()
    all_mod_preds, all_head_preds = [], []
    all_mod_labels, all_head_labels = [], []
    has_labels = False
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast(device_type, enabled=(device_type == 'cuda')):
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
    # Locate the encoder stack across HF model variants:
    #   ModernBERT:    bert.encoder.layers  (+ bert.encoder.final_norm)
    #   BERT/RoBERTa:  bert.encoder.layer
    #   some exports:  bert.layers
    if hasattr(model.bert, 'layers'):
        layers = model.bert.layers
    elif hasattr(model.bert, 'encoder'):
        enc = model.bert.encoder
        if hasattr(enc, 'layers'):
            layers = enc.layers
        elif hasattr(enc, 'layer'):
            layers = enc.layer
        else:
            raise AttributeError("Cannot locate transformer layers in model.bert.encoder")
    else:
        raise AttributeError("Cannot locate transformer layers in model.bert")

    total_layers = len(layers)

    for idx, layer in enumerate(layers):
        if idx >= from_layer:
            for param in layer.parameters():
                param.requires_grad = True

    # Post-encoder LayerNorm; unfreeze so the top-layer representations can
    # adapt. ModernBERT keeps it on the encoder; BERT/RoBERTa often dot no such
    # norm at all (their final norm lives inside the pooler or not at all).
    final_norm = None
    if hasattr(model.bert, 'final_norm'):
        final_norm = model.bert.final_norm
    elif hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'final_norm'):
        final_norm = model.bert.encoder.final_norm
    if final_norm is not None:
        for param in final_norm.parameters():
            param.requires_grad = True

    print(f'Unfroze layers {from_layer}-{total_layers-1} ({total_layers - from_layer} of {total_layers} layers)')