"""CLI entry point for the gauss-only src/ package (train80 + smoke only).

Usage::

    python -m src.run --config config/mae_gauss.json
    python -m src.run --config config/mae_gauss.json --set lora_rank=8
    python -m src.run --config config/mae_gauss.json --smoke --device cpu

``--smoke`` runs a fast pipeline sanity check and exits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except ImportError:
    torch = None

from src.config import Config, coerce_value
from src.utils import get_logger, set_seed, get_device, resolve_paths


def _base_config() -> Dict[str, Any]:
    return {}


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Config not found: {path}')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _build_config(args) -> Config:
    cfg_dict: Dict[str, Any] = _base_config()
    if args.config:
        cfg_dict.update(_load_json(args.config))
    if args.set:
        for s in args.set:
            if '=' not in s:
                raise ValueError(f'--set expects KEY=VALUE, got {s!r}')
            k, v = s.split('=', 1)
            if not k.strip():
                raise ValueError(f'--set empty key in {s!r}')
            cfg_dict[k.strip()] = coerce_value(k.strip(), v)
    return Config(**cfg_dict)


def _lock_device(device_str: str | None) -> str:
    """Resolve the device string, validating CUDA availability.

    ``--device auto`` uses the first available accelerator; an explicit
    ``--device cuda[:N]`` on a CPU-only torch build (or with too few GPUs)
    falls back to ``cpu`` with a warning instead of crashing at ``.to()``.
    """
    if device_str and device_str != 'auto':
        chosen = device_str
    else:
        dev = get_device()
        chosen = str(dev) if dev else 'cpu'
    if chosen.startswith('cuda'):
        if not torch.cuda.is_available():
            warnings.warn(
                f'CUDA requested ({chosen}) but torch has no usable CUDA; using cpu')
            return 'cpu'
        if chosen != 'cuda':
            idx = int(chosen.split(':')[1])
            if idx >= torch.cuda.device_count():
                warnings.warn(
                    f'CUDA device {idx} requested but only '
                    f'{torch.cuda.device_count()} GPU(s) visible; using cuda:0')
                return 'cuda'
    return chosen


def _smoke(logger: logging.Logger, device_str: str) -> None:
    """Quick pipeline sanity check (frozen backbone, 1 batch)."""
    from torch.amp import GradScaler
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from src.config import Config
    from src.data import CompDataset, collate_comp, load_labeled
    from src.marks import Span, find_spans
    from src.model import build_model
    from src.train import train_epoch, evaluate
    from src.losses import GaussLoss

    cfg = Config()
    logger.info('=== SMOKE (device=%s, head=gauss only) ===', device_str)

    # 1. marks
    test = "It was a night watch"
    offsets = [(0, 2), (3, 6), (7, 8), (9, 13), (14, 19), (20, 25), (0, 0)]
    result = find_spans(test, offsets, "night", "watch", "")
    assert result.found and not result.degenerate
    logger.info('[smoke] marks OK')

    # 2. data
    rows = load_labeled(cfg)[:60]
    tok = cfg.build_tokenizer()
    ds = CompDataset(rows, tok, max_len=64)
    assert len(ds) == len(rows)
    logger.info('[smoke] data OK (%d rows tokenized)', len(rows))

    # 3. model
    device = torch.device(device_str)
    model = build_model(cfg, device)
    logger.info('[smoke] model OK')

    # 4a. GaussLoss + forward/backward
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_comp)
    criterion = GaussLoss(bin_sigma=cfg.bin_sigma, use_label_std=cfg.use_label_std,
                          ccc_var_floor=cfg.ccc_var_floor)
    assert criterion.requires_logits
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.head_lr)
    opt.zero_grad()
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=1, num_training_steps=3)
    batch = next(iter(loader))
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.amp.autocast(device_type, enabled=(device.type == 'cuda')):
        mod_pred, head_pred, mod_logits, head_logits = model(batch, with_logits=True)
        allowed = batch['has_label'] & batch['has_mod'] & batch['has_head'] & ~batch['degenerate']
        loss = (
            criterion(mod_pred, batch['mod_avg'], mod_logits, batch['mod_std'],
                      mask=allowed)
            + criterion(head_pred, batch['head_avg'], head_logits, batch['head_std'],
                        mask=allowed)
        )
    scaler.scale(loss).backward()
    logger.info('[smoke] gauss forward + backward OK (loss %.4f)', loss.item())

    # 4a-PV. The overall PV exit must be usable independently of the NN exits:
    # it consumes the Base/Particle span pair and its loss must reach pv_gauss.
    opt.zero_grad()
    with torch.amp.autocast(device_type, enabled=(device.type == 'cuda')):
        _, _, pv_pred, _, _, _ = model(batch, with_logits=True, with_pv=True)
        pv_loss = pv_pred.square().mean()
    scaler.scale(pv_loss).backward()
    pv_grads = [p.grad for p in model.pv_gauss.parameters() if p.grad is not None]
    assert pv_grads and all(torch.isfinite(g).all() for g in pv_grads)
    logger.info('[smoke] PV exit forward + backward OK (loss %.4f)', pv_loss.item())

    # 4b. single-batch evaluate
    with torch.amp.autocast(device_type, enabled=(device.type == 'cuda')):
        model(batch, with_logits=True)
    mod_h, head_h, *_ = evaluate(model, loader, device)
    logger.info('[smoke] evaluate OK (mod_shape=%s, head_shape=%s)',
                mod_h.shape, head_h.shape)

    # 5. eval-mode clamp + finite prediction check
    model.eval()
    with torch.no_grad():
        mod_pred2, head_pred2 = model(batch)
    assert tuple(mod_pred2.shape) == (batch['input_ids'].shape[0],)
    assert tuple(head_pred2.shape) == (batch['input_ids'].shape[0],)
    assert torch.isfinite(mod_pred2).all() and torch.isfinite(head_pred2).all()
    logger.info('[smoke] eval-mode clamp OK (mod_pred=%s, head_pred=%s)',
                tuple(mod_pred2.shape), tuple(head_pred2.shape))

    logger.info('=== SMOKE PASSED ===')


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Compositionality pipeline (gauss-only src/)')
    ap.add_argument('--config', type=str, default=None, help='Path to config JSON file')
    ap.add_argument('--set', action='append', default=[],
                    help='Override config values as key=value (value auto-coerced; repeatable)')
    ap.add_argument('--device', type=str, default='auto',
                    help='Device: auto, cpu, cuda, cuda:0 (default: auto)')
    ap.add_argument('--smoke', action='store_true',
                    help='Run a quick pipeline sanity check (1 batch, no checkpoint)')
    ap.add_argument('--check-config', action='store_true',
                    help='Validate the effective config, print it, and exit without training')
    return ap


def _print_config(cfg: Config) -> None:
    """Print the resolved config as JSON (no device, no output dirs touched)."""
    print('=== EFFECTIVE CONFIG ===')
    print(json.dumps(asdict(cfg), indent=2, sort_keys=True, ensure_ascii=False))
    print('=== CONFIG OK ===')


def _startup_check(logger, cfg: Config, device_str: str) -> None:
    """Log every resolved config field + key environment facts before the
    training phase starts, so a wrong config is visible in the very first logs.
    """
    logger.info('=' * 70)
    logger.info('CONFIG CHECK (start of run, before training phase)')
    logger.info('=' * 70)
    for k, v in asdict(cfg).items():
        logger.info('  cfg.%-22s %s', k, v)
    logger.info('--- environment ---')
    logger.info('  %-28s %s', 'python', sys.version.split()[0])
    try:
        import torch as _t
        logger.info('  %-28s %s (cuda_available=%s)', 'torch', _t.__version__, _t.cuda.is_available())
        if device_str.startswith('cuda') and _t.cuda.is_available():
            try:
                _i = 0 if device_str == 'cuda' else int(device_str.split(':')[1])
                logger.info('  %-28s %s', 'gpu', _t.cuda.get_device_name(_i))
            except Exception:
                pass
            free, total = _t.cuda.mem_get_info(0)
            logger.info('  %-28s %.2f / %.2f GiB free', 'gpu memory', free / 2**30, total / 2**30)
    except Exception as exc:
        logger.info('  %-28s error: %s', 'torch', exc)
    try:
        import transformers as _tf
        logger.info('  %-28s %s', 'transformers', _tf.__version__)
    except Exception as exc:
        logger.info('  %-28s error: %s', 'transformers', exc)
    logger.info('=' * 70)
    logger.info('CONFIG CHECK OK')
    logger.info('=' * 70)


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    cfg = _build_config(args)
    cfg.validate()
    if args.check_config:
        _print_config(cfg)
        return
    device_str = _lock_device(args.device)
    data_dir, output_dir = resolve_paths(cfg)

    logger = get_logger('src', log_dir=str(output_dir), level=logging.INFO)
    logger.info('Config: %s | device=%s', args.config or '(defaults)', device_str)
    cfg.save(output_dir / 'config.json')
    _startup_check(logger, cfg, device_str)
    if args.smoke:
        logger.info('=== SMOKE MODE ===')
        _smoke(logger, device_str)
        return

    device = torch.device(device_str)
    set_seed(cfg.seed)

    if cfg.mode != 'train80':
        raise ValueError(f'src/ only supports mode=train80, got {cfg.mode!r}')

    from src.pipeline import run_train80
    started = time.time()
    metrics = run_train80(cfg, logger, device, data_dir, output_dir)

    logger.info('Done (%.1fs) | val_rho_mean=%.4f', time.time() - started,
                metrics['val_rho_mean'])


if __name__ == '__main__':
    main()
