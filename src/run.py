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
from pathlib import Path
from typing import Any, Dict, List

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
    if device_str and device_str != 'auto':
        return device_str
    dev = get_device()
    return str(dev) if dev else 'cpu'


def _smoke(logger: logging.Logger, device_str: str) -> None:
    """Quick pipeline sanity check (frozen backbone, 1 batch)."""
    import torch
    from torch.amp import GradScaler
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from src.config import Config
    from src.data import CompDataset, collate_comp, load_labeled
    from src.folds import assign_folds
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
    rows = load_labeled(cfg)[:12]
    tok = cfg.build_tokenizer()
    ds = CompDataset(rows, tok, max_len=64)
    folds = assign_folds(rows, 3, cfg.seed)
    assert all('fold' in r for r in folds)
    logger.info('[smoke] data OK (%d rows, folds ok)', len(rows))

    # 3. model
    device = torch.device(device_str)
    model = build_model(cfg, device)
    logger.info('[smoke] model OK')

    # 4a. GaussLoss + forward/backward
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_comp)
    criterion = GaussLoss(bin_sigma=cfg.bin_sigma, use_label_std=cfg.use_label_std,
                          std_alpha=cfg.loss_std_alpha, ccc_var_floor=cfg.ccc_var_floor)
    assert criterion.requires_logits
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.head_lr)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=1, num_training_steps=3)
    diag: Dict[str, Any] = {}
    batch = next(iter(loader))
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
        (mod_pred, head_pred, mod_emb, head_emb,
         mod_logits, head_logits) = model(batch, with_logits=True, with_reps=True)
        allowed = batch['has_label'] & batch['has_mod'] & batch['has_head'] & ~batch['degenerate']
        loss = (
            criterion(mod_pred[allowed], batch['mod_avg'][allowed], mod_logits[allowed],
                      batch['mod_std'][allowed], compound_ids=batch['compound_id'][allowed])
            + criterion(head_pred[allowed], batch['head_avg'][allowed], head_logits[allowed],
                        batch['head_std'][allowed], compound_ids=batch['compound_id'][allowed])
        )
    scaler.scale(loss).backward()
    logger.info('[smoke] gauss forward + backward OK (loss %.4f)', loss.item())

    # 4b. single-batch evaluate
    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
        model(batch, with_logits=True)
    mod_h, head_h = evaluate(model, loader, device)
    logger.info('[smoke] evaluate OK (mod_shape=%s, head_shape=%s)',
                mod_h.shape, head_h.shape)

    # 5. context 6-tuple
    with torch.no_grad():
        (mod_pred2, head_pred2, mod_emb2, head_emb2,
         mod_logits2, head_logits2) = model(batch, with_logits=True, with_reps=True)
    n_feat = mod_logits2.shape[1] if mod_logits2 is not None else 0
    has_ctx = mod_emb2 is not None
    assert has_ctx or n_feat > 0, 'use_context was enabled but no role features produced'
    logger.info('[smoke] context 6-tuple OK (lm_stats=%d, role_feats=%s, emb=%s)',
                n_feat, mod_logits2.shape if mod_logits2 is not None else None,
                mod_emb2.shape if mod_emb2 is not None else None)

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
    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    cfg = _build_config(args)
    cfg.validate()
    device_str = _lock_device(args.device)
    data_dir, output_dir = resolve_paths(cfg)

    logger = get_logger('src', log_dir=str(output_dir), level=logging.INFO)
    logger.info('Config: %s | device=%s', args.config or '(defaults)', device_str)
    cfg.save(output_dir / 'config.json')
    if args.smoke:
        logger.info('=== SMOKE MODE ===')
        _smoke(logger, device_str)
        return

    import torch
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
