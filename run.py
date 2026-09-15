#!/usr/bin/env python3
"""Single CLI entry point for the mmBERT compositionality pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

from mm.config import Config, coerce_value

_COMMANDS = ('train80', 'train5', 'predict', 'warmup', 'probe', 'smoke')
_TRAIN_MODES = ('train80', 'train5', 'predict')

_LOCK_FH = None


def _acquire_run_lock(output_dir) -> bool:
    """Refuse to start a second run.py against the same output dir.

    Kaggle (and accidental notebook re-runs) frequently fire two processes at
    the same GPU; both then contend for the HF cache / CUDA and appear to
    HANG right after the mode header. Lock via flock (POSIX only; harmless
    no-op on Windows for local dev).
    """
    global _LOCK_FH
    try:
        import fcntl
    except ImportError:
        return True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = open(output_dir / 'run.lock', 'w')
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    _LOCK_FH = fh  # keep the file handle alive for the whole process lifetime
    return True


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='run.py', description=__doc__)
    p.add_argument('command', choices=_COMMANDS)
    p.add_argument(
        '--config', type=Path, metavar='PATH',
        help='YAML or JSON config file applied on top of defaults (before --set overrides).'
    )
    p.add_argument(
        '--set', action='append', default=[], metavar='KEY=VALUE',
        help='override a config knob (repeatable). Lists are comma-separated.'
    )
    p.add_argument('--data-path', type=str, help='override data_path')
    p.add_argument('--output-dir', type=str, help='override output_dir')
    p.add_argument('--seed', type=int, help='override seed')
    p.add_argument('--debug', action='store_true', help='full traceback on error')
    return p


def _set_overrides(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for item in args.set:
        if '=' not in item:
            raise ValueError(f'--set expects KEY=VALUE, got {item!r}')
        key, raw = item.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f'--set empty key in {item!r}')
        out[key] = coerce_value(key, raw)
    return out


def _merge(cfg: Config, args: argparse.Namespace) -> Config:
    if args.config is not None:
        if not args.config.is_file():
            raise FileNotFoundError(f'Config file not found: {args.config}')
        cfg = Config.load(args.config)

    overrides = _set_overrides(args)
    if args.data_path is not None:
        overrides['data_path'] = args.data_path
    if args.output_dir is not None:
        overrides['output_dir'] = args.output_dir
    if args.seed is not None:
        overrides['seed'] = args.seed
        
    if args.command in _TRAIN_MODES:
        overrides['mode'] = args.command
    return cfg.update(**overrides)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        cfg = _merge(Config.defaults(), args)
        cfg.validate()
    except (ValueError, FileNotFoundError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        if getattr(args, 'debug', False):
            import traceback
            traceback.print_exc()
        return 1

    if args.command == 'smoke':
        return _run_smoke()

    # THIẾT LẬP MÔI TRƯỜNG TRƯỚC KHI IMPORT TORCH
    if cfg.debug_cuda:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'

    import torch
    from mm.utils import get_device, get_logger, resolve_paths, set_seed

    device = get_device()
    logger = get_logger()
    logger.info('Using device: %s', device)
    if torch.cuda.is_available():
        logger.info('GPU: %s | %.1f GB', torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    try:
        data_dir, output_dir = resolve_paths(cfg)
    except RuntimeError as exc:
        logger.error(str(exc))
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    cfg.save(output_dir / 'config.json')
    logger.info('Working data dir: %s', data_dir)
    logger.info('Output dir: %s', output_dir)
    logger.info('Config:\n%s', cfg.pretty())

    set_seed(cfg.seed)

    if not _acquire_run_lock(output_dir):
        logger.error('Another run.py is already running against %s; refusing '
                     'to start a duplicate process (remove %s/run.lock to force).',
                     output_dir, output_dir)
        return 1

    from mm import pipeline

    started = datetime.now()
    try:
        entry = {
            'train80': pipeline.run_train80,
            'train5': pipeline.run_train5,
            'predict': pipeline.run_predict,
            'warmup': pipeline.run_warmup,
            'probe': pipeline.run_probe,
        }[args.command]
        entry(cfg, logger, device, data_dir, output_dir)
    except Exception as exc:
        logger.error('%s: %s', type(exc).__name__, exc)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    logger.info('Finished in %s', datetime.now() - started)
    return 0


def _run_smoke() -> int:
    root = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, str(root / 'tests' / 'smoke_mm.py')],
        cwd=root,
    )
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())