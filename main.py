#!/usr/bin/env python3
"""Command-line entry point for the ModernBERT Compositionality pipeline.

Usage
-----
    python main.py train5                    # default mode
    python main.py train80 --epochs 8 --batch 16
    python main.py train5 --config config.json
    python main.py predict --predict-mode single

Any option can also be changed permanently by editing the `Config` dataclass
in config.py or by providing a JSON config file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from config import Config


def _setup_src() -> None:
    """Make the src/ package importable from a local checkout or a Kaggle input."""
    from src.utils import find_src_root
    sys.path.insert(0, str(find_src_root()))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Train ModernBERT on the Compositionality shared task '
                    'and generate a submission.',
    )
    parser.add_argument(
        'mode', choices=('train80', 'train5', 'predict'),
        help="train80 = 80/20 split, train5 = stratified 5-fold CV, "
             "predict = trial + submission",
    )
    parser.add_argument(
        '--config', type=Path, metavar='PATH',
        help='JSON config file applied on top of defaults (before CLI flags).',
    )
    parser.add_argument('--epochs', type=int, help='override num_epochs')
    parser.add_argument(
        '--freeze-epochs', type=int, dest='freeze_epochs',
        help='override freeze_epochs (epochs of Phase 1 with frozen encoder)',
    )
    parser.add_argument('--batch', type=int, help='override batch_size')
    parser.add_argument(
        '--accum-steps', type=int, dest='accum_steps',
        help='override accum_steps (gradient accumulation; effective batch = batch * accum)',
    )
    parser.add_argument('--patience', type=int, help='override patience')
    parser.add_argument(
        '--ccc-weight', type=float,
        help='override ccc_weight (0..1; blend of MSE and 1-CCC)',
    )
    parser.add_argument(
        '--lambda-rank', type=float, metavar='L',
        help='override lambda_rank (>0 adds pairwise margin-ranking loss)',
    )
    parser.add_argument(
        '--unfreeze-from', type=int, dest='unfreeze_from_layer',
        help='override unfreeze_from_layer',
    )
    parser.add_argument(
        '--predict-mode', choices=('single', '5fold'),
        help='override predict_mode (only used by mode=predict)',
    )
    parser.add_argument('--data-path', type=str, help='override data_path')
    parser.add_argument('--output-dir', type=str, help='override output_dir')
    parser.add_argument('--seed', type=int, help='override seed')
    return parser


def _merge_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.config is not None:
        if not args.config.is_file():
            raise FileNotFoundError(f'Config file not found: {args.config}')
        raw = json.loads(args.config.read_text(encoding='utf-8'))
        cfg = Config.from_dict(raw)
    overrides = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key not in ('mode', 'config')
    }
    return cfg.update(**overrides)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        _setup_src()
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    import torch

    from src import pipelines
    from src.utils import get_device, get_logger, resolve_paths, set_seed

    try:
        cfg = _merge_overrides(Config.defaults(), args)
        cfg.mode = args.mode
        cfg.validate()
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

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
        return 1

    cfg.save(output_dir / 'config.json')
    logger.info('Working data dir: %s', data_dir)
    logger.info('Output dir: %s', output_dir)
    logger.info('Config:\n%s', cfg.pretty())

    set_seed(cfg.seed)

    started = datetime.now()
    try:
        if args.mode == 'train80':
            pipelines.run_train80(cfg, logger, device, data_dir, output_dir)
        elif args.mode == 'train5':
            pipelines.run_train5(cfg, logger, device, data_dir, output_dir)
        else:
            pipelines.run_predict(cfg, logger, device, data_dir, output_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error('%s: %s', type(exc).__name__, exc)
        return 1

    logger.info('Finished in %s', datetime.now() - started)
    return 0


if __name__ == '__main__':
    sys.exit(main())