"""Small, dependency-light utilities shared across the pipeline.

Only imports torch lazily so that config-only or data-only tooling still
works in minimal environments.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config

_LOG_FORMAT = '%(asctime)s | %(levelname)-7s | %(message)s'
_LOG_DATE = '%H:%M:%S'


def set_seed(seed: int) -> None:
    """Seed every RNG in the process for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available torch device, or None if torch is absent."""
    try:
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        return None


def get_logger(name: str = 'compartment', log_dir: Optional[str | Path] = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'run.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


_KAGGLE_DATASET = Path('/kaggle/input/datasets/ieltsmater/compartment/Compartment')
_KAGGLE_WORKING = Path('/kaggle/working')


def _auto_data_path() -> Path:
    if _KAGGLE_DATASET.is_dir():
        return _KAGGLE_DATASET
    local = Path('dataset')
    if local.is_dir():
        return local
    raise RuntimeError(
        'Training data not found. Mount the Compartment Kaggle dataset or '
        'place dataset/ locally, or set data_path in the config.'
    )


def _auto_output_dir() -> Path:
    if _KAGGLE_WORKING.is_dir():
        return _KAGGLE_WORKING
    return Path('output')


def resolve_paths(cfg: Config):
    """Resolve and create the data + output directories for a run."""
    data_dir = Path(cfg.data_path) if cfg.data_path else _auto_data_path()
    output_dir = Path(cfg.output_dir) if cfg.output_dir else _auto_output_dir()

    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    (output_dir / 'submission').mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir


def find_src_root() -> Path:
    """Locate the directory holding the src/ package (local or Kaggle input)."""
    if Path('src').is_dir():
        return Path(os.path.abspath('.'))

    roots = sorted(
        Path('/kaggle/input').glob('*/src')
    ) + sorted(
        Path('/kaggle/input').glob('*/**/src')
    )
    if roots:
        return roots[0].parent

    raise RuntimeError(
        'src/ package not found. Run from the repo root or add the GitHub '
        'repo as a Kaggle input (Add Input -> GitHub).'
    )