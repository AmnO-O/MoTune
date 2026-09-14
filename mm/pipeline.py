"""End-to-end pipelines orchestrated by `run.py`.

Milestone 1 skeleton: command dispatch exists with validated signatures; the
body of each entry point is filled in by the later milestones:

  - M2: data layer (span alignment on real tokenizers, multilingual loaders)
  - M3: model + losses + trainer (attention pool, LoRA, MLM warmup)
  - M4: probes + whether the A/B grid produces the submission
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from mm.config import Config


def _not_built(name: str) -> Dict[str, float]:
    raise NotImplementedError(
        f'{name} is not wired yet (milestone M3+). Build order: '
        'M2 = mm/data.py span alignment, M3 = model/losses/trainer.'
    )


def run_warmup(cfg: Config, logger: logging.Logger, device,
               data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Phase 0: compound-aware MLM warmup on mmBERT with LoRA, then merge."""
    logger.info('=== warmup: compound-aware MLM (epochs=%d) ===', cfg.warmup_mlm_epochs)
    return _not_built('run_warmup')


def run_train80(cfg: Config, logger: logging.Logger, device,
                data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Compound-level 80/20 split, single checkpoint."""
    logger.info('=== train80 (compound-level 80/20 split) ===')
    return _not_built('run_train80')


def run_train5(cfg: Config, logger: logging.Logger, device,
               data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Stratified compound-grouped 5-fold CV + OOF evaluation."""
    logger.info('=== train5 (stratified %d-fold CV) ===', cfg.n_splits)
    return _not_built('run_train5')


def run_predict(cfg: Config, logger: logging.Logger, device,
                data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Predict the trial split from checkpoint(s) + write the submission."""
    logger.info('=== predict (trial + submission) ===')
    return _not_built('run_predict')


def run_probe(cfg: Config, logger: logging.Logger, device,
              data_dir: Path, output_dir: Path) -> Dict[str, float]:
    """Zero-shot LM-predictability probe before any scoring training."""
    logger.info('=== probe: LM predictability vs gold scores ===')
    return _not_built('run_probe')