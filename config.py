"""Typed configuration for the Compositionality training pipeline.

The `Config` dataclass is the single source of truth for every hyperparameter,
path and mode of operation. It can be built from CLI overrides, a JSON file,
or defaults, and is always serialized next to the run artifacts for
reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

Mode = Literal['train80', 'train5', 'predict']
PredictMode = Literal['single', '5fold']

_MODES = ('train80', 'train5', 'predict')
_PREDICT_MODES = ('single', '5fold')


@dataclass
class Config:
    """All knobs used by the training and prediction pipelines."""

    # === run ===
    mode: Mode = 'train5'
    seed: int = 42

    # === model ===
    model_name: str = 'answerdotai/ModernBERT-base'
    hidden_size: int = 768
    max_length: int = 128
    max_context_length: int = 256
    dropout: float = 0.2
    # output head: 'reg' = scalar regression; 'softmax' = ordinal bins -> E[Y]
    head_mode: str = 'reg'
    num_bins: int = 6       # ordinal bins, centers uniformly spaced over [1, 5]

    # === optimization ===
    batch_size: int = 32
    accum_steps: int = 1       # gradient accumulation (effective batch = batch_size * accum_steps)
    head_lr: float = 5e-4
    encoder_lr: float = 3e-6
    embedding_lr: float = 1e-5     # only the (new, random) marker embeddings
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    num_epochs: int = 10
    freeze_epochs: int = 5
    unfreeze_from_layer: int = 19
    warmup_ratio: float = 0.15
    loss_type: str = 'mse_ccc'
    ccc_weight: float = 0.7
    lambda_rank: float = 0.0     # 0 = off; >0 adds pairwise margin-ranking to the loss
    rank_margin: float = 0.5
    ce_weight: float = 0.0     # 0 = off; >0 adds Gaussian soft-target CE on the ordinal bins
    bin_sigma: float = 0.5     # std (in bin units) of the Gaussian soft target
    use_label_std: bool = True # per-sample Gaussian width from ModStd/HeadStd when available; falls back to bin_sigma
    patience: int = 5
    num_workers: int = 2

    # === split ===
    n_splits: int = 5
    test_size: float = 0.2

    # === predict ===
    predict_mode: PredictMode = '5fold'

    # === paths (None = auto-detect: Kaggle dataset / local ./dataset, ./output) ===
    data_path: Optional[str] = None
    output_dir: Optional[str] = None

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def defaults(cls) -> 'Config':
        return cls()

    @classmethod
    def from_dict(cls, values: Dict[str, Any], strict: bool = False) -> 'Config':
        """Return a new Config with `values` applied on top of defaults.

        With `strict=True`, unknown keys raise rather than being ignored.
        """
        known = {f.name for f in fields(cls)}
        extra = set(values) - known
        if extra and strict:
            raise ValueError(f'Unknown config keys: {sorted(extra)}')
        safe = {k: v for k, v in values.items() if k in known}
        return replace(cls.defaults(), **safe)

    @classmethod
    def load(cls, path: str | Path) -> 'Config':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    def update(self, **values: Any) -> 'Config':
        return self.from_dict(values, strict=True).__class__(
            **{**asdict(self), **values}
        )

    # ------------------------------------------------------------------ #
    # serialization / validation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def pretty(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.pretty() + '\n', encoding='utf-8')

    def validate(self) -> None:
        errors: List[str] = []

        if self.mode not in _MODES:
            errors.append(f'mode must be one of {_MODES}, got {self.mode!r}')
        if self.predict_mode not in _PREDICT_MODES:
            errors.append(f'predict_mode must be one of {_PREDICT_MODES}, got {self.predict_mode!r}')

        if self.batch_size < 1:
            errors.append(f'batch_size must be >= 1, got {self.batch_size}')
        if self.accum_steps < 1:
            errors.append(f'accum_steps must be >= 1, got {self.accum_steps}')
        if self.num_workers < 0:
            errors.append(f'num_workers must be >= 0, got {self.num_workers}')
        if self.num_epochs < 1:
            errors.append(f'num_epochs must be >= 1, got {self.num_epochs}')
        if not 0 <= self.freeze_epochs < self.num_epochs:
            errors.append(f'freeze_epochs must be in [0, num_epochs), got {self.freeze_epochs} vs num_epochs={self.num_epochs}')
        if self.unfreeze_from_layer < 0:
            errors.append(f'unfreeze_from_layer must be >= 0, got {self.unfreeze_from_layer}')
        if self.patience < 1:
            errors.append(f'patience must be >= 1, got {self.patience}')
        if self.head_lr <= 0 or self.encoder_lr <= 0 or self.embedding_lr <= 0:
            errors.append(
                f'lrs must be positive, got head_lr={self.head_lr}, '
                f'encoder_lr={self.encoder_lr}, embedding_lr={self.embedding_lr}'
            )
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            errors.append(f'warmup_ratio must be in [0, 1], got {self.warmup_ratio}')
        if self.lambda_rank < 0:
            errors.append(f'lambda_rank must be >= 0, got {self.lambda_rank}')
        if self.rank_margin <= 0:
            errors.append(f'rank_margin must be > 0, got {self.rank_margin}')
        if self.head_mode not in ('reg', 'softmax'):
            errors.append(f"head_mode must be 'reg' or 'softmax', got {self.head_mode!r}")
        if self.num_bins < 2:
            errors.append(f'num_bins must be >= 2, got {self.num_bins}')
        if self.ce_weight < 0:
            errors.append(f'ce_weight must be >= 0, got {self.ce_weight}')
        if self.bin_sigma <= 0:
            errors.append(f'bin_sigma must be > 0, got {self.bin_sigma}')
        if not 0 < self.test_size < 1:
            errors.append(f'test_size must be in (0, 1), got {self.test_size}')
        if self.n_splits < 2:
            errors.append(f'n_splits must be >= 2, got {self.n_splits}')

        if errors:
            raise ValueError('Invalid configuration:\n  ' + '\n  '.join(errors))