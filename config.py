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
ContextPool = Literal['mean', 'cls', 'mean+cls']
HeadFeatures = Literal['legacy', 'compact']
Phase1Schedule = Literal['linear', 'constant']
RankMarginMode = Literal['clamp', 'dynamic']
ConsistMode = Literal['pull', 'infonce']

_MODES = ('train80', 'train5', 'predict')
_PREDICT_MODES = ('single', '5fold')
_CONTEXT_POOLS = ('mean', 'cls', 'mean+cls')
_HEAD_FEATURES = ('legacy', 'compact')
_PHASE1_SCHEDULES = ('linear', 'constant')
_RANK_MARGIN_MODES = ('clamp', 'dynamic')
_CONSIST_MODES = ('pull', 'infonce')


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
    num_bins: int = 6       # ordinal bins, centers uniformly spaced over [SCORE_MIN, SCORE_MAX]
    # sentence-level context representation fed to each head:
    #   'mean'      = mean-pool over all tokens (original)
    #   'cls'       = ModernBERT [CLS] embedding (attention-condensed summary)
    #   'mean+cls'  = concatenation of both (richest; default)
    context_pool: str = 'mean+cls'
    head_features: str = 'compact'
    head_hidden: int = 128

    # === optimization ===
    batch_size: int = 32
    accum_steps: int = 1       # gradient accumulation (effective batch = batch_size * accum_steps)
    head_lr: float = 1e-4      # was 5e-4: heads were overfitting frozen features within ~5 epochs
    encoder_lr: float = 3e-6
    embedding_lr: float = 1e-5     # only the (new, random) marker embeddings
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    num_epochs: int = 12
    freeze_epochs: int = 4
    unfreeze_from_layer: int = 18
    warmup_ratio: float = 0.15
    # Phase-1 (frozen encoder) LR schedule:
    #   'constant' = warmup then HOLD (markers keep learning; no wasted near-zero tail)
    #   'linear'   = old behaviour: warmup then decay to 0 at the end of phase 1
    phase1_schedule: Phase1Schedule = 'constant'
    loss_type: str = 'mse_ccc'
    ccc_weight: float = 0.7
    # Small-variance batches make 1-CCC saturate at ~1 (denominator explodes);
    # a variance floor keeps the gradient informative on tiny batches.
    ccc_var_floor: float = 0.05
    lambda_rank: float = 0.5     # 0 = off; >0 adds pairwise margin-ranking to the loss
    rank_margin: float = 0.5
    #   'dynamic' = hinge margin = target gap (ClampTérmino pushes extreme pairs hardest)
    #   'clamp'   = hinge margin capped at rank_margin (old behaviour)
    rank_margin_mode: RankMarginMode = 'dynamic'
    # 0 = off; >0 reweights MSE/CCC by 1/(1+alpha*ModStd/HeadStd) so
    # high-disagreement (ambiguous) rows count for less.
    loss_std_alpha: float = 0.0
    # Compound-consistency self-supervised loss (data enrichment, label-free):
    # pulls together the span representations of the SAME compound across its
    # contexts so the model learns "same MWE = same concept". 0 = off.
    lambda_consist: float = 0.0
    #   'pull'     = centroid-variance (recommended; no negatives in tiny batches)
    #   'infonce'  = contrastive with same-compound positive views
    consist_mode: ConsistMode = 'pull'
    consist_temp: float = 0.1        # temperature for consist_mode='infonce'
    # data augmentation that PRESERVES labels (masked spans kept intact):
    # random word-drop / tail-crop of non-marker context words as a fraction
    # of training draws. Tokenize cache is disabled while augment_prob > 0.
    augment_prob: float = 0.0
    # Label-free auxiliary rows (ContextID/Compound/Mod/Head/Context, no
    # ModAvg/HeadAvg), e.g. the NCTTI consistency set. Appended to the train
    # loader: supervised terms ignore them (has_label=False), the compound-
    # consistency term uses them for more cross-context pull. '' = off.
    aux_data_path: str = ''
    ce_weight: float = 0.0     # 0 = off; >0 adds Gaussian soft-target CE on the ordinal bins
    bin_sigma: float = 0.5     # std (in bin units) of the Gaussian soft target
    use_label_std: bool = True # per-sample Gaussian width from ModStd/HeadStd when available; falls back to bin_sigma
    amp_init_scale: float = 1024.0   # GradScaler starting scale (safe operating point for FP16 embeddings)
    amp_growth_interval: int = 256   # clean steps before attempting scale growth (prevents rapid overflow oscillation)
    patience: int = 4
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
        if self.context_pool not in _CONTEXT_POOLS:
            errors.append(
                f"context_pool must be one of {_CONTEXT_POOLS}, got {self.context_pool!r}"
            )
        if self.head_features not in _HEAD_FEATURES:
            errors.append(
                f"head_features must be one of {_HEAD_FEATURES}, got {self.head_features!r}"
            )
        if self.head_hidden < 1:
            errors.append(f'head_hidden must be >= 1, got {self.head_hidden}')
        if self.phase1_schedule not in _PHASE1_SCHEDULES:
            errors.append(
                f"phase1_schedule must be one of {_PHASE1_SCHEDULES}, got {self.phase1_schedule!r}"
            )
        if self.rank_margin_mode not in _RANK_MARGIN_MODES:
            errors.append(
                f"rank_margin_mode must be one of {_RANK_MARGIN_MODES}, got {self.rank_margin_mode!r}"
            )
        if self.consist_mode not in _CONSIST_MODES:
            errors.append(
                f"consist_mode must be one of {_CONSIST_MODES}, got {self.consist_mode!r}"
            )
        if self.consist_temp <= 0:
            errors.append(f'consist_temp must be > 0, got {self.consist_temp}')
        if self.lambda_consist < 0:
            errors.append(f'lambda_consist must be >= 0, got {self.lambda_consist}')
        if self.loss_std_alpha < 0:
            errors.append(f'loss_std_alpha must be >= 0, got {self.loss_std_alpha}')
        if self.ccc_var_floor < 0:
            errors.append(f'ccc_var_floor must be >= 0, got {self.ccc_var_floor}')
        if not 0 <= self.augment_prob < 1:
            errors.append(f'augment_prob must be in [0, 1), got {self.augment_prob}')
        if self.num_bins < 2:
            errors.append(f'num_bins must be >= 2, got {self.num_bins}')
        if self.ce_weight < 0:
            errors.append(f'ce_weight must be >= 0, got {self.ce_weight}')
        if self.bin_sigma <= 0:
            errors.append(f'bin_sigma must be > 0, got {self.bin_sigma}')
        if self.amp_init_scale <= 0:
            errors.append(f'amp_init_scale must be > 0, got {self.amp_init_scale}')
        if self.amp_growth_interval < 1:
            errors.append(f'amp_growth_interval must be >= 1, got {self.amp_growth_interval}')
        if not 0 < self.test_size < 1:
            errors.append(f'test_size must be in (0, 1), got {self.test_size}')
        if self.n_splits < 2:
            errors.append(f'n_splits must be >= 2, got {self.n_splits}')

        if errors:
            raise ValueError('Invalid configuration:\n  ' + '\n  '.join(errors))