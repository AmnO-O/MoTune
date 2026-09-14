"""Typed configuration for the mmBERT compositionality pipeline.

The `Config` dataclass is the single source of truth for every hyperparameter,
path and mode of operation. It is built from a YAML or JSON file plus
`--set key=value` CLI overrides, validated, and always serialized next to the
run artifacts for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

Mode = Literal['train80', 'train5', 'predict']
PredictMode = Literal['single', '5fold']
ContextPool = Literal['mean', 'cls', 'mean+cls']
HeadPool = Literal['mean', 'attn']
HeadMode = Literal['reg', 'softmax']
RankMarginMode = Literal['clamp', 'dynamic']
ConsistMode = Literal['pull', 'infonce']
MlmMaskSpan = Literal['one', 'both']
Phase1Schedule = Literal['constant', 'linear']

_MODES = ('train80', 'train5', 'predict')
_PREDICT_MODES = ('single', '5fold')
_CONTEXT_POOLS = ('mean', 'cls', 'mean+cls')
_HEAD_POOLS = ('mean', 'attn')
_HEAD_MODES = ('reg', 'softmax')
_RANK_MARGIN_MODES = ('clamp', 'dynamic')
_CONSIST_MODES = ('pull', 'infonce')
_MLM_MASK_SPANS = ('one', 'both')
_PHASE1_SCHEDULES = ('constant', 'linear')


@dataclass
class Config:
    """All knobs used by the training and prediction pipelines."""

    # === run ===
    mode: Mode = 'train5'
    seed: int = 42

    # === model ===
    backbone: str = 'jhu-clsp/mmBERT-base'
    hidden_size: int = 768
    # span pooling: 'mean' = mean-pool over span tokens; 'attn' = learned
    # attention pool (recommended: a compound is usually 1-2 tokens)
    head_pool: HeadPool = 'attn'
    context_pool: ContextPool = 'mean+cls'
    # output head: 'reg' = scalar regression; 'softmax' = ordinal bins -> E[Y]
    head_mode: HeadMode = 'reg'
    num_bins: int = 6
    head_hidden: int = 128
    dropout: float = 0.2
    # attach LM-predictability features (avg log P + span entropy of the span
    # tokens under the pretrained MLM head) to the head input. 0 = off.
    use_lm_features: bool = False

    # === data / paths (filenames are resolved under data_path) ===
    data_path: Optional[str] = None
    output_dir: Optional[str] = None
    max_context_length: int = 256
    max_mlm_length: int = 128
    train_file: str = 'en-nn-train.tsv'
    trial_file: str = 'en-nn-trial.tsv'
    # label-free rows appended to the scoring loader (consistency signal only)
    aux_data_paths: List[str] = field(default_factory=list)
    # label-free sentences for the compound-aware MLM warmup (any language)
    mlm_data_paths: List[str] = field(default_factory=list)

    # === phase 0 -- compound-aware MLM warmup (warmup_mlm_epochs > 0 enables) ===
    warmup_mlm_epochs: int = 0
    # 'one' = mask modifier OR head (keeps the other word = the pairing signal)
    # 'both' = mask the whole compound
    mlm_mask_span: MlmMaskSpan = 'both'
    mlm_mask_prob: float = 0.8
    mlm_random_prob: float = 0.1
    warmup_batch_size: int = 32
    warmup_lr: float = 5e-5
    warmup_warmup_ratio: float = 0.1
    # LoRA adapter used during the warmup; MERGED into the base afterwards
    warmup_lora_rank: int = 8
    warmup_lora_alpha: int = 16
    warmup_lora_dropout: float = 0.1

    # === phase 1 -- scoring (encoder frozen, then LoRA) ===
    freeze_epochs: int = 3
    lora_epochs: int = 9
    # A/B escape hatch: also fully unfreeze top layers from this index (0 = off)
    unfreeze_from_layer: int = 0
    # LoRA adapter used during scoring (fresh rank, trained on the spot)
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    # Apply LoRA only to layer index >= this (0 = all 22 layers of mmBERT);
    # top layers carry the compositional semantics.
    lora_from_layer: int = 14

    # === optimization ===
    batch_size: int = 32
    accum_steps: int = 1
    head_lr: float = 1e-4
    encoder_lr: float = 8e-6
    embedding_lr: float = 0.0      # 0 = frozen (mmBERT embedding table is ~197M)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_ratio: float = 0.15
    phase1_schedule: Phase1Schedule = 'constant'
    amp_init_scale: float = 1024.0
    amp_growth_interval: int = 256
    patience: int = 4
    num_workers: int = 2

    # === losses ===
    ccc_weight: float = 0.7
    ccc_var_floor: float = 0.05
    loss_std_alpha: float = 0.0
    lambda_rank: float = 0.5
    rank_margin: float = 0.5
    rank_margin_mode: RankMarginMode = 'dynamic'
    lambda_consist: float = 0.0
    consist_mode: ConsistMode = 'pull'
    consist_temp: float = 0.1
    # compound-center calibration: MSE between predicted and gold compound
    # centroids per batch (teaches between-compound ranking). 0 = off.
    lambda_compound: float = 0.0
    ce_weight: float = 0.0
    bin_sigma: float = 0.5
    use_label_std: bool = True

    # === split ===
    n_splits: int = 5
    test_size: float = 0.2

    # === predict ===
    predict_mode: PredictMode = '5fold'

    # ------------------------------------------------------------------ #
    # derived
    # ------------------------------------------------------------------ #
    @property
    def total_epochs(self) -> int:
        return self.freeze_epochs + self.lora_epochs

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def defaults(cls) -> 'Config':
        return cls()

    @classmethod
    def from_dict(cls, values: Dict[str, Any], strict: bool = False) -> 'Config':
        known = {f.name for f in fields(cls)}
        extra = set(values) - known
        if extra and strict:
            raise ValueError(f'Unknown config keys: {sorted(extra)}')
        safe = {k: v for k, v in values.items() if k in known and v is not None}
        return replace(cls.defaults(), **safe)

    @classmethod
    def load(cls, path: str | Path) -> 'Config':
        path = Path(path)
        if path.suffix.lower() in ('.yml', '.yaml'):
            try:
                import yaml  # lazy: optional dependency
            except ImportError as exc:
                raise ValueError(
                    'PyYAML is required to load a .yaml config file '
                    '(pip install PyYAML, or use a .json config)'
                ) from exc
            raw = yaml.safe_load(path.read_text(encoding='utf-8'))
            if not isinstance(raw, dict):
                raise ValueError(f'YAML config must be a mapping, got {type(raw).__name__}')
            return cls.from_dict(raw)
        return cls.from_dict(json.loads(path.read_text(encoding='utf-8')))

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
        if self.head_pool not in _HEAD_POOLS:
            errors.append(f'head_pool must be one of {_HEAD_POOLS}, got {self.head_pool!r}')
        if self.context_pool not in _CONTEXT_POOLS:
            errors.append(f'context_pool must be one of {_CONTEXT_POOLS}, got {self.context_pool!r}')
        if self.head_mode not in _HEAD_MODES:
            errors.append(f'head_mode must be one of {_HEAD_MODES}, got {self.head_mode!r}')
        if self.ce_weight > 0 and self.head_mode != 'softmax':
            errors.append('ce_weight > 0 (Gaussian soft-target CE) requires head_mode="softmax"')
        if self.rank_margin_mode not in _RANK_MARGIN_MODES:
            errors.append(
                f'rank_margin_mode must be one of {_RANK_MARGIN_MODES}, got {self.rank_margin_mode!r}'
            )
        if self.consist_mode not in _CONSIST_MODES:
            errors.append(f'consist_mode must be one of {_CONSIST_MODES}, got {self.consist_mode!r}')
        if self.mlm_mask_span not in _MLM_MASK_SPANS:
            errors.append(
                f'mlm_mask_span must be one of {_MLM_MASK_SPANS}, got {self.mlm_mask_span!r}'
            )
        if self.phase1_schedule not in _PHASE1_SCHEDULES:
            errors.append(
                f'phase1_schedule must be one of {_PHASE1_SCHEDULES}, got {self.phase1_schedule!r}'
            )

        if self.batch_size < 1:
            errors.append(f'batch_size must be >= 1, got {self.batch_size}')
        if self.accum_steps < 1:
            errors.append(f'accum_steps must be >= 1, got {self.accum_steps}')
        if self.num_workers < 0:
            errors.append(f'num_workers must be >= 0, got {self.num_workers}')
        if not 0 <= self.freeze_epochs:
            errors.append(f'freeze_epochs must be >= 0, got {self.freeze_epochs}')
        if self.lora_epochs < 1:
            errors.append(f'lora_epochs must be >= 1, got {self.lora_epochs}')
        if self.warmup_mlm_epochs < 0:
            errors.append(f'warmup_mlm_epochs must be >= 0, got {self.warmup_mlm_epochs}')
        if self.warmup_mlm_epochs > 0 and not self.mlm_data_paths:
            errors.append('warmup_mlm_epochs > 0 requires mlm_data_paths to be non-empty')
        if not 0 <= self.mlm_mask_prob <= 1:
            errors.append(f'mlm_mask_prob must be in [0, 1], got {self.mlm_mask_prob}')
        if not 0 <= self.mlm_random_prob <= 1:
            errors.append(f'mlm_random_prob must be in [0, 1], got {self.mlm_random_prob}')
        if self.mlm_mask_prob + self.mlm_random_prob > 1:
            errors.append('mlm_mask_prob + mlm_random_prob must be <= 1')

        if self.lora_rank < 1 or self.warmup_lora_rank < 1:
            errors.append('lora_rank/r and warmup_lora_rank must be >= 1')
        if self.lora_alpha < 1 or self.warmup_lora_alpha < 1:
            errors.append('lora_alpha and warmup_lora_alpha must be >= 1')
        if self.unfreeze_from_layer < 0:
            errors.append(f'unfreeze_from_layer must be >= 0, got {self.unfreeze_from_layer}')

        if self.patience < 1:
            errors.append(f'patience must be >= 1, got {self.patience}')
        if self.head_lr <= 0 or self.encoder_lr <= 0:
            errors.append(
                f'lrs must be positive, got head_lr={self.head_lr}, encoder_lr={self.encoder_lr}'
            )
        if self.embedding_lr < 0:
            errors.append(f'embedding_lr must be >= 0, got {self.embedding_lr}')
        if self.warmup_lr <= 0:
            errors.append(f'warmup_lr must be > 0, got {self.warmup_lr}')
        if not 0 <= self.warmup_ratio <= 1:
            errors.append(f'warmup_ratio must be in [0, 1], got {self.warmup_ratio}')

        if self.lambda_rank < 0:
            errors.append(f'lambda_rank must be >= 0, got {self.lambda_rank}')
        if self.lambda_consist < 0:
            errors.append(f'lambda_consist must be >= 0, got {self.lambda_consist}')
        if self.lambda_compound < 0:
            errors.append(f'lambda_compound must be >= 0, got {self.lambda_compound}')
        if self.rank_margin <= 0:
            errors.append(f'rank_margin must be > 0, got {self.rank_margin}')
        if self.consist_temp <= 0:
            errors.append(f'consist_temp must be > 0, got {self.consist_temp}')
        if self.loss_std_alpha < 0:
            errors.append(f'loss_std_alpha must be >= 0, got {self.loss_std_alpha}')
        if self.ccc_var_floor < 0:
            errors.append(f'ccc_var_floor must be >= 0, got {self.ccc_var_floor}')
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


def coerce_value(name: str, raw: Any, cfg: type = Config) -> Any:
    """Coerce a raw CLI `--set` value to the Config field type.

    Strings are coerced to int/float/bool/List[str] when the field expects
    them; everything else passes through and is validated later by
    ``Config.validate``.
    """
    f = next((f for f in fields(cfg) if f.name == name), None)
    if f is None:
        return raw
    if not isinstance(raw, str):
        return raw
    type_ = f.type
    if 'List' in type_ or 'list' in type_:
        return [x.strip() for x in raw.split(',') if x.strip()]
    if type_ == 'int':
        return int(raw)
    if type_ == 'float':
        return float(raw)
    if type_ == 'bool':
        return raw.strip().lower() in ('1', 'true', 'yes', 'on')
    return raw