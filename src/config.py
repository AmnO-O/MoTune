"""Typed configuration for the gauss-only mmBERT compositionality pipeline.

The `Config` dataclass is the single source of truth for every hyperparameter,
path and mode of operation. It is built from a YAML or JSON file plus
`--set key=value` CLI overrides, validated, and always serialized next to the
run artifacts for reproducibility. Only the gauss head path is supported here
(no reg/softmax heads, no MLM warmup phase).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

Mode = Literal['train80']
RankMarginMode = Literal['clamp', 'dynamic']

_MODES = ('train80',)
_RANK_MARGIN_MODES = ('clamp', 'dynamic')


@dataclass
class Config:
    """All knobs used by the gauss-only training pipeline."""

    # === run ===
    mode: Mode = 'train80'
    seed: int = 42

    # === model ===
    backbone: str = 'jhu-clsp/mmBERT-base'
    hidden_size: int = 768
    # Every output uses a dedicated intermediate exit. hidden_states[0] is the
    # embedding output, so 19/20/21,22 select transformer blocks 18/19/20,21.
    # The PV exit predicts one overall distribution from Base and Particle.
    gauss_ctx_mod: Tuple[int, ...] = (19,)       # modifier exit: block 18
    gauss_ctx_head: Tuple[int, ...] = (20,)      # head exit: block 19
    gauss_ctx_pv: Tuple[int, ...] = (21, 22)     # PV exit: blocks 20--21
    head_hidden: int = 128
    dropout: float = 0.2
    # literality feature (ALWAYS ON, no knob): cosine between the contextualised
    # USE embedding of a constituent span and the static prototype (base/lemma)
    # embedding row of its own tokens. High = word keeps its literal meaning in
    # context (e.g. "market" in "flea market"); low = drift/lexicalised (e.g.
    # "tower" in "ivory tower"). The cos column is appended to the head input
    # unconditionally in build_model (head_in += 1).

    # === data / paths (filenames are resolved under data_path) ===
    data_path: Optional[str] = None
    output_dir: Optional[str] = None
    max_context_length: int = 256

    # Train datasets (EN / DE, NN + PV)
    en_nn_train: str = 'en-nn-train.tsv'
    de_nn_train: str = 'de-nn-train.tsv'
    en_pv_train: str = 'en-pv-train.tsv'
    de_pv_train: str = 'de-pv-train.tsv'          # German PV joins the mix by default
    #                                (trennbare Verben, e.g. abhauen; mod=verb, head=particle).
    #                                _match_german_pv locates 100% of spans; ~65% of rows are
    #                                non-degenerate (particle detached) and reach supervised
    #                                losses, the fused one-token rows ("abgehauen") stay masked
    #                                (representation-only). Turn off with --set de_pv_train=.

    # Multi-task Trial Datasets (EN / DE)
    en_nn_trial: str = 'en-nn-trial.tsv'
    de_nn_trial: str = 'de-nn-trial.tsv'
    en_pv_trial: str = 'en-pv-trial.tsv'
    de_pv_trial: str = 'de-pv-trial.tsv'

    # === scoring phases (encoder frozen, then LoRA) ===
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
    lora_from_layer: int = 18

    # === optimization ===
    batch_size: int = 32
    accum_steps: int = 1
    head_lr: float = 1e-4
    encoder_lr: float = 8e-6
    embedding_lr: float = 0.0      # 0 = frozen (mmBERT embedding table is ~197M)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    amp_init_scale: float = 1024.0
    amp_growth_interval: int = 256
    patience: int = 4
    num_workers: int = 2

    # === losses ===
    ccc_weight: float = 0.7
    ccc_var_floor: float = 0.05
    lambda_rank: float = 0.5
    rank_margin: float = 0.5
    rank_margin_mode: RankMarginMode = 'dynamic'
    bin_sigma: float = 0.5
    use_label_std: bool = True

    # === split ===
    test_size: float = 0.2

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
        tuple_fields = {f.name for f in fields(cls) if 'Tuple' in str(f.type)}
        for name in tuple_fields:
            if isinstance(safe.get(name), list):
                safe[name] = tuple(safe[name])
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

    def build_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.backbone)

    def validate(self) -> None:
        errors: List[str] = []

        if self.mode not in _MODES:
            errors.append(f'mode must be one of {_MODES}, got {self.mode!r}')
        if self.rank_margin_mode not in _RANK_MARGIN_MODES:
            errors.append(
                f'rank_margin_mode must be one of {_RANK_MARGIN_MODES}, got {self.rank_margin_mode!r}'
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

        if self.lora_rank < 1:
            errors.append('lora_rank must be >= 1')
        if self.lora_alpha < 1:
            errors.append('lora_alpha must be >= 1')
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
        if not 0 <= self.grad_clip:
            errors.append(f'grad_clip must be >= 0, got {self.grad_clip}')

        if self.lambda_rank < 0:
            errors.append(f'lambda_rank must be >= 0, got {self.lambda_rank}')
        if self.rank_margin <= 0:
            errors.append(f'rank_margin must be > 0, got {self.rank_margin}')
        if self.ccc_var_floor < 0:
            errors.append(f'ccc_var_floor must be >= 0, got {self.ccc_var_floor}')
        if self.bin_sigma <= 0:
            errors.append(f'bin_sigma must be > 0, got {self.bin_sigma}')
        if self.amp_init_scale <= 0:
            errors.append(f'amp_init_scale must be > 0, got {self.amp_init_scale}')
        if self.amp_growth_interval < 1:
            errors.append(f'amp_growth_interval must be >= 1, got {self.amp_growth_interval}')
        if not 0 < self.test_size < 1:
            errors.append(f'test_size must be in (0, 1), got {self.test_size}')

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
    if 'List' in type_ or 'list' in type_ or 'Tuple' in type_ or 'tuple' in type_:
        return [x.strip() for x in raw.split(',') if x.strip()]
    if type_ == 'int':
        return int(raw)
    if type_ == 'float':
        return float(raw)
    if type_ == 'bool':
        return raw.strip().lower() in ('1', 'true', 'yes', 'on')
    return raw
