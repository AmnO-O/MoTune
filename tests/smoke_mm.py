"""Local smoke checks for the mmBERT rebuild -- NO torch / transformers needed.

Run from the repo root:
    python tests/smoke_mm.py        (or: python run.py smoke)

Covers: syntax of every module, config construction/validation/round-trip and
``--set`` coercion, CLI wiring, and the offset-based span matcher on synthetic
token offsets (the real-tokenizer alignment check runs on Kaggle in M2).
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    status = 'OK' if condition else 'FAIL'
    print(f'  [{status}] {label}')
    if not condition:
        FAILURES.append(label)


def sync_parse() -> None:
    print('=== 1. SYNTAX (run.py + mm/*.py) ===')
    files = [ROOT / 'run.py', ROOT / 'tests' / 'smoke_mm.py'] + sorted((ROOT / 'mm').glob('*.py'))
    for path in files:
        try:
            ast.parse(path.read_text(encoding='utf-8'))
            check(True, str(path.relative_to(ROOT)))
        except SyntaxError as exc:
            check(False, f'{path.name}: {exc.msg} @ {exc.lineno}')


def check_config() -> None:
    print('=== 2. CONFIG (defaults / validation / round-trip / --set) ===')
    sys.path.insert(0, str(ROOT))
    from mm.config import Config, coerce_value

    defaults = Config.defaults()
    defaults.validate()
    check(defaults.backbone == 'jhu-clsp/mmBERT-base', f'default backbone = mmBERT-base')
    check(defaults.total_epochs == defaults.freeze_epochs + defaults.lora_epochs,
          'total_epochs == freeze_epochs + lora_epochs')

    bad = [
        ('lambda_rank=-1', lambda: Config.defaults().update(lambda_rank=-1)),
        ('head_pool=bogus', lambda: Config.defaults().update(head_pool='bogus')),
        ('mlm_mask_span=bogus', lambda: Config.defaults().update(mlm_mask_span='bogus')),
        ('freeze_epochs=-1', lambda: Config.defaults().update(freeze_epochs=-1)),
        ('lora_epochs=0', lambda: Config.defaults().update(lora_epochs=0)),
        ('warmup without data', lambda: Config.defaults().update(warmup_mlm_epochs=3)),
        ('embedding_lr=-1', lambda: Config.defaults().update(embedding_lr=-1)),
        ('unknown_key', lambda: Config.defaults().update(unknown_key=1)),
    ]
    for label, fn in bad:
        try:
            fn().validate()
            check(False, f'[reject] {label}')
        except ValueError:
            check(True, f'[reject] {label}')

    ok = Config.defaults().update(
        warmup_mlm_epochs=2, mlm_data_paths=['en-nn-train.tsv', 'de-nn-train.tsv'],
        head_pool='mean', lambda_compound=0.3,
    )
    ok.validate()
    check(ok.warmup_mlm_epochs == 2 and ok.lambda_compound == 0.3,
          'valid warmup config accepted')
    check(ok.total_epochs == ok.freeze_epochs + ok.lora_epochs,
          'total_epochs derived correctly')

    # JSON round-trip
    tmp = Path(tempfile.gettempdir()) / 'mm_cfg_smoke.json'
    defaults.save(tmp)
    check(Config.load(tmp).to_dict() == defaults.to_dict(), 'config JSON round-trip')
    tmp.unlink()

    # YAML path: works if PyYAML is installed, else must fail with a clear error
    tmp = Path(tempfile.gettempdir()) / 'mm_cfg_smoke.yaml'
    tmp.write_text('seed: 7\nfreeze_epochs: 2\nlora_epochs: 4\nmlm_data_paths:\n  - en-nn-train.tsv\n', encoding='utf-8')
    try:
        from mm.config import Config as C2
        y = C2.load(tmp)
        check(y.seed == 7 and y.total_epochs == 6, 'config YAML round-trip')
        check((y.mlm_data_paths == ['en-nn-train.tsv']), 'YAML list field parsed')
    except ValueError as exc:
        check('PyYAML' in str(exc), f'YAML config without PyYAML raises clear error')
    tmp.unlink()

    # --set coercion
    check(coerce_value('mlm_data_paths', 'a.tsv,b.tsv') == ['a.tsv', 'b.tsv'], 'coerce List')
    check(coerce_value('freeze_epochs', '3') == 3, 'coerce int')
    check(coerce_value('embedding_lr', '0.0') == 0.0, 'coerce float')
    check(coerce_value('use_label_std', 'false') is False, 'coerce bool')
    check(coerce_value('backbone', 'x/y') == 'x/y', 'str passes through')
    check(coerce_value('not_a_field', '1') == '1', 'unknown key passes through (validated later)')


def check_cli() -> None:
    print('=== 3. CLI WIRING (run.py) ===')
    sys.path.insert(0, str(ROOT))
    import run as cli
    from mm.config import Config

    tmp_cfg = Path(tempfile.gettempdir()) / 'mm_cli_smoke.json'
    tmp_cfg.write_text('{"head_pool": "mean", "freeze_epochs": 5}', encoding='utf-8')

    args = cli._build_parser().parse_args([
        'train5',
        '--config', str(tmp_cfg),
        '--set', 'freeze_epochs=2', '--set', 'lora_epochs=6',
        '--set', 'mlm_data_paths=de-nn-train.tsv,nctti_en.tsv',
        '--set', 'use_label_std=false',
        '--seed', '7',
    ])
    overrides = cli._set_overrides(args)
    cfg = cli._merge(Config.defaults(), args)

    check(cfg.head_pool == 'mean', 'config file applied on top of defaults')
    check(cfg.freeze_epochs == 2 and cfg.lora_epochs == 6 and cfg.total_epochs == 8,
          'CLI --set ints map onto Config')
    check(cfg.mlm_data_paths == ['de-nn-train.tsv', 'nctti_en.tsv'], 'CLI --set lists map')
    check(cfg.use_label_std is False, 'CLI --set bool maps')
    check(cfg.seed == 7 and cfg.mode == 'train5', 'CLI flags map (seed, mode)')
    check(str(args.config) == str(tmp_cfg), '--config parsed')
    tmp_cfg.unlink()

    bad = cli._build_parser().parse_args(['train5', '--set', 'no_equals'])
    try:
        cli._set_overrides(bad)
        check(False, '--set without = rejected')
    except ValueError:
        check(True, '--set without = rejected')

    # non-training commands must not set cfg.mode (validation only allows
    # train modes) - regressed on Kaggle when probe hit Invalid configuration
    for cmd in ('probe', 'warmup', 'smoke'):
        a = cli._build_parser().parse_args([cmd])
        merged = cli._merge(Config.defaults(), a)
        merged.validate()
        check(merged.mode in ('train80', 'train5', 'predict'),
              f'{cmd} leaves a valid cfg.mode')


def _word_offsets(text: str):
    """Synthetic single-token-per-word offset map for unit tests."""
    out = []
    pos = 0
    for w in text.split(' '):
        s = text.index(w, pos)
        out.append((s, s + len(w)))
        pos = s + len(w)
    return out


def check_marks() -> None:
    print('=== 4. OFFSET SPAN MATCHER (synthetic token offsets) ===')
    sys.path.insert(0, str(ROOT))
    from mm.marks import Span, find_spans

    def tp(sp: Span):
        return (sp.start, sp.end) if sp is not None else None

    # 1) basic contiguous EN compound
    r = find_spans('the night watch is useful', _word_offsets('the night watch is useful'),
                   'night', 'watch')
    check(r.found and tp(r.mod) == (1, 2) and tp(r.head) == (2, 3) and r.adjacent,
          'EN spaced compound -> mod(1,2) head(2,3) adjacent')
    check(not r.degenerate, 'EN spaced compound not degenerate')

    # 2) head inflection (plural)
    r = find_spans('the night watches are', _word_offsets('the night watches are'),
                   'night', 'watch')
    check(r.found and tp(r.head) == (2, 3) and r.adjacent, 'head inflection "watches" matched')

    # 3) head possessive
    r = find_spans("the night watch's value", _word_offsets("the night watch's value"),
                   'night', 'watch')
    check(r.found and tp(r.head) == (2, 3) and r.adjacent, "head possessive \"watch's\" matched")

    # 4) German closed compound, single token
    text = 'Das Abiturzeugnis ist gut'
    r = find_spans(text, _word_offsets(text), 'Abitur', 'Zeugnis')
    check(r.found and tp(r.mod) == (1, 2) and tp(r.head) == (1, 2) and r.adjacent,
          'German closed compound aligned inside one token')
    check(r.degenerate, 'collapse to one token flagged degenerate')

    # 5) German compound, tokenizer SPLITS it -> two adjacent tokens
    text = 'das abitur zeugnis ist'
    r = find_spans(text, _word_offsets(text), 'abitur', 'zeugnis')
    check(r.found and tp(r.mod) == (1, 2) and tp(r.head) == (2, 3) and r.adjacent,
          'German split compound matched as adjacent tokens')

    # 6) non-contiguous (head appears later, unrelated word between)
    r = find_spans('the night sky, watch quietly', _word_offsets('the night sky, watch quietly'),
                   'night', 'watch')
    check(r.found and tp(r.mod) == (1, 2) and tp(r.head) == (3, 4) and not r.adjacent,
          'non-contiguous compound falls back to ordered search')

    # 7) no match -> found=False, no crash on empty spans
    r = find_spans('the night is dark', _word_offsets('the night is dark'), 'banana', 'watch')
    check(not r.found and r.mod.start is None and r.head.start is None,
          'unmatched compound returns empty spans')

    # 8) boundary: "watch" must NOT match inside "watchful"
    r = find_spans('the watchful eye', _word_offsets('the watchful eye'), 'night', 'watch')
    check(not r.found, '"watch" does not match the prefix of "watchful"')

    # 9) punctuation boundary after head
    r = find_spans('time watch.', _word_offsets('time watch.'), 'time', 'watch')
    check(r.found and tp(r.head) == (1, 2) and r.adjacent, 'head at punctuation boundary')

    # 10) capitalization is case-insensitive
    r = find_spans('The Night Watch Is Long', _word_offsets('The Night Watch Is Long'),
                   'night', 'watch')
    check(r.found and tp(r.mod) == (1, 2) and tp(r.head) == (2, 3), 'case-insensitive matching')

    # 11) German 'ß' must NOT expand to 'ss' ("Großstadt" keeps char positions):
    #     casefold would shift offsets and corrupt the token mapping.
    text = 'Die Großstadt Berlin'
    r = find_spans(text, _word_offsets(text), 'Groß', 'stadt')
    check(r.found and tp(r.mod) == (1, 2) and tp(r.head) == (1, 2) and r.degenerate,
          'German "ß" preserves length (no casefold expansion)')
    r = find_spans(text, _word_offsets(text), 'groß', 'Stadt')
    check(r.found and tp(r.mod) == (1, 2) and tp(r.head) == (1, 2),
          '"ß" lowercase input still matches (case-insensitive)')


def check_data() -> None:
    print('=== 5. DATA LOADERS (real local TSVs, no torch) ===')
    sys.path.insert(0, str(ROOT))
    import random
    from mm.config import Config
    from mm.data import _df_to_rows, load_labeled, read_tsv
    from mm.marks import find_spans

    rows_en = _df_to_rows(read_tsv('dataset/en-nn-train.tsv'), 'en-nn', 'en')
    check(len(rows_en) == 3480 and rows_en[0]['has_label'],
          f'en-nn: {len(rows_en)} rows, labeled')

    rows_de = _df_to_rows(read_tsv('dataset/de-nn-train.tsv'), 'de-nn', 'de')
    check(rows_de[0]['compound'] == 'Abiturzeugnis' and rows_de[0]['mod'] == 'Abitur'
          and rows_de[0]['lang'] == 'de',
          'de-nn compound/mod/head/lang parsed')

    rows_pv = _df_to_rows(read_tsv('dataset/en-pv-train.tsv'), 'en-pv', 'en')
    check(rows_pv[0]['mod'] == 'crack' and rows_pv[0]['head'] == 'down'
          and rows_pv[0]['has_label'],
          'en-pv Base/Particle -> mod/head + label')

    rows_nctti = _df_to_rows(read_tsv('dataset/nctti_en.tsv'), 'nctti_en', 'en')
    check(len(rows_nctti) == 539 and not rows_nctti[0]['has_label'],
          'nctti aux: label-free')

    cfg = Config.defaults().update(data_path='dataset')
    labeled = load_labeled(cfg)
    n_compounds = len({r['compound_id'] for r in labeled})
    check(len(labeled) == 3480 and n_compounds > 100,
          f'load_labeled: {len(labeled)} rows / {n_compounds} compound ids')

    def synth_offsets(context: str):
        out, pos = [], 0
        for w in context.split(' '):
            if pos >= len(context):
                break
            s = context.find(w, pos)
            if s < 0:
                break
            out.append((s, s + len(w)))
            pos = s + len(w)
        return out

    def alignment(rows, sample):
        found = deg = 0
        for r in rows[:sample]:
            res = find_spans(r['context'], synth_offsets(r['context']), r['mod'], r['head'])
            found += bool(res.found)
            deg += bool(res.degenerate)
        return found, deg

    n = min(len(rows_en), 2000)
    found, _ = alignment(rows_en, n)
    check(found / n > 0.4, f'en-nn synthetic-offset alignment {found}/{n} ({100*found/n:.0f}%)')

    gn = min(len(rows_de), 2000)
    gfound, gdeg = alignment(rows_de, gn)
    check(gfound / gn > 0.2,
          f'de-nn synthetic-offset alignment {gfound}/{gn} ({100*gfound/gn:.0f}%)')
    check(gdeg > 0, f'de-nn: {gdeg} German closed compounds flagged degenerate')

    # collate_mlm must pad *labels* with -100 (ignore_index), never _PAD_ID=0:
    # padded positions would otherwise contribute CrossEntropy against token 0.
    dl_src = (ROOT / 'mm' / 'data.py').read_text(encoding='utf-8')
    tree = ast.parse(dl_src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == 'collate_mlm')
    seg = ast.get_source_segment(dl_src, fn)
    check("pad_val = -100 if key == 'labels' else _PAD_ID" in seg,
          'collate_mlm: labels padded with -100 (MLM loss-leak guard)')


def check_folds() -> None:
    print('=== 6. FOLDS + GROUP SAMPLER (real TSVs, no torch) ===')
    sys.path.insert(0, str(ROOT))
    from mm.data import _df_to_rows, read_tsv
    from mm.folds import CompoundGroupSampler, assign_folds

    rows = _df_to_rows(read_tsv('dataset/en-nn-train.tsv'), 'en-nn', 'en')
    for r in rows:
        r['compound_id'] = -1

    folded = assign_folds(rows, n_splits=5, seed=42)
    folds = {r['compound']: r['fold'] for r in folded}
    check(0 <= min(folds.values()) and max(folds.values()) <= 4,
          'fold ids in range')
    check(len(set(folds.values())) == 5, 'all 5 folds used')
    per_fold = {}
    for r in folded:
        per_fold[r['fold']] = per_fold.get(r['fold'], 0) + 1
    mx, mn = max(per_fold.values()), min(per_fold.values())
    check(mx - mn <= len(folded) // 2,
          f'fold sizes balanced: {per_fold}')
    # compound exclusivity is inherent (fold is per compound); verify deterministic
    again = assign_folds(rows, n_splits=5, seed=42)
    check([r['fold'] for r in again] == [r['fold'] for r in folded],
          'deterministic given the same seed')
    # every compound appears with a consistent fold even after shuffle of input
    import random as _r
    rng = _r.Random(3)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    re_shuffled = assign_folds(shuffled, n_splits=5, seed=42)
    folds2 = {r['compound']: r['fold'] for r in re_shuffled}
    check(folds == folds2, 'fold assignment invariant to row order')

    # sampler: rows of each compound are emitted together (batchable dense pairs)
    n_lab = [i for i, r in enumerate(rows) if r['has_label']]
    labels_n = {r['compound'] for r in rows if r['has_label']}
    codes = {c: v for v, c in enumerate(sorted(labels_n))}
    cids = [codes[r['compound']] if r['has_label'] else -1 for r in rows]
    samp = CompoundGroupSampler(cids, batch_size=32, seed=7)
    order = list(samp)
    check(len(order) == len(rows), 'sampler yields every row once')
    # compound -1 rows may be sprinkled between groups; check labeled groups
    # still come in contiguous runs
    runs = {}
    for idx in order:
        c = cids[idx]
        if c < 0:
            continue
        if c not in runs:
            runs[c] = [idx]
        elif cids[order[order.index(idx) - 1]] == c:
            runs[c].append(idx)
    check(sum(len(v) for v in runs.values()) == sum(1 for c in cids if c >= 0),
          'all labeled rows reach their compound group')
    samp.set_epoch(1)
    order2 = list(samp)
    check(order != order2, 'set_epoch changes the shuffle')


def check_fixes() -> None:
    print('=== 7. BUG FIXES & CONTRACTS ===')
    sys.path.insert(0, str(ROOT))
    import ast
    import numpy as np
    from mm.config import Config

    # 1) Verify freeze_phase1 / unfreeze_phase2 methods exist on Trainer via AST
    trainer_ast = ast.parse((ROOT / 'mm' / 'trainer.py').read_text(encoding='utf-8'))
    methods = [n.name for n in ast.walk(trainer_ast) if isinstance(n, ast.FunctionDef)]
    check('_freeze_phase1' in methods and '_unfreeze_phase2' in methods,
          'Trainer has two-phase freeze/unfreeze methods')

    # 2) Verify base_model bypass exists in MMBertRegressor forward (kept as a
    # non-registered attribute so state_dict/optimizer are not duplicated)
    model_src = (ROOT / 'mm' / 'model.py').read_text(encoding='utf-8')
    check("object.__setattr__(self, 'base_model', _backbone(self))" in model_src
          and 'outputs = self.base_model(' in model_src,
          'MMBertRegressor bypasses MLM head when with_logits is False')

    # 3) Verify compound_consistency_loss casts rep to float and guards infonce singletons
    loss_src = (ROOT / 'mm' / 'losses.py').read_text(encoding='utf-8')
    check('rep = torch.cat([mod_emb, head_emb], dim=-1).float()' in loss_src,
          'compound_consistency_loss casts rep to float for AMP compatibility')
    check('& (compound_ids[:, None] >= 0)' in loss_src,
          'margin_rank_loss prevents spurious -1 compound pairs')

    # 4) Verify the LoRA layer-window knob is wired end to end
    cfg = Config()
    check(getattr(cfg, 'lora_from_layer', 0) == 18,
          'config exposes lora_from_layer defaulting to 18')
    model_src2 = (ROOT / 'mm' / 'model.py').read_text(encoding='utf-8')
    check("def apply_lora(model: nn.Module, rank: int = 8, alpha: int = 16,"
          "\n               dropout: float = 0.1, targets: Optional[List[str]] = None,"
          "\n               from_layer: int = 0) -> List[LoRAAdapter]:" in model_src2,
          'apply_lora accepts from_layer window')
    pipe_src = (ROOT / 'mm' / 'pipeline.py').read_text(encoding='utf-8')
    check('from_layer=cfg.lora_from_layer' in pipe_src,
          'pipeline passes lora_from_layer to apply_lora')
    tr_src = (ROOT / 'mm' / 'trainer.py').read_text(encoding='utf-8')
    check('from_layer=self.cfg.lora_from_layer' in tr_src,
          'trainer passes lora_from_layer to apply_lora')


def main() -> int:
    sync_parse()
    check_config()
    check_cli()
    check_marks()
    check_data()
    check_folds()
    check_fixes()

    print('=' * 50)
    if FAILURES:
        print(f'RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}')
        return 1
    print('RESULT: all smoke checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())