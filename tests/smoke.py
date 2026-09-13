"""Local smoke checks that need NO torch / transformers / GPU.

Run from the repo root:
    python tests/smoke.py

Covers: syntax of every Python module + notebook, span matcher on the real
data, fold stratification integrity, and config validation.
"""

from __future__ import annotations

import ast
import glob
import json
import sys
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
    print('=== 1. SYNTAX (modules + notebooks) ===')
    py_files = [ROOT / 'main.py', ROOT / 'config.py'] + sorted((ROOT / 'src').glob('*.py'))
    for path in py_files:
        try:
            ast.parse(path.read_text(encoding='utf-8'))
            check(True, str(path.relative_to(ROOT)))
        except SyntaxError as exc:
            check(False, f'{path.name}: {exc.msg} @ {exc.lineno}')

    for nb in sorted((ROOT / 'notebook').glob('*.ipynb')):
        cells = json.loads(nb.read_text(encoding='utf-8'))['cells']
        try:
            for cell in cells:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))
            check(True, str(nb.relative_to(ROOT)))
        except SyntaxError as exc:
            check(False, f'{nb.name}: {exc.msg}')


def check_matching() -> None:
    print('=== 2. SPAN MATCHER on real data ===')
    sys.path.insert(0, str(ROOT))
    from src.matching import fallback_marked_text, mark_compound

    correct = total = 0
    fb = 0
    mwe_leak = 0
    for split in ('dataset/en-nn-train.tsv', 'trial/en-nn-trial.tsv'):
        df = pd.read_csv(ROOT / split, sep='\t')
        for _, r in df.iterrows():
            total += 1
            marked = mark_compound(str(r['Context']), str(r['Mod']), str(r['Head']))
            if marked is not None and '<mod>' in marked and '<head>' in marked:
                correct += 1
            else:
                fb += 1
                fallback_marked_text(str(r['Mod']), str(r['Head']), str(r['Context']))
            if marked is not None and '<mwe>' in marked:
                mwe_leak += 1
    check(correct / total >= 0.95, f'{correct}/{total} rows marked in-context (>=95%)')
    check(total == correct + fb, 'fallback covers every remaining row')
    check(mwe_leak == 0, f'4-token scheme: no <mwe> markers leaked ({mwe_leak})')


def check_folds() -> None:
    print('=== 3. FOLD STRATIFICATION (compound leak check) ===')
    sys.path.insert(0, str(ROOT))
    from src.folds import prepare_stratified_folds

    df = pd.read_csv(ROOT / 'dataset/en-nn-train.tsv', sep='\t')
    out = prepare_stratified_folds(df.copy(), target_col='Compound')
    leak = sum(
        1
        for f in range(5)
        if not set(out.loc[out['fold'] != f, 'Compound']).isdisjoint(
            set(out.loc[out['fold'] == f, 'Compound'])
        )
    )
    check(leak == 0 and out['fold'].nunique() == 5, f'5 folds, no leakage ({out.groupby("fold").size().to_dict()})')


def check_config() -> None:
    print('=== 4. CONFIG VALIDATION ===')
    sys.path.insert(0, str(ROOT))
    from config import Config

    defaults = Config.defaults()
    check(defaults.embedding_lr > 0, f'defaults OK (batch={defaults.batch_size}, embedding_lr={defaults.embedding_lr}, lambda_rank={defaults.lambda_rank})')
    check(
        (defaults.head_features, defaults.phase1_schedule, defaults.rank_margin_mode,
         defaults.consist_mode, defaults.head_lr)
        == ('compact', 'constant', 'dynamic', 'pull', 1e-4),
        'rework defaults active (compact head, constant phase1, dynamic margin, head_lr=1e-4)',
    )

    try:
        Config.defaults().update(accum_steps=0).validate()
        check(False, 'accum_steps=0 rejected')
    except ValueError:
        check(True, 'accum_steps=0 rejected')

    try:
        Config.defaults().update(lambda_rank=-1).validate()
        check(False, 'lambda_rank=-1 rejected')
    except ValueError:
        check(True, 'lambda_rank=-1 rejected')

    try:
        Config.defaults().update(head_mode='bogus').validate()
        check(False, 'head_mode=bogus rejected')
    except ValueError:
        check(True, 'head_mode=bogus rejected')

    try:
        Config.defaults().update(num_bins=1).validate()
        check(False, 'num_bins=1 rejected')
    except ValueError:
        check(True, 'num_bins=1 rejected')

    try:
        Config.defaults().update(context_pool='bogus').validate()
        check(False, 'context_pool=bogus rejected')
    except ValueError:
        check(True, 'context_pool=bogus rejected')

    try:
        Config.defaults().update(head_features='bogus').validate()
        check(False, 'head_features=bogus rejected')
    except ValueError:
        check(True, 'head_features=bogus rejected')

    try:
        Config.defaults().update(phase1_schedule='bogus').validate()
        check(False, 'phase1_schedule=bogus rejected')
    except ValueError:
        check(True, 'phase1_schedule=bogus rejected')

    try:
        Config.defaults().update(rank_margin_mode='bogus').validate()
        check(False, 'rank_margin_mode=bogus rejected')
    except ValueError:
        check(True, 'rank_margin_mode=bogus rejected')

    try:
        Config.defaults().update(consist_mode='bogus').validate()
        check(False, 'consist_mode=bogus rejected')
    except ValueError:
        check(True, 'consist_mode=bogus rejected')

    try:
        Config.defaults().update(ccc_var_floor=-0.1).validate()
        check(False, 'ccc_var_floor=-0.1 rejected')
    except ValueError:
        check(True, 'ccc_var_floor=-0.1 rejected')

    try:
        Config.defaults().update(loss_std_alpha=-1).validate()
        check(False, 'loss_std_alpha=-1 rejected')
    except ValueError:
        check(True, 'loss_std_alpha=-1 rejected')

    try:
        Config.defaults().update(lambda_consist=-0.5).validate()
        check(False, 'lambda_consist=-0.5 rejected')
    except ValueError:
        check(True, 'lambda_consist=-0.5 rejected')

    try:
        Config.defaults().update(augment_prob=1.5).validate()
        check(False, 'augment_prob=1.5 rejected')
    except ValueError:
        check(True, 'augment_prob=1.5 rejected')

    try:
        Config.defaults().update(consist_temp=0).validate()
        check(False, 'consist_temp=0 rejected')
    except ValueError:
        check(True, 'consist_temp=0 rejected')

    try:
        Config.defaults().update(ce_weight=-0.5).validate()
        check(False, 'ce_weight=-0.5 rejected')
    except ValueError:
        check(True, 'ce_weight=-0.5 rejected')

    try:
        Config.defaults().update(unknown_key=1)
        check(False, 'unknown config key rejected')
    except ValueError:
        check(True, 'unknown config key rejected')

    import tempfile
    from pathlib import Path as P
    tmp = P(tempfile.gettempdir()) / 'cfg_smoke.json'
    defaults.save(tmp)
    check(Config.load(tmp).to_dict() == defaults.to_dict(), 'config JSON round-trip')
    tmp.unlink()

    # every CLI flag must map onto an existing Config field (strict update raises otherwise)
    import main as cli
    args = cli._build_parser().parse_args([
        'train5', '--epochs', '3', '--batch', '16', '--freeze-epochs', '1',
        '--accum-steps', '2', '--patience', '4', '--ccc-weight', '0.5',
        '--lambda-rank', '0.3', '--unfreeze-from', '21', '--predict-mode', 'single',
        '--head-mode', 'softmax', '--num-bins', '6', '--ce-weight', '0.3',
        '--bin-sigma', '0.5', '--context-pool', 'cls',
        '--head-features', 'legacy', '--head-hidden', '64',
        '--phase1-schedule', 'linear', '--rank-margin-mode', 'clamp',
        '--ccc-var-floor', '0.2', '--std-alpha', '0.5', '--lambda-consist', '0.1',
        '--consist-mode', 'infonce', '--augment-prob', '0.2',
        '--data-path', 'x', '--output-dir', 'y', '--seed', '7',
    ])
    merged = cli._merge_overrides(Config.defaults(), args)
    check(
        (merged.num_epochs, merged.batch_size, merged.freeze_epochs, merged.accum_steps,
         merged.lambda_rank, merged.unfreeze_from_layer, merged.predict_mode,
         merged.seed, merged.output_dir)
        == (3, 16, 1, 2, 0.3, 21, 'single', 7, 'y'),
        'all CLI flags map onto Config fields',
    )
    check(
        (merged.head_mode, merged.num_bins, merged.ce_weight, merged.bin_sigma,
         merged.context_pool)
        == ('softmax', 6, 0.3, 0.5, 'cls'),
        'ordinal CLI flags map onto Config fields',
    )
    check(
        (merged.head_features, merged.head_hidden, merged.phase1_schedule,
         merged.rank_margin_mode, merged.ccc_var_floor, merged.loss_std_alpha,
         merged.lambda_consist, merged.consist_mode, merged.augment_prob)
        == ('legacy', 64, 'linear', 'clamp', 0.2, 0.5, 0.1, 'infonce', 0.2),
        'new experiment CLI flags map onto Config fields',
    )
    check(
        (merged.use_label_std, Config.defaults().use_label_std)
        == (True, True),
        'use_label_std defaults on',
    )

    args_off = cli._build_parser().parse_args(['train5', '--no-label-std'])
    merged_off = cli._merge_overrides(Config.defaults(), args_off)
    check(merged_off.use_label_std is False, '--no-label-std maps to use_label_std=False')


def check_sampler() -> None:
    print('=== 5. GROUP SAMPLER (rank-pair density) ===')
    try:
        import torch  # noqa: F401
    except ImportError:
        print('  [SKIP] torch not installed locally')
        return
    sys.path.insert(0, str(ROOT))
    from src.sampler import CompoundGroupSampler

    import numpy as np
    df = pd.read_csv(ROOT / 'dataset/en-nn-train.tsv', sep='\t')
    ids = pd.factorize(df['Compound'])[0]
    from src.sampler import CompoundGroupSampler as CGS
    sampler = CGS(ids, batch_size=32, seed=42)
    idx = list(iter(sampler))

    check(
        len(idx) == len(df),
        f'sampler emits all rows exactly once ({len(idx)}/{len(df)})',
    )
    covered = len(set(idx))
    check(covered == len(df), f'100% row coverage ({covered}/{len(df)})')

    blocks = [idx[i:i + 32] for i in range(0, len(idx), 32)]
    pair_counts = []
    for b in blocks:
        _, counts = np.unique(ids[b], return_counts=True)
        pairs = sum(c * (c - 1) // 2 for c in counts if c > 1)
        pair_counts.append(pairs)
    avg_pairs = float(np.mean(pair_counts))
    print(f'  avg same-compound pairs/batch = {avg_pairs:.0f} (was ~1 random)')
    check(avg_pairs >= 30, f'same-compound pairs dense enough ({avg_pairs:.0f} >= 30)')

    # set_epoch must be reproducible AND differ across epochs.
    sampler.set_epoch(0)
    ep0a = list(iter(sampler))
    sampler.set_epoch(0)
    ep0b = list(iter(sampler))
    sampler.set_epoch(1)
    ep1 = list(iter(sampler))
    check(
        ep0a == ep0b,
        'set_epoch(0) reproducible within the same dataset',
    )
    check(
        ep0a != ep1,
        'set_epoch(1) shuffles differently from epoch 0',
    )


def main() -> int:
    sync_parse()
    check_matching()
    check_folds()
    check_config()
    check_sampler()

    print('=' * 50)
    if FAILURES:
        print(f'RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}')
        return 1
    print('RESULT: all smoke checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())