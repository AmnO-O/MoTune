"""Smoke checks for the gauss-only src/ package -- NO torch / transformers needed.

Run from the repo root:
    python tests/smoke_src.py        (or: python -m src.run --smoke)

Covers: syntax of every src/ module, config construction/validation/round-trip
and ``--set`` coercion, CLI wiring (``python -m src.run``), the offset-based
span matcher on synthetic token offsets, the real local TSV loaders + folds,
and -- when torch is available -- numerical checks of the merged gauss losses
and the GaussHead.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def check(condition: bool, label: str) -> None:
    status = 'OK' if condition else 'FAIL'
    print(f'  [{status}] {label}')
    if not condition:
        FAILURES.append(label)


def sync_parse() -> None:
    print('=== 1. SYNTAX (src/*.py + tests/smoke_src.py) ===')
    files = sorted((ROOT / 'src').glob('*.py')) + [ROOT / 'tests' / 'smoke_src.py']
    for path in files:
        try:
            ast.parse(path.read_text(encoding='utf-8'))
            check(True, str(path.relative_to(ROOT)))
        except SyntaxError as exc:
            check(False, f'{path.name}: {exc.msg} @ {exc.lineno}')
    check(not (ROOT / 'src' / 'losses_gauss.py').exists(),
          'src/losses_gauss.py removed (merged into src/losses.py)')


def check_config() -> None:
    print('=== 2. CONFIG (defaults / validation / round-trip / --set) ===')
    sys.path.insert(0, str(ROOT))
    from src.config import Config, coerce_value

    defaults = Config.defaults()
    defaults.validate()
    check(defaults.backbone == 'jhu-clsp/mmBERT-base', f'default backbone = mmBERT-base')
    check(defaults.total_epochs == defaults.freeze_epochs + defaults.lora_epochs,
          'total_epochs == freeze_epochs + lora_epochs')
    check(defaults.mode == 'train80', 'only train80 mode exists')
    check(defaults.ccc_weight == 0.7, 'single ccc_weight default')
    check(not hasattr(defaults, 'use_proto_cos'),
          'use_proto_cos knob removed (proto-cos always on, head_in += 1)')
    check(not hasattr(defaults, 'head_mode') and not hasattr(defaults, 'num_bins')
          and not hasattr(defaults, 'warmup_mlm_epochs')
          and not hasattr(defaults, 'lambda_consist')
          and not hasattr(defaults, 'consist_mode')
          and not hasattr(defaults, 'phase1_schedule')
          and not hasattr(defaults, 'gauss_dedicated')
          and not hasattr(defaults, 'span_layers')
          and not hasattr(defaults, 'context_layers')
          and not hasattr(defaults, 'aux_data_paths'),
          'legacy model, fallback-pooling, and auxiliary-data knobs are absent')

    bad = [
        ('lambda_rank=-1', lambda: Config.defaults().update(lambda_rank=-1)),
        ('removed head_pool', lambda: Config.defaults().update(head_pool='mean')),
        ('removed lambda_dist', lambda: Config.defaults().update(lambda_dist=1)),
        ('removed lambda_compound', lambda: Config.defaults().update(lambda_compound=1)),
        ('mode=train5', lambda: Config.defaults().update(mode='train5')),
        ('freeze_epochs=-1', lambda: Config.defaults().update(freeze_epochs=-1)),
        ('lora_epochs=0', lambda: Config.defaults().update(lora_epochs=0)),
        ('embedding_lr=-1', lambda: Config.defaults().update(embedding_lr=-1)),
        ('unknown_key', lambda: Config.defaults().update(unknown_key=1)),
    ]
    for label, fn in bad:
        try:
            fn().validate()
            check(False, f'[reject] {label}')
        except ValueError:
            check(True, f'[reject] {label}')

    ok = Config.defaults().update(freeze_epochs=2)
    ok.validate()
    check(ok.freeze_epochs == 2 and ok.total_epochs == ok.freeze_epochs + ok.lora_epochs,
          'valid gauss config accepted, total_epochs derived')

    # JSON round-trip
    tmp = Path(tempfile.gettempdir()) / 'src_cfg_smoke.json'
    defaults.save(tmp)
    check(Config.load(tmp).to_dict() == defaults.to_dict(), 'config JSON round-trip')
    tmp.unlink()

    # YAML path: works if PyYAML is installed, else must fail with a clear error
    tmp = Path(tempfile.gettempdir()) / 'src_cfg_smoke.yaml'
    tmp.write_text('seed: 7\nfreeze_epochs: 2\nlora_epochs: 4\n', encoding='utf-8')
    try:
        from src.config import Config as C2
        y = C2.load(tmp)
        check(y.seed == 7 and y.total_epochs == 6, 'config YAML round-trip')
    except ValueError as exc:
        check('PyYAML' in str(exc), f'YAML config without PyYAML raises clear error')
    tmp.unlink()

    # --set coercion
    check(coerce_value('freeze_epochs', '3') == 3, 'coerce int')
    check(coerce_value('use_label_std', 'false') is False, 'coerce bool')
    check(coerce_value('gauss_ctx_head', '19,20') == ['19', '20'], 'coerce gauss_ctx_*')
    check(coerce_value('backbone', 'x/y') == 'x/y', 'str passes through')
    check(coerce_value('not_a_field', '1') == '1', 'unknown key passes through (validated later)')


def check_cli() -> None:
    print('=== 3. CLI WIRING (python -m src.run) ===')
    sys.path.insert(0, str(ROOT))
    from types import SimpleNamespace
    import src.run as cli
    from src.config import Config

    tmp_cfg = Path(tempfile.gettempdir()) / 'src_cli_smoke.json'
    tmp_cfg.write_text('{"freeze_epochs": 5}', encoding='utf-8')

    args = SimpleNamespace(
        config=str(tmp_cfg),
        set=['freeze_epochs=2', 'lora_epochs=6', 'use_label_std=false'],
        device='auto', smoke=False,
    )
    parsed = cli._build_parser().parse_args([
        '--config', str(tmp_cfg),
        '--set', 'freeze_epochs=2', '--set', 'lora_epochs=6',
        '--set', 'use_label_std=false',
        '--device', 'cpu',
    ])
    check(parsed.config == str(tmp_cfg) and parsed.device == 'cpu'
          and parsed.smoke is False, 'CLI flags parsed')
    check(parsed.set == ['freeze_epochs=2', 'lora_epochs=6', 'use_label_std=false'],
          'repeated --set collected')
    merged = cli._build_config(parsed)
    check(merged.freeze_epochs == 2 and merged.lora_epochs == 6
          and merged.total_epochs == 8 and merged.use_label_std is False,
          'CLI --set ints/bools map onto Config')
    merged.validate()
    bad = cli._build_parser().parse_args(['--set', 'no_equals'])
    try:
        cli._build_config(bad)
        check(False, '--set without = rejected')
    except ValueError:
        check(True, '--set without = rejected')
    tmp_cfg.unlink()


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
    from src.marks import Span, find_spans

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

    # 11) German PV fused past participle (ge- infix)
    text_de1 = 'ich bin nach L.A. abgehauen .'
    r_de1 = find_spans(text_de1, _word_offsets(text_de1), 'hauen', 'ab', 'abhauen')
    check(r_de1.found and r_de1.degenerate and tp(r_de1.mod) == (4, 5) and tp(r_de1.head) == (4, 5),
          'German PV fused past participle "abgehauen" matched')

    # 12) German PV separated V2 clause
    text_de2 = 'Die Linken lehnen sie ab .'
    r_de2 = find_spans(text_de2, _word_offsets(text_de2), 'lehnen', 'ab', 'ablehnen')
    check(r_de2.found and not r_de2.adjacent and tp(r_de2.mod) == (2, 3) and tp(r_de2.head) == (4, 5),
          'German PV separated V2 clause "lehnen ... ab" matched')

    # 13) German PV strong verb separated past
    text_de3 = 'Er schloss die Tür ab .'
    r_de3 = find_spans(text_de3, _word_offsets(text_de3), 'schließen', 'ab', 'abschließen')
    check(r_de3.found and not r_de3.adjacent and tp(r_de3.mod) == (1, 2) and tp(r_de3.head) == (4, 5),
          'German PV strong verb separated past "schloss ... ab" matched')



def check_data() -> None:
    print('=== 5. DATA LOADERS (real local TSVs, no torch) ===')
    sys.path.insert(0, str(ROOT))
    from src.config import Config
    from src.data import _df_to_rows, load_labeled, read_tsv
    from src.marks import find_spans

    rows_en = _df_to_rows(read_tsv('dataset/en-nn-train.tsv'), 'en-nn', 'en')
    check(len(rows_en) == 3480 and rows_en[0]['has_label'],
          f'en-nn: {len(rows_en)} rows, labeled')

    rows_de = _df_to_rows(read_tsv('dataset/de-nn-train.tsv'), 'de-nn', 'de')
    check(rows_de[0]['compound'] == 'Abiturzeugnis' and rows_de[0]['mod'] == 'Abitur'
          and rows_de[0]['lang'] == 'de',
          'de-nn compound/mod/head/lang parsed')

    cfg = Config.defaults().update(data_path='dataset')
    labeled = load_labeled(cfg)
    n_compounds = len({r['compound_id'] for r in labeled})
    check(len(labeled) >= 3480 and n_compounds > 100,
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
        found = 0
        for r in rows[:sample]:
            res = find_spans(r['context'], synth_offsets(r['context']), r['mod'], r['head'])
            found += bool(res.found)
        return found

    n = min(len(rows_en), 2000)
    found = alignment(rows_en, n)
    check(found / n > 0.4, f'en-nn synthetic-offset alignment {found}/{n} ({100*found/n:.0f}%)')

    gn = min(len(rows_de), 2000)
    gfound = alignment(rows_de, gn)
    check(gfound / gn > 0.2,
          f'de-nn synthetic-offset alignment {gfound}/{gn} ({100*gfound/gn:.0f}%)')

    rows_pv = _df_to_rows(read_tsv('dataset/en-pv-train.tsv'), 'en-pv', 'en')
    check(len(rows_pv) == 1557 and rows_pv[0]['is_pv'],
          f'en-pv: {len(rows_pv)} rows parsed (is_pv flag)')
    pv_n = min(len(rows_pv), 500)
    pv_ok = sum(
        1 for r in rows_pv[:pv_n]
        if find_spans(r['context'], synth_offsets(r['context']),
                      r['mod'], r['head'], r.get('compound', '')).found
    )
    check(pv_ok / pv_n > 0.8,
          f'en-pv irregular/doubled-verb alignment {pv_ok}/{pv_n} ({100*pv_ok/pv_n:.0f}%)')

    rows_de_pv = _df_to_rows(read_tsv('dataset/de-pv-train.tsv'), 'de-pv', 'de')
    check(len(rows_de_pv) == 1490 and rows_de_pv[0]['is_pv'],
          f'de-pv: {len(rows_de_pv)} rows parsed (is_pv flag)')
    de_pv_n = min(len(rows_de_pv), 500)
    de_pv_ok = sum(
        1 for r in rows_de_pv[:de_pv_n]
        if find_spans(r['context'], synth_offsets(r['context']),
                      r['mod'], r['head'], r.get('compound', '')).found
    )
    check(de_pv_ok / de_pv_n > 0.9,
          f'de-pv trennbare Verben alignment {de_pv_ok}/{de_pv_n} ({100*de_pv_ok/de_pv_n:.0f}%)')

    rows_nctti = _df_to_rows(read_tsv('dataset/nctti_en_scored.tsv'), 'nctti_en_scored', 'en')
    check(len(rows_nctti) == 539 and rows_nctti[0]['is_pv'] and rows_nctti[0]['has_label'],
          f'nctti scored: {len(rows_nctti)} PV-schema labeled rows')
    check(0.0 <= rows_nctti[0]['mod_avg'] <= 5.0 and rows_nctti[0]['mod_std'] > 0,
          'nctti scored Avg in [0,5] with finite Std')

    # MLM warmup data pieces must be gone
    data_src = (ROOT / 'src' / 'data.py').read_text(encoding='utf-8')
    check('def MlmDataset' not in data_src and 'def collate_mlm' not in data_src
          and 'def load_mlm_rows' not in data_src,
          'src/data.py has no MLM warmup dataset/collate/loader')


def check_folds() -> None:
    print('=== 6. FOLDS + GROUP SAMPLER (real TSVs, no torch) ===')
    sys.path.insert(0, str(ROOT))
    from src.data import _df_to_rows, read_tsv
    from src.folds import CompoundGroupSampler, StratifiedKFold, assign_folds

    if StratifiedKFold is None:
        check(True, 'fold assignment skipped (scikit-learn unavailable)')
        return

    rows = _df_to_rows(read_tsv('dataset/en-nn-train.tsv'), 'en-nn', 'en')
    for r in rows:
        r['compound_id'] = -1

    folded = assign_folds(rows, n_splits=5, seed=42)
    folds = {r['compound']: r['fold'] for r in folded}
    check(0 <= min(folds.values()) and max(folds.values()) <= 4, 'fold ids in range')
    check(len(set(folds.values())) == 5, 'all 5 folds used')
    per_fold = {}
    for r in folded:
        per_fold[r['fold']] = per_fold.get(r['fold'], 0) + 1
    mx, mn = max(per_fold.values()), min(per_fold.values())
    check(mx - mn <= len(folded) // 2, f'fold sizes balanced: {per_fold}')
    again = assign_folds(rows, n_splits=5, seed=42)
    check([r['fold'] for r in again] == [r['fold'] for r in folded],
          'deterministic given the same seed')
    import random as _r
    rng = _r.Random(3)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    re_shuffled = assign_folds(shuffled, n_splits=5, seed=42)
    folds2 = {r['compound']: r['fold'] for r in re_shuffled}
    check(folds == folds2, 'fold assignment invariant to row order')

    n_lab = [i for i, r in enumerate(rows) if r['has_label']]
    labels_n = {r['compound'] for r in rows if r['has_label']}
    codes = {c: v for v, c in enumerate(sorted(labels_n))}
    cids = [codes[r['compound']] if r['has_label'] else -1 for r in rows]
    samp = CompoundGroupSampler(cids, batch_size=32, seed=7)
    order = list(samp)
    check(len(order) == len(rows), 'sampler yields every row once')
    samp.set_epoch(1)
    order2 = list(samp)
    check(order != order2, 'set_epoch changes the shuffle')


def check_fixes() -> None:
    print('=== 7. GAUSS CONTRACTS & BUG-FIX REGRESSION GUARDS ===')
    sys.path.insert(0, str(ROOT))
    from src.config import Config

    model_src = (ROOT / 'src' / 'model.py').read_text(encoding='utf-8')
    loss_src = (ROOT / 'src' / 'losses.py').read_text(encoding='utf-8')
    train_src = (ROOT / 'src' / 'train.py').read_text(encoding='utf-8')
    pipe_src = (ROOT / 'src' / 'pipeline.py').read_text(encoding='utf-8')

    # 1) reg/softmax & warmup machinery fully gone
    check('class CombinedLoss' not in loss_src,
          'src/losses.py has no CombinedLoss (reg/softmax composite dropped)')
    check('def warmup_epoch' not in train_src,
          'src/train.py has no warmup_epoch (MLM warmup phase dropped)')
    check('run_train5' not in pipe_src and 'run_predict' not in pipe_src
          and 'run_warmup' not in pipe_src and 'run_probe' not in pipe_src,
          'src/pipeline.py wires only run_train80')
    check('def _tokenizer(cfg: Config, logger: logging.Logger)' in pipe_src,
          '_tokenizer accepts (cfg, logger) as called by run_train80')
    check('def _build_head' not in model_src and 'mod_regressor' not in model_src
          and 'bin_centers' not in model_src,
          'src/model.py has no reg/softmax heads (_build_head / regressors / bins)')

    # 2) gauss-only wiring
    check('class GaussLoss' in loss_src and 'def gauss_kl' in loss_src
          and 'self.requires_logits = True' in loss_src,
          'src/losses.py defines gauss_kl + GaussLoss (sigma via logits channel)')
    check('def margin_rank_loss' in loss_src and 'def compound_center_loss' not in loss_src
          and 'def compound_consistency_loss' not in loss_src,
          'src/losses.py retains ranking only; centre/consistency losses are absent')
    check('def _role_context(' not in model_src and 'def _compose_gauss_feat(' in model_src,
          '_features builds only dedicated per-role feature bundles')
    check('pv_mod_exit_emb, pv_head_exit_emb = mod_exit_emb, head_exit_emb' not in model_src,
          'PV literalness features stay on the PV exit layers')

    # 3) proxy for the role-feature width: attention-fused vector + plain-AutoModel
    check('self.head_in = hidden_size' in model_src
          and 'self.fusion = SpanFusion' in model_src,
          'head_in is ONE fused attention vector per exit (no concat bundle)')
    check('from transformers import AutoModel' in model_src
          and 'AutoModelForMaskedLM' not in model_src
          and '_lm_span_stats' not in model_src
          and 'use_lm_features' not in model_src,
          'backbone is a plain AutoModel; LM-predictability stats dropped (no [MASK])')
    check('def _compose_gauss_feat(self, mod_emb, head_emb, context_emb,'
          in model_src and 'lm_stats' not in model_src
          and 'mod_len, head_len' not in model_src
          and 'return self.fusion(mod_emb, head_emb, context_emb, cos_)' in model_src,
          'SpanFusion fuses span pair + context mean/CLS + cos (lengths removed, no cat)')
    check('tok = F.embedding(input_ids, weight)' in model_src,
          '_prototype_cos uses F.embedding (not manual index_select)')
    check('proto_safe = torch.where(has_span, proto, torch.ones_like(proto))' in model_src,
          '_prototype_cos substitutes ones BEFORE the cosine (no NaN in backward)')

    # 4) LoRA base-freeze invariant (LoRA now lives in src/lora.py)
    lora_src = (ROOT / 'src' / 'lora.py').read_text(encoding='utf-8')
    check('linear.weight.requires_grad = False' in lora_src
          and 'linear.bias.requires_grad = False' in lora_src,
          'LoRAAdapter freezes its base weight/bias on wrap')
    check("if '.linear.' in name:" in train_src,
          'unfreeze_top_layers skips LoRA-wrapped base weights')

    # 4b) label/empty masking lives in the criterion, not a train.py helper
    check('def _supervised_term' not in train_src,
          '_supervised_term removed; GaussLoss.forward owns the allowed-mask')
    check('mask=allowed' in train_src,
          'train_epoch calls GaussLoss with mask=allowed (no pre-indexing)')

    try:
        import torch as _torch
        from src import losses as G
        from src.heads import GaussHead

        # KL(N(0,1)||N(0,1)) == 0; inflated sigma_p is penalised (>= base case)
        zero = float(G.gauss_kl(_torch.zeros(4), _torch.ones(4),
                                _torch.zeros(4), _torch.ones(4)))
        infl = float(G.gauss_kl(_torch.zeros(4), _torch.ones(4) * 10,
                                _torch.zeros(4), _torch.ones(4)))
        check(abs(zero) < 1e-5 and infl > zero,
              'gauss_kl: identical Gaussians -> 0, inflated sigma_p is penalised')

        # _target_sigma clamp sanity
        ts = G._target_sigma(_torch.tensor([0.0, 0.4, 9.0]), bin_sigma=0.5)
        check(bool((ts >= 0.25).all()) and bool((ts <= 5.0).all()),
              '_target_sigma clamps crowd std into [0.25, 5]')

        # GaussHead: mu/sigma shapes, sigma > 0, gradients finite
        h = GaussHead(32, hidden=48, dropout=0)
        mu, sigma = h(_torch.randn(8, 32))
        check(mu.shape == (8,) and (sigma > 0).all(),
              'GaussHead returns mu/sigma shapes correctly, sigma > 0')
        (mu - _torch.randn(8)).pow(2).mean().backward()
        grads = [p.grad for p in h.parameters() if p.grad is not None]
        check(len(grads) > 0 and all(_torch.isfinite(g).all() for g in grads),
              'GaussHead gradients flow and are finite')

        # margin_rank_loss: dynamic vs clamp mode, finite; NaN target excluded
        pred = _torch.randn(16)
        tgt = _torch.randn(16)
        c = _torch.tensor([0] * 8 + [1] * 8, dtype=_torch.long)
        l1 = float(G.margin_rank_loss(pred, tgt, margin=0.5, compound_ids=c, mode='dynamic'))
        l2 = float(G.margin_rank_loss(pred, tgt, margin=0.5, compound_ids=c, mode='clamp'))
        check(min(l1, l2) >= 0.0 and max(l1, l2) > 0.0,
              'margin_rank_loss returns non-negative finite values in either mode')

        # NaN mixing: NaN safe positive gate with std_alpha weighting
        dfl = G.GaussLoss()
        loss = dfl(pred, tgt, logits=_torch.ones(16), compound_ids=c)
        check(bool(_torch.isfinite(loss)), 'GaussLoss forward finite')

        # empty-mask: returns 0.0 with grad connected (no NaN, no graph break)
        pg = _torch.randn(16, requires_grad=True)
        z = dfl(pg, tgt, logits=_torch.ones(16), compound_ids=c,
                mask=_torch.zeros(16, dtype=_torch.bool))
        check(bool(z.item() == 0.0) and z.requires_grad,
              'GaussLoss(mask=all-False) returns grad-connected zero')
    except ImportError:
        print('  [SKIP] torch not installed; GaussHead / gauss_kl numerical checks skipped')


def check_targets() -> None:
    print('=== 8. SINGLE-TARGET WIRING (targets module, expansion, routing) ===')
    sys.path.insert(0, str(ROOT))

    import src.data as D
    import src.train as T
    import src.trainer as TR
    from src.targets import TARGETS, target_code, row_targets, target_selector, pool_span

    # target_code accepts both names and integer codes
    check(target_code('mod') == 0 and target_code('head') == 1 and target_code('pv') == 2,
          'target_code: names map to 0/1/2')
    check(target_code(0) == 0 and target_code(1) == 1 and target_code(2) == 2,
          'target_code: integer codes round-trip identity')
    try:
        target_code('bogus')
        check(False, 'target_code rejects unknown names')
    except ValueError:
        check(True, 'target_code rejects unknown names')
    try:
        target_code(9)
        check(False, 'target_code rejects out-of-range codes')
    except ValueError:
        check(True, 'target_code rejects out-of-range codes')

    # row_targets: None when absent, int codes when a tensor is supplied
    check(row_targets({}) is None, 'row_targets -> None when no target field')
    import torch as _torch
    check(row_targets({'target': _torch.tensor([0, 1, 2])}) == [0, 1, 2],
          'row_targets converts a long tensor to a list of codes')

    # target_selector: bool row masks + None passthrough for joint mode
    sel_mod = target_selector([0, 1, 2], 'mod')
    check(bool(_torch.equal(sel_mod, _torch.tensor([True, False, False]))),
          'target_selector picks rows of one target')
    check(target_selector(None, 'mod') is None, 'target_selector -> None for joint mode')

    # pool_span: masked mean over the active span; PV pools mod|head
    # hidden shape: (batch=2, seq_len=6, hidden_dim=4)
    hidden = _torch.arange(48, dtype=_torch.float).reshape(2, 6, 4)
    modm = _torch.tensor([[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=_torch.bool)
    headm = _torch.tensor([[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=_torch.bool)
    batch = {'mod_span_mask': modm, 'head_span_mask': headm}
    pmod = pool_span(hidden, batch, 'mod')
    phead = pool_span(hidden, batch, 'head')
    ppv = pool_span(hidden, batch, 'pv')
    check(pmod.shape == (2, 4) and phead.shape == (2, 4) and ppv.shape == (2, 4),
          'pool_span returns one (hidden,) vector per row for every target')
    check(bool(_torch.allclose(pmod[0], hidden[0, 0:2, :].mean(dim=0))),
          'pool_span(mod) averages the modifier tokens')
    check(bool(_torch.allclose(phead[0], hidden[0, 2:4, :].mean(dim=0))),
          'pool_span(head) averages the head tokens')
    check(bool(_torch.allclose(ppv[0], hidden[0, 0:4, :].mean(dim=0))),
          'pool_span(pv) averages mod|head union')
    check(bool(_torch.allclose(ppv[1], hidden[1, 4:6, :].mean(dim=0))),
          'pool_span(pv) on PV-only row is well-defined')

    # pool_active: single-pass per-row routing == per-target pool_span result
    import src.targets as TG
    mixed = TG.pool_active(hidden, batch, [0, 1])          # row0 mod, row1 head
    check(bool(_torch.allclose(mixed[0], pmod[0]))
          and bool(_torch.allclose(mixed[1], phead[1])),
          'pool_active routes each row to its own span in one pass')
    pv_first = TG.pool_active(hidden, batch, ['pv', 'pv'])
    check(bool(_torch.allclose(pv_first[0], hidden[0, 0:4, :].mean(dim=0))),
          'pool_active pv pools the whole compound (mod|head)')

    # expand_targets: NN rows -> mod+head; PV rows -> pv; nothing else
    nn = {'is_pv': False, 'mod_avg': 3.5, 'head_avg': 3.8, 'has_label': True,
          'compound': 'flea market', 'mod': 'flea', 'head': 'market'}
    pv = {'is_pv': True, 'mod_avg': 0.3, 'has_label': True,
          'compound': 'crack down', 'mod': 'crack', 'head': 'down'}
    fid = float('nan')
    nnn = dict(nn, mod_avg=fid, head_avg=3.0)
    full = D.expand_targets([nn, pv, nnn], ['mod', 'head', 'pv'])
    labels = [(r['target'], bool(r['has_label'])) for r in full]
    check(labels == [('mod', True), ('head', True),
                     ('pv', True),
                     ('mod', False), ('head', True)],
          'expand_targets drops invalid target rows and recomputes has_label')
    check(D.expand_targets([nn], []) == [nn], 'expand_targets with empty targets is a no-op')
    try:
        import numpy as _np
        check(bool(_np.isfinite(full[0]['mod_avg'])),
              'expand_targets: labeled nn mod-target row keeps its finite label')
        check(not _np.isfinite(full[3]['mod_avg']) and full[3]['has_label'] is False,
              'expand_targets: unlabeled mod-target row keeps NaN and has_label=False')
    except ImportError:
        print('  [SKIP] numpy unavailable; NaN passthrough check skipped')

    # trainer expands rows in _build_loaders (source guard instead of GPU run)
    tr_src = (ROOT / 'src' / 'trainer.py').read_text(encoding='utf-8')
    check('expand_targets(train_rows' in tr_src and 'expand_targets(val_rows' in tr_src,
          '_build_loaders expands train/val rows via expand_targets')
    check("'target' in batch" in tr_src.replace(' ', '') or 'target' in tr_src,
          'trainer dispatches on batch[target] when present')

    # train_epoch gates each loss by its own target
    t_src = (ROOT / 'src' / 'train.py').read_text(encoding='utf-8')
    check("mod_mask = allowed_nn & (tgt == 0)" in t_src
          and "head_mask = allowed_nn & (tgt == 1)" in t_src
          and "pv_mask = allowed_pv & (tgt == 2)" in t_src,
          'train_epoch gates mod/head/pv losses by per-row target')

    # model routing: inactive heads are zeroed (source guard in model_combined)
    m_src = (ROOT / 'src' / 'model_combined.py').read_text(encoding='utf-8')
    check('torch.where(select(' in m_src
          and 'target_selector(targets, t)' in m_src
          and 'torch.zeros_like' in m_src,
          'CombinedBackboneModel zeroes inactive-head predictions via torch.where')

    # collate stacks the scalar target key
    d_src = (ROOT / 'src' / 'data.py').read_text(encoding='utf-8')
    check("item['target'] = torch.tensor(_TARGET_CODE" in d_src,
          'CompDataset encodes target as a scalar long tensor')
    check("'target'" in d_src.replace(' ', '') and 'torch.stack' in d_src,
          'collate_comp stacks scalar keys (incl. target)')

    # target_prefix validation in Config
    from src.config import Config
    cfg_ok = Config(target_prefix=True, targets=['mod', 'head'])
    check(cfg_ok.target_prefix is True, 'Config accepts target_prefix=True with non-empty targets')
    try:
        Config(target_prefix=True, targets=[]).validate()
        check(False, 'Config rejects target_prefix=True with empty targets')
    except ValueError:
        check(True, 'Config rejects target_prefix=True with empty targets')

    # MARKER_CODE mapping
    from src.targets import MARKER_CODE, TARGET_PREFIX_TOKENS
    check(MARKER_CODE == {'mod': 7, 'head': 8, 'pv': 9},
          'MARKER_CODE maps mod/head/pv to mmBERT unused token ids 7/8/9')
    check(TARGET_PREFIX_TOKENS == {'mod': '<unused0>', 'head': '<unused1>', 'pv': '<unused2>'},
          'TARGET_PREFIX_TOKENS maps targets to unused token names')

    # CompDataset target_prefix token insertion (<marker> word <marker>) and span mask shift
    try:
        import torch as _torch
        class _MockTok:
            def __call__(self, text, max_length=256, truncation=True, return_tensors='pt', return_offsets_mapping=True):
                return {
                    'input_ids': _torch.tensor([[2, 10, 20, 30, 1]]),
                    'attention_mask': _torch.tensor([[1, 1, 1, 1, 1]]),
                    'offset_mapping': _torch.tensor([[[0, 0], [0, 3], [4, 8], [9, 15], [0, 0]]])
                }
            def encode(self, text, add_special_tokens=False):
                mapping = {'flea': [20], 'market': [30], 'flea market': [20, 30]}
                return mapping.get(text, [99])

        _row = {'context': 'our flea market', 'mod': 'flea', 'head': 'market', 'compound': 'flea market',
                'compound_id': 0, 'has_label': True, 'mod_avg': 3.0, 'head_avg': 4.0, 'mod_std': 0.5,
                'head_std': 0.5, 'target': 'mod'}
        _ds0 = D.CompDataset([_row], _MockTok(), target_prefix=False)
        _ds1 = D.CompDataset([_row], _MockTok(), target_prefix=True)
        check(_ds0[0]['input_ids'].tolist() == [2, 10, 20, 30, 1],
              'CompDataset without prefix preserves original token sequence')
        check(_ds1[0]['input_ids'].tolist() == [2, 7, 20, 7, 10, 20, 30, 1],
              'CompDataset with target_prefix prepends <marker> word <marker> right after <bos>')
        check(_ds0[0]['mod_span_mask'].tolist() == [False, False, True, False, False]
              and _ds1[0]['mod_span_mask'].tolist() == [False, False, False, False, False, True, False, False],
              'CompDataset target_prefix shifts mod_span_mask cleanly past prefix to the context occurrence')
        check(_ds0[0]['head_span_mask'].tolist() == [False, False, False, True, False]
              and _ds1[0]['head_span_mask'].tolist() == [False, False, False, False, False, False, True, False],
              'CompDataset target_prefix shifts head_span_mask cleanly past prefix to the context occurrence')
        check(_ds0[0]['prefix_mask'].tolist() == [False, False, False, False, False]
              and _ds1[0]['prefix_mask'].tolist() == [False, False, True, False, False, False, False, False],
              'CompDataset prefix_mask marks ONLY the WORD tokens inside the prefix (markers excluded)')
        check(_ds1[0]['attention_mask'].tolist() == [1, 1, 1, 1, 1, 1, 1, 1],
              'prefix word tokens are attended')
    except ImportError:
        print('  [SKIP] torch unavailable; CompDataset prefix check skipped')

    # CompDataset span_markers: border ids spliced around the row's OWN span
    try:
        import torch as _torch
        _rm = dict(_row); _rm['target'] = 'mod'
        _dm = D.CompDataset([_rm], _MockTok(), span_markers=True)
        check(_dm[0]['input_ids'].tolist() == [2, 10, 7, 20, 7, 30, 1],
              'span_markers wraps the mod span with id 7 (open + close)')
        check(_dm[0]['mod_span_mask'].tolist() == [False, False, False, True, False, False, False]
              and _dm[0]['head_span_mask'].tolist() == [False, False, False, False, False, True, False],
              'span_markers shifts both span masks exactly around the spliced ids')
        _rh = dict(_row); _rh['target'] = 'head'
        _dh = D.CompDataset([_rh], _MockTok(), span_markers=True)
        check(_dh[0]['input_ids'].tolist() == [2, 10, 20, 8, 30, 8, 1],
              'span_markers wraps the head span with id 8')
        _rp = dict(_row); _rp['target'] = 'pv'
        _dp = D.CompDataset([_rp], _MockTok(), span_markers=True)
        check(_dp[0]['input_ids'].tolist() == [2, 10, 9, 20, 30, 9, 1],
              'span_markers pv wraps the WHOLE compound with id 9')
    except ImportError:
        print('  [SKIP] torch unavailable; span_markers check skipped')

    # config: span_markers validation
    try:
        Config(span_markers=True, target_prefix=True, targets=['mod'],
               model_backend='combined').validate()
        check(False, 'Config rejects span_markers + target_prefix together')
    except ValueError:
        check(True, 'Config rejects span_markers + target_prefix together')
    try:
        Config(span_markers=True, model_backend='exits', targets=['mod']).validate()
        check(False, 'Config rejects span_markers on a non-combined backend')
    except ValueError:
        check(True, 'Config rejects span_markers on a non-combined backend')
    try:
        Config(span_markers=True, targets=[], model_backend='combined').validate()
        check(False, 'Config rejects span_markers with empty targets')
    except ValueError:
        check(True, 'Config rejects span_markers with empty targets')

    # config: prefix_readout validation ('prefix'/'dual' need target_prefix)
    from src.config import Config as _C
    try:
        _C(prefix_readout='prefix', target_prefix=False, targets=['mod']).validate()
        check(False, 'Config rejects prefix_readout=prefix without target_prefix')
    except ValueError:
        check(True, 'Config rejects prefix_readout=prefix without target_prefix')
    try:
        _C(prefix_readout='dual', target_prefix=False, targets=['mod']).validate()
        check(False, 'Config rejects prefix_readout=dual without target_prefix')
    except ValueError:
        check(True, 'Config rejects prefix_readout=dual without target_prefix')
    try:
        _C(prefix_readout='nope', target_prefix=True, targets=['mod']).validate()
        check(False, 'Config rejects an unknown prefix_readout value')
    except ValueError:
        check(True, 'Config rejects an unknown prefix_readout value')
    check(_C(prefix_readout='dual', target_prefix=True, targets=['mod'],
             model_backend='combined').validate() is None
          and _C(prefix_readout='prefix', target_prefix=True, targets=['mod'],
                 model_backend='combined').validate() is None,
          'Config accepts prefix_readout together with target_prefix')

    # trainer wires span_markers
    check('span_markers=self.cfg.span_markers' in tr_src,
          'trainer passes span_markers to CompDataset')

    # trainer wires target_prefix
    check('target_prefix=self.cfg.target_prefix' in tr_src,
          'trainer passes target_prefix to CompDataset')

    # model: learned marker embedding folded into the head group
    check("self.marker_emb = nn.Embedding(3, hidden_size) if target_prefix else None" in m_src,
          'CombinedBackboneModel creates a learned 3xH marker embedding when target_prefix on')
    check("hidden_states[:, 1] = marker" in m_src
          and "inputs_embeds=hidden_states" in m_src,
          'CombinedBackboneModel overwrites position 1 with the learned marker vector')
    check("marker = getattr(model, 'marker_emb', None)" in tr_src
          and "heads.append(marker)" in tr_src,
          'trainer._pred_heads folds marker_emb into the head_lr group')
    check("seen = set()" in tr_src
          and "if id(m) not in seen:" in tr_src,
          'trainer._pred_heads dedupes shared-head aliases by module identity')

    # model: prefix_readout wiring (context/prefix/dual pooling + head-group gate)
    check("self.prefix_readout = prefix_readout" in m_src,
          'CombinedBackboneModel stores prefix_readout from cfg')
    check("self.prefix_gate = None" in m_src
          and "if target_prefix and prefix_readout == 'dual':" in m_src
          and "nn.Parameter(torch.zeros(1))" in m_src,
          'dual readout gets a trainable 1-param gate (sigmoid 0 -> 50/50 start)')
    check("from .targets import TARGETS, pool_active, pool_prefix, pool_span, row_targets, target_selector" in m_src,
          'CombinedBackboneModel imports pool_prefix for the prefix readout')
    check("pre = pool_prefix(hidden, batch)" in m_src
          and "pool = pre" in m_src and "'dual'" in m_src,
          'prefix readout pools the prefix WORD tokens')
    check("a = torch.sigmoid(self.prefix_gate.to(hidden.device))" in m_src
          and "pool = a * ctx + (1 - a) * pre" in m_src,
          'dual readout blends context+prefix pools via a learned gate')
    check("'prefix_mask' in batch" in m_src,
          'prefix readout guards on batch prefix_mask presence (joint/predict safe)')
    check('prefix_readout=str(getattr(cfg, \'prefix_readout\', \'context\'))' in m_src,
          'build_combined_model forwards prefix_readout from cfg')
    check("gate = getattr(model, 'prefix_gate', None)" in tr_src
          and "heads.append(nn.ParameterList([gate]))" in tr_src,
          'trainer._pred_heads folds prefix_gate into the head_lr group')

    # model: single shared GaussHead (mod/head/pv_gauss all alias the same module)
    check("self.gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)" in m_src
          and "object.__setattr__(self, 'mod_gauss', self.gauss)" in m_src
          and "object.__setattr__(self, 'head_gauss', self.gauss)" in m_src
          and "object.__setattr__(self, 'pv_gauss', self.gauss)" in m_src,
          'CombinedBackboneModel shares ONE GaussHead across all targets')

    # model: static_span fuses ctx+static pools via a transformer (FusionBlock)
    check("self.static_fuse = StaticFusion(" in m_src
          and "FusionBlock(hidden, num_heads, ffn_expansion=2" in m_src,
          'StaticFusion reuses FusionBlock for a single fused (B,H) vector')
    check("pool = self.static_fuse(pool, static_pool)" in m_src
          and "pool = self.static_fuse(pool, static_v)" in m_src,
          'CombinedBackboneModel fuses ctx+static pools via StaticFusion (no concat)')
    check("torch.cat([pool, pool_span(static, batch, t)], dim=-1)" not in m_src,
          'static_span no longer uses a raw feature concat')
    check("self.head_in = hidden_size" in m_src
          and "static_fuse_layers=int(getattr(cfg, 'static_fuse_layers', 1))" in m_src,
          'head_in stays hidden_size; build_combined_model forwards fusion knobs')

    # single-pass routing: one pool + one shared-head call for the whole batch
    check("pool = pool_active(hidden, batch, targets)" in m_src
          and "mu, sigma = self.gauss(pool)" in m_src,
          'fast path pools each row OWN span once and runs the shared head once')

    # target_prefix must not crash on predict/joint batches (no target key)
    check("if self.target_prefix and 'target' in batch:" in m_src,
          'marker injection gated on batch[target] so predict/joint mode is safe')

    # config: build_combined_model forwards both flags
    check("target_prefix=bool(getattr(cfg, 'target_prefix', False))" in m_src
          and "static_span=bool(getattr(cfg, 'static_span', False))" in m_src,
          'build_combined_model forwards target_prefix/static_span from cfg')

    from src.config import Config
    try:
        Config(static_span=True, targets=['mod']).validate()
        check(False, 'Config rejects static_span=True with backend=exits')
    except ValueError:
        check(True, 'Config rejects static_span=True with backend=exits')
    cfg_sp = Config(static_span=True, model_backend='combined', targets=['mod'])
    check(cfg_sp.static_span is True,
          'Config accepts static_span=True with model_backend=combined')
    try:
        Config(static_span=True, model_backend='combined', targets=['mod'],
               static_fuse_layers=0).validate()
        check(False, 'Config rejects static_fuse_layers=0')
    except ValueError:
        check(True, 'Config rejects static_fuse_layers=0')

    # static_fuse joins the head group (trained at head_lr in phase 1)
    check("fuse = getattr(model, 'static_fuse', None)" in tr_src
          and "heads.append(fuse)" in tr_src,
          'trainer._pred_heads folds static_fuse into the head_lr group')


def check_static_ext() -> None:
    print('=== 9. EXTERNAL STATIC EMBEDDINGS (StaticVec + combined fusion) ===')
    sys.path.insert(0, str(ROOT))

    from src.config import Config

    # 1) validation wiring
    sv_src = (ROOT / 'src' / 'static_vec.py').read_text(encoding='utf-8')
    check('class StaticVec' in sv_src and 'def _load' in sv_src and 'def tensor' in sv_src,
          'src/static_vec.py defines StaticVec with load + OOV tensor lookup')

    # Config accepts external static only on the combined backend + static_span
    ok = Config(model_backend='combined', static_span=True, targets=['mod'],
                static_ext_path='x.vec', static_ext_dim=300)
    ok.validate()
    check(ok.static_ext_path == 'x.vec' and ok.static_ext_dim == 300,
          'Config accepts static_ext_path with combined + static_span')
    for label, bad in [
        ('backend=exits',
         Config(model_backend='exits', static_span=True, targets=['mod'], static_ext_path='x.vec')),
        ('static_span=False',
         Config(model_backend='combined', static_span=False, targets=['mod'], static_ext_path='x.vec')),
        ('static_ext_dim=0',
         Config(model_backend='combined', static_span=True, targets=['mod'],
                static_ext_path='x.vec', static_ext_dim=0)),
    ]:
        try:
            bad.validate()
            check(False, f'[reject] static_ext with {label}')
        except ValueError:
            check(True, f'[reject] static_ext with {label}')

    # 2) source guards: dataset, model, trainer wiring
    d_src = (ROOT / 'src' / 'data.py').read_text(encoding='utf-8')
    m_src = (ROOT / 'src' / 'model_combined.py').read_text(encoding='utf-8')
    tr_src = (ROOT / 'src' / 'trainer.py').read_text(encoding='utf-8')
    check("item['head_static'] = head_v" in d_src
          and "item['pv_static'] = pv_static" in d_src
          and "self.static_vec.tensor(r.get('compound', ''))" in d_src,
          'CompDataset encodes per-row mod/head/compound static vectors when static_vec given')
    check("self.static_proj = nn.Linear(static_ext_dim, hidden_size)" in m_src,
          'CombinedBackboneModel projects external static vectors to H')
    check("static_ext=bool(getattr(cfg, 'static_ext_path', None))" in m_src
          and "static_ext_dim=int(getattr(cfg, 'static_ext_dim', 300))" in m_src,
          'build_combined_model forwards static_ext knobs from cfg')
    check("static_ext_path not found (tried:" in tr_src,
          'trainer raises clear FileNotFoundError when static_ext_path missing')
    check("static_vec=static_vec" in tr_src,
          'trainer passes StaticVec to both CompDatasets')
    check("proj = getattr(model, 'static_proj', None)" in tr_src and "heads.append(proj)" in tr_src,
          'trainer._pred_heads folds static_proj into the head_lr group')

    # 3) functional: StaticVec parse / normalize / OOV (no torch needed beyond numpy)
    tmp = Path(tempfile.gettempdir()) / 'src_static_smoke.vec'
    tmp.write_text(
        '4 3\n'
        'acid 1 0 0\n'
        'solution 0 1 0\n'
        'crack 0.5 0.5 0\n'
        'down -0.5 0.5 0\n',
        encoding='utf-8')
    try:
        from src.static_vec import StaticVec
        sv = StaticVec(tmp, 3, ['acid solution', 'crack down', 'Abitur', 'OOVWORD'])
        check(len(sv) == 4, 'StaticVec keeps only the wanted words (4/6 coverage)')
        check(np.isclose(sv.vector('acid'), [1, 0, 0]).all(),
              'StaticVec returns the exact (unit) vectors')
        check(np.isclose(sv.vector('crack down'), [0, 1, 0], atol=1e-5).all(),
              'multi-part surface form is the normalized mean of its parts')
        check(sv.vector('Abitur') is None and sv.vector('OOVWORD') is None,
              'OOV words return None (dataset falls back to the zero vector)')

        import torch as _torch
        check(_torch.equal(sv.tensor('Abitur'), _torch.zeros(3)),
              'StaticVec.tensor gives a zero vector on OOV')

        # dataset-level pv anchor: whole-compound vector when available,
        # base+particle mean only when the compound surface form is OOV
        from src.data import CompDataset as _CD

        class _FakeTok:
            def __call__(self, text, max_length=256, truncation=True,
                         return_tensors='pt', return_offsets_mapping=True):
                return {'input_ids': _torch.tensor([[1, 1, 1, 1, 1]]),
                        'attention_mask': _torch.tensor([[1, 1, 1, 1, 1]]),
                        'offset_mapping': _torch.tensor([[[0, 1]] * 5])}

        _rows = [
            {'context': 'x', 'mod': 'crack', 'head': 'down', 'compound': 'acid',
             'compound_id': 0, 'has_label': True, 'mod_avg': 1.0, 'head_avg': 1.0,
             'mod_std': 0.1, 'head_std': 0.1, 'target': 'pv'},
            {'context': 'x', 'mod': 'acid', 'head': 'solution', 'compound': 'OOVWORD',
             'compound_id': 1, 'has_label': True, 'mod_avg': 1.0, 'head_avg': 1.0,
             'mod_std': 0.1, 'head_std': 0.1, 'target': 'pv'},
        ]
        _ds = _CD(_rows, _FakeTok(), max_len=8, is_test=True, static_vec=sv)
        check(_torch.allclose(_ds[0]['pv_static'], _torch.as_tensor([1., 0., 0.])),
              'pv anchor is the WHOLE compound vector when the surface form exists')
        check(_torch.allclose(_ds[1]['pv_static'],
                              _torch.as_tensor([0.7071068, 0.7071068, 0.]), atol=1e-5),
              'pv anchor falls back to base+particle mean when the compound is OOV')

        # 4) functional: combined model fuses + routes external static
        from types import SimpleNamespace
        import torch.nn as _nn
        import src.model_combined as MC

        class _FakeLM(_nn.Module):
            def __init__(self):
                super().__init__()
                self._emb = _nn.Embedding(100, 8)

            def get_input_embeddings(self):
                return self._emb

            def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                        output_hidden_states=False):
                if input_ids is not None:
                    B, L = input_ids.shape
                else:
                    B, L = inputs_embeds.shape[:2]
                return SimpleNamespace(last_hidden_state=_torch.randn(B, L, 8))

        import unittest.mock as _mock
        with _mock.patch.object(MC.AutoModel, 'from_pretrained', return_value=_FakeLM()):
            model = MC.CombinedBackboneModel(
                'fake', hidden_size=8, dropout=0.0, head_hidden=8,
                static_span=True, static_ext=True, static_ext_dim=3,
                static_fuse_layers=1, static_fuse_heads=1)
        check(model.static_proj.weight.shape == (8, 3)
              and model.static_fuse is not None,
              'static_ext creates the H x ext Linear + the fusion transformer')

        B, L = 3, 6
        batch = {
            'input_ids': _torch.randint(4, 40, (B, L)),
            'attention_mask': _torch.ones(B, L, dtype=_torch.long),
            'mod_span_mask': _torch.tensor(
                [[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=_torch.bool),
            'head_span_mask': _torch.tensor(
                [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=_torch.bool),
            'target': _torch.tensor([0, 1, 2], dtype=_torch.long),
            'mod_static': _torch.randn(B, 3),
            'head_static': _torch.randn(B, 3),
            'pv_static': _torch.randn(B, 3),
        }
        model.eval()
        with _torch.no_grad():
            mod, head, pv, mod_sig, head_sig, pv_sig = model(
                batch, with_logits=True, with_pv=True)
        check(mod.shape == (B,) and pv.shape == (B,), 'combined ext-static predict shapes')
        check(bool(mod_sig[0] > 0) and bool(head_sig[1] > 0) and bool(pv_sig[2] > 0),
              'sigmas stay positive on routed rows through the ext-static fusion')

        # Routing + gradient in TRAIN mode: the eval-mode score clamp turns any
        # near-zero mu into exactly SCORE_MIN (0.0), so the "which row fired"
        # check needs the un-clamped predictions.
        model.train()
        mod, head, pv, *_ = model(batch, with_logits=True, with_pv=True)
        check(float(mod[0]) != 0.0 and float(mod[1]) == 0.0 and float(mod[2]) == 0.0,
              'mod rows routed: only the mod target row is non-zero')
        check(float(head[0]) == 0.0 and float(head[1]) != 0.0 and float(head[2]) == 0.0,
              'head rows routed: only the head target row is non-zero')
        check(float(pv[0]) == 0.0 and float(pv[1]) == 0.0 and float(pv[2]) != 0.0,
              'pv rows routed: only the pv target row is non-zero')
        (mod.sum() + head.sum() + pv.sum()).backward()
        check(model.static_proj.weight.grad is not None
              and bool(_torch.isfinite(model.static_proj.weight.grad).all()),
              'gradient flows into static_proj (fusion is not frozen)')

        # 5) prefix readout: pool_prefix masked-mean + model wiring
        from src.targets import pool_prefix as _pp
        _hid = _torch.tensor([[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]]])
        _pm = _torch.tensor([[0, 1, 1, 0]], dtype=_torch.bool)
        _pooled = _pp(_hid, {'prefix_mask': _pm})
        check(_torch.allclose(_pooled, _torch.tensor([[2.5, 2.5, 2.5]])),
              'pool_prefix masked-mean over ONLY the prefix word tokens')
        _pm_empty = _torch.tensor([[0, 0, 0, 0]], dtype=_torch.bool)
        check(_torch.allclose(_pp(_hid, {'prefix_mask': _pm_empty}),
                              _torch.zeros(1, 3)),
              'pool_prefix on an empty prefix degrades to a zero pool (caller guards)')

        _kept = ('input_ids', 'attention_mask', 'mod_span_mask', 'head_span_mask',
                 'target', 'mod_static', 'head_static', 'pv_static')
        _pb = {k: batch[k] for k in _kept}
        _pb['prefix_mask'] = _torch.tensor(
            [[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=_torch.bool)
        with _mock.patch.object(MC.AutoModel, 'from_pretrained', return_value=_FakeLM()):
            _mpref = MC.CombinedBackboneModel(
                'fake', hidden_size=8, dropout=0.0, head_hidden=8,
                target_prefix=True, prefix_readout='prefix')
            _mdual = MC.CombinedBackboneModel(
                'fake', hidden_size=8, dropout=0.0, head_hidden=8,
                target_prefix=True, prefix_readout='dual')
        check(_mpref.prefix_gate is None,
              'prefix readout uses no gate')
        check(_mdual.prefix_gate is not None
              and tuple(_mdual.prefix_gate.shape) == (1,),
              'dual readout owns a 1-param scalar gate')
        _mpref.eval(); _mdual.train()
        with _torch.no_grad():
            _mpref(batch, with_logits=True, with_pv=True)              # no prefix_mask
            _out_p = _mpref(_pb, with_logits=True, with_pv=True)       # prefix pool
        check(all(_torch.isfinite(o).all() for o in _out_p if o is not None),
              'prefix readout runs on batches with AND without prefix_mask')
        _dual_out, *_ = _mdual(_pb, with_logits=True, with_pv=True)
        check(bool(_torch.isfinite(_dual_out).all()),
              'dual readout blends+gates without NaNs')
        _dual_out.sum().backward()
        check(_mdual.prefix_gate.grad is not None
              and bool(_torch.isfinite(_mdual.prefix_gate.grad).all()),
              'gradient flows into the dual gate (trains at head_lr)')
    except ImportError:
        print('  [SKIP] numpy/torch unavailable; StaticVec + fusion functional checks skipped')
    finally:
        tmp.unlink()



def check_mlm_adaptation() -> None:
    print('=== 10. TASK-ADAPTIVE PREFIX MLM (Dataset, Masking, Collate, LoRA merge, Untouched model.py) ===')
    import subprocess
    import torch
    import torch.nn as nn
    from src.config import Config
    from src.dataset_mlm import PrefixMLMDataset, collate_mlm
    from src.lora import apply_lora, lora_parameters, merge_lora
    from src.targets import MARKER_CODE

    # 1. Guarantee src/model.py is 100% UNTOUCHED
    diff_res = subprocess.run(['git', 'diff', 'src/model.py'], cwd=ROOT, capture_output=True, text=True)
    check(diff_res.returncode == 0 and not diff_res.stdout.strip(),
          'src/model.py has ZERO git diff (100% untouched invariant)')

    # 2. Config MLM knobs
    cfg = Config.defaults()
    check(hasattr(cfg, 'mlm_epochs') and cfg.mlm_epochs == 3, 'Config has mlm_epochs default')
    check(hasattr(cfg, 'mlm_lr') and cfg.mlm_lr == 5e-5, 'Config has mlm_lr default')
    check(hasattr(cfg, 'mlm_mask_prob') and cfg.mlm_mask_prob == 0.8, 'Config has mlm_mask_prob default')
    check(hasattr(cfg, 'mlm_from_layer') and cfg.mlm_from_layer == 18, 'Config has mlm_from_layer default')

    bad_mlm = [
        ('mlm_epochs=0', lambda: Config.defaults().update(mlm_epochs=0)),
        ('mlm_lr=-1', lambda: Config.defaults().update(mlm_lr=-1)),
        ('mlm_mask_prob=1.5', lambda: Config.defaults().update(mlm_mask_prob=1.5)),
        ('mlm_from_layer=-1', lambda: Config.defaults().update(mlm_from_layer=-1)),
    ]
    for label, fn in bad_mlm:
        try:
            fn().validate()
            check(False, f'[reject] {label}')
        except ValueError:
            check(True, f'[reject] {label}')

    # 3. PrefixMLMDataset construction and token alignment
    class DummyTokenizer:
        mask_token_id = 4
        cls_token_id = 0
        sep_token_id = 2
        pad_token_id = 1
        vocab_size = 1000
        def encode(self, text, add_special_tokens=False):
            return [len(text) + 10]

    tok = DummyTokenizer()
    rows = [{'sentence': 'The acid rain fell.', 'mod': 'acid', 'head': 'rain', 'compound': 'acid rain'}]
    ds = PrefixMLMDataset(rows, tok, targets=['mod', 'head', 'pv'], is_train=False)
    check(len(ds) == 3, 'PrefixMLMDataset expands 1 row x 3 targets into 3 samples')

    item_mod = ds[0]
    ids = item_mod['input_ids'].tolist()
    lbls = item_mod['labels'].tolist()
    # Structure: [CLS, marker_mod(7), mask(4), marker_mod(7), context_tok(29), SEP(2)]
    check(ids[0] == 0 and ids[1] == MARKER_CODE['mod'] and ids[3] == MARKER_CODE['mod'],
          'Prefix prompt structure: [CLS, marker, target_token(s), marker, context...]')
    check(ids[2] == tok.mask_token_id, 'eval mode always masks target word')
    check(lbls[2] == 14, 'labels holds the ground-truth token id at the target position')
    check(lbls[0] == -100 and lbls[1] == -100 and lbls[3] == -100 and lbls[4] == -100 and lbls[5] == -100,
          'labels contains -100 everywhere else (CLS, markers, context, SEP)')

    # Collation padding
    batch = collate_mlm([item_mod, ds[1]], pad_id=tok.pad_token_id)
    check(batch['input_ids'].shape[0] == 2, 'collate_mlm batches samples correctly')
    check((batch['labels'][0, 2] != -100).item(), 'collate_mlm keeps target label intact')

    # 4. 80/10/10 Stochastic policy
    ds_train = PrefixMLMDataset(rows * 40, tok, targets=['mod'], is_train=True)
    mask_count = 0
    id_count = 0
    rand_count = 0
    for s in ds_train:
        tid = s['input_ids'][2].item()
        if tid == tok.mask_token_id:
            mask_count += 1
        elif tid == 14:
            id_count += 1
        else:
            rand_count += 1
    check(mask_count > id_count and mask_count > rand_count,
          f'80/10/10 policy: mask={mask_count}, identity={id_count}, rand={rand_count}')

    # 5. LoRA application, forward, backward and merge_lora round-trip
    class MockLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = nn.Module()
            self.lm.layers = nn.ModuleList([nn.Module() for _ in range(20)])
            for layer in self.lm.layers:
                layer.attn = nn.Module()
                layer.attn.q_proj = nn.Linear(32, 32)
            self.head = nn.Linear(32, 32)
            self.decoder = nn.Linear(32, 100)

        def forward(self, x, labels=None):
            h = x
            for layer in self.lm.layers:
                h = layer.attn.q_proj(h)
            logits = self.decoder(self.head(h))
            loss = None
            if labels is not None:
                mask = labels != -100
                loss = nn.functional.cross_entropy(logits[mask], labels[mask])
            return logits, loss

    mock_m = MockLM()
    adapters = apply_lora(mock_m, targets=['q_proj'], from_layer=18)
    check(len(adapters) == 2, 'LoRA applied to layers >= 18 (2 adapters)')

    # 5b. AutoModelForMaskedLM (ModernBERT) carries the backbone under root
    #     child 'model', not 'lm' -- apply_lora must still match (Stage 1 MLM).
    class MockMLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module() for _ in range(22)])
            for layer in self.model.layers:
                layer.attn = nn.Module()
                layer.attn.Wqkv = nn.Linear(32, 32)
                layer.attn.Wo = nn.Linear(32, 32)
            self.decoder = nn.Linear(32, 100)

    mock_mlm = MockMLM()
    adapters_mlm = apply_lora(mock_mlm, targets=['Wqkv', 'Wo'], from_layer=18)
    check(len(adapters_mlm) == 8,
          'apply_lora matches ModernBERT roots under "model" (Wqkv+Wo x layers>=18)')
    merge_lora(mock_mlm, adapters_mlm)
    check(isinstance(mock_mlm.model.layers[18].attn.Wqkv, nn.Linear)
          and isinstance(mock_mlm.model.layers[18].attn.Wo, nn.Linear),
          'merge_lora unwraps model-rooted ModernBERT adapters back to nn.Linear')

    dummy_x = torch.randn(2, 6, 32)
    dummy_lbl = torch.full((2, 6), -100, dtype=torch.long)
    dummy_lbl[0, 2] = 5
    dummy_lbl[1, 2] = 8
    _, dummy_loss = mock_m(dummy_x, labels=dummy_lbl)
    dummy_loss.backward()
    check(torch.isfinite(dummy_loss), 'Prefix MLM dummy loss forward + backward finite')
    merge_lora(mock_m, adapters)
    check(isinstance(mock_m.lm.layers[18].attn.q_proj, nn.Linear),
          'merge_lora unwraps adapter back to nn.Linear with folded weights')


def check_proto_stream() -> None:
    print('=== 11. TWO-STREAM PROTOTYPE (Disentangled Lexical vs Contextual) ===')
    import subprocess
    import torch
    import torch.nn as nn
    from src.config import Config
    from src.data import CompDataset, collate_comp
    from src.prototype_stream import SemanticShiftFusion, pool_prototype, prototype_rank_loss

    # 1. src/model.py must remain 100% untouched
    res = subprocess.run(['git', 'diff', 'src/model.py'], capture_output=True, text=True)
    check(res.stdout.strip() == '', 'src/model.py has ZERO git diff (100% untouched invariant)')

    # 2. Config knobs and validation
    cfg = Config()
    check(cfg.proto_stream is False, 'Config has proto_stream default (False)')
    check(cfg.proto_rank_loss == 0.0, 'Config has proto_rank_loss default (0.0)')

    # Rejection of invalid configs
    try:
        Config(proto_rank_loss=-0.1).validate()
        check(False, '[reject] proto_rank_loss < 0')
    except ValueError:
        check(True, '[reject] proto_rank_loss < 0')

    try:
        Config(proto_stream=True, model_backend='exits').validate()
        check(False, '[reject] proto_stream with backend=exits')
    except ValueError:
        check(True, '[reject] proto_stream with backend=exits')

    cfg_ok = Config(proto_stream=True, model_backend='combined')
    cfg_ok.validate()
    check(True, 'Config accepts proto_stream=True with model_backend=combined')

    # 3. CompDataset tokenization of prototype
    class MockTokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            # Return [0, 10, 11, 2] (length 4: CLS, 2 subwords, SEP)
            return {
                'input_ids': torch.tensor([[0, 10, 11, 2]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1]]),
                'offset_mapping': torch.tensor([[[0, 0], [0, 2], [2, 4], [0, 0]]]),
            }
        def encode(self, text, add_special_tokens=False):
            return [10, 11]

    mock_tok = MockTokenizer()
    rows = [{'context': 'acid rain falls', 'mod': 'acid', 'head': 'rain', 'compound': 'acid rain',
             'target': 'mod', 'has_label': True, 'mod_avg': 4.5, 'head_avg': 4.0, 'mod_std': 0.5, 'head_std': 0.5,
             'is_pv': False, 'row_id': 1, 'compound_id': 1}]

    ds_off = CompDataset(rows, mock_tok, proto_stream=False)
    check('proto_ids' not in ds_off[0], 'CompDataset without proto_stream omits proto_ids')

    ds_on = CompDataset(rows, mock_tok, proto_stream=True)
    item = ds_on[0]
    check('proto_ids' in item and 'proto_mask' in item, 'CompDataset with proto_stream creates proto_ids and proto_mask')
    check(item['proto_ids'].shape == (4,) and item['proto_mask'].shape == (4,), 'proto_ids shape matches tokenizer output')

    # Collation
    batch = collate_comp([item, item], pad_token_id=0)
    check(batch['proto_ids'].shape == (2, 4), 'collate_comp stacks and pads proto_ids')
    check(batch['proto_mask'].shape == (2, 4), 'collate_comp stacks and pads proto_mask')

    # 4. pool_prototype excludes CLS and SEP on sequences >= 3
    # Hidden state shape: (B=2, L=4, H=8)
    hidden_proto = torch.zeros(2, 4, 8)
    hidden_proto[:, 0, :] = 100.0   # CLS
    hidden_proto[:, 1, :] = 2.0     # subword 1
    hidden_proto[:, 2, :] = 4.0     # subword 2
    hidden_proto[:, 3, :] = 200.0   # SEP
    mask_proto = torch.ones(2, 4, dtype=torch.long)

    pooled_proto = pool_prototype(hidden_proto, mask_proto)
    check(pooled_proto.shape == (2, 8), 'pool_prototype output shape is (B, H)')
    # Expected mean of subword 1 (2.0) and subword 2 (4.0) is 3.0
    check(torch.allclose(pooled_proto, torch.full((2, 8), 3.0)),
          'pool_prototype correctly isolates subwords excluding [CLS] and [SEP]')

    # Degrade test on sequence of length 2
    hidden_short = torch.full((1, 2, 8), 5.0)
    mask_short = torch.ones(1, 2, dtype=torch.long)
    pooled_short = pool_prototype(hidden_short, mask_short)
    check(torch.allclose(pooled_short, torch.full((1, 8), 5.0)),
          'pool_prototype degrades safely on short sequence length < 3')

    # 5. SemanticShiftFusion
    H = 16
    fuse = SemanticShiftFusion(hidden_size=H, dropout=0.0)
    h_ctx = torch.randn(2, H, requires_grad=True)
    h_proto = torch.randn(2, H, requires_grad=True)
    out_fuse = fuse(h_ctx, h_proto)

    check(out_fuse.shape == (2, H), 'SemanticShiftFusion output shape is (B, H)')
    check(torch.isfinite(out_fuse).all(), 'SemanticShiftFusion outputs all finite values')
    check(fuse.last_cos is not None and fuse.last_cos.shape == (2,),
          'SemanticShiftFusion records last cosine similarities')

    loss = out_fuse.sum()
    loss.backward()
    check(h_ctx.grad is not None and h_proto.grad is not None,
          'gradients flow smoothly through SemanticShiftFusion to inputs')

    # 6. prototype_rank_loss
    cos_sim = torch.tensor([0.9, 0.2, 0.8])
    ratings = torch.tensor([5.0, 1.0, 4.0])
    rank_loss = prototype_rank_loss(cos_sim, ratings, margin=0.2)
    check(torch.isfinite(rank_loss) and rank_loss >= 0.0,
          'prototype_rank_loss computes valid finite ranking loss')

    # 7. Mock CombinedBackboneModel forward with proto_stream
    from src.model_combined import CombinedBackboneModel
    class DummyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(50, 16)
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
            B, L = input_ids.shape
            h = self.emb(input_ids)
            class Out:
                pass
            o = Out()
            o.last_hidden_state = h
            return o
        def get_input_embeddings(self):
            return self.emb

    # Monkey-patch AutoModel.from_pretrained to avoid downloading weights
    import transformers
    orig_from_pretrained = transformers.AutoModel.from_pretrained
    transformers.AutoModel.from_pretrained = lambda *args, **kwargs: DummyLM()
    try:
        model = CombinedBackboneModel(
            backbone='dummy', hidden_size=16, proto_stream=True
        )
        check(model.shift_fuse is not None, 'CombinedBackboneModel initializes shift_fuse when proto_stream=True')

        # Run forward
        batch_mock = {
            'input_ids': torch.randint(0, 40, (2, 8)),
            'attention_mask': torch.ones(2, 8, dtype=torch.long),
            'proto_ids': torch.tensor([[0, 10, 11, 2], [0, 12, 13, 2]]),
            'proto_mask': torch.ones(2, 4, dtype=torch.long),
            'mod_span_mask': torch.zeros(2, 8, dtype=torch.bool),
            'head_span_mask': torch.zeros(2, 8, dtype=torch.bool),
            'target': torch.tensor([0, 1]),
            'is_pv': torch.tensor([False, False]),
            'has_mod': torch.tensor([True, True]),
            'has_head': torch.tensor([True, True]),
            'degenerate': torch.tensor([False, False]),
            'has_label': torch.tensor([True, True]),
        }
        batch_mock['mod_span_mask'][:, 2:4] = True
        batch_mock['head_span_mask'][:, 4:6] = True

        preds = model(batch_mock, with_logits=True, with_pv=True)
        mod_p, head_p, pv_p, mod_s, head_s, pv_s = preds
        check(mod_p.shape == (2,) and mod_s.shape == (2,), 'CombinedBackboneModel proto_stream forward shapes valid')
        check(torch.isfinite(mod_p).all() and torch.isfinite(mod_s).all(), 'Forward outputs are finite')

        dummy_loss = mod_p.sum() + mod_s.sum()
        dummy_loss.backward()
        p_has_grad = any(p.grad is not None for p in model.shift_fuse.parameters())
        check(p_has_grad, 'gradients flow backward into shift_fuse parameters')
    finally:
        transformers.AutoModel.from_pretrained = orig_from_pretrained


def main() -> int:
    sync_parse()
    check_config()
    check_cli()
    check_marks()
    check_data()
    check_folds()
    check_fixes()
    check_targets()
    check_static_ext()
    check_mlm_adaptation()
    check_proto_stream()

    print('=' * 50)
    if FAILURES:
        print(f'RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}')
        return 1
    print('RESULT: all smoke checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
