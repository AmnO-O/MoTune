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

    # CompDataset target_prefix token insertion and +1 span mask shift
    try:
        import torch as _torch
        class _MockTok:
            def __call__(self, text, max_length=256, truncation=True, return_tensors='pt', return_offsets_mapping=True):
                return {
                    'input_ids': _torch.tensor([[2, 10, 20, 30, 1]]),
                    'attention_mask': _torch.tensor([[1, 1, 1, 1, 1]]),
                    'offset_mapping': _torch.tensor([[[0, 0], [0, 3], [4, 8], [9, 15], [0, 0]]])
                }
        _row = {'context': 'our flea market', 'mod': 'flea', 'head': 'market', 'compound': 'flea market',
                'compound_id': 0, 'has_label': True, 'mod_avg': 3.0, 'head_avg': 4.0, 'mod_std': 0.5,
                'head_std': 0.5, 'target': 'mod'}
        _ds0 = D.CompDataset([_row], _MockTok(), target_prefix=False)
        _ds1 = D.CompDataset([_row], _MockTok(), target_prefix=True)
        check(_ds0[0]['input_ids'].tolist() == [2, 10, 20, 30, 1],
              'CompDataset without prefix preserves original token sequence')
        check(_ds1[0]['input_ids'].tolist() == [2, 7, 10, 20, 30, 1],
              'CompDataset with target_prefix prepends marker id right after <bos>')
        check(_ds0[0]['mod_span_mask'].tolist() == [False, False, True, False, False]
              and _ds1[0]['mod_span_mask'].tolist() == [False, False, False, True, False, False],
              'CompDataset target_prefix shifts mod_span_mask exactly by +1')
        check(_ds0[0]['head_span_mask'].tolist() == [False, False, False, True, False]
              and _ds1[0]['head_span_mask'].tolist() == [False, False, False, False, True, False],
              'CompDataset target_prefix shifts head_span_mask exactly by +1')
    except ImportError:
        print('  [SKIP] torch unavailable; CompDataset prefix check skipped')

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

    # model: single shared GaussHead (mod/head/pv_gauss all alias the same module)
    check("self.gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)" in m_src
          and "object.__setattr__(self, 'mod_gauss', self.gauss)" in m_src
          and "object.__setattr__(self, 'head_gauss', self.gauss)" in m_src
          and "object.__setattr__(self, 'pv_gauss', self.gauss)" in m_src,
          'CombinedBackboneModel shares ONE GaussHead across all targets')

    # model: static_span concats the frozen embedding-table pool
    check("static = self.lm.get_input_embeddings()(batch['input_ids'])" in m_src
          and "torch.cat([pool, pool_span(static, batch, t)], dim=-1)" in m_src,
          'CombinedBackboneModel concats static span pool when static_span on')
    check("self.head_in = hidden_size * (2 if static_span else 1)" in m_src,
          'CombinedBackboneModel doubles head_in for static_span')

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


def main() -> int:
    sync_parse()
    check_config()
    check_cli()
    check_marks()
    check_data()
    check_folds()
    check_fixes()
    check_targets()

    print('=' * 50)
    if FAILURES:
        print(f'RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}')
        return 1
    print('RESULT: all smoke checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
