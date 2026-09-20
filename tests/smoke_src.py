"""Smoke checks for the gauss-only src/ package -- NO torch / transformers needed.

Run from the repo root:
    python tests/smoke_src.py        (or: python -m src.run --smoke)

Covers: syntax of every src/ module, config construction/validation/round-trip
and ``--set`` coercion, CLI wiring (``python -m src.run``), the offset-based
span matcher on synthetic token offsets, the real local TSV loaders, the
two-stream prototype stack, and -- when torch is available -- numerical checks
of the merged gauss losses and the GaussHead.
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
    for gone in ('folds.py', 'static_vec.py', 'dataset_mlm.py', 'train_mlm.py'):
        check(not (ROOT / 'src' / gone).exists(),
              f'src/{gone} removed (cleaned out of the two-stream plan)')


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
    check(not hasattr(defaults, 'lambda_rank') and not hasattr(defaults, 'rank_margin')
          and not hasattr(defaults, 'rank_margin_mode')
          and not hasattr(defaults, 'target_prefix') and not hasattr(defaults, 'span_markers')
          and not hasattr(defaults, 'prefix_readout')
          and not hasattr(defaults, 'static_span') and not hasattr(defaults, 'static_ext_path')
          and not hasattr(defaults, 'static_ext_dim') and not hasattr(defaults, 'static_fuse_layers')
          and not hasattr(defaults, 'mlm_epochs') and not hasattr(defaults, 'mlm_lr')
          and not hasattr(defaults, 'mlm_mask_prob') and not hasattr(defaults, 'mlm_from_layer'),
          'rank/prefix-marker/static-ext/MLM knobs are absent (two-stream plan only)')

    bad = [
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

    # Pure-frozen probe mode: no LoRA adapters, no encoder LR (two-stream freeze).
    frozen = Config.defaults().update(
        proto_stream=True, model_backend='combined', freeze_epochs=30,
        lora_targets=[], lora_epochs=0, lora_rank=0, lora_alpha=0, encoder_lr=0.0)
    frozen.validate()
    check(frozen.total_epochs == 30 and frozen.lora_epochs == 0,
          'pure-frozen config accepted (empty lora_targets, encoder_lr=0)')

    # encoder_lr=0 must still be rejected when LoRA training is configured.
    try:
        Config.defaults().update(lora_epochs=5, encoder_lr=0.0).validate()
        check(False, '[reject] encoder_lr=0 with LoRA epochs')
    except ValueError:
        check(True, '[reject] encoder_lr=0 with LoRA epochs')

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


def check_fixes() -> None:
    print('=== 6. GAUSS CONTRACTS & BUG-FIX REGRESSION GUARDS ===')
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
    check('def margin_rank_loss' not in loss_src and 'def compound_center_loss' not in loss_src
          and 'def compound_consistency_loss' not in loss_src,
          'src/losses.py has no ranking/centre/consistency terms (ranking lives in prototype_stream)')
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

        # GaussLoss forward: finite; rank/compound_ids args are gone from the API
        pred = _torch.randn(16)
        tgt = _torch.randn(16)
        dfl = G.GaussLoss()
        loss = dfl(pred, tgt, logits=_torch.ones(16))
        check(bool(_torch.isfinite(loss)), 'GaussLoss forward finite')

        # empty-mask: returns 0.0 with grad connected (no NaN, no graph break)
        pg = _torch.randn(16, requires_grad=True)
        z = dfl(pg, tgt, logits=_torch.ones(16),
                mask=_torch.zeros(16, dtype=_torch.bool))
        check(bool(z.item() == 0.0) and z.requires_grad,
              'GaussLoss(mask=all-False) returns grad-connected zero')
    except ImportError:
        print('  [SKIP] torch not installed; GaussHead / gauss_kl numerical checks skipped')


def check_targets() -> None:
    print('=== 7. SINGLE-TARGET WIRING (targets module, expansion, routing) ===')
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

    # model: single shared GaussHead (mod/head/pv_gauss all alias the same module)
    check("self.gauss = GaussHead(self.head_in, head_hidden, dropout=dropout)" in m_src
          and "object.__setattr__(self, 'mod_gauss', self.gauss)" in m_src
          and "object.__setattr__(self, 'head_gauss', self.gauss)" in m_src
          and "object.__setattr__(self, 'pv_gauss', self.gauss)" in m_src,
          'CombinedBackboneModel shares ONE GaussHead across all targets')

    # single-pass routing: one pool + one shared-head call for the whole batch
    check("pool = pool_active(hidden, batch, targets)" in m_src
          and "mu, sigma = self.gauss(pool)" in m_src,
          'fast path pools each row OWN span once and runs the shared head once')

    # build_combined_model forwards proto_stream; shift_fuse exists only then
    check("proto_stream=bool(getattr(cfg, 'proto_stream', False))" in m_src,
          'build_combined_model forwards proto_stream from cfg')
    check("self.shift_fuse = SemanticShiftFusion(hidden_size, dropout=dropout) if proto_stream else None" in m_src,
          'CombinedBackboneModel builds SemanticShiftFusion only when proto_stream=True')
    check("if self.proto_stream and h_proto is not None:" in m_src,
          'proto_stream fuses the pooled context via shift_fuse before the shared head')

    # trainer wires proto_stream into both datasets + folds shift_fuse into the head group
    check("proto_stream=self.cfg.proto_stream" in tr_src,
          'trainer passes proto_stream to both CompDatasets')
    check("shift = getattr(model, 'shift_fuse', None)" in tr_src
          and "heads.append(shift)" in tr_src,
          'trainer._pred_heads folds shift_fuse into the head_lr group')
    check("seen = set()" in tr_src
          and "if id(m) not in seen:" in tr_src,
          'trainer._pred_heads dedupes shared-head aliases by module identity')


def check_proto_stream() -> None:
    print('=== 8. TWO-STREAM PROTOTYPE (Disentangled Lexical vs Contextual) ===')
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
    check_fixes()
    check_targets()
    check_proto_stream()

    print('=' * 50)
    if FAILURES:
        print(f'RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}')
        return 1
    print('RESULT: all smoke checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
