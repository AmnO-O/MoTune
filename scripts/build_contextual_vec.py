"""Build a .vec of CONTEXTUAL-encoder snapshots for the static anchors.

Same end product as ``build_static_vec.py`` (a filtered ``.vec`` that the
trainer's ``StaticVec`` reads) but the vectors are CLS-pooled embeddings of
each wanted surface form through a STRONG multilingual transformer (default
``BAAI/bge-m3``, 1024-d). This puts EN + DE in one meaningful semantic space
and bakes German morphology into the vectors, unlike the fastText crawl (whose
EN and DE halves are two separate spaces glued into one file).

Usage (repo root)::

    python scripts/build_contextual_vec.py --out vectors/bge_m3_1024.vec

Then run the combined-static arm with ``static_ext_path`` pointing at it and
``static_ext_dim=1024``.

Encode shapes: CLS pooling (BGE-family convention), L2-normalized, one row
per unique surface token the dataset needs (same wanted set as the fastText
builder). A one-time model download (~2.3 GB) happens on first call.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _load_shared() -> 'module':
    """Import build_static_vec.py's collect_wanted_words/report helpers."""
    spec = importlib.util.spec_from_file_location(
        'bsv', Path(__file__).resolve().parent / 'build_static_vec.py')
    bsv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsv)
    return bsv


def encode_batch(cfg_module, words, model_name: str, device: str) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer
    import numpy as np

    if not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    at = AutoTokenizer.from_pretrained(model_name)
    am = AutoModel.from_pretrained(model_name).to(device).eval()

    out: dict = {}
    with torch.no_grad():
        for i in range(0, len(words), 128):
            ch = words[i:i + 128]
            enc = at(ch, padding=True, truncation=True, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                last = am(**enc).last_hidden_state            # (B, T, D)
                reps = last[:, 0, :].float()                  # CLS pooling
            reps = torch.nn.functional.normalize(reps, p=2, dim=1)
            for w, v in zip(ch, reps.cpu().numpy()):
                out[w] = v.astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='vectors/bge_m3_1024.vec')
    ap.add_argument('--model', default='BAAI/bge-m3')
    ap.add_argument('--device', default='cuda', help='cuda | cpu (auto-falls back)')
    args = ap.parse_args()

    bsv = _load_shared()
    wanted = bsv.collect_wanted_words()
    dim = 1024

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'encoding {len(wanted)} tokens with {args.model}...', flush=True)
    reps = encode_batch(bsv, sorted(wanted), args.model, args.device)
    missed = sorted(set(wanted) - set(reps))
    print(f'encoded {len(reps)}/{len(wanted)} tokens '
          f'({100 * len(reps) / len(wanted):.1f}%); OOV: {len(missed)}')

    with open(out, 'w', encoding='utf-8') as f:
        for w in sorted(wanted & set(reps)):
            f.write(f'{w} ' + ' '.join(f'{v:.6f}' for v in reps[w]) + '\n')
    print('built:', out, f'({out.stat().st_size / 1e6:.2f} MB)')

    bsv.report(out, dim, wanted, 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())