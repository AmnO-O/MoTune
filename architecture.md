# Kiến trúc — luồng dữ liệu & chiều tensor (input → output)

Ký hiệu: `B` = batch, `S` = seq_len, `H` = hidden (mặc định **768** cho mmBERT-base),
`N` = số transformer blocks của backbone (LoRA window, mid-5 đều dựa vào `N`).
Mọi chiều ghi theo thứ tự tensor PyTorch `(…)`. Số lấy trực tiếp từ
`mm/config.py`, `mm/model.py`, `mm/heads.py` (dòng tham chiếu trong ngoặc).

---

## 0. Config backbone — nguồn chân lý về `N` (quyết định mid & LoRA)

| knob | default | dòng | ý nghĩa |
|---|---|---|---|
| `backbone` | `jhu-clsp/mmBERT-base` | config.py:49 | backbone chứa LoRA + dự đoán MLM |
| `hidden_size` | `768` | config.py:50 | = `H` |
| `context_pool` | `mean+cls` | config.py:54 | pool nhánh context; `mean+cls` → dim `2H` |
| `use_lm_features` | `False` | config.py:74 | thêm 4 feats từ LM span stats |
| `use_proto_cos` | `False` | config.py:81 | thêm 1 giá trị cos literalness mỗi nhánh |
| `lora_from_layer` | `18` | config.py:143 | LoRA chỉ lên layer có index ≥ n |
| `span_layers` | `None` (auto) | config.py:61 | 5 lớp giữa cho nhánh span |

---

## 1. Flow chung `_features` — nền cho mọi nhánh (model.py:583-654)

```
input_ids[B,S] / attention_mask[B,S] / mod_span_mask[B,S] / head_span_mask[B,S]
  │  (LM branch, need_lm=True)
  ▼
backbone(input_ids, attention_mask, output_hidden_states=True)
  ▼
outputs.hidden_states: tuple gồm (N+1) phần tử
   [0] embedding   → [B,S,H]
   [1..N] block 1..N → [B,S,H]
```

### a) Nhánh span (literalness) — đọc LỚP GIỮA (model.py:615-634)
- Resolve lớp được chọn (`span_layers=None`):
  - `N+1 > 8` → `mid = (N+1)//2`, lấy **5 lớp**: `range(mid-2, mid+3)`
  - `N+1 ≤ 8` → `(-1,)` = **lớp cuối** (fallback cho backbone nhỏ)
- `span_hidden = mean(stack of 5 lớp [B,S,H]) → [B,S,H]` (model.py:630-631)
- `mod_emb   = mod_pool(span_hidden, mod_span_mask)   → [B,H]`
- `head_emb  = head_role_pool(span_hidden, head_span_mask) → [B,H]`

### b) Nhánh context — đọc LỚP CUỐI (`hidden` = hidden_states[-1]) (model.py:650-666)
- Mặc định `context_layers=None`: context = **lớp cuối** (global + sâu nhất cho mmBERT/ModernBERT — block 21).
- Nếu chỉ định `context_layers=(10,16,22)`: mean-pool 3 hidden-states đó (global attention của mmBERT-base: block 9/15/21) trước khi pool context.
- `mean_emb = _masked_mean(context_hidden, attention_mask) → [B,H]`
- `context_emb = CLS[cls] / mean[mean] / cat(mean,CLS)[mean+cls] → [B, context_dim]`
  - `mean+cls` (mặc định): `[B, 2H]` = `[B,1536]`

### c) Độ dài span (model.py:644-646)
- `mod_len  = (mod_span_mask.sum / seq_len)  → [B,1]`
- `head_len = (head_span_mask.sum / seq_len) → [B,1]`

### d) LM span stats — chỉ khi `use_lm_features=True` (model.py:521-560)
- 1 forward MLM phụ; per span: avg logP + entropy
- `→ [B, 4]` (2 span × 2 thống kê) (model.py:531/535 zeros hoặc cat)

### e) Ghép `features` (model.py:648-653)
```
feats = [mod_emb, head_emb, mod_len, head_len, context_emb]
        (+ lm_stats nếu use_lm_features)
features = cat(feats, dim=1)   # concat theo chiều cuối
feature dim (default) = H + H + 1 + 1 + 2H = 4H + 2 = 3074
feature dim (+lm)     = 3074 + 4 = 3078
```
Trả về: `(mod_emb[B,H], head_emb[B,H], features[B,·], logits)`.

---

## 2. Header `_build_head` / `GaussHead` (mm/heads.py:17-38, mm/model.py:439-458)

Shared MLP trunk + head cụ thể:

```
features[B, 3074]   (default)
   │ mod branch                        │ head branch
   │ + cos nếu use_proto_cos: [B,3075] │ + cos: [B,3075]
   ▼                                   ▼
GaussHead(in=3074, hidden=128)       GaussHead(in=3074, hidden=128)
   │ Linear(3074→128) + Tanh + Dropout   │ Linear(3074→128) + Tanh + Dropout
   │ [B,128]                             │ [B,128]
   ▼                                     ▼
mu:  Linear(128→1) → [B]  logvar: Linear(128→1) → [B]
sigma = softplus(logvar)+floor(0.05) → [B]  (>0 luôn)
```
- `mu`  = điểm dự đoán (clamp [min,max] khi infer)
- `sigma` = độ bất định — cùng băng MLM logits để chung vòng train
- Softmax/ordinal/regression heads nằm độc lập ở `heads.py` (đụng `GaussHead` riêng, không đổi shared)

Tham chiếu: model.py:507-514 (mod/head_gauss + regressor), heads.py:17-38.

---

## 3. Flow warmup (Phase 0) — MLM warmup backbone (mm/warmup.py)

```
(article, stressor processed) → train corpus
  ▼
backbone(input_ids, attention_mask)   [B,S,H]
  ▼  LoRA window: layer ≥ lora_from_layer(=18), trong warmup snE 10 adapters
outputs.logits / losses (LM head trên LỚP CUỐI)   [B,S,V]   (V=vocab)
  ▼
LoRA-MERGE vào backbone (scaling*B@A + wrap)
  ▼
warmup_merged.pt   ← snapshot backbone-only (heads fresh mỗi run)
```
- Span/context pooling **không tham gia** warmup — warmup chỉ đào backbone (MLM).
- Đổi `lora_from_layer` / window → phải **re-run warmup** (ckpt merge phụ thuộc window).
- Đổi `span_layers` / `context_pool` / head → **KHÔNG** cần warmup lại (heads khởi tạo mới; backbone không đổi).

---

## 4. Flow train (Phase 1/2) — co-train heads (mm/trainer.py)

```
batch → _features → (mod_emb, head_emb, features[B,·], logits)
  ▼
GaussHead per-branch → mod_mu,mod_sigma(B,) / head_mu,head_sigma(B,)
  ▼
loss = NLL(logmu_mod, sigma_mod, target_mod) + NLL(head…)   # Gaussian NLL
      + (weighted) LM/MLM loss khi cần
  ▼
LoRA adapters (≥ layer 18) update; backbone <18 đông cứng
```
- Phase 1: warmup backbone (từ `warmup_merged.pt`).
- Phase 2 (train80): lấy snapshot warmup, thêm LoRA, chỉ train 10 adapters + heads.

---

## 5. Flow predict / infer (bounded mu)

```
_mod branch (use_proto? with LM?) → mod_pred = clamp(mod_mu, min, max) (B,)
_head branch → head_pred = clamp(head_mu, min, max) (B,)
sigma → (mu, sigma) hoàn chỉnh N(mu, sigma²)
```

---

## Tóm tắt chiều theo mỗi bước

| bước | input | output |
|---|---|---|
| backbone forward | ids[B,S], mask[B,S] | hidden_states: (N+1)×`[B,S,H]` |
| span mean 5 lớp giữa | 5×`[B,S,H]` | `[B,S,H]` |
| mod/head pool | `[B,S,H]` + span_mask | `[B,H]` |
| context mean+cls | `[B,S,H]` | `[B,2H]` |
| lens | span_masks | mod_len`[B,1]`, head_len`[B,1]` |
| lm stats (tùy chọn) | ids, masks | `[B,4]` |
| features cat | 5 (+1) feats | `[B, 3074]` (default) |
| GaussHead | `[B,3074]` (hoặc +cos `[B,3075]`) | mu`(B,)`, sigma`(B,)` |
| warmup snapshot | backbone | `warmup_merged.pt` (backbone-only) |
| train | `warmup_merged.pt` + LoRA(≥18) | head weights + adapters |
| predict | features | mod_pred, head_pred `(B,)` |
