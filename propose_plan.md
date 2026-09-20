# KIẾN TRÚC TWO-STREAM BI-ENCODER: DYNAMIC PROTOTYPE VS. CONTEXTUAL MEANING CHO BÀI TOÁN COMPOSITIONALITY (MoTune)

> **Tài liệu Thiết kế Kiến trúc & Kế hoạch Đề xuất Chi tiết (`propose_plan.md`)**  
> **Tác giả:** Đội ngũ Nghiên cứu & Phát triển MoTune  
> **Mô hình Backbone:** `jhu-clsp/mmBERT-base` (ModernBERT, 22 layers, 768-d, vocab 256,000)  
> **Mục tiêu:** Hiện thực hóa ý tưởng **Parallel Batching / Two-Stream Disentanglement** để ước lượng độ cấu thành ngữ nghĩa (Compositionality / Idiomaticity) thông qua độ lệch ngữ nghĩa (Semantic Shift) giữa nghĩa nguyên bản (Lexical Prototype) và nghĩa ngữ cảnh (In-Context Meaning).

---

## 1. CƠ SỞ LÝ THUYẾT & ĐỘNG LỰC NGHIÊN CỨU

### 1.1. Bản chất Ngữ nghĩa học của Tính Cấu thành (Semantic Compositionality)
Trong ngữ nghĩa học tính toán (*Schulte im Walde et al.*, *Reddy et al.*, benchmark *CICE/Compartment*), tính cấu thành ngữ nghĩa của một từ tố trong cụm từ (Modifier, Head Noun, hoặc Phrasal Verb) được định nghĩa là **mức độ bảo toàn ngữ nghĩa gốc khi từ tố đó được đặt vào một ngữ cảnh cụ thể**:

$$\text{Compositionality Score} \propto 1 - \text{SemanticShift}(\mathbf{h}_{\text{prototype}}, \mathbf{h}_{\text{context}})$$

* **Trường hợp Nghĩa đen (Literal - Điểm cao, e.g. 4.0 - 5.0):**
  * Ví dụ: *"acid rain"* $\to$ Từ *"acid"* trong câu *"The acid rain damaged the forest"* vẫn mang nghĩa một hợp chất hóa học có tính axit.
  * Vector ngữ cảnh $\mathbf{h}_{\text{context}}$ và vector từ điển $\mathbf{h}_{\text{prototype}}$ gần như trùng khít nhau ($\text{Distance} \approx 0$).
* **Trường hợp Thành ngữ / Ẩn dụ (Idiomatic - Điểm thấp, e.g. 0.0 - 1.5):**
  * Ví dụ: *"red tape"* $\to$ Từ *"red"* trong *"The project was delayed by bureaucratic red tape"* không còn chỉ màu sắc đỏ nữa mà chỉ sự rườm rà về thủ tục.
  * Ngữ cảnh câu đã kéo lệch biểu diễn $\mathbf{h}_{\text{context}}$ đi rất xa khỏi biểu diễn nguyên bản $\mathbf{h}_{\text{prototype}}$ ($\text{Distance} \gg 0$).

---

### 1.2. Phân tích So sánh: 3 Thế hệ Kiến trúc

| Tiêu chí so sánh | Thế hệ 1: External Static (`static_vec.py`) | Thế hệ 2: Target Prefix (`best_plan.md`) | Thế hệ 3: Two-Stream Prototype (Đề xuất này) |
| :--- | :--- | :--- | :--- |
| **Nguồn Prototype** | File từ điển ngoài (FastText / Word2Vec `.vec`) | Prefix chung sequence: `[CLS] <marker> target <marker> Ctx` | **Dynamic Stream:** mmBERT encode riêng `[CLS] target [SEP]` |
| **Không gian biểu diễn (Latent Space)** | **Bị lệch:** 300-d (FastText) $\neq$ 768-d (mmBERT). Phải qua `nn.Linear` để ép chiều. | **Chung:** 768-d (mmBERT). | **Hoàn hảo:** Cùng 1 backbone mmBERT 768-d, chung vocab 256k. |
| **Vấn đề OOV (Từ ngoài từ điển)** | **Nghiêm trọng:** Từ ghép, từ tiếng Đức hiếm bị OOV $\to$ vector 0. | **Không bị:** Dùng BPE/SentencePiece subwords. | **Không bị:** Tokenizer mmBERT xử lý 100% từ vựng. |
| **Tính độc lập (Disentanglement)** | Độc lập tuyệt đối. | **Bị nhiễu:** Attention 22 tầng cho phép target và context "nhìn" nhau từ Layer 0. | **Độc lập tuyệt đối:** Stream 1 chỉ thấy từ đơn, Stream 2 thấy toàn câu. |
| **Phụ thuộc tài nguyên ngoài** | Phải tải file `.vec` 1 - 2 GB lên Kaggle. | Không cần file ngoài. | **Không cần file ngoài:** Tự sinh 100% bằng mmBERT! |
| **Chi phí tính toán** | Thấp. | Chuỗi dài hơn do chèn prefix ($L + 5$). | **Cực thấp:** Stream 1 chỉ có độ dài $L \le 6$ tokens ($O(L^2) \approx 36$). |

---

## 2. KIẾN TRÚC TỔNG THỂ (WORKFLOW MERMAID)

```mermaid
flowchart TD
    subgraph INPUT["DỮ LIỆU ĐẦU VÀO (Per Batch)"]
        R["Mẫu dữ liệu: Target Word ('acid') + Context ('The acid rain fell...')"]
        R --> S1["Stream 1 (Prototype Batch):<br/>[CLS] acid [SEP]<br/>(Chiều dài L1 ≈ 3-6 tokens)"]
        R --> S2["Stream 2 (Context Batch):<br/>[CLS] The acid rain fell on the forest... [SEP]<br/>(Chiều dài L2 ≈ 128-256 tokens)"]
    end

    subgraph BACKBONE["SHARED BACKBONE mmBERT (ModernBERT 22 Layers)"]
        S1 -->|Forward 1 (Siêu nhẹ)| BB["mmBERT-base (Shared Weights + LoRA)"]
        S2 -->|Forward 2 (Standard)| BB
        BB --> H1["Last Hidden State Stream 1 (B, L1, 768)"]
        BB --> H2["Last Hidden State Stream 2 (B, L2, 768)"]
    end

    subgraph POOLING["TRÍCH XUẤT BIỂU DIỄN (Pooling)"]
        H1 --> P1["MeanPool(Target Subwords)<br/>==> h_proto (B, 768)"]
        H2 --> P2["SpanPool(Context Target Span)<br/>==> h_ctx (B, 768)"]
    end

    subgraph INTERACTION["TƯƠNG TÁC NGỮ NGHĨA (Disentangled Semantic Shift)"]
        P1 & P2 --> DIFF["Vector Dịch chuyển (Displacement):<br/>Δh = h_ctx - h_proto"]
        P1 & P2 --> PROD["Tương đồng chiều (Hadamard):<br/>m = h_ctx ⊙ h_proto"]
        P1 & P2 --> COS["Độ tương đồng Cosine:<br/>cos_sim = Cosine(h_ctx, h_proto)"]
        
        P1 & P2 & DIFF & PROD & COS --> FUSE_LAYER["Projection / Fusion Block:<br/>Linear([h_ctx ; h_proto ; Δh ; m ; cos]) -> (B, 768)"]
    end

    subgraph HEAD["DỰ ĐOÁN & HUẤN LUYỆN (Rating & Losses)"]
        FUSE_LAYER --> GAUSS["GaussHead (Shared 768 -> 128 -> 2)"]
        GAUSS --> OUT["Dự đoán Phân phối: μ (Rating [0, 5]), σ² (Độ bất định)"]
        
        OUT --> L_NLL["Loss Chính: Gaussian NLL Loss"]
        COS --> L_CONTRAST["Loss Bổ trợ: Margin Ranking / Contrastive Loss<br/>(Ép cos_sim tương quan thuận với Rating)"]
        L_NLL & L_CONTRAST --> TOTAL_LOSS["Total Loss = L_NLL + λ * L_Contrast"]
    end
```

---

## 3. THIẾT KẾ TOÁN HỌC & CÁC KHỐI CHỨC NĂNG

### 3.1. Trích xuất Vector Prototype ($\mathbf{h}_{\text{proto}}$) và Context ($\mathbf{h}_{\text{ctx}}$)
Với mỗi mẫu dữ liệu trong batch:
- **Stream 1:** Chuỗi input $\mathbf{x}_{\text{proto}} = [\text{[CLS]}, w_1, \dots, w_K, \text{[SEP]}]$ trong đó $\{w_1, \dots, w_K\}$ là $K$ subwords của từ mục tiêu (ví dụ: *"Handschuh"* $\to$ 3 subwords).
  $$\mathbf{h}_{\text{proto}} = \frac{1}{K} \sum_{k=1}^K \mathbf{H}^{(1)}_{k}$$
  $\mathbf{h}_{\text{proto}}$ phản ánh nghĩa ngữ niệm thuần khiết (Isolated Semantic Prototype).

- **Stream 2:** Chuỗi context gốc $\mathbf{x}_{\text{ctx}} = [\text{[CLS]}, c_1, \dots, c_M, \text{[SEP]}]$. Giả sử từ mục tiêu nằm ở span từ token $s_{\text{start}}$ đến $s_{\text{end}}$:
  $$\mathbf{h}_{\text{ctx}} = \frac{1}{s_{\text{end}} - s_{\text{start}}} \sum_{j=s_{\text{start}}}^{s_{\text{end}}} \mathbf{H}^{(2)}_{j}$$
  $\mathbf{h}_{\text{ctx}}$ phản ánh nghĩa ngữ cảnh thực tế đã tương tác với toàn bộ câu văn.

---

### 3.2. Không gian Tương tác Ngữ nghĩa (Semantic Shift Feature Interaction)
Thay vì chỉ tính một đại lượng vô hướng (scalar) là Cosine, ta trang bị cho mô hình toàn bộ tensor đặc trưng sai khác để lớp Head tự học quy luật:

1. **Vector Dịch chuyển (Semantic Displacement):**
   $$\mathbf{d} = \mathbf{h}_{\text{ctx}} - \mathbf{h}_{\text{proto}} \in \mathbb{R}^{768}$$
   *Ý nghĩa:* Cho biết nghĩa của từ đã bị "kéo" theo hướng nào trong không gian tiềm ẩn 768 chiều.
2. **Tương đồng Chiều (Hadamard Alignment):**
   $$\mathbf{m} = \mathbf{h}_{\text{ctx}} \odot \mathbf{h}_{\text{proto}} \in \mathbb{R}^{768}$$
   *Ý nghĩa:* Nhấn mạnh các chiều đặc trưng vẫn giữ nguyên dấu và độ lớn.
3. **Độ tương đồng Cosine chuẩn hóa:**
   $$s = \frac{\mathbf{h}_{\text{ctx}} \cdot \mathbf{h}_{\text{proto}}}{\|\mathbf{h}_{\text{ctx}}\|_2 \|\mathbf{h}_{\text{proto}}\|_2} \in [-1, 1]$$
4. **Vector Tổng hợp (Fused Representation):**
   $$\mathbf{z} = \left[ \mathbf{h}_{\text{ctx}} \,;\,\, \mathbf{h}_{\text{proto}} \,;\,\, \mathbf{d} \,;\,\, \mathbf{m} \,;\,\, s \right] \in \mathbb{R}^{4H + 1} = \mathbb{R}^{3073}$$
   Vector này được chuẩn hóa qua LayerNorm và nén về chiều $H = 768$:
   $$\mathbf{e}_{\text{final}} = \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_{\text{shift}} \mathbf{z} + \mathbf{b}_{\text{shift}}\right)\right) \in \mathbb{R}^{768}$$
   *(Hoặc lựa chọn nâng cao: Dùng transformer block `StaticFusion` sẵn có để cross-attend giữa $\mathbf{h}_{\text{ctx}}$ và $\mathbf{h}_{\text{proto}}$).*

---

### 3.3. Hàm Mất Mát Đa Mục Tiêu (Multi-Task Objective)

1. **Mất mát Hồi quy Gaussian (Supervised NLL Loss):**
   $$\mathcal{L}_{\text{NLL}} = \frac{1}{2} \log \sigma^2 + \frac{(y - \mu)^2}{2\sigma^2}$$
2. **Mất mát Căn chỉnh Tương phản / Thứ hạng (Auxiliary Contrastive / Ranking Loss):**
   Ta mong muốn độ tương đồng cosine $s_i$ giữa prototype và context của mẫu $i$ phải tỷ lệ thuận với điểm rating $y_i$:
   * Với cặp mẫu $(i, j)$ trong cùng 1 batch có cùng vai trò (cùng là `mod` hoặc `head`):
     Nếu $y_i - y_j > \delta$ (mẫu $i$ có nghĩa cấu thành rõ hơn hẳn mẫu $j$), thì ta phạt nếu $s_i$ không lớn hơn $s_j$:
     $$\mathcal{L}_{\text{rank}} = \max\left(0, \, (s_j - s_i) + \gamma \cdot (y_i - y_j)\right)$$
3. **Tổng hàm mất mát:**
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}} + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}}$$
   *(Thiết lập $\lambda_{\text{rank}} = 0.1$ giúp định hình không gian biểu diễn mà không làm lệch dự đoán $\mu$).*

---

## 4. TỐI ƯU HÓA TÍNH TOÁN & BỘ NHỚ TRÊN KAGGLE T4 (14 GB)

Một trong những ưu điểm lớn nhất của kiến trúc đề xuất là **cực kỳ tiết kiệm bộ nhớ**:

### 4.1. Toán học về Memory & Chi phí Attention
- **Stream 2 (Context):** $B = 32$, $L_2 = 256$ $\implies$ Số phép tính Attention $\propto 32 \times 256^2 \approx 2{,}097{,}152$.
- **Stream 1 (Prototype):** $B = 32$, $L_1 \le 6$ $\implies$ Số phép tính Attention $\propto 32 \times 6^2 \approx 1{,}152$.
- **Tỷ lệ chi phí:** Stream 1 chỉ tiêu tốn:
  $$\frac{1{,}152}{2{,}097{,}152} \approx \mathbf{0.055\%}$$
  nghĩa là **chưa tới 1/1000 lượng tính toán attention của Stream 2**!

### 4.2. Cơ chế Thực thi Không Tăng VRAM (Sequential Stream Forward)
Thay vì nối gộp 2 batch làm sequence dài hoặc padding Stream 1 lên 256 (gây lãng phí VRAM), ta thực hiện lần lượt trong cùng step:
```python
# 1. Forward Stream 1 (siêu ngắn, chỉ 6 tokens)
with torch.amp.autocast(device_type):
    out_proto = self.lm(input_ids=batch['proto_ids'], attention_mask=batch['proto_mask'])
    h_proto = pool_prototype(out_proto.last_hidden_state, batch['proto_mask']) # (B, 768)

# 2. Forward Stream 2 (context bình thường)
with torch.amp.autocast(device_type):
    out_ctx = self.lm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    h_ctx = pool_active(out_ctx.last_hidden_state, batch, targets) # (B, 768)

# 3. Tương tác và GaussHead
fused = self.shift_fusion(h_ctx, h_proto) # (B, 768)
mu, sigma = self.gauss(fused)
```
* Bộ nhớ kích hoạt (Activation Memory) của Stream 1 được giải phóng ngay lập tức, chỉ giữ lại tensor vector `(B, 768)` chiếm vỏn vẹn **98 KB VRAM**!
* **Hoàn toàn không có nguy cơ OOM trên GPU T4.**

---

## 5. THIẾT KẾ MÃ NGUỒN CHI TIẾT (MAPPING VÀO THƯ MỤC `src/`)

Để tuân thủ nghiêm ngặt nguyên tắc: **`src/model.py` giữ nguyên 100% không chỉnh sửa**, toàn bộ logic sẽ được đặt gọn gàng trong các module chuyên trách:

### 5.1. `src/data.py` (Mở rộng mã hóa dữ liệu)
- Trong `CompDataset._encode(r)`:
  - Khi bật cấu hình `cfg.proto_stream = True`:
  - Trích xuất từ mục tiêu của row (`mod`, `head`, hoặc `compound`).
  - Tokenize độc lập: `enc_proto = tokenizer(target_word, max_length=16, truncation=True, return_tensors='pt')`.
  - Bổ sung vào item:
    ```python
    item['proto_ids'] = enc_proto['input_ids'].squeeze(0)
    item['proto_mask'] = enc_proto['attention_mask'].squeeze(0)
    ```
- Trong `collate_comp`:
  - Tự động pad `proto_ids` và `proto_mask` theo chiều dài tối đa của từ mục tiêu trong batch (thường chỉ $L \le 6$).

### 5.2. `src/model_combined.py` (Tích hợp Module `ShiftFusion`)
- Tạo một khối lớp nhỏ gọn `SemanticShiftFusion(nn.Module)`:
  ```python
  class SemanticShiftFusion(nn.Module):
      def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
          super().__init__()
          # Input: [h_ctx, h_proto, h_ctx - h_proto, h_ctx * h_proto, cos]
          in_dim = hidden_size * 4 + 1
          self.proj = nn.Sequential(
              nn.Linear(in_dim, hidden_size),
              nn.LayerNorm(hidden_size),
              nn.GELU(),
              nn.Dropout(dropout),
          )
      def forward(self, h_ctx: torch.Tensor, h_proto: torch.Tensor) -> torch.Tensor:
          diff = h_ctx - h_proto
          prod = h_ctx * h_proto
          cos = F.cosine_similarity(h_ctx, h_proto, dim=-1, eps=1e-8).unsqueeze(-1)
          feat = torch.cat([h_ctx, h_proto, diff, prod, cos], dim=-1)
          return self.proj(feat)
  ```
- Trong `CombinedBackboneModel`:
  - Khởi tạo `self.shift_fuse = SemanticShiftFusion(hidden_size)` nếu `proto_stream` bật.
  - Trong `_pred_heads(self)`: Module `self.shift_fuse` tự động được nạp vào nhóm tham số huấn luyện ở `head_lr`.

### 5.3. `src/config.py` (Bổ sung cấu hình & Ràng buộc)
- Thêm các trường cấu hình:
  - `proto_stream: bool = False` (Bật/tắt chế độ Two-Stream Prototype).
  - `proto_rank_loss: float = 0.0` (Trọng số cho Margin Ranking Loss, mặc định 0.0 là thuần Gauss NLL).
- Validation:
  - `proto_stream` tương thích hoàn toàn với `model_backend = "combined"`.
  - Tự động cảnh báo nếu bật đồng thời cả `static_ext` và `proto_stream` (vì `proto_stream` sinh vector tốt hơn hẳn static external).

---

## 6. MA TRẬN SO SÁNH HIỆU QUẢ DỰ KIẾN (BENCHMARK EXPECTATION)

| Phương pháp | Spearman Correlation ($\rho$) mục tiêu | Ưu điểm cốt lõi | Rủi ro kỹ thuật |
| :--- | :---: | :--- | :--- |
| **Baseline (Context Span Only)** | $0.56 - 0.58$ | Đơn giản, đã chạy ổn định. | Không có điểm tựa đối chiếu nghĩa nguyên bản. |
| **Static External (FastText .vec)** | $0.57 - 0.59$ | Có nghĩa nguyên bản. | Lệch không gian vector (300d $\to$ 768d), OOV từ ghép. |
| **Stage 1 Adapted Prefix (`best_plan.md`)** | $0.58 - 0.62$ | Attention tương tác sâu 22 tầng, Context Sink. | Stage 1 pre-train cần tinh chỉnh lr/epochs. |
| **Two-Stream Dynamic Prototype (Đề xuất này)** | **$0.59 - 0.63$** | **Disentanglement sạch sẽ, cùng không gian 768d, không OOV, bổ sung vector độ lệch $\Delta \mathbf{h}$, cực nhẹ.** | **Cần căn chỉnh LayerNorm để tránh hiện tượng Anisotropy.** |

---

## 7. KẾ HOẠCH TRIỂN KHAI TỪNG BƯỚC (ROADMAP)

### Giai đoạn 1: Chuẩn bị & Smoke Test CPU cục bộ
1. Viết Unit Test / Smoke Check cho `proto_ids` trong `tests/smoke_src.py`:
   - Kiểm tra Tokenization của từ mục tiêu đơn lập.
   - Kiểm tra Forward pass 2 luồng giả lập trên CPU.
   - Đảm bảo gradient truyền trơn tru vào `self.shift_fuse` và `self.gauss`.
   - Đảm bảo `src/model.py` **không có bất kỳ thay đổi nào (`git diff` rỗng)**.

### Giai đoạn 2: Cập nhật Config & Pipeline
1. Cập nhật `src/config.py` thêm `proto_stream`, `proto_rank_loss`.
2. Tạo file cấu hình thực nghiệm `config/proto_stream_gauss.json`.
3. Bổ sung trích xuất `proto_ids` trong `src/data.py` và module `SemanticShiftFusion` trong `src/model_combined.py`.

### Giai đoạn 3: Thực nghiệm trên Kaggle T4
1. Thử nghiệm Chế độ 1 (End-to-End Single Stage):
   - Chạy trực tiếp `proto_stream_gauss.json` trên backbone gốc `jhu-clsp/mmBERT-base`.
   - So sánh trực tiếp với kết quả baseline $\rho \approx 0.564$.
2. Thử nghiệm Chế độ 2 (Kết hợp Siêu cấp - Hybrid):
   - Dùng backbone đã được thích ứng từ Stage 1 Prefix MLM (`checkpoints/prefix_mlm_adapted`), kết hợp với cơ chế Two-Stream Prototype.

---

## 8. KẾT LUẬN & ĐỀ XUẤT HÀNH ĐỘNG

Ý tưởng **Two-Stream Dynamic Prototype / Parallel Batching** của bạn là một bước tiến học thuật rất sáng tạo và thực tế:
1. Nó **giải phóng dự án** khỏi sự phụ thuộc vào các file nhị phân tĩnh cồng kềnh (FastText `.vec`).
2. Nó **mô phỏng trung thực 100% cách con người đánh giá tính thành ngữ**: so sánh nghĩa gốc của từ với nghĩa của nó trong câu văn.
3. Về mặt kỹ thuật, nó **cực kỳ an toàn cho VRAM**, không gây OOM và tương thích tuyệt đối với cấu trúc Single-Head Gauss hiện tại.

Kế hoạch trên đã sẵn sàng để chuyển sang giai đoạn thực thi khi bạn phê duyệt!
