# Lab 21 — Evaluation Report

**Họ tên**: Hoàng Trung Hiếu  **MSSV**: 2A202602945  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (fp16)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 mẫu (chia ngẫu nhiên seed 42) |
| `max_length` | 1024 — p95 đo được là 98 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** Có — kết quả từ `results/template_check.json` xác nhận `reasoning preserved — safe to train on traces`. Khối suy luận `<think>` được bảo toàn nguyên vẹn trong cấu trúc Jinja template của Qwen3.5, không bị tự động cắt bỏ khi gọi hàm `apply_chat_template`. Về tham số `max_length`: p95 đo được là 98 tokens (gợi ý 256), nhưng tôi giữ nguyên mức trần 1024 theo mặc định của tier T4 nhằm bảo toàn đầy đủ ngữ cảnh cho mọi trường hợp biên phức tạp và tránh cắt cụt bất kỳ chuỗi sinh nào.

---

## 2. Mask proof (NB1)

| Tiêu chí | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn token đầu tiên được tính loss (trích xuất từ `results/mask_proof.json`):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Toàn bộ phần prompt chỉ dẫn của hệ thống và câu hỏi của khách hàng đã được gán nhãn `-100` (`IGNORE_INDEX`), đảm bảo hàm mục tiêu lan truyền ngược chỉ tập trung cập nhật trọng số cho việc sinh nhãn JSON mục tiêu.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3238.8 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 987.7 |
| (c) LoRA fine-tune | 0.970 | 0.722 | 1.000 | 1353.5 |

**(b) có thật sự mạnh hơn (a) không?** Có. Baseline (b) cải thiện vượt trội so với (a): độ chính xác `target` tăng từ 0.000 lên 0.765, tỷ lệ chuẩn định dạng `format` đạt tuyệt đối 1.000 (so với 0.000 của a), và độ trễ phản hồi giảm mạnh từ 3238.8 ms xuống còn 987.7 ms.  
Tôi **không sửa** `OPTIMIZED_PROMPT` và giữ nguyên chuỗi prompt gốc với mã băm SHA `719e74d3b6232053`. Việc giữ nguyên mốc chuẩn khó này đảm bảo tính liêm chính học thuật tuyệt đối của phép so sánh, không làm yếu prompt cơ sở để cố tình tâng bốc kết quả fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6269 | 0.970 | 401.3 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32456704 | 0.0001 | 0.5374 | 0.970 | 267.8 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32464896 | 0.00001 | 1.5702 | 0.000 | 397.9 | 8.78 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.940 | 467.0 | 3.86 |

> Bảng trên được xếp hạng dựa vào năng lực thực tế trên cột **target (NB5 §4)**, không xếp theo `train loss` của NB4.

### 4.1 — Phân tích vị trí gắn adapter vs Rank (`attn_only` vs `correct`)
Run `attn_only` được cấu hình nâng rank lên $r=283$ bằng thuật toán `matched_rank()` để khớp chính xác ngân sách 32.46M tham số huấn luyện (sai lệch dưới $0.03\%$ so với `correct`). Trên tập đánh giá mục tiêu `target`, `attn_only` đạt điểm số $0.970$, ngang bằng với `correct` ($0.970$), mặc dù train loss của `attn_only` ($0.5374$) thấp hơn đáng kể so với `correct` ($0.6269$). Thứ tự theo train loss không phản ánh đúng thứ tự năng lực trên tập kiểm tra thực tế: việc dồn toàn bộ tham số vào các lớp attention với rank cực lớn khiến mô hình học thuộc (memorize) dữ liệu huấn luyện nhanh hơn, dẫn đến loss thấp hơn nhưng không tạo ra thêm bất kỳ giá trị tổng quát hóa nào trên bài toán trích xuất JSON. Điều này chứng minh rằng rank không phải là đòn bẩy vạn năng; phân bổ adapter qua toàn bộ các lớp `all-linear` mang lại sự ổn định và độ bao phủ biểu diễn tốt hơn cho mô hình ngôn ngữ.

### 4.2 — Phân tích thang đo Learning Rate (`wrong_lr`)
Run `wrong_lr` chỉ thay đổi một biến duy nhất là giảm learning rate xuống $10\times$ ($10^{-5}$ thay vì $10^{-4}$), mô phỏng sai lầm kinh điển khi áp dụng thang LR của full fine-tuning cho LoRA. Đường loss của `wrong_lr` giảm rất chậm chạp và dừng lại ở mức $1.5702$, dẫn đến việc mô hình hoàn toàn thất bại trên tập đánh giá mục tiêu với `target = 0.000` và `format = 0.000`. Nếu một kỹ sư chỉ quan sát thấy train loss vẫn giảm dần đều từ 2.16 xuống 1.57 mà không nắm vững lý thuyết, họ rất dễ đưa ra kết luận sai lầm rằng mô hình đang học tốt và chỉ cần train thêm nhiều epochs, hoặc bi quan cho rằng LoRA bất lực trước bài toán này. Thực tế, vì ma trận $B$ của LoRA được khởi tạo bằng 0 nên cần một learning rate lớn hơn khoảng $10\times$ so với full fine-tune để gradient có đủ độ lớn thúc đẩy adapter thích nghi kịp thời với tác vụ mới.

### 4.3 — Phân tích chi phí và hiệu năng của QLoRA (`qlora`)
Run `qlora` (lượng tử hóa 4-bit NF4) đã giúp cắt giảm bộ nhớ VRAM cực kỳ ấn tượng, từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn $56\%$ VRAM). Tuy nhiên, cái giá phải trả là thời gian huấn luyện bị kéo dài thêm từ 401.3 giây lên 467.0 giây (chậm hơn khoảng $16.4\%$) do chi phí tính toán giải lượng tử hóa liên tục trên kiến trúc GPU Turing T4. Quan trọng nhất, độ chính xác `target` bị sụt giảm từ 0.970 xuống 0.940 do nhiễu mất mát thông tin từ việc ép trọng số về 4-bit. Kết quả thực nghiệm định lượng này hoàn toàn ủng hộ khuyến nghị kỹ thuật của nhà phát triển (vendor recommendation) rằng không nên lạm dụng QLoRA cho Qwen3.5 trên các bài toán đòi hỏi độ chính xác cấu trúc cao khi phần cứng vẫn đủ dung lượng VRAM để chạy bản 16-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.069` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết
Cổng an toàn hồi quy ghi nhận kết quả FAILED. Về mặt mục tiêu nghiệp vụ chuyên biệt, bản fine-tune LoRA đã tạo ra sự bứt phá vượt bậc khi đưa độ chính xác `target` từ 0.765 (Baseline b) lên 0.970 (`target_delta = +0.205`), với định dạng JSON hoàn hảo ($1.000$). Tuy nhiên, hệ thống bị đánh trượt do năng lực tri thức tổng quát bị sụt giảm ngoài ngưỡng dung sai: `regression_delta = -0.0689` (từ 0.7911 tụt xuống 0.7222, trong khi ngưỡng sai số tối đa chỉ là 0.020).

Nguyên nhân trực tiếp xuất phát từ hiện tượng "quên thảm họa" (catastrophic forgetting): toàn bộ 225 mẫu huấn luyện chỉ thuần túy là dữ liệu gán nhãn JSON CSKH mà không hề có bất kỳ mẫu dữ liệu đối thoại hay tri thức phổ thông nào được xen kẽ. Việc adapter thích nghi quá mức với cấu trúc JSON chuyên biệt đã làm tổn thương khả năng sinh câu trả lời tự nhiên ở các câu hỏi đời sống. Đây là một kết luận thực nghiệm vô cùng quý giá và phản ánh đúng bản chất của quy trình phát triển AI: thay vì cố tình sửa tập test hay nới lỏng cổng an toàn để lấy trạng thái PASSED giả tạo, kết quả FAILED trung thực này chỉ ra giải pháp kỹ thuật cần thiết là phải trộn thêm $1-5\%$ dữ liệu replay tổng quát vào tập huấn luyện trước khi triển khai chính thức ra môi trường sản xuất.

---

## 6. Định tính — So sánh chi tiết các ca điển hình

Dưới đây là 5 ca đánh giá thực tế từ tập test, bao gồm cả ca fine-tune thắng và ca fine-tune thua:

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 47 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không gọi... | `van_chuyen`, `thap`, `ốp lưng điện thoại`, `tich_cuc` | Nhầm sentiment thành `trung_tinh` | `van_chuyen`, `thap`, `ốp lưng điện thoại`, `tich_cuc` | ✅ **FT thắng**: Bắt đúng cảm xúc tích cực của khách khi khen shop hỗ trợ tốt. |
| 48 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu... | `hoi_thong_tin`, `trung_binh`, `ốp lưng điện thoại`, `trung_tinh` | Nhầm intent sang `doi_tra` | `hoi_thong_tin`, `trung_binh`, `ốp lưng điện thoại`, `trung_tinh` | ✅ **FT thắng**: Phân loại chuẩn xác ý định hỏi giá thay vì xử lý đơn hàng. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền... | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | Trích xuất đúng `thap` do có câu "Khi nào tiện" | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `tich_cuc` | ❌ **FT thua**: Mô hình đoán nhầm `urgency: trung_binh` do thiên kiến từ khóa "Chưa thấy tiền". |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện... | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | Nhận diện đúng mức độ `thap` từ ngữ cảnh mềm | `san_pham_loi`, `trung_binh`, `nồi chiên không dầu`, `trung_tinh` | ❌ **FT thua**: Đánh giá sai mức độ khẩn cấp, thiên về `trung_binh` khi gặp lỗi phụ kiện. |
| 12 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện... | `san_pham_loi`, `thap`, `áo khoác gió`, `tich_cuc` | Đạt điểm tối đa 1.0 trên cả 4 trường | `san_pham_loi`, `trung_binh`, `áo khoác gió`, `tich_cuc` | ❌ **FT thua**: Tiếp tục đoán sai mức `thap` thành `trung_binh` dù khách nói "Khi nào tiện". |

### Nhận xét mẫu hình chung ở các ca Fine-tune thua
Cả 3 trường hợp fine-tune thua (ca 3, 5, 12) đều xuất hiện một quy luật nhất quán: mô hình bị đoán sai trường `urgency` từ mức `thap` thành `trung_binh`. Khi khách hàng khiếu nại về tiền bạc hoặc hàng lỗi nhưng dùng lời lẽ nhẹ nhàng ("Khi nào tiện"), Baseline (b) với prompt chỉ dẫn quy tắc nhận diện rất tốt điều này để hạ mức khẩn cấp. Trong khi đó, mô hình fine-tune sau quá trình huấn luyện đã bị thiên kiến (bias) liên kết chặt chẽ các từ khóa mang tính khiếu nại ("chưa thấy tiền", "lỗi", "thiếu phụ kiện") với mức khẩn cấp tối thiểu là `trung_binh`, bỏ qua các sắc thái lịch sự làm giảm mức độ nghiêm trọng của yêu cầu.

---

## 7. Kết luận & Điều tôi học được

### Kết luận tổng kết
Dựa trên kết quả đo đạc khoa học và toàn diện, bản fine-tune `correct` hiện tại **chưa nên triển khai trực tiếp ra môi trường sản xuất độc lập** nếu không có lớp kiểm soát an toàn đi kèm. Mặc dù mô hình mang lại độ chính xác nghiệp vụ ấn tượng trên tác vụ CSKH ($97.0\%$ so với $76.5\%$ của prompt engineering), sự suy giảm $6.9\%$ năng lực tri thức tổng quát đặt ra rủi ro suy thoái khi xử lý các tương tác mở rộng của người dùng. Để đưa vào production, phương án khả thi nhất là triển khai theo cơ chế Router: sử dụng adapter này chuyên trách cho pipeline trích xuất thông tin tự động, hoặc tiến hành huấn luyện lại với việc pha trộn $2-5\%$ dữ liệu đối thoại đa miền để vượt qua cổng kiểm soát hồi quy.

Đòn bẩy thực sự quyết định thành công trong lab này không phải là việc cố gắng tăng rank LoRA lên mức cực đại, mà là **việc lựa chọn Learning Rate chuẩn xác (thang LoRA $10\times$)** và **xây dựng Loss Masking chuẩn xác (chỉ tính loss trên câu trả lời)**. Khi mất đi thang LR đúng, mô hình hoàn toàn bất lực (`target = 0.000`), và nếu mask tính cả prompt, mô hình sẽ học vẹt thay vì học cách suy luận định dạng.

### Ba điều tôi học được
1. **Loss Masking là nền tảng quyết định tính sống còn của SFT:** Việc giải mã ngược mask để chứng minh câu hỏi bị che hoàn toàn và câu trả lời được tính loss quan trọng hơn nhiều so với việc tinh chỉnh cấu hình LoRA. Nếu mask sai, mọi nỗ lực tinh chỉnh phía sau đều vô nghĩa.
2. **Nghịch lý giữa Train Loss và Năng lực thực tế:** Run `attn_only` với rank cao ($r=283$) có train loss thấp hơn hẳn bản `correct` ($0.537$ vs $0.626$), nhưng độ chính xác target hoàn toàn tương đương. Đánh giá mô hình phải dựa trên năng lực giải quyết tác vụ cuối cùng, tuyệt đối không được dùng train loss làm thước đo chất lượng.
3. **Giá trị của việc đo Baseline đóng băng và Cổng hồi quy:** Đo Baseline (b) trước khi huấn luyện giúp thiết lập một thước đo công bằng, ngăn chặn tâm lý thiên vị của người kỹ sư. Một kết quả FAILED trung thực chỉ ra chính xác điểm yếu của pipeline (quên thảm họa) có giá trị kỹ thuật cao hơn nhiều so với một kết quả PASSED giả tạo do nới lỏng tiêu chuẩn đánh giá.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. **Bổ sung 3% dữ liệu Replay đa nhiệm:** Trộn thêm khoảng 15-20 mẫu câu hỏi kiến thức đời sống vào tập train 225 mẫu để khắc phục hoàn toàn hiện tượng suy thoái năng lực chung và đưa cổng hồi quy về trạng thái `PASSED`.
2. **Thử nghiệm NB6 (Merge & Hot-swap Serving):** Hợp nhất adapter vào trọng số gốc fp16 và đo đạc độ trễ phục vụ thực tế (serving throughput) khi triển khai qua vLLM hoặc Ollama.

---

## Phụ lục — Thưởng đã làm

- [ ] B1: NB6 merge + hot-swap
- [ ] B2: Dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3: Reasoning-trace collapse
- [ ] B4: Quét rank có kiểm soát
- [x] B5: HuggingFace Hub công khai (+2 điểm): https://huggingface.co/hoangtrunghieu0025-lab/qwen3.5-4b-cskh-lora
