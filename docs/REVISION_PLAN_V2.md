# REVISION_PLAN_V2 (rev.3) — Sửa nền tảng trước khi lên `base`

> Nguồn: phản hồi chuyên gia (3 vòng) + **xác minh trực tiếp trong code**. Cập nhật: 12/10/2026.
> **rev.3** chốt: (1) **quy tắc vị trí SIL cố định** (tránh lệch train/inference);
> (2) A2.4 truyền thêm `vuv_target`; (3) DCT dựng lại contour bằng **cos basis trên relative
> position** (ONNX-safe, âm tiết ngắn dùng 2–3 hệ số); (4) bucket pad **cả chiều âm tiết (×16)**;
> (5) trim **chừa 100 ms** biên; (6) MFA **bỏ hẳn clip score thấp** (không fallback heuristic);
> (7) **gộp nâng cấp vocoder vào bước 2** (bước 7 chỉ còn GTA fine-tune); (8) ngưỡng smoke
> so ở **mức audio** (F0 RMSE + CER), không so val loss cũ.

## 0. Run cũ đã dừng
- Acoustic dừng tại **20.850 bước**: val tăng ngược từ 8k (274,0) → 305,5 (20k) trong khi train giảm
  ⇒ **overfit vào nhãn duration chia đều** (không đơn thuần thiếu dung lượng).
- Giữ `runs/small_codelung/best.pt` (274,0 @8k) + `last.pt` **chỉ để test pipeline** (repr. mel cũ).
- Vocoder cũ chạy nốt cho demo cũ; **sẽ bị thay** bởi vocoder nâng cấp huấn luyện lại trên mel mới.

## 1. A1 — mel loss lấn át — ✅ ĐÃ SỬA
- `masked_l1(..., mean_over_bins=True)` cho mel (chia thêm `size(-1)`); có test đơn vị.
- Sau fix cần cân lại trọng số theo độ lớn thật; f0 chuyển log-F0 z-score (mục A2) để không trội.

## 2. A2 — variance adaptor: có sẵn, nhưng **conditioning đang dùng giá trị DỰ ĐOÁN** (lỗi thứ hai)
Bằng chứng code (`models/vifasttts.py`):
```python
syllable_f0 = self.f0(x)          # DỰ ĐOÁN từ encoder
exp_f0, _, _ = self.length_regulator(syllable_f0.unsqueeze(-1), durations)
variance = torch.cat([exp_f0, exp_vuv, exp_energy], dim=-1)
decoded = self.decoder(expanded + self.variance_proj(variance) + position_proj(...))
```
⇒ Trong train, decoder **nhận f0/vuv/energy dự đoán**; gradient mel chảy ngược vào predictor
(predictor bị kéo học "cái gì tốt cho mel" thay vì F0 thật) và decoder học trên đầu vào chưa chuẩn
⇒ **nguyên nhân thứ hai của mel mờ**, bên cạnh duration.

**A2.4 — BẮT BUỘC:**
- Thêm `model.variance_teacher_forcing: true`: forward nhận thêm **`f0_target`, `vuv_target`,
  `energy_target`** (cấp âm tiết, cùng z-score với lúc suy luận) và **dùng GT để conditioning khi train**;
  suy luận/ONNX dùng giá trị dự đoán.
- Có test (train dùng GT, infer dùng prediction; ONNX export không đổi input).
- Phần còn lại của A2 (làm ở bước 4, sau MFA): **log-F0 z-score toàn cục** + **DCT contour**:
  - Dựng lại contour cấp frame **thuần op cơ bản, ONNX-safe**:
    $f_0(t) = \sum_k c_k \cos(\pi k\, p_t)$ với $p_t$ = **relative position đã có** trong âm tiết.
  - **Âm tiết ngắn (< ~5 frame) chỉ dùng 2–3 hệ số** (hệ số bậc cao vô nghĩa với ít mẫu).
  - Kèm **pyworld (harvest + stonemask)** + nội suy qua vô thanh — riêng pyworld làm sớm ở bước 2.

## 3. Thứ tự thực hiện (rev.3)
| Bước | Việc | Phụ thuộc |
| --- | --- | --- |
| 1 | ✅ A1 mel loss | — |
| 2 | **Mel mới** (log-magnitude clamp 1e-5 + chuẩn hoá toàn cục per-bin) + **pyworld** + **regen cache** → **train vocoder bản nâng cấp luôn** (MPD đầy đủ + MRD + MR-STFT, 100k+, đo lại MPS/CPU) | — (không cần MFA) |
| 3 | **MFA**: lexicon từ parser (prefix vị trí) → `mfa train` → align → **bỏ hẳn clip score thấp** → QC | chạy **riêng**, không song song training |
| 4 | **SIL + log-F0 DCT + conditioning GT** (f0/vuv/energy) cùng một đợt — đều lấy ranh giới từ MFA | cần bước 3 |
| 5 | **Smoke small 15k** với alignment thật; so **ở mức audio** với best.pt 8k | cần bước 4 |
| 6 | **Base enc4/dec6** + LR cosine từ đầu | đạt ngưỡng bước 5 |
| 7 | **GTA fine-tune vocoder** trên mel do acoustic teacher-forcing sinh | cần bước 6 |

Lý do đảo (bước 2↔3): (a) DCT contour tính trên ranh giới âm tiết thật — chia đều thì hệ số DCT cũng
lệch; (b) SIL từ VAD rồi thay bằng sp/sil của MFA là làm hai lần.

## 4. Token SIL — đặc tả đầy đủ (chốt)
### 4.1. Vị trí SIL cố định (khớp train ↔ inference)
- **Luôn chèn SIL ở đầu câu, cuối câu và sau mỗi dấu câu** — **cả khi train lẫn khi suy luận**.
- Nếu MFA **không** thấy khoảng lặng ở vị trí đó (đọc lướt qua dấu phẩy…) → **duration = 0**.
  LengthRegulator kiểu cumsum+gather chạy đúng với 0 frame; `log1p(0) = 0`.
- Các `sp/sil` **không nằm sau dấu câu**: **gộp vào âm tiết đứng trước**; nếu dài bất thường
  (**> 300 ms**) → **đánh dấu clip để QC** (xem `qc_flags` trong báo cáo chuyển TextGrid).
- Lý do: lúc suy luận chỉ có văn bản, không có MFA — SIL ở vị trí tuỳ ý sẽ làm **đầu vào train và
  inference lệch nhau**.

### 4.2. 7 slot cho SIL
- `onset=NONE`, `medial=NONE`, `nucleus=SIL`, `coda=NONE`, **`tone=NONE`** (thêm mới),
  `boundary`/`punctuation` = **giữ nguyên giá trị của vị trí đứng trước** (model biết ngừng sau phẩy
  hay sau chấm).
- **Vocab: 24/2/16/11/7/4/6 = 70**. Không gán NGANG cho SIL (tránh "thanh ngang = im lặng").

### 4.3. Mask loss
- Thêm `speech_mask` cấp âm tiết vào batch → **tone CE + f0/vuv/energy loss mask bỏ SIL**.
- **duration loss KHÔNG mask** SIL (model phải học độ dài im lặng).

### 4.4. Ripple phải làm cùng batch
- Vocab size, param counts, export metadata, test đếm tham số, docs; frontend thêm bước sinh SIL
  theo dấu câu (đầu/cuối câu + sau dấu) khi suy luận.

## 5. MFA — phone set prefix theo vị trí
- `o_ng` (onset) vs `c_ng` (coda); `m_w` (medial) vs `c_w` (coda); `c_c`, `c_ch`, `c_p`, `c_t`...
  (~90–100 nhãn).
- Lý do: cùng ký hiệu nhưng **âm học khác** (coda tắc không bật hơi) → tách mô hình âm học;
  map TextGrid → 5 thành phần **không mơ hồ**.
- `nucleus+tone` gắn liền 1 nhãn (thanh không align thành đoạn riêng).
- **Bỏ hẳn clip score thấp khi train acoustic** — không trộn nhãn heuristic (90% của 51h là thừa đủ).
  Clip bị loại ghi vào danh sách để QC, không đưa vào train/val.

## 6. Chi tiết dễ sai — chốt cách làm
- **Làm tròn duration**: `boundary_i = round(t_i · sr / hop)` cho từng ranh giới;
  `duration_i = boundary_{i+1} − boundary_i` (hiệu các **mốc tích luỹ đã làm tròn**) — tổng khớp
  tuyệt đối số frame của clip.
- **Quy tắc frame tối thiểu (rev.3)**:
  - **SIL được phép 0 frame**.
  - **Chỉ âm tiết có tiếng mới ép tối thiểu 1 frame**; nếu bên cạnh không có SIL để trừ bù thì
    **mượn 1 frame từ âm tiết kề dài hơn**. (Hiếm — mỗi phone MFA thường ≥ ~30 ms.)
- **Trim biên**: trim nhưng **chừa lại ~100 ms** mỗi đầu (tránh MFA căn đầu/cuối câu kém);
  phần chừa này trở thành **SIL đầu/cuối câu** theo §4.1. Trim **trước khi tính mel và trước MFA**
  (hoặc dịch offset TextGrid tương ứng).
- **log-F0 z-score**: mean/std **toàn cục người nói** (không per-câu).
- **Bucket padding**: pad theo **bội số 64 frame** *và* **bội số 16 âm tiết** (cả hai chiều đều tạo
  shape mới cho MPS); frame pad **ngoài mask của mọi loss và attention**.
- **Cache**: đổi mel/extractor ⇒ regen cache (key tự đổi theo config + hash file stats).
- **Khi conditioning GT (A2.4)**: GT f0/vuv/energy truyền vào forward phải cùng z-score với lúc
  suy luận (dùng chung mean/std lưu trong config).

## 7. Ngưỡng nghiệm thu (để quyết định đi tiếp)
| Mốc | Điều kiện qua |
| --- | --- |
| Vocoder sau đổi mel | Copy-synthesis nghe sạch; **CER chênh ≤ 1 điểm** so với audio gốc |
| MFA | **> 90% clip giữ lại**; chênh ranh giới từ so Whisper **trung vị < 50ms**; 50–100 câu nghe đạt |
| Smoke small (15k) | Val không tăng ngược trước 10k; **so ở mức audio với best.pt 8k**: mỗi model dùng **vocoder của nó**, đo **F0 RMSE (cùng extractor pyworld)** + **CER trên cùng tập câu** — phải tốt hơn rõ |
| Base | **CER của TTS ≤ copy-synthesis + 2–3 điểm** |

## 8. Bộ metric tối thiểu (giữ nguyên)
| Metric | Đo gì |
| --- | --- |
| Mel L1 mean/bin + mask VAD | loss "sạch", bỏ im lặng |
| MCD (DTW) | độ gần phổ |
| log-F0 RMSE (cent) + lỗi V/UV | thanh điệu (đo với duration thật) |
| CER round-trip (Whisper/PhoWhisper, có dấu) | đọc đúng chữ + thanh; so copy-synthesis |
| Duration MAE (ms) vs MFA | nhịp |

## 9. Regularization / model (giữ nguyên rev.1–2)
- Dropout: encoder/decoder **0,2**; variance predictors **0,5**; **không SpecAugment** (mel là target).
- Lần train kế: **base enc4/dec6**, LR **warmup 4–6k → cosine từ đầu**, min 5–10%.

## 10. Vocoder (gộp vào bước 2; bước 7 chỉ còn GTA)
- **MPD đầy đủ (periods 2/3/5/7/11) + MRD** (multi-resolution); tham chiếu **Vocos** + BigVGAN.
- **Multi-resolution STFT loss** (hoặc mel loss trọng số cao ~45 — HiFi-GAN dùng 45).
- hidden 256; **crop 64** (đo MPS: nhanh gần gấp đôi crop 128); **100k+ bước**;
  **đã đo lại MPS vs CPU**: MPS nhanh hơn **3,8–4,7×** khi discriminator lớn (ch=8/16/32) —
  chọn `ch=16` (D 7,7M, 0,67 s/bước) làm mặc định; `ch=32` (D 31M) = 1,49 s/bước.
- Cuối cùng: **GTA fine-tune** trên mel do acoustic teacher-forcing sinh (vài chục nghìn bước).
- 1-graph ONNX: iSTFT bằng **ConvTranspose1d trọng số cố định** + chia window-sum (làm sau).

## 11. Các đính chính giữ nguyên
- VITS/ONNX: lập luận trong báo cáo đầu **quá mạnh** (Piper export ONNX bình thường) — đánh đổi thật
  là train GAN nặng trên MPS; vẫn giữ FS2-style.
- OOM validation: nguyên nhân = **cache shape MPS allocator** (empty_cache 40→19GB) → fix bằng
  bucket padding (frame ×64 **và âm tiết ×16**) + `inference_mode`; sau đó đo val toàn bộ mỗi 5k.
- Tiếng Anh: giữ respell theo **cách người nói trong kênh thực đọc**; không token song ngữ.
- MFA: chọn **(b) train bằng lexicon sinh từ parser** + lọc score; Whisper chỉ để QC; chạy riêng.
