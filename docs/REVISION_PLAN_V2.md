# REVISION_PLAN_V2 (rev.2) — Sửa nền tảng trước khi lên `base`

> Nguồn: phản hồi chuyên gia (2 vòng) + **xác minh trực tiếp trong code**. Cập nhật: 12/10/2026.
> **rev.2** tiếp thu 5 điểm sửa: (1) đảo thứ tự MFA lên trước SIL/F0-DCT; (2) A2.4 **bắt buộc**;
> (3) đặc tả đầy đủ 7 slot cho token SIL (thêm tone=NONE → vocab 70) + mask loss;
> (4) phone set MFA **prefix theo vị trí**; (5) chốt cách làm tròn duration / trim biên / z-score /
> bucket padding; (6) thêm **ngưỡng nghiệm thu cụ thể**.

## 0. Run cũ đã dừng
- Acoustic dừng tại **20.850 bước**: val tăng ngược từ 8k (274,0) → 305,5 (20k) trong khi train giảm
  ⇒ **overfit vào nhãn duration chia đều** (không đơn thuần thiếu dung lượng).
- Giữ `runs/small_codelung/best.pt` (274,0 @8k) + `last.pt` **chỉ để test pipeline**.
- Vocoder hiện tại chạy nốt cho demo; **sẽ train lại** sau khi đổi biểu diễn mel (§3 bước 2).

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

**A2.4 — chuyển thành BẮT BUỘC:**
- Thêm `model.variance_teacher_forcing: true`: forward nhận thêm `f0_target`/`energy_target`
  (cấp âm tiết, đã z-score) và **dùng GT để conditioning khi train**; suy luận/ONNX dùng giá trị dự đoán.
- Có test (train dùng GT, infer dùng prediction; ONNX export không đổi input).
- Phần còn lại của A2 (làm ở bước 4, sau MFA): **log-F0 z-score toàn cục** + **DCT contour 4–5 hệ số/âm tiết**
  + **pyworld (harvest+stonemask)** — riêng pyworld làm sớm ở bước 2 vì regen cache.

## 3. Thứ tự thực hiện (rev.2 — đảo MFA lên trước)
| Bước | Việc | Phụ thuộc |
| --- | --- | --- |
| 1 | ✅ A1 mel loss | — |
| 2 | **Mel mới** (log-magnitude clamp 1e-5 + chuẩn hoá toàn cục per-bin) + **pyworld** + **regen cache** → **train lại vocoder** | — (không cần MFA) |
| 3 | **MFA**: lexicon từ parser (prefix vị trí) → `mfa train` → align → lọc 5–10% → QC | chạy **riêng**, không song song training |
| 4 | **SIL + log-F0 DCT + conditioning GT** làm cùng một đợt — đều lấy ranh giới từ MFA | cần bước 3 |
| 5 | **Smoke small 15k** với alignment thật | cần bước 4 |
| 6 | **Base enc4/dec6** + LR cosine từ đầu | đạt ngưỡng bước 5 |
| 7 | **Vocoder nâng cấp** (MPD đầy đủ + MRD + MR-STFT, 100k+) → **GTA fine-tune** | — |

Lý do đảo: (a) DCT contour tính trên ranh giới âm tiết thật — nếu vẫn chia đều thì hệ số DCT cũng lệch
y như nhãn cũ; (b) SIL từ VAD rồi thay bằng sp/sil của MFA là làm hai lần.

## 4. Đặc tả token SIL (chốt 7 slot)
- `onset=NONE`, `medial=NONE`, `nucleus=SIL`, `coda=NONE`, **`tone=NONE`** (thêm mới),
  `boundary`/`punctuation` = **giữ nguyên giá trị của vị trí đứng trước**
  (model biết đây là ngừng sau phẩy hay sau chấm).
- **Vocab: 24/2/16/11/7/4/6 = 70** (không phải 69). Không gán NGANG cho SIL (tránh "thanh ngang = im lặng").
- **Mask loss**: thêm `speech_mask` cấp âm tiết vào batch → **tone CE + f0/vuv/energy loss mask bỏ SIL**;
  **duration loss KHÔNG mask** SIL (cần học độ dài im lặng).
- SIL sinh từ **MFA `sp/sil`**; **trim biên trước**; SIL có duration thật.
- Ripple phải làm cùng batch: vocab size, param counts, export metadata, test đếm tham số, docs.

## 5. MFA — phone set prefix theo vị trí
- `o_ng` (onset) vs `c_ng` (coda); `m_w` (medial) vs `c_w` (coda); `c_c`, `c_ch`, `c_p`, `c_t`... (~90–100 nhãn).
- Lý do: cùng ký hiệu nhưng **âm học khác** (coda tắc không bật hơi) → tách mô hình âm học;
  và map TextGrid → 5 thành phần **không mơ hồ**.
- `nucleus+tone` gắn liền 1 nhãn (thanh không align thành đoạn riêng).

## 6. Chi tiết dễ sai — chốt cách làm
- **Làm tròn duration**: `boundary_i = round(t_i · sr / hop)` cho từng ranh giới;
  `duration_i = boundary_{i+1} − boundary_i` (hiệu các mốc tích luỹ **đã làm tròn**).
  Tổng khớp tuyệt đối số frame của clip. Âm tiết về 0 frame → **ép tối thiểu 1 frame, trừ bù vào SIL kề**.
- **Trim biên**: trim **trước khi tính mel và trước khi chạy MFA** (hoặc dịch offset TextGrid tương ứng) —
  nếu không, ranh giới lệch đúng bằng phần đã cắt.
- **log-F0 z-score**: mean/std **toàn cục người nói** (không per-câu).
- **Bucket padding** (bội số 64 frame): frame pad **ngoài mask của mọi loss và attention**.
- **Cache**: đổi mel/extractor ⇒ regen cache (key tự đổi theo config).
- **Khi conditioning GT (A2.4)**: GT f0/energy truyền vào forward phải cùng z-score với lúc suy luận
  (dùng chung mean/std lưu trong config).

## 7. Ngưỡng nghiệm thu (để quyết định đi tiếp)
| Mốc | Điều kiện qua |
| --- | --- |
| Vocoder sau đổi mel | Copy-synthesis nghe sạch; **CER chênh ≤ 1 điểm** so với audio gốc |
| MFA | **> 90% clip giữ lại**; chênh ranh giới từ so Whisper **trung vị < 50ms**; 50–100 câu nghe đạt |
| Smoke small (15k) | **Val không tăng ngược trước 10k**; F0 RMSE & CER tốt hơn **rõ** so với best.pt 8k |
| Base | **CER của TTS ≤ copy-synthesis + 2–3 điểm** |

## 8. Bộ metric tối thiểu (giữ nguyên)
| Metric | Đo gì |
| --- | --- |
| Mel L1 mean/bin + mask VAD | loss "sạch", bỏ im lặng |
| MCD (DTW) | độ gần phổ |
| log-F0 RMSE (cent) + lỗi V/UV | thanh điệu (đo với duration thật) |
| CER round-trip (Whisper/PhoWhisper, có dấu) | đọc đúng chữ + thanh; so copy-synthesis |
| Duration MAE (ms) vs MFA | nhịp |

## 9. Regularization / model / vocoder (giữ nguyên rev.1)
- Dropout: encoder/decoder **0,2**; variance predictors **0,5**; **không SpecAugment** (mel là target).
- Lần train kế: **base enc4/dec6**, LR **warmup 4–6k → cosine từ đầu**, min 5–10%.
- Vocoder: **MPD đầy đủ (2/3/5/7/11) + MRD**; **multi-resolution STFT loss** (hoặc mel ~45);
  hidden 256; crop 128; **100k+ bước**; đo lại MPS vs CPU khi discriminator to hơn;
  cuối cùng **GTA fine-tune** trên mel do acoustic teacher-forcing sinh.
- 1-graph ONNX: iSTFT bằng **ConvTranspose1d trọng số cố định** + chia window-sum (làm sau, giữ 2 graph trước).

## 10. Các đính chính giữ nguyên
- VITS/ONNX: lập luận trong báo cáo đầu **quá mạnh** (Piper export ONNX bình thường) — đánh đổi thật là
  train GAN nặng trên MPS; vẫn giữ FS2-style.
- OOM validation: không có tích luỹ tensor trong code; nguyên nhân = **cache shape MPS allocator**
  (empty_cache 40→19GB) → fix bằng bucket padding + `inference_mode`; sau đó đo val toàn bộ mỗi 5k.
- Tiếng Anh: giữ respell theo **cách người nói trong kênh thực đọc**; không token song ngữ.
- MFA: chọn **(b) train bằng lexicon sinh từ parser** + lọc score; Whisper chỉ để QC; chạy riêng.
