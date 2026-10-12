# REVISION_PLAN_V2 — Sửa nền tảng trước khi lên `base`

> Nguồn: phản hồi chuyên gia (qua chủ dự án) + **xác minh trực tiếp trong code**. Cập nhật: 12/10/2026.
> Trạng thái: A1 đã sửa; các mục ⬜ là kế hoạch theo thứ tự đã thống nhất.

## 0. Run cũ đã dừng
- Acoustic dừng tại **20.850 bước** theo đề xuất: val **tăng ngược** từ bước 8k (274,0) → 305,5 (20k)
  trong khi train vẫn giảm → **overfit vào nhãn duration chia đều**, không đơn thuần là vấn đề dung lượng.
- Giữ `runs/small_codelung/best.pt` (val 274,0 @8k) + `last.pt` **chỉ để test pipeline**.
- Vocoder hiện tại chạy nốt cho demo; **sẽ phải train lại** sau khi đổi biểu diễn mel (§2).

## 1. Ba vấn đề nền tảng — xác minh & kế hoạch

### A1. Mel loss lấn át các loss khác — ✅ ĐÚNG, đã sửa
- Code cũ: `(pred-target).abs()*mask).sum()/mask.sum()` với mask `[B,T,1]` ⇒ **cộng dồn 100 bin**
  (mel ~270 ≈ 2,7/bin), trong khi duration MSE chỉ ~0,1–0,5 → trọng số "1.0" của các loss khác
  thực tế chỉ ~1/100.
- Sửa: `masked_l1(..., mean_over_bins=True)` cho mel (chia thêm `size(-1)`) — có test đơn vị.
- ⚠️ Sau khi sửa phải cân lại trọng số theo độ lớn thật (f0 đang L1 trên **Hz** sẽ trội → xử lý ở A2).

### A2. Variance adaptor — ⚠️ ĐÍNH CHÍNH: **đã tồn tại từ trước**
Bằng chứng (`src/vifasttts/models/vifasttts.py`):
```python
x = self.encoder(self.embedding(features), padding_mask)
syllable_f0 = self.f0(x)...            # dự đoán từ ENCODER (trước decoder)
exp_f0, _, _ = self.length_regulator(...)   # giãn theo duration
variance = torch.cat([exp_f0, exp_vuv, exp_energy], dim=-1)
decoded = self.decoder(expanded + self.variance_proj(variance)
                       + self.position_proj(positions.unsqueeze(-1)), frame_mask)
```
→ f0/vuv/energy **điều khiển decoder** đúng kiểu FS2 (không phải "head phụ sau decoder");
**relative position trong âm tiết cũng đã cộng** (điểm 12.2(i) đã có sẵn).

Phần **còn thiếu thật** (đồng ý với góp ý):
1. f0 là **scalar trung bình mỗi âm tiết** → mất đường nét thanh (hỏi trũng, ngã gãy). →
   ⬜ **4–5 hệ số DCT của log-F0 z-score** mỗi âm tiết, nội suy ra contour cấp frame, embed theo frame.
2. f0 loss đang L1 trên **Hz** → ⬜ **log-F0 z-score** (ổn định + cân được với mel/duration).
3. Extractor autocorrelation dễ lỗi quãng tám → ⬜ **pyworld (harvest + stonemask)** hoặc RMVPE; nội suy qua vô thanh.
4. (tuỳ chọn) ⬜ teacher forcing GT cho nhánh conditioning khi train.

### A3. Không có token khoảng lặng (SIL) — ✅ ĐÚNG
- Hiện pause bị "nhét" vào âm tiết kề qua chia đều → mel âm tiết lẫn im lặng.
- ⬜ **Token SIL**: nucleus += `SIL` (vocab 68→69); sinh từ **VAD** (trước) / **MFA `sp/sil`** (sau);
  dataset dùng duration thật cho SIL; **trim im lặng đầu/cuối clip**.
- Ripple cần làm một batch: vocab size, param counts, export metadata, test đếm tham số.

## 2. Biểu diễn mel (theo 12.8) — ⬜
- `log(power+1e-6)` → **log-magnitude, clamp 1e-5** (kiểu HiFi-GAN) + **chuẩn hoá toàn cục per-bin**
  (mean/std tính trên dataset, lưu vào config).
- KHÔNG chuẩn hoá per-utterance; không top_db per-câu (mất thông tin độ lớn, không đảo ngược được).
- Đổi mel ⇒ **train lại vocoder**; cache key tự đổi theo mel config.

## 3. MFA pipeline (theo 12.1) — ⬜
- Chọn **(b) train MFA bằng lexicon sinh từ parser của mình** + lọc theo score (d).
- Phone set: `onset / medial / nucleus+tone / coda` (~90 phone); duration âm tiết = tổng phone.
- Scripts cần viết:
  - `tools/mfa_build_corpus.py` — wav + lab từ manifest (text đã respell nhất quán),
  - `tools/mfa_train_align.sh` (+ docs) — train trên 51h rồi align,
  - `tools/textgrid_to_durations.py` — TextGrid → durations (đảm bảo **sum == số frame của clip**),
    sinh đoạn SIL, xuất báo cáo QC.
- QC bắt buộc: Praat 50–100 câu chọn theo độ đa dạng; so Whisper **median lệch < ~50ms**;
  phân phối duration theo role/thanh hợp lý (nặng + coda tắc p/t/c/ch ngắn hơn; hỏi/ngã dài hơn);
  lọc 5–10% clip score thấp → fallback heuristic.
- Whisper **chỉ để QC**, không dùng làm nhãn.
- Chạy **riêng**, không song song training (RAM/CPU) — theo lưu ý vận hành (C).
- (dài hạn, tuỳ chọn e) aligner nội bộ kiểu RAD-TTS/FastPitch — chỉ khi MFA score kém.

## 4. Bộ metric tối thiểu (theo 12.7) — ⬜
| Metric | Đo gì |
| --- | --- |
| Mel L1 **mean/bin + mask VAD** | loss "sạch", bỏ ảnh hưởng im lặng |
| **MCD** (có DTW) | độ gần phổ |
| **log-F0 RMSE (cent) + lỗi V/UV** | thanh điệu (đo với duration thật) |
| **CER round-trip** (Whisper/PhoWhisper, có dấu) | đọc đúng chữ + thanh; so với **copy-synthesis** (TTS ≤ copy + 2–3 điểm) |
| **Duration MAE (ms) vs MFA** | nhịp |

## 5. Regularization (theo 12.14) — ⬜
- Dropout: encoder/decoder **0,2**; variance predictors **0,5** (theo FS2 gốc).
- **Không** SpecAugment (mel là target, làm méo nhãn).
- Đánh giá lại overfit **sau khi có duration thật**.

## 6. Model cho lần train kế (theo 12.4, 12.5) — ⬜
- **`base`**, đảo tỷ lệ **encoder 4 / decoder 6** — **chỉ sau khi có alignment thật**.
- LR: warmup 4–6k → **cosine từ đầu** tới tổng số bước, min 5–10%.
- Smoke `small` ~15k với alignment thật trước: kỳ vọng val giảm đều (không tăng ngược sớm),
  F0 RMSE & CER tốt hơn run 8k cũ — đạt mới lên base.

## 7. Vocoder (theo 12.9, 12.10, 12.12) — ⬜
- **Discriminator 39k là điểm chết**: nâng **MPD đầy đủ (2/3/5/7/11) + MRD**; tham chiếu **Vocos**
  (cũng iSTFT-based) + BigVGAN.
- Thêm **multi-resolution STFT loss** (hoặc mel loss trọng số cao ~45).
- hidden 256 là đủ; giữ crop 128; **steps 100k+** (60k quá ít cho GAN).
- Đo lại **MPS vs CPU** sau khi discriminator lớn hơn (kết luận "CPU nhanh hơn" có thể đảo).
- Cuối cùng (12.10): **fine-tune vocoder trên mel GTA** (acoustic teacher-forcing) vài chục nghìn bước.
- (12.12) 1-graph ONNX: viết iSTFT bằng **ConvTranspose1d trọng số cố định** (IDFT × window) + chia
  window-sum — khả thi; hiện giữ 2 graph cho streaming.

## 8. Điểm cần đính chính trong báo cáo trước (để đối chiếu với AI kia)
- **A2**: model **đã có variance adaptor**; lập luận "head phụ" là do báo cáo trước viết mơ hồ.
- **12.6 OOM validation**: rà code **không thấy tích luỹ tensor** (đã dùng `loss.item()` + `no_grad`);
  nguyên nhân là **cache hình dạng của MPS allocator** (bằng chứng: `empty_cache` đưa 40→19GB tức thời;
  crash đúng lúc validation 113 batch nhiều shape khác nhau).
  → Đồng ý giải pháp **pad theo bucket (bội số 64 frame)** + `inference_mode`, sau đó **đo val toàn bộ
  1.799 mẫu mỗi 5k** là khả thi.
- **12.13 VITS/ONNX**: lập luận "VITS hại ONNX/determinism" trong báo cáo trước **quá mạnh** —
  Piper export ONNX bình thường, tất định khi cố định noise. Đánh đổi thật là **train GAN end-to-end
  nặng trên MPS + khó điều khiển nhịp/F0**; Matcha cần nhiều bước ODE → suy luận chậm, export phức tạp.
  → Vẫn giữ FS2-style; chỉ cân nhắc đổi nếu chạm trần sau tất cả bước trên.
- **12.11 tiếng Anh**: giữ respell, nhưng đảm bảo phản ánh **cách người nói trong kênh thực đọc**
  (phiên âm kiểu Việt), không phải chuẩn Anh; không làm token song ngữ với 51h một giọng.

## 9. Thứ tự thực hiện & tiêu chí nghiệm thu (theo 12.15)
1. ✅ **A1** mel loss mean/bin (đã commit + test).
2. ⬜ **Mel** log-magnitude + chuẩn hoá toàn cục → train lại vocoder (copy-synthesis phải nghe sạch).
3. ⬜ **A3 SIL token** + trim biên; **A2 nâng f0** (log-F0 z-score + DCT contour + extractor pyworld).
4. ⬜ **MFA**: lexicon riêng → train → align → lọc 5–10% → QC 50–100 câu + đối chiếu Whisper.
5. ⬜ **Smoke small 15k** với alignment thật: val giảm đều; F0 RMSE & CER tốt hơn run cũ.
6. ⬜ **Base** (enc4/dec6) + LR cosine từ đầu.
7. ⬜ **Vocoder nâng cấp** (MPD+MRD, MR-STFT, 100k+), rồi **GTA fine-tune**.
8. ⬜ **Metrics đầy đủ** (MCD / F0 / CER / Duration MAE) ở từng mốc.

## 10. Vận hành
- Vocoder hiện tại chạy nốt (test pipeline) — sau khi đổi mel sẽ train lại.
- **MFA chạy riêng**, không song song training (RAM chỉ còn ~0,7GB khi chạy song song — đo được không
  còn đáng tin).
- Run cũ (small 8k/20,8k) **không dùng làm model cuối**.
