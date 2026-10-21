# A0 lượt 3 — Probe Việc 2 (adversarial fine-tune) — 21/10/2026

## Trả lời nhanh câu hỏi của Agent 0

- α=0 vs α=1,5: **nghe mù không phân biệt được** (người dùng xác nhận). ⇒ α trung tính về cảm nhận;
  không dùng α làm biến số nữa (giữ 1,5 như hiện tại cho khỏi đổi pipeline).
- DNSMOS: đã bỏ khỏi so sánh chính (chỉ log), đúng chỉ đạo.
- Centroid: giữ, chỉ so cặp cùng onset+thanh, n≥5.

## Việc 2 — script đã có, chạy được: `tools/finetune_adv.py`

Đúng spec: freeze embedding/encoder/duration/slot/pause/variance (**trainable 2,82M/7,05M** = decoder
+ mel + postnet); G loss mel 1.0 + mel_pre 0.2 + mel_sharp 1.0; **hinge**; crop 64 frame ≥80% vùng
tiếng; G lr 2e-5, D lr 1e-4, betas (0.5,0.9), clip 1.0; EMA 0.999; dừng khẩn theo spec.
Tốc độ: **~0,19 s/bước** ⇒ 20k ≈ 63 phút.

### Probe 1k (d-warmup 250 để thấy cả nhánh G; full run sẽ dùng 1000)

| bước | mel train | D loss | adv | val mel L1 | **GV hữu thanh** | GV 2–6 kHz |
|---|---|---|---|---|---|---|
| 0 (baseline) | — | — | — | **1,5529** | **0,593** | 0,550 |
| 250 | 0,570 | 0,723 | 0 | 1,5529 | 0,593 | 0,550 |
| 500 | 0,501 | 0,548 | 0,823 | 1,5620 | 0,584 | 0,541 |
| 750 | 0,657 | 0,540 | 1,375 | 1,5708 | 0,575 | 0,534 |
| 1000 | 0,682 | **0,101** | 1,355 | 1,5743 | **0,572** | 0,528 |

### Đọc kết quả (trung thực — đây là tín hiệu XẤU)
1. **GV hữu thanh GIẢM** (0,593 → 0,572) — ngược mục tiêu (>0,75).
2. **Val mel L1 tăng** +1,4% (còn trong ngưỡng 5% nhưng đang xấu đi).
3. **D thắng quá nhanh**: D loss 0,723 → **0,101** chỉ sau ~750 bước G; adv của G tăng 0,82 → 1,38
   (G không lừa được D). Warmup D kiểu này làm D mạnh trước khi G học.
4. Ramp adv mới đạt ~0,037 (trọng số 0,1 × 750/2000) mà GV đã giảm ⇒ không phải do adv quá mạnh,
   mà do **D quá mạnh** + **mel_sharp/fm đang kéo lệch**.

### Đề xuất chỉnh (chờ A0 duyệt trước khi chạy 20k)
- **D yếu lại**: d_lr 1e-4 → **5e-5**, hoặc cập nhật D **1 bước mỗi 2 bước G**.
- **Warmup D ngắn lại**: 1000 → **250–400** (đủ để D khởi động, không đủ để áp đảo).
- **adv 0,1 giữ nguyên**, nhưng **fm 2,0 giữ** (fm là thứ giữ cấu trúc; nếu bỏ thì GV còn giảm nữa).
- Thêm theo dõi **val GV ở cả 10 batch** (đã có) + **D accuracy** (tỉ lệ D(real) > D(fake)).
- Nếu sau khi chỉnh mà 2k bước vẫn GV giảm → **dừng Việc 2**, chuyển sang **v3.5** (mask lặng +
  mel stats + f0_delta 0,1 + EMA + slot N/C embedding) và **flow-matching** chỉ khi cần.

## Việc 1b (audit dữ liệu) — chưa làm, cần thêm thời gian

Kế hoạch đã rõ: (2.1) CER real-audio bằng Whisper trên 300 segment val + 100 holdout + tìm top 3%
lệch + 30 segment tệ nhất cho người dùng nghe; (2.2) đếm token coda NH mà MFA chọn phone n (từ
TextGrid), tách theo vần; (2.3) ffprobe codec + so Opus itag 251 cho 2 video + so nhóm RNNoise vs
không RNNoise (2 video ồn).

## Trạng thái khác
- v3.4 (coda/onset-weighted): đang train, sẽ chấm xong ~16:40–17:30 — **giữ làm số tham khảo**, không
  dùng làm gốc (đúng chỉ đạo).
- Gốc cho mọi bước = `runs/acoustic_v32/best.pt` (v3.2 small 59k).
