# Thảo luận lại kiến trúc & ngân sách train — Mac mini M6 32GB + dữ liệu @codelung

> Bản gửi chuyên gia (vòng 4). Trạng thái: **mọi tiến trình đã dừng** để phân tích lại trước khi
> dồn máy chạy lần cuối. Cập nhật: 12/10/2026.

## 0. Trạng thái đã dừng & những gì đã có

- **Đã dừng toàn bộ**: acoustic small (20.850 bước), vocoder cũ (~46k/60k), cache feature
  (1.641/13.225 file — chạy tiếp được, không mất gì). Giữ `best.pt` 8k **chỉ để test pipeline**.
- Đã sửa/đã có từ các vòng trước (không hỏi lại):
  1. **A1** mel loss mean/bin ✅
  2. **A2.4** conditioning GT (f0/vuv/energy khi train, dự đoán khi suy luận) — bắt buộc ✅ (code chờ)
  3. **A3 SIL**: vị trí cố định (đầu/cuối câu + sau dấu câu, duration 0 được) + 7 slot + tone NONE → vocab 70
  4. **Mel mới**: log-magnitude clamp 1e-5 + chuẩn hoá toàn cục per-bin (đã tính stats 400 clip)
  5. **F0 pyworld** harvest + stonemask (đo: ~21× realtime/lõi → 51h ≈ 18 phút với 8 worker)
  6. **Vocoder nâng cấp**: MPD đầy đủ 2/3/5/7/11 + MRD + MR-STFT (đã có code + test + chunked runner MPS)
  7. **Corpus MFA thô đã dựng**: `data/codelung/mfa/` (13.225 wav symlink + labs + lexicon + phones +
     danh sách fallback) — **chưa train MFA**

## 1. Hồ sơ phần cứng (số đo thật)

| Hạng mục | Đo được |
| --- | --- |
| Máy | Mac mini M6 · RAM 32GB unified · CPU ~12 lõi · ổ trống 360GB |
| Acoustic small 6,14M · batch 16 (~29.000 frame/bước) | **0,478 s/bước** model MPS thuần · **0,67–0,99 s** thực tế · fp16 MPS **không lợi** (0,52) |
| Vocoder cũ (MPD-lite 39k, CPU) | 0,166–0,25 s/bước (MPS chậm hơn 2,4×) |
| **Vocoder mới** (MPD+MRD, crop 64, batch 8) | MPS: ch=8 **0,48s** · ch=16 (D 7,7M) **0,67s** · ch=32 (D 31M) **1,49s**; CPU chậm hơn **3,8–4,7×** |
| Bộ nhớ MPS | từng leo **40GB → OOM** (limit báo 42,43GB) dù RAM 32GB — cache theo shape + validation dài; đã chữa bằng `empty_cache` + cap val 40 batch + chạy **khối 5.000 bước** (tiến trình mới) |
| Dự đoán base 16,4M (enc4/dec6, batch 6–8) | 0,75–1,25 s/bước · 120–160k bước ≈ **30–48h** *(ước lượng)* |
| Ưu tiên người dùng | **nhiệt/độ bền máy** khi chạy dài (không cloud) |

## 2. Dữ liệu @codelung (đo mới)

| Hạng mục | Số |
| --- | --- |
| Tổng | 13.225 clip · **51,07h** · 24kHz mono · train 11.426 (**44,4h**) / val 1.799 (6,7h) theo video |
| Clip | 1–20s · TB **13,98s** · p50 14,7s · p90 18,9s · chỉ 13% ≤ 8s (2,3h) |
| Nội dung | video lập trình/công nghệ, **mật độ tiếng Anh cao**; respell chữ-số tốt (chỉ 0,1% clip còn chứa số) |
| Fallback | 1,24% âm tiết nhưng **tập trung: 3.859 clip (29%) có ≥1 từ fallback · 3.217 loại từ** (62% hapax) — đa số là thuật ngữ Anh (window, ordering, voice, smart, touch, chromium, clarify, approve) + typo caption (chffix, endp, maxide, boxy, dust, ter). Train còn **29,6h clip 0-fallback** |
| Lexicon MFA | 6.351 mục (train+val) → **121 phone** (nucleus+tone 90 · onset 23 · medial 1 · coda 10) theo tiền tố vị trí |

## 3. Ngân sách thời gian tham chiếu

| Kịch bản | Ước lượng máy này |
| --- | --- |
| Acoustic small @ 60k / @ 120k (batch 16) | ~12h / ~24h |
| Acoustic base @ 60k / @ 120k (batch 6–8) | **~20–30h / ~40–60h** |
| Vocoder mới ch=16 crop 64 @ 100k (MPS) | **19–24h** |
| Vocoder mới ch=32 @ 100k | 41h+ |
| MFA train + align 51h (CPU) | **chưa biết — hỏi C1** |

## 4. Câu hỏi thảo luận

### A. Ngân sách & chọn model cho lần train "hoàn chỉnh" đầu tiên
- **A1.** Với 44h dữ liệu và máy này, lần đầu nên: (a) small @ ~120k (~24h), (b) base @ ~60k (~20–30h),
  hay (c) **small smoke 15–30k trước** để xác nhận alignment + SIL + log-F0 DCT, đạt mới dồn base?
  *(mình nghiêng (c))*
- **A2.** Quy tắc "enc4/dec6" — nhưng trên MPS chi phí nằm ở **decoder** (T ≈ 3.000 frame/bước, encoder
  chỉ N ≈ 300 âm tiết): giữ 4/6 hay thử 3/7, 4/7 và đo s/bước? Tiêu chí chọn ngoài lý thuyết?
- **A3.** Số bước: theo "tokens/step + epochs" hay đơn giản 60k (small) / 100–160k (base)? Ngưỡng dừng
  thực dụng — theo val audio-level hay loss?
- **A4.** Song song acoustic + vocoder (trước lợi ~2,8h nhưng RAM 31/32GB + nhiệt cao): tiếp tục song song
  có kiểm soát, hay tuần tự? Có nên duty-cycle để tản nhiệt?
- **A5.** Giảm hop 256 → 320 (10,7 → 13,3ms; bớt ~20% frame cả train lẫn inference) để tiết kiệm máy:
  đánh đổi chất lượng tới đâu? (iSTFT hop nội bộ cũng giảm theo)

### B. Dữ liệu & tiền xử lý
- **B1.** 29% clip chứa từ fallback: (a) loại hẳn khỏi MFA + acoustic (còn 29,6h train), (b) mở rộng
  overrides top 300–500 từ phổ biến rồi parse lại (cứu phần lớn 14,8h), (c) cả hai? *(mình nghiêng (c),
  nhưng cần ước lượng công QC đọc thử)*
- **B2.** Clip TB 14s: cắt segment 4–8s khi train acoustic (tăng batch/ổn định, phá ngữ điệu dài) — nên
  không? Nếu cắt thì cắt theo ranh giới từ/âm tiết?
- **B3.** Biên clip theo phụ đề có thể cụt phụ âm: mở rộng ±100–150ms trước MFA (phần thừa thành SIL
  đầu/cuối)? Làm luôn khi dựng lại clip hay chỉ khi align?
- **B4.** Đo SNR/speech-ratio rồi loại 5–10% tệ nhất trước MFA, hay để MFA score tự lọc?
- **B5.** Tách "val thuần Việt" (không từ Anh) để đo generalization thật, tránh điểm ảo?

### C. MFA trên CPU
- **C1.** `mfa train` từ đầu (Kaldi, ~12 lõi, num_jobs 8) cho ~50h/13k clip 14s: **mất khoảng bao lâu**?
  Train trên subset 15–20h sạch nhất rồi align full có hợp lý hơn không?
- **C2.** Phone set hiện tại (4 tiền tố, nucleus+tone 90 nhãn, tổng 121) đủ mịn chưa? Có nên tách
  IÊ/UÔ/ƯƠ thành 2 phone (lúc đó "1 nhãn ≠ 1 âm tiết")?
- **C3.** Chạy Whisper-MLX (MPS) làm QC chéo thời lượng cấp từ (median diff <50ms) và/hoặc lọc clip
  trước MFA — có đáng?
- **C4.** Ngưỡng loại clip theo score (%) nào? Nếu >10% bị loại: nới cấu hình MFA hay chuyển aligner
  nội bộ (RAD-TTS) ngay?

### D. Vocoder
- **D1.** ch=16 (D 7,7M · 0,67s/b · 100k ≈ 19–24h) vs ch=32 (D 31M · 1,49s/b · 41h+): chọn gì?
  Hai pha (ch16 100k → fine-tune ch32 ~20k LR thấp) có đáng?
- **D2.** crop 64 (0,68s) vs 128 (1,36s): ảnh hưởng chất lượng? Curriculum 64→128 nửa sau?
- **D3.** Generator rất nhẹ (iSTFTNet-lite ~3,5M, 2 tầng conv): nâng gần Vocos (ConvNeXt blocks,
  hidden 384–512) trong cùng ngân sách bước, hay giữ lite + dồn ngân sách cho bước?
- **D4.** AMP/fp16 MPS cho GAN (chỉ discriminator) có đáng thử?

### E. Vận hành MPS
- **E1.** Đặt `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (0,85–0,9) thay vì mặc định unbounded để chặn leo 40GB —
  có ảnh hưởng tốc độ?
- **E2.** `empty_cache` mỗi 10 bước có phá tái dùng buffer? Chu kỳ/điều kiện khuyến nghị?
- **E3.** `torch.compile` trên MPS (torch 2.14) cho acoustic/vocoder: đáng thử hay bỏ qua?
- **E4.** GAN dài trên MPS: pitfall (fragmentation theo shape, kernel không tất định) và mẹo ổn định?

### F. Kiến trúc (mở lại có kiểm soát)
- **F1.** Giữ FS2 + Conformer-lite? Hay Transformer thuần (bỏ depthwise conv) hợp MPS/ONNX hơn — có số
  liệu so sánh nào đáng tin?
- **F2.** Postnet mel 2–3 lớp (giảm mismatch train/infer trước GTA) — làm ngay hay để giai đoạn GTA?
- **F3.** Tone classifier aux: giữ hay bỏ khi đã có log-F0 DCT + GT conditioning (tone đã là input)?
- **F4.** Energy: DCT 4–5 hệ số như f0 hay giữ scalar/âm tiết?

## 5. Việc sẽ làm ngay khi có kết luận (không cần bàn)
1. Chạy tiếp **cache feature** (đang dừng 1.641/13.225 — chạy lại vài phút).
2. **`mfa train`** sau khi chốt C1–C4 (chạy riêng, không song song training).
3. **Smoke small** với alignment + SIL + log-F0 DCT để xác nhận trước khi dồn máy cho base.
