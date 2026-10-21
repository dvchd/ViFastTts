# A0 lượt 4 — Việc 2b (lưới probe) + Việc 1b (audit dữ liệu) — 21/10/2026

## 0. Trả lời trực tiếp 5 điểm ở §0 của A0

1. **Đúng**: adv ở bước 500 chỉ ~0,01 mà GV đã giảm ⇒ thủ phạm không phải GAN. P0 dưới đây chứng minh.
2. **Đúng**: fm 2,0 là loss hồi quy L1 trên đặc trưng D, kéo về trung bình. P1 (bỏ fm) khác hẳn P-cũ.
3. **Đúng**: hạ lr D là ngược hướng; giữ D lr 1e-4, D:G 1:1.
4. **Đã làm đối chứng P0** (adv 0, fm 0) — xem bảng.
5. **Đã đo cả LIVE và EMA** ở mọi mốc val (cột riêng trong `val_log.csv`).

## 1. Việc 2b — lưới probe 2k (đúng §1: warmup 300, KHÔNG ramp, D lr 1e-4, 1:1, hinge, crop 64)

Gốc mọi probe: `runs/acoustic_v32/best.pt`. Val: **10 batch cố định** (batch 1) của v32-val;
`val_mel` = L1 mel z toàn row (gồm cả frame lặng — cùng công thức mọi probe);
GV = median(var_pred/var_real) trên frame hữu thanh; GV26 = cùng chỉ số trên dải 2–6 kHz.

Baseline (trước train): **val 1,5529 · GV 0,593 · GV26 0,550**.

| probe | bước | val LIVE | **GV LIVE** | GV26 LIVE | val EMA | GV EMA | GV26 EMA | D loss | D acc | grad ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| P0 adv 0 · fm 0 (đối chứng) | 500 | 1,6070 | **0,548** | 0,493 | 1,5603 | 0,586 | 0,543 | — | — | — |
| P0 | 1000 | 1,5871 | **0,564** | 0,505 | 1,5737 | 0,574 | 0,528 | — | — | — |
| P0 | 1500 | 1,6044 | **0,561** | 0,502 | 1,5808 | 0,576 | 0,522 | — | — | — |
| P0 | 2000 | 1,5883 | **0,575** | 0,528 | 1,5853 | 0,570 | 0,517 | — | — | — |
| P1 adv 0,1 · fm 0 | 500 | 1,6224 | **0,572** | 0,537 | 1,5605 | 0,586 | 0,540 | 0,93 | 1,00 | 2,45 |
| P1 | 1000 | 1,6104 | **0,615** | 0,580 | 1,5770 | 0,589 | 0,540 | 0,26 | 1,00 | 2,46 |
| P1 | 1500 | 1,6295 | **0,641** | 0,597 | 1,5884 | 0,602 | 0,551 | 0,62 | 0,94 | 0,61 |
| **P1** | **2000** | **1,6191** | **0,665** | **0,629** | 1,5964 | 0,616 | 0,569 | 0,88 | 1,00 | 1,42 |
| P2 adv 0,3 · fm 0 | 500 | 1,6541 | **0,664** | 0,642 | 1,5609 | 0,593 | 0,543 | 1,41 | 0,95 | 2,81 |
| P2 | 1000 | 1,6794 | **0,774** | 0,785 | 1,5830 | 0,634 | 0,602 | 0,60 | 0,99 | 2,24 |
| P2 | 1500 | 1,6710 | **0,735** | 0,719 | 1,6022 | 0,686 | 0,654 | 0,70 | 0,98 | 0,75 |
| P2 | 2000 | 1,6702 | **0,792** | 0,729 | 1,6167 | 0,724 | 0,680 | 1,15 | 0,95 | 1,53 |
| P3 adaptive · fm 0 | 500 | 1,6131 | **0,550** | 0,507 | 1,5602 | 0,584 | 0,540 | 0,71 | 1,00 | — |
| P3 | 1000 | 1,5959 | **0,584** | 0,539 | 1,5752 | 0,578 | 0,531 | 0,16 | 1,00 | — |
| P3 | 1500 | 1,6163 | **0,586** | 0,539 | 1,5847 | 0,580 | 0,535 | 0,36 | 1,00 | — |
| P3 | 2000 | 1,6014 | **0,612** | 0,581 | 1,5912 | 0,586 | 0,537 | 0,33 | 1,00 | — |
| P4 adaptive · fm 0,5 | 500 | 1,6124 | **0,558** | 0,511 | 1,5601 | 0,586 | 0,542 | 0,72 | 1,00 | — |
| P4 | 1000 | 1,5950 | **0,589** | 0,539 | 1,5750 | 0,581 | 0,534 | 0,16 | 1,00 | — |
| P4 | 1500 | 1,6159 | **0,590** | 0,540 | 1,5845 | 0,582 | 0,538 | 0,41 | 0,98 | — |
| P4 | 2000 | 1,6018 | **0,611** | 0,580 | 1,5910 | 0,589 | 0,541 | 0,35 | 1,00 | — |

(D loss/D acc/grad ratio: lấy từ `train_log.csv` tại đúng bước val. `—` = P0 không dùng D;
P3/P4 `grad ratio` để trống vì λ đo trên Conv cuối postnet, không đo trên decoder.)

### Quyết định theo §2 của A0

| probe | GV 2k | val 2k | cổng GV ≥ 0,65 & val ≤ +5 % |
|---|---|---|---|
| P0 | 0,575 (↓ so 0,593) | +2,3 % | ✗ (GV giảm) |
| **P1** | **0,665** | **+4,3 %** | **✓ CHỌN** |
| P2 | 0,792 | +7,6 % | ✗ (val vượt 5 %) |
| P3 | 0,612 | +3,1 % | ✗ |
| P4 | 0,611 | +3,2 % | ✗ |

⇒ Chạy **20k với cấu hình P1** (`runs/adv_p1_20k`, đang chạy, checkpoint mỗi 2k, hard-stop ±10 %).

### Vì sao P3/P4 (adaptive) yếu

λ đo được **0,055** (nhỏ hơn nhiều so với 0,1 của P1) ⇒ trọng số adv thực = 0,055, gần P0.
Nguyên nhân: `||grad L_rec||` trên **Conv cuối postnet** nhỏ so với `||grad L_adv||` trên cùng
tenor; công thức VQGAN nhân 0,5 ⇒ λ ~0,05. Muốn adaptive có tác dụng phải đo trên **toàn bộ
decoder** (nơi grad adv lớn) hoặc bỏ hệ số 0,5. **Chưa sửa trong lượt này** (A0 chốt "không chỉnh
lặt vặt"); ghi lại để lượt sau quyết.

### Cảnh báo về thước GV (quan trọng)

GV là **tỉ lệ phương sai** — có thể bị "gian lận": P4 ở bước 400 có GV LIVE **0,949** nhưng val
+10,1 % (model rung/nhiễu chứ không phải texture). Vì vậy GV **không đủ** để chấp nhận; phải kèm
**nghe + AP 2–6 kHz**. P2 (GV 0,79, val +7,6 %) cũng cần kiểm tai trước khi coi là "thắng".

**Đọc nhanh**: P0 (L1 thuần) **tự làm GV giảm** ngay cả khi không có GAN ⇒ một phần mất mát
"độ gai" trong fine-tune là do L1, không phải do adversarial. Điều này khớp giả thuyết A0 và
có nghĩa: muốn so GAN phải so **chênh lệch so với P0**, không so với baseline.

## 2. Việc 1b.2.1 — CER audio THẬT (Whisper small int8, cùng công thức `eval_acoustic`)

Tool mới: `tools/audit_real_cer.py` (cắt đúng `frame_range`, hop 256 @ 24 kHz; CER ký tự;
phân loại lỗi bằng `parse_syllable`; xuất top + trang nghe).

### 2.1a CER thật trên **đúng 100 clip** của CER-100 (cùng id với `v32_cer100.json`)

| | mean | median | p90 |
|---|---|---|---|
| **Audio THẬT** | **0,712** | **0,347** | 1,52 |
| Audio model v3.2 (đã đo trước) | 0,369 | 0,252 | — |

Ghép cặp 100 clip: **median Δ (thật − model) = +0,067**; model có CER thấp hơn audio thật ở **66/100 clip**.
⇒ **CER 0,37 của model KHÔNG phải bằng chứng phát âm kém**: nó *tốt hơn* audio thật theo cùng ASR.
Mean bị thổi bởi đuôi nặng: 4/100 clip Whisper **hallucinate** câu CTA YouTube ("Hãy subscribe…"),
5/100 clip <100 frame có CER trung bình **2,71**.

### 2.1b CER thật trên 300 segment val (cùng bộ bench)

| n | mean | median | p90 | p97 | clip <100 frame | hallucination |
|---|---|---|---|---|---|---|
| 300 | 0,858 | **0,415** | 1,37 | 4,01 | 24 clip (CER tb 4,06) | 6 clip |

⇒ Dùng **median** (0,35–0,42) làm mốc; mean chỉ dùng khi đã lọc clip ngắn/hallucination.

### 2.1c CER thật trên **1.000 segment train** (lấy mẫu ngẫu nhiên, seed 42)

| n | mean | median | p90 | p97 |
|---|---|---|---|---|
| 1.000 | 0,822 | **0,411** | 1,25 | 3,57 |

⇒ Ổn định với val-300 (median 0,415) và holdout-100 (median 0,347): **median thật ≈ 0,41**.
Ước lượng thời gian quét cả corpus (15,7k row): ~8 giờ Whisper CPU ⇒ chưa chạy toàn bộ.

### 2.1d Cách transcript đã được sửa (lịch sử repo)

- `tools/audit_bigrams.py`: tìm **bigram hiếm** có biến thể phổ biến → 145 nghi vấn → duyệt tay
  còn **129 cặp** ("thư dục→thư mục", "hột cái→một cái", "trài khoản→tài khoản"…).
- `tools/apply_transcript_fixes.py` + `resources/transcript_fixes.json`: áp **114/9763 row** (chỉ thay
  từ nguyên vẹn, giữ hoa/thường + dấu câu). Commit `328c8f1`, `a4b7aa2` (sửa bug mất từ đầu cặp 2→2).
- `tools/audit_transcripts.py`: soát chéo bằng Whisper (faster-whisper) từng clip, xếp cặp lệch lặp lại.
- **Không áp lên val/holdout** (giữ mốc so sánh, tránh "tự thưởng" CER).
- Hệ quả còn lại (đo được): các từ vay tiếng Anh viết theo phiên âm ("páp" = public, "vê pê ét" = VPS,
  "quếp" = web, "phôn đờ"…). ASR so sai nhiều nhất ở nhóm này — không phải lỗi nghe của model.

### 2.1e 30 segment lệch nhất (cho người dùng nghe)

- holdout-100: https://cio-rows-rocks-seq.trycloudflare.com/audit_real/
- val-300: https://cio-rows-rocks-seq.trycloudflare.com/audit_real_val/

Mỗi trang: audio THẬT + transcript manifest + Whisper nghe được, xếp theo CER.

## 3. Việc 1b.2.2 — MFA có chọn sai coda NH không?

Tool mới: `tools/audit_mfa_coda.py` (đọc 19.168 TextGrid, khớp phone theo thời gian vào từng word,
so với coda frontend). Toàn bộ corpus:

| coda | tổng | đúng | lệch |
|---|---|---|---|
| **NH** | 18.609 | 17.502 (94,1 %) | **n: 1.107 = 5,9 %** |
| N | 33.156 | 24.840 (74,9 %) | ŋ: 8.168 (24,6 %), ŋm: 148 |
| NG | 27.397 | 11.591 (42,3 %) | ŋm: 15.806 (57,7 %) |
| CH | 3.726 | 2.738 (73,5 %) | t̚: 260, mất phone cuối: 407 |
| T | 18.062 | 9.184 (50,8 %) | k̚: 2.448, mất phone cuối: ~5.000 |

**Kết luận theo cổng của A0**: NH→n = **5,9 % < 20 %** ⇒ **KHÔNG cần re-align** cho v3.5.

**Phát hiện thêm (quan trọng hơn)**: từ điển MFA `data/codelung/vn_models/vietnamese_mfa.dict` là
**đa biến thể Bắc + Nam** (vd "điện" có 2 entry: `ɗ iː˨˩˨ ŋ` prob 0,99 và `ɗ iə˨˩ˀ n` prob 0,42;
"bạn" có `n` 0,99 và `ŋ` 0,38). Aligner chọn theo xác suất **của từng utterance** ⇒ **24,6 % token
coda N bị gán phone ŋ** và 57,7 % NG bị gán `ŋm` — tức biên độ dài coda/nguyên âm của một phần
dữ liệu đang theo **cách phát âm Nam**, trong khi giọng đọc là Bắc. Đây là ứng viên sửa cho v3.5:
**chỉ giữ biến thể Bắc trong dict** rồi re-align (đo lại sau).

## 4. Việc 1b.2.3 — codec nguồn + RNNoise

- **Codec: KHÔNG đo được.** Toàn bộ `data/codelung/audio*/**` là **WAV PCM 24 kHz** (đã decode + resample);
  bản tải gốc (webm/m4a) không còn trong repo. ffprobe chỉ thấy `pcm_s16le 24000`.
- **Thay thế — phổ trung bình 40 clip** (dB so dải 1–4 kHz): 4–6 kHz −15,9 · 6–8 −26,0 · 8–10 −32,1 ·
  10–11,5 −37,4 · 11,5–12 −43,9. **Không có "tường" lowpass** dưới 12 kHz ⇒ không có dấu hiệu codec
  bitrate thấp (AAC ≤96k thường cắt 11–13 kHz). Không cần tải lại Opus 251 để kiểm.
- **RNNoise có phá cấu trúc hài không?** 15 clip/nhóm (nhóm không RNNoise = 2 video "Học làm game"
  4kyqfOypVWI + lRojc8KDJ6k; nhóm RNNoise = 15 clip ngẫu nhiên video khác):

| nhóm | HNR (dB) | AP 2–6 kHz | CPP (dB) | jitter % |
|---|---|---|---|---|
| KHÔNG RNNoise | 4,89 | 0,59 | 0,91 | 3,44 |
| CÓ RNNoise | 5,09 | **0,54** | 1,08 | 4,09 |

⇒ RNNoise **không phá** cấu trúc hài (AP 2–6 kHz *thấp hơn* = ít nhiễu hơn). Lưu ý: hai nhóm khác
video/phòng thu ⇒ đây là so sánh thô, không phải thí nghiệm nhân quả.

## 5. Kết luận & đề xuất

1. **Bỏ CER làm cổng phát âm** (dùng để log): mốc thật median 0,35–0,42; model đã *thấp hơn* thật.
2. **Không re-align vì NH** (5,9 %). Có thể re-align *có chủ đích* vì biến thể Nam trong dict (mục 3).
3. **RNNoise + codec không phải nút thắt** (mục 4).
4. Việc 2: xem bảng probe — chọn cấu hình theo cổng GV ≥ 0,65 @2k + val ≤ +5 %.
