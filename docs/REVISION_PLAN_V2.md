# REVISION_PLAN_V2 (rev.5) — Sửa nền tảng trước khi lên `base`

> Nguồn: phản hồi chuyên gia (5 vòng) + xác minh trong code. Cập nhật: 12/10/2026.
> **rev.5** chốt 3 điểm trước khi khởi động vocoder + chỉnh nhỏ (xem §9). Các mục dưới giữ nguyên
> từ rev.4 và đã được cập nhật dần trong §1/§2.

## 1. Trạng thái code (đã có)
- **A1** mel loss mean/bin ✅ · **A2.4** conditioning GT f0/vuv/energy ✅ (sẽ bật khi retrain)
- **Mel mới** log-magnitude clamp 1e-5 + chuẩn hoá toàn cục per-bin ✅ (`tools/compute_mel_stats.py`)
- **F0 pyworld** harvest+stonemask ✅ (đo ~21× realtime/lõi)
- **Vocoder nâng cấp**: MPD đầy đủ 2/3/5/7/11 + MRD + MR-STFT ✅; **generator Vocos**
  (ConvNeXt hidden 384 × 8 block, **expansion 3× → 12.069.762 tham số**) ✅ — đo lại MPS (rev.6):
  **0,703 s/bước** (G 12,07M + D 7,72M, crop 64) → 100k ≈ **19,5h**
- **Postnet mel** (F2): 5 lớp conv 256 kênh residual + loss cả mel trước/sau ✅ (small 6,14M → **7,38M**;
  base → ~17,7M; large → ~33,4M)
- **B3 VAD nới biên** trong `cut_clips` ✅ (mép khoảng lặng ≤200ms; cờ `tight`/`speech`)
- **B1 công cụ rà soát fallback** ✅ (`tools/fallback_review.py` + gói 400 từ + clip mẫu + merge overrides)
- **E1 watermark MPS** ✅ (0.8/0.7 trong runner khối + script) · **E2 empty_cache** giãn còn mỗi 500
  bước + sau validation ✅ · **E4 grad clip 5.0 + kiểm NaN/Inf mỗi bước (tự resume)** ✅

## 2. Ngân sách máy (đo thật)
| Hạng mục | Số |
| --- | --- |
| Acoustic small 7,38M (postnet) batch 16 | 0,478→~0,50 s/bước (ước tính +4% cho postnet) |
| Vocoder Vocos (G **12,07M** + D 7,72M, crop 64) | **0,703 s/bước MPS** (rev.6) → 100k ≈ **19,5h** |
| MPS vs CPU (discriminator lớn) | MPS nhanh hơn **3,7–4,7×** |
| Trần bộ nhớ | đặt watermark 0.8/0.7 + chunk 5k + bucket padding |

## 3. Dữ liệu (chốt cách xử lý)
| Vấn đề | Quyết định |
| --- | --- |
| Fallback 1,24% (3.217 từ, 29% clip) | Mở rộng overrides **top 400 từ ≥2 lần** (bỏ hapax) → parse lại; clip còn fallback **loại khỏi acoustic + MFA**, **giữ cho vocoder** |
| Clip dài 14s (p90 19s) | Cắt tại **SIL thật của MFA** (ưu tiên sau dấu câu), đoạn **3–10s**, không cắt giữa âm tiết |
| Biên clip | **B3**: nới tới mép khoảng lặng ≤200ms (đã code); đụng tiếng nói câu kề → giữ nguyên + cờ QC |
| Lọc rẻ trước MFA | Chỉ loại lỗi rõ (~2–3%): nhạc nền, clipping, tỉ lệ tiếng nói <50%; còn lại để score MFA lọc |
| Eval | Thêm **~100 câu thuần Việt tự viết** (đủ 6 thanh + coda tắc) làm bộ cố định đo CER/nghe thử |
| Vocoder | Train trên **`train.jsonl` 44,96h** (11.424 clip, **gồm cả clip fallback** — không cần text), **không gồm 4 video val** (rev.5 §1b) |

## 4. Huấn luyện (chốt)
| Hạng mục | Chốt |
| --- | --- |
| A1 | **Smoke small ~20k** với alignment thật; đạt ngưỡng mới lên base |
| A2 | Đo 200 bước cho **4/4, 4/5, 4/6** @ hidden 256; chọn cấu hình **s/bước ≤ ~1,0s** (đoán 4/5) |
| A3 | Tính theo **epoch, batch theo tổng frame ~29k** (rev.5): sau lọc còn ~33–35h → 1 epoch ≈ **530–560 bước**; small 60k ≈ 110 epoch; **base 100k ≈ 180 epoch + early stop**. Val audio-level (CER, F0 RMSE) **mỗi 10k**; val loss toàn tập **mỗi 5k** chỉ để phát hiện overfit |
| A4 | **Không chạy song song acoustic + vocoder** (tranh MPS/RAM). Lịch: **MFA (CPU) ∥ vocoder (MPS)** → xong mới train acoustic một mình. Mac chạy dài liên tục ổn; theo dõi s/bước tụt dần (powermetrics) thay vì nghỉ định kỳ |
| A5 | **Giữ hop 256** (đổi hop phải làm lại stats + vocoder) |
| F1 | Giữ **Conformer-lite** (depthwise conv rẻ, ONNX tốt); nghẽn O(T²) xử lý bằng cắt segment |
| F3 | Giữ tone aux, **trọng số 0,1**, mask SIL |
| F4 | Energy **scalar/âm tiết** |

## 5. Trả lời đã chốt (A–F)
- **B1 (c)**: mở rộng overrides top 400 + loại phần còn lại (acoustic/MFA); vocoder giữ hết.
- **C1**: MFA từ đầu ~**3–8h** (num_jobs 10); train trên **subset ~20h sạch nhất** rồi align toàn bộ;
  chạy thử 1h để ngoại suy.
- **C2**: bộ **121 phone đủ**; không tách IÊ/UÔ/ƯƠ; kiểm tra nhãn nào <50 lần.
- **C3**: Whisper chỉ chạy **~500 clip mẫu** để QC chéo, không lọc.
- **C4**: loại 10% thấp nhất theo log-likelihood/frame + vi phạm cứng (phone >500ms, âm tiết có tiếng
  >1s, sp >300ms không ở dấu câu, align lỗi); nếu loại >10% → kiểm lexicon, tăng beam
  `--beam 100 --retry_beam 400`, chỉ khi đó mới tính RAD-TTS.
- **D1**: **ch=16, 100k bước**, bỏ ch=32, không chia 2 pha. **D2**: crop 64, không curriculum.
- **D3**: generator **Vocos (ConvNeXt, hidden 384, 8 block)** — đã code, +4% s/bước.
- **D4**: **không** AMP/fp16.
- **E3**: bỏ qua `torch.compile`.

## 6. Lịch chạy (rev.4) — ⚠️ **ĐÃ THAY BẰNG §9.3** (giữ làm bản ghi)
1. **Cache feature cho xong** *(điều chỉnh: chạy sau recut B3+overrides để không phải làm 2 lần)*.
2. **Mở rộng overrides** (top 400, nghe clip mẫu) → **recut** (B3) → **parse lại** → **lọc rẻ** (B4).
3. **Song song: MFA (CPU) ∥ vocoder Vocos ch=16 100k (MPS, `train.jsonl` 44,96h)**.
4. **QC MFA** (loại 10% thấp nhất theo luật C4) → **SIL + log-F0 DCT + cắt segment theo SIL** →
   đo **s/bước 4/4, 4/5, 4/6** (chọn ≤1,0s).
5. **Smoke small ~20k** → đạt ngưỡng thì **base ~100k** (early stop theo val audio-level).
6. **GTA fine-tune vocoder** trên mel acoustic teacher-forcing.
> Lịch đang thực thi: **§9.3** (recut → stats 2.000 → cache → vocoder chạy trước; overrides/MFA/lọc
> là làn CPU song song; demo nghe thử ở mốc 5k/20k/50k bằng `tools/vocoder_milestones.py`).

## 7. Ngưỡng nghiệm thu
| Mốc | Điều kiện qua |
| --- | --- |
| Vocoder mới | Copy-synthesis nghe sạch; CER chênh ≤1 điểm so với audio gốc |
| MFA | >90% clip giữ; median lệch so Whisper <50ms; 50–100 câu nghe đạt |
| Smoke small 20k | Không tăng ngược sớm; F0 RMSE + CER tốt hơn rõ so với best 8k (so ở mức audio, cùng bộ 100 câu) |
| Base | CER của TTS ≤ copy-synthesis +2–3 điểm |

## 8. Việc cần người dùng làm
- **Nghe 400 clip mẫu** trong `data/codelung/review/` và điền `data/codelung/review/filled.txt`
  (mỗi dòng `từ|cách đọc`; 7 từ đã có gợi ý sẵn từ quy tắc `-ing`).
  Sau đó: `python tools/fallback_review.py --merge data/codelung/review/filled.txt`.
- Mẹo (rev.5): lỗi caption (chffix/endp/maxide) → sửa về **từ đúng** trong transcript, đừng respell
  lỗi chính tả; thuật ngữ Anh → ghi **cách bạn thực đọc trong video** (ví dụ `window|uyn đâu`);
  làm theo **thứ tự tần suất** — 150–200 từ đầu cứu phần lớn số giờ, phần còn lại có thể bỏ.

## 9. rev.5 — chốt trước khi khởi động vocoder + chỉnh nhỏ
### 9.1. Ba điểm chốt
1. **Đóng băng cấu hình mel** ✅ code: `canonical_mel_config` + `mel_config_hash` (gồm hash nội dung
   file stats) ghi vào **cả checkpoint acoustic và vocoder**; `synthesize` kiểm tra khi load, lệch →
   báo lỗi rõ (checkpoint cũ không có hash thì bỏ qua). **Stats tính lại trên 2.000 clip sau recut**
   rồi mới chốt; từ đó mọi thay đổi mel (n_fft/hop/f_max/clamp/stats) buộc train lại vocoder.
2. **Tách val khỏi vocoder** ✅: vocoder train trên **`train.jsonl` (44,96h — 11.424 clip, gồm cả clip
   fallback)** chứ không dùng val; CER copy-synthesis trên val mới phản ánh đúng.
3. **Audit Vocos** ✅: expansion kênh trung gian **3×** (đúng Vocos gốc, trước là 4×) → generator
   **12.069.762** tham số: net 8 block **7.123.968** · head (384→1.026, k7) **2.758.914** (23%) ·
   up 1.180.032 · post 737.664 · pre 269.184. Giữ head k7 (pha chính xác); lite cũ 3,53M vẫn còn.

### 9.2. Chỉnh nhỏ
- **E4 grad clip**: GAN **5.0** (`vocoder_training.grad_clip`) · acoustic giữ **1.0**
  (`training.grad_clip`) — đã ghi rõ trong config.
- **NaN**: kiểm **gộp mỗi 10 bước** (1 sync, không `.item()` từng bước) → raise để runner
  **resume + đổi seed** (`--seed`), **NaN lặp ≥2 lần thì hạ LR 0.5×** (`--lr-scale`) ✅ code.
- **Recut B3 xong** (13.215 clip · **51,73h**; **4.745 biên được nới** qua âm tiết bị cắt; split tái
  tạo **đúng 4 video val cũ**, không giao nhau) → **dựng lại corpus MFA sau recut** (việc bắt buộc).
- **Cắt segment theo SIL**: tính **mel + F0 trên clip đầy đủ rồi cắt theo frame** (tránh hiệu ứng biên
  STFT); mỗi segment bắt đầu/kết thúc bằng SIL, giữ đúng duration SIL.
- **A3 epoch (cập nhật)**: sau khi loại clip fallback + lọc MFA, acoustic còn ~**33–35h**; batch theo
  **tổng frame ~29k** (không theo số clip) → 1 epoch ≈ **530–560 bước**; 100k bước ≈ **180 epoch**;
  **early stop vẫn là chốt chặn**.

### 9.3. Lịch chạy (máy không chờ người) ✅ đang chạy
1. **Recut B3** ✅ → **mel stats 2.000 clip** → **cache** → **khởi động vocoder ngay**
   (MPS, `train.jsonl` 44,96h, Vocos + MPD/MRD, 100k bước).
2. Trong lúc vocoder chạy: người điền overrides → parse lại text (không cần recut) → lọc rẻ (B4) →
   **dựng lại corpus MFA** → MFA (CPU) ∥ vocoder (MPS).
3. QC MFA → SIL + log-F0 DCT + **cắt segment** (CPU, song song được với vocoder).
4. Đo s/bước **4/4, 4/5, 4/6**: **tạm dừng vocoder** (chunk resume) hoặc chờ xong — **không tranh MPS**.
5. Smoke small 20k → base 100k (early stop audio-level) → GTA fine-tune.

## 10. rev.6 — kiểm tra trước khi để máy chạy 20h
### 10.1. Hai lỗi kỹ thuật
1. **Checkpoint chỉ được ghi sau lần kiểm NaN đạt** ✅ code:
   - Vòng train gộp cờ `finite_ok`/`window_min` không sync; **tại mỗi bước bội số 10** mới `.item()`
     một lần. NaN trong cửa sổ → **raise trước mọi thao tác ghi**.
   - Ghi checkpoint **atomic** (`last.pt.tmp` → `last.pt`) và **xoay bản cũ sang `prev.pt`**;
     `save_every` tự làm tròn về **bội số của 10**.
   - State kèm `clean: true` + `last_loss` hữu hạn; runner `train_vocoder_chunked` chỉ resume từ
     checkpoint sạch (`last.pt` → `prev.pt`), có test.
2. **Cache key đã gồm hash file stats** ✅ xác minh: `feature_cache_key` chứa `mel_norm_hash`
   (sha1 **nội dung** `mel_stats.json`) + toàn bộ mục mel; có test `test_cache_key_sensitive_to_norm_stats`.
   Chuỗi hiện tại chạy **stats → cache** nên key nhất quán. (Không đổi sang "cache mel chưa chuẩn hoá"
   vì sẽ phải sửa 3 điểm gọi và làm lại cache giữa chừng — nếu sau này cần đổi stats, key tự vô hiệu.)

### 10.2. Vận hành
- **MFA ∥ vocoder**: dùng `--num_jobs 8` (không 10) + kiểm **s/bước vocoder không tụt quá ~15%**
  khi MFA chạy; tụt hơn thì giảm `--num_jobs` hoặc tạm dừng MFA.
- **Nghe thử sớm**: `tools/vocoder_milestones.py` (chạy nền, dùng **CPU**) xuất copy-synthesis
  ~10 câu val tại **5k/20k/50k** + metric mel L1/clipping; **20k còn rè kim loại/lỗi pha → dừng sửa**.
- **s/bước cập nhật (generator 12,07M)**: **0,703 s/bước** (MPS, b8 crop64) → 100k ≈ **19,5h**.
- **Clip được nới biên (4.745)**: theo dõi riêng tỷ lệ bị MFA loại (so với nhóm thường) khi có score;
  chưa xử lý riêng — MFA sẽ bắt qua sp dài/score thấp.

