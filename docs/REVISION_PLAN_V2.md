# REVISION_PLAN_V2 (rev.4) — Sửa nền tảng trước khi lên `base`

> Nguồn: phản hồi chuyên gia (4 vòng) + xác minh trong code. Cập nhật: 12/10/2026.
> **rev.4** chốt toàn bộ câu hỏi A–F (xem §5), thêm 3 điểm quan trọng: **vocoder không cần text**
> (train trên đủ 51h, chạy song song MFA), **cắt clip theo SIL** (giảm nghẽn attention decoder),
> **giữ hop 256**.

## 1. Trạng thái code (đã có)
- **A1** mel loss mean/bin ✅ · **A2.4** conditioning GT f0/vuv/energy ✅ (sẽ bật khi retrain)
- **Mel mới** log-magnitude clamp 1e-5 + chuẩn hoá toàn cục per-bin ✅ (`tools/compute_mel_stats.py`)
- **F0 pyworld** harvest+stonemask ✅ (đo ~21× realtime/lõi)
- **Vocoder nâng cấp**: MPD đầy đủ 2/3/5/7/11 + MRD + MR-STFT ✅; **generator Vocos** (ConvNeXt
  hidden 384 × 8 block, 14,4M) ✅ — đo MPS: **0,659 s/bước** (lite 0,633; +4%, trong ngưỡng ≤20%)
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
| Vocoder Vocos (G 14,4M + D 7,7M, crop 64) | **0,659 s/bước MPS** → 100k ≈ 19–20h |
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
| Vocoder | Train trên **toàn bộ 51h** (kể cả clip fallback) — không cần text |

## 4. Huấn luyện (chốt)
| Hạng mục | Chốt |
| --- | --- |
| A1 | **Smoke small ~20k** với alignment thật; đạt ngưỡng mới lên base |
| A2 | Đo 200 bước cho **4/4, 4/5, 4/6** @ hidden 256; chọn cấu hình **s/bước ≤ ~1,0s** (đoán 4/5) |
| A3 | Tính theo **epoch** (1 epoch ≈ 714 bước @ batch 16): small 60k ≈ 84 epoch; **base 100k ≈ 140 epoch + early stop**. Val audio-level (CER, F0 RMSE) **mỗi 10k**; val loss toàn tập **mỗi 5k** chỉ để phát hiện overfit |
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

## 6. Lịch chạy (rev.4)
1. **Cache feature cho xong** *(điều chỉnh: chạy sau recut B3+overrides để không phải làm 2 lần)*.
2. **Mở rộng overrides** (top 400, nghe clip mẫu) → **recut** (B3) → **parse lại** → **lọc rẻ** (B4).
3. **Song song: MFA (CPU) ∥ vocoder Vocos ch=16 100k (MPS trên 51h)**.
4. **QC MFA** (loại 10% thấp nhất theo luật C4) → **SIL + log-F0 DCT + cắt segment theo SIL** →
   đo **s/bước 4/4, 4/5, 4/6** (chọn ≤1,0s).
5. **Smoke small ~20k** → đạt ngưỡng thì **base ~100k** (early stop theo val audio-level).
6. **GTA fine-tune vocoder** trên mel acoustic teacher-forcing.

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
