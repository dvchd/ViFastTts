# Danh mục thước đo & quy trình đánh giá (v2) — `docs/EVAL_TOOLS.md`

> Một chỗ liệt kê **mọi thước** đang dùng, cách chạy, ý nghĩa, cổng, và **bẫy thống kê**.
> Số liệu gốc: `docs/ACOUSTIC_V2.md` §30–§39. Cập nhật: 14/10/2026.

## 0) Nguyên tắc vàng (đã trả giá 2 lần)

1. **Không nâng cấp gì trước khi có thước đo** — và **phải kiểm thước trên dữ liệu THẬT trước khi
   dùng nó để quyết định**. Thước cũ (`s_k`→prior gần nhất) chỉ đạt **0,406 trên audio thật** và làm
   ta kết luận sai hướng (§34.3 → đính chính §33.5/§36.4). Thước mới `tone_ruler` = **0,762** trên audio thật.
2. **Mỗi thước có trần và điểm mù** — ghi rõ khi báo cáo (vd `tone_ruler` **bỏ ~14%** âm tiết ngắn/
   kết thúc tắc = đúng nhóm sắc/nặng khó nhất).
3. **`mel L1` trung bình ≠ tai người** (§37): nó thưởng mel "điển hình/mượt"; tai thưởng **độ nét**.
   Chọn checkpoint cuối = **L1 + độ nét + thanh + rè + CER**, chốt bằng A/B tai người.

## 1) Thước *trong* vòng train (`train_v2.py::_validate`, ghi `history.csv`)

| Cột | Là gì | Ghi chú |
|---|---|---|
| `val_mel_voiced` | mel L1 **hữu thanh** (frame không thuộc khoảng nghỉ G) | dùng để chọn `best.pt` — **chỉ là "neo an toàn"** |
| `val_total` | loss tổng (đã nhân trọng số) | theo dõi sức khoẻ chung |
| `val_rough` | **\|Δmel_model\| / \|Δmel_thật\|** (thời gian + tần số) | **độ nét**; 1,0 = nét bằng thật; base@18k 0,395 → @40k 0,419 |
| `train_mel`, `mel_gap` | L1 train · khoảng cách train−val | phát hiện overfit macro |
| `lr` | LR hiện tại | nhớ: LR phụ thuộc `--max-steps` (cosine) |

**Bẫy**: SE(mel hữu thanh) = **±0,0126 với 40 clip** nhưng **±0,0046 với 300 clip** (§30.1). Chênh
nhỏ hơn 2×SE ⇒ **trong nhiễu**. `no_improve`/`early_stop_patience`: **tắt (0)** cho lịch cosine dài —
val chững ở 45% schedule (LR còn cao) **không phải** hội tụ (§32.1).

## 2) Thước *ngoài* train (offline)

| # | Tool | Đo gì | Lệnh (mẫu) | Tiền tố tốn ~ | Test? |
|---|---|---|---|---|---|
| 1 | `tools/eval_acoustic.py --mode tf_syl` | mel L1 z/raw · mel dB hữu thanh · vuv · **F0 RMSE theo 6 thanh** · energy · dur MAE | `--config <cfg> --checkpoint <ck> --manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl --cache-dir cache/codelung_v2 --clips 300 --device mps --out <json>` | 3–6 phút (MPS) | ✅ |
| 2 | `tools/eval_acoustic.py --mode free` | **CER (Whisper)** free-run trên holdout | thêm `--manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl --clips 20` | 2–4 phút | ✅ |
| 3 | `tools/eval_creak.py` | **"rè"**: pool `hnr/glottal/jitter` trên nucleus, so **THẬT vs synth** theo thanh | `--config <run>/resolved_config.json --checkpoint <ck> --clips 25 --out <json>` | 3–5 phút (CPU) | ⚠️ chưa |
| 4 | `tools/tone_ruler.py --score-real` | accuracy thanh trên **audio thật** (trần thước) | `--train` (4 phút) · `--score-real` · `--manifest … --out …` | 1–4 phút (CPU) | ✅ |
| 5 | `tools/eval_tone_paths.py` (A/B/C) | accuracy thanh **A** audio thật → **B** mel thật→Vocos → **C** free-run | `--config <run>/resolved_config.json --checkpoint <ck> --clips 60 --out <json>` | ~1 phút (CPU) | ⚠️ chưa |
| 6 | `tools/eval_vocoder_fidelity.py` | fidelity mel thật→Vocos (recall phổ) | xem docstring | 2–5 phút | ✅ |
| 7 | `tools/voice_quality.py` | cổng R8-1: HNR/tilt/glottal/perturb (creak) | `--manifest … --synth-dir … --output …` | 1–3 phút | ✅ |
| 8 | `tools/tone_path_diagnosis.py` | chẩn đoán 4 đường thanh (bằng thước cũ — **legacy**) | `--clips 60 --out …` | 1 phút | ✅ |
| 9 | `tools/eval_tone_pairs.py` | minimal-pair thanh + trang A/B mù (**legacy**, thước yếu) | xem docstring | 2–5 phút | ✅ |
| 10 | `tools/clip_quality.py`, `align_quality_report.py` | QC dữ liệu (chất lượng clip, alignment) | xem docstring | — | ✅ |

## 3) Bộ số mốc để so (khi chấm checkpoint mới)

| Thước | B0 | small b8 (46k) | **base 40k** | "tốt hơn" nếu |
|---|---|---|---|---|
| mel L1 z (300 clip) | 1,2758 | 1,1603 | **1,1411** | ↓ ≥ 0,005 (vượt nhiễu) |
| mel dB hữu thanh | 9,79 | 8,64 | **8,51** | ↓ |
| vuv (nucleus) | 0,9918 (–) | 0,9823 / 0,9925 | 0,9823 / 0,9925 | giữ ≥ 0,98 |
| dur MAE (frame) | 4,79 | **4,75** | 4,84 | ↓ |
| **độ nét** (rough) | — | 0,393 | **0,419** (18k: 0,395) | **↑** (mục tiêu ≥ 0,6) |
| CER free (20 holdout) | 0,939 | **0,1977** | 0,2168 | ↓ (nhiễu ±0,03) |
| thanh A/B/C (60 row) | – | C 0,738 (A 0,797) | **C 0,798–0,805** (A 0,799–0,801) | C→A (đã đạt) |
| **rè** Δhnr ngã (thật→synth) | – | – | **−2,75** | **→ 0** (hiện đang xấu) |
| **rè** Δjitter ngã | – | – | **+0,037** | **→ 0** |

## 4) Quy trình "chấm một checkpoint" (đủ bộ, ~15–25 phút)

```bash
CK=runs/<run>/last.pt; CFG=runs/<run>/resolved_config.json
# 1) mel + F0 theo thanh + độ dài (MPS)
.venv/bin/python tools/eval_acoustic.py --config $CFG --checkpoint $CK --mode tf_syl \
  --manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl --cache-dir cache/codelung_v2 \
  --clips 300 --device mps --out artifacts/eval_acoustic/<run>_tf_syl_300.json
# 2) CER free-run trên holdout (MPS)
.venv/bin/python tools/eval_acoustic.py --config $CFG --checkpoint $CK --mode free \
  --manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl --cache-dir cache/codelung_v2 \
  --clips 20 --device mps --out artifacts/eval_acoustic/<run>_free_holdout20.json
# 3) thanh A/B/C (CPU)
.venv/bin/python tools/eval_tone_paths.py --config $CFG --checkpoint $CK --clips 60 \
  --out artifacts/tone_path/tone_paths_<run>.json
# 4) rè ngã/nặng (CPU)
.venv/bin/python tools/eval_creak.py --config $CFG --checkpoint $CK --clips 25 \
  --out artifacts/eval_acoustic/creak_<run>.json
# 5) demo tai người (3 câu cố định, cùng Vocos) → release asset (test tải trước khi báo)
```
**Cổng chấp nhận một run mới**: mel_z ≤ mốc cũ · `rough` **không giảm** · C ≥ 0,78 · Δcreak **không phình**
· CER ≤ 0,22 · (tuỳ chọn) tai người A/B thắng/hoà.

## 5) Bẫy thống kê & điểm mù (đọc trước khi kết luận)

- **Nhiễu val**: 40 clip → SE ±0,0126; 300 clip → ±0,0046. Chênh < 2×SE = nhiễu.
- **60 row đầu dễ hơn trung bình** (A = 0,7955 vs 0,7623 toàn val) ⇒ đừng lấy 0,80 làm "trần".
- **`tone_ruler` mù ~14%**: âm tiết nucleus < 4 frame (kết thúc tắc = sắc/nặng khó nhất) bị bỏ.
- **A/B/C phải so `paired`** (cùng âm tiết) — C free-run bỏ ít hơn (2,8% vs 14,4%) nên tập chấm khác nhau.
- **CER 20 clip** nhiễu ±0,03 ⇒ tăng lên 50–100 clip khi cần quyết định sát nút.
- **Thước cũ (`eval_tone_pairs`, `tone_path_diagnosis`)**: trần 0,46 — chỉ dùng để so với quá khứ.
- **`hnr/jitter` là proxy** (harvest + pyworld) — dùng so *tương đối* giữa các nguồn, không phải số tuyệt đối.
- **Mọi demo phải test tải** (`curl` HTTP 200 + đúng số byte) trước khi báo người nghe.
