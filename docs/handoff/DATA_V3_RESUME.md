# Đêm 18→19/10/2026 — bảng so sánh tự động

**Mốc so sánh**: `b8_filt` (b8 trên val/holdout lọc, cùng nhãn). Mọi số ở `summary_old` + `syl_z`/`mel_gap` (`summary_fixgap`) — cùng `val_filt`.

| Ứng viên | Trạng thái | mel_z | mel dB | rough | tone C | CER-20 | F0 NGÃ | dur MAE | vuv | Δjitter | Δhnr | syl_z | mel_gap | CER-100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **b8_filt (mốc)** | — | 1.1580 | 8.637 | 0.3423 | 0.7374 | 0.1977 | 198.1 | 4.742 | 0.9820 | +0.0294 | -3.10 | 0.6583 | 3.0696 | 0.2126 |
| b8 trên val v3 (mốc cùng tập) | 60000 · val 0.6314 | 1.9422 | 16.856 | 0.3373 | 0.7344 | — | 211.4 | 4.742 | 0.8490 | +0.0235 | -1.91 | 1.8692 | 2.7737 | — |
| v3r best | 60000 · val 2.1008 | 1.7802 | 15.349 | 0.1851 | 0.2297 | 0.7693 | 200.7 | 4.919 | 0.8490 | +0.0080 | -2.84 | 1.6483 | 2.4083 | 0.5470 |
| v3r last (60k) | 60000 · val 2.1008 | 1.9263 | 17.143 | 0.2315 | 0.5110 | 0.4642 | 194.4 | 5.045 | 0.8227 | +0.0358 | -2.41 | 1.8243 | 2.6028 | 0.5470 |
| v3r trên val_filt (cũ) | 60000 · val 2.1008 | 1.4836 | 12.445 | 0.1842 | 0.2116 | 0.7411 | 206.3 | 4.936 | 0.9809 | -0.0090 | -3.46 | 1.0999 | 2.8060 | 0.5470 |
| v3 cũ best@9k (tham chiếu) | 60000 · val 2.1008 | 1.7802 | 15.349 | 0.1851 | 0.2297 | — | 200.7 | 4.919 | 0.8490 | +0.0080 | -2.84 | 1.6483 | 2.4083 | — |

## Cổng cứng (tương đối so mốc)

| Ứng viên | mel_l1_z | mel_db | rough | tone_C | cer | f0_nga | mel_gap | syl_z | mel_z_fix | mel_db_fix | đạt/trượt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b8 trên val v3 (mốc cùng tập) | 🚫 | 🚫 | 🚫 | 🚫 | ❌ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |
| v3r best | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |
| v3r last (60k) | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |
| v3r trên val_filt (cũ) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 1/10 |
| v3 cũ best@9k (tham chiếu) | 🚫 | 🚫 | 🚫 | 🚫 | ❌ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |

## Probe đuôi hiếm (`creak_rarity_probe`) — HNR/jitter theo nhóm

| Ứng viên | hiếm (≤5×) | phổ biến (≥150×) | ngang (mốc nền) |
|---|---|---|---|
| v3r best | -1.43 / 0.100 | -1.24 / 0.093 | -0.92 / 0.086 |
| v3r last (60k) | -1.43 / 0.100 | -1.24 / 0.093 | -0.92 / 0.086 |
| v3r trên val_filt (cũ) | -1.43 / 0.100 | -1.24 / 0.093 | -0.92 / 0.086 |
| v3 cũ best@9k (tham chiếu) | -1.38 / 0.089 | -1.30 / 0.098 | -0.80 / 0.094 |
| **b8 (mốc)** | +0.71 / 0.067 | +1.04 / 0.063 | +1.38 / 0.058 |

## Ghi chú

- Mốc/ứng viên đều chấm trên **cùng** `val_filt` + `holdout_filt` (mốc ghim `eval/baselines/b8_filt.json`).
- `CER-20` = 20 clip holdout (nhiễu ±0,03 — cần xem thêm CER-100 nếu cần chốt sát nút).
- `Δjitter`/`Δhnr` ngã: càng gần 0 càng tốt (mục tiêu → 0); mốc thật ngã: hnr +1,10 · jitter 0,063.
- Cột trạng thái = bước val cuối từ `history.csv` (val-trong-train chỉ 40 hàng — chỉ để tham khảo).
