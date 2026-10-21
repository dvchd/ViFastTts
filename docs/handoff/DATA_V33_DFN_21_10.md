# Đêm 18→19/10/2026 — bảng so sánh tự động

**Mốc so sánh**: `b8_filt` (b8 trên val/holdout lọc, cùng nhãn). Mọi số ở `summary_old` + `syl_z`/`mel_gap` (`summary_fixgap`) — cùng `val_filt`.

| Ứng viên | Trạng thái | mel_z | mel dB | rough | tone C | CER-20 | F0 NGÃ | dur MAE | vuv | Δjitter | Δhnr | syl_z | mel_gap | CER-100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **b8_filt (mốc)** | — | 1.1580 | 8.637 | 0.3423 | 0.7374 | 0.1977 | 198.1 | 4.742 | 0.9820 | +0.0294 | -3.10 | 0.6583 | 3.0696 | 0.2126 |
| b8 trên val v33 (DFN) | 60000 · val 0.6314 | 1.5387 | 11.830 | 0.3271 | 0.7285 | — | 201.2 | 4.891 | 0.9853 | +0.0276 | -1.59 | 0.7555 | 3.1073 | — |
| v33 best | 60000 · val 0.5916 | 1.3110 | 10.250 | 0.3771 | 0.8051 | — | 181.3 | 4.834 | 0.9853 | +0.0462 | -1.94 | 0.7038 | 2.7930 | 0.4435 |
| v33 last | 60000 · val 0.5916 | 1.3084 | 10.224 | 0.3750 | 0.8059 | — | 182.6 | 4.842 | 0.9853 | +0.0387 | -1.97 | 0.7029 | 2.7881 | 0.4435 |

## Cổng cứng (tương đối so mốc)

| Ứng viên | mel_l1_z | mel_db | rough | tone_C | cer | f0_nga | mel_gap | syl_z | mel_z_fix | mel_db_fix | đạt/trượt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b8 trên val v33 (DFN) | 🚫 | 🚫 | 🚫 | 🚫 | ❌ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |
| v33 best | 🚫 | 🚫 | 🚫 | 🚫 | ❌ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |
| v33 last | 🚫 | 🚫 | 🚫 | 🚫 | ❌ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |

## Probe đuôi hiếm (`creak_rarity_probe`) — HNR/jitter theo nhóm

| Ứng viên | hiếm (≤5×) | phổ biến (≥150×) | ngang (mốc nền) |
|---|---|---|---|
| **b8 (mốc)** | +0.71 / 0.067 | +1.04 / 0.063 | +1.38 / 0.058 |

## Ghi chú

- Mốc/ứng viên đều chấm trên **cùng** `val_filt` + `holdout_filt` (mốc ghim `eval/baselines/b8_filt.json`).
- `CER-20` = 20 clip holdout (nhiễu ±0,03 — cần xem thêm CER-100 nếu cần chốt sát nút).
- `Δjitter`/`Δhnr` ngã: càng gần 0 càng tốt (mục tiêu → 0); mốc thật ngã: hnr +1,10 · jitter 0,063.
- Cột trạng thái = bước val cuối từ `history.csv` (val-trong-train chỉ 40 hàng — chỉ để tham khảo).
