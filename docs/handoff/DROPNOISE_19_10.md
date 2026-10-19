# Đêm 18→19/10/2026 — bảng so sánh tự động

**Mốc so sánh**: `b8_filt` (b8 trên val/holdout lọc, cùng nhãn). Mọi số ở `summary_old` + `syl_z`/`mel_gap` (`summary_fixgap`) — cùng `val_filt`.

| Ứng viên | Trạng thái | mel_z | mel dB | rough | tone C | CER-20 | F0 NGÃ | dur MAE | vuv | Δjitter | Δhnr | syl_z | mel_gap | CER-100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **b8_filt (mốc)** | — | 1.1580 | 8.637 | 0.3423 | 0.7374 | 0.1977 | 198.1 | 4.742 | 0.9820 | +0.0294 | -3.10 | 0.6583 | 3.0696 | 0.2126 |
| loại video nhiễu best | 70000 · val 0.6243 | 1.1520 | 8.598 | 0.3524 | 0.7504 | 0.2021 | 191.1 | 4.718 | 0.9820 | +0.0431 | -3.03 | 0.6649 | 3.0488 | 0.2382 |
| loại video nhiễu last | 70000 · val 0.6243 | 1.1675 | 9.001 | 0.3449 | 0.7572 | 0.2143 | 188.8 | 4.859 | 0.9820 | +0.0364 | -2.76 | 0.6551 | 3.1044 | 0.2382 |
| mel_sharp best (tham chiếu) | 70000 · val 0.6308 | 1.1494 | 8.600 | 0.3420 | 0.7439 | 0.2087 | 189.8 | 4.796 | 0.9820 | +0.0452 | -2.79 | 0.6566 | 3.0480 | 0.2465 |
| oversample (tham chiếu) | 70000 · val 0.6517 | 1.1801 | 9.073 | 0.3432 | 0.7474 | 0.2152 | 193.4 | 4.966 | 0.9819 | +0.0459 | -2.87 | 0.6680 | 3.1143 | 0.2862 |

## Cổng cứng (tương đối so mốc)

| Ứng viên | mel_l1_z | mel_db | rough | tone_C | cer | f0_nga | mel_gap | syl_z | mel_z_fix | mel_db_fix | đạt/trượt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loại video nhiễu best | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 9/10 |
| loại video nhiễu last | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | 7/10 |
| mel_sharp best (tham chiếu) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 10/10 |
| oversample (tham chiếu) | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | 4/10 |

## Probe đuôi hiếm (`creak_rarity_probe`) — HNR/jitter theo nhóm

| Ứng viên | hiếm (≤5×) | phổ biến (≥150×) | ngang (mốc nền) |
|---|---|---|---|
| loại video nhiễu best | +0.61 / 0.067 | +0.38 / 0.074 | +0.96 / 0.061 |
| loại video nhiễu last | +0.61 / 0.067 | +0.38 / 0.074 | +0.96 / 0.061 |
| mel_sharp best (tham chiếu) | +0.95 / 0.072 | +1.14 / 0.065 | +1.54 / 0.058 |
| oversample (tham chiếu) | +1.26 / 0.062 | +1.25 / 0.065 | +1.75 / 0.059 |
| **b8 (mốc)** | +0.71 / 0.067 | +1.04 / 0.063 | +1.38 / 0.058 |

## Ghi chú

- Mốc/ứng viên đều chấm trên **cùng** `val_filt` + `holdout_filt` (mốc ghim `eval/baselines/b8_filt.json`).
- `CER-20` = 20 clip holdout (nhiễu ±0,03 — cần xem thêm CER-100 nếu cần chốt sát nút).
- `Δjitter`/`Δhnr` ngã: càng gần 0 càng tốt (mục tiêu → 0); mốc thật ngã: hnr +1,10 · jitter 0,063.
- Cột trạng thái = bước val cuối từ `history.csv` (val-trong-train chỉ 40 hàng — chỉ để tham khảo).
