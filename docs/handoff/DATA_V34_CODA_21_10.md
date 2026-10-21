# Đêm 18→19/10/2026 — bảng so sánh tự động

**Mốc so sánh**: `b8_filt` (b8 trên val/holdout lọc, cùng nhãn). Mọi số ở `summary_old` + `syl_z`/`mel_gap` (`summary_fixgap`) — cùng `val_filt`.

| Ứng viên | Trạng thái | mel_z | mel dB | rough | tone C | CER-20 | F0 NGÃ | dur MAE | vuv | Δjitter | Δhnr | syl_z | mel_gap | CER-100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **b8_filt (mốc)** | — | 1.1580 | 8.637 | 0.3423 | 0.7374 | 0.1977 | 198.1 | 4.742 | 0.9820 | +0.0294 | -3.10 | 0.6583 | 3.0696 | 0.2126 |
| v32 small (mốc) | 60000 · val 0.6011 | 1.4703 | 9.972 | 0.3402 | 0.8127 | — | 183.9 | 4.817 | 0.9850 | +0.0571 | -1.22 | 0.6850 | 3.1042 | 0.3694 |
| v34 coda/onset | 57000 · val 0.6001 | 1.4490 | 9.610 | 0.3362 | 0.7837 | — | 184.9 | 4.746 | 0.9848 | +0.0494 | -1.24 | 0.6834 | 3.0536 | 0.3178 |
| v34 last | 57000 · val 0.6001 | 1.4536 | 9.680 | 0.3365 | 0.8172 | — | 187.4 | 4.805 | 0.9848 | +0.0526 | -1.05 | 0.6863 | 3.0977 | 0.3178 |

## Cổng cứng (tương đối so mốc)

| Ứng viên | mel_l1_z | mel_db | rough | tone_C | cer | f0_nga | mel_gap | syl_z | mel_z_fix | mel_db_fix | đạt/trượt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v32 small (mốc) | 🚫 | 🚫 | 🚫 | 🚫 | ❌ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |
| v34 coda/onset | 🚫 | 🚫 | 🚫 | 🚫 | ❌ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |
| v34 last | 🚫 | 🚫 | 🚫 | 🚫 | ❌ | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 0/10 |

## Probe đuôi hiếm (`creak_rarity_probe`) — HNR/jitter theo nhóm

| Ứng viên | hiếm (≤5×) | phổ biến (≥150×) | ngang (mốc nền) |
|---|---|---|---|
| **b8 (mốc)** | +0.71 / 0.067 | +1.04 / 0.063 | +1.38 / 0.058 |

## Ghi chú

- Mốc/ứng viên đều chấm trên **cùng** `val_filt` + `holdout_filt` (mốc ghim `eval/baselines/b8_filt.json`).
- `CER-20` = 20 clip holdout (nhiễu ±0,03 — cần xem thêm CER-100 nếu cần chốt sát nút).
- `Δjitter`/`Δhnr` ngã: càng gần 0 càng tốt (mục tiêu → 0); mốc thật ngã: hnr +1,10 · jitter 0,063.
- Cột trạng thái = bước val cuối từ `history.csv` (val-trong-train chỉ 40 hàng — chỉ để tham khảo).
