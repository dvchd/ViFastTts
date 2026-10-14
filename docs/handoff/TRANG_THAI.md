# TRANG_THAI — trạng thái hiện hành (nguồn duy nhất, cập nhật mỗi vòng)

> Cập nhật: **vòng 08** · commit gốc `ab42b06` · người cập nhật: Agent 2 · tệp quyết định: `docs/handoff/V08_agent0.md`
> Vòng đang chạy: **V08 — A2 fixtail (chuẩn bị + probe lại)**. Trạng thái: **DỪNG chờ Agent 1 đọc diff**.

## 1. Mốc & hash ghim
| Đối tượng | Hash | Ghi chú |
|---|---|---|
| manifest train fixtail | `965b4798…` (full trong `eval/README.md`) | Σgap/row 294,9 · phủ pause 71,4% |
| manifest val fixtail | `cdbbd659…` | Σgap/row 303,0 · phủ 74,0% |
| manifest val CŨ | `19b6e117…` | dùng cho CỔNG CỨNG (không đổi) |
| holdout CŨ (CER) | `3ca7c79b…` | clip-level 842 hàng |
| mốc `eval/baselines/base40k_old.json` | cần ghim lại (B5) | val cũ |
| mốc `base40k_fixtail.json` | **chưa sinh** (B9) | thay `base40k_fixgap.json` |
| model thước `tone_ruler.pt` | không versioned | train lại: `tools/tone_ruler.py --train` |

## 2. Cổng A2 (tương đối so mốc cùng bản val)
- **CỨNG** (None = TRƯỢT): mel_l1_z(tf@cũ) ≤ mốc×1,01 · mel_db ≤ mốc+0,10 · rough(từ checkpoint@cũ) ≥ mốc−0,01 ·
  tone_C(@cũ) ≥ mốc−0,01 · CER(holdout cũ) ≤ mốc+0,03 · F0 NGÃ ≤ mốc+5¢ · syl_z(tf@fixtail) ≤ mốc×1,01 ·
  mel_gap(free@fixtail) ≤ mốc.
- **MỤC TIÊU**: pause_f1_all ≥ mốc+0,073 · gap_sum_mae_all ≤ mốc×0,714 · length_ratio(free) 1,00±0,05 ·
  gap_z(tf) ≤ mốc×0,85.
- Kết luận bench: TỪ CHỐI / ĐẠT CỨNG–TRƯỢT MỤC TIÊU / ĐẠT CỨNG–MỤC TIÊU CHƯA ĐỦ SỐ / CHẤP NHẬN.

## 3. Hằng số & quyết định đang hiệu lực
- **C_TAIL = 67 frame** (trần lặng đuôi, chỉ đoạn cuối clip; cắt cả nhãn lẫn `frame_range[1]`) — §44.8.
- `--reset-val-state` + `INIT_RESUME/INIT_RESET_VAL_STATE` (watchdog chỉ dùng khi `OUT/last.pt` chưa có).
- sh25 **giữ toàn cục** (không gated) — đóng mục 5c.
- Batch 8 + `SameNBatchSampler`; patience 0 cho cosine dài; `max_steps = lr_decay_steps = 60000`.

## 4. Việc đang chờ (theo mã vòng)
| Mã | Việc | Trạng thái |
|---|---|---|
| V08.B1–B4, B7, B8 | trần fixtail + manifest + watchdog/script + §44.8 + test | ✅ commit `64a47ff` |
| V08.B5 | bench kiểm sha256 mốc vs README + ghim lại `base40k_old.json` | ⏳ chờ |
| V08.B6 | `--validate-at-stop` cho train-v2 | ⏳ chờ |
| V08.B9 | sinh mốc `base40k_fixtail.json` (MPS) + kiểm H1 | ⏳ chờ |
| V08.B10 | probe lại trên fixtail | ⏳ chờ |
| V08.C1 | Agent 1 đọc diff `ab42b06..64a47ff` (Y1–Y6) | ⏳ chờ |
| V08.D1 | người dùng nghe 6 cặp AB sh25 | ✅ đã có kết luận (giữ sh25 toàn cục) |

## 5. Bẫy đã gặp (đừng lặp)
probe no-op (thiếu `--stop-at-step`) · lệnh thật train từ số 0 (thiếu INIT_RESUME) · sha256 ghim trước khi sửa tệp ·
JSON thô lọt lại vào `eval/baselines` · đổi bên đối chứng mà không báo trước · thước yếu che kết luận (§34.3→§36.4).
