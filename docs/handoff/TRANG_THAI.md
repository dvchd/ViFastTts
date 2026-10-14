# TRANG_THAI — trạng thái hiện hành (nguồn duy nhất, Agent 2 cập nhật mỗi vòng)

**Vòng hiện hành**: V10 · **Tệp đặc tả**: `docs/handoff/V10_agent0.md`
**Commit gốc của vòng** (trước khi sửa): `06b5f10` · **Commit cuối**: `<cập nhật khi dừng>
**Trạng thái**: **ĐANG LÀM** (V10.B1–B9; B10 sau khi A2 tới 60k)

> Agent 1 đọc vòng này theo thông tin ở đầu tệp (không cần hash gửi kèm): diff `06b5f10..<commit cuối>`,
> câu hỏi V10.C1–C7 ở mục C của tệp đặc tả. Agent 1 **chỉ đọc, không ghi tệp** vào repo.
> Từ V10.A1: commit ĐẦU TIÊN của mỗi vòng là tệp đặc tả nguyên văn `docs/handoff/V<nn>_agent0.md`.

## 1. Mốc & hash ghim (V10.A3: ghim sha256 theo byte TRONG GIT — `git show HEAD:<tệp> | shasum -a 256`)
| Đối tượng | Hash | Ghi chú |
|---|---|---|
| manifest train fixtail | `965b4798…` (full trong `eval/README.md`) | Σgap/row 294,9 · phủ pause 71,4% |
| manifest val fixtail | `cdbbd659…` | Σgap/row 303,0 · phủ 74,0% |
| manifest val CŨ | `19b6e117…` | dùng cho CỔNG CỨNG (không đổi) |
| holdout CŨ (CER) | `3ca7c79b…` | clip-level 842 hàng |
| mốc `eval/baselines/base40k_old.json` | `3d6fefe5…` | val cũ — CỔNG CỨNG |
| mốc `base40k_fixtail.json` | `7be6fcff…` (sẽ sinh lại ở B7) | thay `base40k_fixgap.json` |
| `base40k_fixgap.json` | `cf770864…` — **KHÔNG DÙNG** | chuyển `artifacts/bench/legacy/` (V10.A5) |
| model thước `tone_ruler.pt` | không versioned | train lại: `tools/tone_ruler.py --train` |

## 2. Cổng A2 (tương đối so mốc cùng bản val)
- **CỨNG** (None = TRƯỢT): mel_l1_z(tf@cũ) ≤ mốc×1,01 · mel_db ≤ mốc+0,10 · rough(từ checkpoint@cũ) ≥ mốc−0,01 ·
  tone_C(@cũ) ≥ mốc−0,01 · CER(holdout cũ) ≤ mốc+0,03 · F0 NGÃ ≤ mốc+5¢ · syl_z(tf@fixtail) ≤ mốc×1,01 ·
  mel_gap(free@fixtail) ≤ mốc.
- **MỤC TIÊU**: pause_f1_all ≥ mốc+0,073 · gap_sum_mae_all ≤ mốc×0,714 ·
  length_ratio(free) theo V10.A9: |1 − lr| ≤ 0,5×|1 − mốc| (≥0,8833 & ≤1,1168) · gap_z(tf) ≤ mốc×0,85.
- **THEO DÕI (V10.A9)**: length_ratio_speech · length_ratio_pause @ val fixtail free-run.
- Kết luận bench: TỪ CHỐI / ĐẠT CỨNG–TRƯỢT MỤC TIÊU / ĐẠT CỨNG–MỤC TIÊU CHƯA ĐỦ SỐ / CHẤP NHẬN.

## 3. Hằng số & quyết định đang hiệu lực
- **C_TAIL = 67 frame** (trần lặng đuôi, chỉ đoạn cuối clip; cắt cả nhãn lẫn `frame_range[1]`) — §44.8.
- `--reset-val-state` + `INIT_RESUME/INIT_RESET_VAL_STATE` (watchdog chỉ dùng khi `OUT/last.pt` chưa có).
- sh25 **giữ toàn cục** (không gated) — đóng mục 5c.
- Batch 8 + `SameNBatchSampler`; patience 0 cho cosine dài; `max_steps = lr_decay_steps = 60000`.
- **V10.A7**: tiêu chí probe từng thành phần loss ±5% → xét lại theo kết quả đo tất định (B4).
- **V10.A8**: history.csv ghi từng thành phần loss + tỉ phần pause (train & validate).
- **V10.A11**: A2 thật chạy khi đủ 6 điều kiện (không cần chờ Agent 0).

## 4. Việc đang chờ (theo mã vòng)
| Mã | Việc | Trạng thái |
|---|---|---|
| V10.B1 | commit đặc tả V10 + V09 vào repo | 🔄 |
| V10.B2 | pin theo git byte + bảng README + gỡ fixgap + xóa khóa null | ⏳ |
| V10.B3 | thành phần loss vào history + test | ⏳ |
| V10.B4 | đo f0p tất định 3 manifest → nhánh | ⏳ |
| V10.B5 | length_ratio_speech/pause + cổng mới | ⏳ |
| V10.B6 | test + ruff + push | ⏳ |
| V10.B7 | sinh lại mốc fixtail + ghim | ⏳ |
| V10.B8 | khởi động A2 thật (đủ 6 điều kiện) | ⏳ |
| V10.B9 | hồ sơ + TRANG_THAI | ⏳ |
| V10.B10 | bench 45k/50k/55k/60k (sau khi A2 xong) | ⏳ |

## 5. Bẫy đã gặp (đừng lặp)
probe no-op (thiếu `--stop-at-step`) · lệnh thật train từ số 0 (thiếu INIT_RESUME) · sha256 ghim trước khi sửa tệp ·
JSON thô lọt lại vào `eval/baselines` · đổi bên đối chứng mà không báo trước · thước yếu che kết luận (§34.3→§36.4) ·
**pin sha256 theo byte ĐĨA** (máy khác checkout autocrlf → lệch) → từ V10.A3 ghim theo byte trong git + `.gitattributes -text` ·
**tài liệu ghi "đã sửa" khi code chưa sửa** (V09.A3) → chỉ ghi khi kèm tên test xanh.
