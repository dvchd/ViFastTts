# TRANG_THAI — trạng thái hiện hành (nguồn duy nhất, Agent 2 cập nhật mỗi vòng)

**Vòng hiện hành**: V10 · **Tệp đặc tả**: `docs/handoff/V10_agent0.md`
**Commit gốc của vòng** (trước khi sửa): `06b5f10` · **Commit cuối**: `<commit hồ sơ B9>`
**Trạng thái**: **A2 ĐANG CHẠY, CHỜ ĐỌC** — A2 khởi động 12:35 14/10 (`runs/acoustic_v2_A2_fixtail`, resume base@40000,
60k, ~0,43 s/bước ⇒ tới 60k ≈ 2,3 giờ); đủ 6 điều kiện V10.A11; B10 (bench 45/50/55/60k) chờ A2 xong.

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
| mốc `eval/baselines/base40k_old.json` | `92e90dad…` | val cũ — CỔNG CỨNG (pin theo byte git) |
| mốc `base40k_fixtail.json` | `a8b74d2e…` | sinh lại bằng code mới (B7) + `length_ratio_speech/pause` |
| `base40k_fixgap.json` | `cf770864…` — **KHÔNG DÙNG** | đã gỡ khỏi git: `artifacts/bench/legacy/` (V10.A5) |
| model thước `tone_ruler.pt` | không versioned | train lại: `tools/tone_ruler.py --train` |

## 2. Cổng A2 (tương đối so mốc cùng bản val)
- **CỨNG** (None = TRƯỢT): mel_l1_z(tf@cũ) ≤ mốc×1,01 · mel_db ≤ mốc+0,10 · rough(từ checkpoint@cũ) ≥ mốc−0,01 ·
  tone_C(@cũ) ≥ mốc−0,01 · CER(holdout cũ) ≤ mốc+0,03 · F0 NGÃ ≤ mốc+5¢ · syl_z(tf@fixtail) ≤ mốc×1,01 ·
  mel_gap(free@fixtail) ≤ mốc.
- **MỤC TIÊU**: pause_f1_all ≥ mốc+0,073 · gap_sum_mae_all ≤ mốc×0,714 ·
  length_ratio(free) theo V10.A9: |1 − lr| ≤ 0,5×|1 − mốc| (≥0,8833 & ≤1,1168) · gap_z(tf) ≤ mốc×0,85.
- **THEO DÕI (V10.A9)**: length_ratio_speech (mốc 0,9343) · length_ratio_pause (mốc 0,5332) @ val fixtail free-run.
- Kết luận bench: TỪ CHỐI / ĐẠT CỨNG–TRƯỢT MỤC TIÊU / ĐẠT CỨNG–MỤC TIÊU CHƯA ĐỦ SỐ / CHẤP NHẬN.

## 3. Hằng số & quyết định đang hiệu lực
- **C_TAIL = 67 frame** (trần lặng đuôi, chỉ đoạn cuối clip; cắt cả nhãn lẫn `frame_range[1]`) — §44.8.
- `--reset-val-state` + `INIT_RESUME/INIT_RESET_VAL_STATE` (watchdog chỉ dùng khi `OUT/last.pt` chưa có).
- sh25 **giữ toàn cục** (không gated) — đóng mục 5c.
- Batch 8 + `SameNBatchSampler`; patience 0 cho cosine dài; `max_steps = lr_decay_steps = 60000`.
- **V10.A7 (đã quyết bằng đo)**: NHÁNH NHIỄU — f0p/f0/f0_delta **+0,000%** giữa 3 manifest; mọi thành phần probe
  trong ±2σ của 500 bước cuối base ⇒ probe fixtail **ĐẠT**; tiêu chí probe ±5% → **±2σ**.
- **V10.A8**: history.csv ghi từng thành phần loss + tỉ phần pause (train & validate).
- **V10.A11**: A2 thật chạy khi đủ 6 điều kiện (không cần chờ Agent 0).

## 4. Việc đang chờ (theo mã vòng)
| Mã | Việc | Trạng thái |
|---|---|---|
| V10.B1–B7 | đặc tả · pin byte git · loss vào history · B4 nhánh nhiễu · length_ratio A9 · test/push · sinh lại mốc (H1 +0,14%) | ✅ `baab50f` `e14d2dd` `168d1fb` `9b073c8` `d83230f` |
| V10.B8 | A2 thật — đủ 6/6 điều kiện, khởi động 12:35 | ✅ **ĐANG CHẠY** |
| V10.B9 | hồ sơ `V10_agent2.md` + TRANG_THAI | ✅ |
| V10.B10 | bench 45k/50k/55k/60k + tỉ số BCE pause val/train | ⏳ chờ A2 tới 60k |
| V10.A10 | SẮC-tắc làm lại (so SẮC-tắc với SẮC-khác cùng ô độ dài + val) | ⏳ sau A2 |

## 4b. A2 đang chạy (số kiểm đầu)
- `runs/acoustic_v2_A2_fixtail` · log `artifacts/dataset_v2/train_v2_A2_fixtail.log` (watchdog resume).
- Dòng đầu history: `# X3 resume: resumed_from=runs/acoustic_v2_base/last.pt@40000` ✅ · cột thành phần loss ✅ · không NaN.
- Bước 41000: val 8,8516 (mel hữu thanh 0,5972 · nét 0,355) · **val_pause_gate 0,934 vs train_pause_gate 0,124 (7,5×)** · val_pause_frac 12,0% vs train 2,6%.
- Nhịp: ~0,43 s/bước · mps 13–15G · validate_every 1000 · save_every 5000 (checkpoint 45/50/55/60k cho B10).

## 5. Bẫy đã gặp (đừng lặp)
probe no-op (thiếu `--stop-at-step`) · lệnh thật train từ số 0 (thiếu INIT_RESUME) · sha256 ghim trước khi sửa tệp ·
JSON thô lọt lại vào `eval/baselines` · đổi bên đối chứng mà không báo trước · thước yếu che kết luận (§34.3→§36.4) ·
**pin sha256 theo byte ĐĨA** (máy khác checkout autocrlf → lệch) → từ V10.A3 ghim theo byte trong git + `.gitattributes -text` ·
**tài liệu ghi "đã sửa" khi code chưa sửa** (V09.A3) → chỉ ghi khi kèm tên test xanh.
