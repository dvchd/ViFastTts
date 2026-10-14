# VÒNG 08 | Báo cáo Agent 2 | Commit: `64a47ff` (code) · `cfe51c9` (handoff) · `ab42b06` (gốc)

**Test**: 2.469 pass · ruff sạch · **Trạng thái: DỪNG** (chờ Agent 1 đọc `ab42b06..64a47ff`)

## Bảng số theo mã câu hỏi
| Mã | Việc | Kết quả |
|---|---|---|
| V08.B1 | `--tail-cap-frames` | ✅ chỉ đoạn cuối clip, cắt cả `frame_range[1]`; 3 test (không cờ = byte cũ · gap 300→67 & T −233 & bất biến đúng · chỗ cắt không đổi) |
| V08.B2 | manifest `*_fixtail` | ✅ train 9.763 · Σgap/row **294,9** (fixgap 409,8) · phủ pause **71,4%** · T max **1.293** · bất biến 100%; val 1.773 · 303,0 · 74,0% · 1.216. **So theo id vs fixgap: 0 lệch** (id/syllables/durations/seg_durations/frame_range[0]/gaps-khác-cuối); `T khác mà không phải đoạn cuối clip = 0`; T giảm: train 4.719 đoạn/1.121.705 frame (max 1.003) · val 819/182.484 (max 726). Hash: train `965b4798…` · val `cdbbd659…` |
| V08.B3 | p → pos_weight | ✅ p không đổi **0,3082** ⇒ pos_weight **2,244** |
| V08.B4 | watchdog + script | ✅ `INIT_RESUME`/`INIT_RESET_VAL_STATE` chỉ khi `OUT/last.pt` chưa có; dry-run in đúng lệnh `--resume runs/acoustic_v2_base/last.pt --reset-val-state --max-steps 60000` |
| V08.B7 | §44.8 | ✅ C_TAIL=67 + nguồn gốc + lý do cắt target + sh25 toàn cục + fixtail thay fixgap |
| V08.B8 | test + push | ✅ |
| V08.B5 | bench kiểm sha256 mốc vs README (lệch ⇒ dừng) + test | ✅ `c3fc66c` |
| V08.B6 | `--validate-at-stop` report-only (không chạm best_val/no_improve) + test | ✅ `c3fc66c` |
| V08.B9 | mốc `base40k_fixtail.json` + kiểm H1 | ✅ **H1 ĐẠT**: syl_z cũ 0,6612 → fixtail **0,6621 (+0,14%)** ≤3% |
| V08.B10 | probe lại fixtail | ✅ cơ chế ĐẠT; **1 điểm lạ** (xem dưới) |

## V08.B9 — số mốc `base40k_fixtail.json` (sha256 `7be6fcff…`)
| Cổng | Mốc fixtail | Ngưỡng tuyệt đối cho A2 |
|---|---|---|
| syl_z (tf, hard) | **0,6621** | ≤ **0,6687** |
| mel_gap (free, hard) | **3,1272** | ≤ 3,1272 |
| gap_z (tf, mục tiêu) | **2,1529** | ≤ **1,830** (×0,85) |
| pause_f1_all | 0,5559 | ≥ **0,6289** (+0,073) |
| gap_sum_mae_all | **156,46** (fixgap 245,15) | ≤ **111,7** (×0,714) |
| length_ratio | 0,7665 (fixgap 0,6481) | 1,00 ± 0,05 |
| (tham khảo) mel_z tổng @fixtail | 1,3311 (fixgap 1,6218) | – |

## V08.B10 — probe fixtail: cơ chế ĐẠT, một thành phần "lạ"
- LR 6,03e-5 @40050 / 6,00e-5 @40100 (= 0,3018×lr_max, lệch <1%) ✅ · không NaN/OOM · RAM đỉnh **9,5 GB** · **0,379–0,388 s/bước** (nhanh hơn fixgap 0,43 do đuôi ngắn hơn) · checkpoint @40100 ✅ · `--validate-at-stop` chạy: val 8,8275 (mel hữu thanh 0,5928 · nét 0,372).
- So 500 bước cuối base 40k: `loss +5,9%` · **`mel +11,7%`** · **`G +26,4%`** (fixgap probe: +33,9% / +41,8% ⇒ **ít hơn nhiều** ✅) · `tho +0,6%` ✅ · `f0d +4,9%` ✅ · `sharp +2,0%` ✅ · **`f0p −18,9%` ❌ ngoài ±5%**.
- **Điểm lạ**: `f0p` −18,9% (GIẢM, tức tốt hơn) — nhưng probe fixgap cũng −12,6% ⇒ nghi **nhiễu lấy mẫu 2 bước** (log_every=50). Nhánh đã ghi ("kết quả lạ ⇒ dừng + báo") ⇒ **DỪNG**, không chạy thật.
- Đề xuất xử lý (chờ Agent 0): (a) chấp nhận B10 (cơ chế đạt + mel/G tốt hơn hẳn), hoặc (b) probe lại với `log_every=10` (10 mẫu) để phân giải `f0p`, hoặc (c) bỏ `f0p` khỏi bộ tiêu chí probe (nó là prior phụ, trọng số 0,1).

## TỰ QUYẾT KHÁC ĐẶC TẢ
1. **Đổi bên đối chứng probe**: "TB 500 bước cuối base 40k" → **"extension cùng bước 40050/40100"** (cùng checkpoint/LR/hyperparams, chỉ khác manifest), vì mốc cũ khác cadence log và base tự dao động ±11%. Số: +10,4% (mốc cũ) vs **+7,6%** (đối chứng đúng).
2. **Nguồn `mel_gap_z`**: sửa free → **tf** (đúng định nghĩa B2) + hiệu chỉnh 2 tệp mốc từ **dữ liệu thô cùng lần chạy** (không chạy lại eval).
3. **B2 kiểm bằng đo** (id-diff script) thay vì suy luận.

## Bằng chứng kèm theo
- `docs/handoff/V08_agent0.md` (quyết định) · `docs/handoff/TRANG_THAI.md` (mốc/hash/cổng/việc chờ)
- `artifacts/dataset_v2/verify_gaps_{train,val}_fixtail.json` · `eval/README.md` (hash ghim)
- Probe cũ (fixgap): `runs/acoustic_v2_A2_fixgap_probe/` (giữ làm đối chứng, không resume từ đó)
