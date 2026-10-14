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
| V08.B5/B6 | bench kiểm sha mốc vs README · `--validate-at-stop` | ⏳ chưa làm (chờ lệnh) |
| V08.B9/B10 | mốc `base40k_fixtail.json` (MPS ~30') · probe lại (~2') | ⏳ chưa làm (chờ duyệt) |

## TỰ QUYẾT KHÁC ĐẶC TẢ
1. **Đổi bên đối chứng probe**: "TB 500 bước cuối base 40k" → **"extension cùng bước 40050/40100"** (cùng checkpoint/LR/hyperparams, chỉ khác manifest), vì mốc cũ khác cadence log và base tự dao động ±11%. Số: +10,4% (mốc cũ) vs **+7,6%** (đối chứng đúng).
2. **Nguồn `mel_gap_z`**: sửa free → **tf** (đúng định nghĩa B2) + hiệu chỉnh 2 tệp mốc từ **dữ liệu thô cùng lần chạy** (không chạy lại eval).
3. **B2 kiểm bằng đo** (id-diff script) thay vì suy luận.

## Bằng chứng kèm theo
- `docs/handoff/V08_agent0.md` (quyết định) · `docs/handoff/TRANG_THAI.md` (mốc/hash/cổng/việc chờ)
- `artifacts/dataset_v2/verify_gaps_{train,val}_fixtail.json` · `eval/README.md` (hash ghim)
- Probe cũ (fixgap): `runs/acoustic_v2_A2_fixgap_probe/` (giữ làm đối chứng, không resume từ đó)
