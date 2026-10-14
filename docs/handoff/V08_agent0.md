# VÒNG 08 | Commit gốc: `ab42b06` | Thay thế: phần giao việc của mọi vòng trước
Agent 2 THỰC THI: Phần B | Agent 1 ĐỌC: Phần C | Người dùng: Phần D

## A. Quyết định (Agent 0)
- **A1. Chặn trần lặng cuối clip**: `C_TAIL = 67 frame` (= trung vị nhóm "chỗ cắt", nhịp nghỉ thật
  ≈0,7 s). Chỉ áp cho **gap cuối của đoạn cuối clip**; cắt **cả nhãn lẫn `frame_range[1]`** (nếu chỉ cắt
  nhãn mà giữ mel ⇒ lặp lỗi cũ). Nối câu: chèn lặng tất định ở tầng ghép audio. Quy tắc 40% cũ **không
  áp lại** sau khi chặn trần.
- **A2.** Manifest mới `*_fixtail` (train + val300); `base40k_fixtail.json` thay `base40k_fixgap.json`;
  `base40k_old.json` giữ (ghim lại sha256).
- **A3 (lỗi chặn W5).** `watch_train_v2.sh` nhận `INIT_RESUME`/`INIT_RESET_VAL_STATE` — **chỉ khi
  `OUT/last.pt` chưa có**; `run_A2_fixgap.sh` chế độ thật truyền `INIT_RESUME=runs/acoustic_v2_base/last.pt`,
  `INIT_RESET_VAL_STATE=1`; thêm `--dry-run`; kiểm sau 1 phút: dòng đầu `history.csv` = `# X3 resume:` @40000.
- **A4.** Ghim lại sha256; bench **tự kiểm** sha tệp mốc vs README, lệch ⇒ dừng.
- **A5.** Probe ĐẠT về cơ chế; **phải probe lại** trên fixtail (LR <1% · không NaN · `tho/f0d/f0p/sharp`
  trong ±5% so 500 bước cuối base 40k · `mel`/`G` báo số, kỳ vọng tăng **ít hơn** fixgap (+33,9%/+41,8%)).
  Thêm `--validate-at-stop` nếu sửa nhỏ.
- **A6.** **sh25 giữ toàn cục** (kết quả nghe AB), đóng mục 5c; rè ngã/nặng thuộc acoustic.
- **A7.** W1–W4, W6–W9 đạt; W11 xác nhận không có trần trong `cut_segments` (chỗ sửa ở A1).

## B. Agent 2 THỰC THI
| Mã | Việc | Trạng thái |
|---|---|---|
| V08.B1 | `cut_segments --tail-cap-frames` + test (3 ca) | ✅ |
| V08.B2 | sinh train/val `*_fixtail` + verify + so theo id + ghim hash + bảng B1 mới | ✅ |
| V08.B3 | đếm lại p ⇒ pos_weight | ✅ (p không đổi 0,3082 ⇒ 2,244) |
| V08.B4 | watchdog + script (INIT_*, --dry-run) + dán đầu ra dry-run | ✅ |
| V08.B5 | bench kiểm sha mốc vs README + test + ghim lại `base40k_old.json` | ⏳ |
| V08.B6 | `--validate-at-stop` + test | ⏳ |
| V08.B7 | §44: C_TAIL + lý do cắt target + sh25 toàn cục + fixtail thay fixgap | ✅ |
| V08.B8 | toàn bộ test + push + báo commit | ✅ `64a47ff` |
| V08.B9 | sinh mốc `base40k_fixtail.json` (MPS) + kiểm H1 ≤3% + ngưỡng tuyệt đối | ⏳ |
| V08.B10 | probe lại (thư mục `runs/acoustic_v2_A2_fixtail_probe`) | ⏳ |
| V08.B11 | DỪNG, báo cáo; SẮC-tắc khi MPS rảnh | ⏳ |

## C. Agent 1 ĐỌC (khi push): Y1–Y6
Y1 trần trong `cut_segments` (chỉ đoạn cuối clip? cắt `frame_range[1]`? giữ hành vi cũ khi không cờ? test?)
Y2 watchdog `INIT_*` chỉ khi `last.pt` chưa có? Y3 lệnh thật có `--resume`+`--reset-val-state`+fixtail?
Y4 bench kiểm sha mốc ở đâu, lệch làm gì, test? Y5 `--validate-at-stop` ảnh hưởng `best_val`/early stop?
Y6 thay đổi ngoài đặc tả B1–B7?

## D. Người dùng
Nghe 6 cặp `ab40k_nga{1..6}_{plain|sh25}.wav` (release `demo-audio-20261013`). Kết luận: sh25 ổn ở ngã,
plain rè hơn tổng thể ⇒ **giữ sh25 toàn cục**.
