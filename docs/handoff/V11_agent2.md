# VÒNG 11 | Báo cáo Agent 2 (đang chạy) | Gốc: `b1ae8ce` · Commit code cuối: `<cập nhật>`

**Test**: xem mục B4/B5 · **Trạng thái**: A2 ĐANG CHẠY + B-I (B1–B5) xong

## Phần B-I (trong lúc A2 chạy — chỉ CPU)

| Mã | Việc | Kết quả |
|---|---|---|
| V11.B1 | commit đặc tả | ✅ `3022f29` (V11_agent0.md nguyên văn; TRANG_THAI → V11) |
| V11S.B1 | đặc tả bổ sung | ✅ `c06959c` (V11_agent0_bosung.md nguyên văn) |
| V11.B2 | σ chuẩn + báo lại 7 thành phần | ✅ `100c50a` — σ ở V10 tính trên **giá trị từng mẫu** (log ghi mỗi 50 bước; 10 mẫu trong 500 bước cuối base) và so với **một mẫu tức thời** của probe ⇒ **đúng thang mẫu↔mẫu**. Khi so với **trung bình m bước**: σ chuẩn = σ_mẫu/√m. Bảng (2σ mẫu → 2σ chuẩn m=20 → probe Δ%): loss 13,22→2,96 (+4,31) · mel 49,38→11,04 (+13,16) · G 70,58→15,78 (+40,78) · tho 17,50→3,91 (−1,13) · f0d 25,07→5,61 (+5,32) · **f0p 59,11→13,22 (−16,40)** · sharp 9,49→2,12 (+2,64). Kiểm chéo: σ của trung bình trượt 2 mẫu khớp σ_mẫu/√2 (lệch 4–20%) ⇒ quy tắc √m dùng được. **Log cũ (50 bước) không dựng được trượt 20 bước** — cột 2σ chuẩn là **ước lượng**; từ A2 dòng `comps:` in trung bình cửa sổ 50 bước ⇒ lần sau đo trực tiếp. §44 đã sửa. **Không ảnh hưởng A2** (V10.B4 đã loại lỗi chuẩn hóa; mel/G tăng là đúng đích A2) |
| V11.B3 | chia val_calib / val_lock | ✅ `e21c2ff` — `tools/split_val_calib_lock.py`: hàng 301–1773, **theo clip cha**, loại 1 clip cha dùng chung với val300. **calib 737 hàng/413 clip (sha1 `147f4b7c…`)** · **lock 735/429 (sha1 `f6db096a…`, KHÓA)**. Ghim cả 4 hàng manifest (train/val fixtail + calib/lock) vào `eval/README.md`. Test `tests/test_val_split.py`: không clip cha nào ở hai tập · không giao val300 · không trùng hàng · cân 0,4–0,6 |
| V11.B4 | gate_threshold + cer_prod | ✅ `c06959c` — **gate_threshold** xuyên suốt: model (`v2_slot5.forward`, cuối danh sách, mặc định 0,5) · export (**input thứ ba cuối** `gate_threshold` float32 [1] — graph cũ 2 input vẫn chạy) · engine (đọc input thứ ba nếu có, feeds thêm) · infer/CLI (`--gate-threshold`). Test: mặc định == 0,5 **từng bit**; ngưỡng cao hơn ⇒ số frame không tăng; engine 3 input truyền gate, 2 input chạy như cũ. **cer_prod** (V11.A5): `--cer-prod` → lead=0 (free-run vốn không truyền lead) + **đệm lặng 150 ms** hai đầu (test 3.600 mẫu @24 kHz); ghi `cer_prod` (cổng vẫn dùng `cer`); bench chạy thêm lượt cer_prod cho base |
| V11S.A3/A4/A5 | bổ sung | ✅ `c06959c` — quy ước **"commit code cuối"** (HEAD luôn là commit hồ sơ; AGENTS.md §8 + TRANG_THAI sửa) · probe **chặn ghi đè** khi đã có last.pt (trừ `--overwrite-probe`; test) · test pin **skip** khi thiếu git/không phải repo (phần README==đĩa vẫn chạy) |
| V11.B5 | CreaPy + UTMOSv2 | ⚠️ **CreaPy: CHƯA LÀM** — không có trên PyPI (`pip install creapy` fail) và không tìm được nguồn chính thức (search rỗng) ⇒ theo đặc tả: ghi CHƯA LÀM, không thay công cụ khác. **UTMOSv2: cài được** (git+github sarulab-speech/UTMOSv2, 1.3.1.dev0; trọng số HF tải bằng curl vì thiếu `wget`) — resample nội bộ về 16 kHz (`dataset/_utils.py:18` ✓, file ta 24 kHz). **sh25/plain: sh25 thắng 6/6** (1,495/0,731/0,880/1,021/0,823/1,385 → 2,483/2,185/2,005/1,107/1,192/2,175) ✅ **khớp tai user** · **40k/18k: chỉ 2/3 cặp** (cau_dai 1,208>1,165 ✅ · hoi_thoai 1,134<1,402 ❌ · sau_thanh 1,385>1,252 ✅) và **trung bình 40k 1,242 < 18k 1,273** ⇒ **KHÔNG khớp tiêu chí ⇒ BỎ** (không dùng làm thước/theo dõi) |

## Phần B-II (sau khi A2 tới 60k) — chưa làm
V11.B6–B11 + V11S.B3 (watchdog: step_of chịu last.pt hỏng · ghi checkpoint nguyên tử · chặn nhánh đầu khi có history.csv — **KHÔNG sửa watch_train_v2.sh khi A2 đang chạy**, đúng V11S.A2).

**Phát hiện + xử lý trước B6 (TỰ QUYẾT #4)**: trainer chỉ ghi `last.pt` (save_every 5000 **ghi đè**, không giữ checkpoint theo mốc)
⇒ B6 cần 45k/50k/55k sẽ **mất** nếu không sao lưu. Đã phóng watcher CPU (`/private/tmp/opencode/ckpt_saver.py`, chỉ đọc log +
`last.pt`, thử lại khi đọc lỗi) sao lưu `runs/acoustic_v2_A2_fixtail/ckpt_{45,50,55}k.pt` khi A2 qua mốc (A2 ở 43.750 lúc phóng
⇒ 45k chưa qua ✓). 60k = `last.pt` cuối. Kế hoạch B6: bench 4 mốc × (`ckpt_45k/50k/55k/last.pt`) + `cer_prod` base 40k.

## TỰ QUYẾT KHÁC ĐẶC TẢ (tạm)
1. UTMOSv2 cài từ GitHub + tải trọng số bằng `curl` (đặc tả không nói cách; `wget` thiếu trên máy).
2. Test gate_threshold: kỳ vọng "số frame không tăng" thay vì "bằng nhau" (ngưỡng cao hơn cắt bớt gap là đúng chức năng — test đầu tiên của tôi sai kỳ vọng, đã sửa).
3. Ghi 4 hàng manifest fixtail/calib/lock vào `eval/README.md` (đặc tả chỉ nói ghim hash).

## CHƯA LÀM / LÀM DỞ (tạm)
- **CreaPy** (V11.B5): không cài được — xem bảng trên.
- **UTMOSv2**: bị loại theo tiêu chí (40k/18k không khớp).
- B6–B11 + V11S.B3: chờ A2 tới 60k.
