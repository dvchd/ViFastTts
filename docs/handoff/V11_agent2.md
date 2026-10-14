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

## Phần bổ sung 2 (V11S2.B1–B5) — xong

| Mã | Việc | Kết quả |
|---|---|---|
| V11S2.B1 | commit đặc tả | ✅ (đặc tả bổ sung 2 nguyên văn vào repo) |
| V11S2.B2 | val f0p A2 vs base + train trượt 500 | ✅ **Agent 0 quyết: KHÔNG dừng A2** (dao động nội tại ±3%; train +0,9%) — theo dõi 48k–60k: chỉ báo nếu 3 mốc liên tiếp >+5% VÀ tăng; cuối cùng để bench quyết bằng F0 NGÃ + tone_C. **BÁO THEO NHÁNH**: base 40k val f0p **0,41400** (CPU, 300 hàng val300 fixtail, batch 1) · A2 theo mốc: 41k 0,4287 · 42k 0,4083 · 43k 0,4206 · 44k 0,4208 · 45k 0,4326 · 46k 0,4289 · **47k 0,4350 (+5,07% — vượt biên ±5% 0,07 điểm %)**; trung bình 7 mốc **+2,8%**. CPU cross-check khớp history **từng chữ số** (45k 0,43262 · 47k 0,43498 ⇒ MPS≈CPU ✓). **TRAIN f0p trượt 500 bước cuối = 0,60364 vs base 0,59810 = +0,9%** (trong ±5%) ⇒ chênh −16,4% ở probe là **nhiễu một cửa sổ** (khớp giả thuyết LR/cửa sổ batch). Nhánh: "TĂNG hơn 5% ⇒ báo ngay, Agent 0 quyết" — đã báo (mốc 47k +5,07%, xu hướng tăng nhẹ, train +0,9%); **A2 vẫn chạy** tới khi có quyết định |
| V11S2.B3 | kiểm 3 điều kiện A5 (bản sao 45k) | ✅ `ckpt_45k.pt`: **step 45000** đúng · `torch.load` OK · **sha256 `db37bbe622937bc6…`** · chép lúc **13:15:41** (bản sao do watcher CŨ tạo — đọc trực tiếp last.pt; **từ 50k dùng watcher mới**: chép trước → `torch.load` trên bản sao → đúng step mới nhận, lỗi thì chép lại; ghi sha256+giờ vào `/private/tmp/opencode/ckpt_report.jsonl`) |
| V11S2.B4 | CreaPy | ✅ **ĐẠT kiểm chứng — chuyển thành thước THEO DÕI** (Agent 0; free-run 4 bản sau B6) — cài vào `~/creapy-venv` (clone `SPSC-TUGraz/creapy` về `~/creapy-src`, KHÔNG vào repo, không thêm dependency — GPL-3.0); tham số theo A4: **min creak 60 ms · ngưỡng 0,8 · STE phân vị 30 mỗi clip**, `gender_model=male` (F0 trung vị giọng user **122 Hz ⇒ nam**). Trên audio THẬT (25 clip giàu ngã/nặng, 786 âm tiết): **ngã+nặng 0,1774 vs ngang+huyền 0,1023 = 1,73× ⇒ creak rõ hơn ⇒ ĐẠT** (ngưỡng 1,15×). Copy-synth 25 clip **xong**: mel thật → Vocos base: **ngã+nặng 0,1893 vs ngang+huyền 0,0792 (2,39×)** — Vocos GIỮ (chí rõ hơn) tương phản creak so với audio thật (1,73×) ⇒ không phải nguồn làm mất creak; **free-run base 40k để sau B6** (cần MPS) |
| V11S2.B5 | hàng val thiếu | ✅ hàng **`4ITcvEa_Acc_0186__s01`** (parent `4ITcvEa_Acc_0186` **có trong val300**) ⇒ bị loại để chống rò rỉ đúng như dự đoán (đã ghi §44) |

## Phần B-II — A2 ĐÃ XONG 60k (15:40) — số theo dõi (báo theo V11S2.A2)

**A2 kết thúc sạch**: step 60000 · best.pt @42000 (best_val 0,5886) · last.pt @60000 · no_improve 18 · watchdog đóng đúng.
`ckpt_45k/50k/55k` ✓ (sha 45k `db37bbe6…`, 55k `9e44e9f7…`) · sự cố 55.850 → resume 55.000 đã xử lý ✓.

| mốc | val f0p | vs base (0,41400) | train_pause_gate | val_pause_gate | val/train | val_pause_frac |
|---|---|---|---|---|---|---|
| 45000 | 0,4326 | **+4,50%** | 0,1038 | 0,9855 | 9,5× | 0,122 |
| 50000 | 0,4678 | **+12,99%** ⚠ | 0,0787 | 1,0934 | 13,9× | 0,127 |
| 55000 | 0,4710 | **+13,77%** ⚠ | 0,0586 | 1,1796 | 20,1× | 0,131 |
| 60000 | 0,4987 | **+20,47%** ⚠ | 0,0479 | 1,2996 | 27,2× | 0,137 |

**⚠ TRIGGER (V11S2.A2)**: val f0p **3+ mốc liên tiếp > +5% VÀ xu hướng tăng rõ** (4,50 → 12,99 → 13,77 → 20,47%) ⇒ **báo ngay**.
Lưu ý kỹ thuật: f0p là loss **kéo contour về prior trung bình của thanh** — nó TĂNG có thể do model học contour **riêng biệt hơn** (xa prior), chưa chắc là F0 xấu đi; **quyết định cuối theo cổng cứng F0 NGÃ (+5 cent) + tone_C** ở B6 (đang chạy).
**⚠ Thêm**: BCE pause val **tăng** (0,986→1,300) trong khi train giảm (0,104→0,048) ⇒ tỉ số val/train **xấu dần** (9,5×→27,2×; base ~20×) — cần bench xác nhận recall/f1 thật.

## V11.B6 — bench A2 4 mốc (đã xong) + V11.B7 cây quyết định

| mốc | mel_z✗ | dB✗ | rough | toneC | cer | cerP | f0nga | melgap | sylz | pF1✗ | gMAE✗ | lr | gapz✗ | sp | pa | Δjit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 45k | 1,1924 | 9,414 | 0,3867 | 0,7858 | 0,2048 | 0,1896 | 182,4 | 3,1122 | 0,6672 | 0,5855 | 124,41 | 0,8829 | 2,1705 | 0,935 | 0,724 | 0,0490 |
| 50k | 1,1916 | 9,194 | 0,3904 | 0,8117 | 0,2037 | 0,1919 | 182,2 | 3,0933 | 0,6676 | 0,5880 | 123,36 | 0,8816 | 2,1658 | 0,932 | 0,721 | 0,0444 |
| **55k** | **1,1848** | 9,258 | 0,3931 | 0,8148 | 0,2030 | 0,2322 | 181,1 | 3,1012 | 0,6681 | **0,5884** | **123,35** | 0,8888 ✅ | **2,1462** | 0,937 | 0,726 | **0,0409** |
| 60k | 1,1954 | 9,281 | 0,3911 | **0,8180** | **0,1961** | 0,2013 | 184,1 | 3,1006 | 0,6685 | 0,5870 | 122,43 | **0,9007** ✅ | 2,1724 | 0,950 | **0,740** | 0,0477 |
| base | **1,1599** | **8,955** | 0,3819 | 0,8045 | 0,2332 | 0,2385 | 180,2 | 3,1272 | 0,6621 | 0,5559 | 156,46 | 0,7665 | 2,1529 | 0,934 | 0,533 | 0,0372 |

**KẾT LUẬN MỖI MỐC: TỪ CHỐI** (mel_l1_z + mel_db trượt ở cả 4 ✗). **V11.B7 = nhánh "TỪ CHỐI" ⇒ DỪNG, báo, KHÔNG chạy A1** (đúng cây V11.A4).

**Phân tích (để Agent 0 quyết, không tự ý đổi luật)**: hai cổng trượt là **mel trên val CŨ** (+1,1–2,0% vs biên 1%; +0,14–0,36 dB vs biên 0,10) — trong khi **`syl_z` (vùng âm tiết) KHÔNG đổi** ✅ ⇒ phần xấu thêm nằm ở **vùng gap của val cũ** — tức so **khác nhãn** (A2 học nhãn fixtail, val cũ giữ nhãn cũ). Trên **val fixtail** (cùng nhãn): mel_z tổng base 1,3311 vs A2 1,3414/1,3404/**1,3311 (bằng đúng)**/1,3457. **6 cổng cứng còn lại đều đạt và TỐT HƠN base** (rough, tone_C, CER, f0_nga, mel_gap, syl_z) + **cải thiện thật**: tone_C 45k 0,786→60k **0,818** (base 0,8045) · CER **0,1961** (base 0,2332) · cer_prod **0,1896–0,2322** (base 0,2385) · **length_ratio 55k/60k ĐẠT cổng A9** ✅ · `length_ratio_pause` 0,533→**0,740 (+39%)** với `speech` gần như không đổi ✅. **Chưa đạt**: pause_f1_all (+0,030/0,073) · gap_mae_all (−22%/28,6%) · gap_z (đứng yên) · Δjitter xấu hơn base ở cả 4 mốc (0,041–0,049 vs 0,0372).
**Nếu Agent 0 quyết bỏ qua 2 cổng mel khác nhãn**: mốc tốt nhất = **55k** (mel_z thấp nhất, pF1/gMAE/Δjit/gapz tốt nhất, lr ĐẠT; hòa mục tiêu với 60k nhưng gap_z thấp hơn ✓).

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


## Phụ lục: nghiên cứu dataset giọng NỮ BẮC mở hoàn toàn (sub-agent, 14/10)

**Kết luận: KHÔNG có bộ nào hội đủ** "mở hoàn toàn + nữ Bắc (Hà Nội) + chất lượng cao + ≥5–10 h để train acoustic".
- Gần nhất: **VLSP 2021 TTS** (~7,5 h, nữ Hà Nội, **research-only**, thu từ vlog có nhiễu) · **VAIS-1000** (~1,7 h nữ Bắc **studio**, **CC-BY-4.0** — quá ngắn để train, tốt cho eval/fine-tune).
- Nhóm **giấy phép sạch cho thương mại/eval**: VAIS-1000 (CC-BY-4.0) · Common Voice vi validated (CC0, lọc `sex=female`; MDC cấm re-host) · FLEURS vi (CC-BY-4.0) · FOSD ~30 h (FPT license mở) · LSVSC 100 h (CC-BY-4.0, rủi ro provenance crawl).
- **Cảnh báo**: các bộ lớn "mở" (viVoice 1.017 h, PhoAudiobook 941 h, Bud500 500 h, VSV-1100 1.100 h, GigaSpeech2 6.039 h…) đều **crawl YouTube/social + NC/không sở hữu bản quyền** ✗ — không dùng cho sản phẩm.
- **Đính chính 2 chi tiết trong báo cáo sub-agent**: (a) giọng hiện tại của dự án là **NAM Bắc** (F0 trung vị 122 Hz), không phải nữ; (b) dataset train thật = **19,3 h** (không phải 12 h).
- **Ý nghĩa kế hoạch**: muốn giọng nữ Bắc *sản phẩm* ⇒ **tự thu 10–20 h** (hoặc VLSP 2021 chỉ để nghiên cứu, không phát hành); còn VAIS/FLEURS/CV dùng được ngay cho **eval chéo người nói** (kiểm khái quát) — giấy phép sạch.
