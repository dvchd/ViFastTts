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

## PHẢN BIỆN NGOÀI (sub-agent, model `opencode-go/muse-spark-1.3-contributor`)

**Kết luận chính của phản biện (đọc `V11_agent2.md` + `ACOUSTIC_V2.md` §37–44 + `EVAL_TOOLS.md`):**
1. **CẮT phần vocoder của E4** (MR-STFT trọng số <1 kHz + discriminator CQT 80–320 Hz): cùng họ "kéo thống kê" như GTA (§41) — GTA đã chứng minh jitter về mức thật nhưng HNR −1,7 dB + rough mượt hơn ✗; và **hop 256 ⇒ mel không chứa jitter mức chu kỳ** ⇒ trần lý thuyết đã thấy. Giữ **F0-mask vùng creak** (phía acoustic, rẻ) + creak theo âm tiết.
2. **A1 (pause ×4) đã dừng là ĐÚNG** — đừng mở lại (BCE pause val/train 9,5×→27,2× đã overfit; tăng trọng số = đổ thêm dầu). Phần còn lại của pause là **chất lượng đặt pause** (F1/MAE), không phải khối lượng.
3. **Sai thứ tự lớn nhất: 2 cổng mel val CŨ đang chặn cải thiện THẬT** ⇒ phải **version cổng theo val hash** (khi đổi nhãn thì đổi thang đo), thay vì để từng vòng phải "xin bỏ qua".
4. Flow-matching/ZipVoice: chỉ là nghiên cứu nền; prototype phải qua cổng **"ONNX tĩnh + RTF CPU"** trước khi đo chất lượng.
5. **1 thí nghiệm duy nhất đáng làm**: chốt **A2-55k** bằng **A/B mù (tai user) + cer_prod + CreaPy + eval_creak**, KHÔNG train thêm; cổng: A/B thắng/hòa · cer_prod ≤ base 0,2385 · tỉ số creak (ngã+nặng)/(ngang+huyền) không thấp hơn base · Δjitter ≤ base+0,01.
6. **Đang bỏ qua**: (a) **duration là nghi phạm số 1** (length_ratio_speech 0,93–0,95 + pause + SẮC-tắc cùng chỉ về phân phối duration; MAE 4,8 frame đứng yên từ B0); (b) chưa có **cổng duration/prosody**; (c) CER 20-clip quá nhiễu để quyết (cần 50–100 — **cer_prod của ta đã 842 clip ✓ vượt yêu cầu**); (d) luật cổng thiếu điều khoản đổi nhãn; (e) checkpoint theo mốc (đã vá ở `a027fd4` ✓).
7. **3 tuần**: W1 chốt ứng viên (không train) + hiệu chuẩn `gate_threshold` trên val_calib · W2 sửa rẻ nhất còn lại (duration/prosody, hạn chế đụng core) · W3 đóng gói + niêm phong eval + đóng băng E4-vocoder/flow-matching thành roadmap.

**Quyết định của phiên chính (tự quản lý)**: ÁP DỤNG 1–7. Cụ thể: cắt phần vocoder của E4 (ghi roadmap) · A1 đóng · thêm **cổng "cùng nhãn" (val fixtail)** cho bench (việc kế tiếp) · W1 = chốt A2-55k bằng A/B mù + CreaPy free-run + eval_creak (không train) · đưa **duration** vào B9 thành chẩn đoán chính · không mở vòng train acoustic/vocoder mới tới khi W1 xong.

## W1 — chốt ứng viên A2-55k (không train) — số đo

**CreaPy 4 bản (25 clip giàu ngã/nặng, 384 âm tiết ngã+nặng / 402 ngang+huyền):**

| bản | ngã+nặng | ngang+huyền | tỉ số |
|---|---|---|---|
| thật | 0,1772 | 0,1026 | 1,73× |
| copy-synth (mel thật → Vocos) | 0,1894 | 0,0794 | 2,38× |
| free-run **base 40k** | 0,3235 | 0,1187 | 2,72× |
| free-run **A2-55k** | 0,3320 | 0,1184 | **2,80×** |

- **Cổng Muse #3 ĐẠT**: tỉ số A2-55k (2,80×) **không thấp hơn** base (2,72×) ✓
- **Phát hiện mới**: free-run creak **cao hơn hẳn thật** (0,32 vs 0,18) trong khi copy-synth ≈ thật (0,19) ⇒ phần creak thừa đến từ **mel DỰ ĐOÁN của acoustic** (cả base lẫn A2), không chỉ từ Vocos — bổ sung cho §39 (jitter +39% ở copy-synth là chuyện khác, thang khác). ⇒ đích E4 phía **acoustic** đúng hơn là phía vocoder (khớp quyết định cắt vocoder-E4).

**Cổng W1 khác**: cer_prod A2-55k **0,2322 ≤ base 0,2385** ✓ (nhưng 60k tốt hơn nhiều: 0,2013) · Δjitter A2-55k 0,0409 ≤ base+0,01 = 0,0472 ✓ · **A/B tai người = cổng còn lại**.
**Đánh đổi 55k vs 60k**: 55k tốt hơn ở mel_z (1,1848 vs 1,1954), pF1 (0,5884), gMAE (123,35), gap_z (2,1462), Δjit (0,0409) · 60k tốt hơn ở CER (0,1961), cer_prod (0,2013), tone_C (0,8180), length_ratio (0,9007). Có thể sinh thêm cặp A/B cho 60k nếu tai bạn muốn so 3 bản.

**Release A/B (đã kiểm tải HTTP 200 + đúng byte)**: `demo-audio-20261014-w1` — 6 cặp câu giàu ngã/nặng, `w1_<i>_{base40k,A2_55k}.wav`
→ `https://github.com/dvchd/ViFastTts/releases/download/demo-audio-20261014-w1/w1_1_base40k.wav` (và 11 file cùng dạng)

## PHẢN BIỆN NGOÀI (sub-agent 2, model mặc định — nghiên cứu E4: creak/F0/vocoder)

**Đã tìm ra nguồn V11.A13 nhắc**: repo **`iamdinhthuan/tone-aware-vocos`** (paper 2026 *under review*, ẩn danh ✗ — mọi số phải đo lại). Số gốc + hệ số cụ thể:
- **C — MR-STFT tăng trọng số ≤1 kHz (w=3×, FFT 512/1024/2048, hop 256, Hann)**: F0 RMSE **3,498→3,340 Hz (−4,5%)**, so baseline **p_holm=9,9e-5** ✓ (bằng chứng MẠNH). Objective Vocos của họ: `45·mel + 0,1·mrd + 1,0·weighted_mrstft` ⇒ **phải tune lại tỉ lệ** cho trainer ta (đề xuất quét {0,25; 0,5; 1; 2}).
- **B — CQT critic 80–320 Hz** (fmin 80, 48 bins @24/oct, log-mag, adv 0,1 + fm 0,1): 3,340→3,271 nhưng **p_holm=0,626 KHÔNG ý nghĩa** ✗ ⇒ **công lao chính là C, không phải B**.
- **A — feature-matching của classifier thanh = ablation ÂM** ✗: thêm A làm F0 RMSE **3,271→3,402** (p=3,7e-5 hướng xấu) ⇒ **không làm**; nếu muốn "tone-aware" thật thì đi **CTC/tone-unit** hoặc **CE trên s_k** (đã có), không feature-matching.
- **F0 vùng creak**: gốc là **Ninh & Yamashita IS 2015 + IEICE 2015** — không phải nội suy thô mà **lan truyền pitch mark** với ngưỡng cụ thể (`MinPR=0,4 · MaxPR=2,7 · CorrThs=0,4 · cửa sổ L=min(P,15 ms)`; C1/B2 tìm tiến, C2 tìm cả hai; >25 ms coi là vô thanh). VME 14,5%→6,8% (test) **nhưng RMSE có tính vòng tròn** ✗ (reference tạo bằng chính phương pháp đó) ⇒ đừng lấy làm mốc.
- **Khuyến nghị cho kiến trúc slot-DCT**: **nội suy có trọng số** hợp hơn mask (DCT cần contour xác định toàn slot) — **lai**: target = contour lan truyền, loss nhân `w_t = 1 − 0,5·c_t` (c_t = xác suất creak/frame) ✓.
- **Thước F0 phân bố (mới)**: bộ 3 tầng — (1) `flatness_T = std_T(model)/std_T(thật)` theo **từng thanh** (<0,9 = xẹp); (2) **W₂** (công thức đóng 1-D) / **JSD** histogram 20–24 bin + bootstrap **theo ÂM TIẾT**; (3) phân bố slope/onset/offset theo rime (ViTone). Đơn vị = âm tiết (không frame ✗); cần **≥200–450 âm tiết/thanh**; phải **kiểm thước trước**: thật-vs-thật ≈0 + xếp đúng cặp 40k>18k ✓.
- **Cảnh báo đúng điểm mù của dự án**: (a) **mọi detector creak tốt hơn trên giọng NỮ** (White JASA 2022) ✗ ta là nam; (b) classifier thanh trên thật chỉ ~58–68% và **recall NGÃ chỉ 6,4%** (ViTone 2026) ✗ ⇒ đừng biến creak/ngã thành cổng cứng, chỉ trọng số mềm ✓.
- **Xếp hạng 5 việc**: 1) C (weighted MR-STFT) · 2) kiểm gate creak vs gán tay ~100 âm tiết · 3) bộ thước F0 phân bố · 4) A/B MASK vs NỘI SUY vs LAI cho target F0 creak · 5) B (CQT) chỉ nếu p_holm<0,05.

**Quyết định tinh chỉnh của phiên chính**: giữ nguyên "cắt B (CQT critic) + không thêm A" ✓; **C chuyển từ "cắt" sang ứng viên rẻ có bằng chứng mạnh** (chỉ-train, 0 chi phí suy luận; cổng: F0 RMSE per-tone giảm ≥3% + `val_rough`/`eval_creak` trong ±5%, nếu trượt ⇒ `stft_coeff=0`) — xếp **sau W1/W2**, cùng đợt với E4; **E4 = nội suy có trọng số (lai) thay vì mask**; thêm **bộ thước F0 phân bố** vào danh sách thước (làm sau khi W1 xong, phải kiểm thước trước ✓).

## W1 — KẾT LUẬN + B9 (chẩn đoán nhanh, không train)

**A/B tai người (6 cặp, nghe lại kỹ): HÒA** ✓ — người dùng xác nhận "về cơ bản là hòa". Ghi chú kèm: **rè to hơn ở A2** (cặp 1, cặp 5 "vừa đọc vừa rè nặng") — khớp số đo (Δjitter 0,0409 vs 0,0372; CreaPy 2,80× vs 2,72×).
**Quyết định**: A2-55k **đạt 4/4 cổng W1** (hòa · cer_prod ✓ · CreaPy ✓ · Δjitter ✓) ⇒ **bản tham chiếu nội bộ** cho E4/ONNX; **base 40k vẫn là bản dùng hằng ngày** tới khi dòng rè xong (A/B hòa + rè là điểm đau #1).
**Sửa nhầm đã hoàn tác**: `"AI" → "ây ai"` GIỮ NGUYÊN (mình tưởng nhầm là lỗi đọc; thực ra người dùng chỉ mô tả + nêu RÈ) — commit `34f9b6c`.

**B9 — Duration (tf, val fixtail, 7.218 âm tiết)**: base 40k **4,99 frame (53 ms)** · A2-55k 5,01 (53 ms) — **không đổi** ✗. Theo nhóm `coda:tone`: xấu nhất NONE|HUYEN **6,79 frame (72 ms, n=803)**, J|SAC 5,80 (n=522), M|NGANG 6,37 — không nhóm nào nổi trội bất thường ⇒ **lỗi rải đều** ⇒ KHÔNG ở trần nhãn (§17 ~18 ms; 53 ms ≈ 3×) ⇒ **đầu duration còn dư địa** (hỗ trợ giả thuyết "thoái hoá về trung bình"); triệu chứng người thấy = nói nhanh 5–7% (length_ratio_speech 0,93–0,95) ⇒ **ứng viên W2**.
**B9 — VUV**: metric dùng **dự đoán thật** (`pred_vuv[active_t] == true_vuv[active_t]`, `eval_acoustic.py:747`) — **không phải lỗi đo** ✓; nhưng **bão hoà** (0,9823 giống nhau ở small/base) ⇒ theo V11.A9: **hạ từ cổng xuống THEO DÕI** ✓.

## HÀNG ĐỢI PHIÊN SAU (tự quản lý, không cần hỏi)
1. **E4 — RÈ (ưu tiên #1)**: target F0 vùng creak = **nội suy có trọng số** (lan truyền pitch mark, ngưỡng MinPR 0,4/MaxPR 2,7/CorrThs 0,4 — Ninh & Yamashita 2015) + loss **C** cho vocoder (MR-STFT ≤1 kHz w=3×, FFT 512/1024/2048, hop 256; tune lại tỉ lệ; cổng F0 RMSE per-tone ↓≥3% + rough/creak ±5%) — chỉ-train, 0 chi phí suy luận.
2. **W2 — duration**: sửa đầu duration (53 ms → mục tiêu ≤40 ms; length_ratio_speech → ≥0,97) mà không giảm CER/tone_C.
3. **Thước F0 phân bố** (flatness_T + W₂/JSD bootstrap theo âm tiết) — phải kiểm thước trước (thật-vs-thật ≈0; 40k>18k).
4. **Cổng cùng nhãn** trong bench (val fixtail: mel_z ≤ mốc×1,01 · dB ≤ mốc+0,10) + version cổng theo val hash.
5. **B10**: speed 0,8/1,25 · năng lượng ngã/nặng · mối nối câu (5 câu liền) — base 40k + A2-55k.
6. **Bộ 30 câu người thật** (người dùng thu) → chuẩn vàng + preference pairs.

## ⚠️ ĐÍNH CHÍNH sau red-team (sub-agent độc lập) — 3 lỗi của mình + 1 phát hiện lớn

**Bằng chứng lưu ở `artifacts/redteam/` (bench42k/ + redteam_*.json).**

1. **Lập luận "mel trượt do so KHÁC NHÃN" — SAI** ✗. Đối chứng **nhãn CŨ**: `runs/acoustic_v2_base_60k/last.pt` (ext57k, cùng cosine, cùng resume base@40000) **cũng trượt** trên val cũ: mel_l1_z **1,1765** (>1,1715) · dB **9,1408** (>9,0547), +3,3σ/+3,7σ. ⇒ **~2/3 mức xấu là hiệu ứng "train tiếp 40k→60k"**, không phải nhãn; §44.7 đã ghi ext57k kém hơn (mình không trích ✗). Nhãn các gap BÊN TRONG val cũ **trùng khít** fixtail (Σ|Δ|=0), chỉ khác lặng đuôi — mà lặng đuôi bị loại khỏi phép so val cũ ⇒ "khác nhãn" không giải thích được ✗.
2. **"syl_z KHÔNG đổi" — SAI** ✗: syl_z tăng **+0,68…0,97%** ở cả 4 mốc & 2 bản val (**+5,4σ**); chỉ *lọt* biên +1% với dư 0,09% (55k) / **0,03% (60k)** ✗.
3. **Bảng của mình đọc sai/sót**: `rough` base = **0,3919** (mình ghi 0,3819 = biên cổng ✗); **thiếu giá trị val CŨ** của mel_gap/syl_z/mel_gap_z (ở val cũ A2 **xấu hơn base cả 4 mốc** ✗); "6 cổng còn lại tốt hơn base" **sai** — f0_nga xấu hơn cả 4 mốc ✗, syl_z xấu hơn ✗, rough xấu hơn 3/4 ✗; chỉ **CER, mel_gap(fixtail), tone_C(50k+)** thực sự tốt hơn ✓.
4. **PHÁT HIỆN LỚN — `best.pt@42000` (checkpoint tốt nhất theo val của chính A2)**: **ĐẠT cả 2 cổng mel** (mel_z **1,1645** ✓ · dB **8,8116** ✓ **tốt hơn base**) + **7/8 cổng cứng** (rough hụt 0,0016; đo trên 60 clip thì ĐẠT) + **thắng 55k ở 11/14 chỉ số**: syl_z 0,6584 ✓(tốt hơn base) · mel_gap 3,0646 ✓ · **CER 0,1853** (tốt nhất) · F1 0,5951 · gMAE 120,15 · lr 0,9114 · Δjit 0,0356 (**tốt hơn base 0,0372**) · cer_prod. ⇒ **Vấn đề không phải "bỏ 2 cổng mel" mà là CHỌN MỐC**: grid 45–60k (watcher bắt đầu 43,75k) đã **bỏ đúng checkpoint tốt nhất** ✗✗ — tranh luận của mình lệch trọng tâm ✗.
5. **Hiệu chuẩn `gate_threshold` (mình đề xuất) — không chữa được phán quyết** ✗: red-team **đo trực tiếp**: trần F1 = **0,6182** (thr 0,05) trên val300, **vẫn < mục tiêu 0,6289** (trên val_calib đạt 0,6339 ⇒ **bẫy "chỉnh và báo cùng tập"** ✗); không chạm được cổng tf (mel/dB/gap_z) ⇒ **không thể đổi TỪ CHỐI**; và **§43.1 đã bác bỏ hướng này bằng số** (trần F1 0,564 vs 0,535) ✗. Điểm được thật: gapMAE 123,3→**110,8 (ĐẠT ≤111,7)** ✓ + lr_pause 0,726→0,887 ✓ (nhưng F1 vẫn hụt ✗). Cần thêm `--gate-threshold` cho `eval_acoustic` (hiện hard-code 0,5 trong thước) nếu muốn dùng.
6. **Bẫy thống kê phải nhớ**: `rough` 12 clip **lật phán quyết** 42k khi lên 60 clip ✗; biên ±1% mel ≈ **1–2σ** nhiễu ghép cặp ✗; `mel_gap` free-run đang so ở lr 0,889 vs 0,767 (vi phạm ghi chú "chỉ tin khi lr≈1") ✗; free-run mel_l1_z val cũ nhiễu độ dài (cửa sổ A2 100% T vs base 90%) ✗.
7. **Lỗi quy trình cần sửa**: `bench_checkpoint.py` default `--val-manifest` vẫn trỏ manifest **fixgap đã bỏ** ✗; config A1 `pause_pos_weight 3,0` vs văn bản **2,244** ✗; `base_60k/resolved_config.json` ghi `max_steps 40000` trong khi chạy cosine 60000 ✗; §44/README còn số H1 cũ (gap_z 2,3862 — bản fixgap) ✗.

**HÀNG ĐỢI SỬA (ưu tiên mới #1)**: (a) **chấm đầy đủ `best.pt@42k`** (rough ≥60 clip + A/B tai người 42k vs base vs 55k) → có thể xoá toàn bộ tranh luận ✓; (b) sửa cổng: rough ≥60 clip · hiệu chuẩn biên mel theo **đối chứng cùng lịch train (ext57k)** nếu vẫn dùng "không hồi quy" · nêu rõ bộ cổng **trộn 2 val** (6 cổng val cũ + mel_gap/syl_z fixtail) ✗; (c) sửa 4 lỗi quy trình ở mục 7; (d) rồi mới tới E4-rè → W2-duration.

## CHỐT LẠI THEO ĐÚNG LUẬT (sau đính chính)

**`best.pt@42k` = checkpoint A2 được chọn** ✓ — là **mốc DUY NHẤT trong 42/45/50/55/60k đạt MỌI cổng cứng**:
mel_z 1,1645 ✓ · dB 8,8116 ✓ (tốt hơn base) · rough **0,3588** ✓ (≥ biên 0,3579; red-team đo 0,3596@60 clip) · tone_C 0,8002 ✓ · cer **0,1853** ✓ (tốt nhất) · f0_nga 182,0 ✓ · mel_gap 3,0646 ✓ · syl_z 0,6584 ✓ (tốt hơn base).
**Cổng mục tiêu**: length_ratio **0,9114 ĐẠT** ✓ · F1 0,5951 ✗ (cần 0,6289; trần hiệu chuẩn ngưỡng 0,6182) · gMAE 120,15 ✗ (cần 111,7; hiệu chuẩn đưa được 110,8 ✓) · gap_z 2,0… ✗.
⇒ Theo **cây V11.A4**: **"ĐẠT CỨNG, TRƯỢT MỤC TIÊU"** ⇒ nhánh **chuẩn bị A1 (không chạy)** ✓ — nhưng theo red-team: F1 cần **nâng khả năng phân biệt của gate** (A1) vì trần ngưỡng = 0,6182; và trước khi chạy A1 phải sửa `pause_pos_weight` (**2,244**, config đang ghi 3,0 ✗) + `max_steps` 60k ✗ + chấp nhận hiệu ứng extension ✗.
**A/B tai người 42k vs base**: release `demo-audio-20261014-w1b` (6 cặp, kiểm tải 200 + đúng byte ✓) — **cổng cuối** để đổi bản dùng hằng ngày.

## W1c — sửa thiếu sót demo: PHẢI áp sh25 khi nghe A/B

**Thiếu sót của mình**: 2 release trước (`w1` 55k vs base, `w1b` 42k vs base) sinh bằng đường **thẳng model→mel→Vocos, KHÔNG áp sh25** ✗ — không phải cấu hình production (user đã chọn sh25 toàn cục). **sh25 gốc không còn trên máy** ✗ (script cũ không lưu) ⇒ dùng bản tương đương `_unsharp` (unsharp 3×3 box-blur mel z, **α=1,5**) trong `artifacts/vocoder_gta/demo_gta.py` ✓.
**Release mới `demo-audio-20261014-w1c`** (6 cặp base vs 42k, **CÓ sh25**, kiểm tải 200 + sha khác bản thường ✓, RMS 0,039 vs 0,031 ✓).
**Nhận xét tai user trên w1b (không sh25)**: base vs 42k "na ná nhau" ✓ (khớp số: 42k ≈ base trên phần lớn chỉ số) · **vẫn rè** ✗ · **lỗi phát âm/tròn chữ GIỐNG NHAU ở cả 2 model** ✗ ⇒ các lỗi phát âm **không phải do checkpoint** mà ở **frontend/đặc trưng chung** ⇒ **thêm mục mới vào hàng đợi**: thu các câu lỗi cụ thể từ user → chẩn đoán frontend (chuẩn hoá/phiên âm) vs acoustic.
