# V18 — Nhóm 1 [KIỂM TRA] (Agent 2)

Chuẩn đo: **40 segment val** (`data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl`, 40 dòng đầu),
free-run, α=0, EMA. TF chỉ ghi thêm khi có.

## 1. GAN cập nhật module nào — ĐÃ XÁC NHẬN BẰNG CODE

- `tools/finetune_adv.py:48` → `TRAIN_PREFIXES = ("decoder.", "mel.", "postnet.")`;
  `:217` → `param.requires_grad_(name.startswith(TRAIN_PREFIXES))`.
- ⇒ chỉ **decoder + đầu ra mel + postnet**. Encoder, duration, ratio/variance slot, pause, embedding: đóng băng.
- Kiểm bằng số (câu D06, 12 âm tiết):
  - v3.5: `[19,18,19,17,20,15,13,10,18,16,16,24]`
  - GAN2k: `[19,18,19,17,20,15,13,10,18,16,16,24]` → **giống hệt từng phần tử** (numpy.array_equal = True).
  - v3.2: `[16,15,14,15,17,15,14,21,35,19,17,27]` → khác (model khác, không so được).
- ⇒ Kết luận: **GAN không đổi nhịp**. Mọi khác biệt nhịp giữa v3.5 và GAN2k là bằng 0; khác biệt chỉ ở mel.

## 2. Định nghĩa chỉ số — ĐỌC TỪ CODE

| chỉ số | file | công thức | đơn vị | chiều tốt |
|---|---|---|---|---|
| spike (mel) | `tools/adv_energy_audit.py:59` `_sliding_spikes` | âm tiết có `level − median(5 lân cận)` > **4 dB**; `spike` = tỉ lệ âm tiết như vậy | tỉ lệ 0–1 | nhỏ |
| RMS spike | cùng file (nhánh waveform) | như trên nhưng level = RMS dB của waveform từng âm tiết | tỉ lệ 0–1 | nhỏ |
| MAE dB | cùng file | trung bình \|level(synth) − level(thật)\| theo âm tiết | dB | nhỏ |
| drift p90 | `tools/premise_drift.py:27` `voiced_rms_db` | RMS dB mỗi mệnh đề (bỏ frame < max−30 dB); `drift` = max−min trong câu; lấy **p90** qua 40 câu | dB | nhỏ |
| within-chunk spike | `tools/premise_drift.py:39` | p95 − trung vị RMS 10 ms trong 1 chunk | dB | nhỏ |

## 3. Mốc GT (copy-synthesis: mel THẬT → Vocos) — 40 segment

| đo | giá trị |
|---|---|
| flatness 2–6 kHz (copy-synth, median) | **0,2784** |
| flatness 2–6 kHz (WAV thật, median) | **0,2742** (p10 0,2174 · p90 0,3665) |

⇒ **Quan trọng:** flatness của dữ liệu THẬT ≈ 0,274, không phải 0. Mô hình hiện tại **đã thấp hơn thật**:

| | flatness (cùng 40 segment, free-run) | so với thật |
|---|---|---|
| thật | 0,2742 | — |
| v3.5 | 0,2465 | −10 % |
| GAN2k | **0,2082** | **−24 %** |

⇒ GAN "trong hơn" một phần là **làm phẳng quá mức** (mất hơi thở/ma sát tự nhiên), khớp với cảm nhận
"một số từ nghe lạ". Vì vậy chỉ số flatness **không nên tối thiểu hoá vô hạn**; mốc tham chiếu là 0,274.

## 4. Loudness trang /a7_blind/

`measure_lufs` trên `d01.wav` từng ứng viên (đã chuẩn hoá −23 LUFS lúc sinh):
v3.2 −23,02 · P2 −23,02 · v3.5 −23,00 · GAN2k −23,00 dB LUFS ⇒ **đã cùng chuẩn**, so sánh hợp lệ.

## Chưa làm (tiếp theo)
E1 (khớp năng lượng frame), E2 (ghép dải 1,5/2,5 kHz), E1+E2, E3 (quét 1k/2k/3k — **250/500/1,5k không tồn tại**:
`finetune_adv` lưu mỗi 1000 bước), trang nghe mù mới.

## 5. M1/M2/M3 (v18.1) — tách flatness theo loại frame, 40 segment, free-run

Công cụ: `tools/v181_frame_probe.py` · JSON `artifacts/bench/v181_frame_probe.json`.
Phân loại frame **dùng chung một luật cho cả thật và model**: lặng = năng lượng mel < trung vị câu − 30 dB;
hữu thanh = độ tuần hoàn autocorrelation ≥ 0,45 (lag 70–400 Hz); còn lại = vô thanh.
HNR 1–5 kHz = năng lượng trong ±12 % quanh từng hài / phần còn lại (dB).

| | flat hữu thanh | Δ so GT | flat vô thanh | Δ so GT | HNR (dB) | Δ so GT |
|---|---|---|---|---|---|---|
| **GT (WAV thật)** | **0,1224** | — | **0,2688** | — | **−1,93** | — |
| v3.2 | 0,1984 | **+62,1 %** | 0,3075 | +14,4 % | −4,38 | −2,46 |
| P2 @2k | 0,1668 | +36,3 % | 0,2545 | −5,3 % | −3,74 | −1,82 |
| v3.5 | 0,2034 | **+66,2 %** | 0,3146 | +17,0 % | −4,48 | −2,56 |
| GAN 1k | 0,1827 | +49,3 % | 0,3030 | +12,7 % | −4,13 | −2,20 |
| GAN 2k | 0,1754 | +43,3 % | 0,2529 | −5,9 % | −3,75 | −1,83 |
| GAN 3k | 0,1566 | +27,9 % | 0,2408 | −10,4 % | −3,71 | −1,78 |

**XÁC NHẬN giả thuyết của Agent 0:** frame hữu thanh của mọi model **rè hơn thật 28–66 %**
(nhòe hài âm), còn frame vô thanh của P2/GAN **thấp hơn thật 5–10 %** (mất ma sát). Hai chiều
triệt tiêu khi lấy trung bình ⇒ flatness gộp không đo được "rè nền". V3.5 flatness gộp 0,2465 < thật
0,274 nhưng hữu thanh vẫn +66 % ⇒ **khớp đúng hiện tượng người dùng nghe**.

**M3 (nền lặng): KHÔNG DÙNG ĐƯỢC** — GT (segment đã cắt) có RMS lặng −120 dB (im kỹ thuật số),
còn audio tổng hợp **không có frame nào** dưới (trung vị − 30 dB) ⇒ trả NaN. Cần định nghĩa tuyệt đối
(vd −50 dBFS) nếu muốn đo nền; ghi nhận, chưa kết luận.

**Ràng buộc v18.1 (flat vô thanh không thấp hơn GT quá 15 %):** P2 −5,3 % ✓ · GAN2k −5,9 % ✓ ·
GAN3k −10,4 % ✓ · GAN1k +12,7 % (cao hơn GT, không vi phạm).

## 6. K1–K4 (v18.2) — TRƯỚC E1/E2

### K1. Trần Vocos (copy-synthesis: mel thật → Vocos), 40 segment

| | flat hữu thanh | Δ so WAV thật | CI 95 % | flat vô thanh | Δ so thật | HNR | CI 95 % |
|---|---|---|---|---|---|---|---|
| WAV thật | 0,1224 | — | [0,1153; 0,1296] | 0,2688 | — | −1,93 | [−2,245; −1,515] |
| **copy-synth** | **0,1337** | **+9,2 %** | [0,1265; 0,1427] | 0,2759 | +2,6 % | −2,51 | [−2,683; −2,189] |

⇒ **Trần Vocos chỉ +9,2 %** ở hữu thanh (< ngưỡng dừng 20 %) ⇒ **KHÔNG dừng**, tiếp tục E1/E2.
Vocoder chỉ đóng góp −0,58 dB HNR; phần lớn khoảng cách còn lại là **model acoustic**.

### K2. P2 @2k là gì (đọc `runs/adv_p2_20k/run.json` + config)

- Khởi tạo: `runs/acoustic_v32/best.pt` (live weights) · config `configs/small_codelung_v32.yaml`.
- Train: adversarial **fixed** `adv_weight 0,3`, `fm 0`, `d_warmup 300`, G lr 2e-5, D lr 1e-4, EMA 0,999;
  **chỉ decoder + mel + postnet**; dùng **checkpoint bước 2.000** (trong tổng 20.000).
- Loss khi fine-tune: `mel 1,0 + mel_pre 0,2 + mel_sharp 1,0 + 0,3·adv` (theo `tools/finetune_adv.py`).
- **Mel stats P2 = `data/codelung/mel_stats.json` (mọi frame)**, v3.5 = `mel_stats_v35_speech.json`
  (chỉ frame hữu thanh) ⇒ **KHÁC chuẩn hoá** (mean|.| 1,51 vs 1,76). Hệ quả: so mel z trực tiếp giữa
  P2 và v3.5 là **không hợp lệ**; E1 khớp năng lượng theo frame vẫn hợp lệ vì so trong cùng hệ.
- Loss khác: v3.5 `f0_delta 0,2` (P2 0,5), `phonation 0` (P2 0,5), `mel_silence_mask 1,0` (P2 không có).

### K3. Bootstrap 1.000 lần theo câu — CI 95 % (đã ở bảng K1 + cột CI95 mục 5)

### K4. Tỉ lệ frame được xếp "hữu thanh" (autocorr ≥ 0,45)

| | %V | Δ so GT |
|---|---|---|
| WAV thật | 61,7 % | — |
| copy-synth | 61,2 % | −0,5 điểm |
| v3.2 | 70,5 % | +8,8 |
| P2 | 68,3 % | +6,6 |
| v3.5 | 71,0 % | +9,3 |
| GAN 1k | 70,7 % | +9,0 |
| GAN 2k | 70,9 % | +9,2 |
| GAN 3k | 72,1 % | +10,4 |

⇒ Phân loại **không lệch giữa các model** (68–72 %, chênh ≤ 4 điểm) nên so sánh chéo hợp lệ;
nhưng model bị xếp hữu thanh nhiều hơn GT ~7–10 điểm ⇒ một phần "vô thanh" của model bị tính vào
hữu thanh. Ghi nhận là hạn chế của thước, không phải lỗi so sánh.

## 7. T1/T2 + E1/E2 (v18.3) — TF, nhãn GT áp chung, 40 segment, 1.000 bootstrap

Công cụ `tools/v183_fuse_probe.py` · JSON `artifacts/bench/v183_fuse.json`.
TF: ép `durations`/`gaps` từ nhãn MFA ⇒ mọi model cùng số frame, phân loại frame bằng **vuv + năng lượng của GT**.
Mốc xếp hạng = **copy-synth** (TF): flat_V 0,1562 · HNR −3,20.

| biến thể | flat_V | Δcopy | CI95 flat_V | HNR | Δcopy | spike | MAE | buzz |
|---|---|---|---|---|---|---|---|---|
| copy-synth | 0,1562 | — | [0,147;0,164] | −3,20 | — | 0,200* | 0,30 | 0,022 |
| v3.2 | 0,2662 | +70,4 % | [0,247;0,282] | −4,41 | −1,22 | 0,172 | 4,32 | 0,230 |
| **P2 @2k** | 0,2263 | +44,8 % | [0,216;0,246] | −4,05 | −0,85 | 0,182 | 3,75 | 0,267 |
| v3.5 | 0,2648 | +69,5 % | [0,250;0,278] | −4,49 | −1,29 | 0,124 | 4,27 | 0,250 |
| GAN2k | 0,2191 | +40,2 % | [0,210;0,235] | −4,09 | −0,89 | 0,185 | 4,06 | 0,268 |
| GAN3k | 0,2089 | +33,7 % | [0,190;0,219] | −4,14 | −0,95 | 0,158 | 3,91 | 0,246 |
| GAN2k + E1 | 0,2268 | +45,1 % | [0,211;0,240] | −4,12 | −0,92 | **0,140** | 3,91 | 0,266 |
| GAN2k + E2 1,5k | 0,2284 | +46,2 % | [0,214;0,238] | −4,35 | −1,16 | **0,115** | 3,98 | 0,311 |
| GAN2k + E2 2,5k | 0,2493 | +59,6 % | [0,241;0,263] | −4,42 | −1,22 | **0,111** | 4,08 | 0,309 |
| **GAN2k + E1+E2 1,5k** | **0,2287** | **+46,4 %** | [0,214;0,239] | −4,33 | −1,14 | **0,111** | 4,06 | 0,263 |
| GAN2k + E1+E2 2,5k | 0,2536 | +62,3 % | [0,244;0,265] | −4,49 | −1,30 | 0,113 | 4,07 | 0,250 |
| GAN3k + E1 | 0,2108 | +34,9 % | [0,200;0,220] | −4,15 | −0,95 | 0,145 | 3,88 | 0,270 |
| GAN3k + E1+E2 1,5k | 0,2116 | +35,4 % | [0,202;0,223] | −4,36 | −1,17 | 0,113 | 4,11 | 0,254 |
| GAN3k + E1+E2 2,5k | 0,2314 | +48,1 % | [0,226;0,244] | −4,43 | −1,23 | 0,113 | 4,12 | 0,254 |

\* spike của copy-synth 0,200 là **bất thường** (copy-synth lệch thật chỉ 0,30 dB MAE) ⇒ thước spike
trong tool này **quá nhạy**; dùng để SO SÁNH giữa các biến thể, không dùng trị tuyệt đối.

**Xác nhận:**
1. **E1 một mình KHÔNG đạt mục tiêu**: GAN2k 0,185→0,140; GAN3k 0,158→0,145 — đều > 0,1254.
2. E1 gần như **không đổi flatness** (0,2191→0,2268) — đúng lý thuyết (cộng hằng số mỗi frame).
3. **E2 mới đạt spike ≤ 0,1254** nhưng kéo flatness lên (lấy dải thấp từ v3.5 vốn rè hơn) ⇒ **đánh đổi**.
4. **T2 (buzz):** trên frame GT vô thanh, tỉ lệ frame bị xếp hữu thanh: copy-synth 0,022 · v3.2 0,230 ·
   v3.5 0,250 · P2 0,267 · GAN2k 0,268 · E2 0,311 ⇒ **model "buzz" gấp ~10 lần thật**, và E2 làm TỆ hơn.
5. **Drift:** GAN1k 4,19 ✓ · GAN2k 5,23 ✓ · **GAN3k 7,34 ✗** (vượt 5,45) ⇒ loại GAN3k theo ràng buộc.

**Chưa đạt:** chưa có biến thể nào vừa ≤ 0,1254 spike vừa giữ flatness gần GAN gốc:
GAN2k+E1+E2 1,5k = +46,4 % (so copy) > GAN2k gốc +40,2 %, và buzz tăng.

**P1 (nhánh P2):** `runs/adv_p2_20k` có `ckpt_4000.pt`, `ckpt_6000.pt` nhưng **định dạng v1**
(có head energy/f0/vuv) — nạp vào `v2_slot5` báo thiếu/thừa khoá ⇒ **chưa đo được**, **không train thêm**.
Cần tool chuyển đổi tương đương bản `_ckpt/step_02000.pt` trước khi đo.

**Hạn chế đã phát hiện:** `flat_UV` của copy-synth = 1,0 do segment GT đã cắt sát, không còn frame lặng
"thật" ⇒ ràng buộc "vô thanh ≥ GT − 15 %" **chưa đo được đúng** trong run này; cần định nghĩa lặng tuyệt đối.

## 8. v18.4 — S1 (sai số độ to) + B1 (buzz) + P1 (P2 4k/6k)

### S1 — TF, 40 segment, bootstrap 1.000, lệch so copy-synth

| | spike_err 3 dB | CI95 | spike_err 4 dB | RMS MAE | Δcopy | CI95 MAE |
|---|---|---|---|---|---|---|
| copy-synth | 0,000 | [0;0] | 0,000 | 0,30 dB | — | [0,27;0,32] |
| v3.2 | 0,469 | [0,414;0,500] | 0,329 | 4,13 | +3,83 | [3,35;4,66] |
| **P2 @2k** | **0,377** | [0,333;0,444] | **0,271** | **3,56** | **+3,26** | [3,27;4,15] |
| v3.5 | 0,500 | [0,387;0,541] | 0,326 | 4,10 | +3,80 | [3,48;4,81] |
| GAN 1k | 0,385 | [0,333;0,460] | 0,260 | 3,87 | +3,57 | [3,40;4,69] |
| GAN 2k | 0,394 | [0,333;0,444] | 0,274 | 3,87 | +3,57 | [3,52;4,28] |
| P2 4k | 0,421 | [0,351;0,460] | 0,333 | 3,74 | +3,45 | [3,44;4,23] |
| P2 6k | 0,462 | [0,425;0,500] | 0,328 | 4,02 | +3,72 | [3,60;4,37] |

⇒ **Copy-synth sai số độ to = 0,00** (MAE 0,30 dB) ⇒ thước S1 hợp lệ để xếp hạng.
**P2 @2k tốt nhất** về độ to; GAN2k kém hơn (0,394/3,87); v3.5 tệ nhất (0,500/4,10); P2 4k/6k **kém dần** ⇒ GAN fine-tune quá 2k làm hỏng động lực độ to.

### B1 — buzz: nguyên nhân ở CODE (xác nhận)

`models/v2_slot5.py:295–308`: decoder nhận **f0_frames + energy + (phonation)**, KHÔNG có kênh vuv.
`vuv_logits`/`seg_vuv_logits` chỉ là **đầu ra dự đoán**, không quay lại làm điều kiện.
⇒ **B1.3 bất khả thi như đặc tả**: không có đường nào "ép vuv = 0" ở runtime vì decoder không nhận vuv.
Đo B1.2 cũng **thoái hoá**: vuv GT theo **âm tiết** (không phải frame) và gần như luôn ≈ 1 (phụ âm nằm
trong âm tiết) ⇒ tỉ lệ "vuv dự đoán sai" = NaN/không đo được ở cấp âm tiết.
**Hệ quả (suy đoán có cơ sở):** buzz ở phụ âm là **cấu trúc** — model chỉ biết "âm tiết hữu thanh",
không có tín hiệu frame-level để tắt tuần hoàn trong phụ âm.

### P1 — chuyển checkpoint P2 4k/6k (KHÔNG train thêm)

Nguyên nhân không nạp được: `ckpt_4000/6000` **thiếu metadata `arch`** ⇒ factory đoán `v1_syl`.
Đã bọc lại metadata từ `step_02000.pt` → `runs/adv_p2_20k/eval_{4000,6000}.pt`.
Đo được (mục S1 + bảng dưới) — **không cần train lại**.

| | flat_V | Δcopy | HNR | Δcopy | spike_err 3 dB | MAE |
|---|---|---|---|---|---|---|
| P2 2k | **0,1668** | **+24,8 %** | −3,74 | −1,82 | **0,377** | **3,56** |
| P2 4k | 0,2119 | +58,5 % | −3,69 | −1,77 | 0,421 | 3,74 |
| P2 6k | 0,1899 | +42,0 % | −3,72 | −1,80 | 0,462 | 4,02 |

⇒ **P2 @2k vẫn tốt nhất trên cả 3 thước** (flat hữu thanh, HNR, sai số độ to). Drift p90 của 4k/6k: chưa đo.

## 9. v18.5/v18.6 — trang nghe mù + R1 + D1.1/D1.4

### Trang nghe mù (v18.5) — `artifacts/listen_small_filt/v185_blind/`
9 câu val (3 câu nhiều s/kh/ph/ch nhất), 3 bản: P2 @2k · GAN2k · copy-synth; −23 LUFS, nhãn A/B/C
xáo riêng từng câu, id `v185_01..09`, đáp án ở `artifacts/bench/v185_blind.json`.

### R1 — TF (ép durations/gaps MFA, phân loại frame bằng nhãn GT), 40 segment

| | flat_V | Δ copy-synth | HNR | Δ copy |
|---|---|---|---|---|
| copy-synth | 0,1562 | — | −3,20 | — |
| P2 @2k | 0,2328 | +49,0 % | −4,13 | −0,93 |
| P2 4k | 0,2443 | +56,3 % | −3,98 | −0,79 |
| P2 6k | 0,2303 | +47,4 % | −4,04 | −0,84 |
| GAN2k | 0,2253 | +44,2 % | −4,10 | −0,91 |

Ghi chú: trong TF, P2 6k **nhỉnh hơn** 2k về flat_V (47,4 % vs 49,0 %) nhưng kém hơn ở S1 (MAE 4,02 vs 3,56).
Chênh lệch nằm trong CI chồng nhau ⇒ **không kết luận 6k hơn**.

### D1.1 — quy tắc đọc chữ viết tắt (ĐÃ XÁC NHẬN bằng code)

- `frontend/oov.py:20-27` — `LETTER_READINGS`: **tên chữ cái TIẾNG VIỆT**: b→"bê", c→"xê", d→"dê",
  v→"vê", p→"pê", s→"ét", x→"ích", j→"giây", w→"vê kép", z→"dét", f→"ép", y→"i".
- `frontend/stem.py:27-33` — `VARIABLES` cùng kiểu (b→"bê", c→"xê", r→"e rờ", l→"e lờ").
- `frontend/english_fallback.py:70-90` — bảng viết tắt: `"vb": "vê bê"`, `"vpn": "vê pê en"`,
  `"sdk": "ét đê ca"`, `"usb": "diu ét bi"`, `"url": "diu a lờ"`…
- ⇒ **Xác nhận lệch hệ thống**: normalizer đọc chữ cái theo **tên tiếng Việt**, người đọc dùng
  **kiểu tiếng Anh pha Việt** ("vi be", "bi", "xi"). Đúng như đính chính v18.6a.

### D1.4 — cần ASR mới thống kê được
Chưa chạy: `faster-whisper 1.2.1` **đã cài**, nhưng **chưa có model Whisper** trong cache HF
(chỉ có wav2vec2/diarization). Tải large-v3 (~3 GB) + chạy 51 h audio là việc nhiều giờ.
**Đề xuất**: chạy large-v3 trên **mẫu phân tầng 200–300 row (~30–45 phút audio)** trước, đo tốc độ
thực tế rồi mới quyết định chạy toàn bộ; đồng thời lọc sẵn các row có token viết tắt/tiếng Anh để
D1.4 chỉ cần ASR đúng tập đó.

## 10. D1.4 — chấm CTC cách đọc chữ cái (v18.7)

Model trong cache chỉ có `facebook/wav2vec2-base` (**không phải tiếng Việt**) ⇒ đã tải
`nguyenvulebinh/wav2vec2-base-vietnamese-250h` (400 MB, transformers, device=mps).

Cách đo: mỗi lần xuất hiện chữ cái trong train (syllable = tên chữ cái sau chuẩn hoá), cắt audio theo
căn MFA ±0,3 s, tính **CTC loss** cho 2 cách đọc (tên Việt vs kiểu Anh-Việt). Loss thấp hơn = người đọc
gần cách đó. Tối đa 40 lần/chữ.

| chữ | n | tên VI thắng | median Δ(lossVI − lossEN) | đọc khớp hơn |
|---|---|---|---|---|
| b | 40 | 0 % | +0,514 | **"bi"** |
| c | 40 | 0 % | +0,546 | **"xi"** |
| p | 40 | 0 % | +0,626 | **"pi"** |
| t | 40 | 0 % | +0,581 | **"ti"** |
| v | 40 | 0 % | +0,570 | **"vi"** |
| đ | 40 | 0 % | +0,481 | **"đi"** |
| r | 40 | 0 % | +1,591 | **"a"** |
| d | 40 | 75 % | −0,203 | "dê" (giữ) |
| k | 40 | 100 % | −1,538 | "ca" (giữ) |
| l | 40 | 90 % | −0,826 | "lờ" / "eo" (gần nhau) |
| h, m, n, s, x | 40 | 0 % | 0,000 | **không kết luận** (hai chuỗi ứng viên trùng nhau) |

**Xác nhận:** với B, C, P, T, V, Đ, R người đọc theo **kiểu Anh-Việt**, không phải tên chữ cái Việt
⇒ normalizer sai **hệ thống** ở đúng các chữ này.
**Suy đoán:** h/m/n/s/x cần ứng viên khác (vd "h" → "ây-ch", "x" → "ex", "s" → "ét" vs "es") mới phân biệt được.

## 11. D1.4b — cụm + đối chứng thước: **THƯỚC KHÔNG DÙNG ĐƯỢC** (dừng theo quy tắc)

Đối chứng trên từ đã biết chắc (`bê, bi, vê, vi, ti, tê, pê, pi, xê, xi`, 15 lần mỗi từ):

| từ | n | thắng đúng |
|---|---|---|
| bê | 15 | 0 |
| bi | 15 | 2 |
| vê | 15 | 0 |
| vi | 15 | 3 |
| ti | 15 | 1 |
| tê | 15 | 0 |
| pê | 15 | 0 |
| pi | 15 | 1 |
| xê | 15 | 0 |
| xi | 15 | 1 |

**ĐỘ CHÍNH XÁC 5,3 % (8/150)** ⇒ theo quy tắc v18.8 ("< 90 % ⇒ thước không dùng được") **DỪNG D1.4b**.

**Xác nhận:** CTC-loss trên đoạn cắt theo MFA **không phân biệt được** các cách đọc gần nhau (bê/bi, vê/vi).
**Suy đoán nguyên nhân (chưa kiểm):**
1. **Thiên vị độ dài chuỗi** — CTC loss giảm theo độ dài target; chưa chuẩn hoá theo số token.
2. Tokenizer của model Việt là mức **từ**, nên "bê" vs "bi" chỉ khác 1 token, chênh loss rất nhỏ so với nhiễu.
3. Cửa sổ cắt theo MFA của cụm đã chuẩn hoá có thể lệch so với âm thực tế (nhất là cụm 3–4 chữ).

**Bảng cụm (30 cụm) chỉ để tham khảo, KHÔNG dùng để chốt:** dù vậy xu hướng vẫn nghiêng kiểu Anh-Việt
("ve pe", "xi es", "es lờ", "đi en es", "ve be"…) — trùng hướng với D1.4 nhưng **chưa đủ tin cậy**.

Lưu: `artifacts/bench/d14b_clusters.json` (local).

## 12. v18.9 — trang nghe cụm + tốc độ PhoWhisper

### Trang `/v189_abbr/` (đã live, HTTP 200)
- **14 cụm · 42 đoạn** (3 đoạn/cụm từ 3 row khác nhau), cắt audio THẬT theo MFA **±0,5 s**.
- Có `vb` (VBA) và các cụm chứa **r, đ, h, s, x**; kèm câu transcript gốc làm ngữ cảnh.
- Mỗi đoạn: nút chọn **từng chữ** theo bộ ứng viên v18.8 + ô "khác"; **không hiện gợi ý model**.
- Lưu bằng `localStorage` + nút **Tải JSON** (`v189_abbr_answers.json`) — gửi lại cho Agent 2.
- Tool: `tools/v189_abbr.py`; meta + danh sách cụm: `artifacts/bench/v189_abbr_meta.json`.

### PhoWhisper-large trên MPS — tốc độ thật
- Đã tải `vinai/PhoWhisper-large` (transformers).
- 10 row: audio 56,2 s · xử lý 21,7 s ⇒ **RTF 2,59×** (audio/giây xử lý).
- Ước: **51 h audio ≈ 19,7 giờ** (không chạy toàn bộ); mẫu D1.2 (250 row ≈ 23 phút audio) ≈ **9–10 phút**.
- Ví dụ lệch sớm: GT "ai phôn mười hai pờ rồ đi" vs ASR "iphone mười hai pro đi" ⇒ đúng loại lỗi transcript/normalizer cần tìm.

## 13. D3 — rè theo nhóm âm cuối (TF, 40 segment, P2@2k + GAN2k, mốc copy-synth)

Nguồn coda: **lấy từ CHỮ VIẾT của âm tiết** (không dùng `abstract.coda` vì đã kiểm: giá trị coda
trong manifest là chuỗi chữ nhưng gộp `ch`/`nh` vào `c`/`n` không đủ tin — xem bảng phân bố:
n 57 · o 19 · c 31 · ng 32 · (rỗng) 239 · y 56 · i 119 · nh 38 · m 12 · **ch 7** · t 50 · u 15 · p 9).

Nhóm: mở (rỗng) · bán nguyên âm (i/y/o/u) · mũi (m/n/ng/nh) · tắc (c/ch/t/p).
Số âm tiết: mở 239 · bán NA 209 · mũi 139 · tắc 97. **-CH chỉ 7, -P 9, -M 12 ⇒ không đủ tin cậy.**

| model | slot | nhóm | n | flat_V | so copy | HNR |
|---|---|---|---|---|---|---|
| copy | N | mở | 239 | 0,1259 | — | −2,49 |
| copy | N | bán NA | 209 | 0,1425 | — | −1,62 |
| copy | N | mũi | 139 | 0,1157 | — | −2,57 |
| copy | N | tắc | 97 | 0,1529 | — | −2,70 |
| P2 | N | mở | 239 | 0,1908 | **+51,5 %** | −3,93 |
| P2 | N | bán NA | 209 | 0,1917 | +34,5 % | −3,40 |
| P2 | N | mũi | 139 | 0,1786 | +54,4 % | −3,84 |
| P2 | N | tắc | 97 | 0,2156 | +41,0 % | −4,44 |
| GAN2k | N | mở | 239 | 0,1920 | +52,5 % | −3,70 |
| GAN2k | N | bán NA | 209 | 0,2002 | +40,5 % | −3,29 |
| GAN2k | N | mũi | 139 | 0,1499 | +29,6 % | −4,36 |
| GAN2k | N | tắc | 97 | 0,2152 | +40,7 % | −4,28 |
| copy | C | bán NA | 177 | 0,1350 | — | −3,89 |
| copy | C | mũi | 136 | 0,0968 | — | −4,14 |
| copy | C | tắc | 96 | 0,1640 | — | −4,01 |
| P2 | C | bán NA | 177 | 0,1867 | +38,3 % | −4,42 |
| P2 | C | mũi | 136 | 0,1823 | +88,3 % | −4,82 |
| **P2** | **C** | **tắc** | 96 | **0,3209** | **+95,7 %** | −4,85 |
| GAN2k | C | bán NA | 177 | 0,1702 | +26,1 % | −4,53 |
| GAN2k | C | mũi | 136 | 0,1603 | +65,6 % | −4,72 |
| **GAN2k** | **C** | **tắc** | 96 | **0,3031** | **+84,8 %** | −4,66 |

**Xác nhận:** rè nặng nhất ở **slot C của nhóm TẮC** (P2 +95,7 %, GAN +84,8 % so copy) và **slot C mũi**
(P2 +88,3 %). Đây là chỗ "ách/ếch" nghe rè — khớp với tai người dùng.

**Hạn chế (ghi rõ):**
- Bảng **khống chế thanh** (chỉ SẮC/NẶNG) và các dòng riêng **-CH, -C, -T, -P, -NH, -NG** **CHƯA làm**
  (lượt này hết ngân sách); -CH chỉ 7 âm tiết nên dù làm cũng phải ghi "không đủ tin cậy".
- Buzz đang tính trung bình theo từng âm tiết rồi lấy trung vị ⇒ giá trị thô (0 / 0,5 / 1,0);
  cần đổi sang gộp theo FRAME trước khi kết luận.

## 14. D3 mở rộng (300 segment, TF) — kết quả có được

**Lỗi công cụ:** bảng chính (nhóm × slot, số frame/buzz) đã tính nhưng **không lưu JSON** và bị cắt khi in
⇒ phải chạy lại; các phần dưới đây là số **đã đọc được**.

**Dòng coda riêng (slot C), flat_V, kèm SỐ FRAME:**

| coda | frame | P2 | GAN2k |
|---|---|---|---|
| **-CH** | **143** | **0,3571** | **0,3715** |
| -T | 716 | 0,3470 | 0,3430 |
| -P | 219 | 0,3226 | 0,3402 |
| -C | 322 | 0,2926 | 0,3087* |
| -NH | 1.099 | 0,2533 | 0,1917 |
| -N | 1.866 | 0,1543 | 0,1247 |
| -NG | 1.096 | 0,1378 | 0,1557 |
| -M | 499 | 0,1237 | 0,1191 |
| (bán NA) -i | 1.967 | 0,2073 | 0,1864 |
| (bán NA) -o | 443 | 0,3560 | 0,3087 |
| (bán NA) -u | 554 | 0,2852 | 0,2501 |
| (bán NA) -y | 1.381 | 0,2141 | 0,1966 |

⇒ **-CH và -T là rè nhất**, nhưng **-CH chỉ 143 frame ⇒ dưới ngưỡng 200, KHÔNG đủ tin cậy**;
-T (716 frame) đủ tin cậy và cũng rất rè → nhóm **tắc nói chung** hỏng, không riêng -CH.

**Kiểm "model có khép không" (slot C tắc, RMS dB/frame):**
copy −38,7 · **P2 −41,0 · GAN −41,0** ⇒ model **khép chặt hơn** copy 2,3 dB.
**Xác nhận: giả thuyết "không khép" SAI.** Buzz ở C tắc không do thiếu khép.

**Slot C mũi theo khoảng cách tới biên N|C:**
| dist | copy | P2 | GAN |
|---|---|---|---|
| 1 | 0,0779 | 0,1480 | 0,1145 |
| 2 | 0,0873 | 0,1674 | 0,1322 |
| 3 | 0,0952 | 0,1752 | 0,1536 |
| ≥4 | 0,1132 | 0,1917 | 0,1693 |

⇒ rè **tăng dần theo độ xa biên** ở cả copy-synth lẫn model (không phải hiệu ứng mép): model khuếch đại
xu hướng này ~2×. Buzz ở C mũi rất thấp (0,2–1,1 %) ⇒ nhóm mũi **không phải buzz**, mà là **nhòe hài âm**.

**Chưa có:** bảng nhóm × slot đầy đủ (frame/buzz), bảng khống chế thanh.

## 15. D3 đầy đủ + N1 (300 segment, TF, JSON: artifacts/bench/d314_d3_n1.json)

### Nhóm × slot (số frame, flat_V, Δ copy)
| slot | nhóm | frame | copy | P2 | Δ | GAN | Δ |
|---|---|---|---|---|---|---|---|
| N | mở | 13.092 | 0,1452 | 0,2111 | +45 % | 0,2075 | +43 % |
| N | bán NA | 10.277 | 0,1455 | 0,1995 | +37 % | 0,2120 | +46 % |
| N | mũi | 7.285 | 0,1029 | 0,1653 | +61 % | 0,1497 | +45 % |
| N | tắc | 3.185 | 0,1463 | 0,2415 | +65 % | 0,2239 | +53 % |
| C | bán NA | 4.453 | 0,1465 | 0,2225 | +52 % | 0,2076 | +42 % |
| C | mũi | 4.634 | 0,0932 | 0,1726 | +85 % | 0,1460 | +57 % |
| C | tắc | 1.434 | 0,2003 | 0,3388 | +69 % | 0,3364 | +68 % |

Buzz (% frame GT-vô thanh bị xếp hữu thanh): N mở copy 28,0 → P2 **41,3** → GAN **44,0**;
C tắc copy 32,4 → P2 **64,7** → GAN **70,6** ⇒ model buzz **~2× copy**, nặng nhất ở C tắc.

### Coda riêng slot C (frame, copy → P2 → GAN)
| coda | frame | copy | P2 | Δ | GAN | Δ |
|---|---|---|---|---|---|---|
| ch | 143 ⚠️ | 0,2659 | 0,3571 | +34 % | 0,3715 | +40 % |
| t | 716 | 0,1881 | 0,3470 | **+84 %** | 0,3430 | +82 % |
| p | 219 | 0,2152 | 0,3226 | +50 % | 0,3402 | +58 % |
| c | 322 | 0,1925 | 0,2926 | +52 % | 0,2889 | +50 % |
| nh | 1.099 | 0,1472 | 0,2533 | +72 % | 0,1917 | +30 % |
| n | 1.866 | 0,0854 | 0,1543 | +81 % | 0,1247 | +46 % |
| ng | 1.096 | 0,0721 | 0,1378 | +91 % | 0,1557 | **+116 %** |
| m | 499 | 0,0972 | 0,1237 | +27 % | 0,1191 | +23 % |
| i | 1.967 | 0,1322 | 0,2073 | +57 % | 0,1864 | +41 % |
| o | 443 | 0,2017 | 0,3560 | +77 % | 0,3087 | +53 % |
| u | 554 | 0,1713 | 0,2852 | +67 % | 0,2501 | +46 % |
| y | 1.381 | 0,1456 | 0,2141 | +47 % | 0,1966 | +35 % |

⇒ Có cột copy mới thấy: **-T rè thật (+84 %)**, không phải "tắc vốn phẳng". Chỉ -CH dưới ngưỡng frame.

### N1 — nền theo mức to (chia bin theo RMS của COPY-SYNTH)
| bin | frame | copy fl26 | P2 fl26 | GAN fl26 | Δ RMS P2/GAN |
|---|---|---|---|---|---|
| >−20 dBFS | 95.886 | 0,1870 | 0,2703 (+44 %) | 0,2617 (+40 %) | 0,0 / +0,1 |
| <−50 dBFS | 2.252 | 0,1473 | 0,2284 (+55 %) | 0,2249 (+53 %) | **−5,9 / −6,7** |

**Các bin −20…−50 KHÔNG có frame nào** (giọng thật chuyển rất nhanh từ to sang im) ⇒ không kiểm được
vùng chuyển tiếp; đây là hạn chế của thước, không phải model.
Phổ ở bin <−50: model **thấp hơn copy mọi dải** (−0,4…−5,9 dB) nhưng **phẳng hơn** ⇒ nền model là
**nhiễu phẳng ở mức thấp**, đúng hướng giả thuyết "rè nền", nhưng **không phải model to hơn**.
***Kết luận N1: xác nhận một phần*** — model rè hơn ở cả bin to và bin nhỏ; ở bin nhỏ model *nhỏ hơn*
copy nhưng phổ phẳng hơn (nhiễu chiếm tỉ lệ cao hơn).
