# V18 — Nhóm 1 [KIỂM TRA] (Agent 2)

> **Cập nhật v18.37:** luật hiện hành chỉ còn **BAK + UTMOS**; **CER/WER đã bị loại khỏi luật**
> (giữ số cũ trong tài liệu làm lịch sử). Production vẫn **P2 EMA**; xem `configs/production_v1831.json`
> (metadata v18.37) và `V1837_agent2.md`. Các kết luận free-run cũ bị giới hạn bởi punctuation/EMA-live/
> vocoder mismatch — không xóa lịch sử, chỉ hạ mức áp dụng.

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

## 16. Q1/Q2 sửa lỗi đo (300 segment, TF, JSON: artifacts/bench/q12_n1_buzz.json)

### Q1 — công thức RMS và bin
- Công thức: `20*log10(sqrt(mean(x^2)))` trên **waveform**, full-scale = 1.0 ⇒ đúng dBFS.
- **Lỗi thật:** điều kiện bin viết `lo < rms <= hi` với dải sắp xếp giảm dần ⇒ mọi frame −20…−50 rơi
  vào nhánh mặc định `b=4`. Nay sửa: `b = 0 if rms>-20 else 1 if rms>-30 else 2 if rms>-40 else 3 if rms>-50 else 4`.
- Histogram 2 dB và bảng N1 đầy đủ đã ghi JSON (phần in bị cắt).

### N1 sau khi sửa (đọc được bin < −50 dBFS)
| model | frame | flat_V | Δ RMS so copy |
|---|---|---|---|
| P2 | 25.398 | **0,4391** | **+12,6 dB** |

⇒ **Đảo kết luận trước:** ở frame rất nhỏ, model **to hơn copy 12,6 dB** và **phẳng hơn nhiều** (0,44 so
copy ~0,15–0,2) ⇒ **nền model đúng là nhiễu phẳng** (giả thuyết "rè nền" ĐƯỢC XÁC NHẬN).

### Q2 — buzz, định nghĩa thống nhất (acoustic, từ WAV thật)
GT vô thanh = frame có **autocorr của WAV THẬT < 0,45**; buzz = frame đó mà model có autocorr ≥ 0,45.
Copy-synth theo định nghĩa này = **0,0 %** (đúng, vì dùng chính WAV thật làm chuẩn).

| | C bán NA | C mũi | C tắc | N mở | N bán NA | N mũi | N tắc |
|---|---|---|---|---|---|---|---|
| P2 | 45,5 % | **67,7 %** | 41,6 % | 45,2 % | 43,9 % | 37,5 % | 49,1 % |
| GAN | 46,7 % | **72,7 %** | 44,3 % | 49,5 % | 42,7 % | 34,8 % | 52,1 % |

⇒ Sửa lại so với D3: **buzz nặng nhất ở C MŨI (68–73 %)**, không phải C tắc. Số cũ (28–32 %) sai vì
định nghĩa "GT vô thanh" theo vuv âm tiết, không phải theo WAV thật.

## 17. G1 — thiết kế GTA fine-tune Vocos (CHƯA train)
- **Dữ liệu:** mel P2@2k chế độ TF (duration MFA) cho **toàn bộ 18.226 row train**, ghép audio thật cùng row;
  **300 segment val giữ riêng** (không đưa vào train).
- **Căn frame–audio:** mel TF có `T` frame ứng với `(T-1)*256 + 1024` mẫu; phải kiểm từng row khớp trước khi ghép
  (lệch ⇒ bỏ row đó, ghi số bỏ).
- **Khởi tạo:** Vocos hiện tại; **giữ nguyên loss gốc** (mel + multi-res STFT + GAN của vocoder), lr thấp.
- **Đánh giá:** đúng bảng D3/coda (flat_V, HNR theo slot so copy-synth) + S1 + drift.
- **Dung lượng:** 15,8 triệu frame × 100 bin × 2 byte (fp16) ≈ **3,2 GB** (+ audio gốc đã có).
- **Thời gian:** 0,653 s/bước (b8·crop64·D ch16 đã đo) ⇒ 5k bước ≈ **55 phút**, 10k ≈ **1,8 giờ** trên Mac mini.
- **Rủi ro:** chỉ hợp với P2 (mel z-space của v3.2); đổi acoustic model ⇒ **phải GTA lại**; nếu Vocos quen
  với nhiễu của P2, nó có thể **học luôn rè** thay vì khử ⇒ phải so với copy-synth chứ không chỉ nghe.

## 18. G0 (thử runtime) + N1 đầy đủ — KẾT QUẢ ÂM TÍNH

JSON: `artifacts/bench/g0.json` (300 segment, TF).

**N1 (cùng một định nghĩa bin cho mọi bản), bin < −50 dBFS:**
| | frame | flat_V |
|---|---|---|
| copy-synth | 25.398 | **0,4828** |
| P2 | 26.798 | 0,4698 |
| GAN | 25.712 | 0,4606 |

⇒ **Sửa lại:** ở bin < −50, flatness của model **KHÔNG cao hơn copy** (0,47 vs 0,48). Khác biệt thật là
**mức to**: P2 **+12,6 dB** so copy (đã đo ở v17). Vậy "rè nền" = **nền quá to**, không phải nền quá "phẳng".

**G0 (thay frame nhỏ bằng profile nền trung bình):**
- Ngưỡng q5 = **−70 dB** ⇒ không thay frame nào (kết quả y hệt bản gốc).
- Ngưỡng q15 = **−25,9 dB** ⇒ thay quá nhiều: flat_V P2 **0,2437 → 0,2518** (tệ hơn), và bin < −50
  thành **flat = 1,0** (hằng số ⇒ nhiễu phẳng tuyệt đối).
⇒ **G0 bản này THẤT BẠI**. Hướng đúng phải là **giảm mức** frame nhỏ (~−12 dB) chứ không thay bằng profile.

**Ghi chú:** ngưỡng năng lượng của mel z-space không tương ứng trực tiếp dBFS của waveform (q5 = −70 dB
trong khi waveform chỉ xuống −80), nên cần hiệu chỉnh ngưỡng theo phân bố RMS thật trước khi thử lại.

## 19. C) mel_silence_mask — đọc code + G0 v2 (expander) (JSON: artifacts/bench/g0v2.json)

### C — code
`models/v2_losses.py:136-152`: khi `loss_weights.mel_silence_mask > 0` và batch có `vuv`,
mel L1 được nhân trọng số `_speech_frame_weight(...)` ⇒ **frame lặng gần như KHÔNG bị phạt**.
⇒ Xác nhận: model **không bị phạt khi phát tiếng ở chỗ phải im**.

**Sửa lại thông tin trong v18.17:** cấu hình **v3.5 BẬT** mask (`mel_silence_mask: 1.0`),
còn **P2/v3.2 KHÔNG có khoá này** (mặc định 0) ⇒ **mask không thể là nguyên nhân nền +12,6 dB của P2**;
phải kiểm chéo Δ RMS bin < −50 của v3.5 (chưa đo).

### G0 v2 — downward expander trên waveform (TF, 300 segment)
Ngưỡng −45 dB (1:2) và −40 dB (1:3), attack 2 ms, release 30 ms, max −12 dB, tất định.

| | flat_V hữu thanh | N1 < −50: frame | N1 flat |
|---|---|---|---|
| copy-synth | 0,1569 | 21.493 | **0,4296** |
| P2 gốc | 0,2437 | 26.798 | 0,4698 |
| P2 + exp −45/1:2 | 0,2441 | 28.444 | 0,4621 |
| P2 + exp −40/1:3 | 0,2448 | 30.365 | 0,4542 |

⇒ Expander **gần như không đổi flat_V** (đúng: flatness bất biến theo mức), chỉ kéo N1 flat về phía copy
một chút. **Chưa đo Δ RMS từng bin** (thiếu) ⇒ chưa kết luận được mức nền đã về 0 dB chưa.

9 WAV mẫu (−23 LUFS) đã ghi vào `artifacts/listen_small_filt/v1817_floor/`.

## 20. W1 (gate) + V1/V3 (miền mel) + G1 GTA 5k + eval (v18.27–v18.28)

JSON: `artifacts/bench/v1827_domain.json` · `w1_gate.json` · `v1828_base_tf.json` · `v1828_gta_tf.json` · `v1828_gta_copy.json` · `v1828_free.json`.
Công cụ: `tools/v1827_domain_probe.py` · `tools/w1_gate_probe.py` · `tools/v1828_gta_eval.py` · `tools/v1828_listen_page.py`.

### V1 — miền mel (đính chính v18.25)
- Production truyền `sharpen_mel(mel, 0)` = **chính tensor model** (`infer.py:327`); α=0 ⇒ `max|diff| = 0`.
- `VocosGenerator` nhận **z** (`models/vocos.py:197` → `denormalize` `:160`), dựng với `stats_path` (`istftnet.py:228`).
- L1(model mel, mel thật z) = **0,992** vs với log thô **2,355** ⇒ model mel là z ✓.
- **Mốc copy-synth đúng miền loss = 0,133** (median/row 0,129) — số 0,709 trước đó sai (miền z + đầu vào là mel P2).

### V3 — sàn trong cùng miền acoustic z (300 val, 90.466 frame)
- p1 model **−4,68** vs thật **−11,29**; p5 **−3,39** vs **−4,23**; min −16,23 vs −15,58.
- L1: toàn bộ 0,992 · có tiếng 0,704 · **lặng 1,790** (26,6 % frame lặng); dịch k: đáy **k=0** (0,7302), đối xứng ✓.
- ⇒ Model **không kẹt sàn**, nhưng **không tạo được nhóm nhỏ frame cực im** của giọng thật (đúng đính chính v18.28).

### W1 — noise gate (2.300 row: 2.000 train + 300 val; 666.291 frame)
- **3,64 % frame im kỹ thuật số** (RMS < −90 dBFS); 22,6 % nền nhiễu (−90…−50).
- Tập trung theo video: `5KnS0TGxWLg` **13,2 %**, `YFfCmXJk` **10,2 %**, `fx303iiDb7M` 1,04 % — 17/20 prefix ~0 %.
- Khoảng lặng im kỹ thuật số: trước −33…−38 dBFS → 4–5 frame xuống dần → **−240 dB (0 tuyệt đối)**; nền nhiễu: phẳng −55…−60 dBFS. Có **461** gap digital / **2.175** gap nền.
- Mel trung bình mỗi frame: **thật** digital **−7,85** · nền −2,42 · hữu thanh +0,03; **model** digital **−4,64** · nền −2,42 · hữu thanh −0,02.
  ⇒ Model **lấy trung gian** giữa im kỹ thuật số và nền nhiễu (đúng giả thuyết W1.4).

### G1 GTA — 5.000 bước, batch 8, d-warmup 500 (`runs/g1_gta5.log`, 3.251 s)
- Mel loss: 1,14 (600) → 0,80 (1.100) → 0,66–0,77 (3.000–5.000); **sàn copy 0,133**.
- `mps` 2,4 G → 3,4 G từ bước 1.000 (sau backward của G) rồi **phẳng tới 5.000**; `empty_cache` mỗi 200 row đã xử lý OOM.
- Một cú nhảy loss ở bước 2.400 (mel 2,55) rồi hồi ngay (1,19 → 0,74).

**TF 300 segment (98.138 frame), bin < −50 = 25.398 frame:**
| bản | flat_V | HNR | S1 median | bin4 Δ RMS (median / mean / energy) | flat bin4 |
|---|---|---|---|---|---|
| copy-synth | 0,1317 | −2,54 | — | — | — |
| P2 | **0,1884** | **−4,05** | **3,80** | **12,4 / 16,9 / 17,5** | **0,4391** |
| GTA 1k | 0,2189 | −4,51 | 3,82 | 15,7 / 20,3 / 19,1 | 0,468 |
| GTA 2k | 0,2306 | −4,65 | 3,92 | 14,7 / 19,2 / 18,5 | 0,5535 |
| GTA 3k | 0,2143 | −4,59 | 3,82 | 13,8 / 17,9 / 18,0 | 0,5526 |
| GTA 4k | 0,1883 | −4,36 | 3,80 | 14,0 / 18,7 / 17,7 | 0,5419 |
| GTA 5k | 0,1966 | −4,38 | 3,77 | 14,5 / 19,4 / 17,6 | 0,5475 |

**Copy-check (100 segment, mel thật → vocoder → L1 với mel thật, log thô):** Vocos gốc **0,133**; GTA 1k **0,529** · 2k 0,383 · 3k **0,341** · 4k 0,375 · 5k 0,369 ⇒ **GTA quên mel thật 2,6–4×**.

**Free-run 40 câu (production):**
| bản | flat_V | HNR | % frame < −50 | mean dB |
|---|---|---|---|---|
| WAV thật | 0,1187 | −1,67 | — | — |
| P2 | 0,1615 | −3,68 | 2,11 | −30,22 |
| P2+exp −45 | 0,1607 | −3,67 | 4,58 | −31,05 |
| GTA 4k | 0,1706 | −3,98 | 1,92 | −29,78 |
| GTA 4k+exp | 0,1697 | −3,96 | 4,88 | −30,62 |

⇒ **Cổng GTA: TRƯỢT** (không bản nào hơn P2 ở flat_V/HNR/N1; copy-check xấu 2,6–4×). Nguyên nhân khả dĩ: mel P2 đã mất chi tiết, vocoder không thể khôi phục — GTA chỉ thêm nhiễu phẳng.

**Expander (P2, 300 TF):** Δ RMS bin < −50 giảm mạnh theo **median** (12,4 → 3,2 / 1,1 dB) nhưng **energy** chỉ 17,5 → 15,5 / 13,5 vì trần −12 dB; **S1 xấu đi** (3,80 → 4,34 / 4,99 dB). Flatness không đổi (0,439) — đúng bản chất gain.

**Trang nghe mù:** https://cio-rows-rocks-seq.trycloudflare.com/v1828_gate/ (9 câu × 5 bản: copy/P2/P2+exp/GTA4k/GTA4k+exp, −23 LUFS, nhãn xáo, đáp án `artifacts/listen_small_filt/v1828_gate/key.json`).

## 21. v18.29 — giải mã trang nghe, L2/L3, ba nhánh fine-tune (F0/Fc/Fd)

JSON: `v1828_l1_scores.json` · `v1829_l2_strat.json` · `v1829_l2_l3.json` · `v1829_l4_tf.json` · `v1829_l4_free.json` · `v1829_gate_map.json` · `v1829_fd_targets*.json`.

### L1 — điểm người dùng (1 = không lỗi … 5 = nặng), 9 câu
| bản | rè TB | hụt hơi TB | lạc TB | số câu rè = 1 |
|---|---|---|---|---|
| copy-synth | **1,00** | **1,00** | **1,00** | 9/9 |
| P2 | 2,89 | 1,00 | 2,33 | 2/9 |
| P2 + exp −45 | 3,00 | 1,00 | 1,89 | 2/9 |
| GTA 4k | 4,56 | 1,00 | 1,33 | 0/9 |
| GTA 4k + exp | 4,56 | 1,00 | 1,33 | 0/9 |
- `hut = 1` **45/45 ô** ⇒ expander không gây hụt hơi theo tai.
- ⚠️ **Cả 9 câu đều thuộc video `2b65WyiFC9k`** (trang chọn nhầm 1 video) — video này **CÓ gate** (11,5 % frame digital ở các row val; train rows của nó 0 %). v18.28 ghi "không gate" là **sai** (đo trên train rows).
- s07 (4 bản `lac=3`): transcript có từ lạ `in xờ pếch`, `đê mô` ⇒ lỗi nội dung do OOV, không phải do nhánh.

### L2 — N1 tách nhóm video (300 row **stratified** 12 prefix, 88.998 frame)
- 9/12 prefix có gate; **bin < −50 chỉ tồn tại ở video có gate** (`n1_nogate` = 0 frame) ⇒ thước N1 là phép đo **chỉ trên video gate**.
- P2: bin4 gate `d_med 14,4 dB` / `energy 17,3 dB` · flat 0,4425. Toàn bộ (gộp): 5,07 / 16,91.
- Free-run 40 câu: WAV thật `<−50` = **29,0 %**, mean **−45,6 dB**; P2 = 2,33 %, −30,5 dB.

### L3 — trần expander (P2, −40 dB, 1:3), 300 row head
| trần | bin4 d_med | d_energy | S1 median |
|---|---|---|---|
| −12 dB | 1,11 | 13,47 | 4,99 |
| −18 dB | −3,86 | 12,79 | 5,67 |
| −24 dB | −8,83 | 12,49 | 6,24 |
⇒ Hạ trần chỉ kéo **median** xuống (thậm chí âm = nhỏ hơn copy), **energy gần như không đổi** (17,5 → 12,5) và **S1 xấu dần**.

### L4 — ba nhánh (2k bước, từ P2@2k, chỉ decoder+mel+postnet, KHÔNG GAN)
| nhánh | flat_V | HNR | S1 median | bin4 gate d_med / energy |
|---|---|---|---|---|
| P2 (mốc) | **0,1878** | **−4,06** | **3,706** | **14,4 / 17,3** |
| F0 (đối chứng) | 0,2190 | −4,51 | 4,009 | 15,5 / 16,0 |
| Fc gate=1 | 0,2195 | −4,50 | 4,038 | 15,2 / 16,1 |
| Fc gate=0 | 0,2200 | −4,50 | 3,998 | 15,6 / 16,0 |
| Fd (đích gate đồng nhất) | 0,2258 | −4,63 | **10,597** | **−15,4 / 13,3** |
- **F0 đã tệ hơn P2** (flat +17 %, HNR −0,45, S1 +8 %) ⇒ fine-tune 2k **không GAN** tự nó làm hỏng các thước này; mọi nhánh phải so với **F0**, không phải P2.
- Fc ≈ F0 (chênh ≤0,4 dB ở bin4 khi gate 1 vs 0) ⇒ **cờ gate 1 bit chưa có tác dụng**.
- Fd: đổi **54,5 % frame** (17.786 row), kéo nền xuống **thấp hơn copy** (d_med −15,4 dB) nhưng **S1 10,6 dB** ⇒ gate cả tiếng nhỏ ⇒ hụt hơi nặng.
- **Cổng "flat_V và S1 không tệ hơn P2 quá 5 %": CẢ 3 NHÁNH TRƯỢT** (F0 +16,6 %/+8,2 % · Fc +16,9 %/+9,0 % · Fd +20,2 %/+186 %).
- Free-run: P2 flat 0,1719 / HNR −3,89 · Fc 0,2045 / −4,36 · Fd 0,2052 / −4,48.

**Trang nghe:** https://cio-rows-rocks-seq.trycloudflare.com/v1829_floor/ — 9 câu (5 không gate + 4 gate) × 5 bản (copy · P2 · P2+exp −40/1:3 trần 18 · Fc gate=1 · Fd), −23 LUFS, nhãn xáo, xuất JSON. Đáp án: `artifacts/listen_small_filt/v1829_floor/key.json`.

## 22. v18.30 — B1 (thước nào khớp tai) · B2 (G1/G2) · B3 (OOV)

JSON: `v1830_metric_corr.json` · `v1830_g12_tf.json` · `v1830_free.json` · `v1830_oov_val.json`.

### B1 — Spearman với điểm rè (45 ô = 9 câu × 5 bản; 1 = không rè)
| thước | rho gộp | rho trong câu (median) |
|---|---|---|
| **inter-harmonic** (mức giữa hài − copy) | −0,478 | **−0,90** |
| **flat 2–8 kHz** | 0,377 | **0,80** |
| HNR | −0,460 | −0,70 |
| S1 (syllable MAE) | 0,421 | 0,50 |
| flat_V (2–6 kHz) | 0,313 | 0,50 |
| N1 bin4 (median Δ) | 0,301 | 0,50 |
| N1 bin4 (energy Δ) | 0,398 | 0,30 |
- Dấu inter-harmonic: mọi model **âm** so copy (ít năng lượng giữa hài hơn), GTA âm nhất (−2,1…−3,2 dB) và bị chấm rè nặng nhất ⇒ **âm hơn = tệ hơn**.
- ⚠️ Chỉ 9 câu (một phần từ 2 video), 45 điểm ⇒ **định hướng**, không kết luận chắc.
- Thước chính đề xuất: **flat 2–8 kHz** (diễn giải rõ) + **HNR**; inter-harmonic cần kiểm trên tập đa video.

### B2 — G1/G2 (2k bước, cùng công thức P2: adv 0,3 · d_warmup 300 · chỉ decoder/mel/postnet)
| bản | flat_V | HNR | S1 median | bin4 Δ median | bin4 energy | bin4 flat |
|---|---|---|---|---|---|---|
| copy | 0,1191 | −2,81 | — | — | — | — |
| P2 | 0,1878 | −4,06 | 3,706 | 5,07 | 16,91 | 0,4369 |
| **G1** | **0,1459** | −3,98 | 3,783 (**+2,1 %**) | 7,87 | 18,01 | 0,3688 |
| G2 (fm 1,0) | 0,1370 | −3,92 | 3,940 (**+6,3 %**) | 8,51 | 18,87 | 0,3570 |
- **G1 ĐẠT cổng** (flat_V −22 %, HNR tốt hơn, S1 +2,1 % ≤ 5 %). **G2 TRƯỢT** (S1 +6,3 %).
- Cả hai **làm nền to hơn** ở bin < −50 (d_med 5,07 → 7,87/8,51; energy 16,9 → 18,0/18,9) dù flat tốt hơn.
- Free-run 40 câu: P2 flat 0,1719/`<−50` 2,33 % · **G1 0,1491/1,59 %** · G2 0,1467/1,46 % ⇒ ở production, G1/G2 **giảm** tỉ lệ frame nền rất nhỏ.
- TỰ QUYẾT KHÁC ĐẶC TẢ: P2 dùng `fm_weight = 0` ⇒ "×2" vẫn = 0; đã chạy G2 với **fm = 1,0** (giá trị chuẩn của feature matching). Nếu A0 muốn 2,0 thì chạy lại 2k (~6 phút).

### B3 — OOV
- Quét 3.378 row val: **0 token OOV** (100 % `status = VALID`), không có chữ ngoại (f/j/w/z).
- `in xờ pếch` / `đê mô` là **phiên âm tiếng Anh** (inspector / demo), không phải OOV.
- Đã hỗ trợ **khoá nhiều từ** trong `custom_reading` (+2 test) nhưng **revert entry** cho "in xờ pếch": đổi 3 → 4 âm tiết làm **lệch durations manifest** (eval TF ném lỗi `d_syl 26 vs active 27`). Muốn dùng phải giữ **đúng số âm tiết** hoặc tái tạo manifest.
- Trang nghe: `--exclude-oov` lọc theo status + cụm phiên âm đã biết (hiện không loại câu nào).

**Trang nghe v1830:** https://cio-rows-rocks-seq.trycloudflare.com/v1830_gan/ — 9 câu (5 không gate + 4 gate) × 4 bản (copy · P2 · G1 · G2), −23 LUFS, nhãn xáo, xuất JSON. Đáp án: `artifacts/listen_small_filt/v1830_gan/key.json`.

## 23. v18.31 — B0 (định nghĩa nền) · A (đo rộng 300 câu) · A2 · B (G1b)

JSON: `v1831_tf.json` · `v1831_free300.json`. Cấu hình production: `configs/production_v1831.json`.

### L0/B0 — vì sao 29,0 % vs 2,33 %
**Không có mâu thuẫn:** hai báo cáo dùng **cùng 40 câu stratified**, cùng định nghĩa (RMS 256 mẫu, 20·log10, **không** chuẩn hoá LUFS). Con số **29,0 % / −45,6 dB là WAV THẬT** (hàng `copy`), còn **P2 = 2,33 % / −30,5 dB** — trùng khít giữa `v1829_l4_free.json` và `v1830_free.json`. Câu chữ v18.29 đặt "WAV thật" và "P2" cùng dòng nên bị đọc gộp.

### L1/A — free-run 300 câu stratified, 12 video (production path)
| bản | flat_V | flat 2–8k | HNR | % frame < −50 | mean dB | duration ratio |
|---|---|---|---|---|---|---|
| WAV thật | 0,1112 | 0,0548 | −2,22 | 27,93 | −44,00 | — |
| P2 | 0,1698 | 0,0741 | −3,79 | 2,19 | −30,04 | 0,566 ± 0,146 |
| **G1** | **0,1425** | 0,0611 | **−3,71** | **1,65** | −30,48 | 0,563 ± 0,148 |
| G2 (fm 1,0) | 0,1430 | **0,0588** | −3,67 | 1,45 | −29,77 | 0,565 ± 0,147 |
| G1b (2k tiếp) | 0,1626 | 0,0656 | −3,92 | 2,33 | −30,02 | 0,565 ± 0,147 |

**Theo video (G1 vs P2):** flat_V thắng **12/12** video; HNR thắng **8/12**.
**Luật A:** G1 ĐẠT (flat 12/12 ≥ 2/3 · HNR 8/12 ≥ 2/3 · S1 +2,1 % ≤ 5 %) ⇒ **G1 = bản thắng**.
**G2:** flat_V 0,1430 **kém** G1 0,35 % (luật đòi ≥3 % tốt hơn) ⇒ **không đổi**.
Duration ratio P2/G1/G2/G1b đều ~0,565 ⇒ không có drift thời lượng mới.

### L2/A2 — nghịch lý nền ở TF: **hệ thống**, không do 1–2 video
- Bin < −50, Δ(G1 − P2) theo video: **+1,0 … +3,6 dB trên 12/12 video**.
- Theo độ dài: ngắn (<150f) **+4,7 dB** · vừa +1,9 · dài (≥300f) +2,0.
⇒ Kết luận: G1 **nâng mức frame lặng trong TF một cách đều** (~+2 dB), không phải lỗi mẫu; nhưng free-run lại **giảm** tỉ lệ frame < −50 (2,19 → 1,65 %). Cơ chế chưa rõ (khác căn chỉnh GT vs pause dự đoán) — **suy đoán**, cần đo riêng nếu muốn.

### L3/B — G1b (2k tiếp từ G1, cùng công thức)
flat_V **0,1626** (kém G1 14 %), HNR −3,92 (kém 0,21), `<−50` 2,33 % (kém 1,65 %) ⇒ **bão hoà ở G1**, dừng nhánh; không chạy G1c.

### Production & rollback
- Production đổi sang **G1** (`runs/v1830_g1/ckpt_2000.pt`), nhãn **"CHƯA XÁC NHẬN TAI"**, ghi tại `configs/production_v1831.json`.
- Rollback 1 bước: truyền `--checkpoint artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt`.

## 24. v18.32 — bộ nghe mù 36 câu + chẩn đoán TF/free-run + kiểm duration ratio

JSON: `v1832_tf_free.json` · items/mapping: `artifacts/listen_small_filt/v1832_blind/{items,mapping}.json`.
Công cụ: `tools/v1832_blind_set.py`, `tools/v1832_tf_free_diag.py`.

### Bộ nghe mù (chờ người dùng chấm)
- **36 câu / 12 video** · 12 ngắn (<150f) · 12 vừa (150–300f) · 12 dài (≥300f) · **36/36 có khoảng nghỉ** · 4 câu từng nghe ở trang cũ.
- Mỗi câu: **A/B** = P2 vs G1 (ẩn tên, đảo thứ tự ngẫu nhiên, mapping riêng), **Thật** = bản ghi gốc để tham chiếu.
- Hai phiên bản: **gốc** (không đổi) và **−23 LUFS**: chỉ **một hệ số gain tuyến tính** + chốt đỉnh −1 dBTP; **không** denoise/gate/compressor/limiter động.
- Chấm 5 mục (rè nền · rõ chữ · tự nhiên · ngắt nghỉ · lỗi phát âm) + thích A/B/hoà; xuất JSON.
- Trang: https://cio-rows-rocks-seq.trycloudflare.com/v1832_blind/

### Chẩn đoán TF vs free-run (36 câu, cùng checkpoint/frontend/vocoder, α=0)
| điều kiện | tổng s | < −50 s (%) | pause % | LUFS | p50 dB | dur ratio | trim ratio | speech ratio |
|---|---|---|---|---|---|---|---|---|
| GT thật | 105,29 | 31,44 (29,86) | 27,70 | −27,52 | −35,67 | — | — | — |
| P2 TF | 104,90 | 33,30 (31,75) | 28,29 | −28,89 | −37,44 | 0,995 | 0,975 | 0,875 |
| P2 free | 65,76 | 1,62 (2,46) | 1,54 | −27,51 | −28,18 | 0,616 | 0,803 | 0,830 |
| G1 TF | 104,90 | 31,83 (30,34) | 28,26 | −28,35 | −35,69 | 0,995 | 0,974 | 0,929 |
| G1 free | 65,64 | 1,23 (1,87) | 1,19 | −28,21 | −28,58 | 0,615 | 0,801 | 0,850 |
| copy-synth | 104,90 | 26,89 (25,63) | 22,95 | −29,35 | −34,06 | — | — | — |

- Khác biệt < −50 giữa TF và free-run **gần như trùng khớp với chênh lệch pause**: TF 28,3 % pause / 31,8 % < −50; free 1,2 % pause / 2,5 % < −50. LUFS hai chế độ tương đương ⇒ không phải lỗi chuẩn hoá mức.
- **Mask**: TF lấy từ copy-synth (RMS < −50), **cố định** giữa P2/G1; free-run không căn frame nên chỉ báo ngưỡng tuyệt đối, **không** dùng mask chung.
- **"duration schedule dùng chung"** = điều kiện TF (durations/gaps GT). Pipeline **không** hỗ trợ free-run với durations GT (`iter_synthesize_sentence` không nhận durations) ⇒ hai điều kiện vẫn khác nhau ở **nguồn duration/gap + hậu xử lý ghép câu**.
- ⇒ **Chưa xác định cơ chế**; không quy nguyên nhân cho duration hay pause. Số liệu chỉ cho thấy pause chiếm ~28 pp trong 30 pp chênh lệch.

### Kiểm duration ratio
- Mẫu số = `frame_range` (gồm lead + gaps); tử số = số frame model. Ratio free/GT = **0,615**.
- Kiểm bằng số: pause GT 27,7 % ⇒ (1−0,277)/(1−0,0154) = **0,734**; nhân speech ratio 0,83–0,85 ⇒ **≈0,61** — khớp.
- Cắt silence đầu/cuối (max−35 dB): trim ratio **0,80**; speech ratio (frame hữu thanh) **0,83–0,93**.
- ⇒ Ratio ~0,56–0,62 **không** chứng minh thời lượng tuyệt đối đúng; phần lớn là **mất pause**, phần còn lại là speech ngắn hơn ~7–17 %. Không sửa duration trong vòng này.

### Quyết định giữ nguyên
- Không dùng < −50 % làm bằng chứng rè tốt/xấu; không đổi thước chính (C5 chưa đủ 30 câu có điểm); G1 giữ nhãn **CHƯA XÁC NHẬN TAI**; rollback P2 nguyên vẹn.

## 25. v18.33 — đánh giá tự động đa tiêu chí (P2 vs G1)

### 25.0 LUẬT QUYẾT ĐỊNH — ĐĂNG KÝ TRƯỚC KHI XEM KẾT QUẢ
- **Giám khảo chính**: DNSMOS **BAK** (nhiễu nền; cao = tốt) và **UTMOSv2** (tự nhiên; cao = tốt).
  **Phụ**: DNSMOS SIG/OVRL, flat_V, HNR, thống kê RMS. **Cổng nội dung**: CER (ASR).
- So **ghép cặp theo câu** (cùng câu, cùng path free-run), KTC 95% bằng **bootstrap theo cụm video**
  (resample 12 video, 2.000 lần) — không coi 300 câu là 300 quan sát độc lập.
- **G1 thắng** nếu đồng thời: (1) ΔBAK (G1−P2) trung vị > 0 **và** CI95 không chứa 0;
  (2) ΔUTMOS ≥ 0 hoặc CI95 không nằm dưới −0,02; (3) ΔCER ≤ +0,002 hoặc CI95 không nằm trên +0,002;
  (4) ≥ 2/3 số video có ΔBAK ≥ 0.
- **P2 thắng** nếu đối xứng (ΔBAK CI95 < 0, hoặc UTMOS/CER xấu đi rõ theo CI).
- **Còn lại**: chưa phân thắng bại — giữ nguyên trạng thái, không training mò.
- Ngưỡng 0,02 MOS và 0,2 điểm % CER là **ngưỡng phát hiện tối thiểu** (độ nhạy), **không** phải
  ngưỡng cảm nhận; không tuyên bố "đã xác nhận bằng tai".

### 25.1 Công cụ — kiểm tra khả dụng (nguồn/phiên bản/giới hạn)
| công cụ | trạng thái | phiên bản / model | ghi chú |
|---|---|---|---|
| DNSMOS (P.835) | **dùng được** | `speechmos` (ONNX đi kèm), 16 kHz | cho ovrl/sig/bak/p808; **bắt buộc 16 kHz** (tự resample polyphase) |
| UTMOSv2 | **dùng được** | `utmosv2` 1.3.1, `fusion_stage3`, fold0 | checkpoint cache `~/.cache/utmosv2`; ~0,5 s/file trên MPS |
| ASR tiếng Việt | **dùng được** | `faster-whisper` small (CTranslate2 int8, CPU) | language=vi, beam 1, không VAD; ~1 s/câu |
| NISQA | **KHÔNG dùng** | `nisqa` 2.0.post2 (PyPI) | gói **không kèm trọng số**; `pip install` kéo torch 2.2.1 làm hỏng môi trường (đã khôi phục torch 2.14.1 + torchvision 0.29.1). Cần tải `nisqa_pretrained_weights` thủ công ⇒ bỏ trong vòng này |

### 25.2 KẾT QUẢ (300 câu / 12 video, free-run production, ghép cặp)
**Giám khảo (trung bình):**
| bản | UTMOSv2 | DNSMOS BAK | SIG | OVRL | CER | WER |
|---|---|---|---|---|---|---|
| P2 | **1,534** | **2,801** | **3,063** | **2,310** | **0,542** | **0,601** |
| G1 | 1,444 | 2,295 | 2,808 | 2,014 | 0,669 | 0,745 |
| WAV thật | 2,248 | 3,981 | 3,350 | 3,035 | 0,664 | 0,748 |

**Δ ghép cặp (G1 − P2), KTC 95% bootstrap theo cụm video:**
| chỉ số | trung vị | CI95 | % câu G1 tốt hơn |
|---|---|---|---|
| BAK | **−0,401** | [−0,456, −0,350] | 16,0 |
| SIG | −0,132 | [−0,141, −0,116] | 27,3 |
| OVRL | −0,225 | [−0,278, −0,198] | 18,3 |
| UTMOS | **−0,102** | [−0,143, −0,058] | 39,0 |
| CER | **+0,063** | [+0,047, +0,096] | — |
| flat_V | −0,028 | [−0,029, −0,024] | 12,0 (thấp hơn = "tốt" theo thước cũ) |
| HNR | +0,110 | [+0,007, +0,188] | 57,3 |

- **BAK thấp hơn ở 12/12 video** (G1 không thắng video nào). Ví dụ 2b65WyiFC9k 2,98→2,23 · BD83pScFukQ 2,74→2,15 · vEtfty4Cc6Y 3,00→2,44.
- CER theo video **hỗn hợp** (4ITcvEa 0,730→0,679 và lRojc8KDJ6k 1,112→0,672 tốt lên; 50swDIypplU 0,266→0,433 và EUNlJU1BuzE 0,394→1,157 xấu đi) nhưng tổng thể xấu đi rõ; ví dụ G1 bỏ từ ("Thì lý thuyết chúng ta sẽ" → "Dù chúng ta sẽ").
- **flat_V/HNR (phụ trợ) ngược chiều với giám khảo** — đây chính là lý do v18.31 chọn G1; đánh giá tự động không xác nhận.

**C. Thời lượng (cùng 300 câu):** P2 571,31 s · G1 570,08 s · thật 952,49 s; pause 1,69 % / 1,21 % / 28,68 %; trim 567,7 / 566,9 / 826,4 s; speech frames 38.099 / 39.267 / 54.155. Không sửa duration.

**Kiểm độ nhạy (8 câu thật/8 video, suy giảm nhân tạo):**
| | clean | noise30 | noise20 | noise10 | drop2 | drop5 | drop10 |
|---|---|---|---|---|---|---|---|
| BAK | 4,07 | 3,58 | 3,04 | 2,23 | 4,00 | 4,08 | 3,95 |
| UTMOS | 2,42 | 2,22 | 2,02 | 1,57 | 2,07 | 1,68 | 1,67 |
| CER (vs bản sạch) | 0 | 0,21 | 0,18 | 0,35 | 0,145 | 0,218 | 0,294 |
- **BAK đơn điệu tuyệt đối với nhiễu** (rho = −1,0 ở 8/8 câu) → dùng được để so nền.
- UTMOS đơn điệu cả nhiễu lẫn cắt (rho −1,0 / −0,5).
- **CER không đơn điệu ở mức nhẹ** (0,21 > 0,18) và sàn ASR trên WAV thật = 0,188 ⇒ chỉ tin khi khác biệt lớn; không dùng CER làm thước rè.

### 25.3 KẾT LUẬN (theo luật §25.0 đăng ký trước)
- (b) **P2 tốt hơn theo đánh giá tự động**: BAK CI95 < 0, UTMOS CI95 < 0, CER CI95 > 0, **0/12 video** G1 tốt hơn về BAK.
- **Đề xuất rollback production về P2** (đã cập nhật `configs/production_v1831.json`; G1 giữ làm nhánh thí nghiệm).
- Nhãn: **"đánh giá tự động"** — **không** phải xác nhận bằng tai. Không đổi thước chính C5.

### 25.4 AUDIT v18.33 (v18.34) — từng mục đúng / lỗi / chưa giải thích

**A. CER/WER — LỖI cách trình bày (không phải lỗi tính):**
- "Sàn ASR 0,188" ở sensitivity là CER(ASR bản sạch) vs **manifest text** trên **8 câu ngắn**, không phải sàn của bộ 300.
  Bộ 300 (cùng reference, cùng normalization) có CER **macro 0,664** cho WAV thật ⇒ hai số **khác tập**, không so được.
- Có **hai loại** CER: (i) vs transcript chuẩn (dùng ở bảng chính); (ii) vs ASR của bản sạch (dùng ở sensitivity).
  Đã tách tên; không gọi chung là "sàn".
- Tổng hợp: bảng chính là **macro theo câu** (`mean` CER từng câu). Micro (tổng edit/tổng ký tự):
  P2 **0,291** · G1 **0,393** · thật **0,413**; median theo câu: 0,264 / 0,406 / 0,422.
  Macro bị chi phối bởi vài câu lỗi nặng; **chiều kết luận không đổi** nhưng độ lớn nhỏ hơn nhiều so với 0,54/0,67.
- Pairing: **300/300** text khớp manifest; **300/300** `frame_range` = Σdurations+Σgaps; **296/300** WAV thật dài đúng
  (4 file ngắn hơn 1–10 ms do file nguồn kết thúc — không ảnh hưởng chấm).
- Quy tắc chấm **giống nhau** cho P2/G1/real (cùng hàm, cùng reference, cùng normalization NFC + bỏ dấu câu).
- Ví dụ "bỏ từ": chỉ được ghi **"ASR trả transcript thiếu/khác từ"** — chưa có bằng chứng độc lập trong audio.
- ⇒ **Tạm loại CER/WER khỏi luật quyết định** (vẫn báo cáo tham khảo).

**B. Chiều tốt/xấu — LỖI nhãn cột:**
- Cột "% câu G1 tốt hơn" cũ tính `delta > 0` cho **mọi** thước ⇒ với flat_V/CER bị ngược.
- Đã thay bằng `eval_stats.outcome_counts(values, higher_better=...)` (+3 test). Bảng đúng:
  BAK 16,0 % thắng · SIG 27,3 · OVRL 18,3 · UTMOS 39,0 · HNR 57,3 · **flat_V 88,0 % thắng** (thấp hơn = tốt) ·
  CER 20,3 % thắng / 61,7 % thua / 18,0 hoà · WER 17,0 / 53,7 / 29,3.
- Không sửa luật sau khi thấy kết quả.

**C. Thời lượng — LỖI nhãn "speech frames":**
- `total = pause + non-pause` đúng (pause = frame RMS < max−35 dB; hop 256 @24 kHz; đơn vị frame).
- "speech frames" cũ thực chất là **voiced frames** (autocorr ≥ 0,45), **không** phải phần bù của pause:
  P2 38.099 voiced / 52.761 non-pause; G1 39.267 / 52.895.
- Pause%: macro 1,69/1,21/28,68 (P2/G1/thật) vs **weighted theo thời lượng** 1,44/0,97/26,34.
- G1 tổng frame 53.411 vs P2 53.534 (−0,23 %) nhưng voiced +3,1 % và pause −257 frame ⇒ khác do **bộ phát hiện voiced/ngưỡng năng lượng**, không được đọc là tốc độ nói.

**D. Sensitivity & thống kê — LỖI phát biểu:**
- Xu hướng **trung bình** ≠ đơn điệu **từng câu**. Với cắt: UTMOS trung bình 2,07→1,68→1,67 nhưng rho từng câu
  [0,5; −0,5; −0,5; −1; −1; −0,5; 0,5; 0,5], median **−0,5** ⇒ **không** đơn điệu từng câu. Đã sửa phát biểu.
- Với nhiễu: BAK rho = **−1,0 ở 8/8 câu** (đơn điệu thật); UTMOS median −1,0 nhưng 2/8 câu chỉ −0,5.
- Bootstrap: resample **12 video có hoàn lại**, gộp mọi câu của video được chọn, tính **trung vị** delta (2.000 lần).
  CI báo cáo là CI của **trung vị** (đúng đại lượng đã in): [−0,456, −0,350]; CI của **trung bình** là [−0,563, −0,448].
  Số câu/video đều 25 nên chưa có trường hợp lệch; phương pháp vẫn xử lý được (gộp theo video được chọn).

**E. Kết luận sau audit:**
- Loại CER/WER khỏi luật ⇒ còn BAK (CI95 < 0), UTMOS (CI95 < 0), **0/12 video** G1 hơn BAK ⇒ **vẫn P2**, nhưng
  **hạ mức tin cậy**: "G1 bị các giám khảo tự động đánh giá thấp hơn P2, đặc biệt ở chất lượng nền theo DNSMOS BAK".
- **Không** nói "đã chứng minh G1 tăng nhiễu nền"; **không** kết luận về GAN nói chung; **không** quy nguyên nhân
  cho loss/vocoder/duration.
- Giữ P2 là **lựa chọn vận hành thận trọng**, không phải "thắng tuyệt đối".

### 25.5 Lịch sử kết luận (ghi để đối chiếu, không xoá)
1. **v18.31**: chọn G1 theo **flat_V/HNR/S1** (đo tự động, chưa có tai) — nhãn "chưa xác nhận tai".
2. **v18.33**: giám khảo tự động (DNSMOS/UTMOSv2/ASR) → **P2 tốt hơn** ở BAK/UTMOS/CER; rollback production về P2.
3. **v18.34 (audit)**: sửa nhãn chiều (flat_V thực ra G1 thắng 88 %), loại **CER/WER** khỏi luật (macro/micro lệch,
   "sàn 0,188" khác tập), sửa "speech frames" → **voiced frames**, sửa phát biểu "UTMOS đơn điệu với cắt".
   Kết luận: **giữ P2** theo BAK+UTMOS (CI95 < 0, 0/12 video) nhưng **hạ mức tin cậy**:
   *"G1 bị các giám khảo tự động đánh giá thấp hơn P2, đặc biệt ở chất lượng nền theo DNSMOS BAK"* —
   **không** phải "đã chứng minh G1 tăng nhiễu nền", **không** kết luận về GAN nói chung, **không** quy nguyên nhân.

## 26. v18.35 — sàng lọc checkpoint hiện có (không training)

### 26.0 LUẬT SÀNG LỌC — ĐĂNG KÝ TRƯỚC KHI XEM KẾT QUẢ
- Bộ sàng lọc **S**: 120 câu, 12 video × 10 (4 ngắn / 3 vừa / 3 dài mỗi video), **loại** mọi câu đã dùng
  ở v1831/v1832/v1833 (300 stratified + 36 blind + 300 head).
- Giám khảo chính: **DNSMOS BAK** và **UTMOSv2**; bổ trợ SIG/OVRL, flat_V, HNR. **Không** dùng CER/WER.
- Ứng viên **ĐẠT** nếu: trung vị ΔBAK > 0 **và** CI95 (bootstrap cụm video) không chứa 0;
  ΔUTMOS CI95 dưới > −0,02; **≥ 2/3 video** có ΔBAK ≥ 0.
- Nhiều ứng viên đạt: chọn ΔBAK trung vị lớn nhất trong nhóm có ΔUTMOS ≥ 0.
- Không ai đạt: giữ P2, kết thúc vòng. Không mở rộng ứng viên/bộ đo.
- Xác nhận (nếu có): **đúng 1** ứng viên trên tập **C** 120 câu rời rạc (khác S, khác các bộ đã dùng),
  cùng luật; chỉ **đề xuất**, không đổi production.

### 26.1 Kiểm kê checkpoint (tồn tại thật, đọc từ file + run.json)
| checkpoint | nguồn khởi tạo | step | loss | ghi chú |
|---|---|---|---|---|
| `adv_p2_20k/ckpt_2000.pt` (**P2**, production) | `acoustic_v32/best.pt` | 2.000 | adv 0,3 · fm 0 · d_warmup 300 · lr 2e-5 | bản đang dùng |
| `adv_p2_20k/eval_4000.pt` | P2-run | 4.000 | như trên | **ứng viên** `p2_4k` |
| `adv_p2_20k/eval_6000.pt` | P2-run | 6.000 | như trên | loại (chỉ giữ 4 ứng viên; 4k đại diện) |
| `v1830_g1/ckpt_1000.pt` | **P2** | +1.000 (tổng 3.000) | adv 0,3 · fm 0 | **ứng viên** `g1_1k` |
| `v1830_g1/ckpt_2000.pt` | P2 | +2.000 | adv 0,3 · fm 0 | G1 — đã bị loại ở v18.33 |
| `v1830_g2/ckpt_1000.pt` | P2 | +1.000 | adv 0,3 · **fm 1,0** | **ứng viên** `g2_1k` |
| `v1830_g2/ckpt_2000.pt` | P2 | +2.000 | adv 0,3 · fm 1,0 | **ứng viên** `g2_2k` |
| `v1830_g1b/*` | **G1** | +2.000 | adv 0,3 · fm 0 | **loại** (không có lý do mới; đã bão hoà) |
| `v1829_f0/fd/fc` | P2 | 2.000 | không GAN / cờ gate | **loại** (đã trượt cổng v18.29) |
| `adv_v35/*`, `acoustic_v35/*` | v3.5 | — | adv 0,3 | **loại** (config/mel-stats v35 khác, không so trực tiếp) |

### 26.2 KẾT QUẢ SÀNG LỌC (110 câu / 11 video; 44 ngắn · 33 vừa · 33 dài; 0 câu trùng bộ cũ)
⚠️ `2b65WyiFC9k` **không có câu nào chưa dùng** (toàn bộ 242 row nằm trong "300 head" cũ) ⇒ bộ sàng lọc còn 11 video.

| ứng viên | BAK trung vị | CI95 | % câu hơn P2 | UTMOS trung vị | CI95 | flat_V | HNR | ĐẠT? |
|---|---|---|---|---|---|---|---|---|
| `g1_1k` (GAN 3k luỹ kế) | −0,012 | [−0,118, +0,017] | 47,3 | **+0,096** | [+0,049, +0,149] | −0,017 | +0,135 | **✗** (BAK CI chứa 0) |
| `g2_1k` (fm 1,0, 1k) | −0,063 | [−0,140, −0,036] | 40,0 | **+0,070** | [+0,023, +0,151] | −0,021 | +0,190 | ✗ (BAK xấu) |
| `p2_4k` | −0,002 | [−0,091, +0,089] | 49,1 | **−0,070** | [−0,181, −0,025] | +0,028 | +0,145 | ✗ (UTMOS xấu) |
| `g2_2k` | **−0,385** | [−0,449, −0,322] | 20,9 | +0,043 | [−0,045, +0,100] | −0,022 | +0,165 | ✗ (BAK xấu mạnh) |

- **Không ứng viên nào ĐẠT** luật đã đăng ký ⇒ **giữ P2, kết thúc vòng**; **không** chạy bước xác nhận (luật chỉ cho 1 bản khi đã đạt sàng lọc).
- **`g1_1k` (diễn giải đã chỉnh theo A0):** *"UTMOS cải thiện; BAK **chưa xác định được chiều khác biệt**; chưa đạt mục tiêu cải thiện chất lượng nền."*
  - **Không** gọi BAK "tương đương P2" chỉ vì CI95 chứa 0; **không** gọi đây là "mâu thuẫn trực tiếp giữa hai giám khảo".
- **Giả thuyết chưa kiểm chứng:** G1@1k có UTMOS +0,10 trên bộ sàng lọc còn G1@2k có −0,10 trên bộ 300 của v18.33 — **hai tập khác nhau**, nên **không** được kết luận "2k bước là quá tay"; chỉ ghi là giả thuyết, không hành động.
- Không dùng CER/WER; không đổi production; không training.

### 26.3 Đính chính thiết kế & số liệu bổ sung (không chạy lại audio)
- **Sai lệch so với thiết kế:** chạy **110 câu / 11 video** thay vì 120/12; bucket thực tế **44 ngắn / 33 vừa / 33 dài**.
- **Thời điểm phát hiện:** thiếu `2b65WyiFC9k` được phát hiện **sau khi chạy xong và đã có điểm**, ở bước kiểm tra phân bố.
  **Không có điều chỉnh thiết kế nào trước khi xem điểm**, và **không chọn lại câu** sau khi thấy điểm; không bổ sung dữ liệu.
  Nguyên nhân: cả 242 row của video này đã nằm trong "300 head" cũ nên bị loại theo luật "không dùng câu đã dùng".
- **Số video có ΔBAK ≥ 0 (trên 11 video thực tế):** `g1_1k` **4/11** · `g2_1k` **2/11** · `p2_4k` **6/11** · `g2_2k` **0/11**.
- **Ngưỡng 2/3 áp dụng trên 11 video:** cần ≥ ⌈2/3 × 11⌉ = **8 video**; code dùng `(2·11+2)//3 = 8`. Cả 4 ứng viên đều **trượt điều kiện BAK** (CI95 và/hoặc số video), nên **không cần thêm dữ liệu** để hoàn thiện hay đổi kết quả sàng lọc.
- **Trạng thái cuối:** P2 = production (lựa chọn vận hành thận trọng) · `g1_1k` lưu checkpoint + kết quả, **chưa đạt luật giảm rè**, không phải bản thắng và **cũng chưa bị chứng minh kém toàn diện** · các ứng viên khác giữ lịch sử, không thử lại nếu không có giả thuyết mới cụ thể · **chưa xác định cơ chế** chất lượng nền kém ở các bản GAN · **không** kết luận GAN nói chung không phù hợp.
