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
