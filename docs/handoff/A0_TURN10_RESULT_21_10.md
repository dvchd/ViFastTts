# A0 lượt 10 → Agent 2: báo cáo Việc 1, 2, 4, 5 (21/10/2026)

Commit gốc: `e1e12b2`. File này thay cho `A0_TURN6_RESULT_21_10.md` cho các việc của v10.

## 1. Việc 1 — sửa thước drift (mốc giọng thật + 40 câu + CI)

### 1.1 Công cụ
- `tools/real_drift.py` (mới): gom row theo `parent`, cắt audio THẬT tại khoảng lặng > 0,64 s,
  bỏ chunk < 0,15 s hoặc < −50 dBFS, cần ≥ 3 chunk; `drift = max − min`, `drift_norm = drift/√n`,
  median/p90 + **CI 90 % bootstrap 1.000 lần**.
- `tools/premise_drift.py` (sửa): 40 câu, thêm CI, `drift_norm`, histogram số chunk,
  `--chunk-loudness` (6d); bản cũ (20 câu, không CI) đã bỏ.
- `examples/drift_texts.json`: **40 câu**, mỗi câu 3–6 mệnh đề; chunk median = 4, min 3, max 5
  (đúng bằng chunk median 4 của giọng thật ⇒ so được trực tiếp).

### 1.2 Mốc giọng thật (val + holdout, audio thật)

| | drift median | **drift p90** | norm | chunk median | n |
|---|---|---|---|---|---|
| **THẬT** | **3,41** [3,29–3,57] | **7,59** [7,29–7,97] | 1,72 | 4 | 1.252 |

Histogram chunk: 3→416 · 4→380 · 5→241 · 6→138 · 7→52 · 8→21 · 9→4.

### 1.3 Model (40 câu, free-run đọc đều, α=0, EMA)

| checkpoint | median [CI] | **p90 [CI]** | norm |
|---|---|---|---|
| v3.2 | 1,66 [1,44–2,01] | **3,39** [2,76–3,76] | 0,86 |
| P1 @2k | 2,01 [1,47–2,33] | 3,58 [2,92–4,37] | 0,97 |
| P1 @4k | 2,56 [2,00–2,87] | 4,41 [3,86–5,10] | 1,24 |
| P1 @6k | 2,43 [2,20–3,01] | 4,60 [4,03–4,81] | 1,30 |
| P1 @8k | 2,80 [2,35–3,37] | 5,05 [4,11–6,63] | 1,44 |
| **P2 @2k** | **2,42** [2,08–2,64] | **4,95** [4,12–6,43] | 1,21 |
| P2 @4k | 3,83 [3,22–4,48] | **7,28** [5,86–8,23] | 1,91 |
| P2 @6k | 4,68 [4,15–5,99] | **7,71** [6,92–9,46] | 2,53 |

#### 1.4 Thước bổ sung: spike TRONG mệnh đề (p95 − trung vị RMS 10 ms của chunk)
Đây mới là thứ tai người dùng nghe ("một từ tự to lên"), tách khỏi biến thiên giữa mệnh đề:

| | spike median | spike p90 |
|---|---|---|
| **THẬT** | **6,83** | 9,33 |
| v3.2 | 5,25 | — |
| P1 @6k | 6,17 | — |
| **P2 @2k** | **6,34** | — |
| P2 @6k | 8,08 | — |

⇒ **P2 @2k gần giọng thật nhất về spike trong mệnh đề** (6,34 vs 6,83); v3.2 **dưới** (5,25 — ít nhấn),
P2 @6k **vượt** (8,08 — nhấn quá). Khớp với tai người dùng chọn P2 @2k. P1 @6k (6,17) không nổi bật
trên thước này, nên "xoá to" mà người dùng nghe ở P1 4k/6k nhiều khả năng là **lệch mức giữa mệnh đề**
(cột drift), không phải spike một từ.

## 1.5 Cổng của A0 KHÔNG THỂ ĐẠT (cần A0 quyết)
Cổng v10: p90 ∈ [thật−1,5; thật+1,5] = **[6,09; 9,09] VÀ ≤ v3.2+2 = 5,39**. Giao của hai điều kiện
là **rỗng** vì giọng thật (7,59) **cao hơn** v3.2 (3,39) — trường hợp A0 chưa xét (A0 chỉ nêu
"nếu thật thấp hơn v3.2"). Không checkpoint nào qua cả hai.

- Nếu chỉ giữ điều kiện "gần thật": **P2 @4k (7,28)** và **P2 @6k (7,71)** là gần nhất; P1 8k (5,05)
  và P2 2k (4,95) còn cách thật ~2,6 dB.
- Nếu chỉ giữ "≤ v3.2+2": P2 2k và P1 8k qua.

**Phát hiện quan trọng:** drift là **biến thiên mức giữa các MỆNH ĐỀ**, không phải "một từ tự to lên".
Giọng thật có p90 7,6 dB ⇒ v3.2 (3,4) **đều quá mức**, không phải chuẩn. Việc P2 4k/6k "trôi" thực ra
là **tiến gần giọng thật hơn**, không phải lỗi. Lỗi tai người dùng nghe ("xoá" to) là **spike trong
mệnh đề** — thước drift không tách được; xem mục 5.

## 2. Việc 2 — chốt P2 @2k + ONNX + loudness cả file (6f)

### 2.1 ONNX
- `artifacts/onnx/v32_adv_p2_2k.onnx` (30 MB, opset 18, fp32) — acoustic từ **P2 @2k EMA**.
- `artifacts/onnx/v32_adv_p2_2k_vocoder.onnx` (52 MB) — Vocos `runs/vocoder_vocos_stftc/last.pt`.
- `synthesize-onnx` chạy được, `arch=v2_slot5`, sr 24 kHz.

### 2.2 Tất định
- md5 **×3 cùng input: giống hệt** (`9051f4d9013ba1b1f3198d7d90fa839f`).
- Đổi intra-op threads 1/2/4 (`VIFASTT_INTRA_THREADS`): 1 và 2 giống nhau; 4 khác **0,53 % mẫu,
  lệch tối đa 1 LSB (−90 dBFS)**. Mặc định (nhiều luồng) vs 1 luồng: 2,5 % mẫu, lệch ≤ 4 LSB (−78 dBFS).
  ⇒ **bit-exact theo từng cấu hình**, khác biệt giữa cấu hình là nhiễu số học dưới −78 dBFS.
  Production nên ghim `VIFASTT_INTRA_THREADS` để md5 ổn định tuyệt đối.

### 2.3 RTF (CPU, 2 luồng, 11,33 s audio)
| phần | thời gian | RTF |
|---|---|---|
| acoustic | 0,026 s | **0,002** |
| vocoder | 0,043 s | **0,004** |
| tổng | 0,069 s | **0,006** (≈166× realtime) |

### 2.4 Loudness cả file (6f)
- `src/vifasttts/runtime/loudness.py`: LUFS theo **ITU-R BS.1770-4** (K-weighting 48 kHz, block 400 ms
  chồng 75 %, gate −70 LUFS + −10 LU, đo trên resample 2×), true-peak oversample 4×.
- **Limiter look-ahead** (5 ms, release 80 ms) thay vì hạ tĩnh: đỉnh −1 dBTP của câu mẫu chỉ là
  **một xung 6 ms**; hạ tĩnh làm LUFS rơi 4,4 dB. Có vòng lặp bù gain để LUFS về đúng target.
- `--target-lufs` (mặc định **−16,0**, `0` = tắt) ở `synthesize`, `synthesize-batch`, `synthesize-onnx`;
  `infer.py` nhận `target_lufs` (mặc định thư viện = None).
- Kết quả câu mẫu: LUFS **−16,07**, true-peak **−1,00 dBTP**, không clip, tất định ×3.
- Test mới: `tests/test_loudness.py` (4 test).

## 3. Việc 3 — probe sửa độ to (E0–E3)

Xem mục 6 (cập nhật sau khi chạy xong).

## 4. Việc 4 — chuẩn hoá độ to theo chunk ở runtime (6d)

- `src/vifasttts/runtime/chunk_loudness.py`: mức frame = `ln Σ_b exp(2·logmel_b)` → dB; dịch chunk
  về mục tiêu, **giới hạn ±4 dB**; đổi dB→ln = `/8,686`.
- Mức mục tiêu đo từ dữ liệu thật: `tools/measure_voiced_level.py` →
  `data/codelung/voiced_level_v2.json` = **24,614 dB** (300 row, 84.321 frame hữu thanh;
  row p10–p90 20,84–26,49).
- Cờ `--chunk-loudness` (mặc định **TẮT**) ở CLI + `Synthesizer`.
- **Đã sửa một lỗi của bản đầu**: cộng hằng số vào z-space (sai, vì std khác từng bin) → kết quả
  vô nghĩa (P2 2k p90 4,95→6,07). Bản đúng: cộng `Δln/std_b`.
- Drift 40 câu, TẮT → BẬT:

| checkpoint | p90 TẮT | **p90 BẬT** | median TẮT | median BẬT |
|---|---|---|---|---|
| v3.2 | 3,39 | **2,55** | 1,66 | 1,39 |
| P2 @2k | 4,95 | **4,22** | 2,42 | 2,68 |
| P2 @6k | 7,71 | **6,15** | 4,68 | 3,74 |

⇒ 6d **làm đều hơn** (đúng vai trò lưới an toàn), nhưng giọng thật p90 = 7,59 ⇒ càng đều càng **xa
giọng thật**. Vì vậy mặc định TẮT là đúng; chỉ bật nếu tai người dùng thấy lỗi "từ tự to" mà không
muốn mất nhấn. Cần trang nghe A/B trước khi chốt.

## 5. Việc 5 — top spike (chỉ báo cáo)

`tools/top_spike_inspect.py` (mới): vẽ mel THẬT quanh âm tiết + khoảng âm tiết theo `durations` +
phone MFA, so mức thật với trung vị segment. Kết quả 10 top spike của v3.2:

| # | từ | Δ dB (model−thật) | thật − trung vị | phone thật |
|---|---|---|---|---|
| 1 | qua | +21,2 | **−22,6** | k w aː˨˨ l |
| 2 | Ở | +17,7 | **−39,2** | ʔ əː˨˩˨ |
| 3 | ét | +15,8 | **−16,9** | ɛː˦˥ t̚ |
| 4 | Ở | +14,8 | **−37,0** | ʔ əː˨˩˨ ɗ |
| 5 | ai | +14,7 | **−24,3** | ʔ aː˨˨ j |
| 6 | Còn | +13,8 | **−26,9** | k ɔː˨˨ n |
| 7 | qua | +12,2 | **−23,5** | k w aː˨ː l |
| 8 | thì | +12,0 | **−18,9** | tʰ iː˨˨ |
| 9 | mình | +11,9 | **−6,7** | m i˨˨ ɲ s |
| 10 | vợt | +11,6 | **−16,9** | v əː˨˩˨ k̚ |

**Kết luận: cả 10 "spike" đều là frame THẬT gần lặng (−6,7…−39,2 dB so trung vị segment)** — không
phải model tự làm to. Nguyên nhân: biên segment/align rơi vào khoảng lặng (đầu/cuối segment), hoặc
âm tiết thật đọc rất nhẹ. **Không sửa dữ liệu** theo đúng chỉ đạo. Ảnh: `artifacts/top_spikes/*.png`
(local, không commit).

## 6. Việc 3 — kết quả E1–E3 (cập nhật)

Khởi tạo từ **P2 @2k EMA** (trọng số sống + EMA = EMA P2@2k), cấu hình P2 (adv 0,3 · fm 0 · hinge),
mỗi nhánh 4k bước, `energy_weight 1,0` (E1/E2/E3), E2 thêm `lowband_weight 2,0`, E3 adv 0,2.
`E0` = P2 @4k/@6k đã có (đo lại bằng thước 40 câu ở mục 1.3).

| nhánh | step | **val EMA** | **GV EMA** | GV live | drift p90 | ghi chú |
|---|---|---|---|---|---|---|
| **E0** (P2 chạy tiếp) | 4k | 1,65 (P2 4k) | 0,66* | — | **7,28** | gần giọng thật nhất |
| E0 | 6k | — | — | — | **7,71** | |
| **E1** (energy) | 2k | 1,6436 | 0,6487 | 0,6835 | 3,94 | |
| **E1** | 4k | **1,6505** | **0,6718** | 0,7253 | **3,76** | **trượt cổng GV ≥ 0,72** |
| **E2** (energy+lowband) | 2k | 1,6268 | 0,6624 | 0,6657 | — | |
| **E2** | 4k | **1,6178** | **0,6722** | 0,6886 | — | **trượt cổng GV ≥ 0,72** |
| E3 (energy, adv 0,2) | — | — | — | — | — | **huỷ theo v11** |

\* GV của P2 4k lấy từ val_log của run P2 20k gốc (cùng cấu hình, EMA).

**Kết luận bước đầu:** E1 **giảm drift p90** (4,95 → 3,76) tức làm giọng **đều hơn**, đi **xa** giọng
thật (7,59), đồng thời **GV EMA rơi xuống 0,672 < 0,72** ⇒ theo điều kiện chọn của v10, E1 **không đạt**.
Điều này khớp với nhận định ở mục 1.5: "sửa độ to" bằng energy loss đang tối ưu **sai hướng** so với
thước drift mới (thước muốn nhiều biến thiên hơn, không phải ít hơn).


## 7. Cần A0 quyết
1. **Cổng drift**: giao rỗng — chọn (a) chỉ "gần thật ±1,5 dB", (b) chỉ "≤ v3.2+2", hay (c) định
   nghĩa mới tách **spike trong mệnh đề** (đề xuất: p95−median RMS trong chunk, đo cả giọng thật).
2. **6d**: bật hay không sau khi nghe A/B.
3. **P2 @2k** có tiếp tục là production candidate không, khi P2 4k/6k gần giọng thật hơn về drift
   nhưng tai người dùng đã chọn 2k.

## 8. Việc 7 — chẩn đoán 2 lỗi chung của mọi model (d04, d06)

### 8.1 Độ phủ âm tiết trong train (2.1)
Train v3.2: **220.384 token / 2.552 type**. Các từ cần soi:

| âm tiết | đúng (kể thanh) | cùng vần bỏ thanh | cùng coda |
|---|---|---|---|
| **ách** | **1** | 516 | 2.401 |
| **treo** | **1** | 721 | 5.203 |
| ếch | 20 | 249 | 2.401 |
| bấm | 37 | 294 | 5.626 |
| tường | 27 | 1.263 | 17.878 |
| ao | 46 | 4.459 | 5.203 |
| rồi | **1.714** | 3.994 | 26.425 |

Trong 40 câu drift có **62 âm tiết < 10 lần**. ⇒ d04 ("ách", "treo" = 1 lần) là **lỗ hổng độ phủ**,
không sửa được bằng loss; "rồi" thì rất phổ biến ⇒ lỗi prosody, không phải thiếu dữ liệu.

### 8.2 "rồi" quá trầm (2.2)
Frontend d06: `rồi` = onset **R**, nucleus **Ô**, coda **J**, tone **HUYEN**, đứng đầu mệnh đề sau
dấu phẩy (trong steady mode = đầu chunk).

| nguồn | semis (so trung vị segment) | contour (cuối−đầu) | creak% | dur |
|---|---|---|---|---|
| **THẬT — đầu mệnh đề** (n=62) | **−1,34** | **−2,50** | **0 %** | 24 |
| THẬT — giữa (n=282) | −1,77 | −1,88 | 0 % | 16 |
| THẬT — cuối (n=91) | −2,47 | −2,35 | 0 % | 19 |
| **Model TF** (n=20, đầu segment) | 0,00 | **−0,78** | **23,1 %** | 27,5 |
| **Model free-run** (n=20) | 0,00 | −1,16 | **25,4 %** | 24,5 |
| **v3.2 · d06 free-run** | 0 (108,4 Hz) | **+2,56 (LÊN)** | **19,3 %** | — |
| **P2 @2k · d06 free-run** | 0 (110,2 Hz) | **+3,13 (LÊN)** | **21,2 %** | — |

**Nén f0 toàn cục** (532/514 âm tiết, TF, v3.2/P2@2k): slope model-vs-thật **0,59 / 0,69**, corr
**0,67 / 0,66**; dải thật p10–p90 −3,1…+3,3 semitone. Tỉ lệ âm tiết lệch |ΔF0| > 3 semitone:
**23 %** (v3.2), 10–26 % (P2@2k). **f0 halving trong nhãn: 0,78 %** (HUYEN 0,5 %) ⇒ **không** phải
nguyên nhân.

**Đọc theo cây quyết định của A0:** TF cũng lệch (creak 23 %, contour −0,78 thay vì −2,5) ⇒ lỗi nằm ở
**đầu f0/decoder**, không phải duration/free-run. Cụ thể: (a) biên độ f0 bị **nén 31–41 %**;
(b) ~20–25 % frame mất cao độ (creak) ở đầu chunk; (c) contour **đi lên** thay vì đi xuống.

### 8.3 "ách / ếch" rè (2.3)
Flatness 2–6 kHz (Wiener entropy; cao = rè). Model (v3.2 / P2@2k) so với **giọng thật cùng từ**:

| từ | model v3.2 | model P2@2k | **thật** | tỉ lệ |
|---|---|---|---|---|
| trên | 0,299 | 0,291 | 0,133 | 2,2× |
| con | 0,294 | 0,256 | 0,123 | 2,3× |
| thì | 0,347 | 0,285 | 0,140 | 2,3× |
| ao | 0,401 | 0,363 | 0,143 | 2,7× |
| cũ | 0,441 | 0,416 | 0,255 | 1,7× |
| **ách** | 0,335 | 0,258 | 0,190 (n=1) | 1,4–1,8× |
| **ếch** | 0,243 | 0,204 | 0,194 (n=19) | 1,1–1,3× |
| treo | 0,285 | 0,297 | 0,270 (n=1) | 1,0–1,1× |
| tường | 0,266 | 0,279 | 0,149 (n=27) | 1,8× |

**Tách nguồn rè:** cho **mel THẬT** qua vocoder rồi đo lại: thật **0,152** → vocoder **0,177**
(chỉ +16 %); model đầy đủ 0,25–0,44 ⇒ **rè đến từ MEL DỰ ĐOÁN**, không phải vocoder.
Với "ách/treo" phần lớn là do **độ phủ 1 lần**; với "trên/con/thì/ao/tường" là lỗi mel toàn cục.

### 8.4 Grad norm từng loss trên encoder (3.1 — đo trước khi đổi f0_delta)
Batch 8×49 âm tiết, mỗi loss bật riêng weight 1,0, |grad| trên `encoder.*`:

| loss | giá trị | grad enc | % tổng |
|---|---|---|---|
| **pause_gate** | 0,993 | **3,848** | **83,4 %** |
| **pause_gap** | 0,475 | **1,843** | **39,9 %** |
| **f0_delta** | 6,438 | **1,323** | **28,7 %** |
| f0_prior | 0,266 | 0,676 | 14,7 % |
| mel_pre | 0,733 | 0,554 | 12,0 % |
| mel | 0,714 | 0,498 | 10,8 % |
| energy | 0,803 | 0,446 | 9,7 % |
| phonation | 0,548 | 0,246 | 5,3 % |
| duration | 0,117 | 0,205 | 4,4 % |
| f0 | 0,574 | 0,146 | 3,2 % |
| vuv | 0,077 | 0,094 | 2,0 % |
| mel_sharp | 0,593 | 0,055 | 1,2 % |
| ratio | 0,866 | 0,037 | 0,8 % |
| tone | 0,001 | 0,028 | 0,6 % |

⇒ **hai đầu pause chiếm 83 % + 40 % grad encoder** (đúng như lo ngại §43.1), f0_delta 28,7 %,
mel chỉ 10,8 %. Hạ f0_delta 0,5→0,1 sẽ đưa nó về ~6 % và trả capacity cho mel — **hợp lý**, nhưng
cần xác nhận vì lỗi f0 hiện tại là **nén biên độ**, không phải delta.

## 9. Sửa thước 4a (v11 §4)

`tools/adv_energy_audit.py`: loại âm tiết có mức THẬT < trung vị segment − 10 dB trước khi tính
spike rate/MAE và top spike. Chạy lại v3.2 (40 segment):

| | spike rate | MAE | RMS spike | RMS MAE | n |
|---|---|---|---|---|---|
| cũ (chưa lọc) | 0,1859 | 2,589 | 0,0776 | 1,769 | 683 |
| **mới (đã lọc)** | **0,1388** | **2,214** | **0,064** | **1,710** | **641** |

Top spike mới: 7,8 / 7,3 / 7,1 / 6,6 / 6,5 dB (là / lịch / páp / hướng / có) — hết hẳn nhóm
+15…+22 dB của frame thật gần lặng. Số cũ **không dùng để so nữa**.

**P2 @2k (đã lọc, cùng 40 segment): spike 0,140 · MAE 2,199 · n=641 — nhưng RMS spike 0,114
so với 0,064 của v3.2 (1,78×) và RMS MAE 1,9 vs 1,71.** Tức GAN giữ nguyên mức mel nhưng làm
**mức waveform lệch nhiều hơn** — khớp với tai người dùng ("2k vẫn méo, 6k méo hơn").
Đây là lý do nên đo cả RMS, và là cảnh báo cho v3.5: cần kiểm RMS spike sau GAN.

## 10. v3.5 — kế hoạch (CHƯA train)

Theo v11 §3, gộp một lần; các mục cần code mới được đánh dấu:

1. **Mask frame lặng khỏi mel L1** — chỉ giữ ±4 frame quanh biên (code: `slot_losses`/`mel_error`).
2. **Mel stats tính chỉ trên frame tiếng nói** — đổi `data/codelung/mel_stats.json` + `mel_hash`
   (đổi khoá cache ⇒ config riêng, không sửa v32).
3. **f0_delta 0,5 → 0,1** — đo được f0_delta chiếm **28,7 %** grad encoder; hạ còn ~6 % để trả
   capacity cho mel. **Cần xác nhận**: lỗi f0 hiện tại là nén biên độ (slope 0,59–0,69), không phải
   delta; nếu hạ quá tay, contour có thể tệ hơn. Đề xuất: 0,3 trước, đo slope/corr, rồi mới 0,1.
4. **Bỏ phonation loss + conditioning** (5,3 % grad encoder; không cải thiện rè).
5. **EMA 0,999 toàn bộ train**.
6. **Slot N nhận embedding coda trực tiếp; slot C nhận embedding nucleus** (code: `slot_ops`/model).
7. **Giảm áp lực 2 đầu pause**: pause_gate + pause_gap chiếm **83 % + 40 %** grad encoder. Đề xuất
   `pause_weight` 0,5 → 0,1 và dùng `pos_weight` (đã có `PAUSE_POS_WEIGHT`) thay vì tăng weight.
8. **Nhãn f0**: halving 0,78 % (HUYEN 0,5 %) ⇒ **không** cần sửa hàng loạt; chỉ log.
9. **KHÔNG energy loss** (E1/E2 đều trượt GV).

Thứ tự train đề xuất: L1 tới hội tụ (60k) → GAN riêng (adv 0,3 · fm 0 · hinge · warmup 300 ·
EMA 0,999) → chọn checkpoint sớm (~2k) theo cổng drift ≤ v3.2+2 (đọc **cận trên CI**) + tai.

## 11. Việc 8a — tách nguồn lỗi prosody (sửa luôn lỗi semis = 0,00)

**Lỗi đo cũ (A0 nghi đúng):** ở chế độ model mình truyền **cả chunk** vào `_stats`, nên "semis" luôn
= 0 và creak/contour là của cả chunk, không phải token "rồi". Đã sửa: cắt đúng token theo
`durations` + `pause_gap` (mục 8.2 ở trên là số CŨ, không dùng).

Đo lại 12 segment có "rồi" ở đầu, tham chiếu f0 trung vị THẬT của segment, f0 đo trên AUDIO:

| model | chế độ | semis | contour | creak | **slope** | corr |
|---|---|---|---|---|---|---|
| **P2 @2k** | tf_dur | −1,85 | −1,87 | 0,02 | **0,79** | 0,75 |
| **P2 @2k** | **tf_full** | −1,09 | −0,87 | 0,03 | **1,02** | 0,80 |
| **P2 @2k** | free | −2,22 | −3,06 | 0,057 | 0,81 | 0,73 |
| v3.2 | tf_dur | −1,85 | −1,52 | 0,025 | 0,81 | 0,76 |
| v3.2 | **tf_full** | −0,99 | −0,39 | 0,045 | **0,59** | 0,66 |
| v3.2 | free | −1,83 | −2,32 | 0,077 | 0,81 | 0,77 |

Giọng thật (đầu mệnh đề): semis **−1,34**, contour **−2,50**, creak **0 %**.
Phonation dự đoán (hnr, glot, jitter) `[-1,47, 0,03, 0,11]` vs thật `[-1,45, 0,14, 0,14]` ⇒ đầu
phonation gần đúng (glottalization hơi thấp).

**Đọc:** với P2 @2k, tf_full đưa slope về **1,02** ⇒ **decoder không lỗi**, lỗi ở đầu dự đoán f0.
Nhưng với v3.2, tf_full lại **0,59** ⇒ đường f0 → mel **không ổn định giữa các checkpoint** (xem 8b).

## 12. Việc 8b — giãn f0 ở runtime: **KẾT QUẢ ÂM** (quan trọng)

Thêm `--f0-scale` (mặc định 1,0): `f0_new = mean_câu + k·(f0 − mean_câu)` trong log-F0, chỉ frame
hữu thanh, trước `variance_proj`. Đo TF-dur trên 40 segment val (P2 @2k):

| k | slope | corr | model p10–p90 |
|---|---|---|---|
| 1,00 | 0,785 | 0,633 | −4,4…+2,1 |
| 1,25 | 0,780 | 0,650 | −4,4…+2,2 |
| 1,45 | 0,783 | 0,634 | −4,4…+2,1 |

**Nguyên nhân:** đầu f0 CÓ đổi (p10/p90 99/126 → 93/133 Hz) nhưng **cao độ audio gần như không đổi**
(84/98/115 → 84/98/114 Hz). Kiểm chứng bằng cách đặt **cột f0 = 0** trong `variance_override`:
audio chỉ giảm 5–9 % (110,9→104,4 · 138,7→129,4 · 135,5→121,2 Hz), **vẫn còn cao độ**.
⇒ Cao độ đầu ra chủ yếu do **hidden state của decoder** quyết định; **kênh f0 chỉ ảnh hưởng ~10 %**.
Vì vậy **không thể sửa prosody bằng scaling ở runtime** — phải sửa kiến trúc (xem mục 13).

## 13. v3.5 — bổ sung do 8a/8b

Thêm vào danh sách v11/v12:
- **Nối f0 vào decoder bằng FiLM/pitch-embedding** (không chỉ concat qua `variance_proj`), và
  **f0 dropout/perturbation ±2 semitone khi train** để decoder buộc phải dùng tín hiệu f0.
  Cổng: slope ≥ 0,75 **ở cả TF-dur và free-run**, không chỉ TF-full.
- Giữ `--f0-scale` trong code nhưng **mặc định 1,0** (đã chứng minh vô tác dụng).
- f0_delta: giá trị **6,4 là z-unit** (L1 trên delta hệ số DCT của log-F0 **đã z-score** theo
  `f0_stats`, chia cho `len(DELTA_POSITIONS)`), **không phải Hz/semitone**. Quy đổi thô: lệch contour
  ~1 semitone ⇒ ~0,5–1,0 đơn vị semitone. Vì vậy hạ 0,5→0,2 là hợp lý; muốn so trực tiếp bằng
  semitone thì phải đổi loss sang thang semitone (chưa làm).

## 14. Việc 8c — kịch bản thu bổ sung v1

`tools/build_recording_script.py` → **`docs/recording_script_v1.md`**:
- **800 câu** + **56 câu cặp tối thiểu**, **9.234 âm tiết** ≈ **34 phút** đọc (2 lần ≈ 68 phút).
- Phủ **600/600** âm tiết hiếm (< 10 lần) đạt ≥ 3 lần.
- Kèm hướng dẫn thu (cùng micro, TẮT RNNoise, WAV 48 kHz 24-bit, đọc đều, 2 lần/câu).
- Hạn chế: câu sinh theo **khuôn** (carrier) để phủ âm tiết, không phải văn xuôi tự nhiên; nếu cần
  tự nhiên hơn thì cần một corpus văn bản để chọn câu (chưa có trong repo).
