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

### 1.4 Cổng của A0 KHÔNG THỂ ĐẠT (cần A0 quyết)
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

(xem bản cập nhật cuối file)

## 7. Cần A0 quyết
1. **Cổng drift**: giao rỗng — chọn (a) chỉ "gần thật ±1,5 dB", (b) chỉ "≤ v3.2+2", hay (c) định
   nghĩa mới tách **spike trong mệnh đề** (đề xuất: p95−median RMS trong chunk, đo cả giọng thật).
2. **6d**: bật hay không sau khi nghe A/B.
3. **P2 @2k** có tiếp tục là production candidate không, khi P2 4k/6k gần giọng thật hơn về drift
   nhưng tai người dùng đã chọn 2k.
