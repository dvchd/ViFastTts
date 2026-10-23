# V18.39 — Agent 2: ổn định đánh giá tự động, không training

> **Trạng thái: DỪNG SAU V18.39, CHỜ ĐỌC.** Không training, không đổi production,
> không chuyển ONNX mới. Giữ **P2 EMA**; crop fixed giữ làm mặc định **vì loại padding**,
> **không** tuyên bố cải thiện chất lượng.
>
> **Commit gốc của vòng**: `5e31677` · **Tệp đặc tả**: yêu cầu A0 → Agent 2 trong chat
> (không có `V1839_agent0.md`) · **Test**: 2.556 passed · **Ruff**: sạch.
> **Working tree** khi chấm: `5e31677` + tool v1839 chưa commit + 2 file test user chưa stage;
> **không dùng worktree sạch**.

---

## 1. Khóa v18.38

- P2 EMA vẫn là production; không đổi checkpoint.
- Crop fixed vẫn là mặc định trong code vì **loại padding** (không phải vì thắng chất lượng).
- Không kéo dài probe, không mở nhánh GAN mới.
- Không chuyển production sang ONNX mới; `artifacts/onnx/v1837/` giữ nguyên.

---

## 2. Điều chỉnh kết luận v18.38

1. **Fixed vs legacy**: v18.38 nói ΔUTMOS −0,0507 (CI95 không chứa 0). Chấm lại 5 seed:
   **−0,0264, CI95 [−0,0554, +0,0190]** ⇒ **không còn ý nghĩa**. Kết luận đúng:
   **chưa có bằng chứng khác biệt rõ**, **không** gọi tương đương.
2. **Fixed vs REC**: v18.38 nói ΔUTMOS +0,1064 (CI95 dương). Chấm lại:
   **+0,0176, CI95 [−0,0003, +0,0530]** ⇒ **chỉ là xu hướng**, chưa chứng minh GAN có lợi
   ổn định. BAK vẫn fixed hơn REC (+0,0201 CI [0,0032, 0,0679]), flat fixed tốt hơn.
   Nhưng **fixed chưa hơn P2**.
3. **HNR giảm nhẹ** (fixed vs P2 −0,028 dB; fixed hơn legacy +0,036 dB) **không tự chứng minh
   chất lượng cảm nhận giảm/tăng**. HNR là proxy; không có tai người.
4. **Bỏ quy tắc “ΔUTMOS < 0,05 không đáng tin”**. Range của một file không phải ngưỡng toàn tập.
   Số liệu đúng: mỗi file lệch trung bình **~0,28 điểm** giữa 5 seed (max 0,45–0,67);
   same-seed lặp **exact 0,0**. Vì vậy phải dùng **trung bình nhiều seed**, không dùng một pass.
5. **CI bootstrap theo video chưa gồm bất định do giám khảo**. Trong tài liệu này:
   CI bootstrap video tính trên **trung bình 5 seed**; **seed range** báo riêng, không cộng vào CI
   và không coi 5 seed × 36 câu là 180 quan sát độc lập.

---

## 3. Audit UTMOSv2 (nguồn, cấu hình, random)

### 3.1 Nguồn thực thi

| Mục | Giá trị |
|---|---|
| Package | `utmosv2 1.3.1.dev0` |
| Config | `fusion_stage3`, fold 0, seed 42 |
| Checkpoint | `~/.cache/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth` |
| Checkpoint SHA256 | `c8149d988e4bbf3f347e6966b5d769de347a5f8c59ffca1dc4bd4bf5b8585e57` |
| Config file SHA256 | `9e30618e51ddbcf77cb2d3bbd66e3f194833c06f3141abb626ba385eb38f1c05` |
| Device | MPS |
| Model mode | `self._model.eval().to(device)` trong `_predict_impl` |
| Gradient | `torch.no_grad()` |
| DataLoader | `shuffle=False`; batch 8 ở `utmos_dir`, batch 1 ở audit per-file |
| Resample | WAV 24 kHz → `librosa.load` native → `librosa.resample` 16 kHz |
| Silence | `remove_silent_section=True` (mặc định) |
| Dataset domain | `predict_dataset="sarulab"` (mặc định) |
| Repetitions | audit: 1 pass/lần; 5 seed; **không** dùng `num_repetitions=5` một lần |

### 3.2 Nguồn ngẫu nhiên — ĐÃ XÁC ĐỊNH

`utmosv2` dùng random crop/mixup **ngay ở prediction**, không seed:

- `utmosv2/dataset/_utils.py:35-37`:
  `start = np.random.randint(0, y.shape[0] - length)`.
- `utmosv2/dataset/ssl.py:48`: SSL crop 3 s dùng `select_random_start`.
- `utmosv2/dataset/multi_spec.py:52-63`: 2 frame × 1,4 s dùng `select_random_start`;
  `mixup_inner=True` dùng `np.random.beta(0.4, 0.4)`.
- Không có `np.random.seed` trong package; `create_model(seed=42)` chỉ dùng seed để chọn
  **tên checkpoint** (`fold0_s42...`), không seed RNG.
- `utmos_dir` cũ (`tools/v1833_judges.py`) gọi `model.predict(input_dir=..., batch_size=8)`
  **không truyền `num_repetitions`** ⇒ mặc định **1**; config `inference.num_tta=5` không được dùng.
  Đây là **lỗi cấu hình chấm**, không phải lỗi model.

### 3.3 Kiểm tra same-seed

Với cùng file, cùng seed:
- 5 seed khác nhau cho điểm khác (ví dụ một file P2: 0,8853 / 0,9498 / 1,3343 …);
- lặp **đúng seed 0** cho **max abs diff = 0,0** trên cả 4 bản/12 câu.
⇒ Nguồn ngẫu nhiên là **RNG của dataset**, không phải MPS/model; seed hóa được.

---

## 4. Kiểm tra lặp 12 câu × 5 seed

### 4.1 Bộ cố định

Chọn **trước khi chấm**: 12 câu, 12 video thật (lấy `parent`, không dùng `id.split("_")[0]`),
4 short / 4 mid / 4 long. File: `artifacts/bench/v1839_utmos_audit/selection.json`.
Mỗi WAV có SHA256, sample rate, số mẫu, độ dài.

### 4.2 Dao động từng file (5 seed)

| Bản | range trung bình | median range | max range |
|---|---:|---:|---:|
| P2 EMA | 0,2926 | 0,2847 | 0,4490 |
| REC | 0,2976 | 0,3207 | 0,6694 |
| legacy | 0,2994 | 0,2644 | 0,5348 |
| fixed | 0,2807 | 0,2807 | 0,5510 |

⇒ Một file có thể lệch **0,28 điểm trung bình**, tối đa **0,67** giữa các seed.
Không được dùng 1 pass để so từng file.

### 4.3 Delta ghép cặp theo seed (12 câu)

| So sánh | mean của 5 seed | seed range | chiều 5/5 seed |
|---|---:|---:|---|
| rec − P2 | −0,1137 | 0,0827 | âm cả 5 |
| legacy − P2 | +0,0278 | 0,0494 | dương cả 5 |
| fixed − P2 | −0,0344 | 0,0454 | âm cả 5 |
| fixed − legacy | −0,0622 | 0,0405 | âm cả 5 |
| fixed − rec | +0,0794 | 0,0932 | dương cả 5 |

- Chiều **trung bình** ổn định trong 5 seed, nhưng **từng câu** hỗn hợp:
  ví dụ fixed−legacy: **0/12** câu dương cả 5 seed, **5/12** câu âm cả 5, **7/12** câu đổi dấu.
  ⇒ Không coi 12 câu × 5 seed là 60 quan sát độc lập.
- Xếp hạng 12 câu **ổn định cả 5 seed**: `legacy > P2 > fixed > rec`.

### 4.4 Same-seed vs different-seed

- Cùng seed 0 lặp lại: **max abs diff 0,0** trên cả 4 bản.
- Khác seed: range trung bình **0,2807–0,2994**; max 0,4490–0,6694.
⇒ Nếu chỉ chạy 1 lần, điểm là **một mẫu random**, không phải ước lượng ổn định.

---

## 5. Chấm lại 36 câu bằng cấu hình đã sửa

Không render lại audio. Dùng đúng WAV của v18.38; mỗi file **trung bình 5 seed [0..4]**,
mỗi seed 1 pass; cùng checkpoint/config/input hash.

- Input set SHA256: `71aa31e13984087c9f738968c670c972d2b17d05519e7972f6452ab2a6dc66b5`
- UTMOS checkpoint/config SHA như §3.1.
- BAK giữ nguyên từ v18.38 (cùng WAV, DNSMOS không dùng random crop).

### 5.1 Mean

| Bản | UTMOS seeded | BAK (v18.38) |
|---|---:|---:|
| P2 EMA | 1,4899 | **3,0341** |
| REC | 1,4568 | 3,0428 |
| legacy | **1,5207** | 3,0169 |
| fixed | 1,4957 | 3,0247 |

### 5.2 Ghép cặp (36 câu, bootstrap theo 12 video)

| So sánh | ΔUTMOS median [CI95] | ΔBAK median [CI95] | UTMOS thắng/thua | BAK thắng/thua |
|---|---:|---:|---:|---:|
| fixed − legacy | **−0,0264** [−0,0554, +0,0190] | +0,0041 [−0,0112, +0,0084] | 15/21 | 20/16 |
| fixed − P2 | **−0,0243** [−0,0384, +0,0112] | −0,0042 [−0,0269, +0,0092] | 15/21 | 18/18 |
| fixed − rec | **+0,0176** [−0,0003, +0,0530] | **+0,0201** [+0,0032, +0,0679] | 21/15 | 21/15 |
| legacy − P2 | **+0,0223** [+0,0045, +0,0461] | −0,0061 [−0,0164, +0,0072] | 26/10 | 17/19 |
| rec − P2 | **−0,0748** [−0,0970, −0,0081] | +0,0048 [−0,0528, +0,0198] | 13/23 | 20/16 |

**Đọc đúng**:
- **fixed vs legacy**: UTMOS CI chứa 0; BAK CI chứa 0 ⇒ **không có bằng chứng khác biệt**,
  không gọi tương đương. HNR fixed hơn, flat fixed kém hơn — tín hiệu trái chiều.
- **fixed vs P2**: UTMOS CI chứa 0; BAK CI chứa 0; HNR fixed kém nhẹ ⇒ **không hơn P2**.
- **fixed vs REC**: BAK CI dương; UTMOS CI sát 0 (cận dưới −0,0003) ⇒ fixed hơn REC trên BAK,
  UTMOS chỉ xu hướng; **vẫn chưa hơn P2**.
- **legacy vs P2**: UTMOS CI dương, nhưng BAK không xác nhận; không đủ để đổi production.

### 5.3 Ổn định thứ hạng 36 câu

Xếp hạng theo **mean UTMOS** ổn định cả 5 seed:
`legacy > fixed > P2 > rec` (cả 5 seed).
Nhưng **paired median** fixed−P2 là **−0,0243** (fixed thấp hơn), trong khi mean fixed−P2 là
**+0,0058**. Mean bị kéo bởi outlier; **không dùng mean-rank làm bằng chứng fixed > P2**.
Đây là lý do báo cả median + mean + per-video.

Per-video (số video có delta dương):
- fixed−legacy: **3/12**; fixed−P2: **6/12**; fixed−rec: **8/12**;
  legacy−P2: **8/12**; rec−P2: **3/12**.

### 5.4 So với v18.38 — kết luận nào giữ / hạ

| Kết luận v18.38 | Trạng thái sau 5 seed |
|---|---|
| fixed vs legacy ΔUTMOS −0,0507 (CI<0) | **HẠ**: −0,0264 CI chứa 0; không còn ý nghĩa |
| fixed vs rec ΔUTMOS +0,1064 (CI>0) | **HẠ**: +0,0176 CI sát 0; chỉ xu hướng |
| legacy vs P2 ΔUTMOS +0,015 | **NÂNG NHẸ**: +0,0223 CI dương, nhưng BAK không xác nhận |
| rec vs P2 ΔUTMOS −0,0864 | **GIỮ**: −0,0748 CI âm |
| fixed vs P2 ΔUTMOS −0,009 | **GIỮ**: −0,0243 CI chứa 0; không hơn P2 |
| BAK/flat/HNR của v18.38 | **GIỮ** (cùng WAV; BAK deterministic) |
| “ΔUTMOS < 0,05 không đáng tin” | **BỎ** — thay bằng 5-seed protocol + per-file range |

---

## 6. Audit đối chứng probe v18.38 (từ source/log)

### 6.1 RNG

- Crop dùng `random.Random(args.seed)` **riêng** (`tools/v1838_probe.py`), không dùng
  `np.random`/torch RNG của dropout.
- Dropout dùng torch global RNG. Ở đầu pha G, probe **reset**
  `torch.manual_seed(seed + d_warmup + 1)` cho cả 3 nhánh vì REC bỏ qua 300 forward D warm-up.
  Đây là **TỰ QUYẾT** của probe, không phải hành vi `finetune_adv` gốc.
- Batch order: materialize 500 batch index; `batch_order_hash` giống nhau.
- D init: một template, copy state; `d_init_sha` giống nhau.
- G init: cùng P2 EMA; `g_init_sha` giống nhau.

### 6.2 Cùng gì / khác gì

| Mục | Kết luận |
|---|---|
| Cùng batch order | **Có** — cùng `batch_indices.json`, hash |
| Cùng crop RNG seed/state | **Có** giữa legacy/fixed; REC không crop |
| Cùng torch RNG state toàn trajectory | **Không** — REC bỏ 300 forward; đã reset ở đầu G |
| Cùng dropout realization | **Bằng chứng gián tiếp**: milestone g0 rec = 1,2448 và g1 rec = 1,2755 giống nhau ở cả 3 nhánh. Không log mask từng bước, nên không khẳng định 200 bước. |
| Cùng module trainable | **Có** — 2.823.496 tham số decoder/mel/postnet |
| Cùng dữ liệu | **Có** — cùng manifest, cùng batch index |

### 6.3 Module đóng băng vẫn có dropout

`model.train()` đặt **tất cả** top-level ở train mode, kể cả module đóng băng:
`encoder`, `duration`, `pause`, `slot_expander`, `ratio_mlp`, `variance_mlp` có dropout 0,1.
`embedding`, `g_token`, `variance_proj`, `position_proj`, `length_regulator` không dropout.
⇒ Không nói “frozen nên không random”; frozen chỉ không có gradient, dropout vẫn hoạt động.

**Không chạy lại training để sửa đối chứng trong vòng này.**

---

## 7. Nguồn dao động & việc tiếp

### Đã xác định
- UTMOSv2 random crop/mixup không seed; một pass là một mẫu random.
- `utmos_dir` cũ dùng `num_repetitions=1`, không dùng TTA 5 trong config.
- Same seed → exact; different seed → per-file range lớn.

### Nghi ngờ
- Khác biệt nhỏ giữa `input_dir` batch 8 (v18.38) và `input_path` batch 1 (v18.39)
  có thể do batching/preprocessing; chưa tách riêng. Không đủ để đổi kết luận.
- MPS numeric không thấy dao động khi cùng seed (max diff 0,0), nên không phải nguồn chính.

### Chưa xác định
- Vì sao mean-rank fixed > P2 nhưng paired median fixed < P2: phân bố lệch/outlier.
  Không dùng mean-rank làm kết luận.
- Ảnh hưởng của `remove_silent_section` và resample 24→16k lên từng file: chưa ablation.

### Cấu hình chấm tái lập
`tools/v1839_utmos_audit.py`:
- 5 seed [0..4], mỗi seed 1 pass;
- `input_path` từng file, 24 kHz, `librosa` resample 16 kHz;
- `remove_silent_section=True`, `num_repetitions=1`, `predict_dataset="sarulab"`;
- checkpoint/config hash ghi trong `audit12.json`/`rescore36.json`.

**Việc tiếp tối thiểu**: nếu cần kết luận UTMOS, dùng 5-seed trung bình + BAK/flat/HNR;
nếu vẫn mâu thuẫn, chấm trên bộ độc lập/30 câu người thật. Không mở training dài.

---

## 8. Test / commit / artifact

- **Test**: `pytest -q` → **2.556 passed**; `ruff check .` sạch.
- **Working tree**: `5e31677` + tool v1839 chưa commit + 2 file test user chưa stage;
  **không** dùng worktree sạch.
- **Không stage/revert** `tests/test_custom_reading.py`, `tests/test_eval_stats.py`.
- **Không đổi** production, ONNX, checkpoint gốc.
- Artifact:
  `artifacts/bench/v1839_utmos_audit/selection.json`,
  `audit12.json`, `rescore36.json`, `rescore36_stats.json`.
- Không dùng smoke ONNX output length làm parity chất lượng.

**Kết luận cuối**: v18.38 giữ **P2**, giữ crop fixed vì loại padding; **không** có bằng chứng
crop-fixed cải thiện chất lượng. UTMOSv2 chỉ dùng được khi seed + trung bình nhiều lượt;
các ΔUTMOS một-pass của v18.38 đã hạ mức.
