# V18.38 — Agent 2: probe crop D có đối chứng (không đổi production)

> **Trạng thái: DỪNG SAU V18.38, CHỜ ĐỌC.** Probe chẩn đoán 1 seed, 200 G update.
> Không đổi production, không đổi kiến trúc/loss, không ép vuv, không train dài.
> Production vẫn **P2 EMA** + Vocos pretrained; artifact ONNX v1837 giữ nguyên.

**Commit gốc khi chạy probe**: `f1baef0` · **Working tree**: có 2 file test của user chưa stage
(`tests/test_custom_reading.py`, `tests/test_eval_stats.py`) + tools v1838 mới; **không dùng
worktree sạch**. Probe dùng code tại commit f1baef0; tool probe/eval có SHA riêng trong
`runs/v1838_crop_probe/report.json`.

---

## 1. Kết luận ngắn

1. **Crop fixed xóa padding thật**: legacy có **35/3.975 row (0,88%)** và **31/500 bước (6,2%)**
   bị padding, tổng **1.098 frame**, max **64 frame**; fixed **0**. Đây là sửa tính đúng.
2. **Không nhánh nào tốt hơn P2 EMA** trên bộ 36 câu dev:
   - fixed vs P2: ΔBAK **−0,0042** CI95 [−0,0269, +0,0092]; ΔUTMOS −0,009 CI95
     [−0,0656, +0,0212]; ΔHNR **−0,028** CI95 [−0,061, −0,0045]; Δflat +0,0019
     (cao hơn = kém hơn) CI95 [−0,0006, +0,0033].
   - legacy vs P2: ΔBAK −0,0061 CI95 [−0,0164, +0,0072]; ΔUTMOS +0,015 CI95
     [−0,0514, +0,0483]; ΔHNR −0,1115 CI95 [−0,201, −0,033].
   - rec vs P2: ΔBAK +0,0048 CI95 [−0,0391, +0,0198]; ΔUTMOS **−0,0864** CI95
     [−0,1391, −0,046]; Δflat **+0,0071** (kém); ΔHNR **−0,157** (kém).
3. **Fixed vs legacy: không có thắng rõ**:
   ΔBAK +0,0041 CI95 [−0,0112, +0,0084]; ΔUTMOS **−0,0507** CI95 [−0,1131, −0,0191]
   (fixed kém hơn); Δflat +0,0011 (fixed phẳng hơn = kém hơn); ΔHNR +0,036
   (fixed hơn). **Không thể nói sửa crop làm GAN tốt hơn.**
4. **Fixed vs REC**: fixed hơn REC về BAK (+0,0201 CI [0,0032, 0,0674]), UTMOS
   (+0,1064 CI [0,0246, 0,135]), flat (−0,005, thấp hơn = tốt hơn), HNR +0,062
   (CI95 chứa 0). Nhưng **fixed chưa hơn P2** ⇒ theo §7: *chưa đủ điều kiện tiếp tục dài
   hoặc thay production*.
5. **Không kết luận “crop không ảnh hưởng”**: probe 200 G update, 1 seed, 36 câu dev;
   chỉ nói **probe ngắn này chưa phân biệt được**.
6. **D accuracy cao không phải chất lượng audio**: D acc cuối legacy 0,786 / fixed 0,823;
   D loss 1,690 / 1,621. Chỉ là năng lực phân biệt của D, không đo tai.
7. **UTMOSv2 có dao động giữa lần chấm**: cùng file P2 EMA (bit-identical với v1837),
   UTMOS mean qua các lần: **1,4842 · 1,5177 · 1,4794 · 1,4642** (range 0,0535);
   BAK ổn định **3,0341**. Vì vậy ΔUTMOS nhỏ hơn ~0,05 **không đáng tin**; đọc BAK/flat/HNR
   + CI95, không so tuyệt đối giữa các lần chấm.

---

## 2. Kiểm tra trước khi chạy

### 2.1 Crop fixed trên batch đại diện

`artifacts/bench/v1837_crop_audit.json` (v18.37) và `runs/v1838_crop_probe/crop_trace.csv`:

| | Legacy (không lengths) | Fixed (lengths=frame_lengths) |
|---|---:|---:|
| Row trong 500 batch | 3.975 | 3.975 |
| Row có padding | **35 (0,88%)** | **0** |
| Bước có ít nhất 1 row padding | **31/500 (6,2%)** | **0/500** |
| Tổng frame padding | **1.098** | **0** |
| Padding tối đa | **64 frame** | 0 |

Ví dụ batch đích N=2 ở v18.37: row 34 frame bị crop `3..67` (64 frame) ⇒ 33 frame padding.
Cách fixed:
- `width = min(64, min(frame_lengths trong batch))`;
- mỗi row start trong `[0, frame_length - width]`;
- row ngắn hơn 64 không bị kéo giãn/lặp; cả batch dùng width ngắn hơn (ví dụ 34).
- D `MelScaleDiscriminator` dùng Conv1d stride 2 + `avg_pool1d(..., ceil_mode=True)` nên
  nhận width thay đổi; **không** lặp waveform/mel, không stretch.

Tỷ lệ row ngắn:
- manifest đầy đủ: **115/18.226 row <64 frame (0,63%)**;
- sau `SlotDataset(min_syllables=2)` dùng cho probe: **42/17.113 (0,25%)**.
- Vì vậy ảnh hưởng legacy không phải mọi row ngắn, mà là **35 row rơi vào batch có row dài**
  (0,88% crop row; 6,2% số bước).

### 2.2 Cùng module / dữ liệu

- Trainable: **2.823.496** tham số, đúng `decoder.` + `mel.` + `postnet.` (không có
  encoder/duration/pause/variance/embedding).
- Cùng dataset, cùng `batch_indices.json` (500 batch), `batch_order_hash =
  c77f9222d814a985a25448a162193b664b93161108c13b098bc0c4ab550ef505`.
- `g_init_sha = 5418ddb25492d184`; `d_init_sha = 002685c40e46c794` cho hai nhánh GAN.
- Legacy/fixed chỉ khác **một tham số**: `sample_crops(lengths=...)` vs `None`.

---

## 3. Thiết kế probe

| Nhánh | D warm-up | G update | Crop | D |
|---|---:|---:|---|---|
| `rec` | — | **200** | — | không |
| `gan_legacy` | 300 | **200** | cũ (`lengths=None`) | có |
| `gan_fixed` | 300 | **200** | fixed (`lengths=frame_lengths`) | có |

- Cùng P2 EMA, cùng thứ tự 500 batch, cùng seed 42, cùng AdamW G (`lr 2e-5`,
  betas 0.5/0.9, wd 0), cùng EMA 0.999.
- Hai nhánh GAN dùng **cùng D init**, cùng optimizer D (`lr 1e-4`), cùng lịch 300 D-only
  rồi D+G mỗi bước.
- **EMA chỉ cập nhật ở bước G** cho cả 3 nhánh (không cập nhật trong 300 D warm-up).
  Đây là **TỰ QUYẾT** khác `finetune_adv.py` (script cũ cập nhật EMA mọi bước, kể cả D warm-up);
  lý do: so generator sau đúng 200 G update, không để EMA “trôi” 300 bước không update.
- **RNG**: reset `torch.manual_seed(seed + d_warmup + 1)` ở đầu pha G cho cả 3 nhánh để
  dropout của generator (kể cả module đóng băng) không lệch do REC bỏ qua 300 forward.
  Python crop RNG riêng, seed 42; đã lưu state.
- **Mode**: `model.train()`, D `train()`; module đóng băng vẫn ở train mode ⇒ **dropout 0,1
  của chúng vẫn hoạt động**. Snapshot mode nằm trong `summary.json`.
- Không dùng feature matching, không energy/lowband, không đổi loss khác.

Thời gian thực chạy: `rec` 30,0 s · `gan_legacy` 61,0 s · `gan_fixed` 60,3 s (MPS).
Số update thực tế: REC 200 G/0 D; legacy 200 G/500 D; fixed 200 G/500 D.

---

## 4. Log huấn luyện

### 4.1 Mốc 0 / 100 / 200 G update

`runs/v1838_crop_probe/train_*.csv` + `summary.json`:

| Nhánh | g | rec | adv thô | adv×0,3 | grad_ratio thô | grad_ratio hiệu dụng | D loss | D acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rec | 0 | 1,2448 | 0 | 0 | — | — | — | — |
| rec | 100 | 1,6164 | 0 | 0 | — | — | — | — |
| rec | 200 | **1,1499** | 0 | 0 | — | — | — | — |
| legacy | 0 | 1,2448 | 0 | 0 | — | — | — | — |
| legacy | 100 | 1,6485 | 0,4925 | 0,1478 | 0,3444 | 0,1033 | 1,1466 | 0,9115 |
| legacy | 200 | 1,2146 | 0,1221 | 0,0366 | **4,3771** | **1,3131** | 1,6897 | 0,7865 |
| fixed | 0 | 1,2448 | 0 | 0 | — | — | — | — |
| fixed | 100 | 1,6568 | 0,5327 | 0,1598 | 0,3287 | 0,0986 | 1,3558 | 0,9062 |
| fixed | 200 | **1,2139** | 0,1642 | 0,0493 | **4,7376** | **1,4213** | 1,6210 | 0,8229 |

- `grad_ratio` là **thô** (chưa nhân 0,3); hiệu dụng = 0,3 × thô.
- Ở g200, gradient adversarial hiệu dụng **>1** ở cả hai nhánh GAN (1,31 / 1,42) ⇒
  không thể nói adv là “nhỏ”; nhưng đây là **một bước**, không chứng minh chất lượng.
- D acc 0,79–0,82 chỉ là D phân biệt real/fake; **không** dùng làm điểm audio.
- Không có NaN/Inf; mọi loss/grad được kiểm.

### 4.2 Crop trace

`runs/v1838_crop_probe/crop_trace.csv` (7.950 row = 2 nhánh × 500 batch; batch cuối có
thể <8 row). Legacy padding 0,88% row / 6,2% step; fixed 0. Trace có `frame_length`,
`crop_start/end`, `padding_frames`, `padding_ratio` cho từng row.

### 4.3 Checkpoint / tái lập

Mỗi nhánh lưu `*_g100.pt`, `*_g200.pt` (và `last` trùng g200) với:
`model`, `ema`, `opt_g`, `opt_d`, `discriminator`, `rng` (Python global + crop + numpy + torch),
`config`, `source_commit`, `checkpoint_sha256`, `batch_order_hash`, `crop_mode`, `weights`.
`resolved.json`, `batch_indices.json`, `rng_start.pt` cũng được lưu.

Đánh giá dùng wrapper **EMA** mới `*_eval_ema.pt` (model = `ema`; không sửa checkpoint gốc).

---

## 5. Đánh giá cuối trên 36 câu dev

### 5.1 Điều kiện

- 36 câu `v1832_blind` (12 video, 3 bucket), text **có dấu câu**, sentence mode, α=0,
  cùng Vocos pretrained.
- **EMA cho cả 4 bản** (P2 EMA + 3 nhánh); không trộn live.
- Không CER, không tai người, không chọn checkpoint từ nhiều mốc.

### 5.2 Mean

`runs/v1838_crop_probe/eval/result.json`:

| Bản | BAK | UTMOS | flat_V | HNR |
|---|---:|---:|---:|---:|
| P2 EMA | **3,0341** | 1,5177 | **0,1673** | **−3,5255** |
| rec | 3,0428 | 1,4310 | 0,1764 | −3,7523 |
| legacy | 3,0169 | **1,5357** | 0,1687 | −3,6753 |
| fixed | 3,0247 | 1,5136 | 0,1696 | −3,6158 |

### 5.3 Ghép cặp vs P2 EMA

| Bản | ΔBAK median [CI95] | video BAK≥0 | ΔUTMOS median [CI95] | Δflat [CI95] | ΔHNR [CI95] |
|---|---:|---:|---:|---:|---:|
| rec | +0,0048 [−0,0391, +0,0198] | 7/12 | **−0,0864** [−0,1391, −0,046] | **+0,0071** (kém) [+0,0070, +0,0086] | **−0,1570** [−0,3565, −0,105] |
| legacy | −0,0061 [−0,0164, +0,0072] | 5/12 | +0,0150 [−0,0514, +0,0483] | +0,0007 [−0,0008, +0,0014] | **−0,1115** [−0,201, −0,033] |
| fixed | −0,0042 [−0,0269, +0,0092] | 7/12 | −0,0090 [−0,0656, +0,0212] | +0,0019 [−0,0006, +0,0033] | **−0,0280** [−0,061, −0,0045] |

### 5.4 Ghép cặp giữa các nhánh

| So sánh | ΔBAK median [CI95] | video BAK≥0 | ΔUTMOS median [CI95] | Δflat [CI95] | ΔHNR [CI95] |
|---|---:|---:|---:|---:|---:|
| fixed − legacy | +0,0041 [−0,0112, +0,0084] | 7/12 | **−0,0507** [−0,1131, −0,0191] | +0,0011 (kém) [+0,0001, +0,0018] | +0,0360 [0,031, 0,080] |
| fixed − rec | **+0,0201** [+0,0032, +0,0674] | 9/12 | **+0,1064** [+0,0246, +0,135] | **−0,0050** (tốt hơn) [−0,0077, −0,0038] | +0,0620 [−0,022, 0,138] |
| legacy − rec | +0,0286 [−0,0003, +0,0712] | 9/12 | +0,0973 [+0,0165, +0,1776] | −0,0090 (tốt hơn) | +0,0725 [−0,034, 0,103] |

**Đọc đúng**:
- Fixed vs legacy **không có thắng rõ**: BAK CI chứa 0; UTMOS fixed kém; flat fixed kém;
  chỉ HNR fixed hơn. Không đủ để nói crop fix cải thiện chất lượng.
- Fixed vs REC: fixed hơn REC trên BAK/UTMOS/flat, nhưng **fixed vs P2 không hơn**
  (BAK/UTMOS CI chứa 0, HNR âm nhẹ) ⇒ chưa đủ điều kiện tiếp tục dài.
- D accuracy cao không đổi kết luận này.

### 5.5 TF diagnostic (tách riêng, không trộn free-run)

`runs/v1838_crop_probe/eval/tf_diagnostic.json`, cùng 36 câu, GT durations/seg/gaps:

| Bản | mel L1 z (all) | mel L1 z (voiced) |
|---|---:|---:|
| P2 EMA | 1,1647 | 0,7259 |
| rec | **1,1561** | **0,7159** |
| legacy | 1,1630 | 0,7236 |
| fixed | 1,1621 | 0,7232 |

Đây chỉ là quan sát mel TF; **không** dùng để chấm free-run và không thay BAK/UTMOS.

### 5.6 Độ bất định của UTMOSv2

Cùng file P2 EMA (bit-identical với v1837):
- v18.37: 1,4842;
- v18.38 eval: 1,5177;
- hai lần chấm lại: 1,4794 · 1,4642.

⇒ range **0,0535**; BAK giữ 3,0341. Vì vậy **ΔUTMOS < ~0,05 không đáng tin**;
các bảng trên dùng CI95 và BAK/flat/HNR làm neo. Không so UTMOS tuyệt đối giữa các lần chấm.

---

## 6. Cách diễn giải theo §7

1. **Fixed tốt hơn legacy?** Không. BAK không khác; UTMOS fixed kém; flat fixed kém;
   chỉ HNR fixed hơn. ⇒ **không có bằng chứng sửa crop có ích trong probe này**.
2. **Fixed tốt hơn legacy nhưng không hơn REC?** Không đúng chiều: fixed **hơn REC**
   trên BAK/UTMOS/flat, nhưng **chưa hơn P2**.
3. **Fixed tốt hơn REC nhưng chưa hơn P2?** Đúng. ⇒ **chưa đủ điều kiện tiếp tục dài
   hoặc thay production**.
4. **Không có khác biệt rõ sau 200 G update?** Với fixed-vs-legacy: đúng trên BAK/UTMOS;
   probe ngắn chưa phân biệt được. **Không** kết luận “crop không ảnh hưởng”.
5. **Một seed** chỉ là bằng chứng sơ bộ; không phải kết quả ổn định.

---

## 7. Thay đổi / artifact / test

### Source
- `tools/v1838_probe.py` (mới): 3 nhánh, log đầy đủ, checkpoint/RNG/D.
- `tools/v1838_eval.py` (mới): EMA-only eval + BAK/UTMOS/flat/HNR + TF diagnostic.
- Không sửa `src/`, không sửa loss/kiến trúc, không đổi `configs/production_v1831.json`.
- v18.37 artifacts giữ nguyên `artifacts/onnx/v1837/`; không overwrite.

### Artifact (local)
`runs/v1838_crop_probe/`:
- `resolved.json`, `batch_indices.json`, `rng_start.pt`, `summary.json`, `report.json`;
- `train_rec.csv`, `train_gan_legacy.csv`, `train_gan_fixed.csv`, `crop_trace.csv`;
- `rec_g100.pt`, `rec_g200.pt`, `gan_legacy_g100.pt`, `gan_legacy_g200.pt`,
  `gan_fixed_g100.pt`, `gan_fixed_g200.pt`;
- `*_eval_ema.pt`;
- `eval/result.json`, `eval/judges.json`, `eval/tf_diagnostic.json`, `eval/items.json`,
  audio 4×36 WAV.

SHA tool: `probe_script_sha256` / `eval_script_sha256` trong `report.json`.

### Test / môi trường
- **Working tree** tại `f1baef0` + tools v1838 chưa commit + 2 file test user chưa stage.
  Không dùng worktree sạch.
- `pytest -q`: **2.556 passed** (chạy trên working tree; gồm test v18.37).
- `ruff check .`: sạch.
- Không stage/revert `tests/test_custom_reading.py`, `tests/test_eval_stats.py`.
- Không dùng smoke output length của ONNX làm bằng chứng parity chất lượng.

---

## 8. Giới hạn / chưa làm

- **1 seed**, 36 câu dev, 12 video; không phải bộ độc lập.
- **200 G update**; không kéo dài, không thử seed 2, không chọn checkpoint từ g100.
- **UTMOSv2 dao động ~0,05**; ΔUTMOS nhỏ không đáng tin.
- **Không chấm tai**, không CER, không ASR gate.
- **Không chạy H3 ép vuv**.
- **Không thay production**, không chuyển ONNX mới, không sửa checkpoint gốc.
- **Không kết luận** “crop fix không ảnh hưởng” — chỉ nói probe này chưa phân biệt được.

**Việc tiếp tối thiểu (chưa làm)**: nếu muốn kiểm crop fix, chạy seed 2 với cùng 500 batch
và cùng eval; nếu vẫn không khác, dừng nhánh crop. Không mở training dài từ probe này.
