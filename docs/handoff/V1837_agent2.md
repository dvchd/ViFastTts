# V18.37 — Agent 2: sửa tính đúng pipeline, kiểm lead, chưa train

> **Trạng thái: DỪNG SAU V18.37, CHỜ ĐỌC.** Không training, không đổi loss/kiến trúc,
> không cài package, không đổi production. Production acoustic vẫn **P2 EMA**.
>
> **Commit gốc của vòng**: `bfde6a7` · **Tệp đặc tả**: yêu cầu A0 → Agent 2 trong chat
> (không có `V1837_agent0.md`) · **Test**: 2.556 passed · **Ruff**: sạch.
>
> Quy ước bằng chứng: **[CODE]** đọc source · **[SỐ]** đọc JSON/log · **[ĐO]** chạy tool
> trong vòng này (không train) · **[CHƯA]** chưa làm.

---

## 1. Kết luận ngắn

1. **ONNX cũ đã được xác định bằng tensor, không bằng tên file**:
   - acoustic `artifacts/onnx/v32_adv_p2_2k.onnx` = **P2 EMA** (180/181 tensor exact;
     lệch lớn nhất 1,57e-05 ở buffer `f0_log_mean`);
   - vocoder `artifacts/onnx/v32_adv_p2_2k_vocoder.onnx` = **E4-C** (63/63 tensor exact),
     **0/63 khớp Vocos pretrained** ⇒ **MISMATCH** với vocoder production.
   - Đã export bản sửa ở **đường dẫn mới** `artifacts/onnx/v1837/` (acoustic P2 EMA +
     Vocos pretrained), có `mel_hash`, `stats_path`, `stats_sha256`, `checkpoint_sha256`.
     **Không ghi đè, không chuyển production.**
2. **Eval text cũ mất dấu câu**: `v1833_render_eval.py` và các tool cùng lứa dựng
   `" ".join(raw)`, không nối `punctuation`. Đã có hàm dùng chung `vifasttts/eval_text.py`;
   raw-join cũ chỉ còn là **legacy có nhãn**.
3. **Trace 12 câu**: legacy **0 gap**, bản phục hồi dấu câu **173 gap** (5/12 câu đổi gap;
   9/12 câu đổi độ dài; số token/âm tiết không đổi). Dấu câu cuối câu **không tạo gap**
   (thiết kế `sentence_gaps` bỏ phần tử cuối) — không phải bug.
4. **Crop D có lỗi thật**: batch đích N=2, row 34 frame bị crop 64 frame ⇒ **33 frame padding**,
   gap mask đánh dấu 33 frame đó là gap, fake có **3.300** giá trị khác 0 trong padding.
   Đã sửa `sample_crops(lengths=...)`: không crop vào padding, row ngắn dùng width =
   min(64, frame_length ngắn nhất). **Chưa chạy probe 200 bước**, chưa kết luận chất lượng.
5. **EMA/live không cứu G1 trên BAK** (bộ 36 câu dev): G1 EMA vẫn thấp hơn P2 EMA
   **−0,0964 BAK** (CI95 [−0,4248, −0,0136]; 4/12 video ≥0), G1 live **−0,3829**
   (1/12 video). UTMOS của G1 EMA +0,0357 nhưng CI95 chứa 0. Đây là kiểm tra confound,
   **không phải xác nhận độc lập**.
6. **Split**: train/val/holdout tách theo **video** (0 giao); nhưng val∩holdout = **9 video**
   (0 clip cha, 0 segment) ⇒ holdout là **unseen-clip trong video đã thấy ở val**, không phải
   unseen-video.
7. **VUV**: trên 36 câu TF, C-mũi có **576 frame**, `vuv` chỉ đánh dấu **26 frame vô thanh**,
   nhưng autocorr WAV thật đánh dấu **124 frame vô thanh**. Target ratio 0,9549; predicted
   prob 0,9606; logit 5,7448. Denominator buzz phụ thuộc định nghĩa; **không ép vuv**.
8. **`grad_ratio` là gradient thô**, chưa nhân `adv_weight=0,3`. P2 @7600: raw 4,608 ⇒
   hiệu dụng ~1,382; @2000: 1,531 ⇒ ~0,459. Loss lớn không tự chứng minh gradient chi phối.

---

## 2. Xác minh production và ONNX

### 2.1 Đường nạp thật

| Đường | Entry point | Acoustic | Vocoder | Ghi chú |
|---|---|---|---|---|
| PyTorch (eval/production) | `Synthesizer(config, checkpoint, vocoder_checkpoint=None)` | `_ckpt/step_02000.pt` (P2 EMA) | `build_generator` tự nạp `vocoder_training.pretrained_path` | `engine.vocoder.pretrained_path` trả đúng file pretrained **[ĐO]** |
| CLI `synthesize` | `vifasttts synthesize --config … --checkpoint …` | P2 EMA | pretrained nếu không truyền `--vocoder-checkpoint` | CLI mặc định chunk mode, `target_lufs=-16` |
| Server | `TtsEngine(config, checkpoint, vocoder_checkpoint=None)` | P2 EMA | pretrained | `server.py` |
| ONNX | `synthesize-onnx --acoustic … --vocoder …` | old ONNX = P2 EMA | old ONNX = **E4-C** | **không có config production ONNX trong repo**; đây là artifact export cũ |

**Kết luận phân loại**:
- **Runtime production (PyTorch)** dùng P2 EMA + Vocos pretrained.
- **ONNX cũ** là **export cũ**, không phải runtime production; nếu ai chạy lệnh ONNX cũ,
  họ đang nghe P2 EMA + **E4-C**, không phải pretrained.
- **ONNX mới** `artifacts/onnx/v1837/` là artifact sửa, **chưa dùng**.

### 2.2 Hash / nguồn / mel

| Đối tượng | Path | SHA256 | Nguồn / trạng thái |
|---|---|---|---|
| P2 EMA | `artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt` | `44dd0da6213f7512810f8f808c3c4b11d627578dd82d40cccaf8d7dcec7764be` | `model` == `ema` của `runs/adv_p2_20k/ckpt_2000.pt`; max diff 0,0 |
| P2 live | `runs/adv_p2_20k/ckpt_2000.pt` | `eb76382a47511160e1811df0863efe30c3daaa2349a266240d8e187d029b817d` | `model`, không phải production |
| G1 | `runs/v1830_g1/ckpt_2000.pt` | `ce685c75a947d1bdd6cf2b3084d11c22a8db9fc149fe28151e8988e2b26e2f90` | live + EMA; không phải production |
| Vocos pretrained | `artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin` | `97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a` | `charactr/vocos-mel-24khz`, zero-shot |
| Mel stats | `data/codelung/mel_stats.json` | `cab53eb6d1b369e08d3adcf349372c27e97b713faa55f6756be0c62f67bcc7bc` | `min_log_energy=-6`, 100 bin |
| Mel hash | config v32 | `e065776b45b74b874c483dfdc68c1f9c5e45918b` | khớp `mel_hash` của P2 |
| ONNX acoustic cũ | `artifacts/onnx/v32_adv_p2_2k.onnx` | `fa5b68caf2202612ba72e23b4ea7b697a4e25fd7f3b0650c1c18a147011d9561` | **P2 EMA**, 180/181 exact |
| ONNX vocoder cũ | `artifacts/onnx/v32_adv_p2_2k_vocoder.onnx` | `941ea50e773c25640f98aa23c010cf4b2f3b4508d1119ec10f2763ec35b26ba2` | **E4-C**, 63/63 exact |
| ONNX acoustic mới | `artifacts/onnx/v1837/p2ema_acoustic.onnx` | `ff6844ec8a28b2c17a40d944d9ebe19476a159b8573c22fdd52c932cbfcdf23f` | P2 EMA, metadata v18.37 |
| ONNX vocoder mới | `artifacts/onnx/v1837/vocos_pretrained_vocoder.onnx` | `ed193b8246ae4078203e2b88fae5bc492b53b9f6ae6fd7652a1278aca02dd3f3` | Vocos pretrained, metadata v18.37 |

Số tham số: acoustic **7.045.035 parameter** (+36 buffer = 7.045.071 state tensor) ·
Vocos **13.531.650** · D mel train-only **715.587**.

### 2.3 Bằng chứng tensor

`artifacts/bench/v1837_provenance.json`:

```text
acoustic old ONNX vs P2 EMA : exact 180/181, max abs 1.57e-05 (f0_log_mean)
acoustic old ONNX vs P2 live: exact   5/181, max abs 8.64e-03
acoustic old ONNX vs G1 live: exact 130/181, max abs 1.60e-02
acoustic old ONNX vs G1 EMA : exact 103/181, max abs 8.02e-03

vocoder old ONNX vs pretrained: exact  0/63, max abs 2.66e-02
vocoder old ONNX vs E4-C       : exact 63/63, max abs 0.0
```

⇒ **Kết luận không dựa vào tên file**: ONNX acoustic là P2 EMA; ONNX vocoder là E4-C.
Bản mới đã được export và kiểm lại: acoustic 180/181 exact với P2 EMA; vocoder 63/63
exact với pretrained. Smoke test ONNX mới: `synthesize_onnx` chạy được, 30.720 mẫu @24 kHz
(`artifacts/bench/v1837_onnx_smoke.wav`).

### 2.4 Metadata export mới

`export.py` nay ghi thêm:

```text
mel_hash, stats_path, stats_sha256, checkpoint, checkpoint_sha256,
export_version=v18.37 (vocoder thêm pretrained_sha256)
```

Test `tests/test_onnx_runtime.py` kiểm `mel_hash`, `checkpoint_sha256`, `export_version`.
**Giới hạn**: metadata mới chỉ có ở artifact v1837; ONNX cũ vẫn thiếu `mel_hash`.

---

## 3. Sửa text đánh giá

### 3.1 Hàm dùng chung

`src/vifasttts/eval_text.py`:

- `build_eval_text(row)` — nối `raw`/`normalized`, gắn `punctuation` vào token trước,
  không thêm khoảng trắng, không lặp dấu (`"bạn." + "." → "bạn."`), giữ dấu nhiều ký tự;
- `legacy_raw_join(row)` — đúng hành vi cũ, **chỉ để đối chứng**;
- `text_record(row)` — `{text, text_legacy, text_mode}` để ghi `items.json`.

Các tool đã dùng hàm chung: `v1833_render_eval.py`, `v1835_screen.py`,
`v1832_tf_free_diag.py`, `v1828_gta_eval.py`. `v1833_render_eval` có
`--text-mode punctuation|legacy`; mặc định punctuation.

### 3.2 Test

`tests/test_eval_text.py` (8 test):
- dấu phẩy/chấm/hỏi; không dấu câu;
- không lặp dấu; dấu nhiều ký tự;
- custom reading `VBA`/`VB` vẫn khớp và đúng thứ tự;
- thiếu token → `ValueError` (không bỏ im lặng);
- `text_record` có cả hai đường;
- `Synthesizer.sentence_gaps`: phẩy 23 frame, chấm/hỏi 52 frame (24 kHz, hop 256).

### 3.3 Lưu text thực

`items.json` của các tool v18.37 và các tool đã sửa ghi:
`text` (thực truyền frontend), `text_legacy`, `text_mode`, `video` (lấy từ `parent`,
không cắt `id.split("_")[0]`).

---

## 4. Kiểm tra nhỏ pause (12 câu)

Chọn **trước khi render**: 4 short / 4 mid / 4 long; mỗi bucket có phẩy, chấm, hỏi,
không dấu; row đầu tiên theo manifest. File: `artifacts/bench/v1837_pause/selection.json`.

Cùng P2 EMA + Vocos pretrained, sentence mode, α=0.

| Đo | Legacy raw-join | Có dấu câu |
|---|---:|---:|
| Tổng gap frame (12 câu) | **0** | **173** |
| Số câu đổi gap | — | 5/12 |
| Số câu đổi độ dài waveform | — | 9/12 |
| Số token/âm tiết | 12/12 giữ nguyên | 12/12 giữ nguyên |

Chi tiết:
- dấu phẩy: gap **23 frame** (~0,245 s);
- chấm/hỏi **bên trong câu**: gap **52 frame** (~0,554 s);
- chấm/hỏi **ở cuối câu**: **0 gap** — `sentence_gaps` bỏ phần tử cuối, đúng thiết kế;
- dù gap = 0, dấu câu vẫn đổi `features` (boundary/punctuation) nên duration/mel có thể đổi;
  ví dụ short `?` dài 17.424 → 13.584 mẫu, long `.` 57.424 → 69.696 mẫu.

Log đầy đủ: `artifacts/bench/v1837_pause/trace.json` (mỗi câu có
`text → parsed_punctuation → gaps → slot_durations → gap_fill_frames → trim → waveform_len`).

### 4.1 CLI chunk và ONNX chunk (3 câu)

`artifacts/bench/v1837_pause/runtime_modes.json` + `runtime_summary.json`:

| Câu | Sentence mode | CLI chunk | ONNX chunk |
|---|---:|---:|---:|
| short phẩy | 23.296 | 39.168 | 39.168 |
| mid phẩy | 40.720 | 52.224 | 52.224 |
| long phẩy | 124.672 | 156.160 | 156.160 |

- CLI chunk và ONNX chunk **trùng độ dài** trên 3 câu này (cùng acoustic/chunk; vocoder khác
  nhưng không đổi độ dài).
- Chênh sentence mode là do **chế độ**: chunk mode cắt tại dấu câu, mỗi mảnh được forward
  riêng + pause runtime; sentence mode đọc cả câu + gap trong graph.
- **Không mặc định các mode phải giống waveform**; chỉ ghi khác biệt.
- ONNX dùng artifact cũ ⇒ vocoder E4-C; đây là **khác biệt provenance**, không phải kết luận
  chất lượng.

---

## 5. EMA/live (36 câu, có giới hạn)

### 5.1 Wrapper

Tạo mới trong `artifacts/checkpoints/v1837/` (không sửa checkpoint gốc):

| Wrapper | Nguồn | Key | Key/shape/buffer |
|---|---|---|---|
| `p2_ema.pt` | `_ckpt/step_02000.pt` | `model` | đủ, khớp P2 |
| `p2_live.pt` | `runs/adv_p2_20k/ckpt_2000.pt` | `model` | đủ, khớp P2 |
| `g1_ema.pt` | `runs/v1830_g1/ckpt_2000.pt` | `ema` | đủ, khớp P2 |
| `g1_live.pt` | `runs/v1830_g1/ckpt_2000.pt` | `model` | đủ, khớp P2 |

`missing_vs_p2=[]`, `extra_vs_p2=[]`, `shape_mismatch_vs_p2=[]` cho cả 4.
P2 EMA wrapper render **bit-identical** với checkpoint gốc trên một câu thử
(33.808 mẫu, `np.array_equal=True`).

### 5.2 Bộ 36 câu

Lấy đúng 36 item của `v1832_blind/items.json` (12 video, 3 bucket), **chọn trước khi chấm**;
text dùng `build_eval_text` (có dấu câu); cùng sentence mode, cùng Vocos pretrained, α=0.
Không dùng GT durations/gaps.

### 5.3 Kết quả

`artifacts/bench/v1837_ema_live/result.json` + `judges.json`:

| Bản | BAK mean | UTMOS mean | SIG | OVRL |
|---|---:|---:|---:|---:|
| P2 EMA | **3,0341** | 1,4842 | 3,1002 | 2,4591 |
| P2 live | 2,8686 | 1,4496 | 3,0398 | 2,3742 |
| G1 EMA | 2,8649 | **1,5318** | 3,0562 | 2,3905 |
| G1 live | 2,5738 | 1,4306 | 2,8209 | 2,1875 |

Ghép cặp theo câu, bootstrap theo video (2.000 lần), so với P2 EMA:

| Ứng viên | ΔBAK median [CI95] | câu BAK tốt hơn | video ΔBAK≥0 | ΔUTMOS median [CI95] |
|---|---:|---:|---:|---:|
| P2 live | **+0,0003** [−0,1361, +0,0719] | 18/36 | 6/12 | −0,0201 [−0,0949, +0,0685] |
| G1 EMA | **−0,0964** [−0,4248, −0,0136] | 12/36 | **4/12** | +0,0357 [−0,1026, +0,1904] |
| G1 live | **−0,3829** [−0,7105, −0,1653] | 5/36 | **1/12** | −0,0832 [−0,1355, +0,0299] |

Đọc đúng:
- **EMA/live là confound thật** (P2 live khác P2 EMA; G1 live khác G1 EMA), nhưng
  **không cứu G1 trên BAK**: G1 EMA vẫn âm, CI95 không chứa 0, chỉ 4/12 video ≥0.
- UTMOS của G1 EMA **không xác định tốt hơn** (CI chứa 0); G1 live UTMOS âm nhẹ.
- P2 live so P2 EMA gần như hoà BAK; UTMOS hơi thấp nhưng CI chứa 0.
- Đây là **36 câu dev**, không phải bộ độc lập; không dùng để tự đổi production.

---

## 6. Crop D (không training)

`artifacts/bench/v1837_crop_audit.json`.

### 6.1 Trước khi sửa — tái hiện lỗi

Batch đích N=2 (1 row ngắn + 7 row dài), `frame_lengths`:
`[34, 130, 160, 176, 173, 172, 131, 144]`.

Legacy (`lengths=None`):
- row 34 frame bị crop `start=3, end=67` (64 frame) ⇒ **33 frame padding**;
- gap mask đánh dấu **33/33** frame padding là gap;
- real padding = 0; fake padding có **3.300** giá trị khác 0;
- 7 row còn lại fully valid.

### 6.2 Sau khi sửa

`sample_crops(..., lengths=frame_lengths)`:
- `width = min(frames, min(lengths))` ⇒ batch đích width = **34**;
- mỗi row start trong `[0, frame_length-width]` ⇒ **0 padding**;
- 4 batch ngẫu nhiên (32 row) + batch đích (8 row) đều **0 padding**;
- training call đã truyền `lengths=moved["frame_lengths"]`.

Test mới `tests/test_crop_sampling.py`:
- row 100/34 frame, crop width 34, row ngắn kết thúc đúng 34;
- không truyền `lengths` vẫn giữ width 64 (hành vi cũ).

**Chưa làm**: probe 200 bước, đo tác động chất lượng. Đây là sửa tính đúng, không phải
kết luận D tốt hơn.

### 6.3 `grad_ratio`

`finetune_adv.py` tính `‖grad(hinge_g)‖ / ‖grad(rec)‖` — **chưa nhân** `adv_weight`/`eff`.
Đã thêm comment tại chỗ. P2:
- @2000: raw **1,5309** ⇒ hiệu dụng ~**0,459** (×0,3);
- @7600: raw **4,6080** ⇒ hiệu dụng ~**1,382**.

⇒ Câu cũ “gradient adversarial lớn hơn 1,5–4,6 lần” phải đọc là **thô**; hiệu dụng thấp hơn.
Không dùng độ lớn loss để suy ra gradient chi phối.

---

## 7. VUV diagnostic (không ép)

`artifacts/bench/v1837_vuv_diag.json`, 36 câu TF, P2 EMA, GT durations/seg/gaps/lead.

| Nhóm C | Frame | vuv GT voiced | vuv GT unvoiced | autocorr unvoiced | target ratio | pred logit | pred prob | BCE seg | frame BCE | buzz % (prob≥0,5 trên autocorr-unvoiced) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mũi | 576 | 550 | **26** | **124** | 0,9549 | 5,7448 | 0,9606 | 0,1529 | 0,1312 | **87,62%** |
| tắc | 110 | 107 | 3 | 28 | 0,9727 | 4,5815 | 0,9721 | 0,0775 | 0,0775 | 94,34% |
| bán NA | 360 | 347 | 13 | 89 | 0,9639 | 5,4478 | 0,9675 | 0,0552 | 0,0988 | 91,30% |

- **Denominator buzz phụ thuộc định nghĩa**: `vuv` chỉ 26 frame vô thanh C-mũi; autocorr
  WAV thật 124 frame. Đây là lý do q12 (denominator autocorr) và target `seg_vuv` (gần 1)
  không thể so trực tiếp.
- Model khớp **target segment** (prob 0,9606 vs ratio 0,9549), nhưng target segment gần như
  “voiced” cho C-mũi; đây là vấn đề **độ phân giải target**, không phải model bỏ tín hiệu.
- **Không ép C-mũi vô thanh; không dùng ±10; không thêm frame-level vuv.**
- n nhỏ (36 câu); `buzz %` ở đây là xác suất segment, **không phải buzz waveform**.

---

## 8. Split: giải quyết mâu thuẫn

`artifacts/bench/v1837_split_stats.json`:

| Split | Segment | Clip cha | Video | Giờ segment |
|---|---:|---:|---:|---:|
| train | 18.226 | 5.807 | 13 | 16,06 |
| val | 3.378 | 1.011 | 12 | 3,03 |
| holdout | 2.611 | 842 | 9 | 2,48 |

Giao:

| Cặp | Video | Clip cha | Segment |
|---|---:|---:|---:|
| train ∩ val | **0** | 0 | 0 |
| train ∩ holdout | **0** | 0 | 0 |
| val ∩ holdout | **9** | **0** | **0** |

⇒ **“val∩holdout=0” đúng ở mức clip/segment, sai ở mức video.** Holdout là **unseen-clip
trong 9 video đã thấy ở val**, không phải unseen-video. Train vẫn tách video hoàn toàn khỏi
val/holdout. Không gọi holdout là “unseen-video” trong báo cáo sau.

Ghi chú phụ: các script cũ dùng `id.split("_")[0]` nên video id bị cắt (`4ITcvEa_Acc` →
`4ITcvEa`, `EM_ObHWvd94` → `EM`, `VeU_85bZlrY` → `VeU`, `z0zSgR_oie4` → `z0zSgR`).
Đã sửa ở `v1833_render_eval.py`, `v1835_screen.py` và thêm `video_id_of_row`; JSON cũ vẫn
giữ nhãn cũ (lịch sử).

---

## 9. Kết luận cũ bị giới hạn bởi gì

Không xóa lịch sử, không tuyên bố mọi kết quả cũ vô hiệu. Các điểm phải đọc kèm:

1. **Punctuation / inference mode** — v18.31, v18.33, v18.35 dùng free-run text `raw join`
   (mất dấu câu), `gaps=0`, sentence mode. Mọi kết luận **pause/duration C→D** bị giới hạn.
   BAK/UTMOS ít bị ảnh hưởng bởi pause hơn nhưng **chưa tách**; không nói “đã chứng minh”.
2. **EMA/live** — P2 được chấm bằng EMA, G1/G2/g1_1k bằng live. Kiểm tra 36 câu cho thấy
   G1 EMA vẫn kém BAK (4/12 video), nên **hướng kết luận BAK của v18.33/v18.35 không bị đảo**;
   nhưng CI/số câu phải ghi rõ là điều kiện mới. UTMOS thì **không xác định**.
3. **Vocoder mismatch** — mọi so sánh ONNX cũ với PyTorch pretrained là **không cùng vocoder**
   (E4-C vs pretrained). Không dùng ONNX cũ để xác nhận chất lượng production.
4. **Video id cũ bị cắt** — nhãn per-video trong JSON cũ có thể sai tên; số nhóm hiện tại
   không bị gộp (không có hai video cùng prefix), nhưng không dùng nhãn cũ để kết luận video.
5. **`grad_ratio` thô** — mọi diễn giải “adv chi phối” phải nhân 0,3 (P2 fixed) trước khi so.
6. **Crop D cũ** — các run P2/G1/G2 đã train với crop có thể chứa padding; không thể sửa
   lịch sử, nhưng **không dùng kết luận D “sạch”** từ các run đó. Chưa đo tác động.

---

## 10. Thay đổi source / test / artifact

### Source
- `src/vifasttts/eval_text.py` (mới) — builder text có dấu câu + legacy.
- `src/vifasttts/data/split.py` — `video_id_of_row`.
- `src/vifasttts/infer.py` — `Synthesizer.sentence_gaps` dùng chung; sentence mode dùng helper.
- `src/vifasttts/export.py` — metadata `mel_hash`/stats/checkpoint; sửa signature vocoder.
- `tools/finetune_adv.py` — `sample_crops(lengths=...)`; training call truyền `frame_lengths`;
  comment `grad_ratio` thô.
- `tools/v1833_render_eval.py`, `v1835_screen.py`, `v1832_tf_free_diag.py`,
  `v1828_gta_eval.py` — dùng builder, ghi text thật + legacy; video id đúng.
- `configs/production_v1831.json` — metadata v18.37, provenance, bỏ CER khỏi gate hiện hành,
  sửa note cũ.
- `docs/EVAL_TOOLS.md`, `docs/handoff/TRANG_THAI.md`, `docs/handoff/A0_V18_GROUP1.md` — cập nhật
  luật/trạng thái, giữ số cũ.

### Tool mới
`v1837_artifact_provenance.py` · `v1837_pause_trace.py` · `v1837_ema_live.py` ·
`v1837_crop_audit.py` · `v1837_vuv_diag.py` · `v1837_split_stats.py`.

### Test
- `tests/test_eval_text.py` (8)
- `tests/test_crop_sampling.py` (2)
- `tests/test_dataset_tools.py` (+2)
- `tests/test_onnx_runtime.py` (+metadata assertions)
- **Full**: `2556 passed` (`pytest -q`), `ruff check .` sạch.

### Artifact (local, không commit)
`artifacts/bench/v1837_provenance.json` · `v1837_pause/*` · `v1837_ema_live/*` ·
`v1837_crop_audit.json` · `v1837_vuv_diag.json` · `v1837_split_stats.json` ·
`artifacts/checkpoints/v1837/*` · `artifacts/onnx/v1837/*` · `v1837_onnx_smoke.wav`.

---

## 11. Chưa làm / việc tiếp

1. **Không chạy H3 ép vuv** (đúng yêu cầu). Chỉ đo; chưa thử can thiệp.
2. **Không chạy probe 200 bước** cho crop; chưa biết tác động chất lượng.
3. **Không chuyển production sang ONNX mới**; không overwrite old ONNX.
4. **Không chấm lại 300 câu** với punctuation; mới có 36 câu EMA/live + 12 câu pause.
5. **Không có bộ độc lập / tai người**; 36 câu là dev, không phải xác nhận.
6. **Không mở rộng G2 / step khác**.
7. **Không sửa** script `v1831` (không có trong `tools/`); chỉ còn JSON lịch sử.
8. **Không sửa** checkpoint gốc; wrapper mới có provenance.

**Đề xuất tối thiểu tiếp theo** (chưa thực thi):
- xác nhận ONNX mới bằng một câu trên runtime đích (đã smoke 1 câu, cần thêm nếu deployment);
- nếu cần kết luận BAK/UTMOS, chấm trên bộ **độc lập** (holdout clip hoặc 30 câu người thật);
- nếu muốn đánh giá tác động crop, probe 200 bước với `lengths` bật, so cùng seed/val;
- chỉ sau đó mới cân nhắc H3 (vuv) hoặc thay đổi kiến trúc.
