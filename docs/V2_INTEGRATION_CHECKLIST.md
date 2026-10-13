# V2 integration checklist — nâng cấp lên `arch = "v2_slot5"` (chỉ-đọc)

> Kiểm kê ngày 13/10/2026 từ code thật (commit `09ff10d`). Không sửa code.
> Quy ước: mỗi dòng mục 1 có `file:line` · hiện tại (1 câu) · v2 cần gì (1 câu).
> Nguồn thiết kế: `docs/ACOUSTIC_V2.md` §1–§14, §10–§12, §21 (`slot_ops.py` đã đóng băng).

## 1. Vị trí phụ thuộc kiến trúc v1 (cần nhánh/khai báo cho v2)

### 1a. `src/vifasttts/models/vifasttts.py` (forward, outputs, LengthRegulator, positions)

- `src/vifasttts/models/vifasttts.py:16` · `LengthRegulator` mở rộng chuỗi cấp âm tiết `[B,N,H]` bằng cumsum+gather. · v2 tái dùng nguyên lớp này cho chuỗi slot `[B,1+5N,H]` (F2: slot duration 0 tự bị bỏ qua, không sửa).
- `src/vifasttts/models/vifasttts.py:48` · `ViFastTtsModel.__init__` chỉ nhận `(config, vocab_sizes)`, không có tham số `arch`. · v2 thêm `arch` (mặc định `"v1_syl"`) + factory đọc `arch` từ checkpoint (§10).
- `src/vifasttts/models/vifasttts.py:56` · `duration` là `ConvPredictor` cấp âm tiết duy nhất. · v2 giữ head này (tầng 1 `d_syl`) + thêm MLP ratio/pause cấp segment trên `concat(h_syl, role_emb, …)` (§1, §3–§4).
- `src/vifasttts/models/vifasttts.py:57` · `f0/vuv/energy` là `ConvPredictor` cấp âm tiết (scalar/âm tiết). · v2 thay bằng MLP variance cấp segment dự đoán `s_k`/`energy`/phonation theo slot O/M/N/C (§6–§7), slot G không cộng variance.
- `src/vifasttts/models/vifasttts.py:60` · Nhánh phonation cứng 3 kênh theo `use_phonation`. · v2 chốt đúng 3 kênh đạt cổng R6 (`hnr`, `jitter`, `glottalization`; `tilt`/`aper` không vào cache) — kiểm comment kênh đã lỗi thời (ghi tilt, thiếu jitter).
- `src/vifasttts/models/vifasttts.py:65` · `variance_proj` nhận 3 (+3) kênh cấp âm tiết. · v2 đổi input thành variance cấp slot (số kênh `s_k` + energy + phonation theo slot, G mask 0) + `variance_proj` mới (warm-start không lấy từ B0, §11).
- `src/vifasttts/models/vifasttts.py:66` · `position_proj` dùng position cấp âm tiết 0..1. · v2 dùng `slot_ops.slot_positions` (`(t+0.5)/n` trong từng slot, slot G = 0) (§1, §21).
- `src/vifasttts/models/vifasttts.py:81` · `_frame_positions` tính position theo span âm tiết (F1: phủ cả gap). · v2 không dùng cho slot (thay bằng `slot_positions`); giữ cho nhánh v1.
- `src/vifasttts/models/vifasttts.py:114` · Chế độ free tự đoán `d_syl` bằng `round(exp(log_d)-1)`, clamp ≥ 1. · v2 thêm sau đó: pause gate (`Where`, `gate > 0.5`), `gap = round(expm1(log_gap))`, scale theo `speed` (§4–§5), rồi `distribute_slots` với cùng ratio.
- `src/vifasttts/models/vifasttts.py:117` · `length_regulator(x, durations)` mở rộng theo `d_syl` cấp âm tiết. · v2 mở rộng theo durations cấp slot `[B,1+5N]` (thêm `G_lead` + slot G = gap, §1) và dựng slots `[B,N,5,H]` (G = `silence_emb` + `Linear(boundary, punct)` của âm tiết trước).
- `src/vifasttts/models/vifasttts.py:118` · `exp_f0/exp_vuv/exp_energy` mở rộng scalar cấp âm tiết ra frame. · v2 thay bằng contour từ `slot_ops.build_f0_contour(s, slot_durations, pos)` (§6, §21).
- `src/vifasttts/models/vifasttts.py:137` · Dict output chốt các key `mel/log_duration/durations/f0/vuv_logits/energy[/mel_pre][/phonation]/tone_logits/lengths`. · v2 giữ nguyên mọi key + ý nghĩa (`durations` vẫn `[B,N]` hữu thanh cấp âm tiết, §10), chỉ thêm `slot_durations` + `pause_gate` (và nội bộ `log_gap`).
- `src/vifasttts/models/vifasttts.py:145` · `tone_logits` từ `Linear` trên `h_syl` cấp âm tiết. · v2 warm-start lấy nguyên head này từ B0 (§11).

### 1b. `src/vifasttts/export.py` (tên/thứ tự output, metadata, dummy input)

- `src/vifasttts/export.py:53` · `_export_fp32` dựng `ViFastTtsModel(config, sizes)` trực tiếp, không đọc `arch` checkpoint. · v2 đọc `arch` từ checkpoint → dựng đúng model + factory (§10).
- `src/vifasttts/export.py:66` · Dummy duy nhất `zeros((1, 8, 7))` cho input `features`. · v2 thêm dummy `speed = float32 [1]` (scalar `[]` kẹt TorchScript, §5) và test export `speed=1.0` byte-identical ngay.
- `src/vifasttts/export.py:67` · `output_names` chốt thứ tự `mel, log_duration, durations, f0, vuv_logits, energy, tone_logits, lengths`. · v2 giữ nguyên tên + thứ tự, chỉ append `slot_durations`, `pause_gate` vào cuối (§10, §14).
- `src/vifasttts/export.py:71` · Nhánh phonation chèn `phonation` sau `energy` (trước `tone_logits`). · v2 không đổi vị trí chèn này; output mới vẫn append cuối để runtime cũ đọc theo tên không vỡ.
- `src/vifasttts/export.py:76` · `input_names=["features"]` duy nhất. · v2 thêm input `speed` (production chỉ `features` + `speed`, §10).
- `src/vifasttts/export.py:77` · `dynamic_axes` chỉ có `features.batch/syllables`. · v2 thêm axis `slots` cho `slot_durations [B, 1+5N]` và `syllables` cho `pause_gate [B,N]` (§1).
- `src/vifasttts/export.py:80` · Metadata chỉ ghi `vifasttts.model/sample_rate/n_mels/use_phonation`. · v2 thêm `arch`, `mel_config_hash`, `f0_stats_hash`, `seg_method`, `seg_method_version` (§10).

### 1c. `src/vifasttts/runtime/onnx_engine.py` (đọc output theo tên, `_stretch_mel`, ghép câu, pause)

- `src/vifasttts/runtime/onnx_engine.py:107` · `OnnxAcousticModel` lấy `input_name` là input đầu tiên (giả định 1 input). · v2 có 2 input (`features`, `speed`) → phải resolve theo tên, không lấy `[0]`.
- `src/vifasttts/runtime/onnx_engine.py:108` · `output_names` đọc động từ session (đã đọc theo tên — tốt). · v2 giữ nguyên cách đọc theo tên; thêm rẽ nhánh theo `arch` trong metadata (§5, §14).
- `src/vifasttts/runtime/onnx_engine.py:125` · `run(features)` chỉ feed `features` int64. · v2 thêm tham số `speed=1.0` (float32 `[1]`) feed vào graph; bỏ `_stretch_mel` khi `arch == v2_slot5` (§5).
- `src/vifasttts/runtime/onnx_engine.py:171` · `_stretch_mel` co giãn mel xấp xỉ nearest-frame (v1: speed 2 forward PyTorch, ONNX chỉ xấp xỉ). · v2 không gọi với graph v2 (speed đã trong graph, gate quyết định ở speed gốc, §5); giữ cho graph v1.
- `src/vifasttts/runtime/onnx_engine.py:238` · `iter_synthesize_onnx` chỉ đọc `outputs["mel"]`. · v2 giữ đúng (không cần đọc `slot_durations`/`pause_gate` ở runtime vì pause đã nướng vào durations trong graph).
- `src/vifasttts/runtime/onnx_engine.py:239` · Nhánh `if speed != 1.0: mel = _stretch_mel(mel, speed)`. · v2 rẽ nhánh: v1 giữ, v2 truyền `speed` vào session (§14: "Runtime rẽ nhánh `_stretch_mel` theo `arch`").
- `src/vifasttts/runtime/onnx_engine.py:243` · Khoảng nghỉ giữa câu ghép ngoài model bằng `pause_seconds(chunk)` (text). · v2 giữ nguyên (§4: nghỉ giữa câu do lớp ghép câu lo; mask gap cuối row chỉ trong model).

### 1d. `src/vifasttts/train.py` (loss, validation, checkpoint, resume/warm-start)

- `src/vifasttts/train.py:40` · `_build_loaders` dùng chung `collate_batch` cho cả train/val. · v2 nối target từ `v2_targets.segment_targets` vào collate (9 tensor) — dataset/collate là điểm nối chính của task 5.
- `src/vifasttts/train.py:66` · `_to_device` chỉ thêm `tone_ids` từ `features[..., 4]`. · v2 không đổi (tone vẫn từ ID thứ 4); các target v2 đã là tensor, chuyển device cùng dict.
- `src/vifasttts/train.py:90` · Validation gọi `model(features, durations=batch["spans"], ...)` (teacher-forcing cấp âm tiết). · v2 cần 3 chế độ eval `tf_syl`/`tf_seg`/`free` (§12); `tf_syl` truyền `d_syl`+gaps, `tf_seg` thêm `seg_durations`, `free` không truyền gì.
- `src/vifasttts/train.py:91` · Validation dùng `acoustic_losses` v1 (mel/duration/f0/vuv/energy/tone). · v2 thêm loss ratio (CE/KL, mask inactive + `d_syl<3`), pause (BCE gate + MSE log_gap khi gate=1), F0 theo `s_k` + mask Addendum 6 (§3–§6, §19.2).
- `src/vifasttts/train.py:237` · Dựng `ViFastTtsModel(config, sizes)` trực tiếp, không factory theo `arch`. · v2 dựng qua factory đọc `arch` (mặc định `v1_syl` khi thiếu, §10).
- `src/vifasttts/train.py:245` · Optimizer `AdamW` mới trên toàn bộ tham số. · v2 warm-start (§11): optimizer mới hoàn toàn, giai đoạn A đóng băng `embedding/encoder/duration` + LR 0.25× cho `decoder/mel/postnet`, grad clip 1.0.
- `src/vifasttts/train.py:249` · Resume load `model` + `optimizer` + `step` thẳng (giả định cùng kiến trúc). · v2 cấm load chéo v1→v2 bằng `strict=False` ngầm; chỉ warm-start có chủ đích lấy subset key (§10–§11) + ghi `warm_start_from = <hash B0>`.
- `src/vifasttts/train.py:305` · Train forward dùng `batch["spans"]` (F1: phủ gap bằng vector âm tiết). · v2 train dùng `seg_durations` + gaps (slot G = gap thật, `G_lead = 0` cho row segment) và **raise** nếu `arch == v2_slot5` mà thiếu `seg_durations` (§3, F3).
- `src/vifasttts/train.py:331` · Dict checkpoint chỉ ghi `model/optimizer/config/step/best_val/mel_hash` (không có `arch`). · v2 thêm `arch="v2_slot5"`, `f0_stats_hash`, `seg_method`, `seg_method_version`, `seg_weights_hash`, `warm_start_from` (B9, §9–§11).
- `src/vifasttts/train.py:270` · AMP chỉ bật cho CUDA (MPS giữ fp32, đo M6 0,52 vs 0,48 s/bước). · v2 giữ nguyên quyết định này khi đo R8/T4.1 (không đổi cùng lúc).

### 1e. `src/vifasttts/data/dataset.py` (collate, target, chỗ nối `seg_durations`)

- `src/vifasttts/data/dataset.py:68` · `allocate_durations` fallback chia đều im lặng khi thiếu forced alignment. · v2 cấm fallback: train v2 thiếu `seg_durations` phải raise (F3, §14), giữ hàm này chỉ cho v1/clip cũ.
- `src/vifasttts/data/dataset.py:139` · `__getitem__` đọc `durations/gaps/lead_frames` nhưng **không đọc `seg_durations`** (grep toàn `src/` không có hit nào ngoài `v2_targets.py`). · v2 nối tại đây: đọc `seg_durations [N,4]` (O,M,N,C) từ manifest `_slot4.jsonl` + gọi `segment_targets(row, features_cắt, f0_stats)` (T2.2).
- `src/vifasttts/data/dataset.py:164` · Pool f0/vuv/energy theo cửa sổ **âm tiết** (mean trên `durations`, `vuv>0.5` nhị phân). · v2 thay bằng pool theo **cửa sổ segment** từ `v2_targets.segment_windows` + `s_k`/mask Addendum 6 (O cần onset hữu thanh, N theo tỷ lệ, M/C ≥ 0,5; `n<2` bỏ).
- `src/vifasttts/data/dataset.py:168` · `frame_phonation` stack cứng `(hnr, tilt, glottalization)`. · v2 đổi sang 3 kênh R6 (`hnr`, `glottalization`, `jitter`; `jitter` tái dùng f0/vuv, không harvest lần hai) — khớp cache v3 + `small_codelung_v2.yaml`.
- `src/vifasttts/data/dataset.py:186` · Dict item chỉ có `features/durations/spans/f0/vuv/energy/mel[/phonation]`. · v2 thêm 9 tensor từ `segment_targets` (`seg_ratio`, `seg_ratio_mask`, `f0_shape`, `f0_weight`, `seg_energy`, `seg_vuv`, `pause_gate`, `pause_log_gap`, `pause_mask`).
- `src/vifasttts/data/dataset.py:200` · `collate_batch` pad `features/durations/spans/f0/vuv/energy/mel[/phonation]` + lengths. · v2 pad thêm 9 tensor segment (`[B,N,4]`, `f0_shape [B,N,4,4]`, pause `[B,N]`) + mask tương ứng.
- `src/vifasttts/data/dataset.py:129` · Cắt feature theo `frame_range` cho row segment (gốc 0 = `frame_range[0]`). · v2 tái dùng quy ước này: `segment_windows(relative=True)` khớp đúng cửa sổ đã cắt (T2.2).

### 1f. `src/vifasttts/cli.py`

- `src/vifasttts/cli.py:168` · Lệnh `train` gọi `train_reference` với manifest/config, không có cờ `arch`. · v2 không cần cờ mới (arch từ config/checkpoint), nhưng cần tài liệu: train v2 bắt buộc manifest `_slot4.jsonl`.
- `src/vifasttts/cli.py:182` · Lệnh `export-onnx` chỉ nhận config+checkpoint+opset+precision. · v2 không thêm cờ (giữ CLI ổn định); khác biệt nằm trong graph (input `speed`, output mới, metadata `arch`).
- `src/vifasttts/cli.py:296` · Lệnh `synthesize` (PyTorch) dùng 2-forward cho speed (`infer.py:113`). · v2 PyTorch có thể dùng speed trong forward như ONNX (tránh làm tròn 2 lần ở chế độ teacher-forced, §5).
- `src/vifasttts/cli.py:321` · Lệnh `synthesize-onnx` nhận `--speed` với help "xấp xỉ bằng co giãn mel". · v2 cập nhật help (v2: speed thật trong graph) + runtime rẽ nhánh theo `arch`.
- `src/vifasttts/cli.py:385` · Lệnh `serve` không có cờ model (đọc từ checkpoint). · v2 không đổi CLI; server dùng `Synthesizer` nên hưởng厂 factory + `iter_synthesize` sau khi model v2 xong.

### 1g. `tests/` (test chốt tên/thứ tự output)

- `tests/test_neural_smoke.py:104` · Smoke test chốt dict output có `mel [B,T,M]` + `durations [B,N]`. · v2 giữ assert này (ý nghĩa `durations` không đổi) + thêm assert `slot_durations`/`pause_gate` shape.
- `tests/test_neural_smoke.py:330` · Test export ONNX chạy session với feed duy nhất `{"features": …}`. · v2 cập nhật feed thêm `speed` + assert thứ tự output cũ giữ nguyên, output mới ở cuối.
- `tests/test_neural_smoke.py:340` · Assert `lengths == durations.sum()`. · v2 mở rộng bất biến `G_lead + Σ(5N) == T` toàn manifest + random (§14).
- `tests/test_neural_smoke.py:411` · Test nhánh phonation chốt shape `[B,N,3]`. · v2 cập nhật 3 kênh R6 (hnr/glottalization/jitter) thay tilt.
- `tests/test_onnx_runtime.py:102` · Test parity Torch==ONNX chốt `mel` + `lengths == durations.sum()`. · v2 thêm parity cho `slot_durations`/`pause_gate` + `speed=1.0` byte-identical 2 lần chạy.
- `tests/test_onnx_runtime.py:93` · Test metadata vocoder chốt hop graph vs hop mel. · v2 thêm test metadata acoustic có `arch/mel_config_hash/f0_stats_hash/seg_method/seg_method_version`.
- `tests/test_slot_ops.py:140` · Test parity Torch==ONNX của `slot_ops` (opset 18, `dynamo=False`, byte-identical). · v2 tái dùng pattern này cho export model (không vòng lặp động/argsort, `Where` cho gate).
- `tests/test_v2_targets.py:1` · 7 test nền T2.2 chốt `segment_windows`/ratio/pause/`s_k`+mask. · v2 (task 5) tiêu thụ trực tiếp output `segment_targets` trong collate — không sửa module này, chỉ nối vào.

### 1h. Tiêu thụ output ngoài 4 file bắt buộc (để không sót)

- `src/vifasttts/losses.py:47` · `acoustic_losses(outputs, batch, weights)` đọc `mel/log_duration/durations/f0/vuv_logits/energy/tone_logits[/phonation][/mel_pre]`. · v2 thêm nhánh loss ratio/pause/`s_k` đọc key mới, giữ mọi key cũ.
- `src/vifasttts/infer.py:101` · `Synthesizer._render` đọc `outputs["durations"]` + `outputs["mel"]`, speed bằng forward lần 2. · v2 rẽ nhánh: v1 giữ 2-forward, v2 truyền speed/durations một lần (teacher-forced không scale, §5).
- `src/vifasttts/infer.py:72` · `Synthesizer.__init__` dựng `ViFastTtsModel` + `load_state_dict` trực tiếp. · v2 dùng factory theo `arch` + `verify_mel_hash` giữ nguyên.
- `src/vifasttts/models/conformer.py:15` · `MAX_POSITIONS = 4096` cho positional encoding. · v2 giữ + thêm assert chia câu trước model (§16: ~43,7 s @ hop 256).
- `src/vifasttts/server.py:56` · Server nạp `Synthesizer` một lần, `synthesize_bytes` validate `speed` 0.25–4.0 (`server.py:72`). · v2 không đổi API; dải speed test trong graph là {0.5, 0.8, 1.0, 1.25, 2.0} (§5).
- `tools/eval_acoustic.py:121` · Eval `tf_syl` đọc `outputs["mel"/"log_duration"/"vuv_logits"/"energy"/"f0"]` + `batch["spans"/"durations"]`. · v2 thêm chế độ `tf_seg`/`free` (§12); `tools/eval_acoustic.py:272` hiện raise `tf_seg` cho v1 — v2 gỡ đúng nhánh này.

## 2. Hợp đồng interface hiện tại (không được phá v1)

- **Output ONNX acoustic (tên + thứ tự)**: `mel, log_duration, durations, f0, vuv_logits, energy, [phonation,] tone_logits, lengths` (`export.py:67`; chèn phonation sau `energy` ở `export.py:71`). · v2: giữ nguyên, append `slot_durations`, `pause_gate` vào cuối (§10).
- **Input ONNX hiện tại**: duy nhất `features` int64 `[B, N, 7]` (7 ID: onset, medial, nucleus, coda, tone, boundary, punctuation — `models/vifasttts.py:40`, dummy `export.py:66`, feed `onnx_engine.py:127`, smoke `test_neural_smoke.py:335`). · v2 thêm `speed` float32 `[1]`.
- **Key metadata đang ghi**: `vifasttts.model, sample_rate, n_mels, use_phonation` (`export.py:80`). · v2 thêm `arch, mel_config_hash, f0_stats_hash, seg_method, seg_method_version` (§10).
- **Key dict output của model**: `mel [B,T,M] · log_duration [B,N] · durations [B,N] (hữu thanh cấp âm tiết, §10) · f0/vuv_logits/energy [B,N] · [mel_pre] · [phonation [B,N,3]] · tone_logits [B,N,6] · lengths [B]` (`models/vifasttts.py:137`). · Nơi tiêu thụ: `losses.py:47` (train/val), `infer.py:101` (mel+durations), `onnx_engine.py:238` (mel), `eval_acoustic.py:121` (mel/log_duration/vuv/energy/f0), smoke/onnx test (mel/lengths/durations).
- **Checkpoint đang ghi**: `model, optimizer, config, step, best_val, mel_hash` (`train.py:331`). · Không có `arch` (mặc định `v1_syl` khi thiếu, §10). · v2 thêm `arch, f0_stats_hash, seg_method(_version), seg_weights_hash, warm_start_from`.
- **Cache key**: `feature_cache_key` = sha1(audio+mtime + block `mel/f0/phonation/sample_rate`) — đổi block = đổi khoá → dùng `configs/small_codelung_v2.yaml`, không sửa config B0 (AGENTS.md §4; R6 §20).

## 3. Điểm nối cho v2 (gợi ý, không sửa code)

- **Nơi thêm `arch`/factory**: `models/vifasttts.py:48` (`__init__` nhận `arch="v1_syl"`) + hàm factory mới (đọc `state["arch"]`, mặc định `v1_syl`); thay 3 chỗ dựng trực tiếp: `train.py:237`, `export.py:62` (`_export_fp32`), `infer.py:72` (`Synthesizer.__init__`).
- **Nơi nhận target mới từ `data/v2_targets.py`**: `data/dataset.py:111` (`__getitem__`: đọc `row["seg_durations"]`, gọi `segment_targets(row, features_đã_cắt, f0_stats)` sau khi cắt `frame_range`) → `data/dataset.py:200` (`collate_batch`: pad 9 tensor `seg_ratio [B,N,4]`, `seg_ratio_mask [B,N,4]`, `f0_shape [B,N,4,4]`, `f0_weight [B,N,4]`, `seg_energy [B,N,4]`, `seg_vuv [B,N,4]`, `pause_gate/pause_log_gap/pause_mask [B,N]`).
- **Nơi thêm input `speed` float32 `[1]`**: `models/vifasttts.py:93` (`forward(..., speed=None)`; `None` = teacher-forced/free mặc định 1.0), `export.py:66` (dummy `ones(1)` float32) + `export.py:76` (`input_names=["features", "speed"]`), `onnx_engine.py:125` (`run(features, speed=1.0)`).
- **Output mới chèn cuối**: `models/vifasttts.py:137` (dict thêm `slot_durations [B,1+5N]`, `pause_gate [B,N]` sau `lengths` về mặt ONNX — dict Python không phụ thuộc thứ tự, nhưng `output_names` ở `export.py:67` phải append đúng cuối) + `export.py:77` (dynamic_axes `slots`, `syllables`).
- **Module model mới** (trong `models/`, tái dùng `slot_ops.py` đóng băng §21, không sửa): `role_emb`, segment expander (O/M/N/C từ `h_syl`, G = `silence_emb` + `Linear(boundary, punct)` âm tiết trước; `G_lead` chung `silence_emb`), MLP ratio/pause/variance cấp segment, `variance_proj`/`position_proj` mới.
- **Loss mới** (`losses.py:47`, thêm nhánh khi batch có key v2): ratio CE/KL (mask inactive + `d_syl<3`, §3), pause BCE (`gap≥3`) + MSE `log1p` khi gate=1 (§4), F0 trên `s_k` với mask Addendum 6 + `n<4` mask `k≥n` (§6, §19.2), mel L1 vẫn tính cả frame slot G (§8).
- **Runtime rẽ nhánh** (`onnx_engine.py:239`, `infer.py:113`): đọc `arch` từ metadata/checkpoint → v1 giữ `_stretch_mel`/2-forward, v2 truyền `speed` vào graph.
- **Warm-start** (`train.py:249`, theo §11): script/hàm riêng load subset B0 (`embedding/encoder/decoder/mel/postnet/tone_classifier/duration`), khởi tạo mới module slot, optimizer mới, 2 giai đoạn (A đóng băng + LR 0.25× + log mel G vs hữu thanh riêng; B mở khóa + warm-up 2k + cosine).

## 4. Top-10 rủi ro khi làm task 5 (rủi ro + cách phòng)

1. **Đảo thứ tự output ONNX** (chèn `phonation`/`slot_durations` sai vị trí) làm vỡ runtime cũ đọc theo index — luôn append output mới vào cuối + test assert thứ tự cũ (`export.py:67`, §14).
2. **Load chéo v1→v2 bằng `strict=False` ngầm** gây shape mismatch im lặng — cấm, chỉ warm-start tường minh subset key + ghi `warm_start_from` (§11).
3. **Fallback `allocate_durations` im lặng** khi thiếu `seg_durations` làm污染 train v2 — raise cứng khi `arch==v2_slot5` thiếu target (F3, §3, §14).
4. **Dùng `f0>0` làm mask hữu thanh** (f0 đã nội suy, 100% vs vuv 71%) — mọi pool/mask F0 dùng `vuv>0.5` (F4) + mask Addendum 6 theo lớp onset (`segments.py`, §19.2).
5. **`speed` scalar `[]` kẹt TorchScript opset 18** — dùng float32 shape `[1]`, test export + byte-identical `speed=1.0` ngay từ đầu (§5, §14).
6. **Scale 2 lần ở teacher-forced** (scale durations truyền từ ngoài rồi scale lại) — `durations` ngoài vào thì không scale (§5); gate quyết định ở speed gốc, tập vị trí pause bất biến trên {0.5,…,2.0}.
7. **Slot duration 0 vẫn gather frame rác** (đầu/cuối chuỗi) — test zero-slot: không frame nào gather vào slot 0 + `G_lead+Σ(5N)==T` toàn manifest (§14).
8. **Trộn ngữ cảnh slot G/inactive bằng Conv trên chuỗi 5N** — predictor cấp segment dùng MLP trên concat, chỉ predictor cấp âm tiết (duration tổng, pause) dùng Conv (§1).
9. **Đụng cache B0 khi bật phonation** (đổi block `phonation` đổi toàn bộ key) — dùng config riêng `small_codelung_v2.yaml`, không sửa config B0 (AGENTS.md §4, §20).
10. **Đánh giá sai mốc do F1** (train phủ gap, infer không gap) — so B0↔v2 bằng `tf_syl` (chính), mask đo từ manifest (bỏ lead/gap), cổng giai đoạn A: mel hữu thanh không tệ hơn B0 quá 20% trên val đầy đủ (§1, §11–§12).
