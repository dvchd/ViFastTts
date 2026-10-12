# Luồng hoạt động của model — as-built (train → inference → export)

> Tài liệu này mô tả **luồng thật trong code** (`src/vifasttts`), kèm tensor shape,
> công thức loss và điểm khác nhau train/inference. Số liệu lấy từ config
> `configs/base_24k.yaml`. Liên quan: `docs/ARCHITECTURE.md` (sơ đồ ngắn),
> `docs/TRAINING.md` (hướng dẫn chạy), `docs/TTS_ARCHITECTURE_RESEARCH.md` (thiết kế
> tương lai), `docs/PROJECT_STATUS.md` (trạng thái tổng).

## 0. TL;DR — bốn luồng

| Luồng | Entry point | Vào | Ra | Trạng thái |
| --- | --- | --- | --- | --- |
| Chuẩn bị dữ liệu | `vifasttts build-manifest` | CSV `id\|text` + WAV | manifest JSONL (text + syllables + audio) | ✅ as-built |
| Train acoustic | `vifasttts train` | manifest JSONL | `best.pt`, `last.pt`, `history.csv` | ✅ as-built (alignment placeholder) |
| Train vocoder | `vifasttts train-vocoder` | manifest JSONL | checkpoint có khóa `vocoder` | ✅ bản tham chiếu |
| Inference | `vifasttts synthesize` | text + checkpoint | WAV PCM 16-bit 24 kHz | ✅ as-built |
| Export | `vifasttts export-onnx` | checkpoint | ONNX (fp32/fp16/int8) | ✅ as-built |

**Ngắn gọn một câu:** text → parse thành **7 ID/âm tiết** → model dự đoán **mel** +
duration/f0/vuv/energy → **duration predictor** quyết định độ dài từng âm tiết →
mel được **vocoder** đổi thành sóng. Train dùng duration *thật* (teacher forcing);
inference dùng duration *tự dự đoán*. Không sampling, chạy tất định.

## 1. Sơ đồ tổng thể

```text
┌─ OFFLINE ─────────────────────────────────────────────────────────────────────┐
│ metadata.csv ──build-manifest──▶ manifest.jsonl ──cache-features──▶ cache/*.pt │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼ (train)                                  ▼ (inference)
┌─ ACOUSTIC MODEL ────────────────────────────┐  ┌─ FRONTEND ───────────────────┐
│ 7 ID/âm tiết ─▶ Embedding ─▶ Encoder        │  │ text ─▶ normalizer ─▶ parser │
│   │                                         │  │      ─▶ 7 ID/âm tiết         │
│   ├─▶ Duration predictor ─▶ [B,N]           │  └──────────────┬───────────────┘
│   ├─▶ F0/VUV/Energy (+Phonation) predictors │                 │
│   ▼                                         │                 ▼
│ LengthRegulator(durations) ─▶ [B,T,H]       │  ┌─ ACOUSTIC MODEL (eval) ──────┐
│ + frame position + variance                 │  │ forward(durations=None)      │
│   ▼                                         │  │  → durations dự đoán         │
│ Decoder ─▶ Mel head ─▶ mel [B,T,100]        │  │  → mel [1,T,100]             │
└─────────────────────────────────────────────┘  └──────────────┬───────────────┘
        │                                                        │
        ▼ (train-vocoder: GAN)                                   ▼
┌─ VOCODER ──────────────────────────────────┐  ┌─ VOCODER (load weights) ─────┐
│ mel thật + wav thật                        │  │ ISTFTNetLite(mel)            │
│ generator iSTFTNetLite vs MPD đầy đủ + MRD │  │  → waveform                  │
│ losses: mel L1 + FM + adversarial          │  └──────────────┬───────────────┘
└────────────────────────────────────────────┘                 ▼
                                                      write_wav PCM16 24 kHz
```

## 2. Đơn vị đầu vào: 7 ID mỗi âm tiết

`frontend/vocab.py` mã hoá mỗi âm tiết thành 7 số nguyên:

| Vị trí | Trường | Số giá trị | Ví dụ `quốc` |
| --- | --- | --- | --- |
| 0 | onset (âm đầu) | 24 | `K` |
| 1 | medial (âm đệm) | 2 | `W` |
| 2 | nucleus (âm chính) | 15 | `Ô` |
| 3 | coda (âm cuối) | 11 | `C` |
| 4 | tone (thanh) | 6 | `SAC` |
| 5 | boundary | 4 | `WORD` / `PHRASE` / `SENTENCE` |
| 6 | punctuation | 6 | `NONE`/`COMMA`/`PERIOD`/`QUESTION`/`EXCLAMATION`/`OTHER` |

Tổng cộng **68 ID** cho toàn bộ tiếng Việt chuẩn — không UNK, không BPE.
Token ngoài tiếng Việt vẫn được mã hoá (xem §6.3): `FALLBACK_REQUIRED` dùng
`(NONE, NONE, A, NONE, NGANG)` để pipeline không crash.

**`boundary` (vị trí 5) để làm gì?** Đây là **nhãn ranh giới ngữ điệu cuối âm tiết**,
sinh tự động từ dấu câu trong câu gốc:

| Giá trị | Sinh ra khi | Ý nghĩa |
| --- | --- | --- |
| `WORD` | mặc định | hết một từ, chưa hết mệnh đề |
| `PHRASE` | âm tiết đứng trước `,` `;` `:` | hết mệnh đề, còn câu |
| `SENTENCE` | âm tiết đứng trước `.` `!` `?` `…` | hết câu |
| `NONE` | chỉ dùng khi **padding** trong collate | không có ranh giới / không áp dụng |

Model dùng embedding của ID này để **học ngắt nghỉ**: kéo dài âm tiết cuối mệnh đề/câu
(final lengthening), hạ đường F0 cuối câu, và chuẩn bị khoảng lặng — các predictor
duration/f0/vuv/energy nhìn thấy nó nên prosody tự nhiên hơn. Ở khâu tổng hợp, việc
chèn khoảng nghỉ (`frontend/chunking.py: pause_seconds`) dựa trên cùng dấu câu, nên hai
tầng nhất quán. Lưu ý: hiện `word_end` luôn `True` (tokenizer chưa gộp từ đa âm tiết),
nên thực tế chỉ có 3 trạng thái WORD/PHRASE/SENTENCE; `NONE` là giá trị padding
(ô số 0 trong batch) và dành chỗ cho tương lai (ví dụ đơn vị dưới âm tiết).

## 3. Luồng dữ liệu (offline)

```text
metadata.csv (id|text)
   └─▶ data/manifest.py: build_manifest
         · parse_text(text) cho từng dòng → danh sách ParsedSyllable (kèm status)
         · ghi JSONL: {id, audio_path, text, syllables:[...]}
   └─▶ (tùy chọn) vifasttts cache-features
         · data/cache.py: extract_features từng WAV; lưu <sha1(audio, config)>.pt
   └─▶ vifasttts validate-manifest
         · kiểm WAV tồn tại, âm tiết rỗng; in coverage tone/coda/fallback
```

Một mẫu manifest chứa **audio_path** + **danh sách âm tiết** đã parse. Trường
`durations` (số frame mel cho mỗi âm tiết) là **tùy chọn** — đến từ forced alignment
(MFA...). Không có thì dataset tự chia đều (placeholder, xem §9).

## 4. Luồng TRAIN acoustic (`train.py: train_reference`)

> Config được `validate_config` kiểm trước khi chạy (báo rõ `thiếu mel.hop_length`...).
> Lệnh nhanh: `vifasttts --version`; phiên âm IPA: `vifasttts g2p "quốc" --dialect south`.

### 4.1 Dataset & collate (`data/dataset.py`)

`TtsDataset.__getitem__` trả:

| Key | Shape | Nguồn |
| --- | --- | --- |
| `features` | `[N, 7]` | `vocab.encode` từng âm tiết |
| `durations` | `[N]` | manifest `durations` (nếu khớp tổng frame), ngược lại `allocate_durations` |
| `mel` | `[T, 100]` | `extract_features`: log-mel (n_fft 1024, hop 256, 24 kHz); `mel.log_magnitude` → log-magnitude clamp 1e-5, `mel.normalize_path` → chuẩn hoá toàn cục per-bin (`tools/compute_mel_stats.py`) |
| `f0`, `vuv`, `energy` | `[N]` | trung bình các frame được gán cho âm tiết theo `durations` |
| `phonation` (tùy chọn) | `[N, 3]` | HNR/tilt/glottalization, chỉ khi `phonation.enabled` |

`collate_batch` pad: `features [B,Nmax,7]`, `mel [B,Tmax,100]`, kèm
`syllable_lengths [B]`, `mel_lengths [B]` để tạo mask.

Quan hệ độ dài: `T = 1 + len(wav)//hop` (STFT center=True); 1 frame mel ≈ 10,67 ms.

### 4.2 Kiến trúc model (`models/vifasttts.py`, `models/conformer.py`)

```text
features [B,N,7]
 └─ ComponentEmbedding: 7 bảng embedding(96) → concat(672) → Linear(256)  → [B,N,256]
 └─ Encoder ConformerLite ×6 (sinusoidal PE + attention additive-mask + depthwise conv)
      → syllable state x [B,N,256]
 ├─ Duration predictor  ConvPredictor(256→1, 3 lớp) → log_duration [B,N]
 ├─ F0 / VUV / Energy predictors                          → [B,N] mỗi cái
 ├─ (tùy chọn) Phonation predictor                        → [B,N,3]
 ├─ Tone classifier Linear(256→6)                → tone_logits [B,N,6] (chỉ để học thêm)
 └─ LengthRegulator(x, durations)
      · expanded [B,T,256], lengths [B], frame_mask [B,T]
      · variance mở rộng theo frame → nối (f0,vuv,energy[,phonation]) → Linear(3→256)
      · frame position 0..1 trong âm tiết → Linear(1→256)
 └─ Decoder ConformerLite ×4 → Mel head Linear(256→100) → mel_pre [B,T,100]
      └─ (khi `model.use_postnet: true`) Postnet 5 lớp conv 256 kênh residual → mel [B,T,100]
```

Số tham số thật (base_24k):

| Phần | Tham số |
| --- | --- |
| Embedding | 178.816 |
| Encoder (6 lớp) | 8.320.512 |
| Decoder (4 lớp) | 5.547.008 |
| Predictors (duration/f0/vuv/energy) | 2.369.540 |
| Còn lại (tone head, mel head, proj...) | 28.778 |
| **Tổng acoustic** | **16.444.654** |
| Vocoder iSTFTNetLite (hidden 256) | 3.527.426 |

### 4.3 Vòng lặp một bước train

```text
batch = collate(...)                 # [B] mẫu
batch = _to_device(batch)            # + tone_ids = features[...,4]
lr    = base_lr * min(1, step/warmup)
out   = model(features, durations=batch["durations"], input_lengths=syllable_lengths)
        # TEACHER FORCING: dùng duration thật để decoder sinh mel đúng độ dài
loss, parts = acoustic_losses(out, batch, loss_weights)
loss.backward(); clip_grad_norm(1.0); optimizer.step()
```

### 4.4 Loss (`losses.py`, trọng số trong config)

| Loss | Công thức | Mask | Trọng số base |
| --- | --- | --- | --- |
| `mel` | L1 trên `[B,T,100]` | `mel_lengths` | 1.0 |
| `duration` | MSE giữa `log_duration` và `log1p(durations thật)` | cấp âm tiết | 1.0 |
| `f0` | L1 | cấp âm tiết | 1.0 |
| `vuv` | BCEWithLogits | cấp âm tiết | 0.5 |
| `energy` | L1 | cấp âm tiết | 0.5 |
| `tone` | CrossEntropy trên `tone_logits^T` vs tone ID trong input | cấp âm tiết | 0.25 |
| `phonation` (tùy chọn) | L1 trên 3 kênh | cấp âm tiết | theo `loss_weights` |

Ghi chú: **tone loss là bài toán phụ** — tone đã nằm trong input (ID thứ 4), classifier
chỉ giúp encoder học biểu diễn phân biệt thanh; ở inference `tone_logits` không được dùng.

### 4.5 Optimizer, validation, checkpoint

- AdamW, `lr 2e-4`, `weight_decay 1e-4`, warmup tuyến tính, grad clip 1.0.
- Thiết bị: `training.device` = `auto` (mặc định: CUDA → Apple MPS → CPU) | `cpu` |
  `cuda` | `mps`. **MPS** = Metal Performance Shaders — backend GPU của Apple (không phải
  tên máy; đừng lẫn với "CUDA MPS" của NVIDIA). Trên Mac Apple Silicon, acoustic model
  chạy MPS nhanh hơn CPU; vocoder mặc định quay về CPU (đo thực tế nhanh hơn — xem
  `docs/TRAINING_ESTIMATES.md`). **Neural Engine không được PyTorch dùng** — nếu muốn
  khai thác ANE phải chuyển sang Core ML.
- AMP tự bật khi CUDA (`training.mixed_precision`).
- Validation mỗi `validate_every` bước: forward **cùng duration thật**, không backward;
  ghi `history.csv` (`step, train_total, train_mel, val_total`); lưu `last.pt` và
  `best.pt` (val thấp nhất), kèm `config` + `step`.
- `resolved_config.json` ghi lại config thực dùng — tái lập thí nghiệm.

## 5. Luồng TRAIN vocoder (`train.py: train_vocoder_reference`)

```text
VocoderDataset: mel [T,100] + wav [T*hop]  (cùng cache feature)
generator  = build_generator: ISTFTNetLite (mặc định) hoặc ISTFTNetVocos (ConvNeXt, hidden 384 × 8)
discriminator = MPD đầy đủ (2/3/5/7/11) + MRD  (config `discriminator: mpd_mrd`;
               mặc định cũ: MultiPeriodDiscriminatorLite)
mỗi bước:
  fake = G(mel); crop real/fake về min length
  ── bước D ──
  real_out = D(real); fake_out = D(fake.detach())
  d_loss = BCE(real→1) + BCE(fake→0)          → opt_d.step()
  ── bước G ──
  fake_out = D(fake)
  g = adv_weight·adversarial(BCE fake→1)
    + fm_weight·feature_matching(L1 real.detach() vs fake)
    + mel_weight·mel L1 (mel của fake vs mel thật — đã chuẩn hoá cùng config)
    + stft_weight·MR-STFT (spectral convergence + log-mag L1)
                                              → opt_g.step()
```

Checkpoint `last.pt` chứa `{"vocoder": ..., "discriminator": ..., "config", "step"}`.
`history_vocoder.csv` ghi `g_total/g_adv/g_mel/g_stft/d_loss`. Trọng số + hidden + ratio nằm ở
nhóm `vocoder_training` của config; **`synthesize` đọc cùng nhóm này** nên phải khớp.

## 6. Luồng INFERENCE (`infer.py: synthesize`)

### 6.1 Từng bước

```text
text ─▶ parse_text(dialect)                    # normalizer → parser → [ParsedSyllable]
     ─▶ không có âm tiết nào → ValueError
     ─▶ vocabs.encode từng âm tiết             # features [1,N,7]
     ─▶ model(features)                        # KHÔNG truyền durations
          · durations = round(exp(log_d)-1).clamp_min(1)   # tự dự đoán
     ─▶ nếu speed ≠ 1: durations = round(durations/speed).clamp_min(1)
                       rồi forward lại với durations mới
     ─▶ mel = outputs["mel"][0]                # [T,100]
     ─▶ vocoder = ISTFTNetLite(...)  ← hidden/upsample_ratio từ config.vocoder_training
          · nạp trọng số theo thứ tự:
            1) file `--vocoder-checkpoint` (khóa "vocoder" hoặc state thô)
            2) checkpoint acoustic nếu có khóa "vocoder"
            3) không có → cảnh báo, dùng trọng số ngẫu nhiên (chỉ để debug)
     ─▶ waveform ─▶ write_wav PCM 16-bit, 24 kHz (stdlib `wave`)
```

### 6.2 Không sampling, không ngẫu nhiên

- `model.eval()` + `torch.no_grad()`; dropout tắt.
- Duration làm tròn và **mỗi âm tiết ≥ 1 frame**; `speed` chỉ co giãn duration.
- Cùng input → cùng output (bit-level trên cùng máy/phiên bản).

### 6.3 Token OOV/fallback

Token không parse được vẫn đi qua model dưới dạng
`(onset=NONE, medial=NONE, nucleus=A, coda=NONE, tone=NGANG)` — model sẽ "đọc"
một âm tiết đại diện. Muốn kiểm soát phát âm OOV: dùng `resolve_oov=True`
(`frontend/oov.py`) hoặc thêm override trong normalizer/English fallback.
Nội dung chỉ có dấu câu hoặc rỗng → `synthesize` raise `ValueError`.

### 6.4 Runtime ONNX (`runtime/`) — không cần PyTorch

```text
text ─▶ frontend/chunking.py (chia câu, ≤ max_chars) ─▶ từng đoạn:
   OnnxAcousticModel (onnxruntime): 7 ID → mel [1,T,M]
   --speed: co giãn mel xấp xỉ (nearest-frame)
   OnnxVocoder (onnxruntime): mel → spec_real/spec_imag
   runtime/istft.py: iSTFT overlap-add numpy → waveform
ghép đoạn + khoảng nghỉ theo dấu câu (pause_seconds)
runtime/postprocess.py: fade mặc định, --peak-normalize → -3 dBFS
▶ WAV
```

Metadata trong file ONNX (do `export.py` ghi) cung cấp `sample_rate/n_fft/hop_length`
cho runtime; nếu thiếu thì truyền tường minh khi khởi tạo `OnnxVocoder`.

Chọn **execution provider** khi tổng hợp ONNX: `auto|cpu|cuda|tensorrt|coreml`
(`--provider`). Trên Mac đo được CPU nhanh nhất cho model này; `coreml` là lựa chọn
thử nghiệm để dùng GPU/ANE. Chi tiết + số đo: `docs/APPLE_SILICON.md`.

### 6.5 Streaming, batch và server

- **Streaming** (thư viện): `iter_synthesize(...)` và `iter_synthesize_onnx(...)` yield
  `(waveform numpy, sample_rate)` theo từng đoạn — gồm cả khoảng nghỉ giữa câu; dùng
  được cho phát dần hoặc tích hợp server. `Synthesizer` (infer) giữ model đã nạp để
  không nạp lại mỗi câu.
- **Batch**: `synthesize_batch(config, checkpoint, rows, output_dir, ...)` nạp model một
  lần cho cả danh sách; CLI `synthesize-batch --input file --output-dir dir` đọc
  `id<TAB>câu` hoặc mỗi dòng một câu (tự sinh `line_0001`).
- **HTTP server** (`server.py`, chỉ stdlib — không thêm dependency):
  - `GET /health` → `{"status": "ok", "version": ...}`; `GET /info` → sample rate...
  - `POST /synthesize` nhận JSON `{"text", "dialect"?, "speed"?, "overrides"?, "peak_normalize"?}`
    → trả `audio/wav`. Giới hạn 2000 ký tự, speed 0.25–4.0, lock tuần tự.
  - CLI: `vifasttts serve --config ... --checkpoint ... [--vocoder-checkpoint ...] [--port 8000]`.
  - Lưu ý: server nghiên cứu/nội bộ (không auth, một model); production nên đặt sau
    reverse proxy + hàng đợi.

## 7. Luồng EXPORT (`export.py`)

- `torch.onnx.export` (TorchScript exporter, `dynamo=False` khi torch hỗ trợ), opset 18.
- Dynamic axes: `features` là `{0: batch, 1: syllables}`; chiều dài frame động theo
  `durations` nhờ LengthRegulator vector hoá (đã kiểm B=1..3, N=3..12 bằng onnxruntime).
- Output ONNX (tên cố định, thứ tự cố định):

```text
mel [B,T,100], log_duration [B,N], durations [B,N], f0 [B,N],
vuv_logits [B,N], energy [B,N], (phonation [B,N,3] nếu use_phonation),
tone_logits [B,N,6], lengths [B]
```

- `--precision fp32|fp16|int8`: fp16 chuyển toàn graph (giữ io types); int8 quantize
  động weight (QUInt8). Bản chuyển ghi từ file trung gian rồi xoá — `output` luôn hoàn chỉnh.
- **Vocoder ONNX** (`export-vocoder-onnx`): wrapper trả `spec_real/spec_imag`
  `[B,bins,T×upsample]` (tách khỏi `aten::complex` vì ONNX không hỗ trợ op này);
  metadata kèm `sample_rate/n_fft/hop_length/upsample_ratio`. Bước iSTFT ghép bằng
  numpy ở runtime (`runtime/istft.py`, khớp `torch.istft(center=True)`).
- Ghi custom metadata cho cả hai model để runtime không cần đọc config.

## 8. Train vs Inference — khác nhau ở đâu

| Điểm | Train | Inference |
| --- | --- | --- |
| Duration | teacher forcing: duration thật (hoặc placeholder chia đều) | tự dự đoán `round(exp(log_d)-1)`, ≥ 1 |
| Frame mask | từ duration thật | từ duration dự đoán |
| Dropout | bật | tắt (`eval()`) |
| Loss/backward | có | không (`no_grad`) |
| Tone classifier | có loss phụ | không dùng output |
| Vocoder | train riêng (GAN) | nạp trọng số có sẵn |
| OOV | fallback nằm trong manifest | fallback encode tại runtime |
| Tốc độ | phụ thuộc batch/GPU | `--speed` co giãn duration |

## 9. Giới hạn as-built (trung thực)

1. **Alignment**: không có `durations` thật thì chia đều theo trọng số/frame — placeholder.
   Muốn chất lượng cần MFA/aligner thật ghi `durations` vào manifest (§4 khảo sát).
2. **Vocoder tham chiếu + nâng cấp**: MPD-lite (mặc định cũ) và **MPD đầy đủ + MRD + MR-STFT**
   (config `discriminator: mpd_mrd`, `stft_weight > 0`) — xem `docs/REVISION_PLAN_V2.md` §10.
3. **Phonation conditioning** mới ở mức reference (`model.use_phonation`): model dự đoán
   3 kênh; chưa có đánh giá nghe/thính học.
4. **ONNX**: đã có runtime đầy đủ (acoustic + vocoder spectrum + iSTFT numpy ở
   `runtime/`); còn lại là **gộp vocoder vào một graph duy nhất** và **calibration
   INT8 đầy đủ** (hiện int8 động). `--speed` trên đường ONNX là xấp xỉ (co giãn mel).
5. **Fallback OOV** đọc âm tiết đại diện nếu không bật `resolve_oov` / `--overrides`.
6. **Một giọng, một phương ngữ mỗi checkpoint**; không speaker/dialect ID.
7. **Server HTTP** là bản nghiên cứu/nội bộ (stdlib, không auth, một model, lock tuần tự);
   production cần reverse proxy + hàng đợi + giới hạn tải.

## 10. Bản đồ file

| Việc | File |
| --- | --- |
| Text → 7 ID | `frontend/{normalizer,pipeline,syllable,vocab}.py` |
| Trích mel/f0/vuv/energy (+phonation) | `data/features.py`, `data/phonation.py` |
| Cache feature | `data/cache.py` |
| Dataset + collate (+vocoder) | `data/dataset.py` |
| Model acoustic | `models/vifasttts.py`, `models/components.py`, `models/conformer.py` |
| Loss acoustic | `losses.py` |
| Vocoder + discriminator + loss GAN | `models/istftnet.py`, `models/discriminators.py` |
| Vòng train | `train.py` |
| Inference (PyTorch) | `infer.py` (chunk + overrides + postprocess) |
| Chia câu | `frontend/chunking.py` |
| Runtime ONNX + iSTFT + postprocess | `runtime/{onnx_engine,istft,postprocess}.py` |
| Streaming/batch (`Synthesizer`) | `infer.py` |
| HTTP server (stdlib) | `server.py` |
| Export (acoustic + vocoder)/quantization | `export.py` |
| CLI | `cli.py` |

## 11. Lệnh chạy nhanh (mỗi luồng)

```bash
# 0) dữ liệu
vifasttts build-manifest --metadata data/metadata.csv --wav-dir data/wavs --output data/manifest.jsonl
vifasttts validate-manifest --manifest data/manifest.jsonl --wav-dir data/wavs
vifasttts cache-features --manifest data/manifest.jsonl --config configs/base_24k.yaml --cache-dir cache

# 1) train acoustic
vifasttts train --config configs/base_24k.yaml --train-manifest data/train.jsonl \
  --val-manifest data/val.jsonl --output runs/base

# 2) train vocoder
vifasttts train-vocoder --config configs/base_24k.yaml --manifest data/train.jsonl --output runs/vocoder

# 3) inference (PyTorch)
vifasttts synthesize --config configs/base_24k.yaml --checkpoint runs/base/best.pt \
  --vocoder-checkpoint runs/vocoder/last.pt --text "Xin chào" --output out.wav

# 4) export
vifasttts export-onnx --config configs/base_24k.yaml --checkpoint runs/base/best.pt \
  --output model.onnx --precision fp16
vifasttts export-vocoder-onnx --config configs/base_24k.yaml \
  --checkpoint runs/vocoder/last.pt --output vocoder.onnx

# 5) inference chỉ với ONNX Runtime (không cần PyTorch)
vifasttts synthesize-onnx --acoustic model.onnx --vocoder vocoder.onnx \
  --text "Xin chào" --output onnx.wav --peak-normalize

# 6) batch + server
vifasttts synthesize-batch --config configs/base_24k.yaml --checkpoint runs/base/best.pt \
  --input data/cau.txt --output-dir data/out
vifasttts serve --config configs/base_24k.yaml --checkpoint runs/base/best.pt --port 8000
```

## 12. Kiểm chứng bằng test

| Test | Phủ gì |
| --- | --- |
| `tests/test_neural_smoke.py::test_end_to_end_train_synth_export` | train 2 bước → synthesize → export ONNX (fp32/int8/fp16) → onnxruntime chạy, kể cả batch/độ dài động |
| `tests/test_neural_smoke.py::test_model_forward_backward_and_masks` | forward + loss + backward + mask |
| `tests/test_neural_smoke.py::test_model_phonation_branch` | nhánh phonation (output + loss) |
| `tests/test_vocoder.py` | MPD-lite/MPD đầy đủ + MRD + MR-STFT, dataset/collate, train smoke, CLI, synthesize dùng vocoder vừa train |
| `tests/test_cache.py` | cache không trích lại; khoá nhạy config |
| `tests/test_onnx_runtime.py` | export acoustic+vocoder → mel ONNX khớp PyTorch → `synthesize-onnx` + streaming + CLI |
| `tests/test_server.py` | engine (validate payload), WAV bytes, HTTP `/health`, `/info`, `POST /synthesize`, lỗi 400/404 |
| `tests/test_runtime_istft.py` | iSTFT numpy khớp `torch.istft(center=True)` |
| `tests/test_chunking.py` | chia câu/mệnh đề/từ, khoảng nghỉ theo dấu câu |
| `tests/test_config_validation.py` | lỗi config nêu đúng đường dẫn khóa |
| `tests/test_postprocess.py` | fade + chuẩn hoá đỉnh đúng đích dBFS |
| `tests/test_features_batched.py` | mel batch khớp từng mẫu (bắt lỗi broadcast) |
| `tests/test_phonation.py` | HNR/tilt/glottalization đúng dải; kênh đi vào dataset |
| `tests/test_losses*` (qua smoke) | công thức loss/trọng số không NaN, backward chạy |

Chạy tất cả: `pytest` (2264 test). Ma trận parser: `python tools/parser_audit.py`.

## 13. Tài liệu liên quan

- `docs/ARCHITECTURE.md` — sơ đồ ngắn as-built + lý do thiết kế.
- `docs/TRAINING.md` — hướng dẫn train + cảnh báo alignment.
- `docs/DATA_PIPELINE.md` — pipeline chuẩn bị dữ liệu 15 bước.
- `docs/DEPLOYMENT.md` — ONNX runtime, quantization.
- `docs/TTS_ARCHITECTURE_RESEARCH.md` — thiết kế tương lai (aligner, phonation, vocoder production).
- `docs/PROJECT_STATUS.md` — trạng thái tổng + roadmap ✅/⬜.
