# Quy trình huấn luyện

> Luồng chi tiết as-built (tensor shape, loss, train vs inference): `docs/MODEL_FLOW.md`.
> Ước lượng **dữ liệu / tham số / thời gian / máy / chi phí** trước khi bắt tay: `docs/TRAINING_ESTIMATES.md`.

## Trạng thái as-built

`vifasttts train` hiện chạy được end-to-end:

1. Đọc manifest JSONL (`audio_path` + `syllables` + tùy chọn `durations`).
2. `data/features.py` trích **mel log**, **log-energy**, **F0 và VUV** (autocorrelation) từ WAV tại chỗ; đọc WAV bằng `soundfile` nếu có, fallback stdlib `wave`.
3. `data/dataset.py` mã hoá âm tiết thành 7 ID, gom target cấp âm tiết (f0/vuv/energy) theo duration, collate có padding + mask.
4. Train với teacher forcing: decoder mở rộng theo **duration thật** để sinh mel đúng độ dài; duration/f0/vuv/energy/tone dự đoán từ syllable state (chuẩn FastSpeech 2).
5. Mask được áp cho encoder, decoder và mọi loss (không tính vào vùng pad).
6. Lưu `best.pt`, `last.pt` (kèm config, step), `history.csv`; có warmup LR, grad clip, AMP nếu chạy CUDA.
7. (Tùy chọn) Cache feature mel/f0/vuv theo hash(audio, config): đặt `training.cache_features: true` hoặc `cache_dir`, hoặc chạy trước `vifasttts cache-features`. (Tùy chọn) Bật `phonation.enabled: true` để trích thêm HNR/spectral tilt/glottalization cấp frame cho nghiên cứu chất giọng.

```bash
vifasttts train --config configs/base_24k.yaml \
  --train-manifest data/train.jsonl --val-manifest data/val.jsonl \
  --output runs/base --max-steps 5000

vifasttts cache-features --manifest data/train.jsonl \
  --config configs/base_24k.yaml --cache-dir cache/train
```

Cache khoá theo `(đường dẫn audio, size, mtime, mel config)` nên đổi dữ liệu/cấu hình sẽ tự
trích lại. Bật phonation thì nhớ tính lại cache.

## Cảnh báo alignment

- Nếu manifest không có trường `durations`, số frame được **chia đều** cho các âm tiết (`data/dataset.py:allocate_durations`). Đây là placeholder deterministic để pipeline chạy được, **không phải** chất lượng production.
- Muốn học duration đúng: dùng forced alignment (ví dụ MFA với model G2P + acoustic tiếng Việt, hoặc aligner tương đương), ghi `durations` (frame cấp mel, hop 256 @ 24 kHz) vào manifest cho từng âm tiết.
- Vocoder chưa được huấn luyện trong `train`; `synthesize` cảnh báo khi checkpoint không có trọng số vocoder.

## Vocoder (tham chiếu)

`vifasttts train-vocoder` huấn luyện vocoder iSTFTNetLite với discriminator MPD-lite
(mel L1 + feature matching + adversarial, tinh thần HiFi-GAN):

```bash
vifasttts train-vocoder --config configs/base_24k.yaml \
  --manifest data/train.jsonl --output runs/vocoder --max-steps 100000
```

Checkpoint lưu khóa `vocoder` nên `vifasttts synthesize` tự dùng ngay (không còn cảnh báo
"chưa có trọng số vocoder"). Cấu hình ở nhóm `vocoder_training` trong config. Đây là bản
**tham chiếu**: cần discriminator lớn hơn, spectral loss và dữ liệu nhiều hơn cho production.

## Phonation (tùy chọn)

Bật `phonation.enabled: true` để trích HNR/spectral tilt/glottalization cấp frame; bật
`model.use_phonation: true` để model **dự đoán thêm 3 kênh** này (giống f0/vuv/energy) và
điều kiện hoá decoder. Thêm trọng số `phonation` vào `loss_weights` nếu muốn học mạnh hơn.
Cache feature tự khoá theo cấu hình phonation nên không lẫn bản có/không có kênh.

## Sáu giai đoạn đề xuất (giữ từ thiết kế gốc)

1. Train iSTFTNet bằng mel thật; train thêm HiFi-GAN baseline để A/B. `train-vocoder` tham chiếu đã có (MPD-lite + mel L1 + FM + adversarial); cần cấu hình lớn + spectral loss trước khi tuyên bố chất lượng.
2. Train acoustic model với duration, F0, energy thật.
3. Train predictors (đóng băng decoder hoặc giảm LR).
4. Suy luận hoàn toàn bằng predicted features.
5. Joint fine-tuning acoustic + vocoder (JETS-style).
6. Export ONNX → kiểm tra tương đương → FP16/INT8 từng phần. `export-onnx --precision fp16|int8` đã chạy được cho acoustic model (kiểm bằng onnxruntime trong `tests/test_neural_smoke.py`); phần vocoder và calibration INT8 đầy đủ vẫn là việc tiếp theo (xem `docs/TTS_ARCHITECTURE_RESEARCH.md` mục 9.12).

## Cấu hình

`configs/base_24k.yaml` giữ nguyên các nhóm `mel`, `model`, `training`, `loss_weights`. Các khóa `training.*` được đọc bởi `train.py`; `loss_weights` được đọc bởi `losses.py`.

Thiết bị: `training.device: auto` (mặc định) → CUDA nếu có, rồi Apple MPS, cuối cùng CPU;
đặt tường minh `cpu|cuda|mps` khi cần. Trên Mac Apple Silicon, acoustic chạy MPS còn
vocoder tự về CPU (đo thực tế nhanh hơn — `docs/TRAINING_ESTIMATES.md` §3.1). AMP chỉ
bật trên CUDA; `torch.stft` complex không có trên MPS nên mel loss của vocoder được
tính trên CPU tự động.

Chọn biến thể theo lượng dữ liệu (ước lượng: `docs/TRAINING_ESTIMATES.md`):

| Dữ liệu | Config | Tham số acoustic |
| --- | --- | --- |
| < 5 giờ | `configs/small_24k.yaml` | ≈ 6,1M |
| 5–30 giờ | `configs/base_24k.yaml` | ≈ 16,4M |
| ≥ 40–50 giờ | `configs/large_24k.yaml` | ≈ 32,2M |

## Checklist trước khi train thật

```bash
vifasttts validate-manifest --manifest data/train.jsonl --wav-dir data/wavs
pytest
```

- Không còn token `RTX3080`-kiểu fallback trong tập train nếu không có chủ đích.
- Coverage thanh điệu × coda đủ (6 thanh, các coda `-p/-t/-c/-ch/-nh/-ng`).
- Mọi WAV 24 kHz mono (hoặc để `load_wav` tự resample).
