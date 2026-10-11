# Quy trình huấn luyện

## Trạng thái as-built

`vifasttts train` hiện chạy được end-to-end:

1. Đọc manifest JSONL (`audio_path` + `syllables` + tùy chọn `durations`).
2. `data/features.py` trích **mel log**, **log-energy**, **F0 và VUV** (autocorrelation) từ WAV tại chỗ; đọc WAV bằng `soundfile` nếu có, fallback stdlib `wave`.
3. `data/dataset.py` mã hoá âm tiết thành 7 ID, gom target cấp âm tiết (f0/vuv/energy) theo duration, collate có padding + mask.
4. Train với teacher forcing: decoder mở rộng theo **duration thật** để sinh mel đúng độ dài; duration/f0/vuv/energy/tone dự đoán từ syllable state (chuẩn FastSpeech 2).
5. Mask được áp cho encoder, decoder và mọi loss (không tính vào vùng pad).
6. Lưu `best.pt`, `last.pt` (kèm config, step), `history.csv`; có warmup LR, grad clip, AMP nếu chạy CUDA.

```bash
vifasttts train --config configs/base_24k.yaml \
  --train-manifest data/train.jsonl --val-manifest data/val.jsonl \
  --output runs/base --max-steps 5000
```

## Cảnh báo alignment

- Nếu manifest không có trường `durations`, số frame được **chia đều** cho các âm tiết (`data/dataset.py:allocate_durations`). Đây là placeholder deterministic để pipeline chạy được, **không phải** chất lượng production.
- Muốn học duration đúng: dùng forced alignment (ví dụ MFA với model G2P + acoustic tiếng Việt, hoặc aligner tương đương), ghi `durations` (frame cấp mel, hop 256 @ 24 kHz) vào manifest cho từng âm tiết.
- Vocoder chưa được huấn luyện trong `train`; `synthesize` cảnh báo khi checkpoint không có trọng số vocoder.

## Sáu giai đoạn đề xuất (giữ từ thiết kế gốc)

1. Train iSTFTNet bằng mel thật; train thêm HiFi-GAN baseline để A/B. Hiện repo mới có `ISTFTNetLite` và loss acoustic + mel L1; cần bổ sung discriminator/spectral loss trước khi tuyên bố chất lượng.
2. Train acoustic model với duration, F0, energy thật.
3. Train predictors (đóng băng decoder hoặc giảm LR).
4. Suy luận hoàn toàn bằng predicted features.
5. Joint fine-tuning acoustic + vocoder (JETS-style).
6. Export ONNX → kiểm tra tương đương → FP16/INT8 từng phần (xem `docs/TTS_ARCHITECTURE_RESEARCH.md` mục 9.12).

## Cấu hình

`configs/base_24k.yaml` giữ nguyên các nhóm `mel`, `model`, `training`, `loss_weights`. Các khóa `training.*` được đọc bởi `train.py`; `loss_weights` được đọc bởi `losses.py`.

## Checklist trước khi train thật

```bash
vifasttts validate-manifest --manifest data/train.jsonl --wav-dir data/wavs
pytest
```

- Không còn token `RTX3080`-kiểu fallback trong tập train nếu không có chủ đích.
- Coverage thanh điệu × coda đủ (6 thanh, các coda `-p/-t/-c/-ch/-nh/-ng`).
- Mọi WAV 24 kHz mono (hoặc để `load_wav` tự resample).
