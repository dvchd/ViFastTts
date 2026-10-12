# Run thật: model `small` trên kênh Code Lủng (@codelung)

> Số liệu **as-built** + cách chạy lại + hướng dẫn triển khai. Cập nhật: 12/10/2026.
> Dữ liệu: `docs/YOUTUBE_DATASET.md` · Ước lượng lý thuyết: `docs/TRAINING_ESTIMATES.md`
> · Luồng model: `docs/MODEL_FLOW.md`.

## 1. Dữ liệu

| | |
|---|---|
| Nguồn | 31 video kênh `@codelung` (YouTube, giấy phép dữ liệu CC BY 4.0) |
| Clip | **13.225** · **51,07 giờ** · 24 kHz mono PCM16 |
| Chất lượng text | fallback **1,22%** âm tiết · **71%** clip 0 fallback · đủ 6 thanh × 11 coda |
| Chia tập | `train.jsonl` 11.426 mẫu (44,38h) · `val.jsonl` 1.799 mẫu (6,68h) — **theo video**, không rò rỉ |
| Cache feature | 13.225 mẫu · **6,7 GB** · đo thật: **33 giây** với `--workers 8` |

## 2. Cấu hình & ngân sách (`configs/small_codelung.yaml`)

| Thành phần | Giá trị |
|---|---|
| Acoustic | `small` **6.139.054** tham số · batch 16 · lr 2e-4 · **60.000 bước** |
| Validate | mỗi 1.000 bước, **cap 40 batch** (chống phình bộ nhớ MPS) |
| Vocoder | iSTFTNetLite 3,5M · batch 8 · **30.000 bước** · `crop_frames: 128` (≈1,36s/bước) |
| Thiết bị | acoustic **MPS** · vocoder **CPU** (đo thật: CPU nhanh hơn cho batch nhỏ) |

## 3. Số đo thật trên Mac mini M6 32GB (máy tham chiếu)

| Hạng mục | Đo được | Ghi chú |
|---|---|---|
| Dataloader **không** cache | **6,34 s/batch** (16 clip ≈ 29.000 frame) | Nghẽn STFT + F0 autocorrelation trên clip 14s → **cache là bắt buộc** |
| Cache feature (8 workers) | **33 giây / 13.225 clip (51 giờ audio)** | 6,7 GB, khoá theo hash(audio, config) |
| Model MPS fp32 | **0,478 s/bước** | Đo riêng model, batch đã nằm trên thiết bị |
| Model MPS fp16 (autocast) | 0,520 s/bước | **Không lợi** → AMP chỉ bật cho CUDA |
| Train thật (cache + quản lý bộ nhớ) | **~0,67 s/bước** | → 60k bước ≈ **~12 giờ** |
| Vocoder CPU (`crop_frames=128`) | **0,166 s/bước** | → 30k bước ≈ **~1,4 giờ** |

### Bài học bộ nhớ MPS (quan trọng cho run dài)

- MPS giữ cache MPSGraph/command buffer theo **hình dạng tensor**; batch dài–ngắn khác
  nhau làm "other allocations" leo dần → **40 GB → OOM** (giới hạn 42,43 GB).
- Điểm vỡ điển hình: **validation dài** (113 batch) — cần cap batch + `empty_cache`.
- Cách chữa trong repo: `torch.mps.empty_cache()` mỗi 10 bước + trong/sau validation,
  `validate_batches: 40`, và **`tools/train_chunked.py`** (mỗi khối 5.000 bước là tiến
  trình mới → xả sạch Metal, tự `--resume` và tự thử lại khi lỗi).
- Log in kèm `mps=x.xG` để theo dõi trực tiếp.

## 4. Chạy lại từ số 0 (hoặc chạy tiếp)

```bash
pip install -e ".[train,audio]"

# (1) Dữ liệu — xem docs/YOUTUBE_DATASET.md; giả sử đã có data/codelung/{train,val}.jsonl

# (2) Cache feature (một lần, 8 tiến trình)
vifasttts cache-features --manifest data/codelung/train.jsonl \
  --config configs/small_codelung.yaml --cache-dir cache/codelung --workers 8
vifasttts cache-features --manifest data/codelung/val.jsonl \
  --config configs/small_codelung.yaml --cache-dir cache/codelung --workers 8

# (3) Acoustic — chạy từng khối 5k bước (tự resume, chống OOM MPS)
python tools/train_chunked.py --config configs/small_codelung.yaml \
  --train-manifest data/codelung/train.jsonl --val-manifest data/codelung/val.jsonl \
  --output runs/small_codelung --total-steps 60000 --chunk 5000

# (4) Vocoder (CPU)
vifasttts train-vocoder --config configs/small_codelung.yaml \
  --manifest data/codelung/train.jsonl --output runs/vocoder_codelung

# (5) Export ONNX + demo
vifasttts export-onnx --config configs/small_codelung.yaml \
  --checkpoint runs/small_codelung/best.pt --output artifacts/small_codelung/model.onnx --precision fp32
vifasttts export-onnx --config configs/small_codelung.yaml \
  --checkpoint runs/small_codelung/best.pt --output artifacts/small_codelung/model.fp16.onnx --precision fp16
vifasttts export-vocoder-onnx --config configs/small_codelung.yaml \
  --checkpoint runs/vocoder_codelung/last.pt --output artifacts/small_codelung/vocoder.onnx
```

Hoặc một lệnh duy nhất: **`bash tools/run_small_codelung.sh`** (cache → acoustic → vocoder → ONNX → demo,
có resume, chạy lại an toàn).

## 5. Kết quả run (as-built)

> Mục này được điền bằng số liệu thật khi run hoàn tất; checkpoint nằm ở
> `runs/small_codelung/` (best.pt/last.pt) và `runs/vocoder_codelung/last.pt`.

| | |
|---|---|
| Acoustic | 60.000 bước · loss train/val: cập nhật sau · `best.pt` chọn theo val |
| Vocoder | 30.000 bước · `history_vocoder.csv` |
| ONNX | `artifacts/small_codelung/` (fp32, fp16, vocoder) — kích thước cập nhật sau |
| Demo | `artifacts/small_codelung/demo.wav` (PyTorch) + `demo_onnx.wav` (ONNX Runtime) |
| RTF | cập nhật sau khi đo trên M6 |

## 6. Triển khai (deployment)

```bash
# Chạy không cần PyTorch, chỉ ONNX Runtime
vifasttts synthesize-onnx --acoustic artifacts/small_codelung/model.onnx \
  --vocoder artifacts/small_codelung/vocoder.onnx \
  --text "Xin chào, đây là bản small." --output out.wav --peak-normalize --provider auto

# HTTP server (POST /synthesize)
vifasttts serve --config configs/small_codelung.yaml \
  --checkpoint runs/small_codelung/best.pt \
  --vocoder-checkpoint runs/vocoder_codelung/last.pt --port 8000
```

- `--provider auto` chọn CPU EP trên máy này (đo trước đó: ONNX CPU nhanh hơn CoreML EP
  cho graph nhỏ — xem `docs/APPLE_SILICON.md`).
- fp16 dùng cho triển khai nhẹ; int8 cần calibration (xem `docs/DEPLOYMENT.md`).

## 7. Hạn chế đã biết & việc tiếp theo

1. **Alignment**: manifest chưa có `durations` thật → frame chia theo trọng số (placeholder).
   Chất lượng sẽ tăng rõ khi chạy **MFA** trên `data/codelung/raw/` rồi retrain (resume
   hoặc train lại với durations thật).
2. Vocoder là bản **tham chiếu** (MPD-lite); production cần discriminator lớn + spectral loss.
3. Sau alignment: có thể nâng lên `base` (16,4M) hoặc `large` (32M) — 51 giờ dữ liệu đủ chỗ.
   **Dự đoán chi tiết batch/bộ nhớ/thời gian cho base/large: `docs/TRAINING_ESTIMATES.md` §3.2**
   (base ~30–42h với batch 6–8; large ~52–72h với batch 4).
4. Fallback còn 1,22% — bổ sung `--overrides` cho các token còn lại trong `oov_report.json`.
5. Batch size cho `small`: giữ **16** (đã kiểm chứng); muốn tối ưu ~5–9% thời gian thì A/B
   **20/24** theo hướng dẫn ở `docs/TRAINING_ESTIMATES.md` §3.2 (20 an toàn, 24 sát ngưỡng bộ nhớ).
