# Triển khai

## Định dạng

ONNX là định dạng mục tiêu. Frontend có thể chạy ngoài graph.

## Windows

ONNX Runtime CPU, CUDA hoặc TensorRT Execution Provider.

## macOS

ONNX Runtime ARM64 hoặc Core ML export riêng.

## Linux ARM

ONNX Runtime ARM64, NEON, mixed precision hoặc INT8.

## Runtime ONNX không cần PyTorch

```bash
vifasttts export-onnx --config configs/base_24k.yaml --checkpoint runs/base/best.pt \
  --output artifacts/model.onnx --precision fp16
vifasttts export-vocoder-onnx --config configs/base_24k.yaml \
  --checkpoint runs/vocoder/last.pt --output artifacts/vocoder.onnx
vifasttts synthesize-onnx --acoustic artifacts/model.onnx --vocoder artifacts/vocoder.onnx \
  --text "Xin chào" --output out.wav --peak-normalize
```

- `runtime/onnx_engine.py` + `runtime/istft.py`: text → 7 ID → mel (onnxruntime) →
  vocoder phổ → iSTFT numpy → WAV; tự chia câu dài, fade đầu/cuối.
- Metadata trong file ONNX (`sample_rate`, `n_fft`, `hop_length`) đủ để runtime chạy
  mà không cần đọc config.
- `--speed` trên đường ONNX là **xấp xỉ** (co giãn mel nearest-frame); muốn chính xác
  thì dùng đường PyTorch (chạy lại duration predictor).

## HTTP server (nghiên cứu/nội bộ)

```bash
vifasttts serve --config configs/base_24k.yaml --checkpoint runs/base/best.pt \
  --vocoder-checkpoint runs/vocoder/last.pt --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Xin chào", "speed": 1.0, "peak_normalize": true}' -o out.wav
curl http://127.0.0.1:8000/health
```

- Chỉ dùng stdlib (`ThreadingHTTPServer`) — không thêm dependency; một model nạp sẵn,
  lock tuần tự, giới hạn 2000 ký tự / speed 0.25–4.0.
- Production: đặt sau reverse proxy, thêm hàng đợi/timeout, cân nhắc chạy nhiều process.
- Batch không cần server: `vifasttts synthesize-batch --input list.txt --output-dir out/`.
- Streaming (phát dần): Python API `iter_synthesize(...)` / `iter_synthesize_onnx(...)`.

### Vì sao chưa gộp vocoder vào một graph ONNX?

ONNX có op `STFT` (opset 17) nhưng **không có ISTFT** (đề xuất còn mở). Muốn gộp phải tự
cài iSTFT bằng op có sẵn — khả thi nhưng tốn kém: ma trận IDFT + overlap-add dạng
ConvTranspose đường chéo cỡ `n_fft²` (~1M tham số cho n_fft=1024) hoặc scatter phức tạp,
lợi ích chưa tương xứng. Vì vậy pipeline hiện tại xuất **2 file** (`model.onnx` +
`vocoder.onnx`) và ghép waveform bằng iSTFT numpy trong `runtime/istft.py` (đã kiểm khớp
`torch.istft`). Đây là lựa chọn có chủ đích, không phải thiếu sót.

## Quantization

Repo hỗ trợ sẵn hai mức từ CLI:

```bash
vifasttts export-onnx --config configs/base_24k.yaml \
  --checkpoint runs/base/best.pt --output artifacts/model.fp16.onnx --precision fp16
vifasttts export-onnx --config configs/base_24k.yaml \
  --checkpoint runs/base/best.pt --output artifacts/model.int8.onnx --precision int8
```

- **FP16**: chuyển toàn graph, giữ kiểu input/output (`keep_io_types`); cần `onnxruntime`
  bản có `transformers.float16`.
- **INT8**: quantize động weight (QUInt8, `onnxruntime.quantization.quantize_dynamic`) —
  hợp với embedding/linear/duration/energy. Đây là mức khởi điểm, **chưa** phải calibration
  đầy đủ: nên đo lại chất lượng với đủ 6 thanh × coda tắc và giữ F0 predictor/vocoder ở
  độ chính xác cao hơn cho tới khi kiểm thử âm thanh hoàn tất.
- Sau khi lượng tử, luôn chạy lại `onnxruntime.InferenceSession` và so hình dạng/giá trị
  với bản fp32 (mẫu trong `tests/test_neural_smoke.py`).
- Nếu bật `model.use_phonation`, graph có thêm đầu ra `phonation` (3 kênh); giữ nguyên
  thứ tự đầu ra khi tích hợp runtime.
