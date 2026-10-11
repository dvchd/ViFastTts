# Triển khai

## Định dạng

ONNX là định dạng mục tiêu. Frontend có thể chạy ngoài graph.

## Windows

ONNX Runtime CPU, CUDA hoặc TensorRT Execution Provider.

## macOS

ONNX Runtime ARM64 hoặc Core ML export riêng.

## Linux ARM

ONNX Runtime ARM64, NEON, mixed precision hoặc INT8.

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
