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

Nên quantize embedding, linear layer, duration và energy trước. Giữ F0 predictor và vocoder ở độ chính xác cao hơn cho đến khi kiểm thử âm thanh hoàn tất.
