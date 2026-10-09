# Roadmap Rust

Python được dùng trong giai đoạn nghiên cứu vì dễ thay đổi và có hệ sinh thái huấn luyện tốt.

Sau khi frontend ổn định, có thể viết package Rust bổ sung cho:

- Unicode NFC và text cleaning.
- Lexer và normalizer deterministic.
- Syllable parser.
- Component vocab encoder.
- Manifest validator.
- ONNX Runtime inference.
- WAV streaming và CLI.

## Ranh giới tương thích

Rust phải sử dụng cùng:

- JSON schema của parser.
- Component vocabulary ID.
- Quy tắc round-trip.
- Bộ unit test golden.
- Model config.

## Kết nối Python

- PyO3 cho extension Python.
- C ABI nếu cần tích hợp đa ngôn ngữ.
- CLI JSON Lines để kiểm thử độc lập.

Rust là lớp tối ưu triển khai, không phải một parser phương ngữ riêng.
