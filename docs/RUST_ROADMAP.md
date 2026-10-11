# Roadmap Rust

Python được dùng trong giai đoạn nghiên cứu vì dễ thay đổi và có hệ sinh thái huấn luyện tốt.

Sau khi frontend ổn định, có thể viết package Rust bổ sung cho:

- Unicode NFC và text cleaning.
- Lexer và normalizer deterministic.
- Syllable parser (gồm chữ hoa/NFD, dấu thanh kép, gạch nối từ vay).
- Compose canonical (`compose_canonical`: đặt dấu trên âm chính, `oo`, `qu`...).
- Profile `open/strict/corpus` + lexicon shape (đọc/ghi JSON).
- Component vocab encoder + kiểm bất biến `parse → compose → parse`.
- Segment Expander + trọng số duration theo role.
- Manifest validator (+ coverage tone/coda/fallback).
- ONNX Runtime inference (input 7 ID → mel; vocoder tách riêng).
- WAV streaming và CLI.

## Ranh giới tương thích

Rust phải sử dụng cùng:

- JSON schema của parser (kể cả `status`, `confidence`, `punctuation`).
- Component vocabulary ID (24/2/15/11/6 + boundary/punctuation).
- Quy tắc round-trip và tone placement (xem `tests/test_spelling_bijection.py`).
- Bảng compatibility (`MEDIAL_NUCLEUS_COMPATIBILITY`, `CODA_TONE_COMPATIBILITY`).
- Bộ unit test golden (đối chiếu từng ca của Python).
- Model config (điểm danh sách output ONNX giống `docs/MODEL_FLOW.md` §7).

## Kết nối Python

- PyO3 cho extension Python.
- C ABI nếu cần tích hợp đa ngôn ngữ.
- CLI JSON Lines để kiểm thử độc lập.

Rust là lớp tối ưu triển khai, không phải một parser phương ngữ riêng.
