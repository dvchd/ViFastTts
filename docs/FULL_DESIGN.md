# ViFastTts Full Design

## Phạm vi

ViFastTts tách frontend tiếng Việt dùng chung khỏi checkpoint âm thanh. Parser không nhận phương ngữ. Normalizer nhận `north`, `central`, `south` chỉ để chọn cách đọc có khác biệt thực tế. Mỗi checkpoint chỉ chứa một người nói và một phương ngữ, nên không có dialect ID, speaker ID hoặc style ID trong neural input.

## Âm tiết

```text
ONSET + MEDIAL + NUCLEUS + CODA + TONE
```

Parser lưu song song chính tả và category trừu tượng. Onset giữ riêng D, GI, R, TR, CH, S, X. Medial gồm NONE/W. Nucleus gồm nguyên âm đơn và IÊ/UÔ/ƯƠ. Coda gồm NONE, M, N, NG, NH, P, T, C, CH, J, W. Tone gồm sáu thanh.

## Chuẩn hoá dấu

Parser chấp nhận cả lối đặt dấu cũ và mới. Normalizer xuất một dạng duy nhất, đặt dấu trên nucleus, ví dụ `hòa` thành `hoà`, `hóa` thành `hoá`, `thủy` thành `thuỷ`.

## Phương ngữ trong normalizer

- Bắc dùng `nghìn`, `linh`, mặc định `tư` sau hàng chục.
- Trung dùng `ngàn`, `lẻ`, mặc định `tư`.
- Nam dùng `ngàn`, `lẻ`, mặc định `bốn`.
- Triệu và tỷ dùng chung.

Các lựa chọn có thể override theo dự án. Parser âm tiết không thay đổi theo profile.

## English fallback

Fallback là từ điển xác định cho các từ công nghệ phổ biến, không phải G2P tiếng Anh tổng quát. Tên sản phẩm, tên người và thuật ngữ chuyên ngành phải được thêm vào override dictionary.

## Neural architecture

- Component embeddings cho từng trường âm tiết.
- Context encoder ở cấp âm tiết.
- Duration, F0, VUV và energy predictors.
- Acoustic decoder sinh mel.
- iSTFT-style vocoder sinh waveform 24 kHz.
- ONNX là định dạng triển khai mục tiêu.

## Rust sau này

Python được giữ cho nghiên cứu và training. Unicode, normalizer, parser, vocabulary mapping, manifest validation, ONNX inference, WAV I/O và CLI có thể viết thêm bằng Rust. Rust phải giữ nguyên schema, vocab ID và golden tests.
