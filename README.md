# ViFastTts

TTS tiếng Việt single-speaker, single-dialect với frontend dùng chung. Xem thư mục `docs`.

## Normalizer theo vùng miền

`north`, `central`, `south` chỉ tác động tới cách đọc tắt có khác biệt thật, ví dụ `nghìn/ngàn`, `linh/lẻ` và lựa chọn `tư/bốn` sau hàng chục. Parser âm tiết không thay đổi theo vùng.

Mọi văn bản được chuẩn hoá vị trí dấu thanh về dạng âm vị học chính: `hòa` thành `hoà`, `hóa` thành `hoá`, `thủy` thành `thuỷ`.
