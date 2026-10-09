# Kiến trúc ViFastTts

ViFastTts dùng frontend tiếng Việt chung và checkpoint single-speaker, single-dialect.

## Input

Mỗi âm tiết là một hàng gồm onset, medial, nucleus, coda, tone, boundary và punctuation. Các embedding thành phần được nối hoặc cộng để tạo một vector âm tiết.

## Encoder

Conformer nhẹ xử lý chuỗi ở cấp âm tiết. Nó không nhận dialect, speaker hoặc style embedding.

## Segment Expander

Mỗi âm tiết được mở thành các role onset, medial, nucleus và coda. Tone là điều kiện của toàn âm tiết, không phải segment có duration riêng.

## Variance predictors

- Duration deterministic.
- F0 có điều kiện theo tone.
- Voiced/unvoiced.
- Energy.
- Phonation có thể được bổ sung sau khi có dữ liệu và metric phù hợp.

## Decoder

Acoustic decoder sinh mel. iSTFTNet giải mã mel thành waveform 24 kHz. HiFi-GAN nên được giữ làm baseline nghiên cứu.

## Mở rộng phương ngữ

Cùng parser được dùng cho mọi phương ngữ. Mỗi phương ngữ dùng một checkpoint riêng, không cần dialect ID.
