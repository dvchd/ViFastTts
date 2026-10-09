# Thiết kế hệ thống TTS tiếng Việt một giọng, chất lượng cao và tốc độ nhanh

**Phiên bản:** 2.0, kiến trúc đã cải tiến  
**Ngày:** 09/10/2026  
**Phạm vi phiên bản đầu:** một người nói, một checkpoint, tiếng Việt, ưu tiên giọng miền Bắc, 24 kHz, chạy nhanh trên CPU, Apple Silicon, NVIDIA và ARM  
**Nguyên tắc mở rộng:** bộ phân tích âm tiết dùng chung cho mọi phương ngữ; có thể huấn luyện checkpoint miền Nam hoặc miền Trung mà không thay parser hay định dạng dữ liệu

---

## Mục lục

1. [Tóm tắt quyết định cuối cùng](#1-tóm-tắt-quyết-định-cuối-cùng)
2. [Mục tiêu và giới hạn](#2-mục-tiêu-và-giới-hạn)
3. [Các nguyên tắc thiết kế](#3-các-nguyên-tắc-thiết-kế)
4. [Kiến trúc tổng thể](#4-kiến-trúc-tổng-thể)
5. [Ba lớp độc lập trong hệ thống](#5-ba-lớp-độc-lập-trong-hệ-thống)
6. [Frontend chuẩn hóa văn bản](#6-frontend-chuẩn-hóa-văn-bản)
7. [Bộ phân tích âm tiết tiếng Việt dùng chung](#7-bộ-phân-tích-âm-tiết-tiếng-việt-dùng-chung)
8. [Năm thuộc tính âm tiết](#8-năm-thuộc-tính-âm-tiết)
9. [Quy trình phân tích một âm tiết](#9-quy-trình-phân-tích-một-âm-tiết)
10. [Các quy tắc chính tả đặc biệt](#10-các-quy-tắc-chính-tả-đặc-biệt)
11. [Bán nguyên âm cuối và nguyên âm đôi](#11-bán-nguyên-âm-cuối-và-nguyên-âm-đôi)
12. [Validator và round-trip](#12-validator-và-round-trip)
13. [Định dạng dữ liệu đầu ra của parser](#13-định-dạng-dữ-liệu-đầu-ra-của-parser)
14. [Vocabulary thành phần](#14-vocabulary-thành-phần)
15. [Biểu diễn đầu vào neural network](#15-biểu-diễn-đầu-vào-neural-network)
16. [Vì sao không cần dialect_id, speaker_id và style_id](#16-vì-sao-không-cần-dialect_id-speaker_id-và-style_id)
17. [Text Encoder](#17-text-encoder)
18. [Segment Expander và alignment](#18-segment-expander-và-alignment)
19. [Duration Predictor](#19-duration-predictor)
20. [F0, thanh điệu và phonation](#20-f0-thanh-điệu-và-phonation)
21. [Energy Predictor](#21-energy-predictor)
22. [Acoustic Decoder](#22-acoustic-decoder)
23. [iSTFTNet và tạo waveform](#23-istftnet-và-tạo-waveform)
24. [Dữ liệu từ video và SRT](#24-dữ-liệu-từ-video-và-srt)
25. [Thiết kế tập dữ liệu](#25-thiết-kế-tập-dữ-liệu)
26. [Quy trình huấn luyện](#26-quy-trình-huấn-luyện)
27. [Hệ thống loss](#27-hệ-thống-loss)
28. [Đánh giá](#28-đánh-giá)
29. [Ngăn đọc sai, nuốt chữ và lặp chữ](#29-ngăn-đọc-sai-nuốt-chữ-và-lặp-chữ)
30. [ONNX và triển khai đa nền tảng](#30-onnx-và-triển-khai-đa-nền-tảng)
31. [Cấu hình mô hình đề xuất](#31-cấu-hình-mô-hình-đề-xuất)
32. [Cấu trúc mã nguồn](#32-cấu-trúc-mã-nguồn)
33. [Lộ trình triển khai](#33-lộ-trình-triển-khai)
34. [Rủi ro và cách xử lý](#34-rủi-ro-và-cách-xử-lý)
35. [Tiêu chí hoàn thành](#35-tiêu-chí-hoàn-thành)
36. [Phụ lục kiểm thử parser](#36-phụ-lục-kiểm-thử-parser)
37. [Tài liệu tham khảo](#37-tài-liệu-tham-khảo)

---

# 1. Tóm tắt quyết định cuối cùng

Hệ thống cuối cùng gồm hai phần lớn độc lập:

1. **Frontend tiếng Việt dùng chung**, không gắn với giọng Bắc, Trung hay Nam.
2. **Mô hình TTS single-speaker, single-dialect**, mỗi checkpoint chỉ học một người nói và một cách phát âm vùng miền.

Kiến trúc neural được đề xuất:

> **Vietnamese common frontend + structured syllable features + syllable-level Conformer + deterministic segment duration + tone-conditioned F0, phonation và energy + lightweight acoustic decoder + iSTFTNet 24 kHz.**

Luồng chính:

```text
Văn bản thô
    ↓
Text Normalizer tiếng Việt
    ↓
Tách câu và âm tiết
    ↓
Vietnamese Syllable Parser dùng chung
    ↓
Mỗi âm tiết = ONSET + MEDIAL + NUCLEUS + CODA + TONE
    ↓
Syllable Composition Embedding
    ↓
Conformer Context Encoder
    ↓
Segment Expander
    ↓
Duration + F0/Phonation + Energy Predictors
    ↓
Acoustic Decoder
    ↓
Mel-spectrogram
    ↓
iSTFTNet
    ↓
Waveform 24 kHz
```

Không có đầu vào:

```text
dialect_id
speaker_id
style_id
emotion_id
reference_audio
```

Lý do là mỗi checkpoint chỉ chứa một giọng. Nếu sau này cần giọng Nam hoặc Trung, ta dùng cùng parser và cùng định dạng dữ liệu nhưng huấn luyện một checkpoint mới.

---

# 2. Mục tiêu và giới hạn

## 2.1 Mục tiêu

- Đọc đúng nội dung tiếng Việt.
- Giảm tối đa nuốt chữ, lặp chữ, thêm chữ và dừng sớm.
- Thanh điệu đúng và ổn định.
- Một người nói cố định.
- Phong cách trung tính, gần kiểu đọc máy dịch hoặc thuyết minh đều.
- Non-autoregressive và deterministic.
- Chạy nhanh trên CPU.
- Xuất ONNX.
- Có thể chạy trên Windows, macOS, Linux x86, Linux ARM và NVIDIA GPU.
- Frontend dùng lại được cho mọi phương ngữ tiếng Việt.

## 2.2 Ngoài phạm vi đầu tiên

- Voice cloning.
- Đa người nói trong một checkpoint.
- Điều khiển cảm xúc.
- Chuyển giọng bằng audio mẫu.
- Tiếng Anh xen tiếng Việt hoàn chỉnh.
- Hát.
- Mô hình đa phương ngữ trong một checkpoint.
- Streaming từng frame có độ trễ cực thấp.

## 2.3 Chất lượng ưu tiên

Thứ tự ưu tiên:

```text
1. Đọc đúng
2. Không bỏ hoặc lặp âm tiết
3. Thanh điệu đúng
4. Chất giọng ổn định
5. Tự nhiên
6. Biểu cảm
```

---

# 3. Các nguyên tắc thiết kế

## 3.1 Parser không quyết định phương ngữ

Parser chỉ trả cấu trúc tiếng Việt trừu tượng:

```text
TR khác CH
S khác X
D khác GI khác R
HỎI khác NGÃ
NH khác NG
CH khác C
```

Không gộp chỉ vì một giọng Bắc cụ thể phát âm gần nhau.

## 3.2 Checkpoint tự học phương ngữ

Cùng đầu vào trừu tượng:

```text
ONSET_R + NUCLEUS_O + CODA_J + TONE_HUYEN
```

checkpoint Bắc, Nam hoặc Trung sẽ tự học cách hiện thực hóa từ dữ liệu của người nói đó.

## 3.3 Frontend giải những gì có thể giải bằng luật

Neural model không nên phải đoán:

- `105` đọc thế nào.
- `15/10` là ngày hay phân số.
- `kg` là đơn vị gì.
- `TP.HCM` mở rộng ra sao.
- `qu` được phân tích thế nào.
- Dấu nặng khác dấu mũ của `ô` ra sao.

## 3.4 Biểu diễn có cấu trúc, không dùng chuỗi ký tự thô

Mỗi âm tiết là một bản ghi có trường rõ ràng. Điều này giúp:

- Vocabulary nhỏ.
- Dễ kiểm thử.
- Giữ đầy đủ cấu trúc tiếng Việt.
- Dễ mở rộng sang checkpoint vùng miền khác.
- Không cho tone một duration giả.

## 3.5 Training phức tạp, inference đơn giản

Các thành phần như aligner và discriminator chỉ dùng khi train. Runtime chỉ giữ các khối cần thiết để sinh âm thanh.

---

# 4. Kiến trúc tổng thể

## 4.1 Inference

```text
Raw text
  ↓
Unicode normalization
  ↓
Context-aware text normalization
  ↓
Sentence and phrase splitting
  ↓
Syllable parsing
  ↓
Structured syllable tensors
  ↓
Syllable composition embeddings
  ↓
Context encoder
  ↓
Segment expansion
  ↓
Duration, F0, phonation and energy
  ↓
Acoustic decoder
  ↓
Mel
  ↓
iSTFTNet
  ↓
PCM audio
```

## 4.2 Training

```text
Text ──► Frontend ──► Structured syllables ──► Encoder
                                                      │
Audio ──► Mel / F0 / Energy / Alignment targets ──────┤
                                                      ▼
                                        Predictors + Acoustic Decoder
                                                      │
                                                      ▼
                                                Predicted Mel
                                                      │
                                                      ▼
                                                  iSTFTNet
                                                      │
                                                      ▼
                                             Predicted Waveform
```

---

# 5. Ba lớp độc lập trong hệ thống

## 5.1 Lớp A: Chuẩn hóa văn bản

Chuyển ký hiệu sang cách đọc chữ:

```text
105 → một trăm linh năm
3,5% → ba phẩy năm phần trăm
10:30 → mười giờ ba mươi phút
```

Lớp này có thể có cấu hình từ vựng theo mục đích xuất bản, nhưng không gắn với âm vị vùng miền.

## 5.2 Lớp B: Phân tích âm tiết tiếng Việt

Chuyển một âm tiết viết thành cấu trúc:

```text
{
  onset,
  medial,
  nucleus,
  coda,
  tone
}
```

Lớp này dùng chung cho mọi checkpoint.

## 5.3 Lớp C: Mô hình âm thanh

Chuyển cấu trúc âm tiết thành giọng của một người cụ thể.

```text
Common parser output
    ├── North speaker checkpoint
    ├── South speaker checkpoint
    └── Central speaker checkpoint
```

Không cần một dialect mapper cứng ở giữa nếu mỗi checkpoint được train riêng toàn bộ.

---

# 6. Frontend chuẩn hóa văn bản

## 6.1 Unicode

- Chuyển về NFC.
- Loại ký tự điều khiển.
- Chuẩn hóa khoảng trắng.
- Chuẩn hóa dấu nháy, dấu ba chấm và dấu nối.
- Không xóa dấu tiếng Việt.

## 6.2 Lexer

Nhận diện:

```text
WORD
INTEGER
DECIMAL
DATE
TIME
PERCENT
CURRENCY
MEASURE
ABBREVIATION
EMAIL
URL
PHONE
VERSION
IP_ADDRESS
PUNCTUATION
```

## 6.3 Parser theo ngữ cảnh

Ví dụ:

```text
ngày 3/2  → ngày ba tháng hai
phân số 3/2 → ba phần hai
tỷ số 3:2 → ba hai hoặc ba trên hai
phiên bản 3.2 → ba chấm hai
```

## 6.4 Chế độ strict

Nếu biểu thức mơ hồ:

```text
Hẹn vào 3/4
```

strict mode trả cảnh báo thay vì tự đoán.

## 6.5 Từ điển override

```yaml
pronunciations:
  HTMX: hát tê em ích
  ONNX: ô en en ích
  Mac Mini: mác mi ni
```

Override chạy trước syllable parser.

---

# 7. Bộ phân tích âm tiết tiếng Việt dùng chung

## 7.1 Đơn vị đầu vào

Parser xử lý từng âm tiết chính tả, không xử lý cả từ đa âm tiết như một khối.

Ví dụ:

```text
học sinh giỏi
```

có ba âm tiết:

```text
học | sinh | giỏi
```

Thông tin chúng thuộc cùng từ hoặc cụm được lưu bằng boundary metadata.

## 7.2 Cấu trúc lõi

```text
VietnameseSyllable {
    onset
    medial
    nucleus
    coda
    tone
}
```

Các trường bổ sung:

```text
original_spelling
normalized_spelling
word_boundary
phrase_boundary
punctuation
source_span
parser_confidence
```

## 7.3 Chính tả và cấu trúc trừu tượng

Luôn lưu hai lớp:

```text
orthography: cách viết cụ thể
abstract: cấu trúc dùng cho model
```

Ví dụ:

```json
{
  "raw": "nghỉ",
  "orthography": {
    "onset": "ngh",
    "nucleus": "i",
    "tone_mark": "hỏi"
  },
  "abstract": {
    "onset": "NG",
    "medial": "NONE",
    "nucleus": "I",
    "coda": "NONE",
    "tone": "HOI"
  }
}
```

---

# 8. Năm thuộc tính âm tiết

## 8.1 ONSET

Phụ âm đầu. Có thể là `NONE`.

Ví dụ trừu tượng:

```text
NONE, B, M, PH, V, T, TH, Đ, N, D, GI, R,
X, S, CH, TR, NH, L, K, KH, NG, G, H
```

Giữ `D`, `GI`, `R` riêng. Giữ `TR`, `CH` riêng. Giữ `S`, `X` riêng.

## 8.2 MEDIAL

Âm đệm trừu tượng:

```text
NONE
W
```

Ví dụ:

```text
hoa  → medial W
hoàn → medial W
huyền → medial W trong hệ phân tích đã chọn
```

Medial là thuộc tính cấu trúc dùng chung, không phải đặc trưng riêng miền Bắc.

## 8.3 NUCLEUS

Âm chính bắt buộc. Gồm nguyên âm đơn, biến thể độ ngắn nếu hệ phân tích cần, và ba nhóm nguyên âm đôi trừu tượng.

Ví dụ:

```text
A, Ă, Â
E, Ê
I
O, Ô, Ơ
U, Ư
IÊ
UÔ
ƯƠ
```

Có thể thêm feature length/quality thay vì tạo quá nhiều category nếu nghiên cứu cho thấy cần thiết.

## 8.4 CODA

Âm cuối, gồm phụ âm và bán nguyên âm:

```text
NONE
M, N, NG, NH
P, T, C, CH
J, W
```

Trong đó:

```text
J thường được viết i hoặc y ở cuối
W thường được viết u hoặc o ở cuối
```

## 8.5 TONE

Luôn giữ sáu nhãn:

```text
NGANG
HUYEN
SAC
HOI
NGA
NANG
```

Không gộp hỏi và ngã trong parser.

---

# 9. Quy trình phân tích một âm tiết

Thứ tự chuẩn:

```text
1. Unicode NFC
2. Tách dấu câu
3. Trích xuất thanh điệu
4. Giữ nguyên chất lượng nguyên âm
5. Nhận diện onset bằng longest match
6. Xử lý qu và gi
7. Nhận diện coda bằng longest match
8. Phân tích phần nguyên âm thành medial và nucleus
9. Chuẩn hóa biến thể chính tả sang category trừu tượng
10. Validate tổ hợp
11. Round-trip kiểm tra
```

## 9.1 Tách thanh nhưng giữ chất lượng nguyên âm

```text
ọ → o + NANG
ộ → ô + NANG
ợ → ơ + NANG
ắ → ă + SAC
ấ → â + SAC
ế → ê + SAC
ứ → ư + SAC
```

Không được biến tất cả thành chữ Latin không dấu.

## 9.2 Longest match onset

Thử chuỗi dài trước:

```text
ngh, ng, gh, gi, kh, nh, ph, th, tr, ch, qu, ...
```

Không thử `n` trước `ng` hoặc `ngh`.

## 9.3 Longest match coda

Thử:

```text
nh, ng, ch, m, n, p, t, c
```

Sau đó xét bán nguyên âm cuối `i/y/u/o` dựa vào phần nguyên âm hợp lệ.

## 9.4 Phân tích phần còn lại

Tra bảng luật:

```text
oa → medial W + nucleus A
oe → medial W + nucleus E
uê → medial W + nucleus Ê
uyê → medial W + nucleus IÊ
```

Biến thể nucleus:

```text
iê, yê, ia, ya → nucleus IÊ
uô, ua → nucleus UÔ
ươ, ưa → nucleus ƯƠ
```

---

# 10. Các quy tắc chính tả đặc biệt

## 10.1 c, k và q

`c` và `k` có thể cùng ánh xạ đến onset trừu tượng `K`, nhưng giữ grapheme gốc.

```text
ca → onset_grapheme c, onset_abstract K
ki → onset_grapheme k, onset_abstract K
```

`q` thường được xử lý trong `qu`.

## 10.2 g và gh

```text
ga, gô → G

ghe, ghê, ghi → G
```

Giữ spelling nhưng cùng onset trừu tượng.

## 10.3 ng và ngh

```text
nga, ngô → NG
nghe, nghê, nghi → NG
```

## 10.4 qu

Convention thống nhất:

```text
quả
→ onset_grapheme QU
→ onset_abstract K
→ medial W
→ nucleus A
→ coda NONE
→ tone HOI
```

```text
quyển
→ onset_abstract K
→ medial W
→ nucleus IÊ
→ coda N
→ tone HOI
```

Không lúc coi `QU` là onset nguyên khối, lúc lại tách `Q + U` theo cách khác trong representation trừu tượng.

## 10.5 gi

Ưu tiên thử `gi` là onset grapheme.

```text
giá
→ onset GI
→ nucleus A
→ tone SAC
```

```text
giếng
→ onset GI
→ nucleus IÊ
→ coda NG
→ tone SAC
```

Cần test riêng cho:

```text
gi, gì, giá, giữa, giết, giếng, giường
```

## 10.6 i và y

Lưu cả grapheme và category trừu tượng.

Ở nucleus:

```text
i/y có thể cùng ánh xạ category I tùy trường hợp
```

Ở coda:

```text
i/y → J
```

## 10.7 Vị trí dấu thanh

Các dạng Unicode hoặc lối đặt dấu cũ phải chuẩn hóa về cùng phân tích.

Ví dụ:

```text
hòa
hoà
```

đều phải cho:

```text
H + W + A + NONE + HUYEN
```

---

# 11. Bán nguyên âm cuối và nguyên âm đôi

## 11.1 `mai`

```text
mai
→ onset M
→ medial NONE
→ nucleus A
→ coda J
→ tone NGANG
```

`i` ở đây là bán nguyên âm cuối trong hệ phân tích âm vị học.

## 11.2 `sau`

```text
sau
→ onset S
→ medial NONE
→ nucleus A
→ coda W
→ tone NGANG
```

## 11.3 `hoa`

```text
hoa
→ onset H
→ medial W
→ nucleus A
→ coda NONE
→ tone NGANG
```

## 11.4 `ngoài`

```text
ngoài
→ onset NG
→ medial W
→ nucleus A
→ coda J
→ tone HUYEN
```

## 11.5 Nguyên âm đôi trừu tượng

```text
tiến → T + NONE + IÊ + N + SAC
mía  → M + NONE + IÊ + NONE + SAC
muốn → M + NONE + UÔ + N + SAC
mua  → M + NONE + UÔ + NONE + NGANG
tưởng → T + NONE + ƯƠ + NG + HOI
mưa   → M + NONE + ƯƠ + NONE + NGANG
```

---

# 12. Validator và round-trip

## 12.1 Bất biến bắt buộc

```text
1. Có đúng một nucleus
2. Có đúng một tone
3. Các dấu chất lượng nguyên âm không bị mất
4. Stop coda P/T/C/CH chỉ đi với SAC hoặc NANG trong âm tiết chuẩn
5. Tổ hợp medial và nucleus hợp lệ
6. Tổ hợp nucleus và coda hợp lệ
7. Không có phần ký tự dư
8. Round-trip về normalized spelling thành công
9. Không gộp khác biệt chính tả cần bảo toàn
10. Kết quả không phụ thuộc vị trí Unicode combining mark ban đầu
```

## 12.2 Minimal pairs

```text
học → H + NONE + O + C + NANG
hộc → H + NONE + Ô + C + NANG
hột → H + NONE + Ô + T + NANG

khanh → KH + NONE + A + NH + NGANG
khang → KH + NONE + A + NG + NGANG

trăm → TR + NONE + Ă + M + NGANG
chăm → CH + NONE + Ă + M + NGANG

bảo → B + NONE + A + W + HOI
bão → B + NONE + A + W + NGA
```

## 12.3 Parser confidence

Nếu âm tiết không nằm trong bảng hợp lệ:

```text
status = INVALID
```

Nếu thuộc từ ngoại lai:

```text
status = FALLBACK_REQUIRED
```

Không âm thầm gán `<UNKNOWN>` rồi sinh audio.

---

# 13. Định dạng dữ liệu đầu ra của parser

Ví dụ `ngoài`:

```json
{
  "raw": "ngoài",
  "normalized": "ngoài",
  "orthography": {
    "onset": "ng",
    "medial": "o",
    "nucleus": "a",
    "coda": "i",
    "tone_location": "a"
  },
  "abstract": {
    "onset": "NG",
    "medial": "W",
    "nucleus": "A",
    "coda": "J",
    "tone": "HUYEN"
  },
  "boundaries": {
    "word_end": true,
    "phrase_end": false,
    "sentence_end": false
  },
  "status": "VALID"
}
```

Ví dụ `hộc`:

```json
{
  "raw": "hộc",
  "abstract": {
    "onset": "H",
    "medial": "NONE",
    "nucleus": "Ô",
    "coda": "C",
    "tone": "NANG"
  }
}
```

---

# 14. Vocabulary thành phần

Không dùng một vocabulary phẳng bắt buộc. Dùng namespace riêng:

```text
onset_vocab
medial_vocab
nucleus_vocab
coda_vocab
tone_vocab
boundary_vocab
punctuation_vocab
```

Ước lượng:

```text
Onset: khoảng 20 đến 30 category
Medial: 2 category
Nucleus: khoảng 12 đến 20 category
Coda: 11 category
Tone: 6 category
Boundary/Punctuation: khoảng 10 đến 20 category
```

Tổng category nhỏ hơn nhiều so với lưu hàng trăm vần nguyên khối.

---

# 15. Biểu diễn đầu vào neural network

## 15.1 Một vị trí cho mỗi âm tiết

Input tensor:

```text
shape = [batch, syllable_length, feature_count]
```

Mỗi hàng:

```text
[
  onset_id,
  medial_id,
  nucleus_id,
  coda_id,
  tone_id,
  word_boundary_id,
  phrase_boundary_id,
  punctuation_id
]
```

## 15.2 Syllable composition embedding

```text
x =
    onset_embedding(onset_id)
  + medial_embedding(medial_id)
  + nucleus_embedding(nucleus_id)
  + coda_embedding(coda_id)
  + tone_embedding(tone_id)
  + boundary_embedding(boundary_id)
  + punctuation_embedding(punctuation_id)
```

Có thể nối rồi chiếu tuyến tính thay vì cộng:

```text
x = Linear(concat(all_component_embeddings))
```

Nên benchmark cả hai. Concatenation giữ thông tin riêng rõ hơn nhưng tốn tham số hơn.

## 15.3 Feature dropout

Không nên dropout ngẫu nhiên tone, nucleus hoặc coda vì đó là thông tin phát âm bắt buộc.

Có thể dropout nhẹ boundary feature trong training để tăng độ bền, nhưng phải kiểm chứng.

---

# 16. Vì sao không cần dialect_id, speaker_id và style_id

## 16.1 Dialect

Một checkpoint chỉ chứa một phương ngữ. Việc chọn file model đã xác định phương ngữ:

```text
north_voice.onnx
south_voice.onnx
central_voice.onnx
```

`dialect` chỉ là metadata trong config, không phải tensor đầu vào.

## 16.2 Speaker

Một checkpoint chỉ có một người đọc. Chất giọng nằm trực tiếp trong trọng số.

Không cần:

```text
speaker embedding
speaker encoder
speaker reference
speaker ID
```

## 16.3 Style

Nếu dataset có cách đọc tương đối thống nhất, phong cách trung bình được học vào trọng số.

Không cần:

```text
style ID
style encoder
style diffusion
emotion ID
```

## 16.4 Metadata vẫn nên có

```json
{
  "language": "vi",
  "dialect": "north",
  "speaker_count": 1,
  "speaker_name": "voice_01",
  "sample_rate": 24000
}
```

Metadata phục vụ quản lý, không điều kiện hóa neural network.

---

# 17. Text Encoder

## 17.1 Lựa chọn

Conformer nhỏ:

```yaml
layers: 6
hidden_size: 256
attention_heads: 4
ffn_size: 1024
conv_kernel_size: 15
```

## 17.2 Vai trò

- Hiểu ngữ cảnh toàn câu.
- Học ảnh hưởng của dấu câu.
- Học nối âm cục bộ.
- Tạo biểu diễn ngữ cảnh cho duration, F0 và energy.

## 17.3 Không cần BERT đa ngôn ngữ

Một ngôn ngữ, vocabulary thành phần nhỏ và một giọng không cần text model lớn.

---

# 18. Segment Expander và alignment

## 18.1 Hai cấp biểu diễn

Text Encoder làm việc cấp âm tiết. Duration cần chi tiết cấp segment:

```text
ONSET segment
MEDIAL segment
NUCLEUS segment
CODA segment
```

Tone không là một segment có duration. Tone điều kiện hóa toàn âm tiết.

## 18.2 Segment mask

Ví dụ `hoa`:

```text
ONSET H: active
MEDIAL W: active
NUCLEUS A: active
CODA: inactive
```

Ví dụ `ăn`:

```text
ONSET: inactive
MEDIAL: inactive
NUCLEUS Ă: active
CODA N: active
```

## 18.3 Alignment

Giai đoạn đầu dùng forced alignment bên ngoài để tạo target gần đúng. Sau baseline ổn định, có thể thêm alignment module kiểu monotonic để joint train.

## 18.4 Duration target

Duration phân bổ cho segment phát âm:

```text
onset_duration
medial_duration
nucleus_duration
coda_duration
pause_duration
```

---

# 19. Duration Predictor

## 19.1 Deterministic

Không dùng stochastic duration trong production mặc định.

```text
cùng input + cùng speed → cùng duration
```

## 19.2 Điều kiện đầu vào

- Contextual syllable state.
- Segment role.
- Onset/medial/nucleus/coda.
- Tone.
- Boundary và punctuation.

Tone phải ảnh hưởng duration vì thanh điệu không chỉ là F0.

## 19.3 Ràng buộc

```text
active segment duration >= 1 frame
inactive segment duration = 0
pause duration nằm trong khoảng hợp lệ
tổng duration theo số âm tiết nằm trong biên thống kê
```

## 19.4 Tốc độ đọc

```text
adjusted_duration = predicted_duration / speed
```

Giới hạn ban đầu:

```text
0.75 <= speed <= 1.35
```

---

# 20. F0, thanh điệu và phonation

## 20.1 Tone không chỉ là pitch

Thanh điệu có thể ảnh hưởng:

- F0 contour.
- Duration.
- Energy.
- Glottalization hoặc phonation.
- Kiểu kết thúc âm tiết.

## 20.2 Kiến trúc

```text
F0 = tone-conditioned contour + sentence intonation residual
```

Predictor nhận:

```text
context state
tone
segment role
duration
punctuation
phrase position
```

## 20.3 Đầu ra

```text
log-F0 theo frame
voiced/unvoiced
phonation features hoặc acoustic conditioning bổ sung
```

## 20.4 Loss phụ

Tone classifier từ acoustic hidden state dự đoán lại sáu thanh. Loss này giúp giữ thanh nhưng không chạy khi inference.

---

# 21. Energy Predictor

Energy predictor nhỏ, deterministic.

Vai trò:

- Giữ phụ âm rõ.
- Nhấn câu vừa phải.
- Tránh giọng hoàn toàn phẳng.
- Hỗ trợ khác biệt do thanh và coda.

Đầu vào chứa tone và boundary.

---

# 22. Acoustic Decoder

## 22.1 Đầu vào

```text
expanded segment features
+ F0 embedding
+ voiced/unvoiced
+ energy embedding
+ tone conditioning
+ frame position
```

## 22.2 Kiến trúc

Conformer hoặc FFT decoder nhẹ:

```yaml
layers: 4
hidden_size: 256
attention_heads: 4
ffn_size: 1024
conv_kernel_size: 15
```

## 22.3 Đầu ra

Mel-spectrogram 100 bins ở 24 kHz.

Dùng mel trong phiên bản đầu vì dễ quan sát, debug và thay vocoder.

---

# 23. iSTFTNet và tạo waveform

## 23.1 Lựa chọn

Dùng iSTFTNet làm vocoder mục tiêu, đồng thời train HiFi-GAN nhỏ làm baseline kiểm chứng.

## 23.2 Thông số audio

```yaml
sample_rate: 24000
n_fft: 1024
win_length: 1024
hop_length: 256
mel_bins: 100
f_min: 0
f_max: 12000
```

## 23.3 Vì sao 24 kHz

- Đủ cho giọng nói.
- Nhanh hơn 48 kHz.
- File nhỏ hơn.
- Giảm tải CPU và ARM.
- Lỗi phát âm không được giải quyết bằng tăng sample rate.

---

# 24. Dữ liệu từ video và SRT

## 24.1 SRT chỉ là gợi ý

SRT có thể lệch thời gian hoặc khác lời nói thực tế. Không dùng trực tiếp mà không căn chỉnh.

## 24.2 Pipeline

```text
Video gốc
  ↓
Trích WAV
  ↓
Đọc SRT
  ↓
Cắt vùng sơ bộ
  ↓
VAD
  ↓
ASR có timestamp
  ↓
So sánh ASR và SRT
  ↓
Forced alignment
  ↓
Cắt clip 2 đến 12 giây
  ↓
Lọc chất lượng
  ↓
Parser coverage check
  ↓
Manifest
```

## 24.3 Audio

- Mono.
- Không clipping.
- Không người khác.
- Không nhạc.
- Không khử nhiễu quá mạnh.
- Giữ 50 đến 150 ms đầu và 100 đến 300 ms cuối.

## 24.4 Transcript

Phải khớp điều thực sự được nói, không chỉ gần nghĩa.

## 24.5 Metadata

```json
{
  "id": "video_003_00125",
  "audio_path": "wavs/video_003_00125.wav",
  "text_raw": "Hôm nay mình thử chiếc máy mới.",
  "text_normalized": "Hôm nay mình thử chiếc máy mới.",
  "source_video": "video_003",
  "duration_sec": 3.84,
  "asr_cer": 0.0,
  "alignment_confidence": 0.97,
  "status": "accepted"
}
```

---

# 25. Thiết kế tập dữ liệu

## 25.1 Quy mô

```text
5 giờ: prototype
10 đến 20 giờ: baseline tốt
30 đến 50 giờ: mục tiêu chính
50 đến 100 giờ: tốt nếu nhất quán
```

## 25.2 Chia theo video

```text
80% source videos → train
10% source videos → validation
10% source videos → test
```

Không chia ngẫu nhiên clip từ cùng một video sang cả train và test.

## 25.3 Coverage

Đếm số lần xuất hiện của:

```text
mỗi onset
mỗi medial
mỗi nucleus
mỗi coda
mỗi tone
mỗi cặp nucleus-coda
mỗi tổ hợp tone-coda
```

Thu thêm 2 đến 5 giờ script có chủ đích nếu video tự nhiên thiếu coverage.

---

# 26. Quy trình huấn luyện

## Giai đoạn 0: Parser và dataset

- Parser pass unit test.
- Không unknown âm tiết trong train.
- Nghe kiểm tra ngẫu nhiên.
- Coverage report.

## Giai đoạn 1: Vocoder

```text
Ground-truth mel → iSTFTNet → waveform
```

Train thêm HiFi-GAN baseline.

## Giai đoạn 2: Acoustic model với ground-truth features

```text
structured text
+ true duration
+ true F0
+ true energy
→ mel
```

## Giai đoạn 3: Predictor

Train duration, F0, voiced/unvoiced và energy.

## Giai đoạn 4: Full predicted inference

```text
text → all predicted features → mel → waveform
```

## Giai đoạn 5: Joint fine-tuning

Fine-tune acoustic model và vocoder để giảm mismatch.

## Giai đoạn 6: Export và tối ưu

- ONNX FP32.
- FP16.
- INT8 từng module.
- Benchmark.
- Distillation nếu cần model Nano.

---

# 27. Hệ thống loss

```text
L_total =
    λ_mel × L_mel
  + λ_duration × L_duration
  + λ_f0 × L_f0
  + λ_vuv × L_vuv
  + λ_energy × L_energy
  + λ_tone × L_tone
  + λ_alignment × L_alignment
  + λ_stft × L_stft
  + λ_adv × L_adversarial
  + λ_fm × L_feature_matching
```

## 27.1 Duration

```text
MSE(log(1 + d_pred), log(1 + d_true))
```

## 27.2 F0

- L1 log-F0.
- Delta-F0.
- Syllable contour loss.
- Voiced/unvoiced BCE.

## 27.3 Tone

Cross entropy sáu thanh từ acoustic hidden state.

## 27.4 Vocoder

- Multi-resolution STFT.
- Mel reconstruction.
- Adversarial.
- Feature matching.

---

# 28. Đánh giá

## 28.1 Nội dung

- ASR round-trip CER.
- Syllable Error Rate.
- Tone Error Rate.
- Missing Segment Rate.
- Repetition Rate.

## 28.2 Tự nhiên

- MOS.
- CMOS giữa checkpoint.
- Listening test bởi người bản xứ đúng vùng.

## 28.3 Âm sắc

- Speaker embedding similarity.
- Kiểm tra câu ngoài tập train.

## 28.4 Tốc độ

```text
RTF = synthesis time / audio duration
```

Đo:

- Cold start.
- Warm start.
- 3 giây, 10 giây, 30 giây.
- Batch 1, 4, 8, 16.
- RAM và VRAM.

---

# 29. Ngăn đọc sai, nuốt chữ và lặp chữ

## 29.1 Frontend xác định

Số, ngày và ký hiệu được mở rộng thành chữ.

## 29.2 Parser không âm thầm fallback

Âm tiết không hợp lệ trả lỗi hoặc yêu cầu từ điển override.

## 29.3 Duration constraints

- Active segment không có duration 0.
- Inactive segment luôn duration 0.
- Pause có giới hạn.
- Tổng thời gian có biên thống kê.

## 29.4 Non-autoregressive

Không dùng attention tuần tự kiểu Tacotron, giảm nguy cơ lặp và dừng sớm.

## 29.5 ASR verification

Trong batch xuất bản:

```text
Text → TTS → ASR → CER
```

Câu lỗi được chia nhỏ hoặc đưa kiểm tra.

---

# 30. ONNX và triển khai đa nền tảng

## 30.1 File phát hành

```text
model-fp32.onnx
model-fp16.onnx
model-int8.onnx
frontend-rules.json
component-vocabs.json
model-config.json
pronunciation-dictionary.json
```

## 30.2 Runtime

```text
Frontend Rust/C++
→ structured tensors
→ ONNX Runtime
→ PCM
```

## 30.3 Windows

- CPU EP.
- CUDA EP.
- TensorRT EP nếu tương thích.

## 30.4 macOS

- ONNX Runtime ARM64.
- Có thể xuất Core ML riêng.
- Benchmark thực tế trước khi chọn backend.

## 30.5 ARM Linux

- ONNX Runtime ARM64.
- NEON.
- INT8 hoặc mixed precision.

## 30.6 Quantization

Quantize trước:

- Embeddings.
- Text Encoder linear layers.
- Duration và energy predictor.

Thận trọng:

- F0 predictor.
- Norm layers.
- Mel projection.
- iSTFTNet.

---

# 31. Cấu hình mô hình đề xuất

```yaml
model_name: ViFastTTS-Base-v2
language: vi
sample_rate: 24000
speakers: 1

input:
  unit: syllable
  features:
    - onset
    - medial
    - nucleus
    - coda
    - tone
    - word_boundary
    - phrase_boundary
    - punctuation
  dialect_conditioning: false
  speaker_conditioning: false
  style_conditioning: false

embedding:
  component_dim: 96
  composition: concat_linear
  output_dim: 256

text_encoder:
  type: conformer
  layers: 6
  hidden_size: 256
  attention_heads: 4
  ffn_size: 1024
  conv_kernel_size: 15
  dropout: 0.1

segment_expander:
  roles:
    - onset
    - medial
    - nucleus
    - coda

alignment:
  external_targets_first: true
  trainable_monotonic_later: true

duration_predictor:
  type: deterministic
  layers: 3
  channels: 256
  kernel_size: 3

f0_predictor:
  type: tone_conditioned
  layers: 4
  channels: 256
  predict_vuv: true
  predict_residual_intonation: true

energy_predictor:
  layers: 2
  channels: 256
  kernel_size: 3

acoustic_decoder:
  type: conformer
  layers: 4
  hidden_size: 256
  attention_heads: 4
  ffn_size: 1024
  mel_bins: 100

waveform_decoder:
  type: istftnet

inference:
  deterministic: true
  speed_min: 0.75
  speed_max: 1.35
```

Mục tiêu quy mô:

```text
40 đến 60 triệu tham số
FP32: 160 đến 240 MB
FP16: 80 đến 120 MB
INT8/mixed: khoảng 45 đến 80 MB
```

---

# 32. Cấu trúc mã nguồn

```text
vifasttts/
├── frontend/
│   ├── normalizer/
│   │   ├── unicode.py
│   │   ├── lexer.py
│   │   ├── numbers.py
│   │   ├── dates.py
│   │   ├── times.py
│   │   ├── units.py
│   │   └── abbreviations.py
│   ├── syllable/
│   │   ├── tone.py
│   │   ├── onset.py
│   │   ├── coda.py
│   │   ├── vowel.py
│   │   ├── special_qu.py
│   │   ├── special_gi.py
│   │   ├── validator.py
│   │   ├── roundtrip.py
│   │   └── parser.py
│   ├── boundaries.py
│   └── vocabs/
├── data/
│   ├── extract_audio.py
│   ├── import_srt.py
│   ├── vad.py
│   ├── asr_align.py
│   ├── forced_align.py
│   ├── quality_filter.py
│   ├── coverage.py
│   └── manifest.py
├── models/
│   ├── component_embeddings.py
│   ├── syllable_composer.py
│   ├── conformer.py
│   ├── segment_expander.py
│   ├── duration.py
│   ├── pitch.py
│   ├── energy.py
│   ├── acoustic_decoder.py
│   ├── istftnet.py
│   └── discriminators.py
├── train/
│   ├── vocoder.py
│   ├── acoustic.py
│   ├── predictors.py
│   ├── joint.py
│   └── distill.py
├── evaluation/
│   ├── cer.py
│   ├── syllable_error.py
│   ├── tone_error.py
│   ├── duration_checks.py
│   └── benchmark.py
├── export/
│   ├── onnx.py
│   ├── quantize.py
│   ├── coreml.py
│   └── validate.py
└── tests/
    ├── test_unicode.py
    ├── test_tone.py
    ├── test_onset.py
    ├── test_coda.py
    ├── test_qu.py
    ├── test_gi.py
    ├── test_roundtrip.py
    ├── test_minimal_pairs.py
    └── test_onnx.py
```

---

# 33. Lộ trình triển khai

## Giai đoạn A: Frontend

1. Unicode và tone extractor.
2. Onset/coda longest match.
3. Bảng medial/nucleus.
4. `qu`, `gi`, `i/y` và bán nguyên âm cuối.
5. Validator.
6. Round-trip.
7. Corpus fuzz test.

## Giai đoạn B: Dataset

1. Trích audio và SRT.
2. VAD và ASR.
3. Forced alignment.
4. Lọc.
5. Coverage report.
6. Thu bổ sung.

## Giai đoạn C: Baseline

1. FastSpeech2 hoặc VITS baseline.
2. Xác minh dataset.
3. Xác minh parser.

## Giai đoạn D: ViFastTTS v2

1. iSTFTNet.
2. Component embeddings.
3. Syllable Conformer.
4. Segment expander.
5. Predictors.
6. Joint fine-tuning.

## Giai đoạn E: Deployment

1. ONNX FP32.
2. FP16.
3. INT8/mixed.
4. Runtime Rust/C++.
5. Benchmark đa nền tảng.

---

# 34. Rủi ro và cách xử lý

## Parser phân tích sai nhưng vẫn hợp lệ

Cách xử lý:

- Round-trip.
- Bảng âm tiết hợp lệ.
- Minimal-pair tests.
- Kiểm tra corpus lớn.

## `qu` và `gi` không nhất quán

Cách xử lý:

- Module chuyên biệt.
- Không rải quy tắc ở nhiều file.
- Snapshot tests.

## Transcript sai

Cách xử lý:

- ASR comparison.
- Forced alignment confidence.
- Manual sampling.

## Audio nhiều điều kiện thu

Cách xử lý:

- Chọn profile chính.
- Chia test theo video.
- Loại profile hiếm hoặc xử lý nhất quán.

## Thanh đúng nhưng giọng phẳng

Cách xử lý:

- Residual intonation.
- Phrase boundaries.
- Dữ liệu câu hỏi và câu dài.

## INT8 làm xấu âm

Cách xử lý:

- Quantize từng module.
- Giữ F0/vocoder ở FP16 hoặc FP32.
- Calibration corpus đầy đủ thanh và coda.

---

# 35. Tiêu chí hoàn thành

## Parser

- 100% unit tests bắt buộc pass.
- Không mất `ă â ê ô ơ ư`.
- Round-trip pass trên bảng âm tiết chuẩn.
- Không gộp minimal pairs.
- Không unknown trên test tiếng Việt chuẩn.

## Model

- Không active segment duration 0.
- Không lặp hoặc nuốt ở ít nhất 99% bộ test chuẩn.
- ASR round-trip CER mục tiêu dưới 2% trên văn bản phổ thông.
- Tone error đủ thấp theo người nghe bản xứ.
- MOS nội bộ mục tiêu ít nhất 4/5.

## Tốc độ mục tiêu

Đây là mục tiêu kỹ thuật cần benchmark thực tế:

```text
CPU desktop: RTF < 0.10
Apple Silicon: RTF < 0.07
RTX 3060 warm: RTF < 0.02
ARM mạnh: real-time hoặc nhanh hơn
```

## Deployment

- PyTorch và ONNX gần tương đương.
- Runtime không cần Python.
- Có FP32 tham chiếu.
- Có bản tối ưu không tăng đáng kể lỗi đọc.

---

# 36. Phụ lục kiểm thử parser

## 36.1 Chất lượng nguyên âm

```text
học → H, NONE, O, C, NANG
hộc → H, NONE, Ô, C, NANG
hợc → H, NONE, Ơ, C, NANG nếu âm tiết được chấp nhận trong corpus mục tiêu
```

## 36.2 Coda

```text
khanh → KH, NONE, A, NH, NGANG
khang → KH, NONE, A, NG, NGANG
hộc → H, NONE, Ô, C, NANG
hột → H, NONE, Ô, T, NANG
```

## 36.3 Onset bảo toàn

```text
trăm → TR, NONE, Ă, M, NGANG
chăm → CH, NONE, Ă, M, NGANG
sương → S, NONE, ƯƠ, NG, NGANG
xương → X, NONE, ƯƠ, NG, NGANG
```

## 36.4 Tone bảo toàn

```text
bảo → B, NONE, A, W, HOI
bão → B, NONE, A, W, NGA
```

## 36.5 Medial

```text
hoa → H, W, A, NONE, NGANG
hoàn → H, W, A, N, HUYEN
ngoài → NG, W, A, J, HUYEN
```

## 36.6 Nucleus đôi

```text
tiến → T, NONE, IÊ, N, SAC
mía → M, NONE, IÊ, NONE, SAC
muốn → M, NONE, UÔ, N, SAC
mua → M, NONE, UÔ, NONE, NGANG
tưởng → T, NONE, ƯƠ, NG, HOI
mưa → M, NONE, ƯƠ, NONE, NGANG
```

## 36.7 Chính tả biến thể

```text
ca và ki → onset abstract K
ga và ghi → onset abstract G
nga và nghi → onset abstract NG
```

## 36.8 Trường hợp đặc biệt

```text
quả → K, W, A, NONE, HOI
quyển → K, W, IÊ, N, HOI
giá → GI, NONE, A, NONE, SAC
giếng → GI, NONE, IÊ, NG, SAC
```

---

# 37. Tài liệu tham khảo

1. FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. https://arxiv.org/abs/2006.04558
2. JETS: Jointly Training FastSpeech2 and HiFi-GAN for End-to-End Text-to-Speech. https://arxiv.org/abs/2203.16852
3. VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. https://arxiv.org/abs/2106.06103
4. VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech. https://arxiv.org/abs/2307.16430
5. Kokoro-82M model card. https://huggingface.co/hexgrad/Kokoro-82M
6. Kokoro discussion về style diffusion và iSTFTNet. https://huggingface.co/hexgrad/Kokoro-82M/discussions/111
7. VieNeu SDK Overview. https://docs.vieneu.io/docs/sdk/overview/
8. VieNeu Streaming Benchmarks. https://github.com/pnnbao97/VieNeu-TTS/blob/main/docs/streaming.md
9. Piper Training Guide. https://tderflinger.github.io/piper-docs/guides/training/
10. Vietnamese syllable structure and spelling rules. https://vietsyllables.com/basics/vietnamese-syllables
11. Vietnamese syllable chart. https://vietnameselab.com/syllable-chart
12. Bảng vần tiếng Việt. https://s.ngonngu.net/syllables/rhymes/

---

# Quyết định cuối cùng trong một đoạn

Bộ frontend phải được xây một lần cho tiếng Việt nói chung, không gắn với vùng miền. Mỗi âm tiết được phân tích thành năm thuộc tính trừu tượng `ONSET`, `MEDIAL`, `NUCLEUS`, `CODA`, `TONE`, đồng thời giữ lớp chính tả để round-trip và debug. Mô hình neural nhận một vector cấu trúc cho mỗi âm tiết, không nhận `dialect_id`, `speaker_id` hoặc `style_id`, vì mỗi checkpoint chỉ chứa một người nói và một phương ngữ. Text Encoder làm việc ở cấp âm tiết, sau đó Segment Expander tạo onset, medial, nucleus và coda để dự đoán duration chi tiết; tone điều kiện hóa duration, F0, phonation, energy và acoustic decoder. Kiến trúc non-autoregressive, deterministic, dùng Conformer nhỏ và iSTFTNet 24 kHz, sau đó export ONNX để chạy nhanh trên nhiều nền tảng.
