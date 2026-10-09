# Thiết kế và xây dựng hệ thống TTS tiếng Việt miền Bắc một giọng, chất lượng cao và tốc độ nhanh

**Phiên bản tài liệu:** 1.0  
**Ngày:** 09/10/2026  
**Mục tiêu:** Xây dựng từ đầu một hệ thống chuyển văn bản thành giọng nói, chỉ hỗ trợ tiếng Việt miền Bắc, chỉ có một người đọc, ưu tiên đọc đúng, ổn định, nhanh trên CPU, Apple Silicon, NVIDIA GPU và ARM.

---

## Mục lục

1. [Tóm tắt quyết định kiến trúc](#1-tóm-tắt-quyết-định-kiến-trúc)
2. [Giải thích TTS theo cách đơn giản](#2-giải-thích-tts-theo-cách-đơn-giản)
3. [Yêu cầu và phạm vi dự án](#3-yêu-cầu-và-phạm-vi-dự-án)
4. [Kiến trúc cuối cùng](#4-kiến-trúc-cuối-cùng)
5. [Thành phần 1: Chuẩn hóa văn bản tiếng Việt Bắc](#5-thành-phần-1-chuẩn-hóa-văn-bản-tiếng-việt-bắc)
6. [Thành phần 2: Phân tích âm tiết và tokenizer](#6-thành-phần-2-phân-tích-âm-tiết-và-tokenizer)
7. [Thành phần 3: Text Encoder dùng Conformer nhỏ](#7-thành-phần-3-text-encoder-dùng-conformer-nhỏ)
8. [Thành phần 4: Căn chỉnh văn bản với âm thanh](#8-thành-phần-4-căn-chỉnh-văn-bản-với-âm-thanh)
9. [Thành phần 5: Dự đoán duration](#9-thành-phần-5-dự-đoán-duration)
10. [Thành phần 6: Dự đoán F0 và thanh điệu](#10-thành-phần-6-dự-đoán-f0-và-thanh-điệu)
11. [Thành phần 7: Dự đoán năng lượng](#11-thành-phần-7-dự-đoán-năng-lượng)
12. [Thành phần 8: Length Regulator và Acoustic Decoder](#12-thành-phần-8-length-regulator-và-acoustic-decoder)
13. [Thành phần 9: iSTFTNet tạo waveform](#13-thành-phần-9-istftnet-tạo-waveform)
14. [Pipeline dữ liệu từ video YouTube và SRT](#14-pipeline-dữ-liệu-từ-video-youtube-và-srt)
15. [Thiết kế tập dữ liệu](#15-thiết-kế-tập-dữ-liệu)
16. [Quy trình huấn luyện từng giai đoạn](#16-quy-trình-huấn-luyện-từng-giai-đoạn)
17. [Loss function và mục đích từng loss](#17-loss-function-và-mục-đích-từng-loss)
18. [Đánh giá chất lượng](#18-đánh-giá-chất-lượng)
19. [Ngăn đọc sai, nuốt chữ, lặp chữ và thêm chữ](#19-ngăn-đọc-sai-nuốt-chữ-lặp-chữ-và-thêm-chữ)
20. [Tối ưu tốc độ và triển khai đa nền tảng](#20-tối-ưu-tốc-độ-và-triển-khai-đa-nền-tảng)
21. [Cấu hình mô hình đề xuất](#21-cấu-hình-mô-hình-đề-xuất)
22. [Cấu trúc dự án](#22-cấu-trúc-dự-án)
23. [Lộ trình thực hiện](#23-lộ-trình-thực-hiện)
24. [Rủi ro và cách xử lý](#24-rủi-ro-và-cách-xử-lý)
25. [Tiêu chí hoàn thành](#25-tiêu-chí-hoàn-thành)
26. [Thuật ngữ](#26-thuật-ngữ)
27. [Tài liệu tham khảo](#27-tài-liệu-tham-khảo)

---

# 1. Tóm tắt quyết định kiến trúc

Kiến trúc cuối cùng được đề xuất là:

> **Vietnamese North Frontend + Onset/Rime/Tone Tokenizer + Conformer Encoder + Deterministic Duration/Pitch/Energy Predictors + Lightweight Acoustic Decoder + iSTFTNet, chạy ở 24 kHz.**

Tên tạm dùng trong tài liệu:

> **ViFastTTS**

Đây không phải tên một dự án mã nguồn mở có sẵn. Nó là tên tạm cho kiến trúc sẽ được tự xây dựng bằng cách tham khảo các ý tưởng tốt từ FastSpeech 2, JETS, Kokoro, VITS và VITS2.

## 1.1 Sơ đồ tổng thể

```text
Văn bản thô
    │
    ▼
Bộ chuẩn hóa tiếng Việt miền Bắc
    │
    ▼
Bộ phân tích âm tiết
    │
    ▼
Token: phụ âm đầu + vần + thanh điệu + dấu câu
    │
    ▼
Conformer Text Encoder
    │
    ├───────────────┬────────────────┐
    ▼               ▼                ▼
Duration         F0/Pitch          Energy
Predictor        Predictor         Predictor
    │               │                │
    └───────────────┴────────────────┘
                    │
                    ▼
             Length Regulator
                    │
                    ▼
          Lightweight Acoustic Decoder
                    │
                    ▼
              Mel-spectrogram
                    │
                    ▼
                 iSTFTNet
                    │
                    ▼
             Waveform PCM 24 kHz
```

## 1.2 Vì sao chọn kiến trúc này?

Mục tiêu của dự án không phải tạo một mô hình biết nhiều giọng, nhiều ngôn ngữ hoặc bắt chước cảm xúc từ audio mẫu. Mục tiêu là một giọng Bắc cố định, đọc rõ, đều, đúng chữ và nhanh.

Vì vậy, kiến trúc cần:

- Không autoregressive, để không phải sinh từng token âm thanh theo thứ tự.
- Duration xác định, để giảm nuốt chữ, lặp chữ và nhịp đọc bất thường.
- F0 được giám sát rõ ràng, vì tiếng Việt là ngôn ngữ thanh điệu.
- Frontend do mình kiểm soát, để đọc đúng số, ngày tháng, đơn vị và viết tắt.
- Không dùng voice cloning, speaker encoder hoặc style diffusion.
- Dễ xuất ONNX và dễ lượng tử hóa.
- Waveform decoder nhẹ, phù hợp CPU và ARM.

FastSpeech 2 dùng kiến trúc non-autoregressive và mô hình hóa rõ duration, pitch và energy thông qua Variance Adaptor [1]. JETS kết hợp FastSpeech 2, alignment module và vocoder trong một hệ thống có thể huấn luyện chung [2]. Kokoro bỏ style diffusion khi suy luận và dùng iSTFTNet, hướng đến tốc độ cao [3].

## 1.3 Những gì không chọn

### Không chọn LLM cộng audio codec làm kiến trúc chính

Các hệ như VieNeu-TTS v3 Turbo có nhiều lợi thế về nhiều giọng, cloning, biểu cảm, streaming và tiếng Việt xen tiếng Anh. Tuy nhiên, chúng vẫn phải sinh audio token theo chuỗi và giải mã bằng codec. Điều đó không cần thiết với một giọng cố định.

### Không chọn mô hình tự hồi quy

Mô hình tự hồi quy tạo đầu ra theo thứ tự, phần sau phụ thuộc phần trước. Nó có thể tự nhiên nhưng dễ gặp lỗi lặp, dừng sớm và chậm hơn.

### Không chọn diffusion nhiều bước

Diffusion có thể cho chất lượng cao nhưng cần nhiều vòng suy luận. Mục tiêu của dự án là tốc độ CPU và ARM, nên không phù hợp.

### Không dùng VITS nguyên bản làm kiến trúc cuối

VITS rất tốt để làm baseline. Tuy nhiên, VAE, normalizing flow và stochastic duration predictor làm mô hình khó debug hơn. Với mục tiêu đọc ổn định, duration xác định và F0 tách riêng dễ kiểm soát hơn [4].

---

# 2. Giải thích TTS theo cách đơn giản

Một hệ thống TTS giống như một người đọc theo sáu bước:

1. Nhìn văn bản và hiểu cách đọc ký hiệu.
2. Chia từ thành các âm tiết.
3. Biết âm tiết nào đọc trước, âm tiết nào đọc sau.
4. Quyết định mỗi âm kéo dài bao lâu.
5. Quyết định cao độ, thanh điệu và độ mạnh.
6. Tạo ra tín hiệu âm thanh.

Ví dụ:

```text
Đầu vào: "Hôm nay là 15/10."
```

Bước chuẩn hóa:

```text
"Hôm nay là ngày mười lăm tháng mười."
```

Bước phân tích âm tiết:

```text
hôm | nay | là | ngày | mười | lăm | tháng | mười
```

Bước tokenizer tách từng âm tiết thành thành phần:

```text
hôm   = H + OM + HUYỀN
nay   = N + AY + NGANG
là    = L + A + HUYỀN
```

Sau đó mô hình dự đoán:

- Âm `hôm` dài bao nhiêu mili giây.
- Đường cao độ của thanh huyền.
- Chỗ nào cần nghỉ.
- Cường độ từng đoạn.
- Dạng phổ âm thanh tương ứng.

Cuối cùng, iSTFTNet biến dạng phổ thành sóng âm có thể phát ra loa.

---

# 3. Yêu cầu và phạm vi dự án

## 3.1 Yêu cầu chức năng

Hệ thống cần:

- Chỉ hỗ trợ tiếng Việt.
- Tối ưu cho một giọng miền Bắc.
- Đọc văn bản thông thường, truyện, mô tả và lời thuyết minh.
- Đọc đúng dấu thanh.
- Đọc đúng số và đơn vị phổ biến.
- Không nuốt âm tiết.
- Không lặp âm tiết.
- Không tự thêm từ.
- Có thể điều chỉnh tốc độ đọc.
- Xuất audio PCM/WAV.
- Có thể xử lý từng câu hoặc một danh sách câu.

## 3.2 Yêu cầu phi chức năng

- Chạy real-time trên CPU desktop.
- Chạy nhanh hơn real-time trên Apple Silicon.
- Chạy tốt trên NVIDIA GPU.
- Có bản dùng được trên ARM64.
- Không phụ thuộc Python khi triển khai cuối.
- Model có thể xuất ONNX.
- Kết quả xác định khi dùng cùng đầu vào và cùng cấu hình.
- Có test tự động cho frontend.

## 3.3 Ngoài phạm vi phiên bản đầu

- Nhiều người nói.
- Giọng miền Trung hoặc miền Nam.
- Voice cloning.
- Điều khiển cảm xúc.
- Tiếng Anh xen tiếng Việt hoàn chỉnh.
- Hội thoại nhiều nhân vật.
- Streaming frame-level cực thấp độ trễ.
- Hát hoặc ngâm thơ.

Việc giới hạn phạm vi giúp toàn bộ dung lượng mô hình tập trung vào đúng một giọng và một ngôn ngữ.

---

# 4. Kiến trúc cuối cùng

## 4.1 Luồng khi huấn luyện

```text
Văn bản gốc ──► Normalizer ──► Tokenizer ──► Text Encoder
                                                   │
Audio thật ──► Spectrogram ──► Alignment ──────────┤
      │                                            │
      ├──► F0 extractor ───────────────────────────┤
      ├──► Energy extractor ───────────────────────┤
      └──► Waveform supervision                    │
                                                   ▼
                                      Duration/F0/Energy Predictors
                                                   │
                                                   ▼
                                            Acoustic Decoder
                                                   │
                                                   ▼
                                             Mel dự đoán
                                                   │
                                                   ▼
                                               iSTFTNet
                                                   │
                                                   ▼
                                           Waveform dự đoán
```

## 4.2 Luồng khi sử dụng

```text
Văn bản
  ↓
Normalizer
  ↓
Tokenizer
  ↓
Text Encoder
  ↓
Duration + F0 + Energy
  ↓
Acoustic Decoder
  ↓
iSTFTNet
  ↓
Audio
```

Khi sử dụng thực tế, không cần audio mẫu, forced aligner, posterior encoder hoặc discriminator. Các thành phần chỉ phục vụ huấn luyện được loại bỏ khỏi gói runtime.

## 4.3 Nguyên tắc thiết kế

1. **Đúng trước, tự nhiên sau.**
2. **Frontend quyết định cách đọc, neural model quyết định cách phát âm.**
3. **Không để mô hình phải đoán những gì có thể giải bằng luật.**
4. **Không gộp mất thông tin chính tả trong tokenizer.**
5. **Mọi bước phải quan sát và kiểm thử được.**
6. **Inference phải đơn giản hơn training.**

---

# 5. Thành phần 1: Chuẩn hóa văn bản tiếng Việt Bắc

## 5.1 Vai trò

Mô hình không nên nhận trực tiếp chuỗi như:

```text
Ngày 15/10, nhiệt độ là 25°C, tăng 3.5%.
```

Nó nên nhận chuỗi đã chuẩn hóa:

```text
Ngày mười lăm tháng mười, nhiệt độ là hai mươi lăm độ C, tăng ba phẩy năm phần trăm.
```

Nếu normalizer sai, mô hình dù tốt đến đâu cũng đọc sai.

## 5.2 Các lớp xử lý

### Bước 1: Chuẩn Unicode

- Chuyển về Unicode NFC.
- Loại ký tự điều khiển không cần thiết.
- Chuẩn hóa khoảng trắng.
- Chuẩn hóa dấu nháy và dấu gạch.
- Giữ nguyên dấu tiếng Việt.

### Bước 2: Tách câu

Không chỉ tách theo dấu chấm. Cần tránh tách sai:

```text
TP.HCM
ThS.
P.GS
v.v.
1.5
example.com
```

### Bước 3: Lexer

Lexer nhận diện loại token:

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

### Bước 4: Parser theo ngữ cảnh

Cùng một chuỗi có thể có nhiều cách đọc:

```text
3/2 trong "ngày 3/2"     → ngày ba tháng hai
3/2 trong "phân số 3/2"  → ba phần hai
3:2 trong "tỷ số 3:2"    → ba hai hoặc ba trên hai
3.2 trong "phiên bản 3.2" → ba chấm hai
```

Parser chọn cách đọc dựa vào từ xung quanh.

## 5.3 Quy tắc tiếng Việt miền Bắc

Dùng từ vựng mặc định:

```text
1000  → một nghìn
1005  → một nghìn không trăm linh năm
105   → một trăm linh năm
15    → mười lăm
21    → hai mươi mốt
24    → hai mươi tư
```

Không dùng lẫn `nghìn/ngàn` hoặc `linh/lẻ` nếu muốn giọng nhất quán.

## 5.4 Chế độ nghiêm ngặt

Nên có hai chế độ:

```text
strict = true
strict = false
```

Trong chế độ nghiêm ngặt, cấu trúc mơ hồ sẽ báo lỗi hoặc cảnh báo thay vì đoán.

Ví dụ:

```text
"Hẹn vào 3/4"
```

Có thể là ngày ba tháng tư hoặc tỷ lệ ba phần tư. Nếu không có ngữ cảnh, hệ thống nên trả cảnh báo.

## 5.5 Kiểm thử normalizer

Mỗi luật phải có unit test:

```text
input:  "105"
output: "một trăm linh năm"

input:  "10:30"
output: "mười giờ ba mươi phút"

input:  "25°C"
output: "hai mươi lăm độ C"

input:  "3,5%"
output: "ba phẩy năm phần trăm"
```

Cần duy trì một bộ test hồi quy. Khi sửa luật số điện thoại, không được làm hỏng luật số thập phân.

---

# 6. Thành phần 2: Phân tích âm tiết và tokenizer

## 6.1 Vì sao không dùng BPE?

BPE phù hợp với mô hình ngôn ngữ vì nó chia văn bản thành mảnh thường gặp. Nhưng TTS cần biết cách phát âm. Một token BPE có thể chứa nhiều âm tiết và không biểu diễn rõ thanh điệu.

## 6.2 Vì sao không dùng ký tự thuần?

Ký tự thuần vẫn có thể hoạt động, nhưng mô hình phải tự học:

- Dấu thanh nằm trên nguyên âm nào.
- `ngh` và `ng` có quan hệ gì.
- `ươ` là một cụm âm chính.
- `qu` và `gi` cần xử lý đặc biệt.

Ta có thể giảm gánh nặng cho mô hình bằng bộ phân tích âm tiết có luật.

## 6.3 Biểu diễn được chọn

Mỗi âm tiết được biểu diễn bằng:

```text
<ONSET> <RIME> <TONE>
```

Trong đó:

- `ONSET`: phụ âm đầu.
- `RIME`: phần vần, gồm âm đệm, âm chính và âm cuối.
- `TONE`: thanh điệu.

Ví dụ:

```text
trường → <ONSET_TR> <RIME_UONG> <TONE_HUYEN>
chuyện → <ONSET_CH> <RIME_UYEN> <TONE_NANG>
người  → <ONSET_NG> <RIME_UOI> <TONE_HUYEN>
học    → <ONSET_H> <RIME_OC> <TONE_NANG>
```

## 6.4 Token đặc biệt

```text
<PAD>
<BOS>
<EOS>
<SYLLABLE_BOUNDARY>
<WORD_BOUNDARY>
<PHRASE_BOUNDARY>
<PAUSE_SHORT>
<PAUSE_MEDIUM>
<PAUSE_LONG>
<COMMA>
<PERIOD>
<QUESTION>
<EXCLAMATION>
<UNKNOWN>
```

## 6.5 Không gộp cặp âm theo giọng Bắc

Vẫn giữ các token khác nhau:

```text
<ONSET_TR> != <ONSET_CH>
<ONSET_S>  != <ONSET_X>
<ONSET_D>  != <ONSET_GI>
<ONSET_R>  != <ONSET_D>
<TONE_HOI> != <TONE_NGA>
```

Dù người đọc có thể phát âm gần giống nhau, việc giữ riêng giúp:

- Không mất thông tin chính tả.
- Dễ kiểm tra lỗi.
- Dễ bổ sung giọng khác trong tương lai.
- Cho phép mô hình học khác biệt nhỏ nếu người nói thực sự tạo ra khác biệt.

## 6.6 Round-trip test

Tokenizer cần hỗ trợ kiểm tra:

```text
text
  ↓
tokens
  ↓
reconstructed syllables
```

Không cần khôi phục đúng dấu câu ban đầu 100%, nhưng phải khôi phục đúng chuỗi âm tiết sau chuẩn hóa.

Nếu đầu vào:

```text
"một trăm linh năm"
```

thì quá trình round-trip không được biến thành:

```text
"một chăm linh lăm"
```

## 6.7 Xử lý từ ngoài từ điển

Tiếng Việt có chính tả tương đối đều. Parser nên xử lý bằng luật và có từ điển ngoại lệ nhỏ cho:

- Tên riêng.
- Từ mượn.
- Chữ viết tắt.
- Từ tiếng Anh phổ biến.
- Cách đọc thương hiệu.

Phiên bản đầu có thể dùng spelling mode:

```text
USB → u ét bê
AI  → ây ai hoặc a i, tùy từ điển dự án
```

---

# 7. Thành phần 3: Text Encoder dùng Conformer nhỏ

## 7.1 Text Encoder làm gì?

Tokenizer chỉ tạo danh sách token. Text Encoder biến mỗi token thành vector có chứa ngữ cảnh.

Ví dụ từ `năm` có thể là:

- Số năm.
- Năm trong thời gian.
- Động từ nằm nếu văn bản bị sai dấu.

Sau normalizer, phần lớn mơ hồ đã được giải quyết. Text Encoder tiếp tục học quan hệ giữa các âm tiết, từ và dấu câu.

## 7.2 Vì sao dùng Conformer?

Conformer kết hợp:

- Self-attention để nhìn toàn câu.
- Convolution để học cấu trúc cục bộ.

TTS cần cả hai:

- Dấu hỏi cuối câu có thể ảnh hưởng ngữ điệu toàn câu.
- Âm tiết lân cận ảnh hưởng cách nối âm và nhịp đọc.

## 7.3 Cấu hình ban đầu

```yaml
text_encoder:
  type: conformer
  layers: 6
  hidden_size: 256
  attention_heads: 4
  ffn_size: 1024
  conv_kernel_size: 15
  dropout: 0.1
```

Có thể thử kernel 31 nếu câu dài và dữ liệu đủ lớn.

## 7.4 Embedding đầu vào

Vector đầu vào nên là tổng của:

```text
token embedding
+ vị trí token
+ vị trí trong âm tiết
+ ranh giới từ
+ ranh giới cụm
+ loại dấu câu
```

Không cần speaker embedding vì chỉ có một người đọc.

Không cần language embedding vì chỉ có tiếng Việt.

Không cần style embedding trong phiên bản đầu.

---

# 8. Thành phần 4: Căn chỉnh văn bản với âm thanh

## 8.1 Alignment là gì?

Giả sử câu có năm âm tiết và audio dài ba giây. Mô hình cần biết mỗi âm tiết ứng với đoạn audio nào.

```text
"xin"  → 0,00 đến 0,35 giây
"chào" → 0,35 đến 0,80 giây
"các"  → 0,80 đến 1,15 giây
"bạn"  → 1,15 đến 1,60 giây
```

Nếu alignment sai, mô hình có thể:

- Kéo dài âm sai.
- Nuốt từ.
- Phát âm chồng.
- Học sai đường F0 của thanh điệu.

## 8.2 Hai lớp alignment

### Alignment ngoại tuyến để lọc dữ liệu

Dùng ASR hoặc CTC forced aligner để kiểm tra transcript và audio có khớp không.

Mục đích:

- Phát hiện transcript thiếu từ.
- Phát hiện audio có thêm lời.
- Phát hiện clip cắt mất đầu hoặc cuối.
- Tính confidence cho từng câu.

### Alignment bên trong khi training

Có thể tham khảo alignment module của JETS hoặc Monotonic Alignment Search. JETS học alignment trong joint training và giảm phụ thuộc vào công cụ bên ngoài [2].

## 8.3 Lộ trình đơn giản

Phiên bản đầu nên:

1. Dùng forced alignment bên ngoài tạo duration gần đúng.
2. Train acoustic model bằng duration đó.
3. Khi baseline hoạt động, thêm trainable alignment module.
4. So sánh hai cách trên cùng tập test.

Cách này dễ debug hơn việc xây mọi thứ cùng lúc.

---

# 9. Thành phần 5: Dự đoán duration

## 9.1 Duration là gì?

Duration là số frame âm thanh dành cho một token hoặc một âm tiết.

Ví dụ:

```text
<ONSET_CH>  = 3 frame
<RIME_AO>   = 12 frame
<TONE_HUYEN> không nhất thiết có duration riêng
```

Có hai cách:

- Dự đoán duration theo phoneme/rime.
- Dự đoán tổng duration âm tiết rồi phân bổ bên trong.

Với tokenizer onset/rime/tone, cách hợp lý là:

- Onset có duration.
- Rime có duration.
- Tone dùng làm điều kiện cho F0, không cần chiếm frame riêng.

## 9.2 Dùng deterministic predictor

Production mặc định dùng duration xác định:

```text
cùng văn bản + cùng tốc độ → cùng duration
```

Điều này giúp:

- Dễ tái tạo lỗi.
- Kết quả ổn định.
- Không tự ngắt nghỉ khác nhau mỗi lần.
- Giảm khả năng token có duration bằng 0.

## 9.3 Ràng buộc an toàn

Sau khi dự đoán, thực hiện:

```text
duration = clamp(duration, min_duration, max_duration)
```

Quy tắc gợi ý:

- Token phát âm không được có duration bằng 0.
- Pause ngắn, vừa, dài có khoảng duration riêng.
- Tổng thời gian câu phải nằm trong phạm vi hợp lý theo số âm tiết.
- Nếu tốc độ vượt ngưỡng, cảnh báo thay vì ép quá mạnh.

## 9.4 Điều chỉnh tốc độ đọc

```text
adjusted_duration = predicted_duration / speed
```

Ví dụ:

```text
speed = 1.0 → bình thường
speed = 1.1 → nhanh hơn
speed = 0.9 → chậm hơn
```

Nên giới hạn khoảng thực tế, ví dụ 0.75 đến 1.35 trong phiên bản đầu.

---

# 10. Thành phần 6: Dự đoán F0 và thanh điệu

## 10.1 F0 là gì?

F0 là tần số cơ bản của giọng. Có thể hiểu đơn giản là đường cao thấp của giọng theo thời gian.

Tiếng Việt cần F0 cho hai mục đích:

1. Thanh điệu của âm tiết.
2. Ngữ điệu của cả câu.

Nếu chỉ dự đoán một đường F0 chung, ngữ điệu câu có thể làm thanh điệu bị méo.

## 10.2 Mô hình đề xuất

```text
F0 cuối cùng = tone prior + residual intonation
```

- `tone prior`: thông tin cơ bản của sáu thanh.
- `residual intonation`: điều chỉnh theo câu hỏi, dấu phẩy, trọng âm và vị trí trong câu.

Không hard-code thanh sắc luôn tăng theo một đường cố định. Mỗi người nói và môi trường ngữ âm có biến thiên. Tone prior chỉ là gợi ý cho neural network.

## 10.3 Đầu vào cho F0 predictor

- Hidden state từ Text Encoder.
- Tone token.
- Vị trí âm tiết trong từ và câu.
- Dấu câu.
- Ranh giới cụm.
- Duration dự đoán.

## 10.4 Đầu ra

- F0 theo frame.
- Voiced/unvoiced mask.
- Có thể thêm F0 trung bình theo âm tiết.

## 10.5 Loss phụ cho thanh điệu

Thêm một tone classifier phụ:

```text
acoustic hidden state → dự đoán lại thanh điệu
```

Nếu hidden state của âm tiết `má` bị nhận thành thanh huyền, mô hình bị phạt.

Loss phụ này không chạy khi inference, nên không làm model production chậm hơn.

---

# 11. Thành phần 7: Dự đoán năng lượng

Energy mô tả độ mạnh tương đối theo thời gian.

Energy predictor giúp:

- Phụ âm rõ hơn.
- Nhịp câu bớt phẳng.
- Dấu câu có chuyển tiếp tự nhiên.
- Tránh mọi âm tiết có cùng độ lớn.

Phiên bản đầu chỉ cần predictor nhỏ:

```yaml
energy_predictor:
  layers: 2
  channels: 256
  kernel_size: 3
  dropout: 0.1
```

Không cần emotion embedding.

---

# 12. Thành phần 8: Length Regulator và Acoustic Decoder

## 12.1 Length Regulator

Text Encoder tạo một vector cho mỗi token. Audio lại có hàng trăm frame. Length Regulator lặp vector theo duration.

Ví dụ:

```text
Token A, duration 3 → A A A
Token B, duration 5 → B B B B B
```

Sau bước này, chuỗi hidden state có độ dài bằng số frame acoustic.

## 12.2 Acoustic Decoder

Acoustic Decoder nhận:

```text
expanded text features
+ F0 embedding
+ energy embedding
+ frame position
```

và tạo mel-spectrogram.

Cấu hình đề xuất:

```yaml
acoustic_decoder:
  type: conformer
  layers: 4
  hidden_size: 256
  attention_heads: 4
  ffn_size: 1024
  conv_kernel_size: 15
  dropout: 0.1
  mel_bins: 100
```

## 12.3 Vì sao dùng mel trước?

Có thể dự đoán latent trực tiếp để nhanh hơn, nhưng mel có lợi thế:

- Dễ quan sát bằng hình ảnh.
- Dễ so sánh với audio thật.
- Có nhiều loss chuẩn.
- Có thể tách lỗi acoustic model và vocoder.
- Có thể thử nhiều vocoder mà không train lại cả frontend.

Sau khi pipeline ổn định mới thử compact latent.

---

# 13. Thành phần 9: iSTFTNet tạo waveform

## 13.1 Nhiệm vụ

Mel-spectrogram chưa phải âm thanh. iSTFTNet biến biểu diễn acoustic thành waveform.

```text
mel → đặc trưng magnitude/phase → inverse STFT → waveform
```

## 13.2 Vì sao chọn iSTFTNet?

Kokoro sử dụng iSTFTNet, đồng thời bỏ style diffusion trong inference. Chủ dự án Kokoro cho biết iSTFTNet có thể nhanh hơn HiFi-GAN khoảng 1,5 đến 2 lần trong bối cảnh so sánh được thảo luận [3].

Lợi ích kỳ vọng:

- Nhanh trên CPU.
- Giảm convolution upsampling nặng.
- Phù hợp ONNX.
- Có thể gộp vào một graph inference.

## 13.3 Baseline bắt buộc

Dù lựa chọn cuối là iSTFTNet, vẫn nên train một HiFi-GAN nhỏ làm baseline.

Nếu mel tốt nhưng audio iSTFTNet xấu, ta biết lỗi nằm ở vocoder.

Nếu cả hai vocoder đều xấu, lỗi nhiều khả năng nằm ở mel hoặc dữ liệu.

## 13.4 Sample rate

Dùng 24 kHz trong phiên bản đầu.

Thông số gợi ý:

```yaml
audio:
  sample_rate: 24000
  n_fft: 1024
  hop_length: 256
  win_length: 1024
  mel_bins: 100
  f_min: 0
  f_max: 12000
```

24 kHz đủ cho giọng nói và giảm chi phí đáng kể so với 48 kHz.

---

# 14. Pipeline dữ liệu từ video YouTube và SRT

## 14.1 Nguyên tắc

SRT là gợi ý timeline, không phải ground truth hoàn hảo.

Một caption SRT có thể:

- Bắt đầu trước khi nói.
- Kết thúc sau khi nói.
- Chứa văn bản đã biên tập khác lời thật.
- Chứa hai câu trong một đoạn.
- Bị thiếu từ đệm.

Do đó không cắt audio theo SRT rồi train ngay.

## 14.2 Pipeline đầy đủ

```text
Video gốc
  ↓
Trích audio WAV
  ↓
Đọc SRT
  ↓
Tạo các vùng audio sơ bộ
  ↓
Voice Activity Detection
  ↓
ASR có word timestamps
  ↓
So sánh ASR với SRT
  ↓
Forced alignment
  ↓
Cắt câu 2 đến 12 giây
  ↓
Lọc chất lượng
  ↓
Chuẩn hóa định dạng audio
  ↓
Kiểm tra thủ công một phần
  ↓
Train/validation/test split
```

## 14.3 Trích audio

Ưu tiên video gốc trước khi upload. Nếu không còn, lấy audio chất lượng cao nhất từ video của chính bạn.

Đầu ra trung gian:

```text
48 kHz hoặc 44,1 kHz
mono
WAV float32 hoặc PCM 16-bit
```

Chỉ resample sang 24 kHz một lần trong pipeline chuẩn.

## 14.4 Cắt clip

Mục tiêu:

- Tối thiểu 1,5 giây.
- Tốt nhất 3 đến 10 giây.
- Tối đa 12 đến 15 giây.
- Một người nói.
- Không mất phụ âm đầu.
- Không mất âm cuối.
- Đầu clip giữ 50 đến 150 ms im lặng.
- Cuối clip giữ 100 đến 300 ms im lặng.

## 14.5 So sánh SRT và ASR

Tính CER hoặc edit distance giữa:

```text
SRT đã chuẩn hóa
ASR transcript đã chuẩn hóa
```

Phân loại:

```text
CER <= 0.03  → có thể tự động chấp nhận
0.03 < CER <= 0.10 → đưa vào hàng kiểm tra
CER > 0.10   → loại hoặc sửa thủ công
```

Ngưỡng cần điều chỉnh theo chất lượng ASR thực tế.

## 14.6 Lọc audio

Loại clip nếu:

- Có tiếng người khác.
- Có nhạc, tiếng chuông hoặc hiệu ứng lớn.
- Bị clipping.
- Âm lượng quá nhỏ.
- Nhiễu nền quá mạnh.
- Reverb quá khác số đông.
- ASR confidence thấp.
- Transcript không khớp.
- Có tiếng ho, hắng giọng hoặc cười che mất từ.

Không khử nhiễu mạnh toàn bộ. Khử mạnh có thể làm mất âm gió và chi tiết giọng.

## 14.7 Metadata

Mỗi mẫu nên có:

```json
{
  "id": "video_003_00125",
  "audio_path": "wavs/video_003_00125.wav",
  "text_raw": "Hôm nay mình thử chiếc máy mới.",
  "text_normalized": "Hôm nay mình thử chiếc máy mới.",
  "duration_sec": 3.84,
  "source_video": "video_003",
  "start_sec": 125.20,
  "end_sec": 129.04,
  "asr_cer": 0.0,
  "alignment_confidence": 0.97,
  "snr_db": 27.4,
  "status": "accepted"
}
```

---

# 15. Thiết kế tập dữ liệu

## 15.1 Số giờ mục tiêu

- 5 giờ sạch: prototype.
- 10 đến 20 giờ: baseline tốt.
- 30 đến 50 giờ: mục tiêu phù hợp cho model chính.
- 50 đến 100 giờ: có ích nếu chất lượng và cách thu nhất quán.

Với một người nói, chất lượng transcript và độ bao phủ âm tiết quan trọng hơn chỉ tăng số giờ.

## 15.2 Chia tập theo video

Không chia ngẫu nhiên từng clip.

Nên chia theo video nguồn:

```text
80% video → train
10% video → validation
10% video → test
```

Lý do: các clip cùng video có cùng micro, phòng và compression. Nếu một video xuất hiện trong cả train và test, kết quả test có thể tốt giả tạo.

## 15.3 Bộ test đặc biệt

Tạo riêng các nhóm:

- Sáu thanh.
- Tất cả vần.
- Cặp tr/ch.
- Cặp s/x.
- Cặp d/gi/r.
- Hỏi/ngã.
- Số nguyên.
- Số thập phân.
- Ngày tháng.
- Giờ.
- Đơn vị.
- Viết tắt.
- Tên riêng.
- Câu hỏi.
- Câu dài.

## 15.4 Thu bổ sung có chủ đích

Video tự nhiên thường không phủ đủ các cấu trúc hiếm. Nên thu thêm 2 đến 5 giờ script được thiết kế riêng.

Bộ script cần ưu tiên:

- Mọi onset và rime.
- Mọi thanh trên nhiều rime khác nhau.
- Số và ký hiệu.
- Tên người, địa danh.
- Câu có dấu phẩy, hai chấm, ngoặc.
- Câu dài 20 đến 40 âm tiết.

---

# 16. Quy trình huấn luyện từng giai đoạn

Không nên bật toàn bộ hệ thống rồi train một lần. Cần tách giai đoạn để biết lỗi nằm ở đâu.

## Giai đoạn 0: Kiểm tra dữ liệu

- Nghe ngẫu nhiên 500 clip.
- Kiểm tra transcript.
- Vẽ phân bố độ dài.
- Vẽ phân bố âm lượng.
- Kiểm tra số lần xuất hiện của từng onset, rime và tone.
- Tìm token chưa có trong vocabulary.

Điều kiện qua giai đoạn:

- Không có token không xác định trong train.
- Tỷ lệ transcript sai trong mẫu nghe thủ công đủ thấp.
- Không có clip quá dài hoặc quá ngắn bất thường.

## Giai đoạn 1: Train vocoder độc lập

```text
Mel thật → iSTFTNet → waveform
```

Mục tiêu là kiểm tra vocoder có thể tái tạo audio sạch từ mel thật.

Nếu vocoder chưa tốt, không chuyển sang mel dự đoán.

Song song, train HiFi-GAN baseline.

## Giai đoạn 2: Train acoustic model với ground-truth features

```text
Text tokens
+ ground-truth duration
+ ground-truth F0
+ ground-truth energy
→ mel
```

Mục tiêu:

- Xác minh tokenizer.
- Xác minh Text Encoder.
- Xác minh Acoustic Decoder.

Nếu mel dự đoán tốt khi dùng ground-truth features, kiến trúc acoustic cơ bản hoạt động.

## Giai đoạn 3: Train predictors

Train riêng hoặc cùng acoustic model:

- Duration predictor.
- F0 predictor.
- Voiced/unvoiced predictor.
- Energy predictor.

Đánh giá từng predictor bằng metric riêng.

## Giai đoạn 4: Inference hoàn toàn bằng giá trị dự đoán

```text
Text
→ predicted duration
→ predicted F0
→ predicted energy
→ mel
→ waveform
```

So sánh với giai đoạn dùng ground-truth features. Khoảng chênh lệch chỉ ra predictor nào yếu.

## Giai đoạn 5: Joint fine-tuning

Joint fine-tune acoustic model và iSTFTNet để giảm mismatch giữa mel dự đoán và mel thật. JETS cho thấy lợi ích của việc huấn luyện chung acoustic generator và vocoder cùng alignment module [2].

Không nên joint train từ ngày đầu vì rất khó debug.

## Giai đoạn 6: Distillation và deployment optimization

- Export FP32 ONNX.
- So sánh PyTorch và ONNX.
- Tạo FP16.
- Quantize INT8 từng phần.
- Benchmark trên CPU, Mac, NVIDIA và ARM.
- Distill sang model nhỏ nếu cần.

---

# 17. Loss function và mục đích từng loss

Tổng loss có thể là:

```text
L_total =
    λ_mel       × L_mel
  + λ_duration  × L_duration
  + λ_f0        × L_f0
  + λ_vuv       × L_voiced_unvoiced
  + λ_energy    × L_energy
  + λ_tone      × L_tone_classification
  + λ_align     × L_alignment
  + λ_adv       × L_adversarial
  + λ_fm        × L_feature_matching
  + λ_stft      × L_multi_resolution_stft
```

## 17.1 Mel loss

So sánh mel dự đoán và mel thật.

Có thể dùng:

- L1 loss.
- SSIM-like loss tùy chọn.
- Multi-scale mel loss.

## 17.2 Duration loss

Dự đoán log-duration thường ổn định hơn duration thô:

```text
L_duration = MSE(log(1 + d_pred), log(1 + d_true))
```

## 17.3 F0 loss

- L1 trên log-F0 ở frame voiced.
- Delta-F0 loss để giữ hình dạng đường cong.
- Syllable contour loss để giữ thanh điệu.

## 17.4 Voiced/unvoiced loss

Binary cross entropy cho việc frame có giọng hay không.

## 17.5 Tone classification loss

Cross entropy cho sáu thanh. Đây là loss phụ chuyên biệt quan trọng cho tiếng Việt.

## 17.6 Vocoder losses

- Adversarial loss.
- Feature matching loss.
- Multi-resolution STFT loss.
- Mel reconstruction loss.

Không nên đặt adversarial weight quá lớn từ đầu vì có thể tạo tiếng sắc nhưng phát âm kém ổn định.

---

# 18. Đánh giá chất lượng

## 18.1 Không chỉ dùng một chỉ số

TTS có thể nghe hay nhưng đọc sai. Hoặc đọc đúng nhưng giọng máy. Cần nhiều nhóm metric.

## 18.2 Độ chính xác nội dung

### ASR round-trip CER

```text
văn bản → TTS → audio → ASR → transcript
```

So sánh transcript ASR với văn bản chuẩn hóa.

### Syllable Error Rate

Phù hợp tiếng Việt hơn word error trong một số trường hợp.

### Tone Error Rate

Dùng classifier hoặc người nghe đánh giá thanh điệu.

### Missing Token Rate

Tỷ lệ token phát âm có duration bằng 0 hoặc không xuất hiện trong alignment đầu ra.

## 18.3 Độ tự nhiên

### MOS

Người nghe chấm 1 đến 5.

### CMOS

Người nghe so sánh hai model A và B.

### MUSHRA nhỏ

Có thể dùng nếu so nhiều vocoder hoặc nhiều checkpoint.

## 18.4 Độ giống người nói

Dù chỉ có một người, vẫn nên đo:

- Speaker embedding cosine similarity.
- Người quen với giọng đánh giá.
- So sánh các câu chưa xuất hiện trong train.

## 18.5 Tốc độ

Đo:

```text
RTF = thời gian sinh / thời lượng audio
```

Cần benchmark:

- Câu 3 giây.
- Câu 10 giây.
- Đoạn 30 giây.
- Batch 1, 4, 8, 16.
- Cold start.
- Warm start.
- Peak RAM.

## 18.6 Test theo nền tảng

- Windows CPU.
- Windows NVIDIA.
- macOS Apple Silicon.
- Linux x86_64.
- Linux ARM64.
- Raspberry Pi 5 nếu là mục tiêu.

---

# 19. Ngăn đọc sai, nuốt chữ, lặp chữ và thêm chữ

## 19.1 Lớp 1: Frontend xác định

Mọi số và ký hiệu được chuyển thành chữ trước khi neural model nhìn thấy.

## 19.2 Lớp 2: Tokenizer kiểm tra được

- Không có unknown token im lặng.
- Round-trip pass.
- Mỗi âm tiết hợp lệ có đúng một tone.
- Mỗi âm tiết có rime.

## 19.3 Lớp 3: Duration validation

- Không token phát âm nào có duration bằng 0.
- Duration không vượt giới hạn.
- Tổng duration hợp lý.
- Pause không chiếm phần lớn câu.

## 19.4 Lớp 4: Không dùng attention tự hồi quy

Kiến trúc non-autoregressive loại bỏ một nguồn lỗi lặp và dừng sớm thường gặp ở Tacotron-like models.

## 19.5 Lớp 5: ASR kiểm tra đầu ra

Đối với batch tạo audio dài:

```text
TTS output
  ↓
Vietnamese ASR
  ↓
CER check
```

Nếu CER vượt ngưỡng:

- Tạo lại với tốc độ bình thường.
- Chia câu ngắn hơn.
- Đưa vào hàng kiểm tra.
- Không tự động xuất bản.

## 19.6 Lớp 6: Từ điển override

Cho phép người dùng thêm:

```yaml
pronunciations:
  "HTMX": "hát tê em ích"
  "ONNX": "ô en en ích"
  "Mac Mini": "mác mi ni"
```

Từ điển này nằm trước tokenizer.

---

# 20. Tối ưu tốc độ và triển khai đa nền tảng

## 20.1 Định dạng chuẩn

Dùng ONNX làm format triển khai chính.

Các file phát hành:

```text
vifasttts-fp32.onnx
vifasttts-fp16.onnx
vifasttts-int8.onnx
vocab.json
normalizer_rules.json
model_config.json
pronunciation_dictionary.json
```

Có thể tách frontend khỏi graph neural:

```text
Frontend Rust/C++
→ token IDs
→ ONNX Runtime
→ PCM audio
```

## 20.2 Windows CPU

- ONNX Runtime CPU.
- FP32 làm bản tham chiếu.
- INT8 cho CPU có hỗ trợ phù hợp.
- Điều chỉnh số thread.

## 20.3 Windows NVIDIA

- ONNX Runtime CUDA.
- FP16.
- TensorRT Execution Provider nếu graph hỗ trợ.
- Batch inference cho video dài.

## 20.4 macOS Apple Silicon

- ONNX Runtime ARM64 để đồng nhất.
- Có thể xuất thêm Core ML.
- FP16 hoặc mixed precision.
- Benchmark CPU và Core ML, không mặc định Core ML luôn nhanh hơn.

## 20.5 Linux ARM64

- ONNX Runtime ARM64.
- NEON.
- INT8 hoặc mixed precision.
- Build runtime tối giản chỉ với operator cần dùng.

## 20.6 Phần nên quantize trước

- Token embedding.
- Text Encoder linear layers.
- Duration predictor.
- Energy predictor.

## 20.7 Phần cần thận trọng khi quantize

- F0 predictor.
- Layer norm.
- Final mel projection.
- iSTFTNet.

Lượng tử hóa waveform decoder quá mạnh có thể tạo noise và làm âm sắc xấu.

## 20.8 Streaming

Kiến trúc này sinh song song theo câu. Streaming phiên bản đầu có thể làm ở cấp đoạn:

1. Chia văn bản thành câu hoặc mệnh đề.
2. Sinh câu đầu.
3. Phát câu đầu trong khi sinh câu tiếp.
4. Ghép bằng crossfade nhỏ.

Frame-level streaming có thể làm sau, nhưng không cần cho batch lồng tiếng video.

---

# 21. Cấu hình mô hình đề xuất

## 21.1 Bản Base

```yaml
model_name: ViFastTTS-Base
language: vi
accent: north
speakers: 1
sample_rate: 24000

frontend:
  tokenizer: onset_rime_tone
  strict_normalization: true
  preserve_orthographic_contrasts: true

mel:
  n_fft: 1024
  win_length: 1024
  hop_length: 256
  bins: 100
  f_min: 0
  f_max: 12000

text_encoder:
  type: conformer
  layers: 6
  hidden_size: 256
  attention_heads: 4
  ffn_size: 1024
  conv_kernel_size: 15
  dropout: 0.1

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
  conv_kernel_size: 15

waveform_decoder:
  type: istftnet

inference:
  deterministic: true
  default_speed: 1.0
  min_speed: 0.75
  max_speed: 1.35
```

Mục tiêu quy mô:

- 40 đến 60 triệu tham số.
- FP32 khoảng 160 đến 240 MB.
- FP16 khoảng 80 đến 120 MB.
- INT8 khoảng 45 đến 75 MB, tùy graph và metadata.

## 21.2 Bản Mobile/Nano sau này

```yaml
text_encoder_layers: 4
hidden_size: 192
attention_heads: 4
acoustic_decoder_layers: 3
ffn_size: 768
```

Mục tiêu:

- 18 đến 30 triệu tham số.
- INT8 khoảng 25 đến 40 MB.
- Chạy tốt hơn trên ARM.

Chỉ distill sang Nano sau khi Base đạt chất lượng.

---

# 22. Cấu trúc dự án

```text
vifasttts/
├── README.md
├── pyproject.toml
├── configs/
│   ├── base_24k.yaml
│   ├── nano_24k.yaml
│   └── dataset.yaml
├── frontend/
│   ├── normalizer/
│   │   ├── unicode.py
│   │   ├── sentence_splitter.py
│   │   ├── lexer.py
│   │   ├── number_parser.py
│   │   ├── date_time_parser.py
│   │   ├── unit_parser.py
│   │   └── abbreviations.py
│   ├── syllable_parser/
│   │   ├── parser.py
│   │   ├── onsets.py
│   │   ├── rimes.py
│   │   └── tones.py
│   ├── tokenizer.py
│   └── vocab.json
├── data/
│   ├── import_srt.py
│   ├── extract_audio.py
│   ├── vad_segment.py
│   ├── asr_align.py
│   ├── quality_filter.py
│   ├── build_manifest.py
│   └── split_by_video.py
├── models/
│   ├── conformer.py
│   ├── text_encoder.py
│   ├── alignment.py
│   ├── duration_predictor.py
│   ├── f0_predictor.py
│   ├── energy_predictor.py
│   ├── length_regulator.py
│   ├── acoustic_decoder.py
│   ├── istftnet.py
│   ├── discriminators.py
│   └── vifasttts.py
├── losses/
│   ├── acoustic.py
│   ├── duration.py
│   ├── pitch.py
│   ├── tone.py
│   ├── stft.py
│   └── adversarial.py
├── train/
│   ├── train_vocoder.py
│   ├── train_acoustic.py
│   ├── train_predictors.py
│   ├── joint_finetune.py
│   └── distill.py
├── inference/
│   ├── synthesize.py
│   ├── batch.py
│   ├── verify_asr.py
│   └── audio_join.py
├── export/
│   ├── export_onnx.py
│   ├── quantize_int8.py
│   ├── export_coreml.py
│   └── validate_export.py
├── evaluation/
│   ├── cer.py
│   ├── tone_error.py
│   ├── duration_checks.py
│   ├── benchmark.py
│   └── listening_test/
└── tests/
    ├── test_normalizer.py
    ├── test_syllable_parser.py
    ├── test_tokenizer_roundtrip.py
    ├── test_duration_constraints.py
    ├── test_onnx_equivalence.py
    └── fixtures/
```

Nên tách frontend thành package độc lập. Sau này có thể viết lại frontend bằng Rust hoặc C++ mà không ảnh hưởng code training.

---

# 23. Lộ trình thực hiện

## Giai đoạn A: Frontend, 4 đến 8 tuần

### Tuần 1 đến 2

- Định nghĩa cách đọc chuẩn miền Bắc.
- Tạo lexer.
- Chuẩn hóa Unicode và câu.
- Xử lý số cơ bản.

### Tuần 3 đến 4

- Date, time, unit, percent, currency.
- Viết tắt và từ điển override.
- Unit test.

### Tuần 5 đến 6

- Syllable parser.
- Danh sách onset, rime, tone.
- Tokenizer.
- Round-trip test.

### Tuần 7 đến 8

- Corpus fuzz test.
- Test hàng triệu câu văn bản nếu có.
- Sửa trường hợp ngoại lệ.

## Giai đoạn B: Dataset, 4 đến 10 tuần

- Trích audio và SRT.
- VAD.
- ASR alignment.
- Lọc tự động.
- Kiểm tra thủ công.
- Tạo 5 giờ sạch đầu tiên.
- Sau đó mở rộng lên 20 đến 50 giờ.

## Giai đoạn C: Baseline, 4 đến 6 tuần

- Train VITS/Piper hoặc FastSpeech2 baseline.
- Xác nhận dữ liệu có thể tạo TTS.
- Không tối ưu kiến trúc quá sớm.

## Giai đoạn D: ViFastTTS Base, 8 đến 16 tuần

- Train iSTFTNet.
- Train acoustic model.
- Train predictors.
- Full inference.
- Joint fine-tune.

## Giai đoạn E: Đánh giá và sửa lỗi, 4 đến 8 tuần

- Bộ test phát âm.
- ASR round-trip.
- Listening test.
- Tìm lỗi theo token/vần/thanh.
- Bổ sung dữ liệu khó.

## Giai đoạn F: Deployment, 4 đến 8 tuần

- ONNX FP32.
- FP16.
- INT8 từng phần.
- Benchmark đa nền tảng.
- Runtime C++ hoặc Rust.
- API batch.

Tổng thời gian cho một người làm bán thời gian có thể từ 6 đến 12 tháng. Một nhóm nhỏ có thể song song hóa frontend, dữ liệu và model.

---

# 24. Rủi ro và cách xử lý

## 24.1 Transcript không khớp audio

**Biểu hiện:** nuốt chữ, phát âm sai, alignment hỗn loạn.

**Xử lý:** ASR round-trip, forced alignment, kiểm tra thủ công, loại clip nghi ngờ.

## 24.2 Audio từ nhiều thời kỳ khác nhau

**Biểu hiện:** giọng lúc gần, lúc xa, âm sắc thay đổi.

**Xử lý:** gom video theo profile, ưu tiên profile chính, thêm channel/recording condition trong training nếu cần, hoặc loại profile hiếm.

## 24.3 Frontend quá tham vọng

**Biểu hiện:** nhiều luật chồng nhau, sửa một lỗi gây lỗi khác.

**Xử lý:** lexer/parser theo loại token, unit test, strict mode, version hóa rules.

## 24.4 Thanh điệu đúng nhưng ngữ điệu máy

**Xử lý:** cải thiện residual F0 predictor, dấu câu, phrase boundary, thêm dữ liệu câu dài và câu hỏi.

## 24.5 Audio sạch nhưng vẫn rè

**Xử lý:** kiểm tra vocoder bằng mel thật, kiểm tra sample rate, STFT config, discriminator và clipping.

## 24.6 ONNX chậm hơn PyTorch

**Xử lý:** profile operator, tránh dynamic shape quá mức, fuse graph, giới hạn chiều dài, dùng execution provider phù hợp.

## 24.7 INT8 làm hỏng giọng

**Xử lý:** quantize từng module, giữ F0 và vocoder ở FP16/FP32, calibration bằng câu đa dạng.

## 24.8 Thiếu âm tiết hiếm

**Xử lý:** phân tích coverage, sinh script bổ sung, thu thêm có chủ đích.

---

# 25. Tiêu chí hoàn thành

Một bản Base có thể coi là đạt khi:

## Chức năng

- Normalizer pass toàn bộ test bắt buộc.
- Tokenizer không có unknown trên test chuẩn.
- Không có token phát âm duration bằng 0.
- Sinh được câu dài ít nhất 30 đến 50 âm tiết.
- Có batch inference.

## Độ chính xác

Mục tiêu ban đầu, cần hiệu chỉnh theo ASR:

- ASR round-trip CER dưới 2% trên văn bản thông thường sạch.
- Không có lỗi lặp hoặc nuốt chữ trong ít nhất 99% câu test chuẩn.
- Tone Error Rate đủ thấp theo người nghe bản xứ.
- Toàn bộ số, ngày và đơn vị trong test bắt buộc được đọc đúng.

## Chất lượng nghe

- MOS nội bộ ít nhất 4/5 trên tập văn bản thông thường.
- Không có noise hệ thống rõ ràng.
- Âm sắc ổn định giữa các câu.

## Tốc độ mục tiêu

Các con số sau là mục tiêu kỹ thuật, không phải benchmark đã được chứng minh:

- CPU desktop: RTF dưới 0,10.
- Apple Silicon: RTF dưới 0,07.
- RTX 3060: RTF dưới 0,02 sau warm-up.
- Raspberry Pi 5: RTF dưới 0,50 với bản phù hợp.

VieNeu-TTS v3 Turbo hiện công bố CPU RTF khoảng 0,55 đến 0,61 ở FP32 và khoảng 0,35 ở INT8 trên CPU 6 nhân; v3 Nano được mô tả ở khoảng 0,11 đến 0,22 trên CPU desktop [5][6]. Các mốc ViFastTTS phía trên là mục tiêu ước tính dựa trên việc mô hình chỉ có một giọng, một ngôn ngữ, 24 kHz và suy luận hoàn toàn non-autoregressive.

## Triển khai

- PyTorch và ONNX output gần tương đương.
- Có FP32 chính xác.
- Có FP16 cho GPU/Mac nếu phù hợp.
- Có bản INT8 không làm tăng lỗi phát âm đáng kể.
- Runtime không cần Python.

---

# 26. Thuật ngữ

## Acoustic model

Mô hình biến token văn bản thành biểu diễn âm thanh như mel-spectrogram.

## Alignment

Sự tương ứng giữa token văn bản và đoạn thời gian trong audio.

## Autoregressive

Sinh phần tử tiếp theo dựa trên các phần tử đã sinh trước đó.

## CER

Character Error Rate, tỷ lệ lỗi ký tự giữa hai transcript.

## Conformer

Kiến trúc kết hợp self-attention và convolution.

## Duration

Thời gian hoặc số frame của một token phát âm.

## F0

Tần số cơ bản, thể hiện cao độ của giọng.

## Frontend

Phần xử lý văn bản trước neural network, gồm normalizer, parser, G2P và tokenizer.

## G2P

Grapheme-to-Phoneme, chuyển chữ viết thành biểu diễn phát âm.

## iSTFT

Inverse Short-Time Fourier Transform, biến biểu diễn phổ trở lại waveform.

## Mel-spectrogram

Biểu diễn năng lượng âm thanh theo thời gian và tần số trên thang mel.

## Non-autoregressive

Tạo nhiều vị trí đầu ra song song thay vì lần lượt từng vị trí.

## ONNX

Định dạng mô hình trung gian để chạy trên nhiều runtime và nền tảng.

## Onset

Phụ âm đầu của âm tiết.

## Rime

Phần vần của âm tiết, gồm phần sau phụ âm đầu.

## RTF

Real-Time Factor. RTF 0,1 nghĩa là tạo 10 giây audio mất 1 giây.

## Tone

Thanh điệu. Tiếng Việt có sáu thanh trong hệ thống chữ viết chuẩn.

## Vocoder

Mô hình biến biểu diễn acoustic thành waveform.

## Waveform

Dãy số biểu diễn tín hiệu âm thanh theo thời gian.

---

# 27. Tài liệu tham khảo

1. FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. https://arxiv.org/abs/2006.04558
2. JETS: Jointly Training FastSpeech2 and HiFi-GAN for End-to-End Text-to-Speech. https://arxiv.org/abs/2203.16852
3. Trao đổi kiến trúc Kokoro, bỏ style diffusion và dùng iSTFTNet. https://huggingface.co/hexgrad/Kokoro-82M/discussions/111
4. VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. https://arxiv.org/abs/2106.06103
5. VieNeu SDK Overview, thông tin Turbo và Nano. https://docs.vieneu.io/docs/sdk/overview/
6. VieNeu Streaming Benchmarks. https://github.com/pnnbao97/VieNeu-TTS/blob/main/docs/streaming.md
7. VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech. https://arxiv.org/abs/2307.16430
8. Kokoro-82M model card. https://huggingface.co/hexgrad/Kokoro-82M
9. Piper training guide. https://tderflinger.github.io/piper-docs/guides/training/
10. VieNeu-TTS v3 Turbo model description. https://huggingface.co/ncnlam/VieNeu-TTS-v3-Turbo

---

# Phụ lục A: Thứ tự ưu tiên thực tế

Nếu nguồn lực hạn chế, hãy làm theo thứ tự:

1. Dataset sạch.
2. Normalizer đúng.
3. Tokenizer đúng.
4. Baseline FastSpeech2 hoặc VITS chạy được.
5. Duration ổn định.
6. F0 và thanh điệu.
7. Vocoder chất lượng.
8. Joint fine-tuning.
9. ONNX.
10. INT8 và ARM.

Không nên bắt đầu bằng tối ưu GPU hoặc viết iSTFTNet trước khi dataset và frontend đáng tin cậy.

# Phụ lục B: Bộ thử nghiệm tối thiểu

```text
Tôi có một trăm linh năm quyển sách.
Hôm nay là ngày mười lăm tháng mười.
Nhiệt độ ngoài trời là hai mươi lăm độ C.
Tỷ lệ tăng trưởng là ba phẩy năm phần trăm.
Trăm năm trong cõi người ta.
Chăm chỉ học tập sẽ mang lại kết quả tốt.
Sương xuống trên những mái nhà cũ.
Xương cá cần được loại bỏ cẩn thận.
Bạn đã sẵn sàng chưa?
Xin chào, rất vui được gặp các bạn.
```

Cần bổ sung hàng nghìn câu bao phủ onset, rime, tone và cấu trúc normalizer.

# Phụ lục C: Quyết định cuối cùng

Nếu phải tóm tắt toàn bộ tài liệu trong một câu:

> Hãy xây một hệ thống non-autoregressive chuyên tiếng Việt Bắc, dùng frontend có luật, tokenizer onset/rime/tone, Conformer nhỏ, duration xác định, F0 có điều kiện theo thanh điệu, acoustic decoder nhẹ và iSTFTNet 24 kHz, sau đó triển khai bằng ONNX.
