# Kiến trúc ViFastTts

ViFastTts dùng frontend tiếng Việt chung và checkpoint single-speaker, single-dialect. Tài liệu này mô tả **trạng thái as-built** trong `src/vifasttts`; đề xuất nâng cấp dài hạn nằm ở `docs/TTS_ARCHITECTURE_RESEARCH.md` (mục 9).

## Sơ đồ as-built

```text
raw text
  → frontend/unicode (NFC, quotes)
  → frontend/stem (LaTeX, hoá học, đơn vị)          [tùy chọn stem_formulas]
  → frontend/normalizer (số/ngày/giờ/phân số/khoảng/tiền/%), dialect profile
  → frontend/pipeline (token, chấm câu, fallback; resolve_oov tùy chọn)
  → frontend/syllable (ONSET+MEDIAL+NUCLEUS+CODA+TONE, chính tả + trừu tượng)
  → frontend/vocab (7 ID: onset, medial, nucleus, coda, tone, boundary, punctuation)
  → models/vifasttts.ViFastTtsModel
       ComponentEmbedding (concat + Linear)
       ConformerLite encoder + sinusoidal positional encoding + padding mask
       Duration/F0/VUV/Energy predictors (ConvPredictor)
       LengthRegulator vector hoá (cumsum + gather, không vòng lặp Python)
       frame-position 0..1 trong âm tiết
       ConformerLite decoder + mel head 100 bins
  → models/istftnet.ISTFTNetLite (ConvTranspose ×4 rồi iSTFT) → PCM 24 kHz
```

## Điểm quan trọng

- **Input**: mỗi âm tiết là 7 ID rời rạc; không có `dialect_id`, `speaker_id`, `style_id`, không audio tham chiếu.
- **Deterministic**: không sampling, không flow; dropout tắt ở eval; duration làm tròn, mỗi âm tiết ≥ 1 frame.
- **Mask**: encoder nhận `input_lengths` (train); decoder nhận frame mask sinh ra từ LengthRegulator nên không attend vào vùng pad.
- **LengthRegulator**: biểu diễn bằng `cumsum` + so sánh + `gather`, export ONNX với **độ dài frame động** — đã kiểm chứng bằng onnxruntime cho các tổ hợp (B=1..3, N=3..12, tổng frame khác nhau).
- **Frame position**: mỗi frame biết vị trí 0..1 trong âm tiết chứa nó (hỗ trợ ranh giới đầu/giữa/cuối nguyên âm, mục 9.8 của khảo sát).
- **Vocoder**: `ISTFTNetLite` upsample mel ×4 trước khi dự đoán STFT; đây vẫn là bản tham chiếu (chưa có discriminator/spectral loss trong huấn luyện).
- **Export ONNX**: exporter TorchScript, 8 output có tên (`mel`, `log_duration`, `durations`, `f0`, `vuv_logits`, `energy`, `tone_logits`, `lengths`), dynamic axes cho batch/số âm tiết. Dùng `exp-1` thay `expm1` để export được. Vocoder chưa export trong graph (iSTFT không có op chuẩn); chiến lược đề xuất ở mục 9.12 của khảo sát (tự cài iSTFT bằng ConvTranspose hoặc xuất phổ cho runtime).

## Vì sao đầu vào neural là 5 thành phần âm tiết, không phải IPA

Trong tài liệu, tầng biểu diễn này gọi là **dạng âm vị học** (bộ âm vị 5 thành phần; xem `docs/GLOSSARY.md`). Quyết định thiết kế này được khóa bằng test trong `tests/test_component_faithfulness.py` và được khuyến nghị cho một-giọng/đơn ngữ (xem `docs/TTS_ARCHITECTURE_RESEARCH.md` mục 10).

**Đầu vào là 5 thành phần + boundary/punctuation**, không phải chuỗi ký tự thô hay IPA:
`ONSET + MEDIAL + NUCLEUS + CODA + TONE`, mỗi trường là một ID rời rạc trong vocab nhỏ (24/2/14/11/6).

- **Không cần G2P ở runtime.** Chính tả tiếng Việt gần như ngữ âm, nên parser đã cho biểu diễn phonemic-native trực tiếp từ văn bản; thêm bước G2P chỉ thêm một bề mặt lỗi (OOV, quy ước thanh điệu của từng hệ G2P) mà không thêm thông tin.
- **Tone là biến rời rạc 6 giá trị**, không phải ký hiệu phụ trên chuỗi IPA. Duration/F0/nhịp điệu điều kiện hoá trực tiếp trên tone; IPA phải mã hoá thanh bằng quy ước riêng (ví dụ sea-g2p mã hoá huyền/hỏi/ngã/nặng bằng 2/4/5/6 còn ngang/sắc dựa vào chất lượng nguyên âm) — dễ sai lệch khi so khớp.
- **Không mất thông tin chính tả.** Các cặp phân biệt quan trọng như `uy` vs `ui`, `d` vs `gi`, `tr` vs `ch`, `s` vs `x`, hỏi vs ngã vị trí dấu cũ/mới đều được khóa bằng test; IPA Bắc gộp `d/gi` thành `/z/` và làm mờ các khác biệt này.
- **Phủ từ bản ngữ không UNK**: 24 × 2 × 14 × 11 × 6 = 44.352 tổ hợp phủ toàn bộ âm tiết tiếng Việt; từ vay/OOV dùng fallback + override thay vì phình vocab.
- **Đổi lại, kiến trúc là đơn ngữ.** Nếu cần đa ngôn ngữ, hướng đúng là thêm language token + bộ token hoá theo ngôn ngữ (hoặc IPA có kiểm soát) như một lớp riêng, không đổi thân Conformer.

Lớp IPA trong `frontend/g2p.py` và các hệ ngoài (sea-g2p, viG2P, WikiPron, MFA) là **công cụ phân tích build-time**: audit lexicon, round-trip, gợi ý respelling, chuẩn bị alignment — xem `docs/G2P_VIETNAMESE.md`. Chiều ánh xạ là **5 thành phần → IPA** (broad/narrow theo Bắc/Trung/Nam, dựa trên Kirby 2011/Thompson 1965; xem `docs/FRONTEND.md`), không phải văn bản → IPA, nên không có G2P ngoài nào chen vào đường runtime.

## Frontend

- Parser giữ song song chính tả (`Orthography`) và category trừu tượng (`AbstractSyllable`).
- Tone là thuộc tính toàn âm tiết, không phải segment có duration.
- Token không parse được giữ nguyên thô với `ParseStatus.FALLBACK_REQUIRED` (không bao giờ crash, không nuốt chữ số).
- G2P tùy chọn (`frontend/g2p.py`) chuyển âm tiết đã parse sang chuỗi âm vị IPA gần đúng — dùng cho nghiên cứu/lexicon, không thay vocab ID.
- OOV resolver (`frontend/oov.py`) theo quy trình 5 bước viG2P, opt-in ở `parse_text(..., resolve_oov=True)`.

## Lộ trình (v2.1, xem khảo sát)

Các hạng mục dưới đây **chưa** có trong code và được đánh dấu thứ tự ưu tiên trong `docs/TTS_ARCHITECTURE_RESEARCH.md`:

1. **Segment Expander**: duration cấp role (onset/medial/nucleus/coda) thay cho duration cấp âm tiết, cần alignment thật (MFA) — mục 9.4.
2. **Alignment**: MFA giai đoạn đầu; forward-sum + Viterbi + static prior về sau — mục 9.4.
3. **Phonation**: các kênh đo được (spectral tilt H1–H2, HNR, glottalization) — mục 9.7.
4. **Vocoder production**: discriminator + spectral loss, HiFi-GAN baseline A/B — mục 9.9 và `docs/TRAINING.md` giai đoạn 1.
5. **ONNX vocoder + quantization từng phần** — mục 9.12.

## Mở rộng phương ngữ

Cùng parser được dùng cho mọi phương ngữ. Mỗi phương ngữ dùng một checkpoint riêng, không cần dialect ID; normalizer là nơi duy nhất biết `north/central/south`.
