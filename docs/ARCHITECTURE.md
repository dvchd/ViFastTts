# Kiến trúc ViFastTts

ViFastTts dùng frontend tiếng Việt chung và checkpoint single-speaker, single-dialect. Tài liệu này mô tả **trạng thái as-built** trong `src/vifasttts`; **luồng chi tiết train → inference → export** nằm ở `docs/MODEL_FLOW.md`; trạng thái tổng hợp và nhật ký quyết định ở `docs/PROJECT_STATUS.md`; đề xuất nâng cấp dài hạn ở `docs/TTS_ARCHITECTURE_RESEARCH.md` (mục 9).

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
- **Vocoder**: `ISTFTNetLite` upsample mel ×4 trước khi dự đoán STFT. Đã có vòng train tham chiếu
  (`vifasttts train-vocoder`, MPD-lite + mel L1 + feature matching + adversarial — `docs/MODEL_FLOW.md` §5);
  bản production cần discriminator lớn hơn + spectral loss + dữ liệu nhiều.
- **Export ONNX**: exporter TorchScript, 8 output có tên (`mel`, `log_duration`, `durations`, `f0`,
  `vuv_logits`, `energy`, `tone_logits`, `lengths`) — 9 nếu bật `model.use_phonation`; dynamic axes cho
  batch/số âm tiết. Dùng `exp-1` thay `expm1` để export được. Hỗ trợ `--precision fp16|int8`.
  Kèm custom metadata (sample_rate...).
- **Runtime ONNX**: `runtime/` chạy acoustic + vocoder ONNX bằng onnxruntime (không cần
  PyTorch), ghép waveform bằng iSTFT numpy, có chia câu dài + fade/peak-normalize —
  `vifasttts synthesize-onnx`; chi tiết `docs/MODEL_FLOW.md` §6.4.

## Vì sao đầu vào neural là 5 thành phần âm tiết, không phải IPA

Trong tài liệu, tầng biểu diễn này gọi là **dạng âm vị học** (bộ âm vị 5 thành phần; xem `docs/GLOSSARY.md`). Quyết định thiết kế này được khóa bằng test trong `tests/test_component_faithfulness.py` và được khuyến nghị cho một-giọng/đơn ngữ (xem `docs/TTS_ARCHITECTURE_RESEARCH.md` mục 10).

**Đầu vào là 5 thành phần + boundary/punctuation**, không phải chuỗi ký tự thô hay IPA:
`ONSET + MEDIAL + NUCLEUS + CODA + TONE`, mỗi trường là một ID rời rạc trong vocab nhỏ (24/2/15/11/6).

- **Không cần G2P ở runtime.** Chính tả tiếng Việt gần như ngữ âm, nên parser đã cho biểu diễn phonemic-native trực tiếp từ văn bản; thêm bước G2P chỉ thêm một bề mặt lỗi (OOV, quy ước thanh điệu của từng hệ G2P) mà không thêm thông tin.
- **Tone là biến rời rạc 6 giá trị**, không phải ký hiệu phụ trên chuỗi IPA. Duration/F0/nhịp điệu điều kiện hoá trực tiếp trên tone; IPA phải mã hoá thanh bằng quy ước riêng (ví dụ sea-g2p mã hoá huyền/hỏi/ngã/nặng bằng 2/4/5/6 còn ngang/sắc dựa vào chất lượng nguyên âm) — dễ sai lệch khi so khớp.
- **Không mất thông tin chính tả.** Các cặp phân biệt quan trọng như `uy` vs `ui`, `d` vs `gi`, `tr` vs `ch`, `s` vs `x`, hỏi vs ngã vị trí dấu cũ/mới đều được khóa bằng test; IPA Bắc gộp `d/gi` thành `/z/` và làm mờ các khác biệt này.
- **Phủ từ bản ngữ không UNK**: vocab 24 × 2 × 15 × 11 × 6 ID; không gian âm tiết trừu tượng hợp lệ
  là **28.800** (coda tắc chỉ đi với sắc/nặng), tương ứng 6.336 shape (onset × vần) và 264 vần
  (15 nucleus gồm `OO` từ vay); từ vay/OOV dùng fallback + override thay vì phình vocab.
  Số liệu audit chi tiết: `docs/SYLLABLE_SYSTEMS_RESEARCH.md` mục 5.
- **Đổi lại, kiến trúc là đơn ngữ.** Nếu cần đa ngôn ngữ, hướng đúng là thêm language token + bộ token hoá theo ngôn ngữ (hoặc IPA có kiểm soát) như một lớp riêng, không đổi thân Conformer.

Lớp IPA trong `frontend/g2p.py` và các hệ ngoài (sea-g2p, viG2P, WikiPron, MFA) là **công cụ phân tích build-time**: audit lexicon, round-trip, gợi ý respelling, chuẩn bị alignment — xem `docs/G2P_VIETNAMESE.md`. Chiều ánh xạ là **5 thành phần → IPA** (broad/narrow theo Bắc/Trung/Nam, dựa trên Kirby 2011/Thompson 1965; xem `docs/FRONTEND.md`), không phải văn bản → IPA, nên không có G2P ngoài nào chen vào đường runtime.

## Frontend

- Parser giữ song song chính tả (`Orthography`) và category trừu tượng (`AbstractSyllable`).
- Tone là thuộc tính toàn âm tiết, không phải segment có duration.
- Token không parse được giữ nguyên thô với `ParseStatus.FALLBACK_REQUIRED` (không bao giờ crash, không nuốt chữ số).
- G2P tùy chọn (`frontend/g2p.py`) chuyển âm tiết đã parse sang chuỗi âm vị IPA gần đúng — dùng cho nghiên cứu/lexicon, không thay vocab ID.
- OOV resolver (`frontend/oov.py`) theo quy trình 5 bước viG2P, opt-in ở `parse_text(..., resolve_oov=True)`.

## Lộ trình (v2.1, xem khảo sát)

Các hạng mục thiết kế trong `docs/TTS_ARCHITECTURE_RESEARCH.md` và trạng thái as-built:

1. ✅ **Segment Expander** — `frontend/segments.py` (duration cấp role, tách `IÊ/UÔ/ƯƠ`).
2. ⬜ **Alignment thật** (MFA; forward-sum + Viterbi + static prior về sau) — hiện chia đều/trọng số.
3. ✅ **Phonation** — trích `data/phonation.py` + điều kiện hoá `model.use_phonation` (reference).
4. ✅ **Vocoder tham chiếu** — `train-vocoder` (MPD-lite); ⬜ bản production + A/B HiFi-GAN.
5. ✅ **ONNX + quantization** FP16/INT8 động; ⬜ ONNX vocoder + calibration đầy đủ.

Chi tiết luồng: `docs/MODEL_FLOW.md`; trạng thái đầy đủ: `docs/PROJECT_STATUS.md`.

## Mở rộng phương ngữ

Cùng parser được dùng cho mọi phương ngữ. Mỗi phương ngữ dùng một checkpoint riêng, không cần dialect ID; normalizer là nơi duy nhất biết `north/central/south`.
