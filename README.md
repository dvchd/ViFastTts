# ViFastTts — Text-to-Speech tiếng Việt

Toolkit nghiên cứu và triển khai **TTS tiếng Việt một giọng**: hiểu văn bản tiếng Việt,
sinh giọng nói, huấn luyện và xuất ONNX để chạy nhanh trên CPU/GPU. Mỗi checkpoint là
một người nói, một phương ngữ — không cần `speaker_id`, `dialect_id` hay audio mẫu.

> **Đọc nhanh:** Muốn biết toàn cảnh trong 1 trang → mục 2–3 bên dưới.
> Muốn hiểu sâu từng bước → `docs/MODEL_FLOW.md` (model) và `docs/SYLLABLE_GUIDE.md` (âm tiết).

## 1. Dự án làm gì?

Ba khối chính, ghép lại thành một câu TTS hoàn chỉnh:

```text
Văn bản  ──▶  FRONTEND (tiếng Việt)  ──▶  MODEL (mel + thời lượng)  ──▶  VOCODER  ──▶  WAV
```

| Khối | Nhiệm vụ | Điểm mạnh |
| --- | --- | --- |
| **Frontend** | Chuẩn hoá số/ngày/giờ/đơn vị, tách âm tiết, chuyển thành **7 ID/âm tiết** | Phủ toàn bộ tiếng Việt chuẩn, không UNK; 3 vùng miền Bắc/Trung/Nam |
| **Model acoustic** | Từ 7 ID → dự đoán mel + thời lượng + f0/vuv/energy | Non-autoregressive, chạy tất định, xuất ONNX được |
| **Vocoder** | Từ mel → sóng 24 kHz | iSTFTNet nhẹ, có vòng huấn luyện GAN tham chiếu |

**Khác biệt cốt lõi:** thay vì đưa chuỗi ký tự hay BPE vào model, dự án mô tả mỗi âm tiết
bằng 5 thành phần ngôn ngữ học `âm đầu + âm đệm + âm chính + âm cuối + thanh điệu`
(cộng dấu câu/chấm câu — xem mục 7). Nhờ vậy: phát âm ổn định, dấu thanh không bao giờ sai
vị trí, và không có "từ lạ" với tiếng Việt chuẩn.

## 2. Trạng thái hiện tại (số liệu thật)

| Hạng mục | Kết quả |
| --- | --- |
| Bộ test | **2223 test pass**; `ruff` + `mypy` sạch |
| Parser âm tiết | Vét cạn **49.166 tổ hợp = 0 lỗi**; 39K từ thật: 6.617/6.633 parse OK |
| Kho âm tiết | 24 âm đầu · 2 âm đệm · 15 âm chính · 11 âm cuối · 6 thanh → **264 vần**, **28.800 âm tiết trừu tượng** |
| Model | Acoustic **16,4 triệu** tham số; vocoder **3,5 triệu** tham số |
| Sinh giọng | `vifasttts synthesize` chạy end-to-end (WAV 16-bit, 24 kHz); văn bản dài tự chia câu |
| Runtime ONNX | `vifasttts synthesize-onnx` chạy acoustic + vocoder ONNX, iSTFT numpy (không cần PyTorch) |
| Dịch vụ | `vifasttts serve` (HTTP stdlib: `POST /synthesize` → WAV) · `synthesize-batch` (file → nhiều WAV) · streaming qua `iter_synthesize` |
| Xuất ONNX | fp32 / fp16 / int8, batch và độ dài **động** (đã kiểm bằng onnxruntime) |
| Huấn luyện | Train acoustic + train vocoder (GAN tham chiếu) + cache feature |
| Dữ liệu YouTube | `vifasttts youtube-fetch`: tải audio + phụ đề kênh của bạn → clip 24 kHz mono + manifest + SRT mở + dataset card (CC BY 4.0), báo cáo từ lạ (OOV) + `--overrides` |

⚠️ **Chưa có sẵn trọng số huấn luyện** trong repo. Muốn có giọng nói thật, cần audio + transcript
và (khuyến nghị) alignment thật bằng MFA — xem `docs/TRAINING.md`.

## 3. Luồng hoạt động trong 30 giây

```text
1) CHUẨN BỊ:   WAV + CSV  →  manifest.jsonl  (+ cache feature, kiểm coverage)
2) TRAIN:      manifest   →  acoustic model (dự đoán mel/duration/f0)  → best.pt
                           vocoder riêng (mel → sóng)                 → vocoder.pt
3) SUY LUẬN:   text  →  7 ID/âm tiết  →  mel dự đoán  →  vocoder  →  out.wav
4) XUẤT:       best.pt  →  model.onnx  (chạy bằng ONNX Runtime, fp32/fp16/int8)
```

Điểm cần nhớ để hiểu code:

- **Train dùng thời lượng thật** (teacher forcing); **suy luận tự dự đoán** thời lượng.
- **Không sampling** — cùng đầu vào cho cùng đầu ra.
- Token không phải tiếng Việt vẫn chạy được (fallback), không bao giờ crash.

Chi tiết đầy đủ (tensor shape, công thức loss, số tham số): `docs/MODEL_FLOW.md`.

## 4. Cài đặt

Yêu cầu Python ≥ 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,audio,train,onnx]"
```

Bản tối thiểu (chỉ frontend/CLI, không cần torch):

```bash
pip install -e .
```

## 5. Thử nhanh

```bash
# Xem một câu được tách thành âm tiết và 5 thành phần thế nào
vifasttts parse "Hôm nay trời đẹp."
vifasttts parse-syllable "quốc"
vifasttts check-profile "voan bình" --profile strict   # cảnh báo chính tả/từ vay
vifasttts g2p "quốc" --dialect south                   # phiên âm IPA (phân tích)

# So sánh phiên âm IPA Bắc/Nam (chỉ để phân tích)
python examples/g2p_demo.py

# Lấy dữ liệu từ kênh YouTube của bạn (demo 1 video ngắn, cần: pip install -e ".[youtube]")
vifasttts youtube-fetch "https://www.youtube.com/@codelung/videos" --out data/codelung --demo

# Phiên bản CLI
vifasttts --version
```

Kiểm chứng toàn bộ pipeline bằng test (tự tạo dữ liệu tí hon, train 2 bước, sinh WAV, xuất ONNX):

```bash
pytest
```

## 6. Các lệnh chính

| Nhóm | Lệnh |
| --- | --- |
| Frontend | `vifasttts parse "<câu>"` · `parse-syllable <từ>` · `check-profile "<câu>" --profile strict` · `g2p "<câu>" [--dialect south]` |
| Dữ liệu | `build-manifest` · `validate-manifest` (kèm coverage) · `build-lexicon` · `cache-features` · `youtube-fetch` (tải kênh của bạn) |
| Huấn luyện | `vifasttts train ...` · `vifasttts train-vocoder ...` |
| Sinh giọng | `vifasttts synthesize --text "..." --output out.wav [--dialect south] [--speed 0.9] [--overrides file.json] [--peak-normalize]` |
| Batch / dịch vụ | `synthesize-batch --input list.txt --output-dir out/` · `serve --host 127.0.0.1 --port 8000` |
| Xuất | `vifasttts export-onnx ... [--precision fp16\|int8]` · `export-vocoder-onnx ...` |
| Runtime ONNX | `vifasttts synthesize-onnx --acoustic model.onnx --vocoder vocoder.onnx --text "..." --output out.wav [--provider cpu\|cuda\|coreml\|tensorrt]` |
| Khác | `vifasttts --version` |
| Kiểm tra | `pytest` · `ruff check .` · `mypy src/vifasttts` · `tools/parser_audit.py` · `tools/frontend_benchmark.py` · `tools/dataset_report.py` · `tools/split_manifest.py` |

<details>
<summary>Ví dụ đầy đủ (bấm để mở)</summary>

```bash
# Dữ liệu: metadata.csv có dạng  id|text
vifasttts build-manifest --metadata data/metadata.csv --wav-dir data/wavs --output data/manifest.jsonl
vifasttts validate-manifest --manifest data/manifest.jsonl --wav-dir data/wavs

# Huấn luyện
vifasttts train --config configs/base_24k.yaml \
  --train-manifest data/train.jsonl --val-manifest data/val.jsonl --output runs/base
vifasttts train-vocoder --config configs/base_24k.yaml \
  --manifest data/train.jsonl --output runs/vocoder

# Sinh giọng + xuất ONNX
vifasttts synthesize --config configs/base_24k.yaml \
  --checkpoint runs/base/best.pt --vocoder-checkpoint runs/vocoder/last.pt \
  --text "Xin chào, đây là hệ thống thử nghiệm." --output out.wav
vifasttts export-onnx --config configs/base_24k.yaml \
  --checkpoint runs/base/best.pt --output artifacts/model.onnx --precision fp16
vifasttts export-vocoder-onnx --config configs/base_24k.yaml \
  --checkpoint runs/vocoder/last.pt --output artifacts/vocoder.onnx

# Chạy lại chỉ với ONNX Runtime (không cần PyTorch)
vifasttts synthesize-onnx --acoustic artifacts/model.onnx --vocoder artifacts/vocoder.onnx \
  --text "Xin chào" --output onnx.wav --peak-normalize

# Batch và dịch vụ HTTP
vifasttts synthesize-batch --config configs/base_24k.yaml --checkpoint runs/base/best.pt \
  --input data/cau.txt --output-dir data/wavs_out
vifasttts serve --config configs/base_24k.yaml --checkpoint runs/base/best.pt --port 8000
# → curl -X POST localhost:8000/synthesize -d '{"text":"Xin chào"}' -o out.wav
#   (kèm GET /health, GET /info)
```

</details>

Trên **Mac Apple Silicon**: `training.device: auto` (mặc định) chạy acoustic bằng **MPS**,
vocoder bằng **CPU** (đo thực tế nhanh hơn trên máy nhỏ); số đo và ước lượng thời gian cho
Mac mini M6 32GB nằm ở `docs/TRAINING_ESTIMATES.md` §3.1.

## 7. Âm tiết tiếng Việt trong dự án (dễ hiểu)

Mỗi âm tiết (một "tiếng") gồm 5 phần + dấu câu. Ví dụ `quốc`:

| Phần | Trong `quốc` | Có bao nhiêu giá trị | Ví dụ khác |
| --- | --- | --- | --- |
| Âm đầu (onset) | `K` | 24 | `b`, `ch`, `ng`, `qu`... |
| Âm đệm (medial) | `W` | 2 (không có / có) | `oa`, `uy`, `qu`... |
| Âm chính (nucleus) | `Ô` | 15 | `a`, `ie`, `uo`, `oo`... |
| Âm cuối (coda) | `C` | 11 | `ng`, `nh`, `i`, `u`... |
| Thanh điệu | `SAC` (sắc) | 6 | ngang, huyền, hỏi, ngã, nặng |

Cộng thêm **chấm câu** và **ranh giới từ/câu** → mỗi âm tiết là **7 con số** đưa vào model.
Giải thích từ đầu cho người không chuyên: `docs/SYLLABLE_GUIDE.md`.

## 8. Cấu trúc repo

```text
src/vifasttts/
  frontend/   chuẩn hoá text, tách âm tiết, vocab, G2P (phân tích), lexicon/profile, segments, chunking
  data/       manifest, features (mel/f0/vuv/energy/phonation), dataset, cache, subtitles + youtube (tải kênh của bạn)
  models/     acoustic model, conformer, iSTFTNet vocoder, discriminator
  runtime/    chạy ONNX (acoustic + vocoder), iSTFT numpy, postprocess
  train.py    vòng huấn luyện acoustic + vocoder
  infer.py    text → WAV (PyTorch)
  export.py   ONNX (acoustic + vocoder) + quantization
configs/      base_24k.yaml + small_24k.yaml + large_24k.yaml + overrides_tech_vi.json (từ lạ cho youtube-fetch)
tools/        parser_audit.py, frontend_benchmark.py
tests/        2223 test: parser, normalizer, G2P, model, train/infer/export, runtime ONNX, server, youtube
docs/         toàn bộ tài liệu (xem mục 10)
```

## 9. Giới hạn hiện tại & việc còn lại

- **Alignment**: chưa có thời lượng thật → mặc định chia theo trọng số (placeholder). Cần MFA/aligner để chất lượng tốt hơn (kể cả khi lấy dữ liệu nhanh từ YouTube của bạn bằng `youtube-fetch` — thời lượng mới ở mức câu).
- **Vocoder**: bản tham chiếu (GAN nhẹ), chưa phải cấu hình production.
- **ONNX**: đã có runtime đầy đủ (acoustic + vocoder spectrum + iSTFT numpy). Điểm còn để ngỏ: gộp vocoder vào **một graph duy nhất** và calibration INT8 đầy đủ.
- **OOV**: từ ngoài tiếng Việt đọc bằng fallback — muốn kiểm soát thì bật `resolve_oov=True`, dùng `--overrides file.json` hoặc thêm override.
- **Một checkpoint = một giọng, một phương ngữ** (đúng thiết kế; muốn đa giọng phải tách lớp riêng).

Roadmap chi tiết có đánh dấu ✅/⬜: `docs/PROJECT_STATUS.md`.

## 10. Tài liệu — nên đọc gì trước?

| Nhóm | Tài liệu | Nội dung |
| --- | --- | --- |
| **Bắt đầu ở đây** | `docs/PROJECT_STATUS.md` | Trạng thái tổng, quyết định quan trọng, việc còn lại |
| | `docs/MODEL_FLOW.md` | Luồng model chi tiết: data → train → inference → export |
| | `docs/SYLLABLE_GUIDE.md` | Hướng dẫn tách âm tiết cho người không chuyên |
| | `docs/GLOSSARY.md` | Thuật ngữ (âm đầu/đệm/chính/cuối, toàn ánh/song ánh...) |
| **Frontend** | `docs/FRONTEND.md` | Parser, profile/lexicon, segments, G2P, OOV |
| | `docs/NORMALIZER.md` | Số/ngày/giờ theo vùng miền, dấu thanh, i/y |
| | `docs/STEM_NORMALIZER.md` | Toán/LaTeX, hoá học, đơn vị vật lý |
| | `docs/G2P_VIETNAMESE.md` | Khảo sát sea-g2p, viG2P... và bảng IPA nội bộ |
| **Model & dữ liệu** | `docs/ARCHITECTURE.md` | Sơ đồ as-built, quyết định thiết kế |
| | `docs/DATA_PIPELINE.md` | 15 bước chuẩn bị dữ liệu |
| | `docs/YOUTUBE_DATASET.md` | Lấy dataset sạch từ kênh YouTube của bạn (demo + chạy thật, giấy phép mở) |
| | `docs/TRAINING.md` | Hướng dẫn train + cảnh báo alignment |
| | `docs/TRAINING_RUN_CODELUNG.md` | **Run thật `small` trên kênh @codelung**: số đo MPS, cách chạy lại, triển khai |
| | `docs/TRAINING_ESTIMATES.md` | **Ước lượng** dữ liệu/tham số/thời gian/máy/chi phí trước khi train |
| | `docs/APPLE_SILICON.md` | **Mac/M-series**: MPS, MLX, ANE, provider ONNX (đo thật + quyết định) |
| | `docs/DEPLOYMENT.md` | ONNX Runtime, quantize, nền tảng |
| **Nghiên cứu** | `docs/SYLLABLE_SYSTEMS_RESEARCH.md` | So sánh mô hình tách/gộp âm tiết (quốc tế + Việt Nam) và audit biên |
| | `docs/TTS_ARCHITECTURE_RESEARCH.md` | Khảo sát họ kiến trúc TTS + đề xuất tương lai |
| **Thiết kế gốc** | `docs/DESIGN_V1_MIEN_BAC.md`, `docs/DESIGN_V2.md`, `docs/FULL_DESIGN.md` | Bản thiết kế ban đầu (tham khảo lịch sử) |
| **Khác** | `docs/RUST_ROADMAP.md` | Kế hoạch port Rust (tối ưu triển khai) |
| | `docs/LICENSES.md` | Giấy phép các thành phần tham chiếu |

## 11. Giấy phép

MIT cho code trong repo. Dữ liệu giọng nói, lexicon và checkpoint do người dùng tạo có
giấy phép riêng; các công cụ tham chiếu (sea-g2p, MFA, vPhon...) giữ giấy phép gốc —
chi tiết ở `docs/LICENSES.md`. Dataset lấy từ kênh YouTube của bạn mặc định ghi giấy phép
mở **CC BY 4.0** (ghi nguồn kênh khi tái sử dụng); đổi bằng `--license`, ví dụ
`--license "CC0 1.0"`.
