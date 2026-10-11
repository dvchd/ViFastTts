# ViFastTts

ViFastTts là dự án nghiên cứu và triển khai TTS tiếng Việt một giọng, một checkpoint, ưu tiên độ chính xác phát âm, thanh điệu ổn định và tốc độ suy luận cao.

## Triết lý

- Frontend dùng chung cho tiếng Việt, không gắn cứng với miền Bắc, Trung hay Nam.
- Mỗi checkpoint chỉ chứa một người nói và một phương ngữ.
- Không cần `dialect_id`, `speaker_id`, `style_id` hoặc audio tham chiếu ở runtime.
- Mỗi âm tiết được mô tả bằng `ONSET + MEDIAL + NUCLEUS + CODA + TONE`.
- Mô hình non-autoregressive và deterministic.
- Conformer ở cấp âm tiết, Length Regulator ở cấp frame, iSTFTNet ở đầu ra.
- ONNX là định dạng triển khai chính (đã kiểm chứng export với batch/độ dài động).

## Trạng thái

Những phần chạy được ngay trong repository:

- Chuẩn hóa Unicode, tách câu, tách token.
- Parser âm tiết tiếng Việt có cấu trúc, fallback không crash.
- Normalizer theo vùng miền (`north`, `central`, `south`): số, ngày/tháng/năm, giờ, phân số, khoảng giá trị, phần trăm, tiền tệ, điện thoại, La Mã, STEM (Toán/Hoá/Lý), English fallback có kiểm chứng "reading phải là âm tiết Việt hợp lệ".
- Lớp G2P tùy chọn (`frontend/g2p.py`, chỉ dùng cho phân tích — không nằm trong đường runtime) và OOV resolver (`frontend/oov.py`, opt-in qua `parse_text(..., resolve_oov=True)`).
- Vocabulary thành phần, manifest JSONL và `validate-manifest`.
- Mô hình PyTorch: Conformer + positional encoding + variance predictors + LengthRegulator vector hoá + frame-position + iSTFTNet có upsampling.
- Vòng huấn luyện thật: trích mel/energy/F0/VUV từ WAV, collate có mask, teacher forcing duration, lưu `best.pt`/`last.pt`.
- Suy luận end-to-end `vifasttts synthesize` (WAV 16-bit, không cần thư viện ngoài).
- Export ONNX acoustic model (đã kiểm chứng bằng onnxruntime: batch, số âm tiết và tổng frame động).

Để có giọng nói chất lượng thực tế bạn vẫn cần dataset audio + transcript và alignment thật (MFA hoặc aligner tương đương); repository dùng alignment tạm (chia đều frame) và không kèm trọng số đã huấn luyện. Xem `docs/TRAINING.md` và `docs/TTS_ARCHITECTURE_RESEARCH.md`.

## Cài đặt

Yêu cầu Python >= 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,audio,train,onnx]"
```

Trên Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,audio,train,onnx]"
```

## Thử parser

```bash
vifasttts parse "Hôm nay trời đẹp."
vifasttts parse-syllable "ngoài"
vifasttts check-profile "voan bình" --profile strict   # cảnh báo chính tả/ngữ âm
python examples/g2p_demo.py   # phiên âm IPA Bắc/Nam, broad/narrow, 6 thanh
```

## Normalizer theo vùng miền

`north`, `central`, `south` chỉ tác động tới cách đọc số và từ có khác biệt thật, ví dụ `nghìn/ngàn`, `linh/lẻ` và lựa chọn `tư/bốn` sau hàng chục. Parser âm tiết không thay đổi theo vùng.

- Bắc: `nghìn`, `linh`, mặc định `tư` sau hàng chục.
- Trung: `ngàn`, `lẻ`, mặc định `tư`.
- Nam: `ngàn`, `lẻ`, mặc định `bốn`.

Mọi văn bản được chuẩn hoá vị trí dấu thanh về dạng âm vị học chính: `hòa` thành `hoà`, `hóa` thành `hoá`, `thủy` thành `thuỷ`.

Thứ tự phân loại token: exact override → acronym viết hoa chính xác → tên sản phẩm → số La Mã viết hoa chuẩn (≥2 ký tự) → âm tiết tiếng Việt → English fallback không phân biệt hoa thường → giữ nguyên.

Ví dụ mới: `25/12/2023` → "ngày hai mươi lăm tháng mười hai năm hai nghìn không trăm hai mươi ba"; `14h30` → "mười bốn giờ ba mươi phút"; `2/3` → "hai phần ba"; `10-20%` → "mười đến hai mươi phần trăm"; `1.005.000` → "một triệu không trăm linh năm nghìn".

## Kiểm tra dự án

```bash
pytest
ruff check .
mypy src/vifasttts
python tools/parser_audit.py                       # audit parser: ma trận ~49k tổ hợp
python tools/parser_audit.py --wordlist data.txt   # audit trên danh sách từ (1 từ/dòng)
python tools/parser_audit.py --wordlist data.txt --profile strict   # cảnh báo chính tả/ngữ âm
python tools/parser_audit.py --wordlist data.txt --write-lexicon data/lexicon.json
python tools/frontend_benchmark.py --text "Chào bạn"                # so sánh biểu diễn char/slot/phoneme/rime
```

## Tạo manifest từ WAV và metadata

```bash
vifasttts build-manifest \
  --metadata data/metadata.csv \
  --wav-dir data/wavs \
  --output data/manifest.jsonl

vifasttts validate-manifest --manifest data/manifest.jsonl --wav-dir data/wavs
```

`validate-manifest` kiểm tra WAV tồn tại và in **coverage** (số âm tiết, fallback, phân bố tone/coda, thanh còn thiếu) để đối chiếu với `docs/DATA_PIPELINE.md` bước 10.

Sinh lexicon shape để siết dữ liệu bằng `profile` (xem `docs/FRONTEND.md`):

```bash
vifasttts build-lexicon --manifest data/manifest.jsonl --output data/lexicon.json
vifasttts check-profile "khoai lang" --profile corpus --lexicon data/lexicon.json
```

Định dạng `metadata.csv`:

```text
sample_0001|Hôm nay trời đẹp.
sample_0002|Tôi đang thử hệ thống mới.
```

## Huấn luyện

```bash
vifasttts train \
  --config configs/base_24k.yaml \
  --train-manifest data/train.jsonl \
  --val-manifest data/val.jsonl \
  --output runs/base \
  --max-steps 200000
```

Manifest có thể chứa trường `durations` (số frame mỗi âm tiết từ forced alignment). Nếu thiếu, duration được chia đều theo số frame — chỉ là placeholder, xem cảnh báo trong `docs/TRAINING.md`.

Vocoder tham chiếu (iSTFTNetLite + MPD-lite) tách riêng:

```bash
vifasttts train-vocoder \
  --config configs/base_24k.yaml \
  --manifest data/train.jsonl \
  --output runs/vocoder \
  --max-steps 100000
```

`synthesize` tự dùng trọng số `vocoder` trong checkpoint nếu có.

Để train nhanh hơn ở epoch sau, tính trước feature vào cache:

```bash
vifasttts cache-features \
  --manifest data/train.jsonl \
  --config configs/base_24k.yaml \
  --cache-dir cache/train
```

hoặc đặt `training.cache_features: true` trong config (dùng `<run>/feature_cache`).

## Tổng hợp

```bash
vifasttts synthesize \
  --config configs/base_24k.yaml \
  --checkpoint runs/base/best.pt \
  --text "Xin chào, đây là hệ thống thử nghiệm." \
  --output out.wav
```

## Export ONNX

```bash
vifasttts export-onnx \
  --config configs/base_24k.yaml \
  --checkpoint runs/base/best.pt \
  --output artifacts/vifasttts.onnx
```

Thêm `--precision fp16` hoặc `--precision int8` để xuất bản lượng tử hoá (xem `docs/DEPLOYMENT.md`).

## Hướng Rust sau này

Phiên bản đầu dùng Python để dễ nghiên cứu, kiểm thử và thay đổi kiến trúc. Sau khi quy tắc ổn định, các phần xử lý thuần có thể viết thêm bằng Rust:

- Unicode normalization.
- Text lexer và normalizer.
- Syllable parser và OOV resolver.
- Vocabulary mapping.
- Manifest validation.
- ONNX Runtime inference.
- WAV I/O và CLI.

PyO3 hoặc C ABI có thể được dùng để gọi Rust từ Python. Việc viết lại bằng Rust là tối ưu triển khai, không thay đổi định dạng parser hoặc checkpoint.

## Tài liệu

- `docs/ARCHITECTURE.md` — kiến trúc as-built và lộ trình.
- `docs/PROJECT_STATUS.md` — trạng thái tổng hợp, nhật ký quyết định, kết quả audit, việc còn lại.
- `docs/GLOSSARY.md` — thuật ngữ: cấu trúc âm tiết (nguồn gốc, âm đầu–đệm–chính–cuối), dạng âm vị học, ngữ âm; quy ước đặt tên.
- `docs/SYLLABLE_GUIDE.md` — hướng dẫn tách âm tiết cho người không chuyên (5 bước, ví dụ, bài tập, tài liệu tham khảo).
- `docs/FRONTEND.md` — parser, G2P, OOV.
- `docs/DATA_PIPELINE.md`
- `docs/TRAINING.md`
- `docs/DEPLOYMENT.md`
- `docs/RUST_ROADMAP.md`
- `docs/G2P_VIETNAMESE.md` — sea-g2p, viG2P và các hệ G2P tiếng Việt; cách tích hợp.
- `docs/SYLLABLE_SYSTEMS_RESEARCH.md` — so sánh mô hình tách/gộp âm tiết (lý thuyết quốc tế + nghiên cứu Việt Nam) và audit biên.
- `docs/TTS_ARCHITECTURE_RESEARCH.md` — khảo sát toàn bộ họ kiến trúc TTS và đề xuất v2.1.
- `docs/FULL_DESIGN.md` — tách frontend dùng chung / checkpoint riêng, âm tiết, dấu, dialect, English fallback, neural.
- `docs/NORMALIZER.md` — số theo vùng miền, dấu canonical, fallback tiếng Anh, thứ tự token.
- `docs/STEM_NORMALIZER.md` — Toán/LaTeX, Hoá học, đơn vị vật lý.
- `docs/DESIGN_V1_MIEN_BAC.md` — thiết kế chi tiết v1 miền Bắc (tài liệu gốc).
- `docs/DESIGN_V2.md` — thiết kế hệ thống v2 cải tiến (tài liệu gốc).
- `docs/LICENSES.md` — ghi chú giấy phép các thành phần tham chiếu.

## Giấy phép

MIT. Dữ liệu giọng nói và checkpoint do người dùng tạo có thể có giấy phép riêng.
