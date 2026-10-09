# ViFastTts

ViFastTts là dự án nghiên cứu và triển khai TTS tiếng Việt một giọng, một checkpoint, ưu tiên độ chính xác phát âm, thanh điệu ổn định và tốc độ suy luận cao.

## Triết lý

- Frontend dùng chung cho tiếng Việt, không gắn cứng với miền Bắc, Trung hay Nam.
- Mỗi checkpoint chỉ chứa một người nói và một phương ngữ.
- Không cần `dialect_id`, `speaker_id`, `style_id` hoặc audio tham chiếu ở runtime.
- Mỗi âm tiết được mô tả bằng `ONSET + MEDIAL + NUCLEUS + CODA + TONE`.
- Mô hình non-autoregressive và deterministic.
- Conformer ở cấp âm tiết, Segment Expander ở cấp thành phần âm tiết, iSTFTNet ở đầu ra.
- ONNX là định dạng triển khai chính.

## Trạng thái

Repository này là một nền tảng đầy đủ để tiếp tục nghiên cứu và huấn luyện. Những phần có thể chạy ngay:

- Chuẩn hóa Unicode và tách câu cơ bản.
- Parser âm tiết tiếng Việt có cấu trúc.
- Normalizer theo vùng miền (`north`, `central`, `south`), chuẩn hoá dấu thanh, số, La Mã, tiếng Anh fallback và công thức STEM.
- Orthography canonical, dialect profile, roman numerals, English fallback, STEM (Toán/Hoá/đơn vị).
- Vocabulary thành phần.
- Manifest và kiểm tra dữ liệu.
- Mô hình PyTorch ViFastTts (Conformer + Length Regulator + variance predictors + iSTFTNet).
- Loss, training loop, inference và export ONNX.
- Unit test, CLI và tài liệu.

Để có giọng nói thực tế, bạn vẫn cần dataset audio và transcript, alignment target, sau đó huấn luyện checkpoint. Repository không kèm trọng số đã huấn luyện.

## Cài đặt

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
```

## Normalizer theo vùng miền

`north`, `central`, `south` chỉ tác động tới cách đọc số và từ có khác biệt thật, ví dụ `nghìn/ngàn`, `linh/lẻ` và lựa chọn `tư/bốn` sau hàng chục. Parser âm tiết không thay đổi theo vùng.

- Bắc: `nghìn`, `linh`, mặc định `tư` sau hàng chục.
- Trung: `ngàn`, `lẻ`, mặc định `tư`.
- Nam: `ngàn`, `lẻ`, mặc định `bốn`.

Mọi văn bản được chuẩn hoá vị trí dấu thanh về dạng âm vị học chính: `hòa` thành `hoà`, `hóa` thành `hoá`, `thủy` thành `thuỷ`. Parser chấp nhận cả hai lối viết, normalizer chỉ xuất một dạng canonical.

Thứ tự phân loại token: exact override → acronym viết hoa chính xác → tên sản phẩm → số La Mã viết hoa chuẩn (≥2 ký tự) → âm tiết tiếng Việt → English fallback không phân biệt hoa thường → giữ nguyên.

## Kiểm tra dự án

```bash
pytest
ruff check .
mypy src/vifasttts
```

## Tạo manifest từ WAV và metadata

```bash
vifasttts build-manifest \
  --metadata data/metadata.csv \
  --wav-dir data/wavs \
  --output data/manifest.jsonl
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
  --output runs/base
```

## Export ONNX

```bash
vifasttts export-onnx \
  --config configs/base_24k.yaml \
  --checkpoint runs/base/best.pt \
  --output artifacts/vifasttts.onnx
```

## Hướng Rust sau này

Phiên bản đầu dùng Python để dễ nghiên cứu, kiểm thử và thay đổi kiến trúc. Sau khi quy tắc ổn định, các phần xử lý thuần có thể viết thêm bằng Rust:

- Unicode normalization.
- Text lexer và normalizer.
- Syllable parser.
- Vocabulary mapping.
- Manifest validation.
- ONNX Runtime inference.
- WAV I/O và CLI.

PyO3 hoặc C ABI có thể được dùng để gọi Rust từ Python. Việc viết lại bằng Rust là tối ưu triển khai, không thay đổi định dạng parser hoặc checkpoint.

## Tài liệu

- `docs/ARCHITECTURE.md`
- `docs/FRONTEND.md`
- `docs/DATA_PIPELINE.md`
- `docs/TRAINING.md`
- `docs/DEPLOYMENT.md`
- `docs/RUST_ROADMAP.md`
- `docs/FULL_DESIGN.md` — tách frontend dùng chung / checkpoint riêng, âm tiết, dấu, dialect, English fallback, neural.
- `docs/NORMALIZER.md` — số theo vùng miền, dấu canonical, fallback tiếng Anh, thứ tự token.
- `docs/STEM_NORMALIZER.md` — Toán/LaTeX, Hoá học, đơn vị vật lý.
- `docs/DESIGN_V1_MIEN_BAC.md` — thiết kế chi tiết v1 miền Bắc (tài liệu gốc).
- `docs/DESIGN_V2.md` — thiết kế hệ thống v2 cải tiến (tài liệu gốc).

## Giấy phép

MIT. Dữ liệu giọng nói và checkpoint do người dùng tạo có thể có giấy phép riêng.
