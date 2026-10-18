# Frontend tiếng Việt

## Mục tiêu

Bảo toàn cấu trúc chính tả và tạo **dạng âm vị học** trung lập phương ngữ (thuật ngữ: `docs/GLOSSARY.md`).

## Cấu trúc

```text
ONSET + MEDIAL + NUCLEUS + CODA + TONE
```

## Quy tắc quan trọng

- Chỉ loại dấu thanh, không làm mất ă â ê ô ơ ư.
- Longest match cho onset và coda.
- Giữ D, GI, R riêng.
- `gi` dùng chung chữ i: trước `ê`, cuối từ hoặc phụ âm thì i thuộc vần (`gì` = GI + I, `giếng` = GI + IÊ, cùng vần với `tiếng`); trước nguyên âm khác (`già`, `giữ`, `giường`) thì i là digraph /z/ câm.
- Giữ TR, CH riêng.
- Giữ S, X riêng.
- Giữ HỎI, NGÃ riêng.
- `qu` ánh xạ K + W khi sau `u` là nguyên âm **không tròn môi**: `a ă â e ê i/y ơ` (`qua`, `quăn`, `quân`, `que`, `quê`, `quy`, `quơ`).
- `qu` + `ô` (`quốc`): âm đệm /w/ không kết hợp với âm tròn môi (u, ô, o) → chữ `u` thuộc nguyên âm đôi `uô`: `quốc` = K + NONE + UÔ + C (onset chính tả `q`; đồng âm `cuốc` — xem `docs/SYLLABLE_GUIDE.md` §5).
- `quo`, `quu` là fallback (không phải vần thật).
- `q` trần không hợp lệ (`qa`, `qi`, `qy`... là fallback); riêng `quốc` có onset chính tả `q` nội bộ (u thuộc vần — xem trên).
- `i/y` cuối có thể ánh xạ J.
- `u/o` cuối có thể ánh xạ W.
- `uy` là medial W + nucleus I, **không** phải U + coda J: `khuy` (KH/W/I) khác `khui` (KH/NONE/U/J); dấu thanh rơi trên y (`thuỷ`, `tuỳ`).
- `ui` là nucleus U + coda J; dấu thanh rơi trên u (`tủi`, `củi`, `chùi`).
- `uya` (khuya) dùng medial W + nucleus IÊ; `uyê` (nguyễn) dùng W + IÊ + coda.
- `oo` trong từ vay đã Việt hoá (xoong, boong, moóc) dùng **nucleus `OO` riêng** (o dài ɔː), phân biệt với `o` thường; dấu đặt trên chữ o thứ hai.
- `ay`, `au` (và `oay`) dùng **nucleus `Ă`** (a ngắn [a]) còn `ai`, `ao` (và `oai`) dùng **`A`** (a dài [aː]): `tay` ≠ `tai`, `rau` ≠ `rao`, `quay` ≠ `quai` — nguồn: Help:IPA/Vietnamese ("ai" [aːj] vs "ay" [aj]).
- `iê/yê/ia/ya`, `uô/ua`, `ươ/ưa` dùng nucleus trừu tượng chung.

## Lưu ý

Parser tham chiếu trong repository bao phủ các quy tắc lõi và bộ test tối thiểu. Để siết dữ liệu
trước khi huấn luyện production, dùng **profile** + **lexicon** (mục trên) thay vì sửa parser:
`strict` bắt biến thể chính tả, `corpus` bắt shape lạ so với lexicon dự án.

## Kiểm tra parser

- `python tools/parser_audit.py` — vét cạn 49.104 tổ hợp hợp lệ (âm đầu × vần × âm cuối × thanh),
  kiểm tra bất biến `parse → compose → parse`; thoát mã 1 nếu có vấn đề (chạy được trong CI).
- `pytest tests/test_parser_matrix.py tests/test_parser_edges.py` — cùng tính chất ở dạng
  test thường trực, kèm corpus từ thật/khó, chữ ngoài Latin và edge Unicode.
- Tokenizer giữ cả chữ ngoài Latin (CJK, Cyrillic, Thái...) làm token fallback thay vì nuốt mất.
- Các "alias" đã cố ý: `uy` luôn là W + I (`khuy` ≠ `khui`); `giê` luôn là GI + IÊ
  (`giếng` cùng vần `tiếng`). `oo` **không còn** là alias: đã tách thành nucleus `OO` riêng (ɔː)
  vì người bản ngữ phân biệt `xoong` ≠ `xong`; đây là một nguyên âm dài, không phải nguyên âm đôi.

## Profile chặt/chùng (`frontend/lexicon.py`)

Parser giữ permissive; profile thêm lớp kiểm tra tùy chọn (không từ chối dữ liệu, chỉ cảnh báo):

- `open` — mặc định, không cảnh báo.
- `strict` — luật chính tả/ngữ âm bản ngữ: `c` trước i/e/ê phải là `k`; `k` trước a/o/u/ơ/ư/ă/â phải là `c`; `g/gh`, `ng/ngh` (đã tính âm đệm nên `ngoe`, `nguyễn`, `nghiêng` hợp lệ); phụ âm môi + /w/ (`voan`, `boa` — từ vay); coda `nh/ch` chỉ sau a/ê/i; `iê` đứng một mình phải viết `yê`.
- `corpus` — thêm ràng buộc shape (onset + vần) phải có trong `Lexicon`.

```python
from vifasttts.frontend.lexicon import Lexicon, check_syllable
from vifasttts.frontend.syllable import parse_syllable

lex = Lexicon.from_words(["ba", "mẹ", "khoai"])      # hoặc from_manifest / load
check_syllable(parse_syllable("khoai"), "corpus", lex)  # -> []
check_syllable(parse_syllable("voan"), "strict")        # -> ["labial-w"]
```

CLI: `vifasttts check-profile "..." --profile strict`, `vifasttts build-lexicon ...`.
Audit: `tools/parser_audit.py --wordlist FILE --profile strict|corpus [--lexicon lex.json]`
(`--write-lexicon` để lưu lexicon sinh từ wordlist).

Đo trên ngữ liệu 39K+74K: strict chỉ báo ~35 biến thể từ thật (chủ yếu `ngịu`/`ngẹt` —
chuẩn là `nghịu`/`nghẹt`, từ vay môi+/w/, chữ K đầu từ dân tộc) → corpus chuẩn gần như sạch.
Corpus profile với lexicon 39K phát hiện ~290 shape mới trong 74K → nên sinh lexicon từ
corpus lớn nhất có thể trước khi bật.

## Segment Expander (`frontend/segments.py`)

Mở 5 thành phần thành chuỗi segment cấp âm vị để phân bổ duration và điều kiện hoá phonation
(xem `docs/TTS_ARCHITECTURE_RESEARCH.md` §9.4):

- `IÊ/UÔ/ƯƠ` → hai đích (`i/u/ɯ` + `ə`); `OO` có trọng số dài hơn (1.2).
- Trọng số role: onset 0.4, âm đệm 0.3, âm chính 1.0, coda 0.35–0.6; tone 0 (không chiếm frame).
- `distribute_frames(frames, segments)` chia frame theo trọng số (largest remainder, tổng đúng,
  mỗi segment dương ≥ 1 frame khi đủ frame).
- Biến thể phương ngữ/kiểu đọc: Nam `OO→ɔ`; Bắc narrow `ưu/ươu → [iw]/[iəw]`.

## Phép chiếu vần (rime projection)

`AbstractSyllable.rime` = `(medial, nucleus, coda)` và `.rime_key` — phép chiếu không mất
thông tin sang mô hình "onset + vần + thanh". Mọi biểu diễn thô hơn (vần+thanh, tonal
phoneme, whole-syllable) đều là merge tất định từ 5 thành phần — xem
`docs/SYLLABLE_SYSTEMS_RESEARCH.md` mục 6.

## Benchmark biểu diễn (`frontend/benchmark.py`)

```bash
python tools/frontend_benchmark.py --text "Chào bạn, hôm nay trời đẹp quá!"
```

In số đơn vị/âm tiết, p95 độ dài câu và vocab cho 4 biểu diễn char/slot/phoneme/rime.
Ví dụ thực đo: char 3.29, slot 5.0, phoneme 4.0, rime 2.0 đơn vị/âm tiết — dùng để cân
nhắc unit đầu vào model.

## Chia câu cho văn bản dài (`frontend/chunking.py`)

`chunk_text(text, max_chars=400)` tách theo câu → mệnh đề → từ; `pause_seconds()` gợi ý
khoảng nghỉ (0.15s cuối câu, 0.08s cuối mệnh đề, 0.05s còn lại). Cả `synthesize` (PyTorch)
lẫn `synthesize-onnx` dùng chung để tránh đọc câu dài một hơi. `infer.load_overrides(file)`
nạp từ điển "mặt chữ → cách đọc" và `runtime/postprocess.py` lo fade/peak-normalize.

## Lớp G2P tùy chọn (`frontend/g2p.py`) — IPA chuẩn theo phương ngữ

Ánh xạ 5 thành phần sang IPA theo Kirby (2011)/Thompson (1965) và bảng đối chiếu trong `docs/G2P_VIETNAMESE.md`:

- `syllable_to_phonemes(parsed, dialect="north", *, with_tone=True, tone_format="contour", style="broad")`.
  - `style="broad"` (mặc định): âm vị gọn (`t`, `tʰ`, `k`...). `style="narrow"`: thêm chi tiết ngữ âm `t̪`, `t̪ʰ`, `p̚ t̚ k̚`, `k̟̚`, `ŋ̟`; Bắc: trước `nh/ch`, `a`/`ê` → `ăj`/`ə̆j` (Henderson 1965; Kirby 2011: `anh` = `ăj ŋ̟`).
  - `tone_format="contour"` (mặc định, ký hiệu Chao) | `"number"` (1 ngang, 2 huyền, 3 ngã, 4 hỏi, 5 sắc, 6 nặng — quy ước viG2P để đối chiếu) | `"number8"` (như "number" nhưng 7 = sắc nhập, 8 = nặng nhập cho âm tiết kết thúc `p/t/c/ch` — quy ước gắn nhãn Praat) | `"none"`.
- `text_to_phonemes(...)` trả một tuple cho mỗi âm tiết; token OOV trả tuple rỗng để tầng ngoài quyết định.
- Bảng theo vùng:
  - **Bắc (Hà Nội)**: gộp `d/gi/r → z`, `ch/tr → tɕ`, `s/x → s`; thanh ˧, ˨˩, ˧˥, ˧˩˧, ˧ˀ˥, ˧ˀ˨ʔ.
  - **Nam (TP.HCM)**: `d/gi → j`, `r → ʐ`, `s → ʂ`, `x → s`, `tr → ʈ`, `ch → c`; hỏi–ngã gộp `˨˩˦`, nặng `˨˩˨`, ngang `˦`; `OO` gộp về `ɔ` (`xoong` ≈ `xong`).
  - **Trung**: xấp xỉ (âm đầu kiểu Nam, thanh điệu tạm theo Bắc) — chờ dữ liệu Huế, xem `docs/G2P_VIETNAMESE.md` mục 6.5.
- Narrow Bắc: `ưu/ươu → [iw]/[iəw]` (Thompson 1965); broad giữ âm vị /ɯw/ɯəw/.
- Nguyên âm theo Kirby: `a ă ɤ̆ ɛ e i ɔ o ɤ u ɯ`; thêm `OO = ɔː` (o dài, từ vay); vần đôi `iə uə ɯə`; không thêm trọng âm (tiếng Việt không có stress từ vựng).
- Đây vẫn là kênh nghiên cứu/đối chiếu (lexicon, round-trip, kiểm tra biến thể), **không** thay vocab ID ở runtime. Kế hoạch tích hợp sea-g2p/viG2P, WikiPron, MFA nằm ở `docs/G2P_VIETNAMESE.md`.

## OOV resolver (`frontend/oov.py`)

Quy trình 5 bước theo tinh thần viG2P (viết lại, không dùng code GPL):

1. Exact acronym / product / English fallback có sẵn.
2. Token giống từ tiếng Anh → respelling trong từ điển.
3. Token viết hoa toàn bộ (≥ 2 chữ cái, không chứa chữ số) → đánh vần bằng tên chữ cái tiếng Việt: `VNG` → "vê en giê".
4. Còn lại trả `None` → giữ `FALLBACK_REQUIRED` kèm `raw` để log và bổ sung override.

Bật khi cần bằng `parse_text(text, resolve_oov=True)`; mặc định tắt để hành vi không đổi. Các âm tiết sinh từ đánh vần có `confidence=0.5` và giữ `raw` là token gốc.
