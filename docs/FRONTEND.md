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
- `qu` ánh xạ K + W.
- `q` chỉ tồn tại trong `qu`; `qa`, `qi`, `qy`... là fallback (không phải âm đầu hợp lệ tiếng Việt).
- `i/y` cuối có thể ánh xạ J.
- `u/o` cuối có thể ánh xạ W.
- `uy` là medial W + nucleus I, **không** phải U + coda J: `khuy` (KH/W/I) khác `khui` (KH/NONE/U/J); dấu thanh rơi trên y (`thuỷ`, `tuỳ`).
- `ui` là nucleus U + coda J; dấu thanh rơi trên u (`tủi`, `củi`, `chùi`).
- `uya` (khuya) dùng medial W + nucleus IÊ; `uyê` (nguyễn) dùng W + IÊ + coda.
- `oo` trong từ vay đã Việt hoá (xoong, boong, moóc) dùng **nucleus `OO` riêng** (o dài ɔː), phân biệt với `o` thường; dấu đặt trên chữ o thứ hai.
- `ay`, `au` (và `oay`) dùng **nucleus `Ă`** (a ngắn [a]) còn `ai`, `ao` (và `oai`) dùng **`A`** (a dài [aː]): `tay` ≠ `tai`, `rau` ≠ `rao`, `quay` ≠ `quai` — nguồn: Help:IPA/Vietnamese ("ai" [aːj] vs "ay" [aj]).
- `iê/yê/ia/ya`, `uô/ua`, `ươ/ưa` dùng nucleus trừu tượng chung.

## Lưu ý

Parser tham chiếu trong repository bao phủ các quy tắc lõi và bộ test tối thiểu. Trước khi huấn luyện production, cần mở rộng bảng âm tiết hợp lệ bằng corpus lớn và thêm test cho tên riêng, từ vay mượn, chính tả cũ và trường hợp hiếm.

## Kiểm tra parser

- `python tools/parser_audit.py` — vét cạn ~51k tổ hợp (âm đầu × vần × âm cuối × thanh),
  kiểm tra bất biến `parse → compose → parse`; thoát mã 1 nếu có vấn đề (chạy được trong CI).
- `pytest tests/test_parser_matrix.py tests/test_parser_edges.py` — cùng tính chất ở dạng
  test thường trực, kèm corpus từ thật/khó, chữ ngoài Latin và edge Unicode.
- Tokenizer giữ cả chữ ngoài Latin (CJK, Cyrillic, Thái...) làm token fallback thay vì nuốt mất.
- Các "alias" đã cố ý: `uy` luôn là W + I (`khuy` ≠ `khui`); `giê` luôn là GI + IÊ
  (`giếng` cùng vần `tiếng`). `oo` **không còn** là alias: đã tách thành nucleus `OO` riêng (ɔː)
  vì người bản ngữ phân biệt `xoong` ≠ `xong`; đây là một nguyên âm dài, không phải nguyên âm đôi.

## Lớp G2P tùy chọn (`frontend/g2p.py`) — IPA chuẩn theo phương ngữ

Ánh xạ 5 thành phần sang IPA theo Kirby (2011)/Thompson (1965) và bảng đối chiếu trong `docs/G2P_VIETNAMESE.md`:

- `syllable_to_phonemes(parsed, dialect="north", *, with_tone=True, tone_format="contour", style="broad")`.
  - `style="broad"` (mặc định): âm vị gọn (`t`, `tʰ`, `k`...). `style="narrow"`: thêm chi tiết ngữ âm `t̪`, `t̪ʰ`, `p̚ t̚ k̚`, `k̟̚`, `ŋ̟`.
  - `tone_format="contour"` (mặc định, ký hiệu Chao) | `"number"` (1 ngang, 2 huyền, 3 ngã, 4 hỏi, 5 sắc, 6 nặng — quy ước viG2P để đối chiếu) | `"none"`.
- `text_to_phonemes(...)` trả một tuple cho mỗi âm tiết; token OOV trả tuple rỗng để tầng ngoài quyết định.
- Bảng theo vùng:
  - **Bắc (Hà Nội)**: gộp `d/gi/r → z`, `ch/tr → tɕ`, `s/x → s`; thanh ˧, ˨˩, ˧˥, ˧˩˧, ˧ˀ˥, ˧ˀ˨ʔ.
  - **Nam (TP.HCM)**: `d/gi → j`, `r → ʐ`, `s → ʂ`, `x → s`, `tr → ʈ`, `ch → c`; hỏi–ngã gộp `˨˩˦`, nặng `˨˩˨`, ngang `˦`.
  - **Trung**: xấp xỉ (âm đầu kiểu Nam, thanh điệu tạm theo Bắc) — chờ dữ liệu Huế, xem `docs/G2P_VIETNAMESE.md` mục 6.5.
- Nguyên âm theo Kirby: `a ă ɤ̆ ɛ e i ɔ o ɤ u ɯ`; thêm `OO = ɔː` (o dài, từ vay); vần đôi `iə uə ɯə`; không thêm trọng âm (tiếng Việt không có stress từ vựng).
- Đây vẫn là kênh nghiên cứu/đối chiếu (lexicon, round-trip, kiểm tra biến thể), **không** thay vocab ID ở runtime. Kế hoạch tích hợp sea-g2p/viG2P, WikiPron, MFA nằm ở `docs/G2P_VIETNAMESE.md`.

## OOV resolver (`frontend/oov.py`)

Quy trình 5 bước theo tinh thần viG2P (viết lại, không dùng code GPL):

1. Exact acronym / product / English fallback có sẵn.
2. Token giống từ tiếng Anh → respelling trong từ điển.
3. Token viết hoa toàn bộ (≥ 2 chữ cái, không chứa chữ số) → đánh vần bằng tên chữ cái tiếng Việt: `VNG` → "vê en giê".
4. Còn lại trả `None` → giữ `FALLBACK_REQUIRED` kèm `raw` để log và bổ sung override.

Bật khi cần bằng `parse_text(text, resolve_oov=True)`; mặc định tắt để hành vi không đổi. Các âm tiết sinh từ đánh vần có `confidence=0.5` và giữ `raw` là token gốc.
