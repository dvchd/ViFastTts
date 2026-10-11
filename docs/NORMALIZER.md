# Normalizer theo vùng miền

Normalizer nhận `north`, `central` hoặc `south`. Profile chỉ đổi những từ đọc số và viết tắt có khác biệt thực tế:

- Bắc: `nghìn`, `linh`, ưu tiên `tư` sau hàng chục.
- Trung: `ngàn`, `lẻ`, ưu tiên `tư`.
- Nam: `ngàn`, `lẻ`, mặc định `bốn`.

`triệu` và `tỷ` dùng chung. Parser âm tiết không nhận dialect và không gộp các đối lập chính tả.

## Chính tả dấu thanh

Normalizer dùng vị trí dấu trên âm chính:

```text
hòa → hoà
hóa → hoá
thủy → thuỷ
tùy → tuỳ
hủy → huỷ
qúy → quý
```

Quy tắc hiện đại: `oa/oe/uy` đặt dấu trên chữ thứ hai (`hoà`, `khoẻ`, `thuỷ`);
`qu` coi `u` thuộc âm đầu nên dấu rơi trên nguyên âm sau (`quỳ`, `quý`).
Dấu thanh **chỉ** nằm trên chữ nguyên âm; parser bóc dấu ở bất kỳ đâu rồi
`compose_canonical` đặt lại đúng âm chính (kể cả khi dấu lỡ nằm trên phụ âm).

Cả hai lối viết được parser chấp nhận, nhưng output normalizer chỉ có một dạng canonical.

## Số và nhóm số

- Nhóm ba chữ số có hàng trăm bằng 0 được đọc đúng chuẩn: `1.005.000` → "một triệu không trăm linh năm nghìn", `1.050.000` → "một triệu không trăm năm mươi nghìn", `1.010.101` → "một triệu không trăm mười nghìn một trăm linh một".
- Số lớn đọc đệ quy theo `tỷ`: `2.500.000.000.000` → "hai nghìn năm trăm tỷ"; số ≥ 10^12 không còn crash.
- Dấu phân cách nghìn kiểu Việt (`.` và khoảng trắng) được xử lý trước; `3.14` vẫn là số thập phân.

## Ngày, giờ, phân số, khoảng

| Input | Output |
| --- | --- |
| `25/12/2023`, `25-12-2023`, `25.12.2023` | "ngày hai mươi lăm tháng mười hai năm hai nghìn không trăm hai mươi ba" |
| `ngày 25/12/2023` | giữ đúng một chữ "ngày" |
| `14:30` | "mười bốn giờ ba mươi phút" |
| `14:05` | "mười bốn giờ linh năm phút" |
| `14h30`, `18h` | "mười bốn giờ ba mươi phút", "mười tám giờ" |
| `2/3` | "hai phần ba" |
| `3-5`, `10-20%` | "ba đến năm", "mười đến hai mươi phần trăm" |
| `1.500-2.000` | "một nghìn năm trăm đến hai nghìn" |

Ngày không hợp lệ (`2024-13-40`) và giờ không hợp lệ (`14:99`) được giữ nguyên để rơi vào fallback thay vì đọc sai. Phân số được xử lý trước phần trăm để `1/2%` không bị cắt nửa chừng.

## Fallback tiếng Anh

Từ điển tích hợp chỉ dành cho từ công nghệ thường gặp. Mọi dự án nên có `english_overrides` riêng. Fallback không phải G2P tiếng Anh tổng quát và không được dùng để đoán tên riêng không biết.

Bất biến được kiểm bằng test: **mọi reading trong từ điển phải parse được thành âm tiết Việt hợp lệ** (nếu không, pipeline sẽ rơi lại vào `FALLBACK_REQUIRED` ngay sau khi chuẩn hoá). Các reading lỗi cũ như `blu tút`, `brao dơ`, `eo` đã được sửa.

Thế hệ mạng di động `5G`/`4G` đọc là "năm giê"/"bốn giê".

## Thứ tự phân loại token

```text
exact override
→ exact uppercase acronym
→ exact product name
→ canonical uppercase Roman numeral
→ Vietnamese syllable
→ case-insensitive English fallback
→ unchanged token
```

Quy tắc này phân biệt:

```text
AI → ây ai
Ai → Ai
ai → ai
VI → sáu
Vi → Vi
vi → vi
```

Số La Mã chỉ nhận dạng chuỗi viết hoa chuẩn có ít nhất hai ký tự, trong khoảng 1 đến 3999. Dạng sai như `IIII`, `VX`, `IC`, `VIVI` không được chuyển. Có thể tắt bằng `roman_numerals=False` hoặc override từng token.

## Giữ chữ số

Tokenizer của pipeline giữ nguyên chữ số trong token thay vì nuốt mất: `RTX3080` → một token fallback `RTX3080`, `v2` → `v2`. Các token này giữ `raw` để debug và có thể xử lý bằng OOV resolver (`resolve_oov=True`) nếu phù hợp.

## Biến thể chính tả tự do i/y (qui/quy, mĩ/mỹ, lí/lý)

Cách viết `i`/`y` sau phụ âm đầu là điểm **chưa thống nhất** của chính tả tiếng Việt hiện đại:

- **Quyết định 240/QĐ (1984)** của Bộ Giáo dục — *Quy định về chính tả tiếng Việt và thuật ngữ tiếng Việt* — là văn bản quy phạm đầu tiên về chính tả.
- **Quyết định 1989/2018/QĐ-BGDĐT** (chính tả trong Chương trình, SGK phổ thông) yêu cầu viết `lí luận`, `mĩ thuật` (dùng **i**).
- **Nghị định 30/2020/NĐ-CP** (công tác văn thư, đang áp dụng cho văn bản hành chính) — phụ lục có ví dụ `Triều Lý`, `kỷ niệm` (dùng **y**).
- Hai nhóm quy định còn mâu thuẫn nhau; trường hợp điển hình là `Qui Nhơn` hay `Quy Nhơn` đều xuất hiện trong văn bản hành chính và tài liệu ngôn ngữ học.
- Ngoài ra, chữ Quốc ngữ dùng `y` cho bán nguyên âm trong `ay/áy`, và `i` trong các tổ hợp như `iê` (`uyên`, `yên`, `yêu`...); nhưng thực tế vẫn song song `mỹ lệ`/`mị dân`...

**Chính sách của ViFastTts:** parser/normalizer **giữ nguyên** lựa chọn i/y của người viết (chỉ chuẩn hoá vị trí dấu thanh), để round-trip bảo toàn chính tả; không tự ý đổi `qui`↔`quy` vì cả hai đang hợp lệ theo các văn bản khác nhau. Khi cần một quy ước duy nhất cho runtime, hãy xử lý ở tầng ứng dụng/override.

Nguồn: Wikipedia tiếng Việt, *Chính tả tiếng Việt* (dẫn QĐ 240/QĐ 1984; QĐ 1989/2018/QĐ-BGDĐT; NĐ 30/2020/NĐ-CP) — <https://vi.wikipedia.org/wiki/Chính_tả_tiếng_Việt>.
