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
```

Cả hai lối viết được parser chấp nhận, nhưng output normalizer chỉ có một dạng canonical.

## Fallback tiếng Anh

Từ điển tích hợp chỉ dành cho từ công nghệ thường gặp. Mọi dự án nên có `english_overrides` riêng. Fallback không phải G2P tiếng Anh tổng quát và không được dùng để đoán tên riêng không biết.


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

Số La Mã chỉ nhận dạng chuỗi viết hoa chuẩn có ít nhất hai ký tự, trong khoảng 1 đến 3999. Dạng sai như `IIII`, `VX`, `IC`, `VIVI` không được chuyển. Ký tự đơn `I`, `V`, `X`, `L`, `C`, `D`, `M` được giữ nguyên để tránh xung đột với tên biến, mục lục và chữ viết tắt. Có thể tắt bằng `roman_numerals=False` hoặc override từng token.
