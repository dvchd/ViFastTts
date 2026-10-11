# STEM Normalizer

Lớp STEM chạy trước normalizer số và từ vựng.

## Toán và LaTeX

Hỗ trợ `$...$`, `$$...$$`, `\\(...\\)`, `\\[...\\]`, phân số, căn, số mũ, chỉ số, chữ Hy Lạp, tổng, tích phân, toán tử so sánh, tập hợp và ngoặc.

## Hoá học

Hỗ trợ `\\ce{...}`, `\\chem{...}` và công thức độc lập như `H2O`, `CO2`, `NaCl`, `H2SO4`, `Ca(OH)2`. Nguyên tố được đọc theo tên tiếng Việt và chỉ số được đọc bằng number normalizer.

## Vật lý và đơn vị

Hỗ trợ số kèm đơn vị SI phổ biến, lũy thừa và đơn vị ghép: `m/s`, `m/s2`, `m2`, `m3`, `Hz`, `Pa`, `J`, `W`, `V`, `A`, `Ω`, `°C`, `mol`, `L`.

## Ranh giới an toàn

- Nên dùng delimiters LaTeX cho biểu thức toán phức tạp.
- Nên dùng `\\ce{}` cho công thức hoá học hiếm hoặc dễ nhập nhằng.
- Có thể tắt toàn bộ bằng `NormalizerOptions(stem_formulas=False)`.
- Đây là normalizer đọc thành lời, không phải hệ đại số máy tính và không tính kết quả biểu thức.

## Guard chống nhận nhầm (mới)

- Công thức độc lập chỉ được đọc khi có **≥ 2 nguyên tố đã biết**, hoặc là đơn chất hai/ba nguyên tử phổ biến (`O2`, `O3`, `H2`, `N2`, `F2`, `Cl2`, `Br2`, `I2`), hoặc là ion có điện tích tường minh (`Fe3+`, `O2-`, `NH4+`).
- Mã sản phẩm/phiên bản được giữ nguyên: `MP3`, `N95`, `K2`, `B2B`, `RTX3080`, `v2`.
- Trong `\\ce{}`/`\\chem{}` (người dùng chỉ định rõ) guard được nới để đọc mọi công thức.
- Token chứa chữ số không bị tokenizer nuốt mất; chúng thành fallback thô nếu không parse được.
