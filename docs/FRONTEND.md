# Frontend tiếng Việt

## Mục tiêu

Bảo toàn cấu trúc chính tả và tạo biểu diễn trừu tượng không phụ thuộc vùng miền.

## Cấu trúc

```text
ONSET + MEDIAL + NUCLEUS + CODA + TONE
```

## Quy tắc quan trọng

- Chỉ loại dấu thanh, không làm mất ă â ê ô ơ ư.
- Longest match cho onset và coda.
- Giữ D, GI, R riêng.
- Giữ TR, CH riêng.
- Giữ S, X riêng.
- Giữ HỎI, NGÃ riêng.
- `qu` ánh xạ K + W.
- `i/y` cuối có thể ánh xạ J.
- `u/o` cuối có thể ánh xạ W.
- `iê/yê/ia/ya`, `uô/ua`, `ươ/ưa` dùng nucleus trừu tượng chung.

## Lưu ý

Parser tham chiếu trong repository bao phủ các quy tắc lõi và bộ test tối thiểu. Trước khi huấn luyện production, cần mở rộng bảng âm tiết hợp lệ bằng corpus lớn và thêm test cho tên riêng, từ vay mượn, chính tả cũ và trường hợp hiếm.
