# Hướng dẫn cấu trúc âm tiết tiếng Việt (cho người không chuyên ngôn ngữ)

> Tài liệu này dành cho kỹ sư, sinh viên hoặc bất kỳ ai muốn hiểu vì sao
> ViFastTts tách mỗi âm tiết thành 5 thành phần `âm đầu – âm đệm – âm chính –
> âm cuối + thanh điệu`. Không cần nền ngôn ngữ học. Thuật ngữ đầy đủ và
> nguồn gốc khái niệm: `docs/GLOSSARY.md`.

## 1. "Tiếng" = một âm tiết

Tiếng Việt viết tách từng tiếng: **"Hôm nay trời đẹp" có 4 tiếng = 4 âm tiết**.
Mỗi tiếng đọc lên được chia thành 5 ô:

```text
[âm đầu] [âm đệm] [âm chính] [âm cuối]  +  thanh điệu
 có thể    có thể    BẮT BUỘC    có thể      luôn có (ngang nếu không dấu)
 trống     trống                 trống
```

- Ô **âm chính** luôn tồn tại — đó là hạt nhân của âm tiết.
- Bốn ô còn lại có thể trống; trong dự án ghi là `NONE`.
- Đây là mô hình chuẩn của ngôn ngữ học (xem `docs/GLOSSARY.md`), không phải
  quy ước riêng của ViFastTts.

## 2. Năm ô là gì

| Ô | Còn gọi là | Nội dung | Ví dụ |
| --- | --- | --- | --- |
| **âm đầu** | phụ âm đầu | b, m, ph, v, t, th, đ, n, l, x, s, d, gi, r, ch, tr, nh, ng/ngh, g/gh, kh, h, k/c/q, p | `m` trong "ma"; trống trong "ăn", "uống" |
| **âm đệm** | bán âm `/w/` | viết bằng "o"/"u" trong `oa, oă, oe, uâ, uê, uy, qu` | "hoa" có `W`; đa số tiếng không có |
| **âm chính** | nguyên âm hạt nhân | a ă â e ê i/y o ô ơ u ư; bộ đôi `iê/yê/ia/ya`, `uô/ua`, `ươ/ưa` | `a` trong "ma" (không bao giờ trống) |
| **âm cuối** | phụ âm cuối hoặc bán âm cuối | m, n, ng, nh, p, t, c, ch; `i/y` (= `/j/`), `u/o` (= `/w/`) | `n` trong "an"; trống trong "ma", "hoa" |
| **thanh điệu** | 6 thanh | ngang (không dấu), huyền, hỏi, ngã, sắc, nặng | "ma" (ngang), "má" (sắc), "mạ" (nặng) |

## 3. Cách tách một tiếng trong 5 bước

Ví dụ thực hành với **"quốc"**:

1. **Tách thanh điệu**: nhìn dấu → "quốc" có dấu sắc trên `ô` ⇒ thanh **sắc**; bỏ dấu còn `quôc`.
2. **Tìm âm đầu**: lấy phụ âm đầu **dài nhất** ở đầu tiếng (ưu tiên `ngh, ng, nh, ch, gh, gi, kh, ph, qu, th, tr` trước khi xét phụ âm đơn) ⇒ `qu` (đọc là `k` + `w`).
3. **Tìm âm cuối**: lấy phụ âm cuối dài nhất ở cuối (`nh, ng, ch` trước, rồi `m, n, p, t, c`); nếu không có thì xét bán âm cuối `i, y, u, o` ⇒ `c`.
4. **Phần còn lại là vần**: `uô` — mở đầu bằng `u` nhưng `u` ở đây là **âm đệm** (vì `qu` tạo `/w/`), còn lại `ô` là **âm chính**.
5. **Ghi 5 ô**: `(K, W, Ô, C, sắc)`.

## 4. Những điểm "hack não" cần nhớ

- **`qu` = `k` + `w`**: "quốc" = `K W Ô C`; không tách thành `q` + `uông`.
- **`gi` dùng chung chữ `i`**: "già" = `GI` + `A`; "giếng" = `GI` + `IÊ` + `NG` (i thuộc vần);
  "gì" = `GI` + `I`. Quy tắc: sau `gi`, nếu gặp `ê`, hết tiếng, hoặc phụ âm thì `i` thuộc vần;
  nếu gặp nguyên âm khác (`a, u, ư, o...`) thì `i` câm ("già", "giữ", "giường").
- **`uy` khác `ui`**: "thuỷ" = `W` + `I` (dấu trên `y`); "tủi" = `U` + `J` (dấu trên `u`).
- **`i/y` cuối đều là `/j/`**: "tai" = `A` + `J` (a dài), "tay" = `Ă` + `J` (ă ngắn) — khác ở **nguyên âm**, giống ở âm cuối; tương tự "rao" ≠ "rau", "mai" ≠ "may".
- **`anh/ach` khác `an/ac`**: âm cuối là `nh/ch`, không phải `n/c`.
- **`oo` trong từ vay**: "xoong", "boong", "moóc" — chữ `oo` là **một nguyên âm o dài** `[ɔː]`
  viết đôi (không phải nguyên âm đôi như `iê/uô/ươ`), và được tách thành **nucleus `OO` riêng**
  (khác `O` thường) vì người bản ngữ phân biệt "xoong" ≠ "xong"; dấu đặt trên chữ `o` thứ hai ("moóc").
- **Dấu thanh theo chuẩn hiện đại**: "hoà", "thuỷ", "tuỳ" (dấu trên `a/y`) thay cho lối cũ
  "hòa", "thủy", "tùy" — dự án luôn chuẩn hoá về lối hiện đại.

### Vì sao "hoà" đánh dấu trên `a` mà không phải trên `o`?

Theo hệ thống âm vị, **dấu thanh thuộc về âm chính (nucleus)**, không thuộc âm đệm.
Trong "hoa":

```text
  h   +   o   +   a
  │       │       └── ÂM CHÍNH A (nguyên âm hạt nhân)  ← dấu thanh nằm ở đây
  │       └────────── ÂM ĐỆM W /w/ (bán âm, viết bằng o/u)
  └────────────────── ÂM ĐẦU H
```

Âm tiết là bộ âm vị `(H, W, A, NONE, HUYEN)`; `compose_canonical` áp dấu lên âm chính
`A` → **hoà**. Lối cũ "hòa" đặt dấu lên âm đệm — đọc lên **giống hệt** (cùng bộ âm vị)
nhưng khác quy ước ghi. Các nhóm tương tự:

| Vần | Dấu đặt trên | Ví dụ |
| --- | --- | --- |
| `oa`, `oe`, `uy` | chữ thứ hai (`a/e/y` = âm chính) | hoà, khoẻ, thuỷ |
| `qu` | nguyên âm sau `u` (`u` thuộc âm đầu K+W) | quả, quý |
| `ia`, `ua`, `ưa` (mở) | chữ đầu (`i/u/ư`) | bìa, mùa, chửa |
| `iê`, `uô`, `ươ` | `ê/ô/ơ` | tiếng, muốn, tưởng |
| `oo` (từ vay) | chữ `o` thứ hai | moóc |

Đây là lý do **parser chấp nhận cả hai lối viết** (để không mất chữ gốc) nhưng
normalizer luôn xuất lối hiện đại; có thể giữ nguyên lối người viết bằng
`NormalizerOptions(canonical_tone_spelling=False)` (xem `docs/NORMALIZER.md`).

## 5. Bảng ví dụ đầy đủ (sinh trực tiếp từ parser của dự án)

| Tiếng | Âm đầu | Âm đệm | Âm chính | Âm cuối | Thanh |
| --- | --- | --- | --- | --- | --- |
| `ăn` | NONE | NONE | Ă | N | ngang |
| `uống` | NONE | NONE | UÔ | NG | sắc |
| `hoa` | H | W | A | NONE | ngang |
| `khuya` | KH | W | IÊ | NONE | ngang |
| `giếng` | GI | NONE | IÊ | NG | sắc |
| `quốc` | K | W | Ô | C | sắc |
| `anh` | NONE | NONE | A | NH | ngang |
| `xoong` | X | NONE | OO | NG | ngang |
| `nghiêng` | NG | NONE | IÊ | NG | ngang |
| `chuyện` | CH | W | IÊ | N | nặng |
| `khuy` | KH | W | I | NONE | ngang |
| `tủi` | T | NONE | U | J | hỏi |
| `thuỷ` | TH | W | I | NONE | hỏi |
| `già` | GI | NONE | A | NONE | huyền |
| `gì` | GI | NONE | I | NONE | huyền |
| `ướp` | NONE | NONE | ƯƠ | P | sắc |

## 6. Sáu thanh điệu

| Tiếng | Thanh | Dấu |
| --- | --- | --- |
| ma | ngang | không dấu |
| mà | huyền | dấu huyền |
| mả | hỏi | dấu hỏi |
| mã | ngã | dấu ngã |
| má | sắc | dấu sắc |
| mạ | nặng | dấu nặng |

Thứ tự nhớ theo SGK: **ngang – huyền – hỏi – ngã – sắc – nặng**.

## 7. Bài tập (đáp án ở mục 5)

Tách 5 ô cho các tiếng: `nghiêng`, `chuyện`, `khuy`, `hòa`, `ướp`, `giặt`.

<details>
<summary>Bấm xem đáp án</summary>

| Tiếng | Âm đầu | Âm đệm | Âm chính | Âm cuối | Thanh |
| --- | --- | --- | --- | --- | --- |
| `nghiêng` | NG | NONE | IÊ | NG | ngang |
| `chuyện` | CH | W | IÊ | N | nặng |
| `khuy` | KH | W | I | NONE | ngang |
| `hòa` | H | W | A | NONE | huyền |
| `ướp` | NONE | NONE | ƯƠ | P | sắc |
| `giặt` | GI | NONE | Ă | T | nặng |

</details>

## 8. Liên hệ với dự án

- Mỗi âm tiết trong code mang 3 dạng: chính tả, **dạng âm vị học** (5 ô), và
  dạng ngữ âm (IPA). Ví dụ "giếng" → bộ âm vị `(GI, NONE, IÊ, NG, SẮC)` →
  IPA Bắc `z iə ŋ ˧˥`.
- Model neural **không** nhận chữ hay IPA, chỉ nhận ID số của 5 ô
  (ví dụ `[9, 0, ...]`); nhãn chữ chỉ để người đọc. Xem `docs/GLOSSARY.md`.
- Ô trống ghi `NONE`; âm tiết chưa tách được (số, từ ngoại lai) giữ trạng thái
  `FALLBACK_REQUIRED`.

## 9. Đọc thêm

- Wikipedia (EN), *Syllable* — mục *Chinese segmentation* mô tả mô hình
  initial–medial–final + tone: <https://en.wikipedia.org/wiki/Syllable>
- Wikipedia (EN), *Vietnamese phonology* — âm vị học tiếng Việt (Hà Nội/Sài Gòn):
  <https://en.wikipedia.org/wiki/Vietnamese_phonology>
- Wikipedia (VI), *Âm tiết*: <https://vi.wikipedia.org/wiki/Âm_tiết>
- L. C. Thompson, *A Vietnamese Reference Grammar* (1965; tái bản 1988, University of
  Hawai'i Press): <https://uhpress.hawaii.edu/title/a-vietnamese-reference-grammar/>
- Đoàn Thiện Thuật, *Ngữ âm tiếng Việt*, NXB ĐHQG Hà Nội (giáo trình đại học;
  không có bản điện tử chính thức — tra thư viện khi cần trích dẫn)
- Clements & Keyser (1983), *CV Phonology: A Generative Theory of the Syllable* —
  nền tảng lý thuyết σ/onset/rime/nucleus/coda
- `docs/GLOSSARY.md`, `docs/FRONTEND.md`, `docs/G2P_VIETNAMESE.md` — tài liệu nội bộ
