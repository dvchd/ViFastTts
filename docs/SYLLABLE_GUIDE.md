# Hướng dẫn cấu trúc âm tiết tiếng Việt (cho người không chuyên ngôn ngữ)

> Tài liệu này dành cho kỹ sư, sinh viên hoặc bất kỳ ai muốn hiểu vì sao
> ViFastTts tách mỗi âm tiết thành 5 thành phần `âm đầu – âm đệm – âm chính –
> âm cuối + thanh điệu`. Không cần nền ngôn ngữ học. Thuật ngữ đầy đủ và
> nguồn gốc khái niệm: `docs/GLOSSARY.md`.
>
> Bản 16/10/2026: chỉnh theo góp ý rà soát của ThS Đinh Thị Hằng
> (Phòng Việt ngữ học, Viện Ngôn ngữ học) — bảng tách 4 tầng "thành phần →
> âm vị → con chữ → ví dụ"; sửa `quốc` = /k/ + /uô/ + /k/; bổ sung âm đệm
> `uơ/uyê/uya`; ghi chú `anh/ach` theo Đoàn Thiện Thuật.

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

## 2. Âm vị và con chữ

Các thành phần âm tiết được **cấu tạo từ các âm vị**; con chữ chỉ là cách
*thể hiện* âm vị đó. Âm vị được ghi giữa hai gạch chéo (`/f/`, `/z/`…), còn
con chữ để trong ngoặc kép. Một âm vị có thể được viết bằng nhiều con chữ:

| Âm vị | Con chữ | Ví dụ |
| --- | --- | --- |
| `/f/` | `ph` | phở |
| `/z/` | `d`, `gi` (giọng Bắc; Nam Bộ: `d` đọc /j/) | da, già |
| `/ŋ/` | `ng`, `ngh` | ngà, nghé |
| `/ɣ/` | `g`, `gh` | gà, ghế |

→ Khi đọc bảng dưới, luôn tách hai tầng: **âm vị** (cái máy dùng) và
**con chữ** (cái người viết).

## 3. Năm ô là gì

| Thành phần âm tiết | Âm vị cấu tạo | Con chữ thể hiện | Ví dụ |
| --- | --- | --- | --- |
| **âm đầu** (phụ âm đầu) | các âm vị phụ âm (như `/f/ /z/ /ŋ/ /ɣ/`…) + **âm tắc thanh hầu /ʔ/** không có con chữ | b, m, ph, v, t, th, đ, n, x, s, d, gi, l, tr, r, ch, nh, c/k/q, ng/ngh, kh, g/gh, h — thêm: `p` chỉ trong từ vay ("đèn pin", "sâm panh"; một số tác giả không xếp vào hệ thống) | `m` trong "ma"; "ăn", "anh", "in" mở đầu bằng /ʔ/ (không viết chữ nào) |
| **âm đệm** | bán âm **/w/** | "o", "u" trong `oa, oă, oe, uâ, uê, uy, uơ, uyê, uya`; "u" trong `qu` khi nguyên âm sau **không tròn môi** (qua, quê, quy, quân, quơ…) | "hoa" (H W A); "huơ"; "khuyên"; "khuya"; đa số tiếng không có |
| **âm chính** | **11 nguyên âm đơn** + **3 nguyên âm đôi** (mỗi âm vị đôi gồm hai thành phần nhưng có giá trị một âm vị — xem §5) | đơn: a, ă, â, e, ê, i/y, o, ô, ơ, u, ư (`i` và `y` cùng một âm vị /i/) — đôi: `iê/ia/yê/ya` = /iə/; `uô/ua` = /uə/; `ươ/ưa` = /ɯə/ — thêm: `oo` ("xoong") là âm o dài từ vay, **không phải** âm đôi | "ma" (A); "biển", "mía" (IÊ); "muốn", "mùa" (UÔ); "mương", "mưa" (ƯƠ) |
| **âm cuối** | phụ âm cuối `/m n ŋ p t k/` + bán âm cuối `/j w/` (`nh`, `ch` là **biến thể** của /ŋ/, /k/ — xem §5) | m, n, ng, nh, p, t, c, ch; `i/y` (= /j/); `u/o` (= /w/) | "an" (N); "tai" (J); "rau" (W) |
| **thanh điệu** | 6 thanh vị | ngang (không dấu), huyền, hỏi, ngã, sắc, nặng | ma, mà, mả, mã, má, mạ |

## 4. Cách tách một tiếng trong 5 bước

Ví dụ thực hành với **"quốc"**:

1. **Tách thanh điệu**: nhìn dấu → "quốc" có dấu sắc trên `ô` ⇒ thanh **sắc**; bỏ dấu còn `quôc`.
2. **Tìm âm đầu**: chữ `q` (đọc /k/; `q` luôn đi với `u`) ⇒ âm đầu là **K**.
3. **Tìm âm cuối**: lấy phụ âm cuối dài nhất ở cuối tiếng (`nh, ng, ch` trước, rồi `m, n, p, t, c`) ⇒ `c`.
4. **Phần còn lại là vần**: `uô` — chú ý: `u` ở đây **không phải âm đệm**, vì âm đệm /w/ không đứng trước âm tròn môi (u, ô, o); `uô` là **nguyên âm đôi** /uə/ (giống "cuốc", "muốn").
5. **Ghi 5 ô**: `(K, NONE, UÔ, C, sắc)`.

> Nếu tách "quốc" = `qu` + `ô` + `c` (u làm âm đệm) thì phạm luật "âm đệm /w/
> không kết hợp với âm tròn môi" — và sẽ tách khác "cuốc", trong khi "quốc"
> **đồng âm với "cuốc"** (xem §5).

## 5. Những điểm "hack não" cần nhớ

- **Âm đệm /w/ không kết hợp với nguyên âm tròn môi (u, ô, o)** — nó chỉ đứng
  trước âm **không tròn môi**: `a ă â e ê i/y ơ`. Đây là chìa khoá của các ca bên dưới.
- **`qu` = /k/ + /w/** khi sau `u` là âm **không tròn môi**: `qua`, `quăn`,
  `quân`, `que`, `quê`, `quy`, `quơ`, `quất` (= `K W Â T`). Khi sau `qu` là
  **`ô`** (tròn môi): chữ `u` thuộc vần — **`quốc` = K + UÔ + C**, không tách
  `K W Ô C`; "quốc" đồng âm "cuốc".
- **`uô` là một khối, `uâ`/`uơ` tách được**: nhìn nguyên âm sau chữ `u` —
  gặp **`ô`** (tròn môi) thì `u` thuộc **nguyên âm đôi** (`muốn`, `mùa`,
  `quốc`); gặp âm **không tròn môi** thì `u` là **âm đệm /w/** (`tuần` =
  T W Â N; `huơ` = H W Ơ). Bằng chứng: "muốn", "buổi", "vua" — `uô` đứng được
  sau m, b, v, còn /w/ thì không.
- **`gi` dùng chung một chữ `i`**: `gi` = /z/. Nếu vần cũng bắt đầu bằng /i/
  thì hai chữ i trùng nhau, chính tả chỉ ghi **một**: `gì` = GI + I,
  `giếng` = GI + IÊ + NG (cùng vần với "tiếng"). Còn `già` = GI + A: chữ i
  nằm trong âm đầu, không đọc riêng. (Cách nhớ: "cái gì", "giữ gìn".)
- **`uy` khác `ui`**: "thuỷ" = `W` + `I` (dấu trên `y`); "tủi" = `U` + `J` (dấu trên `u`).
- **`i/y` cuối đều là `/j/`**: "tai" = `A` + `J` (a dài), "tay" = `Ă` + `J` (ă ngắn) — khác ở **nguyên âm**, giống ở âm cuối; tương tự "rao" ≠ "rau", "mai" ≠ "may".
- **`anh/ach`**: theo Đoàn Thiện Thuật (*Ngữ âm tiếng Việt*), `a` trong `anh/ach`
  là một âm vị **e ngắn**, còn `nh/ch` là **biến thể** của /ŋ/, /k/ (không phải
  /n/, /c/). SGK truyền thống tách `anh` = a + nh; dự án giữ cách SGK cho ổn định
  chính tả — nhưng "anh" khác "an" (xem bảng ví dụ §6).
- **`oo` trong từ vay**: "xoong", "boong", "moóc" — chữ `oo` là **một nguyên âm o dài** `[ɔː]`
  viết đôi (không phải nguyên âm đôi như `iê/uô/ươ`), và được tách thành **nucleus `OO` riêng**
  (khác `O` thường) vì người bản ngữ phân biệt "xoong" ≠ "xong"; dấu đặt trên chữ `o` thứ hai ("moóc").
- **Dấu thanh theo chuẩn hiện đại**: "hoà", "thuỷ", "tuỳ" (dấu trên `a/y`) thay cho lối cũ
  "hòa", "thủy", "tùy" — dự án luôn chuẩn hoá về lối hiện đại (có thể tắt). Lối này được
  **Quyết định 1989/QĐ-BGDĐT (Điều 8)** quy định cho SGK. Ngoài đời cả hai lối cùng
  tồn tại (chưa có quy định thống nhất cấp quốc gia; công cụ gõ như UniKey cho chọn cả hai);
  với dữ liệu/tìm kiếm, cách gõ ảnh hưởng đến kết quả — vì vậy dự án chuẩn hoá
  `hoà/thuỷ/tuỳ` nhưng có tuỳ chọn giữ nguyên (`NormalizerOptions(canonical_tone_spelling=False)`;
  xem `docs/NORMALIZER.md`).

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
| `uâ`, `uê`, `uơ` | nguyên âm sau `u` (`u` là âm đệm) | quân, huế, thuở |
| `qu` | nguyên âm sau `u` (trừ "quốc" — xem §5) | quả, quý |
| `ia`, `ua`, `ưa` (mở) | chữ đầu (`i/u/ư`) | bìa, mùa, chửa |
| `iê`, `uô`, `ươ` | `ê/ô/ơ` | tiếng, muốn, tưởng |
| `oo` (từ vay) | chữ `o` thứ hai | moóc |

Đây là lý do **parser chấp nhận cả hai lối viết** (để không mất chữ gốc) nhưng
normalizer luôn xuất lối hiện đại; có thể giữ nguyên lối người viết bằng
`NormalizerOptions(canonical_tone_spelling=False)` (xem `docs/NORMALIZER.md`).

## 6. Bảng ví dụ đầy đủ (sinh trực tiếp từ parser của dự án)

| Tiếng | Âm đầu | Âm đệm | Âm chính | Âm cuối | Thanh |
| --- | --- | --- | --- | --- | --- |
| `ăn` | NONE | NONE | Ă | N | ngang |
| `uống` | NONE | NONE | UÔ | NG | sắc |
| `hoa` | H | W | A | NONE | ngang |
| `khuya` | KH | W | IÊ | NONE | ngang |
| `giếng` | GI | NONE | IÊ | NG | sắc |
| `quốc` | K | NONE | UÔ | C | sắc |
| `quất` | K | W | Â | T | sắc |
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

## 7. Sáu thanh điệu

| Tiếng | Thanh | Dấu |
| --- | --- | --- |
| ma | ngang | không dấu |
| mà | huyền | dấu huyền |
| mả | hỏi | dấu hỏi |
| mã | ngã | dấu ngã |
| má | sắc | dấu sắc |
| mạ | nặng | dấu nặng |

Thứ tự nhớ theo SGK: **ngang – huyền – hỏi – ngã – sắc – nặng**.

## 8. Bài tập (đáp án ở mục 6)

Tách 5 ô cho các tiếng: `nghiêng`, `chuyện`, `khuy`, `hòa`, `ướp`, `giặt`, `quốc`, `quất`.

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
| `quốc` | K | NONE | UÔ | C | sắc |
| `quất` | K | W | Â | T | sắc |

</details>

## 9. Liên hệ với dự án

- Mỗi âm tiết trong code mang 3 dạng: chính tả, **dạng âm vị học** (5 ô), và
  dạng ngữ âm (IPA). Ví dụ "giếng" → bộ âm vị `(GI, NONE, IÊ, NG, SẮC)` →
  IPA Bắc `z iə ŋ ˧˥`; **"quốc" → `(K, NONE, UÔ, C, SẮC)` → `k uə k ˧˥`**
  (giống hệt "cuốc" — hai từ đồng âm).
- Model neural **không** nhận chữ hay IPA, chỉ nhận ID số của 5 ô
  (ví dụ `[9, 0, ...]`); nhãn chữ chỉ để người đọc. Xem `docs/GLOSSARY.md`.
- Ô trống ghi `NONE`; âm tiết chưa tách được (số, từ ngoại lai) giữ trạng thái
  `FALLBACK_REQUIRED`.

## 10. Đọc thêm

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
