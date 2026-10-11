# Thuật ngữ (Glossary)

Một nơi duy nhất để tra cứu tên gọi các tầng biểu diễn, tránh nhầm giữa
**chữ viết**, **âm vị** và **ngữ âm**.

## Khái niệm tổng quát và nguồn gốc

`ONSET, MEDIAL, NUCLEUS, CODA, TONE` không phải quy ước riêng của dự án — đây là
**mô hình chuẩn của ngôn ngữ học** cho âm tiết:

- Tên khái niệm phổ quát: **cấu trúc âm tiết** (syllable structure) / **thành tố âm tiết** (syllable constituents).
- Mô hình phương Tây (tổng quát): **âm đầu – vần** (onset–rime), vần = âm chính + âm cuối.
  Ký hiệu chuẩn: σ (âm tiết), ω (onset), ρ (rime), ν (nucleus), κ (coda), τ (tone).
- Mô hình Đông Nam Á / Trung Quốc (áp cho tiếng Việt): **initial + medial + final + tone**,
  trong đó *final* = medial + nucleus + coda. Ký hiệu: ι (initial), μ (medial), φ (final), τ (tone).
- Cách gọi tiếng Việt (chuẩn ngữ văn phổ thông): **âm tiết = âm đầu + vần + thanh điệu**;
  **vần = âm đệm + âm chính + âm cuối**.

| Dự án | Mô hình ĐNÁ | Cách gọi tiếng Việt | Mô hình phương Tây |
| --- | --- | --- | --- |
| `ONSET` | initial ι | âm đầu | onset ω |
| `MEDIAL` | medial μ | âm đệm | (nằm trong onset của mô hình Tây) |
| `NUCLEUS` | nucleus ν | âm chính | nucleus ν |
| `CODA` | coda κ | âm cuối | coda κ |
| `TONE` | tone τ | thanh điệu | tone (suprasegmental, không thuộc onset/rime) |

**Điểm dễ nhầm:** "vần" tiếng Việt = *final* của mô hình ĐNÁ (gồm cả âm đệm),
**không đồng nhất với "rime"** trong mô hình phương Tây (rime chỉ gồm âm chính + âm cuối).
Vì vậy tài liệu dự án dùng:

- "cấu trúc âm tiết (âm đầu–đệm–chính–cuối + thanh điệu)" khi nói **khái niệm**;
- "bộ âm vị 5 thành phần" khi nói **giá trị** của một âm tiết.

Nguồn tham khảo:

- Wikipedia (EN), *Syllable* — mục *Chinese segmentation*: mô hình initial–medial–final + tone. <https://en.wikipedia.org/wiki/Syllable>
- Wikipedia (EN), *Vietnamese phonology* — Kirby (2011), Thompson (1965) cho âm vị học tiếng Việt. <https://en.wikipedia.org/wiki/Vietnamese_phonology>
- Wikipedia (VI), *Âm tiết* — khái niệm chung, dẫn Clements & Keyser (1983). <https://vi.wikipedia.org/wiki/Âm_tiết>
- Thompson (1965), *A Vietnamese Reference Grammar* (tái bản 1988, University of Hawai'i Press): <https://uhpress.hawaii.edu/title/a-vietnamese-reference-grammar/>
- Giáo trình *Ngữ âm tiếng Việt* (Đoàn Thiện Thuật; ĐH Tổng hợp Hà Nội 1977, tái bản NXB ĐHQG Hà Nội)
  và SGK Ngữ văn phổ thông — mô hình "âm đầu + vần + thanh điệu", "vần = âm đệm + âm chính + âm cuối".
  (Giáo trình chuẩn phổ biến; khi trích dẫn học thuật nên đối chiếu bản in — không có bản điện tử chính thức.)
- Clements & Keyser (1983), *CV Phonology: A Generative Theory of the Syllable*; Chomsky & Halle (1968);
  Kahn (1976); Selkirk (1982) — lý thuyết cấu trúc âm tiết.
- **Hướng dẫn học tập cho người không chuyên:** `docs/SYLLABLE_GUIDE.md` (cách tách 5 bước, ví dụ, bài tập).

## Ba tầng biểu diễn của một âm tiết

Mỗi `ParsedSyllable` mang đồng thời ba dạng của cùng một âm tiết:

| Tầng | Tên tài liệu | Tên tiếng Anh | Trong code | Ví dụ: "giếng" |
| --- | --- | --- | --- | --- |
| 1 | **Dạng chính tả** | Orthography | `Orthography` | onset `gi`, nucleus `iê`, coda `ng`, dấu sắc trên `ê` |
| 2 | **Dạng âm vị học** | Phonological form | `AbstractSyllable` (`ParsedSyllable.abstract`) | `(GI, NONE, IÊ, NG, SẮC)` |
| 3 | **Dạng ngữ âm (IPA)** | Phonetic form | `frontend/g2p.py` | Bắc `z iə ŋ ˧˥`, Nam `j iə ŋ ˧˥` |

Quy tắc dùng tên:

- Nói "**dạng âm vị học**" (thay cho "trừu tượng"/"abstract") khi muốn nói tới tầng 2.
- Nói "**bộ âm vị 5 thành phần**" (hoặc ngắn: "bộ âm vị") cho giá trị 5 trường.
- Nói "**dạng ngữ âm**" khi nói tới IPA; không gọi nó là "phiên âm" chung chung.
- "**Âm tiết**" (`ParsedSyllable`) là *vật chứa* cả ba dạng, không phải tầng 2.

## Năm thành phần — dùng thẳng thuật ngữ ngữ văn

| Nhãn trong code | Tiếng Việt (chuẩn ngữ văn) | Tiếng Anh |
| --- | --- | --- |
| `ONSET` | âm đầu | onset |
| `MEDIAL` | âm đệm | medial |
| `NUCLEUS` | âm chính | nucleus |
| `CODA` | âm cuối | coda |
| `TONE` | thanh điệu | tone |

## Nhãn âm vị (giá trị category)

- Kiểu viết: chữ in hoa, có thể kèm dấu tiếng Việt: `K`, `W`, `IÊ`, `UÔ`, `ƯƠ`, `NG`, `SAC`...
- `NONE` = âm tiết không có thành phần đó (ví dụ âm tiết mở không có `CODA`).
- Mỗi nhãn thuộc **đúng một bảng**, và **vai trò quyết định nhãn**:
  - `K` (âm đầu) vs `C` (âm cuối) — cùng `/k/` nhưng khác vị trí;
  - `I` (âm chính) vs `J` (bán âm cuối `/j/`);
  - `U` (âm chính) vs `W` (bán âm cuối `/w/`).
- Nhãn **không phải IPA**: chúng là category âm vị học trung lập phương ngữ,
  được giữ ổn định để làm khoá vocab cho checkpoint/ONNX/Rust.
  Xem lý do đầy đủ ở `docs/ARCHITECTURE.md` và `docs/FRONTEND.md`.

## Bảng đối chiếu nhanh code ↔ tài liệu

| Code | Tài liệu gọi là |
| --- | --- |
| `ParsedSyllable` | âm tiết (vật chứa 3 dạng + trạng thái parse) |
| `Orthography` | dạng chính tả |
| `AbstractSyllable` | dạng âm vị học (đề xuất đổi tên code: `PhonologicalForm`) |
| `AbstractSyllable.onset/medial/nucleus/coda/tone` | âm đầu / âm đệm / âm chính / âm cuối / thanh điệu |
| `ComponentVocabs` | vocab năm thành phần (ID số cho model) |
| `frontend/g2p.py` | dạng ngữ âm (IPA, một chiều, theo vùng) |
| `ParseStatus.FALLBACK_REQUIRED` | âm tiết chưa chuẩn hoá được (chữ số, từ ngoại lai, URL...) |

## Tránh dùng

- "trừu tượng"/"abstract" một mình — quá chung, dễ lẫn với lớp lập trình.
- "mã âm tiết" — dễ hiểu nhầm thành ID số của model.
- "khuôn âm tiết" — chỉ nói cấu trúc, không nói đây là các category âm vị.
- "phiên âm" khi đang nói về dạng âm vị học — phiên âm là việc của IPA (tầng 3).
