# Nghiên cứu mô hình tách/gộp âm tiết — lý thuyết quốc tế và thực nghiệm tiếng Việt

> **Câu hỏi nghiên cứu:** (1) Có hệ thống tách âm tiết (parse) và gộp âm tiết
> (compose) nào *tối ưu hơn* mô hình 5 thành phần hiện tại của ViFastTts không?
> (2) Nếu không, parser còn bug hay trường hợp biên nào?
>
> **Trả lời ngắn:** Không có hệ ngoại lai nào tốt hơn cho bài toán của repo —
> mọi hệ thay thế đều là *phép chiếu thô hơn* của mô hình 5 thành phần (mục 6).
> Audit vét cạn + ngữ liệu 39K/74K không tìm thấy bug mới; còn lại là các giới
> hạn **có chủ đích** (parser permissive) đã được liệt kê và định lượng (mục 7).
> Cập nhật: 11/10/2026. Quy ước chữ ↔ âm vị: `docs/GLOSSARY.md`; hướng dẫn
> nhập môn: `docs/SYLLABLE_GUIDE.md`; G2P: `docs/G2P_VIETNAMESE.md`.

## 1. Tiêu chí so sánh

Một mô hình tách/gộp âm tiết cho frontend TTS tiếng Việt cần thỏa:

| # | Tiêu chí | Ý nghĩa |
| --- | --- | --- |
| C1 | **Phủ (coverage)** | Không UNK cho tiếng Việt chuẩn; nhận diện được cả cách viết hợp lệ khác nhau của cùng âm |
| C2 | **Khả nghịch (song ánh)** | `chữ → đơn vị → chữ` trả lại đúng chữ canonical (tone-position chuẩn hoá) |
| C3 | **Trung lập phương ngữ** | Một âm vị trừu tượng cho mọi vùng; khác biệt Bắc/Nam để tầng G2P xử lý |
| C4 | **Tone tách rời (autosegmental)** | Thanh là thành phần riêng, không dính vào chữ cái |
| C5 | **Kích thước hữu hạn, đếm được** | Vocab nhỏ, kiểm chứng bằng vét cạn được |
| C6 | **Chịu được OOV/từ vay** | Không crash; có đường fallback tường minh |
| C7 | **Triển khai được** | Thuật toán tất định, O(1)–O(len), dễ port Rust |
| C8 | **Nền tảng ngôn ngữ học** | Khớp phân tích được công nhận của tiếng Việt |

## 2. Mô hình hiện tại của ViFastTts (as-built)

Ba tầng: **chính tả** (`Orthography`) → **âm vị học** (`AbstractSyllable`:
`ONSET/MEDIAL/NUCLEUS/CODA/TONE`) → **ngữ âm** (`g2p.py`, một chiều).

- Inventory: **24 onset / 15 nucleus / 11 coda / 6 tone**; nucleus gồm 12 nguyên
  âm đơn + `IÊ/UÔ/ƯƠ` + `OO` (từ vay, tách riêng vì người bản ngữ phân biệt
  `xoong ≠ xong`).
- Không gian lý thuyết: **264 rime** (15 + 9 với âm đệm W, × 11 coda),
  **1.200 cặp (rime, thanh)**, **6.336 shape** (onset × rime),
  **28.800 âm tiết trừu tượng**; audit chữ viết kiểm **49.166 tổ hợp chính tả**.
- Bất biến đang được test trong CI (`tests/test_parser_matrix.py`):
  `parse → compose → parse` ổn định; đầu ra tuân bảng
  `MEDIAL_NUCLEUS_COMPATIBILITY`/`CODA_TONE_COMPATIBILITY`; coda tắc chỉ
  sắc/nặng.
- Định vị lý thuyết: đây là mô hình **2 tầng "âm tiết = âm đầu + vần + thanh;
  vần = âm đệm + âm chính + âm cuối"** (Đoàn Thiện Thuật) được triển khai thành
  5 slot, cộng thêm tách `IÊ/UÔ/ƯƠ` có alias chính tả và **thanh là thành phần
  riêng** (kiểu autosegmental).

## 3. Bức tranh lý thuyết quốc tế

### 3.1 Các mô hình cấu trúc âm tiết

| Mô hình | Nội dung | Áp dụng cho tiếng Việt? |
| --- | --- | --- |
| **Onset–rime** (Fudge 1969; Selkirk 1982; Kenstowicz 1994) | Âm tiết = onset + vần; vần = hạt nhân + coda. Vần là đơn vị của vần thơ, âm luật | Có — nhưng thô hơn mô hình 5 thành phần: mất thông tin âm đệm/âm chính/âm cuối |
| **Mora** (Hyman 1985; Hayes 1989; Otake, Hatano & Cutler 1993) | Âm tiết được đếm theo mora (Nhật); người Nhật tách lời nói theo mora chứ không theo âm tiết | Không — tiếng Việt không có trọng âm từ vựng, không đếm mora |
| **Sonority Sequencing** (Clements 1990; Parker 2002) | Độ vang tăng dần về hạt nhân | Chỉ là *điều kiện lọc*: bị phản ví dụ ngay ở tiếng Anh (s+stop; Smith 2012), và Duanmu (2009) cho thấy không có sơ đồ âm tiết "chung cho mọi ngôn ngữ" — luôn cần điều kiện riêng |
| **Maximal Onset** (Kahn 1976; Selkirk 1982) | Phụ âm giữa được gán tối đa vào onset sau | Không cần: chữ Quốc ngữ đã tách sẵn ranh giới âm tiết; và ngay tiếng Anh, giải thuật lý thuyết hay lệch người bản ngữ (Goslin & Frauenfelder 2001) |
| **CV phonology / autosegmental** (Clements & Keyser 1983; Goldsmith 1976) | Tầng CV + tầng thanh riêng | **Có** — lý do thanh là thành phần riêng, không phải thuộc tính của một chữ cái |
| **Government Phonology** (Kaye, Lowenstamm & Vergnaud 1985, 1990) | Nguyên âm rỗng, cấp phép | Học thuật; không thực dụng cho frontend |
| **Optimality Theory** (Prince & Smolensky 1993) | Xếp hạng ràng buộc | Không phải thuật toán tất định; khó kiểm chứng vét cạn |

**Nhận xét then chốt:** hầu hết lý thuyết trên giải bài toán *tìm ranh giới âm
tiết trong chuỗi âm vị liên tục* (tiếng Anh, tiếng Đức...). Tiếng Việt **không
có** bài toán đó ở tầng chữ: mỗi chữ, ngăn bởi khoảng trắng, là một âm tiết. Bài
toán đúng của frontend Việt là **giải mã cấu trúc bên trong một âm tiết viết
sẵn** và **gộp lại** — một bài toán *parsing + canonical rendering*, không phải
*syllabification*.

Các mô tả ngôn ngữ Đông Nam Á lục địa thường thêm **medial** giữa initial và
rime rồi gộp medial+nucleus+coda thành *final* (Wikipedia "Syllable", dẫn
Hannas 1997; Henderson 1966) — tức truyền thống khu vực đã dùng sẵn cách nhìn
4–5 slot mà repo triển khai, chỉ khác ở chỗ repo tách `IÊ/UÔ/ƯƠ` thành nucleus
đơn vị và tách thanh thành thành phần riêng.

### 3.2 Phương pháp tách âm tiết trong NLP

- **Rule-based** (sonority, legality, NLTK `LegalitySyllableTokenizer`): tốt khi
  ngôn ngữ có quy tắc rõ; cần bảng legality theo ngôn ngữ.
- **Data-driven** (Daelemans & van den Bosch 1992 — IGTree; các mô hình HMM/SVM/neural
  sau này): cần corpus ranh giới âm tiết; thành công với tiếng Anh/Đức vì có
  bài toán ranh giới thật.
- **Lexicon/DP** (Thái, Khmer, Lào — chữ không dấu cách): phải cắt chuỗi thành
  âm tiết bằng từ điển + quy hoạch động; đó là bài toán *segmentation* thật sự.
- **Universal tools** (Epitran, Phonemizer/espeak-ng, LingPy): lookup + luật.
  Epitran tiếng Việt dùng bảng ánh xạ theo **cả âm tiết có dấu**, không tách
  thành phần (xem mục 4.6). Các G2P neural đa ngữ (DeepPhonemizer,
  CharsiuG2P/ByT5) phủ ~100 ngôn ngữ **kể cả tiếng Việt**, nhưng đầu ra chỉ là
  chuỗi âm vị — **không mô hình hoá vị trí dấu thanh**, đúng chỗ mà frontend
  chuyên biệt của repo làm được.

Với chữ Quốc ngữ, cách đúng là **constrained decoding của một ngữ pháp nhỏ**
(24 × 264 × thanh): tất định, kiểm chứng được bằng vét cạn — đúng như repo đang làm.

**Chuẩn so sánh hiện có:** Bartlett, Kondrak & Cherry (ACL 2008; NAACL 2009)
đặt chuẩn benchmark tách âm vị; Marchand & Damper (2006), Marchand, Adsett &
Damper (2007) so sánh các giải thuật tự động cho tiếng Anh; Daelemans, van den
Bosch & Weijters (1997) hệ thống hoá IGTree. Krantz, Dulin & De Palma
(2018/2019) thừa nhận "định nghĩa âm tiết hiếm khi được đồng thuận, ngay trong
một ngôn ngữ" — điều này củng cố lựa chọn *đặc tả tường minh theo ngôn ngữ*
thay vì hy vọng một giải thuật "universal" thắng cho tiếng Việt.

### 3.3 Ngôn ngữ khác rút ra được gì

- **Trung Quốc**: chữ Hán đa âm tiết, pinyin tách sẵn theo âm tiết; tone là
  thành tố riêng của pinyin, có **thuật toán đặt dấu tường minh** (ưu tiên
  a/e; `ou`→`o`; trường hợp `iu/ui` đặc biệt) và dạng số `zhong1` — đúng là
  mô hình "onset + vần + thanh + tone-as-digit" mà công cụ Việt (vPhon,
  sea-g2p, misaki) đã vay.
- **Thái/Lào/Khmer**: không dấu cách → syllable segmentation dựa từ điển; không
  áp dụng cho tiếng Việt nhưng củng cố ý tưởng **lexicon validation**.
- **Nhật**: mora/kana — người Nhật tách lời nói theo mora chứ không theo âm
  tiết (Otake, Hatano & Cutler 1993); **Hàn**: khối Hangul — mỗi khối là một âm
  tiết và G2P Hàn tách lại thành onset/nucleus/coda jamo, tương tự trực giác
  "một chữ = một âm tiết" của tiếng Việt.
- **Tiếng Anh**: bài toán ranh giới + G2P bất quy tắc; frontend tiếng Việt *dễ
  hơn về segmentation* nhưng *khó hơn về chính tả dấu thanh* (vị trí dấu, i/y,
  c/k/q...). Chuẩn Kahn 1976 + Goslin & Frauenfelder 2001 cho thấy ngay cả
  tiếng Anh, giải thuật lý thuyết cũng hay lệch với người bản ngữ — "maximal
  onset" không phải chân lý.

### 3.4 Tổng hợp khảo sát quốc tế

1. **Boundary discovery (SSP/MOP/legality, NLTK/LingPy/neural) gần như vô hiệu
   với tiếng Việt**, vì ranh giới âm tiết trùng khoảng trắng trong chữ Quốc
   ngữ; các giải thuật đó giải bài toán tiếng Việt không có (mục 3.1–3.2).
2. **Rime+tone grouping** được lý thuyết autosegmental/mora hậu thuẫn và các
   engine độc lập hội tụ về đó: espeak-ng gộp tone vào entry vần; vPhon và
   misaki (Kokoro-vi) mã hoá trực tiếp `(C)(w)V(G|C₂)+T`.
3. **Tone-as-digit** (kiểu pinyin) là thứ công cụ Việt vay hiệu quả nhất.
4. **G2P universal chỉ nên là fallback** cho token không phải tiếng Việt.
5. **Chiều gộp** (đặt dấu thanh, Telex/VNI, NFC) **không có tool quốc tế nào
   làm thay** — đây là đóng góp riêng của frontend Việt.

Cảnh báo độ tin cậy: **không có benchmark head-to-head nào** chấm điểm các
framework này trên tách âm tiết/đặt dấu tiếng Việt; các kết luận trên dựa trên
đọc mã nguồn + tài liệu. Và việc chọn lối dấu "cũ/mới" là **quyết định chính
sách, không phải ngôn ngữ học** — mọi triển khai 5 thành phần cần để nó cấu
hình được (repo đã có `canonical_tone_spelling`).

## 4. Nghiên cứu tiếng Việt

### 4.1 Mô hình chuẩn và các biến thể

- **Đoàn Thiện Thuật, *Ngữ âm tiếng Việt*** (1977): chuẩn hóa mô hình 2 tầng
  "âm tiết = âm đầu + vần + thanh điệu; vần = âm đệm + âm chính + âm cuối"
  (theo công thức Polivanov). Sách giáo khoa hiện hành dùng đúng bộ thuật ngữ
  này (QĐ 1989/QĐ-BGDĐT 2018).
- **Thompson (1965)**: mô tả C₁VC₂ cho Hà Nội; bàn về "nuclear models" trong
  phân tích IC (Language 41).
- **Nguyễn Quang Hồng (1994)**: coi âm đệm và thanh điệu là **siêu đoạn**;
  vần tách "vần cái" (nucleus+coda); đếm 21 âm đầu, 124 vần.
- **Vương Hữu Lễ & Hoàng Dũng**: (C₁)(W)V(C₂), 8 kiểu âm tiết.
- **Kirby (2011, JIPA)**: (C)(w)V(C) + thanh; kho tư liệu Hà Nội hiện đại.
- **Cao Xuân Hạo**: phản đối mô hình đoạn tính — cho rằng [w] và thanh là
  **điệu vị phi đoạn**; đề xuất mô tả 8 thanh vị. Đây là cảnh báo lý thuyết:
  mọi biểu diễn segmental (như 5 ID) là **xấp xỉ dùng cho máy**, cần giữ thanh
  như thành phần riêng để ít sai lệch nhất.

**Repo chọn gì và vì sao:** mô hình 5 thành phần + thanh riêng = bản "duỗi"
segment của mô hình Đoàn Thiện Thuật (chuẩn sư phạm), đồng thời tôn trọng phi
đoạn tính kiểu Cao Xuân Hạo ở mức có thể cho máy (thanh tách riêng; [w] là slot
mà không phải feature gộp vào phụ âm đầu).

### 4.2 Tranh luận quan trọng

| Vấn đề | Các quan điểm | Lựa chọn của repo |
| --- | --- | --- |
| `/w/` là âm đệm hay nguyên âm đôi? | Đoàn Thiện Thuật: slot âm đệm; Cao Xuân Hạo: nét [+tròn môi]; Phạm Andrea Hòa (2009): **segment độc lập**, nhưng "triphthong không phải một đơn vị" | Slot `MEDIAL=W` tách khỏi nucleus; đuôi `j/w` là coda |
| `qu` | Phần lớn: /k/ + /w/; một số: /kw/ | Onset `qu` → K + W (chính tả gộp trong onset, âm vị tách) |
| `gi` + i | `giếng` = /z+iəŋ/; chỉ ~7 âm tiết viết `g` trước i (`gì`, `gìn`, `giết`, `giêng`, `giềng`, `giếng`, `gịa`) | GI là một onset; chữ i dùng chung (`gì` = GI + I, `giếng` = GI + IÊ + NG) |
| `i`/`y` | Một âm vị /i/, hai cách viết; QĐ 240/1984 *không* đặt quy tắc i/y; QĐ 2018 yêu cầu i sau âm đầu (`lí luận`), tên riêng tự do | Giữ nguyên như người viết (round-trip); không tự ý đổi |
| `/ɔ/`–`/ɔː/` | Haudricourt (1952): `xoong` [sɔːŋ] ≠ `xong` [sɔŋ]; Nguyễn Lân Trung (2010); nhiều giáo trình coi o/ô là hai âm vị | Nucleus `OO` riêng, tách khỏi `O` |
| Số lượng nguyên âm | 9+3 (Haudricourt 1952; Kirby 2011), 10, 11 (Thompson 1965), 14+19 (Emerich 2012) | 12 đơn + 3 đôi + OO = 15 nucleus (phủ hợp 9+3 cộng các biến thể chính tả ă/â) |
| Thanh Nam bộ | hỏi–ngã gộp (5 thanh); Phạm 2003: tiếng Việt nghiêng về **register language** (phonation) hơn là tone thuần | Giữ 6 thanh trừu tượng; gộp hỏi/ngã ở tầng G2P Nam |

### 4.3 Chính tả

- **Vị trí dấu thanh:** lối "mới" (`hoà`, `thuỷ`, `khoẻ`) được **QĐ
  1989/2018/QĐ-BGDĐT (Điều 8)** quy định cho chương trình/SGK phổ thông. Lưu ý
  phổ biến: **QĐ 240/QĐ (1984) không chứa quy tắc `hòa/hoà`** — văn bản đó chỉ
  quy định các trường hợp chính tả chưa thống nhất và tên riêng nước ngoài.
  Thực tế văn bản nhà nước trộn hai lối (Quốc hội: "cộng hoà"; cổng Chính phủ:
  "cộng hòa") → repo chuẩn hoá về lối mới nhưng có option giữ nguyên.
- **c/k/q, g/gh, ng/ngh**: phân bố bổ khuyết theo chính tả (không phải luật
  ngữ âm) — xem mục 7 về việc parser *không* enforce lớp này (có chủ đích).
- **Bùi Hiền 2017** (*Tiếq Việt*) bị phản đối rộng rãi — nhắc để thấy tiếng
  Việt không thiếu đề xuất cải cách, nhưng frontend phải bám chính tả hiện hành.

### 4.4 Ngữ âm học phân bố (phonotactics)

- `/w/` (âm đệm) **không đi sau phụ âm môi** /ɓ f v m/ trong từ bản ngữ; các từ
  như `voan`, `boa`, `moa`, `phuy` là **từ vay** (chủ yếu Pháp/Anh).
- Coda `nh/ch` chỉ sau `a, ê, i` (Thompson 1965); coda tắc `p/t/c/ch` chỉ đi
  với **sắc/nặng**.
- Danh sách vần đầy đủ: Thompson (1965), Wikipedia "Vietnamese phonology"
  (bảng theo Kirby 2011), Huỳnh Công Tín (2003) cho Sài Gòn.

### 4.5 Đếm âm tiết: lý thuyết vs thực dùng

| Nguồn | Số lượng | Ghi chú |
| --- | --- | --- |
| Hannas 1997 | ~4.500–4.800 nói được / ~6.200 viết được | trích qua Wikipedia |
| Trần Thị Minh Phương (1991) | 16.882 → 6.142 "âm tiết hành chức" | luận án, qua ngonngu.net |
| Hoàng Phê (từ điển) | 6.760 chính tả / 6.616 âm vị | Nguyen et al. 2000 |
| HCMUS CLC | 6.835 âm tiết phân biệt | từ điển học thuật |
| Vietlex / danh sách cộng đồng | 7.729 / 6.653–7.884 | tùy tiêu chí đếm |
| COCOSDA-2018 | ~19.000 khả năng / ~6.500 dùng | |
| **Repo (bảng)** | **6.336 shape / 28.800 âm tiết trừu tượng** | vét cạn được, không UNK |

Các con số xấp xỉ nhau (~6.100–7.900 tùy tiêu chí); repo nằm đúng dải này và
mạnh hơn ở chỗ **kiểm chứng vét cạn** thay vì phụ thuộc từ điển.

### 4.6 Bức tranh công cụ tiếng Việt (đối chiếu nhanh)

| Công cụ | Đơn vị tách | Thanh | Phương ngữ | Ghi chú |
| --- | --- | --- | --- | --- |
| **vPhon** (Kirby) | dict `onsets/nuclei/onglides/offglides/codas/tones` riêng | Chao / Gedney / số | N (mặc định), C, S | Tham chiếu học thuật, GPL-3.0 |
| **Viphoneme (viG2P)** | Chuỗi `/d/ɯə/k/6/` = onset+nucleus+coda + số thanh | Số cuối âm tiết | Hà Nội-biased | Build trên vPhon + Vinorm |
| **Vi_G2P** | **Tonal phoneme**: thanh gộp vào ký hiệu nguyên âm | Gộp vào âm vị | 1 phương ngữ | Cho Kaldi lexicon |
| **sea-g2p** | Từ điển + DP cho OOV, IPA + số thanh | Số | Không đổi được | Apache-2.0; dùng bởi VieNeu-TTS |
| **misaki** (Kokoro-vi) | `(C)(w)V(G|C₂)+T`, dict theo slot | Tone riêng | Bắc | Bằng chứng công nghiệp dùng đúng 5 slot |
| **Epitran vie-Latn** | Lookup **cả âm tiết có dấu** | Chao gắn vào âm tiết | Trộn | Có lỗi huyền trùng ngang |
| **espeak-ng `vi`** | Entry vần + tone digit; 3 voice | Số 1–6 | N/central/south | De-facto trong Piper/VITS-vi |
| **MeloTTS-Vi** | 45 phoneme + **8 tone ID** (embedding riêng) | Embedding riêng | Bắc | VITS2 + PhoBERT |
| **Vinorm/vietnormalizer** | Chuẩn hoá về **chuỗi âm tiết cách nhau** | — | — | Text layer |
| **vigo** (IME) | Chiều **gộp**: CVC + luật đặt dấu (Telex/VNI) | Luật vị trí dấu | — | Bản đối chiếu nghịch |
| **CharsiuG2P / DeepPhonemizer** | Chuỗi âm vị đa ngữ (có tiếng Việt) | **Không mô hình hoá** dấu thanh | — | Chỉ nên làm fallback |
| **ViSpeechFormer (2026)** | Khai thác quan hệ gần 1–1 chữ↔âm vị cho ASR | Suy từ chữ | — | Xác nhận tính "thân thiện" của chữ Quốc ngữ |

**Tổng hợp:** (1) tầng chữ gần như luôn **syllable-atomic** (mỗi âm tiết một
token); (2) tầng âm vị hội tụ về **onset + nucleus + coda + tone**, với thanh
là **số/embedding riêng**; (3) neural TTS hiện đại (VietTTS BPE, F5-TTS-Vi
char) bỏ qua tách thành phần, đổi lấy mất kiểm soát dấu thanh/phương ngữ;
(4) các hệ khác căn bản chỉ có ba: **initial+rime+tone**, **tonal phoneme**,
**whole-syllable/BPE** — tất cả đều là phép chiếu của 5 thành phần (mục 6.2).

## 5. Kiểm chứng thực nghiệm (audit mới của repo)

Ngữ liệu: `duyet/vietnamese-wordlist` (GPL-2.0; **chỉ dùng phân tích, không
vendor**): Viet39K = 39.071 dòng / **6.633 âm tiết duy nhất**; Viet74K = 73.901
dòng / **9.061 âm tiết duy nhất**. Chạy `tools/parser_audit.py` (ma trận +
wordlist) và các script audit mở rộng; kết quả:

| Kiểm chứng | Kết quả |
| --- | --- |
| Ma trận parse→compose→parse | 49.166 tổ hợp, **0 vấn đề** |
| Wordlist 39K | **6.617/6.633** parse; 16 fallback = typo/từ vay (đã phân loại) |
| Wordlist 74K | 1.382 fallback — chủ yếu từ vay kỹ thuật (`axit`, `album`...), từ dân tộc (`hđrung`, `kpă`...), mojibake, chữ cái đơn; **không có từ Kinh chuẩn nào bị rơi** |
| Rime ngữ liệu dùng | 180/264 rime "sinh được"; **0 rime ngoài bảng** |
| Shape (onset × rime) | 2.663/6.336 (42%) xuất hiện; **637 shape (23%) chỉ 1 từ** (hợp 39K+74K) — bằng chứng cần lexicon cho dữ liệu train |
| Cặp (onset, nucleus) chưa từng gặp | 6: `D+OO, GI+Ê, NH+OO, P+Ư, TR+OO, Đ+OO` (OO chỉ sống trong vài từ vay: xoong, boong, moóc...) |
| Onset × W | Mọi onset đều có từ với W trong 74K, nhưng **nhóm môi cực hiếm**: B=4, M=5, PH=1, V=2, P=2 — đúng giới hạn ngữ âm môi+/w/ (toàn từ vay) |
| Coda × nucleus | `NH`/`CH` chỉ gặp sau `A, I, Ê`; `W` coda không gặp sau O/Ô/UÔ/Ơ — khớp Thompson 1965 |
| Chữ HOA / NFD | **0 lệch** trên toàn bộ 9.061 âm tiết (case + Unicode chuẩn hoá) |
| Round-trip compose | 90 lệch — **tất cả** là chuẩn hoá vị trí dấu (`hòa→hoà`) hoặc chữ hỏng dấu kép; không có lệch cấu trúc |
| G2P sweep | Mọi tổ hợp hợp lệ × 3 phương ngữ × broad/narrow × 3 kiểu thanh: **0 lỗi**; 63 ký hiệu IPA |
| Biên hỗn hợp | `a-xít`, `pa-lăng`, `In-đô-nê-xi-a` tách đúng thành âm tiết; `!!!`, chuỗi rỗng không crash; CJK/Cyrillic/Thái giữ nguyên fallback |
| Dấu thanh kép | `hòá` bị từ chối; `hòà` (cùng một dấu) canonical hoá về `hoà` |

Các script audit có thể tái chạy: `tools/parser_audit.py [--wordlist FILE]`,
`pytest tests/test_parser_matrix.py tests/test_parser_edges.py`.

## 6. Trả lời: có hệ thống nào tối ưu hơn không?

### 6.1 Bảng so sánh các lựa chọn biểu diễn

| Biểu diễn | Đơn vị/âm tiết | Ví dụ `khuỷu`, `quốc` | Nghịch đảo chữ được? | Phủ chuẩn? | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| **5 thành phần (repo)** | 5 slot + thanh | `KH+W+I+W+HOI`, `K+W+Ô+C+SAC` | Có (cần alias chính tả) | Có | Tối đa thông tin, kiểm chứng vét cạn |
| Onset + vần + thanh | 3 đơn vị (vần gộp) | `KH + uyu + HOI` | Được nhưng vần nhiều (hàng trăm) | Có | Nghiên cứu ASR 2026 dùng để giảm lỗi cấp vần |
| Phoneme string + tone | 3–5 âm vị | `kʰ w i w ˧˩˧` | Không (mất chính tả) | Có | espeak-ng/sea-g2p |
| Tonal phoneme | tone gộp vào ký hiệu | `kʰ wi w4` | Không | Có | Vi_G2P (Kaldi lexicon) |
| Cả âm tiết | 1 token | `khuỷu` (ID) | Có nếu có từ điển | **Không** (6.000+ token, OOV) | Epitran lookup |
| Char/BPE | ký tự/subword | `k h u ỷ u` | Có | Có | VietTTS, F5-TTS-Vi — nhưng mất kiểm soát dấu thanh/phương ngữ |

### 6.2 Luận điểm "độ mịn + phép chiếu" (vì sao 5 thành phần là tối ưu)

Mọi biểu diễn khác là **hàm chiếu tất định** từ 5 thành phần:

```text
5 thành phần ──merge(medial,nucleus,coda)──▶ onset + vần + thanh
             ──merge(tone vào nucleus)─────▶ tonal phoneme
             ──merge(tất cả)───────────────▶ whole-syllable
             ──expand(IÊ→iə, UÔ→uə, ƯƠ→ɯə)─▶ phoneme string + tone
```

Chiều ngược lại **không khôi phục được** thông tin đã gộp (ví dụ whole-syllable
không biết âm đệm là gì; tonal phoneme không tách thanh ra để đổi quy ước).
Vì vậy 5 thành phần là **điểm mịn nhất còn hữu hạn và neo theo chính tả**; mọi
hệ khác đều là "view" của nó. Đây là cơ sở lý thuyết cho tính tối ưu — không
phải sở thích.

### 6.3 Khi nào nên dùng view khác

- **BPE/char** cho LM sinh văn bản thuần: không cần kiểm soát phát âm.
- **Onset+vần+thanh** cho mô hình prosody/thanh điệu cấp vần, hoặc khi cần
  chuỗi ngắn hơn (một nghiên cứu ASR 2026 đã dùng để đo lỗi theo thành phần).
- **Phoneme string** khi cần so khớp với hệ bên ngoài (sea-g2p, espeak) — repo
  đã có `g2p.py` làm projection này.
- **5 thành phần** khi cần kiểm soát dấu thanh, phương ngữ, OOV, và dataset
  QA — đúng use case của repo.

### 6.4 Giới hạn bằng chứng (trung thực khoa học)

Khảo sát tooling (mục 4.6) **không tìm thấy benchmark công khai nào**
so sánh trực tiếp unit frontend (char/BPE vs phoneme vs rime vs 5-slot) cho
tiếng Việt. Vì vậy không thể khẳng định "chất lượng TTS của 5 thành phần cao
hơn" — chỉ có thể khẳng định **tính đúng đắn, khả nghịch và coverage** của nó.
Đề xuất thí nghiệm A/B nằm ở mục 8.

## 7. Bug và trường hợp biên còn lại

### 7.1 Kết luận săn bug

**Không tìm thấy bug mới** trong parser/composer sau: vét cạn ma trận, 9.061 âm
tiết ngữ liệu, chữ HOA/NFD, dấu kép, gạch nối, G2P sweep, biên hỗn hợp. Các
điểm dưới đây là **hành vi có chủ đích** (permissive) hoặc **việc nên siết ở
tầng ứng dụng**, cần ghi nhận thay vì "fix" ngầm.

### 7.2 Per-missive có chủ đích (không phải bug)

1. **Rime "sinh được" nhưng chưa gặp** (84 loại): `E+CH/NH`, `IÊ+CH/NH`,
   `O+CH/NH/W`, `OO+CH/M/N/NH/P/T/W`, `U+CH/NH`, `UÔ+CH/NH/P/W`,
   `Â+CH/NH`, `Ê+J`, `Ô+CH/NH/W`, `Ă+CH/NH`, `Ơ+C/CH/NH/W`, `Ư+CH/NH/P`,
   `ƯƠ+CH/NH`, và các tổ hợp `W+...`. Đây là "không gian lý thuyết rộng hơn
   thực dùng" — quyết định có chủ đích để không chặn từ mới/từ vay.
2. **6 cặp (onset, nucleus) chưa từng gặp**: `D+OO, GI+Ê, NH+OO, P+Ư, TR+OO,
   Đ+OO` — phần lớn vì `OO` chỉ sống trong vài từ vay; `GI+Ê` không tồn tại
   trong tiếng Việt (`dê` mới là /ze/). Không cần luật cứng.
3. **Phân bố chính tả c/k/q, g/gh, ng/ngh không enforce**: parser nhận `cen`,
   `ka`, `kơi` (toàn từ ngoại/ngoại lệ trong ngữ liệu). Muốn siết: dùng profile
   strict (7.3).
4. **Từ vay vi phạm ngữ âm học bản ngữ**: `boa`, `voan`, `moa`, `phuy`,
   `poa`... (môi + /w/). Cố ý nhận để đọc từ vay; lớp strict có thể cảnh báo.
5. **Chính tả "lỏng" khác**: `iên` (đứng một mình, chuẩn là `yên`), `tuyn`,
   `ơt`; tone lỡ nằm trên phụ âm/âm đệm vẫn được tái định vị (`qúy → quý`,
   `tủi` giữ đúng vì `ui` = U+J); `hòà` cùng-một-dấu canonical hoá, `hòá`
   khác-dấu bị từ chối.

### 7.3 Đề xuất siết (strict profile — chưa triển khai)

Cơ chế: sinh **lexicon** từ ngữ liệu + luật, rồi bật kiểm tra tùy chọn:

```text
check_lexicon(syllable, profile) với profile ∈ {open, strict, corpus}
  open   : như hiện tại (mặc định)
  strict : + onset×(nucleus) hợp lệ; + c/k/q, g/gh, ng/ngh đúng phân bố;
           + giới hạn môi+/w/ (trừ khi token là từ vay có trong override)
  corpus : + chỉ nhận shape đã có trong lexicon (dùng cho QA dataset train)
```

Strict không ảnh hưởng model input (vẫn 7 ID); nó chỉ là bộ lọc cảnh báo/log.
Đây là con đường duy nhất còn lại để tăng độ chặt mà không phá coverage.

### 7.4 Biên đã kiểm và đạt

- Chữ HOA có dấu (`HÒA`, `ĐƯỜNG`, `QUỐC`) — 0 lệch abstract.
- NFD (macOS gõ dấu tách rời) — parse qua `normalize_text` NFC toàn chuỗi.
- Từ vay có gạch nối (`a-xít`, `pa-lăng`, `In-đô-nê-xi-a`) — tách đúng âm tiết.
- Chữ ngoài Latin (CJK, Cyrillic, Thái) — giữ nguyên fallback, không nuốt.
- Token số/emoji/URL/email — đi qua normalizer, không crash.
- `hòà` (một dấu dán hai lần) → `hoà`; `hòá` (hai dấu khác) → từ chối.
- Coda tắc + thanh bằng/hỏi/ngã (`ooc`, `moc`) → từ chối; `moóc`, `mọc` → nhận.

### 7.5 Ngoài phạm vi parser (ghi nhận cho roadmap)

- **G2P Trung** đang xấp xỉ (thanh Bắc, âm đầu Nam); Huế cần dữ liệu riêng.
- **Nam bộ**: `OO`/`O` gộp trong khẩu ngữ (`xoong`/`xong` nghe gần nhau) —
  bảng G2P Nam chưa thêm biến thể; `ưu/ươu` Hà Nội ngả [iw/iəw] đã ghi chú.
- **Segment Expander** (IÊ→i+ə, UÔ→u+ə, ƯƠ→ɯ+ə) chưa implement — cần cho
  duration cấp âm vị.
- Đối thủ cũng có lỗi riêng đáng biết: Epitran vie-Latn có **huyền trùng ngang**
  (`à→a˧`, `á→a˧`) — lý do không thể mặc định tin lookup bên ngoài.
- **Không có benchmark frontend** công khai → cơ hội nghiên cứu (mục 8).

## 8. Đề xuất cập nhật roadmap

1. **Strict/lexicon profile** (7.3) — ưu tiên 1; dùng Viet39K/74K + lexicon dự án.
2. **Rime projection** công khai (`abstract.rime`), kèm test đối chiếu với biểu
   diễn "onset + vần + thanh" cho mô hình prosody.
3. **Segment Expander** (đã có trong roadmap) — nối 5 thành phần với phoneme
   string; là "view" chuẩn cho so khớp hệ ngoài.
4. **Dialect G2P**: thêm biến thể Nam cho `OO`, ghi chú `ưu/ươu`; muốn đủ Huế
   cần WikiPron `hue` (~30.242 mục) hoặc MFA Huế.
5. **Benchmark A/B frontend** (char/BPE vs phoneme vs vần vs 5-slot) trên cùng
   corpus (VAIS-1000/PhoAudiobook) với normalizer cố định — đóng gap học thuật.
6. Duy trì audit trong CI: `tools/parser_audit.py` + `pytest`.

## 9. Kết luận

1. **Không** có hệ thống tách/gộp ngoại lai nào tối ưu hơn cho tiếng Việt so
   với mô hình 5 thành phần của repo: mọi hệ khác là phép chiếu thô hơn (6.2).
2. **Không** tìm thấy bug mới; các giới hạn còn lại là permissive có chủ đích,
   đã định lượng, và có thể siết bằng strict profile (7.2–7.3).
3. Việc cần làm tiếp không phải "đổi mô hình" mà là **thêm lớp** (lexicon,
   dialect, expander) và **thêm bằng chứng** (benchmark).

## 10. Nguồn tham khảo

### Tiếng Việt — âm vị học & chính tả

1. Đoàn Thiện Thuật, *Ngữ âm tiếng Việt* (1977) — mô hình 2 tầng:
   <https://vnu.edu.vn/ngu-am-tieng-viet-cua-gs-doan-thien-thuat-post12501.html>
2. QĐ 1989/2018/QĐ-BGDĐT, *Quy định về chính tả trong chương trình, SGK phổ
   thông* (Điều 8 — vị trí dấu thanh):
   <https://lophocnguvan.com/quy-dinh-ve-chinh-ta-trong-chuong-trinh-sach-giao-khoa-giao-duc-pho-thong/>
3. QĐ 240/QĐ (1984) — văn bản gốc (không chứa quy tắc hòa/hoà):
   <https://luatvietnam.vn/giao-duc/quyet-dinh-240-qd-1984-ve-chinh-ta-tieng-viet-va-ve-thuat-ngu-tieng-viet-160366-d1.html>
4. L. C. Thompson, *A Vietnamese Grammar* (1965):
   <https://eric.ed.gov/?id=ED015448>; "Nuclear Models in Vietnamese
   Immediate-Constituent Analysis", *Language* 41(4): 610–618, DOI 10.2307/411528.
5. J. P. Kirby, "Vietnamese (Hanoi Vietnamese)", *JIPA* 41(3), 2011:
   <https://doi.org/10.1017/S0025100311000181>
6. Cao Xuân Hạo — phi đoạn tính của âm vị: <https://ngonngu.net/avh_bieudien3/72>;
   cấu trúc âm tiết: <https://ngonngu.net/tietvi_cautruc/117>
7. Phạm, Andrea Hòa, "The identity of non-identified sounds…" *TWPL* 34 (2009):
   <https://twpl.library.utoronto.ca/index.php/twpl/article/download/6803/3844/10796>
8. Phạm, Andrea Hòa, *Vietnamese Tone: A New Analysis* (Routledge, 2003):
   <https://www.routledge.com/Vietnamese-Tone-A-New-Analysis/Horn-Pham/p/book/9780415861366>
9. Emerich, Giang Huong, *The Vietnamese Vowel System* (PhD, UPenn, 2012):
   <https://eric.ed.gov/?id=ED552129>
10. Haudricourt (1952) — tái bản của Ferlus & Michaud về nguyên âm ngắn,
    gồm `xoong`/`xong`:
    <https://shs.hal.science/halshs-01631474/file/Haudricourt1952_ShortVowels.pdf>
11. Nguyễn Lân Trung, *Synergies Pays riverains du Mékong* 2 (2010):
    <http://gerflint.fr/Base/Mekong2/nguyen_lan_trung.pdf>
12. Vietnamese phonology / Vietnamese alphabet (Wikipedia — bảng theo Kirby,
    Thompson, Huỳnh Công Tín):
    <https://en.wikipedia.org/wiki/Vietnamese_phonology>,
    <https://en.wikipedia.org/wiki/Vietnamese_alphabet>
13. Đào Tiến Thi — chuyện i/ngắn-y dài:
    <http://chimviet.free.fr/baivo/daotienthi/daotienthin_ChuyenINganYDai.htm>
14. Trần Thị Minh Phương (1991) — đếm âm tiết hành chức:
    <https://ngonngu.net/tietvi_cautruc/117>
15. HCMUS CLC — thống kê từ điển/âm tiết:
    <https://www.clc.hcmus.edu.vn/?page_id=229>
16. Trần & Vallée (2009), *JSEALS* 1 — phân bố CVC 69,47%:
    <https://www.gipsa-lab.grenoble-inp.fr/~nathalie.vallee/publications.html>

### Tiếng Việt — công cụ & ngữ liệu

17. vPhon (Kirby): <https://github.com/kirbyj/vPhon>
18. Viphoneme (viG2P): <https://github.com/v-nhandt21/Viphoneme>
19. Vi_G2P (tonal phoneme): <https://github.com/v-nhandt21/Vi_G2P>
20. sea-g2p: <https://github.com/pnnbao97/sea-g2p>
21. Epitran: <https://github.com/dmort27/epitran>
22. espeak-ng `vi` rules: `dictsource/vi_rules` (kho mã nguồn espeak-ng)
23. MeloTTS-Vi: <https://github.com/manhcuong02/MeloTTS_Vietnamese>
24. Vinorm: <https://github.com/v-nhandt21/Vinorm>;
    vietnormalizer: <https://github.com/nghimestudio/vietnormalizer>
25. vigo (IME, chiều gộp): <https://github.com/haidinhtuan/vigo>
26. VietTTS: <https://github.com/dangvansam/viet-tts>
27. VITS-vi / sherpa-onnx: <https://github.com/phatjkk/vits-tts-vietnamese>
28. F5-TTS Viet: <https://github.com/hynt/F5-TTS-Vietnamese-100h>
29. VietVoice-TTS: <https://github.com/nguyenvulebinh/VietVoice-TTS>
30. duyet/vietnamese-wordlist (GPL-2.0, chỉ phân tích):
    <https://github.com/duyet/vietnamese-wordlist>
31. Danh sách âm tiết cộng đồng:
    <https://github.com/vietnameselanguage/syllable>
32. "Syllabic-Structure Decoder for ASR in Vietnamese" (initial+rhyme+tone):
    arXiv 2605.27874

### Quốc tế — lý thuyết & công cụ

33. Fudge, "Syllables", *Journal of Linguistics* (1969); Selkirk, *The Syntax
    of Words* (1982); Kenstowicz, *Phonology in Generative Grammar* (1994).
34. Clements, "The role of the sonority cycle in core syllabification" (1990);
    Parker, *Quantifying the Sonority Hierarchy* (PhD, 2002); Smith, "The
    obstruent sonority paradox…" (2012):
    <https://users.castle.unc.edu/~jlsmith/home/pdf/smith2012_mfm20_handout.pdf>;
    Duanmu, *Syllable Structure: The Limits of Variation* (OUP, 2009):
    <https://academic.oup.com/book/41740>
35. Goldsmith, *Autosegmental Phonology* (MIT diss., 1976):
    <https://people.cs.uchicago.edu/~jagoldsm/Papers/dissertation.pdf>
36. Clements & Keyser, *CV Phonology* (MIT Press, 1983); Kahn,
    *Syllable-based Generalizations in English Phonology* (1976):
    <https://seas.elte.hu/w/!syllables/_media/kahn76-thesis.pdf>; Goslin &
    Frauenfelder (2001) — giải thuật lý thuyết lệch người bản ngữ (dẫn trong
    tài liệu NLTK).
37. Hyman, *A Theory of Phonological Weight* (1985); Hayes,
    "Compensatory Lengthening in Moraic Phonology" (1989):
    <https://brucehayes.org/papers/HayesCompensatoryLengthening1989.pdf>;
    Otake, Hatano & Cutler (1993):
    <https://doi.org/10.1006/jmla.1993.1014>; Labrune, *The Phonology of
    Japanese* (2012).
38. Kaye, Lowenstamm & Vergnaud (1990):
    <https://doi.org/10.1017/S0952675700001184>; Prince & Smolensky, *Optimality
    Theory* (1993); Féry & van de Vijver (eds.), *The Syllable in Optimality
    Theory* (2003): <https://doi.org/10.1017/CBO9780511497926>
39. Daelemans, van den Bosch & Weijters, "IGTree" (1997):
    <https://doi.org/10.1023/A:1006506017891>; Bartlett, Kondrak & Cherry (ACL
    2008): <https://aclanthology.org/P08-1065/>; *On the Syllabification of
    Phonemes* (NAACL 2009): <https://aclanthology.org/N09-1035/>; Marchand &
    Damper (2006): <https://doi.org/10.1017/S1351324905004043>; Marchand,
    Adsett & Damper (2007): <https://eprints.soton.ac.uk/264285/>; Krantz,
    Dulin & De Palma (2018): <https://arxiv.org/abs/1807.05518> và (2019)
    <https://arxiv.org/abs/1909.13362>; Tits (2023):
    <https://doi.org/10.18653/v1/2023.emnlp-industry.8>
40. NLTK `LegalitySyllableTokenizer` / `SyllableTokenizer`:
    <https://www.nltk.org/api/nltk.tokenize.syllable.html> (port từ syllabipy:
    <https://github.com/henchc/syllabipy>); LingPy `syllabify`:
    <https://lingpy.org/reference/lingpy.sequence.html>
41. Epitran — Mortensen, Dalmia & Littell (LREC 2018):
    <https://github.com/dmort27/epitran>; Phonemizer — Bernard & Titeux (2021):
    <https://doi.org/10.21105/joss.03958>; CharsiuG2P — Zhu, Zhang & Jurgens
    (2022): <https://arxiv.org/abs/2204.03067>; DeepPhonemizer:
    <https://github.com/as-ideas/DeepPhonemizer>
42. Pinyin — tone digit + thuật toán đặt dấu: Wikipedia "Pinyin"; pypinyin:
    <https://github.com/mozillazg/python-pinyin>
43. Thai — Chormai, Prasertsom & Cheevaprawatdomrong (COLING 2020):
    <https://doi.org/10.18653/v1/2020.coling-main.407>; PyThaiNLP:
    <https://github.com/PyThaiNLP/pythainlp>
44. Khmer/Lào/Burmese — khmercut: <https://github.com/seanghay/khmercut>;
    chamkho: <https://github.com/veer66/chamkho>; Srithirath & Seresangtakul
    (2013): <https://doi.org/10.1109/ECTICON.2013.6559585>; sylbreak:
    <https://github.com/ye-kyaw-thu/sylbreak>
45. Syllable-based TTS — Kopeček (1997); Do, Tran & Nguyen (2011):
    <https://doi.org/10.1145/2069216.2069249>; Phung, Luong & Akagi (2012)
46. misaki (Kokoro TTS, frontend Vi theo đúng 5 slot):
    <https://github.com/hexgrad/misaki>
47. UGTPHON (2026) — benchmark G2P cho văn bản người dùng, có tiếng Việt:
    <https://arxiv.org/pdf/2609.27205>
48. ViSpeechFormer (2026) — quan hệ gần 1–1 chữ↔âm vị:
    arXiv 2602.10003; Nhan, Tri & Nam (NICS 2020):
    <https://doi.org/10.1109/NICS51282.2020.9335905>; Dang, Vuong & Phan
    (2022): <https://arxiv.org/abs/2209.02971>
49. Quy tắc đặt dấu thanh (lối cũ/mới, `ê/ơ` luôn nhận dấu, `gi/qu` là một
    chữ): <https://vi.wikipedia.org/wiki/Quy_tắc_đặt_dấu_thanh_của_chữ_Quốc_ngữ>;
    Unicode L2/24-111 (dấu ngã/perispomene):
    <https://www.unicode.org/L2/L2024/24111-vietnamese-apex.annot.pdf>
