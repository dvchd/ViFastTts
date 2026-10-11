# G2P tiếng Việt và khu vực Đông Nam Á

> Ghi chú phương pháp: mọi URL trong tài liệu này đã được fetch trực tiếp ngày **11/10/2026**, trừ các trường hợp ghi rõ "chưa xác minh". Số liệu (số sao, phiên bản, ngày phát hành) là giá trị quan sát tại thời điểm truy cập, không phải giá trị cố định. Với sea-g2p, ngoài việc đọc README/PyPI, tài liệu còn cài đặt thử bản 0.10.0 vào virtualenv riêng và chạy kiểm chứng thực nghiệm (kết quả ghi ở mục 2).

## 1. Tổng quan và tiêu chí đánh giá

Tiếng Việt có chính tả gần như ngữ âm: một chữ cái/đồ thị thường ứng với tối đa một âm vị, và ngược lại. Vì vậy G2P tiếng Việt chủ yếu là bài toán **rule-based + từ điển**, không phải bài toán học máy phức tạp như tiếng Anh. Các khó khăn thật sự nằm ở:

- **Thanh điệu**: 6 thanh, phải được mã hóa rõ ràng trong output (IPA kèm dấu thanh, chữ số 1–6, hay Chao tone letters).
- **Biến thể phương ngữ Bắc/Trung/Nam**: hỏi/ngã, r/d/gi, s/x, tr/ch, vần iê/ia, uô/ua, ươ/ưa...
- **Từ ngoài từ điển (OOV)**: tên riêng, từ vay mượn, viết tắt, tên sản phẩm, teencode/user-generated text.
- **Code-switching Việt–Anh**: xử lý cụm tiếng Anh xen giữa câu tiếng Việt mà không phá vỡ ranh giới âm tiết.
- **Định dạng output**: IPA thô, IPA có tách âm tiết, hay mã riêng của từng hệ.

Tiêu chí đánh giá được dùng trong tài liệu: cách tiếp cận, định dạng input/output, bộ âm vị, xử lý thanh điệu, xử lý phương ngữ, xử lý OOV/từ vay/viết tắt, giấy phép, chất lượng/tốc độ, và khả năng tái sử dụng trong một TTS frontend Python.

Bối cảnh Đông Nam Á: **sea-g2p** bao phủ Việt – Thái – Indonesia (kèm code-switching tiếng Anh); Thái còn có FastThaiG2P (arXiv 2608.12814, xác minh link) và model MFA Thai; Indonesia ngoài sea-g2p có thể dùng rule-based riêng. Tài liệu này tập trung vào tiếng Việt, sea-g2p và viG2P theo yêu cầu.

## 2. sea-g2p

### 2.1 Nhận dạng và liên kết

| Mục | Giá trị |
| --- | --- |
| Repo | https://github.com/pnnbao97/sea-g2p |
| Tác giả | Phạm Nguyễn Ngọc Bảo (pnnbao97) |
| PyPI | https://pypi.org/project/sea-g2p/ |
| Giấy phép | Apache-2.0 (hiển thị trên trang GitHub; PyPI khai báo "Apache Software License") |
| Đo tại 11/10/2026 | 113 sao, 23 fork; bản mới nhất 0.10.0 phát hành 23/09/2026; release đầu 0.1.0 ngày 09/03/2026 |
| Ngôn ngữ | vi, th, id (tiếng Anh là code-switching tích hợp cho cả ba) |
| Bài báo | **Không tìm thấy paper chính thức**: truy vấn arXiv API cho "sea-g2p" và "VieNeu" không trả kết quả; tìm kiếm web cũng không có. Nguồn tài liệu duy nhất là README và PyPI. |

sea-g2p là engine phonemization của VieNeu-TTS (https://github.com/pnnbao97/VieNeu-TTS, Apache-2.0, 2.820 sao tại 11/10/2026).

### 2.2 Cài đặt (đã kiểm chứng)

```bash
pip install sea-g2p          # Python >= 3.10, wheel dựng sẵn cho macOS arm64/x86_64,
                             # Linux x86_64/aarch64, Windows; abi3 nên chạy trên 3.10+
```

Đã cài thử `sea-g2p==0.10.0` vào venv Python 3.12 ngày 11/10/2026: chạy được ngay, không cần từ điển ngoài (wheel ~28 MB, từ điển nhị phân `sea_g2p.bin` đi kèm).

Có thêm **C ABI** cho host C/C++ (`include/sea_g2p.h`, `sea_g2p_open`, `sea_g2p_phonemize`, `sea_g2p_normalize`...) hoặc build Rust:

```bash
cargo build --release --no-default-features --features capi
```

### 2.3 Cách dùng và định dạng input/output

```python
from sea_g2p import SEAPipeline
pipeline = SEAPipeline(lang="vi")
pipeline.run("Giá SP500 hôm nay là 4.200,5 điểm.")
# -> 'zˈaːɜ ˈɛɜt̪ pˈe nˈam tʃˈam hˈom nˈaj lˌaː2 bˈoɜn ŋˈi2n hˈaːj tʃˈam fˈəɪ4 nˈam ɗˈiɛ4m.'
```

- API công khai xác minh bằng `dir()`: `SEAPipeline(lang).run(text|list)`, `Normalizer(lang).normalize(text|list)`, `G2P(lang).convert(...)` và `G2P.phonemize_batch(...)`. Truyền list sẽ tự chạy song song (Rayon).
- **Input**: văn bản UTF-8 thô, có sẵn normalizer 17 tầng cho tiếng Việt (số, ngày, đơn vị, công thức, viết tắt...). Có thẻ điều khiển `<en>...</en>` (giữ nguyên cho bộ đọc tiếng Anh) và `<math>...</math>`.
- **Output**: chuỗi IPA, **mỗi âm tiết cách nhau một dấu cách** — thuận lợi cho TTS dạng âm tiết. Không có dấu câu đánh dấu ranh giới âm tiết ngoài khoảng trắng.

Ví dụ thực nghiệm (chạy bản 0.10.0, 11/10/2026):

| Input | Output |
| --- | --- |
| `ma má mà mả mã mạ` | `mˈaː mˈaːɜ mˌaː2 mˈaː4 mˈaː5 mˈaː6` |
| `Tôi tên là Nguyễn Văn A` | `t̪ˈoj t̪ˈen lˌaː2 ŋwˈiɛ5n vˈan ˈaː` |
| `Zalo Facebook iPhone TikTok` | `zˈɑːloʊ fˈeɪsbʊk ˈaɪ fˈoʊn tˈɪk tˈɑːk` |
| `Công ty cổ phần VNG` | `kˈoŋ t̪ˈi kˈo4 fˈə2n vˈe nˈəː2 ɣˈəː2` (đánh vần V-N-G) |
| `blorptang` (từ giả) | `bˈiːˈɛl ˈɔːɹp t̪ˈaːŋ` (đánh vần + đọc kiểu Anh + đọc theo luật Việt) |

### 2.4 Bộ âm vị và thanh điệu

- Output dùng **IPA thật**, có dấu răng `̪` (ví dụ `t̪`), dấu trọng âm `ˈ`/`ˌ`, dấu dài `ː`. Các ký hiệu quan sát được trên câu thử: `p b m f v t̪ tʰ ɗ n l s ʃ z ɣ x k h ŋ ɲ tʃ ...`, nguyên âm `a aː ă ɜ e ɛ ə i ɪ o ɔ u ʊ ɯ ...`, bán nguyên âm cuối `j w`.
- **Thanh điệu** dùng chữ số: huyền = 2, hỏi = 4, ngã = 5, nặng = 6. Đáng lưu ý: **ngang và sắc không dùng chữ số**; khác biệt ngang/sắc thể hiện qua chất lượng nguyên âm (ví dụ `ma` → `mˈaː` vs `má` → `mˈaːɜ`; `già` → `zˈaː2` vs `giá` → `zˈaːɜ`). Đây là điểm cần kiểm tra kỹ nếu dùng sea-g2p làm chuẩn đối chiếu thanh điệu — không nên giả định mapping 1–6 đầy đủ.
- README nói rõ quy ước chữ số cho thanh tiếng Việt được cố ý tách khỏi quy ước Chao tone letters của tiếng Thái để hai hệ dùng chung một inventory.

### 2.5 OOV, tên riêng, từ vay, code-switching, phương ngữ

- **OOV**: "Never gives up on a word" theo README; từ ngoài từ điển được đọc bằng luật chính tả, đánh vần, hoặc chuyển sang bộ đọc tiếng Anh (ví dụ thực tế ở mục 2.3). Hành vi này ổn định nhưng có thể pha trộn (vừa đánh vần vừa đọc kiểu Anh).
- **Tên riêng**: đọc theo luật; không có cơ chế "từ điển tên riêng" riêng ngoài việc thêm vào dictionary ở tầng build (không có API override công khai được ghi trong README).
- **Viết tắt/sản phẩm**: acronym được đánh vần theo tên chữ cái tiếng Việt (`VNG` → `vˈe nˈəː2 ɣˈəː2`); có bảng viết tắt trong normalizer.
- **Từ vay/tiếng Anh**: chạy qua bộ đọc tiếng Anh tích hợp; có thể ép bằng thẻ `<en>...</en>`.
- **Phương ngữ**: chỉ có `lang="vi"`, **không có tham số chọn Bắc/Trung/Nam** trong API. Output không có nhãn phương ngữ.

### 2.6 Chất lượng, hiệu năng, điểm mạnh/yếu

- Hiệu năng do README công bố (chưa benchmark độc lập): full pipeline tiếng Việt ~37.000 câu/s; G2P ~415.000 câu/s; normalizer ~41.000 câu/s (CPython 3.12, Windows 11, đa luồng).
- Kiến trúc: Rust core, từ điển nhị phân mmap, string pooling, tìm kiếm nhị phân; bản Python là binding PyO3.
- **Chưa tìm thấy đánh giá độc lập (benchmark bên thứ ba)** cho sea-g2p. Benchmark UGTPhon (arXiv 2609.27205, 2026) chỉ đánh giá Epitran và viG2P cho tiếng Việt, không có sea-g2p.
- **Mạnh**: nhanh, zero-dependency khi cài, Apache-2.0 (thân thiện thương mại), có normalizer mạnh, code-switching Việt–Anh, có C ABI cho runtime native, output tách theo âm tiết.
- **Yếu**: không có paper/đánh giá độc lập; không chọn phương ngữ; quy ước thanh ngang/sắc không tường minh; không có API thêm từ điển/override tại runtime; chất lượng trên tên riêng/UGT chưa được kiểm chứng công khai.

## 3. viG2P

Tên "viG2P" được dùng cho hai dòng code khác nhau; cần phân biệt rõ.

### 3.1 viG2P gốc — Viphoneme (v-nhandt21 / Đỗ Trí Nhân, AILAB HCMUS)

| Mục | Giá trị |
| --- | --- |
| Repo | https://github.com/v-nhandt21/Viphoneme |
| PyPI | https://pypi.org/project/viphoneme/ |
| Bài báo | Tri, Nguyen Minh; Nam, Cao Xuan et al., "Vietnamese Speech Synthesis with End-to-End Model and Text Normalization", NICS 2020. Link IEEE: https://ieeexplore.ieee.org/document/9335905/ (**chưa xác minh trực tiếp**: IEEE Xplore chặn fetch công cụ; thông tin tác giả/tựa xác minh chéo qua README Viphoneme và kết quả tìm kiếm học thuật) |
| Giấy phép | Nội dung LICENSE là mẫu **MIT** kèm dòng "This project is belong to AILAB, Ho Chi Minh University of Science"; PyPI khai báo license "AILAB" — cần tự xác nhận nếu dùng thương mại |
| Phiên bản | PyPI viphoneme 3.0.0, phát hành 23/10/2023 (đo 11/10/2026); repo 110 sao, 20 fork, commit cuối 21/06/2024 |
| Cách tiếp cận | **Rule-based** dựa trên vPhon + kiến thức âm vị học; pipeline: vinorm (chuẩn hoá số/viết tắt) → underthesea (tokenize) → `vi2IPA`; từ tiếng Anh qua `eng_to_ipa` |
| Cài đặt | `pip install vinorm underthesea eng_to_ipa` rồi `pip install viphoneme` |

Ví dụ input → output (lấy từ README, đã fetch):

```
vi2IPA("Được viết vào 6/4/2020, có thể xử lí những trường hợp chứa English")
→ dɯək6 viət5 vaw2 ʂăw5 tʰaŋ5 bon5 năm1 haj1 ŋin2 xoŋ͡m1 ʈăm1 haj1_mɯəj1 , kɔ5_tʰe4 sɯ4_li5 ɲɯŋ3 ʈɯəŋ2_hɤp6 cɯə5 ˈɪŋlɪʃ
```

- **Output**: IPA kèm **chữ số thanh 1–6** đặt sau âm tiết (trong ví dụ: ngang=1, huyền=2, ngã=3, hỏi=4, sắc=5, nặng=6). Dấu `_` nối các âm tiết trong cùng một từ; khoảng trắng giữa các từ. Có `vi2IPA_split(text, delimiter)` để tách từng ký hiệu âm vị, và `Parsing`/`syms` để lấy danh sách ký hiệu.
- **Bộ âm vị**: IPA có phân biệt Hà Nội (ví dụ `ʂ` cho s, `s` cho x, `ʈ` cho tr), `phonologize` cho OOV.
- **Phương ngữ**: kế thừa từ vPhon nhưng README không mô tả API chọn vùng; mặc định nghiêng về Bắc (cần kiểm chứng nếu dùng chính xác phương ngữ).
- **OOV**: README tự cảnh báo "there is too much more task to use it for production, such as OOV (You can handle it by using MFA from Kaldi or Transformer Aligner)"; vinorm v2 giữ từ lạ lại để espeak xử lý nhưng espeak không tách âm tiết, đầu ra dính liền (ví dụ espeak: `ve1kɛɜpte1ɔ7`).
- **Đánh giá độc lập**: benchmark UGTPhon (arXiv 2609.27205, 2026) đo trên tiếng Việt: viG2P đạt PER_ipa **25.4**, PER_canonical **27.3**, PER_non-canonical **78.4** (Δ +51.1) — tốt nhất trong nhóm rule-based nhưng rất yếu trên văn bản phi chuẩn (teencode).
- **Mạnh**: chuẩn hoá + output sạch theo âm tiết, có mã thanh đầy đủ 1–6, Python API đơn giản, nhiều sao/lượt cài (PyPI có lịch sử 2022–2023), có sẵn hướng MFA cho OOV.
- **Yếu**: giấy phép không chuẩn (AILAB), phụ thuộc nặng (`vinorm`, `underthesea`, `eng_to_ipa`), không còn cập nhật từ 2023, không có API chọn phương ngữ, OOV dựa vào espeak.

### 3.2 Bản mở rộng Việt–Anh — hoang1007/vig2p

| Mục | Giá trị |
| --- | --- |
| Repo | https://github.com/hoang1007/vig2p |
| Giấy phép | **Không có file LICENSE và không khai báo trong `pyproject.toml`** (đã kiểm tra danh sách file repo) — chưa công bố giấy phép, không nên dùng trong sản phẩm |
| Quan sát 11/10/2026 | 0 sao, 17 commit; phụ thuộc `cmudict` (GPL-3.0-or-later) và `phonemizer` (GPL-3.0) — rủi ro giấy phép nếu nhúng |
| Cài đặt | `pip install git+https://github.com/hoang1007/vig2p.git` |
| Cách tiếp cận | Kế thừa Viphoneme, bổ sung bộ máy song ngữ; case-sensitive để phân biệt viết tắt (`B` → "bê", `b` → "bờ") |

Ví dụ (từ README, đã fetch):

```
vig2p("năm hai nghìn không trăm hai mươi hai , việt nam đăng cai tổ chức seagame")
→ năm1 haj1 ŋin2 xoŋ1 ʈăm1 haj1 mɯəj1 haj1 , viət6 nam1 dăŋ1 kaj1 to4 cɯk5 siːɡeɪm

vig2p("anh có một vài issues muốn discuss với em", separator="/")
→ ɛ/ɲ/1/ /k/ɔ/5/ /m/o/t/6/ /v/a/j/2/ /ɪ/ʃ/uː/z/ /m/uə/n/5/ /d/ɪ/s/k/ʌ/s/ /v/ɤ/j/5/ /ɛ/m/1
```

**Chính sách OOV 5 bước** (README mô tả, đáng tái sử dụng như *quy trình*, không phải code): (1) từ theo luật tiếng Việt → đọc theo luật; (2) không phải từ Việt chuẩn → tra CMU dict, nếu có → đọc như tiếng Anh; (3) không có trong CMU và viết hoa → coi là viết tắt; (4) nếu không thì tách thành các tiểu từ và đọc từng phần theo tiếng Việt; (5) cuối cùng giao cho espeak.

**Đánh giá**: không tìm thấy benchmark riêng (0 sao, không paper). Chỉ nên tham khảo thuật toán OOV và ví dụ, không dùng code vì giấy phép chưa rõ.

## 4. Các hệ G2P tiếng Việt khác

### 4.1 vPhon (kirbyj)

- Link: https://github.com/kirbyj/vPhon — GPL-3.0, 56 sao, 22 fork; phiên bản 2.1.1 theo README; commit cuối 29/05/2024.
- Cách tiếp cận: rule-based thuần Python (không cần cài đặt), nhận UTF-8 qua STDIN/stdout hoặc interactive.
- Output: IPA, mặc định là dạng **narrow/surface Bắc** với chữ số thanh kiểu Gedney dạng superscript (`A1, B2...`); tuỳ chọn `--chao` (Chao tone letters), `-8` (mã số 1–8), `--phonemic` (dạng underlying theo Pham 2006), `--delimit`, `--ortho`.
- Phương ngữ: `-d n` (Hà Nội), `-d c` (Huế), `-d s` (Sài Gòn), `-d o` (spelling pronunciation thử nghiệm). Trung/Nam gộp hỏi–ngã.
- OOV: input không đọc được sẽ được bọc trong ngoặc vuông `[hacao]`, `[79,5mb]` — rất tốt để **phát hiện** OOV trong audit.
- Mạnh: tài liệu âm vị học kỹ, là nền tảng lý thuyết cho Viphoneme; 3 phương ngữ. Yếu: GPL-3.0 (không nhúng vào sản phẩm đóng), CLI thô, từ 2024 không cập nhật, không có API Python chính thức ngoài gọi script.

### 4.2 Epitran (dmort27)

- Link: https://github.com/dmort27/epitran — MIT (GitHub API xác nhận; PyPI 1.35.3 khai báo "MIT-Modern-Variant", phát hành 08/10/2026); 841 sao.
- Đã xác minh có cặp `vie-Latn` và đã fetch `epitran/data/map/vie-Latn.csv`: mapping đồ thị→IPA có gắn **Chao tone letters** trực tiếp trong map (`a` → `a˧`, `ai` → `aj˧`, `gì` → `<GI>˧`...). Cần bật `tones=True` khi tạo `Epitran` vì mặc định Epitran **loại bỏ** tone IPA — nếu quên, output tiếng Việt sẽ mất thanh.
- OOV: ký tự không có trong map sẽ được truyền nguyên vào output (không báo lỗi).
- Đánh giá độc lập (UGTPhon 2026, tiếng Việt): PER_ipa **32.7**, PER_canonical **34.7**, PER_non-canonical **75.1** (Δ +40.4).
- Mạnh: dễ cài, license MIT, thuần mapping dễ đọc. Yếu: rule map không xử lý được nhiều biến thể/UGT, không tách âm tiết, không chọn phương ngữ.

### 4.3 phonemizer + eSpeak NG

- Links: https://github.com/bootphon/phonemizer (GPL-3.0, PyPI 3.4.0 phát hành 31/07/2026) và https://github.com/espeak-ng/espeak-ng (GPL-3.0, 6.931 sao).
- espeak-ng có **3 giọng Việt**: `vi` (Bắc), `vi-vn-x-central` (Trung), `vi-vn-x-south` (Nam) — xác minh trong `docs/languages.md`. `phonemizer` gọi espeak-ng và trả IPA.
- Output: IPA với chữ số thanh kiểu espeak; Viphoneme README lưu ý espeak **không tách âm tiết**, output dính liền (ví dụ `ve1kɛɜpte1ɔ7`) và không có API tách theo âm tiết Việt.
- Mạnh: sẵn có ngoài hệ sinh thái, hỗ trợ 3 phương ngữ, tự chuẩn hoá số/ngày. Yếu: GPL-3.0/Biến thể GPL ảnh hưởng phân phối, output không theo mô hình âm tiết, toàn bộ xử lý nằm ngoài Python (gọi binary).

### 4.4 Vinorm — chỉ là chuẩn hoá văn bản, KHÔNG phải G2P

- Link: https://github.com/v-nhandt21/Vinorm — 67 sao; LICENSE là mẫu MIT kèm ghi chú AILAB HCMUS (giống Viphoneme); PyPI `vinorm` 2.0.7 phát hành 24/12/2020.
- Chức năng: chuyển **non-standard words** (số, ngày, giờ, tiền tệ, viết tắt) thành âm tiết đọc được; `from vinorm import TTSnorm`. Không sinh âm vị, không có IPA, không xử lý thanh điệu.
- Khác biệt cốt lõi so với G2P: Vinorm là tầng **trước** G2P, tương ứng phần `normalizer` của ViFastTts; không thể thay thế parser/G2P.
- Bản mới hơn được README trỏ tới: https://github.com/NoahDrisort/vinorm (**chưa xác minh** — không fetch). Có bản C++ cùng tác giả (chưa xác minh).

### 4.5 VietNormalizer (2026) — cũng chỉ là chuẩn hoá

- Link: https://arxiv.org/abs/2603.04145 — "VietNormalizer: An Open-Source, Dependency-Free Python Library for Vietnamese Text Normalization in TTS and NLP Applications" (đã fetch trang arXiv). Thư viện zero-dependency, pipeline rule-based chuyển NSW về chữ đọc được. **Không phải G2P**; thông tin giấy phép/repo chưa xác minh.

### 4.6 MFA / ViMFA — G2P từ điển + căn chỉnh

- Montreal Forced Aligner: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner (MIT, 1.902 sao, commit gần 08/10/2026).
- **Model G2P tiếng Việt dựng sẵn** (đã fetch trang mfa-models): Vietnamese (Hanoi), (Ho Chi Minh City), (Hue), các bản v2.0.0/v2.0.0a/v3.0.0, phoneset "MFA", giấy phép **CC BY 4.0**. Acoustic model tiếng Việt: Vietnamese CV/MFA acoustic v2.0.0, v2.0.0a, v3.0.0.
- Quy trình: `mfa train_g2p phoneme_dict/viIPA.txt g2p_model/viIPA.zip` rồi `mfa g2p`; từ điển dạng `<word>\t<phoneme>` (mỗi dòng một âm tiết).
- ViMFA (https://github.com/v-nhandt21/ViMFA, MIT, 15 sao) cung cấp script và hai bộ từ điển ARPA/IPA cho tiếng Việt để train G2P, cùng hướng dẫn align/adapt acoustic model. README nêu rõ mục tiêu: phonologize cho **OOV** trong TTS.
- Vai trò phù hợp: tạo alignment/duration ở tầng training, kiểm tra OOV, không phải runtime frontend.

### 4.7 CharsiuG2P — neural đa ngữ

- Link: https://github.com/lingjzhu/CharsiuG2P — MIT, 396 sao; paper ByT5 (arXiv 2204.03067).
- Mô hình ByT5 "100 ngôn ngữ" trên HuggingFace; đã xác minh `lang_list.txt` có **`vie-n`, `vie-c`** và thư mục `dicts/` có thêm `vie-s.tsv`. Input là từng từ kèm tiền tố ngôn ngữ (`<vie-n>: ...`), output IPA có Chao tone letters; ví dụ dữ liệu `vie-n.tsv`: `a di đà phật → a˧˧zi˧˧da˨kiŋ˧˧fɤ̆t˨ˀ˩`.
- Tiếng Việt viết tách từ nên không cần segmenter đặc biệt; phù hợp làm teacher/đối chiếu neural, nhưng nặng (300M ByT5), cần PyTorch, và chất lượng phụ thuộc dữ liệu WikiPron/espeak nghiêng về chuẩn.

### 4.8 WikiPron tiếng Việt — nguồn từ điển phát âm (không phải G2P engine)

- Link: https://github.com/CUNY-CL/wikipron — code Apache-2.0, 377 sao. Dữ liệu lấy từ Wiktionary nên theo điều khoản Wiktionary (không phải Apache-2.0; thường là CC BY-SA).
- Đã fetch `data/scrape/summary.tsv`: `vie_latn_hanoi_narrow.tsv` **27.684** mục; `vie_latn_hue_narrow.tsv` **30.242**; `vie_latn_saigon_narrow.tsv` **32.559** (kèm bản `_filtered`). Mỗi dòng dạng `word<TAB>IPA (narrow)`, có nhãn phương ngữ — rất giá trị để xây lexicon và audit parser.

### 4.9 Các ghi chú khác

- **seq2seq tiếng Việt**: tìm thấy repo generic https://github.com/QuyAnh2005/Grapheme-To-Phoneme (2 sao, không có LICENSE, dataset ví dụ không phải tiếng Việt) — chỉ là template Encoder-Decoder LSTM, không dùng được ngay.
- **ViSpeechFormer** (arXiv 2602.10003, 2026): framework ASR tiếng Việt mô hình hoá **phoneme** tường minh thay vì ký tự; là bằng chứng cho giá trị của một "phoneme layer" trong pipeline tiếng Việt (dù là ASR, không phải G2P).
- **FastThaiG2P** (arXiv 2608.12814, 2026): G2P Thái dạng từ điển + luật, output IPA/Kokoro, độ trễ ~0,15 ms/câu — tham chiếu khi mở rộng sang SEA.

## 5. Bảng so sánh

| Tên | Link | Cách tiếp cận | Output | License | Phương ngữ | OOV | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sea-g2p | https://github.com/pnnbao97/sea-g2p | Rule + từ điển nhị phân (Rust), normalizer tích hợp | IPA, mỗi âm tiết cách nhau dấu cách; thanh huyền/hỏi/ngã/nặng = 2/4/5/6; ngang & sắc không dùng số | Apache-2.0 | Không có API chọn vùng (`lang="vi"`) | Đọc theo luật/đánh vần/tiếng Anh; không API override runtime | Nhanh (~37k câu/s pipeline theo README); dùng bởi VieNeu-TTS; có C ABI; không paper |
| viG2P (Viphoneme) | https://github.com/v-nhandt21/Viphoneme | Rule-based trên nền vPhon + vinorm + underthesea + eng_to_ipa | IPA + chữ số thanh 1–6, `_` nối âm tiết trong từ | MIT template + ghi chú AILAB (PyPI: "AILAB") | Không có API chọn vùng | Từ lạ để espeak; README khuyến nghị dùng MFA cho OOV | Benchmark 2026: PER_ipa 25.4, PER_nc 78.4; không cập nhật từ 2023 |
| viG2P (hoang1007) | https://github.com/hoang1007/vig2p | Rule + CMU dict + espeak, case-sensitive | IPA + số thanh, tùy chọn separator giữa âm vị | **Chưa công bố license**; phụ thuộc GPL | Không | 5 bước: luật Việt → CMU → viết tắt → tách tiểu từ → espeak | 0 sao; chỉ nên tham khảo quy trình OOV |
| vPhon | https://github.com/kirbyj/vPhon | Rule-based thuần Python | IPA; Gedney mặc định, `--chao`, `-8`, `--phonemic`, `--delimit` | GPL-3.0 | `-d n/c/s/o` | Bọc `[từ]` — dễ phát hiện OOV | Nền tảng lý thuyết; CLI stdin/stdout; không cập nhật từ 2024 |
| Epitran | https://github.com/dmort27/epitran | Mapping CSV + pre/post-processor | IPA; **cần `tones=True`**, tone = Chao letters | MIT (PyPI: MIT-Modern-Variant) | Không | Ký tự lạ đi nguyên vào output | Map `vie-Latn` đã xác minh; PER 2026: 32.7/75.1 |
| phonemizer + eSpeak NG | https://github.com/bootphon/phonemizer | Gọi espeak-ng | IPA, chữ số thanh kiểu espeak; không tách âm tiết | GPL-3.0 | `vi`, `vi-vn-x-central`, `vi-vn-x-south` | espeak tự xử lý (đánh vần) | Chuẩn hoá số/ngày có sẵn; license copyleft |
| Vinorm | https://github.com/v-nhandt21/Vinorm | Normalization (NSW) rule + từ điển | **Chữ tiếng Việt đọc được** (không phải phoneme) | MIT template + ghi chú AILAB | Không | Giữ từ lạ (v2) để tầng sau xử lý | Tầng trước G2P, tương ứng normalizer của ViFastTts |
| VietNormalizer | https://arxiv.org/abs/2603.04145 | Normalization rule, zero-dependency | Chữ tiếng Việt | Chưa xác minh | Không | Chưa xác minh | 2026, chỉ chuẩn hoá |
| MFA (model Việt) | https://mfa-models.readthedocs.io/en/latest/g2p/index.html | G2P từ điển (train từ lexicon) | Chuỗi phoneme theo phoneset MFA | Model: CC BY 4.0; tool: MIT | Hanoi / HCMC / Hue | Sinh phát âm cho từ ngoài từ điển khi align | Tốt cho training/alignment, không phải runtime |
| ViMFA | https://github.com/v-nhandt21/ViMFA | Script train MFA + từ điển viIPA/viARPA | phoneme | MIT | Theo dữ liệu | "phonologize for OOV" | 15 sao; hướng dẫn đầy đủ |
| CharsiuG2P | https://github.com/lingjzhu/CharsiuG2P | Neural ByT5 100 ngôn ngữ | IPA + Chao tones; theo từng từ | MIT (data theo nguồn gốc) | `vie-n`, `vie-c` (data có `vie-s`) | Neural, có thể sai với từ hiếm | Nặng; tốt làm teacher/đối chiếu |
| WikiPron (vi) | https://github.com/CUNY-CL/wikipron | Từ điển phát âm từ Wiktionary | `word<TAB>IPA narrow` | Code Apache-2.0; data theo Wiktionary | Hanoi / Huế / Sài Gòn | Không (chỉ dữ liệu) | 27.684 / 30.242 / 32.559 mục — nguồn lexicon tốt |

## 6. Khuyến nghị tích hợp vào ViFastTts

Bối cảnh: ViFastTts coi parser `ONSET + MEDIAL + NUCLEUS + CODA + TONE` là nguồn sự thật cho chính tả và category trừu tượng; neural input là 7 trường vocab ID, không phải chuỗi ký tự thô; phương ngữ nằm ở normalizer; English fallback là **từ điển respelling xác định**, không phải G2P tiếng Anh tổng quát (`docs/FULL_DESIGN.md`). Vì vậy G2P ngoài nên đóng vai trò **công cụ build-time và kiểm chứng**, không phải runtime engine.

### 6.1 Nên dùng G2P để làm gì

1. **Xây và audit lexicon/bảng âm tiết hợp lệ (ưu tiên cao nhất)**
   - Dùng WikiPron vi (Hanoi/Huế/Sài Gòn) làm nguồn từ + IPA có nhãn phương ngữ; chạy `parse_text`/`safe_parse_syllable` trên từng mục để kiểm tra mọi từ parse được và mọi âm tiết nằm trong inventory. Mục tiêu: mở rộng bảng âm tiết hợp lệ bằng corpus lớn như `docs/FRONTEND.md` đã ghi.
   - Dùng sea-g2p (Apache-2.0) như "ý kiến thứ hai": so IPA của WikiPron với IPA của sea-g2p cho cùng danh sách từ; chênh lệch là ứng viên cần review (biến thể phương ngữ, lỗi dữ liệu, hoặc lỗi parser).
2. **Kiểm chứng round-trip nội bộ (deterministic, không cần G2P ngoài)**
   - Với mỗi từ chuẩn hoá: `parse → compose_canonical → normalize` rồi so khớp lại với input; mismatch ghi log và bổ sung test. Đây là cách kiểm tra parser/orthography không phụ thuộc bên thứ ba; G2P chỉ dùng để giải thích các ca mismatch (ví dụ `gi`/`d`, coda `y/i`).
3. **Sinh ứng viên cho override dictionary và English fallback**
   - ViFastTts hiện dùng respelling tiếng Việt (ví dụ `facebook → phây búc`) trong `english_fallback.py`. sea-g2p có thể chạy offline trên danh sách tên sản phẩm/tên riêng/danh từ công nghệ để gợi ý respelling; con người duyệt rồi mới đưa vào `ACRONYM_READINGS`/`EXACT_PRODUCT_NAMES`/override. Không gọi G2P ở runtime.
4. **OOV resolver (thiết kế mới, dùng G2P làm tham chiếu)**
   - Hiện token không parse được trả `ParseStatus.FALLBACK_REQUIRED` với `nucleus=token`. Đề xuất thêm resolver theo đúng tinh thần 5 bước của viG2P (viết lại quy trình, không copy code): (a) tra override/acronym/English fallback; (b) nếu giống từ tiếng Anh → đọc theo bảng fallback; (c) nếu viết hoa toàn bộ → đánh vần; (d) nếu không → tách tiểu từ rồi đánh vần; (e) log + human review.
   - Dùng vPhon để phát hiện OOV (`[từ]` trong output) và sea-g2p để lấy gợi ý đọc cho các ca khó.
5. **Phoneme layer tùy chọn (chỉ cho training/đối chiếu, không thay vocab hiện tại)**
   - Nếu cần initial duration/alignment: dùng MFA với model G2P + acoustic tiếng Việt (Hanoi/HCMC/Hue, CC BY 4.0) để bootstrap alignment, hoặc dùng IPA của sea-g2p/WikiPron làm target phụ khi phân tích lỗi.
   - ViSpeechFormer (arXiv 2602.10003) cho thấy hướng phoneme có giá trị trong tiếng Việt, nhưng ViFastTts vẫn nên giữ component embeddings; phoneme layer chỉ là kênh phụ trợ, có thể bật/tắt và không ảnh hưởng vocab ID.
6. **Kiểm thử biến thể phương ngữ**
   - Dùng dữ liệu có nhãn (WikiPron hanoi/hue/saigon; Charsiu vie-n/c/s; MFA Hanoi/HCMC/Hue; espeak `vi`/`vi-vn-x-central`/`vi-vn-x-south`; vPhon `n/c/s`) để viết test cho các khác biệt thực tế: hỏi/ngã, r/d/gi, s/x, tr/ch, iê/ia, uô/ua, ươ/ưa. Dialect vẫn chỉ nằm ở normalizer như thiết kế.

### 6.2 Không nên dùng G2P để làm gì

- **Không thay parser cấu trúc âm tiết** hay suy ONSET/MEDIAL/NUCLEUS/CODA từ output IPA: G2P trả chuỗi âm vị, đã mất thông tin chính tả (D/GI/R, TR/CH, S/X, vị trí dấu thanh cũ/mới) — thứ mà ViFastTts cần.
- **Không đưa chuỗi ký tự IPA thô làm input neural**: phá vỡ thiết kế component vocab, làm vocab phình to, mất ổn định khi đổi engine.
- **Không dùng code GPL-3.0 trong runtime/sản phẩm** (vPhon, phonemizer, espeak-ng, `cmudict`): chỉ dùng ở công cụ nghiên cứu/đối chiếu ngoài repo runtime. Ưu tiên sea-g2p (Apache-2.0), MFA models (CC BY 4.0), WikiPron code (Apache-2.0, lưu ý data theo Wiktionary).
- **Không dùng để đoán tên riêng không biết trong production**: cả sea-g2p lẫn viG2P đều đọc theo luật/đánh vần và có thể sai; thiết kế hiện tại đã đúng khi yêu cầu "tên sản phẩm, tên người... phải thêm vào override" (`docs/FULL_DESIGN.md`).
- **Không kỳ vọng xử lý teencode/UGT**: benchmark UGTPhon 2026 cho thấy PER non-canonical của viG2P là 78.4 và Epitran 75.1 — cần tầng normalizer/override riêng.

### 6.3 Luồng dữ liệu đề xuất

```
RUNTIME (giữ nguyên):
raw text → normalizer (dialect: north|central|south) → tokenizer
         → parser ONSET+MEDIAL+NUCLEUS+CODA+TONE
         → nếu OK: vocab.encode (7 trường)
         → nếu FALLBACK_REQUIRED: OOV resolver (override → acronym/English fallback
            → đánh vần/tách tiểu từ) → vocab.encode + log để bổ sung override

BUILD-TIME ONLY (không nằm trong ONNX/runtime):
WikiPron (vi, 3 vùng) + sea-g2p (Apache-2.0)
   → audit: coverage inventory, round-trip, đối chiếu IPA, gợi ý respelling
   → báo cáo + đề xuất cập nhật tests/override
MFA (G2P + acoustic Việt, CC BY 4.0)
   → alignment/duration bootstrap cho training (tuỳ chọn phoneme layer)
```

### 6.4 Việc cần làm cụ thể

1. Viết `tools/` script build-time (không thêm vào dependency runtime): cài `sea-g2p==0.10.0` (pin version) trong môi trường riêng; xuất bảng `âm tiết → IPA` và danh sách token OOV từ corpus nội bộ.
2. Thêm test cho cặp tối thiểu thanh điệu (`ma/má/mà/mả/mã/mạ`) và các cặp phương ngữ (r/d/gi, s/x, tr/ch, hỏi/ngã) dựa trên dữ liệu WikiPron.
3. Xây quy trình audit định kỳ: `parse → compose_canonical → so khớp`; log mismatch kèm phiên bản engine/pin commit.
4. Mở rộng override dictionary bằng dữ liệu (humans-in-the-loop), không nhúng output G2P trực tiếp.
5. Ghi `docs/LICENSES.md` (hoặc mục trong README) liệt kê: sea-g2p Apache-2.0; WikiPron code Apache-2.0/data theo Wiktionary; MFA tool MIT/models CC BY 4.0; các công cụ GPL chỉ dùng offline.
6. Nếu cần alignment: thử `mfa g2p` với model Vietnamese (Hanoi) trên manifest hiện có để lấy duration khởi tạo.

### 6.5 Bảng IPA nội bộ của ViFastTts (`frontend/g2p.py`)

`frontend/g2p.py` ánh xạ 5 thành phần sang IPA broad/narrow theo Kirby (2011) và Thompson (1965) (nguồn: Wikipedia "Vietnamese phonology"), có profile theo vùng:

| Hiện tượng | Bắc (Hà Nội) | Trung (xấp xỉ) | Nam (TP.HCM) |
| --- | --- | --- | --- |
| `d`, `gi`, `r` | đều /z/ | /j/, /j/, /ʐ/ | /j/, /j/, /ʐ/ |
| `ch`, `tr` | đều /tɕ/ | /c/, /ʈ/ | /c/, /ʈ/ |
| `s`, `x` | đều /s/ | /ʂ/, /s/ | /ʂ/, /s/ |
| thanh hỏi / ngã | ˧˩˧ / ˧ˀ˥ | như Bắc (tạm) | gộp ˨˩˦ |
| thanh nặng | ˧ˀ˨ʔ | như Bắc (tạm) | ˨˩˨ |
| thanh ngang | ˧ | ˧ | ˦ |

- Nguyên âm: `a ă ɤ̆ ɛ e i ɔ o ɤ u ɯ`; vần đôi `iə uə ɯə` (Kirby 2011); thêm `OO = ɔː` (o dài trong từ vay: xoong, moóc) và tách `A` (a dài, ai/ao) khỏi `Ă` (ă ngắn, ay/au) theo Help:IPA/Vietnamese ("ai" [aːj] vs "ay" [aj]).
- Narrow thêm: `t̪ t̪ʰ`, `p̚ t̚ k̚`, `k̟̚`, `ŋ̟`.
- **Quy ước số thanh phải chuẩn hoá khi đối chiếu**: bảng nội bộ dùng thứ tự viG2P (1 ngang, 2 huyền, 3 ngã, 4 hỏi, 5 sắc, 6 nặng); VNI/Telex dùng 1 sắc, 2 huyền, 3 hỏi, 4 ngã, 5 nặng; sea-g2p chỉ dùng số cho huyền/hỏi/ngã/nặng (2/4/5/6) và phân biệt ngang/sắc bằng chất lượng nguyên âm. Vì vậy so khớp liên hệ thống cần ánh xạ quy ước trước, không so ký tự thô.
- Trung: chưa có bảng Huế đủ tin cậy; muốn chốt cần dữ liệu WikiPron `hue` hoặc MFA Huế (`docs/G2P_VIETNAMESE.md` mục 4.6).

## 7. Tài liệu tham khảo

Tất cả truy cập ngày **11/10/2026**, trừ khi ghi chú khác.

1. sea-g2p — GitHub: https://github.com/pnnbao97/sea-g2p
2. sea-g2p — PyPI: https://pypi.org/project/sea-g2p/
3. VieNeu-TTS: https://github.com/pnnbao97/VieNeu-TTS
4. viG2P gốc (Viphoneme) — GitHub: https://github.com/v-nhandt21/Viphoneme
5. viG2P gốc — PyPI: https://pypi.org/project/viphoneme/
6. Tri, N. M., Nam, C. X. et al., "Vietnamese Speech Synthesis with End-to-End Model and Text Normalization", NICS 2020: https://ieeexplore.ieee.org/document/9335905/ (**chưa xác minh trực tiếp**; IEEE Xplore chặn fetch)
7. viG2P mở rộng Việt–Anh: https://github.com/hoang1007/vig2p
8. vPhon: https://github.com/kirbyj/vPhon
9. Epitran: https://github.com/dmort27/epitran; map `vie-Latn`: https://raw.githubusercontent.com/dmort27/epitran/master/epitran/data/map/vie-Latn.csv
10. phonemizer: https://github.com/bootphon/phonemizer
11. eSpeak NG và danh sách ngôn ngữ: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
12. Vinorm: https://github.com/v-nhandt21/Vinorm
13. VietNormalizer (arXiv 2603.04145): https://arxiv.org/abs/2603.04145
14. MFA model G2P tiếng Việt: https://mfa-models.readthedocs.io/en/latest/g2p/index.html; acoustic: https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
15. Montreal Forced Aligner: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
16. ViMFA: https://github.com/v-nhandt21/ViMFA
17. CharsiuG2P: https://github.com/lingjzhu/CharsiuG2P; paper: https://arxiv.org/abs/2204.03067
18. WikiPron: https://github.com/CUNY-CL/wikipron; bảng thống kê scrape: https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/summary.tsv
19. UGTPhon benchmark (arXiv 2609.27205): https://arxiv.org/abs/2609.27205
20. FastThaiG2P (arXiv 2608.12814): https://arxiv.org/abs/2608.12814
21. ViSpeechFormer (arXiv 2602.10003): https://arxiv.org/abs/2602.10003
22. Repo seq2seq tham khảo (không chuyên Việt): https://github.com/QuyAnh2005/Grapheme-To-Phoneme
