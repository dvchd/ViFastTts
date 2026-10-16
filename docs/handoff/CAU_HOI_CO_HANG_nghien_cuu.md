# Nghiên cứu 3 câu hỏi gửi cô Hằng (Viện Ngôn ngữ học) — hồ sơ đối chiếu

> Ngày 16/10/2026. Mục đích: chuẩn bị **đối chiếu** câu trả lời của cô với (a) nguồn
> phiên chính tra được, (b) phản biện độc lập của sub-agent **khác họ model** (chỉ đọc,
> không sửa tệp). Hồ sơ này KHÔNG thay thế ý kiến chuyên gia — dùng để so.
> Sub-agent: `opencode-go/muse-spark-1.3-contributor` (tên chữ cái + anh/ach) ·
> `opencode-go/gemini-4-argon-contributor` (Việt hoá đọc tiếng Anh) — giá lúc dùng:
> muse-spark **$0,1/$0,2**/M token · gemini-4-argon **$4/$20**/M token (**vượt ngân sách
> sub-agent** — bài học này đã thành rule trong `AGENTS.md` §8: các phiên sau chỉ dùng model
> có input ≤ $0,2/M, mặc định `muse-spark`).
> Kiểm chứng: mọi ví dụ `english_fallback.py` dưới đây đã đối chiếu trực tiếp với tệp
> (`CASE_INSENSITIVE_ENGLISH`, 671 mục) — không qua trung gian.

## 0. TL;DR

1. **Tên chữ cái** ("bê, xê, dê, đê, e-lờ, em-mờ…") = tên chữ **vay từ tiếng Pháp**,
   dùng để đọc chữ rời/viết tắt (VTV = vê-tê-vê, HTV = hắt-tê-vê). Riêng **H = hát/hắt
   chưa có nguồn giải thích** ("ache" tiếng Pháp và "agá" Bồ Đào Nha đều không khớp
   thẳng) → câu hỏi cho cô. Không có văn bản pháp quy nào quy định **cách đọc** viết tắt
   (QĐ 1989 và QĐ 240 chỉ quy định cách VIẾT).
2. **Việt hoá đọc tiếng Anh (kiểu Katakana)**: **không có chuẩn nhà nước** cho từ thông
   dụng; chính sách hiện hành ngược lại — giữ nguyên dạng chữ Latin (QĐ 1989 Điều 7 cấm
   phiên âm thuật ngữ; Điều 5 chỉ SGK tiểu học được phiên âm tên riêng). Nhưng **có
   nghiên cứu loanword adaptation chặt chẽ** (Nguyễn Đăng Liêm 1970; Barker 1969;
   Kang–Phạm–Storme 2016; Nguyen Huynh Trang 2021) + paper ASR 2025 làm đúng việc dự án
   đang làm (Vietlish: "inbox"→"in bóc", "message"→"mét xít"). Quy luật chính: chèn "ờ"
   cho cụm phụ âm đầu; **cắt** phụ âm thừa trong cụm cuối (khác Katakana!); vô thanh hoá
   cuối b/d/g→p/t/c; xát cuối → tắc (s→t, v→p); `-ing`→`-inh`; thanh điệu theo coda +
   trọng âm (sắc cho coda tắc; ngang/huyền theo trọng âm).
3. **anh/ach**: **đính chính** — Thompson (1965/1967) là phe "coda riêng /c, ɲ/";
   phe "nh/ch là biến thể của /ŋ, k/ + a là /ɛ/" là **Henderson (1965) + Kirby (2011)**
   (và Đoàn Thiện Thuật *theo dẫn của cô* — nguyên văn chưa xác minh). Khuyến nghị TTS:
   **giữ nhãn SGK (a + nh) nhưng đích âm thanh [ăjŋ̟]/[ăjk̟̚]**. Repo hiện in
   broad "a ɲ" / narrow "a ŋ̟" — thiếu đôi hoá.

---

## 1. TÊN CHỮ CÁI — "B là bê, H là hát" từ đâu ra?

### 1.1 Chốt được (nguồn)

- `vi.wikipedia` **Chữ Quốc ngữ**, mục *Tên gọi các chữ cái*:
  "Tất cả tên gọi các chữ cái trong tiếng Việt đều được bắt nguồn từ **tiếng Pháp**.
  Chúng thường được dùng để đánh vần hoặc gọi tên viết tắt như VTV (vê-tê-vê),
  HTV (hắt-tê-vê)…" — kèm bảng: a, ă(á), â(ớ), bê, xê, dê, đê, e, ê, gờ, hắt, i (ngắn),
  ca, e-lờ, em-mờ, en-nờ, o, pê, quy, e-rờ, ét-sì, tê, u, vê, ích-xì, y (dài).
  F/J/W/Z = ép(-phờ)/gi/vê kép/dét — bài nêu rõ nguồn gốc từ "effe/ji/double vé/zède"
  (tiếng Pháp). Y từng có tên "i-cờ-rét" (i grec, đã bớt dùng).
  → <https://vi.wikipedia.org/wiki/Chữ_Quốc_ngữ>
- Đối chiếu bảng tên chữ Pháp (bé, cé, dé, gé, pé, té, vé; elle/emme/enne/erre/esse/ixe)
  — khớp từng chữ một, trừ H. → <https://en.wikipedia.org/wiki/French_orthography>
- Ba "hệ" tên chữ theo thế hệ: **"a bờ cờ"** (đọc âm — từ phong trào Bình dân học vụ)
  / **"a bê xê"** (theo tiếng Pháp) / **"ây bi xi"** (theo tiếng Anh) — cùng bài vi.wiki,
  mục "Từ giữa thế kỷ 20".
- Từ 2003, ngành giáo dục phân vai: **tên chữ** để đọc chữ rời/viết tắt, **âm chữ** để
  ghép vần lớp 1 (báo Tuổi Trẻ/Thanh Niên; sách *Syllabaire quốc-ngữ* 1909 đã dạy
  song song "bê-bờ", "xê-cờ"…).
- **Không có văn bản pháp quy nào quy định cách ĐỌC**: QĐ 1989/QĐ-BGDĐT và Thông tư
  01/2011/TT-BNV chỉ quy định cách VIẾT/trình bày. Chuẩn đọc là **thông lệ nhà trường +
  yêu cầu của giới ngôn ngữ với báo chí** (GS Nguyễn Minh Thuyết 2016: TPP="tê-pê-pê",
  MH370="em-mờ hát…"; ngoại lệ: viết tắt đã thành từ → đọc như từ — UNESCO="u-nê-xcô",
  FIFA="phi-pha"; viết tắt "bung được" → bung: TP.HCM="thành phố Hồ Chí Minh").
  → <https://infonet.vietnamnet.vn/dai-phat-thanh-truyen-hinh-can-doc-ten-chu-cai-tieng-viet-cho-dung-159849.html>

### 1.2 Còn mở (để hỏi cô / cần văn bản gốc)

- **Gốc chữ H "hát/hắt"**: không khớp "ache" /aʃ/ tiếng Pháp; giả thuyết "agá" (Bồ) chưa
  có nguồn xác nhận; cũng có thể là sáng tạo nội sinh. `vi.wikipedia` **H** ghi cả ba
  biến thể **hắt / hát / hờ**. → câu hỏi số 1 cho cô.
- **Ai chuẩn hoá bảng tên chữ đọc lên**: báo chí hay quy cho de Rhodes, nhưng từ điển
  1651 (23 chữ, không có ă/â/ê/ô/ơ/ư độc lập) không ghi tên chữ đọc lên. Cần bản SGK
  Pháp–Việt trước 1945 để chốt — mức tin cậy hiện tại: trung bình.
- Biến thể đang lưu hành: `hắt` vs `hát`; G = "gờ" hay "giê"; Q = "quy" hay "cu"
  (Dân trí 2014 gọi "quy" là cách đọc phía Nam ảnh hưởng tiếng Pháp).

### 1.3 Hệ quả cho dự án — 2 bảng tên chữ đang LỆCH nhau (đã kiểm trên code)

- `src/vifasttts/frontend/oov.py::LETTER_READINGS`: d = "dê" (đúng), **có** đ/l/o;
  style ngắn "lờ, rờ, em, en, ét, ích"; j = "giây" (biến thể kiểu ANH /dʒeɪ/ — có thật
  theo vi.wiki "J": *"đọc là 'giây'… theo tiếng Anh hoặc 'gi'… theo tiếng Pháp"*).
- `src/vifasttts/frontend/stem.py::VARIABLES`: **d = "đê" (SAI — "đê" là tên của Đ)**,
  **thiếu đ, l, o**; style dài "e rờ, i dài".
- → Việc cần làm: hợp nhất 1 bảng, sửa lỗi "d", chốt style (chuẩn trường học:
  e-lờ, e-rờ, em-mờ, en-nờ, ét-xì, ích-xì, i ngắn/dài) — chờ cô xác nhận danh sách.

---

## 2. VIỆT HOÁ ĐỌC TỪ TIẾNG ANH (kiểu Katakana) — có chuẩn không?

### 2.1 Chính sách: KHÔNG có chuẩn cho từ thông dụng

- **QĐ 240/QĐ 1984**: thuật ngữ nước ngoài "không nên lấy yêu cầu đồng hoá theo ngữ âm
  và chữ viết tiếng Việt làm tiêu chí chỉ đạo"; bác lối phiên âm tuỳ tiện từng âm tiết.
  → <https://luatvietnam.vn/giao-duc/quyet-dinh-240-qd-1984-ve-chinh-ta-tieng-viet-va-ve-thuat-ngu-tieng-viet-160366-d1.html>
- **QĐ 1989/QĐ-BGDĐT**: Điều 5 — tên riêng nước ngoài giữ nguyên dạng; **chỉ SGK Tiểu
  học** dùng phiên âm gạch nối (Tô-mát Ê-đi-xơn, Mát-xcơ-va, Pa-ri); Điều 7 — cấm phiên
  âm thuật ngữ khoa học (viết `hydrogen`, `concerto`).
- Từ điển Anh–Việt chuẩn chỉ ghi **IPA quốc tế**; "phiên âm bồi" chỉ có ở blog dạy học.
- Báo đài không thống nhất (VOV2: SGK viết "Anbe Anhxtanh", báo viết "Albert Einstein",
  phát thanh viên đọc pha trộn) — dự thảo "quy định cách đọc tên riêng nước ngoài"
  (2007, Nguyễn Văn Khang tham gia) đến nay vẫn chưa ban hành.

### 2.2 Nghiên cứu loanword adaptation (có — và đây là "chuẩn ngầm" học thuật)

- Nguyễn Đăng Liêm (1970), *A Contrastive Phonological Analysis of English and
  Vietnamese*; Barker (1969) French loanwords; Alves (2009); Kang–Phạm–Storme (2016);
  Nguyen Huynh Trang (2021, OT) — danh mục URL đầy đủ trong báo cáo sub-agent B
  (lưu bản đầy đủ ở artifacts phiên; tóm tắt dưới đây).
- Quy luật quan sát được (dùng làm spec cho pipeline):
  1. Khuôn âm tiết Việt `C(w)V C + T`; **không cụm phụ âm**; coda ∈ {p t c ch · m n ng nh
     · i/y · u/o}.
  2. Cụm phụ âm **đầu** → chèn âm tiết "ờ" (epenthesis): `frame`→"phờ rêm",
     `stream`→"xờ trim", `block`→"bờ lốc" (âm chèn thường mang thanh huyền/ngang).
  3. Cụm phụ âm **cuối** → **cắt bớt** (khác Katakana!): `text`→"tếch", `list`→"lít",
     `link`→"linh".
  4. Vô thanh hoá cuối: b/d/g → p/t/c + thanh SẮC/NẶNG ("tab"→"táp", "code"→"cốt").
  5. Âm xát cuối → tắc gần nhất: f/v→p ("off"→"óp"); s/z/θ/ð/ʃ→t/ch ("boss"→"bót",
     "with"→"quít"); θ→th; ð→đ; ʃ→s.
  6. `/l/` cuối → n ("bill"→"bin", "module"→"mô đun") hoặc rụng sau nguyên âm đôi
     ("mail"→"meo"); `-əl/-le` → "-ồ" ("google"→"gu gồ").
  7. `-ing` → **`-inh`** ("booking"→"búc kinh") vì tiếng Việt không có vần "ing".
  8. /w/ → "qu-" (Nam) hoặc "u-" (Bắc); /j/ → "d-/gi-" hoặc "i-"; nguyên âm:
     æ→a/ă/e, ʌ→ă/â, ɜː→ơ, eɪ→ây/ê, oʊ→âu/ô.
  9. Thanh: coda tắc → >90% SẮC (NẶNG khi không trọng âm đuôi); mở/vang → trọng âm:
     NGANG, schwa cuối: HUYỀN; **hỏi/ngã gần như không bao giờ** cho từ vay.
- Hệ TTS/ASR Việt: **2 trường phái** — (a) code-switch bằng IPA tiếng Anh chèn vào chuỗi
  IPA Việt (Viphoneme dùng eng_to_ipa; vig2p cmudict→espeak; sea-g2p trong VieNeu-TTS
  đọc thẳng "Facebook"→`fˈeɪsbʊk`); (b) **respell về âm tiết Việt** trước bộ âm vị Việt
  (Vinorm/Vbee — "dự đoán cách đọc từ viết tắt, từ vay mượn"). ViFastTts chọn (b) —
  bắt buộc bởi kiến trúc 5-slot.
- Paper ASR 2025 tự xây bản đồ tương tự: Vietlish ("inbox"→"in bóc",
  "message"→"mét xít", "a"→"ây") — <https://arxiv.org/abs/2508.19270>;
  TSPC xây "unified Vietnamese phoneme" — <https://arxiv.org/abs/2509.05983>.
  → Xác nhận: **không ai có "Katakana sẵn"; ai cũng tự xây bảng ánh xạ** như dự án.

### 2.3 Đánh giá `english_fallback.py` (đã đối chiếu từng mục với tệp)

- **Nhóm tốt**: `work`→"quớc", `framework`→"phờ rêm quớc", `network`→"nét quớc",
  `keyword`→"ki quớt", `google`→"gu gồ", `youtube`→"diu túp", `wifi`→"quai phai";
  `booking/coding/running/learning` → "…kinh/đinh/ninh" (nhất quán -inh).
- **Nhóm bất nhất**:
  - /taɪ/: `type`→"típ" (kiểu Pháp) vs `typescript`→"thái xờ ríp" vs `time`→"thai".
  - Coda /d/: `load/reload`→"lốt" nhưng `download`→"đao lô", `upload`→"áp lô" (mất coda).
  - `-ing`: `monitoring`→"mo ni tơ **ring**", `rendering`→"ren đờ **ring**",
    `huggingface`→"hắc **ghing** phây" — lệch nhóm "-inh".
  - /w/: `win`→"uyn" nhưng `windows`→"quin đô", `with`→"quít" (lệch ngay trong cùng w-).
  - Đọc kiểu "mặt chữ Pháp" lẫn với đọc IPA: `data`→"đa ta", `module`→"mô đun",
    `status`→"xờ ta tút", `tutorial`→"tu tô ri an".
- **Nhóm SAI rõ (~10 mục, nên sửa)**:
  | key | hiện tại | vấn đề |
  |---|---|---|
  | `share` / `sale` | "sêu" | /ʃeə(r)/ không ai đọc "sêu"; `share` nên "se/xe", và trùng với `sale` |
  | `sell` | "xêu" | /sel/ → "xen/xeo"? "xêu" lệch âm cuối |
  | `catching` | "cách tinh" | phải "cách chinh" (mất /tʃ/ của "-ching") |
  | `hover` | "háp vơ" | thừa /p/ (đúng: "hơ vơ") |
  | `disable` | "đi xây bồ" | mất /s/ ("đít xây bồ") |
  | `record` | "ri cồ" | mất coda /d/ (đúng: "ri cọt") |
  | `result` | "ri dần" | lệch từ (gần với "reason"); `result` = "ri zăn/dăn" |
  | `tool` / `tools` | "tu" | mất /l→n/ (đúng: "tun") |
  | `subscribe` | "súp sờ cờ rai" | tách 4 âm tiết, dễ vỡ nhận diện (nên "sắp sờ rai") |
  | `windows` | "quin đô" | /wɪn-/ Bắc nên "uyn-"; vần "-ows" → "đâu/đô" tuỳ chọn |
- **Ghi chú phương ngữ (quan trọng)**: giọng mẫu của dự án = **Bắc**; các chuỗi
  "quít/quớc/quin" sẽ được giọng Bắc đọc [kw…] (thêm /k/) trong khi tiếng Anh là /w/;
  "diu túp" sẽ ra [z-] (Bắc) thay vì [j-]. → Cần quy ước w/j cho Bắc:
  /w/→"u-" (`uýt`, `uyn`, `uớc`?) hay chấp nhận "qu-"; /j/→"i-" (`iu túp`) hay "d-"
  (`diu túp`). Đây là câu hỏi số 4 cho cô.

### 2.4 PHẢN BIỆN NGƯỢC (rủi ro của phương án Việt hoá)

- Katakana thành công vì người Nhật **quen nhìn** chuỗi katakana trong đời sống; người
  Việt **chưa bao giờ thấy** chuỗi "súp sờ cờ rai" — nên với từ dài/nhiều âm chèn, bản
  Việt hoá dễ "vỡ nhận diện", nhất là khi model đọc các âm chèn "sờ/cờ" **không ngắn
  như thật** (cần: âm chèn ~30–50 ms, thanh huyền/ngang nhẹ — hiện là việc của duration
  model, chưa có cơ chế riêng).
- Từ 1–2 âm tiết quen thuộc thì Việt hoá **tự nhiên hơn** code-switch trong câu Việt;
  từ dài/nhiều từ Anh liền nhau có thể để đọc tiếng Anh (hoặc IPA) dễ nghe hơn.
- → Khuyến nghị: **cấu hình được** (Việt hoá cho từ đã có trong bảng; giữ/đọc theo IPA
  cho phần còn lại) + **không hứa "chuẩn quốc gia"** vì không tồn tại.

---

## 3. ANH/ACH — "a + nh" hay "/ɛ/ + ŋ"?

### 3.1 Hai phe + ĐÍNH CHÍNH

- **Phe "coda riêng"**: Thompson (1965; *A Vietnamese Grammar*, ERIC) và Thompson (1967,
  *Language* 43) — cuối `-ch/-nh` là âm vị **/c, ɲ/** độc lập, đối lập cả /t,k/ và /n,ŋ/.
  → **Đính chính: Thompson KHÔNG thuộc phe /ɛ/** (bản nháp trước từng xếp lẫn — sửa hồ sơ).
- **Phe "allophone"**: Henderson (1965) — [c]/[ɲ] cuối thực chất là **tiền-velar
  [k̟, ŋ̟], không tiếp xúc lợi**; Kirby (2011, JIPA) — Hà Nội: `/ɛŋ/→[ăjŋ̟]`,
  `/ɛk/→[ăjk̟̚]`; lập luận: lỗ hổng phân bố [k/ŋ] sau [i,e], phương ngữ Trung bảo thủ
  [ɛɲ], hành vi trong từ láy. Đoàn Thiện Thuật *theo dẫn của cô* + tài liệu TV1-CGD —
  **nguyên văn chưa xác minh được** (không truy cập được sách; cần số trang khi trích).
- Nam Bộ: `-nh/-ch` nhập vào `-n/-t` (anh → [ăn]) — lưu ý cho profile Nam.

### 3.2 Khuyến nghị cho TTS (kèm trade-off)

- **Nhãn đầu vào: GIỮ kiểu SGK** (`A` + `NH/CH`) — khả nghịch chính tả tuyệt đối; mọi
  phe đồng ý mặt chính tả; đổi sang /ɛ/+ŋ/k là mất thông tin một chiều (không tái sinh
  được "anh" vs "eng").
- **Đích âm thanh: [ăjŋ̟] / [ăjk̟̚]** (nguyên âm rút ngắn + offglide [j] + coda prevelar).
  Không tổng hợp [aɲ] ("a dài + nh" — lỗi ngoại ngữ).
- **Repo hiện tại**: g2p broad `a ɲ`, narrow `a ŋ̟` — **thiếu rút ngắn [ă] và offglide
  [j]**. Việc "g2p narrow cho anh/ach" nằm trong hàng đợi (chờ cô xác nhận trước khi đổi).
- **Kiểm chứng đề xuất sau này**: tổng hợp "anh/ách/ênh" rồi đo formant — F2 phải gần
  nhóm "eng" hơn "ang" + có F2 rising trước coda.

---

## 4. PHẢN BIỆN NGOÀI — đối chiếu chéo (sub-agent, khác họ model)

| Điểm | Phiên chính (DeepSeek V4.1 Flash) | Muse Spark 1.3 | Gemini 4 Argon | Kết luận |
|---|---|---|---|---|
| Tên chữ từ tiếng Pháp | ✓ (vi.wiki + bảng Pháp) | ✓ (đối chiếu từng chữ) | — | **Đồng thuận cao** |
| H "hát/hắt" | Chưa giải thích; nghi ngoại lệ | Giả thuyết "agá" **chưa xác minh** + có thể nội sinh | — | **Cùng đánh dấu MỞ** → hỏi cô |
| Viết tắt | Không có văn bản quy định đọc | Tên chữ + ngoại lệ từ vựng hoá/bung | — | **Đồng thuận** |
| Katakana-like chuẩn? | Không có; SGK chỉ tên riêng | — | Không có; loạn 3 cách (SGK/báo/PTV) | **Đồng thuận** |
| Quy luật Anh→Việt | Mới có "Vietlish" papers | — | Hệ thống hoá 9 nhóm + tone rules | Bổ sung cho nhau |
| `english_fallback.py` | (chưa soát sâu) | — | **~10 lỗi + 5 nhóm bất nhất** | Đã **kiểm chứng lại trên tệp** — đúng |
| Thompson | Từng gán lẫn vào phe /ɛ/ | **Đính chính: Thompson = phe coda riêng** | — | **Sửa theo đính chính** |
| anh/ach cho TTS | Nhãn SGK + đích [ăjŋ̟] | Cùng khuyến nghị + cảnh báo Bắc không đồng nhất | — | **Đồng thuận** |

- Khác biệt duy nhất đáng chú ý: mức độ "cả gan" của khuyến nghị sản phẩm (Gemini đề
  xuất pipeline rule-based thay bảng tay; phiên chính nghiêng về "sửa bảng tay trước,
  pipeline sau khi có cô chốt quy ước"). Đây là lựa chọn thứ tự công việc, không phải
  mâu thuẫn dữ kiện.

## 5. CÂU HỎI TINH (bản cập nhật để hỏi cô)

1. ★ **Chữ H "hát/hắt"**: gốc từ đâu ạ ("ache" tiếng Pháp, "agá" Bồ Đào Nha, hay do
   người Việt tạo: âm "hờ" + vần "-át")? Có bảng tên chữ gốc nào (SGK Pháp–Việt trước
   1945, sách vần Công giáo) để chốt ạ?
2. ★ Khi máy đọc **viết tắt** (TP.HCM, HĐND): nên dùng bảng **tên chữ** nào; và với viết
   tắt nước ngoài (WHO, USB) thì đọc tên chữ Việt hay giữ tiếng Anh?
3. ★ **Cụm phụ âm đầu** tiếng Anh ("frame" → "phờ rêm"): âm chèn "ờ" — có phải một âm
   tiết thật không, nên gán thanh gì và **ngắn cỡ nào** để nghe tự nhiên?
4. ★ Với **giọng Bắc**: ghi /w/, /j/, `-ing`, `-əl` bằng chữ gì cho đỡ lệch —
   "uýt" hay "quít"; "iu túp" hay "diu túp"; "-inh"; "gu gồ" hay "gu gờ"?
5. ★ **anh/ach**: giữ nhãn SGK + đích [ăjŋ̟] như khuyến nghị đúng chưa? Tương phản
   ăn–anh–ăng ở người Hà Nội trẻ còn vững không?

## 6. VIỆC NÊN LÀM TIẾP

- ✅ **ĐÃ ÁP DỤNG 16/10** (các lỗi khách quan, không phụ thuộc quy ước đang chờ cô):
  1. `english_fallback.py`: sửa 14 mục — `share→se`, `sell→xen`, `catching→cách chinh`,
     `hover→hơ vơ`, `disable→đít xây bồ`, `record→ri cọt`, `result→ri dăn`,
     `tool/tools→tun`, `download→đao lốt`, `upload→áp lốt`; đồng bộ `-ing→-inh`:
     `monitoring→mo ni tơ rinh`, `rendering→ren đờ rinh`, `huggingface→hắc ghinh phây`.
  2. `stem.py`: sửa bug `d→"dê"` (trước ghi sai thành "đê"), thêm `đ/l/o`.
  3. `g2p.py` narrow (Bắc): trước `nh/ch`, `a→ăj`, `ê→ə̆j` (Henderson 1965; Kirby 2011);
     `anh` narrow = `ăj ŋ̟` — kèm test.
- ⏳ **Chờ cô/product chốt**: quy ước `/w/`, `/j/` cho giọng Bắc ("quít" vs "uýt"; "diu" vs "iu");
  kiểu đọc mặt chữ vs IPA (`data/module/status/tutorial`); cụm phụ âm 3 âm ("subscribe");
  style bảng tên chữ (tên-chuẩn vs ngắn) + đồng bộ `oov.py` ↔ `stem.py`.
- (Nhãn `anh/ach` giữ SGK — đã thống nhất, không đổi; đo formant kiểm chứng sau khi có model.)
