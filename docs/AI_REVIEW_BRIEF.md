# ViFastTts — Hồ sơ bàn giao để AI khác review (snapshot có cầu nối tới repo)

> **Mục đích.** Tài liệu này mô tả **toàn bộ các phần cốt lõi của repository ViFastTts ở trạng thái
> as-built**, đủ tự chứa (self-contained) để một AI/người review không có quyền truy cập repo vẫn
> đánh giá được kiến trúc, dữ liệu, quyết định thiết kế, điểm mạnh/yếu. Người dùng sẽ giữ vai trò
> trung gian; **agent đã viết tài liệu này (gọi là "agent cầu nối") có thể xác minh mọi chi tiết
> trong repo** (grep/đọc file/chạy test/đo thử) theo yêu cầu.
>
> **Snapshot lấy lúc:** 2026-10-12, HEAD `1bd127f` (rev.6; đứng sau `3e832c7` brief, `c9896f7` reparse,
> `379884c` B4, `6b1ffea` rev.5, `17ae651` rev.4).
> **Trạng thái working tree lúc soạn:** sạch — chính sách "checkpoint sạch" rev.6 §1a và demo nghe
> thử theo mốc đã được commit ở `1bd127f` (mô tả ở §2.4).

---

## 0. Cách dùng tài liệu này (giao thức review)

### 0.1. Quy ước nhãn độ tin cậy
Mọi số liệu quan trọng trong tài liệu này được gắn một trong bốn nhãn:

| Nhãn | Nghĩa |
| --- | --- |
| **[TEST]** | Đã kiểm chứng bằng test tự động trong repo (chạy được, có kết quả) |
| **[ĐO]** | Số đo thực trên máy tham chiếu (Mac mini M6 32GB), ghi lại trong docs/repo |
| **[TÀI LIỆU]** | Lấy từ tài liệu/hướng dẫn trong repo, chưa tự chạy lại |
| **[KẾ HOẠCH]/[ƯỚC LƯỢNG]** | Chưa xảy ra hoặc là suy đoán có cơ sở |

### 0.2. Điều nên làm khi review
1. **Review kiến trúc & giả định**: vì sao 5 thành phần + 7 ID thay vì IPA/BPE; vì sao NAR + teacher
   forcing; vì sao tách vocoder thành 2 graph ONNX. Các quyết định này được ghi rõ ở §12.
2. **Tìm lỗ hổng logic / rủi ro dữ liệu**: alignment placeholder, xử lý fallback, chia val, nguy cơ
   overfit đã quan sát được (§13.2).
3. **Chỉ ra chỗ tài liệu nói một đằng code làm một nẻo** — mục §15 đã liệt kê sẵn các chỗ lệch tìm
   thấy trong đợt audit này; hãy soi thêm.
4. **Đặt câu hỏi kiểm chứng được**: ưu tiên dạng "file X dòng Y có đúng là…?", "đo lại Z giúp tôi".

### 0.3. Điều không nên đề xuất (ràng buộc cứng của repo)
- ✋ **Không đề xuất đổi cấu hình mel** (n_fft/hop/f_max/clamp/stats) tùy tiện: mel đã được "đóng
  băng" bằng hash trong checkpoint (§5.2); đổi sẽ buộc **train lại vocoder từ đầu**.
- ✋ **Không đề xuất chạy acoustic + vocoder song song ngay bây giờ**: máy tham chiếu đã chốt lịch
  "MFA (CPU) ∥ vocoder (MPS)" — acoustic chạy một mình (§13.3).
- ✋ **Không đề xuất hạ hop 256**: đã đo và chốt giữ hop 256 (đổi hop = làm lại stats + vocoder).
- ✋ Không kỳ vọng repo có sẵn trọng số tốt: mọi checkpoint hiện có **chỉ để test pipeline** (§13.2).
- ✓ Nếu đề xuất cần số liệu, ghi rõ **cần đo gì, trên phần cứng nào**, để agent cầu nối chạy và trả số.

### 0.4. Cách gửi yêu cầu xác minh cho agent cầu nối
Định dạng đề nghị:
```
CÂU HỎI: <câu hỏi cụ thể>
GIẢ THUYẾT: <điều đang nghi ngờ>
CẦN KIỂM: <file/hàm/test/lệnh nên xem hoặc chạy>
```
Agent cầu nối sẽ trả kết quả kèm **trích dẫn file:dòng** trong repo.

### 0.5. TL;DR tiếng Anh (cho review nhanh)
ViFastTts is a Vietnamese single-speaker TTS toolkit: a deterministic text frontend that decomposes
each Vietnamese syllable into 5 phonological components (onset, medial, nucleus, coda, tone),
encoded as 7 discrete IDs per syllable (+ boundary + punctuation); a non-autoregressive
FastSpeech2-style acoustic model (Conformer-lite encoder/decoder, duration/F0/VUV/energy predictors,
optional postnet + phonation channels); and an iSTFTNet/Vocos vocoder trained adversarially
(MPD-full + MRD + MR-STFT). Pipelines: manifest → train (MPS/CUDA/CPU) → synthesize → ONNX export
(fp32/fp16/int8) → PyTorch-free ONNX runtime (numpy iSTFT). 2257 tests pass; mypy clean; ruff
currently flags one F841 inside an uncommitted WIP (clean-checkpoint policy). Real dataset: 13,215
YouTube clips (51.7 h, CC BY 4.0). Current known gaps: real forced alignment (MFA) not yet run,
duration labels are placeholder, SIL token and log-F0 DCT are specced but not implemented, INT8
calibration incomplete.

---

## 1. Dự án là gì (30 giây)

**ViFastTts** = toolkit nghiên cứu/triển khai **TTS tiếng Việt một giọng**, mỗi checkpoint là
một người nói, một phương ngữ — **không** `speaker_id`, `dialect_id`, audio tham chiếu.

```text
Văn bản ──▶ FRONTEND (tiếng Việt) ──▶ MODEL acoustic (mel + duration/f0/vuv/energy) ──▶ VOCODER ──▶ WAV 24 kHz
```

Ba khối:

| Khối | Nhiệm vụ | Cốt lõi kỹ thuật |
| --- | --- | --- |
| **Frontend** | Chuẩn hoá text → tách âm tiết → **7 ID/âm tiết** | Parser âm tiết 5 thành phần, normalizer số/ngày/giờ + fallback Anh/OOV |
| **Acoustic model** | 7 ID → mel + duration/f0/vuv/energy (+phonation tùy chọn) | NAR kiểu FastSpeech2, Conformer-lite, LengthRegulator vector hoá export ONNX |
| **Vocoder** | mel → waveform 24 kHz | iSTFTNetLite (3,5M) hoặc Vocos-style ConvNeXt (12,07M); GAN MPD-lite hoặc MPD đầy đủ + MRD + MR-STFT |

**Khác biệt cốt lõi:** thay vì ký tự/BPE/IPA, mỗi âm tiết là bộ 5 category ngôn ngữ học
`(ONSET, MEDIAL, NUCLEUS, CODA, TONE)` → 7 ID rời rạc. Nhờ vậy: phát âm ổn định, không UNK với
tiếng Việt chuẩn, thanh điệu không bao giờ sai vị trí, vocab cực nhỏ (68 ID), export ONNX gọn.

**Tính chất:** tất định (không sampling); train dùng duration thật (teacher forcing), suy luận tự
dự đoán duration; `--speed` co giãn duration; chunk câu dài tự động; OOV không bao giờ crash.

---

## 2. Snapshot trạng thái & môi trường

### 2.1. Chất lượng kỹ thuật tại snapshot

| Hạng mục | Kết quả | Nhãn |
| --- | --- | --- |
| Test | **2257 passed** trong ~18 giây (`python -m pytest -q`) | [TEST] (agent cầu nối chạy lại) |
| `ruff check .` | **1 lỗi F841** ở `src/vifasttts/train.py:548` — thuộc WIP chưa commit (§2.4); trước đó sạch | [TEST] |
| `mypy src/vifasttts` | Sạch, 49 source file | [TEST] |
| Parser audit vét cạn | 49.166 tổ hợp hợp lệ — **0 lỗi**; wordlist 39K từ: 6.617/6.633 parse OK | [TÀI LIỆU] |
| Audit mở rộng Viet74K | 73.901 mục: 0 lỗi cấu trúc, 0 rime ngoài bảng | [TÀI LIỆU] |

> ⚠️ **Phát hiện của đợt audit này (đáng sửa):** lệnh `pytest` trần (đúng như README hướng dẫn)
> **fails collection** với 4 module `tests/test_fallback_review.py`, `tests/test_clip_quality.py`,
> `tests/test_mfa_corpus.py`, `tests/test_reparse_overrides.py` vì chúng `import tools.*` mà
> `tools` không nằm trên `sys.path` (không có `__init__.py`, không có `conftest.py`).
> Chạy `python -m pytest` thì qua đủ 2257. Xem §15.1.

### 2.2. Máy tham chiếu (mọi số đo MPS)
Mac mini **M6, 32GB RAM hợp nhất**, CPU 12 nhân, GPU 12 nhân, Neural Engine Dual 16-core (PyTorch
**không** dùng ANE), Metal 4 [TÀI LIỆU: docs/APPLE_SILICON.md, docs/TRAINING_ESTIMATES.md §3.1].

### 2.3. Trạng thái dữ liệu & run trên máy lúc snapshot

| Thứ | Trạng thái |
| --- | --- |
| `data/codelung/` (~17 GB) | 13.215 clip · **51,73h** · 24 kHz mono; manifest 0 lỗi sau recut B3; train.jsonl 11.424 mẫu (~44,96h), val.jsonl 1.791 mẫu (chia theo video) |
| `data/codelung/mel_stats.json` | Stats chuẩn hoá per-bin tính lại trên **2.000 clip** sau recut (2.683.036 frame) — đã chốt để đóng băng mel |
| `data/codelung/mfa/` | Corpus MFA **đã dựng** (wavs symlink + labs + lexicon + phones.txt; phone set có tiền tố `o_/m_/n_/c_`, tổng 121 phone) — **chưa train MFA** |
| `data/codelung/review/` | Gói 400 từ fallback + clip mẫu cho người dùng nghe (chờ điền `filled.txt`) |
| `runs/small_codelung/` | Run acoustic `small` cũ **dừng ở 20.850 bước** (val tăng ngược từ 8k: 274,0 → 305,5 do overfit nhãn duration chia đều); `best.pt` @8k chỉ để test pipeline |
| `runs/vocoder_codelung/` | Vocoder cũ (MPD-lite, log-power) dừng ~46.190 bước — cũng chỉ để test |
| `cache/codelung/` | Cache feature đang được **tính lại** (đã xoá bản cũ); lúc soạn: ~1.300/~13.215 file |
| Pipeline nền đang chạy | Chuỗi: `compute_mel_stats (xong) → cache train → cache val → train_vocoder_chunked → runs/vocoder_codelung_v2 (100k bước, chunk 5k)` |

### 2.4. WIP chưa commit lúc soạn (rev.6 §1a — chính sách checkpoint sạch)
- `src/vifasttts/train.py`: kiểm NaN/Inf gộp mỗi 10 bước; **chỉ lưu checkpoint ở bước vừa kiểm
  sạch**; lưu atomic qua `last.pt.tmp`; xoay bản cũ thành `prev.pt`; ghi thêm `clean: true`,
  `last_loss`; cuối run không lưu nếu cửa sổ hiện tại có NaN.
- `tools/train_vocoder_chunked.py`: `_pick_checkpoint()` chọn `last.pt → prev.pt`, bỏ qua
  checkpoint `clean: false` hoặc `last_loss` không hữu hạn.
- Lỗi lint duy nhất của repo nằm trong WIP này: `last_clean_loss` gán mà không dùng
  (`train.py:548`) → nên đổi thành `_ = window_value` hoặc dùng lại biến.

*(Nếu các thay đổi này đã được commit sau snapshot, đánh giá chúng như phần chính thức của repo.)*

---

## 3. Bản đồ repository

### 3.1. Cây thư mục
```text
ViFastTts/
  pyproject.toml          # package "vifasttts" 0.1.0, Python ≥3.11, extras: dev/audio/f0/train/onnx/youtube/all
  README.md               # tổng quan + lệnh
  configs/                # 4 config YAML + overrides_tech_vi.json (từ lạ cho youtube-fetch)
  src/vifasttts/          # ~7,4k dòng Python (49 file .py)
    frontend/             # unicode, stem, normalizer, pipeline, syllable, tone, rules, orthography,
                          #   vocab, dialect, roman, g2p, english_fallback, oov, lexicon, segments,
                          #   chunking, benchmark
    data/                 # manifest, features, phonation, cache, dataset, split, report, subtitles, youtube
    models/               # components, conformer, vifasttts (acoustic), istftnet, discriminators
    runtime/              # onnx_engine, istft (numpy), postprocess
    train.py              # vòng train acoustic + vocoder
    infer.py              # text → WAV (PyTorch), Synthesizer, batch, streaming
    export.py             # ONNX acoustic + vocoder + fp16/int8
    server.py             # HTTP stdlib: /synthesize, /health, /info
    cli.py                # Typer CLI
    config.py, device.py, losses.py, types.py
  tools/                  # 12 công cụ (xem §10.3)
  tests/                  # 56 file, 2257 test
  docs/                   # 25 tài liệu (bản đồ ở §16)
  examples/               # parse.py, g2p_demo.py, minimal_pairs.txt
  data/ cache/ runs/      # dữ liệu thật (bị .gitignore; xem §2.3)
```

### 3.2. File lõi → vai trò (kèm số dòng lúc snapshot)

| File | ~LOC | Vai trò |
| --- | --- | --- |
| `frontend/syllable.py` | 167 | Tách 5 thành phần; mọi luật đặc thù `gi/qu/uy/ay…`; `validate()` bất biến âm vị |
| `frontend/rules.py` | 71 | Bảng chính tả ↔ âm vị + compatibility (dùng cho validator/audit) |
| `frontend/normalizer.py` | 320 | Số/ngày/giờ/tiền/%/khoảng; thứ tự ưu tiên token; dialect |
| `frontend/pipeline.py` | 91 | `parse_text`: normalizer → token → syllable → gắn dấu câu/ranh giới; fallback không crash |
| `frontend/vocab.py` | 53 | `ComponentVocabs` 7 bảng → encode 7 ID |
| `frontend/g2p.py` | 255 | Chiếu 5 thành phần → IPA Bắc/Trung/Nam, broad/narrow — **chỉ phân tích, không runtime** |
| `frontend/lexicon.py` | 208 | Profile `open/strict/corpus` + Lexicon shape |
| `frontend/segments.py` | 185 | Segment Expander (tách IÊ/UÔ/ƯƠ, trọng số duration theo role, largest remainder) |
| `frontend/english_fallback.py` | 282 | Bảng ~670 từ công nghệ + 31 acronym + 6 tên sản phẩm; bất biến "mọi reading parse được" |
| `frontend/oov.py` | 49 | Resolver OOV 4 bước opt-in (acronym/product/English/đánh vần chữ cái) |
| `frontend/chunking.py` | 89 | Chia câu → mệnh đề → từ; `pause_seconds` 0.15/0.08/0.05s |
| `data/features.py` | 269 | Mel (log-power cũ / log-magnitude mới + chuẩn hoá toàn cục), energy, F0 (autocorr/pyworld), **mel hash** |
| `data/dataset.py` | 260 | TtsDataset/VocoderDataset, collate + mask, `allocate_durations` (placeholder) |
| `data/cache.py` | 118 | Cache feature `.pt` khoá SHA1(audio+size+mtime+mel config+stats hash+f0+phonation) |
| `data/manifest.py` | 91 | build/validate manifest + coverage report |
| `data/phonation.py` | 127 | HNR, spectral tilt, glottalization cấp frame (tùy chọn) |
| `data/youtube.py` | 809 | `youtube-fetch`: yt-dlp + ffmpeg subprocess, phụ đề json3, cắt clip + QC, recut B3, respell mã |
| `models/vifasttts.py` | 147 | Model acoustic + `LengthRegulator` vector hoá (cumsum + gather) |
| `models/conformer.py` | 120 | ConformerLite (PE hình sin, self-attention additive mask viết tay, depthwise conv) |
| `models/istftnet.py` | 159 | `ISTFTNetLite` + `ISTFTNetVocos` (ConvNeXt); `build_generator()` |
| `models/discriminators.py` | 234 | MPD-lite, MPD đầy đủ 2/3/5/7/11, MRD, MR-STFT, các hàm loss GAN |
| `train.py` | 584 | Vòng train acoustic + vocoder; LR schedule; early stop; resume; MPS hardening |
| `infer.py` | 248 | `Synthesizer`, `synthesize`, `iter_synthesize`, `synthesize_batch`; kiểm mel hash khi load |
| `export.py` | 204 | Export ONNX acoustic (fp32/fp16/int8) + vocoder (spectrum) + metadata |
| `runtime/onnx_engine.py` | 248 | Chạy ONNX không cần PyTorch; chọn provider; chunk câu; `--speed` xấp xỉ |
| `runtime/istft.py` | 55 | iSTFT numpy khớp `torch.istft(center=True, hann)` |
| `runtime/postprocess.py` | 41 | Fade 10/20 ms, peak-normalize −3 dBFS |
| `server.py` | 197 | HTTP stdlib, lock tuần tự, giới hạn 2000 ký tự / speed 0.25–4.0 |
| `cli.py` | 401 | Toàn bộ lệnh CLI (Typer) |
| `losses.py` | 78 | `masked_l1` (+ mean_over_bins) và `acoustic_losses` |

---

## 4. Frontend — nguồn sự thật của tiếng Việt

### 4.1. Ba tầng biểu diễn của một âm tiết (`types.py`)
Mỗi `ParsedSyllable` mang đồng thời:
1. `Orthography` — chính tả gốc (`gi/qu`, `i/y`, vị trí dấu cũ/mới…).
2. `AbstractSyllable` — **dạng âm vị học**: 5 category trung lập phương ngữ `(onset, medial, nucleus, coda, tone)` —
   đây là khoá vocab cho model. Có property `rime = (medial, nucleus, coda)`.
3. Dạng ngữ âm IPA — sinh **một chiều** bởi `frontend/g2p.py` (Bắc/Trung/Nam), chỉ dùng phân tích.

`ParseStatus = VALID | INVALID | FALLBACK_REQUIRED`; token lạ giữ `raw`, `confidence=0`, ID mặc định
`(NONE, NONE, A, NONE, NGANG)` để pipeline không bao giờ crash.

### 4.2. Đường ống parse (`frontend/pipeline.py`)
```text
text
 → normalize_text(dialect, options)      # unicode NFC → giờ 14h30 → STEM → dấu nghìn →
                                         # ngày → giờ:phút → điện thoại → tiền tệ → phân số →
                                         # khoảng → % → số thập phân/số nguyên → 5G/4G → words
 → token hoá (\w+ | ký tự không phải chữ)
 → per token chữ: safe_parse_syllable(token)
      • OK → ParsedSyllable
      • FAIL + resolve_oov=True → thử respell (acronym/product/English/chữ cái) rồi parse lại
      • FAIL → _fallback_syllable (giữ raw, FALLBACK_REQUIRED)
 → dấu câu: gắn vào âm tiết trước; ".!?" → SENTENCE, ",;:" → PHRASE (word_end luôn True)
```

### 4.3. Quy tắc parser đặc thù (đã kiểm bằng test) — tóm tắt để review
| Ca | Cách xử lý | Ý nghĩa |
| --- | --- | --- |
| `gi` dùng chung chữ i | `gì/gìn` → onset GI + nucleus I; `giếng` → GI + IÊ (cùng vần `tiếng`); `già/giữ/giường` → digraph `gi` câm | Tránh mất /z/ |
| `qu` | = `K` + `W`; sau `qu` chỉ hợp lệ `a ă â e ê i/y ô ơ`; `qua` = K+W+A | `quốc` = K+W+Ô+C |
| `uy` vs `ui` | `uy` = medial W + nucleus I (`thuỷ` — dấu trên y); `ui` = nucleus U + coda J (`tủi` — dấu trên u) | Hai vần khác nhau thật |
| `iê/yê/ia/ya`, `uô/ua`, `ươ/ưa` | Cùng nucleus trừu tượng IÊ/UÔ/ƯƠ; chính tả giữ ở Orthography | Song ánh ở cặp (tuple + chính tả dư) |
| `oo` từ vay | **Nucleus `OO` riêng** [ɔː] (`xoong` ≠ `xong`), dấu trên chữ o thứ hai | Phản hồi người bản ngữ |
| `ay/au` vs `ai/ao` | `ay/au` → **Ă ngắn** [a]; `ai/ao` → **A dài** [aː] (`tay≠tai`, `rau≠rao`) | Help:IPA/Vietnamese |
| Coda tắc P/T/C/CH | Chỉ hợp lệ với thanh SẮC/NẶNG | `validate()` chặn |
| Âm đệm W | Chỉ kết hợp `A Ă Â E Ê I Ô Ơ IÊ` | `MEDIAL_NUCLEUS_COMPATIBILITY` |
| `q` trần | Không hợp lệ (`qa/qi` → fallback) | — |
| Token ngoài Latin | Giữ nguyên thành fallback (CJK, Cyrillic…) | Không nuốt chữ |

Vị trí dấu thanh: chuẩn hoá về **lối hiện đại** (`hoà`, `thuỷ`, `tuỳ`) theo QĐ 1989/2018/QĐ-BGDĐT;
có thể tắt bằng `canonical_tone_spelling=False`. Parser chấp nhận cả hai lối; model không bị ảnh
hưởng vì chỉ thấy tuple.

### 4.4. Vocab 7 ID (`frontend/vocab.py`)
| Vị trí | Trường | Số giá trị | Ghi chú |
| --- | --- | --- | --- |
| 0 | onset | 24 | NONE + 23 phụ âm đầu |
| 1 | medial | 2 | NONE / W |
| 2 | nucleus | 15 | A Ă Â E Ê I O Ô Ơ U Ư IÊ UÔ ƯƠ **OO** |
| 3 | coda | 11 | NONE M N NG NH P T C CH J W |
| 4 | tone | 6 | NGANG HUYEN SAC HOI NGA NANG |
| 5 | boundary | 4 | NONE(padding) / WORD / PHRASE / SENTENCE |
| 6 | punctuation | 6 | NONE COMMA PERIOD QUESTION EXCLAMATION OTHER |

Tổng vocab **68 ID**. Không UNK, không BPE. Không gian âm tiết trừu tượng hợp lệ **28.800**
(6.336 shape, 264 vần) — đối chiếu Kirby 2011 + lexicon 39K từ.

**Chưa có token SIL:** kế hoạch A3 (REVISION_PLAN_V2 rev.4) mô tả SIL ở vị trí cố định + tone NONE
→ vocab ~70, nhưng **chưa có trong code** (`ComponentVocabs.default()` vẫn 24/2/15/11/6 + 4 + 6).
Đây là việc đang chờ sau MFA (§15.2).

### 4.5. Normalizer (`frontend/normalizer.py`)
- **Thứ tự xử lý quan trọng** (đọc code theo đúng thứ tự): unicode → `14h30` → STEM (LaTeX/hoá
  học/đơn vị) → dấu nghìn kiểu Việt → ngày `d/m/y` → giờ `h:mm` → điện thoại (+84 → 0) → tiền
  (`đồng/đ/vnd/$`) → phân số → khoảng `3-5`/`10-20%` → phần trăm → số thập phân → số nguyên →
  `5G/4G` → từ (override → acronym → product → Roman → âm tiết Việt → English fallback).
- **Dialect** chỉ đổi cách đọc số: Bắc `nghìn/linh/tư`, Trung `ngàn/lẻ/tư`, Nam `ngàn/lẻ/bốn`.
- Phân loại hoa/thường chính xác: `AI→ây ai`, `Ai→Ai`, `ai→ai`; `VI→sáu`, `Vi/Vi→giữ nguyên`.
- Số La Mã chỉ nhận chuỗi hoa chuẩn ≥2 ký tự, 1–3999 (`IIII/VX/IC` không đổi).
- Ngày/giờ không hợp lệ **giữ nguyên** để rơi fallback thay vì đọc sai.

### 4.6. STEM, English fallback, OOV
- `frontend/stem.py`: bảng Hy Lạp, lệnh toán LaTeX, biến số, nguyên tố hoá học, đơn vị vật lý;
  `read_chemical_formula(strict=True)` tránh nhầm tên sản phẩm (RTX3080, MP3, N95, K2).
- `english_fallback.py`: 3 lớp bảng — acronym (phân biệt hoa thường), từ tiếng Anh (không phân biệt),
  tên sản phẩm. **Bất biến được test:** mọi reading phải parse được thành âm tiết Việt hợp lệ.
- `oov.py`: bước cuối đánh vần tên chữ cái (VNG → "vê en giê"), `confidence=0.5`, opt-in qua
  `parse_text(resolve_oov=True)`; mặc định tắt.

### 4.7. Profile & Lexicon, Segment Expander, Chunking
- `lexicon.py`: `open` (mặc định, không cảnh báo) / `strict` (luật chính tả bản ngữ: c/k, g/gh,
  ng/ngh, môi+W, coda nh/ch, iê→yê…) / `corpus` (shape phải có trong Lexicon). Chỉ **cảnh báo**,
  không chặn dữ liệu. CLI: `check-profile`, `build-lexicon`.
- `segments.py`: mở 5 thành phần thành chuỗi segment cấp âm vị, trọng số duration
  (onset 0.4, medial 0.3, nucleus 1.0, coda 0.35–0.6, tone 0), `distribute_frames` largest-remainder.
  Đây là "view" cho aligner tương lai, không thay đầu vào 7 ID.
- `chunking.py`: tách câu → mệnh đề → từ; nghỉ 0.15s cuối câu / 0.08s cuối mệnh đề / 0.05s còn lại.
- `benchmark.py`: đo số đơn vị/âm tiết giữa char 3,3 / slot 5,0 / phoneme 4,0 / rime 2,0 [ĐO].

### 4.8. G2P IPA (chỉ để phân tích/đối chiếu)
`frontend/g2p.py` chiếu **5 thành phần → IPA**, ba profile Bắc (Kirby 2011), Trung (xấp xỉ Huế),
Nam (Huỳnh Công 2003): Bắc gộp d/g/r → /z/, ch/tr → /tɕ/, s/x → /s/; Nam tách tương ứng; Trung dùng
âm đầu kiểu Nam + thanh Bắc (chưa có bảng Huế). Narrow: `t̪`, `p̚ t̚ k̚`, `k̟̚ ŋ̟`; Bắc narrow
`ưu/ươu → [iw]/[iəw]`. `--style narrow`, `--tone-format contour|number|none`. **Module này bị cấm
import bởi `models/`, `train.py`, `infer.py`, `pipeline.py`** (không nằm trên đường runtime).

---

## 5. Dữ liệu & đặc trưng

### 5.1. Manifest & dataset
- `build-manifest`: CSV `id|text` + WAV → JSONL `{id, audio_path, text, syllables:[ParsedSyllable...]}`.
- Trường `durations` (số frame mel/âm tiết) **tùy chọn** từ forced alignment. Không có →
  `allocate_durations()` chia đều (deterministic, placeholder — **nói rõ không phải production**).
- `TtsDataset` trả: `features [N,7]`, `durations [N]`, `f0/vuv/energy [N]` (trung bình các frame
  theo duration), `mel [T,100]`, `phonation [N,3]` (nếu bật). `collate_batch` pad + trả
  `syllable_lengths`, `mel_lengths`.
- `VocoderDataset` trả `mel [T,100]` + `wav [T*hop]` (cắt/pad đúng theo `T*hop`, center=True).
- Quan hệ: `T = 1 + len(wav)//hop`, hop 256 @ 24 kHz → 1 frame ≈ 10,67 ms.

### 5.2. Mel mới (đóng băng biểu diễn) — trạng thái "đã chốt"
- Cấu hình chạy thật (`small_codelung.yaml`): `n_fft=1024, hop=256, n_mels=100, f_min=0,
  f_max=12000`, **`log_magnitude: true`** (floor `log_floor=1e-5`, kiểu HiFi-GAN thay log-power 1e-6)
  và **`normalize_path: data/codelung/mel_stats.json`** (chuẩn hoá toàn cục per-bin `(mel−mean)/std`;
  stats từ 2.000 clip/2,68M frame, cache LRU theo mtime).
- **Đóng băng bằng hash:** `mel_config_hash()` = SHA1 của (mel config canonical + hash nội dung file
  stats) ghi vào **mọi checkpoint acoustic/vocoder** (`mel_hash`); khi `synthesize`/load mà lệch →
  `ValueError` rõ ràng; checkpoint cũ không có hash → bỏ qua (tương thích ngược). 3 test riêng.
- Cache feature cũng hash nội dung stats nên đổi stats là cache tự vô hiệu.
- Hệ quả (ràng buộc review): đổi bất kỳ tham số mel nào ⇒ **phải train lại vocoder**.

### 5.3. F0 / VUV / Energy / Phonation
- F0: mặc định cũ `autocorr` (rẻ, dễ lỗi quãng tám); config chạy thật dùng **`f0.method: pyworld`**
  = harvest + stonemask, interpolate qua vô thanh, đo ~21× thời gian thực/lõi [ĐO]; cần extra `[f0]`.
- VUV: từ ngưỡng voicing; Energy: log-power trung bình (scalar/âm tiết — quyết định F4).
- Phonation (tùy chọn, mặc định tắt): HNR (autocorr, dB), spectral tilt (dB/oct), glottalization
  (tỉ lệ năng lượng <80 Hz). Bật bằng `phonation.enabled` + `model.use_phonation` (predictor 3 kênh
  + loss `phonation`); chưa có đánh giá thính học.

### 5.4. Cache feature
`.pt` theo SHA1(đường dẫn + size + mtime + mel config + hash stats + f0 config + phonation config);
`precompute_features(workers=N)` dùng ProcessPool; đo thật 13.225 clip/51h = **33 giây với 8 workers**,
6,7 GB [ĐO]. Cache là **bắt buộc trên thực tế**: dataloader không cache = 6,34 s/batch (STFT + F0
trên clip 14s) vs model 0,478 s/bước [ĐO].

### 5.5. Dataset YouTube (`youtube-fetch`)
Flow: yt-dlp liệt kê video (cache `.videos.json`) → tải phụ đề json3 → làm sạch caption cuộn/nhãn
nhạc → tải audio mono 24 kHz → gộp câu theo max_gap/duration/chars → cắt clip có pad 60/120 ms +
fade 10/20 ms → QC (im lặng RMS < −45 dBFS, trùng chữ, fallback > 34%, mật độ phụ đề < 2 ký tự/s)
→ xuất audio/metadata.csv/manifest/subs SRT/stats/dataset card (mặc định **CC BY 4.0**)/`.done.json`.
- Resume bằng `.done.json`; retry 3 lần backoff 6/12s; `--recut` cắt lại từ raw+subs;
  `--respell-codes` (`v2→vê hai`, `MP3→em pê ba`, `n8n→en tám en`).
- **B3 nới biên VAD** (`_adjust_boundary`): nới tới mép khoảng lặng ≤200 ms; 4.745 biên được nới
  qua âm tiết bị cắt; cờ `speech/tight` trong clips.jsonl/stats.
- **B4 lọc rẻ** (`tools/clip_quality.py`, mới): speech-ratio <50%, clipping >1%, RMS <−45 dBFS →
  `quality.jsonl` + `quality_drop.txt`; `--apply` ghi đè manifest và lưu `excluded`.
- **reparse_overrides** (`tools/reparse_overrides.py`, mới): mở rộng overrides → parse lại text
  (không recut audio), cập nhật metadata/manifest + chia lại split.
- Làm sạch nội dung @codelung: bảng ~670 từ công nghệ + 31 acronym + 183 overrides kênh; suy từ ASR
  sai bằng ngữ cảnh (`vouer→voucher`, `clock→Claude`, `queen→Qwen`, `kch→cache`…); fallback từ
  **7,1% → 1,22%** âm tiết, 71% clip 0-fallback, đủ 6 thanh/11 coda [TÀI LIỆU].

### 5.6. MFA corpus builder (`tools/mfa_build_corpus.py`)
Sinh `wavs/` (symlink), `labs/` (câu respell = trường `normalized`), `lexicon.txt`,
`phones.txt`, `syllable_index.json`, `fallback_samples.txt/words.txt`. Phone set có tiền tố vị trí:
`o_<onset>`, `m_<medial>`, `n_<nucleus>_<tone>`, `c_<coda>` — thanh gắn liền nucleus, không align
thanh thành đoạn riêng. Corpus đã dựng, **chưa train MFA** (kế hoạch ~3–8h CPU, subset ~20h sạch
nhất rồi align toàn bộ) [KẾ HOẠCH].

---

## 6. Model acoustic

### 6.1. Kiến trúc (`models/vifasttts.py` + `models/conformer.py`)
```text
features [B,N,7]
 └─ ComponentEmbedding: 7 bảng embedding(component_dim) → concat → Linear(hidden)   [B,N,H]
 └─ Encoder ConformerLite ×enc_layers (PE hình sin + self-attention additive mask + depthwise conv + FFN thang 0.5)
      → syllable state x [B,N,H]
 ├─ Duration predictor ConvPredictor(H→1, duration_layers)  → log_duration [B,N]
 ├─ F0 / VUV / Energy predictors ConvPredictor(H→1)         → [B,N] mỗi cái
 ├─ (tùy chọn) Phonation predictor ConvPredictor(H→3)       → [B,N,3]
 ├─ Tone classifier Linear(H→6)                             → tone_logits (aux, không dùng ở inference)
 └─ LengthRegulator(x, durations)
      · expanded [B,T,H], lengths [B], frame_mask [B,T]   (cumsum + gather, KHÔNG vòng lặp Python)
      · variance mở rộng theo frame (f0,vuv,energy[,phonation]) → Linear(3/6→H)
      · frame position 0..1 trong âm tiết → Linear(1→H)
 └─ Decoder ConformerLite ×dec_layers → Mel head Linear(H→100) → mel_pre [B,T,100]
      └─ (use_postnet) Postnet 5 lớp conv residual (256 kênh) → mel [B,T,100]
```
Chi tiết export-ONNX đáng chú ý: `durations = round(exp(log_d) − 1).clamp_min(1)` (**exp−1 thay
expm1** vì ONNX thiếu op tương ứng); LengthRegulator nhận `durations` nên **độ dài frame động**
theo input — đã kiểm bằng onnxruntime (B=1..3, N=3..12).

### 6.2. Bảng tham số (tính chính xác từ config)
| Biến thể | hidden / enc–dec / ffn / heads | Tham số acoustic | Vocoder tương ứng |
| --- | --- | --- | --- |
| tiny (test) | 16 / 1–1 / 32 / 2 | 14.142 | — |
| small | 192 / 4–2 / 768 / 4 | **6.139.054** (+postnet → **~7,38M**) | lite 192 / Vocos 384 |
| base (mặc định) | 256 / 6–4 / 1024 / 4 | **16.444.654** | lite 256 |
| large | 320 / 8–5 / 1280 / 8 | **32.156.654** | lite 384 |
| xl (ví dụ) | 384 / 10–6 / 1536 / 8 | 55.428.974 | — |

Vocoder **iSTFTNetLite hidden 256 = 3.527.426** (bản tham chiếu cũ). Generator **Vocos-style**
(`ISTFTNetVocos`, expansion 3×, hidden 384, 8 block) sau audit rev.5 = **12.069.762**
(net 7.123.968 · head 384→1026 k7 2.758.914 · up 1.180.032 · post 737.664 · pre 269.184).
Discriminator MPD đầy đủ + MRD: ch=16 → 7,7M (dùng cho run thật); ch=32 → ~31M.

### 6.3. Train vs Inference — bảng khác biệt then chốt
| Điểm | Train | Inference |
| --- | --- | --- |
| Duration | teacher forcing: duration thật (hoặc placeholder chia đều) | tự dự đoán `round(exp(log_d)−1)`, mỗi âm tiết ≥1 frame |
| Mask | từ duration thật | từ duration dự đoán |
| Dropout / grad | bật / có | tắt (`eval()` + `no_grad`) |
| Tone classifier | có loss phụ | không dùng output |
| Vocoder | train riêng (GAN) | nạp trọng số có sẵn (hoặc cảnh báo random) |
| `--speed` | — | chia duration dự đoán cho speed rồi forward lại |
| Tất định | seed cố định | cùng input → cùng output |

---

## 7. Loss & vòng huấn luyện

### 7.1. Acoustic loss (`losses.py` + `loss_weights`)
| Loss | Công thức | Mask | Trọng số base / small_codelung |
| --- | --- | --- | --- |
| `mel` | L1 trên `[B,T,100]`, **mean_over_bins=True** (sửa A1) | `mel_lengths` | 1.0 / 1.0 |
| `mel_pre` | L1 trên mel trước postnet (F2) | `mel_lengths` | — / 0.2 |
| `duration` | MSE giữa `log_duration` và `log1p(durations thật)` | cấp âm tiết | 1.0 |
| `f0` | L1 | cấp âm tiết | 1.0 |
| `vuv` | BCEWithLogits | cấp âm tiết | 0.5 |
| `energy` | L1 | cấp âm tiết | 0.5 |
| `tone` | CrossEntropy (aux) | cấp âm tiết | 0.25 / **0.1** (F3) |
| `phonation` | L1 3 kênh (tùy chọn) | cấp âm tiết | theo config |

### 7.2. Vòng lặp acoustic (`train.py: train_reference`)
- AdamW, lr 2e-4 (warmup tuyến tính), weight_decay 1e-4, grad clip 1.0.
- Thiết bị: `training.device` = `auto` (CUDA → MPS → CPU) | `cpu` | `cuda` | `mps`.
- **AMP chỉ bật khi CUDA** (`mixed_precision`); MPS giữ fp32 vì đo thật không lợi (0,52 vs 0,478 s/bước).
- LR schedule: `constant` (mặc định) hoặc `cosine` (`min_lr_ratio`, `lr_decay_steps` **toàn cục** —
  quan trọng khi chạy theo khối, tránh reset LR đầu mỗi khối).
- Early stop: `early_stop_patience: N` mốc val không cải thiện → ghi `early_stop.json`, chunked
  runner đọc để dừng. Resume `--resume` khôi phục model + optimizer + step + best_val; `history.csv`
  ghi tiếp.
- MPS hardening: `empty_cache` mỗi ~500 bước + sau validation; `validate_batches: 40`; watermark
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` / LOW=0.7 trong runner; checkpoint theo khối 5.000 bước
  (`tools/train_chunked.py` — mỗi khối là tiến trình mới để xả sạch bộ nhớ Metal).
- `resolved_config.json` ghi config thực dùng.

### 7.3. Vòng lặp vocoder (`train.py: train_vocoder_reference`)
```text
mỗi bước (VocoderDataset: mel [T,100] + wav [T*hop], random crop crop_frames=64):
  fake = G(mel_real)
  ── D ──  real/fake detach → discriminator_loss(BCE real→1, fake→0) → grad clip 5.0 → step
  ── G ──  g = adv·BCE(fake→1) + fm·feature_matching + mel_weight·mel L1
                + stft_weight·MR-STFT (spectral convergence + log-mag L1)  → grad clip → step
```
- Trọng số thật trong `small_codelung.yaml`: `mel_weight: 5.0` (đã hạ từ 45 sau khi mel được chuẩn
  hoá toàn cục), `feature_matching_weight: 2.0`, `adversarial_weight: 1.0`, `stft_weight: 2.0`,
  `grad_clip: 5.0` (GAN; acoustic giữ 1.0).
- `torch.stft` complex không chạy MPS → mel loss của vocoder **tính trên CPU** rồi autograd truyền
  ngược qua copy thiết bị về MPS (không cần cấu hình).
- Ổn định hóa: (rev.5) kiểm NaN/Inf gộp mỗi 10 bước, một sync; NaN → raise để runner resume + đổi
  seed (`--seed`) và hạ LR 0.5× nếu lặp (`--lr-scale`). (WIP rev.6) chỉ lưu checkpoint tại bước
  kiểm sạch, atomic + `prev.pt` (§2.4).
- Vocoder dùng `train.jsonl` **gồm cả clip fallback** (GAN không cần text); val tách riêng để đo
  CER copy-synthesis đúng.

### 7.4. Số đo huấn luyện tham chiếu [ĐO]
| Hạng mục | Số |
| --- | --- |
| Acoustic small batch 16 (~29k frame/bước) | 0,478 s/bước model MPS; ~0,67–0,75 s/bước thực tế |
| Base (dự đoán) | 0,75–1,25 s/bước · 120–160k bước ≈ 30–48h |
| Vocoder Vocos + MPD/MRD ch=16 crop 64 | **0,659 s/bước MPS** → 100k ≈ 19–20h; MPS nhanh hơn CPU 3,8–4,7× |
| Vocoder lite cũ MPD-lite | 0,166 s/bước CPU (MPS chậm hơn 2,4×) |
| Cache 8 workers | 33 giây/13.225 clip (51h) |
| Inference base (acoustic+vocoder) | RTF ≈ 0,08 trên CPU 2 luồng [ĐO cũ] |

---

## 8. Vocoder & discriminator

### 8.1. Generator (`models/istftnet.py`)
- `ISTFTNetLite` (mặc định cũ): pre conv → 2 lớp conv → ConvTranspose ×upsample_ratio → post →
  head dự đoán `(real, imag)` cho iSTFT. `hop_length` trong graph = `hop/ratio` (hop 256, ratio 4 → 64).
- `ISTFTNetVocos`: 8 `ConvNeXtBlock` (dwconv k7 → LayerNorm → pwconv ×expansion → GELU → pwconv →
  gamma) hidden 384; **expansion 3×** (đúng Vocos gốc; trước là 4×, đã audit rev.5).
- `build_generator(vt, mel_config)` chọn theo `vocoder_training.generator: lite|vocos`.
- Output export: `(spec_real, spec_imag)` thay vì tensor complex (`aten::complex` không hỗ trợ ONNX);
  iSTFT ghép ở runtime bằng numpy.

### 8.2. Discriminator & loss (`models/discriminators.py`)
- `MultiPeriodDiscriminatorLite`: 3 period (2/3/5), 3 tầng conv — mặc định cũ, smoke test.
- `MultiPeriodMultiResolutionDiscriminator`: **MPD đầy đủ** periods 2/3/5/7/11 (kênh c/4c/16c/32c/32c,
  như HiFi-GAN) **+ MRD** `SpectralDiscriminator` trên STFT real/imag, 3 độ phân giải mặc định
  `(1024,256,1024), (2048,512,2048), (512,128,512)`.
- Các hàm loss: `discriminator_loss` (BCE), `generator_adversarial_loss`, `feature_matching_loss`
  (L1, real detach), `multi_resolution_stft_loss` (spectral convergence + log-mag L1).

### 8.3. Kết luận đo chọn cấu hình (đã chốt — REVISION_PLAN_V2 §5)
- ch=16, **100k bước**, bỏ ch=32, không chia 2 pha; crop 64, không curriculum; generator Vocos;
  **không** AMP/fp16 cho GAN; bỏ `torch.compile`.

---

## 9. Inference, export, runtime, deployment

### 9.1. Đường PyTorch (`infer.py`)
`Synthesizer` nạp acoustic + vocoder 1 lần (kiểm `mel_hash` cả hai), `iter_synthesize` yield từng
đoạn + khoảng nghỉ; `synthesize` ghép + fade + tùy chọn peak-normalize → WAV PCM16 24 kHz;
`synthesize_batch` cho nhiều câu; `load_overrides` nạp JSON "mặt chữ → cách đọc".
Thứ tự nạp vocoder: `--vocoder-checkpoint` → khóa `vocoder` trong checkpoint acoustic → cảnh báo
random (chỉ debug).

### 9.2. Export ONNX (`export.py`)
- `torch.onnx.export` TorchScript exporter (`dynamo=False` khi torch hỗ trợ), opset 18 mặc định.
- Acoustic outputs (đúng thứ tự): `mel [B,T,100], log_duration [B,N], durations [B,N], f0 [B,N],
  vuv_logits [B,N], energy [B,N], [phonation [B,N,3] nếu bật], tone_logits [B,N,6], lengths [B]`.
  Dynamic axes: batch + số âm tiết; độ dài frame động theo durations.
- Metadata nhúng: `sample_rate`, `n_mels` (acoustic); `n_fft`, `hop_length`, `upsample_ratio`
  (vocoder) để runtime không cần config.
- Precision: `fp32` | `fp16` (`convert_float_to_float16 keep_io_types`) | `int8` (dynamic QUInt8).
  Ghi qua file trung gian rồi rename — `output` luôn hoàn chỉnh.

### 9.3. Runtime ONNX không cần PyTorch (`runtime/`)
`OnnxAcousticModel` (7 ID → outputs) + `OnnxVocoder` (mel → real/imag → `istft` numpy overlap-add
khớp `torch.istft(center=True)`) + postprocess. `--provider auto|cpu|cuda|coreml|tensorrt`;
trên Mac **CPU nhanh hơn CoreML EP** (CoreML chỉ nhận ~129/807 node do shape động) [ĐO]; CoreML
cache ở `~/.cache/vifasttts/coreml` (env `VIFASTTTS_COREML_CACHE`). `--speed` trên đường ONNX là
**xấp xỉ** (co giãn mel nearest-frame). Chưa gộp vocoder vào 1 graph vì ONNX thiếu op ISTFT —
là lựa chọn có chủ đích, ghi trong `docs/DEPLOYMENT.md`.

### 9.4. Server & batch
`server.py` stdlib: `GET /health`, `GET /info`, `POST /synthesize` (JSON → `audio/wav`); giới hạn
2000 ký tự, speed 0.25–4.0, lock tuần tự, không auth — dùng nội bộ/nghiên cứu.

---

## 10. Config, CLI, tools

### 10.1. Config (`configs/`)
| File | Đặc điểm chính |
| --- | --- |
| `base_24k.yaml` | Mặc định cho 5–30h: 16,4M; mel log-power cũ; F0 autocorr; MPD-lite; `use_postnet: false` |
| `small_24k.yaml` | <5h: 6,1M |
| `large_24k.yaml` | ≥40–50h: 32,2M |
| `small_codelung.yaml` | Run thật @codelung (chi tiết dưới) |
| `overrides_tech_vi.json` | ~183 từ OOV công nghệ cho `youtube-fetch --overrides` |

`small_codelung.yaml` — bản chạy thật (đã đo/chốt):
- `mel.log_magnitude: true` + `mel.normalize_path: data/codelung/mel_stats.json`;
  `f0.method: pyworld` (70–400 Hz, interpolate).
- Acoustic small + `use_postnet: true`; batch 16; **max_steps 60k**; warmup 6k; **cosine** tới 5%
  base_lr với `lr_decay_steps: 60000`; `early_stop_patience: 25`; `validate_every: 1000` +
  `validate_batches: 40`; `save_every: 1000`; `num_workers: 6`; `cache_dir: cache/codelung`;
  `loss_weights.tone: 0.1`.
- Vocoder: **100k bước**, generator **vocos** (blocks 8, expansion 3, hidden 384), discriminator
  **mpd_mrd** ch=16, `crop_frames: 64`, `stft_weight: 2.0`, `mel_weight: 5.0`, `grad_clip: 5.0`,
  `device: auto` (→ MPS cho MPD+MRD).

### 10.2. CLI chính (`vifasttts ...`)
| Nhóm | Lệnh |
| --- | --- |
| Frontend | `parse`, `parse-syllable`, `check-profile`, `g2p` |
| Dữ liệu | `build-manifest`, `validate-manifest`, `build-lexicon`, `cache-features`, `youtube-fetch` |
| Train | `train`, `train-vocoder` (có `--resume`, `--max-steps`, `--seed`, `--lr-scale`) |
| Sinh giọng | `synthesize`, `synthesize-batch`, `serve` |
| Export/runtime | `export-onnx` (`--precision fp32\|fp16\|int8`), `export-vocoder-onnx`, `synthesize-onnx` (`--provider`) |
| Khác | `--version` |

### 10.3. Tools (`tools/`)
| Tool | Việc |
| --- | --- |
| `parser_audit.py` | Vét cạn 49.166 tổ hợp + audit wordlist + coverage shape/rime; `--profile strict\|corpus` |
| `frontend_benchmark.py` | So sánh char/slot/phoneme/rime |
| `compute_mel_stats.py` | Tính mean/std per-bin (chuẩn hoá toàn cục); song song `--workers` |
| `fallback_review.py` | Gói 400 từ fallback + clip mẫu + `suggest_reading`, merge `filled.txt` vào overrides |
| `clip_quality.py` | **B4**: speech-ratio/clipping/RMS/flatness → drop list; `--apply` |
| `reparse_overrides.py` | Parse lại text sau khi mở rộng overrides (không recut) + chia lại split |
| `mfa_build_corpus.py` | Corpus MFA từ manifest (wavs/labs/lexicon/phones/syllable_index) |
| `dataset_report.py` / `split_manifest.py` | Báo cáo per-video / chia train–val theo video |
| `train_chunked.py` / `train_vocoder_chunked.py` | Chạy theo khối 5k bước, tự resume, watermark MPS, xử lý NaN |
| `run_small_codelung.sh` | Pipeline 5 bước (cache → acoustic → vocoder → ONNX → demo); `PARALLEL=1` để chạy song song |

---

## 11. Test & QA

- **2257 test**, 56 file, chạy ~18 giây; phủ: parser/matrix/edges, normalizer (số, ngày, dialect,
  precedence), STEM, G2P, lexicon/profile, OOV/English fallback, bijection chính tả, tone placement,
  segments/projection/benchmark, chunking, features (batched, representation, mel hash), cache,
  phonation, manifest/dataset/cấu hình, **neural smoke end-to-end** (train 2 bước → synth → export
  ONNX fp32/int8/fp16 → onnxruntime chạy batch/độ dài động), vocoder (MPD-lite/MPD+MRD/MR-STFT,
  dataset, train smoke, CLI), runtime ONNX + iSTFT numpy khớp torch, server, postprocess, device,
  youtube/subtitles offline (yt-dlp giả), clip_quality (B4), reparse_overrides, MFA corpus.
- Lệnh kiểm tra: **`python -m pytest -q`** (an toàn) · `ruff check .` · `mypy src/vifasttts` ·
  `python tools/parser_audit.py`.
- Bằng chứng audit parser (từ docs, chưa chạy lại ở đợt này): ma trận 49.166 tổ hợp 0 lỗi;
  wordlist 39K: 6.617/6.633 OK (16 fallback là typo/từ vay); Viet74K: 0 lỗi cấu trúc.

---

## 12. Nhật ký quyết định (đã chốt — không mở lại trừ khi có bằng chứng mới)

1. **Đầu vào neural = 5 thành phần + boundary/punctuation (7 ID), không IPA/BPE/ký tự.**
   Lý do: chính tả Việt gần ngữ âm; IPA Bắc gộp d/gi/r, ch/tr, s/x → mất phân biệt; tone là biến
   rời rạc; vocab nhỏ, không UNK; G2P chỉ dùng build-time.
2. **Song ánh nằm ở cặp "chữ canonical ↔ (tuple + chính tả dư)"**, không ở tuple thuần (đồng âm/
   biến thể i-y). Nên `ParsedSyllable` giữ cả hai tầng.
3. **Inventory chốt:** 24 onset / 2 medial / 15 nucleus (gồm `OO`) / 11 coda / 6 tone.
   `q` trần không hợp lệ; `uy` = W+I; `oo` là nucleus riêng; `ay/au` = Ă ngắn vs `ai/ao` = A dài.
4. **Tone spelling:** mặc định lối hiện đại (QĐ 1989/2018/QĐ-BGDĐT), cài đặt được; chỉ ảnh hưởng tầng chữ.
5. **Kiến trúc NAR FastSpeech2 + Conformer-lite** (depthwise conv rẻ, ONNX tốt), có frame-position,
   variance adaptor; tone classifier chỉ là loss phụ (F3: giữ, trọng số 0.1, mask SIL sau này);
   energy scalar (F4); bỏ `torch.compile` (E3).
6. **Mel mới (A1/§2 REVISION_PLAN):** log-magnitude clamp 1e-5 + chuẩn hoá toàn cục per-bin; A1 mel
   loss mean/bin. **Mel đóng băng bằng hash** ở checkpoint từ rev.5; đổi mel = train lại vocoder.
7. **F0:** bỏ autocorrelation → pyworld harvest+stonemask; **contour DCT/log-F0 chưa code** (đang chờ).
8. **Vocoder production path:** Vocos generator + MPD đầy đủ 2/3/5/7/11 + MRD + MR-STFT; ch=16,
   crop 64, 100k bước, MPS; D4 không AMP/fp16 cho GAN; GTA fine-tune để sau.
9. **Dữ liệu:** cắt clip tại SIL thật của MFA (3–10s, không cắt giữa âm tiết) — *chờ MFA*;
   B3 nới biên ≤200ms đã làm; lọc rẻ B4 trước MFA; clip còn fallback loại khỏi acoustic/MFA nhưng
   **giữ cho vocoder**; eval ~100 câu thuần Việt tự viết (CER + nghe) — *chưa làm, [KẾ HOẠCH]*.
10. **Vận hành Mac:** acoustic MPS, vocoder MPS khi discriminator lớn; không chạy song song
    acoustic+vocoder trong lịch rev.4; watermark 0.8/0.7; chunk 5k; early stop + cosine LR.
11. **ONNX:** 2 graph (acoustic + vocoder spectrum) + iSTFT numpy là chủ đích; coreml EP thử nghiệm,
    CPU EP nhanh nhất trên Mac; INT8 động là mức khởi điểm.
12. **Một checkpoint = một giọng, một phương ngữ** (đúng thiết kế).
13. **Dữ liệu YouTube của chủ kênh** hợp lệ; mặc định CC BY 4.0; SRT mở cho mọi người.

---

## 13. Hiện trạng thực tế, rủi ro & việc còn lại

### 13.1. Điểm mạnh đã kiểm chứng
- Frontend tiếng Việt rất sâu: parser + audit vét cạn, song ánh chính tả, i/y/gi/qu/uy/OO/A-Ă,
  vị trí dấu thanh, profile QA, OOV opt-in, bất biến English fallback.
- Pipeline đầy đủ có test end-to-end: train → synth → export ONNX → onnxruntime → server.
- Hạ tầng vận hành dài hơi cho Mac (MPS memory, chunked runner, resume, early stop, NaN policy).
- Dữ liệu thật 51,7h có giấy phép mở + quy trình làm sạch/QC lặp lại được (B3/B4/overrides).
- Tài liệu dày, trung thực về giới hạn (MODEL_FLOW §9, PROJECT_STATUS §6, REVISION_PLAN §7).

### 13.2. Rủi ro chính (theo mức độ)
1. **Alignment chưa có thật → duration nhãn chia đều là nguyên nhân overfit đã quan sát**
   (val 274,0 @8k → 305,5 @20,8k trong khi train giảm). Mọi đánh giá chất lượng acoustic hiện tại
   **không có nghĩa**. Chốt chặn: smoke small ~20k với alignment thật rồi mới lên base.
2. **Vocoder mới chưa chạy xong** (đang khởi động 100k bước trên MPS); chưa có số CER copy-synthesis.
3. **SIL + log-F0 DCT + cắt segment theo SIL chưa được code** — bảng A3/vocab ~70 trong tài liệu
   là đặc tả, không phải as-built. Nếu smoke small chạy trước khi có SIL/DCT, kết quả sẽ khó so
   với ngưỡng nghiệm thu đã đặt.
4. **INT8 chưa calibration đầy đủ**; chất lượng int8/vocoder int8 chưa đo.
5. **Rủi ro doc↔code** (§15) — dễ khiến review dựa trên thông tin sai.
6. **Dữ liệu đa dạng hạn chế**: một kênh công nghệ, mật độ tiếng Anh cao, 51,7h→ sau lọc còn
   ~33–35h cho acoustic; giọng/ngữ điệu một người nói.
7. **Test collection qua `pytest` trần bị lỗi** → CI/bài mới dễ đỏ oan (§15.1).

### 13.3. Lịch chạy đã chốt (REVISION_PLAN_V2 rev.5 §9.3) — "máy không chờ người"
1. Recut B3 ✅ → mel stats 2.000 clip ✅ → cache feature (đang chạy) → **vocoder Vocos ch=16 100k
   (MPS, train.jsonl 44,96h)**.
2. Trong lúc vocoder chạy: điền overrides → reparse text → lọc rẻ B4 → dựng lại corpus MFA → MFA
   (CPU) ∥ vocoder (MPS).
3. QC MFA (loại 10% thấp nhất + vi phạm cứng) → SIL + log-F0 DCT + cắt segment → đo s/bước cho
   4/4, 4/5, 4/6 (chọn ≤ ~1,0 s) — **không tranh MPS với vocoder**.
4. Smoke small ~20k (alignment thật) → đạt ngưỡng → base ~100k + early stop audio-level → GTA
   fine-tune vocoder.
- **Ngưỡng nghiệm thu** (đã chốt): vocoder copy-synthesis sạch, CER chênh ≤1 điểm; MFA giữ >90%
  clip, lệch Whisper median <50ms; smoke small tốt hơn rõ best 8k trên bộ 100 câu; base có CER
  ≤ copy-synthesis +2–3 điểm.

### 13.4. Việc cần người dùng làm
- Nghe 400 clip mẫu trong `data/codelung/review/`, điền `filled.txt` (`từ|cách đọc`, ưu tiên theo
  tần suất; 150–200 từ đầu cứu phần lớn số giờ) → `python tools/fallback_review.py --merge ...`.
- Viết ~100 câu eval thuần Việt (đủ 6 thanh + coda tắc) làm bộ cố định đo CER.

---

## 14. Gợi ý trọng tâm cho AI review (danh sách câu hỏi mở)

### A. Acoustic & alignment
1. Với ngưỡng hiện tại, cấu hình smoke small 20k + SIL + log-F0 DCT có đủ để phát hiện lỗi hội tụ
   trước khi dồn máy cho base 100k không? Nên thêm chỉ số nào (ví dụ duration MAE theo âm tiết,
   tỉ lệ silence/ngắt nghỉ) ngoài val loss + CER?
2. Chia frame theo `durations` (mean f0/vuv/energy trong đoạn) có làm mất contour thật trong âm
   tiết dài không? Nếu có, SIL/frame-level targets nên đi tới đâu?
3. `round(exp(log_d)−1).clamp_min(1)` + `speed` chia duration có gây lệch pha với mel khi speed
   cực đoan (0.5/2.0) không? Có nên thay bằng co giãn mel (như đường ONNX) để nhất quán?

### B. Dữ liệu
4. Chiến lược "overrides top 400 + loại clip còn fallback khỏi acoustic" có làm lệch phân bố
   (mất từ vay/tiếng Anh) và ảnh hưởng generalization? Cần đo gì?
5. VAL chia theo video (1.791 mẫu ~6,7h) nhưng vẫn là cùng kênh/1 người → val loss có phản ánh
   thật? Bộ 100 câu tự viết nên thiết kế thế nào để đo đúng (đủ thanh/coda, độ dài, từ vay)?
6. B4 dùng ngưỡng cố định (speech <0.5, clipping >1%, RMS <−45 dBFS): có chỗ nào nên theo phân vị
   thay vì ngưỡng tuyệt đối không?

### C. Vocoder & GAN
7. Sau khi mel được chuẩn hoá toàn cục, `mel_weight=5.0` đã hợp lý chưa, hay nên cân theo std?
   Rủi ro loss mel lấn át adversarial/STFT?
8. MPD periods gồm 2/3/5/7/11 + MRD + crop 64 trên MPS: cấu hình này có nguy cơ overfit discriminator
   với 44,96h? Cần thêm regularizer/metric nào?
9. Generator Vocos 12,07M vs lite 3,53M: có nên lite + train lâu hơn cho bước đầu để giảm độ phức
   tạp vận hành (MPS memory/time) không? Tiêu chí quyết định?

### D. ONNX/deploy
10. Lộ trình INT8 calibration đầy đủ (6 thanh × coda tắc) nên đo chất lượng bằng gì (PESQ?
    CER-after-ASR? nghe)? Có cần phân tầng precision theo submodule không (F0/vocoder giữ fp16)?
11. Gộp 2 graph ONNX bằng iSTFT viết tay (IDFT ma trận) có đáng làm không, hay giữ numpy iSTFT
    là điểm dừng hợp lý?

### E. Code/tài liệu (những chỗ đã tìm thấy — §15)
12. Xác nhận và đề xuất thứ tự sửa cho các lệch doc↔code ở §15; có chỗ nào ảnh hưởng *quyết định*
    chứ không chỉ chữ nghĩa không?

---

## 15. Phát hiện của đợt audit này: lệch doc ↔ code & vấn đề nhỏ

### 15.1. Test collection với `pytest` trần (nên sửa)
`pytest` (như README hướng dẫn) fail collection 4 module vì `import tools.*` không có `tools` trên
`sys.path`: `test_fallback_review.py`, `test_clip_quality.py`, `test_mfa_corpus.py`,
`test_reparse_overrides.py`. `python -m pytest` chạy đủ 2257.
**Gợi ý:** thêm `pyproject.toml → [tool.pytest.ini_options] pythonpath = ["src", "."]`, hoặc
`tests/conftest.py` chèn repo root vào `sys.path`, hoặc chuyển `tools/` thành package có
`__init__.py`. Cần chốt vì CI/dokumen đều ghi `pytest`.

### 15.2. SIL token & log-F0 DCT "hồn nhiên" trong tài liệu kế hoạch
`ARCH_DISCUSSION_M6.md` liệt A3 SIL (vị trí cố định, 7 slot, tone NONE, vocab 70) và log-F0 DCT
trong nhóm "đã sửa/đã có từ các vòng trước" — nhưng **code không có** (không có `SIL`/`DCT` trong
`src/`; vocab vẫn 68). `REVISION_PLAN_V2.md` §6 xếp SIL/DCT vào bước *sau* MFA, nên hiểu đúng là
**đặc tả đã chốt, code chưa làm**. Đề xuất: sửa lại câu chữ ở ARCH_DISCUSSION_M6 để tránh review nhầm.

### 15.3. AMP: TRAINING.md nói "CUDA và MPS", code chỉ bật CUDA
`docs/TRAINING.md` dòng ~19: "grad clip, AMP (CUDA **và MPS**)" — sai với `train.py`:
`use_amp = mixed_precision and device.type == "cuda"` (kèm comment đo thật MPS fp16 không lợi).
MODEL_FLOW viết đúng ("AMP tự bật khi CUDA"). Nên sửa TRAINING.md.

### 15.4. Số tham số Vocos lệch giữa các mục trong cùng tài liệu
`REVISION_PLAN_V2.md` §1 và §2 còn ghi "Vocos ... 14,4M", trong khi §9 (rev.5, sau audit expansion
4×→3×) ghi **12.069.762**. Config `small_codelung.yaml` đã ghi 12,07M. Nên đồng bộ §1/§2.

### 15.5. README còn mô tả vocoder 3,5M là "vocoder" chính
`README.md` §2 ghi "Acoustic 16,4 triệu; vocoder 3,5 triệu" — đúng cho bản lite, nhưng run thật
hiện dùng Vocos 12,07M + MPD/MRD 7,7M. Nên ghi rõ "vocoder lite 3,5M (tham chiếu) / Vocos 12,07M
(run thật)".

### 15.6. ARCHITECTURE.md còn roadmap "⬜ ONNX vocoder"
`docs/ARCHITECTURE.md` §Lộ trình item 5 ghi "⬜ ONNX vocoder" nhưng `export-vocoder-onnx` +
`runtime/onnx_engine.py` đã as-built từ trước (và tests/test_onnx_runtime.py xanh). Chỉ là chữ
nghĩa lỗi thời.

### 15.7. `PROJECT_STATUS.md` ghi "working tree không còn thay đổi" — nay không còn đúng
Repo đang commit liên tục + có WIP (rev.6 §1a). Nên ghi kèm hash hoặc bỏ câu khẳng định đó.

### 15.8. Lint đỏ 1 lỗi trong WIP (xem §2.4)
`ruff check .` → `F841 Local variable last_clean_loss is assigned to but never used`
(`src/vifasttts/train.py:548`). Trước WIP này ruff sạch.

---

## 16. Bản đồ tài liệu trong `docs/` (nên đọc gì thêm)

| Nhóm | Tài liệu | Nội dung |
| --- | --- | --- |
| Trạng thái | `PROJECT_STATUS.md` | Trạng thái tổng + nhật ký quyết định + roadmap ✅/⬜ (cập nhật 12/10/2026) |
| Luồng | `MODEL_FLOW.md` | Data → train → inference → export, shape, loss, số tham số |
| | `ARCHITECTURE.md` | Sơ đồ as-built ngắn + vì sao 5 thành phần |
| Frontend | `FRONTEND.md`, `SYLLABLE_GUIDE.md`, `GLOSSARY.md` | Parser/profile/segments/G2P; hướng dẫn âm tiết cho người không chuyên; thuật ngữ |
| | `NORMALIZER.md`, `STEM_NORMALIZER.md`, `G2P_VIETNAMESE.md` | Số/ngày/giờ theo vùng; STEM; khảo sát G2P + bảng IPA |
| Dữ liệu/train | `DATA_PIPELINE.md`, `YOUTUBE_DATASET.md`, `TRAINING.md` | 15 bước dữ liệu; youtube-fetch; train + cảnh báo alignment |
| | `TRAINING_ESTIMATES.md`, `TRAINING_RUN_CODELUNG.md`, `APPLE_SILICON.md` | Ước lượng data/giờ/máy/chi phí; run thật @codelung; khảo sát Mac/MLX/ANE |
| Kế hoạch | `REVISION_PLAN_V2.md` (rev.5), `ARCH_DISCUSSION_M6.md` | Kế hoạch sửa nền tảng A–F + lịch chạy; hồ sơ hỏi–đáp chuyên gia (lịch sử) |
| Nghiên cứu | `SYLLABLE_SYSTEMS_RESEARCH.md`, `TTS_ARCHITECTURE_RESEARCH.md` | So sánh mô hình âm tiết + audit; khảo sát họ kiến trúc TTS + đề xuất v2 |
| Khác | `DEPLOYMENT.md`, `RUST_ROADMAP.md`, `LICENSES.md` | ONNX/quantize; kế hoạch port Rust; giấy phép |
| Lịch sử | `DESIGN_V1_MIEN_BAC.md`, `DESIGN_V2.md`, `FULL_DESIGN.md` | Thiết kế gốc (tham khảo) |

---

## 17. Phụ lục: lệnh tái lập nhanh (cho agent cầu nối xác minh)

```bash
# Chất lượng
python -m pytest -q                 # 2257 passed (~18s) — dùng -m, không dùng pytest trần
ruff check .                        # hiện đỏ 1 lỗi F841 trong WIP (train.py:548)
mypy src/vifasttts                  # sạch

# Frontend
vifasttts parse "Hôm nay trời đẹp."
vifasttts parse-syllable "quốc"
vifasttts g2p "quốc" --dialect south
python tools/parser_audit.py --wordlist <file>

# Dữ liệu + train (trên data/codelung)
python tools/dataset_report.py --dataset data/codelung
python tools/split_manifest.py --manifest data/codelung/manifest.jsonl --out-dir data/codelung --val-videos 4
vifasttts cache-features --manifest data/codelung/train.jsonl --config configs/small_codelung.yaml \
  --cache-dir cache/codelung --workers 8
python tools/train_chunked.py --config configs/small_codelung.yaml \
  --train-manifest data/codelung/train.jsonl --val-manifest data/codelung/val.jsonl \
  --output runs/small_codelung --total-steps 60000 --chunk 5000
python tools/train_vocoder_chunked.py --config configs/small_codelung.yaml \
  --manifest data/codelung/train.jsonl --output runs/vocoder_codelung_v2 --total-steps 100000 --chunk 5000

# Export + ONNX runtime + server
vifasttts export-onnx --config configs/small_codelung.yaml --checkpoint runs/small_codelung/best.pt \
  --output artifacts/model.onnx --precision fp16
vifasttts export-vocoder-onnx --config configs/small_codelung.yaml \
  --checkpoint runs/vocoder_codelung_v2/last.pt --output artifacts/vocoder.onnx
vifasttts synthesize-onnx --acoustic artifacts/model.onnx --vocoder artifacts/vocoder.onnx \
  --text "Xin chào" --output out.wav --peak-normalize --provider cpu
vifasttts serve --config configs/small_codelung.yaml --checkpoint runs/small_codelung/best.pt \
  --vocoder-checkpoint runs/vocoder_codelung_v2/last.pt --port 8000
```

```python
# API Python: parse + segments + synthesize streaming
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.segments import expand_text, distribute_frames
from vifasttts.infer import Synthesizer

items = parse_text("Khuya nay, trời trở gió.", dialect="north")
segs = expand_text(items)                    # view cấp âm vị + trọng số duration
engine = Synthesizer(config, "runs/.../best.pt", "runs/.../last.pt", device="cpu")
for wav, sr in engine.iter_synthesize("Xin chào, đây là streaming."):
    ...                                       # phát dần từng đoạn
```

---

*Hết hồ sơ. Nếu cần xác minh bất kỳ mục nào, hãy hỏi theo định dạng §0.4 — agent cầu nối sẽ trả
câu trả lời kèm trích dẫn file:dòng trong repository.*
