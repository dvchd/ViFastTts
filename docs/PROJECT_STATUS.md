# Trạng thái dự án (PROJECT_STATUS)

> **Tài liệu "compact"** — tóm tắt phiên làm việc, trạng thái as-built, nhật ký
> quyết định, kết quả audit và việc còn lại. Cập nhật: 12/10/2026.
> Nguồn sự thật as-built: `docs/ARCHITECTURE.md`; thuật ngữ: `docs/GLOSSARY.md`.

## 1. TL;DR

- ViFastTts = frontend tiếng Việt + model TTS non-autoregressive một giọng +
  pipeline train/synthesize/export ONNX.
- **2257 test pass**; `ruff` + `mypy` sạch; working tree không còn thay đổi.
- **Kết luận chuyên gia (rev.4)** đã chốt toàn bộ A–F và lịch chạy — xem `docs/REVISION_PLAN_V2.md`.
  Đã code xong: mel log-magnitude + chuẩn hoá toàn cục, F0 pyworld, vocoder **MPD đầy đủ + MRD +
  MR-STFT + generator Vocos** (đo 0,659 s/bước MPS), **postnet** (small → 7,38M), VAD nới biên clip (B3),
  công cụ rà soát fallback (B1), watermark MPS 0.8/0.7 + grad clip + NaN/Inf tự resume.
- **Run `small` @codelung dừng ở 20.850 bước** (val tăng ngược từ 8k do nhãn chia đều) — không dùng
  làm model cuối, chỉ giữ để test pipeline; kế hoạch mới ở `docs/REVISION_PLAN_V2.md` (rev.4).
- Audit parser: **ma trận 49.166 tổ hợp hợp lệ — 0 vấn đề**; từ điển 39k từ →
  6.633 âm tiết, **6.617 parse OK**, 16 fallback đều là typo/từ vay (đã phân loại).
- Audit mở rộng (11/10/2026): Viet74K 73.901 mục — chữ HOA/NFD/dấu kép/gạch nối
  **0 lỗi cấu trúc**; **0 rime ngoài bảng**; đo được mức "nhận rộng" (180/264 rime,
  2.663/6.336 shape có trong ngữ liệu) — chi tiết: `docs/SYLLABLE_SYSTEMS_RESEARCH.md`.
  `tools/parser_audit.py --wordlist` nay in luôn coverage (shape/rime/hapax).
- Export ONNX chạy với batch/số âm tiết/độ dài frame động (kiểm chứng onnxruntime);
  thêm `--precision fp16|int8` (đã kiểm bằng onnxruntime).
- **Runtime ONNX không cần PyTorch**: `synthesize-onnx` (acoustic + vocoder ONNX,
  iSTFT numpy), chia câu dài, `--overrides`, `--peak-normalize`; vocoder ONNX xuất
  spectrum + metadata.
- **Dịch vụ & batch**: `vifasttts serve` (HTTP stdlib: `POST /synthesize`, `/health`,
  `/info`), `synthesize-batch` (file → nhiều WAV), streaming Python
  `iter_synthesize`/`iter_synthesize_onnx`; library dùng `warnings` thay `print`.
- **Thiết bị**: `training.device` `auto|cpu|cuda|mps`; trên Mac Apple Silicon tự dùng
  MPS cho acoustic, CPU cho vocoder (đo thật: 0,31 vs 0,53 s/bước acoustic;
  vocoder CPU nhanh hơn MPS 2,4×) — xem `docs/TRAINING_ESTIMATES.md` §3.1.
- **Apple Silicon / ANE**: khảo sát + đo thật (MLX ~1,2× train; CoreML EP; ANE chưa
  dùng được từ PyTorch) và quyết định "không fork luồng train, chỉ đổi provider ONNX":
  `docs/APPLE_SILICON.md`.
- Đã triển khai hết lớp frontend/data của roadmap: profile `open/strict/corpus` +
  lexicon, `AbstractSyllable.rime`, Segment Expander + `distribute_frames`, benchmark
  biểu diễn, cache feature, kênh phonation, G2P Nam cho `OO`/`ưu/ươu`.
- **Dữ liệu YouTube**: `vifasttts youtube-fetch` lấy audio + phụ đề (json3) từ kênh
  của bạn, làm sạch caption cuộn, cắt clip 1–20 s có QC (im lặng/trùng/fallback),
  xuất manifest + SRT mở + dataset card (mặc định CC BY 4.0) — `docs/YOUTUBE_DATASET.md`.

## 2. Kiến trúc chốt

Ba tầng cho mỗi âm tiết (chi tiết: `docs/GLOSSARY.md`):

```text
dạng chính tả (Orthography) → dạng âm vị học (AbstractSyllable: 5 thành phần)
    → dạng ngữ âm (IPA, frontend/g2p.py, một chiều)
```

- Model **chỉ nhận 7 ID** (5 thành phần + boundary + punctuation); chữ viết và IPA
  không đi vào model.
- **Song ánh** nằm ở cặp *chữ canonical ↔ (tuple + chính tả dư)* = `ParsedSyllable`.
  Tuple thuần là toàn ánh nhưng không đơn ánh (đồng âm/biến thể tự do).
- Inventory: **24 onset / 15 nucleus / 11 coda / 6 tone** — phủ đủ tiếng Việt chuẩn
  (đối chiếu Kirby 2011; xác minh trên từ điển 39k từ).
- Các **phép chiếu** từ mô hình 5 thành phần (đều tất định, không mất thông tin gốc):
  `rime` (đệm+chính+cuối) cho view "onset+vần+thanh"; `segments.py` mở ra chuỗi
  segment cấp âm vị + trọng số duration; `g2p.py` chiếu IPA. Lớp **profile**
  (`open/strict/corpus` + lexicon) kiểm chính tả/ngữ âm mà không chặn dữ liệu.

## 3. Nhật ký quyết định chính

- `gi` dùng chung chữ i: `gì`/`gìn` = GI + I, `giếng` = GI + IÊ (cùng vần `tiếng`);
  `già`/`giữ` giữ digraph câm.
- `uy` = W + I khác `ui` = U + J; dấu hỏi của `ui` rơi trên u (`tủi`).
- `uya` (khuya) = W + IÊ; `oo` (xoong, moóc) tách thành **nucleus `OO` riêng** (ɔː)
  theo phản hồi người bản ngữ (`xoong` ≠ `xong`).
- `ay/au` = **Ă ngắn [a]**, `ai/ao` = **A dài [aː]** (Help:IPA/Vietnamese);
  `tai ≠ tay`, `rao ≠ rau`, `quai ≠ quay`.
- `q` trần không hợp lệ (`qa/qi` fallback); sau `qu` chỉ hợp lệ với
  `a ă â e ê i/y ô ơ` (`quốc` = K+W+Ô+C).
- Tokenizer giữ chữ ngoài Latin (CJK, Cyrillic...) làm fallback, không nuốt mất.
- Biến thể i/y tự do (`qui/quy`, `mĩ/mỹ`) **giữ nguyên** như người viết; vị trí dấu
  thanh chuẩn hoá theo lối hiện đại, **cài đặt được** qua
  `NormalizerOptions.canonical_tone_spelling` — chỉ ảnh hưởng tầng chữ, model không đổi.
- G2P IPA: broad/narrow; Bắc/Nam chính xác, Trung xấp xỉ; thanh Chao/số/none;
  tham chiếu Kirby 2011 / Thompson 1965 / Huỳnh Công 2003 (xem `docs/G2P_VIETNAMESE.md`).
- OOV resolver opt-in (`resolve_oov=True`) theo quy trình 5 bước viG2P; English
  fallback có bất biến "mọi reading phải parse được".
- Dữ liệu YouTube: dùng **yt-dlp + imageio-ffmpeg** qua subprocess (extra `[youtube]`),
  phụ đề **json3** làm timing (không dùng SRT chưa căn), QC im lặng/trùng/fallback/thưa,
  resume bằng `.done.json`; dataset card + giấy phép mặc định **CC BY 4.0** (kênh
  tham chiếu `@codelung` — audio mở, ai dùng cũng được).
- Làm sạch dữ liệu (kênh `@codelung`): bảng fallback frontend ~670 từ công nghệ + 31 acronym
  + 183 overrides kênh, `--respell-codes` (tách chữ-số, `MP3→em pê ba`, `1.13.1→một mười ba một`),
  cờ `--recut` (làm lại không tải); suy từ ASR sai bằng ngữ cảnh (`vouer→voucher`, `redmi→README`,
  `clock→Claude`, `queen→Qwen`, `kch→cache`, `hip→heap`...) và sửa cách đọc sai (compose, sonnet,
  godot, apache, kubernetes); **kết quả thật: 31 video → 13.225 clip · 51,07 giờ ·
  fallback 1,22% âm tiết (từ 7,1%) · 71% clip 0 fallback · đủ 6 thanh/11 coda ·
  0 vấn đề manifest**; `train.jsonl`/`val.jsonl` chia theo video (44,38h/6,68h).

## 4. Kết quả audit (bằng chứng)

| Audit | Kết quả |
| --- | --- |
| Ma trận (âm đầu × vần × âm cuối × thanh), `tools/parser_audit.py` | 49.166 tổ hợp, **0 vấn đề**; gồm bất biến bảng compatibility |
| Wordlist Viet39K (duyet/vietnamese-wordlist, GPL-2.0 — chỉ phân tích) | 6.633 âm tiết; **6.617 OK**; 16 fallback = typo (`vòmg`, `quuyết`, `sưót`), từ vay viết liền (`palăng`, `gram`, `tout`, `têt`, `gip`, `basoi`, `xit`), chữ biểu cảm phi chuẩn, chữ cái đơn |
| Va chạm tuple trên wordlist | 50 nhóm: biến thể vị trí dấu cũ/mới, i/y tự do, đồng âm thật hiếm (`gen/ghen`, `ca/ka`, `diệp/dịệp`) |
| Inventory thực dùng | đủ 24/15/11/6 (gồm `OO`, `Ă` — xác nhận các bổ sung là cần thiết) |
| Export ONNX | batch 1–3, N 3–12, tổng frame động — khớp PyTorch |

## 5. Cheatsheet lệnh

```bash
# frontend
vifasttts parse "Hôm nay trời đẹp."
vifasttts parse-syllable "giếng"
vifasttts check-profile "voan bình" --profile strict
vifasttts g2p "quốc" --dialect south  # IPA từng âm tiết (phân tích)
vifasttts --version
python examples/g2p_demo.py            # IPA Bắc/Nam, broad/narrow, 6 thanh
python tools/frontend_benchmark.py --text "Chào bạn"   # char/slot/phoneme/rime

# dữ liệu
vifasttts build-manifest --metadata data/metadata.csv --wav-dir data/wavs --output data/manifest.jsonl
vifasttts validate-manifest --manifest data/manifest.jsonl --wav-dir data/wavs   # kèm coverage tone/coda
vifasttts build-lexicon --manifest data/manifest.jsonl --output data/lexicon.json
vifasttts cache-features --manifest data/train.jsonl --config configs/base_24k.yaml --cache-dir cache/train
vifasttts youtube-fetch "https://www.youtube.com/@codelung/videos" --out data/codelung --demo       # demo 1 video
vifasttts youtube-fetch "https://www.youtube.com/@codelung/videos" --out data/codelung --sleep 1.5  # cả kênh
python tools/dataset_report.py --dataset data/codelung                                # báo cáo per-video
python tools/split_manifest.py --manifest data/codelung/manifest.jsonl \
  --out-dir data/codelung --val-videos 4                                              # chia train/val theo video

# huấn luyện / suy luận / export
vifasttts train --config configs/base_24k.yaml --train-manifest ... --val-manifest ... --output runs/base
vifasttts train-vocoder --config configs/base_24k.yaml --manifest data/train.jsonl --output runs/vocoder
vifasttts synthesize --config configs/base_24k.yaml --checkpoint runs/base/best.pt --text "Xin chào" --output out.wav
vifasttts synthesize-batch --config configs/base_24k.yaml --checkpoint runs/base/best.pt --input data/cau.txt --output-dir out/
vifasttts serve --config configs/base_24k.yaml --checkpoint runs/base/best.pt --port 8000
vifasttts export-onnx --config configs/base_24k.yaml --checkpoint runs/base/best.pt --output model.onnx --precision fp16
vifasttts export-vocoder-onnx --config configs/base_24k.yaml --checkpoint runs/vocoder/last.pt --output vocoder.onnx

# runtime ONNX (không cần PyTorch)
vifasttts synthesize-onnx --acoustic model.onnx --vocoder vocoder.onnx --text "Xin chào" --output onnx.wav

# kiểm tra
pytest && ruff check . && mypy src/vifasttts
python tools/parser_audit.py                        # ma trận ~49k
python tools/parser_audit.py --wordlist words.txt   # audit từ điển (+ --profile strict|corpus)
```

## 6. Việc còn lại (roadmap)

Đã xong trong phiên này (✅) và phần còn lại (⬜):

1. ✅ **Segment Expander** (`frontend/segments.py`): mở `IÊ/UÔ/ƯƠ` thành hai đích,
   trọng số duration theo role, `distribute_frames`. ⬜ Aligner thật (MFA/forward-sum)
   và học duration từ alignment — hiện vẫn placeholder chia theo trọng số/frame.
2. ✅ **Phonation** extraction (`data/phonation.py`: HNR, spectral tilt, glottalization,
   bật bằng `phonation.enabled`) + **conditioning**: `model.use_phonation: true` cho
   model dự đoán và điều kiện hoá 3 kênh (loss `phonation`). ⬜ Vocoder-conditioned
   phonation và đánh giá chất giọng nghe được.
3. ✅ **Vocoder tham chiếu**: `train-vocoder` (iSTFTNetLite + MPD-lite: mel L1 + feature
   matching + adversarial), checkpoint khóa `vocoder` cho `synthesize`; test smoke +
   CLI. ⬜ Discriminator lớn hơn, spectral loss, A/B HiFi-GAN — chất lượng production.
4. ✅ **Quantization** khởi điểm: `--precision fp16|int8` + test onnxruntime.
   ✅ **Runtime ONNX** không cần PyTorch (`runtime/`: acoustic + vocoder spectrum +
   iSTFT numpy + postprocess; CLI `synthesize-onnx`, chia câu, `--overrides`).
   ✅ **Streaming/batch/server**: `iter_synthesize(_onnx)`, `synthesize-batch`,
   `vifasttts serve` (HTTP stdlib).
   ⬜ Calibration INT8 đầy đủ (6 thanh × coda tắc) — cần bộ hiệu chỉnh + đo chất lượng;
   ⬜ gộp vocoder vào một graph ONNX (ONNX thiếu op ISTFT — xem `docs/DEPLOYMENT.md`).
5. ✅ **Strict/lexicon**: 3 mức `open/strict/corpus`, `Lexicon` (from_words/from_manifest,
   save/load), CLI `check-profile`/`build-lexicon`, tool audit `--profile`. ⬜ Mở rộng
   OOV resolver cho từ vay viết liền (`palăng`, `a-xít`...).
6. ✅ **G2P phương ngữ**: Nam gộp `OO→ɔ`; Bắc narrow `ưu/ươu → [iw]/[iəw]`.
   ⬜ Bảng thanh Huế (WikiPron hue ~30.242 mục / MFA Huế).
7. ✅ **Cc narrow `ưu/ươu → [iw]/[iəw]`.
   ⬜ Bảng thanh Huế (WikiPron hue ~30.242 mục / MFA Huế).
7. ✅ **Cache feature/duration** (`data/cache.py`, CLI `cache-features`,
   `training.cache_features`) khoá theo hash(audio, config).
8. ✅ **Benchmark biểu diễn** (`frontend/benchmark.py`, `tools/frontend_benchmark.py`):
   char 3.3 / slot 5.0 / phoneme 4.0 / rime 2.0 đơn vị mỗi âm tiết (đo mẫu).
   ⬜ Thí nghiệm A/B huấn luyện thật (char/BPE vs phoneme vs vần vs 5 slot) — research gap.
9. ✅ **Cache feature tầng phonation**: cache đã lưu đủ mel/energy/f0/vuv/phonation
   (khoá hash tính cả cấu hình phonation).
10. 📄 **Tài liệu luồng model as-built**: `docs/MODEL_FLOW.md` (data → train → infer → export,
   tensor shape, loss, số tham số, khác biệt train/infer) và `docs/TRAINING_ESTIMATES.md`
   (ước lượng dữ liệu/thời gian/máy/chi phí + công thức tự hiệu chỉnh).
11. ✅ **Dataset YouTube tự động** (`data/subtitles.py` + `data/youtube.py`, CLI
    `youtube-fetch`): phụ đề json3 làm timing, cắt clip + QC (im lặng/trùng/fallback/thưa),
    báo cáo OOV + `--overrides` (mẫu `configs/overrides_tech_vi.json`), manifest/SRT/dataset
    card mở (CC BY 4.0), resume `.done.json`, test offline bằng yt-dlp giả. ⬜ Căn word-level
    (whisper/MFA) để có `durations` cấp âm tiết; ⬜ publish Hugging Face; ⬜ VAD/speaker
    tách nhạc – nhiều người nói.

- Batch size (dự đoán, ghi ở `docs/TRAINING_ESTIMATES.md` §3.2): giữ **16** cho `small` (đã kiểm
  chứng, đỉnh MPS 25GB); 20/24 chỉ lợi ~5–9% thời gian và 24 sát ngưỡng bộ nhớ — cần A/B thật.
  `base` dự đoán cần batch 6–8 và ~30–42h acoustic; `large` batch 4 và ~52–72h.
- **Dừng run `small` ở 20.850 bước** theo phản hồi chuyên gia: val tăng ngược từ 8k (274,0 → 305,5)
  trong khi train giảm ⇒ overfit vào **nhãn duration chia đều**, không phải thiếu dung lượng.
  Giữ `best.pt` (8k) để test pipeline. Chuyển sang **`docs/REVISION_PLAN_V2.md`**: (A1) sửa mel loss
  mean/bin ✅; (A2) f0 — đính chính *variance adaptor đã có*, phần thiếu là **contour DCT + log-F0 +
  extractor pyworld**; (A3) thêm **token SIL**; đổi **mel log-magnitude + chuẩn hoá toàn cục**;
  **MFA train bằng lexicon sinh từ parser** + QC/lọc score; chỉ lên `base` (enc4/dec6) sau alignment thật.

## 7. Commit tiêu biểu trong phiên

```text
36586c3 Chốt: quy ước chữ (tùy/tuỳ) cài đặt được, nhưng không ảnh hưởng model
21dd79f Chuẩn hoá uy đồng bộ với oa/oe + test dấu thanh chỉ nằm trên nguyên âm
6f651df Audit parser trên từ điển 39K: đầy đủ cho tiếng Việt chuẩn
5b29e04 Tách nucleus OO riêng + khoá tính khả nghịch chính tả (song ánh)
3f3a1a5 Parser: siết "q" trần, giữ chữ ngoài Latin; thêm audit vét cạn + corpus edge
d3f2558 Docs: khảo sát G2P/TTS, glossary thuật ngữ, hướng dẫn cấu trúc âm tiết
314bc40 Frontend + neural: nâng cấp toàn diện (534 test pass)
```
