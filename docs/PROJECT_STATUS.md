# Trạng thái dự án (PROJECT_STATUS)

> **Tài liệu "compact"** — tóm tắt phiên làm việc, trạng thái as-built, nhật ký
> quyết định, kết quả audit và việc còn lại. Cập nhật: 11/10/2026.
> Nguồn sự thật as-built: `docs/ARCHITECTURE.md`; thuật ngữ: `docs/GLOSSARY.md`.

## 1. TL;DR

- ViFastTts = frontend tiếng Việt + model TTS non-autoregressive một giọng +
  pipeline train/synthesize/export ONNX.
- **834 test pass**; `ruff` + `mypy` sạch; working tree không còn thay đổi.
- Audit parser: **ma trận 49.166 tổ hợp hợp lệ — 0 vấn đề**; từ điển 39k từ →
  6.633 âm tiết, **6.617 parse OK**, 16 fallback đều là typo/từ vay (đã phân loại).
- Export ONNX chạy với batch/số âm tiết/độ dài frame động (kiểm chứng onnxruntime).

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
python examples/g2p_demo.py            # IPA Bắc/Nam, broad/narrow, 6 thanh

# dữ liệu
vifasttts build-manifest --metadata data/metadata.csv --wav-dir data/wavs --output data/manifest.jsonl
vifasttts validate-manifest --manifest data/manifest.jsonl --wav-dir data/wavs   # kèm coverage tone/coda

# huấn luyện / suy luận / export
vifasttts train --config configs/base_24k.yaml --train-manifest ... --val-manifest ... --output runs/base
vifasttts synthesize --config configs/base_24k.yaml --checkpoint runs/base/best.pt --text "Xin chào" --output out.wav
vifasttts export-onnx --config configs/base_24k.yaml --checkpoint runs/base/best.pt --output model.onnx

# kiểm tra
pytest && ruff check . && mypy src/vifasttts
python tools/parser_audit.py                        # ma trận ~49k
python tools/parser_audit.py --wordlist words.txt   # audit từ điển
```

## 6. Việc còn lại (roadmap)

1. **Segment Expander** duration cấp role + aligner (MFA/forward-sum) — §9.4 khảo sát.
2. **Phonation** (spectral tilt, HNR, glottalization) — §9.7.
3. **Vocoder production**: discriminator + spectral loss, A/B HiFi-GAN; iSTFT-based
   ONNX (ConvTranspose) cho phần vocoder — §9.9/§9.12.
4. **Quantization** FP16/INT8 từng phần, calibration đủ 6 thanh × coda tắc.
5. **Strict mode / lexicon** cho chính tả không hợp lệ; mở rộng OOV resolver cho
   từ vay viết liền (`palăng`, `a-xít`...).
6. **G2P Trung**: bảng thanh Huế (WikiPron hue / MFA Huế); tùy chọn quy ước i/y.
7. **Cache feature/duration** cho training quy mô lớn (hiện trích tại chỗ).

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
