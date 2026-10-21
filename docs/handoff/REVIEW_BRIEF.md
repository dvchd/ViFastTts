# ViFastTts — Brief tổng hợp để tham vấn (21/10/2026)

> Mục đích: file này tự chứa đủ bối cảnh + số đo + các lần thử để một AI khác đọc và tư vấn hướng
> đi tiếp. Mọi con số đều đo trên máy thật (Mac mini M6), không phải ước lượng — trừ chỗ ghi rõ.

---

## 0. Tóm tắt một đoạn

Xây TTS tiếng Việt **1 giọng** (giọng chủ dự án), **tất định**, chạy **ONNX CPU**, kiểm soát pause
tuyệt đối. Kiến trúc non-AR: text → âm tiết → mel 100 bin @ hop 256 (10,67 ms) → vocoder Vocos.
Chất lượng đã dùng được, **ràng buộc lớn nhất còn lại là "rè"** — một lớp **nhiễu bất tuần hoàn
2–6 kHz** phủ đều lên giọng. Đã chứng minh: vocoder không phải thủ phạm (copy-synthesis mel thật →
Vocos gần như y bản thật), nên thủ phạm là **mel do model dự đoán** (mờ/thiếu chi tiết). Ngoài ra
có 2 lỗi phát âm: **coda NH đọc thành N** ("xanh→xan") và **mờ onset/dính từ** ở biên âm tiết.

---

## 1. Ràng buộc sản phẩm

- 1 giọng (chủ dự án), tiếng Việt, có thể chêm từ tiếng Anh phiên âm.
- **Tất định**: cùng text → cùng file WAV.
- **ONNX CPU** (không GPU khi chạy thật), 24 kHz.
- Kiểm soát pause: chế độ "đọc đều" — pause cố định theo dấu câu (phẩy 0,25 s · chấm 0,55 s),
  chặn pause do model tự nghĩ.
- Dữ liệu hiện có: **10,7 giờ tiếng** (16,06 h nếu tính cả lặng), 1 người nói, 25 video.

## 2. Kiến trúc hiện tại (`v2_slot5`, small 7,05 M)

```
Văn bản → frontend: âm tiết → 7 mã (onset/medial/nucleus/coda/thanh/ranh giới/dấu câu)
        → embedding → encoder 4 lớp (192, 4 head, ffn 768, conv kernel 15)
        → 3 đầu dự đoán:
            - duration (log_duration, mỗi âm tiết bao nhiêu frame)
            - slot (O/M/N/C, chia lòng âm tiết — `seg_durations [N,4]`)
            - pause (gate + log_gap, 334k tham số)
        → trải slot thành frame → decoder 2 lớp + postnet → **mel 100 bin**
        → **Vocos** (train RIÊNG trên mel thật) → WAV 24 kHz
```

- Mel: `n_fft 1024`, `win 1024`, **hop 256 (10,67 ms)**, 100 bin, 24 kHz.
- Loss: **L1 trên mel** (+ duration MSE, ratio CE, f0 shape, vuv, energy, phonation, tone, pause).
- `v2_slot5` small = **7,05 M**; base = 17,1 M (256/6/4/1024) — **base không thắng small** (mục 9).
- Vocoder Vocos: iSTFT-based, nhẹ, ONNX tốt.

## 3. Dữ liệu

| | |
|---|---|
| Nguồn | 25 video YouTube của chủ dự án (thu qua **OBS noise suppression**) |
| Audio | `audio_v2`: chuẩn hoá âm lượng + high-pass + gate; nền nhiễu trung vị **−80 dB** |
| 2 video ồn | `4kyqfOypVWI` (−52,5 dB) + `lRojc8KDJ6k` (−51,0 dB) — video 1 chiếm **5,1% train** |
| Align | **MFA** (Montreal Forced Aligner), nhãn `hybrid+rule t2.1-hybrid-v1` → durations/gaps/seg4 |
| Segment | cắt tại gap >0,64 s (clause-level), gap nội bộ max 60 frame, đuôi ≤60 frame |
| Token | 220.384 âm tiết · **2.552 kiểu** · <5 lần: 41% kiểu (0,8% token) · ≥150 lần: 11% kiểu (78,4% token) · **98,2% token thuộc kiểu ≥10 lần** |

## 4. Kết quả các bản (cùng val 300 segment của v3.2, cùng thước)

| | b8 (small, 46k) | **v3.2 small (59k)** | v3.2 base (30k best) |
|---|---|---|---|
| mel_l1_z | 1,560 | **1,470** | 1,462 |
| mel dB hữu thanh | 10,34 | 9,97 | **9,69** |
| độ nét `rough` | 0,315 | **0,340** | 0,324 |
| thanh C | 0,729 | **0,813** | 0,785 |
| Δjitter ngã | **0,032** | 0,057 | 0,045 |
| ΔHNR ngã | **−1,00** | −1,22 | −1,14 |
| gap MAE (dự đoán pause) | 73,6 | 26,9 | **25,7** |
| giữ coda mũi (20 clip) | 108/202 | **150/202** | — |
| CER-100 (holdout riêng) | — | 0,3694 (trung vị 0,252) | — |

Ghi chú: `b8` = small train batch 8, 60k, best 46k. `v3.2` = cùng kiến trúc/hyperparam, **chỉ đổi cách
cắt segment** (bỏ pause dài khỏi row) — cải thiện mel/coda/gap rõ nhưng **rè không đổi**.

## 5. Vấn đề 1 — RÈ (ưu tiên cao nhất)

### 5.1 Hiện tượng
Tai người: "rè đều", thỉnh thoảng rõ ở một số từ; không theo thanh cố định.

### 5.2 Số đo (thuyết phục nhất)
| Đo | Thật | Synth | Ghi chú |
|---|---|---|---|
| **Aperiodicity dải 2–6 kHz** (D4C, frame hữu thanh) | 0,525 | **0,567 (+0,042)** | dải tai nhạy với "xì/rè" |
| Aperiodicity 0–2 kHz | 0,0475 | 0,0456 | **không đổi** |
| HNR toàn dải | 5,14 dB | 4,90 | −0,24 |
| Năng lượng 6–12 kHz | 0,0004 | 0,0002 | đỉnh **mềm hơn**, không sáng hơn |
| **Copy-synthesis (mel THẬT → cùng Vocos)** | — | **ΔHNR −0,07 dB · Δjitter −0,30%** | vocoder **không** phải thủ phạm |
| ΔCPP / ΔHNR theo âm tiết (dashboard, 104 âm tiết) | — | −0,47 / −0,59 dB | 0/104 tụt >2 dB |

⇒ Rè = **lớp nhiễu mỏng 2–6 kHz do mel model dự đoán thiếu chi tiết**, không phải vocoder, không
phải mất tuần hoàn toàn cục.

### 5.3 Đã thử & kết quả
| Phương án | Kết quả | Kết luận |
|---|---|---|
| Điều kiện hoá phonation (hnr/jitter/glottalization + head) | tai không đổi | ❌ bác bỏ |
| Làm nét mel post-hoc sh25 (α=0→1,5→2,0) | rough +11%, HNR +0,14, jitter không đổi; tai: chỉ che nhẹ | ⚠️ không chữa |
| **DFN toàn corpus rồi train** | HNR 5,14→5,12 · AP 0,532→0,536; model train trên DFN ra **rè hơn** (AP 0,635 vs 0,561) | ❌ |
| Bỏ video ồn | mel đẹp hơn nhưng creak/CER tệ hơn (CER 0,238 vs 0,213) | ❌ |
| Oversample âm tiết hiếm | HNR +0,71→+1,26 nhưng tai trung tính | ⚠️ phụ |
| **Adversarial (T4.1)** | phân kỳ 3 lần ở probe | ❌ bỏ (chưa thử fine-tune từ checkpoint tốt) |
| **Model to (base 17M)** | val 0,618 vs small 0,597; **rè không giảm** | ❌ dung lượng bão hoà |
| Đổi segment (v3.2) | mel/coda/gap tốt hơn; rè không đổi | ⚠️ |
| GTA vocoder | chưa chạy | ⏳ |

### 5.4 Loại trừ
- **Không phải 24 kHz**: copy-synthesis 24 kHz nghe rõ, gần y bản thật.
- **Không phải vocoder**: copy-synthesis sạch (ΔHNR −0,07 dB).
- **Không phải dữ liệu bẩn/DFN**: nền −80 dB; DFN không đổi phonation; train DFN còn tệ hơn.
- **Không phải dung lượng**: base 17M không giảm rè.
- **Không phải alignment thô**: audit MFA — slot coda chứa murmur (năng lượng mũi +5,5 dB so với
  gap −15,7 dB; flatness 0,055 thấp nhất) ⇒ MFA cắt coda đúng ở mức thô.

## 6. Vấn đề 2 — Coda NH → N ("xanh→xan", "lạnh→lạn", "quanh→quan")

- Người nghe xác nhận trên câu mới; cả b8, v3.2 small, v3.2 base đều mắc.
- Đo cặp tối thiểu (carrier "Đọc ___"):
  | | mũi 250–500 Hz xanh/xan | centroid xanh/xan |
  |---|---|---|
  | b8 | 0,53 / 0,11 | 377 / 701 Hz |
  | v3.2 small | 3,90 / 0,73 | 272 / 489 Hz |
  | v3.2 base | 1,46 / 0,17 | 326 / 606 Hz |
  ⇒ Murmur **có** nhưng **centroid của xanh THẤP hơn xan** (ngược với /ɲ/ cần cộng hưởng cao hơn
  /n/) → tai nghe thành "n".
- Giữ coda mũi tổng: b8 108/202 · v3.2 150/202 (thật 202).
- Đang thử: **v3.4 coda/onset-weighted mel loss** (coda +3,0×, onset +2,0×). @11k: năng lượng mũi
  tổng 188 (sát thật 202) nhưng **N vượt gấp đôi** (189 vs thật 95) và **centroid vẫn sai 2/3 cặp**.
- Câu hỏi mở: weighted loss thô quá? cần **auxiliary coda classifier** (ép mel phân biệt N/NG/NH/M)?
  hay **làm mềm biên ±1 frame** (MFA sai ±10,7 ms = 20–30% độ dài coda 3–6 frame)?

## 7. Vấn đề 3 — Mờ onset / "trộn từ"

- Đo **spectral flux tại biên** âm tiết: thật **0,395** → synth **0,250 (−37%)**; **43/85 biên**
  mờ hơn thật >30%.
- Ví dụ "các|cái": flux thật 0,747 → synth 0,416 (−45%); năng lượng đuôi "các" không tắt hết
  trước biên.
- Cùng gốc với rè (mel mờ ở vùng ngắn).

## 8. Vấn đề 4 — Pause (đã giải quyết phần lớn)

- b8 (học pause từ dữ liệu cũ): pause 0,62–1,10 s, sai chỗ; 64,4% row cũ có gap nội bộ >0,64 s.
- v3.2: segment cắt tại gap >0,64 s ⇒ model tự nghĩ ngắn; chế độ **steady** (runtime) chèn pause cố
  định theo dấu câu. Phát hiện: model v3.2 **tự đệm pause cuối mệnh đề** (0,9–1,3 s khi ghép) →
  đã sửa bằng **cắt lặng thừa hai đầu chunk** (`_trim_edge_silence`), còn đúng 0,29/0,59 s.
- Đầu pause 334k tham số (4,7%); có thể bypass (runtime truyền `gaps` cố định) → không cần học.

## 9. Bài học v3/v3.1 (đã đóng)

- v3 = DFN toàn bộ + **cắt lặng 0,7 s rồi NỐI audio**; v3.1 = bỏ DFN, vẫn nối.
- Kết quả: cả hai trượt y hệt (model mờ, nét 0,19, thanh C 0,44, audio như nhiễu).
- Timeline khớp **0 frame**, seg=dur 40/40 ⇒ **không phải lệch nhãn**; thủ phạm là **sửa audio**
  (mối nối mel + pause dồn về 66 frame). ⇒ Bài học: **không sửa audio**; pause là chuyện runtime.

## 10. So sánh với VieNeu-TTS (tham chiếu ngoài)

- **v3 Turbo**: LLM backbone (~0,5B, cộng đồng nói Qwen2.5-0.5B) + codec MOSS-Audio-Tokenizer-Nano,
  **48 kHz**, 10.000 h song ngữ, 25 giọng preset, clone 3–8 s, RTF CPU ~0,55.
- **v3 Nano**: **48M flow-matching** (ONNX CPU), **24 kHz**, 11 giọng + clone, RTF 0,11–0,22.
  Graph: `text_encoder` → `duration_predictor` → `vector_estimator` (flow) → `codec_decoder`.
  **Latent 64 ms/frame (thô hơn mel mình 6×) mà sạch hơn** ⇒ điểm mấu chốt là **codec decoder train
  cùng flow**, không phải độ phân giải.
- Tai chủ dự án: preset Nano **không rè, rõ, ăn đứt model mình**. Nhưng:
  - **Clone giọng chủ dự án** (ref 8 s từ corpus): Nano **méo/mất giọng**, tệ hơn v3.2; Turbo rõ chữ
    hơn nhưng **giọng chưa giống** + **pause sai** (không kiểm soát được).
  - Lưu ý thiên lệch: bản mình xuất **−26 dBFS**, Nano **−16 dBFS** (to hơn 10 dB) — sau khi cân
    mức, so lại mới công bằng.
- Kết luận tạm: zero-shot clone **không đủ** cho giọng này (cộng đồng cũng báo ~50% similarity);
  muốn giọng đúng cần **LoRA fine-tune** (10–30 phút audio, cần GPU).

## 11. Hướng đã/đang chạy

1. **v3.4** (đang train, ETA ~16:40): small + coda/onset-weighted mel loss — nhắm coda NH/N + biên.
2. **Adversarial fine-tune** từ v3.2 small — nhắm lớp rè 2–6 kHz (chưa chạy).
3. **GTA vocoder** — fine-tune Vocos trên mel do model sinh.
4. **Thí nghiệm synthetic** (chưa chạy): VieNeu Turbo đọc **đúng transcript thật** (cùng text, cùng
   số giờ) → MFA align → train small **chỉ trên synthetic** (không mix) → so rè với bản real.
   Mục đích: xác định **rè do dữ liệu/giọng hay do kiến trúc**. Pilot 40 phút dự kiến ~3 h.
5. (xa) Viết lại pipeline theo hướng flow-matching + decoder train cùng (kiểu Nano) hoặc distill.

---

## 12. CÂU HỎI CHO AI LỚN

> Trả lời giúp theo từng số, ưu tiên cái nào làm trước, và nếu thấy giả định nào sai thì nói thẳng.

**Q1 (rè — quan trọng nhất).** Với chuỗi bằng chứng: vocoder tái tạo mel thật gần như hoàn hảo
(ΔHNR −0,07 dB), nhưng mel **do model dự đoán** làm sinh lớp bất tuần hoàn +0,042 ở 2–6 kHz, và
model to hơn không cải thiện — lựa chọn nào hiệu quả nhất để giảm lớp nhiễu này trong kiến trúc
**mel + Vocos, 7M, 24 kHz, hop 256, L1 loss**?
(a) adversarial fine-tune từ checkpoint tốt (xin công thức ổn định: trọng số adv, loại discriminator,
grad clip, EMA, số bước, tỉ lệ D/G);
(b) fine-tune vocoder GTA trên mel do model sinh;
(c) đổi loss (multi-scale STFT trên mel? perceptual? noise-aware weighting?);
(d) tăng độ phân giải (hop 128 → train lại vocoder + acoustic);
(e) bỏ mel, chuyển sang flow-matching + codec decoder train cùng (kiểu VieNeu Nano).
Cho biết thứ tự và lý do.

**Q2 (coda NH/N).** Coda tiếng Việt chỉ 3–6 frame; đo được murmur **có** nhưng **sai chỗ cộng hưởng**
(centroid xanh < xan). Nên:
(a) tiếp tục weighted loss với trọng số theo lớp coda (N/NG/NH/M) thay vì một boost chung;
(b) thêm **auxiliary classifier** trên mel dự đoán (ép phân biệt coda);
(c) làm mềm biên nhãn ±1 frame (MFA sai ±10,7 ms);
(d) dùng representation khác (ví dụ thêm đặc trưng formant/chuyển tiếp làm input)?
Có kinh nghiệm nào về ngôn ngữ nhiều coda (Việt/Quảng Đông/Hàn) không?

**Q3 (thí nghiệm synthetic).** Thiết kế: cùng transcript, cùng số giờ, teacher = TTS chất lượng cao,
MFA re-align, train **chỉ synthetic** vs **chỉ real**, cùng model/loss. Thiết kế này có chỗ nào sai
không? Có nên thêm nhánh **mix** hoặc **teacher-student distillation** ngay không?

**Q4 (chiến lược).** Với ràng buộc tất định/ONNX CPU/1 giọng, nên:
(a) LoRA fine-tune VieNeu Turbo trên 10,7 h (Apache 2.0) rồi dùng luôn;
(b) giữ pipeline mình + adversarial/GTA;
(c) distill VieNeu → small của mình;
(d) viết lại theo flow-matching/codec?
Cho biết rủi ro của từng hướng và cách đánh giá công bằng.

**Q5 (loss/pause).** Nếu pause do runtime quyết định (cố định theo dấu câu), có nên bỏ đầu pause
(334k) và **mask frame lặng khỏi mel loss**? Lợi ích thực tế (chất lượng/tốc độ) có đáng một lần
retrain không?

**Q6 (dữ liệu).** 10,7 h, 2.552 kiểu âm tiết, 98,2% token thuộc kiểu ≥10 lần. Có nên thêm giờ, hay
chỉ thêm **văn bản đa dạng** (phủ đuôi hiếm)? Ngưỡng giờ hợp lý cho small 7M và base 17M?

**Q7 (thước đo).** Bộ thước hiện tại: mel L1 z/dB, `rough` (std mel theo dải), CPP, HNR (D4C),
jitter, AP 2–6 kHz, coda nasal ratio, spectral flux biên, CER (Whisper). Có thước nào **tương quan
tốt hơn với cảm nhận "rè"** của người nghe không (DNSMOS P.835? NISQA? SRMR?)? Có nên dùng thước
chuẩn P.835 cho các lần so sánh không?

**Q8 (pipeline).** Nhìn vào mô tả pipeline (frontend âm tiết → slot O/M/N/C → mel + Vocos, MFA
hybrid, segment cắt tại gap >0,64 s, cache mel/f0/vuv/energy/hnr/glottalization/jitter, L1 +
duration + ratio + f0 shape + vuv + energy + phonation + tone + pause) — có **lỗi thiết kế rõ
ràng** nào không? Có thành phần nào nên bỏ/gộp?

**Q9 (alignment).** MFA per-phone sai ±10,7 ms; coda 3–6 frame ⇒ 20–30% sai. Có aligner/model nào
tốt hơn cho tiếng Việt (MFA với model riêng? WhisperX? force-align bằng CTC?) hoặc cách **học nhãn
mềm** (soft boundaries) không?

**Q10 (roadmap).** Trong 2–4 tuần, với 1 người + 1 Mac mini M6 (MPS), nên làm gì theo thứ tự để
"rè" giảm rõ nhất mà không phá tính tất định/ONNX?

---

## 13. Phụ lục — vị trí code/số liệu

| Thành phần | File/thư mục |
|---|---|
| Kiến trúc + loss v2 | `src/vifasttts/models/` (`vifasttts_v2.py`, `v2_losses.py`, `slot_ops.py`) |
| Train | `src/vifasttts/train_v2.py` |
| Inference + steady mode | `src/vifasttts/infer.py`, `src/vifasttts/frontend/chunking.py` |
| ONNX | `src/vifasttts/runtime/onnx_engine.py`, `src/vifasttts/export.py` |
| Thước rè/coda | `tools/roughness_dashboard.py`, `tools/coda_realization.py` |
| Configs | `configs/small_codelung_v32.yaml` (v3.2), `base_codelung_v32.yaml`, `small_codelung_v34.yaml` |
| Runs | `runs/acoustic_v2_b8`, `runs/acoustic_v32`, `runs/acoustic_v32_base`, `runs/acoustic_v33`, `runs/acoustic_v34` |
| Dữ liệu | `data/codelung/audio_v2/`, `data/codelung/v32/` (manifest clause-level) |
| Tài liệu chi tiết | `docs/ACOUSTIC_V2.md` (§1–§53), `docs/EVAL_TOOLS.md` |

*Số liệu trong file này đo ngày 20–21/10/2026 trên M6; các so sánh đều ghi rõ val/thước.*
