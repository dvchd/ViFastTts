# A0 lượt 2 — Kết quả Việc 1 (1.1–1.6), 21/10/2026

Checkpoint dùng: `runs/acoustic_v32/best.pt` (v3.2 small, 59k). Mọi số là **[ĐO]** trên M6.

## 1.1 GV theo 3 loại frame + map bin→Hz (center filterbank)

| loại frame | n frame | GV median (100 bin) | GV dải 1–5 kHz |
|---|---|---|---|
| **hữu thanh** | 13.500 | **0,565** | **0,586** |
| vô thanh | 323 | 0,208 | 0,214 |
| **lặng (G/lead/gap)** | 6.139 | **0,123** | 0,128 |

**Kết luận**: GV toàn cục 0,333 bị **frame lặng kéo xuống** đúng như A0 dự đoán. Vùng tiếng nói thật
(**hữu thanh**) = **0,565** — sát ngưỡng mục tiêu 0,6 hơn nhiều. ⇒ Khi đánh giá adversarial phải đo
**GV vùng hữu thanh**, không dùng GV toàn cục.
File: `artifacts/eval_acoustic/gv_by_type.csv` (cột bin, hz, voiced, unvoiced, silence).

## 1.2 Mel stats: toàn frame (hiện tại) vs chỉ frame tiếng

- Lệch mean per-bin: **median |Δ| = 1,214** (max 3,97).
- Tỉ lệ std (speech / current): **median 0,939** (min 0,740 · max 1,562).

⇒ Frame lặng **làm lệch mean ~1,2 z** và **phình std ~6%**. Đây là khác biệt **đáng kể** ⇒ đưa vào
v3.5 (tính lại stats chỉ trên frame tiếng). Chưa đổi gì ở checkpoint hiện tại.

## 1.3 Nghe mù α=0 vs α=1,5 (10 câu, TF)

Trang: `artifacts/listen_small_filt/alpha_blind/index.html` — mỗi hàng 2 bản **A/B ngẫu nhiên**
(seed 13), đáp án trong mục "Đáp án" (bấm sau khi nghe). Cả 2 bản đã cân peak −3 dBFS.
Chờ tai người dùng.

## 1.4 F2/centroid cuối nucleus — **AUDIO THẬT trước**, synth sau

### Audio thật (corpus val), centroid dải 500–4.000 Hz, 50% cuối nucleus
| rime | n | centroid |
|---|---|---|
| **NONE\|A\|NH ("anh")** | 7 | **1.129 Hz** |
| **NONE\|A\|N ("an")** | 24 | **967 Hz** |
| NONE\|I\|N ("in") | 33 | 1.891 Hz |
| B\|A\|N | 10 | 862 Hz |
| L\|A\|N | 7 | 1.158 Hz |
| X\|A\|NH ("xanh") | 4 | 976 Hz |

**Thai thật có phân biệt**: "anh" (1.129) **>** "an" (967) đúng chiều /ɲ/ > /n/ ⇒ **giữ thước centroid**
(không bỏ như nhánh "nếu thật cũng không khác").

### Synth (TF, cùng rime) — so với thật
Nhiều rime bị **hạ centroid** (mờ dải cao): M|I|NH −64 Hz · K|A|C −95 · B|A|N −45 · Đ|I|NH −141 ·
K|Ô|T −126 · D|Â|N −467. Và cặp tối thiểu tách rời: xanh 395 < xan 450 (ngược chiều thật).

## 1.5 Dict + acoustic model đã dùng để align corpus

**[CODE]** `docs/DATASET_V2.md` §: lệnh align dùng
`data/codelung/vn_models/vietnamese_mfa.dict` + model `vietnamese_mfa`
(`mfa align data/codelung/mfa_v2c ... vietnamese_mfa ...`).
⚠️ Dict này **có biến thể "nh → n"** (`xanh … s a˨˨ n`, `lạnh … l a˨˩˨ n`) bên cạnh "ɲ".
⇒ Biên coda của một phần từ "nh" có thể được đặt theo phone **n**. Chỉ báo cáo, chưa align lại.

## 1.6 UTMOS / DNSMOS

- `utmos` (bản gốc): **cài thất bại** (fairseq không build được trên py3.12/torch hiện tại).
- `speechmos` (**DNSMOS P.835**, onnxruntime): **cài được**, chạy trên M6. Kết quả 10 câu val:

| nguồn mel | SIG | BAK | OVRL |
|---|---|---|---|
| copy (mel thật → Vocos) | **3,38** | 3,96 | **3,07** |
| teacher-forced | 3,28 | **4,07** | 3,04 |
| free-run | 3,27 | 4,05 | 3,02 |

⇒ DNSMOS **không phân biệt** copy vs model (lệch ≤0,1; SIG của model thấp hơn 0,1, BAK cao hơn 0,1).
Giá trị tuyệt đối thấp (OVRL ~3,0) vì DNSMOS vốn train cho thoại/nhiễu. **Giữ làm thước tham khảo**,
không dùng làm cổng chính. Mẫu wav: `artifacts/eval_acoustic/dnsmos_samples/`.

---

## Đề xuất tiếp (đúng thứ tự A0)

1. **Việc 2 — adversarial fine-tune từ v3.2 best** theo đúng spec: freeze embedding/encoder/duration/
   slot/pause/variance; train decoder+postnet; G lr 2e-5, D lr 1e-4, hinge, crop 64 frame vùng tiếng,
   adv 0.1 ramp 2k, feature matching 2.0, EMA 0.999, 20k bước, probe 1k trước.
   **Thước đo mới**: GV **hữu thanh** (mục tiêu >0,6 từ 0,565) + mel L1 val ≤ +5% + CER + tai (blind,
   cân −23 LUFS) + DNSMOS tham khảo.
2. **Việc 3 — v3.5** gộp: mask lặng khỏi mel L1 (±4 frame quanh biên), mel stats trên frame tiếng,
   f0_delta 0.5→0.1 (sau khi đo grad norm từng loss trên encoder), bỏ phonation, EMA 0.999,
   slot N nhận embedding coda + slot C nhận embedding nucleus, adversarial sau 30k nếu việc 2 đạt.
3. Chưa làm: GTA vocoder, synthetic VieNeu, hop 128, flow-matching.
