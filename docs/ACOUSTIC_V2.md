# Acoustic v2 — thiết kế chốt (Plan + Addendum 1–5)

> **Trạng thái: ĐÓNG thiết kế (13/10/2026).** Thứ tự ưu tiên nguồn gốc:
> **Addendum 5 > 4 > 3 > 2 > 1 > Plan**. Tài liệu này là bản tổng hợp chốt để triển khai;
> số đo thực thi ở §15. Nền nghiên cứu: `docs/TTS_ARCHITECTURE_RESEARCH.md` §9 (v2.1),
> `docs/REVISION_PLAN_V2.md`.

## 0. Mục tiêu & nguyên tắc

- **Prosody từ cấp âm tiết → cấp segment** (O/M/N/C) + **khoảng lặng tường minh** (slot G):
  ngã (creak), hỏi (dip-break), nặng (ngắn/tense), coda tắc, iê/uô/ươ + medial w.
- Giữ **deterministic NAR, 1 forward, ONNX 1 graph tĩnh** (C1–C10); không đưa
  AR/diffusion nhiều bước vào đường chính.
- **Không nâng cấp gì trước khi có thước đo** — B0 (bản hiện tại, 20k bước) là mốc so sánh.
- **Vocos là lớp thay được**: acoustic chỉ cam kết "hợp đồng mel" (`mel_config_hash`).
- Chống mel mờ bằng **adversarial train-only**, không đổi inference.

Ba phát hiện nền (đã kiểm chứng trên code/dữ liệu thật):

- **F1 — B0 lệch train/infer**: train phủ gap bằng vector âm tiết (`spans = durations+gaps`),
  infer không có gap → free-run của B0 bị bất lợi hệ thống. **Không sửa B0**; so sánh chính
  bằng `tf_syl`; R1 chỉ dùng mel `tf_syl`.
- **F2 — 5 slot dùng lại `LengthRegulator`**: slot duration 0 tự bị bỏ qua (đã kiểm chứng).
- **F3 — fallback `allocate_durations` im lặng**: v2 loại row không forced; **raise** nếu train v2
  thiếu `seg_durations` (không fallback sang prior).

## 1. Kiến trúc 5 slot `[O, M, N, C, G]` + `G_lead`

```text
h_syl [B,N,H]
 → slots [B,N,5,H]   (O/M/N/C từ segment expander; G = vector "lặng" học được)
 → reshape [B,5N,H] + durations [B,5N]  (inactive = 0, G = gap)
 → LengthRegulator hiện tại (KHÔNG sửa) — slot duration 0 tự bị bỏ qua
```

- Chuỗi slot toàn câu có thêm `G_lead` ở đầu → độ dài `1 + 5N` (cho holdout còn `lead > 0`).
- **Bất biến toàn cục**: `G_lead + Σ(5N slot) == T` cho mọi row (segment: `G_lead = 0`).
- Slot G:
  - vector = `silence_emb` + `Linear(boundary_emb, punct_emb)` của **âm tiết đứng trước**;
  - **không** cộng variance F0/energy/phonation; `vuv = 0`; position = 0;
  - train: `gaps` thật (gap cuối row segment = 0 sẵn); infer: pause predictor (§4).
- `G_lead` dùng **chung `silence_emb`** với slot G (segment train có `G_lead = 0` nên vector riêng
  không bao giờ nhận gradient); boundary = `SENTENCE`.
- Predictor cấp segment = **MLP** trên `concat(h_syl, role_emb, emb thành phần của role)` — không dùng
  Conv trên chuỗi 5N (kernel sẽ trộn slot G/inactive vào ngữ cảnh). Predictor cấp âm tiết giữ Conv:
  **duration tổng**, **pause**.
- Position: `(t+0.5)/n` **trong từng slot**; slot G cố định 0.
- Output ONNX thêm: `slot_durations [B, 1+5N]` (axis `slots`), `pause_gate [B,N]` (axis `syllables`).

## 2. Chia slot tĩnh (B2) — một hàm tensor duy nhất

1. `n_act` = số slot active; `k = min(d_syl, n_act)`.
2. Cấp 1 frame cho `k` slot active **ưu tiên cao nhất** — thứ tự cố định **N, O, C, M**.
3. Phần còn lại `r = d_syl − k` chia theo `ratio` (chỉ trên active) bằng floor + largest remainder.
4. Hòa phá bằng chỉ số role cố định.

- Xếp hạng 4 phần tử bằng **ma trận so sánh 4×4 + sum** (hoặc TopK k=4 cố định) — không argsort động.
- **3 bất biến (A1)**: inactive = 0 · active ≥ 1 nếu đủ frame · `Σd_seg == d_syl` tuyệt đối.
- `frontend/segments.py:distribute_frames` sửa theo **cùng thuật toán** (một nguồn sự thật).
- Test: random 1e5 + Torch==ONNX + 3 bất biến + `Σ(5N)+G_lead==T` toàn manifest.
- Interface đóng băng sau R4 trong `models/slot_ops.py`:
  - `slot_positions(slot_durations, T) → pos` (`(t+0.5)/n`, slot G trả 0);
  - `build_f0_contour(s, slot_durations, pos) → f0` (công thức §6);
  - `distribute_slots(d_syl, ratio, active) → d_seg`.

## 3. Duration 2 tầng, loss & 4 chế độ

- Tầng 1: `d_syl` hữu thanh cấp âm tiết (khớp MFA word tier) — loss `log1p` MSE **luôn tính**.
- Tầng 2: `ratio` softmax có mask active → chia bằng B2. Loss CE/KL, **mask inactive + `d_syl < 3`** (B5).

| Chế độ | `seg_durations` | Loss ratio | Loss duration tổng | Loss pause |
|---|---|---|---|---|
| Train v2 | Có (từ `_seg.jsonl`) | Tính (mask inactive + d<3) | Luôn | Luôn (mask gap cuối row) |
| `tf_seg` | Có | Chỉ báo cáo | Chỉ báo cáo | Chỉ báo cáo |
| `tf_syl` | `None` (model tự chia) | Không | Chỉ báo cáo | Chỉ báo cáo |
| `free` | `None` | Không | Không | Không |

- **Raise** nếu `arch == v2_slot5` đang train mà `seg_durations is None` — không fallback.

## 4. Pause predictor — gate + log_gap

- `gate`: BCE, target `gap ≥ 3 frame` (~35 ms; tinh chỉnh theo phân bố thật ở R5b).
- `log_gap`: `log1p` + MSE, **chỉ tính khi gate = 1**.
- Inference (tất định, `Where`):
  ```text
  has_pause = gate > 0.5
  gap       = has_pause ? round(expm1(log_gap)) : 0
  ```
- Mask gap cuối row (rìa đã cắt); khoảng nghỉ **giữa các câu** do lớp ghép câu ngoài model lo.

## 5. Speed trong graph (`float32 [1]`)

- `speed` là **input scalar của graph** — lần đầu speed thật trong ONNX (v1: 2 forward PyTorch,
  ONNX chỉ xấp xỉ bằng `_stretch_mel`).
- Công thức (gate quyết định ở speed gốc):
  ```text
  has_pause = pause_gate > 0.5
  gap       = has_pause ? round(expm1(log_gap)) : 0
  gap'      = has_pause ? max(1, round(gap / speed)) : 0
  d_syl'    = max(1, round(d_syl / speed))
  # giữ nguyên ratio, chia lại bằng B2
  ```
- Teacher-forced (`durations` truyền từ ngoài): **không scale** (tránh làm tròn 2 lần).
- Test: `speed=1.0` byte-identical; tập vị trí pause **bất biến** trên {0.5, 0.8, 1.0, 1.25, 2.0};
  `gate=0` → 0 frame kể cả speed 0.8; `gate=1` → ≥1 frame kể cả speed 1.25.
- ONNX: `speed` **float32 shape [1]** (scalar `[]` kẹt TorchScript opset 18) — test export ngay ở R4.
- Runtime: graph có input `speed` → bỏ `_stretch_mel`; rẽ nhánh theo `arch` trong metadata.

## 6. F0: hệ số hình dạng (B1) + mask hữu thanh (F4)

- Model dự đoán **hệ số hình dạng độc lập độ dài**:
  ```text
  s_k = c_k / √n        (c_k là hệ số DCT trực giao hiện có)
  f(pos) = s0 + √2 · Σ_{k=1..3} s_k · cos(πk·pos),   pos = (t+0.5)/n
  ```
  → không có `n` trong công thức dựng (ONNX elementwise); đổi speed không đổi biên độ contour.
- `s0 == mean` (test tới 1e-6; thực đo ≤ 2,8e-17 với float64).
- Stats **hai pass** trên `s_k` (R5b: tính trên frame **nucleus**, mask `vuv`), đóng băng
  `f0_stats_hash` vào checkpoint.
- `n < 4` → mask hệ số `k ≥ n`. Có **delta loss**. **Prior mềm theo thanh** từ thống kê thật (R5).
- **F4 — mask hữu thanh**: `f0.interpolate: true` lấp cả vùng vô thanh → `f0 > 0` **không phải** mask
  hợp lệ (đo trên clip mẫu: `f0>0` = 100% vs `vuv>0.5` = 71%). Dùng **`vuv`** ở mọi nơi pool F0
  (dataset v2, R5b, eval per-tone). `f0_stats.json` cũ phải tính lại (đằng nào cũng phải làm lại cho
  `s_k` + nucleus).
- Mask loss F0: slot O/C chỉ tính khi voiced ratio ≥ 0.5; nucleus luôn tính với trọng số theo tỷ lệ
  hữu thanh (đo bằng `vuv`).

## 7. Phonation — R6 quyết định

- R6 so **3 ứng viên** trên 100 file (ngã/nặng vs ngang): (1) 3 kênh hiện tại; (2) aperiodicity
  pyworld D4C; (3) độ bất ổn chu kỳ F0. Cổng: **AUC ≥ 0.7**; không đạt → không bật phonation ở v2.
- Lý do nghi ngờ kênh cũ: `glottalization` (<80 Hz) chỉ ~4 bin ở n_fft 1024/24k; HNR frame ~43 ms
  dễ làm mờ creak ngắn.
- Chỉ bật trong **lần re-cache v3**.

## 8. Chống mel mờ — adversarial train-only (T4.1)

- Mel discriminator đa thang + feature matching, chấm trên mel **sau postnet**; `mel_pre` giữ L1 0.2.
- Duration discriminator kiểu VITS2. **Tách D khỏi checkpoint/graph inference**.
- Bật sau khi L1 hội tụ (~20–30k bước); đo s/step MPS trước (R8); copy pattern warm-up D + grad clip 5.0.
- Frame slot G **vẫn tính mel L1** (học lặng sạch), **không** tính F0/energy/phonation.

## 9. Cache v3 & manifest `_seg.jsonl`

- **Cache feature v3** (một lần): mel/energy/f0/vuv/**phonation** — tất định theo audio + config.
  Giữ mtime (`rsync -a`); **cân nhắc chuyển content-hash ngay trong lần v3** (sha1 ~8,8 GB ≈ vài chục giây).
- **Manifest `_seg.jsonl`**: `seg_durations` + `seg_method` (ipa/rule/hybrid/aligner) +
  `seg_method_version` + hash bảng map/weight; checkpoint ghi lại các giá trị này (B9).
  `seg_durations` thuộc **manifest**, không phải cache (cache key theo file audio).
- Row không forced: **loại khỏi train v2** (hiện 0% — xem §15).

## 10. Tương thích ngược (checkpoint `arch` / ONNX)

- Checkpoint có trường `arch`: `"v1_syl"` (mặc định khi thiếu) | `"v2_slot5"`; **factory** đọc `arch`.
- Không load v1→v2 bằng `strict=False` ngầm — chỉ warm-start có chủ đích (§11).
- ONNX: **giữ nguyên tên + thứ tự output cũ**; thêm `slot_durations`, `pause_gate` vào cuối.
  Metadata thêm `arch`, `mel_config_hash`, `f0_stats_hash`, `seg_method`, `seg_method_version`.
- Runtime đọc output **theo tên**; input production chỉ `features` + `speed`; teacher-forced chỉ PyTorch.
- Ý nghĩa output `durations` **không đổi**: `[B,N]` hữu thanh cấp âm tiết.

## 11. Warm-start v2 từ B0

- Lấy từ B0: `embedding`, `encoder`, `decoder`, `mel`, `postnet`, `tone_classifier`, `duration`.
- Khởi tạo mới: `role_emb`, segment expander, MLP ratio/variance, pause predictor, `silence_emb`,
  `variance_proj`, `position_proj`.
- Giai đoạn A (~2k step): **đóng băng** `embedding`, `encoder`, `duration`; train module mới +
  `decoder/mel/postnet` với LR **0.25×**; grad clip 1.0; log mel riêng **G vs hữu thanh**.
- Giai đoạn B: mở khóa toàn bộ, warm-up LR lên base trong 2k step rồi cosine như config.
- Cổng cuối giai đoạn A: **trên val** (trung bình 200 step cuối hoặc 1 val đầy đủ) — mel hữu thanh
  không tệ hơn B0 quá **20%**; **không** đánh giá dựa trên loss 500 step đầu.
- Optimizer **mới hoàn toàn**; checkpoint ghi `warm_start_from = <hash B0>`.

## 12. Eval: `tf_syl` / `tf_seg` / `free` + mask vùng đo

| Chế độ | Dùng thật | Dự đoán | Dùng để |
|---|---|---|---|
| `tf_syl` | `d_syl`, `gaps`, `lead` | ratio, variance | **So B0 với v2** (chính) |
| `tf_seg` | + `seg_durations` | variance | Chẩn đoán v2 |
| `free` | — | tất cả | CER, RTF, RAM, nghe |

- Tập val cố định, seed cố định, ghi danh sách clip vào repo.
- **Mask vùng đo lấy từ manifest** (không từ output model):
  - mel L1 / MCD: frame `[lead, T)` (bỏ vùng lead);
  - F0 RMSE / VUV theo thanh: chỉ frame **hữu thanh của âm tiết** (`durations`), bỏ gap + lead.
- RTF: **acoustic ONNX + vocoder ONNX + iSTFT numpy + chia câu**, CPU M6, ghi số luồng, tách phần.
- Minimal pairs: đặt trong **câu khung** cố định, chấm **âm tiết đích** sau align, báo độ chính xác
  thanh theo nhóm + **trang nghe A/B mù**.
- **R1 chỉ dùng mel `tf_syl`**.

## 13. Lộ trình thi hành

1. Song song: T1.3 (B0 9k→20k) · T1.1 (mel contract) · T1.2 (eval_acoustic + mask) · R5a · punct test.
2. B0 + R1.
3. R2 (3 phương án weak label; ưu tiên **lai**: biên onset|rhyme từ MFA + weight rule trong rhyme)
   → sửa hàm chia (B2) → manifest `_seg.jsonl` → T2.1 → R5b.
4. R6 → cache v3 (một lần).
5. R4 → đóng băng `slot_ops.py`.
6. Task 5 slot (gộp T2.2+T2.3) + warm-start + tương thích ngược → cổng.
7. T3.2 → T3.3 → cổng.
8. R3 → T2.4 (điều kiện, không chặn).
9. R8 → T4.1 → cổng.
10. T5.1 GTA. 11. R7 sau Phase 3; Phase 6 (CFM) chỉ khi điều kiện mở.

## 14. Checklist PR

- [ ] Unit test + **3 bất biến** duration; `G_lead + Σ(5N) == T` toàn manifest + random 1e5.
- [ ] Zero-slot: không frame nào gather vào slot 0 (kể cả đầu/cuối chuỗi).
- [ ] `s0 == mean` + `build_f0_contour` khớp `idct4_frames` sau quy đổi.
- [ ] Punct 4 case (dấu sau → âm tiết trước; dấu mở đầu → sau; OOV respell; nhiều dấu).
- [ ] ONNX: export tĩnh, parity Torch/ONNX, **byte-identical** 2 lần chạy, output cũ giữ tên/thứ tự.
- [ ] Load được checkpoint `v1_syl` cũ và infer như trước.
- [ ] Loss raise khi train v2 thiếu `seg_durations`.
- [ ] Tập vị trí pause không đổi theo speed {0.5, 0.8, 1.0, 1.25, 2.0}; `speed` float32 [1]; test export ở R4.
- [ ] Runtime rẽ nhánh `_stretch_mel` theo `arch`; `silence_emb` chung `data_ptr` cho G/G_lead.
- [ ] Báo số row bị loại (không forced); bảng metric `tf_syl` + `free` so B0.

## 15. Preflight trên M6 (13/10/2026) — kết quả

Môi trường: commit `2f6878a` · Python 3.12.15 · torch 2.14.1 · soundfile 0.14.0.
Script: `tools/check_v2_preflight.py` (5 kiểm tra; chạy trước mọi tính năng).

1. **forced**: 0% không forced trên cả 5 manifest — holdout 842 · train 5.811 · val 1.013 ·
   train_seg 9.763 `{forced 9.674 + forced_khong_gap 89}` · val_seg 1.773 `{forced 1.761 + forced_khong_gap 12}`.
   89+12 row là segment **1 âm tiết** (`gaps=[]`) — đúng nhánh `elif` của dataset. `approx_resample = 0`.
2. **F1 xác nhận** (row `4kyqfOypVWI_0001__s00`): `spans == durations + gaps`; frame lặng mang vector
   âm tiết (`max|Δ| = 0`); `pos` chạy 0→0,988 trên cả span; `exp_f0` đồng nhất qua span.
3. **LengthRegulator**: 3/3 OK — `(2,0,3)→[0,0,2,2,2]`, `(0,2,3)→[1,1,2,2,2]`, `(2,3,0)→[0,0,1,1,1]`.
4. **DCT**: 7/7 OK — max |s0−mean| = 2,8e-17; max |B1−idct| = 4,4e-16 (float64, n=1..40).
5. **Holdout lead**: 842 row · lead>0: 533 (63,3%) · p50 = 18 · max = 35 frame.

Ghi chú (không chặn): ca `len(gaps)=N-2` + `Σd==T` ra `forced_khong_gap` thay vì `gaps_sai_do_dai`
(dataset lỗi shape ở ca này) — không có row nào như vậy trên dữ liệu thật.

**T1.1 (mel contract)** — `tests/test_vocos_mel_contract.py`: bank ≤ 6,6e-6; mel max|Δ| ≤ 1,2e-4
(dư lượng fp32; fail threshold 1e-3; mean|Δ| < 1e-5); round-trip melL1 0,145–0,154; len_ratio 0,9992–0,9994.
**Punct test** — `tests/test_punct_attachment.py`: 4/4 pass (luật gộp `?!`→QUESTION, `...`→PERIOD ghi trong test).

## 16. Ghi chú nhỏ đã chốt

- Nucleus đôi IÊ/UÔ/ƯƠ: v2 dùng **1 slot N**; tách 2 đích giữ ở `frontend/segments.py` (chưa nối vào v2).
- `"..."` giữa câu hiện gắn `SENTENCE` — theo dõi ở v2.1 cùng SIL token.
- Giới hạn `MAX_POSITIONS = 4096` (≈43,7 s @ hop 256) — thêm assert + chia câu trước model.
- Row segment N=1 (`gaps=[]`) là bình thường; v2: G slot = 0.

## 17. Kết quả R2 — weak label O/M/N/C (13/10/2026)

Tool: `tools/r2_weak_labels.py` · báo cáo: `artifacts/dataset_v2/r2_report.json`
(200 clip `mfa_v2c_vn_aligned` = 7.224 âm tiết, seed 42; đối chiếu thêm dict 3.631 mục).

| Phương án | Coverage (corpus) | Coverage (dict) | Ghi chú |
|---|---|---|---|
| `ipa` (biên từ phone tier) | 94,4% | 91,5% | fail chủ yếu: **coda tắc không có phone riêng** (C 114 · T 123 · P 17 · CH 15) |
| **`lai` (biên onset\|rhyme từ IPA + weight rule trong rhyme)** | **98,3%** | **100,0%** | ≥ cổng 85% (B4) → **chọn làm mặc định** |
| `rule` (weight thuần) | 100% | 100% | luôn chạy; nhưng timing thô hơn |

- Nhóm fail 1,7% còn lại: **từ vay / đọc chữ cái / âm tiết cụt** không có nguyên âm mang thanh
  ("pi", "bo", "gu", "vi", "min", "tu"...; median 110 ms, p90 220 ms) → **fallback weight rule**.
- Bảng phone→role thực nghiệm (top; đầy đủ trong JSON): onset `k m l ɗ n tʰ s tɕ…` · medial `w` ·
  nucleus (nguyên âm mang thanh) · coda `j n ŋ w ɲ ŋm t̚…`.
- **Định lượng B5**: trên âm tiết IPA đầy đủ, weight rule lệch IPA **nucleus +15,5 ms ·
  coda −15,2 ms** (mean |Δ| ≈ 18 ms ≈ 1,7 frame) → ratio predictor chỉ học lại luật;
  timing thật (nặng/coda tắc/iê-uô-ươ) cần biên IPA hoặc aligner (T2.4).
- **Chốt cho T2.1**: dùng **hybrid** — slot theo cấu trúc abstract (inactive = 0 frame);
  âm tiết không tìm được biên rhyme → weight rule; min-1 frame cho slot active theo B2.

## 18. T2.1 — `seg_durations` weak label (13/10/2026)

Tool: `tools/add_seg_durations.py` (hybrid §17 + fallback rule) → `data/codelung/<stem>_slot4.jsonl`,
mỗi row thêm `seg_durations [N,4]` (thứ tự **O, M, N, C**) + `seg_method` + `seg_method_version`
+ `seg_weights_hash` (B9). Report: `artifacts/dataset_v2/seg_durations_report.json`.

| Tập | Row | Âm tiết | Lỗi `Σ seg ≠ d_syl` | Fallback (rule) | Methods |
|---|---|---|---|---|---|
| train | 9.763 | 222.737 | **0** | 1,4% | hybrid 7.576 + hybrid+rule 2.187 |
| val | 1.773 | 41.521 | **0** | 1,6% | hybrid 1.332 + hybrid+rule 441 |
| holdout | 842 | 35.112 | **0** | 1,5% | hybrid 531 + hybrid+rule 311 |

- Slot active: 1 slot 2,3% · 2 slot 36,8% · 3 slot 59,0% · **4 slot 1,9%** (medial chỉ 2,5%
  âm tiết — corpus vlog; M share khi active ~15,4%).
- Role share trung bình của `d_syl`: O 35,8% · M 0,4% · N 48,5% · C 15,3%.
- Nucleus share theo thanh (median): hỏi 0,50 · huyền 0,467 · ngang 0,467 · ngã 0,474 ·
  sắc 0,45 · **nặng 0,417** — khác biệt chủ yếu do thành phần coda, đúng cảnh báo B5:
  luật trọng số không biết nét thanh → timing thật cần IPA/aligner (T2.4).
- Bất biến đã verify độc lập trên cả 3 file: `Σ seg == d_syl` **100%**;
  `lead + Σd + Σg == T` giữ nguyên (không đụng durations/gaps).

## 19. R5b + Addendum 6 — thống kê cấp segment & mask F0 thật (13/10/2026)

### 19.1 R5b — số đo (`tools/r5b_segment_stats.py`, 2.500/9.763 row seed 42)

Nguồn model: `data/codelung/f0_stats_v2.json` (đóng băng `f0_stats_hash` khi train);
báo cáo đầy đủ: `artifacts/dataset_v2/r5b_stats.json`.

| Đại lượng | Giá trị |
|---|---|
| log-F0 **nucleus** (vuv>0,5) | mean **4,804** · std **0,168** (454.642 frame) |
| log-F0 all (đối chiếu) | mean 4,847 · std 0,224 (bản cũ: 4,871/0,262 — rộng hơn vì gồm f0 nội suy) |
| `s_k` chuẩn hoá = **nucleus** | mean [0,053 · 0,091 · 0,036 · 0,017] · std [**0,902 · 0,308 · 0,170 · 0,120**] |
| energy theo segment | mean **−1,025** · std 1,768 (147.430 segment); nucleus −0,132/0,980 |

- Energy: bản cũ (−4,58/4,67) đo trên âm tiết **chia đều cả clip** (thiếu alignment) — lần này mới
  đo đúng trên cửa sổ segment thật (O/N/C tách nhau) → **thay thang chuẩn hoá** cho v2.
- Số frame hữu thanh trung bình/segment: O 5,5 · M 2,9 · N 8,1 · C 4,2.

- **Prior mềm theo thanh** (nucleus, `s0..s3`): ngang +0,387·+0,093·+0,021·+0,019 ·
  huyền −0,349·+0,141 · sắc +0,420·−0,044 · hỏi −0,843·+0,351 (dip) · ngã +0,275·−0,084·+0,102
  (creak) · nặng −0,080·+0,172 — dấu hình dạng **hợp ngữ âm** (huyền/nặng đi xuống, sắc đi lên,
  hỏi lõm, ngã cong) → dùng làm prior T3.2.
- Duration segment: O median 53,3 ms · N 74,7 · C 42,7 (p90 N 160 ms); nucleus sắc/nặng 64 ms
  ngắn hơn ngang/huyền 74,7 ms (đúng R5a).
- Voiced ratio theo role (số âm tiết): O 0,955 · M 0,989 · N 0,986 · C 0,979 — **mọi role ≥ 0,97**.

### 19.2 Addendum 6 — `vuv` không đủ làm mask F0 cho slot O (probe định lượng)

Probe: `tools/probe_f0_validity.py` · 400 row = 7.991 âm tiết ·
`artifacts/dataset_v2/f0_validity_probe.json`. Kết quả theo nhóm cấu trúc:

| Nhóm (cửa sổ segment) | `vuv=1` | ratio ≥ 0,5 | jitter f0 (median · p90) | rel-energy median |
|---|---|---|---|---|
| onset hữu thanh (11 lớp) | 97,8% | 99,3% | **0,0253 · 0,134** | −0,5 dB |
| onset vô thanh (11 lớp) | 90,2% | 93,8% | **0,0785 · 0,338** (3,1×) | −2,9 dB |
| coda sonorant (M/N/NG/NH/W/J) | 97,3% | 98,2% | 0,0203 · 0,109 | −0,6 dB |
| coda tắc (P/T/C/CH) | 98,0% | 98,6% | 0,0389 · 0,175 (1,9×) | −2,4 dB |

- **Phát hiện**: pyworld harvest+stonemask bật `vuv` cho ~90% frame của onset vô thanh
  (closure ngắn trong speech liên tục) kèm f0 nhiễu 3-4× (ví dụ /k/ "các": 164→206→139 Hz).
  **Năng lượng không tách được** (frame < −15 dB so nucleus = **0,0%** cho mọi nhóm) — mọi
  "cổng năng lượng" đều vô dụng; đây cũng là một phần gốc rễ F0 nhiễu của B0.
- **Quyết định (mask F0 v2)**:
  1. **O**: chỉ tính F0 loss khi `onset ∈ F0_VOICED_ONSETS` (B, D, G, GI, L, M, N, NG, NH, R, V —
     sonorant/implosive/ɣ, jitter ≤ 0,03) **và** `voiced ratio(O) ≥ 0,5`.
  2. **N**: luôn tính, trọng số = voiced ratio (§6 giữ nguyên).
  3. **M/C**: `ratio ≥ 0,5` giữ nguyên §6 — coda tắc tiếng Việt **không bung hơi**, f0 đo được là
     đuôi nguyên âm (jitter 1,9× sonorant nhưng tuyệt đối nhỏ) → vẫn học được (không cần luật lớp).
  4. Hằng số đóng băng tại `frontend/segments.py` (`F0_VOICED_ONSETS`, `F0_VOICED_CODAS`).
- **Không đổi cache** (mask suy từ cấu trúc, tất định từ text) → không ảnh hưởng lần re-cache v3.
- **Theo dõi**: nếu R6 (D4C aperiodicity) đạt cổng, cân nhắc `f0_valid` theo aperiodicity như
  lớp gác thứ hai cho onset hữu thanh jitter cao (B 0,039 · D 0,056 — vẫn nhận, nhưng re-đo sau R6).
- Giá trị s_k báo cáo ở §19.1 vẫn tính trên **mọi** segment (chuẩn hoá phục vụ T3.2); trọng số
  loss F0 ở T3.2 dùng đúng mask §19.2.

## 20. R6 — chọn kênh phonation (13/10/2026)

Tool: `tools/r6_phonation.py` · 100 clip (seed 42) · dương = ngã+nặng (444) vs âm = ngang (913) ·
đo trên **nucleus** hữu thanh (≥ 4 frame) · report `artifacts/dataset_v2/r6_phonation.json`.
`power = max(AUC, 1−AUC)`; cổng **≥ 0,7**.

| Kênh | AUC gộp | chỉ cuối câu (7%) | ngã | nặng | clip-z | **power** | Cổng |
|---|---|---|---|---|---|---|---|
| `hnr` (autocorr, dB) | 0,131 | **0,057** | 0,123 | 0,135 | 0,155 | **0,869** | ✅ |
| `jitter` (bất ổn chu kỳ) | 0,864 | **0,942** | 0,886 | 0,853 | 0,835 | **0,864** | ✅ |
| `glottalization` (< 80 Hz) | 0,763 | 0,691 | 0,776 | 0,756 | 0,747 | **0,763** | ✅ |
| `aper` D4C (1–4 kHz) | 0,424 | 0,466 | 0,282 | 0,499 | 0,423 | 0,576 | ❌ |
| `tilt` (dB/oct) | 0,478 | 0,379 | 0,571 | 0,430 | 0,496 | 0,521 | ❌ |

- **Kết quả**: creak thể hiện rõ nhất qua **HNR thấp** và **jitter cao** (power ~0,87), mạnh nhất ở
  **cuối câu** (hnr 0,943 · jitter 0,966) — đúng bản chất creak phrase-final. `glottalization`
  đạt cổng nhưng yếu hơn. **Kênh cũ `tilt` vô dụng** (0,521) và D4C 1–4 kHz không đạt (0,576).
- **Bẫy đo đã bắt**: bản đầu tính `aper` dải 0–4 kHz → power 0,704 (tưởng đạt). Đo bằng tín hiệu
  tổng hợp thấy cheaptrick **thoái hoá dưới 1 kHz** (sine 150 Hz thuần: 0–1 kHz = 1,12 do "hài
  vắng" bị tính aperiodic; 1–4 kHz = 0,00) → dải 0–4 kHz bị thành phần thoái hoá chi phối.
  Sửa về chuẩn 1–4 kHz (`estimate_aperiodicity` mặc định) → rớt cổng đúng thực chất.
- **Chốt (cache v3 + v2)**: bật phonation với **3 kênh đạt cổng** — `hnr` · `glottalization` ·
  `jitter` (khớp `use_phonation: true` 3 chiều của variance head). `tilt`/`aper` giữ trong code
  (`data/phonation.py`) làm chẩn đoán, **không** vào cache để tránh phí.
- **Config riêng `configs/small_codelung_v2.yaml`**: vì `feature_cache_key` gồm cả block
  `phonation`, đổi config sẽ đổi khoá → tách file để **cache v2 (B0) không bị đè** (2 bộ khoá
  chung thư mục `cache/codelung_v2`).
- Ghi chú vận hành: `jitter` tái dùng `f0/vuv` sẵn có (không harvest lần hai); `extract_features`
  truyền f0/vuv vào `extract_phonation_frames` — xem `tests/test_phonation_r6.py` (6 test).

## 21. R4 — đóng băng `models/slot_ops.py` (13/10/2026)

Interface ba hàm, thuần torch (ONNX-export được, không vòng lặp động / không argsort):

| Hàm | Chữ ký | Nội dung |
|---|---|---|
| `distribute_slots` | `([B,N] long, [B,N,4] float, [B,N,4]) → [B,N,4] long` | B2: k=min(d_syl,n_act) frame cho slot active ưu tiên **N,O,C,M**; dư chia theo ratio (floor + largest remainder); hòa phá theo chỉ số role (O,M,N,C) |
| `slot_positions` | `([B,S] long, gap_mask?, max_frames?) → [B,T] float` | `(t+0,5)/n` trong từng slot; slot G = 0 |
| `build_f0_contour` | `([B,S,4], [B,S] long, [B,T]) → [B,T]` | §6: `f = s0 + √2 Σ s_k cos(πk·pos)` |

- Xếp hạng bằng **ma trận so sánh 4×4 + sum** (đúng yêu cầu B2) — không TopK/argsort động.
- **Test** (`tests/test_slot_ops.py`, 9 test): bất biến trên **1e5 mẫu ngẫu nhiên** (Σ == d_syl ·
  inactive = 0 · active ≥ 1 khi đủ frame · thứ tự ưu tiên khi thiếu frame · hòa phá tất định ·
  ratio chưa softmax vẫn bất biến); `slot_positions`/`build_f0_contour` khớp `idct4_frames` và
  `s0 == mean`; **parity Torch == ONNX** (opset 18, `dynamo=False`, chạy 2 lần byte-identical).
- `frontend/segments.py:distribute_frames` cập nhật **cùng thứ tự ưu tiên B2** (N,O,C,M) —
  test `test_distribute_frames_frontend_cung_uu_tien_b2`.
- Bug đã bắt khi test 1e5: mask `active` ở trục sai trong ma trận so sánh (che nhầm trục j) —
  sửa `unsqueeze(-2)`; nếu chỉ chạy vài ca nhỏ đã có thể lọt.
- Runtime speed (§5): `d_syl' = max(1, round(d_syl/speed))` rồi gọi lại `distribute_slots`
  với **cùng ratio** — không cần lưu d_seg.

## 22. T2.2 (nền tảng) — module mục tiêu cấp segment (13/10/2026)

`src/vifasttts/data/v2_targets.py` (+ `tests/test_v2_targets.py`, 7 test) — một nguồn sự thật cho:

- **Cửa sổ segment** `segment_windows(row, relative=…)`: toạ độ tuyệt đối (R5b, T2.1) hoặc theo
  feature đã cắt (dataset v2); `tools/r5b_segment_stats.py` đã gọi lại từ đây.
- **Ratio target + mask** (B5: mask inactive + `d_syl < 3`) và **pause target** (§4: gate `gap ≥ 3`,
  `log1p(gap)`, mask gap cuối row).
- **Pool** energy/`vuv` theo cửa sổ role; **`s_k` + trọng số loss F0** đúng Addendum 6
  (N = tỷ lệ hữu thanh; O cần `F0_VOICED_ONSETS ∧ ratio ≥ 0,5`; M/C ratio ≥ 0,5; `n < 2` → bỏ).
- `segment_targets(row, features, f0_stats)` gộp 9 tensor cho collate (task 5 slot tiêu thụ).

## 23. Cache v3 + task 5 (đợt 1) — model `v2_slot5` + factory/warm-start (13/10/2026)

### 23.1 Cache v3 đã build + verify 100%

Build: `.venv/bin/vifasttts cache-features --config configs/small_codelung_v2.yaml` cho 3 manifest
(train 9.763 row / **5.807 file**, val 1.773 / 1.011, holdout 842 / 842). Verify bằng
`tools/verify_cache_v3.py` (sub-agent) — **0 file thiếu · 0 lỗi cấu trúc · 0 mismatch recompute**
(20 file/manifest trích lại so cache, tất định theo audio+config); exit 0. Kênh:
`mel, energy, f0, vuv, hnr, glottalization, jitter`.

### 23.2 `models/v2_slot5.py` + `models/factory.py` (đợt 1 của task 5)

- **Model v2** (§1–§6): hidden cấp segment `[O,M,N,C,G]` (segment expander = MLP trên
  `concat(h_syl, role_emb, emb thành phần)`); `G` = `silence_emb + Linear(boundary_emb, punct_emb)`;
  `G_lead` chung `silence_emb` + boundary SENTENCE; ratio softmax có mask active; pause head
  (gate + log_gap) cấp âm tiết; hệ số `s_k` per segment (4) + vuv + energy + phonation (3);
  `speed` float32 `[1]` **trong graph** (chỉ scale khi tự dự đoán — §5); output v1 giữ nguyên +
  `slot_durations` + `pause_gate` ở cuối; output mới cho eval: `seg_ratio`, `seg_shape`, `f0` [B,T].
- **Factory**: `arch_of_checkpoint` (thiếu → `v1_syl`), `build_model(config, sizes, arch)`;
  `warm_start_v2_from_v1` copy **tường minh** (§11: embedding/encoder/decoder/mel/postnet/
  tone_classifier/duration), raise khi shape lệch — không load chéo ngầm.
- **Test** (`tests/test_v2_slot5.py`, 8 test): bất biến `G_lead + Σslot == T` (khớp cả list),
  `Σ d_seg == d_syl` + inactive = 0 ở free, `speed=1.0` **byte-identical**, vị trí pause bất biến
  {0.5, 1.0, 1.25}, gate=1 → ≥1 frame kể cả speed 1,25 (và đúng `round(gap/speed)`), gate=0 → 0 frame
  kể cả 0.8, teacher-forced **không** scale, warm-start so khớp từng tensor, factory arch,
  **ONNX export parity** + độ dài động N=4→6.
- **Bẫy đã bắt khi test**: (1) `segment_to_slot` thiếu `G_lead` → lệch chỉ số 1 slot (f0/vuv/energy
  gán nhầm slot — chỉ lộ vì test kiểm tra *giá trị*, không chỉ shape); (2) export wrapper phải để
  model là **submodule** (closure làm tham số bị bake thành constant → lỗi "requires grad as constant")
  và bọc `torch.no_grad()` (v2 có slice/index Parameter — v1 không có).
- **Chưa xong (đợt 2)**: dataset v2 (nối `v2_targets` vào collate + raise khi thiếu `seg_durations`),
  loss v2, `export.py`/runtime rẽ nhánh `arch` (+ speed), eval `tf_seg`/`tf_syl` v2, warm-start
  train từ B0 → cổng so B0.
- Checklist tích hợp chi tiết: `docs/V2_INTEGRATION_CHECKLIST.md` (sub-agent, 127 dòng).

## 24. Task 5 (đợt 2) — dataset v2 + loss v2 + export/runtime rẽ nhánh (13/10/2026)

### 24.1 Dataset v2 — `data/dataset_v2.py` (sub-agent + siết lại)

`SlotDataset` (manifest `_slot4` + cache): cắt feature theo row, `features [N,7]`, `durations/gaps/lead`,
`seg_durations [N,4]`, targets từ `v2_targets` (12 key); **raise** khi thiếu `seg_durations`/
`frame_range`-quá-ngắn (không fallback im lặng — F3); row clip đầy đủ (không `frame_range`, holdout)
dùng trọn tensor. `collate_slot_batch` yêu cầu cùng N (raise nếu khác), pad trục frame + `frame_lengths`.
Cache **chỉ đọc** (miss → `extract_features`, không ghi — tránh race nhiều worker).

### 24.2 `v2_targets` bổ sung: `f0_k_mask` + phonation

- `f0_k_mask [N,4,4]`: hệ số `s_k` nào có thật (`k < n`) — loss phải mask phần thiếu (B1).
- `seg_phonation [N,4,3]` + mask (voiced ≥ 2 frame ∧ ratio ≥ 0,5): pool `hnr/glottalization/jitter`
  (cache v3) theo segment — thiếu kênh thì trả 0/mask rỗng (dataset cũ vẫn chạy).

### 24.3 Loss v2 — `models/v2_losses.py` (+7 test)

11 thành phần: `mel` (L1 mask frame, chia bin) · `mel_pre` · `duration` (MSE log1p) · **`ratio`**
(CE, mask inactive **trước softmax** + `d_syl < 3`, gộp theo âm tiết) · **`f0`** (L1 hệ số hình dạng
z-score, mask Addendum 6 × `f0_k_mask`) · `vuv` (BCE theo segment) · `energy` · `phonation` ·
`tone` · **`pause_gate`** (BCE) + **`pause_gap`** (MSE, chỉ khi gate = 1). Trọng số mặc định ở
`DEFAULT_WEIGHTS`, config `small_codelung_v2.yaml` ghi rõ 11 key.

### 24.4 Export/runtime rẽ nhánh `arch` (sub-agent) + **bug v1 postnet đã sửa**

- v2: `_export_v2_fp32` (2 input `features`+`speed`, 10 output, metadata `arch` + `seg_*`/`f0_*`),
  precision ≠ fp32 → raise (chưa hỗ trợ); runtime `run(features, speed=...)` (chỉ gửi input thứ hai
  khi arch v2) + `synthesize_onnx` bỏ `_stretch_mel` khi v2. Test: `tests/test_export_v2.py` (6 test).
- **Bug tiềm ẩn v1 phát hiện qua test**: `export_onnx` v1 với `use_postnet=True` (đúng ca B0) sinh
  **9 output, lệch tên 1 slot** (`tone_logits`←mel_pre, `lengths`←tone_logits, `lengths` thật bị đặt
  tên tự động `onnx::ReduceMax_*`) do dict output có `mel_pre`. Đã sửa bằng wrapper
  `_select_outputs_module` (loại `mel_pre` khỏi graph, giữ nguyên 8 tên/thứ tự) + test khoá
  **tên khớp giá trị** cho cả `use_postnet` True/False.
- (Đã ghi ở §21) v2 export cần `torch.no_grad()` + wrapper là **submodule** — nếu không, slice/index
  Parameter bị bake thành constant.

## 25. Task 5 (đợt 3) — trainer v2 + 2 bug kernel MPS (13/10/2026)

### 25.1 `train_v2.py` + CLI `train-v2`

`train_slot_v2(config, train, val, out, warm_start=…, max_steps=…, resume=…)` (§11):
- **Giai đoạn A** (`phase_a_steps`, mặc định 2000): đóng băng `embedding/encoder/duration`,
  LR = base × 0,25, grad clip 1.0; log mel **tách G vs hữu thanh** (`_gap_frame_mask`).
- **Giai đoạn B**: mở khóa toàn bộ → warm-up LR lên base trong `phase_b_warmup_steps` rồi cosine.
- Val `tf_seg` (d_syl/seg/gaps/lead thật); `best.pt` chọn theo **mel hữu thanh**; checkpoint ghi
  `arch`, `seg_method/version/hash` (từ manifest), `f0_stats` + hash 16 ký tự, `warm_start_from`.
- Data: `SlotDataset` + `collate_slot_batch` (batch 1); **lọc row 1 âm tiết khi train** (lý do 25.2).
- Warm-start B0: copy **193 tensor** tường minh, hash `41153e59a9b53ef9` (khớp B0_BASELINE) ✓.
- Smoke thật trên M6: 60 bước giai đoạn A — loss 12,56 → 5,33 · mel 3,16 → 0,98 ·
  mel hữu thanh 1,69 → 0,80 · **0,14 s/bước** (batch 1, ~0,2 s/bước kể cả đọc cache → 20k bước ≈ 1,5 h).

### 25.2 Hai bug kernel MPS đã bắt (torch 2.14.1, M6) + cách xử lý

1. **SDPA + dropout trên MPS lỗi khi q/k/v không `requires_grad`** — đúng ca giai đoạn A đóng băng
   (encoder chết grad): `scaled_dot_product_attention for MPS does not support dropout`.
   Cách đo: cùng tensor với `requires_grad=True` → chạy; không grad → lỗi.
   **Sửa**: `SelfAttention` dùng nhánh attention viết tay khi `training ∧ dropout>0 ∧ device=mps`
   (eval/export vẫn SDPA với dropout 0 → ONNX không đổi).
2. **`ConvPredictor` backward với N = 1 abort tiến trình** (Metal:
   `MPSNDArrayConvolution… channel mismatch`) — head `pause` của v2 + row 1 âm tiết. Đã thử
   zero-pad tường minh **cả từng lớp conv** → vẫn abort ⇒ **tránh ca đó**: `SlotDataset(min_syllables=2)`
   lọc 89 row N=1 (0,9% train); val/eval/holdout chỉ forward nên **không** ảnh hưởng.
3. Regression test: `tests/test_mps_kernel_fallbacks.py` (attention, chạy khi có MPS) +
   `test_dataset_v2.py::test_min_syllables_loc_row_n_1`. **Không** đặt ca N=1-backward vào test
   (abort giết cả tiến trình pytest) — chỉ ghi nhận trong docstring.

## 26. Task 5 cổng — run 1 + bug thang F0 → run 2 **ĐẠT cổng** (13/10/2026)

### 26.1 Kết quả chốt (run 2, 20k bước; 300 clip val + 20 clip holdout, cùng tool)

| Metric | B0 best | v2 run 1 | **v2 run 2** | Cổng §11 (≤1,2× B0) |
|---|---|---|---|---|
| `mel_l1_z` (tf_syl, nhỏ = tốt) | 1,276 | 1,179 | **1,182** | ✅ **−7,4 %** |
| `mel_l1_z` (tf_seg) | — | 1,136 | **1,140** | ✅ |
| speech mel (raw dB) | 9,79 | 8,76 | 8,78 | ✅ (−10,3 %, trong 20 %) |
| **F0 median theo thanh** (cents) | 135·129·163·223·197·154 | ~~7.729–9.195~~ | **129·107·152·153·184·150** | ✅ **tốt hơn cả 6 thanh** |
| VUV (seg / nucleus) | 0,992 / — | 0,982 | 0,982 / **0,9925** | ✅ |
| duration MAE | 51,1 ms | 50,8 | **50,8 ms** | ✅ |
| **free CER** (holdout 20) | **0,964** | 0,230 | **0,236** | ✅ **−75 %** |
| RTF free | 0,0068 | 0,0035 | **0,0027** | ✅ nhanh 2,5× |
| `slot_sum_mae_frames` (Σ slot == T) | — | 0,0 | **0,0** | ✅ bất biến B2 |

F0 mean theo thanh (tf_seg): ngang 178 · huyền 144 · sắc 207 · hỏi 198 · ngã 251 · nặng 216 cents —
các thanh khó (hỏi/ngã/nặng) đều **tốt hơn B0** (223/197/154 → 145/182/148 median).

### 26.2 Bug thang F0 (đã sửa, `64b7cf4`) — vì sao phải retrain

- `s_k` target tính trên log-F0 **đã z-score** (`f0_stats_v2`), nhưng model dựng contour thiếu biến
  đổi ngoài `f0_log_mean + f0_log_std × …` → f0 ra ≈ 1,0 log (~3 Hz) thay vì ~4,8 (~140 Hz)
  (sai ≈8.100 cents). Sửa đúng §6; sửa thêm output `energy` bị nhân đôi thang (head học raw).
- 2 test khoá: `test_f0_output_dung_thang_log_tu_nhien` (đơn vị + frame lead/G = 0) ·
  `test_energy_output_khong_nhan_lai_thang`.
- Fix đổi thang đầu vào `variance_proj` → **retrain** (`runs/acoustic_v2_slot5b`, cùng warm-start B0,
  cùng hyperparam): mel giữ nguyên (0,6514 vs 0,6508), F0 về đúng thang và **tốt hơn B0**.

### 26.3 Export ONNX với checkpoint thật (`artifacts/onnx/v2b_acoustic.onnx`)

- 0,83 s export; input `['features','speed']`; 11 output (8 tên cũ + `phonation` + `slot_durations`
  + `pause_gate` cuối); metadata `arch=v2_slot5`, `seg_method=hybrid+rule`, `f0_stats_hash=66e6809c…`
  (khớp checkpoint). Chạy thật: **byte-identical 2 lần**, `speed=0,5` → dài đúng gấp đôi, `Σ slot == T`,
  `pause_gate` học được ranh giới nghỉ (0,76–0,81) vs trong cụm (0,08–0,46).

### 26.4 Kết luận cổng

**Task 5 ĐẠT toàn bộ cổng** (mel hữu thanh · F0 theo thanh · duration · VUV · free CER · RTF · bất biến
slot). Chuyển sang T3.2 (prior mềm theo thanh + delta loss) → T3.3 (eval thanh minimal pairs) → R8/T4.1
(adversarial train-only).

## 27. T3.2 + T3.3 — prior/delta loss và chấm thanh (13/10/2026)

### 27.1 T3.2 — fine-tune run 3 (`runs/acoustic_v2_slot5c`, 20k→30k, resume từ run 2)

Loss mới: `f0_delta` (khớp **độ dốc contour** theo vị trí chuẩn hoá — độc lập độ dài/s0) +
`f0_prior` (prior mềm theo thanh, z-space, chỉ nucleus hữu thanh). Mel giữ nguyên (đúng thiết kế:
0,6397–0,6493 trong lúc train).

| Metric (300 clip val; free: 20 holdout) | v2 run 2 | **v2 run 3** |
|---|---|---|
| `mel_l1_z` (tf_syl) | 1,182 | **1,175** |
| `mel_l1_z` (tf_seg) | 1,140 | **1,130** |
| F0 median 6 thanh (cents) | 129·107·152·153·184·150 | 130·110·164·150·194·156 |
| duration MAE | 50,8 ms | 51,0 ms |
| **free CER** | 0,236 | **0,220** |
| thanh từ **f0 model** (36 pair, prior gần nhất) | 21/36 = 0,583 | **22/36 = 0,611** |
| — ngang 6/6 · huyền 3/6 · sắc 4→**5**/6 · hỏi 4→**5**/6 · **ngã 0/6** · nặng 4→3/6 | | |

→ Gain nhỏ nhưng đúng hướng (CER −6,7%; sắc/hỏi +1). **Ngã vẫn 0/6** (model thiếu độ *cong* đặc trưng
dip+rise — prior hiện nặng nhất trong bảng vẫn là chưa học được s2 của ngã; cần T4.1 hoặc tăng
trọng số prior / dùng phonation cho creak).

### 27.2 T3.3 — tool chấm thanh minimal pairs

- `tools/eval_tone_pairs.py`: 6 base × 6 thanh trong câu khung "Tôi nói ___ nhé"; hai nguồn đo:
  **`--source model`** (f0 model, chẩn đoán "ý định") và **`--source audio`** (F0 trích từ audio
  tái tạo, harvest→dio→autocorr chọn bộ nhiều hữu thanh nhất); cửa sổ **nucleus** (khớp định nghĩa
  prior; có `--window syllable` chẩn đoán); phân loại = **prior gần nhất** (cùng thống kê R5b);
  kèm **trang nghe A/B mù** (local, không upload) + answer-key JSON.
- Đo audio trên cùng 36 pair (run 2): **7–8/36 ≈ 0,19–0,22** — thấp hơn nhiều so với 0,58–0,61 khi
  đo từ f0 model ⇒ **lỗi/độ mờ nằm ở đường mel→audio**, không phải ý định của model.
- Bug đã bắt khi tích hợp: nhánh `--source model` truyền *log-F0* vào hàm nhận *Hz* (double-log →
  mọi thanh thành hỏi); đã sửa + xác nhận số liệu hợp lý.

### 27.3 Vocoder — đo độ trung thành (`tools/eval_vocoder_fidelity.py`)

| Nguồn mel → Vocos pretrained | mel_l1 (so mel thật) | voiced recall |
|---|---|---|
| **mel thật** (trần của vocoder) | **0,107** | **93,5 %** |
| mel do acoustic run 2 sinh | 1,747 (16× xa hơn) | 86,1 % |

| Vocoder (trên mel thật, 12 clip) | mel_l1 | voiced recall | F0 | RTF | params |
|---|---|---|---|---|---|
| Vocos pretrained | **0,121** | **96,1 %** | **74,8¢** | 0,007 | 13,5M |
| lite nội bộ (`runs/vocoder_codelung`, hợp đồng mel **thô** đời trước) | 2,411 | 78,7 % | 259,1¢ | **0,002** | 2,4M |

**Kết luận**: Vocos **không phải** nút thắt chính (đọc mel thật gần như trong suốt); nút thắt là
**mel do model sinh còn mềm** (1,75 vs 0,107). Thứ tự: **T4.1 adversarial trước** (làm nét mel từ gốc,
đo lại bằng harness này) → **chỉ fine-tune Vocos khi** khoảng cách model-mel↔mel-thật vẫn lớn *kèm*
recall/CER kém đi (fine-tune 2 bước: mel thật → GTA mel sinh). Không train lại vocoder nhỏ từ đầu
(lite kém ~20×, cần nhiều ngày M6; RTF hiện đã tốt 0,007 khi cần bản nhẹ sẽ tính sau).

## 28. T4.1 + điều tra "vì sao chưa rõ" (13/10/2026)

### 28.1 Run làm nét mel (5d) — làm được gì, không được gì

Loss `mel_sharp` = khớp **chênh lệch liền kề** (thời gian + tần số) fake↔real — thay cho adversarial
(đã thử: phân kỳ ở batch 1 kể cả sau spectral norm/BCE/ramp; **giữ trong code nhưng tắt**).
11k bước (27k→38k), best val mel hữu thanh **0,6368** (5c: 0,6397):

| | 5c | 5d | mốc |
|---|---|---|---|
| Gồ ghề (fake/thật) | 0,354 | **0,331** | 1,0 |
| mel_l1 ↔ mel thật | 1,334 | 1,336 | — |
| Voiced recall (mel model → Vocos) | 0,825 | **0,853** | 0,935 |

→ **Loss phía acoustic gần hết dư địa cho độ nét** (giảm 6,5% so với khoảng cách 67%).

### 28.2 Tai người (user) vs số đo — khớp nhau

| Clip demo | CER (Whisper) | Nhận xét |
|---|---|---|
| Hội thoại | **0,051** | chữ gần đúng hết; "mang máng" là do **timbre đục** |
| Câu dài | 0,180 | đa số đúng chữ |
| **6 thanh "ma…mạ"** | **0,865** | **không ngữ cảnh** ⇒ đo trực tiếp **độ rõ dấu** (điểm yếu đã biết) |

### 28.3 Điều tra giả thuyết (user: "từ hay gặp thì rõ hơn")

- Tần suất 2.567 âm tiết train: "cái" 9.050 · "là" 9.049 · **"mình" 8.737** (top 3) …
- Lỗi mel tf_seg trên 2.460 âm tiết val: hiếm (<p25) **0,617** · vừa 0,590 · hay gặp (>p75) **0,563**;
  Spearman(log-freq, lỗi) = **−0,148** → **có thật nhưng yếu** (hiếm lỗi hơn ~9,6%).
- **ĐÍNH CHÍNH (đo lại 14/10, đúng thước val trên 20 row)**: lỗi mel hữu thanh
  **train 0,557 vs val 0,612 → gap +9,7%** (đo theo *mọi frame*: 1,03 vs 1,31 = +27%, phần lớn do
  các đoạn **lặng G bị thuộc nhẹ**). Trước đó ghi "train ≈ val" là do đọc log theo batch bị nhiễu — sai.
  Chẩn đoán đúng: **overfit nhẹ đã bắt đầu ở phần macro/hữu thanh** (size small sắp tới hạn) + phần
  **texture chi tiết vẫn underfit** (gồ ghề thiếu 62–69% ngay cả trên train) ⇒ scale lên base là đúng;
  nếu base overfit nhanh thì **dữ liệu (17,1h) mới là chốt** → cần thêm video.
- Ảnh hưởng ưu tiên: **bước (chắc chắn)** · **dung lượng (cần thử)** · **đa dạng dữ liệu (nhỏ, cần thêm nguồn)**.

### 28.4 Kế hoạch acoustic-only (user chốt: **không fine-tune Vocos**)

1. **Kéo dài 38k → 60k** (`runs/acoustic_v2_slot5e`, cosine mới) — trả lời "chưa đủ bước".
2. **Medium (hidden 256, `configs/small_codelung_v2_medium.yaml`)**: đo s/bước rồi probe 1–2k bước để so
   đường loss khởi đầu với small → nếu khả quan mới chạy dài (2–3× chậm).
3. **Dữ liệu** (tuỳ chọn, cần nguồn từ user): `youtube-fetch` để tăng độ phủ; giữ gate metric như cũ.

## 29. Kinh nghiệm vận hành gom từ phiên 13/10/2026 (đọc trước khi đổi gì)

### 29.1 Batch & thông lượng train (bài học lớn nhất — bỏ sót từ đầu)

- **Batch 1 không phải "an toàn" mà là nút thắt**: `collate_slot_batch` cần **cùng N** trong batch;
  pipeline v2 chạy batch 1 suốt 56k bước → mỗi bước chỉ 1 mẫu. Đo lại:
  **batch 1 = 0,229 s/mẫu · batch 8 = 0,0375 (nhanh 6,1×) · batch 16 = 0,047 (tệ hơn 25%)**.
  → **8 là điểm ngọt đo được** (không phải quy tắc chung; đổi model/pipeline phải **đo lại**).
- Số liệu epoch (quan trọng): train **hiệu dụng 17,1 giờ** (9.674 row dùng được) — không phải 51h corpus.
  60k bước × batch 1 = **6,2 epoch**; với batch 8 = **~50 epoch**. B0 (batch 16, 20k bước) đã "thấy"
  **33 epoch** trong khi v2 mới **5,8 epoch mà vẫn thắng** ⇒ kiến trúc ổn, vấn đề là **thông lượng**.
- Cách làm: `SameNBatchSampler` (gộp index theo N, xáo trộn trong nhóm + giữa batch; val giữ batch 1
  cho metric ổn định). RAM không phải giới hạn (máy 32 GB; batch 8 dùng 3,9–7,4 GB MPS).
- Hệ quả về LR: lịch LR canh cho batch 1; khi tăng batch nên để LR như cũ trước, theo dõi rồi mới chỉnh.

### 29.2 Quy trình bắt buộc khi đổi thứ gì trong train (đã cứu nhiều lần)

1. **Probe 60–100 bước** trước khi chạy dài (đã bắt 2 lần adversarial phân kỳ, 1 lần LR ở sàn do
   `max_steps` nhỏ làm lịch cosine hết sớm — nhớ: LR phụ thuộc `max_steps`).
2. Đo **s/bước + s/mẫu + RAM MPS** ngay trong probe.
3. Chỉ chạy dài khi loss/mel ổn định; sau đó **eval đủ bộ** (mel/thanh/CER/fidelity/gồ ghề) + so mốc.
4. Mọi kết luận phải có số; kết quả trung thực kể cả khi không cải thiện (vd làm nét mel chỉ 0,354→0,331).

### 29.3 Bẫy môi trường MPS (tóm tắt; chi tiết §25 + AGENTS §7)

- SDPA + dropout trên MPS lỗi khi q/k/v không `requires_grad` (giai đoạn đóng băng) → nhánh viết tay.
- `ConvPredictor` backward với **N=1** abort tiến trình → lọc `min_syllables=2` khi train; **không**
  đặt ca N=1-backward vào test (abort giết cả pytest).
- `torch.onnx.export`: v2 cần `torch.no_grad()` + wrapper **submodule** (không closure).
- Watchdog MPS: đặt **cả hai** env `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` **và** `..._LOW_...=0.7`
  (chỉ đặt HIGH → lỗi "invalid low watermark ratio").

### 29.4 GAN/đối kháng ở batch 1 (đã thử, đã bỏ — có mã nguồn giữ lại)

- Hai lỗi logic đã bắt: `adv_start` phải **tương đối theo bước resume**; `lsgan` không bị chặn khi
  logits D phình to (→ dùng **BCE** + spectral norm + ramp + d_lr thấp).
- Kể cả vậy, vẫn phân kỳ khi trọng số adversarial ~0,5 ⇒ **shelve** ở batch 1. Ghi chú: sau khi có
  batch 8, có thể **thử lại** (D ổn định hơn khi batch lớn) — nhưng phải qua đúng quy trình §29.2.

### 29.5 Demo audio & bằng chứng nghe

- Demo qua **GitHub Release asset** (không commit vào lịch sử): release `demo-audio-20261013`
  (B0/v2c/v2d) — xoá bằng `gh release delete <tag> --yes`.
- **Tai người + CER phải khớp nhau**: hội thoại CER 0,051 (nghe "mang máng" do timbre),
  clip 6 thanh CER 0,865 (không ngữ cảnh ⇒ đo trực tiếp độ rõ dấu) — dùng làm thước cho các lần sau.

## 30. Chọn epoch/val theo SỐ ĐO (14/10/2026) — "5h" chỉ còn là tham khảo

### 30.1 Nhiễu val & early stop (bài học trực tiếp từ run b8)

- Đo nhiễu đúng thước chọn `best` (**mel hữu thanh**): std giữa clip **0,0794** →
  **SE(40 clip) = ±0,0126** (95%: ±0,0246) · **SE(300 clip) = ±0,0046** (95%: ±0,0090).
- Hệ quả đã thấy thật: val b8 nhảy `0,6267 (46k) → 0,6387 (50k) → 0,6277 (51k) → 0,6326 (53k)`
  — **dao động trong nhiễu**, "best tại 46k" KHÔNG kết luận được nếu chỉ có 40 clip.
- Đã bổ sung: `train_v2` có **early-stop patience** (lưu `no_improve` trong checkpoint; test riêng);
  config base dùng **val 300 clip + patience 5** (5 mốc = 5k bước).

### 30.2 Quy tắc ngón tay (<5h/5–30h/≥40h) = **tham khảo**, không phải luật

- Con số 5h/30h/40h trong `TRAINING_ESTIMATES.md` là **heuristic nội bộ** (cân bằng overfit↔dung lượng),
  **không** suy ra từ dữ liệu của mình và **không** liên quan ngôn ngữ. Từ 14/10/2026: **số đo thực tế
  là trọng tài**; heuristic chỉ để chọn điểm khởi đầu.
- Bộ số đo thay thế: gap theo epoch (1,2% → 5,7%), gồ ghề (0,33/1,0), âm tiết hiếm (+9,6%),
  nhiễu val (§30.1), và độ dốc val cuối schedule.

### 30.3 Cách chọn epoch (đo được, không đoán)

- **1 epoch** trên train = 9.674 row ≈ 17,1h; với batch 8 ≈ **1.209 bước/epoch**.
- Quyết định theo **epoch × val**: cứ chạy khối (cosine); val vượt nhiễu & không cải thiện `patience`
  mốc → dừng/scale. **Không** dùng train loss làm tín hiệu (nó giảm cả khi overfit).
- Bằng chứng cho small hiện tại: đạt plateau thực dụng quanh **~40–50 epoch** (gap +5,7% @38 epoch,
  val đi ngang 0,626–0,638 từ 46k); base đặt **40.000 bước ≈ 33 epoch** + early stop để tự trả lời.

### 30.4 Lọc dữ liệu (coverage ↔ độ sạch) — thiết kế ablation khi cần

- Đổi 17h → 5h sạch + ×3,4 epoch: **bớt nhoè** (phân bố hẹp) nhưng **mất phủ** (âm tiết/dấu hiếm),
  trần dung lượng vẫn giữ ⇒ dự đoán *hay hơn trên văn bản giống corpus, tệ hơn trên val đầy đủ*.
- Cách dùng đúng: **cắt đuôi tệ nhất** (5–10% clip ồn/align kém), không bỏ 12h. Nếu chạy ablation:
  chấm điểm sạch/clip (speech-frame, năng lượng vùng lặng, điểm alignment) → train bản lọc top-X%
  → so **cùng val 300 clip + gồ ghề + CER**.

## 31. Small đạt ngưỡng (run b8) → khởi động BASE (14/10/2026)

### 31.1 Kết quả run b8 (small, batch 8, 60k bước; best.pt @46k)

| Thước (300 val clip) | B0 | 5d | **b8 best (46k)** | b8 last (60k) |
|---|---|---|---|---|
| mel_l1_z mean | 1,2758 | 1,1745 | **1,1603** | 1,1627 |
| mel_l1_z median | 0,8636 | 0,7900 | 0,7934 | 0,7790 |
| mel dB hữu thanh | 9,79 | 8,72 | 8,64 | 8,64 |
| vuv (nucleus) | 0,9918 (–) | 0,9823/0,9925 | 0,9823/0,9925 | 0,9823/0,9925 |
| dur_mae (frame) | 4,79 | 4,78 | 4,75 | 4,76 |
| free CER (20 holdout) | 0,964 | 0,236 | **0,198** | – |

- best (46k) vs last (60k) chênh **0,0025 ≈ nhiễu** (§30.1) ⇒ dùng `best.pt`; val đi ngang 46k→60k
  (với patience 5 sẽ tự dừng ~51k — tiết kiệm ~45 phút mà artifact không đổi).
- ⇒ **small đạt ngưỡng thực dụng**: mel tốt hơn B0 9,1% (mean), CER 0,198 (B0 0,964).

### 31.2 Demo CER — 3 câu cố định, cùng Vocos (`artifacts/demo/`, release `demo-audio-20261013`)

| Câu | b0 | v2c | v2d | **v2b8** |
|---|---|---|---|---|
| câu dài (Xin chào các bạn…) | 0,730 | 0,169 | 0,034 | **0,022** |
| hội thoại (Dạ vâng, cái này…) | 0,747 | 0,013 | 0,013 | 0,051 |
| 6 thanh (ma má mà mả mã mạ) | – | 0,461 | 0,371 | 0,528 |

- Câu dài: **tốt nhất từ trước tới nay** (0,022); hội thoại tương đương v2d (chênh 1 chữ).
- **6 thanh vẫn yếu** (0,371→0,528 — trong nhiễu 1 clip) và khớp tai người ("gần như không hiểu").
  Thanh điệu là **đích riêng tiếp theo**: cần (a) thêm dữ liệu dấu hiếm, hoặc (b) nhánh F0 chuyên dụng —
  cả hai đều *sau* khi biết kết quả base.

### 31.3 BASE khởi động (tối 14/10)

- Probe (§29.2): batch 8 → **0,374–0,415 s/bước**, RAM MPS **7,3–8,4 GB** (32 GB thoải mái) ⇒ không hạ batch.
- Lệnh chạy: `tools/watch_train_v2.sh runs/acoustic_v2_base <log> 40000 configs/small_codelung_v2_base.yaml …`
  → **40.000 bước ≈ 33 epoch**, cosine 40k, val 300 clip (SE ±0,0046), `early_stop_patience: 5`, tự resume khi crash.
- Giả thuyết cần đo: base giảm val **> nhiễu** so small 0,6267 ⇒ dung lượng là nút thắt → cân nhắc large;
  base **overfit nhanh** (gap phình) ⇒ **dữ liệu là nút thắt** → xin thêm nguồn (17,1h hiện tại có trần).

## 32. Chẩn đoán "dùng thực tế được chưa" + dư địa cải tiến (14/10/2026, đang chạy base)

### 32.1 Base đang thắng small rõ (giữa lịch trình)

| Mốc base | val mel hữu thanh |
|---|---|
| 17k | 0,6067 |
| **18k** | **0,5777** (best) |
| 19k–25k | 0,5833–0,5968 (đi ngang/nhích nhẹ) |

- So small best **0,6267** → base **−7,8%** ⇒ **dung lượng 6,1M đúng là nút thắt macro**.
- **Bài học early-stop (§30.1) — bản sửa**: val "chững" ở 18k nhưng **LR lúc đó còn 1,2e-4**
  (45% schedule). Với lịch cosine, chững *trước* khi LR anneal xong **không** phải hội tụ ⇒
  `early_stop_patience` phải **TẮT (0)** cho run cosine dài; chỉ dùng khi LR đã cạn.
  (Đã sửa: config base patience 0; watchdog **tôn trọng `early_stop.json`** — trước đó vòng
  watchdog × patience gây "resume 1k bước/lần".)

### 32.2 ĐO độ "gồ ghề" mel: dung lượng KHÔNG sửa được texture (12 clip val, CPU)

| Nguồn mel | \|Δ_t\| (% thật) | \|Δ_f\| (% thật) |
|---|---|---|
| Thật | 0,3897 | 0,3483 |
| small b8 (6,1M) | 39,6% (mượt hơn 60%) | 31,6% (mượt hơn 68%) |
| **base 16,4M @18k** | **39,8%** (mượt hơn 60%) | **34,5%** (mượt hơn 65%) |

⇒ **Base cải thiện macro** (val −7,8%) **nhưng texture gần như không đổi**. Nghĩa là:
- "Nghe mượt/thiếu thật" **không phải** do model nhỏ; do **hàm mục tiêu** (L1 log-mel → lấy trung bình
  có điều kiện = làm mượt) + **dữ liệu** (17,1h nhiều nguồn → trung bình càng mờ), không do dung lượng.
- ⇒ Thang dư địa (đo được, không đoán):
  1. **GTA fine-tune Vocos** (2–5k bước ≈ 30–55 phút): học *đảo* độ mượt hệ thống — **đòn rẻ nhất
     trúng texture** (Vocos với mel thật chỉ 0,107/recall 93,5% ⇒ nó không phải nút thắt, nhưng
     *fine-tune trên mel model* thì bù được méo hệ thống). Cổng: fidelity harness + CER + thanh + tai.
  2. **Adversarial ở batch 8**: buộc mel *tự* có texture (batch 1 đã phân kỳ; batch 8 ổn định hơn).
  3. **Dữ liệu**: texture + dấu hiếm — nút thắt dài hạn (17,1h là ít cho TTS).
  4. **large 32M**: thêm macro/phủ, ít tác dụng texture (theo bảng trên).
  5. Tăng trọng số `mel_sharp` (knob rẻ, kỳ vọng nhỏ: 0,354→0,331 trước đây).
- Ghi chú dùng thực tế: CER free 0,198 (holdout) = đọc *hiểu được nhưng đục*; muốn "dùng được"
  cho một giọng cụ thể → **fine-tune 1 speaker** (vài giờ dữ liệu sạch) là đòn lớn nhất/giờ công.

## 33. Thanh điệu: chẩn đoán từng khâu + sửa thước đo (14/10/2026)

### 33.1 Đường đi của thanh (60 clip val ≈ 1.462 âm tiết; bộ phân loại T3.3)

| Đường | Accuracy | n (paired) |
|---|---|---|
| A: audio THẬT → harvest F0 → phân loại | 0,460 | 1106 |
| B: mel THẬT → Vocos → F0 | 0,463 | 1106 |
| C: free-run (text→model→Vocos) → F0 | **0,481** | 1106 |
| D: f0 head của model | 0,764 | 1106 |

- **B ≈ A** (Vocos + mel không làm mất thanh) và **C ≥ A** (pipeline giữ thanh *ngang* audio thật).
- ⚠️ **Trần của chính bộ phân loại trên audio thật chỉ 0,46** ⇒ các số cũ (0,19–0,22) đo trong
  *bối cảnh minimal-pair* là **hiện tượng của thước đo**, không phải bằng chứng model mất thanh.
  → Phải làm **thước mới** trước khi kết luận (`tools/tone_ruler.py`, đang xây).

### 33.2 Lỗi biểu diễn contour (đo lại bằng hàm chuẩn `dct4_frames/idct4_frames`)

Lỗi bình phương trung bình (cent) khi dựng lại contour F0 **THẬT** của âm tiết với K hệ số:

| Thanh | K=2 | K=4 | K=6 | K=8 | K=12 | n |
|---|---|---|---|---|---|---|
| NGANG | 39 | 19 | 11 | 6 | 3 | 1718 |
| HUYỀN | 26 | 13 | 7 | 5 | 2 | 1595 |
| SẮC | 46 | 23 | 14 | 8 | 4 | 1488 |
| HỎI | 70 | 34 | 19 | 12 | 4 | 457 |
| **NGÃ** | **126** | **65** | **33** | 20 | 10 | 304 |
| NẶNG | 88 | 44 | 22 | 10 | 2 | 606 |

- **Sửa lỗi của tôi**: phép đo đầu (transform tự viết) sai chuẩn hoá → lỗi không giảm khi tăng K;
  số trên dùng hàm chuẩn của repo. Ghi lại để nhớ: *đo bằng hàm của repo, không tự chế*.
- **Kết luận**: ở K=4 contour thật chỉ mất **13–65 cent**, trong khi model sai **111–197 cent**
  (§31.1) ⇒ **lỗi DỰ ĐOÁN, không phải lỗi basis**. K=6 chỉ giúp NGÃ (65→33) — knob rẻ, đáng thử sau.

### 33.3 F0 RMSE theo thanh (300 val clip; trung vị, cent — 100 cent = 1 nửa cung)

| | NGANG | HUYỀN | SẮC | HỎI | NGÃ | NẶNG |
|---|---|---|---|---|---|---|
| B0 | 135 | 129 | 163 | 223 | 197 | 154 |
| **b8 (small)** | **128** | **111** | 165 | **152** | 197 | 154 |

⇒ Đúng thứ tự khó mà tai người nghe thấy: NGANG/HUYỀN tốt nhất; **NGÃ (197) · SẮC (165) ·
NẶNG (154) · HỎI (152)** kém nhất. Small chỉ hơn B0 ở HỎI (−71), còn NGÃ/SẮC/NẶNG gần như đứng.

### 33.4 Kế hoạch sửa thanh (theo thứ tự, có cổng đo)

1. **Base** (đang chạy) → đo lại bảng §33.3 xem dung lượng có giúp NGÃ/SẮC/NẶNG không.
2. **Thước mới** `tone_ruler` (≥0,75 trên audio thật) — điều kiện để *xác nhận* mọi cải tiến thanh.
3. **Loss phân biệt thanh**: CE mềm giữa `s_k` dự đoán và prior 6 thanh (thay vì chỉ L1 có trọng số 0,1)
   → tối ưu trực tiếp *khả năng phân biệt*, không phải khớp hệ số.
4. **K=6** cho contour (stats + đầu f0) — kỳ vọng ~30 cent cho NGÃ.
5. **GTA fine-tune Vocos** (tai user xác nhận `sh15/sh25` tốt hơn rõ) — texture mel giữ dấu tốt hơn.
6. **Dữ liệu dấu hiếm**: NGÃ chỉ 304 âm tiết val (~5,4k train, 6× ít hơn NGANG) — nút thắt dài hạn.
