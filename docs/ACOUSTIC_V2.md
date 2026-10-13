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
