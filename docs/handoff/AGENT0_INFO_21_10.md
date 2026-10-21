# Agent 0 — Thông tin bổ sung (A: đọc code/config · B: đo trên máy) — 21/10/2026

> Trả lời danh sách yêu cầu A1–A5, B1–B7. Số đo trên Mac mini M6 (MPS), checkpoint
> `runs/acoustic_v32/best.pt` (v3.2 small, bước 59k) trừ chỗ ghi khác. Ký hiệu: **[CODE]** = đọc
> trực tiếp source, **[ĐO]** = đo trên máy, **[CHƯA]** = chưa làm / cần thêm.

---

## A1. Decoder + postnet

**[CODE]** Encoder và decoder **cùng một lớp** `ConformerLite` (`models/conformer.py`):
- Mỗi block: FFN(thang 0.5) → self-attention → depthwise Conv1d(kernel **15**) + pointwise → FFN(0.5) → LayerNorm.
- Attention: **global** (additive mask, không giới hạn cửa sổ); PE hình sin cố định theo **chỉ số frame tuyệt đối** (max 4096).
- small: encoder 4 block, decoder 2 block, hidden 192, 4 head (head_dim 48), ffn 768, dropout 0.1.
- base: 6/4 block, hidden 256, ffn 1024, component_dim 96.

**Trải slot → frame** (`v2_slot5.py`, `LengthRegulator` + `slot_positions`):
- Mỗi âm tiết → **5 slot** `[O, M, N, C, G]`; mỗi slot có vector hidden riêng = `MLP(concat(h_syl, role_emb, emb_thành phần))`;
  slot G = `Linear(boundary_emb, punct_emb) + silence_emb` (không phụ thuộc âm tiết).
- Frame = hidden của slot được **lặp theo độ dài slot** (không nội suy), cộng thêm:
  `position_proj(vị trí trong slot, chuẩn hoá 0..1)` + `variance_proj(f0, vuv, energy[, phonation])`.
- Có **2 loại vị trí**: PE tuyệt đối theo frame (trong ConformerLite) **và** vị trí tương đối trong slot.
  ⇒ Vị trí frame không bị "nội suy" như FastSpeech; mỗi frame biết mình ở đâu trong slot.

**Slot N có thấy coda?** **[CODE]** Có. `segment_in` của mỗi slot = `h_syl` (encoder trên **cả 7 thành phần** gồm coda)
+ role embedding + embedding của thành phần tương ứng. Slot N dùng `h_syl` đầy đủ ⇒ thông tin coda có mặt
(qua encoder + role). Không có đường tắt cấm.

**Postnet**: 4×(Conv1d 256, kernel **5**, LeakyReLU) + Conv1d→100, **cộng residual** vào mel_pre
(`mel = mel_pre + postnet(mel_pre)`). Receptive field postnet ≈ **17 frame (~180 ms)**; decoder
Conformer ≈ global (attention) + local ±7 frame/lớp.

**L1 đặt ở đâu**: cả **mel sau postnet** (trọng số 1.0) **và mel_pre** (0.2). **[CODE]**

**Dropout**: encoder/decoder/MLP đều 0.1; **tắt khi eval** (`model.eval()`), inference không ảnh hưởng.

**Bảng tham số (small 7,05M)** **[ĐO]**:
| module | tham số | % |
|---|---|---|
| encoder | 3,128M | 44,4% |
| decoder | 1,564M | 22,2% |
| postnet | 1,240M | 17,6% |
| pause | 334k | 4,7% |
| duration | 334k | 4,7% |
| slot_expander | 127k | 1,8% |
| variance/ratio/embedding | ~272k | 3,9% |
| còn lại | ~50k | 0,7% |

## A2. Mel và loss

**[CODE]** Mel: `log-magnitude`, floor **1e-7** kiểu **clamp** (trùng `safe_log` của Vocos), `n_fft 1024`,
`win 1024`, **hop 256 (10,67 ms)**, 100 bin, `f_min 0`, `f_max 12000`, 24 kHz, `center=True`.
Chuẩn hoá: **z-score toàn cục per-bin** (mean/std 100 phần tử trong `mel_stats_v2.json`, tính trên 2.000 clip).
Loss L1 tính **trong không gian z** (không phải log thô/dB).

**Danh sách loss + trọng số** (config v3.2; v3.4 **thêm** `mel_onset_boost 2.0`, `mel_coda_boost 3.0`,
mặc định 0 = hành vi cũ):
```
mel 1.0 · mel_pre 0.2 · mel_sharp 1.0 (khớp |Δ| thời gian+tần số của mel)
duration 1.0 · ratio 1.0 · f0 1.0 · f0_delta 0.5 · f0_prior 0.1 · tone_disc 0.0 (tắt)
vuv 0.5 · energy 0.5 · phonation 0.5 · tone 0.1 · pause_gate 0.5 · pause_gap 0.5
```

**Mask frame lặng có tính vào mel loss?** **[CODE]** **Có.** `frame_lengths` phủ toàn row (lead + âm tiết + gap);
mel L1 lấy trung bình trên **mọi frame hợp lệ**, kể cả lặng. Tách `mel_gap` / `mel_voiced` chỉ để **log**.
(v3.4 chỉ tăng trọng số frame onset/coda, không loại lặng.)

## A3. Frontend và vần NH/CH

**[ĐO]** Parse 12 từ (dialect north), mã 7 thành phần:
| từ | O | M | N | C | T | ids |
|---|---|---|---|---|---|---|
| xanh | X | NONE | A | **NH** | NGANG | 12,0,0,4,0,1,0 |
| xan | X | NONE | A | **N** | NGANG | 12,0,0,2,0,1,0 |
| lạnh | L | NONE | A | **NH** | NANG | 17,0,0,4,5,1,0 |
| lạn | L | NONE | A | **N** | NANG | 17,0,0,2,5,1,0 |
| quanh | K | W | A | **NH** | NGANG | 18,1,0,4,0,1,0 |
| quan | K | W | A | **N** | NGANG | 18,1,0,2,0,1,0 |
| ếch | NONE | NONE | Ê | **CH** | SAC | 0,0,4,8,2,1,0 |
| ết | NONE | NONE | Ê | **T** | SAC | 0,0,4,6,2,1,0 |
| inh | NONE | NONE | I | **NH** | NGANG | 0,0,5,4,0,1,0 |
| in | NONE | NONE | I | **N** | NGANG | 0,0,5,2,0,1,0 |
| ách | NONE | NONE | A | **CH** | SAC | 0,0,0,8,2,1,0 |
| át | NONE | NONE | A | **T** | SAC | 0,0,0,6,2,1,0 |

- **Nucleus của "anh" = "an"** (cùng A) — khác nhau chỉ ở coda NH(4) vs N(2). ⇒ Không có "nguyên âm mũi" riêng;
  tương phản nằm hoàn toàn ở coda.
- **MFA dictionary**: `data/codelung/lexicon.dict` (dùng cho align) có **một biến thể**: `xanh … Ox NaNgang Cnh`,
  `ách … NaSac Cch`. Nhưng `data/codelung/vn_models/vietnamese_mfa.dict` có **nhiều biến thể**, trong đó có
  biến thể **"nh → n"** (`lạnh … l a˨˩˨ n`, `xanh … s a˨˨ n`) bên cạnh "ɲ". ⚠️ Nếu align dùng dict này, một số
  từ "nh" có thể được align bằng phone **n** → biên coda lệch nhẹ. **[CHƯA]** xác nhận dict nào đã dùng
  cho lần align corpus hiện tại.
- MFA model: `data/codelung/vn_models/vietnamese_mfa_g2p.zip` (G2P tiếng Việt). Accent: chưa xác nhận
  trong code; corpus là giọng Bắc.

## A4. Lịch sử adversarial T4.1 (3 lần phân kỳ)

**[CODE]** Config hiện tại (`adversarial`, đang **tắt**):
- D = `MultiResolutionDiscriminator` (mel-domain), **spectral norm**, 3 scale `[1,2,4]`, 64 kênh, 3 lớp.
- Loss **BCE** (ghi chú: `lsgan` đã gây phân kỳ vì logits D phình to).
- `d_learning_rate 1e-4` (đã hạ từ 2e-4 sau lần phân kỳ đầu), `grad_clip_d 5.0`,
  `d_warmup_steps 1000`, `ramp_steps 2000` (adv 0→1), `mel_adv_weight 1.0`, `feature_matching_weight 1.0`.
- D **tách khỏi checkpoint/ONNX** (train-only).

**[CHƯA]** Chi tiết 3 lần phân kỳ (log D-loss/G-loss từng lần, giá trị đỉnh) — nằm trong log cũ,
chưa trích ra. Đã biết: cả 3 lần phân kỳ **ở probe tới ~7,7e6**, trước khi vào run chính.

## A5. Train và runtime

**[CODE]** AdamW (`weight_decay 1e-4`), lr **2e-4**, phase A 2000 bước (đóng băng embedding/encoder/duration,
lr 0,25×), phase B warmup 2000 → cosine tới 0. **Không EMA**. **fp32** (không autocast/GradScaler trên MPS).
Batch **8 segment** (`SameNBatchSampler` gộp cùng N), 60k bước.
ONNX: TorchScript exporter, `dynamo=False`, opset **18**; **acoustic và vocoder là 2 file riêng**
(`synthesize-onnx` nhận 2 đường dẫn). **[CHƯA]** đo lại RTF ONNX; số cũ trong docs: **RTF ≈ 0,08**
(CPU 2 luồng, acoustic+vocoder+iSTFT). **[ĐO]** **tất định PyTorch**: cùng input, 3 lần chạy → md5 giống hệt
(`d1a72ec6f6c4`).

---

## B1. Tách nguồn rè — cùng Vocos, 3 nguồn mel (20 segment val)

**[ĐO]** (α làm nét 1,5 như production):
| nguồn mel | HNR | AP 2–6 kHz |
|---|---|---|
| **copy** (mel THẬT → Vocos) | 4,70 dB | **0,5927** |
| **teacher-forced** (model, đúng duration) | 5,05 dB | **0,5531** |
| **free-run** (model tự dự đoán duration) | 5,02 dB | **0,5547** |
| mel L1 (TF vs thật, z) | — | 1,473 |

**Đọc kết quả (quan trọng, hơi bất ngờ):**
1. **(2) ≈ (3)** ⇒ rè/khác biệt **không đến từ duration** — đúng giả thuyết.
2. **Copy có AP 2–6 kHz CAO HƠN cả TF/free** (0,593 vs 0,553) ⇒ **chính Vocos thêm bất tuần hoàn**
   (~+0,07 so với audio thật 0,525 đo trước đó) — nhưng tai người nghe copy là **sạch**.
   ⇒ **AP 2–6 kHz không phải proxy đáng tin một mình** cho cảm nhận "rè"; nó bị ảnh hưởng bởi nội dung/level.
3. Vì vậy cần **thêm thước**: GV ratio (B2) + flow/spectrogram + tai.

## B2. GV ratio theo bin (teacher-forced)

**[ĐO]** `var(pred)/var(real)` trên 20 segment, từng bin: **median 0,333** (min 0,182 · max 0,407).
⇒ Mel model **co lại phương sai còn 1/3** — over-smoothing định lượng rõ. CSV:
`artifacts/eval_acoustic/gv_ratio.csv` (100 dòng `bin,gv_ratio`).
**[CHƯA]** chú thích bin nào ứng 2–6 kHz (cần đọc filterbank center; xấp xỉ tuyến tính bin 17–50, GV
trung bình dải này ≈ 0,30).

## B3. Spectrogram

**[ĐO]** 3 ảnh PNG (`artifacts/eval_acoustic/spectrograms/seg00..02.png`), mỗi ảnh 3 hàng
(real / TF / free) × 2 cột (mel đầy đủ · zoom 2–6 kHz xấp xỉ). Matplotlib đã cài vào `.venv`.

## B4. Coda NH/N — độ dài slot (nhãn vs dự đoán free-run)

**[ĐO]** 20 segment:
| coda | n | slot C nhãn (median) | slot C dự đoán (median) |
|---|---|---|---|
| N | 36 | 5 | 4 |
| NG | 15 | 4 | 4 |
| NH | 24 | 5 | 4 |
| M | 6 | 6 | 5 |
⇒ Độ dài coda dự đoán ≈ nhãn (ngắn hơn ~1 frame) — **không phải lỗi độ dài**; vấn đề là **nội dung phổ**
(centroid ngược chiều, đo trước: xanh 395 Hz < xan 450 Hz).
**[CHƯA]** quỹ đạo F2 trong 50% cuối nguyên âm (cần tracker formant — LPC/cheaptrick).

## B5. Loss components tại best checkpoint (v3.2, bước 60k — log `comps`)

**[ĐO]** giá trị thô (× trọng số): mel **0,709**×1 · mel_pre 0,756×0,2 · mel_sharp 0,565×1 ·
duration 0,073×1 · ratio 0,883×1 · f0 0,477×1 · **f0_delta 5,466×0,5 = 2,73** · f0_prior 0,313×0,1 ·
vuv 0,070×0,5 · energy 0,705×0,5 · phonation 0,477×0,5 · tone 0,002×0,1 · pause_gate 0,195×0,5 ·
pause_gap 0,095×0,5 · tone_disc 1,605×**0** (tắt).
⚠️ **f0_delta là số hạng lớn nhất** (2,73 sau trọng số, gấp ~4 lần mel) — có thể làm model ưu tiên
contour f0 hơn chi tiết mel. Cần xem có nên giảm trọng số này không.
Curve train/val đầy đủ: `runs/acoustic_v32/history.csv` (val tốt nhất 59k = 0,5974; train mel cuối 0,70).

## B6. Runtime

- **[ĐO]** Tất định PyTorch: 3 lần cùng input → md5 giống hệt (mục A5).
- **[CHƯA]** RTF ONNX hiện tại (chưa export lại trong lượt này); số cũ **[ĐO cũ] ≈ 0,08** (CPU 2 luồng,
  acoustic+vocoder+iSTFT, 24 kHz). Cần đo lại tách phần encoder/decoder/vocoder khi có export.
- Ngưỡng chấp nhận: RTF ≤ 0,3 cho realtime 1 luồng (mục tiêu sản phẩm), ≤ 0,1 là tốt.

## B7. Nguồn dữ liệu — **[CHƯA], cần người dùng trả lời]**

1. Audio gốc: file **OBS gốc** hay tải lại từ YouTube? Nếu tải: codec/bitrate (Opus/AAC, 128k/256k)?
2. OBS noise suppression: **RNNoise / Speex / NVIDIA Noise Removal**? Bật cho **cả 25 video**?
3. Transcript: **gõ tay** hay **ASR (Whisper?)**; ước lượng tỉ lệ lỗi chính tả/dấu?
4. Có bản thu **không qua OBS** (raw) cho bất kỳ video nào không? (để so OBS vs raw như đã bàn)

---

## Tóm tắt cho Agent 0 — 3 điểm mới quan trọng nhất

1. **GV ratio 0,333**: mel model co phương sai còn 1/3 — over-smoothing là định lượng được; **AP 2–6 kHz
   không đủ tin** (copy của mel thật còn AP cao hơn). Cần thước bổ sung (GV, flow, tai).
2. **Coda: độ dài đúng, nội dung sai** (centroid ngược chiều) ⇒ hướng sửa phải nhắm **phổ/formant**,
   không phải duration.
3. **f0_delta chiếm ưu thế loss** (2,73 sau trọng số vs mel 0,71) — nghi vấn cân bằng loss, cần ý kiến.
