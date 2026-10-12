# REBUILD_PLAN_M6 — Kế hoạch dựng lại để training ổn định trên Mac mini M6

> **Bối cảnh:** R5-1 vừa phát hiện val/train lệch vượt ngưỡng; truy nguyên cho thấy dữ liệu **có giọng
> khách mời** và **chênh loudness lớn**. Chủ dự án chốt: *không tiếc dữ liệu/checkpoint cũ, ưu tiên
> kiến trúc + quy trình ổn định nhất cho Mac mini M6, được phép dựng lại từ đầu.*
> Tài liệu này là đề xuất tổng thể; các bước đánh dấu **[CẦN CHỐT]** là chỗ cần bạn/chuyên gia quyết.

---

## 1. Bằng chứng: vì sao phải sửa dữ liệu trước khi bàn kiến trúc

### 1.1. R5-1 — val 14 video nhỏ vs train 17 video còn lại (all clips của video)

| Chỉ số | val | train | Lệch | Ngưỡng | Kết |
| --- | ---: | ---: | ---: | ---: | :---: |
| Nhịp đọc (âm tiết/s) | 2,79 | 2,70 | +3,5% | 10% | ✓ |
| RMS trung vị (dBFS) | −21,5 | −26,8 | **+5,27 dB** | 3 dB | ✗ |
| F0 trung vị (frame hữu thanh) | 139,0 Hz | 126,9 Hz | **+1,58 semitone** | 1,0 | ✗ |
| Tỉ lệ clip fallback | 29,8% | 29,1% | ~0 | — | ✓ |

→ **Không lock N=14 theo cách chọn "video nhỏ nhất"**. Phương án dự phòng "12 nhỏ + 2 cỡ trung" của
review cũng không giải quyết gốc vì nguyên nhân không nằm ở kích thước video.

### 1.2. Nguyên nhân thật: dữ liệu ĐA GIỌNG + loudness không đồng nhất
Quét F0 trung vị cấp clip (từ cache, 11.424 clip):

| Nhóm | Video | Đặc điểm |
| --- | --- | --- |
| **Giọng khách ~99–100% clip** | `LY8_RaLDJT4` (159 Hz) · `W1m6birJBj0` (162) · `pLXBrJ-M-Xo` (187) · `_U4H5FSIb-c` (219) | 600 clip · 2,54h — loạt "Học Từ Thiện" có khách |
| **Trộn 20–25% clip lạ** | `50swDIypplU` (IoT, 21%) · `o8RjAmBtyZo` (22%) · `6GXw85XfiKo` (25%) | ~190 clip |
| **Lẻ tẻ 1–5%** | `fx303iiDb7M` 5% · `VGYWEeV2CzM` 2–3% · vài video 1% | ~100 clip |
| **Chủ kênh (chuẩn)** | 23 video còn lại: F0 trung vị **121–137 Hz** | dải hẹp, rất nhất quán |

Thêm nữa, **loudness lệch tới ~22 dB ngay trong nhóm chủ kênh** (RMS trung vị từ −16,6 dB ở
`X_65x81R4KY`/`4ITcvEa_Acc` đến −38,8 dB ở `5KnS0TGxWLg`; các video "Vibe Coding/VPS" thu bằng
thiết bị khác). Với TTS một giọng, đây là hai nguồn nhiễu hệ thống:
- Giọng khách → model học giọng trộn, F0/energy/duration bị "pha"; val/test đo sai đối tượng.
- Loudness lệch → target energy/mel lệch theo video; decoder conditioning bị nhiễu.

**Kết luận:** trước khi bàn base/large, phải làm **lọc người nói + chuẩn hoá loudness**. May mắn là
chưa có gì đáng tiếc: MFA chưa train, duration vẫn placeholder, checkpoint cũ chỉ để test pipeline.
→ Chủ dự án đã chốt: **bỏ hẳn loạt "Học Từ Thiện"**; phần clip lẻ "giọng khác" đã được kiểm tra
**tự động** bằng NVIDIA Nemotron 3 Diarization — kết quả và danh sách loại: `docs/R6_DIARIZATION_RESULTS.md`.

---

## 2. Kiến trúc — giữ FastSpeech2-NAR + ConformerLite, nâng cấp "ổn định"

### 2.1. Vì sao KHÔNG đổi họ kiến trúc
| Phương án | Đánh giá cho dự án này (1 giọng, tất định, ONNX, M6) |
| --- | --- |
| **FS2-NAR + ConformerLite (giữ)** | Tất định, export ONNX đã kiểm, suy luận 1 lượt, hợp dữ liệu 30–50h. ✅ |
| VITS (flow + GAN, ngẫu nhiên) | Không tất định, ONNX khó, train nặng — loại. |
| Matcha-TTS (flow matching) | Có thể tất định bằng ODE vài bước, nhưng phức tạp hơn, không cần thiết cho mục tiêu hiện tại. |
| AR codec-LM (VALL-E…) | Cần dữ liệu cực lớn, một giọng không hợp — loại. |
| Diffusion mel | Chậm suy luận/ONNX không gọn — loại. |

**Ổn định không đến từ việc đổi kiến trúc**, mà từ: (1) dữ liệu sạch một giọng + loudness chuẩn,
(2) cắt segment ngắn khi train, (3) nhãn duration thật (MFA) + SIL, (4) ngân sách bộ nhớ MPS cố định.

### 2.2. Danh sách nâng cấp kiến trúc sẽ dựng (đã có đặc tả từ các vòng trước)
1. **SIL token** (3 slot: onset SIL, nucleus SIL, tone NONE; vocab 71) chèn tất định từ text.
2. **F0 log + z-score + DCT 4 hệ số** (mỗi âm tiết), hệ số z-score riêng; decoder dựng contour bằng
   IDCT `(t+0.5)/n`; mask theo tỉ lệ hữu thanh; test export ONNX đã xác nhận khả thi.
3. **Energy z-score** (scalar/âm tiết), target SIL 0-frame bị mask.
4. **`duration_scale` là input** của graph (speed chính xác cho cả PyTorch lẫn ONNX; bỏ `_stretch_mel`).
5. **Postnet mel** giữ bật; tone loss = 0 (thay bằng classifier đóng băng sau).
6. **Đóng băng biểu diễn**: `mel_hash` + `f0_stats_hash` trong checkpoint/metadata ONNX (stats bake
   vào buffer — graph tự chứa).
7. **Vocoder**: giữ Vocos-hybrid (ConvNeXt ×8, hidden 384) + MPD đầy đủ + MRD + MR-STFT, ch=16,
   crop 64 — cấu hình đã đo 0,659 s/bước MPS.

### 2.3. Điểm MỚI quan trọng nhất cho "ổn định trên M6": **train trên segment 3–10s cắt theo SIL**
Hiện tại clip dài 14–20s làm batch 16 ≈ 29k frame → đỉnh MPS ~25 GB, phải chunk 5k + watermark +
val cap mới không OOM. Đó là nguồn bất ổn số 1 khi train dài.
- Cắt segment tại SIL theo ranh giới câu (regex đã có + MFA xác nhận), độ dài mục tiêu 3–10s.
- Ước tính bộ nhớ: base (hidden 256, dec 4) với segment ~6s (≈560 frame) × batch 8 ≈ 4,5k frame/bước
  → **bộ nhớ giảm ~4–6×** so với hiện tại (≈8–14 GB driver), không cần watermark gắt.
- Batching theo **ngân sách frame** (bucket theo độ dài, tổng ~10–14k frame/bước) thay vì batch cố định.
- Lợi ích kèm: s/bước thấp hơn, phản hồi nhanh, đỡ overfit ngữ điệu dài, val ổn định.

---

## 3. Kế hoạch dữ liệu mới (làm trước, CPU)

```text
raw/<video>.wav + subs  ──(1) recut/filter──▶ clips một giọng
        ──(2) speaker filter──▶ keep/drop + report
        ──(3) loudness normalize──▶ WAV đồng nhất
        ──(4) split theo video + lock──▶ train/val/test (+holdout)
        ──(5) MFA (CPU) ──▶ TextGrid
        ──(6) mfa_to_durations + SIL + assert──▶ durations cấp âm tiết
        ──(7) cắt segment 3–10s tại SIL ──▶ train/val segments
        ──(8) rebuild cache theo config mới
```

**(1)–(2) Lọc giọng [ĐÃ CHỐT: bỏ hẳn loạt "Học Từ Thiện"]**
- Bỏ 6 video: `KIrbA-wEURg` · `LY8_RaLDJT4` · `W1m6birJBj0` · `_U4H5FSIb-c` · `nEPr-jIokLw` ·
  `pLXBrJ-M-Xo` — tổng **937 clip · 4,05h** — khỏi cả train lẫn val, lưu vào `excluded_speakers.jsonl`
  (giữ file để sau này có thể làm multi-speaker).
- **Pool còn lại: 25 video · 12.278 clip · 47,68h** (chủ kênh). Lưu ý `UJ_AOlM3cr0` ("Học làm web —
  Dự Án Học Từ Thiện") **vẫn giữ** vì là video chủ kênh.
- **Lọc âm thanh không phải giọng chủ kênh — TỰ ĐỘNG HOÁ bằng NVIDIA Nemotron 3 Diarization**
  (chi tiết: `docs/R6_DIARIZATION_RESULTS.md`; tool: `tools/diarize_filter.py`): quy trình 2 lượt —
  anchor ~6s giọng bạn ghép trước mỗi clip (spk0 = bạn), lượt A chọn anchor **theo từng video**
  (bắt buộc: anchor từ video khác mic/phòng báo nhầm 134 clip), lượt B cho kết quả cuối.
  Kết quả: **0,090h** tiếng lạ trong pool (0,19%); **loại 48 clip** (≥0,5s)
  (`data/codelung/review/diar/drop_other_ge_0p5.txt`) → **12.230 clip · 47,48h sạch giọng bạn**.
  Kiểm chứng: anchor đúng spk0 100% clip, không clip nào vượt cửa sổ model, chạy lại tái lập tuyệt đối.
  Các clip F0 cao 163–173 Hz là **giọng bạn** (model xác nhận, không lọc oan).

**(3) Chuẩn hoá loudness:** chuẩn hoá từng clip về **RMS mục tiêu −26 dBFS** (peak-safe, giữ fade);
hoặc tối thiểu là gain theo video. Ghi lại gain vào metadata. Sau bước này energy target mới có phân
bố nhất quán giữa các video.

**(4) Split mới:** sau khi lọc, chạy lại kiểm kiểu R5-1 (F0/RMS/nhịp) rồi mới lock. Phương án trên
**25 video chủ kênh** (val = các video ít giờ nhất, cap theo clip):

| N video val | cap | val | holdout | train | train sau bỏ fallback |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 100 | 1.560 clip/6,52h | 3.273/13,03h | 7.445/28,13h | 19,54h |
| 14 | 100 | 1.360/5,67h | 2.517/10,02h | 8.401/32,00h | 22,02h |
| **12** | **150** | **1.676/6,87h** | 1.328/5,17h | 9.274/35,65h | 24,42h |
| 10 | 150 | 1.376/5,64h | 845/3,33h | 10.057/38,72h | 26,36h |
| 8 | 200 | ~1.280/5,3h | ~280/1,2h | ~10.930/41,5h | ~28,6h |

- Ngưỡng 40h cũ (heuristic) **xung đột** với việc giữ ≥10 video val sau khi bỏ series: muốn ≥40h
  thì val chỉ còn ~8 video. Đề xuất **bỏ ngưỡng 40h** và chọn **N=12, cap 150** (val 6,87h — thừa
  cho 640 batch val + 200 clip eval; train 35,65h; vẫn 12 nguồn video). Nếu muốn giữ 40h → N=8/cap200.
- test/holdout: giữ đúng vai trò đã chốt (val cho tone classifier + test cuối, chạy 1 lần);
  `split_lock.json` + `eval_clips_v1.json` (200 clip sạch fallback, phân tầng theo video, hash text).

**(5)–(7) MFA + SIL + segments:** giữ nguyên toàn bộ đặc tả đã chốt ở các vòng trước
(converter TextGrid→durations, quy đổi biên tích luỹ + assert, SIL theo dấu câu, ngừng thở ≥150ms →
loại clip, mask loss…). Chạy MFA **sau khi lọc giọng** để không mất công align clip khách.

**(8) Cache:** dựng lại theo config mới; cache key hiện đã tách khỏi cấu hình F0-normalize nên ổn định.

---

## 4. Cấu hình training đề xuất cho M6 (sau khi có dữ liệu mới)

| Hạng mục | Chốt đề xuất | Ghi chú |
| --- | --- | --- |
| Model v1 | `base` 16,4M (6 enc/4 dec, hidden 256) + postnet | dữ liệu ~30h+ đủ; small chỉ để smoke |
| Smoke | `small` 20k bước, segment 3–6s | xác nhận alignment + SIL + DCT trước khi dồn máy |
| Chính | `base` 100–160k bước, segment 3–10s | early stop theo metric audio-level |
| Batching | ngân sách ~10–14k frame/bước, bucket theo độ dài | thay batch cố định 16 |
| Thiết bị | MPS fp32 (không AMP); vocoder MPS khi MPD/MRD | như đã đo |
| LR | warmup 6–15k, cosine → 5% base | đã có `lr_schedule: cosine` |
| Grad clip | acoustic 1.0 · GAN 5.0 | như đã chốt |
| Checkpoint | chính sách "sạch" rev.6 + `mel_hash` + `f0_stats_hash` + `split_lock` | đã có 1 phần |
| Val | mỗi 1–2k bước, cap batch; metric audio-level mỗi 10k | xem §5 |
| Bộ nhớ | mục tiêu đỉnh driver ≤ 20 GB | dư an toàn cho macOS |
| Đa tiến trình | chỉ 1 train MPS + MFA CPU song song | không chạy 2 job MPS |

**Ước lượng thời gian (để lên kế hoạch, sẽ chốt bằng đo thật):** base + segment ≈ 0,25–0,5 s/bước
⇒ 100k ≈ 7–14h; vocoder 100k ≈ 19–20h (không đổi). Tổng vòng mới ≈ 1,5–2 ngày máy.

## 5. Metric & cổng nghiệm thu (không đổi so với đã chốt, thêm mốc dữ liệu)

- **Cổng dữ liệu:** lọc giọng xong, R5-1 kiểu mới đạt: F0 lệch <1 semitone, RMS <3 dB, nhịp <10%.
- **Metric model:** mel L1/bin · duration MAE (ms, bỏ SIL) · tỉ lệ 1-frame · log-F0 RMSE (frame hữu
  thanh, tách 6 thanh) · VUV error · tone acc (classifier đóng băng) · CER (PhoWhisper/mlx-whisper,
  100 câu tự viết) · SIL duration MAE.
- **Cổng smoke:** các metric audio-level tốt hơn rõ so với baseline hiện tại (đã có baseline
  mel L1 0,401 · duration 55,8 frame/âm tiết · mel std 2,78 trong `docs/demos/`).
- **Cổng base:** CER ≤ copy-synthesis +2–3 điểm; nghe 50–100 câu đạt.

## 6. Việc cần chốt trước khi thực thi
1. ~~Lọc clip lẻ giọng khác~~ ✅ **đã xong tự động** (Nemotron 3 Diarization, 2 lượt anchor theo video — `docs/R6_DIARIZATION_RESULTS.md`); chốt loại **48 clip** như §3.
2. **Chuẩn hoá loudness:** clip→−26 dBFS (khuyến nghị) hay chỉ gain theo video? Có cho phép ghi đè WAV không?
3. **Quy mô dựng lại:** recut lại từ `raw/` (sạch nhất, dùng overrides + QC mới) hay chỉ lọc/normalize trên clip hiện có (nhanh hơn ~1 ngày)?
4. **Split:** chốt **N=12/cap150** (khuyến nghị — bỏ ngưỡng 40h) hay **N=8/cap200** (giữ 40h)?
5. **Speaker-conditioning tương lai:** giữ mặc định *không* (một checkpoint = một giọng)?

> **Cập nhật 12/10 (đã thực thi):** mục 2–4 đã chốt & chạy — loudness **từng clip → −26 dBFS**
> (hết lệch 22 dB), split **N=12/cap150**, lọc 6 video khách + 48 clip diarization, dựng
> `train_v2/val_v2/holdout_v2(+clean)`, `eval_clips_v1` (200 clip), corpus **MFA v2** (8.723 clip,
> 0 fallback) và stats mel mới. Chi tiết + file: **`docs/DATASET_V2.md`**. MFA train+align đang chạy nền.

## 7. Tóm tắt một câu
Kiến trúc FS2+Conformer giữ nguyên và đã đúng hướng; vấn đề ổn định nằm ở **dữ liệu đa giọng +
loudness lệch + clip quá dài**. Làm lại dữ liệu (lọc giọng → chuẩn loudness → split khớp phân bố →
MFA/SIL → segment 3–10s) rồi train base với batching theo frame là con đường ổn định nhất cho M6 —
và vì chưa có gì đáng tiếc, đây là thời điểm rẻ nhất để làm.
