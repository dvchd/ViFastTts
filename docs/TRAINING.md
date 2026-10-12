# Quy trình huấn luyện

> Luồng chi tiết as-built (tensor shape, loss, train vs inference): `docs/MODEL_FLOW.md`.
> Ước lượng **dữ liệu / tham số / thời gian / máy / chi phí** trước khi bắt tay: `docs/TRAINING_ESTIMATES.md`.

## Trạng thái as-built

`vifasttts train` hiện chạy được end-to-end:

1. Đọc manifest JSONL (`audio_path` + `syllables` + tùy chọn `durations`).
2. `data/features.py` trích **mel**, **log-energy**, **F0 và VUV** từ WAV tại chỗ; mel mặc định là
   log-power như cũ, cấu hình mới (`mel.log_magnitude` + `mel.normalize_path` + `f0.method: pyworld`)
   dùng log-magnitude chuẩn hoá toàn cục + pyworld harvest/stonemask (xem `docs/REVISION_PLAN_V2.md` §2).
   Đọc WAV bằng `soundfile` nếu có, fallback stdlib `wave`.
3. `data/dataset.py` mã hoá âm tiết thành 7 ID, gom target cấp âm tiết (f0/vuv/energy) theo duration, collate có padding + mask.
4. Train với teacher forcing: decoder mở rộng theo **duration thật** để sinh mel đúng độ dài; duration/f0/vuv/energy/tone dự đoán từ syllable state (chuẩn FastSpeech 2).
5. Mask được áp cho encoder, decoder và mọi loss (không tính vào vùng pad).
6. Lưu `best.pt`, `last.pt` (kèm config, step **và optimizer**), `history.csv`; có warmup LR,
   grad clip, AMP (CUDA **và MPS**); log in kèm tốc độ `(x.xxx s/bước)`. **Resume**:
   `--resume runs/.../last.pt` tiếp tục đúng bước/optimizer (job dài không mất tiến độ).
   LR có 2 chế độ: `training.lr_schedule: constant` (mặc định) hoặc `cosine`
   (`min_lr_ratio`, `lr_decay_steps` — giảm dần về 5% base_lr, giúp thoát vùng val chững).
   **Early stop**: `training.early_stop_patience: N` — dừng khi val không cải thiện N mốc
   liên tiếp, ghi `early_stop.json` (chunked runner đọc file này để dừng vòng lặp khối).
7. (Tùy chọn) Cache feature mel/f0/vuv theo hash(audio, config): đặt `training.cache_features: true` hoặc `cache_dir`, hoặc chạy trước `vifasttts cache-features --workers 8` (ProcessPool, nhanh hơn nhiều cho dataset chục giờ). (Tùy chọn) Bật `phonation.enabled: true` để trích thêm HNR/spectral tilt/glottalization cấp frame cho nghiên cứu chất giọng.

```bash
vifasttts train --config configs/base_24k.yaml \
  --train-manifest data/train.jsonl --val-manifest data/val.jsonl \
  --output runs/base --max-steps 5000

# Chạy tiếp nếu bị ngắt (resume từ last.pt, giữ nguyên optimizer + số bước)
vifasttts train --config configs/base_24k.yaml \
  --train-manifest data/train.jsonl --val-manifest data/val.jsonl \
  --output runs/base --resume runs/base/last.pt

vifasttts cache-features --manifest data/train.jsonl \
  --config configs/base_24k.yaml --cache-dir cache/train --workers 8
```

Cache khoá theo `(đường dẫn audio, size, mtime, mel config)` nên đổi dữ liệu/cấu hình sẽ tự
trích lại. Bật phonation thì nhớ tính lại cache.

## Cảnh báo alignment

- Nếu manifest không có trường `durations`, số frame được **chia đều** cho các âm tiết (`data/dataset.py:allocate_durations`). Đây là placeholder deterministic để pipeline chạy được, **không phải** chất lượng production.
- Muốn học duration đúng: dùng forced alignment (ví dụ MFA với model G2P + acoustic tiếng Việt, hoặc aligner tương đương), ghi `durations` (frame cấp mel, hop 256 @ 24 kHz) vào manifest cho từng âm tiết.
- Vocoder chưa được huấn luyện trong `train`; `synthesize` cảnh báo khi checkpoint không có trọng số vocoder.

## Vocoder (tham chiếu)

`vifasttts train-vocoder` huấn luyện vocoder iSTFTNetLite (mel L1 + feature matching + adversarial,
tinh thần HiFi-GAN) với discriminator MPD-lite mặc định, hoặc **MPD đầy đủ 2/3/5/7/11 + MRD +
MR-STFT** khi `vocoder_training.discriminator: mpd_mrd`:

```bash
vifasttts train-vocoder --config configs/base_24k.yaml \
  --manifest data/train.jsonl --output runs/vocoder --max-steps 100000

# Resume tương tự acoustic:
vifasttts train-vocoder --config configs/base_24k.yaml \
  --manifest data/train.jsonl --output runs/vocoder --resume runs/vocoder/last.pt
```

> **Pipeline đầy đủ cho run thật** (cache → acoustic → vocoder → ONNX → demo) đóng gói trong
> `tools/run_small_codelung.sh`; kết quả đo + hướng dẫn triển khai: `docs/TRAINING_RUN_CODELUNG.md`.

### Chạy song song acoustic + vocoder (tối ưu thời gian)

**Vì sao được**: hai model **độc lập** — acoustic học `text → mel` (chạy GPU/MPS), vocoder học
`mel thật → sóng` (chạy CPU trên Mac, không đụng bộ nhớ Metal). Thứ tự nào cũng được; chạy song
song tiết kiệm đúng bằng thời gian train vocoder (ở run `small`: ~2,8 giờ).

```bash
# 1) Acoustic (MPS, từng khối 5k bước, tự resume)
python tools/train_chunked.py --config configs/small_codelung.yaml \
  --train-manifest data/codelung/train.jsonl --val-manifest data/codelung/val.jsonl \
  --output runs/small_codelung --total-steps 60000 --chunk 5000 > logs/acoustic.log 2>&1 &

# 2) Vocoder (CPU; giới hạn thread để nhường CPU cho acoustic)
OMP_NUM_THREADS=6 vifasttts train-vocoder --config configs/small_codelung.yaml \
  --manifest data/codelung/train.jsonl --output runs/vocoder_codelung > logs/vocoder.log 2>&1 &

wait   # đợi cả hai xong
```

**Số đo thật trên Mac mini M6 32GB (chạy song song):**

| | Chạy một mình | Chạy song song | Ghi chú |
| --- | --- | --- | --- |
| Acoustic (MPS) | 0,74 s/bước | ~0,80 s/bước | Chậm hơn ~8% do tranh CPU |
| Vocoder (CPU) | 0,166 s/bước | ~0,24 s/bước | Chậm hơn ~40% (vẫn rất nhanh) |
| RAM hệ thống | ~10–22 GB dùng | ~10–27 GB dùng | Còn 5–20 GB trống — an toàn |
| CPU | ~4–8% | ~35–45% | Còn dư nhiều |
| **Tổng thời gian** | 12,5h + 2,8h = **15,3h** | **~12,6h** | **Tiết kiệm ~2,8 giờ** |

**An toàn & xử lý sự cố:**
- Cả hai đều **resume được** (`--resume` / chunked tự thử lại) — không mất tiến độ nếu crash.
- Nếu RAM căng (máy < 16GB) hoặc muốn acoustic nhanh nhất: giảm `vocoder_training.num_workers`
  (6 → 2–4) hoặc `OMP_NUM_THREADS` (8 → 4), hoặc chạy **tuần tự** như mặc định của script.
- Trên máy **CUDA**: cả hai cùng tranh GPU → chạy song song ít lợi hơn; nên tuần tự hoặc giới hạn
  `vocoder_training.device: cpu`.
- Bật song song trong script trọn gói: `PARALLEL=1 bash tools/run_small_codelung.sh`.

Checkpoint lưu khóa `vocoder` nên `vifasttts synthesize` tự dùng ngay (không còn cảnh báo
"chưa có trọng số vocoder"). Cấu hình ở nhóm `vocoder_training` trong config; `crop_frames`
(mặc định 0) bật **random crop ngắn** mỗi bước — chuẩn GAN vocoder, giữ bước train nhanh và ổn
định dù clip dài 3–20 giây. Đo trên Mac M6 (batch 8): crop 64 · MPD+MRD ch=16 → **0,67 s/bước trên
MPS** (1,35 s/bước CPU); ch=32 (D 31M) → 1,49 s/bước MPS — **MPS nhanh hơn CPU 3,8–4,7× khi
discriminator lớn**, kết luận "CPU nhanh hơn" trước đây đã đảo chiều (`docs/REVISION_PLAN_V2.md` §10).
GAN cần **100k+ bước**; chạy theo khối bằng `python tools/train_vocoder_chunked.py` (chống OOM MPS).

## Phonation (tùy chọn)

Bật `phonation.enabled: true` để trích HNR/spectral tilt/glottalization cấp frame; bật
`model.use_phonation: true` để model **dự đoán thêm 3 kênh** này (giống f0/vuv/energy) và
điều kiện hoá decoder. Thêm trọng số `phonation` vào `loss_weights` nếu muốn học mạnh hơn.
Cache feature tự khoá theo cấu hình phonation nên không lẫn bản có/không có kênh.

## Sáu giai đoạn đề xuất (giữ từ thiết kế gốc)

1. Train iSTFTNet bằng mel thật; train thêm HiFi-GAN baseline để A/B. `train-vocoder` đã có cả
   MPD-lite lẫn **MPD đầy đủ + MRD + MR-STFT** (`discriminator: mpd_mrd`) — bản nâng cấp đang chạy
   cho run @codelung (`docs/REVISION_PLAN_V2.md` §3 bước 2).
2. Train acoustic model với duration, F0, energy thật.
3. Train predictors (đóng băng decoder hoặc giảm LR).
4. Suy luận hoàn toàn bằng predicted features.
5. Joint fine-tuning acoustic + vocoder (JETS-style).
6. Export ONNX → kiểm tra tương đương → FP16/INT8 từng phần. `export-onnx --precision fp16|int8` đã chạy được cho acoustic model (kiểm bằng onnxruntime trong `tests/test_neural_smoke.py`); phần vocoder và calibration INT8 đầy đủ vẫn là việc tiếp theo (xem `docs/TTS_ARCHITECTURE_RESEARCH.md` mục 9.12).

## Cấu hình

`configs/base_24k.yaml` giữ nguyên các nhóm `mel`, `model`, `training`, `loss_weights`. Các khóa `training.*` được đọc bởi `train.py`; `loss_weights` được đọc bởi `losses.py`.

Thiết bị: `training.device: auto` (mặc định) → CUDA nếu có, rồi Apple MPS, cuối cùng CPU;
đặt tường minh `cpu|cuda|mps` khi cần. Trên Mac Apple Silicon, acoustic chạy MPS còn
vocoder tự về CPU (đo thực tế nhanh hơn — `docs/TRAINING_ESTIMATES.md` §3.1). AMP chỉ
bật trên CUDA; `torch.stft` complex không có trên MPS nên mel loss của vocoder được
tính trên CPU tự động.

Chọn biến thể theo lượng dữ liệu (ước lượng: `docs/TRAINING_ESTIMATES.md`):

| Dữ liệu | Config | Tham số acoustic |
| --- | --- | --- |
| < 5 giờ | `configs/small_24k.yaml` | ≈ 6,1M |
| 5–30 giờ | `configs/base_24k.yaml` | ≈ 16,4M |
| ≥ 40–50 giờ | `configs/large_24k.yaml` | ≈ 32,2M |

## Checklist trước khi train thật

```bash
vifasttts validate-manifest --manifest data/train.jsonl --wav-dir data/wavs
pytest
```

- Không còn token `RTX3080`-kiểu fallback trong tập train nếu không có chủ đích.
- Coverage thanh điệu × coda đủ (6 thanh, các coda `-p/-t/-c/-ch/-nh/-ng`).
- Mọi WAV 24 kHz mono (hoặc để `load_wav` tự resample).
