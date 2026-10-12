# Ước lượng train: dữ liệu, tham số, thời gian, máy, chi phí

> ⚠️ **TẤT CẢ SỐ TRONG TÀI LIỆU NÀY LÀ ƯỚC LƯỢNG**, không phải cam kết. Chúng dựa trên:
> (1) **số đo thật** của repo trên một máy CPU 2 luồng (Apple Silicon, torch 2.14) ở mục 3;
> (2) kích thước tham số **tính chính xác** từ config (mục 2);
> (3) kinh nghiệm công bố của các hệ TTS tương tự (FastSpeech2/VITS/iSTFTNet) — phần này là
> **suy đoán có cơ sở**. Hãy hiệu chỉnh lại bằng phép đo 20 bước trên máy thật của bạn (mục 7).
> Cập nhật: 11/10/2026.

## 1. Ước lượng dữ liệu

### 1.1 Cần bao nhiêu audio?

| Mục tiêu | Dữ liệu (một người nói) | Kỳ vọng |
| --- | --- | --- |
| Chạy thử / demo kỹ thuật | **1–3 giờ** | Nghe rõ từ nhưng đều đều, dễ méo thanh điệu hiếm |
| Nghe được, dùng nội bộ | **5–10 giờ** | Tự nhiên vừa phải; cần đủ 6 thanh × coda |
| Chất lượng tốt (khuyến nghị) | **10–20 giờ** | Mức phổ biến của các paper single-speaker (LJSpeech = 24h) |
| Chất lượng cao / thương mại | **40–100+ giờ** | Lợi ích tăng chậm; nên tăng model (mục 2) song song |

Quy đổi nhanh cho tiếng Việt (~4–5 âm tiết/giây):

- **1 giờ** ≈ 14.000–18.000 âm tiết ≈ 3.500–4.000 từ ≈ 1.200 câu (câu ~3 giây).
- **10 giờ** ≈ 150.000 âm tiết ≈ 12.000 câu.
- Điều kiện quan trọng hơn số giờ: **phủ đủ 6 thanh và các coda** — kiểm bằng
  `vifasttts validate-manifest` (in coverage tone/coda/fallback).
- **Bắt buộc có duration thật** (MFA hoặc aligner tương đương) để chất lượng tốt; không có
  thì pipeline chia đều frame (placeholder, xem `docs/TRAINING.md`).

### 1.2 Dung lượng ổn định

| Thứ | Công thức | 10 giờ |
| --- | --- | --- |
| WAV 24 kHz mono PCM16 | 48 KB/s | ≈ 1,7 GB |
| Cache mel (100 bins, float32, hop 256) | ≈ 37,5 KB/s | ≈ 1,4 GB |
| **Tổng (audio + cache)** | ≈ 85,5 KB/s | **≈ 3,1 GB** |

### 1.3 Thời gian chuẩn bị dữ liệu (ước lượng)

| Việc | Ước lượng | Ghi chú |
| --- | --- | --- |
| Cắt/QC audio + transcript | 1–3× thời lượng audio (người) | Phần tốn người nhất |
| Forced alignment (MFA) | ~0,3–1× thời lượng audio (đa luồng CPU) | 10h audio → ~3–10h CPU 8 nhân; **không cần GPU** |
| Chạy `cache-features` | ~0,05–0,1× thời lượng audio | 10h → ~0,5–1h CPU; giảm mạnh thời gian train |

## 2. Ước lượng tham số model (tính chính xác từ config)

Số dưới đây **đo từ code** với vocab 24/2/15/11/6 (+boundary/punctuation):

| Biến thể | hidden/layers (enc–dec)/ffn/heads | Tham số acoustic | Config | Ghi chú |
| --- | --- | --- | --- | --- |
| tiny (test) | 16 / 1–1 / 32 / 2 | **14.142** | — | Chỉ để smoke test, không dùng thật |
| small | 192 / 4–2 / 768 / 4 | **6.139.054** | `configs/small_24k.yaml` | Khi dữ liệu < 5 giờ |
| **base (mặc định)** | 256 / 6–4 / 1024 / 4 | **16.444.654** | `configs/base_24k.yaml` | 5–30 giờ dữ liệu |
| large | 320 / 8–5 / 1280 / 8 | **32.156.654** | `configs/large_24k.yaml` | ≥ 40–50 giờ |
| xl (ví dụ) | 384 / 10–6 / 1536 / 8 | **55.428.974** | — | Chỉ khi có > 100 giờ + nhiều GPU |

Vocoder: **3.527.426** tham số (iSTFTNetLite hidden 256) — dùng chung cho mọi biến thể trên.

**Chọn kích thước theo dữ liệu (quy tắc ngón tay):**

- < 5 giờ → `small` (6,1M): giảm overfit.
- 5–30 giờ → `base` (16,4M): điểm cân bằng của repo.
- ≥ 40 giờ và đã nghe ổn → `large` (32,2M), đồng thời tăng vocoder (hidden 384+).

Config sẵn có: `configs/small_24k.yaml`, `configs/base_24k.yaml`, `configs/large_24k.yaml`
(vocoder hidden tương ứng 192/256/384).

**Dấu chân triển khai (base):**

| Định dạng | Acoustic | + Vocoder |
| --- | --- | --- |
| fp32 | ≈ 66 MB | ≈ 80 MB |
| fp16 | ≈ 33 MB | ≈ 40 MB |
| int8 (weight-only) | ≈ 17 MB | (chưa hỗ trợ) ≈ 20 MB |

## 3. Số đo thật trên máy tham chiếu (CPU 2 luồng)

Máy đo: Apple Silicon, `torch.get_num_threads() == 2`, Python 3.12, torch 2.14, KHÔNG GPU.
Base config, batch 8, câu 30 âm tiết × 20 frame = 600 frame/sample.

| Việc | Kết quả đo | Quy đổi |
| --- | --- | --- |
| Train acoustic (fwd+bwd+step) | **0,53 s/bước** (4.800 frame mel ≈ 51 s audio/batch) | ≈ 96× thời gian thực trên CPU 2 luồng |
| Train vocoder (G+D mỗi bước) | **0,38 s/bước** (19.200 mẫu ≈ 0,8 s audio) | ≈ 2,1× thời gian thực |
| Inference (acoustic+vocoder, base) | **0,019 s** cho 0,23 s audio | **RTF ≈ 0,08** (nhanh hơn ~12× thời gian thực) |

### 3.1 Mac Apple Silicon (MPS) — đo trên Mac mini M6, 32 GB

**Phần cứng (đối chiếu máy + trang spec chính thức `apple.com/mac-mini/specs`):**

| Thành phần | Thông số |
| --- | --- |
| CPU | 12 nhân: 2 **Super** + 4 **Performance** + 6 **Efficiency** |
| GPU | **12 nhân**, Metal 4 |
| Neural Engine | **Dual 16-core** + "Neural Accelerators" (khối AI chuyên dụng) |
| RAM | 32 GB hợp nhất; băng thông **153 GB/s** (bản 16 GB) / **170 GB/s** (bản 24–32 GB) |
| Điện tối đa | 155 W (liên tục) |

Nguồn thông số: trang spec chính thức <https://www.apple.com/mac-mini/specs/> (truy cập
11/10/2026) — dòng "Apple M6 chip": 12-core CPU (2 super + 4 performance + 6 efficiency),
12-core GPU, Neural Accelerators, Dual 16-core Neural Engine, băng thông 153 GB/s
(16 GB) và 170 GB/s (24–32 GB). Máy đo trong repo: bản 32 GB.

**Phân biệt thuật ngữ (rất dễ lẫn):**

| Tên | Là gì | Trong PyTorch |
| --- | --- | --- |
| **CPU** | Bộ xử lý trung tâm, chạy mọi thứ | `device="cpu"` |
| **GPU** | Xử lý song song; Apple GPU chạy qua Metal | `device="mps"` |
| **MPS (Apple)** | *Metal Performance Shaders* — thư viện/API của Apple trên Metal; **không phải tên máy** | backend của `mps`, tương đương vai trò CUDA |
| **CUDA** | Nền tảng GPU của NVIDIA (Mac không có) | `device="cuda"` |
| **CUDA MPS** | *Multi-Process Service* của NVIDIA — **trùng viết tắt, khác hoàn toàn** | – |
| **Neural Engine (ANE)** | Khối AI chuyên dụng của Apple | **PyTorch KHÔNG dùng ANE**; muốn dùng phải qua Core ML/CoreMLTools |

Cùng batch đo như trên:

| Việc | CPU (2 luồng) | CPU (12 luồng) | MPS (GPU) | Kết luận |
| --- | --- | --- | --- | --- |
| Train acoustic | 0,53 s/bước | 0,74 s/bước | **0,31 s/bước** | MPS nhanh nhất (~1,7× CPU 2 luồng); 12 luồng chậm hơn 2 luồng vì overhead đồng bộ trên batch nhỏ |
| Train vocoder (G+D) | **0,39 s/bước** | – | 0,93 s/bước | CPU nhanh hơn 2,4× → **mặc định vocoder dùng CPU trên Mac** (`vocoder_training.device=mps` để ép) |
| Vocoder inference | – | – | ~0,004 s cho 2×50 frame mel | Rất nhanh, dùng tốt cho tổng hợp |

**Đo bổ sung trên dữ liệu thật (run `small` @codelung, clip 14s, batch 16 ≈ 29.000 frame):**

| Hạng mục | Đo được | Ghi chú |
| --- | --- | --- |
| Model MPS fp32 | **0,478 s/bước** | Batch đã trên thiết bị; fp16 autocast 0,52 → không lợi, AMP chỉ cho CUDA |
| Train thật (cache + empty_cache) | **~0,67–0,75 s/bước** | 60k bước ≈ 12–12,5h |
| Dataloader **không** cache | **6,34 s/batch** | STFT + F0 autocorrelation → **cache feature là bắt buộc** với clip dài |
| Cache feature 8 workers | **33 giây / 51 giờ audio** | 6,7 GB cho 13.225 clip |
| Vocoder CPU + crop 128 frame | **0,166 s/bước** | Random crop ngắn thay vì clip 14s (nhanh ~50×) |
| Bộ nhớ MPS | leo tới **40 GB → OOM** | Phải `empty_cache` định kỳ + cap validation + chunked runner — xem `docs/TRAINING_RUN_CODELUNG.md` §3 |

### 3.2 Dự đoán `base`/`large` trên dataset @codelung (Mac mini M6 32GB) — *ước lượng*

> **Tất cả số ở mục này là DỰ ĐOÁN**, neo vào số đo thật của `small` và quy luật tỉ lệ;
> chưa chạy A/B thật (cách tự đo ở cuối mục).

**Mô hình bộ nhớ (MPS driver allocated):**

- Tham số + gradient + optimizer: `base` chỉ ≈ **0,26 GB** (16,4M × 4B × 4) — không đáng kể.
- **Activation chiếm gần hết**, tỉ lệ với `frame mỗi bước × hidden × số layer decoder`
  (encoder chạy ở cấp âm tiết nên rất nhỏ).
- Neo đo thật: `small` (hidden 192, decoder 2 layer) @batch 16 = 29.300 frame → đỉnh **25 GB**
  driver (≈17 GB thật; lúc driver 21 GB thì GPU `In use system memory` chỉ 14,7 GB).
- Hệ số so với `small`: **base ≈ 2,7×** (256/192 × 4/2) · **large ≈ 4,2×** (320/192 × 5/2).

**Bảng dự đoán (cùng ~84 epoch trên 44,4h train):**

| Model | Batch | Frame/bước | Đỉnh MPS (dự đoán) | s/bước (dự đoán) | Bước cần | Tổng acoustic | Rủi ro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small (đo thật) | 16 | 29.300 | 25 GB | **0,75s** | 60k | **~12,5h** | ✅ đang chạy ổn |
| small (tối ưu nhẹ) | 20 | 36.700 | ~31 GB | ~0,89s | 48k | ~11,9h | ✅ thấp |
| small (tối đa an toàn) | 24 | 44.000 | ~37 GB | ~1,03s | 40k | ~11,4h | ⚠️ sát ngưỡng |
| small (quá lớn) | 32 | 58.600 | ~48 GB | – | – | – | ❌ OOM |
| **base** | 6 | ~11.000 | ~25 GB | ~0,75–1,1s | 160k | **~33–48h** | ✅ an toàn |
| **base** | 8 | ~14.700 | ~34 GB | ~0,90–1,25s | 120k | **~30–42h** | ⚠️ có thể swap nhẹ |
| base | 12 | ~22.000 | ~51 GB | – | – | – | ❌ OOM |
| large | 4 | ~7.300 | ~26 GB | ~0,8–1,1s | 240k | **~52–72h** | ✅ nhưng quá lâu |

- Khoảng dao động của `s/bước` phản ánh hệ số base/large trên frame chưa chắc chắn
  (2,5–4× tuỳ mức tối ưu kernel); **cần A/B thật để chốt**.
- **RAM hệ thống (unified 32GB)**: nên giữ đỉnh driver ≤ **~30–32 GB** (thật ~20 GB) để tránh swap;
  base batch 6 nằm ngưỡng an toàn, batch 8 chớm rủi ro, batch 12 chắc chắn OOM.
- **Khuyến nghị**: muốn lên `base` → dùng **batch 6–8 + `tools/train_chunked.py`**,
  dự trù **~1,5 ngày** cho acoustic (chưa kể vocoder ~1,4h + export). `large` chỉ nên tính
  khi đã có alignment thật và chấp nhận ≥2 ngày.

**Cách chọn batch cho `small` (kết luận thực dụng):**

- **Mặc định giữ 16** — đã kiểm chứng qua run 60k bước, đỉnh bộ nhớ 25 GB, không rủi ro.
- Muốn tối ưu: **20 là điểm cân bằng** (dự đoán lợi ~5%, đỉnh ~31 GB); **24 lợi ~9%**
  nhưng sát ngưỡng — phải A/B + chunked. **Không cần số bội của 2**: trên Apple GPU, song song
  chủ yếu theo trục frame (T), batch chỉ ảnh hưởng phụ.
- **Cách tự đo A/B** (300 bước, ~4 phút): đổi `training.batch_size`, chạy
  `vifasttts train ... --max-steps 300` với `log_every: 50`, so `s/bước` và `mps=` đỉnh trong log;
  giữ cấu hình nào có `s/clip` thấp nhất mà đỉnh ≤ ~30 GB.

Ghi chú: ANE (Dual 16-core Neural Engine) **không được dùng** trong pipeline hiện tại —
PyTorch MPS chỉ chạy trên GPU; muốn khai thác ANE cần chuyển model sang Core ML (hướng
tương lai, đặc biệt cho inference trên máy người dùng).

Khảo sát đầy đủ các thư viện Apple (MLX, Core ML, provider ONNX) + số đo so sánh và
quyết định "không fork luồng train": `docs/APPLE_SILICON.md`.

**Ước lượng thời gian train trên Mac mini M6 32 GB** (từ số đo/bước, cộng ~10–20% cho
dataloader nếu **không** cache feature):

| Kịch bản | Acoustic | Vocoder | Tổng ước lượng |
| --- | --- | --- | --- |
| Demo (100k + 100k bước) | ~8,6 h | ~11 h | **~20–24 h** |
| Chuẩn (200k + 150k bước) | ~17 h | ~16 h | **~1,5–2 ngày** |
| Cao cấp (500k+) | ~43 h | ~54 h | **~4–5 ngày** |

**Kết luận cho Mac mini M6 32 GB:** train được thoải mái với `small`/`base` và dữ liệu
5–20 giờ; 32 GB hợp nhất là đủ (model chỉ ~16M tham số, activation nhỏ). Nên:

1. `cache-features` trước (giảm nghẽn đọc/STFT).
2. Để `device: auto`: acoustic chạy MPS, vocoder tự về CPU.
3. `num_workers` 4–6 là hợp lý trên máy 12 nhân (đừng đặt quá cao).
4. Muốn nhanh hơn nhiều (3–10×) thì mới cần GPU rời/NVIDIA; chi phí xem mục 6.
5. `torch.stft` complex không chạy trên MPS — pipeline đã tự tính mel loss của vocoder
   trên CPU rồi truyền ngược qua thiết bị, nên không cần cấu hình gì thêm.

## 4. Ước lượng thời gian train theo cấu hình máy

Hệ số tăng tốc GPU so với CPU 2 luồng ở mục 3 (ước lượng thô cho workload nhỏ, TF32/AMP):

| Máy | Hệ số ước lượng | Ghi chú |
| --- | --- | --- |
| CPU 8–16 luồng | × 3–6 | Khả thi cho dataset nhỏ, chậm |
| RTX 3060 12 GB / T4 | × 20–40 | Tối thiểu nên có |
| RTX 4090 24 GB / A10G | × 50–100 | Khuyến nghị |
| A100 40/80 GB | × 60–120 | Chỉ cần cho large/xl + nhiều thí nghiệm |

**Kịch bản (ước lượng, đã tính cả nghẽn dataloader nhẹ nhàng):**

| Kịch bản | Dữ liệu | Model | Bước (acoustic + vocoder) | 4090 ước lượng | A100 ước lượng | CPU 2 luồng |
| --- | --- | --- | --- | --- | --- | --- |
| Demo | 2–5 h | base | 100k + 100k | **4–10 h** | 3–7 h | 3–7 ngày |
| Chuẩn (khuyến nghị) | 10–20 h | base | 200–400k + 150–300k | **10–25 h** | 7–15 h | 1–3 tuần |
| Cao cấp | 50–100 h | large/xl | 500k–1M + 300–600k | 3–7 ngày | **1,5–4 ngày** | không khả thi |

Số bước tham khảo: mô hình NAR kiểu này thường cần **100k–300k bước** để nghe được và
**300k–1M bước** để chín; dừng dựa trên validation + nghe thử, không chỉ theo bước.

## 5. Máy cấu hình khuyến nghị

| Mức | Cấu hình | Dùng cho |
| --- | --- | --- |
| Tối thiểu (dev) | CPU bất kỳ, 8 GB RAM | Chạy test (2224 test), parse, thử ONNX |
| **Mac Apple Silicon** | ≥ 16 GB RAM hợp nhất (M-series); đo trên Mac mini M6 32 GB | Train small/base cỡ vừa; acoustic dùng MPS, vocoder CPU — xem mục 3.1 |
| Vừa (khuyến nghị) | 1× GPU ≥ 12 GB VRAM (3060/4070), 32 GB RAM, SSD 100 GB | Train base với 10–20 h |
| Thoải mái | 1× 4090 24 GB / A10G, 64 GB RAM | Train + nhiều thí nghiệm A/B, vocoder lớn |
| Lớn | A100/H100 40–80 GB | large/xl, dataset 50 h+, tuning nhiều |

Lưu ý: **cache feature trước khi train** (`vifasttts cache-features` hoặc
`training.cache_features: true`) để dataloader không còn là nút thắt; dung lượng cache
xem mục 1.2.

## 6. Ước lượng chi phí (tham khảo 2026)

Giá cloud GPU thay đổi liên tục — dải tham khảo (Vast.ai/RunPod/Lambda, có thể lệch ±50%):

| GPU | Giá ước lượng | Ghi chú |
| --- | --- | --- |
| T4 | $0,3–0,5/giờ | Rẻ, chậm |
| RTX 4090 | $0,4–0,9/giờ | Hiệu quả nhất cho base |
| A100 80 GB | $1,2–2,5/giờ | Cho large/xl |

**Kịch bản chi phí (chỉ GPU, chưa tính người):**

| Kịch bản | GPU-giờ (đã nhân ×3 cho thí nghiệm lặp) | Chi phí 4090 | Chi phí A100 |
| --- | --- | --- | --- |
| Demo (5 h, base) | 12–30 | **$5–27** | $15–75 |
| Chuẩn (20 h, base) | 30–75 | **$12–68** | $36–190 |
| Cao cấp (100 h, large) | 200–600 | $80–540 | **$240–1.500** |

**Điện + máy local (Việt Nam, tham khảo ~3.000đ/kWh):** hệ 4090 ~500 W → ~1.500đ/giờ;
một lần train chuẩn (15 giờ) ≈ **~25.000đ tiền điện** — rẻ hơn cloud nếu đã có máy,
nhưng chi phí máy (~30–50 triệu) lớn hơn nhiều lần.

**Mac mini M6 (155 W tối đa, thực tế khi train ~60–100 W):** ~200–465đ/giờ;
train chuẩn (~33 giờ acoustic+vocoder) ≈ **~7.000–15.000đ tiền điện** — rẻ nhất nếu
đã có máy (không tính khấu hao).

**Chi phí thật nằm ở con người:** chuẩn bị + QC dữ liệu, alignment, nghe thử, tune
thường chiếm **20–120 giờ người** cho một giọng tốt — lớn hơn toàn bộ tiền GPU.

## 7. Cách tự hiệu chỉnh ước lượng trên máy bạn

```bash
# 1) Làm nóng + đo s/bước thật (log_every=1 để thấy nhịp)
vifasttts train --config configs/base_24k.yaml \
  --train-manifest data/train.jsonl --val-manifest data/val.jsonl \
  --output runs/probe --max-steps 20

# 2) Công thức
#    giờ_train = bước_dự_kiến × giây/bước ÷ 3600
#    (nhân thêm ×3–5 nếu còn thí nghiệm/tune)

# 3) Giảm thời gian train
vifasttts cache-features --manifest data/train.jsonl \
  --config configs/base_24k.yaml --cache-dir cache/train

# 4) Vocoder tương tự
vifasttts train-vocoder --config configs/base_24k.yaml \
  --manifest data/train.jsonl --output runs/vocoder --max-steps 20
```

Ghi lại giây/bước thực tế vào bảng dưới đây rồi thay số vào mục 4:

| Máy của bạn | Giây/bước acoustic | Giây/bước vocoder | Ngày đo |
| --- | --- | --- | --- |
| (điền) | | | |

## 8. Kết luận khuyến nghị cho repo này

1. **Bắt đầu**: 10–20 giờ dữ liệu một người nói, model `base` (16,4M), một GPU 12–24 GB,
   MFA alignment, bật cache feature.
2. **Ngân sách thử nghiệm hợp lý**: $25–90 GPU (hoặc ~vài chục nghìn đồng tiền điện nếu
   chạy máy nhà) + 20–40 giờ người cho dữ liệu/nghe thử.
3. **Chỉ tăng quy mô khi**: đã nghe thấy giới hạn ở `base` (không phải lỗi dữ liệu/alignment),
   coverage 6 thanh × coda đầy đủ, và có ≥ 40–50 giờ dữ liệu sạch.
4. **Không nên**: train từ đầu trên CPU cho dataset hàng chục giờ; dùng `tiny` cho sản phẩm.

## 9. Liên quan

- `docs/TRAINING.md` — hướng dẫn chạy + cảnh báo alignment.
- `docs/MODEL_FLOW.md` — luồng model, số tham số, loss.
- `docs/DATA_PIPELINE.md` — 15 bước chuẩn bị dữ liệu.
- `docs/DEPLOYMENT.md` — ONNX/quantization, dấu chân triển khai.
- `docs/TTS_ARCHITECTURE_RESEARCH.md` — nếu muốn đổi kiến trúc trước khi đầu tư.
