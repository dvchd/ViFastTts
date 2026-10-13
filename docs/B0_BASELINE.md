# B0 — mốc chuẩn acoustic v1 (13/10/2026)

> B0 theo B10: checkpoint `runs/small_codelung_v2/best.pt` (bước 8.000, best val 13,9615);
> ghi kèm `last.pt` (bước 20.000). Đo bằng `tools/eval_acoustic.py` (T1.2):
> `tf_syl` trên 300 clip `val_clean_v2_dur_seg` + `free` trên 20 clip `eval_clips_v1`.

## 0. Truy vết

- Config: `configs/small_codelung.yaml` · dữ liệu: `train/val_clean_v2_dur_seg.jsonl`.
- sha256 (16 ký tự đầu): best.pt `41153e59a9b53ef9` · last.pt `06b636e9a650f3a7` · f0_stats.json `2af359d7824479cd`.
- `mel_config_hash` (trong checkpoint): `e065776b45b74b87` — giống nhau ở best/last.
- LR schedule cosine tới 60k (toàn cục); resume 9k→20k không reset optimizer; val chững từ bước 8k.

## 1. Bảng B0

| Chỉ số (mean) | best.pt (8k) | last.pt (20k) | Tham chiếu |
|---|---|---|---|
| mel L1 z | 1,276 | 1,220 | train mel ~0,8 |
| mel L1 raw (log-magnitude) | 1,696 | 1,620 | copy-synthesis 0,15 |
| mel MAE (dB) | 14,7 | 14,1 | copy 1,3 dB |
| VUV accuracy | 0,992 | 0,991 | — |
| energy MAE | 0,714 | 0,726 | — |
| duration MAE | 51,1 ms | 51,9 ms | ~32% âm tiết 160 ms |
| F0 NGANG / HUYEN / SAC (cents, median) | 134,9 / 129,3 / 162,8 | 138,7 / 133,1 / 166,6 | — |
| F0 HOI / NGA / NANG (cents, median) | 222,6 / 196,5 / 153,5 | 222,6 / 204,6 / 164,4 | — |
| free CER (20 clip) | 0,939 | 0,538 | **orig 0,486** (sàn Whisper) |
| free token-match | 0,000 | 0,234 | **orig 0,567** |
| RTF (PyTorch CPU) | 0,009 | 0,009 | — |
| RAM đỉnh | 1,71 GiB | 1,69 GiB | — |

## 2. Phân rã loss val (20 batch × 8, model last.pt)

| Thành phần | ×trọng số | % tổng |
|---|---|---|
| **f0** | **11,913** | **82%** |
| mel (+ mel_pre) | 2,050 | 14% |
| energy | 0,354 | 2% |
| duration | 0,145 | 1% |
| vuv | 0,021 | 0% |
| tone | 0,000 | 0% |
| **TỔNG** | **14,483** | |

## 3. Kết luận

1. B0 yếu toàn diện; **f0 (Hz thô) chiếm 82% loss** → mel/duration bị bỏ đói. Target f0 còn nhiễu vì
   lấy mean qua cả frame nội suy (F4 — không mask `vuv`).
2. `best.pt` (8k, "best val") **tệ hơn** `last.pt` (20k) trên audio (CER 0,94 vs 0,54; token-match 0,00 vs 0,23)
   → val loss không phải proxy tốt cho chất lượng nghe.
3. Vocoder **không** phải nút cổ chai (copy ≈ orig, xem `artifacts/listen_b0/DIAGNOSTICS.md`);
   **acoustic là nút cổ chai**.
4. Free-run: CER trung bình ≈ sàn + 0,05 (last) nhưng nhiều clip lặp/từ giả; mel không tạo được vùng lặng
   (pred z-std 1,5 vs real 4,5 trên clip đầy đủ).
5. **Khuyến nghị**: đưa **log-F0 z-score + mask `vuv`** (T3.2/F4) vào ngay lần train v2 đầu tiên thay vì
   đợi Phase 3. Probe f0-fix đang chạy để định lượng (`configs/small_codelung_f0fix.yaml`, log
   `artifacts/eval_acoustic/f0fix_probe.log`).

## 4. File kèm

- `artifacts/eval_acoustic/b0_best_tf_syl.json` · `b0_last_tf_syl.json` · `b0_best_free.json` · `b0_last_free.json`
- `artifacts/listen_b0/DIAGNOSTICS.md` + `artifacts/listen_b0/index.html` (nghe thử).
