# Báo cáo chạy đêm 21/10 (v17) — cập nhật dần

Trạng thái lúc ghi: **A1 xong · A2 đang chạy (cache) · A3–A7 chờ**.

## 1. Tóm tắt

| bước | trạng thái | kết quả / ghi chú |
|---|---|---|
| A1 sửa nhãn đầu/cuối | **xong** | cuối **0,0 %** token (không có frame lặng để chuyển — độ dài cuối là *phrase-final lengthening* thật, đúng như v16 §0.3); đầu **6,6 %** (1.211/18.226 row, chuyển frame lặng đầu slot O sang lead) |
| A2 cache + mel stats | **đang chạy** | stats mới trên **556.230 frame hữu thanh** → `data/codelung/mel_stats_v35_speech.json`; cache `cache/codelung_v35` (4 worker) |
| A3 code (item 7, augmentation, ghép row) | **xong 2/3** | item 7 (cross-slot coda/nucleus) + augmentation biên (p 0,3/0,3, 1–5 âm tiết) + sampler theo N SAU cắt; **ghép row: bỏ** (lùi, ghi ở §4) |
| A4 smoke 500 bước | **đạt** | loss 16,97 → ~14,2 (dao động ±2 do batch nhỏ), **0,17–0,19 s/bước**, MPS 5,8 G, không NaN, `ema: True` + `ema_decay 0,999` trong `last.pt` |
| A5 train 60k | **ĐANG CHẠY** | `runs/acoustic_v35` (log `runs/acoustic_v35_train.log`), ước 3,3–3,7 h |
| A6 GAN 3k | chờ | |
| A7 eval + trang nghe mù | chờ | |
| B1 12a | **xong (v16)** | C1 chunk vs C2/C3/C4; flatness giữa câu C1 0,39–0,41 > C2/C3/C4 0,27–0,29 |
| B2 drift p90 C2/C4 | chưa | |
| B3 9a 4 chế độ | chưa | |
| B4 ablation duration | chưa | |

## 2. A1 — chi tiết

- Ngưỡng: năng lượng frame < (trung vị hữu thanh segment − 25 dB), sàn 3 frame/slot.
- **Cuối segment: 0 token bị sửa.** Kiểm tra cho thấy không có run frame năng lượng thấp ở cuối
  slot cuối ⇒ nhãn dài cuối segment **không phải do nhét lặng**; giữ nguyên theo v16 §0.3
  (không sửa đại trà).
- **Đầu segment: 6,6 %** row có frame lặng đầu slot O → chuyển sang `lead_frames`
  (độ dài âm đầu trung vị 15 → 15 frame; chỉ các row có hơi thở/lặng đầu bị sửa).
- Manifest mới: `data/codelung/v35/train_edgelabel.jsonl` (không sửa bản gốc).

## 3. A2 — chi tiết

- Mel stats mới: mean/std từng bin trên frame `vuv ≥ 0,5` của 6.000 row train (lấy mẫu cách 3),
  **556.230 frame**. Mean tuyệt đối trung bình 1,76 · std trung bình 1,68.
- Config `configs/small_codelung_v35.yaml`: `mel.normalize_path` → stats mới, `cache_dir` →
  `cache/codelung_v35` (mel_hash đổi ⇒ cache riêng, **không đụng v32**).
- Cache đang build cho train (18.226 row) + val (3.378 row) với 4 worker.

## 4. Lùi / sai lệch cần ghi nhận

- v3.5 lần 1 (đã huỷ) thiếu item 2 + item 7; run hiện tại sẽ có item 2 (stats/cache mới).
- Item 7 (embedding coda/nucleus) và augmentation/ghép row **chưa làm** — nếu không kịp, train với
  phần còn lại và ghi rõ (theo v17 §A3 "lùi").

### 4b. Cập nhật A3 (đã làm)

- **Item 7 — XONG:** `cross_slot_context: true`; slot N += `coda_to_nucleus(coda)`, slot C +=
  `nucleus_to_coda(nucleus)` (cùng âm tiết, cộng vào khối component của `segment_in`). Mặc định
  `false` ⇒ checkpoint/ONNX cũ không đổi. Test: shape + gradient chảy về hai projection.
- **Augmentation — XONG:** cắt 1–5 âm tiết đầu/cuối, p 0,3 mỗi phía độc lập, sàn 4 âm tiết,
  cắt tại biên âm tiết, **giữ feature boundary gốc**, bất biến `T = Σdur + Σgap` được test.
- **Bẫy đã sửa:** augmentation đổi N sau `__getitem__` ⇒ `collate_slot_batch` vỡ batch
  (`[4,5,6,7]`). Sửa: kế hoạch cắt tính trước theo index (`_plans`), `effective_lengths()` đưa cho
  `SameNBatchSampler` (thêm tham số `lengths`) ⇒ batch gộp theo N **sau** cắt.
- **Ghép row (p 0,2 nối 2 row cùng `parent`) — BỎ (lùi):** cần dựng lại index theo `parent` +
  cộng nhãn 2 row (kể cả gap thật giữa chúng) trong thời gian không cho phép; train với phần còn lại.
- **Bug phụ đã sửa:** `TtsDataset` (v1) crash `KeyError: 'tilt'` khi cache v3 chỉ có `jitter`
  (lộ ra khi chạy nhầm `vifasttts train` thay vì `train-v2`) — nay chọn kênh phonation có trong cache.
- **Kiểm A2:** 20 row ngẫu nhiên, mel z trên frame tiếng: **mean −0,121 · std 1,011** ✓ (mean lệch nhẹ
  do row ngắn ít frame hữu thanh; toàn cục tính trên 556k frame là 0).

