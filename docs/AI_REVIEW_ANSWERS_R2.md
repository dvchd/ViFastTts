# AI_REVIEW_ANSWERS_R2 — Trả lời vòng review 2 (xác minh + trạng thái vận hành)

> **Snapshot:** 2026-10-12, HEAD `524dda8`; **2264 test pass** (`python -m pytest`), `ruff` + `mypy` sạch.
> **Trạng thái quan trọng:** chuỗi nền **đang bị tạm dừng an toàn ngay ranh giới `cache train → cache val`**
> (SIGSTOP tiến trình shell `69421`), cache train đã xong 11.424/11.424; **vocoder v2 chưa hề chạy**.
> Cách resume / hủy: xem §Vận hành cuối tài liệu.

---

## R2-1 — VocoderDataset có đọc cache không, hay gọi extract_features (kể cả pyworld) mỗi item?

**KẾT LUẬN: GIẢ THUYẾT SAI khi cache ấm.** `VocoderDataset` **có** đọc cache khi được truyền `cache_dir`,
và `train_vocoder_reference` truyền `cache_dir` từ config (`small_codelung.yaml` có `cache_dir: cache/codelung`).

**Bằng chứng code:**
- `dataset.py:220` — `self.cache = FeatureCache(cache_dir, config) if cache_dir is not None else None`
- `dataset.py:231–237` — `cache.get(row)` → chỉ gọi `extract_features` **khi cache miss**
- `train.py:427` — `dataset = VocoderDataset(manifest, config, cache_dir=_resolve_cache_dir(training, out))`
- `train.py:103–110` — `_resolve_cache_dir`: ưu tiên `training.cache_dir`, nếu không có thì `<output>/feature_cache`

**Số đo thật** (trên máy, cache đã ấm 100% cho `train.jsonl`):

| Phép đo | Kết quả |
| --- | --- |
| 1 item cache hit | **0,002 s** (torch.load file .pt 0,60 MB) |
| 1 item cache miss (`extract_features` đầy đủ, gồm pyworld) | **1,360 s/clip** |
| DataLoader batch 8, `num_workers=6` | **4,2 ms/batch** (0,5 ms/clip) |
| DataLoader batch 8, `num_workers=0` | 10,7 ms/batch |
| Model + GAN MPS (tham chiếu) | **659 ms/bước** |

→ Dataloader không còn là nút thắt khi cache ấm (4 ms so với 659 ms). Rủi ro chỉ còn nếu có cache miss
(ví dụ clip mới, hoặc đổi cấu hình làm đổi khoá cache). Cache train hiện **đã đủ 11.424/11.424**.

**Ghi chú kèm:** khi miss, `extract_features` vẫn tính pyworld F0 mà vocoder không dùng
(`features.py:244–256`) — nếu sau này có miss thường xuyên (ví dụ thêm dữ liệu), nên làm nhánh
mel-only cho vocoder hoặc namespace cache riêng; **không cần làm ngay** vì cache đã ấm.

---

## R2-2 — Cache lưu F0 cấp frame hay cấp âm tiết?

**KẾT LUẬN: GIẢ THUYẾT ĐÚNG — cache lưu F0 cấp FRAME (Hz), nên DCT/log/z-score làm được trong
dataset/loss mà không phải tính lại cache.**

Ví dụ file cache thật (`cache/codelung/700915d4….pt`, 0,60 MB, load 5,4 ms):

| Key | Shape | dtype | Dải giá trị |
| --- | --- | --- | --- |
| `mel` | `[1441, 100]` | float32 | −2,63 … 2,62 (đã chuẩn hoá toàn cục) |
| `energy` | `[1441]` | float32 | −13,82 … 0,05 (log) |
| `f0` | `[1441]` | float32 | **78,59 … 348,98 Hz** (frame, pyworld + interpolate) |
| `vuv` | `[1441]` | float32 | 0/1 |

**Cảnh báo quan trọng (ủng hộ đề xuất của review):** khoá cache là hash của **toàn bộ** mục
`f0` + `phonation` + `mel` + hash nội dung stats (`data/cache.py:20–50`, dòng `44: "f0": config.get("f0", {})`)
→ **thêm bất kỳ trường nào vào `f0:` sẽ làm vô hiệu toàn bộ 13.215 file cache**. Đúng như review nói:
chuẩn hoá F0 nên đặt stats trong mục config riêng (hoặc file stats riêng ngoài khoá cache) và áp ở
dataset/loss.

---

## R2-3 — LengthRegulator có chạy đúng với duration = 0 (SIL) không, cả torch lẫn ONNX?

**KẾT LUẬN: ĐÚNG về mặt engine — không NaN, cả torch lẫn ONNX (nếu durations là input).**
Nhưng **đường export sản phẩm hiện tại không thể nhận SIL=0** vì clamp_min(1) nội bộ — sẽ phải sửa
khi làm A3, đúng như review đã lưu ý.

**Torch** (tiny model, `durations=[[3,0,2]]`):
- `mel (1, 5, 20)`, `lengths [5]`, toàn bộ finite.
- `frame_positions = [0.0, 0.333, 0.667, 0.0, 0.5]` — finite (không chia 0 nhờ
  `span.clamp_min(1)` tại `models/vifasttts.py:90–91`; index clamp `N−1` tại `:31`).
- Frame của âm tiết duration=0 không được sinh ra; các frame sau thuộc âm tiết kế tiếp — đúng ngữ nghĩa.

**ONNX:** graph export chuẩn **không nhận `durations`** (tính nội bộ
`round(exp(log_d) − 1).clamp_min(1)` — `models/vifasttts.py:116`) nên không thể bơm 0. Để kiểm đúng
cơ chế, đã export một **wrapper nhận `(features, durations)`** rồi chạy onnxruntime với `[[3,0,2]]`:
- `mel (1,5,20)` khớp torch, `lengths [5]`, `max|diff| = 2,4e-7`, finite.
→ LengthRegulator xuất ONNX được và xử lý 0 đúng; **việc cần sửa khi thêm SIL là quy tắc clamp/đầu ra**
(SIL được phép 0 frame, hoặc đưa `durations`/scale thành input graph).

---

## R2-4 — split_manifest / reparse_overrides có giữ danh sách video val cố định không?

**KẾT LUẬN: GIẢ THUYẾT ĐÚNG — cả hai luôn CHIA LẠI; chỉ tất định khi tham số (val_videos, seed)
không đổi.** Không có cơ chế lock.

**Bằng chứng code:**
- `data/split.py:20–37` — `pick_val_videos`: chọn theo hash `sha1(f"{seed}:{video}")` (tất định với
  cùng seed + số lượng), `:40–48` `split_rows` chia lại từ đầu, `:51–96` `write_split` **ghi đè**
  `train.jsonl`/`val.jsonl`.
- `tools/split_manifest.py:30` — luôn gọi `write_split(...)` với tham số CLI.
- `tools/reparse_overrides.py:73` — gọi `write_split(manifest, root, val_videos=val_videos, val_ratio=0.05, seed=seed)`
  với **mặc định `--val-videos 4 --seed 13`** — tức nếu chạy reparse sau khi đã chia val thành 16 video
  mà không truyền tham số mới, nó sẽ **ghi đè về split 4 video cũ**.

**Hệ quả với kế hoạch:** phải (a) chia lại val + ghi `split_lock.json` **trước khi** v2 chạy, và
(b) sửa `write_split`/2 tool trên để **đọc lock và tôn trọng** danh sách video val (không tự chọn lại).
Lưu ý: 4 video val hiện tại là **4 video đầu trong thứ tự hash** của `seed=13`; nếu chọn 16 video
cùng seed, 4 video cũ vẫn nằm trong val (chỉ thêm 12 video mới) — không có video val nào quay lại train.

---

## R2-5 — Còn bao lâu đến khi cache xong và vocoder v2 tự khởi động?

**Trả lời tại thời điểm đo (15:49–16:05):**
- Tốc độ cache thực đo: **~4,9–5,7 file/giây** (172 file/30 s lúc đầu).
- ETA gốc: còn ~1.450 file train (~4–5 phút) + 1.791 file val (~5–6 phút) → **v2 sẽ tự chạy sau ~10 phút**.
- **Đã xử lý:** `kill -STOP 69421` (shell chuỗi nền) **ngay trước khi quá muộn**; cache train khi đó
  vẫn đang chạy và đã chạy xong bình thường.
- Hiện trạng: **cache train 11.424/11.424 (100%)**; shell ở trạng thái `Ts` (stopped), không còn tiến
  trình con; **cache val chưa chạy; v2 chưa tạo thư mục** → an toàn để sửa trước khi resume.

**Cách điều khiển:**
```bash
kill -CONT 69421    # chạy tiếp chuỗi: cache val (~6 phút) → vocoder v2 100k bước
# hoặc
kill 69421          # bỏ chuỗi; sau đó chủ động chạy cache val + train_vocoder_chunked bằng tay
pgrep -fl cache-features   # kiểm tra không còn tiến trình cache
```

---

## Xác nhận nhanh các đề xuất gấp của review (theo trạng thái repo)

| Đề xuất | Trạng thái xác minh |
| --- | --- |
| 1. Chia lại val (16 video × ≤150 clip) + `split_lock.json` + tool tôn trọng lock | **Chưa có gì** — hiện chia 4 video/1.791 clip (972·437·322·60); `write_split` luôn chia lại (R2-4). Cần làm **trước khi resume chuỗi**. |
| 2. Log gradient từng thành phần mỗi 100 bước vào `history_vocoder.csv` | **Chưa có** — `history_vocoder.csv` hiện ghi `step,g_total,g_adv,g_mel,g_stft,d_loss`; công thức `autograd.grad(w·loss_k, fake)` là rẻ và khả thi: `mel` đi qua `mel_fake` (tính từ `fake`), `adv/fm/stft` đi trực tiếp qua `fake_crop`. Lưu ý trên MPS: mel loss tính trên CPU rồi nhân ngược qua copy — gradient vẫn chảy tới `fake_crop` trên MPS. |
| 3. Chuẩn hoá F0 **không** làm vô hiệu cache | **Đúng cảnh báo** — khoá cache chứa nguyên mục `f0:` (`cache.py:44`). Đặt stats/chuẩn hoá ở mục config riêng, giữ `f0:` nguyên → cache 13.215 file (train 11.424 đã xong) không bị xoá. |
| 4. VocoderDataset & cache | Xem R2-1: dùng cache, 4 ms/batch khi ấm; chỉ miss mới tốn 1,36 s/clip. |
| 5. `best.pt` ở 12k không phải 8k | Đã xác nhận ở vòng 1 (val thấp nhất = 277,499 tại bước 12.000; `last.pt` 21k = 301,19). |

### Gợi ý thiết kế `split_lock.json` (khớp R2-4)
```json
{"version": 1, "seed": 13, "val_videos": 16, "max_clips_per_video": 150,
 "videos": ["5KnS0TGxWLg", "4kyqfOypVWI", "..."],
 "hash": "sha1(canonical json của {videos, max_clips_per_video})"}
```
`write_split` đọc lock nếu tồn tại → dùng đúng `videos` (và cap clip/video nếu có), chỉ chia lại
train/val từ manifest; xoá lock (`--force`) mới cho phép chia mới.

---

## Vận hành (để chuyển cho người/AI vận hành)

1. Đang dừng: shell `69421` (`Ts`) — chuỗi `cache val → vocoder v2` chưa chạy. Cache train đủ.
2. Thứ tự khuyến nghị trước khi resume:
   a. Chia lại val 16 video (≤150 clip/video) + ghi `split_lock.json` + test.
   b. Sửa `write_split` + `reparse_overrides`/`split_manifest` đọc lock.
   c. Thêm log gradient định kỳ 100 bước vào vocoder + test (không phá format history cũ: ghi header
      mới kèm cột `g_adv/g_fm/g_mel/g_stft` grad norms).
   d. (Tuỳ chọn) `pytest -q` + `ruff` + `mypy` rồi mới resume.
3. Resume: `kill -CONT 69421` → cache val → v2 tự chạy (100k bước, ~19–20h MPS).

*Trả lời bởi agent cầu nối. Khi v2 chạy ≥2k bước, có thể yêu cầu trích xuất log gradient/thanh phần
loss để chốt `mel_weight` theo quy tắc đã bàn.*
