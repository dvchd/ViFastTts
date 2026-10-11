# Dataset từ kênh YouTube của bạn (`youtube-fetch`)

> Lấy **audio + phụ đề** từ chính kênh của bạn → **clip sạch 24 kHz mono** + `manifest.jsonl`
> + **SRT mở cho mọi người** + dataset card có giấy phép. Chạy được cả **demo 1 video**
> lẫn **chạy thật cả kênh**, chạy lại an toàn (resume).
>
> Kênh tham chiếu của dự án: **Code Lủng/Code Bug `@codelung`** — chủ kênh công bố dữ liệu
> audio mở, ai dùng cũng được (mặc định ghi giấy phép **CC BY 4.0**).

## 1. Vì sao hợp lệ?

- Bạn là **chủ kênh**: tải và tái sử dụng nội dung của mình là hợp lệ; repo không phân phối
  lại video gốc, chỉ tạo dataset clip theo giấy phép bạn chọn (`--license`).
- Chỉ dùng cho **nội dung bạn sở hữu** hoặc có quyền; điều khoản của YouTube vẫn áp dụng
  cho việc tải (xem `docs/LICENSES.md` mục "Dữ liệu YouTube tự tải").
- Công cụ nằm ngoài package: gọi `yt-dlp` + `ffmpeg` qua subprocess, không nhúng code
  bên thứ ba vào ViFastTts.

## 2. Cài đặt

```bash
pip install -e ".[youtube]"    # yt-dlp + imageio-ffmpeg (ffmpeg tích hợp, khỏi brew)
```

Nếu muốn dùng ffmpeg hệ thống: `brew install ffmpeg`, hoặc đặt
`VIFASTTTS_FFMPEG=/đường/dẫn/ffmpeg`. Tương tự `VIFASTTTS_YTDLP` cho yt-dlp.

## 3. Lệnh demo (1 video) và chạy thật (cả kênh)

```bash
# Demo: chọn 1 video ngắn 60–900 giây, chạy hết luồng trong vài phút
vifasttts youtube-fetch "https://www.youtube.com/@codelung/videos" --out data/codelung --demo

# Chạy thật cả kênh (nghỉ 1,5 giây giữa các request cho lịch sự)
vifasttts youtube-fetch "https://www.youtube.com/@codelung/videos" \
  --out data/codelung --sleep 1.5 --license "CC BY 4.0"

# Xem trước danh sách video, chưa tải gì
vifasttts youtube-fetch "https://www.youtube.com/@codelung/videos" --out data/codelung --dry-run

# Chạy tiếp nếu bị ngắt — video đã xong được bỏ qua tự động
vifasttts youtube-fetch "https://www.youtube.com/@codelung/videos" --out data/codelung
```

Kết quả cuối in dạng "Tổng kết": số video xử lý, số clip, tổng giờ, số bị lọc, số âm tiết
theo coverage. Dataset card ở `data/codelung/dataset_card.md`.

> **Lần chạy thật tham chiếu** (kênh `@codelung`, 12/10/2026): 31 video → **13.217 clip ·
> 51,05 giờ · fallback 1,40% âm tiết (7,1% trước khi làm sạch) · 68% clip 0 fallback ·
> đủ 6 thanh/11 coda · 0 vấn đề manifest**; `train.jsonl`/`val.jsonl` chia theo video
> (44,37h / 6,68h) bằng `tools/split_manifest.py`.

## 4. Luồng hoạt động

```text
yt-dlp        liệt kê video (cache .videos.json, lọc độ dài/tiêu đề)
yt-dlp        tải phụ đề json3 (timing chính xác) ──▶ cues sạch (bỏ [Music], cắt caption cuộn)
yt-dlp+ffmpeg bestaudio ─▶ raw/<id>.wav (mono 24 kHz PCM16)  [giữ để căn chỉnh lại]
gộp câu       max_gap / max_duration / max_chars ──▶ câu có [start, end]
cắt clip      ±pad 60/120 ms + fade 10/20 ms + lọc im lặng/trùng/fallback cao
xuất          audio/<id>_NNNN.wav · metadata.csv · manifest.jsonl · subs/<id>.srt
              stats.json · dataset_card.md · LICENSE-DATASET.txt · .done.json
```

Chi tiết làm sạch:

- **Caption cuộn** của YouTube (event sau chứa lại chữ event trước) được cắt phần lặp,
  nên không bị lặp chữ trong transcript.
- Bỏ nhãn tiếng động `[Music]`, `[Âm nhạc]`, `(music playing)`, `♪`, mũi tên `>>`,
  tag `<c>`/karaoke, `&nbsp;`; giữ nội dung thật trong ngoặc như `(phần 1)`.
- **Gộp câu**: khoảng nghỉ ≤ `--max-gap` (mặc định 0,35 s) thì gộp; dừng gộp khi câu trước
  đã hết câu (`. ! ? …`) và nghỉ đủ lớn — tránh dính hai câu vào một clip.
- **Lọc**: dưới `--min-duration` (1 s), im lặng dưới `--min-rms-db` (-45 dBFS), trùng chữ
  (`--dedupe`), fallback > `--max-fallback-ratio` (0,34 — chủ yếu là câu tiếng Anh/typography lạ),
  và **video phụ đề quá thưa** < `--min-caption-density` (2 ký tự/s — video nhiều nhạc/ít lời;
  phụ đề nói thật thường 11–16 ký tự/s).
- **Text** trong `metadata.csv` đã chuẩn hoá số/ngày/giờ theo vùng (`--dialect`), đúng dạng
  "đọc lên" để train; SRT giữ dạng chữ người đọc.
- **Từ lạ (OOV)**: bảng chuẩn frontend phủ sẵn ~580 từ công nghệ + 31 acronym + tên chữ cái
  (`code→cốt`, `user→iu dơ`, `b→bê`...); phần còn lại in top từ lạ + ghi `oov_report.json`
  để điền vào `--overrides` (mẫu: `configs/overrides_tech_vi.json` — `clot→cờ lốt`,
  `clon→cờ lon`, `grock→gờ rốc`...). Clip quá nhiều từ lạ bị lọc bằng `--max-fallback-ratio`.
- **Mã chữ-số** (`--respell-codes`, mặc định bật): tách chữ-số dính liền và đọc lại —
  `Chương2→Chương hai`, `MP3→em pê ba`, `3D→ba dê`, `v2→vê hai`, `500k→năm trăm ca`,
  `n8n→en tám en`. Tắt bằng `--no-respell-codes` nếu muốn giữ nguyên mã.
- **Lỗi tạm thời** (HTTP 429/5xx, timeout) được thử lại 3 lần với backoff 6 s/12 s rồi mới
  bỏ qua video; chạy lại lệnh sẽ tiếp tục từ chỗ dừng (`.done.json`).

## 5. Tuỳ chọn chính

| Tuỳ chọn | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `<channel>` | — | URL kênh (`/videos`), playlist hoặc 1 video |
| `--out` | `data/youtube` | Thư mục dataset |
| `--limit` | 0 (tất cả) | Số video tối đa |
| `--demo` | tắt | Thử lần lượt video ngắn nhất (tối đa 5) tới khi có ít nhất 1 clip |
| `--langs` | `vi,vi-orig` | Phụ đề ưu tiên, cách nhau dấu phẩy |
| `--dialect` | `north` | Cách đọc số/ngày: north/central/south |
| `--min-duration` / `--max-duration` | 1 / 20 giây | Độ dài clip |
| `--max-gap` / `--max-chars` | 0,35 s / 220 | Điều kiện gộp câu |
| `--min-video-duration` / `--max-video-duration` | 0 / 0 | Lọc video theo độ dài (0 = không giới hạn) |
| `--title-filter` | — | Regex lọc tiêu đề video |
| `--keep-wav/--no-keep-wav` | giữ | Giữ WAV gốc để căn chỉnh lại sau |
| `--dedupe/--no-dedupe` | lọc | Lọc clip trùng chữ |
| `--min-rms-db` | -45 | Bỏ clip im lặng |
| `--max-fallback-ratio` | 0,34 | Bỏ clip quá nhiều token lạ |
| `--min-caption-density` | 2 ký tự/s | Bỏ video phụ đề quá thưa (nhiều nhạc/ít lời) |
| `--overrides` | — | File JSON `{"mặt chữ": "cách đọc"}` cho từ lạ (OOV) |
| `--respell-codes/--no-respell-codes` | bật | Làm sạch mã chữ-số (`v2→vê hai`, `MP3→em pê ba`) |
| `--license` | `CC BY 4.0` | Giấy phép ghi vào dataset card (ví dụ `CC0 1.0`) |
| `--sleep` | 1,0 s | Nghỉ giữa các request |
| `--dry-run` / `--refresh` | tắt | Xem trước không tải / bỏ cache danh sách video |
| `--recut` | tắt | Xoá kết quả cũ, cắt lại từ `raw/` + `subs/` (không tải lại) — dùng sau khi sửa overrides |

## 6. Cấu trúc output

```text
data/codelung/
  audio/<video_id>_<NNNN>.wav   clip 24 kHz mono PCM16, fade đầu/cuối
  subs/<video_id>.srt           phụ đề sạch đã gộp câu (mở cho mọi người)
  subs/<video_id>.json3         phụ đề gốc tải về (để đối chiếu)
  raw/<video_id>.wav            audio gốc 24 kHz (nếu giữ)
  metadata.csv                  id|text  (định dạng cho build-manifest)
  manifest.jsonl                text + âm tiết đã parse (đưa thẳng vào train)
  clips.jsonl                   bản ghi chi tiết từng clip (start/end/rms/fallback)
  stats.json                    thống kê + coverage + top OOV
  oov_report.json               từ lạ cần điền vào overrides
  dataset_card.md               nguồn, giấy phép, cách tái tạo, thống kê
  LICENSE-DATASET.txt           giấy phép dữ liệu
  .done.json / .videos.json     trạng thái resume + danh sách video
```

## 7. Giấy phép mở

- Mặc định ghi **CC BY 4.0** (ai dùng cũng được, chỉ cần ghi nguồn kênh).
- Muốn "mở hoàn toàn không điều kiện": `--license "CC0 1.0"`.
- Dataset card tự ghi nguồn kênh, thời điểm sinh, công cụ và câu lệnh tái tạo — đủ để
  người khác trích dẫn.

## 8. Sau khi lấy: kiểm tra và train

```bash
# kiểm tra WAV tồn tại + âm tiết, in coverage thanh/coda/fallback
vifasttts validate-manifest --manifest data/codelung/manifest.jsonl --wav-dir data/codelung/audio

# (khuyến nghị) siết chính tả nguồn bằng lexicon sinh từ chính dataset
vifasttts build-lexicon --manifest data/codelung/manifest.jsonl --output data/codelung/lexicon.json

# tính trước feature rồi mới train
vifasttts cache-features --manifest data/codelung/manifest.jsonl \
  --config configs/base_24k.yaml --cache-dir cache/codelung

# thấy oov_report.json có nhiều từ lạ thì điền cách đọc rồi cắt lại từ cache (không tải lại)
# ví dụ overrides: {"code": "cốt", "copy": "cô pi"} — mẫu: configs/overrides_tech_vi.json
vifasttts youtube-fetch "https://www.youtube.com/@codelung/videos" --out data/codelung \
  --overrides configs/overrides_tech_vi.json --recut

# báo cáo dataset + chia train/val theo video (tránh rò rỉ)
python tools/dataset_report.py --dataset data/codelung
python tools/split_manifest.py --manifest data/codelung/manifest.jsonl \
  --out-dir data/codelung --val-videos 4
```

## 9. Giới hạn (đọc trước khi train)

- **Thời lượng là cấp câu** theo phụ đề YouTube, chưa phải cấp âm tiết. Muốn chất lượng cao,
  chạy **forced alignment (MFA)** trên `raw/` rồi bổ sung trường `durations` — xem
  `docs/DATA_PIPELINE.md` bước 6 và `docs/TRAINING.md`.
- Phụ đề **tự động** có thể sai chính tả hoặc nhận dạng nhầm; các clip fallback cao đã bị
  lọc nhưng vẫn nên nghe thử một mẫu.
- Video nhiều **nhạc/tiếng động** hoặc **nhiều người nói** không được tách tự động — có thể
  lọc bằng `--title-filter` hoặc cắt thủ công.
- Chia **train/val theo video nguồn** (cột `video` trong `clips.jsonl`) để tránh rò rỉ.
- Video rất dài sẽ nạp cả WAV vào RAM (~5,8 MB/phút, ≈ 350 MB/giờ ở 24 kHz float32) —
  dùng `--max-video-duration` nếu máy ít RAM.

## 10. Roadmap

- ⬜ Căn chỉnh từ mức (whisper word timestamps / MFA) để có `durations` cấp âm tiết.
- ⬜ Lệnh xuất/đẩy dataset lên Hugging Face (`dataset_card.md` đã sẵn sàng).
- ⬜ Phát hiện nhạc/nhiều người nói bằng VAD + speaker embedding trước khi cắt.
