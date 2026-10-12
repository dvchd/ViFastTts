# R6 — Lọc âm thanh không phải giọng chủ kênh (NVIDIA Nemotron 3 Diarization)

> **TL;DR:** đã tự động hoá hoàn toàn bước "loại giọng khác/TTS/demo/nhạc" cho dataset một-giọng.
> Model **NVIDIA Nemotron 3 Diarization** (23/09/2026) chạy native qua `transformers` trên CPU Mac
> mini M6. Pool chủ kênh (25 video) sau lọc: **12.230 clip · 47,48h**; chỉ **48 clip** bị loại vì
> chứa **0,090h** âm thanh khác (0,19%). Không cần nghe thủ công.
>
> **Bài học quan trọng (v2):** anchor lấy từ **video khác** gây báo nhầm giọng chủ kênh thành
> "người khác" khi khác mic/phòng — ở lượt đầu đã báo nhầm **134 clip**. Vì vậy quy trình chuẩn
> **bắt buộc 2 lượt**, lượt cuối dùng **anchor lấy từ chính video của clip**.

---

## 1. Model & cách chạy

| | |
| --- | --- |
| Model | `nvidia/Nemotron-3-Diarization` (release 23/09/2026; open weights, OpenMDW-1.1) |
| Kiến trúc | 100M tham số, 31-layer Transformer encoder (streaming Sortformer), tối đa 8 người nói |
| Chạy | `transformers>=5.19` (native) + `librosa`; CPU; không cần stack NeMo |
| Cài | `pip install -e ".[diar]"` (extra mới trong `pyproject.toml`; torch lấy từ extra `train`) |
| Tốc độ đo | ~15,4 clip/s trên Mac mini M6 (một lượt toàn bộ 12–13k clip ≈ **13–15 phút**) |

## 2. Nguyên lý & vì sao phải anchor theo từng video

Model trả về hoạt động của tối đa 8 "kênh speaker" xếp theo **thứ tự xuất hiện trong audio**.
Với một clip đơn lẻ thì không biết ai là chủ kênh. Cách định danh:

1. Ghép **anchor ~6s giọng chủ kênh** lên trước clip → kênh `spk0` luôn là chủ kênh (nói trước).
2. Mọi `spk>0` xuất hiện trong vùng clip = **âm thanh không phải chủ kênh** (TTS/AI/demo/khách),
   kèm mốc thời gian chính xác để cắt nếu cần.

**Vì sao anchor phải theo từng video:** giọng chủ kênh ở video dùng mic/phòng khác (RMS lệch tới
~22 dB giữa các video) có thể bị model coi là người khác nếu anchor lấy từ video khác điều kiện.
Thực đo trên kênh tham chiếu: lượt A (anchor toàn cục từ 1 video) báo 149 clip "có tiếng khác",
nhưng khi kiểm lại bằng anchor cùng video thì **134 clip là giọng chủ kênh** (giữ đúng); chỉ
**15 clip** thực sự là giọng khác. Vì vậy quy trình chuẩn gồm 2 lượt như dưới.

## 3. Quy trình chuẩn (2 lượt) + lệnh tái hiện

```bash
pip install -e ".[diar]"

python tools/diarize_filter.py \
  --manifest data/codelung/manifest.jsonl \
  --anchor-video EM_ObHWvd94 \
  --exclude-video KIrbA-wEURg,LY8_RaLDJT4,W1m6birJBj0,_U4H5FSIb-c,nEPr-jIokLw,pLXBrJ-M-Xo \
  --out-dir data/codelung/review/diar
```

- **Lượt A (anchor toàn cục):** chọn 2 clip sạch của `--anchor-video`, ghép 6s làm anchor, diarize
  toàn bộ → `pass_a.jsonl`. Mục đích: tìm clip sạch để chọn anchor riêng cho mỗi video.
- **Lượt B (anchor theo video):** mỗi video dùng anchor ~6s ghép từ 2 clip sạch **của chính video
  đó** (tiêu chí: `other_s<0,05`, `owner_s>2s`, `unlabeled_frac<0,3`, ưu tiên clip ~12s) → kết quả
  cuối `clip_other.jsonl` + `drop_other_ge_0p5.txt` + `summary.json`.
- Script **resume được** (chạy lại bỏ qua clip đã có), ghi mọi thứ ra JSONL.

**Tiêu chí kiểm chứng đi kèm (đã đạt trên kênh tham chiếu):**

| Kiểm tra | Ngưỡng | Kết quả thực đo |
| --- | --- | --- |
| Anchor có đúng là `spk0` (dominant + độ phủ) | 100% clip dom=0; cov ≥0,95 | **12.278/12.278 dom=0**; cov min 0,945 (1 clip, kiểm lại bằng anchor dự phòng → giữ) |
| Clip dài hơn cửa sổ anchor | anchor 6s + clip ≤ 27,2s | 300 clip >20s nhưng **max 20,6s** → 6+20,6=26,6s ✓, không clip nào vượt |
| Anchor khác điều kiện thu âm | re-check bằng anchor cùng video | 134/149 clip đổi về đúng giọng chủ kênh; 15 còn lại (50sw) là TTS thật |
| Tái lập | chạy lại cho cùng kết quả | 13.215/13.215 clip khớp `other_s` tuyệt đối giữa 2 lần chạy |

## 4. Kết quả cuối cùng (pool 25 video chủ kênh)

| Ngưỡng `other_s` | Số clip | Thời lượng clip | Thời lượng lạ |
| ---: | ---: | ---: | ---: |
| ≥ 0,2s | 52 | 0,217h | 0,090h |
| **≥ 0,5s (chốt)** | **48** | **0,201h** | **0,090h** |
| ≥ 1,0s | 36 | 0,161h | 0,087h |

- Tổng tiếng khác trong pool: **0,090h (0,19%)** — so với 3,45h toàn dataset (93% nằm ở 6 video
  loạt khách đã loại).
- Phân bố 48 clip cần loại: `50swDIypplU` 35 (TTS thiết bị IoT "Tiểu Trí/XiaoZhi") ·
  `8Pq2FwmIEU8` 8 · `nh4wz4wlK4Q` 2 · `5KnS0TGxWLg` 2 · `EM_ObHWvd94` 1; 11 clip hoàn toàn
  không có giọng chủ kênh.
- Loạt "Học Từ Thiện" bị model bắt 70–86% thời lượng là người khác (kể cả KIrbA/nEPr mà F0 không
  phân biệt được) → bỏ là đúng.
- Kiểm tra "âm thanh không-lời-nói" (`unlabeled_frac` cao): top clip đều là **câu ngắn/reaction
  của chính chủ kênh** ("này.", "Không có.", "Ê"…) — không lộ nhạc/hiệu ứng hệ thống.

**Kết quả chốt:**

```text
Pool chủ kênh ban đầu : 12.278 clip · 47,68h
Loại (other_s ≥ 0,5s) :     48 clip · 0,090h tiếng lạ
Pool cuối             : 12.230 clip · 47,48h   ← dùng cho vocoder + tách split
Ước lượng acoustic    : 32,3h sau khi bỏ clip fallback (3.507 clip)
                        ~29h sau lọc MFA (~10%) — sẽ chốt bằng số thật
```

## 5. Artifact & công cụ

| | |
| --- | --- |
| Tool tái hiện | `tools/diarize_filter.py` (+ extra `[diar]` trong `pyproject.toml`) |
| Kết quả kênh tham chiếu | `data/codelung/review/diar/clip_other.jsonl` (mọi clip: owner_s/other_s/segments), `drop_other_ge_0p5.txt` (48 id), `summary.json` |
| Thời gian chạy | ~13–15 phút/lượt trên M6 CPU (2 lượt ~30 phút cho 13k clip) |
| Ghi chú | Model tách *người nói*; chưa tách *nhạc nền dưới tiếng nói* — nếu cần, thêm bộ phát hiện nhạc (spectral flatness, inaSpeechSegmenter) ở vòng sau. |

## 6. Ảnh hưởng tới kế hoạch

1. Bỏ hẳn phương án nghe thủ công và phương án ECAPA — thay bằng quy trình 2 lượt ở trên.
2. `REBUILD_PLAN_M6.md` cập nhật: pool cuối **12.230 clip/47,48h**; acoustic ước lượng **32,3h → ~29h**.
3. Split/R6-3 chọn val trên pool cuối này; mel stats + cache + vocoder v2 tính lại sau khi lọc.
