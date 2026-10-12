# R6 — Tách & lọc giọng tự động bằng NVIDIA Nemotron 3 Diarization

> **TL;DR**: Đã tự động hoá hoàn toàn bước "lọc âm thanh không phải giọng chủ kênh" — **không cần
> nghe thủ công**. Model NVIDIA Nemotron 3 Diarization (23/09/2026) chạy trên Mac mini M6 CPU: xử lý
> **13.215 clip / 51,7h trong 14,3 phút**. Kết quả: pool chủ kênh (25 video) chỉ có **0,22h** âm thanh
> lạ, tập trung ở **149 clip** — đã xuất danh sách loại trừ.
> Thay thế hoàn toàn phương án "xuất 30 clip cho chủ kênh nghe" và bỏ phương án ECAPA.

---

## 1. Model & cách dùng

| | |
| --- | --- |
| Model | **NVIDIA Nemotron 3 Diarization** (`nvidia/Nemotron-3-Diarization`, 23/09/2026) |
| Quy mô | 100M tham số, 31-layer Transformer encoder, tối đa 8 người nói, top Diarization-Bench |
| Chạy | `transformers 5.19` (native, không cần stack NeMo), CPU, fp32 |
| Tốc độ đo được | RTF ≈ 200–340× một clip; **15,4 clip/s** cho toàn dataset (Mac mini M6, CPU) |
| Giấy phép | OpenMDW-1.1 (thương mại được) |

**Kỹ thuật định danh (điểm mấu chốt):** nhãn speaker của model là *thứ tự xuất hiện trong từng đoạn
audio*, nên với 1 clip đơn lẻ không biết ai là chủ kênh. Giải pháp: **ghép 6 giây "anchor" giọng chủ
kênh lên trước mỗi clip** rồi diarize `[anchor + clip]` → kênh `spk0` luôn là chủ kênh (xuất hiện
đầu tiên), mọi `spk>0` trong vùng clip là **âm thanh không phải giọng bạn** (TTS/AI/demo/nhạc/khách),
kèm **mốc thời gian chính xác** để cắt nếu muốn. Tổng độ dài ≤ 26s (< cửa sổ offline 27,2s) nên
không mất định danh giữa chừng.

## 2. Kiểm chứng trên 9 ca đã biết (trước khi chạy toàn bộ)

| Ca | F0 | Kết quả diarize | Kết luận |
| --- | ---: | --- | --- |
| EM_ObHWvd94 (chủ kênh) | 127 | chỉ spk0 | ✓ |
| `6GXw85XfiKo_0348` | 173 | **chỉ spk0** | giọng bạn, cao giọng — GIỮ |
| `o8RjAmBtyZo_0019` | 163 | **chỉ spk0** | giọng bạn — GIỮ |
| `50swDIypplU_0146` | 265 | spk0 (bạn) + **spk1 (12,9–27,9s)** | TTS thiết bị IoT — loại |
| `50swDIypplU_0134` | 269 | **spk1** gần hết + spk0 cuối | TTS + bạn trộn |
| `_U4H5FSIb-c_0017` (loạt khách) | 195 | **spk1** toàn clip | khách — xác nhận |
| `_U4H5FSIb-c_0128` (loạt khách) | 116 | **spk1** | khách (pitch thấp) — xác nhận |
| KIrbA/nEPr (loạt khách, F0 giống bạn ~123–133) | — | **spk1 70–83%** | model bắt được khách mà F0 bó tay |

→ F0 cao chưa chắc là người khác (có thể là bạn hào hứng/đọc game); **diarization mới là bằng chứng**.

## 3. Kết quả toàn bộ dữ liệu (13.215 clip · 51,7h)

- Tổng thời lượng không phải chủ kênh: **3,45h (6,67%)** — nhưng **93% nằm ở 6 video loạt khách** đã chốt bỏ.
- **Pool chủ kênh (25 video · 12.278 clip · 47,68h): chỉ 0,22h** âm thanh lạ.

| Ngưỡng `other_s` | Số clip | Thời lượng clip | Thời lượng lạ |
| ---: | ---: | ---: | ---: |
| ≥ 0,05s | 167 | 0,571h | 0,219h |
| ≥ 0,50s | **149** | **0,511h** | **0,218h** |
| ≥ 1,00s | 125 | 0,447h | 0,213h |
| ≥ 2,00s | 89 | 0,371h | 0,199h |

- Trong 149 clip (≥0,5s): **53 clip hoàn toàn không có giọng bạn**; còn lại trộn bạn + demo/TTS.
- Tập trung nhất: `50swDIypplU` (IoT "Tiểu Trí/XiaoZhi" phát TTS), `fx303iiDb7M_0460`,
  `X_65x81R4KY_0467` (demo game), rải rác `vEtfty4Cc6Y`, `o8RjAmBtyZo`, `6GXw85XfiKo`…
- 9/31 video **không có đoạn lạ nào**; hầu hết video chủ kênh nhiễm <1,6%.

**Quy tắc chốt đề xuất:** sau khi bỏ 6 video loạt khách, **loại 149 clip có `other_s ≥ 0,5s`**
(mất 0,51h trong đó chỉ 0,22h là tiếng lạ) → còn **12.129 clip · 47,17h sạch giọng chủ kênh**.
Với clip trộn, mốc thời gian lạ đã có sẵn trong `clip_other.jsonl` để **cắt tinh** ở bước segment
(MFA) nếu muốn giữ thêm ~96 clip.

## 4. Artifact đã lưu

| File | Nội dung |
| --- | --- |
| `data/codelung/review/diar/clip_other.jsonl` | Toàn bộ 13.215 kết quả: `owner_s`, `other_s`, `other_ratio`, danh sách segment `[start, end, speaker]` |
| `data/codelung/review/diar/drop_other_ge_0p5.txt` | 149 id đề xuất loại (pool chủ kênh) |

Công cụ: `transformers 5.19` + `librosa` (cài trong `.venv` dạng tool phân tích, không vào
`pyproject.toml`); model tự tải về cache HuggingFace (~400MB). `.venv-speaker` + model CAM++ cũ
**không còn cần** — có thể xoá.

## 5. Ảnh hưởng tới kế hoạch rebuild

1. Bỏ phương án ECAPA và bỏ bước nghe 30 clip thủ công — thay bằng pipeline diarization ở trên
   (chạy lại được bất cứ lúc nào, ~15 phút).
2. Bước (2) trong `REBUILD_PLAN_M6.md` nay là: **drop loạt khách + drop 149 clip** → pool 12.129 clip/47,17h.
3. Hạn chế còn lại: model tách *người nói*, chưa tách *nhạc nền dưới tiếng nói*; nếu về sau thấy
   BGM lọt vào, thêm bộ phát hiện nhạc (spectral flatness/inaSpeechSegmenter) — chưa cần ở vòng này.
4. Các bước sau giữ nguyên: loudness theo video → tìm val tất định (R6-3) → split lock → mel stats →
   cache → vocoder v2; song song: overrides → reparse → MFA → converter + SIL → segment → F0 stats.
