# Pipeline dữ liệu

1. Ưu tiên video/WAV gốc; tải kênh bằng `youtube-fetch` (audio + phụ đề json3 làm timing).
2. Trích audio mono 24 kHz; làm sạch caption cuộn, gộp câu, cắt clip + nới biên (B3).
3. So transcript với SRT/VAD; cắt clip 1–20 s.
4. Lọc rẻ chất lượng clip (im lặng, clipping, RMS thấp, trùng chữ, fallback cao) — `tools/clip_quality.py` (B4).
5. QC từ lạ (OOV): `tools/fallback_review.py` → điền `filled.txt` → merge overrides → `--recut` (không tải lại).
6. **Lọc âm thanh không phải giọng chủ kênh** (TTS/AI/demo/nhạc/khách): `tools/diarize_filter.py`
   (NVIDIA Nemotron 3 Diarization, quy trình 2 lượt anchor theo từng video) → `drop_other_ge_0p5.txt`.
7. Chuẩn hoá loudness theo video: −26 dBFS trên frame có tiếng; clip lệch >±6 dB sửa riêng; chặn
   peak >−1 dBFS; **không ghi đè** bản gốc (xuất `<ds>_v2/`). *(đang hoàn thiện — REBUILD_PLAN_M6 §3)*
8. Tạo manifest + coverage report (`build-manifest`, `validate-manifest`).
9. Chia train/valid/test theo **video nguồn** + ghi `split_lock.json` để cố định val.
   *(lock đang hoàn thiện; hiện chia tất định bằng `tools/split_manifest.py`)*
10. Mel stats toàn cục (`tools/compute_mel_stats.py`) — biểu diễn mel "đóng băng" bằng hash trong checkpoint.
11. (Tùy chọn) `cache-features` tính trước mel/f0/vuv (+ phonation).
12. Siết chính tả nguồn: `build-lexicon` + `tools/parser_audit.py --profile strict|corpus`.
13. **Bắt buộc theo thứ tự:** mọi thay đổi overrides phải reparse **TRƯỚC** khi chạy MFA.
14. Forced alignment MFA trên corpus sinh từ parser (`tools/mfa_build_corpus.py`); train subset sạch
    trước rồi align toàn bộ.
15. Converter TextGrid → `durations` cấp âm tiết + SIL: quy đổi biên tích luỹ, assert `sum == T`.
    *(đang hoàn thiện)*
16. Cắt segment 3–10 s tại SIL (ưu tiên cuối câu, khoảng lặng ≥100 ms; lưu `(clip_id, frame_start,
    frame_end)` và cắt trực tiếp từ cache). *(đang hoàn thiện)*
17. F0 stats (mean/std log-F0 trên frame hữu thanh, split train) cho chuẩn hoá + DCT4. *(đang hoàn thiện)*
18. (Tùy chọn) Bật `phonation.enabled: true` để trích thêm HNR/spectral tilt/glottalization cấp frame
    (nghiên cứu chất giọng); nhớ tính lại cache sau khi đổi cấu hình.

Không train trực tiếp bằng SRT chưa căn chỉnh.

Manifest do `build-manifest` sinh ra chứa `audio_path`, `text` và danh sách `syllables` đã parse; có thể bổ sung trường `durations` (số frame mỗi âm tiết) từ forced alignment ở bước 14–15 — vòng train đọc trường này nếu có.

## Nguồn: kênh YouTube của bạn (tùy chọn)

Nếu dữ liệu nằm trên kênh YouTube bạn sở hữu, dùng `vifasttts youtube-fetch` để chạy sẵn các bước
1–3 ở mức câu: tải audio + phụ đề json3 → làm sạch caption cuộn → cắt clip 1–20 giây có fade →
`metadata.csv` + `manifest.jsonl` + SRT mở + dataset card (giấy phép mặc định CC BY 4.0). Các bước
4–5 dùng `tools/clip_quality.py` + `tools/fallback_review.py`; bước 6 dùng
`tools/diarize_filter.py` (lọc TTS/AI/demo/khách tự động, NVIDIA Nemotron 3 Diarization). Forced
alignment (bước 14–15) vẫn chạy riêng để có `durations` cấp âm tiết. **Quy trình tái hiện đầy đủ
kèm lệnh**: `docs/YOUTUBE_DATASET.md` mục 5 và 9; số liệu thực: `docs/R6_DIARIZATION_RESULTS.md`.
