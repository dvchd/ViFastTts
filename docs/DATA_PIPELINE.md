# Pipeline dữ liệu

1. Ưu tiên video hoặc WAV gốc.
2. Trích audio mono.
3. Dùng SRT làm vùng thời gian sơ bộ.
4. VAD và ASR có timestamp.
5. So sánh transcript với SRT.
6. Forced alignment.
7. Cắt 2 đến 12 giây.
8. Loại clip có người khác, nhạc, clipping hoặc transcript sai.
9. Chuẩn hóa sample rate một lần.
10. Tạo manifest và coverage report.
11. Chia train/validation/test theo video nguồn.
12. Chạy `vifasttts validate-manifest` để kiểm tra WAV tồn tại và âm tiết trước khi train.

Không train trực tiếp bằng SRT chưa căn chỉnh.

Manifest do `build-manifest` sinh ra chứa `audio_path`, `text` và danh sách `syllables` đã parse; có thể bổ sung trường `durations` (số frame mỗi âm tiết) từ forced alignment ở bước 6 — vòng train đọc trường này nếu có.
