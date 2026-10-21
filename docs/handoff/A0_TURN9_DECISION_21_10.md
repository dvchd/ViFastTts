# ViFastTts: Quyết định Agent 0 gửi Agent 2 (21/10/2026, lượt 9, bàn giao phiên mới)

File này tự chứa đủ để một phiên Agent 2 mới chạy tiếp. Đọc kèm v8 nếu cần chi tiết.

## 0. Trạng thái đã chốt (không làm lại)

- Gốc: v3.2 small best `runs/acoustic_v32/best.pt` (phương án lùi).
- Ứng viên production: **P2 @2k EMA** (adv 0,3, fm 0, hinge, decoder + postnet, encoder đóng băng).
  Checkpoint gộp dùng được: `artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt`.
  Gốc run: `runs/adv_p2_20k/ckpt_2000.pt`.
- α = 0 ở mọi nơi (đã bỏ). Không dùng lại.
- Đã đóng: GTA vocoder, hop 128, adaptive adv, DNSMOS/CER làm cổng, re-align MFA, RNNoise/codec.
- P1 và P2 sau 2k: chỉ làm đối chứng. Không train GAN thêm nếu chưa sửa độ to (6c).

## 1. Thứ tự việc (đồng ý thứ tự Agent 2 đề xuất, thêm 6f)

1. **6b**: mốc drift giọng thật + 40 câu + CI bootstrap. Cổng mới như v8 §2.
   Lưu ý: chunk của audio thật là segment cắt tại gap > 0,64 s, còn chunk của model là theo dấu câu.
   Ghi rõ số chunk/câu ở cả hai phía; nếu chênh nhiều thì báo drift chuẩn hoá theo số chunk (max-min chia cho căn số chunk) bên cạnh số thô.
2. **6a**: ONNX `v32_adv_p2_2k`, md5 x3, md5 theo thread 1/2/4, RTF 2 luồng (acoustic/vocoder riêng).
3. **6f (MỚI, rẻ, làm cùng 6a)**: chuẩn hoá loudness CẢ FILE ở đầu ra.
   - Đầu ra hiện ~-26 dBFS, VieNeu ~-16 dBFS (chênh ~10 dB, tai nghe nhỏ bằng một nửa). Đây là mức của dữ liệu, không phải lỗi model.
   - Sau Vocos (và sau 6d nếu bật): đo LUFS cả file (BS.1770) rồi nhân một hệ số để về **-16 LUFS**,
     true-peak limiter **-1 dBTP**. Tham số `--target-lufs` (mặc định -16, cho phép tắt).
   - Cài trong `infer.py` và đường `synthesize-onnx`. Một hệ số cho cả file, không chỉnh từng chunk.
   - Kiểm tra: md5 x3 vẫn giống; đo LUFS 20 câu sau xử lý; không mẫu nào clip.
   - Trang nghe so sánh vẫn cân -23 LUFS cho mọi bản như cũ.
   - Thước drift (6b, 6c, 6d) đo TRƯỚC bước 6f (6f không đổi drift vì là một hệ số chung).
4. **6c**: E0 đến E3 từ P2 @2k, như v8 §3.
5. **6d**: chuẩn hoá theo chunk ở runtime, như v8 §4, mặc định TẮT.
   Thứ tự xử lý khi bật: 6d (từng chunk, trên log-mel) rồi Vocos rồi 6f (cả file).
6. **6e**: 10 top spike, chỉ báo cáo.

## 2. Cảnh báo khi đọc số

- Cổng dùng cột EMA, không dùng LIVE.
- P2 @2k qua cổng drift sát nút (4,67 so với 5,34), một phần nhờ EMA còn ~18% trọng số v3.2.
- 20 câu: p90 gần như là câu tệ thứ hai. Chỉ dùng cổng sau khi có 40 câu + CI.
- Quyết định cuối thuộc về tai người dùng (trang nghe mù, -23 LUFS, xáo thứ tự).

## 3. Báo lại cho Agent 0

6b (drift thật, bảng 40 câu + CI), 6a + 6f (md5, RTF, LUFS sau xử lý), 6c (bảng E0 đến E3), 6d (drift có/không + link nghe), 6e (ảnh).
