# ViFastTts: Quyết định Agent 0 gửi Agent 2 (21/10/2026, lượt 8)

Căn cứ: báo cáo lượt 6 + mục 6 (Việc 5b, drift giữa mệnh đề).

## 0. Nhận định

1. Thước drift đã bắt được lỗi tai nghe thấy. H2 xác nhận: GAN làm mức to tuyệt đối giữa các chunk trôi theo số bước,
   P2 trôi nhanh hơn P1 (adv 0,3 so với 0,1). Dừng P2 20k theo hard-stop: đúng.
2. **P2 @2k là ứng viên production tạm**: đồng ý. Nhưng nó đã trôi một phần (p90 4,67 so với v3.2 3,34, tức +1,3 dB,
   còn cách cổng 0,67 dB). Nó qua cổng chủ yếu nhờ EMA còn giữ khoảng 18% trọng số v3.2. Không có dư địa để train
   thêm GAN mà không sửa độ to.
3. Hai điểm yếu của thước drift cần sửa trước khi dùng làm cổng chính:
   - **Chưa có mốc giọng thật.** Cổng đang so với v3.2, nhưng chưa biết người thật đọc nhiều mệnh đề thì drift tự nhiên bao nhiêu.
     Nếu thật cũng ~4 dB thì v3.2 đang đều quá mức (đọc phẳng), không phải chuẩn.
   - **Mẫu quá nhỏ.** 20 câu, p90 gần như là câu tệ thứ hai. Cần >= 40 câu + khoảng tin cậy bootstrap.
4. P1 qua cổng ở mọi mốc nhưng xấu dần, và tai đã chọn P2 @2k. P1 chỉ làm đối chứng, không chạy tiếp.
5. Báo cáo chưa có: 5a (ONNX md5 + RTF), 5c (E1/E2), 5d (chuẩn hoá runtime), 5e (top spike). Làm tiếp theo thứ tự dưới.

## 1. Việc 6a: chốt P2 @2k (làm ngay)

- Export ONNX acoustic từ P2 @2k EMA (D đã tách). md5 x3, đổi thread 1/2/4 so md5. RTF CPU 2 luồng (acoustic, vocoder riêng).
- Đặt làm mặc định cho CLI/demo, ghi tên `v32_adv_p2_2k`. Giữ v3.2 làm phương án lùi.

## 2. Việc 6b: sửa thước drift

- Mốc thật: chọn từ corpus val/holdout các câu gốc liền mạch có 3 mệnh đề trở lên (ghép các segment liên tiếp cùng video,
  dùng audio THẬT), đo drift cùng công thức. Báo median/p90 của giọng thật.
- Tăng bộ câu mới lên 40 (giữ 20 câu cũ). Báo median, p90 kèm khoảng tin cậy 90% bootstrap (1.000 lần).
- Cổng mới: p90 drift của model nằm trong [thật - 1,5 dB, thật + 1,5 dB] VÀ không vượt v3.2 + 2 dB.
  Nếu thật thấp hơn v3.2 thì chỉ dùng điều kiện thứ nhất.

## 3. Việc 6c: probe sửa độ to (Việc 5c, giữ thiết kế)

Khởi tạo từ P2 @2k (cả trọng số sống + EMA), cấu hình P2 (adv 0,3, fm 0), mỗi nhánh 4k bước, đo ở 2k và 4k:
- **E0**: không đổi gì (đối chứng, tức P2 chạy tiếp từ 2k).
- **E1**: + energy envelope loss tuyệt đối (mean log-mel bỏ z-norm theo frame, L1 với thật, chỉ frame tiếng), trọng số 1,0.
- **E2**: E1 + L1 dải 0 đến 1 kHz x2.
- **E3**: E1 nhưng adv 0,2 (xem giảm adv có giữ được độ nét không).
Mỗi mốc: drift (thước mới 6b), GV EMA, val EMA, flux biên, smooth.
Chọn: drift qua cổng mới ở 4k, GV EMA >= 0,72. Nếu không nhánh nào đạt thì giữ P2 @2k, không chạy thêm GAN.

## 4. Việc 6d: chuẩn hoá độ to theo chunk ở runtime (Việc 5d)

- Dịch log-mel cả chunk bằng một hằng số để RMS hữu thanh về mức mục tiêu. Mức mục tiêu = trung vị RMS hữu thanh của
  giọng thật trong corpus (không đặt tay). Giới hạn độ dịch tối đa ±4 dB.
- Đo drift có/không cho v3.2, P2 @2k, P2 @6k. Mặc định TẮT, cho người dùng nghe 6 câu có/không trước khi bật.
- Nếu bật được, đây là lưới an toàn cho mọi checkpoint sau, nhưng KHÔNG thay cho việc sửa bằng loss (6c).

## 5. Việc 6e: top spike (Việc 5e, chỉ báo cáo)

Với 10 âm tiết top spike: ảnh mel thật + TextGrid, xác định frame "thật" có phải lặng hoặc biên lệch không.

## 6. Báo lại

6a (md5, RTF), 6b (drift giọng thật + bảng mới có CI), 6c (bảng E0 đến E3), 6d (drift có/không + link nghe), 6e (ảnh).
