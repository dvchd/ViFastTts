# VÒNG 11 BỔ SUNG 2 | Đặc tả: docs/handoff/V11_agent0_bosung2.md | Gốc: b1ae8ce
Bổ sung cho V11_agent0.md và V11_agent0_bosung.md, không thay thế.
Agent 2 THỰC THI Phần B | Agent 1 ĐỌC Phần C.
Căn cứ: báo cáo B-I của Agent 2 (V11.B1–B5 + bổ sung).

=====================================================================
A. QUYẾT ĐỊNH
=====================================================================
V11S2.A1 B1, B3, B4 và các mục bổ sung (A3, A4, A5): DUYỆT.
         Câu hỏi nhỏ: val có 1.773 hàng, trừ 300 còn 1.473, nhưng calib 737 + lock 735
         = 1.472. Thiếu 1 hàng. Nếu là hàng thuộc clip cha nằm vắt qua ranh giới val300
         và bị loại để tránh rò rỉ thì đúng; ghi rõ id hàng và lý do vào §44.

V11S2.A2 f0p của probe NẰM NGOÀI 2σ chuẩn (−16,4% so với biên 13,2%).
         Theo V10.A7 đây là "NHÁNH KHÁC" (không đổi theo manifest nhưng ngoài 2σ), lẽ ra
         phải dừng. KHÔNG dừng A2, vì:
         - V10.B4 đo tất định: cùng mô hình, f0p trùng 0,000% giữa 3 manifest. Dữ liệu
           fixtail KHÔNG làm đổi giá trị loss f0p. Mọi chênh lệch là do mô hình thay đổi
           trong lúc train (LR khởi động lại khoảng 0,30 × lr_max so với gần mức nhỏ nhất
           ở cuối base) hoặc do khác cửa sổ batch.
         - f0p GIẢM, không tăng.
         Nhưng phải xác minh bằng số của chính A2, việc rẻ, chỉ đọc history:
         - VAL f0p: B4 chứng minh f0p bất biến theo manifest, nên val f0p của A2 (val
           fixtail) so được TRỰC TIẾP với val f0p của base 40k trên cùng 300 id.
           Báo val f0p tại 41k, 42k, 43k, 44k (và các mốc tiếp theo) so với base 40k.
         - TRAIN f0p: trung bình trượt 500 bước của A2 so với 500 bước cuối base.
         Nhánh định trước:
         - val f0p của A2 trong ±5% so với base: chênh ở probe là do LR/cửa sổ batch.
           Ghi nhận, không hành động.
         - val f0p giảm hơn 5% và ổn định: mô hình học F0 prior tốt hơn nhờ LR cao.
           Ghi nhận (có lợi), đối chiếu F0 NGÃ ở bench B6.
         - val f0p TĂNG hơn 5%: báo ngay, Agent 0 quyết có dừng A2 không.
         Tiêu chí probe từ nay: σ chuẩn = σ/√m như Agent 2 đã làm rõ. Ghi chú vào §44:
         công thức giả định các bước độc lập; loss train có tự tương quan nên biên thật
         rộng hơn một chút. Lệch ngoài 2σ là lý do để KIỂM, không tự động là lý do dừng.

V11S2.A3 UTMOSv2: BỎ theo tiêu chí định trước (40k/18k chỉ khớp 2/3 và trung bình ngược
         chiều tai người dùng). Ghi vào §44. Không thử lại với cấu hình khác để tránh chọn
         cấu hình theo kết quả.

V11S2.A4 CreaPy: có nguồn. Kho GitHub SPSC-TUGraz/creapy (bản gốc trên GitLab TU Graz),
         giấy phép GPL-3.0, chưa lên PyPI, cài bằng clone rồi pip install -e.
         - Chỉ dùng làm công cụ đo, cài vào môi trường riêng (ví dụ .venv-creapy), KHÔNG
           đưa mã nguồn vào repo, KHÔNG thêm vào dependencies của ViFastTts (GPL-3.0).
         - Tham số mặc định chỉnh cho hội thoại tiếng Đức. Một dự án sinh viên TU Graz
           trên giọng ĐỌC phải chỉnh: độ dài creak tối thiểu 60 ms, ngưỡng xác suất creak
           0,8, ngưỡng năng lượng ngắn hạn ở phân vị 30%; F1 cao nhất khoảng 0,60 trên
           giọng khỏe. Dùng bộ tham số này làm điểm khởi đầu, ghi rõ trong báo cáo.
         - Nếu giọng người dùng là nam, thử thêm mô hình train trên giọng nam có sẵn.
         - Bước kiểm chứng giữ nguyên V11.A12: trên audio thật, ngã/nặng phải có tỉ lệ
           creak cao hơn rõ rệt so với ngang/huyền. Không đạt: bỏ, không chỉnh tham số
           thêm để "cho đạt".

V11S2.A5 TỰ QUYẾT #4 (watcher sao lưu checkpoint): chấp nhận, kèm 2 điều kiện.
         Phát hiện này quan trọng: các đặc tả trước giả định "checkpoint mỗi 5k" là tệp
         riêng, thực tế trainer chỉ ghi đè last.pt. Fallback ở V11S.A2 ("thử checkpoint 5k
         gần nhất") vì vậy chưa có gì để lùi về.
         Điều kiện cho watcher hiện tại:
         - Bản sao chỉ hợp lệ khi torch.load trên bản sao thành công VÀ step trong tệp
           đúng 45000 / 50000 / 55000. Trainer ghi last.pt không nguyên tử, watcher có
           thể chép đúng lúc đang ghi. Bản sao lỗi: chép lại sau vài giây.
         - Watcher không được giữ khóa hay đọc lâu last.pt (chép xong mới torch.load trên
           bản sao).
         Báo trong V11_agent2.md: mỗi bản sao có step, sha256, thời điểm chép.
         Nếu lỡ mất một mốc (ví dụ 45k chép lỗi mà trainer đã ghi đè): bench các mốc còn
         lại, ghi rõ mốc thiếu; không train lại để lấy mốc.

V11S2.A6 Mở rộng V11S.B3 (sau khi A2 tới 60k): trainer tự lưu ckpt_<step>.pt ở mỗi
         save_every, ghi nguyên tử (tệp tạm, fsync, đổi tên), giữ tất cả (khoảng 4 tệp mỗi
         run). last.pt vẫn ghi như cũ (nguyên tử). Fallback của watchdog dùng các tệp này.
         Test: save_every 2 trong 6 bước tạo ckpt_2, ckpt_4, ckpt_6; đọc được cả ba.
         Sau thay đổi này bỏ watcher sao lưu riêng.

=====================================================================
B. AGENT 2 THỰC THI
=====================================================================
V11S2.B1 Commit tệp này nguyên văn.
V11S2.B2 Ngay (CPU, chỉ đọc history): bảng val f0p A2 theo mốc so với base 40k, và train
         f0p trung bình trượt 500 bước; áp nhánh V11S2.A2.
V11S2.B3 Ngay: kiểm 3 điều kiện V11S2.A5 với bản sao đã có (nếu đã có 45k).
V11S2.B4 Ngay (CPU): cài CreaPy theo V11S2.A4, chạy bước kiểm chứng trên audio thật.
         Đạt thì chạy copy-synth và free-run (free-run cần MPS: để sau B6).
V11S2.B5 Ghi rõ hàng val thiếu (V11S2.A1).
V11S2.B6 V11S2.A6 gộp vào V11S.B3, làm sau khi A2 tới 60k.
Thứ tự còn lại giữ như V11: B6 → B7 → V11S.B3 (+ V11S2.A6) → B9 → B10 → B11.

=====================================================================
C. AGENT 1 ĐỌC (thêm vào V11.C1–C7 và V11S.C1)
=====================================================================
V11S2.C1 train_v2.py: có chỗ nào lưu checkpoint theo step riêng không, hay chỉ last.pt và
         best.pt? save_every trong config thực sự điều khiển gì?
V11S2.C2 Sau V11S2.A6: ghi nguyên tử có fsync trước rename không; tên tệp có đúng step;
         fallback watchdog đọc các tệp này theo thứ tự nào?
V11S2.C3 val_calib / val_lock: hàng thiếu là hàng nào, có phải do lọc theo clip cha không?
