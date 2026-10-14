# VÒNG 11 | Đặc tả: docs/handoff/V11_agent0.md | Commit gốc: b1ae8ce
Thay thế: mọi phần giao việc còn mở của V10 (V10.B10 chuyển thành V11.B6)
Agent 2 THỰC THI: Phần B | Agent 1 ĐỌC: Phần C | Người dùng: Phần D
Căn cứ: báo cáo V10 của Agent 2 (b1ae8ce), báo cáo đọc V10.C1–C7 của Agent 1,
báo cáo rà soát acoustic 10 điểm của Agent 1, tìm kiếm tài liệu của Agent 0 và Agent 1.
Quy tắc: commit đầu tiên của vòng là tệp này nguyên văn.

=====================================================================
A. QUYẾT ĐỊNH
=====================================================================
V11.A1 V10.B1–B9: DUYỆT. A2 đang chạy đúng (resume @40000, đủ cột thành phần loss,
       không NaN). KHÔNG dừng A2.
       Tín hiệu sớm @41k: BCE pause val/train 7,5 lần (base khoảng 20 lần). Chưa kết
       luận, chỉ ghi nhận.

V11.A2 Tiêu chí ±2σ của probe: cần làm rõ thang so sánh.
       2σ báo cáo rất rộng (f0p 59,1%; G 70,6%), nên gần như mọi lệch đều lọt.
       Nếu σ tính trên giá trị TỪNG BƯỚC mà so với TRUNG BÌNH 20 bước của probe thì sai
       thang: σ của trung bình 20 bước nhỏ hơn khoảng √20 ≈ 4,5 lần.
       Quyết định:
       - σ chuẩn = độ lệch chuẩn của các trung bình trượt 20 bước trong 500 bước cuối base.
       - Báo lại 7 thành phần probe fixtail theo σ chuẩn (chỉ ghi nhận).
       - KHÔNG ảnh hưởng A2: V10.B4 đã loại lỗi chuẩn hóa bằng phép đo tất định
         (f0p trùng 0,000% giữa 3 manifest), và mel, G tăng là đúng đích A2.
       - Áp σ chuẩn cho mọi probe từ nay (A1 nếu có). Sửa văn bản tiêu chí trong §44.

V11.A3 Tự quyết của Agent 2 ở V10: chấp nhận cả 4. Vá mốc B7 từ raw là hợp lệ vì test
       pin chạy mọi lần test bảo đảm README == byte git == byte đĩa.

V11.A4 Cây quyết định sau B6 (bench A2), định trước để không mất một vòng:
       - KẾT LUẬN "TỪ CHỐI" (có cổng cứng trượt): DỪNG, báo. Không chạy A1.
       - "CHẤP NHẬN": A2 là gốc mới. Không chạy A1.
       - "ĐẠT CỨNG, TRƯỢT MỤC TIÊU", phần trượt chỉ ở pause_f1_all hoặc gap_mae_all:
         chuẩn bị A1 (V11.B8) nhưng KHÔNG chạy; Agent 0 duyệt sau khi đọc bảng.
       - Trượt mục tiêu chỉ ở length_ratio: không chạy A1 (A1 không nhắm vào đó).
       Chọn checkpoint A2 tốt nhất trong 45/50/55/60k: đạt nhiều cổng mục tiêu nhất
       trong số các mốc đạt mọi cổng cứng; hòa thì lấy mốc có gap_z thấp hơn.

V11.A5 CER phải đo giống production (rà soát mục 4 của Agent 1).
       Production: mô hình nhận lead = 0, tầng ghép chèn lặng cố định. CER holdout hiện
       đưa lead tự nhiên vào mô hình (63% hàng lead > 0), tức đo trên G_lead chưa train.
       Quyết định: thêm trường cer_prod (lead = 0, đệm lặng cố định 150 ms đầu và cuối
       khi ghép audio trước ASR). Vòng này cer_prod chỉ THEO DÕI, cổng vẫn dùng cer cũ để
       base và A2 đo cùng cách. Vòng sau cer_prod thay cer trong cổng.

V11.A6 Chia lại tập val để tránh khớp quá mức vào val300.
       Val300 đang dùng cho chọn checkpoint, cổng, quét ngưỡng gate và sắp tới E4.
       Chia hàng 301 đến 1.773 của val fixtail THEO CLIP CHA (mọi đoạn cùng một clip vào
       cùng một tập, tránh rò rỉ):
         val_calib: khoảng một nửa, dùng để hiệu chuẩn ngưỡng và tham số;
         val_lock:  phần còn lại, KHÓA, chỉ dùng cho đánh giá cuối mỗi dòng (sau khi chốt
                    A2/A1, sau E4). Mỗi lần dùng val_lock ghi vào §44.
       Ghim hash cả hai. Không ai chạy eval trên val_lock trong vòng này.

V11.A7 Ngưỡng gate cứng 0,5 (rà soát mục 2): đưa thành tham số runtime gate_threshold,
       mặc định 0,5 (giữ hành vi cũ từng byte). Hiệu chuẩn trên val_calib, không trên
       val300. Chưa đổi mặc định vòng này.

V11.A8 Duration MAE đứng yên qua mọi run (rà soát mục 1).
       Agent 1 nghi chạm trần nhãn. Agent 0 cho rằng chưa chắc: 4,84 frame (khoảng 52 ms)
       lớn hơn nhiều so với độ lệch nhãn khoảng 18 ms ở §17. Giả thuyết thay thế: bộ dự
       đoán thoái hóa về trung bình.
       Đo trước khi bàn T2.4 (V11.B9). Không đổi aligner trong vòng này.

V11.A9 VUV 0,9823 giống hệt ở small và base tới 4 chữ số (rà soát mục 3): nghi lỗi đo trước
       khi nghi bão hòa. Kiểm (V11.B9). Nếu metric tính từ target hoặc mặt nạ không phụ
       thuộc dự đoán: sửa. Nếu đúng là bão hòa: chuyển từ cổng sang theo dõi.

V11.A10 Tiêu chí E4 (rà soát mục 5): tone_C chỉ là rào chắn không thoái lui. Cổng mục tiêu
       E4 = nhóm SẮC-tắc (teacher-forcing, so SẮC-khác cùng ô độ dài nucleus) + F0 trung vị
       NGÃ (cent) + creak theo âm tiết nếu CreaPy qua kiểm chứng. Chi tiết ở đặc tả E4.

V11.A11 Việc theo dõi bổ sung (rà soát mục 6, 7, 8, 10): năng lượng trên âm tiết ngã/nặng so
       với thật; chất lượng ở speed 0,8 và 1,25; gián đoạn F0/năng lượng tại mối nối câu;
       tỉ phần loss sau A2. Làm sau A2, không chặn gì, thứ tự ở V11.B10.

V11.A12 Công cụ đo mới, chỉ được dùng sau khi qua kiểm chứng:
       - CreaPy: trên audio THẬT, tỉ lệ frame creak của ngã/nặng phải cao hơn rõ rệt so
         với ngang/huyền. Không đạt: bỏ (công cụ phát triển trên hội thoại tiếng Đức vùng
         Áo, chưa kiểm trên tiếng Việt).
       - UTMOSv2: phải xếp hạng KHỚP tai người dùng ở cả hai lần nghe đã có (40k hơn 18k;
         sh25 hơn plain). Kiểm tần số mẫu (mặc định 16 kHz). Khớp cả hai: thêm làm theo
         dõi, chưa làm cổng. Không khớp: bỏ.
       Mọi số liệu trích từ tài liệu ngoài chỉ ghi vào docs sau khi đã đo lại trên dữ liệu
       dự án.

V11.A13 Roadmap cập nhật (ghi §44, không thực thi):
       - E4: thử cả hai cách xử lý F0 vùng creak khi tạo target: MASK (T6.3) và NỘI SUY
         qua vùng creak (theo hướng lan truyền pitch mark của Ninh và Yamashita 2015).
         Creak điều kiện ở cấp âm tiết theo thanh, không theo frame, không toàn cục.
       - Vocoder: trước vocoder điều kiện phonation, thử 2 loss chỉ dùng khi train của
         tone-aware-vocos (MR-STFT tăng trọng số dải dưới 1 kHz; discriminator Constant-Q
         dải 80 đến 320 Hz). Không đổi kiến trúc, không tăng chi phí suy luận. Bỏ loss
         feature của bộ phân loại thanh (tác giả báo làm F0 xấu đi).
       - A1 dự phòng nếu recall pause vẫn thấp sau A1: đầu gap dạng phân lớp theo vài mức
         độ dài thay vì hồi quy (hướng Yang và cộng sự, ICASSP 2023).
       - PhoAudiobook: KHÔNG dùng lúc này (nhiều người nói, trộn phương ngữ, giấy phép chỉ
         nghiên cứu/giáo dục, cấm phân phối lại).

=====================================================================
B. AGENT 2 THỰC THI
=====================================================================
--- B-I. TRONG LÚC A2 CHẠY (chỉ CPU, không chạy eval trên MPS) ---
V11.B1 Commit đầu: docs/handoff/V11_agent0.md nguyên văn.
V11.B2 Theo V11.A2: báo σ chuẩn (trung bình trượt 20 bước, 500 bước cuối base) và 7 thành
       phần probe fixtail theo σ chuẩn. Ghi rõ σ ở V10 đã tính trên thang nào. Sửa §44.
V11.B3 Chia val_calib / val_lock theo V11.A6 (theo clip cha), ghim hash, test: không clip cha
       nào xuất hiện ở hai tập; hai tập không giao val300.
V11.B4 Code (không chạy MPS):
       - gate_threshold runtime (V11.A7) cho infer, export, runtime; mặc định 0,5; test
         đầu ra trùng từng byte với hiện tại khi không truyền.
       - cer_prod trong bench (V11.A5), chỉ theo dõi; test lead = 0 và đệm lặng đúng 150 ms.
       - Mọi chỗ khác chỉ thêm trường, không đổi trường cũ.
       Toàn bộ test, ruff, push.
V11.B5 CPU, dùng audio đã có:
       - CreaPy kiểm chứng trên audio thật (V11.A12), 25 clip giàu ngã/nặng. Đạt thì chạy
         thêm bản copy-synth và free-run base 40k của cùng 25 clip, gộp theo âm tiết và
         theo thanh.
       - UTMOSv2 trên các cặp đã nghe (40k/18k, sh25/plain) theo V11.A12.
       Không cài được hoặc quá chậm trên CPU: ghi vào CHƯA LÀM, không thay công cụ khác.

--- B-II. SAU KHI A2 TỚI 60k ---
V11.B6 Bench A2 tại 45k / 50k / 55k / 60k (--baseline-old base40k_old.json,
       --baseline-fixgap base40k_fixtail.json). Báo bảng 4 mốc × (cứng, mục tiêu, theo dõi),
       dòng KẾT LUẬN, BCE pause val/train mỗi mốc, length_ratio_speech/pause, cer_prod.
       Chạy thêm cer_prod cho base 40k (cùng cách) để có mốc theo dõi.
V11.B7 Áp cây quyết định V11.A4, ghi rõ nhánh và checkpoint chọn.
V11.B8 Chỉ khi rơi nhánh chuẩn bị A1: tạo config A1 (fixtail, pos_weight tính lại trên train
       fixtail, pause_gate/gap 2,0, mass 10,0, resume từ checkpoint A2 đã chọn), dán
       --dry-run. KHÔNG probe, KHÔNG chạy.
V11.B9 Chẩn đoán (MPS, sau B6):
       - Duration (V11.A8): MAE theo nhóm độ dài thật (ngắn / vừa / dài theo tercile) và
         theo loại vị trí (onset, nucleus, coda nếu có), cho base 40k và A2 chọn. Báo thêm
         độ lệch chuẩn của dự đoán so với của thật (thoái hóa về trung bình nếu nhỏ hơn rõ).
         Báo con số §17 "18 ms" đo cái gì, so với cái gì.
       - VUV (V11.A9): dán dòng tính metric; chạy thử với một đầu ra VUV hằng số (toàn 1)
         trên val300, báo giá trị. Nếu gần 0,98 thì thước gần như không phân biệt được.
       - SẮC-tắc làm lại (V10.A10): so SẮC-tắc với SẮC-khác cùng ô độ dài nucleus, train
         trong mẫu và val300, base 40k và A2 chọn; báo n, mel_z, dB, F0 cent mỗi ô.
V11.B10 Theo dõi (MPS, sau B9, làm được tới đâu báo tới đó): năng lượng trên âm tiết ngã/nặng
       so với thật; CER/F0/duration ở speed 0,8 và 1,25; gián đoạn F0/năng lượng tại mối nối
       khi ghép 5 câu liên tiếp với lặng chèn cố định. Đều trên base 40k và A2 chọn.
V11.B11 DỪNG. Cập nhật V11_agent2.md (theo mã, TỰ QUYẾT KHÁC ĐẶC TẢ, CHƯA LÀM / LÀM DỞ) và
       TRANG_THAI.md (gốc b1ae8ce, commit cuối, "DỪNG, CHỜ ĐỌC").

=====================================================================
C. AGENT 1 ĐỌC (pull HEAD, đọc toàn bộ source, trả lời trong chat)
=====================================================================
V11.C1 Free-run trên holdout: lead đi vào mô hình thế nào, G_lead có được dùng không? cer_prod
       có thật sự đặt lead = 0 và chỉ thêm lặng ở tầng ghép không?
V11.C2 Metric VUV: tính từ đầu ra dự đoán hay từ target, mặt nạ nào? Dán dòng.
V11.C3 Duration MAE trong eval: tính trên tập nào, đơn vị gì, theo âm tiết hay theo vị trí?
       Con số 18 ms ở §17 đến từ đâu trong code hoặc tài liệu?
V11.C4 gate_threshold: mọi nơi dùng ngưỡng gate (infer, export ONNX, runtime) đã đọc tham
       số chưa, còn hằng 0,5 nào sót không? Mặc định có giữ đúng hành vi cũ không?
V11.C5 Chia val_calib / val_lock: có thật theo clip cha không, có giao val300 không, hash
       ghim ở đâu, có đường code nào vô tình chạy eval trên val_lock không?
V11.C6 Tiêu chí σ chuẩn trong §44 và code probe (nếu có): đúng thang trung bình trượt chưa?
V11.C7 Mọi vấn đề khác trên toàn bộ source.

=====================================================================
D. NGƯỜI DÙNG (không bắt buộc, không chặn gì)
=====================================================================
Đề xuất tự thu khoảng 30 câu ngắn cố định (cặp tối thiểu thanh điệu, nhóm SẮC-tắc, câu
nhiều ngã/nặng). Vì mô hình train trên chính giọng bạn, đây sẽ là tham chiếu thật cho CER,
F0 và creak, chính xác hơn mọi thước tự động. Nếu đồng ý, Agent 0 soạn danh sách câu ở
đặc tả E4.
