# VÒNG 09 | Đặc tả: docs/handoff/V09_agent0.md | Commit gốc: 475bdb4
Thay thế: phần giao việc còn mở của V08 (B9, B10, B11 chuyển sang vòng này)
Agent 2 THỰC THI: Phần B | Agent 1 ĐỌC: Phần C | Người dùng: Phần D
Căn cứ: báo cáo tiến độ V08 của Agent 2 (commit 475bdb4) và báo cáo đọc code
Y1–Y6 của Agent 1.

=====================================================================
A. QUYẾT ĐỊNH
=====================================================================
V09.A1 Y1, Y5, Y6: đạt. Trần lặng cuối clip, --validate-at-stop và phạm vi
       thay đổi đúng đặc tả.

V09.A2 LỖI CHẶN VẪN MỞ (Y2, Y3): watch_train_v2.sh đọc INIT_RESUME và
       INIT_RESET_VAL_STATE nhưng hai nhánh chạy không dùng, nên lần khởi động đầu
       vẫn train từ số 0. Dry-run in ra lệnh khác với lệnh thật chạy.
       Nguyên nhân gốc: dry-run và lệnh thật dựng ở hai chỗ khác nhau.
       Sửa: dựng lệnh train-v2 MỘT LẦN thành một mảng bash, dry-run in đúng mảng
       đó, chế độ thật chạy đúng mảng đó. Không còn echo mô tả riêng.

V09.A3 §44.8 ghi W5 "đã sửa" trong khi code chưa sửa. Quy tắc từ vòng này:
       - Tài liệu và báo cáo chỉ được ghi "đã sửa" khi kèm tên test khóa lỗi đó
         và test đó đang chạy xanh.
       - Báo cáo Agent 2 có thêm mục "CHƯA LÀM / LÀM DỞ", ngoài mục
         "TỰ QUYẾT KHÁC ĐẶC TẢ".
       Sửa §44.8 cho đúng thực tế.

V09.A4 Pin sha256 cũ (Y4): code kiểm pin đúng, nhưng README vẫn ghim số cũ
       (ghim 3d6fefe5… / cf770864… so với thực bbd2b667… / e50167dc…). Mục
       "ghim lại base40k_old.json" của V08.B5 chưa làm.
       - Ghim lại base40k_old.json theo sha thực.
       - base40k_fixgap.json: ghi "thay bởi fixtail, không dùng cho cổng" trong
         README, ghim sha thực để lưu vết.
       - Đổi hành vi: tệp truyền qua --baseline-* mà CHƯA ghim trong README thì
         DỪNG (không chỉ cảnh báo). Tệp mới đang được sinh bằng --emit-baseline thì
         không bị kiểm (vì chưa có pin).

V09.A5 B9 đang chạy nền: phải xác định lần chạy đó có hợp lệ không.
       - Nếu lệnh có --baseline-old hoặc --baseline-fixgap: theo Y4 nó đã hoặc sẽ
         dừng vì pin lệch. Chạy lại sau khi ghim (A4).
       - Nếu lệnh chỉ --emit-baseline, không truyền --baseline-*: kết quả mốc dùng
         được, nhưng PHẦN SO SÁNH H1 phải đọc số val cũ từ base40k_old.json đã ghim
         lại, không từ tệp chưa xác thực.
       - Mốc B9 phải ghi git_commit = commit thật đã sinh ra nó.
       Báo rõ trường hợp nào đã xảy ra, dán lệnh thật đã chạy.

V09.A6 B10 (probe fixtail) VẪN ĐƯỢC CHẠY: probe gọi train-v2 trực tiếp, không qua
       watch (Y3 xác nhận lệnh probe đúng). Tiêu chí đạt giữ như V08.A5.

V09.A7 Chạy thật A2: CHƯA DUYỆT. Điều kiện: A2 sửa xong + test khóa + Agent 1 xác
       nhận ở V09.C1–C3 + B9 và B10 đạt.

=====================================================================
B. AGENT 2 THỰC THI (theo thứ tự, dừng ở B8)
=====================================================================
V09.B1 Báo ngay (không cần chờ): lệnh thật của lần B9 đang chạy nền và trường hợp
       nào ở V09.A5. Nếu B9 không hợp lệ: dừng nó.

V09.B2 Ghim lại sha theo V09.A4. Đổi "chưa ghim" thành dừng + test.
       Commit bằng đường dẫn tường minh.

V09.B3 Sửa watch_train_v2.sh theo V09.A2:
       - Một hàm dựng mảng lệnh; dry-run in mảng; chế độ thật exec mảng.
       - Lần khởi động đầu (OUT/last.pt chưa có): mảng có --resume $INIT_RESUME, và
         có --reset-val-state nếu INIT_RESET_VAL_STATE=1.
       - Khởi động lại (OUT/last.pt đã có): --resume OUT/last.pt, KHÔNG có
         --reset-val-state.
       - Biến TRAIN_CMD (mặc định lệnh train-v2 thật) để test thay bằng script giả.
       Test (bash hoặc pytest gọi subprocess), dùng script giả ghi lại đối số:
         (a) khởi động đầu: đối số chứa "--resume runs/acoustic_v2_base/last.pt" và
             "--reset-val-state";
         (b) đã có last.pt: đối số chứa "--resume <OUT>/last.pt", không chứa
             "--reset-val-state";
         (c) đầu ra --dry-run trùng từng ký tự với đối số script giả nhận ở (a).
       Dán đầu ra --dry-run của run_A2_fixgap.sh chế độ thật vào báo cáo.

V09.B4 Sửa §44.8 theo thực tế (V09.A3). Thêm mục "CHƯA LÀM / LÀM DỞ" vào mẫu báo
       cáo V09_agent2.md.

V09.B5 Chạy toàn bộ test, ruff, push.

V09.B6 B9 (mốc fixtail), nếu B1 cho thấy cần chạy lại: chạy sau B2. Nhánh đã định:
       H1 = |syl_z(fixtail) − mel_l1_z(val cũ, từ base40k_old.json)| / mel_l1_z ≤ 3%:
       tiếp B7; > 3%: DỪNG, báo số. Ghim sha base40k_fixtail.json. Báo ngưỡng tuyệt
       đối cổng mục tiêu A2 suy ra từ mốc mới, và bảng 3 nhóm gap trên fixtail
       (V08.B2) nếu chưa báo.

V09.B7 B10 probe fixtail (V09.A6), thư mục runs/acoustic_v2_A2_fixtail_probe,
       --validate-at-stop. Báo: LR bước đầu/cuối và lr_at, từng thành phần loss so
       500 bước cuối base 40k (%), BCE pause train/val, tỉ phần loss pause,
       giây/bước, RAM đỉnh. Đạt hay trượt theo V08.A5.

V09.B8 DỪNG. Cập nhật V09_agent2.md và TRANG_THAI.md (vòng V09, commit gốc 475bdb4,
       commit cuối, trạng thái "DỪNG, CHỜ ĐỌC"). MPS rảnh sau B7: chạy SẮC-tắc.

=====================================================================
C. AGENT 1 ĐỌC (tự pull, đọc toàn bộ source, trả lời trong chat)
=====================================================================
V09.C1 watch_train_v2.sh: chỉ ra hàm dựng mảng lệnh. Dry-run và chế độ thật có dùng
       chung đúng một mảng không? Còn đường chạy nào dựng lệnh riêng không?
V09.C2 Lần khởi động đầu và khởi động lại: đối số thực tế của từng nhánh (theo code,
       không theo echo). Có đường nào reset val lần hai khi khởi động lại không?
V09.C3 Test của B3: có thực sự chạy script (không chỉ đọc chuỗi)? Có khóa (a), (b),
       (c) không?
V09.C4 README: sha ghim của base40k_old.json và base40k_fixtail.json có khớp sha
       thực của tệp ở HEAD không (tự tính)? Tệp --baseline-* chưa ghim có bị dừng
       không, test nào khóa?
V09.C5 base40k_fixtail.json: git_commit ghi trong tệp có tồn tại và chứa code bench
       đã sửa không? val_manifest_hash có khớp hash fixtail ghim trong README không?
V09.C6 §44 và TRANG_THAI.md: có chỗ nào ghi "đã sửa" hoặc "xong" mà code hoặc test
       không xác nhận không?
V09.C7 Mọi vấn đề khác thấy được trên toàn bộ source.

=====================================================================
D. NGƯỜI DÙNG
=====================================================================
Ghi chú AGENTS.md trong stash nằm trên máy của Agent 1, không ảnh hưởng repo.
Nếu ghi chú đó có quy tắc bạn muốn giữ, gửi nội dung cho Agent 0 để đưa vào đặc tả,
Agent 2 sẽ thêm vào AGENTS.md. Nếu không cần thì bỏ.
