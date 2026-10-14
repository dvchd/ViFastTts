# VÒNG 10 | Đặc tả: docs/handoff/V10_agent0.md | Commit gốc: 06b5f10
Thay thế: mọi phần giao việc còn mở của V09
Agent 2 THỰC THI: Phần B | Agent 1 ĐỌC: Phần C
Căn cứ: báo cáo V09 của Agent 2 (06b5f10) và báo cáo đọc code của Agent 1 trên
HEAD 06b5f10.

=====================================================================
A. QUYẾT ĐỊNH
=====================================================================
V10.A1 Tệp đặc tả không có trong repo. Agent 1 báo không tồn tại
       docs/handoff/V09_agent0.md, nên đã kiểm theo Y1–Y6 cũ thay vì V09.C1–C7.
       TRANG_THAI.md trỏ tới một tệp không có. Quy tắc từ vòng này:
       commit ĐẦU TIÊN của mỗi vòng là tệp đặc tả nguyên văn
       (docs/handoff/V<nn>_agent0.md), trước mọi thay đổi code. Bổ sung luôn
       V09_agent0.md cho đầy đủ lịch sử.

V10.A2 Watch script (Y2/Y3): Agent 1 xác nhận đã sửa thật ở 76f891b (một mảng
       CMD, INIT_* chỉ lần đầu, dry-run in đúng mảng, test 3 ca). ĐÓNG lỗi chặn.

V10.A3 sha256 mốc: hai agent báo NGƯỢC nhau.
         Agent 2: base40k_old 3d6fefe5…, base40k_fixtail 7be6fcff…, khớp README.
         Agent 1: thực bbd2b667… và 5c903494…, lệch README.
       Cả hai tệp đều lệch trên máy Agent 1, đều khớp trên máy Agent 2. Nhiều khả
       năng nhất: chuyển đổi xuống dòng khi checkout (git core.autocrlf hoặc
       .gitattributes text=auto trên máy Agent 1) làm đổi byte tệp JSON. Khi đó pin
       đúng trên M6 nhưng bench sẽ dừng trên mọi máy checkout khác. Đây là lỗi
       thiết kế pin, không phải lỗi ghim.
       Sửa:
       - .gitattributes: eval/baselines/*.json -text (không chuyển xuống dòng).
       - Tệp mốc ghi bằng xuống dòng LF, JSON chuẩn hóa (sort_keys, indent cố định,
         không khoảng trắng cuối dòng, có dòng trống cuối tệp).
       - README ghim sha256 của byte trong git (git show HEAD:<tệp> | sha256sum),
         Agent 2 dán lệnh và kết quả vào báo cáo.
       - Bench kiểm sha trên byte tệp như hiện tại (đúng khi -text có hiệu lực).
       Agent 1 kiểm lại ở V10.C1 sau khi đã làm mới bản checkout.

V10.A4 README có 2 dòng mâu thuẫn cho base40k_fixtail.json (một dòng pin 7be6…,
       một dòng pin cf77… ghi chữ fixgap). Dọn thành bảng một dòng mỗi tệp:
       tên tệp | sha256 | val_manifest_hash | trạng thái (DÙNG / KHÔNG DÙNG).

V10.A5 base40k_fixgap.json vẫn được track (Agent 1 báo khoảng 13 nghìn dòng, tức
       không phải tệp mốc gọn). Gỡ khỏi git (git rm --cached), giữ trên đĩa ở
       artifacts/bench/legacy/. README giữ một dòng KHÔNG DÙNG kèm sha và đường dẫn
       artifacts. .gitignore chỉ cho phép base*_old.json và base*_fixtail.json.
       Kiểm luôn base40k_fixtail.json có phải tệp gọn không (chỉ khóa cấp cao +
       summary, summary_fixgap); nếu chứa dữ liệu thô thì sinh lại dạng gọn.

V10.A6 summary_fixgap["cer"] = null: XÓA khóa, không để null. Test khóa đổi thành
       "khóa không tồn tại". Áp cho tone_C và creak tương tự.

V10.A7 f0p −18,9% ở probe fixtail: KHÔNG chọn (a), (b), (c) của Agent 2. Đo tất
       định trước, không train (V10.B4). Kết quả quyết định:
       - NHÁNH LỖI: cùng checkpoint, cùng batch, f0p đổi > 2% giữa các manifest
         (cũ / fixgap / fixtail): chuẩn hóa loss phụ thuộc số frame gap. DỪNG,
         không chạy A2, báo công thức chuẩn hóa (tập tin:dòng) và số.
       - NHÁNH NHIỄU: f0p đổi ≤ 2% giữa các manifest VÀ −18,9% nằm trong 2σ của
         f0p 500 bước cuối base: tiêu chí probe cho từng thành phần đổi từ ±5%
         thành ±2σ (σ đo trên 500 bước cuối base). Probe fixtail coi là ĐẠT.
       - NHÁNH KHÁC (đổi ≤ 2% nhưng ngoài 2σ): DỪNG, báo số.

V10.A8 Thiếu số đo pause (cả hai agent cùng báo): ghi TỪNG thành phần loss (tên đúng
       như trong v2_losses, gồm pause_gate, pause_gap) vào history.csv ở cả bước
       train (trung bình theo log_every) lẫn validate, và in ra log. Thêm cột
       tỉ phần pause = (pause_gate + pause_gap) / tổng. Thay đổi chỉ thêm cột, không
       đổi giá trị loss. Test: tổng các thành phần có trọng số bằng loss tổng (sai số
       ≤ 1e-5).

V10.A9 Cổng length_ratio: 1,00 ± 0,05 không đạt được bằng A2. Base trên val cũ
       (không có lặng đuôi) cũng chỉ 0,90, tức mô hình nói nhanh hơn thật khoảng 10%
       ở phần tiếng nói, điều A2 không sửa. Thay bằng:
       - MỤC TIÊU: |1 − length_ratio| ≤ 0,5 × |1 − mốc| (mốc fixtail 0,7665, tức
         length_ratio ≥ 0,8833 và ≤ 1,1168).
       - THEO DÕI: length_ratio_speech = tổng frame âm tiết dự đoán / thật, và
         length_ratio_pause = tổng frame gap dự đoán / thật, cả hai @ val fixtail
         free-run. Kỳ vọng A2: pause tăng rõ, speech gần như không đổi.
       Thêm vào eval_acoustic nếu chưa có (chỉ thêm trường, không đổi trường cũ);
       vì có chạm eval, phải sinh lại base40k_fixtail.json bằng code mới.

V10.A10 SẮC-tắc: kết luận tạm "giới hạn biểu diễn, không phải thiếu dữ liệu" (train
       lệch khoảng 13% ngay trong mẫu, ngang val +13,8%; nucleus < 4 không tệ hơn rõ).
       Chưa đủ để chốt vì so sai nhóm (phải so SẮC-tắc với SẮC-khác cùng ô độ dài) và
       thiếu số val. Làm lại khi MPS rảnh sau A2, không chặn gì.

V10.A11 Chạy thật A2: DUYỆT CÓ ĐIỀU KIỆN, để không mất thêm một vòng 3 giờ. Agent 2
       được khởi động khi ĐỦ cả 6 điều kiện, không cần chờ Agent 0:
       (1) test xanh, ruff sạch, đã push;
       (2) sha mốc khớp README theo cách tính ở V10.A3;
       (3) V10.B4 rơi vào NHÁNH NHIỄU;
       (4) base40k_fixtail.json đã sinh lại bằng code mới (V10.A9), H1 ≤ 3%;
       (5) đầu ra --dry-run chế độ thật có "--resume runs/acoustic_v2_base/last.pt",
           "--reset-val-state", manifest fixtail, "--max-steps 60000";
       (6) sau 1 phút chạy: dòng đầu history.csv là "# X3 resume:" với step 40000 và
           có các cột thành phần loss mới.
       Thiếu bất kỳ điều kiện nào: DỪNG, báo.
       Agent 1 đọc code SONG SONG khi A2 đang chạy. Nếu Agent 1 báo lỗi chặn, Agent 0
       sẽ ra lệnh dừng; checkpoint 5k nào trước lỗi vẫn dùng được tùy loại lỗi.

=====================================================================
B. AGENT 2 THỰC THI (theo thứ tự)
=====================================================================
V10.B1 Commit đầu tiên: docs/handoff/V10_agent0.md (nguyên văn tệp này) và
       docs/handoff/V09_agent0.md (bản Agent 0 đã gửi ở V09). TRANG_THAI.md trỏ
       V10, gốc 06b5f10.
V10.B2 Pin theo V10.A3 + bảng README theo V10.A4 + gỡ tệp theo V10.A5 + xóa khóa
       theo V10.A6. Báo: git config core.autocrlf trên M6, nội dung .gitattributes,
       lệnh và kết quả sha256 theo byte trong git cho từng tệp mốc.
V10.B3 Ghi thành phần loss theo V10.A8 + test.
V10.B4 Đo f0p tất định (script tạm, không commit, mps, không train):
       - checkpoint runs/acoustic_v2_base/last.pt @40000, model.eval(), cố định seed;
       - cùng 50 batch train (cùng danh sách id, cùng thứ tự) dựng từ 3 manifest:
         cũ, fixgap, fixtail;
       - báo mỗi thành phần loss (đặc biệt f0p) trung bình trên 50 batch, cho 3 manifest;
       - báo trung bình và σ của f0p trên 500 bước cuối history base 40k;
       - dán công thức chuẩn hóa f0p (tập tin:dòng: chia cho gì).
       Kết luận nhánh theo V10.A7.
V10.B5 Thêm length_ratio_speech, length_ratio_pause (V10.A9) + test; cổng mục tiêu
       length_ratio mới trong bench + test.
V10.B6 Toàn bộ test, ruff, push.
V10.B7 Sinh lại base40k_fixtail.json bằng code mới (mps). Kiểm H1. Ghim sha theo
       V10.A3. Báo bảng mốc đầy đủ và ngưỡng tuyệt đối mọi cổng A2.
V10.B8 Nếu đủ 6 điều kiện V10.A11: khởi động A2 thật bằng scripts/run_A2_fixgap.sh
       chế độ thật. Dán --dry-run và 5 dòng đầu history.csv vào báo cáo. Không chạy
       eval trên MPS khi A2 đang chạy. Báo ngay nếu NaN hoặc val tăng 3 lần liên tiếp.
V10.B9 Cập nhật V10_agent2.md, TRANG_THAI.md (gốc 06b5f10, commit cuối, trạng thái
       "A2 ĐANG CHẠY, CHỜ ĐỌC" hoặc "DỪNG, CHỜ ĐỌC"). Báo cáo có đủ: kết quả theo mã,
       TỰ QUYẾT KHÁC ĐẶC TẢ, CHƯA LÀM / LÀM DỞ.
V10.B10 Khi A2 tới 60k: bench 45k / 50k / 55k / 60k với --baseline-old base40k_old.json
       --baseline-fixgap base40k_fixtail.json; báo bảng 4 mốc × (cứng, mục tiêu, theo
       dõi), dòng KẾT LUẬN, tỉ số BCE pause val/train mỗi mốc. DỪNG. Không chạy A1.

=====================================================================
C. AGENT 1 ĐỌC (pull HEAD, đọc toàn bộ source, trả lời trong chat)
=====================================================================
Lưu ý: trước khi tính sha, làm mới bản checkout của eval/baselines/ (sau khi
.gitattributes có hiệu lực) và báo git config core.autocrlf trên máy mình.
V10.C1 sha256 hai tệp mốc theo byte trên đĩa VÀ theo git show HEAD:<tệp> | sha256sum.
       Khớp README không? Nếu khác nhau giữa hai cách, báo cả hai.
V10.C2 eval/baselines/ còn tệp nào được track? Mỗi tệp có gọn không (liệt kê khóa cấp
       cao)? summary_fixgap còn khóa cer/tone_C/creak không?
V10.C3 Thành phần loss ghi vào history: có đủ mọi thành phần của v2_losses không? Ghi
       ở train và validate? Có thay đổi giá trị loss không? Test tổng khớp ở đâu?
V10.C4 Công thức chuẩn hóa của f0p (và các loss theo frame khác): chia cho số frame
       hữu thanh, số frame có mặt nạ, hay tổng T? Thêm frame gap có làm đổi giá trị
       trung bình khi mô hình không đổi không? Dán dòng.
V10.C5 length_ratio_speech / length_ratio_pause: định nghĩa, mặt nạ dùng, trường cũ
       có bị đổi không. Cổng length_ratio mới đúng chiều không.
V10.C6 Nếu A2 đã khởi động: theo code, lệnh thật chạy có đúng 4 đối số ở V10.A11(5)
       không? Có đường nào khởi động lại sẽ reset val lần hai không?
V10.C7 Mọi vấn đề khác trên toàn bộ source (gồm các tệp V10 chạm và TRANG_THAI.md).
