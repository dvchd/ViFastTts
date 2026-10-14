# VÒNG 11 BỔ SUNG | Đặc tả: docs/handoff/V11_agent0_bosung.md | Gốc: b1ae8ce
Bổ sung cho V11_agent0.md, không thay thế. Agent 2 THỰC THI Phần B.
Căn cứ: báo cáo đọc V10.C1–C7 của Agent 1 trên HEAD b1ae8ce.

=====================================================================
A. QUYẾT ĐỊNH
=====================================================================
V11S.A1 V10.C1–C6: ĐẠT theo đọc code độc lập. Pin, gỡ tệp, cột thành phần loss,
        chuẩn hóa f0, length_ratio tách, lệnh thật đều xác nhận. Vòng V10 ĐÓNG.

V11S.A2 Watchdog gặp last.pt hỏng (Agent 1 mục 4): ĐÂY LÀ MỤC QUAN TRỌNG NHẤT.
        Hiện tại torch.load lỗi → step_of trả 0 → rơi vào nhánh khởi động đầu →
        resume từ base 40k + reset val, ghi nối vào history.csv cũ. Tức là một lần
        ghi checkpoint dở dang có thể âm thầm xóa toàn bộ tiến độ A2 và làm bẩn history.
        Sửa:
        - step_of: chỉ trả 0 khi last.pt KHÔNG tồn tại. Tồn tại mà đọc lỗi: thử
          checkpoint 5k gần nhất còn đọc được; không có thì DỪNG có mã lỗi, không
          tự khởi động lại.
        - Ghi checkpoint nguyên tử: ghi last.pt.tmp, fsync, rồi đổi tên thành last.pt.
        - Nhánh khởi động đầu chặn nếu thư mục đã có history.csv.
        Test: last.pt hỏng + có ckpt 5k → resume từ 5k; hỏng + không có ckpt → dừng;
        nhánh đầu + history.csv có sẵn → dừng.
        THỜI ĐIỂM: KHÔNG sửa watch_train_v2.sh khi watchdog của A2 còn đang chạy (bash
        đọc script dần theo dòng, sửa tại chỗ có thể làm hỏng tiến trình đang chạy).
        Phần ghi nguyên tử trong train_v2.py cũng chỉ có hiệu lực ở lần khởi động sau.
        Làm sau khi A2 tới 60k, trước mọi run dài tiếp theo (A1 nếu có).
        Trong lúc A2 chạy: nếu watchdog khởi động lại, báo ngay và kiểm dòng
        "# X3 resume" mới trong history.csv.

V11S.A3 TRANG_THAI "commit cuối" luôn lệch một commit (Agent 1 mục 2): không thể tự ghi
        hash của chính commit chứa nó. Đổi quy ước: trường ghi "commit code cuối" (commit
        cuối trước commit hồ sơ). HEAD luôn là commit hồ sơ; Agent 1 đọc khoảng
        <gốc>..HEAD. Bỏ commit phụ chỉ để điền hash.

V11S.A4 Probe ghi đè im lặng (mục 3): chế độ probe chặn nếu thư mục probe đã có last.pt,
        trừ khi truyền --overwrite-probe. Test.

V11S.A5 Test pin cần git (mục 1): nếu không có binary git hoặc không phải repo git thì
        pytest.skip với lý do rõ ràng; phần README == byte đĩa vẫn chạy.

V11S.A6 Hiệu chuẩn của Agent 1 (mục 5): ghi nhận. Hai lần dương tính giả (U8, W10) đều do
        môi trường hoặc đọc giữa chừng, đã tự đính chính bằng bằng chứng. Không cần hành động.

=====================================================================
B. AGENT 2 THỰC THI
=====================================================================
V11S.B1 Commit tệp này nguyên văn cùng commit đầu của V11 (hoặc ngay sau nếu V11 đã bắt đầu).
V11S.B2 Gộp A3, A4, A5 vào V11.B4 (chỉ CPU, làm được trong lúc A2 chạy; A4 chạm script
        run_A2_fixgap.sh, KHÔNG chạm watch_train_v2.sh).
V11S.B3 A2 (watchdog + ghi nguyên tử): làm sau V11.B6, trước V11.B8. Nếu V11.B7 không rơi
        nhánh A1 thì vẫn làm, trước run dài kế tiếp.
V11S.B4 Báo cáo gộp vào V11_agent2.md.

=====================================================================
C. AGENT 1 ĐỌC (thêm vào V11.C1–C7)
=====================================================================
V11S.C1 step_of / watchdog: còn đường nào khiến last.pt hỏng bị coi như "chưa có"? Ghi
        nguyên tử có fsync trước rename không? Nhánh đầu có chặn history.csv có sẵn không?
