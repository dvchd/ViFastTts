# VÒNG 09 | Báo cáo Agent 2 | Commit cuối: `5d5827a` · Commit gốc vòng: `475bdb4`

**Test**: 2.474 pass · ruff sạch · **Trạng thái: DỪNG, CHỜ ĐỌC**

## Bảng số theo mã câu hỏi
| Mã | Việc | Kết quả |
|---|---|---|
| V09.B1 | Lệnh B9 đã chạy + trường hợp V09.A5 | ✅ Lệnh: `bench_checkpoint.py --config runs/acoustic_v2_base/resolved_config.json --checkpoint runs/acoustic_v2_base/last.pt --tag base40k_fixtail --step 40000 --device mps --clips 300 --val-manifest data/codelung/val_..._fixtail.jsonl --holdout-manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl --baseline-old eval/baselines/base40k_old.json --out-dir artifacts/bench --emit-baseline eval/baselines/base40k_fixtail.json` → **trường hợp A5-3**: có `--baseline-old`; pin README lúc đó **khớp** sha thực ⇒ B9 **hợp lệ**; `git_commit` trong tệp = `475bdb4` ✓ (commit tại thời điểm chạy) |
| V09.B2 | Ghim lại sha + "chưa ghim ⇒ dừng" | ✅ `base40k_old` 3d6fefe5… **khớp**; `base40k_fixtail` 7be6fcff… **khớp** + đã vào git (`.gitignore` bổ sung `base*_fixtail.json`); `base40k_fixgap` ghim sha thực cf770864… + ghi "thay bởi fixtail, KHÔNG dùng cho cổng". Bench: **chưa ghim ⇒ SystemExit** (đổi từ cảnh báo), test khóa |
| V09.B3 | Watchdog một mảng lệnh | ✅ Viết lại `tools/watch_train_v2.sh`: hàm `build_cmd` dựng **mảng `CMD`**; `DRY_RUN=1` in `"${CMD[*]}"`, chế độ thật `"${CMD[@]}"`; `TRAIN_CMD` ghi đè (test); `WATCH_MAX_ATTEMPTS`. **Test 3 ca** (`tests/test_watch_train_v2.py`): (a) lần đầu có `--resume runs/acoustic_v2_base/last.pt` + `--reset-val-state`; (b) có `last.pt` ⇒ `--resume <OUT>/last.pt`, **không** reset; (c) dry-run == `<TRAIN_CMD> <đối số script giả nhận>`. `run_A2_fixgap.sh dry-run` **gọi đúng watchdog** (in đúng mảng) |
| V09.B4 | §44.8 theo thực tế | ✅ Sửa câu "W5 đã sửa" ⇒ "bản V08 **chưa** sửa được (patch không áp), sửa ở V09.B3" + test khóa |
| V09.B5 | Test + push | ✅ 2.474 pass |
| V09.B6 | B9 (mốc fixtail) + H1 | ✅ (đã chạy V08.B9) **H1 ĐẠT**: syl_z cũ 0,6612 → fixtail **0,6621 (+0,14% ≤3%)**; mốc: syl_z 0,6621 · gap_z 2,1529 · gap_mae_all **156,46** · pause_f1_all 0,5559 · len_ratio 0,7665; ngưỡng A2: syl_z ≤0,6687 · gap_z ≤1,830 · gap_mae_all ≤111,7 · pause_f1_all ≥0,6289 · len_ratio 1,00±0,05 |
| V09.B7 | B10 probe fixtail | ✅ (đã chạy V08.B10) LR 6,03e-5/6,00e-5 (=0,3018×lr_max) · không NaN/OOM · RAM đỉnh 9,5 GB · 0,379–0,388 s/bước · checkpoint @40100 + `--validate-at-stop` (val 8,8275 · mel hữu thanh 0,5928 · nét 0,372). So 500 bước cuối base 40k: `loss +5,9%` · **`mel +11,7%`** · **`G +26,4%`** (fixgap probe +33,9%/+41,8% ⇒ **ít hơn** ✅) · `tho +0,6%` · `f0d +4,9%` · `sharp +2,0%` · **`f0p −18,9%` ❌ ngoài ±5%** ⇒ theo nhánh "kết quả lạ": **DỪNG, báo** |
| V09.B8 | Hồ sơ + SẮC-tắc | ✅ tệp này + TRANG_THAI (V09) + SẮC-tắc **đang chạy nền** (`/private/tmp/opencode/sac_coda.py`, coda×độ dài nucleus, val300 + train in-sample) |

## TỰ QUYẾT KHÁC ĐẶC TẢ
1. `.gitignore` bổ sung `base*_fixtail.json` (đặc tả A4 chỉ nói ghim sha — nhưng không cho phép tên này thì không commit được tệp mốc).
2. `run_A2_fixgap.sh` thêm `--validate-at-stop` vào lệnh probe (A6 nói giữ tiêu chí V08.A5; `--validate-at-stop` là V08.B6 nên tôi bật luôn để có BCE pause).
3. `base40k_fixgap.json` ghim sha thực **và** đánh dấu không dùng (A4 chỉ nói ghim lại; tôi thêm nhãn để không ai dùng nhầm).

## CHƯA LÀM / LÀM DỞ
- **BCE pause train/val của probe**: log `train-v2` không in riêng BCE pause; `--validate-at-stop` chỉ ghi `val_total/val_mel_voiced/val_rough` ⇒ **chưa có số BCE pause** cho probe (cần thêm cột hoặc in riêng — chưa làm).
- **`f0p` lạ (−18,9%)**: chưa phân giải (nghi nhiễu 2 mẫu do `log_every=50`); chờ Agent 0 chọn (a) chấp nhận, (b) probe lại với `log_every=10`, (c) bỏ `f0p` khỏi tiêu chí probe.
- **SẮC-tắc**: đang chạy nền, chưa có số (nhường MPS cho B9/B10 theo lệnh; giờ mới chạy).
- **Chạy thật A2**: chưa (đúng V09.A7).
