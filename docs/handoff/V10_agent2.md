# VÒNG 10 | Báo cáo Agent 2 | Commit cuối: `403969e` · Commit gốc vòng: `06b5f10`

**Test**: 2.481 pass · ruff sạch · **Trạng thái: A2 ĐANG CHẠY, CHỜ ĐỌC** (khởi động 12:35, ~0,43 s/bước, tới 60k ≈ 2,3 giờ)

## Bảng số theo mã câu hỏi

| Mã | Việc | Kết quả |
|---|---|---|
| V10.B1 | Commit đặc tả đầu tiên | ✅ `baab50f` — `V10_agent0.md` + `V09_agent0.md` (bổ sung lịch sử) + TRANG_THAI trỏ V10/gốc `06b5f10`. Từ vòng này commit đầu = đặc tả nguyên văn |
| V10.B2 | Pin theo byte git + bảng README + gỡ fixgap + xóa khóa null | ✅ `e14d2dd` (+`d83230f` cho mốc mới). **`git config core.autocrlf` trên M6 = KHÔNG ĐẶT** (lý do đĩa==git ở đây, máy Agent 1 bật `true` ⇒ lệch). `.gitattributes` = `eval/baselines/*.json -text`. Mốc chuẩn hóa (LF + `sort_keys` + indent 2 + dòng trống cuối, `_write_json_normalized`). sha256 **theo byte trong git** (`git show HEAD:<tệp> \| shasum -a 256`): `base40k_old.json` = `92e90dad…`, `base40k_fixtail.json` = `a8b74d2e…` (bản cũ `53d85abd…` trước khi sinh lại B7). Gỡ `base40k_fixgap.json` khỏi git → `artifacts/bench/legacy/` (kèm 6 tệp raw `base40k_fixgap_*`); `.gitignore` chỉ cho `base*_old.json` + `base*_fixtail.json`. Khóa `cer: null` trong `summary_fixgap` **xóa hẳn** (+ emit tự xóa khóa null). Bảng README một dòng/tệp: tên \| sha256 \| val_manifest_hash \| DÙNG/KHÔNG DÙNG. **Test pin chạy MỌI lần test** (gợi ý Agent 1 #4): `test_pin_readme_khop_byte_trong_git` (README == byte git == byte đĩa) |
| V10.B3 | Thành phần loss vào history | ✅ `168d1fb` — `LOSS_COMPONENTS` (đúng tên `v2_losses.DEFAULT_WEIGHTS`, gồm `pause_gate`/`pause_gap`) + `history_header()` + `_pause_fraction()`; `_validate` gom mọi thành phần; train = trung bình cửa sổ `log_every`; **10 cột cũ giữ nguyên**, cột mới ở cuối; in log `comps: … pause_frac=…`. Test khóa: `Σ w·thành phần == loss tổng` (≤1e-5), tên khớp `DEFAULT_WEIGHTS`, header, `_pause_fraction`. Test cũ đọc `lr` theo header (không `row[-1]`) |
| V10.B4 | Đo f0p tất định → nhánh | ✅ **NHÁNH NHIỄU**. f0p/f0/f0_delta **+0,000%** giữa 3 manifest (cùng 50 batch × 8 dòng, cùng id/thứ tự); mel/pause đổi (đúng thiết kế A2 — xem §44.9). Công thức: họ f0 chia theo **mask hữu thanh/nucleus** — `v2_losses.py:124` `/f0_mask.sum()`, `:128` `/delta_mask.sum()×3`, `:135` `/nucleus_weight.sum()` (**KHÔNG chia T**) ⇒ thêm frame gap không đổi. So ±2σ (500 bước cuối base 40k, n=10): loss +4,31% (2σ 13,22%) · mel +13,16% (49,38%) · G +40,78% (70,58%) · tho −1,13% (17,50%) · f0d +5,32% (25,07%) · **f0p −16,40% (59,11%)** · sharp +2,64% (9,49%) — **cả 7 TRONG 2σ** ⇒ probe fixtail **ĐẠT**, tiêu chí ±5% → **±2σ**. Cách chia batch: `SameNBatchSampler` gộp theo **N = len(durations)** (`dataset_v2.py:167–188`); B4 gom cùng danh sách id theo N (50 batch × 8, giữ thứ tự manifest) ⇒ thành phần batch **giống hệt** giữa 3 manifest |
| V10.B5 | length_ratio tách phần + cổng A9 | ✅ `9b073c8` — `eval_acoustic` thêm `length_ratio_speech` (Σ frame âm tiết dự đoán/thật) + `length_ratio_pause` (Σ frame gap dự đoán/thật); cổng mới `\|1−lr\| ≤ 0,5×\|1−mốc\|` (mốc 0,7665 ⇒ [0,8833; 1,1168]) + in theo dõi; test khóa (0,8833 đạt · 0,8832 trượt · mốc None ⇒ NO_BASE) |
| V10.B6 | Test + ruff + push | ✅ 2.481 pass · ruff sạch |
| V10.B7 | Sinh lại mốc fixtail + ghim | ✅ `d83230f` — số cũ **không đổi** (syl_z 0,6621 · mel_gap 3,1272 · gap_z 2,1529 · pause_f1_all 0,5559 · gap_mae_all 156,46 · len_ratio 0,7665) + **2 trường A9: speech 0,9343 · pause 0,5332** (đúng kỳ vọng: pause chỉ 53% khối lượng thật). **H1 = +0,136% ≤ 3% ĐẠT** (so `syl_z` cũ 0,6612 → fixtail 0,6621; **đính chính**: công thức H1 đúng là syl_z↔syl_z — bản V09_agent2 ghi nhầm "mel_l1_z val cũ") |
| V10.B8 | Khởi động A2 (6 điều kiện) | ✅ **ĐỦ 6/6**: (1) test xanh+push · (2) pin khớp · (3) B4 = NHIỄN · (4) mốc sinh lại bằng code mới + H1 ≤3% · (5) dry-run đúng 4 đối số · (6) dòng đầu history `# X3 resume: …@40000` + cột thành phần mới (kiểm sau 3′30″, validate đầu ở 41k mới flush). Không NaN · 0,43 s/bước · mps 13–15G. Số đầu (bước 41000): val 8,8516 (mel hữu thanh 0,5972 · nét 0,355) · **train_pause_gate 0,124 vs val_pause_gate 0,934 (7,5×)** · train_pause_frac 2,6% vs **val_pause_frac 12,0%** |
| V10.B9 | Hồ sơ | ✅ tệp này + TRANG_THAI (trạng thái A2 ĐANG CHẠY) |
| V10.B10 | Bench 45/50/55/60k | ⏳ chờ A2 tới 60k (~2,3 giờ); **không chạy A1** |

## Ngưỡng tuyệt đối cổng A2 (từ mốc fixtail mới)
| Cổng | Mốc | Ngưỡng A2 |
|---|---|---|
| syl_z (tf, CỨNG) | 0,6621 | ≤ 0,6687 (×1,01) |
| mel_gap (free, CỨNG) | 3,1272 | ≤ 3,1272 |
| gap_z (tf, MỤC TIÊU) | 2,1529 | ≤ 1,830 (×0,85) |
| pause_f1_all | 0,5559 | ≥ 0,6289 (+0,073) |
| gap_sum_mae_all | 156,46 | ≤ 111,7 (×0,714) |
| length_ratio (A9) | 0,7665 | ≥ 0,8833 & ≤ 1,1168 |
| (theo dõi A9) | speech 0,9343 · pause 0,5332 | pause ↑ rõ, speech ≈ không đổi |

## Dán lệnh (V10.B8)
```
$ bash scripts/run_A2_fixgap.sh dry-run
.venv/bin/vifasttts train-v2 --config configs/small_codelung_v2_base_A2_fixgap.yaml \
  --train-manifest data/codelung/train_clean_v2_dur_seg_slot4_fixtail.jsonl \
  --val-manifest data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl \
  --output runs/acoustic_v2_A2_fixtail --resume runs/acoustic_v2_base/last.pt \
  --reset-val-state --max-steps 60000

$ head -3 runs/acoustic_v2_A2_fixtail/history.csv
# X3 resume: resumed_from=runs/acoustic_v2_base/last.pt@40000, val_hash_old=-, val_hash_new=cdbbd659116a…
step,phase,train_total,train_mel,mel_gap,mel_voiced,val_total,val_mel_voiced,val_rough,lr,train_pause_frac,train_mel,…,train_pause_gate,train_pause_gap,val_mel,…,val_pause_gate,val_pause_gap,val_pause_frac
41000,B,5.60196,0.75325,1.05501,0.44747,8.85156,0.59720,0.35502,5.6022e-05,0.02589,…,0.12449,0.16557,1.28266,…,0.93425,1.19025,0.12001
```

## TỰ QUYẾT KHÁC ĐẶC TẢ
1. **Vá mốc B7 cho 2 trường A9**: `_eval_block` của bench thiếu ánh xạ `length_ratio_speech/pause` (eval đã xuất đúng) ⇒ tôi lấy 2 số từ **chính raw của B7** (`artifacts/bench/base40k_fixtail_fixgap_free.json`) vá vào mốc, **đồng thời sửa bench** (`_scalar` + ánh xạ) để lần emit sau tự đúng. Không chạy lại 30′ bench.
2. **Emit xóa khóa null** áp cho cả `summary` lẫn `summary_fixgap` (A6 chỉ nói `summary_fixgap`).
3. **Gỡ luôn 6 tệp raw `base40k_fixgap_*.json`** khỏi `eval/baselines/` → `artifacts/bench/legacy/` (A5 chỉ nói `base40k_fixgap.json`; giữ `eval/baselines` đúng nghĩa "chỉ mốc gọn").
4. **§44.9 viết ở `3bd6d04`** (trước B7) — số B4 + thiết kế pin/dọn mốc/A8/A9.
5. Test cũ `test_stop_at_step…` đổi đọc `lr` theo **header** (cột mới thêm sau cột `lr` ⇒ `row[-1]` sai).

## CHƯA LÀM / LÀM DỞ
- **V10.B10** (bench 45/50/55/60k + tỉ số BCE pause val/train) — chờ A2 tới 60k; **không chạy eval trên MPS khi A2 đang chạy** (tuân thủ).
- **SẮC-tắc làm lại (A10)**: chưa — phải so SẮC-tắc với **SẮC-khác cùng ô độ dài** + bổ sung số **val**; để sau A2 (không chặn).
- **Đính chính H1**: công thức trong `V09_agent2.md` ghi nhầm (syl_z vs mel_l1_z); bản đúng ở tệp này + §44.9.
