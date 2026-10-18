# RETRAIN SMALL trên dữ liệu lọc — phiên tự động 18/10/2026

> Mục tiêu (yêu cầu người dùng): lọc lại dữ liệu audio hiện có (một số video "rè sẵn")
> → **train lại small TỪ ĐẦU** với quy trình tốt nhất → đo bằng bộ thước, chốt bằng số.
> Quyết định dựa dữ liệu thực tế, không dựa kinh nghiệm cũ.

## 1. Trạng thái

| Pha | Kết quả |
|---|---|
| QC clip | ✅ `tools/audit_clip_quality.py` — 7.660 clip (3 manifest), **10 giây** (8 workers). Corpus **sạch bất ngờ**: 0 clip clipping, SNR p1 = 30,8 dB, nền p1 = −88,6 dB → "rè-sẵn" KHÔNG phải nền nhiễu |
| Phát hiện chính | Nhóm **hiss/sizzle** = `hf_ratio_quiet` (cao tần ở khung lặng) + `flatness` — tập trung ở ~15 video (nhiều nhất `8Pq2FwmIEU8` 26 clip) |
| Luật lọc | Loại khi `hf_ratio_quiet` z > 8 **HOẶC** `flatness` z > 12 (median/MAD từ corpus) → **104 clip (1,36%)** |
| Lọc manifest | ✅ `tools/filter_manifests.py` → `*_filt.jsonl`: train 9.763→**9.672** row (−82 clip) · val 1.773→**1.762** (−10) · holdout 842→**830** (−12). Ô hiếm giữ tỉ lệ (NGA 10.113→10.067; NẶNG+tắc 9.554→9.504) |
| Probe | ✅ 120 bước: loss ổn định (không NaN), **0,23 s/bước**, MPS ~4G |
| Train | 🔄 **ĐANG CHẠY NỀN** — `runs/acoustic_v2_small_filt` · log `artifacts/dataset_v2/train_v2_small_filt.log` · 0→60k bước (config `small_codelung_v2.yaml`: batch 8, cosine tới 60k, patience 25, KHÔNG oversample) · dự kiến **~5–6h** |
| Sự cố & tự phục hồi | Process bị **giết im lặng** ở step 27.000 lúc 13:20 (không crash/jetsam/sleep — nghi phiên harness dọn nền, xem §6) → dựng `tools/watchdog_resume_train.sh` chạy qua **launchd**; 14:40:12 chạy lại, nối đúng lịch cosine. **Gỡ job khi xong: `launchctl remove vifasttts.small_filt.watchdog`** |
| Đo | ⏳ chờ train xong (xem §4) |

## 2. Quyết định đã chốt (kèm lý do)

1. **Một biến số duy nhất**: dữ liệu lọc, giữ nguyên recipe b8 (60k, cosine, batch 8, seed 42,
   không oversample, không adversarial) → so sánh sạch. Oversample (E4b −13% creak nhưng từng bị
   TỪ CHỐI cổng mel) để thành **run #2** nếu run #1 đạt.
2. **KHÔNG dùng HNR thấp làm tín hiệu loại** — creak giọng là đặc trưng cần giữ (`eval_creak` so với
   mốc thật); chỉ loại hiss/sizzle (artifact thu).
3. **Không sửa bản gốc**: mọi bản lọc là `_filt.jsonl` mới; cache v3 dùng lại nguyên (chỉ bỏ index).
4. Manifests nền = `train/val_clean_v2_dur_seg_slot4` (bản b8 đã dùng, trước fixtail) — không dùng
   `typo1` (đang lỗi sync text/syllables, sẽ xử riêng).

## 3. Commit liên quan

- `446460a` — tools QC + filter (+ test); artifacts: `artifacts/clip_qc/{clip_quality,drop_list,filter_report}.json`
- `<watchdog>` — `tools/watchdog_resume_train.sh` (tự resume qua launchd) + §6 sự cố
- (sẽ cập nhật) — kết quả train + bench

## 4. Kế hoạch đo sau khi train xong (chuẩn bị sẵn)

1. **Mốc trên val MỚI**: bench `runs/acoustic_v2_b8/best.pt` với `--val-manifest val_*_filt.jsonl`
   → baseline (mốc) **cùng nhãn** cho mọi cổng. (Không chạy trong lúc máy đang train — bench dùng MPS.)
2. **Bench model mới**: `bench_checkpoint.py --config runs/acoustic_v2_small_filt/resolved_config.json
   --checkpoint …/best.pt --val-manifest val_filt --device mps` → so mốc b8-filt:
   cổng: mel_z ≤ ×1,01 · dB ≤ +0,10 · rough ≥ −0,01 · tone_C ≥ −0,01 · CER ≤ +0,03 · creak không phình.
3. `eval_creak` (mốc thật 0,1893) + `tone_ruler` + rough 60 clip cho cả b8 và bản mới (A/B số).
4. Demo A/B wavs (b8 vs mới) cho tai người — 5 phút nghe là điểm chạm người duy nhất.
5. Chốt: nếu đạt cổng → đổi "daily driver" (ghi rõ), cập nhật hồ sơ + commit; nếu trượt → báo cáo số,
   giữ b8, đề xuất bước sau (oversample run #2 / mel_sharp toàn cục).

## 5. Bàn giao (khi xong)

- Kết quả + bảng cổng ở mục mới của tệp này (hoặc `docs/ACOUSTIC_V2.md` nếu là kết luận lâu dài).
- Lệnh tái lập đầy đủ (QC → lọc → train → bench) ghi ở đây.

## 6. Sự cố 18/10 13:20 — process bị giết im lặng (đã tự phục hồi)

- Train chết ở **step 27.000 lúc 13:20:03** (mtime `ckpt_27000.pt`/`last.pt`). Không crash report,
  không jetsam, không sleep/wake, log sạch → **bị giết từ bên ngoài** (nghi phiên/harness dọn tiến
  trình nền), không phải lỗi code.
- `last.pt` nguyên vẹn (step 27.000 · best_val 0,6305) → resume nối đúng **lịch cosine** (LR tính
  theo bước tuyệt đối; KHÔNG đổi `max_steps` khi resume).
- Lớp tự phục hồi: `tools/watchdog_resume_train.sh` chạy qua **launchd** (thoát khỏi process-group
  của phiên), tự `--resume last.pt` khi chết (tối đa 8 lần), chống chạy trùng; ghi dấu vào log.
- 14:40:12 resume thành công từ 27.000 → 14:44:59 tới 28.000 (0,30 s/bước) — **ETA ~17:30**.
- **Việc phải làm khi run xong: `launchctl remove vifasttts.small_filt.watchdog`** (launchd tự chạy
  lại job sau khi thoát — không gỡ thì lặp vô hại nhưng rác log).
