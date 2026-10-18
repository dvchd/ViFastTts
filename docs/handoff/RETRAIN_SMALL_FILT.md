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
| Train | ✅ **XONG** 60k lúc 17:35 — `runs/acoustic_v2_small_filt` (best val 0,6277 @38k; số 40-hàng chỉ để theo dõi) · recipe b8 (batch 8, cosine 60k, seed 42, KHÔNG oversample) |
| Sự cố & tự phục hồi | Process bị **giết im lặng** ở step 27.000 lúc 13:20 (không crash/jetsam/sleep — nghi phiên harness dọn nền, xem §6) → dựng `tools/watchdog_resume_train.sh` chạy qua **launchd**; 14:40:12 chạy lại, nối đúng lịch cosine. Đã gỡ job khi train xong |
| Đo | ✅ **XONG** (18/10 tối) — bảng số + cổng ở §5; mốc b8 trên val lọc đã ghim (`eval/baselines/b8_filt.json`) |

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
- `12b8f5f` — `tools/watchdog_resume_train.sh` (tự resume qua launchd) + §6 sự cố
- `45909aa` + `a03af29` — mốc b8 trên val lọc (`eval/baselines/b8_filt.json` + ghim sha256)
- `58ad65b` — kết quả train + bench (§5 + ACOUSTIC_V2 §46)

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

## 5. Kết quả (18/10 tối) — **TRƯỢT CỔNG** ⇒ giữ b8; tai người vẫn quyết định

Số đo cùng `val_filt`/`holdout_filt` (mốc `eval/baselines/b8_filt.json`, sha ghim trong `eval/README.md`;
chi tiết + đọc đúng ở `docs/ACOUSTIC_V2.md` §46):

| Chỉ số | b8 (46k) | small_filt 38k | small_filt 60k | hướng |
|---|---|---|---|---|
| mel_z | 1,1580 | **1,1561** (≈ nhiễu) | 1,1849 ✗ | ↓ |
| mel dB | **8,64** | 9,17 | 9,16 | ↓ |
| độ nét rough | 0,3423 | **0,3583** | 0,3567 | ↑ |
| thanh C | 0,7374 | 0,7816 | **0,7955** | ↑ |
| CER-100 | **0,2126** | 0,2498 | 0,2503 (base40k 0,2615) | ↓ |
| F0 NGÃ | 198,1 | **179,4** | 179,7 | ↓ |
| dur MAE | **4,74** | 4,94 | 5,03 | ↓ |
| rè Δjitter ngã | **+0,0294** | +0,0440 | +0,0388 | →0 |
| rè Δhnr ngã | −3,10 | −3,12 | **−2,86** | →0 |

- **Cổng EVAL_TOOLS §4**: 38k trượt **CER** + **rè**; 60k trượt thêm **mel_z** ⇒ theo luật: **giữ b8**.
- **Tai người 18/10 (nghe hết 3 bản A/B qua tunnel)**: *"ngang ngang nhau — chỗ nét vẫn nét, chỗ rè vẫn rè"*
  ⇒ không bản nào thắng tai; cổng cũng trượt ⇒ **giữ b8 làm daily driver**; kết luận bước này:
  **lọc dữ liệu chưa tạo khác biệt nghe được** (mức thắng thanh +0,04–0,06 acc **dưới ngưỡng nghe**).
- **Đòn kế tiếp (18/10 tối)**: adversarial T4.1 retry **phân kỳ lần 3** ở probe (600 bước, loss → 7,7
  triệu — chi tiết `docs/ACOUSTIC_V2.md` §46.4) ⇒ bỏ; chuyển **`mel_sharp` toàn cục ×2**
  (`configs/small_codelung_v2_melsharp.yaml`, resume b8/best@46k → 70k, probe ĐẠT) — run
  `runs/acoustic_v2_b8_melsharp` (watchdog `vifasttts.melsharp`). Kèm: **quét α làm nét cho tai**
  (α=1,5 vs 3,0) qua tunnel — kết quả tai quyết định bước tiếp.
- **Tai 18/10 tối (α + cặp tối thiểu)**: α=1,5 ≈ 3,0 (3 hơi sắc hơn) — không sửa rè; *"tôi đọc chữ"
  nghe sạch, một số từ đuôi nghe rè* ⇒ chẩn đoán §46.5: **đuôi phân bố mỏng** (41% âm tiết <5 lần;
  bão ×1 · hộc ×0 · khanh ×0…) + **hiệu ứng thanh** (synth ngã/nặng lệch −3,1/−1,6 dB hnr) lớn hơn
  hiệu ứng hiếm (~0,33 dB, xu hướng). Chain đã hẹn giờ: sau `mel_sharp` → **oversample** (E4b recipe,
  run `runs/acoustic_v2_b8_oversample`, job `vifasttts.oversample`). Chờ người dùng: thu âm đuôi hiếm
  / link kênh.
- Nhưng **thanh/F0/nét thắng rõ** (C 0,738→0,78–0,80 ≈ base40k; F0 NGÃ −19 cents) — đúng điểm đau ⇒
  **tai người A/B vẫn là chốt cuối** (§37).
- CER-100 đo lại vì bẫy ±0,03: b8 0,2126 < 38k 0,2498 ≈ 60k 0,2503 < base40k 0,2615 ⇒ bản lọc ở mức
  **của base40k** (bản tai user chọn), không phải bất thường — nhưng kém b8 thật (+0,037).
- **Demo nghe** (không commit): `artifacts/listen_small_filt/index.html` — b8@46k vs 38k vs 60k, cùng
  Vocos + sh25 α=1,5, 4 clip giàu ngã/nặng + 5 cặp tối thiểu + orig/copy.
- **Bước sau (chờ tai + chọn)**: (1) giữ b8 → adversarial resume / mel_sharp toàn cục; (2) theo hướng
  lọc → xác minh phương sai (seed 2) hoặc lọc+oversample; (3) luôn theo cổng + tai (§37).

## 5b. Bàn giao

- Kết quả + bảng cổng: §5 ở trên + `docs/ACOUSTIC_V2.md` §46 (kết luận lâu dài).
- Lệnh tái lập đầy đủ (QC → lọc → train → bench) ghi ở §4 + §6 (watchdog).

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
