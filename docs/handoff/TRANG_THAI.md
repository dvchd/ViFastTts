# TRANG_THAI — trạng thái hiện hành (nguồn duy nhất, Agent 2 cập nhật mỗi vòng)

> **V18.39 (23/10/2026)** — audit UTMOSv2: `utmosv2` dùng **random crop + mixup bằng np.random không seed** ở cả prediction; cùng seed **exact**, khác seed mỗi file lệch trung bình **~0,28 điểm** (max 0,45–0,67). Chấm lại 36 WAV cũ bằng **5 seed [0..4]**: UTMOS P2 1,4899 · REC 1,4568 · legacy 1,5207 · fixed 1,4957; ΔUTMOS **fixed−legacy −0,0264 CI [−0,0554, +0,019]** (mất ý nghĩa so với 1 pass), **fixed−rec +0,0176 CI [−0,0003, +0,053]** (chỉ xu hướng), **legacy−P2 +0,0223 CI [+0,0045, +0,0461]** nhưng BAK không xác nhận. BAK/flat/HNR của v18.38 giữ; các ΔUTMOS của v18.38 **hạ mức**. CI bootstrap video **chưa gồm** bất định do giám khảo; seed range báo riêng. **Giữ P2, crop fixed là mặc định vì loại padding, không tuyên bố cải thiện chất lượng.** Chi tiết: `V1839_agent2.md`, `artifacts/bench/v1839_utmos_audit/`.
> **V18.38 (23/10/2026)** — probe crop D **3 nhánh** từ P2 EMA (300 D warm-up + 200 G update; REC 200 G): **không nhánh nào tốt hơn P2** trên bộ 36 câu dev. `gan_fixed` vs `gan_legacy`: ΔBAK +0,0041 CI95 [−0,0112, +0,0084] (không khác), ΔUTMOS −0,0507 (**1 pass chưa seed — xem v18.39**), ΔHNR +0,036 (fixed hơn), Δflat +0,0011 (fixed phẳng hơn = kém hơn). `gan_fixed` vs `rec`: BAK +0,0201, UTMOS +0,1064 (**1 pass chưa seed — xem v18.39**) nhưng **chưa hơn P2**. Crop cũ: 35/3.975 row (0,88%) và 31/500 bước có padding, 1.098 frame; crop fixed: 0. **Giữ P2, không mở training dài.** Chi tiết: `V1838_agent2.md`, `runs/v1838_crop_probe/report.json`.
> **V18.37 (23/10/2026)** — kiểm tra tính đúng pipeline, **không training, không đổi production**. Xác nhận bằng tensor: ONNX acoustic cũ = **P2 EMA** (180/181 tensor exact); ONNX vocoder cũ = **E4-C** (63/63 tensor exact, 0/63 pretrained) ⇒ **mismatch**, đã export bản sửa ở `artifacts/onnx/v1837/` (chưa dùng). Sửa text eval giữ dấu câu (`vifasttts/eval_text.py`); raw-join cũ chỉ còn là legacy. Trace 12 câu: legacy **0 gap**, bản có dấu câu **173 gap**; CLI/ONNX chunk dài hơn sentence mode (ghi rõ, không ép giống). Audit crop D: bản cũ có row 34 frame bị crop 64 ⇒ **33 frame padding**; đã sửa `sample_crops(lengths=...)`, test mới. Bộ split: val∩holdout = **9 video**, nhưng **0 clip cha/segment** ⇒ holdout là unseen-clip, không phải unseen-video. **Luật hiện hành chỉ còn BAK + UTMOS; CER đã bị loại, số cũ giữ trong lịch sử.** Chi tiết: `V1837_agent2.md`, JSON `artifacts/bench/v1837_*.json`.
> **V18.35 (23/10/2026)** — sàng lọc 4 checkpoint còn tồn tại (g1_1k, g2_1k, p2_4k, g2_2k) trên 110 câu/11 video: **không bản nào đạt luật** (trượt điều kiện BAK) ⇒ **giữ P2**, không đổi production, không training. `g1_1k`: UTMOS cải thiện, **BAK chưa xác định được chiều khác biệt**, chưa đạt mục tiêu cải thiện nền — không phải bản thắng, cũng chưa bị chứng minh kém toàn diện. Chi tiết: `A0_V18_GROUP1.md` §26.
> **V18.34 (23/10/2026)** — audit v18.33: sửa nhãn chiều (flat_V G1 thắng 88 %), **loại CER/WER khỏi luật** (macro/micro lệch; "sàn 0,188" là mẫu 8 câu khác tập), "speech frames" = **voiced frames**, UTMOS không đơn điệu từng câu với cắt. **Giữ P2** (BAK+UTMOS CI95 < 0, 0/12 video) nhưng **hạ mức kết luận**: "G1 bị giám khảo tự động đánh giá thấp hơn P2, đặc biệt DNSMOS BAK". Chi tiết: `A0_V18_GROUP1.md` §25.4–25.5, `artifacts/bench/v1834_audit.json`.
> **V18.33 (23/10/2026)** — đánh giá TỰ ĐỘNG 300 câu/12 video: **P2 tốt hơn G1** (BAK −0,40 CI[−0,46,−0,35]; UTMOS −0,10; CER +0,063; 0/12 video G1 hơn BAK) ⇒ **production rollback về P2**, G1 giữ làm thí nghiệm. Nhãn "đánh giá tự động", KHÔNG phải xác nhận tai. Chi tiết: `A0_V18_GROUP1.md` §25, `artifacts/bench/v1833_decision.json`.
> **V18.32 (23/10/2026)** — bộ nghe mù 36 câu (12 video) tại `artifacts/listen_small_filt/v1832_blind/` + chẩn đoán TF/free-run (`artifacts/bench/v1832_tf_free.json`); cơ chế nghịch lý nền **chưa xác định**; G1 vẫn CHƯA XÁC NHẬN TAI.
> **V18.31 (23/10/2026, lịch sử)** — nhánh khi đó là **rè nền / GAN acoustic**; báo cáo đầy đủ ở
> `docs/handoff/A0_V18_GROUP1.md` §20–§23 (mục cũ V11 bên dưới là lịch sử, không còn hiệu lực).
> **Production acoustic (v18.37)**: **P2 EMA** `artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt`
> (SHA256 `44dd0da6…`; `model` == `ema` của `runs/adv_p2_20k/ckpt_2000.pt`) ·
> config `configs/small_codelung_v32.yaml` · metadata `configs/production_v1831.json` (đã cập nhật v18.37).
> **Vocoder production**: Vocos pretrained `artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin`
> (SHA256 `97ec976a…`). **ONNX cũ không phải production**: vocoder ONNX cũ khớp E4-C, xem
> `artifacts/bench/v1837_provenance.json`; bản sửa mới ở `artifacts/onnx/v1837/`.
> **Trạng thái**: DỪNG SAU V18.37 — không training; các kết luận BAK/UTMOS cũ bị giới hạn bởi
> punctuation/inference mode/EMA-live/vocoder mismatch; xem `V1837_agent2.md` §9.
> **Trang nghe cũ**: `/v1829_floor/`, `/v1830_gan/` (lịch sử; không phải production).

**Vòng hiện hành**: **V18.39** · **Tệp đặc tả**: yêu cầu A0 → Agent 2 trong chat (không có `V1839_agent0.md`) ·
**Commit gốc của vòng**: `5e31677` · **Trạng thái**: **DỪNG SAU V18.39, CHỜ ĐỌC** — không training, giữ P2.
> Các mục V11/V10 bên dưới là **lịch sử**, không còn hiệu lực cho quyết định production.
> Quy ước V11S.A3 vẫn dùng: HEAD là commit hồ sơ; đọc diff `<gốc>..HEAD`.
**Việc tiếp (chỉ đề xuất, chưa làm)**: xem `V1839_agent2.md` §7 — không mở training dài từ probe này;
nếu muốn kiểm crop fix, chạy seed 2 với cùng 500 batch và **cùng cấu hình UTMOS 5 seed**; nếu vẫn không khác,
dừng nhánh crop. BAK/UTMOS trên bộ độc lập/30 câu người thật vẫn chưa làm.

> Agent 1 đọc vòng này theo thông tin ở đầu tệp (không cần hash gửi kèm): diff `bfde6a7..<commit cuối>`,
> đối chiếu `V1837_agent2.md` §2–§9. Agent 1 **chỉ đọc, không ghi tệp** vào repo.
> Quy ước cũ: commit đầu vòng là tệp đặc tả; vòng này đặc tả nằm trong chat, không có `V1837_agent0.md`.

## 1. Mốc & hash ghim (V10.A3: ghim sha256 theo byte TRONG GIT — `git show HEAD:<tệp> | shasum -a 256`)
| Đối tượng | Hash | Ghi chú |
|---|---|---|
| manifest train fixtail | `965b4798…` (full trong `eval/README.md`) | Σgap/row 294,9 · phủ pause 71,4% |
| manifest val fixtail | `cdbbd659…` | Σgap/row 303,0 · phủ 74,0% |
| manifest val CŨ | `19b6e117…` | dùng cho CỔNG CỨNG (không đổi) |
| holdout CŨ (CER) | `3ca7c79b…` | clip-level 842 hàng |
| mốc `eval/baselines/base40k_old.json` | `2bc3ca93…` | val cũ — CỔNG CỨNG (pin theo byte git); rough 60 clip 0,3679 |
| mốc `base40k_fixtail.json` | `e29509d7…` | sinh lại (B7) + `length_ratio_speech/pause`; rough 60 clip 0,3679/0,3419; mel_z_fix 1,3311 · dB_fix 9,879 |
| `base40k_fixgap.json` | `cf770864…` — **KHÔNG DÙNG** | đã gỡ khỏi git: `artifacts/bench/legacy/` (V10.A5) |
| model thước `tone_ruler.pt` | không versioned | train lại: `tools/tone_ruler.py --train` |

## 2. Cổng A2 (tương đối so mốc cùng bản val)
- **CỨNG** (None = TRƯỢT): mel_l1_z(tf@cũ) ≤ mốc×1,01 · mel_db ≤ mốc+0,10 · rough(từ checkpoint@cũ) ≥ mốc−0,01 ·
  tone_C(@cũ) ≥ mốc−0,01 · CER(holdout cũ) ≤ mốc+0,03 · F0 NGÃ ≤ mốc+5¢ · syl_z(tf@fixtail) ≤ mốc×1,01 ·
  mel_gap(free@fixtail) ≤ mốc.
- **MỤC TIÊU**: pause_f1_all ≥ mốc+0,073 · gap_sum_mae_all ≤ mốc×0,714 ·
  length_ratio(free) theo V10.A9: |1 − lr| ≤ 0,5×|1 − mốc| (≥0,8833 & ≤1,1168) · gap_z(tf) ≤ mốc×0,85.
- **THEO DÕI (V10.A9)**: length_ratio_speech (mốc 0,9343) · length_ratio_pause (mốc 0,5332) @ val fixtail free-run.
- Kết luận bench: TỪ CHỐI / ĐẠT CỨNG–TRƯỢT MỤC TIÊU / ĐẠT CỨNG–MỤC TIÊU CHƯA ĐỦ SỐ / CHẤP NHẬN.

## 3. Hằng số & quyết định đang hiệu lực
- **C_TAIL = 67 frame** (trần lặng đuôi, chỉ đoạn cuối clip; cắt cả nhãn lẫn `frame_range[1]`) — §44.8.
- `--reset-val-state` + `INIT_RESUME/INIT_RESET_VAL_STATE` (watchdog chỉ dùng khi `OUT/last.pt` chưa có).
- sh25 **giữ toàn cục** (không gated) — đóng mục 5c.
- Batch 8 + `SameNBatchSampler`; patience 0 cho cosine dài; `max_steps = lr_decay_steps = 60000`.
- **V10.A7 (đã quyết bằng đo)**: NHÁNH NHIỄU — f0p/f0/f0_delta **+0,000%** giữa 3 manifest; mọi thành phần probe
  trong ±2σ của 500 bước cuối base ⇒ probe fixtail **ĐẠT**; tiêu chí probe ±5% → **±2σ**.
- **V10.A8**: history.csv ghi từng thành phần loss + tỉ phần pause (train & validate).
- **V10.A11**: A2 thật chạy khi đủ 6 điều kiện (không cần chờ Agent 0).

## 4. Việc đang chờ (theo mã vòng)
| Mã | Việc | Trạng thái |
|---|---|---|
| V11.B1 | commit đặc tả V11 | 🔄 |
| V11.B2 | σ chuẩn (trung bình trượt 20 bước) + báo lại 7 thành phần probe + sửa §44 | ⏳ |
| V11.B3 | chia val_calib / val_lock theo clip cha + ghim hash + test | ⏳ |
| V11.B4 | `gate_threshold` runtime + `cer_prod` (theo dõi) + test/push | ⏳ |
| V11.B5 | CreaPy + UTMOSv2 kiểm chứng (CPU) | ⏳ |
| V11.B6 | bench A2 45/50/55/60k + cer_prod base | ⏳ chờ A2 tới 60k |
| V11.B7–B10 | cây quyết định A4 · A1 (nếu nhánh) · chẩn đoán duration/VUV/SẮC-tắc · theo dõi | ⏳ |
| V11.B11 | hồ sơ `V11_agent2.md` + TRANG_THAI | ⏳ |
| V10.A10 | SẮC-tắc làm lại — chuyển thành V11.B9 | ⏳ |

## 4b. A2 đang chạy (số kiểm đầu)
- `runs/acoustic_v2_A2_fixtail` · log `artifacts/dataset_v2/train_v2_A2_fixtail.log` (watchdog resume).
- Dòng đầu history: `# X3 resume: resumed_from=runs/acoustic_v2_base/last.pt@40000` ✅ · cột thành phần loss ✅ · không NaN.
- Bước 41000: val 8,8516 (mel hữu thanh 0,5972 · nét 0,355) · **val_pause_gate 0,934 vs train_pause_gate 0,124 (7,5×)** · val_pause_frac 12,0% vs train 2,6%.
- Nhịp: ~0,43 s/bước · mps 13–15G · validate_every 1000 · save_every 5000 (checkpoint 45/50/55/60k cho B10).

## 5. Bẫy đã gặp (đừng lặp)
probe no-op (thiếu `--stop-at-step`) · lệnh thật train từ số 0 (thiếu INIT_RESUME) · sha256 ghim trước khi sửa tệp ·
JSON thô lọt lại vào `eval/baselines` · đổi bên đối chứng mà không báo trước · thước yếu che kết luận (§34.3→§36.4) ·
**pin sha256 theo byte ĐĨA** (máy khác checkout autocrlf → lệch) → từ V10.A3 ghim theo byte trong git + `.gitattributes -text` ·
**tài liệu ghi "đã sửa" khi code chưa sửa** (V09.A3) → chỉ ghi khi kèm tên test xanh.
