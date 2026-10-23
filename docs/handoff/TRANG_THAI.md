# TRANG_THAI — trạng thái hiện hành (nguồn duy nhất, Agent 2 cập nhật mỗi vòng)

> **V18.33 (23/10/2026)** — đánh giá TỰ ĐỘNG 300 câu/12 video: **P2 tốt hơn G1** (BAK −0,40 CI[−0,46,−0,35]; UTMOS −0,10; CER +0,063; 0/12 video G1 hơn BAK) ⇒ **production rollback về P2**, G1 giữ làm thí nghiệm. Nhãn "đánh giá tự động", KHÔNG phải xác nhận tai. Chi tiết: `A0_V18_GROUP1.md` §25, `artifacts/bench/v1833_decision.json`.
> **V18.32 (23/10/2026)** — bộ nghe mù 36 câu (12 video) tại `artifacts/listen_small_filt/v1832_blind/` + chẩn đoán TF/free-run (`artifacts/bench/v1832_tf_free.json`); cơ chế nghịch lý nền **chưa xác định**; G1 vẫn CHƯA XÁC NHẬN TAI.
> **V18.31 (23/10/2026)** — nhánh hiện hành là **rè nền / GAN acoustic**; báo cáo đầy đủ ở
> `docs/handoff/A0_V18_GROUP1.md` §20–§23 (mục cũ V11 bên dưới là lịch sử, không còn hiệu lực).
> **Production acoustic**: G1 `runs/v1830_g1/ckpt_2000.pt` (nhãn **CHƯA XÁC NHẬN TAI**) ·
> rollback: `artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt` · cấu hình: `configs/production_v1831.json`.
> **Trạng thái**: ĐANG LÀM — G1 đạt cổng thước (flat_V 12/12 video, HNR 8/12, S1 +2,1 %); G2 không đổi; G1b bão hoà.
> **Trang nghe**: `/v1829_floor/`, `/v1830_gan/` (chờ người dùng chấm).

**Vòng hiện hành**: V11 (bổ sung 1+2) · **Tệp đặc tả**: `V11_agent0.md` + `V11_agent0_bosung.md` + `V11_agent0_bosung2.md`
**Commit gốc của vòng** (trước khi sửa): `b1ae8ce` · **Commit code cuối**: `34f9b6c`
> Quy ước V11S.A3: HEAD luôn là commit hồ sơ (không tự ghi hash của chính nó); đọc `<gốc>..HEAD`.
> **Phiên chính TỰ QUẢN LÝ** từ 14/10 (Agent 0 hết hạn mức): quyết định tự đưa + ghi vào `V11_agent2.md`
> mục "PHẢN BIỆN NGOÀI" (sub-agent, ít nhất 1 model khác họ = `muse-spark-1.3-contributor`).
**Trạng thái**: **ĐANG LÀM (W1 xong, chuyển W2/E4)** — A2 xong 60k ✓ · B6/B7 xong (TỪ CHỐI hình thức do 2 cổng mel val CŨ khác nhãn; 6 cổng còn lại đạt + tốt hơn base) · W1 xong: A/B tai người **HÒA** ⇒ A2-55k = **tham chiếu nội bộ**, base 40k = bản dùng hằng ngày; rè = ưu tiên #1 · hạ tầng xong (`ckpt_<step>.pt` nguyên tử + watchdog chịu last.pt hỏng) · B9 xong (duration 53 ms không ở trần nhãn ⇒ W2; VUV bão hoà ⇒ hạ xuống theo dõi).
**Hàng đợi phiên sau**: E4-rè (nội suy có trọng số + loss C) → W2-duration → thước F0 phân bố → cổng cùng nhãn → B10 → bộ 30 câu người thật (chi tiết ở cuối `V11_agent2.md`).

> Agent 1 đọc vòng này theo thông tin ở đầu tệp (không cần hash gửi kèm): diff `06b5f10..<commit cuối>`,
> câu hỏi V10.C1–C7 ở mục C của tệp đặc tả. Agent 1 **chỉ đọc, không ghi tệp** vào repo.
> Từ V10.A1: commit ĐẦU TIÊN của mỗi vòng là tệp đặc tả nguyên văn `docs/handoff/V<nn>_agent0.md`.

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
