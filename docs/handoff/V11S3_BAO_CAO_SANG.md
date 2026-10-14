# BÁO CÁO SÁNG 15/10/2026 — phiên chạy đêm V11S3

> Bạn ngủ, máy chạy. Tệp này là thứ đọc đầu tiên. Số nào chưa có sẽ ghi rõ **đang chạy**.

## 1. TL;DR (3 dòng)
1. **Rè: chưa sửa xong** — nhưng đã **tìm ra đúng cơ chế bằng đo, và phát hiện thước cũ bị lật ngược**
   (chi tiết §4 — đây là phần giá trị nhất của đêm).
2. **Lỗi đọc chữ: đã sửa 5/5** + tìm ra **129 cặp lỗi chính tả trong dữ liệu train** (vd "thư **dục**"
   → "thư mục") — đây là gốc của "lỗi phát âm giống nhau ở cả 2 model".
3. **Việc cần bạn**: (a) nghe demo A/B (link dưới) khi có; (b) **thu 60 câu** trong
   `resources/cau_thu_am_v1.md` (nhắm đúng ô dữ liệu trống); (c) quyết định về giọng AI (mục 6).

## 2. Dữ liệu: thiếu gì và lỗi gì (đo trên 222.737 âm tiết + 7.342 clip)

| Nhóm | Tỉ lệ trong train | Ghi chú |
|---|---|---|
| **NGÃ** | **4,5%** | đúng thanh bị rè nhất |
| NẶNG | 9,7% | |
| coda tắc (p/t/c/ch) | 1,1–5,0% | SẮC-tắc là ô tệ nhất (§42.1) |
| ô `M:NGÃ` | **2 lần** / 222.737 | gần như không tồn tại |

- **Lỗi chính tả transcript**: `tools/audit_bigrams.py` (bigram hiếm ↔ biến thể phổ biến) → 145 nghi vấn
  → duyệt tay **129 cặp thật** ("hột cái→một cái", "trài khoản→tài khoản", "hập trình→lập trình"…).
  `tools/apply_transcript_fixes.py` đổi **114/9763 row** của manifest fixtail → đã sinh
  `data/codelung/train_clean_v2_dur_seg_slot4_fixtail_typo1.jsonl` (dùng cho lần train sau).
- **Danh sách 60 câu thu âm** (`resources/cau_thu_am_v1.md`): NGÃ+coda vang · câu hỏi (model đang
  ngã giọng cuối câu hỏi ✗ §42.3) · số/ký hiệu/viết tắt · tiếng Anh. Ưu tiên nhóm 1 + nhóm 4.

## 3. Rè: trạng thái từng đòn (số thật, mốc THẬT = creak tuyệt đối ngã+nặng **0,1893**)

| Đòn | Kết quả | Kết luận |
|---|---|---|
| **E4-C** fine-tune Vocos (MR-STFT ≤1 kHz ×3) | mel_l1 −11,3% nhưng F0 RMSE chỉ −0,36% (cần −3%), octave +9,6% | **TRƯỢT** → không dùng (giữ Vocos pretrained) |
| **E4a** làm sạch nhãn F0 vùng creak (A2@42k→50k) | bench **TỪ CHỐI** (2 cổng mel val CŨ trượt = hiệu ứng extension; 9 cổng đạt; tone_C 0,8152 > base 0,8045) · rè NẶNG jitter −64% ✓ · NGÃ ≈ không đổi · creak tuyệt đối ngã+nặng 0,3295 ≈ base 0,3235 | **không sửa rè tầng waveform**; gain khác **không phân biệt được** với "chạy tiếp 8k bước" (phản biện Muse) |
| **E4b** oversample nhóm hiếm (NGÃ×2 · NẶNG×1,5 · coda tắc×1,5) | *đang chạy → điền số khi xong* | |
| **E4c** làm nét mel vùng creak (`mel_sharp` ×(1+3c_t)) | *đang chạy → điền số khi xong* | |

## 4. Phát hiện cơ chế + thước đo (đêm nay)

**Đo liều–đáp ứng trên mel (không train)** — `tools/creak_dose_response.py`, 25 clip creak:

| Mel đưa vào Vocos | ratio | **creak tuyệt đối ngã+nặng** |
|---|---|---|
| THẬT (copy-synth) | 2,390 | **0,1893** (mốc) |
| MODEL (base 40k) | 1,924 | **0,2775 (+47%)** |
| MODEL **làm nét** α=1,5 → 3,0 | 2,63 → 2,70 | **0,2097 → 0,1804** ✓ |

- ⇒ **"Mel mờ ⇒ rè" (như §45.5 phát biểu) là SAI** ✗ — làm mờ mel thật *không* làm creak ngã+nặng tệ
  riêng; nhưng **"làm nét ⇒ giảm creak" ĐÚNG** ✓✓ (0,2775 → 0,1804, về sát mốc thật) — đúng cơ chế
  **sh25** mà tai bạn đã xác nhận.
- ⇒ **Thước `ratio` bị lật ngược** ✗✗ (mel model có ratio 1,92 < mel thật 2,39 nhưng creak tuyệt đối
  cao hơn 47%). **Từ nay chấm rè bằng creak TUYỆT ĐỐI ngã+nặng + Δhnr/Δjitter** (ratio chỉ tham khảo).

## 5. Việc đã sửa/hạ tầng trong đêm (commit)
- 5/5 lỗi frontend (`TP.HCM`, `file wav`, `COVID-19`, `B→bê`, `2+2=4`, email) · `b983a76`
- E4 code (làm sạch F0 creak + MR-STFT trọng số) · `d47fe36` · E4c code · `0dcf217`
- **Bench**: 2 cổng CÙNG NHÃN (mel_z_fix/mel_db_fix) · `rough` 12→60 clip (12 clip lệch +6%, **lật phán
  quyết**) · bench ghi `device` · mốc ghim lại sha256 · `ef3cb6c` `db99df4`
- **Thước mới**: `creak_rate.py` (CreaPy, tái lập đúng 2,390/1,717) · `free_run_clips.py` ·
  `measure_creak.sh` · `vocoder_gate.sh` · `creak_dose_response.py` · `copy_synth_clips.py`
- **Soát dữ liệu**: `audit_bigrams.py` + `apply_transcript_fixes.py` + `resources/transcript_fixes.json`
- **Bench base fresh tái lập mốc CHÍNH XÁC từng số** ✓ (xác nhận mốc/hash/quy trình ổn định)

## 6. Giọng AI cho dữ liệu đặc thù (theo yêu cầu của bạn)
Chi tiết: `docs/handoff/V11S3_giong_tong_hop.md`. Tóm tắt:
- **VieNeu-TTS-v2** (10.000+ h, clone 3–5 s, **chạy MPS/CPU trên Mac**) = hợp nhất; F5-TTS-ViVoice
  chất lượng cao nhưng **cần GPU NVIDIA + giấy phép NC**; XTTS-v2 nhẹ nhưng cộng đồng duy trì.
- **Thứ tự khuyến nghị**: ① bạn thu 60 câu (chắc ăn nhất) → ② dùng chính 60 câu đó làm mẫu clone,
  sinh ~200 câu biên → **lọc + đo** trước, chỉ vào train kèm ablation (bài học GTA §41: audio AI vào
  train từng làm hỏng độ nét).

## 7. Việc tiếp theo (đã xếp)
1. Đọc số E4b/E4c (sáng nay) → nếu E4c giảm creak tuyệt đối mà không mất thanh/CER ⇒ thử tiếp
   `mel_sharp_creak_boost` lớn hơn hoặc **tăng `mel_sharp` toàn cục** (liều–đáp ứng cho thấy nét ⇒ giảm creak).
2. Train lại với **manifest typo1** (đã sửa 114 row) — kiểm CER có giảm không.
3. Thu 60 câu → thêm vào train với trọng số cao cho ô hiếm.
4. Bộ thước F0 phân bố (queue cũ) + `rough` hiệu chuẩn lại.

---

## 8. KẾT QUẢ CUỐI ĐÊM (E4b/E4c) + CHẨN ĐOÁN KIẾN TRÚC (trả lời câu hỏi "do đâu?")

### 8.1 Số cuối (creak TUYỆT ĐỐI ngã+nặng · mốc THẬT 0,1893 · nhiễu ±9% từ bootstrap)

| Bản | creak tuyệt đối | ratio | Bench |
|---|---|---|---|
| THẬT (Vocos-render) | 0,1893 | 2,390 | — |
| base 40k | 0,3235 | 2,725 | ĐẠT CỨNG ✓ |
| **A2 42k (đang dùng)** | **0,3302** | 2,587 | **ĐẠT CỨNG ✓** (re-bench cổng mới) |
| A2 55k | 0,3320 | 2,804 | — |
| E4a 50k (làm sạch nhãn F0) | 0,3295 (≈) | 2,785 | TỪ CHỐI ✗ |
| **E4b 50k (oversample ô hiếm)** | **0,2818** ✓✓ (−13%) | 3,217 | TỪ CHỐI ✗ (3 cổng mel; mel_db_fix +1,25%) |
| E4c 50k (làm nét mel lúc train) | 0,3218 (≈) | 2,529 | TỪ CHỐI ✗ |

**So sánh SẠCH**: E4a và E4b cùng base (A2@42k) + cùng lịch cosine → 50k ⇒ **biến số duy nhất là oversample**
⇒ **rè −13% quy được về oversample** ✓; E4c (làm nét *lúc train*) **không** ăn thua ⇒ sửa **post-hoc**
(sh25) hiệu quả hơn sửa lúc train ✗✓. Kiểm liều–đáp ứng: làm nét mel model α=3,0 ⇒ creak 0,2775→**0,1804**
(về sát mốc thật 0,1893) ⇒ **demo α=1,5 (đang dùng) vs α=3,0** để tai user quyết.

### 8.2 Chẩn đoán: nút cổng nằm ở đâu? (xếp theo bằng chứng)

| Nghi vấn | Bằng chứng đo được | Kết luận |
|---|---|---|
| **Training chưa đủ?** | 1 epoch = 1.209 bước (batch 8) ⇒ 40k bước ≈ **33 epoch**; val mel hữu thanh **phẳng từ ~18k** (tốt nhất 0,5777@18k vs 0,5910@40k); train tiếp 40k→60k còn **xấu** | ✗ **Không** — đã bão hoà, còn nhích overfit |
| **Tham số chưa đủ?** | small→base: mel −1,7% (1,1603→1,1411) + tách thanh tốt hơn ✓; nhưng base **phẳng**; ước lượng "large cần ≥40h" | ✗ ở quy mô hiện tại — thêm tham số một mình vô ích |
| **Kiến trúc thiếu?** | (a) mel model **mờ hơn thật 63%** (rough 0,37) và liều–đáp ứng **chứng minh độ nét điều khiển creak**; (b) acoustic chiếm **~11×** sai số so vocoder (mel 1,36 vs roundtrip 0,12) | ✓✓ **Đòn lớn nhất còn lại** (mục tiêu L1 + decoder, không phải vocoder) |
| **Data trần thấp?** | duration label noise ~18 ms (lỗi model 53 ms = 3× ⇒ chưa chạm trần); typo 129 cặp (~1,2% row); trần **representation** (hop 256 không chứa jitter chu kỳ) | ✓ một phần (nhỏ) + trần kỹ thuật đã biết |
| **Data chưa đủ (lượng)?** | 17,1h · 1 giọng; model bão hoà ở 17h ⇒ muốn to hơn phải thêm giờ | ✓ cho bước scale kế tiếp (≥40h) |
| **Thiếu data đặc thù/chất lượng cao?** | ô hiếm: NGÃ 4,5% · NẶNG 9,7% · coda tắc 1–5% · **M:NGÃ = 2/222.737**; **E4b (oversample đúng ô đó) → rè −13%** trong khi E4a (cùng bước, không oversample) → 0% | ✓✓ **Chốt số 1** |

### 8.3 Việc nên làm (theo hiệu quả/giá)
1. **Thu 60 câu** (`resources/cau_thu_am_v1.md`) → thêm vào train nhắm ô hiếm (bằng chứng E4b −13%).
2. **Demo làm nét α=3,0 vs 1,5** (không cần train) — nếu tai user ưng ⇒ thêm mức α vào runtime.
3. Nâng `mel_sharp` **toàn cục** (hiện 1,0; E4c chỉ thử *cục bộ* và không ăn thua) — 1 run 1h.
4. Dài hạn: thêm ≥40h để nuôi model lớn; adversarial/diffusion cho mel (đã park).
5. **Không** nên: train dài thêm (đã 33 epoch) · tăng tham số một mình.
