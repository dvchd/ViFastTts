# AI_REVIEW_ANSWERS_R4 — Kết quả R4-1 + xác nhận 5 quyết định + chốt lại A1/A5

> **Agent cầu nối · snapshot HEAD `95076eb`** (không đổi trong lúc phân tích); mọi tiến trình đã dừng;
> cache train nguyên 11.424; chưa có v2. Toàn bộ số dưới đây là phân tích read-only trên
> `manifest.jsonl` + `clips.jsonl` + `mfa/fallback_samples.txt`.

---

## 1. R4-1 — Phân bố clip/giờ theo từng video (31 video, sort theo clip tăng dần)

| video | clip | giờ | val cũ? | video | clip | giờ | val cũ? |
| --- | ---: | ---: | :---: | --- | ---: | ---: | :---: |
| lRojc8KDJ6k | 60 | 0,22 | ✓ | EM_ObHWvd94 | 359 | 1,42 | |
| LY8_RaLDJT4 | 86 | 0,36 | | BD83pScFukQ | 424 | 1,65 | |
| nEPr-jIokLw | 101 | 0,46 | | 6GXw85XfiKo | 436 | 1,84 | |
| _k5A863eLQE | 119 | 0,50 | | 4kyqfOypVWI | 437 | 1,81 | ✓ |
| W1m6birJBj0 | 142 | 0,56 | | nh4wz4wlK4Q | 450 | 1,87 | |
| 50swDIypplU | 147 | 0,56 | | UpxLPtoKwo8 | 506 | 2,00 | |
| VeU_85bZlrY | 155 | 0,70 | | MsUY5E3U6x8 | 571 | 2,31 | |
| pLXBrJ-M-Xo | 171 | 0,81 | | UJ_AOlM3cr0 | 594 | 2,36 | |
| _U4H5FSIb-c | 201 | 0,81 | | YFfCmXJk_jI | 642 | 2,52 | |
| o8RjAmBtyZo | 230 | 0,86 | | X_65x81R4KY | 706 | 2,85 | |
| KIrbA-wEURg | 236 | 1,04 | | 6Ll5RtjE41I | 859 | 3,43 | |
| 4ITcvEa_Acc | 244 | 1,02 | | fx303iiDb7M | 873 | 3,41 | |
| z0zSgR_oie4 | 285 | 1,19 | | VGYWEeV2CzM | 963 | 3,36 | |
| vEtfty4Cc6Y | 322 | 1,17 | ✓ | 5KnS0TGxWLg | 972 | 3,57 | ✓ |
| 2b65WyiFC9k | 327 | 1,39 | | 8Pq2FwmIEU8 | 1265 | 4,32 | |
| EUNlJU1BuzE | 332 | 1,36 | | | | | |

**Tổng: 13.215 clip · 51,73h.** Phân bố rất lệch: 8 video lớn (≥700 clip) chiếm ~43% số clip, 15 video
nhỏ (≤332 clip) chỉ ~11h.

### Hệ quả trực tiếp: luật chọn của R4 **không đủ video**
- Dải **80–200 clip chỉ có 7 video** (LY8…, nEPr…, _k5A…, W1m6…, 50sw…, VeU…, pLX…).
- Nới **60–250 chỉ có 12 video** → vẫn **thiếu 4 so với mục tiêu 16**.
- Nới tiếp 60–350 mới đủ 16 video, nhưng khi đó train tụt dưới ngưỡng nghiệm thu (xem §2).
- Điểm tích cực: **train giờ chỉ phụ thuộc tập video được chọn làm val**, cap 100 chỉ chia val/holdout
  (không ảnh hưởng train). Vì vậy cách tối ưu lãng phí là **chọn N video có tổng giờ nhỏ nhất**.

## 2. Các phương án chọn N video val (cap 100, tất định "tổng giờ tăng dần")

| N | val clip | val h | holdout clip | holdout h | train clip | **train h** | train sau bỏ fallback | cache bù |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **16** | 1.546 | 6,49 | 1.612 | 6,52 | 10.057 | **38,72** ✗ | 26,36 | 1.569 |
| 15 | 1.446 | 6,07 | 1.385 | 5,56 | 10.384 | **40,11** ✓ (sát) | 27,19 | 1.569 |
| **14** | **1.346** | **5,66** | 1.153 | 4,61 | 10.716 | **41,46** ✓ | 28,03 | 1.569 |
| 13 | 1.246 | 5,22 | 968 | 3,86 | 11.001 | 42,66 ✓ | 28,88 | 1.569 |
| 12 | 1.146 | 4,82 | 746 | 3,09 | 11.323 | 43,82 ✓ | 29,75 | 1.791 |
| 10 | 946 | 3,96 | 466 | 1,89 | 11.803 | 45,89 ✓ | 31,11 | 1.791 |

*(N=12 chính là pool 60–250; N=10/12 có cache bù 1.791 vì 4 video val cũ đều không được chọn.)*

**Đề xuất: N = 14** — cân bằng giữa đa dạng video (14 nguồn) và dư địa nghiệm thu
(train 41,46h > 40h với biên ~1,5h; val 1.346 clip ≈ 5,66h; holdout 4,61h; train sau bỏ fallback 28,0h).
N=15 chỉ dư 0,11h — quá sát. Nếu ưu tiên train tối đa cho acoustic thì N=12/10 (đổi lại val ít video hơn).

**Danh sách N=14 (chốt nếu bạn đồng ý):**
`lRojc8KDJ6k · LY8_RaLDJT4 · nEPr-jIokLw · _k5A863eLQE · W1m6birJBj0 · 50swDIypplU · VeU_85bZlrY · pLXBrJ-M-Xo · _U4H5FSIb-c · o8RjAmBtyZo · 4ITcvEa_Acc · KIrbA-wEURg · vEtfty4Cc6Y · z0zSgR_oie4`
(N=15 thêm `EUNlJU1BuzE`; N=16 thêm `2b65WyiFC9k`)

**Ghi chú quan trọng:**
- **Hai video val cũ lớn (5KnS0TGxWLg 972 clip · 4kyqfOypVWI 437 clip) quay lại train** — đúng như bạn
  dự kiến; chúng là phần chính của **cache bù** (1.409 clip train + 160 clip val = 1.569 file).
  Thời gian bù ước tính: 1.569 × 1,36 s ÷ 8 worker ≈ **4–5 phút**.
- **val mới có 408/1.346 clip chứa fallback** (30%, tương đương tỉ lệ chung). Đề xuất **200 clip eval
  chọn từ 938 clip val KHÔNG fallback** (vẫn đủ 200, tất định bằng hash) — nếu bạn muốn giữ đủ tỉ lệ
  thực tế thì nói rõ, khi đó metric CER nên báo tách 2 nhóm.

## 3. Năm quyết định — xác nhận + chi tiết thực thi

**1) D6 = (b), mask energy tại SIL-0-frame: XÁC NHẬN.** Quy tắc mask đầy đủ sẽ cài đúng như sau:
| Loại | f0 | tone | phonation | vuv | energy | duration |
| --- | --- | --- | --- | --- | --- | --- |
| SIL (bất kỳ) | mask | mask (ignore_index) | mask | – | – | **giữ** |
| SIL-0-frame | mask | mask | mask | **mask** | **mask** | giữ |
| SIL ≥1 frame | mask | mask | mask | giữ (target 0) | giữ (thấp) | giữ |
| Âm tiết lời nói | giữ | giữ | giữ | giữ | giữ | giữ |
Lý do "sau C4 energy z-score, mặc định 0 = trung bình" đúng — đây chính là căn cứ để mask.

**2) Orthography = (a) + tách hẳn SIL khỏi parser: XÁC NHẬN.** Đã test hiện trạng: `parse_text("<sil>")`,
`parse_text("sil")`, `parse_text("SIL")`, `parse_text("x <sil> y")` → đều cho token **FALLBACK_REQUIRED**,
không có SIL. Đề xuất thực thi: thêm cờ `is_sil: bool = False` vào `ParsedSyllable` (tương thích JSONL cũ);
SIL tạo trong `pipeline.py` sau parse; `validate()/compose_canonical()/test song ánh` lọc `is_sil`;
**bỏ dòng "nhánh SIL trong syllable.validate()"** khỏi bảng điểm chạm D1 như bạn nói; thêm test regression
khẳng định `"<sil>"` vẫn là fallback sau khi có SIL.

**3) E3 trên copy-synthesis: XÁC NHẬN** kèm 3 điều kiện: (i) chỉ dùng **vocoder v2 bản cuối** + split
train, ghi hash checkpoint vocoder vào metadata classifier; (ii) **đóng băng** classifier đánh giá — không
train lại khi GTA fine-tune; (iii) báo kèm accuracy trên **audio thật của val** để biết độ lệch miền.
Lịch: E3 sau khi v2 xong (~20h); smoke trước đó chỉ thiếu tone acc, bổ sung sau — đã ghi vào kế hoạch.

**4) C3 bake stats vào buffer: XÁC NHẬN** kèm 2 bổ sung: (i) model chạy hoàn toàn trong **không gian z**:
predictor → hệ số chuẩn hoá → ×std từng hệ số → IDCT → contour z cấp decoder; graph cần 4 giá trị std;
mean/std log-F0 bake thêm để đổi Hz cho metric; (ii) theo mẫu `mel_hash`: `f0_stats_hash` ghi **checkpoint
+ metadata ONNX**, train **raise** nếu stats trong config lệch buffer, buffer `persistent=True`, có test
`load_state_dict` ghi đè giá trị khởi tạo.

**5) lock_version: XÁC NHẬN.** Ghi `split_lock: {version, hash}` vào `resolved_config.json` và **dict
checkpoint** (cạnh `mel_hash`) cho **cả acoustic lẫn vocoder**; không đưa vào tên run; `eval_metrics.py`
**từ chối so hai kết quả khác lock_version**, chỉ cho phép khi `--allow-cross-lock`.

## 4. Xác nhận các ghi chú R3

| Ghi chú | Xác nhận |
| --- | --- |
| B1 từ nay chỉ đo gradient theo `fake_crop` | ✓ (R1 dùng norm theo tham số — chỉ so trong R3; các bảng mới sẽ ghi rõ đơn vị) |
| `compute_f0_stats.py` phải tính lại sau MFA, trên biên âm tiết thật + split train mới | ✓ (số c0/c1=3,9× chỉ để xác nhận thang, sẽ tái đo) |
| `mfa_to_durations.py`: labs tách theo âm tiết để word tier 1:1 | ✓ xác nhận từ `mfa_build_corpus.py` (mỗi dòng labs = " ".join(normalized) theo từng âm tiết → 1 word = 1 âm tiết); sẽ thêm assert đếm |
| Clip có spn/`<unk>` hoặc số word ≠ số âm tiết → loại + log | ✓ |
| `sil` và `sp` xử lý như nhau | ✓ |
| Report thêm tỉ lệ phone 1 frame | ✓ (đã có trong spec converter) |
| 3.4: `log(max(f0, f_min))` chỉ frame hữu thanh; voiced-ratio <0,3 mask **trước** DCT | ✓ (không tính rồi mới lọc) |
| 3.7: `eval_clips_v1.json`, 200 clip từ val mới sau re-split | ✓ (+ đề xuất lọc fallback ở §2) |

## 5. Thứ tự thực hiện (cập nhật)

1. **Chốt N** (đề xuất **N=14**) → danh sách video val ở §2. *(Cần bạn xác nhận chọn N nào + eval clips lọc fallback hay không.)*
2. Checklist mục 1–4: lock + re-split + gradient log + test (A2/A3 giữ nguyên; A1/A5 thay bằng N đã chốt).
   Khi re-split: ghi `split_lock.json` + `holdout_rest.jsonl` (holdout 1.153 clip/4,61h với N=14).
3. **Cache bù**: train +1.409 file, val +160 file (~4–5 phút) — rồi mới khởi động tay v2.
4. Song song v2: MFA train → `mfa_to_durations` → C1–C4 (stats sau MFA) → D (SIL, đơn giản hoá theo câu 2)
   → metric + 100 câu → E2/E3.
5. Smoke chỉ khi đủ điều kiện như R3.

*Câu hỏi duy nhất còn lại trước khi code mục 3: **N = 14 hay giá trị khác**, và **eval 200 clip có lọc
fallback** (đề xuất có) hay không.*
