# DATASET V2 — trạng thái thực thi (12/10/2026)

> Chủ dự án giao "làm dữ liệu sạch nhất/tốt nhất theo ý agent". Tài liệu này ghi **quyết
> định + kết quả thực thi** của bước dựng lại dữ liệu (REBUILD_PLAN_M6 §3). Nguồn sự thật
> kỹ thuật: `docs/REBUILD_PLAN_M6.md`, `docs/R6_DIARIZATION_RESULTS.md`, `docs/VOCODER_VOCOS.md`.

## 1. Quyết định (chốt 12/10, tối)

| Mục | Chốt | Lý do |
| --- | --- | --- |
| Phạm vi dựng lại | **Lọc + chuẩn hoá tại chỗ trên clip hiện có** (không recut lại từ `raw/`) | Recut B3 vừa chạy xong 15:21 hôm nay — cùng kết quả cắt; recut lại không đổi chất lượng mà tốn ~1 ngày |
| Lọc giọng | Bỏ **6 video loạt khách** (937 clip) + **48 clip nhiễm** (R6 diarization, `other_s ≥ 0,5s`) | R6_DIARIZATION_RESULTS: 0,090h tiếng lạ; loạt "Học Từ Thiện" bị bắt 70–86% là người khác |
| Loudness | **Từng clip → −26 dBFS RMS trên frame có tiếng**; peak-safe ≤ −1 dBFS; boost ≤ +14 dB | Gốc R5-1: lệch tới 22 dB giữa video + trong video; per-clip xử lý cả hai nguồn |
| Audio v2 | Ghi `data/codelung/audio_v2/` (PCM_16), **giữ nguyên** `audio/` (đối chiếu) | Không phá dữ liệu gốc; diff/revert dễ |
| Split | **12 video ít giờ nhất** = val/test; val **cap 150 clip/video**; phần dư = **holdout** (không vào train); 13 video còn lại = train | REBUILD_PLAN §3(4): N=12/cap150 — giữ ≥10 video val, val 6,6h thừa cho 640 batch + eval |
| Bản clean | `fallback_ratio == 0` (theo `clips.jsonl`) dùng cho **acoustic + MFA**; bản đầy đủ cho **vocoder** | Vocoder không cần text; acoustic/MFA cần nhãn đọc chuẩn |
| eval_clips_v1 | **200 clip** từ holdout, ưu tiên 100% sạch, vòng tròn qua video, kèm `text_sha1` | Bộ cố định so CER/nghe giữa các mốc; không nằm trong train |
| Speaker-conditioning | Giữ **không** (một checkpoint = một giọng) | Đúng giới hạn dự án |

## 2. Kết quả

**Lọc + chuẩn hoá** (`tools/normalize_audio_v2.py`):

- Giữ **12.230 clip** · 47,48h (đúng pool R6) — loại 937 clip series khách + 48 clip diarization.
- `audio_v2/` 7,7 GB; gain p5/p50/p95 = **−10,9 / −1,8 / +10,2 dB**; 22 clip chạm trần +14 dB;
  333 clip giảm >12 dB. Sau chuẩn hoá **mọi video về −26,00 dBFS** (hết lệch 22 dB).
- File: `manifest_v2.jsonl`, `loudness.jsonl`, `excluded_speakers.jsonl`, `excluded_other.jsonl`.

**Split** (`tools/build_split_v2.py`) — khớp dự kiến REBUILD_PLAN (±0,1h):

| Nhóm | Clip | Giờ | Dùng cho |
| --- | ---: | ---: | --- |
| `train_v2.jsonl` | 9.262 | 35,60 | vocoder (+ acoustic sau lọc fallback) |
| `val_v2.jsonl` | 1.641 | 6,63 | validation |
| `holdout_v2.jsonl` | 1.327 | 5,25 | test cuối (1 lần) |
| `train_clean_v2.jsonl` | 6.658 | **24,39** | **acoustic + MFA train** |
| `val_clean_v2.jsonl` | 1.140 | 4,42 | val acoustic |
| `holdout_clean_v2.jsonl` | 925 | 3,50 | test cuối |
| `eval_clips_v1.jsonl` | 200 (100% clean) | ~0,8 | CER/nghe cố định (9 video holdout) |

12 video val: `lRojc8KDJ6k, 50swDIypplU, _k5A863eLQE, VeU_85bZlrY, o8RjAmBtyZo, 4ITcvEa_Acc,
vEtfty4Cc6Y, z0zSgR_oie4, EUNlJU1BuzE, 2b65WyiFC9k, EM_ObHWvd94, BD83pScFukQ` (khóa: `split_lock.json`).

**MFA corpus v2** (`tools/mfa_build_corpus.py` → `data/codelung/mfa_v2b/` sau khi merge overrides):
**9.619 clip** · 0 fallback · 2.877 từ · 120 phone (gồm train+val+holdout clean; align toàn bộ).

**Mel stats mới (biểu diễn Vocos)**: tính trên `train_clean_v2`, 2.000 clip, **chỉ frame có
tiếng** (`--min-log-energy -6`; không lọc thì 5,15% phần tử bị kẹp ở floor) → clamp 0%;
`data/codelung/mel_stats.json` (bản cũ sao lưu `mel_stats_v1_backup.json`).

## 2b. Điền overrides fallback (12/10, tối — nhờ Muse Spark 1.3 Contributor soạn nháp)

- 400 từ trong `review/filled.txt`: **387 từ có cách đọc** (đã tự kiểm bằng validator của repo —
  100% parse được), **13 từ bỏ trống** vì không đoán được (`xxz, chxyz, onm, cáiing, vure, bupo,
  yamo, murai, flir, fen, ter, chffix, maxide` — clip chứa chúng tiếp tục bị loại khỏi acoustic/MFA).
- Đồng bộ văn phong với bảng gốc (`window → uyn đâu`, `word → quớt`, `network → nét quớc`,
  `process → pờ rô séch`, …); cụm phụ âm tách bằng "ờ" (`product → pờ rô đắc`).
- Merge: `overrides_tech_vi.json` **183 → 570 mục**; reparse text: **1.546 dòng đổi**.
- **Fallback: 1,22% → 0,80% âm tiết**; `train_clean_v2` **24,39h → 27,28h** (thêm 684 clip);
  val_clean 4,42→4,90h; holdout_clean 3,50→3,93h. Corpus MFA: 8.723 → **9.619 clip** (36,1h).

## 3. Đang chạy nền (đêm 12/10)

**Lỗi đã tìm & sửa — layout corpus MFA (quan trọng):** MFA ghép transcription với audio
theo tên file **trong cùng một thư mục** (`find_exts` trong `corpus/helper.py`). Corpus cũ
tách `labs/` + `wavs/` ⇒ MFA đọc text **rỗng** cho cả 9.619 utterance ⇒
`filter_training_utterances` lọc sạch (0 job) ⇒ `IndexError` khi train. Đã sửa
`tools/mfa_build_corpus.py` sang **layout phẳng** (`<id>.lab` cạnh `<id>.wav`), cập nhật test,
và **kiểm chứng bằng probe 120 clip**: 120/120 utterance có text, 0 ignored, model vào tới
"monophone - Iteration 11 of 40" (không còn lỗi). Xoá các work-dir MFA cũ trong `~/Documents/MFA`.

**Bài học vận hành (13/10):** đêm 12/10 MFA đã train xong (monophone 23:37 → triphone 23:53 →
LDA 00:09, "Training complete!") nhưng bị **kill ngay lúc xuất model/align** (nghi máy ngủ —
`Interrupted system call`). Lần chạy lại **KHÔNG được truyền `--clean`** (xóa mất các stage đã
train) và **phải chạy dưới `caffeinate`**:

```bash
nohup caffeinate -is bash -c 'micromamba run -p ./.mamba-mfa mfa train data/codelung/mfa_v2b \
    data/codelung/mfa_v2b/lexicon.txt data/codelung/mfa_v2b_model.zip \
    --speaker_characters 11 --num_jobs 8 -v' >/dev/null 2>&1 &
```

Chuỗi hiện tại (log `artifacts/dataset_v2/chain_v2d.log`):

1. **`mfa train`** trên `mfa_v2b` (9.619 clip, layout phẳng, `--num_jobs 8`) → `data/codelung/mfa_v2b_model.zip`
2. **`mfa align`** mfa_v2b → `data/codelung/mfa_v2b_aligned/*.TextGrid`

(Đêm 12/10 MFA train xong nhưng bị kill lúc xuất model — nghi máy ngủ; lần resume đầu lại lỡ truyền
`--clean` nên phải train lại từ đầu. Từ đó: **chạy dưới `caffeinate -is`** và thêm
`tools/watch_and_finalize.sh` — watcher tự canh, tự resume/retry khi chain chết (train không `--clean`,
align `--no_overwrite`), rồi tự chạy finalize + ghi `artifacts/dataset_v2/FINAL_REPORT.md`.)

Cache feature v2 đã xong: 10.903 file = 9.262 train + 1.641 val.

Xong bước 2 → việc kế: converter TextGrid→durations cấp âm tiết + SIL + assert (theo đặc tả
cũ) → cắt segment 3–10s tại SIL → smoke small → base.

## 3b. Phát hiện & sửa lớn (13/10 sáng): tên phone `_` làm hỏng alignment

**Triệu chứng:** converter `mfa_to_durations` giữ được đúng 94/7.342 clip train — hầu hết bị
loại vì `phone > 0.5s` và `ngừng thở … giữa câu`. Truy vết từng lớp:

1. **TextGrid export cắt nhãn phone:** MFA coi mọi hậu tố sau `_` cuối cùng là "dấu vị trí"
   (`helper.split_phone_position` = `rsplit("_", 1)`) ⇒ `n_a_huyen` → base `n_a`, `c_m` → `c` …
   DB MFA giữ `kaldi_label` đầy đủ nhưng TextGrid xuất ra dùng base phone ⇒ **mất tone + gộp
   coda trong mọi TextGrid**.
2. **Chất lượng alignment hỏng thật** (không chỉ ở nhãn): thống kê DB v2b — **22,3% word
   interval ≤ 55 ms** (sát sàn HMM min-duration), p50 = 140 ms, 61,8% clip có >20% từ sát sàn.
   QC chéo Whisper 10 clip (`tools/qc_whisper_align.py`, model `small`): coverage 0,448 ·
   median|Δstart| 0,06 s · duration ratio 0,693. Cùng 10 clip với model VN pretrained
   (`vietnamese_mfa`): từ sát sàn chỉ **1,7%** (p50 180 ms) ⇒ lỗi ở **model custom** (nghi án
   chính: tên phone `_` làm hỏng logic base-phone của MFA trong train/align).
3. **Không phải lỗi dữ liệu:** corpus wav byte-identical với `audio_v2`; VAD + Whisper xác nhận
   text↔audio khớp (Whisper transcribe đúng nội dung; word start lệch trung vị 0,06–0,07 s).

**Đã sửa (13/10):**

- `tools/mfa_build_corpus.py`: tên phone **không còn `_`** — CamelCase `Och`, `NaHuyen`, `Cw`,
  `NăNgang`… + chống xung đột tên theo từng thành phần + chặn tên >10 ký tự
  (`kaldi_label` = `String(10)` của MFA). Corpus mới **`data/codelung/mfa_v2c`** (9.619 clip ·
  2.877 từ · 120 phone · 0 fallback).
- `tools/mfa_to_durations.py`: **D5-b** — pause ≥150 ms giữa câu là **điểm cắt segment**,
  không loại clip; chỉ loại khi "ngừng thở" ≥ `--max-mid-pause-seconds` (mặc định 1,5 s).
  `phone > 0,5 s` chỉ áp cho phone **hữu thanh** (bỏ `sil`/`sp`/`spn`/`''`) — trước đó luật
  này loại oan gần hết clip.
- Thêm `tools/qc_whisper_align.py` (C3): QC chéo timing word MFA ↔ Whisper, đo coverage /
  median|Δ| / duration ratio / match rate.

**Đang chạy (13/10 ~06:25):** train **subset 800 clip** `mfa_v2c_sub` để nghiệm thu hướng sửa
(log `artifacts/dataset_v2/mfa_train_v2c_sub.log`). Khi xong: align `mfa_v2c_test` (10 clip) +
QC lại; nếu đạt (từ sát sàn ~≤5%, coverage ≥0,6) → train full `mfa_v2c` → align → finalize.

**Thước đo nghiệm thu:** `tools/align_quality_report.py` — `floor_rate` (tỉ lệ từ ≤55 ms sát
sàn min-duration), `coverage` (thời lượng word/span), `word_dur_median_s`, `long_phone_rate`.
Baseline trên 10 clip test (beam 100/400):

| nguồn | floor_rate p50 (mean) | coverage p50 | word dur p50 |
|---|---|---|---|
| v2b (model cũ, tên `_`) | 0,205 (0,234) | 0,458 | 0,16 s |
| `vietnamese_mfa` (pretrained) | **0,021** (0,017) | 0,552 | 0,19 s |
| v2c subset (đang train) | — | — | — |

Ngưỡng đạt cho v2c: **floor_rate p50 ≤ 0,05 · coverage ≥ 0,5**.

**Plan B (dự phòng):** model `vietnamese_mfa` align tốt trên dữ liệu của mình nhưng dictionary
của nó chỉ phủ **47% từ vựng** (thiếu cả từ phổ biến — dict là từ vựng của corpus huấn luyện
riêng, không đầy đủ) và **không có G2P tiếng Việt** để bù OOV ⇒ muốn dùng phải thêm bước map
120 phone → IPA tự viết. Chỉ triển khai nếu Plan A không đạt.

## 4. MFA — vì sao phải train lại (trả lời câu hỏi)

- Bộ **phone tùy chỉnh** (120 nhãn theo vị trí + thanh) — không pretrained model nào khớp;
  muốn tách thời gian đúng 4 thành phần âm tiết (onset/medial/nucleus/coda; thanh nằm trong
  nhãn nucleus) thì phải train acoustic model MFA **trên chính lexicon/phones của mình**.
- MFA train rẻ (một lần, CPU ~vài giờ), không cần GPU; sau đó **align là dùng lại** — chỉ
  phải align lại khi text/lexicon đổi (ví dụ sau đợt reparse overrides).

## 5. Công cụ dữ liệu đã thêm (12/10, tối) & việc còn lại

| Tool | Việc | Test |
| --- | --- | --- |
| `tools/mfa_to_durations.py` | TextGrid → `durations` (frames) + `pauses` (pause ≥150 ms ở ranh giới câu); loại clip lỗi (ngừng thở giữa câu, phone >500 ms, âm tiết >1 s, lệch từ); assert `sum == T` | `tests/test_mfa_to_durations.py` (6) |
| `tools/cut_segments.py` | Clip → segment 3–10 s, cắt tại biên âm tiết, ưu tiên pause; row có `frame_range` | `tests/test_cut_segments.py` (4) |
| `frame_range` trong `TtsDataset` | Cắt mel/f0/vuv/energy từ clip đầy đủ theo frame (tránh hiệu ứng biên STFT — đúng spec §9.2) | idem |
| `dct4_frames`/`idct4_frames` (`data/features.py`) | DCT-II trực giao `(t+0.5)/n`; n<4 → k=n (đúng R3 §C2) | `tests/test_dct_stats.py` (4) |
| `tools/compute_f0_stats.py` | Stats log-F0 scalar + **DCT4 z-score từng hệ số** (2 pass) + energy. Preview 500 clip: log-F0 4,866/0,262 · DCT std [4,50, 2,39, 1,65, 1,22] (chốt lại khi có durations thật) | idem |
| `data/eval/cau_eval_v1.txt` | **101 câu eval thuần Việt**: 0 fallback · đủ 6 thanh (286/177/278/101/62/149) · đủ coda tắc (P30·T46·C59·CH13) | `tests/test_eval_set.py` (3) |

Việc còn lại:

1. **Chạy converter trên TextGrid thật** (khi `mfa align` xong) → `*_dur.jsonl` + report drops;
   đã có `--mfa-db` (lọc 10% score thấp nhất theo `alignment_log_likelihood/num_frames`, C4).
2. **Cắt segment** từ `*_dur.jsonl` → `*_seg.jsonl` (3–10 s) + kiểm tra `TtsDataset` load được segment.
3. **Chạy `compute_f0_stats.py`** trên `train_clean_v2_dur.jsonl` → `f0_stats.json` (bản chốt).
4. **SIL** (workstream code kế tiếp theo `AI_REVIEW_ANSWERS_R3/R4`: `is_sil` trong `ParsedSyllable`,
   chèn ở `pipeline.py`, vocab 71, mask ở `losses.py`, `where(is_sil, 0, …)` khi dự đoán duration,
   guard g2p/segments/lexicon/coverage) — 35 chỗ gọi `parse_text` trong test sẽ cần cập nhật.
5. **DCT4 target trong dataset/model** (#8) — phần dữ liệu (công thức + stats) đã sẵn ở trên.
6. Acoustic: smoke small → base (theo REVISION_PLAN_V2); vocoder: bước 0 → GTA sau acoustic.

## 6. Lệnh tái lập (đã dùng)

```bash
python tools/normalize_audio_v2.py --exclude-clips data/codelung/review/diar/drop_other_ge_0p5.txt --workers 8
python tools/build_split_v2.py
python tools/mfa_build_corpus.py --manifest data/codelung/train_clean_v2.jsonl \
    --manifest data/codelung/val_clean_v2.jsonl --manifest data/codelung/holdout_clean_v2.jsonl \
    --out data/codelung/mfa_v2
python tools/compute_mel_stats.py --manifest data/codelung/train_clean_v2.jsonl \
    --config configs/small_codelung.yaml --out data/codelung/mel_stats.json --sample 2000 --min-log-energy -6
vifasttts cache-features --manifest data/codelung/train_v2.jsonl --config configs/small_codelung.yaml \
    --cache-dir cache/codelung_v2 --workers 8
micromamba run -p .mamba-mfa mfa train data/codelung/mfa_v2 data/codelung/mfa_v2/lexicon.txt \
    data/codelung/mfa_v2_model.zip --speaker_characters 11 --num_jobs 8 --clean -v
micromamba run -p .mamba-mfa mfa align data/codelung/mfa_v2 data/codelung/mfa_v2/lexicon.txt \
    data/codelung/mfa_v2_model.zip data/codelung/mfa_v2_aligned --speaker_characters 11 --num_jobs 8 --clean -v
```

**v2c (13/10 — tên phone bỏ `_`, thay thế hoàn toàn v2b/v2):** corpus dựng lại bằng cùng lệnh
trên với `--out data/codelung/mfa_v2c`; chuỗi train+align chạy tự động:

```bash
nohup caffeinate -is bash tools/chain_mfa_v2c.sh >/dev/null 2>&1 &          # train → align → DONE
nohup caffeinate -is bash tools/watch_and_finalize.sh >/dev/null 2>&1 &     # canh + finalize + report
# hoặc finalize tay:
TEXTGRID_DIR=data/codelung/mfa_v2c_aligned MFA_DB=$HOME/Documents/MFA/mfa_v2c/mfa_v2c.db \
    bash tools/finalize_dataset_v2.sh
# QC chéo Whisper (C3):
.venv/bin/python tools/qc_whisper_align.py --textgrid-dir data/codelung/mfa_v2c_aligned \
    --corpus data/codelung/mfa_v2c --sample 10 --out artifacts/dataset_v2/qc_whisper_v2c.json
