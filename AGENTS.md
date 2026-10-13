# ViFastTts — hướng dẫn cho agent

> TTS tiếng Việt **non-autoregressive, tất định**, chạy trên **Mac mini M6** (train MPS,
> inference ONNX CPU). Đọc file này trước khi sửa code; thiết kế + số đo nằm trong `docs/`.

## 1. Tài liệu nguồn (đọc theo thứ tự)

1. `docs/ACOUSTIC_V2.md` — **thiết kế chốt + số đo thực thi** acoustic v2 (§0–§14 thiết kế;
   §15 preflight; §17–§22 kết quả R2/T2.1/R5b/Addendum 6/R6/R4/T2.2). Mâu thuẫn → addendum mới hơn thắng.
2. `docs/B0_BASELINE.md` — baseline B0 (mốc so mọi thứ): `runs/small_codelung_v2/best.pt` (8k bước)
   + `last.pt` (20k); speech mel ~9,6–9,8 dB; free CER 0,54; 82% loss val là f0.
3. `PROJECT_STATUS.md` — trạng thái tổng; `docs/DATASET_V2.md` + `docs/TTS_ARCHITECTURE_RESEARCH.md` — nền.

**Nguyên tắc vàng: không nâng cấp gì trước khi có thước đo; mọi đề xuất phải kèm số đo + cổng.**

## 2. Môi trường & lệnh

- Python: `.venv/bin/python` (3.12.15 · torch 2.14.1 · pyworld · onnxruntime · typer · ruff).
- **Chạy lệnh từ repo root** (working dir `/Users/.../ViFastTts`).
- Lint (phải sạch): `.venv/bin/ruff check .` → tự sửa `--fix`.
- Test (phải pass 100%, ~2.350 test ≈ 20 s): `.venv/bin/python -m pytest -q`.
- CLI: `.venv/bin/vifasttts <command>` (`cache-features`, `train`, `export`, `synthesize-onnx`…).
- Push cần: `git -c credential.helper='!gh auth git-credential' push` (dòng
  `fatal: failed to store: -61` là vô hại, push vẫn thành công).

## 3. Bố cục

- `src/vifasttts/frontend/` — text → âm tiết; `segments.py` (role/weights, ưu tiên B2, mask F0 Addendum 6).
- `src/vifasttts/data/` — manifest, features (mel/f0/vuv/energy), cache, dataset, phonation, `v2_targets.py`.
- `src/vifasttts/models/` — `vifasttts.py` (v1), `slot_ops.py` (đóng băng R4), conformer, vocos, istftnet.
- `src/vifasttts/runtime/`, `export.py`, `train.py`, `cli.py`.
- `tools/` — script đo nghiên cứu (mỗi R/T một tool, ghi JSON vào `artifacts/`); `tests/` — unit test.
- `data/codelung/`, `cache/`, `runs/`, `artifacts/` — **local, KHÔNG commit** (đã gitignore).

## 4. Dữ liệu & cache (local)

- Manifest train/val: `data/codelung/{train,val}_clean_v2_dur_seg.jsonl` (row **segment**),
  bản `..._slot4.jsonl` thêm `seg_durations [N,4]` theo thứ tự **O, M, N, C**; holdout:
  `holdout_clean_v2_dur[_slot4].jsonl`. Row segment có `frame_range` (toạ độ clip cha) + `durations` + `gaps`.
- `seg_durations` thuộc **manifest** (không phải cache); checkpoint phải ghi `seg_method` /
  `seg_method_version` / `seg_weights_hash`.
- Cache: `cache/codelung_v2/` — khoá `feature_cache_key` = sha1(audio+mtime + block `mel`,
  `f0`, `phonation`, `sample_rate`). **Đổi các block đó = đổi toàn bộ khoá** → dùng config riêng,
  KHÔNG sửa `configs/small_codelung.yaml` (B0) khi thêm phonation — bản v2 là
  `configs/small_codelung_v2.yaml` (phonation enabled).
- Kênh cache v3: `mel, energy, f0, vuv, hnr, glottalization, jitter` (R6 — §20).

## 5. Quy ước kỹ thuật bắt buộc

- **Frame**: hop 256 @ 24 kHz; `T = 1 + len(wav)//hop`; mel 100 log-magnitude (hợp đồng Vocos).
- **F4**: mask hữu thanh luôn dùng `vuv > 0,5` — KHÔNG dùng `f0 > 0` (f0 đã nội suy qua vô thanh).
- **F0 mask (Addendum 6)**: slot O chỉ tính F0 loss khi `onset ∈ F0_VOICED_ONSETS` ∧ ratio ≥ 0,5;
  M/C ratio ≥ 0,5; N luôn (trọng số = tỷ lệ hữu thanh). Hằng số ở `frontend/segments.py`.
- **DCT**: `dct4_frames`/`idct4_frames` (trực giao, pos = (t+0,5)/n). Model dùng hệ số hình dạng
  `s_k = c_k/√n` (`slot_ops.build_f0_contour`; `s0 == mean`).
- **Chia slot B2**: `models/slot_ops.distribute_slots` ↔ `frontend/segments.distribute_frames`
  (cùng thuật toán, ưu tiên **N, O, C, M**). Bất biến: inactive = 0 · active ≥ 1 nếu đủ frame · Σ = d_syl.
- **ONNX**: opset 18, `dynamo=False`; **giữ nguyên tên + thứ tự output cũ**, chỉ **thêm output mới vào cuối**;
  `speed` là input float32 shape `[1]`; op trong graph phải export được (không vòng lặp động / argsort).
- **`arch`** trong checkpoint: `"v1_syl"` (mặc định khi thiếu) | `"v2_slot5"`; factory đọc `arch`;
  không load chéo bằng `strict=False` ngầm — chỉ warm-start có chủ đích (§11 ACOUSTIC_V2).
- `MAX_POSITIONS = 4096` frame (≈43,7 s @ hop 256) — chia câu trước model.
- **Batch > 1**: dùng `SameNBatchSampler` (gộp cùng N; `collate_slot_batch` không pad N). Batch tối ưu
  phải **ĐO** — trên M6: batch 1 = 0,229 s/mẫu · **8 = 0,0375** · 16 = 0,047 (không suy diễn; xem §29).
- **Trước mọi thay đổi train**: probe 60–100 bước (loss ổn định? s/bước? RAM MPS?) + nhớ LR phụ thuộc
  `max_steps` (đặt max_steps nhỏ → cosine hết sớm → LR ở sàn).
- Comment/docstring **tiếng Việt**, ngắn, nêu lý do.

## 6. Quy trình làm việc & commit

- Mọi thay đổi: chạy `ruff check .` + `pytest -q` trước khi commit; commit dạng
  `v2(<task>): <kết quả ngắn>`; kết thúc bằng push (lệnh ở §2).
- Một R/T xong phải có: **số đo + so mốc** (B0 hoặc số cũ) + cổng đạt/không + file thay đổi
  → ghi vào `docs/ACOUSTIC_V2.md` (số liệu trong doc; JSON thô ở `artifacts/` — local).
- **Không commit** `data/`, `cache/`, `runs/`, `artifacts/`.
- **Demo audio**: *không* commit vào lịch sử repo; khi user cần nghe từ xa → đẩy bằng
  **GitHub Release asset** (vd `gh release create demo-audio-<ngày> artifacts/demo/*.wav`),
  xoá sạch về sau bằng `gh release delete <tag> --yes` (không cần force-push).
  **Upload xong PHẢI kiểm chứng** (đã từng "nói upload" mà quên chạy): `gh release view <tag>
  --json assets` + tải thử `curl -sL -o /dev/null -w '%{http_code} %{size_download}' <url>`.
  Link tải trực tiếp cho user: `https://github.com/dvchd/ViFastTts/releases/download/<tag>/<file>`.
  Nhãn file ghi rõ model + biến thể (`v2b8_sh15_*` = model b8, làm nét α=1,5) + cập nhật phần
  **notes** của release thành mục lục mỗi khi thêm bản mới.
- Báo cáo trung thực: nêu cả trường hợp trượt cổng + cảnh báo rõ.
- Quy trình 3 vai: Agent 0 (kiến trúc, ra addendum khi số đo khác kỳ vọng) · Agent 1 (soát code) ·
  Agent 2 (chạy trên M6). Sub-agent khi được giao việc **không tự commit** — trả file + cách chạy cho phiên chính.

## 7. Hiện trạng (13/10/2026, sau commit `09ff10d`)

- **Xong**: R2 (weak label lai 98,3%) · **T2.1** (`seg_durations`, 0 lỗi bất biến trên 299.370 âm tiết) ·
  **R5a/R5b** (stats cấp segment, `data/codelung/f0_stats_v2.json`) · **Addendum 6** (mask F0 theo lớp onset;
  `vuv` không đủ tin cho O) · **R6** (phonation: hnr 0,869 · jitter 0,864 · glottalization 0,763 —
  tilt/aper rớt) · **R4** (`models/slot_ops.py` đóng băng + 1e5 bất biến + parity ONNX) ·
  **T2.2 nền tảng** (`data/v2_targets.py`) · **cache v3** (đã build + verify 100%, `tools/verify_cache_v3.py`) ·
  **task 5 đợt 1** (`models/v2_slot5.py` + `models/factory.py` + 8 test gồm ONNX parity) ·
  **đợt 2** (dataset v2 + loss v2 + export/runtime rẽ nhánh `arch`; sửa bug lệch tên output v1 postnet) ·
  **đợt 3** (`train_v2.py` + CLI `train-v2` + 2 bug kernel MPS đã vá/tránh — §25).
- **Đang chạy**: train v2 (`runs/v2_slot5`, warm-start B0, 20k bước ≈ 1,5 h) → cổng §11 (mel hữu thanh
  giai đoạn A không tệ hơn B0 quá 20%).
- **Song song**: eval v2 (`tf_seg`/`tf_syl`/`free` + mask Addendum 6 + F0 theo thanh) — subagent.
- **Bẫy MPS đã gặp (đừng quên)**: SDPA+dropout cần grad; `ConvPredictor` backward N=1 abort → lọc
  `min_syllables=2` khi train; mọi test N=1-backward **không** được chạy trên MPS (giết tiến trình pytest).
