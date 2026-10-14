#!/usr/bin/env bash
# Đo "rè" một checkpoint bằng ĐÚNG bộ thước chuẩn hoá (§38–§45) trong MỘT lệnh:
#   1) `eval_creak.py`      — HNR/glottalization/jitter theo thanh, free-run (CPU)
#   2) `free_run_clips.py`  — wav free-run 25 clip giàu ngã/nặng + cửa sổ slot_durations
#   3) `creak_rate.py`      — CreaPy: tỉ lệ creak (ngã+nặng)/(ngang+huyền)
#
# Mốc đã kiểm chứng (cùng một thước, 25 clip): THẬT 1,717 · copy-synth Vocos pretrained 2,390 ·
# free-run base 40k 2,725 · free-run A2-55k 2,80. Đích E4: free-run **về phía 1,717**.
#
# Dùng: tools/measure_creak.sh <tag> <config> <checkpoint> [device] [clips]
#   tag: tên ngắn (vd E4a_50k) — ghi artifacts/creak25/free_<tag>/ + JSON cùng tên.
#   device: mps|cpu (mặc định cpu — chạy được song song, không tranh MPS với train).
set -euo pipefail
TAG=${1:?tag}; CONFIG=${2:?config}; CKPT=${3:?checkpoint}; DEVICE=${4:-cpu}; CLIPS=${5:-40}
cd "$(dirname "$0")/.."
CLIPS_JSON=artifacts/creak25/clips25.json
FREE_DIR="artifacts/creak25/free_${TAG}"
WIN="artifacts/creak25/free_${TAG}_windows.json"

echo "== 1/3 eval_creak ($TAG, $CLIPS clip, $DEVICE) =="
.venv/bin/python tools/eval_creak.py --config "$CONFIG" --checkpoint "$CKPT" \
  --clips "$CLIPS" --device "$DEVICE" --out "artifacts/eval_acoustic/creak_${TAG}.json"

echo "== 2/3 free_run_clips ($TAG) =="
.venv/bin/python tools/free_run_clips.py --clips "$CLIPS_JSON" --config "$CONFIG" \
  --checkpoint "$CKPT" --out "$FREE_DIR" --windows-out "$WIN" --device "$DEVICE"

echo "== 3/3 creak_rate ($TAG) =="
CREAPY_PY="${CREAPY_PY:-$HOME/creapy-venv/bin/python}"
if [ -x "$CREAPY_PY" ]; then
  "$CREAPY_PY" tools/creak_rate.py --clips "$CLIPS_JSON" --synth-dir "$FREE_DIR" \
    --windows "$WIN" --out "artifacts/creak25/free_${TAG}.json"
else
  echo "⚠ thiếu $CREAPY_PY (CreaPy GPL-3.0, cài ngoài repo) — bỏ bước 3"
fi
echo "→ xong $TAG (JSON ở artifacts/creak25/ + artifacts/eval_acoustic/)"
