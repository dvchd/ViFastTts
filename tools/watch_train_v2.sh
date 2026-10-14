#!/usr/bin/env bash
# Watchdog cho `train-v2`: tự chạy lại kèm `--resume` cho tới khi đạt `max_steps`
# (chống crash/NaN khi chạy dài qua đêm). Đặt đủ env watermark MPS (bài học §29.3).
#
# Dùng: tools/watch_train_v2.sh <out_dir> <log_file> <max_steps> <config> <train_manifest> <val_manifest>
#   INIT_RESUME=<ckpt>          resume từ checkpoint NGOÀI — CHỈ ở lần khởi động đầu (chưa có OUT/last.pt)
#   INIT_RESET_VAL_STATE=1      thêm --reset-val-state — CHỈ ở lần khởi động đầu
#   DRY_RUN=1                   chỉ in lệnh train-v2 sẽ chạy (đúng mảng CMD) rồi thoát
#   TRAIN_CMD=<lệnh>            ghi đè lệnh train (test dùng script giả)
#   WATCH_MAX_ATTEMPTS=<n>      số lần thử tối đa (mặc định 60; test đặt 1)
#
# V09.A2: lệnh train-v2 dựng MỘT LẦN thành mảng `CMD` (`build_cmd`) — dry-run in đúng mảng, chế độ
# thật chạy đúng mảng.
# V11S.A2: `last.pt` HỎNG không được coi là "chưa có" — lùi về `ckpt_<step>.pt` mới nhất còn đọc
# được; không có gì đọc được thì DỪNG (mã 3), không tự khởi động lại từ số 0. Lần khởi động đầu
# còn chặn nếu thư mục đã có `history.csv` (mã 4).
set -u
OUT=$1; LOG=$2; MAX=$3; CONFIG=$4; TRAIN=$5; VAL=$6
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

INIT_RESUME="${INIT_RESUME:-}"
INIT_RESET_VAL_STATE="${INIT_RESET_VAL_STATE:-}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_CMD="${TRAIN_CMD:-.venv/bin/vifasttts train-v2}"
ATTEMPTS="${WATCH_MAX_ATTEMPTS:-60}"

# Bước trong MỘT tệp checkpoint; rỗng nếu không đọc được.
_step_of_file() {
  .venv/bin/python -c "import torch; s=torch.load('$1', map_location='cpu', weights_only=False); print(int(s.get('step', 0)))" 2>/dev/null
}

# Đặt STEP + STEP_FILE (tệp dùng để resume). STEP=ERR nếu last.pt tồn tại mà hỏng và không có ckpt nào đọc được.
probe_state() {
  STEP=0; STEP_FILE=""
  [ -f "$OUT/last.pt" ] || return 0
  STEP=$(_step_of_file "$OUT/last.pt")
  if [ -n "$STEP" ]; then STEP_FILE="$OUT/last.pt"; return 0; fi
  for file in $(ls -1 "$OUT"/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -nr); do
    step=$(_step_of_file "$file")
    if [ -n "$step" ]; then STEP=$step; STEP_FILE="$file"; return 0; fi
  done
  STEP="ERR"
}

# ── Dựng lệnh train-v2 MỘT LẦN (mảng) ──────────────────────────────────────────
build_cmd() {
  # shellcheck disable=SC2206  # TRAIN_CMD có thể nhiều từ (test dùng script giả)
  CMD=($TRAIN_CMD --config "$CONFIG" --train-manifest "$TRAIN" --val-manifest "$VAL" --output "$OUT")
  if [ "$STEP" -gt 0 ]; then
    CMD+=(--resume "$STEP_FILE")            # khởi động lại: KHÔNG reset val (hash đã khớp)
  elif [ -n "$INIT_RESUME" ]; then
    CMD+=(--resume "$INIT_RESUME")          # lần đầu: resume từ checkpoint ngoài
    [ "$INIT_RESET_VAL_STATE" = "1" ] && CMD+=(--reset-val-state)
  fi
  CMD+=(--max-steps "$MAX")
}

if [ "$DRY_RUN" = "1" ]; then
  probe_state
  if [ "$STEP" = "ERR" ]; then
    echo "LỖI: $OUT/last.pt hỏng và không có ckpt_*.pt đọc được — sẽ DỪNG (mã 3)"
    exit 3
  fi
  build_cmd
  echo "${CMD[*]}"
  exit 0
fi

for attempt in $(seq 1 "$ATTEMPTS"); do
  probe_state
  if [ "$STEP" = "ERR" ]; then
    echo "[watch $(date '+%H:%M')] LỖI: last.pt hỏng + không có ckpt_*.pt đọc được — DỪNG (mã 3)" >> "$LOG"
    exit 3
  fi
  if [ "$STEP" -eq 0 ] && [ -s "$OUT/history.csv" ]; then
    echo "[watch $(date '+%H:%M')] LỖI: $OUT/history.csv đã có mà last.pt không đọc được — DỪNG (mã 4)" >> "$LOG"
    exit 4
  fi
  if [ "$STEP" -ge "$MAX" ]; then
    echo "[watch $(date '+%H:%M')] đạt $STEP >= $MAX — kết thúc" >> "$LOG"
    break
  fi
  # Trainer đã tự quyết dừng sớm (patience) → TÔN TRỌNG, không resume lại (nếu không sẽ lặp 1k bước/lần).
  if [ -f "$OUT/early_stop.json" ]; then
    echo "[watch $(date '+%H:%M')] có early_stop.json — trainer đã dừng sớm, kết thúc" >> "$LOG"
    break
  fi
  echo "[watch $(date '+%H:%M')] lần $attempt: resume từ $STEP ($STEP_FILE) → $MAX" >> "$LOG"
  build_cmd
  "${CMD[@]}" >> "$LOG" 2>&1
  sleep 5
done
echo "[watch $(date '+%H:%M')] dừng (bước cuối $(probe_state >/dev/null 2>&1; echo "$STEP"))" >> "$LOG"
