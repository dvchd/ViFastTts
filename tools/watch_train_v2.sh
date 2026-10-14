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
# V09.A2: lệnh train-v2 được dựng MỘT LẦN thành mảng `CMD` (hàm `build_cmd`); chế độ thật chạy
# đúng mảng đó, DRY_RUN in đúng mảng đó ⇒ không còn hai chỗ dựng lệnh khác nhau.
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

step_of() {
  if [ -f "$OUT/last.pt" ]; then
    .venv/bin/python -c "import torch; s=torch.load('$OUT/last.pt', map_location='cpu', weights_only=False); print(int(s.get('step', 0)))" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

# ── Dựng lệnh train-v2 MỘT LẦN (mảng) ──────────────────────────────────────────
# $1 = step hiện tại của OUT (0 ⇒ lần khởi động đầu)
build_cmd() {
  local step=$1
  # shellcheck disable=SC2206  # TRAIN_CMD có thể nhiều từ (test dùng script giả)
  CMD=($TRAIN_CMD --config "$CONFIG" --train-manifest "$TRAIN" --val-manifest "$VAL" --output "$OUT")
  if [ "$step" -gt 0 ]; then
    CMD+=(--resume "$OUT/last.pt")          # khởi động lại: KHÔNG reset val (hash đã khớp)
  elif [ -n "$INIT_RESUME" ]; then
    CMD+=(--resume "$INIT_RESUME")          # lần đầu: resume từ checkpoint ngoài
    [ "$INIT_RESET_VAL_STATE" = "1" ] && CMD+=(--reset-val-state)
  fi
  CMD+=(--max-steps "$MAX")
}

if [ "$DRY_RUN" = "1" ]; then
  build_cmd "$(step_of)"
  echo "${CMD[*]}"
  exit 0
fi

for attempt in $(seq 1 "$ATTEMPTS"); do
  STEP=$(step_of)
  if [ "$STEP" -ge "$MAX" ]; then
    echo "[watch $(date '+%H:%M')] đạt $STEP >= $MAX — kết thúc" >> "$LOG"
    break
  fi
  # Trainer đã tự quyết dừng sớm (patience) → TÔN TRỌNG, không resume lại (nếu không sẽ lặp 1k bước/lần).
  if [ -f "$OUT/early_stop.json" ]; then
    echo "[watch $(date '+%H:%M')] có early_stop.json — trainer đã dừng sớm, kết thúc" >> "$LOG"
    break
  fi
  echo "[watch $(date '+%H:%M')] lần $attempt: resume từ $STEP → $MAX" >> "$LOG"
  build_cmd "$STEP"
  "${CMD[@]}" >> "$LOG" 2>&1
  sleep 5
done
echo "[watch $(date '+%H:%M')] dừng (bước cuối $(step_of))" >> "$LOG"
