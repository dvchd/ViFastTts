#!/usr/bin/env bash
# Watchdog cho `train-v2`: tự chạy lại kèm `--resume` cho tới khi đạt `max_steps`
# (chống crash/NaN khi chạy dài qua đêm). Đặt đủ env watermark MPS (bài học §29.3).
#
# Dùng: tools/watch_train_v2.sh <out_dir> <log_file> <max_steps> <config> <train_manifest> <val_manifest>
set -u
OUT=$1; LOG=$2; MAX=$3; CONFIG=$4; TRAIN=$5; VAL=$6
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

step_of() {
  if [ -f "$OUT/last.pt" ]; then
    .venv/bin/python -c "import torch; s=torch.load('$OUT/last.pt', map_location='cpu', weights_only=False); print(int(s.get('step', 0)))" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

for attempt in $(seq 1 60); do
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
  if [ "$STEP" -gt 0 ]; then
    .venv/bin/vifasttts train-v2 --config "$CONFIG" --train-manifest "$TRAIN" \
      --val-manifest "$VAL" --output "$OUT" --resume "$OUT/last.pt" --max-steps "$MAX" >> "$LOG" 2>&1
  else
    .venv/bin/vifasttts train-v2 --config "$CONFIG" --train-manifest "$TRAIN" \
      --val-manifest "$VAL" --output "$OUT" --max-steps "$MAX" >> "$LOG" 2>&1
  fi
  sleep 5
done
echo "[watch $(date '+%H:%M')] dừng (bước cuối $(step_of))" >> "$LOG"
