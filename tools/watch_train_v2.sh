#!/usr/bin/env bash
# Watchdog cho train acoustic v2: chạy train_chunked (tự resume), nếu runner chết thì
# khởi động lại (tối đa 40 lần); dừng khi đủ bước hoặc early-stop.
#
#   nohup caffeinate -is bash tools/watch_train_v2.sh >/dev/null 2>&1 &
#
# Log: artifacts/small_codelung/train_v2_full2.log
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=artifacts/small_codelung/train_v2_full2.log
mkdir -p artifacts/small_codelung runs
TOTAL="${TOTAL:-60000}"
OUT=runs/small_codelung_v2

log() { echo "[watchdog $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
current_step() {
    .venv/bin/python - <<'PY'
import torch
from pathlib import Path
p = Path('runs/small_codelung_v2/last.pt')
print(int(torch.load(p, map_location='cpu', weights_only=False).get('step', 0)) if p.exists() else 0)
PY
}

log "bắt đầu watchdog (total=$TOTAL)"
for i in $(seq 1 40); do
    if [ -f "$OUT/early_stop.json" ]; then log "early-stop — dừng"; break; fi
    step=$(current_step)
    if [ "$step" -ge "$TOTAL" ]; then log "đủ bước ($step) — dừng"; break; fi
    log "lần $i: chạy từ bước $step"
    caffeinate -is .venv/bin/python tools/train_chunked.py \
        --config configs/small_codelung.yaml \
        --train-manifest data/codelung/train_clean_v2_dur_seg.jsonl \
        --val-manifest data/codelung/val_clean_v2_dur_seg.jsonl \
        --output "$OUT" --total-steps "$TOTAL" --chunk 5000 >> "$LOG" 2>&1
    log "runner thoát rc=$? · hiện ở bước $(current_step)"
    sleep 30
done
log "watchdog kết thúc"
