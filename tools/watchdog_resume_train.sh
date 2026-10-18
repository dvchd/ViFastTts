#!/bin/bash
# tools/watchdog_resume_train.sh — giữ một run train-v2 sống qua đêm: nếu process chết thì tự resume từ last.pt.
#
# Dùng (chạy qua launchd để không chết theo phiên terminal):
#   tools/watchdog_resume_train.sh <out_dir> <config> <train_manifest> <val_manifest> <log_file> [max_retry]
# Ví dụ submit:
#   launchctl submit -l vifasttts.small_filt.watchdog -- /bin/bash \
#     "$PWD/tools/watchdog_resume_train.sh" runs/acoustic_v2_small_filt configs/small_codelung_v2.yaml \
#     data/codelung/train_clean_v2_dur_seg_slot4_filt.jsonl data/codelung/val_clean_v2_dur_seg_slot4_filt.jsonl \
#     artifacts/dataset_v2/train_v2_small_filt.log 8
# Gỡ khi xong: launchctl remove <label>
#
# Lý do (18/10): process train small_filt bị giết im lặng lúc 13:20 (không crash report/jetsam)
# → cần lớp tự phục hồi; resume từ last.pt nối đúng lịch cosine (LR tính theo bước tuyệt đối).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT="$1"; CONFIG="$2"; TRAIN="$3"; VAL="$4"; LOG="$5"; MAX_RETRY="${6:-8}"

if [ ! -f "$OUT/last.pt" ]; then
  echo "[watchdog] $(date '+%F %H:%M:%S') THIẾU $OUT/last.pt — không tự chạy từ đầu, dừng." >> "$LOG"
  exit 1
fi

# Chống 2 watchdog chạy song song (pgrep không tự khớp chính nó, trừ shell gọi script).
others=$(pgrep -f "watchdog_resume_train.sh" | grep -v "^$$\$" | wc -l | tr -d ' ')
if [ "$others" -gt 0 ]; then
  echo "[watchdog] $(date '+%F %H:%M:%S') đã có watchdog khác đang chạy — thoát." >> "$LOG"
  exit 0
fi

step_now() {
  .venv/bin/python - "$OUT/last.pt" 2>/dev/null <<'PY' || echo "?"
import sys
import torch
try:
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(int(state.get("step", 0)))
except Exception:
    print("?")
PY
}

i=1
while [ "$i" -le "$MAX_RETRY" ]; do
  # Chống chạy trùng: nếu run này đã có tiến trình train-v2 thì chờ nó kết thúc trước.
  while pgrep -f "vifasttts train-v2.*$OUT" >/dev/null 2>&1; do sleep 60; done
  echo "[watchdog] $(date '+%F %H:%M:%S') lần $i — resume từ step $(step_now)" >> "$LOG"
  .venv/bin/vifasttts train-v2 --config "$CONFIG" \
    --train-manifest "$TRAIN" --val-manifest "$VAL" \
    --output "$OUT" --resume "$OUT/last.pt" >> "$LOG" 2>&1
  code=$?
  echo "[watchdog] $(date '+%F %H:%M:%S') train kết thúc code=$code ở step $(step_now)" >> "$LOG"
  [ "$code" -eq 0 ] && exit 0
  sleep 60
  i=$((i + 1))
done
echo "[watchdog] hết $MAX_RETRY lần thử — dừng." >> "$LOG"
exit 1
