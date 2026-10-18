#!/bin/bash
# tools/watchdog_resume_train.sh — giữ một run train-v2 sống: nếu process chết thì tự resume từ last.pt.
#
# Dùng: tools/watchdog_resume_train.sh <out_dir> <config> <train_manifest> <val_manifest> <log_file> \
#        [max_retry] [cờ thêm cho train-v2 ...]
# Ví dụ (pha-2: resume từ checkpoint của run KHÁC — seed `last.pt` của thư mục mới trước khi chạy):
#   mkdir -p runs/acoustic_v2_b8_melsharp
#   cp runs/acoustic_v2_b8/best.pt runs/acoustic_v2_b8_melsharp/last.pt
#   launchctl submit -l vifasttts.melsharp -- /bin/bash "$PWD/tools/watchdog_resume_train.sh" \
#     runs/acoustic_v2_b8_melsharp configs/small_codelung_v2_melsharp.yaml \
#     data/codelung/train_clean_v2_dur_seg_slot4.jsonl data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
#     artifacts/dataset_v2/train_v2_b8_melsharp.log 8 --max-steps 70000
# Gỡ khi xong: launchctl remove <label>
#
# Lý do (18/10): process train small_filt bị giết im lặng lúc 13:20 (không crash report/jetsam)
# → cần lớp tự phục hồi; resume từ last.pt nối đúng lịch cosine (LR tính theo bước tuyệt đối).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT="$1"; CONFIG="$2"; TRAIN="$3"; VAL="$4"; LOG="$5"; MAX_RETRY="${6:-8}"
shift 6 2>/dev/null || true   # phần còn lại = cờ thêm (vd --max-steps 70000 --adversarial)

if [ ! -f "$OUT/last.pt" ]; then
  echo "[watchdog] $(date '+%F %H:%M:%S') THIẾU $OUT/last.pt — không tự chạy từ đầu, dừng." >> "$LOG"
  exit 1
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

CMD=(.venv/bin/vifasttts train-v2 --config "$CONFIG" \
  --train-manifest "$TRAIN" --val-manifest "$VAL" \
  --output "$OUT" --resume "$OUT/last.pt")
if [ "$#" -gt 0 ]; then CMD+=("$@"); fi

i=1
while [ "$i" -le "$MAX_RETRY" ]; do
  # Chống chạy trùng: nếu run này đã có tiến trình train-v2 thì chờ nó kết thúc trước.
  while pgrep -f "vifasttts train-v2.*$OUT" >/dev/null 2>&1; do sleep 60; done
  echo "[watchdog] $(date '+%F %H:%M:%S') lần $i — resume từ step $(step_now)" >> "$LOG"
  "${CMD[@]}" >> "$LOG" 2>&1
  code=$?
  echo "[watchdog] $(date '+%F %H:%M:%S') train kết thúc code=$code ở step $(step_now)" >> "$LOG"
  [ "$code" -eq 0 ] && exit 0
  sleep 60
  i=$((i + 1))
done
echo "[watchdog] hết $MAX_RETRY lần thử — dừng." >> "$LOG"
exit 1
