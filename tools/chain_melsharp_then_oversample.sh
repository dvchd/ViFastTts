#!/bin/bash
# tools/chain_melsharp_then_oversample.sh — tự động nối tiếp: chờ run mel_sharp xong → probe → chạy
# run oversample (E4b recipe, b8 lineage). Chạy qua launchd; tự gỡ job cuối để không bị chạy lại:
#   launchctl submit -l vifasttts.chain -- /bin/bash "$PWD/tools/chain_melsharp_then_oversample.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_melsharp_oversample.log
PROBE_LOG=artifacts/dataset_v2/probe_oversample.log
echo "[chain] $(date '+%F %H:%M:%S') bắt đầu chờ mel_sharp xong" >> "$LOG"

# 1) chờ run mel_sharp kết thúc, gỡ job của nó (tránh launchd chạy lại vô ích)
while pgrep -f "vifasttts train-v2.*b8_melsharp" >/dev/null 2>&1; do sleep 60; done
sleep 30
launchctl remove vifasttts.melsharp 2>/dev/null || true
echo "[chain] $(date '+%F %H:%M:%S') mel_sharp xong — bắt đầu probe oversample" >> "$LOG"

# 2) probe 200 bước (đúng config sẽ chạy; cổng: exit 0 + không NaN)
.venv/bin/vifasttts train-v2 --config configs/small_codelung_v2_oversample.yaml \
  --train-manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
  --val-manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
  --output runs/_os_probe --resume runs/acoustic_v2_b8/best.pt --max-steps 70000 \
  --stop-at-step 46200 > "$PROBE_LOG" 2>&1
code=$?
if [ "$code" -ne 0 ] || grep -qi "nan" "$PROBE_LOG"; then
  echo "[chain] probe LỖI (code=$code) — DỪNG, không chạy run thật (xem $PROBE_LOG)" >> "$LOG"
  launchctl remove vifasttts.chain 2>/dev/null || true
  exit 1
fi
echo "[chain] probe ĐẠT — phóng run thật" >> "$LOG"

# 3) run thật (watchdog riêng; seed last.pt = b8/best@46k)
mkdir -p runs/acoustic_v2_b8_oversample
cp runs/acoustic_v2_b8/best.pt runs/acoustic_v2_b8_oversample/last.pt
launchctl submit -l vifasttts.oversample -- /bin/bash "$PWD/tools/watchdog_resume_train.sh" \
  runs/acoustic_v2_b8_oversample configs/small_codelung_v2_oversample.yaml \
  data/codelung/train_clean_v2_dur_seg_slot4.jsonl data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
  artifacts/dataset_v2/train_v2_b8_oversample.log 8 --max-steps 70000 >> "$LOG" 2>&1
echo "[chain] $(date '+%F %H:%M:%S') đã phóng vifasttts.oversample" >> "$LOG"
launchctl remove vifasttts.chain 2>/dev/null || true
