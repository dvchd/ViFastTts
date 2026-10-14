#!/usr/bin/env bash
# scripts/run_A2_fixgap.sh — A2 = fixgap + loss pause CŨ (một biến), resume base 40k (§44).
# Kiểm sha1 manifest khớp hash đã ghim TRƯỚC khi chạy (R5); sai thì dừng.
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN=data/codelung/train_clean_v2_dur_seg_slot4_fixgap.jsonl
VAL=data/codelung/val_clean_v2_dur_seg_slot4_fixgap.jsonl
CFG=configs/small_codelung_v2_base_A2_fixgap.yaml
OUT=runs/acoustic_v2_A2_fixgap
check() { local f=$1 want=$2 got; got=$(shasum -a 1 "$f" | cut -d' ' -f1); [ "$got" = "$want" ] || { echo "SHA1 LỆCH: $f"; echo "  got  $got"; echo "  want $want"; exit 1; }; }
check "$TRAIN" fdb622940e0de9560a83050f0c8467cd34dea445
check "$VAL" 5ec5f139ce56a763d90af7de7a59956f712ee88c
echo "sha1 manifest OK"
if [ "${1:-}" = "probe" ]; then
  echo "== PROBE 100 bước =="
  .venv/bin/vifasttts train-v2 --config "$CFG" --train-manifest "$TRAIN" --val-manifest "$VAL" \
    --output "$OUT" --resume runs/acoustic_v2_base/last.pt --reset-val-state --max-steps 100
else
  echo "== A2 THẬT tới 60k =="
  tools/watch_train_v2.sh "$OUT" artifacts/dataset_v2/train_v2_A2_fixgap.log 60000 "$CFG" "$TRAIN" "$VAL"
fi
