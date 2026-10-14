#!/usr/bin/env bash
# scripts/run_A2_fixgap.sh — A2 = fixgap + loss pause CŨ (một biến). (§44/A2)
#   probe : --max-steps 60000 --stop-at-step 40100, thư mục RIÊNG *_probe (LR y như run thật)
#   thật  : resume runs/acoustic_v2_base/last.pt@40000 (KHÔNG resume từ probe)
# Kiểm sha1 manifest khớp hash ghim trước khi chạy; chặn nếu thư mục đích đã có last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN=data/codelung/train_clean_v2_dur_seg_slot4_fixgap.jsonl
VAL=data/codelung/val_clean_v2_dur_seg_slot4_fixgap.jsonl
CFG=configs/small_codelung_v2_base_A2_fixgap.yaml
BASE=runs/acoustic_v2_base/last.pt
MODE="${1:-}"
check() { local f=$1 want=$2 got; got=$(shasum -a 1 "$f" | cut -d' ' -f1); [ "$got" = "$want" ] || { echo "SHA1 LỆCH: $f"; echo "  got  $got"; echo "  want $want"; exit 1; }; }
check "$TRAIN" fdb622940e0de9560a83050f0c8467cd34dea445
check "$VAL" 5ec5f139ce56a763d90af7de7a59956f712ee88c
echo "sha1 manifest OK"

case "$MODE" in
  probe)
    OUT=runs/acoustic_v2_A2_fixgap_probe
    echo "== PROBE tới bước TUYỆT ĐỐI 40100 (thư mục $OUT) =="
    .venv/bin/vifasttts train-v2 --config "$CFG" --train-manifest "$TRAIN" --val-manifest "$VAL" \
      --output "$OUT" --resume "$BASE" --reset-val-state \
      --max-steps 60000 --stop-at-step 40100
    ;;
  run)
    OUT=runs/acoustic_v2_A2_fixgap
    if [ -f "$OUT/last.pt" ] && [ "${2:-}" != "--continue" ]; then
      echo "CHẶN: $OUT/last.pt đã tồn tại — nối nhầm run? Dùng '/bin/zsh run --continue' nếu cố ý."
      exit 2
    fi
    echo "== A2 THẬT tới 60k (thư mục $OUT, resume $BASE) =="
    tools/watch_train_v2.sh "$OUT" artifacts/dataset_v2/train_v2_A2_fixgap.log 60000 "$CFG" "$TRAIN" "$VAL"
    ;;
  *)
    echo "Dùng: $0 probe | $0 run [--continue]"; exit 1;;
esac
