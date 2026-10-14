#!/usr/bin/env bash
# scripts/run_A2_fixgap.sh — A2 (fixtail): đủ gap + TRẦN lặng đuôi 67 + loss pause CŨ. (§44, V09)
#   probe   : train-v2 trực tiếp — --max-steps 60000 --stop-at-step 40100 --validate-at-stop
#   run     : qua watchdog, lần khởi động đầu resume runs/acoustic_v2_base/last.pt + --reset-val-state
#   dry-run : in ĐÚNG mảng lệnh mà chế độ run sẽ chạy (cùng watchdog, DRY_RUN=1) — V09.A2
set -euo pipefail
cd "$(dirname "$0")/.."
TRAIN=data/codelung/train_clean_v2_dur_seg_slot4_fixtail.jsonl
VAL=data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl
CFG=configs/small_codelung_v2_base_A2_fixgap.yaml
BASE=runs/acoustic_v2_base/last.pt
PROBE_OUT=runs/acoustic_v2_A2_fixtail_probe
RUN_OUT=runs/acoustic_v2_A2_fixtail

check() { local f=$1 want=$2 got; got=$(shasum -a 1 "$f" | cut -d' ' -f1); [ "$got" = "$want" ] || {
  echo "SHA1 LỆCH: $f"; echo "  got  $got"; echo "  want $want"; exit 1; }; }
check "$TRAIN" 965b4798b23266c6006be658dd1e93e13d8072ac   # cập nhật theo eval/README.md
check "$VAL" cdbbd659116ae9f948c56152be5b08ea7fa08115

case "${1:-}" in
  probe)
    # V11S.A4: chặn ghi đè im lặng — chỉ chạy khi thư mục probe CHƯA có last.pt (hoặc cố ý).
    if [ -f "$PROBE_OUT/last.pt" ] && [ "${2:-}" != "--overwrite-probe" ]; then
      echo "CHẶN: $PROBE_OUT/last.pt đã tồn tại — dùng '$0 probe --overwrite-probe' nếu cố ý."
      exit 2
    fi
    echo "== PROBE tới bước 40100 =="
    .venv/bin/vifasttts train-v2 --config "$CFG" --train-manifest "$TRAIN" --val-manifest "$VAL" \
      --output "$PROBE_OUT" --resume "$BASE" --reset-val-state \
      --max-steps 60000 --stop-at-step 40100 --validate-at-stop ;;
  dry-run)
    echo "== DRY-RUN: đúng mảng lệnh mà chế độ run sẽ chạy =="
    INIT_RESUME="$BASE" INIT_RESET_VAL_STATE=1 DRY_RUN=1 \
      tools/watch_train_v2.sh "$RUN_OUT" /dev/null 60000 "$CFG" "$TRAIN" "$VAL" ;;
  run)
    if [ -f "$RUN_OUT/last.pt" ] && [ "${2:-}" != "--continue" ]; then
      echo "CHẶN: $RUN_OUT/last.pt đã tồn tại — dùng '$0 run --continue' nếu cố ý."; exit 2; fi
    echo "== A2 THẬT tới 60k (fixtail) =="
    INIT_RESUME="$BASE" INIT_RESET_VAL_STATE=1 tools/watch_train_v2.sh "$RUN_OUT" \
      artifacts/dataset_v2/train_v2_A2_fixtail.log 60000 "$CFG" "$TRAIN" "$VAL" ;;
  *) echo "Dùng: $0 probe | run [--continue] | dry-run"; exit 1 ;;
esac
