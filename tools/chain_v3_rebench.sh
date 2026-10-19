#!/bin/bash
# tools/chain_v3_rebench.sh — sau khi resume v3 xong (34k→60k): chấm lại toàn bộ + CER + demo + báo cáo.
#
# Chạy qua launchd: launchctl submit -l vifasttts.v3rebench -- /bin/bash "$PWD/tools/chain_v3_rebench.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_v3_rebench.log
V3=data/codelung/v3
say() { echo "[v3r] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }
PY=.venv/bin/python
TRAIN_M=$V3/train_clean_v2_dur_seg_slot4.jsonl
VAL_M=$V3/val_clean_v2_dur_seg_slot4.jsonl
HOLD_M=$V3/holdout_clean_v2_dur.jsonl
FILT_VAL=data/codelung/val_clean_v2_dur_seg_slot4_filt.jsonl
FILT_HOLD=data/codelung/holdout_clean_v2_dur_slot4_filt.jsonl

say "chờ resume v3 xong"
waited=0
while pgrep -f "train-v2.*acoustic_v3_clean" >/dev/null 2>&1 && [ "$waited" -lt 300 ]; do sleep 60; waited=$((waited + 1)); done
launchctl remove vifasttts.v3resume 2>/dev/null || true
say "resume xong (step $(tail -1 runs/acoustic_v3_clean/history.csv | cut -d, -f1) · val $(tail -1 runs/acoustic_v3_clean/history.csv | cut -d, -f8))"

best=runs/acoustic_v3_clean/best.pt
[ -f "$best" ] || best=runs/acoustic_v3_clean/last.pt

bench() {  # $1=ckpt $2=tag $3=val $4=holdout [$5=baseline]
  [ -f "artifacts/bench/$2.json" ] && return 0
  say "bench $2"
  local cmd=($PY tools/bench_checkpoint.py --run runs/acoustic_v3_clean --checkpoint "$1" --tag "$2" --device mps
             --val-manifest "$3" --val-manifest-old "$3" --holdout-manifest "$4")
  if [ -n "${5:-}" ]; then
    cmd+=(--baseline-old "$5" --baseline-fixgap "$5")
  fi
  "${cmd[@]}" >> "$LOG" 2>&1 || say "bench $2 LỖI"
}
bench "$best" v3r_best "$VAL_M" "$HOLD_M"
bench runs/acoustic_v3_clean/last.pt v3r_last "$VAL_M" "$HOLD_M"
bench "$best" v3r_onfilt "$FILT_VAL" "$FILT_HOLD" eval/baselines/b8_filt.json

if [ ! -f artifacts/eval_acoustic/v3r_cer100.json ]; then
  say "CER-100 (holdout v3 đã sửa)"
  $PY tools/eval_acoustic.py --config runs/acoustic_v3_clean/resolved_config.json --checkpoint runs/acoustic_v3_clean/last.pt \
    --mode free --manifest "$HOLD_M" --cache-dir cache/codelung_v3 --clips 100 --device mps \
    --out artifacts/eval_acoustic/v3r_cer100.json >> "$LOG" 2>&1 || say "CER-100 LỖI"
fi
if [ ! -f artifacts/creak_rarity/probe_v3r.json ]; then
  say "rarity probe"
  $PY tools/creak_rarity_probe.py --config runs/acoustic_v3_clean/resolved_config.json --checkpoint runs/acoustic_v3_clean/last.pt \
    --manifest "$TRAIN_M" --out artifacts/creak_rarity/probe_v3r.json --device cpu >> "$LOG" 2>&1 || say "probe LỖI"
fi
if [ ! -f artifacts/listen_small_filt/v3r/index.html ]; then
  say "demo v3r (last + best)"
  mkdir -p artifacts/listen_small_filt/v3r_ckpts
  cp runs/acoustic_v3_clean/last.pt artifacts/listen_small_filt/v3r_ckpts/v3r_last.pt
  cp "$best" artifacts/listen_small_filt/v3r_ckpts/v3r_best.pt
  $PY tools/listen_demo.py --config runs/acoustic_v3_clean/resolved_config.json \
    --checkpoint runs/acoustic_v2_b8/best.pt \
    --checkpoint artifacts/listen_small_filt/v3r_ckpts/v3r_last.pt \
    --checkpoint artifacts/listen_small_filt/v3r_ckpts/v3r_best.pt \
    --manifest artifacts/demo_small_filt/manifest.jsonl --clips 4 --sharpen-alpha 1.5 \
    --minimal-pairs artifacts/demo_small_filt/minimal_pairs.txt --device mps \
    --out artifacts/listen_small_filt/v3r >> "$LOG" 2>&1 || say "demo LỖI"
fi

say "report"
$PY tools/night_report.py --out docs/handoff/DATA_V3_RESUME.md --baseline b8_filt \
  --row "b8 trên val v3 (mốc cùng tập)|b8_onv3|b8v3|runs/acoustic_v2_b8" \
  --row "v3r best|v3r_best|v3r|runs/acoustic_v3_clean" \
  --row "v3r last (60k)|v3r_last|v3r|runs/acoustic_v3_clean" \
  --row "v3r trên val_filt (cũ)|v3r_onfilt|v3r|runs/acoustic_v3_clean" \
  --row "v3 cũ best@9k (tham chiếu)|v3_best|v3|runs/acoustic_v3_clean" >> "$LOG" 2>&1 || say "report LỖI"

date '+%F %H:%M:%S' > artifacts/dataset_v2/v3r_done.marker
say "===== DONE ====="
launchctl remove vifasttts.v3rebench 2>/dev/null || true
