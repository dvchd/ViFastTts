#!/bin/bash
# tools/chain_v32_base.sh — v3.2 BASE: cùng data/cắt segment v3.2, model base (256/6/4, ~15M).
#
# Trả lời: dung lượng model (7M → 15M) có giảm rè (AP 2–6 kHz), mờ onset (spectral flux) và coda
# hay không — trên cùng val với b8/v3.2 small, cùng cache (audio không đổi).
#
#   launchctl submit -l vifasttts.v32base -- /bin/bash "$PWD/tools/chain_v32_base.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_v32_base.log
V=data/codelung/v32
M=artifacts/dataset_v2
PY=.venv/bin/python
say() { echo "[v32base] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }
TRAIN_M=$V/train_clean_v2_dur_seg_slot4.jsonl
VAL_M=$V/val_clean_v2_dur_seg_slot4.jsonl
HOLD_M=$V/holdout_clean_v2_dur_seg_slot4.jsonl
CFG=configs/base_codelung_v32.yaml

say "===== bắt đầu ====="
# S1 — train 40k từ đầu (patience 0; resume nếu có last.pt)
if [ ! -f $M/v32base_s1.ok ]; then
  mkdir -p runs/acoustic_v32_base
  say "S1 train bắt đầu"
  if [ -f runs/acoustic_v32_base/last.pt ]; then
    /bin/bash tools/watchdog_resume_train.sh runs/acoustic_v32_base $CFG "$TRAIN_M" "$VAL_M" \
      artifacts/dataset_v2/train_v32base.log 6 >> "$LOG" 2>&1
  else
    .venv/bin/vifasttts train-v2 --config $CFG --train-manifest "$TRAIN_M" \
      --val-manifest "$VAL_M" --output runs/acoustic_v32_base >> "$LOG" 2>&1
  fi
  touch $M/v32base_s1.ok
  say "S1 train xong ($(tail -1 runs/acoustic_v32_base/history.csv | cut -d, -f1) bước)"
fi

# S2 — chấm: base (best/last) trên cùng val v32 + coda + demo steady
if [ ! -f $M/v32base_s2.ok ]; then
  best=runs/acoustic_v32_base/best.pt; [ -f "$best" ] || best=runs/acoustic_v32_base/last.pt
  for spec in "$best|v32base_best" "runs/acoustic_v32_base/last.pt|v32base_last"; do
    ckpt=${spec%%|*}; tag=${spec#*|}
    [ -f "artifacts/bench/$tag.json" ] && continue
    say "bench $tag"
    $PY tools/bench_checkpoint.py --run runs/acoustic_v32_base --checkpoint "$ckpt" --tag "$tag" \
      --clips 300 --val-manifest "$VAL_M" --val-manifest-old "$VAL_M" --holdout-manifest "$HOLD_M" \
      --holdout 40 --tone-clips 80 --creak-clips 40 --device mps --no-cer >> "$LOG" 2>&1 \
      || say "bench $tag LỖI"
  done
  if [ ! -f artifacts/eval_acoustic/v32base_coda.json ]; then
    $PY tools/coda_realization.py --config runs/acoustic_v32_base/resolved_config.json \
      --checkpoint "$best" --manifest "$VAL_M" --cache-dir cache/codelung_v2 --clips 20 \
      --device mps --out artifacts/eval_acoustic/v32base_coda.json >> "$LOG" 2>&1 || say "coda LỖI"
  fi
  if [ ! -f artifacts/listen_small_filt/steady_v32base/index.html ]; then
    $PY tools/steady_demo.py --config runs/acoustic_v32_base/resolved_config.json \
      --checkpoint "$best" --device mps --out artifacts/listen_small_filt/steady_v32base \
      >> "$LOG" 2>&1 || say "demo LỖI"
  fi
  if [ ! -f artifacts/eval_acoustic/v32base_cer100.json ]; then
    $PY tools/eval_acoustic.py --config runs/acoustic_v32_base/resolved_config.json \
      --checkpoint "$best" --mode free --manifest "$HOLD_M" --cache-dir cache/codelung_v2 \
      --clips 100 --device mps --out artifacts/eval_acoustic/v32base_cer100.json \
      >> "$LOG" 2>&1 || say "CER-100 LỖI"
  fi
  touch $M/v32base_s2.ok; say "S2 xong"
fi

$PY tools/night_report.py --out docs/handoff/DATA_V32_BASE_21_10.md --baseline b8_filt \
  --row "b8 trên val v32|b8_on_v32val|b8v32|runs/acoustic_v2_b8" \
  --row "v32 small (7M)|v32_best|v32|runs/acoustic_v32" \
  --row "v32 base (15M)|v32base_best|v32base|runs/acoustic_v32_base" >> "$LOG" 2>&1 || say "report LỖI"
date '+%F %H:%M:%S' > $M/v32base_done.marker
say "===== DONE ====="
launchctl remove vifasttts.v32base 2>/dev/null || true
