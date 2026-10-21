#!/bin/bash
# tools/chain_v34.sh — v3.4: dữ liệu v3.2 + **coda/onset-weighted mel loss** (small).
#
# Vì sao: coda chỉ 3–6 frame/row nên L1 bỏ qua ⇒ model đọc NH thành N ("xanh"→"xan") và
# dễ "trộn từ" ở biên (§53). Trọng số: onset +2,0×, coda +3,0× (config `loss_weights`).
# Chạy trên SMALL vì base 17M không thắng (val 0,618 vs 0,597).
#
#   launchctl submit -l vifasttts.v34 -- /bin/bash "$PWD/tools/chain_v34.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_v34.log
V=data/codelung/v32
M=artifacts/dataset_v2
PY=.venv/bin/python
say() { echo "[v34] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }
TRAIN_M=$V/train_clean_v2_dur_seg_slot4.jsonl
VAL_M=$V/val_clean_v2_dur_seg_slot4.jsonl
HOLD_M=$V/holdout_clean_v2_dur_seg_slot4.jsonl
CFG=configs/small_codelung_v34.yaml

say "===== bắt đầu ====="
# S1 — train 60k từ đầu (patience 0; resume nếu có last.pt)
if [ ! -f $M/v34_s1.ok ]; then
  mkdir -p runs/acoustic_v34
  say "S1 train bắt đầu"
  if [ -f runs/acoustic_v34/last.pt ]; then
    /bin/bash tools/watchdog_resume_train.sh runs/acoustic_v34 $CFG "$TRAIN_M" "$VAL_M" \
      artifacts/dataset_v2/train_v34.log 6 >> "$LOG" 2>&1
  else
    .venv/bin/vifasttts train-v2 --config $CFG --train-manifest "$TRAIN_M" \
      --val-manifest "$VAL_M" --output runs/acoustic_v34 >> "$LOG" 2>&1
  fi
  touch $M/v34_s1.ok; say "S1 train xong ($(tail -1 runs/acoustic_v34/history.csv | cut -d, -f1) bước)"
fi

# S2 — chấm: v34 (best/last) + coda + minimal pairs + demo steady + CER
if [ ! -f $M/v34_s2.ok ]; then
  best=runs/acoustic_v34/best.pt; [ -f "$best" ] || best=runs/acoustic_v34/last.pt
  for spec in "$best|v34_best" "runs/acoustic_v34/last.pt|v34_last"; do
    ckpt=${spec%%|*}; tag=${spec#*|}
    [ -f "artifacts/bench/$tag.json" ] && continue
    say "bench $tag"
    $PY tools/bench_checkpoint.py --run runs/acoustic_v34 --checkpoint "$ckpt" --tag "$tag" \
      --clips 300 --val-manifest "$VAL_M" --val-manifest-old "$VAL_M" --holdout-manifest "$HOLD_M" \
      --holdout 40 --tone-clips 80 --creak-clips 40 --device mps --no-cer >> "$LOG" 2>&1 \
      || say "bench $tag LỖI"
  done
  if [ ! -f artifacts/eval_acoustic/v34_coda.json ]; then
    $PY tools/coda_realization.py --config runs/acoustic_v34/resolved_config.json \
      --checkpoint "$best" --manifest "$VAL_M" --cache-dir cache/codelung_v2 --clips 20 \
      --device mps --out artifacts/eval_acoustic/v34_coda.json >> "$LOG" 2>&1 || say "coda LỖI"
  fi
  # minimal pairs NH/N/NG/M: làm trang cho người nghe + đo centroid
  if [ ! -f artifacts/listen_small_filt/nh_v34/steady/xanh.wav ]; then
    $PY tools/steady_demo.py --config runs/acoustic_v34/resolved_config.json --checkpoint "$best" \
      --texts artifacts/nh_pairs.json --device mps \
      --out artifacts/listen_small_filt/nh_v34 >> "$LOG" 2>&1 || say "minimal pairs LỖI"
  fi
  if [ ! -f artifacts/listen_small_filt/steady_v34/index.html ]; then
    $PY tools/steady_demo.py --config runs/acoustic_v34/resolved_config.json --checkpoint "$best" \
      --device mps --out artifacts/listen_small_filt/steady_v34 >> "$LOG" 2>&1 || say "demo LỖI"
  fi
  if [ ! -f artifacts/eval_acoustic/v34_cer100.json ]; then
    $PY tools/eval_acoustic.py --config runs/acoustic_v34/resolved_config.json --checkpoint "$best" \
      --mode free --manifest "$HOLD_M" --cache-dir cache/codelung_v2 --clips 100 --device mps \
      --out artifacts/eval_acoustic/v34_cer100.json >> "$LOG" 2>&1 || say "CER-100 LỖI"
  fi
  touch $M/v34_s2.ok; say "S2 xong"
fi

$PY tools/night_report.py --out docs/handoff/DATA_V34_CODA_21_10.md --baseline b8_filt \
  --row "v32 small (mốc)|v32_best|v32|runs/acoustic_v32" \
  --row "v34 coda/onset|v34_best|v34|runs/acoustic_v34" \
  --row "v34 last|v34_last|v34|runs/acoustic_v34" >> "$LOG" 2>&1 || say "report LỖI"
date '+%F %H:%M:%S' > $M/v34_done.marker
say "===== DONE ====="
launchctl remove vifasttts.v34 2>/dev/null || true
