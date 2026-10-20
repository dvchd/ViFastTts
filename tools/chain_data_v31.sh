#!/bin/bash
# tools/chain_data_v31.sh — Data v3.1: audio v2 (ĐÚNG bản b8) + CHỈ cap pause 0,7s → train từ đầu.
# Một biến số duy nhất so với b8: cap pause (mục tiêu "đọc đều"). Chạy qua launchd:
#   launchctl submit -l vifasttts.v31 -- /bin/bash "$PWD/tools/chain_data_v31.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_data_v31.log
V=data/codelung/v31
M=artifacts/dataset_v2
say() { echo "[v31] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }
PY=.venv/bin/python
TRAIN_M=$V/train_clean_v2_dur_seg_slot4.jsonl
VAL_M=$V/val_clean_v2_dur_seg_slot4.jsonl
HOLD_M=$V/holdout_clean_v2_dur.jsonl

say "===== bắt đầu ====="
# S2 build: audio v2 + cap pause + seg4
if [ ! -f $M/v31_s2.ok ]; then
  $PY tools/v3_build_clips.py \
    --manifests data/codelung/train_clean_v2_dur.jsonl,data/codelung/val_clean_v2_dur.jsonl,data/codelung/holdout_clean_v2_dur.jsonl \
    --old-seg data/codelung/train_clean_v2_dur_seg_slot4.jsonl,data/codelung/val_clean_v2_dur_seg_slot4.jsonl,data/codelung/holdout_clean_v2_dur_slot4.jsonl \
    --denoised-dir data/codelung/audio_v2 --out-dir $V --audio-prefix data/codelung/v31/audio >> "$LOG" 2>&1 \
    || { say "S2 LỖI"; exit 1; }
  $PY - <<'PYEOF' >> "$LOG" 2>&1
import json
from pathlib import Path
p = Path("data/codelung/v31/holdout_clean_v2_dur.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
for r in rows:
    if "seg_durations" not in r and "seg4" in r:
        r["seg_durations"] = r["seg4"]
p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print("[v31] holdout seg_durations OK", len(rows))
PYEOF
  touch $M/v31_s2.ok; say "S2 xong"
fi
# S3 cắt segment + S4 ghép seg4
if [ ! -f $M/v31_s3.ok ]; then
  $PY tools/cut_segments.py --manifest $V/train_clean_v2_dur.jsonl --manifest $V/val_clean_v2_dur.jsonl --out-dir $V >> "$LOG" 2>&1 || { say "S3 LỖI"; exit 1; }
  $PY tools/v3_segments_slot4.py --clip-manifests $V/train_clean_v2_dur.jsonl,$V/val_clean_v2_dur.jsonl --seg-dir $V >> "$LOG" 2>&1 || { say "S4 LỖI"; exit 1; }
  touch $M/v31_s3.ok; say "S3+S4 xong"
fi
# S5 QC
if [ ! -f $M/v31_s5.ok ]; then
  $PY tools/audit_clip_quality.py $TRAIN_M $VAL_M $HOLD_M --report artifacts/clip_qc/v31_report.json --workers 8 >> "$LOG" 2>&1 || { say "S5 LỖI"; exit 1; }
  $PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "S5 CỔNG TRƯỢT"; exit 1; }
import json, statistics
from pathlib import Path
d = json.loads(Path("artifacts/clip_qc/v31_report.json").read_text())
rows = [r for r in d["rows"].values() if isinstance(r, dict) and "noise_floor_db" in r]
nf = statistics.median([r["noise_floor_db"] for r in rows])
clip = max(r["clip_frac"] for r in rows)
print(f"[v31-qc] clip={len(rows)} · nền p50={nf:.1f} dB · clip_frac max={clip:.4f}")
assert nf < -65 and clip < 0.01, "QC trượt cổng"
PYEOF
  touch $M/v31_s5.ok; say "S5 xong"
fi
# S6 cache v31
for spec in "train|$TRAIN_M" "val|$VAL_M" "holdout|$HOLD_M"; do
  name=${spec%%|*}; man=${spec#*|}
  if [ ! -f $M/v31_cache_$name.ok ]; then
    .venv/bin/vifasttts cache-features --manifest "$man" --config configs/small_codelung_v31.yaml \
      --cache-dir cache/codelung_v31 --workers 6 >> "$LOG" 2>&1 || { say "S6 cache $name LỖI"; exit 1; }
    touch $M/v31_cache_$name.ok
  fi
done
say "S6 xong (cache v31)"
# S7 train từ đầu (patience 0)
if [ ! -f $M/v31_s7.ok ]; then
  mkdir -p runs/acoustic_v31
  say "S7 train bắt đầu"
  if [ -f runs/acoustic_v31/last.pt ]; then
    /bin/bash tools/watchdog_resume_train.sh runs/acoustic_v31 configs/small_codelung_v31.yaml \
      "$TRAIN_M" "$VAL_M" artifacts/dataset_v2/train_v31.log 6 >> "$LOG" 2>&1
  else
    .venv/bin/vifasttts train-v2 --config configs/small_codelung_v31.yaml --train-manifest "$TRAIN_M" \
      --val-manifest "$VAL_M" --output runs/acoustic_v31 >> "$LOG" 2>&1
  fi
  touch $M/v31_s7.ok; say "S7 xong ($(tail -1 runs/acoustic_v31/history.csv | cut -d, -f1) bước)"
fi
# S8 chấm
bench() {
  [ -f "artifacts/bench/$2.json" ] && return 0
  say "bench $2"
  local cmd=($PY tools/bench_checkpoint.py --run runs/acoustic_v31 --checkpoint "$1" --tag "$2" --device mps
             --val-manifest "$3" --val-manifest-old "$3" --holdout-manifest "$4")
  if [ -n "${5:-}" ]; then cmd+=(--baseline-old "$5" --baseline-fixgap "$5"); fi
  "${cmd[@]}" >> "$LOG" 2>&1 || say "bench $2 LỖI"
}
if [ ! -f $M/v31_s8.ok ]; then
  best=runs/acoustic_v31/best.pt; [ -f "$best" ] || best=runs/acoustic_v31/last.pt
  bench "$best" v31_best "$VAL_M" "$HOLD_M"
  bench runs/acoustic_v31/last.pt v31_last "$VAL_M" "$HOLD_M"
  bench "$best" v31_onfilt data/codelung/val_clean_v2_dur_seg_slot4_filt.jsonl data/codelung/holdout_clean_v2_dur_slot4_filt.jsonl eval/baselines/b8_filt.json
  if [ ! -f artifacts/eval_acoustic/v31_cer100.json ]; then
    $PY tools/eval_acoustic.py --config runs/acoustic_v31/resolved_config.json --checkpoint "$best" \
      --mode free --manifest "$HOLD_M" --cache-dir cache/codelung_v31 --clips 100 --device mps \
      --out artifacts/eval_acoustic/v31_cer100.json >> "$LOG" 2>&1 || say "CER-100 LỖI"
  fi
  if [ ! -f artifacts/listen_small_filt/v31/index.html ]; then
    mkdir -p artifacts/listen_small_filt/v31_ckpts
    cp "$best" artifacts/listen_small_filt/v31_ckpts/v31.pt
    $PY tools/listen_demo.py --config runs/acoustic_v31/resolved_config.json \
      --checkpoint runs/acoustic_v2_b8/best.pt \
      --checkpoint artifacts/listen_small_filt/v31_ckpts/v31.pt \
      --manifest artifacts/demo_small_filt/manifest.jsonl --clips 4 --sharpen-alpha 1.5 \
      --minimal-pairs artifacts/demo_small_filt/minimal_pairs.txt --device mps \
      --out artifacts/listen_small_filt/v31 >> "$LOG" 2>&1 || say "demo LỖI"
  fi
  touch $M/v31_s8.ok; say "S8 xong"
fi
# S9 báo cáo
$PY tools/night_report.py --out docs/handoff/DATA_V31_20_10.md --baseline b8_filt \
  --row "b8 trên val v31|b8_onv31|b8v31|runs/acoustic_v2_b8" \
  --row "v31 best|v31_best|v31|runs/acoustic_v31" \
  --row "v31 last|v31_last|v31|runs/acoustic_v31" \
  --row "v31 trên val_filt (cũ)|v31_onfilt|v31|runs/acoustic_v31" >> "$LOG" 2>&1 || say "report LỖI"
date '+%F %H:%M:%S' > $M/v31_done.marker
say "===== DONE ====="
launchctl remove vifasttts.v31 2>/dev/null || true
