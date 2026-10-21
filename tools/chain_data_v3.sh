#!/bin/bash
# tools/chain_data_v3.sh — Data v3 end-to-end (19/10): denoise → build → cut+slot4 → QC → cache → train → chấm.
#
# Chạy qua launchd (idempotent; chết thì chạy lại tự tiếp):
#   launchctl submit -l vifasttts.v3 -- /bin/bash "$PWD/tools/chain_data_v3.sh"
# Gỡ khi xong: launchctl remove vifasttts.v3
#
# Các bước (marker trong artifacts/dataset_v2/v3_*.ok):
#   S1 chờ DFN denoise (job vifasttts.v3dn) xong → chạy bù (idempotent)
#   S2 build clip v3 (normalize + cap 0,7s + seg4)
#   S3 cut segment 3–10s → S4 ghép seg_durations (slot4)
#   S5 QC + cổng (nền nhiễu trung vị < −65 dB, clipping ~0)
#   S6 cache v3 (train/val/holdout) → S7 TRAIN TỪ ĐẦU 0→60k (watchdog)
#   S8 chấm: bench v3 trên val v3 (mốc so = b8 chấm cùng val v3) + val_filt cũ + CER-100 + demo /v3/
#   S9 tự gỡ job
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_data_v3.log
V3=data/codelung/v3
M=artifacts/dataset_v2
say() { echo "[v3] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }
PY=.venv/bin/python
DFNPY=$HOME/dvchd/.dfn-venv/bin/python
TRAIN_M=$V3/train_clean_v2_dur_seg_slot4.jsonl
VAL_M=$V3/val_clean_v2_dur_seg_slot4.jsonl
HOLD_M=$V3/holdout_clean_v2_dur.jsonl

say "===== bắt đầu ====="

# S1 — denoise
if [ ! -f $M/v3_s1.ok ]; then
  while pgrep -f "v3_denoise_clips" >/dev/null 2>&1; do sleep 60; done
  launchctl remove vifasttts.v3dn 2>/dev/null || true
  $DFNPY tools/v3_denoise_clips.py \
    --manifests data/codelung/train_clean_v2_dur.jsonl,data/codelung/val_clean_v2_dur.jsonl,data/codelung/holdout_clean_v2_dur.jsonl \
    --out-dir $V3/denoised --workers 4 >> "$LOG" 2>&1 || { say "S1 LỖI"; exit 1; }
  touch $M/v3_s1.ok; say "S1 xong ($(ls $V3/denoised | wc -l | tr -d ' ') clip)"
fi

# S2 — build clip v3
if [ ! -f $M/v3_s2.ok ]; then
  $PY tools/v3_build_clips.py \
    --manifests data/codelung/train_clean_v2_dur.jsonl,data/codelung/val_clean_v2_dur.jsonl,data/codelung/holdout_clean_v2_dur.jsonl \
    --old-seg data/codelung/train_clean_v2_dur_seg_slot4.jsonl,data/codelung/val_clean_v2_dur_seg_slot4.jsonl,data/codelung/holdout_clean_v2_dur_slot4.jsonl \
    --denoised-dir $V3/denoised --out-dir $V3 >> "$LOG" 2>&1 || { say "S2 LỖI"; exit 1; }
  touch $M/v3_s2.ok; say "S2 xong"
fi

# S3 + S4 — cắt segment + slot4
if [ ! -f $M/v3_s3.ok ]; then
  $PY tools/cut_segments.py --manifest $V3/train_clean_v2_dur.jsonl --manifest $V3/val_clean_v2_dur.jsonl \
    --out-dir $V3 >> "$LOG" 2>&1 || { say "S3 LỖI"; exit 1; }
  $PY tools/v3_segments_slot4.py \
    --clip-manifests $V3/train_clean_v2_dur.jsonl,$V3/val_clean_v2_dur.jsonl \
    --seg-dir $V3 >> "$LOG" 2>&1 || { say "S4 LỖI"; exit 1; }
  touch $M/v3_s3.ok; say "S3+S4 xong"
fi

# S5 — QC + cổng
if [ ! -f $M/v3_s5.ok ]; then
  $PY tools/audit_clip_quality.py $TRAIN_M $VAL_M $HOLD_M --report artifacts/clip_qc/v3_report.json --workers 8 >> "$LOG" 2>&1 || { say "S5 QC LỖI"; exit 1; }
  $PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "S5 CỔNG TRƯỢT — DỪNG"; exit 1; }
import json, statistics
from pathlib import Path
d = json.loads(Path("artifacts/clip_qc/v3_report.json").read_text())
rows = [r for r in d["rows"].values() if isinstance(r, dict) and "noise_floor_db" in r]
nf = statistics.median([r["noise_floor_db"] for r in rows])
clip = max(r["clip_frac"] for r in rows)
snr = statistics.median([r["snr_db"] for r in rows])
print(f"[v3-qc] clip={len(rows)} · nền p50={nf:.1f} dB · SNR p50={snr:.1f} · clip_frac max={clip:.4f}")
assert nf < -65 and clip < 0.01, "QC trượt cổng"
PYEOF
  touch $M/v3_s5.ok; say "S5 xong"
fi

# S6 — cache v3
for spec in "train|$TRAIN_M" "val|$VAL_M" "holdout|$HOLD_M"; do
  name=${spec%%|*}; man=${spec#*|}
  if [ ! -f $M/v3_cache_$name.ok ]; then
    .venv/bin/vifasttts cache-features --manifest "$man" --config configs/small_codelung_v3.yaml \
      --cache-dir cache/codelung_v3 --workers 6 >> "$LOG" 2>&1 || { say "S6 cache $name LỖI"; exit 1; }
    touch $M/v3_cache_$name.ok
  fi
done
say "S6 xong (cache v3)"

# S7 — train từ đầu (mới → chạy trực tiếp; đã có last.pt (chết giữa) → watchdog resume)
if [ ! -f $M/v3_s7.ok ]; then
  mkdir -p runs/acoustic_v3_clean
  say "S7 train bắt đầu"
  if [ -f runs/acoustic_v3_clean/last.pt ]; then
    /bin/bash tools/watchdog_resume_train.sh runs/acoustic_v3_clean configs/small_codelung_v3.yaml \
      "$TRAIN_M" "$VAL_M" artifacts/dataset_v2/train_v3_clean.log 6 >> "$LOG" 2>&1
  else
    .venv/bin/vifasttts train-v2 --config configs/small_codelung_v3.yaml \
      --train-manifest "$TRAIN_M" --val-manifest "$VAL_M" \
      --output runs/acoustic_v3_clean >> "$LOG" 2>&1
  fi
  touch $M/v3_s7.ok; say "S7 xong ($(tail -1 runs/acoustic_v3_clean/history.csv | cut -d, -f1) bước)"
fi

# S8 — chấm: mốc b8 trên val v3 + v3 trên val v3 + v3 trên val_filt + CER-100 + demo
bench() {  # $1=run $2=ckpt $3=tag $4=val $5=holdout [$6=baseline]
  [ -f "artifacts/bench/$3.json" ] && return 0
  say "bench $3"
  local cmd=($PY tools/bench_checkpoint.py --run "$1" --checkpoint "$2" --tag "$3" --device mps
             --val-manifest "$4" --val-manifest-old "$4" --holdout-manifest "$5")
  if [ -n "${6:-}" ]; then
    cmd+=(--baseline-old "$6" --baseline-fixgap "$6")
  fi
  "${cmd[@]}" >> "$LOG" 2>&1 || say "bench $3 LỖI"
}
if [ ! -f $M/v3_s8.ok ]; then
  best=runs/acoustic_v3_clean/best.pt; [ -f "$best" ] || best=runs/acoustic_v3_clean/last.pt
  bench runs/acoustic_v2_b8 runs/acoustic_v2_b8/best.pt b8_onv3 "$VAL_M" "$HOLD_M"
  bench runs/acoustic_v3_clean "$best" v3_best "$VAL_M" "$HOLD_M"
  bench runs/acoustic_v3_clean runs/acoustic_v3_clean/last.pt v3_last "$VAL_M" "$HOLD_M"
  bench runs/acoustic_v3_clean "$best" v3_onfilt data/codelung/val_clean_v2_dur_seg_slot4_filt.jsonl data/codelung/holdout_clean_v2_dur_slot4_filt.jsonl eval/baselines/b8_filt.json
  if [ ! -f artifacts/eval_acoustic/v3_cer100.json ]; then
    $PY tools/eval_acoustic.py --config runs/acoustic_v3_clean/resolved_config.json --checkpoint "$best" \
      --mode free --manifest "$HOLD_M" --cache-dir cache/codelung_v3 --clips 100 --device mps \
      --out artifacts/eval_acoustic/v3_cer100.json >> "$LOG" 2>&1 || say "CER-100 LỖI"
  fi
  if [ ! -f artifacts/creak_rarity/probe_v3.json ]; then
    $PY tools/creak_rarity_probe.py --config runs/acoustic_v3_clean/resolved_config.json --checkpoint "$best" \
      --manifest "$TRAIN_M" --out artifacts/creak_rarity/probe_v3.json --device cpu >> "$LOG" 2>&1 || say "probe LỖI"
  fi
  if [ ! -f artifacts/listen_small_filt/v3/index.html ]; then
    mkdir -p artifacts/listen_small_filt/v3_ckpts
    cp "$best" artifacts/listen_small_filt/v3_ckpts/v3.pt
    $PY tools/listen_demo.py --config runs/acoustic_v3_clean/resolved_config.json \
      --checkpoint artifacts/listen_small_filt/v3_ckpts/v3.pt \
      --manifest artifacts/demo_small_filt/manifest.jsonl --clips 4 --sharpen-alpha 1.5 \
      --minimal-pairs artifacts/demo_small_filt/minimal_pairs.txt --device mps \
      --out artifacts/listen_small_filt/v3 >> "$LOG" 2>&1 || say "demo LỖI"
  fi
  touch $M/v3_s8.ok; say "S8 xong"
fi

# S9 — báo cáo + kết thúc
$PY tools/night_report.py --out docs/handoff/DATA_V3_19_10.md --baseline b8_filt \
  --row "b8 trên val v3|b8_onv3|b8v3|runs/acoustic_v2_b8" \
  --row "v3 best|v3_best|v3|runs/acoustic_v3_clean" \
  --row "v3 last|v3_last|v3|runs/acoustic_v3_clean" \
  --row "v3 trên val_filt (cũ)|v3_onfilt|v3|runs/acoustic_v3_clean" >> "$LOG" 2>&1 || say "report LỖI"
date '+%F %H:%M:%S' > $M/v3_done.marker
say "===== DONE ====="
launchctl remove vifasttts.v3 2>/dev/null || true
