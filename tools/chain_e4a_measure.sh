#!/usr/bin/env bash
# Chuỗi đo E4a qua đêm: đợi train xong → creak (CPU) → tone paths (CPU) → bench E4a (MPS)
# → bench lại base40k "fresh" (MPS, cùng thước/thiết bị) → soát bigram dữ liệu (CPU).
# Chạy: nohup bash tools/chain_e4a_measure.sh <pid_train> > artifacts/e4a_measure.log 2>&1 &
set -u
PID=${1:?pid train}
cd "$(dirname "$0")/.."
LOG=artifacts/e4a_measure.log
say() { echo "[chain $(date '+%H:%M')] $*" | tee -a "$LOG"; }

say "đợi tiến trình $PID kết thúc…"
while ps -p "$PID" >/dev/null 2>&1; do sleep 60; done
sleep 15
say "train xong — bắt đầu đo"

CFG=runs/acoustic_v2_E4a/resolved_config.json
CKPT=runs/acoustic_v2_E4a/last.pt
B=eval/baselines

say "1/5 creak (eval_creak + free-run + CreaPy)"
bash tools/measure_creak.sh E4a_50k "$CFG" "$CKPT" cpu 40 >> "$LOG" 2>&1 || say "⚠ creak lỗi"

say "2/5 tone paths (60 clip, CPU)"
.venv/bin/python tools/eval_tone_paths.py --config "$CFG" --checkpoint "$CKPT" --clips 60 \
  --out artifacts/tone_path/abc_E4a_50k.json >> "$LOG" 2>&1 || say "⚠ tone paths lỗi"

say "3/5 bench E4a@50000 (MPS, 300 clip)"
.venv/bin/python tools/bench_checkpoint.py --config "$CFG" --checkpoint "$CKPT" \
  --tag E4a_50000 --step 50000 --device mps --clips 300 \
  --val-manifest data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl \
  --holdout-manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl \
  --baseline-old "$B/base40k_old.json" --baseline-fixgap "$B/base40k_fixtail.json" \
  --out-dir artifacts/bench >> "$LOG" 2>&1 || say "⚠ bench E4a lỗi"

say "4/5 bench lại base40k (MPS) — so cùng thước, kiểm luôn độ nhạy thiết bị"
.venv/bin/python tools/bench_checkpoint.py --config runs/acoustic_v2_base/resolved_config.json \
  --checkpoint runs/acoustic_v2_base/last.pt --tag base40k_fresh --step 40000 --device mps \
  --clips 300 --val-manifest data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl \
  --holdout-manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl \
  --baseline-old "$B/base40k_old.json" --baseline-fixgap "$B/base40k_fixtail.json" \
  --out-dir artifacts/bench >> "$LOG" 2>&1 || say "⚠ bench base lỗi"

say "5/5 soát bigram dữ liệu train (CPU)"
.venv/bin/python tools/audit_bigrams.py --manifest data/codelung/train_clean_v2.jsonl \
  --out artifacts/dataset_v2/audit_bigrams.json >> "$LOG" 2>&1 || say "⚠ audit lỗi"

say "XONG chuỗi đo"
