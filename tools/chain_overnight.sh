#!/usr/bin/env bash
# Chuỗi đêm V11S3 (sau E4b): đo E4b → re-bench A2 best@42k theo CỔNG MỚI (60 clip + cùng nhãn)
# → chạy E4c (mel_sharp creak boost) → đo E4c → ghi tóm tắt.
# Chạy: nohup bash tools/chain_overnight.sh > artifacts/overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG=artifacts/overnight.log
B=eval/baselines
say() { echo "[overnight $(date '+%H:%M')] $*" | tee -a "$LOG"; }

wait_step() {  # $1 = log, $2 = mốc bước — chờ tới khi log có "Đã huấn luyện v2 <step>"
  while ! grep -q "Đã huấn luyện v2 $2" "$1" 2>/dev/null; do sleep 60; done
}

measure() {  # $1 tag · $2 config · $3 ckpt
  local tag=$1 cfg=$2 ckpt=$3
  say "đo $tag: creak"
  bash tools/measure_creak.sh "$tag" "$cfg" "$ckpt" cpu 40 >> "$LOG" 2>&1 || say "⚠ creak $tag lỗi"
  say "đo $tag: tone paths"
  .venv/bin/python tools/eval_tone_paths.py --config "$cfg" --checkpoint "$ckpt" --clips 60 \
    --out "artifacts/tone_path/abc_${tag}.json" >> "$LOG" 2>&1 || say "⚠ tone paths $tag lỗi"
  say "đo $tag: bench (MPS)"
  .venv/bin/python tools/bench_checkpoint.py --config "$cfg" --checkpoint "$ckpt" \
    --tag "$tag" --step "$(echo "$tag" | grep -oE '[0-9]+$' || echo 0)" --device mps --clips 300 \
    --val-manifest data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl \
    --holdout-manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl \
    --baseline-old "$B/base40k_old.json" --baseline-fixgap "$B/base40k_fixtail.json" \
    --out-dir artifacts/bench >> "$LOG" 2>&1 || say "⚠ bench $tag lỗi"
}

say "chờ E4b xong (log train_v2_E4b.log)"
wait_step artifacts/dataset_v2/train_v2_E4b.log 50000
sleep 20
say "E4b xong — đo"
measure E4b_50000 runs/acoustic_v2_E4b/resolved_config.json runs/acoustic_v2_E4b/last.pt

say "re-bench A2 best@42k theo cổng MỚI (production candidate)"
.venv/bin/python tools/bench_checkpoint.py --config runs/acoustic_v2_A2_fixtail/resolved_config.json \
  --checkpoint runs/acoustic_v2_A2_fixtail/best.pt --tag A2_42k_regate --step 42000 --device mps \
  --clips 300 --val-manifest data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl \
  --holdout-manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl \
  --baseline-old "$B/base40k_old.json" --baseline-fixgap "$B/base40k_fixtail.json" \
  --out-dir artifacts/bench >> "$LOG" 2>&1 || say "⚠ bench A2 lỗi"

say "khởi động E4c (mel_sharp creak boost)"
INIT_RESUME=runs/acoustic_v2_A2_fixtail/best.pt INIT_RESET_VAL_STATE=1 \
  nohup tools/watch_train_v2.sh runs/acoustic_v2_E4c artifacts/dataset_v2/train_v2_E4c.log 50000 \
  configs/small_codelung_v2_E4c.yaml \
  data/codelung/train_clean_v2_dur_seg_slot4_fixtail.jsonl \
  data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl >> "$LOG" 2>&1 &

sleep 60
say "chờ E4c xong"
wait_step artifacts/dataset_v2/train_v2_E4c.log 50000
sleep 20
say "E4c xong — đo"
measure E4c_50000 runs/acoustic_v2_E4c/resolved_config.json runs/acoustic_v2_E4c/last.pt

say "XONG CHUỖI ĐÊM"
