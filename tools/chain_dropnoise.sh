#!/bin/bash
# tools/chain_dropnoise.sh — chờ run "loại video nhiễu" xong → tự chấm + demo + báo cáo.
#
# Run: `runs/acoustic_v2_b8_dropnoise` (train bỏ 448 row video 4kyqfOypVWI, resume b8@46k→70k).
# Chuỗi: chờ 70k → gỡ job run → bench best+last (mốc b8_filt) → CER-100 → probe đuôi hiếm →
# demo nghe (b8 vs run này) → báo cáo docs/handoff/DROPNOISE_19_10.md → tự gỡ job.
# Submit: launchctl submit -l vifasttts.chain_dn -- /bin/bash "$PWD/tools/chain_dropnoise.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_dropnoise.log
RUN=runs/acoustic_v2_b8_dropnoise
say() { echo "[chain-dn] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }

step_of() {
  .venv/bin/python - "$1/last.pt" 2>/dev/null <<'PY' || echo 0
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))
except Exception:
    print(0)
PY
}
ckpt_step() {
  .venv/bin/python - "$1" 2>/dev/null <<'PY' || echo 0
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))
except Exception:
    print(0)
PY
}

say "bắt đầu chờ dropnoise"
waited=0
while [ "$(step_of $RUN)" -lt 70000 ] && [ "$waited" -lt 300 ]; do sleep 60; waited=$((waited + 1)); done
launchctl remove vifasttts.dropnoise 2>/dev/null || true
say "dropnoise: step $(step_of $RUN)"

best="$RUN/best.pt"
[ -f "$best" ] || best="$RUN/last.pt"
for spec in "dn_best|$best" "dn_last|$RUN/last.pt"; do
  tag=${spec%%|*}; ck=${spec#*|}
  if [ ! -f "artifacts/bench/$tag.json" ]; then
    say "bench $tag"
    .venv/bin/python tools/bench_checkpoint.py --run $RUN --checkpoint "$ck" --tag "$tag" \
      --step "$(ckpt_step "$ck")" --device mps \
      --val-manifest data/codelung/val_clean_v2_dur_seg_slot4_filt.jsonl \
      --val-manifest-old data/codelung/val_clean_v2_dur_seg_slot4_filt.jsonl \
      --holdout-manifest data/codelung/holdout_clean_v2_dur_slot4_filt.jsonl \
      --baseline-old eval/baselines/b8_filt.json --baseline-fixgap eval/baselines/b8_filt.json \
      >> "$LOG" 2>&1 || say "bench $tag LỖI"
  fi
done

if [ ! -f artifacts/eval_acoustic/dn_cer100.json ]; then
  say "CER-100 dn"
  .venv/bin/python tools/eval_acoustic.py --config $RUN/resolved_config.json --checkpoint "$best" \
    --mode free --manifest data/codelung/holdout_clean_v2_dur_slot4_filt.jsonl --cache-dir cache/codelung_v2 \
    --clips 100 --device mps --out artifacts/eval_acoustic/dn_cer100.json >> "$LOG" 2>&1 || say "CER-100 LỖI"
fi

if [ ! -f artifacts/creak_rarity/probe_dn.json ]; then
  say "rarity probe dn"
  .venv/bin/python tools/creak_rarity_probe.py --config $RUN/resolved_config.json --checkpoint "$best" \
    --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl --out artifacts/creak_rarity/probe_dn.json \
    --device cpu >> "$LOG" 2>&1 || say "probe LỖI"
fi

if [ ! -f artifacts/listen_small_filt/dropnoise/index.html ]; then
  say "demo dropnoise"
  mkdir -p artifacts/listen_small_filt/dropnoise_ckpts
  cp "$best" artifacts/listen_small_filt/dropnoise_ckpts/dn.pt
  .venv/bin/python tools/listen_demo.py --config runs/acoustic_v2_b8/resolved_config.json \
    --checkpoint runs/acoustic_v2_b8/best.pt \
    --checkpoint artifacts/listen_small_filt/dropnoise_ckpts/dn.pt \
    --manifest artifacts/demo_small_filt/manifest.jsonl --clips 4 --sharpen-alpha 1.5 \
    --minimal-pairs artifacts/demo_small_filt/minimal_pairs.txt --device mps \
    --out artifacts/listen_small_filt/dropnoise >> "$LOG" 2>&1 || say "demo LỖI"
fi

say "report"
.venv/bin/python tools/night_report.py --out docs/handoff/DROPNOISE_19_10.md --baseline b8_filt \
  --row "loại video nhiễu best|dn_best|dn|runs/acoustic_v2_b8_dropnoise" \
  --row "loại video nhiễu last|dn_last|dn|runs/acoustic_v2_b8_dropnoise" \
  --row "mel_sharp best (tham chiếu)|msh_best|msh|runs/acoustic_v2_b8_melsharp" \
  --row "oversample (tham chiếu)|os_best|os|runs/acoustic_v2_b8_oversample" \
  >> "$LOG" 2>&1 || say "report LỖI"

date '+%F %H:%M:%S' > artifacts/dataset_v2/dropnoise_done.marker
say "DONE"
launchctl remove vifasttts.chain_dn 2>/dev/null || true
