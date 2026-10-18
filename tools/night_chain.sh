#!/bin/bash
# tools/night_chain.sh — chuỗi thí nghiệm qua đêm 18→19/10/2026 (idempotent; chạy lại an toàn qua launchd).
#
# Trình tự:
#   S0 chờ run mel_sharp (đang chạy riêng) đạt 70k → gỡ job của nó + gỡ chain cũ
#   S1 probe + run  oversample          (E4b: NGÃ×2 · NẶNG×1,5 · coda×1,5) → runs/acoustic_v2_b8_oversample
#   S2 probe + run  oversample+melsharp (kết hợp 2 đòn)                    → runs/acoustic_v2_b8_os_melsharp
#   S3 probe + run  oversample mạnh     (NGÃ×3 · NẶNG×2,5 · coda×2,5)      → runs/acoustic_v2_b8_os_strong
#   S4 bench best+last của cả 4 run (cùng mốc b8_filt trên val lọc)
#   S5 CER-100 (best) · S6 probe đuôi hiếm (best) · S7 demo nghe (b8 + 4 best)
#   S8 báo cáo markdown → docs/handoff/OVERNIGHT_18_19.md
#   S9 tự gỡ job launchd (vifasttts.night)
#
# Submit:
#   launchctl submit -l vifasttts.night -- /bin/bash "$PWD/tools/night_chain.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/night_chain.log
say() { echo "[night] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }

step_of() {
  .venv/bin/python - "$1/last.pt" 2>/dev/null <<'PY' || echo 0
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", 0)))
except Exception:
    print(0)
PY
}

wait_step() {  # $1=run dir $2=mốc $3=timeout phút
  local waited=0
  while [ "$(step_of "$1")" -lt "$2" ] && [ "$waited" -lt "$3" ]; do sleep 60; waited=$((waited + 1)); done
  [ "$(step_of "$1")" -ge "$2" ]
}

probe() {  # $1=config $2=probe log $3=probe out dir → 0 nếu đạt
  if [ -f "$2.ok" ]; then return 0; fi
  .venv/bin/vifasttts train-v2 --config "$1" \
    --train-manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
    --val-manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
    --output "$3" --resume runs/acoustic_v2_b8/best.pt --max-steps 70000 \
    --stop-at-step 46200 > "$2" 2>&1
  if [ $? -eq 0 ] && ! grep -qi "nan" "$2"; then touch "$2.ok"; return 0; fi
  say "PROBE LỖI ($1) — xem $2"
  return 1
}

run_train() {  # $1=out $2=config $3=log [cờ thêm...]
  local OUT="$1" CFG="$2" RLOG="$3"
  shift 3
  if [ "$(step_of "$OUT")" -ge 70000 ]; then say "$OUT đã đủ 70k — bỏ qua"; return 0; fi
  mkdir -p "$OUT"
  [ -f "$OUT/last.pt" ] || cp runs/acoustic_v2_b8/best.pt "$OUT/last.pt"
  say "train $OUT (từ step $(step_of "$OUT"))"
  /bin/bash tools/watchdog_resume_train.sh "$OUT" "$CFG" \
    data/codelung/train_clean_v2_dur_seg_slot4.jsonl data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
    "$RLOG" 8 "$@" >> "$LOG" 2>&1
  say "xong $OUT (step $(step_of "$OUT"))"
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

best_or_last() { if [ -f "$1/best.pt" ]; then echo "$1/best.pt"; else echo "$1/last.pt"; fi; }

bench_one() {  # $1=run $2=ckpt $3=tag
  if [ -f "artifacts/bench/$3.json" ]; then return 0; fi
  say "bench $3"
  .venv/bin/python tools/bench_checkpoint.py --run "$1" --checkpoint "$2" --tag "$3" \
    --step "$(ckpt_step "$2")" --device mps \
    --val-manifest data/codelung/val_clean_v2_dur_seg_slot4_filt.jsonl \
    --val-manifest-old data/codelung/val_clean_v2_dur_seg_slot4_filt.jsonl \
    --holdout-manifest data/codelung/holdout_clean_v2_dur_slot4_filt.jsonl \
    --baseline-old eval/baselines/b8_filt.json --baseline-fixgap eval/baselines/b8_filt.json \
    >> "$LOG" 2>&1 || say "bench $3 LỖI"
}

RUNS="msh|runs/acoustic_v2_b8_melsharp os|runs/acoustic_v2_b8_oversample osm|runs/acoustic_v2_b8_os_melsharp oss|runs/acoustic_v2_b8_os_strong"

say "===== bắt đầu ====="
# S0 — chờ mel_sharp; gỡ chain cũ (đã bị thay bởi chain này) + job melsharp
launchctl remove vifasttts.chain 2>/dev/null || true
wait_step runs/acoustic_v2_b8_melsharp 70000 240 || say "melsharp chưa đủ 70k sau 4h — vẫn tiếp tục"
launchctl remove vifasttts.melsharp 2>/dev/null || true
say "melsharp: step $(step_of runs/acoustic_v2_b8_melsharp)"

# S1..S3 — 3 run mới (probe trước, chỉ chạy khi probe đạt)
if probe configs/small_codelung_v2_oversample.yaml artifacts/dataset_v2/night_probe_os.log runs/_night_probe_os; then
  run_train runs/acoustic_v2_b8_oversample configs/small_codelung_v2_oversample.yaml \
    artifacts/dataset_v2/train_v2_b8_oversample.log --max-steps 70000
fi
if probe configs/small_codelung_v2_os_melsharp.yaml artifacts/dataset_v2/night_probe_osm.log runs/_night_probe_osm; then
  run_train runs/acoustic_v2_b8_os_melsharp configs/small_codelung_v2_os_melsharp.yaml \
    artifacts/dataset_v2/train_v2_b8_os_melsharp.log --max-steps 70000
fi
if probe configs/small_codelung_v2_oversample_strong.yaml artifacts/dataset_v2/night_probe_oss.log runs/_night_probe_oss; then
  run_train runs/acoustic_v2_b8_os_strong configs/small_codelung_v2_oversample_strong.yaml \
    artifacts/dataset_v2/train_v2_b8_os_strong.log --max-steps 70000
fi

# S4 — bench best + last cho cả 4 run
for spec in $RUNS; do
  short=${spec%%|*}; run=${spec#*|}
  best=$(best_or_last "$run")
  bench_one "$run" "$best" "${short}_best"
  bench_one "$run" "$run/last.pt" "${short}_last"
done

# S5 — CER-100 (best)
for spec in $RUNS; do
  short=${spec%%|*}; run=${spec#*|}
  if [ ! -f "artifacts/eval_acoustic/${short}_cer100.json" ]; then
    say "CER-100 $short"
    .venv/bin/python tools/eval_acoustic.py --config "$run/resolved_config.json" \
      --checkpoint "$(best_or_last "$run")" --mode free \
      --manifest data/codelung/holdout_clean_v2_dur_slot4_filt.jsonl --cache-dir cache/codelung_v2 \
      --clips 100 --device mps --out "artifacts/eval_acoustic/${short}_cer100.json" \
      >> "$LOG" 2>&1 || say "CER-100 $short LỖI"
  fi
done

# S6 — probe đuôi hiếm (best; CPU)
for spec in $RUNS; do
  short=${spec%%|*}; run=${spec#*|}
  if [ ! -f "artifacts/creak_rarity/probe_${short}.json" ]; then
    say "rarity probe $short"
    .venv/bin/python tools/creak_rarity_probe.py --config "$run/resolved_config.json" \
      --checkpoint "$(best_or_last "$run")" --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
      --out "artifacts/creak_rarity/probe_${short}.json" --device cpu >> "$LOG" 2>&1 \
      || say "rarity probe $short LỖI"
  fi
done

# S7 — demo nghe (b8 + 4 best; copy checkpoint ra tên riêng để không trùng thư mục con)
if [ ! -f artifacts/listen_small_filt/morning/index.html ]; then
  say "demo morning"
  mkdir -p artifacts/listen_small_filt/morning_ckpts
  DEMO_CK=()
  for spec in $RUNS; do
    short=${spec%%|*}; run=${spec#*|}
    cp "$(best_or_last "$run")" "artifacts/listen_small_filt/morning_ckpts/${short}.pt"
    DEMO_CK+=(--checkpoint "artifacts/listen_small_filt/morning_ckpts/${short}.pt")
  done
  .venv/bin/python tools/listen_demo.py --config runs/acoustic_v2_b8/resolved_config.json \
    --checkpoint runs/acoustic_v2_b8/best.pt "${DEMO_CK[@]}" \
    --manifest artifacts/demo_small_filt/manifest.jsonl --clips 4 --sharpen-alpha 1.5 \
    --minimal-pairs artifacts/demo_small_filt/minimal_pairs.txt --device mps \
    --out artifacts/listen_small_filt/morning >> "$LOG" 2>&1 || say "demo LỖI"
fi

# S8 — báo cáo
say "report"
.venv/bin/python tools/night_report.py --out docs/handoff/OVERNIGHT_18_19.md --baseline b8_filt \
  --row "mel_sharp best|msh_best|msh|runs/acoustic_v2_b8_melsharp" \
  --row "mel_sharp last|msh_last|msh|runs/acoustic_v2_b8_melsharp" \
  --row "oversample best|os_best|os|runs/acoustic_v2_b8_oversample" \
  --row "oversample last|os_last|os|runs/acoustic_v2_b8_oversample" \
  --row "os+melsharp best|osm_best|osm|runs/acoustic_v2_b8_os_melsharp" \
  --row "os+melsharp last|osm_last|osm|runs/acoustic_v2_b8_os_melsharp" \
  --row "os mạnh best|oss_best|oss|runs/acoustic_v2_b8_os_strong" \
  --row "os mạnh last|oss_last|oss|runs/acoustic_v2_b8_os_strong" \
  >> "$LOG" 2>&1 || say "report LỖI"

date '+%F %H:%M:%S' > artifacts/dataset_v2/night_done.marker
say "===== DONE ====="
launchctl remove vifasttts.night 2>/dev/null || true
