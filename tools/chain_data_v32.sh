#!/bin/bash
# tools/chain_data_v32.sh — Data v3.2: audio/cache b8 NGUYÊN BẢN + cắt segment tại gap >0,64 s.
#
# Không sửa 1 mẫu audio nào (khác v3.1): pause dài trở thành RANH GIỚI segment; trong mỗi row
# không còn gap > 60 frame. Pause lúc đọc do runtime quyết định (chế độ steady: phẩy 0,25 s ·
# chấm 0,55 s). Cache dùng lại `cache/codelung_v2` (khoá theo audio, audio không đổi).
#
#   launchctl submit -l vifasttts.v32 -- /bin/bash "$PWD/tools/chain_data_v32.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_data_v32.log
V=data/codelung/v32
M=artifacts/dataset_v2
PY=.venv/bin/python
say() { echo "[v32] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }
TRAIN_M=$V/train_clean_v2_dur_seg_slot4.jsonl
VAL_M=$V/val_clean_v2_dur_seg_slot4.jsonl
HOLD_M=$V/holdout_clean_v2_dur_seg_slot4.jsonl
CFG=configs/small_codelung_v32.yaml

say "===== bắt đầu ====="

# S1 — dữ liệu: prep clip+seg4 → cắt tại gap >60 → ghép seg4 (idempotent)
if [ ! -f $M/v32_s1.ok ]; then
  mkdir -p $V
  $PY tools/v32_prep_clip_manifest.py --clip-manifest data/codelung/train_clean_v2_dur.jsonl \
    --seg-manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
    --out $V/train_clean_v2_dur.jsonl >> "$LOG" 2>&1 || { say "S1 prep train LỖI"; exit 1; }
  $PY tools/v32_prep_clip_manifest.py --clip-manifest data/codelung/val_clean_v2_dur.jsonl \
    --seg-manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
    --out $V/val_clean_v2_dur.jsonl >> "$LOG" 2>&1 || { say "S1 prep val LỖI"; exit 1; }
  $PY tools/v32_prep_clip_manifest.py --clip-manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl \
    --holdout --out $V/holdout_clean_v2_dur.jsonl >> "$LOG" 2>&1 || { say "S1 prep holdout LỖI"; exit 1; }
  $PY tools/cut_segments.py --manifest $V/train_clean_v2_dur.jsonl \
    --manifest $V/val_clean_v2_dur.jsonl --manifest $V/holdout_clean_v2_dur.jsonl \
    --out-dir $V --max-gap-frames 60 >> "$LOG" 2>&1 || { say "S1 cut LỖI"; exit 1; }
  $PY tools/v3_segments_slot4.py \
    --clip-manifests $V/train_clean_v2_dur.jsonl,$V/val_clean_v2_dur.jsonl,$V/holdout_clean_v2_dur.jsonl \
    --seg-dir $V >> "$LOG" 2>&1 || { say "S1 seg4 LỖI"; exit 1; }
  # kiểm bất biến + trần gap (fail-fast trước khi train 5 h)
  $PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "S1 audit TRƯỢT"; exit 1; }
import json
from pathlib import Path
for name in ("train", "val", "holdout"):
    rows = [json.loads(l) for l in Path(f"data/codelung/v32/{name}_clean_v2_dur_seg_slot4.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        assert r["frame_range"][1] - r["frame_range"][0] == sum(r["durations"]) + sum(r["gaps"]), r["id"]
        assert max(r["gaps"], default=0) <= 60, r["id"]
        assert len(r["seg_durations"]) == len(r["durations"]), r["id"]
    print(f"[v32-audit] {name}: {len(rows)} segment · gap max 60 · T==Σdur+Σgap OK")
PYEOF
  touch $M/v32_s1.ok; say "S1 dữ liệu xong"
fi

# S2 — train từ đầu 60k (patience 0; tự resume nếu có last.pt)
if [ ! -f $M/v32_s2.ok ]; then
  mkdir -p runs/acoustic_v32
  say "S2 train bắt đầu"
  if [ -f runs/acoustic_v32/last.pt ]; then
    /bin/bash tools/watchdog_resume_train.sh runs/acoustic_v32 $CFG "$TRAIN_M" "$VAL_M" \
      artifacts/dataset_v2/train_v32.log 6 >> "$LOG" 2>&1
  else
    .venv/bin/vifasttts train-v2 --config $CFG --train-manifest "$TRAIN_M" \
      --val-manifest "$VAL_M" --output runs/acoustic_v32 >> "$LOG" 2>&1
  fi
  touch $M/v32_s2.ok; say "S2 train xong ($(tail -1 runs/acoustic_v32/history.csv | cut -d, -f1) bước)"
fi

# S3 — chấm: v32 (best/last) + b8 trên CÙNG val v32 + CER-100 + demo steady
if [ ! -f $M/v32_s3.ok ]; then
  best=runs/acoustic_v32/best.pt; [ -f "$best" ] || best=runs/acoustic_v32/last.pt
  for spec in "$best|v32_best|runs/acoustic_v32" \
              "runs/acoustic_v32/last.pt|v32_last|runs/acoustic_v32" \
              "runs/acoustic_v2_b8/best.pt|b8_on_v32val|runs/acoustic_v32"; do
    ckpt=${spec%%|*}; tag=$(echo "$spec" | cut -d'|' -f2); run=$(echo "$spec" | cut -d'|' -f3)
    [ -f "artifacts/bench/$tag.json" ] && continue
    say "bench $tag"
    $PY tools/bench_checkpoint.py --run "$run" --checkpoint "$ckpt" --tag "$tag" --clips 300 \
      --val-manifest "$VAL_M" --val-manifest-old "$VAL_M" --holdout-manifest "$HOLD_M" \
      --holdout 40 --tone-clips 80 --creak-clips 40 --device mps --no-cer >> "$LOG" 2>&1 \
      || say "bench $tag LỖI"
  done
  if [ ! -f artifacts/eval_acoustic/v32_cer100.json ]; then
    $PY tools/eval_acoustic.py --config runs/acoustic_v32/resolved_config.json --checkpoint "$best" \
      --mode free --manifest "$HOLD_M" --cache-dir cache/codelung_v2 --clips 100 --device mps \
      --out artifacts/eval_acoustic/v32_cer100.json >> "$LOG" 2>&1 || say "CER-100 LỖI"
  fi
  for variant in "b8:runs/acoustic_v2_b8/resolved_config.json:runs/acoustic_v2_b8/best.pt" \
                 "v32:runs/acoustic_v32/resolved_config.json:$best"; do
    name=${variant%%:*}; rest=${variant#*:}
    cfg=${rest%%:*}; ckpt=${rest#*:}
    [ -f artifacts/listen_small_filt/steady_$name/index.html ] && continue
    $PY tools/steady_demo.py --config "$cfg" --checkpoint "$ckpt" --device mps \
      --out artifacts/listen_small_filt/steady_$name >> "$LOG" 2>&1 || say "steady demo $name LỖI"
  done
  touch $M/v32_s3.ok; say "S3 xong"
fi

$PY tools/night_report.py --out docs/handoff/DATA_V32_21_10.md --baseline b8_filt \
  --row "b8 trên val v32|b8_on_v32val|b8v32|runs/acoustic_v2_b8" \
  --row "v32 best|v32_best|v32|runs/acoustic_v32" \
  --row "v32 last|v32_last|v32|runs/acoustic_v32" >> "$LOG" 2>&1 || say "report LỖI"
date '+%F %H:%M:%S' > $M/v32_done.marker
say "===== DONE ====="
launchctl remove vifasttts.v32 2>/dev/null || true
