#!/bin/bash
# tools/chain_data_v33.sh — v3.3: DFN TOÀN BỘ audio + cắt segment tại gap >0,64 s (KHÔNG sửa audio).
#
# Mục đích: phép thử DFN **sạch** — v3 thất bại vì khâu NỐI audio, chưa từng thử DFN với cách cắt
# segment đúng (v3.2). Ở đây audio = `data/codelung/v3/denoised/*.wav` (DFN đã chạy, cùng độ dài,
# cùng mức −26 dBFS), timeline giữ nguyên, chỉ đổi chỗ cắt segment. Chờ v3.2 chấm xong mới chạy
# (tránh tranh MPS với training khác).
#
#   launchctl submit -l vifasttts.v33 -- /bin/bash "$PWD/tools/chain_data_v33.sh"
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=artifacts/dataset_v2/chain_data_v33.log
V=data/codelung/v33
M=artifacts/dataset_v2
PY=.venv/bin/python
say() { echo "[v33] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }
TRAIN_M=$V/train_clean_v2_dur_seg_slot4.jsonl
VAL_M=$V/val_clean_v2_dur_seg_slot4.jsonl
HOLD_M=$V/holdout_clean_v2_dur_seg_slot4.jsonl
CFG=configs/small_codelung_v33.yaml

say "===== bắt đầu (chờ v3.2 chấm xong) ====="
while [ ! -f $M/v32_s3.ok ]; do sleep 120; done
say "v3.2 đã chấm xong — bắt đầu v3.3"

# S1 — manifest clip-level trỏ audio DFN + cắt segment + seg4 (không sửa audio)
if [ ! -f $M/v33_s1.ok ]; then
  mkdir -p $V
  $PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "S1 prep LỖI"; exit 1; }
import json
from pathlib import Path
for split in ("train", "val", "holdout"):
    src = Path(f"data/codelung/v32/{split}_clean_v2_dur.jsonl")
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        name = Path(str(r["audio_path"])).name
        denoised = Path("data/codelung/v3/denoised") / name
        if not denoised.exists():
            continue
        r = dict(r)
        r["audio_path"] = f"data/codelung/v3/denoised/{name}"
        out.append(r)
    dst = Path(f"data/codelung/v33/{split}_clean_v2_dur.jsonl")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"[v33] {split}: {len(rows)} → {len(out)} clip (audio DFN)")
PYEOF
  $PY tools/cut_segments.py --manifest $V/train_clean_v2_dur.jsonl \
    --manifest $V/val_clean_v2_dur.jsonl --manifest $V/holdout_clean_v2_dur.jsonl \
    --out-dir $V --max-gap-frames 60 >> "$LOG" 2>&1 || { say "S1 cut LỖI"; exit 1; }
  $PY tools/v3_segments_slot4.py \
    --clip-manifests $V/train_clean_v2_dur.jsonl,$V/val_clean_v2_dur.jsonl,$V/holdout_clean_v2_dur.jsonl \
    --seg-dir $V >> "$LOG" 2>&1 || { say "S1 seg4 LỖI"; exit 1; }
  $PY - <<'PYEOF' >> "$LOG" 2>&1 || { say "S1 audit TRƯỢT"; exit 1; }
import json
from pathlib import Path
for name in ("train", "val", "holdout"):
    rows = [json.loads(l) for l in Path(f"data/codelung/v33/{name}_clean_v2_dur_seg_slot4.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        assert r["frame_range"][1] - r["frame_range"][0] == sum(r["durations"]) + sum(r["gaps"]), r["id"]
        assert max(r["gaps"], default=0) <= 60, r["id"]
        assert "v3/denoised" in r["audio_path"], r["id"]
    print(f"[v33-audit] {name}: {len(rows)} segment · gap ≤60 · T==Σdur+Σgap OK · audio DFN OK")
PYEOF
  touch $M/v33_s1.ok; say "S1 dữ liệu xong"
fi

# S2 — cache v33 (audio DFN)
for spec in "train|$TRAIN_M" "val|$VAL_M" "holdout|$HOLD_M"; do
  name=${spec%%|*}; man=${spec#*|}
  if [ ! -f $M/v33_cache_$name.ok ]; then
    .venv/bin/vifasttts cache-features --manifest "$man" --config $CFG \
      --cache-dir cache/codelung_v33 --workers 6 >> "$LOG" 2>&1 || { say "S2 cache $name LỖI"; exit 1; }
    touch $M/v33_cache_$name.ok
  fi
done
say "S2 cache xong"

# S3 — train 60k (patience 0; resume nếu có last.pt)
if [ ! -f $M/v33_s3.ok ]; then
  mkdir -p runs/acoustic_v33
  say "S3 train bắt đầu"
  if [ -f runs/acoustic_v33/last.pt ]; then
    /bin/bash tools/watchdog_resume_train.sh runs/acoustic_v33 $CFG "$TRAIN_M" "$VAL_M" \
      artifacts/dataset_v2/train_v33.log 6 >> "$LOG" 2>&1
  else
    .venv/bin/vifasttts train-v2 --config $CFG --train-manifest "$TRAIN_M" \
      --val-manifest "$VAL_M" --output runs/acoustic_v33 >> "$LOG" 2>&1
  fi
  touch $M/v33_s3.ok; say "S3 train xong ($(tail -1 runs/acoustic_v33/history.csv | cut -d, -f1) bước)"
fi

# S4 — chấm: v33 (best/last) + b8 trên CÙNG val v33 + coda + CER-100 + demo steady
if [ ! -f $M/v33_s4.ok ]; then
  best=runs/acoustic_v33/best.pt; [ -f "$best" ] || best=runs/acoustic_v33/last.pt
  for spec in "$best|v33_best|runs/acoustic_v33" \
              "runs/acoustic_v33/last.pt|v33_last|runs/acoustic_v33" \
              "runs/acoustic_v2_b8/best.pt|b8_on_v33val|runs/acoustic_v33"; do
    ckpt=${spec%%|*}; tag=$(echo "$spec" | cut -d'|' -f2); run=$(echo "$spec" | cut -d'|' -f3)
    [ -f "artifacts/bench/$tag.json" ] && continue
    say "bench $tag"
    $PY tools/bench_checkpoint.py --run "$run" --checkpoint "$ckpt" --tag "$tag" --clips 300 \
      --val-manifest "$VAL_M" --val-manifest-old "$VAL_M" --holdout-manifest "$HOLD_M" \
      --holdout 40 --tone-clips 80 --creak-clips 40 --device mps --no-cer >> "$LOG" 2>&1 \
      || say "bench $tag LỖI"
  done
  for tag in v33_best b8_v33; do
    if [ "$tag" = "v33_best" ]; then ckpt=$best; out=artifacts/eval_acoustic/v33_coda.json
    else ckpt=runs/acoustic_v2_b8/best.pt; out=artifacts/eval_acoustic/b8_v33_coda.json; fi
    [ -f "$out" ] && continue
    $PY tools/coda_realization.py --config runs/acoustic_v33/resolved_config.json --checkpoint "$ckpt" \
      --manifest "$VAL_M" --cache-dir cache/codelung_v33 --clips 20 --device mps --out "$out" \
      >> "$LOG" 2>&1 || say "coda $tag LỖI"
  done
  if [ ! -f artifacts/eval_acoustic/v33_cer100.json ]; then
    $PY tools/eval_acoustic.py --config runs/acoustic_v33/resolved_config.json --checkpoint "$best" \
      --mode free --manifest "$HOLD_M" --cache-dir cache/codelung_v33 --clips 100 --device mps \
      --out artifacts/eval_acoustic/v33_cer100.json >> "$LOG" 2>&1 || say "CER-100 LỖI"
  fi
  if [ ! -f artifacts/listen_small_filt/steady_v33/index.html ]; then
    $PY tools/steady_demo.py --config runs/acoustic_v33/resolved_config.json --checkpoint "$best" \
      --device mps --out artifacts/listen_small_filt/steady_v33 >> "$LOG" 2>&1 || say "demo LỖI"
  fi
  touch $M/v33_s4.ok; say "S4 xong"
fi

$PY tools/night_report.py --out docs/handoff/DATA_V33_DFN_21_10.md --baseline b8_filt \
  --row "b8 trên val v33 (DFN)|b8_on_v33val|b8v33|runs/acoustic_v2_b8" \
  --row "v33 best|v33_best|v33|runs/acoustic_v33" \
  --row "v33 last|v33_last|v33|runs/acoustic_v33" >> "$LOG" 2>&1 || say "report LỖI"
date '+%F %H:%M:%S' > $M/v33_done.marker
say "===== DONE ====="
launchctl remove vifasttts.v33 2>/dev/null || true
