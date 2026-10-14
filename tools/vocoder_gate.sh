#!/usr/bin/env bash
# Chấm cổng vocoder (§45 / E4-C) trong MỘT lệnh — 3 tầng thước:
#   1) `eval_vocoder.py`     — fidelity copy-synth: mel_l1 · logf0_rmse_cents[_no_oct] · vuv_err
#   2) `voice_quality.py`    — chất giọng theo thanh: HNR/perturb (creak có bị làm mượt không)
#   3) copy-synth 25 clip ngã/nặng + `creak_rate.py` — tỉ lệ creak (mốc THẬT 1,717)
#
# Dùng: tools/vocoder_gate.sh <tag> <config> <checkpoint> [manifest] [clips] [device]
#   tag: tên ngắn (vd stftc_5k) → artifacts/vocoder_eval/<tag>/ + artifacts/creak25/copy_synth_<tag>.json
#   checkpoint: .pt của train-vocoder HOẶC pytorch_model.bin (HF pretrained = mốc 0).
set -euo pipefail
TAG=${1:?tag}; CONFIG=${2:?config}; CKPT=${3:?checkpoint}
MANIFEST=${4:-data/codelung/val.jsonl}; CLIPS=${5:-50}; DEVICE=${6:-cpu}
cd "$(dirname "$0")/.."
OUT="artifacts/vocoder_eval/${TAG}"
CREAK_DIR="artifacts/creak25/copy_synth_${TAG}"

echo "== 1/3 fidelity (eval_vocoder) =="
.venv/bin/python tools/eval_vocoder.py --config "$CONFIG" --checkpoint "$CKPT" \
  --manifest "$MANIFEST" --output "$OUT" --clips "$CLIPS" --device "$DEVICE"

echo "== 2/3 chất giọng (voice_quality) =="
.venv/bin/python tools/voice_quality.py --manifest "$MANIFEST" \
  --synth-dir "$OUT/wavs" --output "$OUT/voice_quality.json"

echo "== 3/3 creak 25 clip ngã/nặng (copy_synth_clips + creak_rate) =="
.venv/bin/python tools/copy_synth_clips.py --clips artifacts/creak25/clips25.json \
  --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl --config configs/small_codelung_v2.yaml \
  --out "$CREAK_DIR" --checkpoint "$CKPT" --device "$DEVICE"
CREAPY_PY="${CREAPY_PY:-$HOME/creapy-venv/bin/python}"
if [ -x "$CREAPY_PY" ]; then
  "$CREAPY_PY" tools/creak_rate.py --clips artifacts/creak25/clips25.json \
    --synth-dir "$CREAK_DIR" --out "artifacts/creak25/copy_synth_${TAG}.json"
else
  echo "⚠ thiếu $CREAPY_PY — bỏ creak_rate"
fi
echo "→ xong $TAG (fidelity: $OUT/metrics.json · chất giọng: $OUT/voice_quality.json)"
