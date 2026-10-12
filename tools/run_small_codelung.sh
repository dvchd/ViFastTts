#!/usr/bin/env bash
# Chạy trọn pipeline "small" trên dataset kênh @codelung:
#   cache feature → train acoustic → train vocoder → export ONNX → demo WAV.
# Chạy lại an toàn: bước nào xong sẽ bỏ qua/tiếp tục (resume từ last.pt).
#
#   bash tools/run_small_codelung.sh
#
# Xem kết quả đo thật + hạn chế: docs/TRAINING_RUN_CODELUNG.md
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=configs/small_codelung.yaml
DATA=data/codelung
CACHE=cache/codelung
RUN_AC=runs/small_codelung
RUN_VC=runs/vocoder_codelung
ART=artifacts/small_codelung
PY=.venv/bin/vifasttts

mkdir -p "$ART" "$CACHE" runs

echo "[1/5] Cache feature (workers 8, bỏ qua file đã có)"
"$PY" cache-features --manifest "$DATA/train.jsonl" --config "$CFG" --cache-dir "$CACHE" --workers 8
"$PY" cache-features --manifest "$DATA/val.jsonl" --config "$CFG" --cache-dir "$CACHE" --workers 8

echo "[2/5] Train acoustic (MPS) — từng khối 5k bước, tự resume (chống OOM Metal)"
TOTAL=$(.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$CFG'))['training']['max_steps'])")
.venv/bin/python tools/train_chunked.py --config "$CFG" \
  --train-manifest "$DATA/train.jsonl" --val-manifest "$DATA/val.jsonl" \
  --output "$RUN_AC" --total-steps "$TOTAL" --chunk 5000

echo "[3/5] Train vocoder (CPU, random crop) — resume nếu đã có last.pt"
RES=""
[ -f "$RUN_VC/last.pt" ] && RES="--resume $RUN_VC/last.pt"
"$PY" train-vocoder --config "$CFG" --manifest "$DATA/train.jsonl" --output "$RUN_VC" $RES

echo "[4/5] Export ONNX (fp32 + fp16 + vocoder)"
"$PY" export-onnx --config "$CFG" --checkpoint "$RUN_AC/best.pt" \
  --output "$ART/model.onnx" --precision fp32
"$PY" export-onnx --config "$CFG" --checkpoint "$RUN_AC/best.pt" \
  --output "$ART/model.fp16.onnx" --precision fp16
"$PY" export-vocoder-onnx --config "$CFG" --checkpoint "$RUN_VC/last.pt" \
  --output "$ART/vocoder.onnx"

echo "[5/5] Demo (PyTorch + ONNX Runtime)"
"$PY" synthesize --config "$CFG" --checkpoint "$RUN_AC/best.pt" \
  --vocoder-checkpoint "$RUN_VC/last.pt" --peak-normalize \
  --text "Xin chào, đây là bản small huấn luyện trên kênh Code Lủng. Hôm nay trời đẹp." \
  --output "$ART/demo.wav"
"$PY" synthesize-onnx --acoustic "$ART/model.onnx" --vocoder "$ART/vocoder.onnx" \
  --text "Xin chào" --output "$ART/demo_onnx.wav" --peak-normalize

echo "XONG. Artifacts: $ART"
