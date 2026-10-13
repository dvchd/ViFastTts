#!/usr/bin/env bash
# Chạy một phát sau khi `mfa align` xong: TextGrid → durations → segments → f0 stats → verify.
#
#   bash tools/finalize_dataset_v2.sh
#
# Đầu ra (data/codelung/):
#   *_dur.jsonl          manifest có durations (từ TextGrid)
#   *_seg.jsonl          segment 3–10s (frame_range, để acoustic train)
#   mfa_durations_report.json · cut_segments_report.json · f0_stats.json
set -euo pipefail
cd "$(dirname "$0")/.."

TEXTGRID_DIR=${TEXTGRID_DIR:-data/codelung/mfa_v2b_aligned}
MFA_DB=${MFA_DB:-$HOME/Documents/MFA/mfa_v2b/mfa_v2b.db}
CONFIG=${CONFIG:-configs/small_codelung.yaml}
CACHE=${CACHE:-cache/codelung_v2}
OUT=data/codelung

echo "[finalize] 1/4 TextGrid → durations"
extra_db=()
[ -f "$MFA_DB" ] && extra_db=(--mfa-db "$MFA_DB" --score-drop-fraction 0.10)
.venv/bin/python tools/mfa_to_durations.py \
    --textgrid-dir "$TEXTGRID_DIR" \
    --manifest "$OUT/train_clean_v2.jsonl" \
    --manifest "$OUT/val_clean_v2.jsonl" \
    --manifest "$OUT/holdout_clean_v2.jsonl" \
    "${extra_db[@]}" \
    --out-dir "$OUT"

echo "[finalize] 2/4 cắt segment 3–10s"
.venv/bin/python tools/cut_segments.py \
    --manifest "$OUT/train_clean_v2_dur.jsonl" \
    --manifest "$OUT/val_clean_v2_dur.jsonl" \
    --out-dir "$OUT"

echo "[finalize] 3/4 stats log-F0 + DCT4 + energy"
.venv/bin/python tools/compute_f0_stats.py \
    --manifest "$OUT/train_clean_v2_dur.jsonl" \
    --config "$CONFIG" --out "$OUT/f0_stats.json" \
    --cache-dir "$CACHE" --sample 2000 --workers 8

echo "[finalize] 4/4 verify: TtsDataset load 1 batch segment"
.venv/bin/python - <<'EOF'
import sys
import yaml
sys.path.insert(0, "src")
from vifasttts.data.dataset import TtsDataset, collate_batch
from torch.utils.data import DataLoader
config = yaml.safe_load(open("configs/small_codelung.yaml"))
dataset = TtsDataset("data/codelung/train_clean_v2_dur_seg.jsonl", config, cache_dir="cache/codelung_v2")
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_batch)
batch = next(iter(loader))
print("  batch:", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})
frames = int(batch["mel_lengths"][0])
n = int(batch["syllable_lengths"][0])
spans = int(batch["spans"][0][:n].sum())
durations = int(batch["durations"][0][:n].sum())
assert frames == spans, f"mel frames {frames} != sum spans {spans}"
print(f"  OK: sum(spans) == mel frames · hữu thanh {durations}/{frames} frame ({durations / frames:.0%})")
EOF

echo "[finalize] XONG — xem báo cáo:"
echo "  $OUT/mfa_durations_report.json · $OUT/cut_segments_report.json · $OUT/f0_stats.json"
