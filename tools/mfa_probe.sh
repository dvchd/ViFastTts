#!/usr/bin/env bash
# Probe MFA (train + align) trên subset ~1h để ngoại suy thời gian cho corpus đầy đủ (C1).
#
# Chạy SAU khi cache feature xong (nhường CPU), song song vocoder (MPS).
#   bash tools/mfa_probe.sh [số_clip]        # mặc định 240 clip ≈ 1h
# Biến môi trường: JOBS (mặc định 8), SPEAKER_CHARS (mặc định 11 = độ dài id YouTube),
#                  CORPUS (data/codelung/mfa), PROBE (data/codelung/mfa_probe).
#
# Kết quả: in thời gian train/align (giây) + số giờ probe để tự ngoại suy:
#   thời_gian_corpus_đầy_đủ ≈ thời_gian_probe × (44,96h / giờ_probe)
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-8}
SPEAKER_CHARS=${SPEAKER_CHARS:-11}
CORPUS=${CORPUS:-data/codelung/mfa}
PROBE=${PROBE:-data/codelung/mfa_probe}
TARGET_CLIPS=${1:-240}
MFA_BIN=${MFA_BIN:-micromamba run -p "$PWD/.mamba-mfa" mfa}

rm -rf "$PROBE"
mkdir -p "$PROBE/corpus"
.venv/bin/python - "$CORPUS" "$PROBE/corpus" "$TARGET_CLIPS" <<'EOF'
import shutil
import sys
import wave
from pathlib import Path

root, out, target = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
labs = sorted(root.glob("labs/*.lab"))
step = max(1, len(labs) // target)
picked = labs[::step][:target]
seconds = 0.0
for lab in picked:
    sample_id = lab.stem
    shutil.copy2(lab, out / f"{sample_id}.lab")
    source = root / "wavs" / f"{sample_id}.wav"
    shutil.copy2(source, out / f"{sample_id}.wav")
    with wave.open(str(source), "rb") as handle:
        seconds += handle.getnframes() / handle.getframerate()
print(f"probe: {len(picked)} clip · {seconds/3600:.2f} giờ")
EOF

start=$(date +%s)
$MFA_BIN train --clean --num_jobs "$JOBS" --speaker_characters "$SPEAKER_CHARS" \
    "$PROBE/corpus" "$CORPUS/lexicon.txt" "$PROBE/model"
train_seconds=$(( $(date +%s) - start ))

start=$(date +%s)
$MFA_BIN align --clean --num_jobs "$JOBS" --beam 100 --retry_beam 400 \
    "$PROBE/corpus" "$CORPUS/lexicon.txt" "$PROBE/model" "$PROBE/aligned"
align_seconds=$(( $(date +%s) - start ))

echo "MFA_PROBE train=${train_seconds}s align=${align_seconds}s jobs=$JOBS"
