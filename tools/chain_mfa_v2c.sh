#!/usr/bin/env bash
# Chuỗi MFA v2c (13/10/2026 — sau khi sửa tên phone bỏ `_`):
#   mfa train (resume-safe, KHÔNG --clean) → mfa align → ghi mốc DONE.
#
# Chạy dưới caffeinate (máy ngủ từng kill job):
#   nohup caffeinate -is bash tools/chain_mfa_v2c.sh >/dev/null 2>&1 &
#
# Log: artifacts/dataset_v2/chain_v2c.log · mfa_train_v2c.log · mfa_align_v2c.log
# Sau DONE: align TextGrid ở data/codelung/mfa_v2c_aligned → chạy finalize
#   TEXTGRID_DIR=data/codelung/mfa_v2c_aligned MFA_DB=$HOME/Documents/MFA/mfa_v2c/mfa_v2c.db \
#       bash tools/finalize_dataset_v2.sh
set -uo pipefail
cd "$(dirname "$0")/.."

D=artifacts/dataset_v2
CORPUS=data/codelung/mfa_v2c
MODEL=data/codelung/mfa_v2c_model.zip
ALIGNED=data/codelung/mfa_v2c_aligned
MFA_ENV=$PWD/.mamba-mfa

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$D/chain_v2c.log"; }

log "bắt đầu: mfa train (resume-safe)"
micromamba run -p "$MFA_ENV" mfa train "$CORPUS" "$CORPUS/lexicon.txt" "$MODEL" \
    --speaker_characters 11 --num_jobs 8 -v >> "$D/mfa_train_v2c.log" 2>&1
train_status=$?
if [ ! -f "$MODEL" ]; then
    log "TRAIN FAILED (exit $train_status, không có $MODEL) — xem $D/mfa_train_v2c.log"
    exit 1
fi
log "train xong (exit $train_status) → align"
micromamba run -p "$MFA_ENV" mfa align "$CORPUS" "$CORPUS/lexicon.txt" "$MODEL" "$ALIGNED" \
    --speaker_characters 11 --num_jobs 8 --clean -v > "$D/mfa_align_v2c.log" 2>&1
align_status=$?
textgrids=$(ls "$ALIGNED"/*.TextGrid 2>/dev/null | wc -l | tr -d ' ')
log "align xong (exit $align_status) · TextGrid $textgrids"
log "DONE $(date '+%H:%M:%S')"
