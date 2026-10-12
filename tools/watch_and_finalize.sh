#!/usr/bin/env bash
# Canh chuỗi MFA (train + align) và tự chạy finalize khi xong — chạy nền dưới caffeinate.
#
#   nohup caffeinate -is bash tools/watch_and_finalize.sh >/dev/null 2>&1 &
#
# Mặc định chạy cho corpus v2c (13/10/2026); override bằng env: CORPUS, MODEL, ALIGNED, MFA_DB.
# Logic: mỗi 60s kiểm tra có tiến trình mfa đang chạy không.
# Nếu không có tiến trình nào và chưa hoàn tất trong 3 tick liên tiếp → coi như chain chết:
#   - thiếu model zip   → chạy lại `mfa train` (resume, KHÔNG --clean, tối đa 2 lần)
#   - thiếu TextGrid    → chạy lại `mfa align` (--no_overwrite, tối đa 2 lần)
# Xong model + đủ TextGrid → chạy `tools/finalize_dataset_v2.sh` → ghi `FINAL_REPORT.md`.
# Timeout an toàn: 20 giờ. Mọi hành động ghi vào artifacts/dataset_v2/watcher.log.
set -uo pipefail
cd "$(dirname "$0")/.."

D=artifacts/dataset_v2
CORPUS=${CORPUS:-data/codelung/mfa_v2c}
MODEL=${MODEL:-data/codelung/mfa_v2c_model.zip}
ALIGNED=${ALIGNED:-data/codelung/mfa_v2c_aligned}
MFA_DB=${MFA_DB:-$HOME/Documents/MFA/mfa_v2c/mfa_v2c.db}
TRAIN_LOG=${TRAIN_LOG:-$D/mfa_train_v2c.log}
ALIGN_LOG=${ALIGN_LOG:-$D/mfa_align_v2c.log}
EXPECTED=$(ls "$CORPUS"/*.lab 2>/dev/null | wc -l | tr -d ' ')
MFA_ENV=/Users/mps/dvchd/ViFastTts/.mamba-mfa

mkdir -p "$D"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$D/watcher.log"; }
textgrid_count() { ls "$ALIGNED"/*.TextGrid 2>/dev/null | wc -l | tr -d ' '; }
mfa_running() { pgrep -f "mfa train" >/dev/null 2>&1 || pgrep -f "mfa align" >/dev/null 2>&1; }

log "watcher bắt đầu · corpus $EXPECTED clip · model=$([ -f "$MODEL" ] && echo có || echo chưa) · TextGrid=$(textgrid_count)"
stall=0; train_retries=0; align_retries=0
for _ in $(seq 1 1200); do  # 1200 × 60s = 20h
    if mfa_running; then stall=0; sleep 60; continue; fi
    # Không có tiến trình mfa nào: hoàn tất khi đủ model + gần đủ TextGrid
    # (MFA có thể bỏ sót ~1% utterance ở tầng thu thập alignment — chấp nhận ≥95%).
    if [ -f "$MODEL" ] && [ "$(textgrid_count)" -ge $((EXPECTED * 95 / 100)) ]; then
        log "hoàn tất: model OK + $(textgrid_count)/$EXPECTED TextGrid"
        break
    fi
    stall=$((stall + 1))
    if [ "$stall" -lt 3 ]; then sleep 60; continue; fi
    stall=0
    if [ ! -f "$MODEL" ] && [ "$train_retries" -lt 2 ]; then
        train_retries=$((train_retries + 1))
        log "chain chết, chưa có model → resume train (lần $train_retries, KHÔNG --clean)"
        micromamba run -p "$MFA_ENV" mfa train "$CORPUS" "$CORPUS/lexicon.txt" "$MODEL" \
            --speaker_characters 11 --num_jobs 8 -v >> "$D/mfa_train_retry.log" 2>&1
        continue
    fi
    if [ -f "$MODEL" ] && [ "$(textgrid_count)" -lt "$EXPECTED" ] && [ "$align_retries" -lt 2 ]; then
        align_retries=$((align_retries + 1))
        log "chain chết/align thiếu ($(textgrid_count)/$EXPECTED) → align lại (lần $align_retries, no overwrite)"
        micromamba run -p "$MFA_ENV" mfa align "$CORPUS" "$CORPUS/lexicon.txt" "$MODEL" "$ALIGNED" \
            --speaker_characters 11 --num_jobs 8 --no_overwrite -v >> "$D/mfa_align_retry.log" 2>&1
        continue
    fi
    log "hết retry — model=$([ -f "$MODEL" ] && echo có || echo thiếu) · TextGrid=$(textgrid_count)/$EXPECTED"
    break
done

if [ ! -f "$MODEL" ] || [ "$(textgrid_count)" -lt $((EXPECTED * 95 / 100)) ]; then
    {
        echo "# MFA v2 — CHƯA HOÀN TẤT ($(date '+%Y-%m-%d %H:%M'))"
        echo
        echo "- model zip: $([ -f "$MODEL" ] && echo có || echo thiếu)"
        echo "- TextGrid: $(textgrid_count)/$EXPECTED"
        echo
        echo '## log train (30 dòng cuối)'
        echo '```'; tail -30 "$TRAIN_LOG" 2>/dev/null | cut -c1-120; echo '```'
        echo '## log align (30 dòng cuối)'
        echo '```'; tail -30 "$ALIGN_LOG" 2>/dev/null | cut -c1-120; echo '```'
    } > "$D/FINAL_REPORT.md"
    log "KHÔNG hoàn tất — xem $D/FINAL_REPORT.md"
    exit 1
fi

log "MFA xong: model OK + $(textgrid_count)/$EXPECTED TextGrid → chạy finalize"
if ! TEXTGRID_DIR="$ALIGNED" MFA_DB="$MFA_DB" bash tools/finalize_dataset_v2.sh >> "$D/finalize.log" 2>&1; then
    log "finalize LỖI — xem $D/finalize.log (30 dòng cuối)"
    { echo "# FINALIZE LỖI ($(date '+%Y-%m-%d %H:%M'))"; echo '```';
      tail -30 "$D/finalize.log" | cut -c1-140; echo '```'; } > "$D/FINAL_REPORT.md"
    exit 1
fi

log "finalize xong → tổng hợp FINAL_REPORT.md"
.venv/bin/python - "$D" "$EXPECTED" "$ALIGNED" "$MODEL" <<'PY' >> "$D/FINAL_REPORT.md"
import json, sys
from pathlib import Path
d, expected, aligned, model = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
print(f"# MFA v2c + finalize — HOÀN TẤT ({__import__('datetime').datetime.now():%Y-%m-%d %H:%M})")
print()
print(f"- TextGrid: {len(list(aligned.glob('*.TextGrid')))}/{expected}")
print(f"- model zip: {model.stat().st_size/1e6:.0f} MB")
for name in ("mfa_durations_report.json", "cut_segments_report.json"):
    path = Path("data/codelung") / name
    if path.exists():
        payload = json.loads(path.read_text())
        print(f"\n## {name}\n```json\n{json.dumps(payload, ensure_ascii=False, indent=1)[:3000]}\n```")
f0 = Path("data/codelung/f0_stats.json")
if f0.exists():
    payload = json.loads(f0.read_text())
    keep = {k: payload[k] for k in ("clips", "syllables", "f0_log_mean", "f0_log_std", "dct_std", "energy_mean", "energy_std")}
    print(f"\n## f0_stats.json (tóm tắt)\n```json\n{json.dumps(keep, ensure_ascii=False, indent=1)}\n```")
print("\n## File kết quả")
for pattern in ("*_dur.jsonl", "*_seg.jsonl"):
    for path in sorted(Path("data/codelung").glob(pattern)):
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        print(f"- {path} ({lines} dòng)")
PY
log "HOÀN TẤT — xem $D/FINAL_REPORT.md"
