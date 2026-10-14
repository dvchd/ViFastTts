#!/usr/bin/env python
"""tools/bench_checkpoint.py — chấm MỘT checkpoint bằng bộ thước đa tiêu chí tiếng Việt + scoreboard.

Vì sao (docs/ACOUSTIC_V2.md §37, §42, §43): loss val chỉ là 1 con số trung bình và đã 2 lần dẫn tới
kết luận sai (L1 ≠ tai người; thước trộn). Benchmark này gom **đúng các hiện tượng tiếng Việt**:
mel tổng thể · độ nét · thanh (A/B/C + per-thanh) · rè creak (Δhnr/Δjitter ngã/nặng) · nhịp nghỉ
(pause_f1/gap_sum_mae/length_ratio) · ngữ điệu cuối câu · CER holdout · nhóm coda:SẮC.

Chạy (MPS nhanh hơn CPU ~3–4×; đừng chạy khi đang train trên MPS):
  .venv/bin/python tools/bench_checkpoint.py --run runs/acoustic_v2_base --step 40000 --device mps
  .venv/bin/python tools/bench_checkpoint.py --config runs/acoustic_v2_base/resolved_config.json \
      --checkpoint runs/acoustic_v2_base/last.pt --tag base40k --device cpu --clips 60 --no-cer

Ghi: `artifacts/bench/<tag>.json` (đầy đủ) + nối 1 dòng vào `artifacts/bench/scoreboard.csv` + in bảng.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = "data/codelung/val_clean_v2_dur_seg_slot4.jsonl"
HOLDOUT = "data/codelung/holdout_clean_v2_dur_slot4.jsonl"
CACHE = "cache/codelung_v2"

#: Cổng tham chiếu (docs/EVAL_TOOLS.md §3; "tốt hơn" nếu (so, ngưỡng) đúng chiều).
GATES = {
    "mel_l1_z": ("<=", 1.16),
    "mel_db_speech": ("<=", 8.7),
    "rough": (">=", 0.40),
    "tone_C": (">=", 0.78),
    "cer": ("<=", 0.22),
    "pause_f1": (">=", 0.60),
    "gap_sum_mae": ("<=", 85.0),
    "creak_djitter_nga": ("<=", 0.02),
}


def _run_tool(args: list[str], out: Path, label: str) -> dict | None:
    """Chạy 1 tool CLI; lỗi/thiếu file ⇒ None + cảnh báo (không làm sập benchmark)."""
    t0 = time.time()
    print(f"  · {label} …", flush=True)
    proc = subprocess.run([sys.executable, *args], check=False, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        print(f"    ⚠ {label} lỗi (exit {proc.returncode}): {' / '.join(tail)}")
        return None
    print(f"    ✓ {label} ({time.time() - t0:.0f}s)")
    return json.loads(out.read_text(encoding="utf-8"))


def _mean(block: dict | None, key: str) -> float | None:
    if not block:
        return None
    value = block.get(key)
    if isinstance(value, dict):
        return float(value.get("mean", value.get("median", 0.0)))
    return float(value) if isinstance(value, (int, float)) else None


def _rough_from_history(run_dir: Path, step: int | None) -> float | None:
    """`val_rough` gần nhất (≤ step) trong history.csv — do train_v2 ghi từ §37."""
    path = run_dir / "history.csv"
    if not path.exists():
        return None
    best = None
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("val_rough") in (None, ""):
                continue
            current = int(float(row.get("step", 0)))
            if step is not None and current > step:
                continue
            best = (current, float(row["val_rough"]))
    return best[1] if best else None


def _gate(key: str, value: float | None) -> str:
    if value is None:
        return "–"
    compare, limit = GATES[key]
    ok = value <= limit if compare == "<=" else value >= limit
    return "✅" if ok else "❌"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark đa tiêu chí cho 1 checkpoint (v2).")
    parser.add_argument("--run", help="thư mục run (tự suy config/checkpoint + history)")
    parser.add_argument("--config", help="resolved_config.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--step", type=int, default=None, help="mốc để lấy val_rough + đặt tên tag")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--clips", type=int, default=300, help="clip cho tf_syl + free (pause/ngữ điệu)")
    parser.add_argument("--holdout", type=int, default=20, help="clip holdout cho CER")
    parser.add_argument("--tone-clips", type=int, default=60, help="clip cho thước thanh A/B/C")
    parser.add_argument("--creak-clips", type=int, default=25, help="clip cho thước rè")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cer", action="store_true", help="bỏ CER (nhanh; dùng khi chỉ soi acoustic)")
    parser.add_argument("--out-dir", default="artifacts/bench")
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run)
        config = str(run_dir / "resolved_config.json")
        checkpoint = args.checkpoint or str(run_dir / "last.pt")
        tag = args.tag or f"{run_dir.name}" + (f"_{args.step}" if args.step else "")
    else:
        if not (args.config and args.checkpoint):
            parser.error("cần --run hoặc (--config và --checkpoint)")
        run_dir = Path(args.config).parent
        config, checkpoint = args.config, args.checkpoint
        tag = args.tag or Path(checkpoint).stem

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark {tag} · device={args.device} · clips={args.clips}")

    tf = _run_tool(
        ["tools/eval_acoustic.py", "--config", config, "--checkpoint", checkpoint, "--mode", "tf_syl",
         "--manifest", VAL, "--cache-dir", CACHE, "--clips", str(args.clips), "--device", args.device,
         "--group-by", "coda:tone", "--out", str(out_dir / f"{tag}_tf.json")],
        out_dir / f"{tag}_tf.json", "mel + F0 theo thanh + nhóm coda:SẮC")
    free = _run_tool(
        ["tools/eval_acoustic.py", "--config", config, "--checkpoint", checkpoint, "--mode", "free",
         "--manifest", VAL, "--cache-dir", CACHE, "--clips", str(args.clips), "--device", args.device,
         "--no-cer", "--out", str(out_dir / f"{tag}_free.json")],
        out_dir / f"{tag}_free.json", "free-run: nhịp nghỉ + ngữ điệu")
    cer = None
    if not args.no_cer:
        cer = _run_tool(
            ["tools/eval_acoustic.py", "--config", config, "--checkpoint", checkpoint, "--mode", "free",
             "--manifest", HOLDOUT, "--cache-dir", CACHE, "--clips", str(args.holdout),
             "--device", args.device, "--out", str(out_dir / f"{tag}_cer.json")],
            out_dir / f"{tag}_cer.json", "CER holdout (Whisper)")
    tone = _run_tool(
        ["tools/eval_tone_paths.py", "--config", config, "--checkpoint", checkpoint,
         "--clips", str(args.tone_clips), "--out", str(out_dir / f"{tag}_tone.json")],
        out_dir / f"{tag}_tone.json", "thanh A/B/C (tone_ruler)")
    creak = _run_tool(
        ["tools/eval_creak.py", "--config", config, "--checkpoint", checkpoint,
         "--clips", str(args.creak_clips), "--out", str(out_dir / f"{tag}_creak.json")],
        out_dir / f"{tag}_creak.json", "rè creak (ngã/nặng)")

    tone_c = None
    if tone:
        # JSON của eval_tone_paths: {"paired_abc": {"n": …, "paths": {"A"|"B"|"C": {"accuracy": …}}}}
        paired = (tone.get("paired_abc") or {}).get("paths") or {}
        block = paired.get("C") if isinstance(paired, dict) else None
        if isinstance(block, dict) and "accuracy" in block:
            tone_c = float(block["accuracy"])
    creak_nga = ((creak or {}).get("groups", {}).get("NGA") or {}).get("delta_synth_minus_real", {})
    row = {
        "tag": tag,
        "step": args.step,
        "mel_l1_z": _mean(tf, "mel_l1_z"),
        "mel_db_speech": _mean(tf, "mel_mae_db_speech"),
        "vuv": _mean(tf, "vuv_accuracy"),
        "dur_mae": _mean(tf, "duration_mae_frames"),
        "rough": _rough_from_history(run_dir, args.step),
        "tone_C": tone_c,
        "cer": _mean(cer, "cer"),
        "pause_f1": _mean(free, "pause_f1"),
        "gap_sum_mae": _mean(free, "gap_sum_mae_frames"),
        "length_ratio": _mean(free, "length_ratio"),
        "mel_gap": _mean(free, "mel_gap"),
        "creak_djitter_nga": creak_nga.get("jitter"),
        "creak_dhnr_nga": creak_nga.get("hnr"),
    }
    report = {"tag": tag, "config": config, "checkpoint": checkpoint, "summary": row,
              "raw": {"tf": tf, "free": free, "cer": cer, "tone": tone, "creak": creak}}
    (out_dir / f"{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    scoreboard = out_dir / "scoreboard.csv"
    fields = list(row.keys())
    write_header = not scoreboard.exists()
    with scoreboard.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n== {tag} ==")
    print(f"  mel_z {row['mel_l1_z'] or float('nan'):.4f} {_gate('mel_l1_z', row['mel_l1_z'])}"
          f" · dB {row['mel_db_speech'] or float('nan'):.2f} {_gate('mel_db_speech', row['mel_db_speech'])}"
          f" · vuv {row['vuv'] or float('nan'):.4f} · dur {row['dur_mae'] or float('nan'):.2f}")
    print(f"  độ nét {row['rough'] if row['rough'] is not None else '–'} {_gate('rough', row['rough'])}"
          f" · thanh C {row['tone_C'] if row['tone_C'] is not None else '–'} {_gate('tone_C', row['tone_C'])}"
          f" · CER {row['cer'] if row['cer'] is not None else '–'} {_gate('cer', row['cer'])}")
    print(f"  pause_f1 {row['pause_f1'] if row['pause_f1'] is not None else '–'} {_gate('pause_f1', row['pause_f1'])}"
          f" · gap_mae {row['gap_sum_mae'] if row['gap_sum_mae'] is not None else '–'} {_gate('gap_sum_mae', row['gap_sum_mae'])}"
          f" · len_ratio {row['length_ratio'] if row['length_ratio'] is not None else '–'}"
          f" · mel_gap {row['mel_gap'] if row['mel_gap'] is not None else '–'}")
    print(f"  rè ngã: Δjitter {row['creak_djitter_nga'] if row['creak_djitter_nga'] is not None else '–'}"
          f" {_gate('creak_djitter_nga', row['creak_djitter_nga'])} · Δhnr {row['creak_dhnr_nga']}")
    print(f"→ {out_dir / f'{tag}.json'} · scoreboard: {scoreboard}")


if __name__ == "__main__":
    main()
