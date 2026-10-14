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



def hard_gates(row: dict, base: dict | None) -> dict[str, tuple[bool, str]]:
    """CỔNG CỨNG (§44): None = TRƯỢT và ghi rõ "thiếu dữ liệu"."""

    def at_most(key: str, limit: float) -> tuple[bool, str]:
        value = row.get(key)
        if value is None:
            return False, "thiếu dữ liệu"
        return value <= limit, f"{value:.4f}≤{limit}"

    def at_least(key: str, limit: float) -> tuple[bool, str]:
        value = row.get(key)
        if value is None:
            return False, "thiếu dữ liệu"
        return value >= limit, f"{value:.4f}≥{limit}"

    cer_base = (base or {}).get("cer")
    if row.get("cer") is None or cer_base is None:
        cer_gate = (False, "thiếu dữ liệu")
    else:
        cer_gate = (row["cer"] <= cer_base + 0.03, f"{row['cer']:.4f}≤{cer_base + 0.03:.4f}")
    return {
        "mel_l1_z≤1.16": at_most("mel_l1_z", 1.16),
        "mel_db≤8.7": at_most("mel_db_speech", 8.7),
        "rough≥0.40": at_least("rough", 0.40),
        "tone_C≥0.79": at_least("tone_C", 0.79),
        "cer≤mốc+0.03": cer_gate,
    }


def guard_gates(row: dict, base: dict) -> dict[str, tuple[bool, str]]:
    """RÀO CHẮN so mốc (§44): không được tệ hơn mốc quá ngưỡng."""

    def pair(key: str) -> tuple[float | None, float | None]:
        return row.get(key), base.get(key)

    mel_gap_new, mel_gap_base = pair("mel_gap")
    mel_z_new, mel_z_base = pair("mel_l1_z")
    tone_new, tone_base = pair("tone_C")
    nga_new, nga_base = pair("f0_nga")
    return {
        "mel_gap không tăng": (mel_gap_new is None or mel_gap_base is None or mel_gap_new <= mel_gap_base + 1e-6,
                               f"{mel_gap_new} vs {mel_gap_base}"),
        "mel_l1_z≤mốc×1.01": (mel_z_new is not None and mel_z_base is not None and mel_z_new <= mel_z_base * 1.01,
                              f"{mel_z_new} vs {mel_z_base}"),
        "tone_C≥mốc−0.01": (tone_new is not None and tone_base is not None and tone_new >= tone_base - 0.01,
                            f"{tone_new} vs {tone_base}"),
        "F0 NGÃ≤mốc+5c": (nga_new is None or nga_base is None or nga_new <= nga_base + 5.0,
                          f"{nga_new} vs {nga_base}"),
    }


def target_gates(row: dict, base: dict) -> dict[str, tuple[bool, str]]:
    """CỔNG MỤC TIÊU pause (§44): tương đối so mốc; None = "chưa đo" (không tính là trượt)."""
    f1_new, f1_base = row.get("pause_f1_all"), base.get("pause_f1_all")
    gap_new, gap_base = row.get("gap_sum_mae_all"), base.get("gap_sum_mae_all")
    ratio = row.get("length_ratio")
    return {
        "pause_f1_all≥mốc+0.073": (f1_new is not None and f1_base is not None and f1_new >= f1_base + 0.073,
                                   f"{f1_new} vs {f1_base}"),
        "gap_sum_mae_all≤mốc×0.714": (gap_new is not None and gap_base is not None and gap_new <= gap_base * 0.714,
                                      f"{gap_new} vs {gap_base}"),
        "length_ratio 1.00±0.05": (ratio is not None and abs(ratio - 1.0) <= 0.05, f"{ratio}"),
    }


def rough_from_checkpoint(config: str, checkpoint: str, clips: int = 12) -> float | None:
    """Tính lại `rough` từ checkpoint (khi history.csv thiếu) — 12 clip val đầu, CPU."""
    import warnings

    import torch
    import yaml

    warnings.filterwarnings("ignore")
    from vifasttts.data.dataset_v2 import SlotDataset, collate_slot_batch
    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.factory import build_model, load_model_state

    try:
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8")) if config.endswith(".yaml") \
            else json.loads(Path(config).read_text(encoding="utf-8"))
        f0_stats = json.loads(Path(ROOT / "data/codelung/f0_stats_v2.json").read_text(encoding="utf-8"))
        vocab = ComponentVocabs.default()
        sizes = [len(vocab.onset), len(vocab.medial), len(vocab.nucleus), len(vocab.coda),
                 len(vocab.tone), len(vocab.boundary), len(vocab.punctuation)]
        model = build_model(cfg, sizes, arch="v2_slot5")
        load_model_state(model, torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)["model"])
        model.load_f0_stats(f0_stats)
        model.eval()
        dataset = SlotDataset(str(ROOT / VAL), cfg, f0_stats, cache_dir=str(ROOT / CACHE))

        def delta(mel: torch.Tensor) -> float:
            dt = (mel[:, 1:] - mel[:, :-1]).abs().mean()
            df = (mel[:, :, 1:] - mel[:, :, :-1]).abs().mean()
            return float(dt + df)

        ratios = []
        with torch.no_grad():
            for index in range(min(clips, len(dataset))):
                batch = collate_slot_batch([dataset[index]])
                outputs = model(batch["features"], durations=batch["durations"],
                                seg_durations=batch["seg_durations"], gaps=batch["gaps"],
                                lead=batch["lead"])
                frames = min(outputs["mel"].size(1), batch["mel"].size(1))
                ratios.append(delta(outputs["mel"][:, :frames]) / max(delta(batch["mel"][:, :frames]), 1e-6))
        return sum(ratios) / len(ratios)
    except Exception as error:  # noqa: BLE001 — rough là thông tin phụ, không được làm sập bench
        print(f"    ⚠ không tính lại được rough: {error}")
        return None


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
    parser.add_argument("--baseline", help="JSON bench mốc (so Δ + cổng cứng/rào chắn/mục tiêu A2 §44)")
    parser.add_argument("--val-manifest", default=VAL)
    parser.add_argument("--holdout-manifest", default=HOLDOUT)
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
         "--manifest", args.val_manifest, "--cache-dir", CACHE, "--clips", str(args.clips),
         "--device", args.device, "--group-by", "coda:tone", "--out", str(out_dir / f"{tag}_tf.json")],
        out_dir / f"{tag}_tf.json", "mel + F0 theo thanh + nhóm coda:SẮC")
    free = _run_tool(
        ["tools/eval_acoustic.py", "--config", config, "--checkpoint", checkpoint, "--mode", "free",
         "--manifest", args.val_manifest, "--cache-dir", CACHE, "--clips", str(args.clips),
         "--device", args.device, "--no-cer", "--out", str(out_dir / f"{tag}_free.json")],
        out_dir / f"{tag}_free.json", "free-run: nhịp nghỉ + ngữ điệu")
    cer = None
    if not args.no_cer:
        cer = _run_tool(
            ["tools/eval_acoustic.py", "--config", config, "--checkpoint", checkpoint, "--mode", "free",
             "--manifest", args.holdout_manifest, "--cache-dir", CACHE, "--clips", str(args.holdout),
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
        "mel_gap_z": _mean(free, "mel_gap_z"),
        "pause_f1_all": _mean(free, "pause_f1_all"),
        "gap_sum_mae_all": _mean(free, "gap_sum_mae_all_frames"),
        "pause_precision": _mean(free, "pause_precision"),
        "f0_nga": (((tf or {}).get("f0_rmse_cents_by_tone") or {}).get("NGA") or {}).get("median"),
        "creak_djitter_nga": creak_nga.get("jitter"),
        "creak_dhnr_nga": creak_nga.get("hnr"),
    }
    import hashlib
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=False,
                            capture_output=True, text=True).stdout.strip()
    val_sha = hashlib.sha1(Path(ROOT / args.val_manifest).read_bytes()).hexdigest()
    if row.get("rough") is None:
        row["rough"] = rough_from_checkpoint(config, checkpoint)
    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        base = baseline.get("summary", {})
        base_sha = baseline.get("val_manifest_sha1")
        if base_sha and base_sha != val_sha:
            raise SystemExit(
                f"LỖI: hash val của mốc ({base_sha[:12]}…) khác hash val của run "
                f"({val_sha[:12]}…) — không so được (chỉ so pause/mel_gap trên CÙNG bản val).")
        gates = {"hard": hard_gates(row, base), "guard": guard_gates(row, base),
                 "target": target_gates(row, base)}
        print(f"\n== So với mốc {baseline_path} (val_sha {str(base_sha)[:12]}…) ==")
        for group, label in (("hard", "CỔNG CỨNG"), ("guard", "RÀO CHẮN"), ("target", "CỔNG MỤC TIÊU")):
            marks = " · ".join(f"{'✅' if ok else '❌'} {name} [{detail}]"
                               for name, (ok, detail) in gates[group].items())
            print(f"  {label}: {marks}")
        print(f"  ngưỡng tuyệt đối suy ra: pause_f1_all ≥ "
              f"{(base.get('pause_f1_all') or 0) + 0.073:.4f} · gap_sum_mae_all ≤ "
              f"{(base.get('gap_sum_mae_all') or 0) * 0.714:.1f}")
        print(f"  THAM KHẢO (bộ cũ, bỏ gap cuối dòng): pause_f1 {row['pause_f1']} [cổng cũ 0,60] · "
              f"gap_sum_mae {row['gap_sum_mae']} [cổng cũ 85] · pause_precision {row['pause_precision']}")
    else:
        gates = {"hard": hard_gates(row, None)}

    report = {"tag": tag, "config": config, "checkpoint": checkpoint, "summary": row, "gates": gates,
              "git_commit": commit, "val_manifest": args.val_manifest, "val_manifest_sha1": val_sha,
              "seed": "deterministic (không lấy mẫu; bootstrap thước thanh seed cố định)",
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
