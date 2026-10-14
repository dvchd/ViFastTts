#!/usr/bin/env python
"""tools/bench_checkpoint.py — chấm MỘT checkpoint bằng bộ thước đa tiêu chí tiếng Việt + scoreboard.

Thiết kế theo Agent 0 §44 (Lượt 1b): cổng cứng là **TƯƠNG ĐỐI** so mốc base 40k cùng bản dữ liệu,
ba trạng thái (đạt / trượt / chưa đo / chưa có mốc), và **hai mốc**:
  · `--baseline-old`   : val CŨ + holdout CŨ  → cấp dữ liệu cho CỔNG CỨNG
  · `--baseline-fixgap`: val FIXGAP           → cấp dữ liệu cho mel_gap + CỔNG MỤC TIÊU pause
Lý do: val fixgap là **thang đo khác** cho mel/pause (§44.1) — chỉ so trên cùng một bản val.

Chạy (MPS nhanh hơn CPU; đừng chạy khi đang train trên MPS):
  .venv/bin/python tools/bench_checkpoint.py --config runs/acoustic_v2_A2_fixgap/resolved_config.json \\
      --checkpoint runs/acoustic_v2_A2_fixgap/last.pt --tag a2_45k --step 45000 --device mps \\
      --baseline-old eval/baselines/base40k_old.json --baseline-fixgap eval/baselines/base40k_fixgap.json

Ghi: JSON đầy đủ + scoreboard vào `artifacts/bench/` (local, không commit); tệp mốc gọn vào
`eval/baselines/` khi truyền `--emit-baseline <path>`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_VAL = "data/codelung/val_clean_v2_dur_seg_slot4.jsonl"
FIX_VAL = "data/codelung/val_clean_v2_dur_seg_slot4_fixgap.jsonl"
OLD_HOLDOUT = "data/codelung/holdout_clean_v2_dur_slot4.jsonl"
CACHE = "cache/codelung_v2"

#: Ngưỡng TUYỆT ĐỐI cũ — chỉ để IN THAM KHẢO (base 40k @ val cũ đã trượt dB/CER, quyết định 1.2).
REFERENCE_ABSOLUTE = {"mel_l1_z": 1.16, "mel_db": 8.7, "rough": 0.40, "tone_C": 0.79, "cer": 0.22}
#: Ngưỡng TƯƠNG ĐỐI của cổng cứng (so mốc cùng bản val).
RELATIVE_HARD = {"mel_l1_z": ("mul", 1.01), "mel_db": ("add", 0.10), "rough": ("sub", -0.01),
                 "tone_C": ("sub", -0.01), "cer": ("add", 0.03), "f0_nga": ("add", 5.0),
                 "mel_gap": ("add", 0.0),  # mel_gap ≤ mốc (giảm mới tốt)
                 "syl_z": ("mul", 1.01)}  # §44/B3: vùng âm tiết @ fixgap không xấu quá 1%
PASS, FAIL, NO_BASE, NO_DATA = "pass", "fail", "nobase", "nodata"
SYMBOL = {PASS: "✅", FAIL: "❌", NO_BASE: "🚫", NO_DATA: "⏳"}


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


def _sha1_file(path: str | Path) -> str:
    return hashlib.sha1(Path(ROOT / path).read_bytes()).hexdigest()


def _rough_from_history(run_dir: Path, step: int | None) -> float | None:
    """`val_rough` gần nhất trong history.csv — CHỈ in tham khảo (A4: cổng cứng luôn tính lại)."""
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
            best = float(row["val_rough"])
    return best


def rough_from_checkpoint(config: str, checkpoint: str, val_manifest: str, clips: int = 12) -> float | None:
    """Tính `rough` từ checkpoint trên ĐÚNG bản val của cổng (A4 — không dùng hằng số)."""
    import warnings

    import torch
    import yaml

    warnings.filterwarnings("ignore")
    from vifasttts.data.dataset_v2 import SlotDataset, collate_slot_batch
    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.factory import build_model, load_model_state

    try:
        path = Path(ROOT / config)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix in (".yaml", ".yml") \
            else json.loads(path.read_text(encoding="utf-8"))
        f0_stats = json.loads((ROOT / "data/codelung/f0_stats_v2.json").read_text(encoding="utf-8"))
        vocab = ComponentVocabs.default()
        sizes = [len(vocab.onset), len(vocab.medial), len(vocab.nucleus), len(vocab.coda),
                 len(vocab.tone), len(vocab.boundary), len(vocab.punctuation)]
        model = build_model(cfg, sizes, arch="v2_slot5")
        load_model_state(model, torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)["model"])
        model.load_f0_stats(f0_stats)
        model.eval()
        dataset = SlotDataset(str(ROOT / val_manifest), cfg, f0_stats, cache_dir=str(ROOT / CACHE))

        def delta(mel: torch.Tensor) -> float:
            return float((mel[:, 1:] - mel[:, :-1]).abs().mean()
                         + (mel[:, :, 1:] - mel[:, :, :-1]).abs().mean())

        ratios = []
        with torch.no_grad():
            for index in range(min(clips, len(dataset))):
                batch = collate_slot_batch([dataset[index]])
                outputs = model(batch["features"], durations=batch["durations"],
                                seg_durations=batch["seg_durations"], gaps=batch["gaps"], lead=batch["lead"])
                frames = min(outputs["mel"].size(1), batch["mel"].size(1))
                ratios.append(delta(outputs["mel"][:, :frames]) / max(delta(batch["mel"][:, :frames]), 1e-6))
        return sum(ratios) / len(ratios)
    except Exception as error:  # noqa: BLE001 — rough là thông tin phụ, không được làm sập bench
        print(f"    ⚠ không tính lại được rough: {error}")
        return None


def _relative_gate(key: str, row_value: float | None, base_value: float | None) -> tuple[str, str]:
    """Một cổng cứng tương đối; ba trạng thái: đạt / trượt / chưa có mốc (A2/A3)."""
    if row_value is None:
        return FAIL, "thiếu dữ liệu"
    if base_value is None:
        return NO_BASE, "chưa có mốc"
    mode, offset = RELATIVE_HARD[key]
    if mode == "mul":
        ok = row_value <= base_value * offset
        limit = f"{base_value * offset:.4f}"
    elif mode == "add":
        ok = row_value <= base_value + offset
        limit = f"{base_value + offset:.4f}"
    else:  # "sub": row ≥ base + offset (offset âm)
        ok = row_value >= base_value + offset
        limit = f"{base_value + offset:.4f}"
    return (PASS if ok else FAIL), f"{row_value:.4f} vs {limit}"


def hard_gates(row_old: dict, base_old: dict | None, row_fix: dict, base_fix: dict | None,
               rough_old: float | None) -> dict[str, tuple[str, str]]:
    """CỔNG CỨNG tương đối (§44): mel/dB/rough/tone_C/CER/F0 NGÃ theo val CŨ; mel_gap theo val FIXGAP."""
    base_old = base_old or {}
    base_fix = base_fix or {}
    values = {
        "mel_l1_z": (row_old.get("mel_l1_z"), base_old.get("mel_l1_z")),
        "mel_db": (row_old.get("mel_db_speech"), base_old.get("mel_db_speech")),
        "rough": (rough_old, base_old.get("rough")),
        "tone_C": (row_old.get("tone_C"), base_old.get("tone_C")),
        "cer": (row_old.get("cer"), base_old.get("cer")),
        "f0_nga": (row_old.get("f0_nga"), base_old.get("f0_nga")),
        "mel_gap": (row_fix.get("mel_gap"), base_fix.get("mel_gap")),
        "syl_z": (row_fix.get("syl_z"), base_fix.get("syl_z")),
    }
    return {key: _relative_gate(key, value, limit) for key, (value, limit) in values.items()}


def target_gates(row_fix: dict, base_fix: dict | None) -> dict[str, tuple[str, str]]:
    """CỔNG MỤC TIÊU pause (tương đối, val FIXGAP): None = "chưa đo" (không tính là trượt — A3)."""
    base = base_fix or {}

    def gate(row_value: float | None, base_value: float | None, check) -> tuple[str, str]:
        if row_value is None:
            return NO_DATA, "chưa đo"
        if base_value is None:
            return NO_BASE, "chưa có mốc"
        return (PASS if check(row_value, base_value) else FAIL), f"{row_value:.4f} vs {base_value}"

    return {
        "pause_f1_all≥mốc+0.073": gate(row_fix.get("pause_f1_all"), base.get("pause_f1_all"),
                                      lambda r, b: r >= b + 0.073),
        "gap_sum_mae_all≤mốc×0.714": gate(row_fix.get("gap_sum_mae_all"), base.get("gap_sum_mae_all"),
                                          lambda r, b: r <= b * 0.714),
        "length_ratio 1.00±0.05": gate(row_fix.get("length_ratio"), 1.0,
                                       lambda r, _b: abs(r - 1.0) <= 0.05),
        # §44/B3: A2 có nhãn cho các khung gap ⇒ vùng gap phải giảm ≥ 15% (ước lượng đầu, chỉnh sau).
        "gap_z≤mốc×0.85": gate(row_fix.get("mel_gap_z"), base.get("mel_gap_z"),
                               lambda r, b: r <= b * 0.85),
    }


def _eval_block(config: str, checkpoint: str, val_manifest: str, holdout: str | None, args,
                tag: str, with_cer: bool) -> tuple[dict, dict]:
    """Chạy tf_syl + free (và CER nếu `with_cer`) trên một bản val; trả (row, raw)."""
    tf = _run_tool(
        ["tools/eval_acoustic.py", "--config", config, "--checkpoint", checkpoint, "--mode", "tf_syl",
         "--manifest", val_manifest, "--cache-dir", CACHE, "--clips", str(args.clips),
         "--device", args.device, "--group-by", "coda:tone", "--out", str(args.out_dir / f"{tag}_tf.json")],
        args.out_dir / f"{tag}_tf.json", f"{tag}: mel + F0 theo thanh + nhóm coda:SẮC")
    free = _run_tool(
        ["tools/eval_acoustic.py", "--config", config, "--checkpoint", checkpoint, "--mode", "free",
         "--manifest", val_manifest, "--cache-dir", CACHE, "--clips", str(args.clips),
         "--device", args.device, "--no-cer", "--out", str(args.out_dir / f"{tag}_free.json")],
        args.out_dir / f"{tag}_free.json", f"{tag}: free-run (nhịp nghỉ + ngữ điệu)")
    cer = None
    if with_cer and not args.no_cer:
        cer = _run_tool(
            ["tools/eval_acoustic.py", "--config", config, "--checkpoint", checkpoint, "--mode", "free",
             "--manifest", holdout, "--cache-dir", CACHE, "--clips", str(args.holdout),
             "--device", args.device, "--out", str(args.out_dir / f"{tag}_cer.json")],
            args.out_dir / f"{tag}_cer.json", f"{tag}: CER holdout (Whisper)")
    row = {
        "mel_l1_z": _mean(tf, "mel_l1_z"), "mel_db_speech": _mean(tf, "mel_mae_db_speech"),
        "vuv": _mean(tf, "vuv_accuracy"), "dur_mae": _mean(tf, "duration_mae_frames"),
        "f0_nga": (((tf or {}).get("f0_rmse_cents_by_tone") or {}).get("NGA") or {}).get("median"),
        "cer": _mean(cer, "cer"),
        "pause_f1": _mean(free, "pause_f1"), "gap_sum_mae": _mean(free, "gap_sum_mae_frames"),
        "pause_f1_all": _mean(free, "pause_f1_all"), "gap_sum_mae_all": _mean(free, "gap_sum_mae_all_frames"),
        "pause_precision": _mean(free, "pause_precision"),
        "length_ratio": _mean(free, "length_ratio"),
        "mel_gap": _mean(free, "mel_gap"), "mel_gap_z": _mean(free, "mel_gap_z"),
        "syl_z": _mean(tf, "syl_z"), "syl_db": _mean(tf, "syl_db"),
    }
    return row, {"tf": tf, "free": free, "cer": cer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark đa tiêu chí cho 1 checkpoint (v2, §44).")
    parser.add_argument("--run", help="thư mục run (tự suy config/checkpoint + history)")
    parser.add_argument("--config", help="resolved_config.json (hoặc .yaml)")
    parser.add_argument("--checkpoint")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--clips", type=int, default=300)
    parser.add_argument("--holdout", type=int, default=20)
    parser.add_argument("--tone-clips", type=int, default=60)
    parser.add_argument("--creak-clips", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cer", action="store_true")
    parser.add_argument("--val-manifest", default=FIX_VAL, help="val FIXGAP (mặc định)")
    parser.add_argument("--val-manifest-old", default=OLD_VAL)
    parser.add_argument("--holdout-manifest", default=OLD_HOLDOUT, help="holdout CŨ (mặc định)")
    parser.add_argument("--baseline-old", help="mốc val CŨ (cổng cứng)")
    parser.add_argument("--baseline-fixgap", help="mốc val FIXGAP (mel_gap + cổng mục tiêu)")
    parser.add_argument("--emit-baseline", help="ghi tệp mốc GỌN (eval/baselines/...) rồi kết thúc")
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

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark {tag} · device={args.device} · clips={args.clips}")

    row_old, raw_old = _eval_block(config, checkpoint, args.val_manifest_old, args.holdout_manifest,
                                   args, f"{tag}_old", with_cer=True)
    row_fix, raw_fix = _eval_block(config, checkpoint, args.val_manifest, None, args, f"{tag}_fixgap",
                                   with_cer=False)
    tone = _run_tool(
        ["tools/eval_tone_paths.py", "--config", config, "--checkpoint", checkpoint,
         "--manifest", args.val_manifest_old, "--clips", str(args.tone_clips),
         "--out", str(args.out_dir / f"{tag}_tone.json")],
        args.out_dir / f"{tag}_tone.json", "thanh A/B/C (tone_ruler)")
    creak = _run_tool(
        ["tools/eval_creak.py", "--config", config, "--checkpoint", checkpoint,
         "--manifest", args.val_manifest_old, "--clips", str(args.creak_clips),
         "--out", str(args.out_dir / f"{tag}_creak.json")],
        args.out_dir / f"{tag}_creak.json", "rè creak (ngã/nặng)")
    tone_c = None
    if tone:
        paired = (tone.get("paired_abc") or {}).get("paths") or {}
        block = paired.get("C") if isinstance(paired, dict) else None
        if isinstance(block, dict) and "accuracy" in block:
            tone_c = float(block["accuracy"])
    creak_nga = ((creak or {}).get("groups", {}).get("NGA") or {}).get("delta_synth_minus_real", {})
    for row in (row_old, row_fix):
        row["tone_C"] = tone_c
        row["creak_djitter_nga"] = creak_nga.get("jitter")
        row["creak_dhnr_nga"] = creak_nga.get("hnr")
    row_old["rough"] = rough_from_checkpoint(config, checkpoint, args.val_manifest_old)
    row_fix["rough"] = rough_from_checkpoint(config, checkpoint, args.val_manifest)

    base_old = base_fix = None
    if args.baseline_old:
        base_old = json.loads(Path(args.baseline_old).read_text(encoding="utf-8"))
        _check_baseline_hash(base_old, args.val_manifest_old, args.baseline_old)
    if args.baseline_fixgap:
        base_fix = json.loads(Path(args.baseline_fixgap).read_text(encoding="utf-8"))
        _check_baseline_hash(base_fix, args.val_manifest, args.baseline_fixgap)

    gates = {"hard": hard_gates(row_old, base_old, row_fix, base_fix, row_old["rough"]),
             "target": target_gates(row_fix, base_fix)}
    print("\n== CỔNG CỨNG (tương đối, val cũ + mel_gap val fixgap) ==")
    for name, (state, detail) in gates["hard"].items():
        print(f"  {SYMBOL[state]} {name} [{detail}]")
    print("== CỔNG MỤC TIÊU A2 (val fixgap) ==")
    for name, (state, detail) in gates["target"].items():
        print(f"  {SYMBOL[state]} {name} [{detail}]")
    print("== THAM KHẢO (không phải cổng) ==")
    print(f"  tuyệt đối cũ: {REFERENCE_ABSOLUTE}")
    print(f"  val cũ   : mel_z {row_old['mel_l1_z']} · dB {row_old['mel_db_speech']} · "
          f"CER {row_old['cer']} · pause_f1 {row_old['pause_f1']} · gap_mae {row_old['gap_sum_mae']}")
    print(f"  val fixgap: mel_z {row_fix['mel_l1_z']} · dB {row_fix['mel_db_speech']} · "
          f"pause_f1_all {row_fix['pause_f1_all']} · gap_mae_all {row_fix['gap_sum_mae_all']} · "
          f"precision {row_fix['pause_precision']} · len_ratio {row_fix['length_ratio']} · "
          f"mel_gap {row_fix['mel_gap']} · rough {row_fix['rough']}")
    print(f"  rè ngã: Δjitter {row_fix['creak_djitter_nga']} · Δhnr {row_fix['creak_dhnr_nga']}"
          f" · (thanh+rè chấm @ val CŨ, manifest tường minh)")
    print(f"  vùng (tf @ fixgap): syl_z {row_fix['syl_z']} · syl_db {row_fix['syl_db']} · "
          f"gap_z {row_fix['mel_gap_z']}")
    if base_fix:
        f1_base, gap_base = base_fix.get("summary", {}).get("pause_f1_all"), base_fix.get("summary", {}).get("gap_sum_mae_all")
        if f1_base is not None and gap_base is not None:
            print(f"  ngưỡng tuyệt đối suy ra: pause_f1_all ≥ {f1_base + 0.073:.4f} · "
                  f"gap_sum_mae_all ≤ {gap_base * 0.714:.1f}")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=False,
                            capture_output=True, text=True).stdout.strip()
    report = {"tag": tag, "git_commit": commit, "config": config, "checkpoint": checkpoint,
              "step": args.step, "seed": "deterministic (không lấy mẫu; bootstrap thước thanh seed cố định)",
              "val_manifest": args.val_manifest, "val_manifest_hash": _sha1_file(args.val_manifest),
              "val_manifest_old": args.val_manifest_old, "val_manifest_hash_old": _sha1_file(args.val_manifest_old),
              "holdout_manifest": args.holdout_manifest, "holdout_manifest_hash": _sha1_file(args.holdout_manifest),
              "summary_old": row_old, "summary_fixgap": row_fix, "gates": gates,
              "history_rough_thamkhao": _rough_from_history(run_dir, args.step),
              "raw": {"old": raw_old, "fixgap": raw_fix, "tone": tone, "creak": creak}}
    report_path = args.out_dir / f"{tag}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    scoreboard = args.out_dir / "scoreboard.csv"
    for suffix, row in (("old", row_old), ("fixgap", row_fix)):
        fields = ["tag", "step", "mel_l1_z", "mel_db_speech", "rough", "tone_C", "cer",
                  "pause_f1_all", "gap_sum_mae_all", "length_ratio", "mel_gap", "f0_nga",
                  "creak_djitter_nga"]
        entry = {"tag": f"{tag}_{suffix}", "step": args.step, **{k: row.get(k) for k in fields[2:]}}
        write_header = not scoreboard.exists()
        with scoreboard.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(entry)

    if args.emit_baseline:
        raw_paths = {name: str(args.out_dir / f"{tag}_{name}.json") for name in ("old_tf", "old_free", "old_cer",
                                                                                 "fixgap_tf", "fixgap_free",
                                                                                 "tone", "creak")}
        compact = {"tag": tag, "git_commit": commit, "checkpoint": checkpoint, "step": args.step,
                   "val_manifest_hash": report["val_manifest_hash"],
                   "val_manifest_hash_old": report["val_manifest_hash_old"],
                   "holdout_manifest_hash": report["holdout_manifest_hash"],
                   "seed": report["seed"], "summary": row_old, "summary_fixgap": row_fix,
                   "raw_paths": raw_paths,
                   "raw_sha256": {name: (hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                         if (ROOT / path).exists() else None) for name, path in raw_paths.items()}}
        Path(args.emit_baseline).write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"→ tệp mốc gọn: {args.emit_baseline}")
    print(f"→ {report_path} · scoreboard: {scoreboard}")


def _check_baseline_hash(baseline: dict, val_manifest: str, path: str) -> None:
    """Mốc phải cùng bản val đang chấm (A6: đọc được khoá cũ `val_manifest_sha1`)."""
    expected = baseline.get("val_manifest_hash") or baseline.get("val_manifest_sha1")
    actual = _sha1_file(val_manifest)
    if expected and expected != actual:
        raise SystemExit(f"LỖI: mốc {path} thuộc val khác ({str(expected)[:12]}… vs {actual[:12]}…) — "
                         "chỉ so trên CÙNG một bản val (§44.1).")


if __name__ == "__main__":
    main()
