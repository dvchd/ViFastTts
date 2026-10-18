#!/usr/bin/env python
"""tools/night_report.py — tổng hợp bảng so sánh các run đêm 18→19/10 (đọc artifacts/bench +
eval_acoustic + creak_rarity; in markdown để người dùng đọc sáng hôm sau).

Dùng:
  .venv/bin/python tools/night_report.py --out docs/handoff/OVERNIGHT_18_19.md \\
      --baseline b8_filt \\
      --row "mel_sharp best|msh_best|msh|runs/acoustic_v2_b8_melsharp" \\
      --row "mel_sharp last|msh_last|msh|runs/acoustic_v2_b8_melsharp"

Mỗi `--row` = `nhãn|tag bench|short|thư mục run` (short dùng tìm `{short}_cer100.json` +
`probe_{short}.json`; thư mục run để lấy bước val cuối từ history.csv — có thể bỏ).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "artifacts" / "bench"
CER_DIR = ROOT / "artifacts" / "eval_acoustic"
PROBE_DIR = ROOT / "artifacts" / "creak_rarity"

METRICS = [
    ("mel_l1_z", "mel_z", "{:.4f}"),
    ("mel_db_speech", "mel dB", "{:.3f}"),
    ("rough", "rough", "{:.4f}"),
    ("tone_C", "tone C", "{:.4f}"),
    ("cer", "CER-20", "{:.4f}"),
    ("f0_nga", "F0 NGÃ", "{:.1f}"),
    ("dur_mae", "dur MAE", "{:.3f}"),
    ("vuv", "vuv", "{:.4f}"),
    ("creak_djitter_nga", "Δjitter", "{:+.4f}"),
    ("creak_dhnr_nga", "Δhnr", "{:+.2f}"),
    ("syl_z", "syl_z", "{:.4f}"),
    ("mel_gap", "mel_gap", "{:.4f}"),
]
GATE_ORDER = ["mel_l1_z", "mel_db", "rough", "tone_C", "cer", "f0_nga",
              "mel_gap", "syl_z", "mel_z_fix", "mel_db_fix"]
SYMBOL = {"pass": "✅", "fail": "❌", "nobase": "🚫", "nodata": "⏳"}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def bench_metrics(tag: str) -> tuple[dict, dict]:
    payload = load_json(BENCH_DIR / f"{tag}.json") or {}
    merged = dict(payload.get("summary_old") or {})
    merged.update({k: v for k, v in (payload.get("summary_fixgap") or {}).items()
                   if k in ("syl_z", "mel_gap")})
    return merged, (payload.get("gates", {}) or {}).get("hard", {})


def cer100(short: str) -> float | None:
    payload = load_json(CER_DIR / f"{short}_cer100.json") or {}
    return (payload.get("cer") or {}).get("mean")


def probe_row(short: str) -> dict | None:
    payload = load_json(PROBE_DIR / f"probe_{short}.json")
    if not payload:
        return None
    return {name: payload["groups"].get(name, {}) for name in ("rare", "common", "ngang")}


def last_val(run_dir: str | None) -> str:
    if not run_dir:
        return "—"
    path = ROOT / run_dir / "history.csv"
    if not path.exists():
        return "—"
    rows = [r for r in csv.reader(path.read_text(encoding="utf-8").splitlines()) if r and r[0] != "step"]
    if not rows:
        return "—"
    step, val = rows[-1][0], rows[-1][7] if len(rows[-1]) > 7 else "?"
    return f"{step} · val {val[:6]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tổng hợp bảng so sánh run đêm (markdown).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", default="b8_filt")
    parser.add_argument("--row", action="append", default=[])
    args = parser.parse_args()

    base_metrics, _ = bench_metrics(args.baseline)
    lines = ["# Đêm 18→19/10/2026 — bảng so sánh tự động", ""]
    lines.append(f"**Mốc so sánh**: `{args.baseline}` (b8 trên val/holdout lọc, cùng nhãn). "
                 "Mọi số ở `summary_old` + `syl_z`/`mel_gap` (`summary_fixgap`) — cùng `val_filt`.")
    lines.append("")

    table = ["| Ứng viên | Trạng thái | " + " | ".join(label for _, label, _ in METRICS)
             + " | CER-100 |",
             "|---|---|" + "---|" * (len(METRICS) + 1)]
    base_cells = [fmt.format(base_metrics.get(key, float("nan"))) if isinstance(base_metrics.get(key), (int, float))
                  else "—" for key, _, fmt in METRICS]
    base_cer = cer100("b8")
    base_cer_cell = f"{base_cer:.4f}" if base_cer is not None else "—"
    table.append(f"| **{args.baseline} (mốc)** | — | " + " | ".join(base_cells)
                 + f" | {base_cer_cell} |")
    gate_lines = ["", "## Cổng cứng (tương đối so mốc)", "",
                  "| Ứng viên | " + " | ".join(GATE_ORDER) + " | đạt/trượt |", "|---|" + "---|" * (len(GATE_ORDER) + 1)]
    probe_lines = ["", "## Probe đuôi hiếm (`creak_rarity_probe`) — HNR/jitter theo nhóm", "",
                   "| Ứng viên | hiếm (≤5×) | phổ biến (≥150×) | ngang (mốc nền) |", "|---|---|---|---|"]

    for spec in args.row:
        parts = (spec.split("|") + ["", "", ""])[:4]
        label, tag, short, run_dir = parts[0], parts[1], parts[2], parts[3]
        metrics, gates = bench_metrics(tag)
        cells = [f"{fmt.format(metrics[key]):s}" if isinstance(metrics.get(key), (int, float)) else "—"
                 for key, _, fmt in METRICS]
        cand_cer = cer100(short)
        cand_cer_cell = f"{cand_cer:.4f}" if cand_cer is not None else "—"
        table.append(f"| {label} | {last_val(run_dir) if run_dir else '—'} | " + " | ".join(cells)
                     + f" | {cand_cer_cell} |")
        gate_cells = [f"{SYMBOL.get(gates.get(name, ['nodata'])[0], '?')}" for name in GATE_ORDER]
        passed = sum(1 for name in GATE_ORDER if gates.get(name, ["nodata"])[0] == "pass")
        gate_lines.append(f"| {label} | " + " | ".join(gate_cells) + f" | {passed}/{len(GATE_ORDER)} |")
        probe = probe_row(short)
        if probe:
            def cell(group: str, probe: dict = probe) -> str:
                g = probe.get(group) or {}
                if not isinstance(g.get("hnr"), (int, float)):
                    return "—"
                return f"{g['hnr']:+.2f} / {g['jitter']:.3f}"
            probe_lines.append(f"| {label} | {cell('rare')} | {cell('common')} | {cell('ngang')} |")
    probe_base = probe_row("b8")
    if probe_base:
        def cell_b(group: str) -> str:
            g = probe_base.get(group) or {}
            return f"{g['hnr']:+.2f} / {g['jitter']:.3f}" if isinstance(g.get("hnr"), (int, float)) else "—"
        probe_lines.append(f"| **b8 (mốc)** | {cell_b('rare')} | {cell_b('common')} | {cell_b('ngang')} |")

    lines += table + gate_lines + probe_lines
    lines += ["", "## Ghi chú", "",
              "- Mốc/ứng viên đều chấm trên **cùng** `val_filt` + `holdout_filt` (mốc ghim `eval/baselines/b8_filt.json`).",
              "- `CER-20` = 20 clip holdout (nhiễu ±0,03 — cần xem thêm CER-100 nếu cần chốt sát nút).",
              "- `Δjitter`/`Δhnr` ngã: càng gần 0 càng tốt (mục tiêu → 0); mốc thật ngã: hnr +1,10 · jitter 0,063.",
              "- Cột trạng thái = bước val cuối từ `history.csv` (val-trong-train chỉ 40 hàng — chỉ để tham khảo).",
              ""]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
