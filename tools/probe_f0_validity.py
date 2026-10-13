"""tools/probe_f0_validity.py — đo chất lượng mask hữu thanh `vuv` cho F0 v2 (trước Addendum 6).

Câu hỏi: F4 chốt mask F0 bằng `vuv` (pyworld). Nhưng đo nhanh trên "các" (/k/) thấy closure
**vô thanh** vẫn bị đánh `vuv=1` kèm f0 nhiễu (164→206→139 Hz) → tỷ lệ hữu thanh role O = 96,9%
(vô lý với tiếng Việt nhiều onset vô thanh). Probe này định lượng theo **onset abstract**:

- tỷ lệ frame `vuv=1` trong cửa sổ O (segment onset);
- tỷ lệ âm tiết có `ratio(O) ≥ 0,5` (ngưỡng mask loss F0 của §6 cho O/C);
- năng lượng frame O (vuv=1) so với median năng lượng nucleus cùng âm tiết (dB log-power);
- độ gồ ghề f0: median |Δ log2 f0| giữa 2 frame liền kề (vuv=1 cả hai) trong O — so với nucleus;

Kết quả ghi `artifacts/dataset_v2/f0_validity_probe.json` + in bảng theo nhóm onset.

Chạy:
    python tools/probe_f0_validity.py --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
        --config configs/small_codelung.yaml --cache-dir cache/codelung_v2 \
        --sample 400 --workers 8 --out artifacts/dataset_v2/f0_validity_probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from r5b_segment_stats import segment_windows

# onset trừu tượng → nhóm phát âm
VOICELESS_ONSETS = {"PH", "T", "TH", "X", "S", "CH", "TR", "K", "KH", "H", "P"}
VOICED_ONSETS = {"B", "M", "V", "N", "D", "GI", "R", "NH", "L", "NG", "G"}


CODA_SONORANT = {"M", "N", "NG", "NH", "W", "J"}
CODA_STOP = {"P", "T", "C", "CH"}


def group_of(onset: str) -> str:
    """Nhóm của onset trừu tượng."""
    if onset in VOICELESS_ONSETS:
        return "voiceless"
    if onset in VOICED_ONSETS:
        return "voiced"
    return "other"


def coda_group_of(coda: str) -> str:
    """Nhóm của coda trừu tượng."""
    if coda in CODA_SONORANT:
        return "sonorant"
    if coda in CODA_STOP:
        return "stop"
    return "other"


def _worker(payload: tuple[dict[str, Any], dict[str, Any], str | None]) -> dict[str, list[float]]:
    """Một row: trả list giá trị thô theo onset (sẽ gộp ở main process)."""
    from vifasttts.data.cache import FeatureCache
    from vifasttts.data.features import extract_features

    row, config, cache_dir = payload
    cache = FeatureCache(cache_dir, config) if cache_dir else None
    extracted = cache.get(row) if cache is not None else None
    if extracted is None:
        extracted = extract_features(row["audio_path"], config)
        if cache is not None:
            cache.put(row, extracted)

    frame_f0 = extracted["f0"].numpy().astype(np.float64)
    frame_vuv = extracted["vuv"].numpy().astype(np.float64)
    frame_energy = extracted["energy"].numpy().astype(np.float64)
    out: dict[str, list[float]] = defaultdict(list)

    def measure(keys: tuple[str, ...], low: int, high: int, nucleus_energy: float) -> None:
        """Đo một cửa sổ segment: vuv/jitter/năng lượng tương đối so nucleus."""
        low, high = max(low, 0), min(high, frame_f0.size)
        if high <= low:
            return
        window_vuv = frame_vuv[low:high] > 0.5
        window_energy = frame_energy[low:high]
        window_f0 = frame_f0[low:high]
        voiced = window_vuv & (window_f0 > 1e-3)
        ratio = float(voiced.mean())
        for key in keys:
            out[f"{key}:frames"] += [float(voiced.size)]
            out[f"{key}:vuv_frames"] += [float(voiced.sum())]
            out[f"{key}:ratio"] += [ratio]
            out[f"{key}:rel_energy"] += list(window_energy[voiced] - nucleus_energy)
        if int(voiced.sum()) >= 2:
            values = np.log2(window_f0[voiced])
            jitter = np.abs(np.diff(values))
            for key in keys:
                out[f"{key}:jitter"] += list(jitter)

    for syllable, windows in zip(row["syllables"], segment_windows(row)):
        by_role = {role: (low, high) for role, low, high in windows}
        if "N" not in by_role:
            continue
        n_low, n_high = by_role["N"]
        n_low = max(n_low, 0)
        n_high = min(n_high, frame_energy.size)
        if n_high <= n_low:
            continue
        nucleus_energy = float(np.median(frame_energy[n_low:n_high]))
        onset = syllable["abstract"]["onset"]
        if onset != "NONE" and "O" in by_role:
            o_low, o_high = by_role["O"]
            measure((onset, f"group:{group_of(onset)}"), o_low, o_high, nucleus_energy)
        coda = syllable["abstract"]["coda"]
        if coda != "NONE" and "C" in by_role:
            c_low, c_high = by_role["C"]
            measure((f"coda:{coda}", f"coda_group:{coda_group_of(coda)}"), c_low, c_high, nucleus_energy)
    return out


def summarize(payload: dict[str, list[float]]) -> dict[str, Any]:
    """Gộp list giá trị → bảng thống kê theo onset/nhóm."""
    onsets: dict[str, dict[str, Any]] = {}
    for key, values in payload.items():
        name, _, field = key.rpartition(":")
        if not name:
            continue
        entry = onsets.setdefault(name, {"class": name.split(":")[-1]})
        array = np.asarray(values, dtype=np.float64)
        if field == "frames":
            entry["syllables"] = int(array.size)
            entry["frames"] = int(array.sum())
        elif field == "vuv_frames":
            entry["vuv_frames"] = int(array.sum())
        elif field == "ratio":
            entry["ratio_ge05_frac"] = round(float((array >= 0.5).mean()), 4)
            entry["ratio_mean"] = round(float(array.mean()), 4)
        elif field == "rel_energy":
            if array.size:
                entry["rel_energy_median"] = round(float(np.median(array)), 2)
                entry["rel_energy_p10"] = round(float(np.percentile(array, 10)), 2)
                entry["rel_energy_lt_-15_frac"] = round(float((array < -15).mean()), 3)
        elif field == "jitter":
            if array.size:
                entry["jitter_median"] = round(float(np.median(array)), 4)
                entry["jitter_p90"] = round(float(np.percentile(array, 90)), 4)
    for entry in onsets.values():
        frames = entry.get("frames", 0)
        entry["vuv_frac"] = round(entry.get("vuv_frames", 0) / frames, 4) if frames else 0.0
    return onsets


def main() -> None:
    """Chạy probe, ghi JSON + in bảng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    picked = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    payloads = [(row, config, args.cache_dir) for row in picked]
    merged: dict[str, list[float]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for part in pool.map(_worker, payloads):
            for key, values in part.items():
                merged[key] += values
    table = summarize(merged)

    order = ([f"group:{name}" for name in ("voiced", "voiceless", "other")]
             + [f"coda_group:{name}" for name in ("sonorant", "stop", "other")]
             + sorted((name for name in table if not name.startswith(("group:", "coda_group:"))),
                      key=lambda name: -table[name].get("syllables", 0)))
    report = {"manifest": args.manifest, "sample": len(picked), "seed": args.seed,
              "rows": len(rows), "table": {name: table[name] for name in order if name in table}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== Probe f0_validity · {len(picked)} row → {out} ==\n")
    header = f'{"onset":14s} {"syl":>6s} {"vuv%":>6s} {"ratio≥.5":>9s} {"relE med":>9s} {"relE<-15":>9s} ' \
             f'{"jit med":>8s} {"jit p90":>8s}'
    print(header)
    for name in order:
        if name not in table:
            continue
        entry = table[name]
        print(f'{name:14s} {entry.get("syllables", 0):6d} {entry.get("vuv_frac", 0):6.1%} '
              f'{entry.get("ratio_ge05_frac", 0):9.1%} {entry.get("rel_energy_median", 0):9.1f} '
              f'{entry.get("rel_energy_lt_-15_frac", 0):9.1%} {entry.get("jitter_median", 0):8.4f} '
              f'{entry.get("jitter_p90", 0):8.4f}')


if __name__ == "__main__":
    main()
