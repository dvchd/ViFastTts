"""tools/v1834_audit.py — audit v18.33 (không chấm lại audio, chỉ tính lại từ JSON).

Kiểm:
1. CER/WER: macro vs micro, reference, normalization, pairing text/audio, frame_range.
2. Chiều tốt/xấu: win/lose/tie theo đúng chiều từng metric.
3. Thời lượng: total = pause + non-pause; voiced ≠ phần bù; pause% macro vs weighted.
4. Sensitivity: xu hướng trung bình vs đơn điệu từng câu (nhiễu và cắt).
5. Bootstrap: CI cho trung vị và trung bình; cách xử lý số câu/video.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import v1828_gta_eval as E

from vifasttts.eval_stats import outcome_counts

HOP = 256
SR = 24000
AUDIO = Path("artifacts/bench/v1833_audio")


def norm(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def micro_cer(refs, hyps) -> float:
    num = den = 0
    for ref, hyp in zip(refs, hyps):
        r = norm(ref).replace(" ", "")
        h = norm(hyp).replace(" ", "")
        num += edit(r, h)
        den += len(r)
    return num / max(1, den)


def micro_wer(refs, hyps) -> float:
    num = den = 0
    for ref, hyp in zip(refs, hyps):
        r = norm(ref).split()
        h = norm(hyp).split()
        num += edit(r, h)
        den += len(r)
    return num / max(1, den)


def main() -> None:
    items = {item["id"]: item for item in json.loads((AUDIO / "items.json").read_text(encoding="utf-8"))}
    judges = json.loads(Path("artifacts/bench/v1833_judges.json").read_text(encoding="utf-8"))
    analysis = json.loads(Path("artifacts/bench/v1833_analysis.json").read_text(encoding="utf-8"))
    sens = json.loads(Path("artifacts/bench/v1833_sens/sensitivity.json").read_text(encoding="utf-8"))

    out: dict = {}

    # ---------- 1. CER/WER ----------
    cer = {}
    for name in ("p2", "g1", "real"):
        rows = judges["per_file"][name]
        ids = sorted(rows)
        refs = [items[i]["text"] for i in ids]
        hyps = [rows[i]["asr"] for i in ids]
        cer[name] = {
            "n": len(ids),
            "macro_cer": round(float(np.mean([rows[i]["cer"] for i in ids])), 4),
            "micro_cer": round(float(micro_cer(refs, hyps)), 4),
            "macro_wer": round(float(np.mean([rows[i]["wer"] for i in ids])), 4),
            "micro_wer": round(float(micro_wer(refs, hyps)), 4),
            "median_cer": round(float(np.median([rows[i]["cer"] for i in ids])), 4),
        }
    # pairing: text ↔ audio
    manifest = {json.loads(x)["id"]: json.loads(x) for x in
                Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    pairing = []
    for key in list(items)[:300]:
        row = manifest[key]
        frames = int(row["frame_range"][1]) - int(row["frame_range"][0])
        real_dur = sf.info(AUDIO / "real" / f"{key}.wav").duration
        pairing.append({
            "id": key,
            "frames_gt": frames,
            "durations_sum": int(sum(row["durations"])) + int(sum(row["gaps"])) + int(row.get("lead_frames") or 0),
            "real_duration_s": round(real_dur, 4),
            "expected_s": round(frames * HOP / SR, 4),
            "text_match": items[key]["text"] == " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"]),
        })
    out["cer"] = cer
    out["pairing"] = {
        "checked": len(pairing),
        "frames_equal_duration_sum": sum(1 for p in pairing if p["frames_gt"] == p["durations_sum"]),
        "real_duration_matches": sum(1 for p in pairing if abs(p["real_duration_s"] - p["expected_s"]) < 1e-3),
        "text_matches": sum(1 for p in pairing if p["text_match"]),
    }

    # ---------- 2. chiều tốt/xấu ----------
    deltas = {}
    for field, higher_better in (("d_bak", True), ("d_sig", True), ("d_ovrl", True), ("d_utmos", True),
                                 ("d_hnr", True), ("d_cer", False), ("d_wer", False), ("d_flat", False)):
        values = [r[field] for r in analysis_rows(judges, items)]
        deltas[field] = {"higher_better": higher_better, **outcome_counts(values, higher_better=higher_better)}
    out["direction"] = deltas

    # ---------- 3. thời lượng ----------
    dur = {}
    for name in ("p2", "g1", "real"):
        total = pause = voiced = 0
        weighted_pause = 0.0
        for key in sorted(judges["per_file"][name]):
            wave, _ = sf.read(AUDIO / name / f"{key}.wav", dtype="float32")
            frames = int(len(wave) // HOP)
            if frames == 0:
                continue
            blocks = np.asarray(wave[: frames * HOP], dtype=np.float64).reshape(frames, HOP)
            db = 20 * np.log10(np.sqrt(np.maximum((blocks ** 2).mean(axis=1), 1e-24)) + 1e-24)
            pause_mask = db < float(np.max(db)) - 35.0
            v = int(E.frame_analysis(wave)["voiced"].sum())
            total += frames
            pause += int(pause_mask.sum())
            voiced += v
            weighted_pause += 100.0 * float(pause_mask.mean()) * frames
        dur[name] = {
            "total_frames": total,
            "pause_frames": pause,
            "nonpause_frames": total - pause,
            "voiced_frames": voiced,
            "pause_pct_macro": analysis["duration"][name]["pause_pct"],
            "pause_pct_weighted": round(weighted_pause / max(1, total), 3),
            "sum_check_ok": pause <= total,
            "voiced_minus_nonpause": voiced - (total - pause),
        }
    out["duration"] = dur

    # ---------- 4. sensitivity: trung bình vs từng câu ----------
    rows = sens["rows"]
    def rho(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return float("nan")
        rx = np.argsort(np.argsort(x[m])).astype(float)
        ry = np.argsort(np.argsort(y[m])).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
        return float((rx * ry).sum() / den) if den else float("nan")

    sens_out = {}
    for kind, sev in (("noise", {"noise30": 1, "noise20": 2, "noise10": 3}),
                      ("drop", {"drop2": 1, "drop5": 2, "drop10": 3})):
        ids = sorted({r["id"] for r in rows if r["kind"] in sev})
        block = {}
        for metric, get in (("bak", lambda r: r["dnsmos"]["bak_mos"]), ("utmos", lambda r: r["utmos"]),
                            ("cer_clean", lambda r: r["cer_vs_clean"])):
            per = []
            for key in ids:
                sub = sorted([r for r in rows if r["id"] == key and r["kind"] in sev], key=lambda r: sev[r["kind"]])
                per.append(rho([sev[r["kind"]] for r in sub], [get(r) for r in sub]))
            means = {k: float(np.mean([get(r) for r in rows if r["kind"] == k])) for k in sev}
            block[metric] = {
                "mean_by_level": {k: round(v, 3) for k, v in means.items()},
                "per_sentence_rho": [round(v, 2) for v in per],
                "median_rho": round(float(np.nanmedian(per)), 2),
                "monotone_all": bool(np.all(np.isfinite(per)) and np.all(np.array(per) < 0)),
            }
        sens_out[kind] = block
    out["sensitivity"] = sens_out

    # ---------- 5. bootstrap: trung vị vs trung bình ----------
    rows_a = analysis_rows(judges, items)
    videos = sorted({r["video"] for r in rows_a})
    rng = np.random.default_rng(0)
    boot_med, boot_mean = [], []
    for _ in range(2000):
        chosen = rng.choice(len(videos), len(videos), replace=True)
        selected = {videos[i] for i in chosen}
        subset = [r["d_bak"] for r in rows_a if r["video"] in selected]
        boot_med.append(float(np.median(subset)))
        boot_mean.append(float(np.mean(subset)))
    out["bootstrap"] = {
        "videos": len(videos),
        "sentences_per_video": sorted({sum(1 for r in rows_a if r["video"] == v) for v in videos}),
        "median_ci95": [round(float(np.percentile(boot_med, 2.5)), 4), round(float(np.percentile(boot_med, 97.5)), 4)],
        "mean_ci95": [round(float(np.percentile(boot_mean, 2.5)), 4), round(float(np.percentile(boot_mean, 97.5)), 4)],
        "reported_in_v1833": analysis["summary"]["d_bak"]["ci95"],
        "reported_quantity": "median delta (analysis dùng np.nanmedian)",
    }

    Path("artifacts/bench/v1834_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


_FLAT_CACHE: dict[tuple[str, str], tuple[float, float]] = {}


def _flat_hnr(name: str, key: str) -> tuple[float, float]:
    """flat_V và HNR của một file (tính lại từ WAV — không chấm lại bằng giám khảo)."""
    cache_key = (name, key)
    if cache_key not in _FLAT_CACHE:
        wave, _ = sf.read(AUDIO / name / f"{key}.wav", dtype="float32")
        metrics = E.frame_analysis(wave)
        flat = float(np.median(metrics["flat"][metrics["voiced"]])) if metrics["voiced"].any() else float("nan")
        hnr = float(np.nanmedian(metrics["hnr"])) if np.isfinite(metrics["hnr"]).any() else float("nan")
        _FLAT_CACHE[cache_key] = (flat, hnr)
    return _FLAT_CACHE[cache_key]


def analysis_rows(judges, items):
    p2, g1 = judges["per_file"]["p2"], judges["per_file"]["g1"]
    rows = []
    for key in p2:
        flat_p2, hnr_p2 = _flat_hnr("p2", key)
        flat_g1, hnr_g1 = _flat_hnr("g1", key)
        rows.append({
            "video": items[key]["video"],
            "d_bak": g1[key]["dnsmos"]["bak_mos"] - p2[key]["dnsmos"]["bak_mos"],
            "d_sig": g1[key]["dnsmos"]["sig_mos"] - p2[key]["dnsmos"]["sig_mos"],
            "d_ovrl": g1[key]["dnsmos"]["ovrl_mos"] - p2[key]["dnsmos"]["ovrl_mos"],
            "d_utmos": g1[key]["utmos"] - p2[key]["utmos"],
            "d_cer": g1[key]["cer"] - p2[key]["cer"],
            "d_wer": g1[key]["wer"] - p2[key]["wer"],
            "d_flat": flat_g1 - flat_p2,
            "d_hnr": hnr_g1 - hnr_p2,
        })
    return rows


if __name__ == "__main__":
    main()
