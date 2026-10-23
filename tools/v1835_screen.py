"""tools/v1835_screen.py — v18.35: sàng lọc checkpoint hiện có (không training).

Bộ sàng lọc: 120 câu cân bằng theo video (12 × 10) và độ dài (4 ngắn / 3 vừa / 3 dài mỗi video),
**loại** các câu đã dùng ở v1831/v1832/v1833 (300 stratified + 36 blind + 300 head cũ).
Giám khảo: DNSMOS BAK + UTMOSv2 (chính), SIG/OVRL + flat_V/HNR (bổ trợ). Không dùng CER/WER.

Luật (đăng ký trước):
- Ứng viên ĐẠT sàng lọc nếu: trung vị ΔBAK (ứng viên − P2) > 0 với CI95 bootstrap cụm video
  không chứa 0; ΔUTMOS CI95 dưới > −0,02; ≥ 2/3 số video có ΔBAK ≥ 0.
- Nhiều ứng viên đạt: chọn ΔBAK trung vị lớn nhất trong nhóm có ΔUTMOS ≥ 0.
- Không ai đạt: giữ P2, kết thúc.
- Xác nhận: đúng 1 ứng viên trên tập C rời rạc (120 câu khác), cùng luật; chỉ đề xuất, không đổi production.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import v1828_gta_eval as E

from vifasttts.eval_stats import outcome_counts

HOP = 256
CFG32 = "configs/small_codelung_v32.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
CANDIDATES = {
    "g1_1k": "runs/v1830_g1/ckpt_1000.pt",
    "g2_1k": "runs/v1830_g2/ckpt_1000.pt",
    "p2_4k": "runs/adv_p2_20k/eval_4000.pt",
    "g2_2k": "runs/v1830_g2/ckpt_2000.pt",
}


def used_ids() -> set[str]:
    used: set[str] = set()
    for path in ("artifacts/bench/v1833_audio/items.json", "artifacts/listen_small_filt/v1832_blind/items.json"):
        if Path(path).exists():
            for item in json.loads(Path(path).read_text(encoding="utf-8")):
                used.add(item.get("id") or item.get("row"))
    # 300 head cũ
    rows = [json.loads(x) for x in Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    used.update(row["id"] for row in rows[:300])
    return used


def pick_rows(n_per_video: int, used: set[str], exclude: set[str] | None = None) -> list[dict]:
    rows = [json.loads(x) for x in Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        if row["id"] in used or (exclude and row["id"] in exclude):
            continue
        groups.setdefault(str(row["id"]).split("_")[0], []).append(row)
    picked: list[dict] = []
    for items in groups.values():
        short = [r for r in items if (r["frame_range"][1] - r["frame_range"][0]) < 150]
        mid = [r for r in items if 150 <= (r["frame_range"][1] - r["frame_range"][0]) < 300]
        long_ = [r for r in items if (r["frame_range"][1] - r["frame_range"][0]) >= 300]
        quota = [("short", short, 4), ("mid", mid, 3), ("long", long_, 3)]
        take = []
        for _name, pool, want in quota:
            take.extend(pool[:want])
        if len(take) < n_per_video:
            rest = [r for r in items if r not in take]
            take.extend(rest[: n_per_video - len(take)])
        picked.extend(take[:n_per_video])
    return picked


def judge_dir(directory: Path, device: str) -> dict:
    import utmosv2

    model = utmosv2.create_model(pretrained=True, config="fusion_stage3", device=device)
    out = model.predict(input_dir=str(directory), device=device, batch_size=8, verbose=False)
    return {Path(item["file_path"]).stem: round(float(item["predicted_mos"]), 4) for item in out}


def dnsmos(wave: np.ndarray) -> dict:
    from scipy.signal import resample_poly
    from speechmos import dnsmos as dns

    w16 = resample_poly(wave.astype(np.float64), 16000, 24000).astype(np.float32)
    return {key: round(float(value), 4) for key, value in dns.run(w16, sr=16000).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--n-per-video", type=int, default=10)
    parser.add_argument("--candidates", default="g1_1k,g2_1k,p2_4k,g2_2k")
    parser.add_argument("--best", default=None, help="mode confirm: tag ứng viên duy nhất")
    parser.add_argument("--out", default="artifacts/bench/v1835_screen")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer

    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    out = Path(args.out)
    used = used_ids()
    tags = [tag for tag in args.candidates.split(",") if tag]
    if args.mode == "screen":
        rows = pick_rows(args.n_per_video, used)
        models = {"p2": P2, **{tag: CANDIDATES[tag] for tag in tags}}
    else:
        screen_ids = {item["id"] for item in json.loads((out / "items.json").read_text(encoding="utf-8"))}
        rows = pick_rows(args.n_per_video, used, exclude=screen_ids)
        models = {"p2": P2, args.best: CANDIDATES[args.best]}

    for name in models:
        (out / name).mkdir(parents=True, exist_ok=True)
    engines = {name: Synthesizer(cfg, path, device=args.device) for name, path in models.items()}
    items = []
    for row in rows:
        text = " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])
        for name, engine in engines.items():
            pieces = [a for a, _ in engine.iter_synthesize_sentence(text)]
            wave = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            sf.write(out / name / f"{row['id']}.wav", np.asarray(wave, dtype=np.float32), 24000)
        items.append({"id": row["id"], "video": str(row["id"]).split("_")[0],
                      "frames": int(row["frame_range"][1]) - int(row["frame_range"][0])})
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    scores = {}
    for name in models:
        utmos = judge_dir(out / name, args.device)
        per = {}
        for item in items:
            key = item["id"]
            wave, _ = sf.read(out / name / f"{key}.wav", dtype="float32")
            metrics = E.frame_analysis(wave)
            per[key] = {
                "utmos": utmos.get(key),
                "dnsmos": dnsmos(wave),
                "flat_v": round(float(np.median(metrics["flat"][metrics["voiced"]])), 4) if metrics["voiced"].any() else None,
                "hnr": round(float(np.nanmedian(metrics["hnr"])), 2) if np.isfinite(metrics["hnr"]).any() else None,
            }
        scores[name] = per

    def paired(tag: str) -> dict:
        rows_p = []
        for item in items:
            key = item["id"]
            a, b = scores["p2"][key], scores[tag][key]
            rows_p.append({
                "video": item["video"], "key": key,
                "d_bak": b["dnsmos"]["bak_mos"] - a["dnsmos"]["bak_mos"],
                "d_sig": b["dnsmos"]["sig_mos"] - a["dnsmos"]["sig_mos"],
                "d_ovrl": b["dnsmos"]["ovrl_mos"] - a["dnsmos"]["ovrl_mos"],
                "d_utmos": b["utmos"] - a["utmos"],
                "d_flat": b["flat_v"] - a["flat_v"],
                "d_hnr": b["hnr"] - a["hnr"],
            })
        videos = sorted({r["video"] for r in rows_p})
        rng = np.random.default_rng(0)
        boot = {field: [] for field in ("d_bak", "d_utmos", "d_sig", "d_ovrl", "d_flat", "d_hnr")}
        for _ in range(2000):
            chosen = rng.choice(len(videos), len(videos), replace=True)
            subset = [r for r in rows_p if r["video"] in {videos[i] for i in chosen}]
            for field in list(boot):
                boot[field].append(float(np.nanmedian([r[field] for r in subset])))
        summary = {}
        for field, higher_better in (("d_bak", True), ("d_utmos", True), ("d_sig", True), ("d_ovrl", True),
                                     ("d_flat", False), ("d_hnr", True)):
            values = np.array([r[field] for r in rows_p], dtype=float)
            summary[field] = {
                "median": round(float(np.nanmedian(values)), 4),
                "ci95": [round(float(np.percentile(boot[field], 2.5)), 4),
                         round(float(np.percentile(boot[field], 97.5)), 4)],
                **outcome_counts(values, higher_better=higher_better),
            }
        per_video = {video: round(float(np.median([r["d_bak"] for r in rows_p if r["video"] == video])), 4)
                     for video in videos}
        passed = (summary["d_bak"]["ci95"][0] > 0
                  and summary["d_utmos"]["ci95"][0] > -0.02
                  and sum(1 for v in per_video.values() if v >= 0) >= (2 * len(videos) + 2) // 3)
        return {"summary": summary, "per_video_bak": per_video, "passes": bool(passed),
                "videos_bak_nonneg": sum(1 for v in per_video.values() if v >= 0), "videos": len(videos)}

    result = {"mode": args.mode, "items": len(items), "videos": len({i["video"] for i in items}),
              "candidates": tags if args.mode == "screen" else [args.best], "results": {}}
    for tag in (tags if args.mode == "screen" else [args.best]):
        result["results"][tag] = paired(tag)
    (out / f"{args.mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
