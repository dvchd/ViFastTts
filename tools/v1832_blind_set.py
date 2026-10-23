"""tools/v1832_blind_set.py — v18.32: bộ nghe mù P2 vs G1 (≥30 câu, ≥6 video).

Chuẩn bị:
- Chọn câu theo video (12 video), phủ ngắn/vừa/dài, có khoảng nghỉ, câu nhỏ, câu từng nghe thấy rè.
- Render **P2** và **G1** bằng đúng production path (`iter_synthesize_sentence`, α=0).
- **WAV thật** làm tham chiếu riêng (không nằm trong A/B).
- Hai phiên bản: **gốc** (không đổi gì) và **cân mức** (một hệ số gain tuyến tính về −23 LUFS,
  có chốt đỉnh −1 dBTP; KHÔNG denoise/gate/compressor/limiter động).
- Ẩn tên checkpoint, đảo A/B mỗi câu, mapping lưu riêng (`mapping.json`, không nhúng vào HTML).
- Trang chấm 5 mục: rè nền · rõ chữ · tự nhiên · ngắt nghỉ · lỗi phát âm (1–5) + thích A/B/hoà.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

HOP = 256
CFG32 = "configs/small_codelung_v32.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
G1 = "runs/v1830_g1/ckpt_2000.pt"
TARGET_LUFS = -23.0
CEILING_DBTP = -1.0


def match_gain(wave: np.ndarray, target_lufs: float = TARGET_LUFS) -> tuple[np.ndarray, dict]:
    """Một hệ số gain tuyến tính về `target_lufs`, chốt đỉnh `CEILING_DBTP` (không limiter)."""
    from vifasttts.runtime.loudness import measure_lufs, true_peak_dbtp

    lufs = measure_lufs(wave, 24000)
    gain_db = target_lufs - lufs if np.isfinite(lufs) else 0.0
    peak = true_peak_dbtp(wave, 24000)
    if np.isfinite(peak) and peak + gain_db > CEILING_DBTP:
        gain_db = CEILING_DBTP - peak
    out = (wave.astype(np.float64) * (10 ** (gain_db / 20.0))).astype(np.float32)
    return out, {"lufs_in": round(float(lufs), 2), "gain_db": round(float(gain_db), 2),
                 "peak_in_dbtp": round(float(peak), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default="data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--out", default="artifacts/listen_small_filt/v1832_blind")
    parser.add_argument("--per-video", type=int, default=3)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer

    rows = [json.loads(x) for x in Path(args.val).read_text(encoding="utf-8").splitlines() if x.strip()]
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(str(row["id"]).split("_")[0], []).append(row)

    # câu từng nghe thấy rè (trang v1828/v1830)
    prior: set[str] = set()
    for path in ("artifacts/listen_small_filt/v1828_gate/key.json",
                 "artifacts/listen_small_filt/v1830_gan/key.json"):
        if Path(path).exists():
            key = json.loads(Path(path).read_text(encoding="utf-8"))
            prior.update(item["row"] for item in key["items"].values())

    def bucket(row: dict) -> str:
        frames = int(row["frame_range"][1]) - int(row["frame_range"][0])
        return "short" if frames < 150 else ("mid" if frames < 300 else "long")

    def has_pause(row: dict) -> bool:
        return sum(int(g) for g in row.get("gaps") or []) > 0

    picked: list[dict] = []
    for items in by_video.values():
        chosen = 0
        for want in ("short", "mid", "long"):
            candidates = [r for r in items if bucket(r) == want and len(r["syllables"]) >= 4]
            if not candidates:
                continue
            candidates.sort(key=lambda r: (r["id"] not in prior, not has_pause(r), r["id"]))
            picked.append(candidates[0])
            chosen += 1
            if chosen >= args.per_video:
                break
    # bù nếu thiếu
    if len(picked) < 30:
        for row in rows:
            if row not in picked and len(row["syllables"]) >= 4:
                picked.append(row)
            if len(picked) >= 30:
                break

    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    p2 = Synthesizer(cfg, P2, device=args.device)
    g1 = Synthesizer(cfg, G1, device=args.device)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "audio").mkdir(parents=True)
    rng = random.Random(args.seed)
    items, mapping = [], {"checkpoints": {"P2": P2, "G1": G1}, "items": {}}

    def render(engine: Synthesizer, row: dict) -> np.ndarray:
        text = " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])
        pieces = [audio for audio, _ in engine.iter_synthesize_sentence(text)]
        return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)

    for index, row in enumerate(picked, start=1):
        sid = f"q{index:02d}"
        start, end = row["frame_range"]
        real, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP, frames=(end - start) * HOP)
        real = np.asarray(real, dtype=np.float32)
        p2_wave = render(p2, row)
        g1_wave = render(g1, row)
        order = ["P2", "G1"]
        rng.shuffle(order)
        labels = {"A": order[0], "B": order[1]}
        mapping["items"][sid] = {"row": row["id"], "labels": labels,
                                 "text": " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])}
        entry = {"sid": sid, "row": row["id"], "video": str(row["id"]).split("_")[0],
                 "frames_gt": int(end) - int(start), "bucket": bucket(row),
                 "has_pause": has_pause(row), "prior_page": row["id"] in prior,
                 "text": mapping["items"][sid]["text"], "files": {}, "meta": {}}
        for label, name in labels.items():
            wave = p2_wave if name == "P2" else g1_wave
            matched, meta = match_gain(wave)
            for variant, data in (("orig", wave), ("matched", matched)):
                path = f"audio/{sid}_{label}_{variant}.wav"
                sf.write(out / path, data, 24000)
                entry["files"][f"{label}_{variant}"] = path
                if variant == "matched":
                    entry["meta"][label] = meta
        for variant, data in (("orig", real), ("matched", match_gain(real)[0])):
            path = f"audio/{sid}_real_{variant}.wav"
            sf.write(out / path, data, 24000)
            entry["files"][f"real_{variant}"] = path
        entry["duration_s"] = {"p2": round(len(p2_wave) / 24000, 2), "g1": round(len(g1_wave) / 24000, 2),
                               "real": round(len(real) / 24000, 2)}
        items.append(entry)

    (out / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    def select(sid: str, label: str, variant: str) -> str:
        return f"audio/{sid}_{label}_{variant}.wav"

    blocks = []
    for entry in items:
        sid = entry["sid"]
        rows_html = []
        for label in ("A", "B"):
            rows_html.append(
                f"<td><div class='lbl'>{label}</div>"
                f"<div class='sub'>gốc</div><audio controls preload='none' src='{select(sid, label, 'orig')}'></audio>"
                f"<div class='sub'>−23 LUFS</div><audio controls preload='none' src='{select(sid, label, 'matched')}'></audio>"
                + "".join(
                    f"<div class='score'>{name} <select data-sid='{sid}' data-label='{label}' data-k='{key}'>{''.join(f'<option value={v}>{v}</option>' for v in range(1, 6))}</select></div>"
                    for name, key in (("rè nền", "re"), ("rõ chữ", "ro"), ("tự nhiên", "tu"),
                                      ("ngắt nghỉ", "ngat"), ("lỗi phát âm", "loi"))
                )
                + "</td>"
            )
        real_cell = (
            f"<td class='ref'><div class='lbl'>Thật</div>"
            f"<div class='sub'>gốc</div><audio controls preload='none' src='{select(sid, 'real', 'orig')}'></audio>"
            f"<div class='sub'>−23 LUFS</div><audio controls preload='none' src='{select(sid, 'real', 'matched')}'></audio></td>"
        )
        pref = (f"<div class='pref'>Thích: "
                f"<select data-sid='{sid}' data-label='pref' data-k='pref'>"
                f"<option value=''>—</option><option value='A'>A</option><option value='B'>B</option>"
                f"<option value='hoa'>hoà</option></select></div>")
        blocks.append(
            f"<h3>{sid} · {entry['video']} · {entry['bucket']} · {entry['frames_gt']}f"
            f"{' · có nghỉ' if entry['has_pause'] else ''}{' · từng nghe' if entry['prior_page'] else ''}</h3>"
            f"<div class='txt'>{entry['text']}</div>"
            f"<table><tr>{''.join(rows_html)}{real_cell}</tr></table>{pref}"
        )

    html = f"""<!doctype html><meta charset='utf-8'><title>v18.32 nghe mù P2 vs G1</title>
<style>
body{{font:15px system-ui;background:#111;color:#eee;max-width:1400px;margin:20px auto}}
td{{padding:8px;vertical-align:top;border-bottom:1px solid #333}}audio{{width:240px}}
.lbl{{font-weight:700}}.sub{{font-size:12px;color:#999;margin-top:4px}}
.score{{font-size:12px;color:#bbb;margin-top:3px}}select{{background:#222;color:#eee;border:1px solid #444;border-radius:4px}}
.ref{{background:#181818}}.txt{{color:#bbb;font-size:13px;margin:4px 0 6px}}
h3{{margin-top:26px}}button{{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:10px 16px;font-size:15px}}
</style>
<h2>v18.32 — nghe MÙ: A/B (2 bản model) + Thật</h2>
<p>Mỗi câu: A/B là hai bản model (ẩn tên), mỗi bản có <b>gốc</b> và <b>−23 LUFS</b>; cột <b>Thật</b> là bản ghi gốc để tham chiếu.
Chấm <b>1 = không lỗi … 5 = lỗi nặng</b> cho rè nền / lỗi phát âm; <b>1 = kém … 5 = tốt</b> cho rõ chữ / tự nhiên / ngắt nghỉ.</p>
<p><b>Không</b> denoise/gate/compressor/limiter: bản −23 LUFS chỉ nhân một hệ số gain, chốt đỉnh −1 dBTP.</p>
{''.join(blocks)}
<p><button id='export'>Xuất kết quả JSON</button> <span id='status'></span></p>
<script>
const KEY='v1832_scores';
function collect(){{const out={{}};document.querySelectorAll('select[data-sid]').forEach(s=>{{
  const id=s.dataset.sid, lab=s.dataset.label, k=s.dataset.k;
  out[id]=out[id]||{{}}; out[id][lab]=out[id][lab]||{{}}; out[id][lab][k]=s.value;}});return out;}}
document.querySelectorAll('select').forEach(s=>s.addEventListener('change',()=>localStorage.setItem(KEY,JSON.stringify(collect()))));
document.getElementById('export').addEventListener('click',()=>{{
  const blob=new Blob([JSON.stringify(collect(),null,1)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v1832_scores.json';a.click();
  document.getElementById('status').textContent='đã xuất';}});
</script>
"""
    (out / "index.html").write_text(html, encoding="utf-8")
    summary = {
        "items": len(items),
        "videos": len({entry["video"] for entry in items}),
        "buckets": {key: sum(1 for entry in items if entry["bucket"] == key) for key in ("short", "mid", "long")},
        "with_pause": sum(1 for entry in items if entry["has_pause"]),
        "prior_page": sum(1 for entry in items if entry["prior_page"]),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
