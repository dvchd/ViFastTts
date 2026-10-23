"""tools/v1833_render_eval.py — v18.33: render P2/G1 cho bộ 300 câu stratified (free-run).

Cùng quy tắc chọn câu như v18.31 (`--stratify`: round-robin theo prefix video), cùng
production path (`iter_synthesize_sentence`, α=0), cùng checkpoint/config đã ghi ở
`configs/production_v1831.json` (P2 = rollback, G1 = production candidate).

Ghi WAV float32 24 kHz + `items.json` (id, video, text, bucket, đường dẫn).
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

HOP = 256
CFG32 = "configs/small_codelung_v32.yaml"
MODELS = {
    "p2": "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt",
    "g1": "runs/v1830_g1/ckpt_2000.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default="data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--out", default="artifacts/bench/v1833_audio")
    parser.add_argument("--segments", type=int, default=300)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer

    rows = [json.loads(x) for x in Path(args.val).read_text(encoding="utf-8").splitlines() if x.strip()]
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        groups.setdefault(str(row["id"]).split("_")[0], []).append(row)
    picked, cursor = [], {key: 0 for key in groups}
    while len(picked) < args.segments and any(cursor[k] < len(groups[k]) for k in groups):
        for key, items in groups.items():
            if cursor[key] < len(items) and len(picked) < args.segments:
                picked.append(items[cursor[key]])
                cursor[key] += 1

    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    out = Path(args.out)
    for name in (*MODELS, "real"):
        (out / name).mkdir(parents=True, exist_ok=True)
    engines = {name: Synthesizer(cfg, path, device=args.device) for name, path in MODELS.items()}
    items = []
    for row in picked:
        text = " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])
        start, end = row["frame_range"]
        real, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP, frames=(end - start) * HOP)
        sf.write(out / "real" / f"{row['id']}.wav", np.asarray(real, dtype=np.float32), 24000)
        for name, engine in engines.items():
            pieces = [a for a, _ in engine.iter_synthesize_sentence(text)]
            wave = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            sf.write(out / name / f"{row['id']}.wav", np.asarray(wave, dtype=np.float32), 24000)
        items.append({
            "id": row["id"], "video": str(row["id"]).split("_")[0],
            "text": text, "text_manifest": row.get("text", text),
            "frames_gt": int(end) - int(start),
            "bucket": "short" if (end - start) < 150 else ("mid" if (end - start) < 300 else "long"),
        })
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"items": len(items), "videos": len({i['video'] for i in items})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
