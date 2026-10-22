"""tools/v1828_listen_page.py — v18.28: render 9 câu × N bản + trang nghe mù (xáo nhãn).

Mỗi câu: copy-synth (TF, mốc) + P2 gốc + P2+expander + GTA tốt nhất (+ GTA+expander).
Tất cả chuẩn hoá **−23 LUFS**, α=0, tên tệp ngẫu nhiên, nhãn A..E xáo theo câu; đáp án ở `key.json`
(không nhúng vào HTML).

Chấm: (a) rè nền, (b) hụt hơi ở khoảng lặng, (c) từ lạc quẻ — thang 1–5.
Trang có nút xuất JSON kết quả (tải về máy).
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
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

HOP = 256
CFG32 = "configs/small_codelung_v32.yaml"
CFG35 = "configs/small_codelung_v35.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
V35 = "runs/acoustic_v35/best_ema.pt"
LABELS = "ABCDEFGH"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", default="data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--min-syllables", type=int, default=9)
    parser.add_argument("--gta", default=None, help="checkpoint Vocos GTA tốt nhất (vocoder-only)")
    parser.add_argument("--with-exp", action="store_true", help="thêm bản GTA + expander −45/1:2")
    parser.add_argument("--out", default="artifacts/listen_small_filt/v1828_gate")
    parser.add_argument("--target-lufs", type=float, default=-23.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from vifasttts.data.cache import FeatureCache
    from vifasttts.infer import Synthesizer, _apply_target_lufs
    from vifasttts.runtime.postprocess import sharpen_mel

    rows = [json.loads(x) for x in Path(args.val).read_text(encoding="utf-8").splitlines() if x.strip()]
    picked = []
    seen = set()
    for row in rows:
        if row.get("parent") in seen or len(row["syllables"]) < args.min_syllables:
            continue
        seen.add(row.get("parent"))
        picked.append(row)
        if len(picked) >= args.rows:
            break
    if len(picked) < args.rows:
        picked = rows[: args.rows]

    cfg32 = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    cfg35 = yaml.safe_load(Path(CFG35).read_text(encoding="utf-8"))
    cache35 = FeatureCache("cache/codelung_v35", cfg35)
    copy_engine = Synthesizer(cfg35, V35, device=args.device)
    p2_engine = Synthesizer(cfg32, P2, device=args.device)
    gta_engine = Synthesizer(cfg32, P2, vocoder_checkpoint=args.gta, device=args.device) if args.gta else None

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "audio").mkdir(parents=True)
    rng = random.Random(args.seed)
    key: dict = {"variants": {}, "items": {}}

    def render_copy(row: dict) -> np.ndarray:
        start, end = row["frame_range"]
        mel = cache35.get(row)["mel"][start:end].to(copy_engine.device)
        with torch.no_grad():
            audio = copy_engine.vocoder(sharpen_mel(mel, 0.0).unsqueeze(0))[0]
        return np.asarray(audio.float().cpu()).reshape(-1)

    def render_free(engine: Synthesizer, row: dict) -> np.ndarray:
        from vifasttts.frontend.pipeline import parse_text

        text = " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])
        del parse_text
        pieces = [audio for audio, _ in engine.iter_synthesize_sentence(text)]
        return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)

    def expander(x: np.ndarray, threshold_db: float = -45.0, ratio: float = 2.0) -> np.ndarray:
        block = 64
        n = len(x) // block
        if n == 0:
            return x
        rms = np.sqrt(np.mean(x[: n * block].reshape(n, block) ** 2, axis=1) + 1e-12)
        level = 20 * np.log10(rms + 1e-12)
        atten = np.where(level < threshold_db, np.minimum((threshold_db - level) * (ratio - 1.0), 12.0), 0.0)
        a_att = float(np.exp(-block / (24000 * 0.002)))
        a_rel = float(np.exp(-block / (24000 * 0.030)))
        gain = np.empty(n)
        prev = 0.0
        for i in range(n):
            coeff = a_att if -atten[i] < prev else a_rel
            prev = coeff * prev + (1.0 - coeff) * (-atten[i])
            gain[i] = prev
        result = x.copy()
        result[: n * block] *= (10 ** (gain / 20.0)).repeat(block)
        return result

    for index, row in enumerate(picked, start=1):
        sid = f"s{index:02d}"
        text = " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])
        variants = {
            "copy": render_copy(row),
            "p2": render_free(p2_engine, row),
            "p2_exp45": expander(render_free(p2_engine, row)),
        }
        if gta_engine is not None:
            gta_audio = render_free(gta_engine, row)
            variants["gta"] = gta_audio
            if args.with_exp:
                variants["gta_exp45"] = expander(gta_audio)
        names = list(variants)
        rng.shuffle(names)
        entries = []
        for label_index, name in enumerate(names):
            label = LABELS[label_index]
            token = f"{sid}_{rng.randrange(16**6):06x}"
            audio = _apply_target_lufs(variants[name].astype(np.float32), 24000, args.target_lufs)
            sf.write(out / "audio" / f"{token}.wav", audio, 24000)
            entries.append({"label": label, "file": f"audio/{token}.wav", "role": name})
            key["variants"].setdefault(name, 0)
            key["variants"][name] += 1
        key["items"][sid] = {"row": row["id"], "text": text, "entries": entries}

    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    options = "".join(f"<option value='{value}'>{value}</option>" for value in range(1, 6))
    cards = []
    for sid, item in key["items"].items():
        columns = "".join(
            f"<td><div class='lbl'>{entry['label']}</div><audio controls preload='none' "
            f"src='{entry['file']}'></audio>"
            f"<div class='score'>rè <select data-sid='{sid}' data-label='{entry['label']}' data-k='re'>{options}</select> "
            f"hụt hơi <select data-sid='{sid}' data-label='{entry['label']}' data-k='hut'>{options}</select> "
            f"lạc <select data-sid='{sid}' data-label='{entry['label']}' data-k='lac'>{options}</select></div></td>"
            for entry in item["entries"]
        )
        text = item["text"]
        short = text if len(text) <= 110 else text[:107] + "…"
        cards.append(f"<h3>{sid} — {short}</h3><table><tr>{columns}</tr></table>")

    html = f"""<!doctype html><meta charset='utf-8'><title>v18.28 nghe mù — GTA + expander</title>
<style>
body{{font:15px system-ui;background:#111;color:#eee;max-width:1250px;margin:20px auto}}
td{{padding:8px;vertical-align:top;border-bottom:1px solid #333}}audio{{width:210px}}
.lbl{{font-weight:700;margin-bottom:4px}}.score{{font-size:12px;color:#bbb;margin-top:4px}}
select{{background:#222;color:#eee;border:1px solid #444;border-radius:4px}}
button{{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:10px 16px;font-size:15px;cursor:pointer}}
h3{{margin-top:26px}}code{{color:#9ae6b4}}
</style>
<h2>v18.28 — nghe MÙ: 5 bản/câu (copy-synth · P2 · P2+expander · GTA · GTA+expander)</h2>
<p>−23 LUFS · α=0 · nhãn xáo theo từng câu. Chấm <b>1 = tệ … 5 = tốt</b>:
<b>rè nền</b> (5 = không rè) · <b>hụt hơi</b> (5 = không hụt) · <b>từ lạc quẻ</b> (5 = không lạc).</p>
<div id='cards'>{''.join(cards)}</div>
<p><button id='export'>Xuất kết quả JSON</button> <span id='status'></span></p>
<script>
const KEY = 'v1828_scores';
function save() {{
  const out = {{}};
  document.querySelectorAll('select[data-sid]').forEach(s => {{
    const id = s.dataset.sid;
    out[id] = out[id] || {{}};
    out[id][s.dataset.label] = out[id][s.dataset.label] || {{}};
    out[id][s.dataset.label][s.dataset.k] = Number(s.value);
  }});
  localStorage.setItem(KEY, JSON.stringify(out));
}}
document.querySelectorAll('select').forEach(s => s.addEventListener('change', save));
document.getElementById('export').addEventListener('click', () => {{
  save();
  const blob = new Blob([localStorage.getItem(KEY) || '{{}}'], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'v1828_scores.json';
  a.click();
  document.getElementById('status').textContent = 'đã xuất';
}});
</script>
"""
    (out / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({"out": str(out), "items": len(key["items"]),
                      "variants": key["variants"], "gta": args.gta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
