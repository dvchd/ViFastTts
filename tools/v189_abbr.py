"""tools/v189_abbr.py — trang nghe cụm viết tắt (v18.9): audio THẬT, không gợi ý của model.

Mỗi cụm: 3 đoạn từ 3 row khác nhau, cắt theo MFA ±0,5 s; mỗi đoạn có nút chọn cách đọc TỪNG CHỮ
(bộ ứng viên v18.8) + ô "khác". Trang lưu localStorage và có nút tải JSON để gửi lại.
"""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

CAND = {
    "b": ["bê", "be", "bi"], "c": ["xê", "xi", "xe"], "d": ["dê", "đê", "đi", "di"],
    "đ": ["đê", "đi"], "p": ["pê", "pi", "pe"], "t": ["tê", "ti", "te"],
    "v": ["vê", "vi", "ve"], "r": ["rờ", "e rờ", "a", "a rờ"], "l": ["lờ", "e lờ", "eo"],
    "h": ["hát", "hắt", "ết", "ây", "ếch"], "m": ["em", "mờ"], "n": ["en", "nờ"],
    "s": ["ét", "ét xì", "es"], "x": ["ích", "ếch", "ex", "xờ"], "k": ["ca", "cây", "kây"],
    "a": ["a", "ây"], "u": ["u", "diu"], "i": ["i", "ai"],
}
READ = {v[0]: k for k, v in CAND.items()}
READ.update({"a": "a", "i": "i", "u": "u"})
OUT = Path("artifacts/listen_small_filt/v189_abbr")
TRAIN = Path("data/codelung/v35/train_edgelabel.jsonl")


def main() -> None:
    rows = [json.loads(x) for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    clusters: dict[str, list[dict]] = {}
    for row in rows:
        syl, durs = row["syllables"], row["durations"]
        gaps = list(row["gaps"]) + [0]
        cursor = int(row["frame_range"][0]) + int(row.get("lead_frames") or 0)
        i = 0
        while i < len(syl):
            raw = str(syl[i].get("raw") or "").strip().lower()
            if raw in READ:
                j, letters = i, []
                while j < len(syl) and len(letters) < 4:
                    token = str(syl[j].get("raw") or "").strip().lower()
                    if token not in READ:
                        break
                    letters.append(READ[token])
                    j += 1
                if len(letters) >= 2:
                    key = "".join(letters)
                    span = (cursor, cursor + sum(durs[i:j]) + sum(gaps[i:j - 1]))
                    clusters.setdefault(key, []).append({
                        "row": row["id"], "audio": row["audio_path"], "span": span,
                        "text": " ".join(str(s.get("raw")) for s in syl),
                    })
                cursor += sum(durs[i:j]) + sum(gaps[i:j - 1])
                i = j
            else:
                cursor += durs[i] + gaps[i]
                i += 1

    ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    chosen = [k for k, _ in ranked[:12]]
    if "vb" not in chosen and "vb" in clusters:
        chosen.append("vb")
    for letter in ("r", "đ", "h", "s", "x"):
        if any(letter in key for key in chosen):
            continue
        for key, _ in ranked:
            if letter in key:
                chosen.append(key)
                break
    chosen = list(dict.fromkeys(chosen))[:15]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(exist_ok=True)
    items = []
    for key in chosen:
        picks, used = [], set()
        for occurrence in clusters[key]:
            if occurrence["row"] in used:
                continue
            used.add(occurrence["row"])
            start, end = occurrence["span"]
            with sf.SoundFile(occurrence["audio"]) as handle:
                sr = handle.samplerate
                pad = int(0.5 * sr)
                handle.seek(max(0, start * 256 - pad))
                frames = (end - start) * 256 + 2 * pad
                audio = handle.read(frames, dtype="float32", always_2d=False)
            name = f"{key}_{len(picks)}.wav"
            sf.write(OUT / "audio" / name, audio, sr)
            picks.append({"file": f"audio/{name}", "text": occurrence["text"][:120]})
            if len(picks) == 3:
                break
        items.append({"cluster": key, "count": len(clusters[key]), "picks": picks})

    page = ["<!doctype html><meta charset='utf-8'><title>v18.9 nghe cụm viết tắt</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1200px;"
             "margin:20px auto}.clip{border:1px solid #333;padding:10px;margin:10px 0}"
             "audio{width:340px}label{margin-right:10px;white-space:nowrap}"
             "input[type=text]{width:120px}</style>"),
            "<h2>Nghe cụm viết tắt và chọn cách đọc</h2>",
            ("<p>Mỗi đoạn là audio THẬT cắt theo căn MFA ±0,5 s. Chọn cách đọc của TỪNG CHỮ rồi bấm "
             "“Tải JSON” ở cuối trang để gửi lại. Không có gợi ý nào của model trên trang.</p>")]
    for group in items:
        page.append(f"<h3>Cụm “{group['cluster'].upper()}” — {group['count']} lần trong train</h3>")
        for idx, pick in enumerate(group["picks"]):
            cid = f"{group['cluster']}_{idx}"
            page.append(f"<div class='clip' data-id='{cid}'><div>Câu: {pick['text']}…</div>"
                        f"<audio controls src='{pick['file']}'></audio><div class='choices'>")
            for letter in group["cluster"]:
                options = "".join(
                    f"<label><input type='radio' name='{cid}_{letter}' value='{opt}'> {opt}</label>"
                    for opt in CAND.get(letter, [letter]))
                page.append(f"<div><b>{letter.upper()}</b>: {options}"
                            f"<label>khác <input type='text' name='{cid}_{letter}_other'></label></div>")
            page.append("</div></div>")
    page.append("""<button id='save'>Tải JSON</button><pre id='log'></pre>
<script>
const KEY='v189_abbr_answers';
let store = JSON.parse(localStorage.getItem(KEY) || '{}');
document.querySelectorAll('.clip').forEach(clip => {
  const id = clip.dataset.id;
  clip.querySelectorAll('input').forEach(inp => {
    inp.addEventListener('change', () => {
      store[id] = store[id] || {};
      const [cid, letter] = inp.name.split('_');
      if (inp.type === 'radio') store[id][letter] = inp.value;
      else store[id][letter + '_other'] = inp.value;
      localStorage.setItem(KEY, JSON.stringify(store));
    });
  });
});
document.getElementById('save').onclick = () => {
  const blob = new Blob([JSON.stringify(store, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'v189_abbr_answers.json'; a.click();
  document.getElementById('log').textContent = 'đã tạo file, gửi lại cho Agent 2';
};
</script>""")
    (OUT / "index.html").write_text("\n".join(page), encoding="utf-8")
    meta = {"clusters": items, "candidates": CAND}
    Path("artifacts/bench/v189_abbr_meta.json").parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts/bench/v189_abbr_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(group["picks"]) for group in items)
    print(f"{len(items)} cụm · {total} đoạn · {OUT/'index.html'}")


if __name__ == "__main__":
    main()
