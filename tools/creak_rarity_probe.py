#!/usr/bin/env python
"""tools/creak_rarity_probe.py — kiểm định giả thuyết "âm tiết HIẾM ⇒ rè" (18/10/2026).

Bối cảnh: tai người dùng nghe cặp tối thiểu ("Tôi đọc chữ ___ một lần.") thấy *một số* từ rè,
một số không; các từ rè nặng nhất gần như KHÔNG có trong train (bão ×1 · hộc ×0 · khanh ×0…).
Tool này kiểm soát biến số: **cùng câu khung, cùng thanh (NGÃ/NẶNG)**, chỉ khác **tần suất
xuất hiện trong train** — synth bằng model + Vocos, đo phonation (`hnr`/`jitter`/`glottalization`)
trên khung hữu thanh (hàm chuẩn của repo: `data/phonation.py`).

Dùng:
  .venv/bin/python tools/creak_rarity_probe.py --config runs/acoustic_v2_b8/resolved_config.json \\
      --checkpoint runs/acoustic_v2_b8/best.pt \\
      --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \\
      --out artifacts/creak_rarity/probe_b8.json --device mps

Nhóm: RARE (count ≤ --rare-max) vs COMMON (count ≥ --common-min) cùng thanh NGA/NANG, thêm
NGANG-common làm mốc nền. Kết quả JSON + bảng in; wav từng từ ở `<out>.wav/`.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARRIER = "Tôi đọc chữ {word} một lần."
#: Chỉ nhận âm tiết thuần chữ Việt (loại tên riêng/respelling tiếng Anh khỏi mẫu).
VIET_RE = re.compile(r"^[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
                     r"ùúủũụưừứửữựỳýỷỹỵ]+$")


def build_freq(manifest: str) -> tuple[Counter, dict[str, str]]:
    """Đếm tần suất từng âm tiết (`normalized`) + thanh chủ đạo của nó."""
    counts: Counter = Counter()
    tones: dict[str, Counter] = defaultdict(Counter)
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for syl in row.get("syllables", []):
            word = str(syl.get("normalized") or syl.get("raw") or "").lower().strip()
            if not word or not VIET_RE.match(word):
                continue
            counts[word] += 1
            tone = (syl.get("abstract") or {}).get("tone")
            if tone:
                tones[word][tone] += 1
    tone_of = {w: c.most_common(1)[0][0] for w, c in tones.items()}
    return counts, tone_of


def measure(wav_path: Path, config: dict) -> dict:
    """Đo phonation trung bình trên khung hữu thanh (hàm chuẩn repo)."""
    from vifasttts.data.features import estimate_f0_pyworld, load_wav
    from vifasttts.data.phonation import extract_phonation_frames

    sample_rate = int(config.get("sample_rate", 24000))
    hop = int(config.get("mel", {}).get("hop_length", 256))
    f0cfg = dict(config.get("f0", {}) or {})
    wav = load_wav(wav_path, sample_rate)
    f0, vuv = estimate_f0_pyworld(wav, sample_rate, hop,
                                  f_min=float(f0cfg.get("f_min", 70.0)),
                                  f_max=float(f0cfg.get("f_max", 400.0)),
                                  interpolate=bool(f0cfg.get("interpolate", True)))
    frames = extract_phonation_frames(wav, config, f0=f0, vuv=vuv)
    mask = vuv > 0.5
    return {k: float(frames[k][mask].mean()) if mask.any() else float("nan")
            for k in ("hnr", "jitter", "glottalization")}


def main() -> None:
    parser = argparse.ArgumentParser(description='Kiểm định "âm tiết hiếm ⇒ rè" (cùng khung, cùng thanh).')
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-per-group", type=int, default=12)
    parser.add_argument("--rare-max", type=int, default=5)
    parser.add_argument("--common-min", type=int, default=150)
    parser.add_argument("--sharpen-alpha", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import yaml

    from vifasttts.infer import Synthesizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    counts, tone_of = build_freq(args.manifest)
    rng = random.Random(args.seed)

    def pick(tone_set: set[str], lo: int, hi: int) -> list[str]:
        cands = sorted(w for w, c in counts.items() if tone_of.get(w) in tone_set and lo <= c <= hi)
        rng.shuffle(cands)
        return cands[: args.n_per_group]

    groups = {
        "rare": pick({"NGA", "NANG"}, 0, args.rare_max),
        "common": pick({"NGA", "NANG"}, args.common_min, 10**9),
        "ngang": pick({"NGANG"}, args.common_min, 10**9),
    }

    out = Path(args.out)
    wav_dir = out.with_suffix(out.suffix + ".wav")
    wav_dir.mkdir(parents=True, exist_ok=True)
    engine = Synthesizer(config, args.checkpoint, device=args.device)

    report: dict = {"checkpoint": args.checkpoint, "manifest": args.manifest,
                    "sharpen_alpha": args.sharpen_alpha, "groups": {}, "per_word": []}
    for name, words in groups.items():
        rows = []
        for word in words:
            target = wav_dir / f"{name}_{word}.wav"
            engine.synthesize(CARRIER.format(word=word), target, sharpen_alpha=args.sharpen_alpha)
            metrics = measure(target, config)
            row = {"group": name, "word": word, "train_count": counts[word], "tone": tone_of[word],
                   **metrics}
            rows.append(row)
            report["per_word"].append(row)
            print(f"  [{name:6s}] {word:10s} ×{counts[word]:<5} hnr {metrics['hnr']:+6.2f} "
                  f"jitter {metrics['jitter']:.3f} glott {metrics['glottalization']:.4f}")
        summary = {}
        for key in ("hnr", "jitter", "glottalization"):
            values = [r[key] for r in rows if r[key] == r[key]]
            summary[key] = round(sum(values) / len(values), 4) if values else None
        report["groups"][name] = {"n": len(rows), "words": words, **summary}
        print(f"== {name}: n={len(rows)} · hnr {summary['hnr']:+.2f} · jitter {summary['jitter']:.3f} "
              f"· glott {summary['glottalization']:.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out} · wav: {wav_dir}/")


if __name__ == "__main__":
    main()
