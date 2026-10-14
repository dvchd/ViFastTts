"""tools/check_synth_quality.py — lọc dữ liệu TỔNG HỢP trước khi đưa vào train (V11S3).

Hai tầng kiểm (bài học GTA §41 — audio AI vào train phải qua cổng):
1. **ASR vòng tròn**: transcribe lại bằng faster-whisper → so với text đã yêu cầu (CER ký tự).
   Lọc bỏ câu sai rõ (model đọc nhầm/nuốt chữ).
2. **Phân bố đặc trưng so giọng THẬT**: F0 trung vị · HNR trung vị · jitter trung vị · tỉ lệ hữu
   thanh · mean/std mel — so với phân bố của chính giọng gốc (lấy từ cache v2). Lệch lớn ⇒
   dữ liệu tổng hợp thuộc "miền" khác (phòng/mic/độ sạch) ⇒ rủi ro khi trộn vào train.

Chạy:
    .venv/bin/python tools/check_synth_quality.py --manifest artifacts/synth_data/pilot.jsonl \
        --limit 50 --out artifacts/synth_data/pilot_quality.json

Kết quả JSON: `{summary, per_clip:[{id, cer, f0_med, hnr_med, jitter_med, voiced, mel_mean, mel_std}],
real_reference:{...}, pass}` — `pass` = tỉ lệ câu đạt (CER ≤ 0,15 + trong dải giọng thật ±3σ).
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _transcribe(paths: list[str]) -> list[str]:
    """Transcribe danh sách wav bằng faster-whisper (CPU, int8) — trả text thường hoá."""
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    engine = WhisperModel("small", device="cpu", compute_type="int8")
    out = []
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            count = round(audio.size * 16000 / rate)
            audio = np.interp(np.linspace(0, audio.size - 1, count), np.arange(audio.size), audio)
        segments, _ = engine.transcribe(audio.astype("float32"), language="vi", beam_size=5)
        out.append(" ".join(segment.text for segment in segments))
    return out


#: Lớp ÂM ĐẦU ĐỒNG ÂM phương ngữ Bắc — ASR hay nhầm, KHÔNG phải TTS đọc sai (đo 15/10:
#: "rảnh"→"giảnh", "dẫm"→"rậm"). So phải gộp lớp, nếu không CER giả rất cao.
_ONSET_CLASS = {"r": "R", "d": "R", "gi": "R", "s": "S", "x": "S", "tr": "T", "ch": "T"}


def _syllable_key(word: str) -> str:
    """Khoá âm tiết bỏ khác biệt phương ngữ: (lớp đầu · vần · thanh) — không parse được ⇒ chữ thường."""
    from vifasttts.frontend.orthography import compose_canonical
    from vifasttts.frontend.syllable import safe_parse_syllable

    item = safe_parse_syllable(word)
    if item is None:
        return word.lower()
    onset, o, abstract = (item.orthography.onset or "").lower(), item.orthography, item.abstract
    rime = compose_canonical(None, o.medial, o.nucleus, o.coda, "NGANG")
    return f"{_ONSET_CLASS.get(onset, onset)}|{rime}|{abstract.tone.name}"


def _cer(a: str, b: str) -> float:
    """Tỉ lệ âm tiết lệch giữa text yêu cầu và text ASR — đã gộp lớp âm đầu đồng âm."""
    import difflib

    left = [_syllable_key(w) for w in _WORD_RE.findall(a.lower())]
    right = [_syllable_key(w) for w in _WORD_RE.findall(b.lower())]
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    matched = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in matcher.get_opcodes() if tag == "equal")
    return 1.0 - matched / max(len(left), 1)


def _features(path: str) -> dict:
    """Trích mel/f0/vuv/hnr/jitter của 1 wav bằng hàm CHUẨN của repo."""

    from vifasttts.config import load_config
    from vifasttts.data.features import extract_features

    config = load_config("configs/small_codelung_v2.yaml")
    feats = extract_features(path, config)
    vuv = feats["vuv"].float()
    voiced = vuv > 0.5
    f0 = feats["f0"].float()[voiced]
    hnr = feats["hnr"].float()[voiced]
    jitter = feats["jitter"].float()[voiced]
    mel = feats["mel"].float()
    return {
        "f0_med": float(f0.median()) if f0.numel() else None,
        "hnr_med": float(hnr.median()) if hnr.numel() else None,
        "jitter_med": float(jitter.median()) if jitter.numel() else None,
        "voiced": float(voiced.float().mean()),
        "mel_mean": float(mel.mean()), "mel_std": float(mel.std()),
    }


def _real_reference(limit: int = 120) -> dict:
    """Phân bố giọng THẬT: lấy `limit` clip đầu của train_clean_v2 từ cache v2."""
    import numpy as np

    from vifasttts.config import load_config
    from vifasttts.data.cache import FeatureCache

    config = load_config("configs/small_codelung_v2.yaml")
    cache = FeatureCache("cache/codelung_v2", config)
    rows = [json.loads(l) for l in Path("data/codelung/train_clean_v2.jsonl").read_text(
        encoding="utf-8").splitlines() if l][:limit]
    f0s, hnrs, jits, voiceds, means, stds = [], [], [], [], [], []
    for row in rows:
        feats = cache.get(row)
        if feats is None:
            continue
        vuv = feats["vuv"].float()
        voiced = vuv > 0.5
        if int(voiced.sum()) < 20:
            continue
        f0s.append(float(feats["f0"].float()[voiced].median()))
        hnrs.append(float(feats["hnr"].float()[voiced].median()))
        jits.append(float(feats["jitter"].float()[voiced].median()))
        voiceds.append(float(voiced.float().mean()))
        means.append(float(feats["mel"].float().mean()))
        stds.append(float(feats["mel"].float().std()))
    return {
        "n_clips": len(f0s),
        "f0_med": [float(np.median(f0s)), float(np.std(f0s))],
        "hnr_med": [float(np.median(hnrs)), float(np.std(hnrs))],
        "jitter_med": [float(np.median(jits)), float(np.std(jits))],
        "voiced": [float(np.median(voiceds)), float(np.std(voiceds))],
        "mel_mean": [float(np.median(means)), float(np.std(means))],
        "mel_std": [float(np.median(stds)), float(np.std(stds))],
    }


def main() -> None:
    """CLI: chấm chất lượng dữ liệu tổng hợp, ghi JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest JSONL dữ liệu tổng hợp")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--cer-max", type=float, default=0.15)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l][: args.limit]
    print(f"chấm {len(rows)} câu · ASR vòng tròn …", flush=True)
    heard = _transcribe([row["audio_path"] for row in rows])
    per_clip = []
    for row, text in zip(rows, heard):
        item = {"id": row["id"], "text": row["text"], "heard": text, "cer": round(_cer(row["text"], text), 4)}
        item.update(_features(row["audio_path"]))
        per_clip.append(item)
        print(f"  {row['id']} CER {item['cer']:.3f} · f0 {item['f0_med']} · hnr {item['hnr_med']}", flush=True)

    real = _real_reference()
    def _inside(value, band, z=3.0):
        if value is None:
            return False
        return abs(value - band[0]) <= z * max(band[1], 1e-6)
    passed = [item for item in per_clip if item["cer"] <= args.cer_max
              and _inside(item["f0_med"], real["f0_med"]) and _inside(item["hnr_med"], real["hnr_med"])]
    summary = {
        "n": len(per_clip), "pass": len(passed), "pass_rate": round(len(passed) / max(len(per_clip), 1), 3),
        "cer_mean": round(sum(item["cer"] for item in per_clip) / max(len(per_clip), 1), 4),
        "f0_med_mean": round(sum(item["f0_med"] for item in per_clip if item["f0_med"]) / max(1, sum(1 for i in per_clip if i["f0_med"])), 1),
        "hnr_med_mean": round(sum(item["hnr_med"] for item in per_clip if item["hnr_med"]) / max(1, sum(1 for i in per_clip if i["hnr_med"])), 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "real_reference": real, "per_clip": per_clip},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nđạt {summary['pass']}/{summary['n']} = {summary['pass_rate'] * 100:.0f}% · CER trung bình {summary['cer_mean']}")
    print(f"giọng THẬT: f0 {real['f0_med'][0]:.0f}±{real['f0_med'][1]:.0f} Hz · hnr {real['hnr_med'][0]:.2f}±{real['hnr_med'][1]:.2f} dB · "
          f"jitter {real['jitter_med'][0]:.3f} · hữu thanh {real['voiced'][0]:.2f}")
    print(f"giọng SYNTH: f0 {summary['f0_med_mean']} Hz · hnr {summary['hnr_med_mean']} dB")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
