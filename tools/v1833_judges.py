"""tools/v1833_judges.py — v18.33: giám khảo tự động (DNSMOS, UTMOSv2, ASR) + kiểm độ nhạy.

Chế độ:
  --mode eval         : chấm P2/G1/real (300 câu) → `v1833_judges.json`
  --mode sensitivity  : tạo bản suy giảm nhân tạo (nhiễu, giọng bị cắt) trên WAV thật,
                        kiểm từng giám khảo có phản ứng đúng hướng và ổn định giữa các câu.

Ghi rõ preprocessing: 24 kHz → 16 kHz bằng `scipy.resample_poly` (polyphase), không gain.
DNSMOS: `speechmos` (ONNX, 16 kHz, không chuẩn hoá thêm).
UTMOSv2: `utmosv2` fusion_stage3, đầu vào 16 kHz do thư viện tự xử lý.
ASR: `faster-whisper` small (CTranslate2 int8, CPU), language=vi, beam 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SR = 24000
SR16 = 16000


def to16(wave: np.ndarray) -> np.ndarray:
    return resample_poly(np.asarray(wave, dtype=np.float64), SR16, SR).astype(np.float32)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref, hyp = normalize_text(ref), normalize_text(hyp)
    if not ref:
        return float("nan")
    return edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")) / len(ref.replace(" ", ""))


def wer(ref: str, hyp: str) -> float:
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    if not r:
        return float("nan")
    return edit_distance(r, h) / len(r)


def dnsmos_scores(wave16: np.ndarray) -> dict:
    from speechmos import dnsmos

    out = dnsmos.run(wave16, sr=SR16)
    return {key: round(float(value), 4) for key, value in out.items()}


def utmos_dir(directory: Path, device: str = "mps") -> dict:
    import utmosv2

    model = utmosv2.create_model(pretrained=True, config="fusion_stage3", device=device)
    out = model.predict(input_dir=str(directory), device=device, batch_size=8, verbose=False)
    return {Path(item["file_path"]).stem: round(float(item["predicted_mos"]), 4) for item in out}


def asr_text(wave16: np.ndarray, model) -> str:
    segments, _info = model.transcribe(wave16, language="vi", beam_size=1, vad_filter=False,
                                       condition_on_previous_text=False)
    return " ".join(seg.text for seg in segments).strip()


def load_asr():
    from faster_whisper import WhisperModel

    return WhisperModel("small", device="cpu", compute_type="int8")


def build_sensitivity(src: Path, out: Path, per_video: int = 1, seed: int = 3) -> list[dict]:
    """Tạo bản nhiễu (SNR 30/20/10 dB) và bản cắt (2/5/10 % thời lượng) từ WAV thật."""
    rng = np.random.default_rng(seed)
    items = json.loads((src / "items.json").read_text(encoding="utf-8"))
    picked, seen = [], set()
    for item in items:
        if item["video"] in seen:
            continue
        seen.add(item["video"])
        picked.append(item)
        if len(picked) >= per_video * 8:
            break
    variants = []
    for item in picked:
        wave, _ = sf.read(src / "real" / f"{item['id']}.wav", dtype="float32")
        speech_rms = float(np.sqrt(np.mean(wave ** 2) + 1e-12))
        variants.append({"id": item["id"], "video": item["video"], "kind": "clean",
                         "wave": wave, "text": item["text"]})
        for snr in (30, 20, 10):
            noise = rng.standard_normal(wave.size).astype(np.float32)
            noise *= speech_rms / (np.sqrt(np.mean(noise ** 2) + 1e-12) * 10 ** (snr / 20.0))
            variants.append({"id": item["id"], "video": item["video"], "kind": f"noise{snr}",
                             "wave": np.clip(wave + noise, -1, 1).astype(np.float32),
                             "text": item["text"]})
        for pct in (2, 5, 10):
            out_wave = wave.copy()
            total = int(wave.size * pct / 100.0)
            hole = int(0.1 * SR)
            filled = 0
            while filled < total:
                start = int(rng.integers(0, max(1, wave.size - hole)))
                out_wave[start:start + hole] = 0.0
                filled += hole
            variants.append({"id": item["id"], "video": item["video"], "kind": f"drop{pct}",
                             "wave": out_wave, "text": item["text"]})
    for variant in variants:
        directory = out / variant["kind"]
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(directory / f"{variant['id']}.wav", variant["wave"], SR)
    (out / "variants.json").write_text(json.dumps(
        [{key: variant[key] for key in ("id", "video", "kind", "text")} for variant in variants],
        ensure_ascii=False, indent=1), encoding="utf-8")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eval", "sensitivity"), default="eval")
    parser.add_argument("--audio", default="artifacts/bench/v1833_audio")
    parser.add_argument("--sens", default="artifacts/bench/v1833_sens")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if args.mode == "eval":
        src = Path(args.audio)
        items = json.loads((src / "items.json").read_text(encoding="utf-8"))
        by_id = {item["id"]: item for item in items}
        asr = load_asr()
        result: dict = {"items": len(items), "judges": {}, "per_file": {}}
        for name in ("p2", "g1", "real"):
            directory = src / name
            utmos = utmos_dir(directory, args.device)
            rows = {}
            for path in sorted(directory.glob("*.wav")):
                wave, _ = sf.read(path, dtype="float32")
                w16 = to16(wave)
                scores = dnsmos_scores(w16)
                text = asr_text(w16, asr)
                ref = by_id[path.stem]["text"]
                rows[path.stem] = {
                    "utmos": utmos.get(path.stem),
                    "dnsmos": scores,
                    "asr": text,
                    "cer": round(cer(ref, text), 4),
                    "wer": round(wer(ref, text), 4),
                }
            result["judges"][name] = {
                "n": len(rows),
                "utmos_mean": round(float(np.mean([r["utmos"] for r in rows.values() if r["utmos"] is not None])), 4),
                "bak_mean": round(float(np.mean([r["dnsmos"]["bak_mos"] for r in rows.values()])), 4),
                "sig_mean": round(float(np.mean([r["dnsmos"]["sig_mos"] for r in rows.values()])), 4),
                "ovrl_mean": round(float(np.mean([r["dnsmos"]["ovrl_mos"] for r in rows.values()])), 4),
                "cer_mean": round(float(np.nanmean([r["cer"] for r in rows.values()])), 4),
                "wer_mean": round(float(np.nanmean([r["wer"] for r in rows.values()])), 4),
            }
            result["per_file"][name] = rows
        out = Path(args.out or "artifacts/bench/v1833_judges.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(result["judges"], ensure_ascii=False, indent=1))
        return

    # sensitivity
    src, out = Path(args.audio), Path(args.sens)
    variants = build_sensitivity(src, out)
    asr = load_asr()
    rows = []
    for variant in variants:
        w16 = to16(variant["wave"])
        rows.append({"id": variant["id"], "video": variant["video"], "kind": variant["kind"],
                     "dnsmos": dnsmos_scores(w16), "asr": asr_text(w16, asr),
                     "text": variant["text"]})
    for kind in sorted({row["kind"] for row in rows}):
        directory = out / kind
        scores = utmos_dir(directory, args.device)
        for row in rows:
            if row["kind"] == kind:
                row["utmos"] = scores.get(row["id"])
    # CER so với bản sạch (tự tham chiếu) và so với text gốc
    clean_asr = {row["id"]: row["asr"] for row in rows if row["kind"] == "clean"}
    if not clean_asr:  # chưa có clean → dùng WAV thật
        for item in json.loads((src / "items.json").read_text(encoding="utf-8")):
            if item["id"] in {row["id"] for row in rows}:
                wave, _ = sf.read(src / "real" / f"{item['id']}.wav", dtype="float32")
                clean_asr[item["id"]] = asr_text(to16(wave), asr)
    for row in rows:
        row["cer_vs_clean"] = round(cer(clean_asr.get(row["id"], ""), row["asr"]), 4)
        row["cer_vs_text"] = round(cer(row["text"], row["asr"]), 4)
    summary = {}
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        summary[kind] = {
            "n": len(subset),
            "utmos": round(float(np.mean([r["utmos"] for r in subset if r.get("utmos") is not None])), 4),
            "bak": round(float(np.mean([r["dnsmos"]["bak_mos"] for r in subset])), 4),
            "sig": round(float(np.mean([r["dnsmos"]["sig_mos"] for r in subset])), 4),
            "ovrl": round(float(np.mean([r["dnsmos"]["ovrl_mos"] for r in subset])), 4),
            "cer_clean": round(float(np.nanmean([r["cer_vs_clean"] for r in subset])), 4),
            "cer_text": round(float(np.nanmean([r["cer_vs_text"] for r in subset])), 4),
        }
    result = {"rows": rows, "summary": summary,
              "note": "Dữ liệu suy giảm NHÂN TẠO, chỉ dùng kiểm độ nhạy — không dùng để chọn checkpoint."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
