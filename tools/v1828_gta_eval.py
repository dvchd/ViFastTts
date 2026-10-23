"""tools/v1828_gta_eval.py — v18.28: eval GTA + baseline trên N1/flat_V/HNR/S1 (+ copy-check).

Vì sao: quyết định chọn checkpoint cuối dùng **N1 bin < −50 Δ RMS** (bin cố định theo copy-synth),
đo cả **TF 300 segment** và **free-run production 40 câu**, kèm flat_V/HNR, S1 và kiểm Vocos GTA
có "quên" mel thật không (copy-synth: mel thật → vocoder → so mel).

Chế độ:
  --mode tf     : teacher forcing (durations GT) — N1 bin theo copy-synth từng frame
  --mode free   : free-run (production, sentence mode) — N1 theo bin tuyệt đối dBFS, so WAV thật
  --mode copy   : chỉ copy-check (mel thật → vocoder → L1 với mel thật, log thô)

JSON ghi **trước** khi in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SR, HOP, NFFT = 24000, 256, 1024
CFG32 = "configs/small_codelung_v32.yaml"
CFG35 = "configs/small_codelung_v35.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
V35 = "runs/acoustic_v35/best_ema.pt"
GTA_DIR = "artifacts/vocoder_gta_p2"
BIN_EDGES = (-20.0, -30.0, -40.0, -50.0)

VARIANTS = {
    "copy": {"copy": True},
    "p2": {"ckpt": P2, "cfg": CFG32},
    "p2_exp45": {"ckpt": P2, "cfg": CFG32, "exp": (-45.0, 2.0, -12.0)},
    "p2_exp40": {"ckpt": P2, "cfg": CFG32, "exp": (-40.0, 3.0, -12.0)},
    "p2_exp40_m18": {"ckpt": P2, "cfg": CFG32, "exp": (-40.0, 3.0, -18.0)},
    "p2_exp40_m24": {"ckpt": P2, "cfg": CFG32, "exp": (-40.0, 3.0, -24.0)},
    "v35": {"ckpt": V35, "cfg": CFG35},
    "f0": {"ckpt": "runs/v1829_f0/ckpt_2000.pt", "cfg": CFG32},
    "g1": {"ckpt": "runs/v1830_g1/ckpt_2000.pt", "cfg": CFG32},
    "g2": {"ckpt": "runs/v1830_g2/ckpt_2000.pt", "cfg": CFG32},
    "fd": {"ckpt": "runs/v1829_fd/ckpt_2000.pt", "cfg": CFG32},
    "fc": {"ckpt": "runs/v1829_fc/ckpt_2000.pt", "cfg": "configs/small_codelung_v32_gate.yaml", "gate": 1},
    "fc_g0": {"ckpt": "runs/v1829_fc/ckpt_2000.pt", "cfg": "configs/small_codelung_v32_gate.yaml", "gate": 0},
    **{f"gta{k}": {"ckpt": P2, "cfg": CFG32, "voc": f"{GTA_DIR}/gta_{k}k.pt"} for k in (1, 2, 3, 4, 5)},
}
#: biến thể GTA + expander (dựng sau dict để không lặp)
for _k in (1, 2, 3, 4, 5):
    VARIANTS[f"gta{_k}_exp45"] = {"ckpt": P2, "cfg": CFG32, "voc": f"{GTA_DIR}/gta_{_k}k.pt", "exp": (-45.0, 2.0)}
    VARIANTS[f"gta{_k}_exp40"] = {"ckpt": P2, "cfg": CFG32, "voc": f"{GTA_DIR}/gta_{_k}k.pt", "exp": (-40.0, 3.0)}


def frame_analysis(wave: np.ndarray) -> dict:
    """Đo theo frame 10,67 ms: RMS dBFS, flatness 2–6 kHz, tuần hoàn, HNR."""
    count = int(len(wave) // HOP)
    if count == 0:
        return {"db": np.zeros(0), "flat": np.zeros(0), "voiced": np.zeros(0, dtype=bool),
                "hnr": np.zeros(0), "count": 0}
    padded = np.pad(wave, (0, max(0, (count - 1) * HOP + NFFT - len(wave))))
    frames = np.stack([padded[i * HOP:i * HOP + NFFT] for i in range(count)])
    window = np.hanning(NFFT)
    spec = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    band = (freqs >= 2000) & (freqs <= 6000)
    power = spec[:, band] ** 2 + 1e-12
    flat = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
    centered = frames - frames.mean(axis=1, keepdims=True)
    autoc = np.fft.irfft(np.abs(np.fft.rfft(centered, 2048, axis=1)) ** 2, axis=1)[:, :1024]
    autoc = autoc / (autoc[:, :1] + 1e-12)
    lo, hi = int(SR / 400), int(SR / 70)
    peak = autoc[:, lo:hi].argmax(axis=1) + lo
    period = autoc[np.arange(count), peak]
    f0 = SR / np.maximum(peak, 1)
    raw = np.stack([padded[i * HOP:i * HOP + HOP] for i in range(count)])
    rms = np.sqrt(np.mean(raw ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    silence = db < (np.median(db) - 30.0)
    voiced = (period >= 0.45) & ~silence
    hnr = np.full(count, np.nan)
    for index in np.flatnonzero(voiced):
        f_0 = float(f0[index])
        if not 50.0 <= f_0 <= 500.0:
            continue
        row = spec[index]
        mask = (freqs >= 1000) & (freqs <= 5000)
        total = float(np.sum(row[mask] ** 2))
        half = 0.12 * f_0
        harmonic = 0.0
        harmonic_index = 1
        while harmonic_index * f_0 <= 5000:
            harmonic += float(np.sum(row[(freqs >= harmonic_index * f_0 - half)
                                         & (freqs <= harmonic_index * f_0 + half) & mask] ** 2))
            harmonic_index += 1
        hnr[index] = 10 * np.log10(max(harmonic, 1e-12) / max(total - harmonic, 1e-12))
    return {"db": db, "flat": flat, "voiced": voiced, "hnr": hnr, "count": count}


def expander(x: np.ndarray, threshold_db: float, ratio: float, *,
             attack_ms: float = 2.0, release_ms: float = 30.0, max_db: float = -12.0) -> np.ndarray:
    """Downward expander tất định (block 64 mẫu, gain one-pole attack/release)."""
    block = 64
    n = len(x) // block
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * block].reshape(n, block) ** 2, axis=1) + 1e-12)
    level = 20 * np.log10(rms + 1e-12)
    # chỉ giảm khi dưới ngưỡng; trần giảm = −max_db (max_db âm)
    atten = np.where(level < threshold_db, np.minimum((threshold_db - level) * (ratio - 1.0), -max_db), 0.0)
    # hệ số one-pole tính theo BLOCK (64 mẫu), không phải theo mẫu
    a_att = float(np.exp(-block / max(1.0, SR * attack_ms / 1000.0)))
    a_rel = float(np.exp(-block / max(1.0, SR * release_ms / 1000.0)))
    gain = np.empty(n)
    prev = 0.0
    for i in range(n):
        coeff = a_att if -atten[i] < prev else a_rel
        prev = coeff * prev + (1.0 - coeff) * (-atten[i])
        gain[i] = prev
    out = x.copy()
    out[: n * block] *= (10 ** (gain / 20.0)).repeat(block)
    return out


def syllable_levels(wave: np.ndarray, seg_index: np.ndarray) -> np.ndarray:
    count = min(len(wave) // HOP, len(seg_index))
    if count == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(wave[: count * HOP].reshape(count, HOP) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    levels = []
    for syl in range(int(seg_index[:count].max()) + 1):
        mask = seg_index[:count] == syl
        if mask.any():
            levels.append(float(np.mean(db[mask])))
    return np.asarray(levels)


def seg_index_of(row: dict) -> np.ndarray:
    """Chỉ số âm tiết cho từng frame (durations + gaps của row)."""
    start, end = row["frame_range"]
    seg_index = np.zeros(end - start, dtype=np.int64)
    cursor = 0
    for idx, (dur, gap) in enumerate(zip(row["durations"], list(row["gaps"]) + [0])):
        seg_index[cursor:cursor + dur] = idx
        cursor += dur + gap
    return seg_index


def bin_members(ref_db: np.ndarray) -> list[np.ndarray]:
    """5 bin cố định theo mức copy-synth: >−20, >−30, >−40, >−50, ≤−50."""
    masks = [ref_db > BIN_EDGES[0]]
    for index in range(1, 4):
        masks.append((ref_db > BIN_EDGES[index]) & (ref_db <= BIN_EDGES[index - 1]))
    masks.append(ref_db <= BIN_EDGES[-1])
    return masks


def summary(metrics: dict) -> dict:
    """Gộp các chỉ số frame của một biến thể."""
    voiced, flat, hnr = metrics["voiced"], metrics["flat"], metrics["hnr"]
    non_silent_unvoiced = (~voiced) & (metrics["db"] >= (np.median(metrics["db"]) - 30.0))
    return {
        "frames": int(metrics["count"]),
        "flat_v": round(float(np.median(flat[voiced])), 4) if voiced.any() else None,
        "flat_uv": round(float(np.median(flat[non_silent_unvoiced])), 4) if non_silent_unvoiced.any() else None,
        "hnr": round(float(np.nanmedian(hnr)), 2) if np.isfinite(hnr).any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tf", "free", "copy"), default="tf")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--segments", type=int, default=300)
    parser.add_argument("--val", default="data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--stratify", action="store_true",
                        help="lấy mẫu round-robin theo prefix video (tránh dồn 1 video)")
    parser.add_argument("--gate-split", action="store_true",
                        help="tách N1 theo nhóm video có gate (digital > 1 %%) / không gate")
    args = parser.parse_args()

    from vifasttts.data.cache import FeatureCache
    from vifasttts.data.features import extract_mel_energy_batched
    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.infer import Synthesizer
    from vifasttts.runtime.postprocess import sharpen_mel

    rows = [json.loads(x) for x in Path(args.val).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.stratify:
        from collections import OrderedDict

        groups: OrderedDict[str, list[dict]] = OrderedDict()
        for row in rows:
            groups.setdefault(str(row["id"]).split("_")[0], []).append(row)
        picked, cursor = [], {key: 0 for key in groups}
        while len(picked) < args.segments and any(cursor[k] < len(groups[k]) for k in groups):
            for key, items in groups.items():
                if cursor[key] < len(items) and len(picked) < args.segments:
                    picked.append(items[cursor[key]])
                    cursor[key] += 1
        rows = picked
    else:
        rows = rows[: args.segments]
    tags = [tag.strip() for tag in args.variants.split(",") if tag.strip()]
    result: dict = {"meta": {"mode": args.mode, "segments": len(rows), "alpha": 0, "variants": tags}, "table": {}}

    # ---- copy-check: Vocos GTA có quên mel thật không (mel thật → vocoder → L1 mel) ----
    if args.mode == "copy":
        from vifasttts.models.istftnet import build_generator
        from vifasttts.models.vocos import load_vocos_pretrained

        cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
        floors = {}
        for tag in tags:
            spec = VARIANTS[tag]
            if "voc" not in spec:
                continue
            gen = build_generator(cfg["vocoder_training"], cfg["mel"])
            load_vocos_pretrained(gen, "artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin")
            gen.load_state_dict(torch.load(spec["voc"], map_location="cpu", weights_only=False)["vocoder"])
            gen.to(args.device).eval()
            loss_cfg = gen.loss_mel_config(cfg["mel"])
            values = []
            with torch.no_grad():
                for row in rows:
                    start, end = row["frame_range"]
                    wav, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP, frames=(end - start) * HOP)
                    wav_t = torch.from_numpy(np.asarray(wav)).unsqueeze(0).to(args.device)
                    mel_z, _ = extract_mel_energy_batched(wav_t, cfg["mel"])
                    mel_raw, _ = extract_mel_energy_batched(wav_t, loss_cfg)
                    audio = gen(sharpen_mel(mel_z, 0.0))
                    out_raw, _ = extract_mel_energy_batched(audio[..., : (end - start) * HOP], loss_cfg)
                    n = min(mel_raw.size(1), out_raw.size(1))
                    values.append(float((out_raw[:, :n] - mel_raw[:, :n]).abs().mean()))
            floors[tag] = round(float(np.mean(values)), 4)
            del gen
            if args.device == "mps":
                torch.mps.empty_cache()
        result["copy_floor"] = floors
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return

    engines: dict[tuple[str, str | None], Synthesizer] = {}

    def engine_for(cfg_path: str, ckpt: str, voc: str | None) -> Synthesizer:
        key = (ckpt, voc)
        if key not in engines:
            cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
            engines[key] = Synthesizer(cfg, ckpt, vocoder_checkpoint=voc, device=args.device)
        return engines[key]

    def render(engine: Synthesizer, row: dict, gate: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        text = " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])
        if args.mode == "tf":
            seg_index = seg_index_of(row)
            items = parse_text(text, resolve_oov=True)
            feats = torch.tensor([[engine.vocabs.encode(it) for it in items]], dtype=torch.long, device=engine.device)
            gaps = list(row["gaps"]) + [0] if len(row["gaps"]) == len(row["durations"]) - 1 else list(row["gaps"])
            with torch.no_grad():
                extra = {}
                if gate is not None:
                    extra["gate"] = torch.tensor([int(gate)], dtype=torch.long, device=engine.device)
                out = engine.model(feats, speed=torch.tensor([1.0], device=engine.device),
                                   durations=torch.tensor([row["durations"]], dtype=torch.long, device=engine.device),
                                   gaps=torch.tensor([gaps], dtype=torch.long, device=engine.device),
                                   lead=torch.tensor([0], dtype=torch.long, device=engine.device), **extra)
                audio = engine.vocoder(sharpen_mel(out["mel"][0], 0.0).unsqueeze(0))[0]
            return np.asarray(audio.float().cpu()).reshape(-1), seg_index
        pieces = [audio for audio, _ in engine.iter_synthesize_sentence(text)]
        return (np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)), None

    gate_prefix: dict[str, bool] = {}
    if args.gate_split and args.mode == "tf":
        from collections import defaultdict

        stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            start, end = row["frame_range"]
            audio, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP,
                               frames=(end - start) * HOP)
            audio = np.asarray(audio, dtype=np.float32)
            n = len(audio) // HOP
            if n == 0:
                continue
            blocks = audio[: n * HOP].reshape(n, HOP).astype(np.float64)
            db = 20 * np.log10(np.sqrt(np.maximum((blocks ** 2).mean(axis=1), 1e-24)))
            prefix = str(row["id"]).split("_")[0]
            stats[prefix][0] += int((db < -90.0).sum())
            stats[prefix][1] += n
        gate_prefix = {key: (value[0] / max(1, value[1]) > 0.01) for key, value in stats.items()}
        result["gate_prefix"] = {key: round(100.0 * value[0] / max(1, value[1]), 2)
                                 for key, value in stats.items()}

    # ---- mốc copy-synth (TF) hoặc WAV thật (free) ----
    cfg35 = yaml.safe_load(Path(CFG35).read_text(encoding="utf-8"))
    cache35 = FeatureCache("cache/codelung_v35", cfg35)
    copy_engine = Synthesizer(cfg35, V35, device=args.device)
    refs: list[dict] = []
    for row in rows:
        if args.mode == "tf":
            start, end = row["frame_range"]
            mel = cache35.get(row)["mel"][start:end].to(copy_engine.device)
            with torch.no_grad():
                audio = copy_engine.vocoder(sharpen_mel(mel, 0.0).unsqueeze(0))[0]
            wave = np.asarray(audio.float().cpu()).reshape(-1)
        else:
            start, end = row["frame_range"]
            wave, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP, frames=(end - start) * HOP)
            wave = np.asarray(wave, dtype=np.float32)
        metrics = frame_analysis(wave)
        metrics["syl"] = syllable_levels(wave, seg_index_of(row)) if args.mode == "tf" else None
        refs.append(metrics)
    copy_summary = summary({"db": np.concatenate([r["db"] for r in refs]),
                            "flat": np.concatenate([r["flat"] for r in refs]),
                            "voiced": np.concatenate([r["voiced"] for r in refs]),
                            "hnr": np.concatenate([r["hnr"] for r in refs]),
                            "count": int(sum(r["count"] for r in refs))})
    copy_summary["label"] = "copy-synth" if args.mode == "tf" else "WAV thật"
    ref_db_all = np.concatenate([r["db"] for r in refs]) if refs else np.zeros(0)
    copy_summary["levels"] = {
        "frames": int(ref_db_all.size),
        "below_minus50_pct": round(float(100.0 * np.mean(ref_db_all < -50.0)), 3) if ref_db_all.size else None,
        "mean_db": round(float(np.mean(ref_db_all)), 2) if ref_db_all.size else None,
    }
    result["table"]["copy"] = copy_summary

    for tag in tags:
        if VARIANTS[tag].get("copy"):
            continue
        spec = VARIANTS[tag]
        engine = engine_for(spec["cfg"], spec["ckpt"], spec.get("voc"))
        per_row = []
        for row in rows:
            wave, seg_index = render(engine, row, gate=spec.get("gate"))
            if spec.get("exp"):
                exp = spec["exp"]
                wave = expander(wave, exp[0], exp[1], max_db=(exp[2] if len(exp) > 2 else -12.0))
            metrics = frame_analysis(wave)
            metrics["seg_index"] = seg_index
            metrics["syl"] = syllable_levels(wave, seg_index) if seg_index is not None else None
            per_row.append(metrics)

        merged_db = np.concatenate([m["db"] for m in per_row])
        merged_flat = np.concatenate([m["flat"] for m in per_row])
        merged_voiced = np.concatenate([m["voiced"] for m in per_row])
        merged_hnr = np.concatenate([m["hnr"] for m in per_row])
        entry = summary({"db": merged_db, "flat": merged_flat, "voiced": merged_voiced,
                         "hnr": merged_hnr, "count": int(merged_db.size)})
        # N1
        if args.mode == "tf":
            def _empty():
                return [{"frames": 0, "d_sum": 0.0, "p_model": 0.0, "p_ref": 0.0, "flat": [], "diffs": []}
                        for _ in range(5)]

            def _add(bins, index, metrics, ref, mask, n):
                bins[index]["frames"] += int(mask.sum())
                bins[index]["d_sum"] += float(np.sum(metrics["db"][:n][mask] - ref["db"][:n][mask]))
                bins[index]["p_model"] += float(np.sum(10.0 ** (metrics["db"][:n][mask] / 10.0)))
                bins[index]["p_ref"] += float(np.sum(10.0 ** (ref["db"][:n][mask] / 10.0)))
                bins[index]["flat"].append(metrics["flat"][:n][mask])
                bins[index]["diffs"].append(metrics["db"][:n][mask] - ref["db"][:n][mask])

            def _report(bins):
                return {
                    str(index): {
                        "frames": payload["frames"],
                        "d_rms": round(payload["d_sum"] / payload["frames"], 2) if payload["frames"] else None,
                        "d_rms_energy": round(10.0 * float(np.log10(max(payload["p_model"], 1e-30) / max(payload["p_ref"], 1e-30))), 2) if payload["frames"] else None,
                        "d_rms_median": round(float(np.median(np.concatenate(payload["diffs"]))), 2) if payload["diffs"] else None,
                        "flat": round(float(np.median(np.concatenate(payload["flat"]))), 4) if payload["flat"] else None,
                    }
                    for index, payload in enumerate(bins)
                }

            bins = _empty()
            bins_gate = _empty() if (args.gate_split and gate_prefix) else None
            for metrics, ref, row in zip(per_row, refs, rows):
                n = min(metrics["db"].size, ref["db"].size)
                is_gate = bool(gate_prefix.get(str(row["id"]).split("_")[0], False))
                for index, mask in enumerate(bin_members(ref["db"][:n])):
                    if not mask.any():
                        continue
                    _add(bins, index, metrics, ref, mask, n)
                    if bins_gate is not None and is_gate:
                        _add(bins_gate, index, metrics, ref, mask, n)
            entry["n1"] = _report(bins)
            if bins_gate is not None:
                entry["n1_gate"] = _report(bins_gate)
            s1 = []
            for metrics, ref in zip(per_row, refs):
                if metrics["syl"] is None or ref["syl"] is None:
                    continue
                n = min(metrics["syl"].size, ref["syl"].size)
                if n == 0:
                    continue
                diff = np.abs(metrics["syl"][:n] - ref["syl"][:n])
                s1.append((float(np.mean(diff)), float(np.mean(diff > 3))))
            if s1:
                entry["s1_mae"] = round(float(np.mean([v[0] for v in s1])), 3)
                entry["s1_spike3"] = round(float(np.mean([v[1] for v in s1])), 4)
                # v18.4 báo MEDIAN theo segment ⇒ thêm median để so trực tiếp mốc cũ
                entry["s1_mae_median"] = round(float(np.median([v[0] for v in s1])), 3)
                entry["s1_spike3_median"] = round(float(np.median([v[1] for v in s1])), 4)
        else:
            # free-run: bin tuyệt đối dBFS, so WAV thật (không căn frame)
            entry["levels"] = {
                "frames": int(merged_db.size),
                "below_minus50_pct": round(float(100.0 * np.mean(merged_db < -50.0)), 3),
                "mean_db": round(float(np.mean(merged_db)), 2),
            }
            entry["levels_ref"] = {
                "frames": copy_summary["frames"],
                "below_minus50_pct": round(float(100.0 * np.mean(np.concatenate([r["db"] for r in refs]) < -50.0)), 3),
                "mean_db": round(float(np.mean(np.concatenate([r["db"] for r in refs]))), 2),
            }
        result["table"][tag] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
