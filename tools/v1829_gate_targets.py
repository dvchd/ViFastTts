"""tools/v1829_gate_targets.py — nhánh **Fd**: đồng nhất đích bằng noise gate tất định.

Vì sao (A0 v18.29): audio gốc có video bị gate (im kỹ thuật số) và video không (nền nhiễu).
Model không đoán được cờ ⇒ lấy trung bình ⇒ nền to. Fd áp **cùng một gate** lên waveform của
MỌI row (ngưỡng −50 dBFS, attack/release mượt như W1 đo) rồi **trích lại mel đích**; input/duration
giữ nguyên. Nhờ vậy target lặng là **một mức duy nhất** cho mọi row.

Ghi ra:
  <out-wav>/<id>.wav          (crop đã gate, PCM16)
  <out-manifest>.jsonl        (audio_path trỏ wav mới, frame_range [0,T])
  <cache-dir>/<sha1>.pt       (feature cũ, chỉ thay `mel` bằng mel của audio đã gate)
  <report>.json               (số frame bị đổi, số row, ngưỡng…)
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

HOP = 256


def gate(x: np.ndarray, threshold_db: float = -50.0, *,
         attack_ms: float = 20.0, release_ms: float = 40.0, block: int = 64) -> np.ndarray:
    """Gate cứng có ramp: dưới ngưỡng → gain 0, trên → 0 dB; attack/release one-pole theo block."""
    n = len(x) // block
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * block].reshape(n, block) ** 2, axis=1) + 1e-12)
    level = 20 * np.log10(rms + 1e-12)
    target = np.where(level < threshold_db, -90.0, 0.0)  # −90 dB ≈ tắt hẳn
    a_att = float(np.exp(-block / (24000 * attack_ms / 1000.0)))
    a_rel = float(np.exp(-block / (24000 * release_ms / 1000.0)))
    gain = np.empty(n)
    prev = 0.0
    for i in range(n):
        coeff = a_att if target[i] < prev else a_rel
        prev = coeff * prev + (1.0 - coeff) * target[i]
        gain[i] = prev
    out = x.copy()
    out[: n * block] *= (10 ** (gain / 20.0)).repeat(block)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small_codelung_v32.yaml")
    parser.add_argument("--manifest", default="data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--cache-src", default="cache/codelung_v2")
    parser.add_argument("--cache-dst", default="cache/codelung_gated")
    parser.add_argument("--wav-dir", default="/private/tmp/opencode/gated_wav")
    parser.add_argument("--manifest-out", default="/private/tmp/opencode/gated_train_slot4.jsonl")
    parser.add_argument("--report", default="artifacts/bench/v1829_fd_targets.json")
    parser.add_argument("--threshold", type=float, default=-50.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from vifasttts.data.cache import FeatureCache
    from vifasttts.data.features import extract_mel_energy_batched

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mel_cfg = dict(config["mel"])
    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    src = FeatureCache(args.cache_src, config)
    dst = FeatureCache(args.cache_dst, config)
    wav_dir = Path(args.wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    changed_frames = total_frames = kept = 0
    report_rows = 0
    manifest_rows = []
    for row in rows:
        feats = src.get(row)
        if feats is None:
            continue
        start, end = row["frame_range"]
        frames = int(end) - int(start)
        with sf.SoundFile(row["audio_path"]) as handle:
            handle.seek(start * HOP)
            audio = handle.read(frames * HOP, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < frames * HOP:
            continue
        gated = gate(audio, args.threshold)
        # ghi wav crop đã gate
        out_wav = wav_dir / f"{row['id']}.wav"
        sf.write(out_wav, gated, 24000, subtype="PCM_16")
        wav_t = torch.from_numpy(gated).unsqueeze(0)
        mel, _ = extract_mel_energy_batched(wav_t, mel_cfg)
        mel = mel[0, :frames]
        new_feats = dict(feats)
        old_mel = feats["mel"][start:end]
        n = min(mel.size(0), old_mel.size(0))
        delta = (mel[:n] - old_mel[:n]).abs().mean(dim=1)
        changed_frames += int((delta > 0.05).sum())
        total_frames += n
        new_feats["mel"] = mel
        new_row = dict(row)
        new_row["audio_path"] = str(out_wav)
        new_row["frame_range"] = [0, frames]
        new_row["parent_audio"] = row["audio_path"]
        dst.put(new_row, new_feats)
        manifest_rows.append(new_row)
        kept += 1
        report_rows += 1

    out_manifest = Path(args.manifest_out)
    with out_manifest.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "rows": kept,
        "frames": total_frames,
        "changed_frames": changed_frames,
        "changed_pct": round(100.0 * changed_frames / max(1, total_frames), 3),
        "threshold_db": args.threshold,
        "manifest_out": str(out_manifest),
        "cache_dst": args.cache_dst,
        "mode": "Fd (gate mọi row) — mel đích trích lại, input/duration giữ nguyên",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
