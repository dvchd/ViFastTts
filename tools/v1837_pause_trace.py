"""v18.37 — trace pause: raw-join legacy vs text phục hồi dấu câu (sentence mode).

Chọn **12 câu cố định trước khi render**: 3 bucket độ dài × 4 kiểu dấu câu
(phẩy / chấm / hỏi / không dấu), lấy row đầu tiên theo thứ tự manifest (không dựa điểm).
Cùng P2 EMA + cùng Vocos pretrained, α=0, sentence mode.

Ghi chuỗi trace:
text → parsed punctuation → gaps → slot_durations → gap fill → trim → waveform length.

3 câu đầu (mỗi bucket 1 câu) được chạy thêm qua **CLI chunk** và **ONNX chunk** để ghi
khác biệt mode; không so waveform bit và không đổi artifact ONNX.
"""

from __future__ import annotations

import json
import subprocess
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
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
ONNX_ACOUSTIC = "artifacts/onnx/v32_adv_p2_2k.onnx"
ONNX_VOCODER = "artifacts/onnx/v32_adv_p2_2k_vocoder.onnx"
OUT = Path("artifacts/bench/v1837_pause")


def signature(row: dict) -> str:
    marks = set()
    for syllable in row["syllables"]:
        punct = str(syllable.get("punctuation") or "")
        for mark in ",.?":
            if mark in punct:
                marks.add(mark)
    return "".join(sorted(marks)) or "none"


def bucket(row: dict) -> str:
    frames = int(row["frame_range"][1]) - int(row["frame_range"][0])
    return "short" if frames < 150 else ("mid" if frames < 300 else "long")


def select(rows: list[dict]) -> list[dict]:
    """12 câu: mỗi bucket lấy 1 câu cho mỗi signature, theo thứ tự manifest."""
    picked = []
    for name in ("short", "mid", "long"):
        for sig in (",", ".", "?", "none"):
            for row in rows:
                if bucket(row) == name and signature(row) == sig:
                    picked.append(row)
                    break
    if len(picked) != 12:
        raise RuntimeError(f"không đủ 12 câu: {len(picked)}")
    return picked


def trace_one(engine, row: dict, text: str, tag: str, *, fill_gap: bool = True) -> dict:
    """Chạy đúng các bước của `iter_synthesize_sentence` và ghi trace."""
    from vifasttts.frontend.normalizer import NormalizerOptions
    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.infer import _trim_edge_silence
    from vifasttts.runtime.postprocess import sharpen_mel

    options = NormalizerOptions(dialect="north", exact_overrides={})
    items = parse_text(text, dialect="north", options=options, resolve_oov=True)
    features = torch.tensor([[engine.vocabs.encode(item) for item in items]],
                            dtype=torch.long, device=engine.device)
    gaps = engine.sentence_gaps(items, engine.sample_rate, HOP)
    with torch.no_grad():
        speed = torch.tensor([1.0], dtype=torch.float32, device=engine.device)
        out = engine.model(features, speed=speed, gate_threshold=1.01,
                           gaps=torch.tensor(gaps, dtype=torch.long,
                                             device=engine.device).unsqueeze(0))
        mel = out["mel"][0]
        slots = out["slot_durations"][0].tolist()
        mel_frames = int(mel.size(0))
        filled = int(sum(slots[5::5])) if fill_gap else 0
        if fill_gap:
            mel = engine._fill_gap_silence(mel, out["slot_durations"][0])
        wave = np.asarray(engine.vocoder(sharpen_mel(mel, 0.0).unsqueeze(0))[0]
                          .float().cpu()).reshape(-1)
    wave_before_trim = int(wave.size)
    wave_trim = _trim_edge_silence(wave, engine.sample_rate)
    out_path = OUT / "audio" / f"{row['id']}_{tag}.wav"
    sf.write(out_path, np.asarray(wave_trim, dtype=np.float32), engine.sample_rate)
    return {
        "id": row["id"], "tag": tag, "text": text,
        "parsed_punctuation": [getattr(item, "punctuation", None) for item in items],
        "items": len(items),
        "gaps": gaps.tolist(), "gap_frames": int(gaps.sum()),
        "gap_slots": [index for index, value in enumerate(slots) if value > 0 and index % 5 == 0],
        "slot_durations": slots, "slot_total": int(sum(slots)),
        "mel_frames": mel_frames,
        "expected_waveform_len": (mel_frames - 1) * HOP,
        "gap_fill_frames": filled,
        "waveform_len_before_trim": wave_before_trim,
        "waveform_len_after_trim": int(wave_trim.size),
        "trimmed_samples": wave_before_trim - int(wave_trim.size),
        "audio": str(out_path),
    }


def run_cli(row: dict, text: str) -> dict:
    path = OUT / "audio" / f"{row['id']}_cli_chunk.wav"
    command = [
        str(_ROOT / ".venv/bin/vifasttts"), "synthesize",
        "--config", CFG32, "--checkpoint", P2,
        "--text", text, "--output", str(path),
        "--target-lufs", "0", "--device", "cpu",
    ]
    proc = subprocess.run(command, check=False, cwd=_ROOT, capture_output=True, text=True, timeout=180)
    info = sf.info(str(path)) if path.exists() else None
    return {
        "id": row["id"], "text": text, "returncode": proc.returncode,
        "stderr_tail": proc.stderr.strip().splitlines()[-3:],
        "audio": str(path) if path.exists() else None,
        "samples": int(info.frames) if info else None,
        "sample_rate": int(info.samplerate) if info else None,
        "mode": "CLI chunk (Synthesizer.synthesize, gate 0.5, fade, target_lufs=0)",
    }


def run_onnx(row: dict, text: str) -> dict:
    from vifasttts.runtime.onnx_engine import synthesize_onnx

    path = OUT / "audio" / f"{row['id']}_onnx_chunk.wav"
    try:
        samples, rate = synthesize_onnx(
            text, ONNX_ACOUSTIC, path, vocoder_path=ONNX_VOCODER,
            provider="cpu", target_lufs=None, sample_rate=24000,
        )
        error = None
    except (ImportError, RuntimeError, ValueError, OSError) as exc:  # pragma: no cover - môi trường thiếu onnx
        samples, rate, error = None, None, f"{type(exc).__name__}: {exc}"
    return {
        "id": row["id"], "text": text, "audio": str(path) if path.exists() else None,
        "samples": samples, "sample_rate": rate, "error": error,
        "mode": "ONNX chunk (synthesize_onnx, fade, target_lufs=None)",
        "onnx_acoustic": ONNX_ACOUSTIC, "onnx_vocoder": ONNX_VOCODER,
        "note": "vocoder ONNX cũ = E4-C (xem v1837_provenance.json), không phải pretrained",
    }


def main() -> None:
    from vifasttts.eval_text import build_eval_text, legacy_raw_join
    from vifasttts.infer import Synthesizer

    rows = [json.loads(line) for line in
            Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = select(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(parents=True, exist_ok=True)

    selection = []
    for row in picked:
        selection.append({
            "id": row["id"], "bucket": bucket(row), "signature": signature(row),
            "text": build_eval_text(row), "text_legacy": legacy_raw_join(row),
        })
    (OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps({"selected": len(selection),
                      "buckets": {b: sum(1 for x in selection if x["bucket"] == b)
                                  for b in ("short", "mid", "long")}}, ensure_ascii=False))

    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    engine = Synthesizer(cfg, P2, device="cpu")
    trace = []
    for row in picked:
        for tag, text in (("legacy", legacy_raw_join(row)), ("punct", build_eval_text(row))):
            trace.append(trace_one(engine, row, text, tag))
    out = {"selection": selection, "trace": trace}
    (OUT / "trace.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # 3 câu: mỗi bucket 1 câu có dấu câu (không lấy 'none').
    cli_rows = []
    for name in ("short", "mid", "long"):
        for row in picked:
            if bucket(row) == name and signature(row) != "none":
                cli_rows.append(row)
                break
    cli = [run_cli(row, build_eval_text(row)) for row in cli_rows]
    onnx = [run_onnx(row, build_eval_text(row)) for row in cli_rows]
    (OUT / "runtime_modes.json").write_text(
        json.dumps({"cli_chunk": cli, "onnx_chunk": onnx}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps({"trace": len(trace), "cli": len(cli), "onnx": len(onnx)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
