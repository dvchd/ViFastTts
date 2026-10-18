"""Test tools/audit_clip_quality.py — đo chỉ số QC trên WAV tổng hợp."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from tools.audit_clip_quality import analyze_clip, collect_clips, summarize

SR = 24000


def _write(path, data) -> None:
    sf.write(str(path), np.asarray(data, dtype=np.float32), SR, subtype="PCM_16")


def _sine(seconds: float, freq: float = 440.0, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_analyze_clip_clean_vs_noisy(tmp_path):
    rng = np.random.default_rng(0)
    # "Phòng" có nền nhiễu riêng (im lặng số bị thước bỏ qua — cần room tone thật).
    quiet_room = rng.normal(0, 1e-3, int(SR * 0.4))
    clean = np.concatenate([quiet_room, _sine(1.2)])
    _write(tmp_path / "clean.wav", clean)

    loud_room = rng.normal(0, 0.03, int(SR * 0.4))
    noisy = np.concatenate([loud_room, _sine(1.2, amp=0.2) + rng.normal(0, 0.05, int(SR * 1.2))])
    _write(tmp_path / "noisy.wav", noisy)

    clipped = np.clip(_sine(1.0, amp=1.5) * 2.0, -1.0, 1.0)
    _write(tmp_path / "clipped.wav", clipped)

    clean_m = analyze_clip(str(tmp_path / "clean.wav"))
    noisy_m = analyze_clip(str(tmp_path / "noisy.wav"))
    clipped_m = analyze_clip(str(tmp_path / "clipped.wav"))

    assert clean_m["snr_db"] > noisy_m["snr_db"]      # nền sạch hơn → SNR cao hơn
    assert clean_m["flatness"] < noisy_m["flatness"]  # phổ "dẹt" hơn khi nhiễu
    assert clipped_m["clip_frac"] > 0.05              # chạm biên nhiều


def test_collect_and_summarize(tmp_path):
    import json

    _write(tmp_path / "a.wav", _sine(0.8))
    manifest = tmp_path / "m.jsonl"
    rows = [{"parent": "p1", "audio_path": str(tmp_path / "a.wav")}] * 2
    rows.append({"audio_path": str(tmp_path / "a.wav")})  # không parent → khoá theo path
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    clips = collect_clips([str(manifest)])
    assert set(clips) == {"p1", str(tmp_path / "a.wav")}

    scanned = {key: analyze_clip(path) for key, path in clips.items()}
    summary = summarize(scanned)
    assert summary["snr_db"]["n"] == 2
    assert summary["dur_s"]["p50"] > 0.5
