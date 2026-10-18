"""Test cho `tools/creak_rarity_probe.py` — đếm tần suất âm tiết + lọc chữ Việt (18/10/2026)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from creak_rarity_probe import VIET_RE, build_freq


def _row(words):
    return {"syllables": [{"normalized": w, "abstract": {"tone": t}} for w, t in words]}


def test_build_freq_counts_and_tones(tmp_path):
    manifest = tmp_path / "m.jsonl"
    lines = [json.dumps(_row([("học", "NANG"), ("và", "HUYEN")]), ensure_ascii=False) for _ in range(3)]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts, tone_of = build_freq(str(manifest))
    assert counts["học"] == 3 and counts["và"] == 3
    assert tone_of["học"] == "NANG" and tone_of["và"] == "HUYEN"


def test_viet_re_loai_ky_tu_la(tmp_path):
    assert VIET_RE.match("học") and VIET_RE.match("giếng")
    assert not VIET_RE.match("ai2") and not VIET_RE.match("x-ray")
