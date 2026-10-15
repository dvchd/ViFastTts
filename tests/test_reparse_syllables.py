"""Test tools/reparse_syllables.py — cập nhật `abstract`/`orthography` sau khi parser đổi."""
from __future__ import annotations

import json
from pathlib import Path

from tools.reparse_syllables import reparse_manifest
from vifasttts.data.manifest import build_manifest


def _manifest(tmp_path: Path) -> Path:
    metadata = tmp_path / "meta.csv"
    metadata.write_text("s1|quốc và cuốc\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(str(metadata), str(tmp_path / "wav"), str(manifest))
    return manifest


def _load(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]


def _simulate_old_quoc(rows: list[dict]) -> None:
    """Giả lập manifest cũ: "quốc" = K+W+Ô+C (trước khi sửa parser 16/10)."""
    for syllable in rows[0]["syllables"]:
        if syllable["normalized"] == "quốc":
            syllable["orthography"] = {
                "onset": "qu", "medial": "u", "nucleus": "ô", "coda": "c", "tone_location": "o",
            }
            syllable["abstract"] = {
                "onset": "K", "medial": "W", "nucleus": "Ô", "coda": "C", "tone": "SAC",
            }


def test_reparse_cap_nhat_quoc_va_giu_nguyen_cac_am_tiet_khac(tmp_path: Path):
    manifest = _manifest(tmp_path)
    rows = _load(manifest)
    _simulate_old_quoc(rows)
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )

    stats = reparse_manifest(manifest, write=True)
    assert stats["changed"] == 1 and stats["stale"] == 0

    rows = _load(manifest)
    quoc = next(s for s in rows[0]["syllables"] if s["normalized"] == "quốc")
    assert quoc["abstract"] == {
        "onset": "K", "medial": "NONE", "nucleus": "UÔ", "coda": "C", "tone": "SAC",
    }
    assert (tmp_path / "manifest.jsonl.pre-reparse.bak").exists()


def test_reparse_khong_doi_thi_khong_ghi(tmp_path: Path):
    manifest = _manifest(tmp_path)
    before = manifest.read_text(encoding="utf-8")
    stats = reparse_manifest(manifest)
    assert stats["changed"] == 0 and stats["stale"] == 0
    assert manifest.read_text(encoding="utf-8") == before
    assert not (tmp_path / "manifest.jsonl.pre-reparse.bak").exists()


def test_reparse_phau_thuat_giu_nguyen_truong_khac(tmp_path: Path):
    """Trường khác (boundaries, punctuation...) dù lệch cũng giữ nguyên, chỉ đếm `stale`."""
    manifest = _manifest(tmp_path)
    rows = _load(manifest)
    _simulate_old_quoc(rows)
    rows[0]["syllables"][0]["punctuation"] = ","
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )

    stats = reparse_manifest(manifest, write=True)
    assert stats["changed"] == 1 and stats["stale"] == 1

    rows = _load(manifest)
    first = rows[0]["syllables"][0]
    assert first["punctuation"] == ","  # giữ nguyên giá trị cũ
    assert first["abstract"]["nucleus"] == "UÔ"  # abstract vẫn được cập nhật


def test_reparse_tu_choi_khi_token_chua_dong_bo(tmp_path: Path):
    """`text` đã sửa nhưng `syllables` chưa sync (raw lệch) → từ chối."""
    manifest = _manifest(tmp_path)
    rows = _load(manifest)
    rows[0]["syllables"][0]["raw"] = "chước"
    rows[0]["syllables"][0]["normalized"] = "chước"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )

    try:
        reparse_manifest(manifest, write=True)
    except ValueError as error:
        assert "token đổi" in str(error)
    else:  # pragma: no cover - phải raise
        raise AssertionError("Phải từ chối vì raw lệch")
    assert not (tmp_path / "manifest.jsonl.pre-reparse.bak").exists()
