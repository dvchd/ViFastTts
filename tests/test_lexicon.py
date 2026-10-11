"""Test lexicon + profile (open/strict/corpus) và các luật chính tả."""
from __future__ import annotations

from pathlib import Path

import pytest

from vifasttts.frontend.lexicon import (
    ISSUE_DESCRIPTIONS,
    Lexicon,
    check_syllable,
    profile_report,
    strict_warnings,
)
from vifasttts.frontend.syllable import parse_syllable


def _issues(word: str) -> list[str]:
    return strict_warnings(parse_syllable(word))


def test_open_profile_never_warns():
    for word in ("cen", "voan", "iên", "ka"):
        assert check_syllable(parse_syllable(word), "open") == []


def test_strict_accepts_standard_spellings():
    for word in ("qua", "kính", "kể", "cá", "ghế", "ghi", "gà", "nghỉ", "nghe", "ngà",
                 "quốc", "quơ", "xoong", "giếng", "anh", "ênh", "inh", "uynh", "yên", "ỉa",
                 "ngoe", "nguyễn", "nghiêng", "nghiêm", "ghê", "goe"):
        assert _issues(word) == [], word


def test_strict_flags_ng_ngh_before_wrong_vowel():
    # "ngịu"/"ngẹt": chuẩn chính tả là "nghịu"/"nghẹt".
    assert "spelling-ng-before-front" in _issues("ngịu")
    assert "spelling-ng-before-front" in _issues("ngẹt")


def test_strict_flags_c_before_front():
    assert "spelling-c-before-front" in _issues("cen")
    assert "spelling-c-before-front" in _issues("cê")
    assert "spelling-c-before-front" in _issues("ci")


def test_strict_flags_k_before_back():
    assert "spelling-k-before-back" in _issues("ka")
    assert "spelling-k-before-back" in _issues("kơi")
    assert "spelling-k-before-back" in _issues("ku")


def test_strict_flags_g_gh_ng_ngh_distribution():
    assert "spelling-g-before-front" in _issues("gê")
    assert "spelling-gh-before-back" in _issues("gha")
    assert "spelling-ng-before-front" in _issues("ngi")
    assert "spelling-ngh-before-back" in _issues("ngha")


def test_strict_flags_labial_w():
    assert "labial-w" in _issues("voan")
    assert "labial-w" in _issues("boa")
    assert _issues("quan") == []


def test_strict_flags_coda_nh_ch_after_nonfront():
    assert "coda-nh-ch-after-nonfront" in _issues("unh")
    assert "coda-nh-ch-after-nonfront" in _issues("cânh")
    assert _issues("oanh") == []


def test_strict_flags_standalone_ie():
    assert "standalone-ie" in _issues("iên")
    assert "standalone-ie" in _issues("iêng")
    assert _issues("yên") == []


def test_strict_flags_open_double_vowel():
    assert "open-double-vowel-spelling" in _issues("iê")
    assert "open-double-vowel-spelling" in _issues("uô")
    assert "open-double-vowel-spelling" in _issues("ươ")
    assert _issues("ua") == [] and _issues("ưa") == [] and _issues("ia") == []


def test_corpus_profile_requires_lexicon():
    with pytest.raises(ValueError):
        check_syllable(parse_syllable("ba"), "corpus")


def test_corpus_profile_membership():
    lexicon = Lexicon.from_words(["ba", "mẹ", "khoai"])
    assert check_syllable(parse_syllable("ba"), "corpus", lexicon) == []
    issues = check_syllable(parse_syllable("khoai"), "corpus", lexicon)
    assert issues == []
    issues = check_syllable(parse_syllable("nga"), "corpus", lexicon)
    assert "shape-not-in-lexicon" in issues


def test_lexicon_roundtrip(tmp_path: Path):
    lexicon = Lexicon.from_words(["ba", "mẹ", "xoong"])
    path = tmp_path / "lexicon.json"
    lexicon.save(path)
    loaded = Lexicon.load(path)
    assert loaded.shapes == lexicon.shapes
    assert ("X", "NONE", "OO", "NG") in loaded.shapes


def test_lexicon_from_manifest(tmp_path: Path):
    from vifasttts.data.manifest import build_manifest

    metadata = tmp_path / "meta.csv"
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    metadata.write_text("s1|Chào bạn\n", encoding="utf-8")
    manifest = tmp_path / "m.jsonl"
    build_manifest(metadata, wav_dir, manifest)
    lexicon = Lexicon.from_manifest(manifest)
    assert ("CH", "NONE", "A", "W") in lexicon.shapes  # "Chào" (coda W từ "o")
    assert lexicon.has(parse_syllable("bạn").abstract)


def test_profile_report_counts():
    report = profile_report(["ba", "cen", "voan", "khoai"], "strict")
    assert report["total"] == 4
    assert report["flagged"] == 2
    assert report["issues"]["spelling-c-before-front"] == 1
    assert report["issues"]["labial-w"] == 1
    assert report["examples"]["labial-w"] == ["voan"]


def test_all_issue_codes_have_descriptions():
    report = profile_report(["cen", "ka", "gê", "gha", "ngi", "ngha", "voan", "unh", "iên", "uô"], "strict")
    for code in report["issues"]:
        assert code in ISSUE_DESCRIPTIONS, code
