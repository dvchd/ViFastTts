"""Test `tools/eval_tone_pairs.py` — chấm thanh minimal pair (CPU, không audio thật)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import eval_tone_pairs as tp

from vifasttts.types import Tone

#: Prior giả tách rõ theo dấu s1 (tăng ↔ s1 âm ↔ SẮC, giảm ↔ s1 dương ↔ HUYỀN).
PRIOR = np.array([
    [0.0, 0.0, 0.0, 0.0],   # NGANG
    [0.0, 1.0, 0.0, 0.0],   # HUYEN
    [0.0, -1.0, 0.0, 0.0],  # SAC
    [5.0, 5.0, 5.0, 5.0],   # HOI
    [-5.0, -5.0, 5.0, 5.0],  # NGA
    [5.0, -5.0, -5.0, 5.0],  # NANG
])


def _sk_of_contour(contour: np.ndarray) -> np.ndarray:
    """Contour z-score → s_k z-score với stats trung tính (mean 0, std 1)."""
    return tp.contour_to_sk_z(np.exp(np.asarray(contour, dtype=float)),
                              f0_log_mean=0.0, f0_log_std=1.0,
                              shape_mean=np.zeros(4), shape_std=np.ones(4))


def test_classify_rising_sac_falling_huyen_flat_ngang():
    rising = np.linspace(-1.0, 1.0, 24)
    falling = rising[::-1]
    flat = np.zeros(24)
    assert tp.classify_shape(_sk_of_contour(rising), PRIOR) == "SAC"
    assert tp.classify_shape(_sk_of_contour(falling), PRIOR) == "HUYEN"
    assert tp.classify_shape(_sk_of_contour(flat), PRIOR) == "NGANG"


def test_classify_tie_break_theo_thu_tu_tones():
    tied = np.zeros((6, 4))  # mọi prior cách đều → lấy đầu (NGANG)
    assert tp.classify_shape(np.zeros(4), tied) == "NGANG"


def test_tone_prior_z_dung_cong_thuc_load_f0_stats():
    stats = {"shape_mean": [1.0, 0.0, 0.0, 0.0], "shape_std": [2.0, 1.0, 1.0, 1.0],
             "shape_tone_prior": {"SAC": {"mean": [3.0, -1.0, 0.5, 0.25]}}}
    priors = tp.tone_prior_z_from_stats(stats)
    assert priors.shape == (6, 4)
    np.testing.assert_allclose(priors[2], [1.0, -1.0, 0.5, 0.25])
    assert (priors[[0, 1, 3, 4, 5]] == 0.0).all()  # thanh thiếu prior → 0


def test_apply_tone_ma_du_6_thanh():
    expected = {"NGANG": "ma", "SAC": "má", "HUYEN": "mà", "HOI": "mả",
                "NGA": "mã", "NANG": "mạ"}
    for tone, syl in expected.items():
        assert tp.apply_tone_to_base("ma", tone) == syl
        assert tp.apply_tone_to_base("ma", Tone(tone)) == syl


def test_smoke_cau_khung_dem_am_tiet_va_index_dich():
    for tone in tp.TONES:
        syl = tp.apply_tone_to_base("ma", tone)
        items, index = tp.parse_target_index(tp.frame_sentence(syl), syl)
        assert len(items) == 4  # Tôi / nói / đích / nhé
        assert index == 2
        assert items[2].abstract.tone == Tone(tone)


def test_smoke_toan_bo_base_mac_dinh_parse_duoc():
    for base in tp.DEFAULT_BASES:
        for tone in tp.TONES:
            syl = tp.apply_tone_to_base(base, tone)
            items, index = tp.parse_target_index(tp.frame_sentence(syl), syl)
            assert items[index].abstract.tone == Tone(tone)


def test_target_frame_window_bo_slot_g():
    slots = [0, 2, 3, 4, 5, 1, 7, 7, 7, 7, 2]  # lead + 2 âm tiết × 5 slot
    # mặc định = nucleus (khớp prior T3.2): bỏ lead/G, chỉ lấy O+M rồi N
    assert tp.target_frame_window(slots, 0) == (5, 4)
    assert tp.target_frame_window(slots, 1) == (29, 7)  # lead(0) + O..G âm 0 (15) + O+M (14) → N
    # chế độ chẩn đoán = cả 4 slot nội dung (bỏ G)
    assert tp.target_frame_window(slots, 0, "syllable") == (0, 14)
    assert tp.target_frame_window(slots, 1, "syllable") == (15, 28)


def test_ab_page_khong_lo_ten_thanh_va_key_du():
    entries = [{"uid": f"ma-{tone}", "base": "ma", "tone": tone, "wav_b64": "AAA="}
               for tone in tp.TONES]
    html, key = tp.build_ab_page(entries, seed=7)
    for tone in tp.TONES:
        assert tone not in html  # mù: không ghi tên thanh
    assert [item["display"] for item in key["items"]] == [1, 2, 3, 4, 5, 6]
    assert {item["uid"] for item in key["items"]} == {e["uid"] for e in entries}
    html2, _ = tp.build_ab_page(entries, seed=7)  # seed cố định → thứ tự ổn định
    assert html2 == html
