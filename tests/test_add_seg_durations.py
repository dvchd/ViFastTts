"""Test `tools/add_seg_durations.py`: bất biến chia slot + biên hybrid (T2.1)."""

from __future__ import annotations

from tools.add_seg_durations import boundary_time, distribute, role_weights, split_syllable
from vifasttts.types import AbstractSyllable, Tone


def _abstract(onset: str = "NONE", medial: str = "NONE", nucleus: str = "A",
              coda: str = "NONE", tone: Tone = Tone.NGANG) -> AbstractSyllable:
    return AbstractSyllable(onset=onset, medial=medial, nucleus=nucleus, coda=coda, tone=tone)


def test_distribute_bat_bien():
    weights = role_weights(_abstract(onset="H", nucleus="A", coda="C"))
    for frames in range(40):
        result = distribute(frames, weights)
        assert sum(result.values()) == frames
        assert set(result) == {"O", "M", "N", "C"}
        if frames >= 3:
            assert result["O"] >= 1 and result["N"] >= 1 and result["C"] >= 1
            assert result["M"] == 0  # inactive = 0


def test_distribute_thieu_frame_uu_tien_n_o_c_m():
    weights = role_weights(_abstract(onset="H", medial="W", nucleus="A", coda="C"))
    assert distribute(2, weights) == {"O": 1, "M": 0, "N": 1, "C": 0}
    assert distribute(1, weights) == {"O": 0, "M": 0, "N": 1, "C": 0}


def test_split_syllable_hybrid_onset_co_dinh():
    weights = role_weights(_abstract(onset="H", nucleus="IÊ", coda="N", tone=Tone.NANG))
    result = split_syllable(20, weights, onset_frames=5)
    assert result["O"] == 5
    assert sum(result.values()) == 20
    assert result["N"] >= 1 and result["C"] >= 1


def test_split_syllable_onset_zero_muon_1_frame():
    weights = role_weights(_abstract(onset="K", medial="W", nucleus="A", coda="N"))
    result = split_syllable(10, weights, onset_frames=0)
    assert result["O"] == 1  # mượn 1 frame từ slot lớn nhất (giữ bất biến active ≥ 1)
    assert sum(result.values()) == 10


def test_split_syllable_fallback_rule():
    weights = role_weights(_abstract(onset="H", nucleus="A", coda="N"))
    assert split_syllable(15, weights, onset_frames=None) == distribute(15, weights)


def test_split_syllable_d_ngan_uu_tien():
    weights = role_weights(_abstract(onset="H", medial="W", nucleus="A", coda="C"))
    assert split_syllable(2, weights, 1) == {"O": 1, "M": 0, "N": 1, "C": 0}
    assert split_syllable(1, weights, 1) == {"O": 0, "M": 0, "N": 1, "C": 0}


def test_boundary_time():
    phones = [(0.0, 0.05, "h"), (0.05, 0.25, "aː˨˨"), (0.25, 0.35, "n")]
    assert boundary_time(phones, _abstract(onset="H", nucleus="A", coda="N")) == 0.05

    with_medial = [(0.0, 0.05, "h"), (0.05, 0.10, "w"), (0.10, 0.30, "aː˨˨")]
    assert boundary_time(with_medial, _abstract(onset="H", medial="W", nucleus="A")) == 0.05

    vowel_initial = [(0.0, 0.02, "ʔ"), (0.02, 0.20, "aː˨˨")]
    assert boundary_time(vowel_initial, _abstract(onset="NONE", nucleus="A")) == 0.0

    no_vowel = [(0.0, 0.05, "k"), (0.05, 0.10, "p")]
    assert boundary_time(no_vowel, _abstract(onset="K", nucleus="A", coda="P")) is None
