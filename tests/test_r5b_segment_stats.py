"""Test `tools/r5b_segment_stats.py` + mask F0 v2 (Addendum 6 — docs/ACOUSTIC_V2.md §19)."""

from __future__ import annotations

import numpy as np

from tools.r5b_segment_stats import pause_histogram, segment_shape, segment_windows
from vifasttts.frontend.segments import F0_VOICED_CODAS, F0_VOICED_ONSETS


def _row(**overrides) -> dict:
    row = {
        "durations": [3, 4], "gaps": [2], "frame_range": [10, 21],
        "seg_durations": [[1, 0, 1, 1], [0, 0, 2, 2]],
    }
    row.update(overrides)
    return row


def test_segment_windows_gaps_va_frame_range():
    windows = segment_windows(_row())
    assert windows[0] == [("O", 10, 11), ("N", 11, 12), ("C", 12, 13)]
    # âm tiết 2 bắt đầu sau gap 2 frame: 13 + 2 = 15
    assert windows[1] == [("N", 15, 17), ("C", 17, 19)]


def test_segment_windows_lead_frames_khong_frame_range():
    row = _row(durations=[4], gaps=[], seg_durations=[[0, 0, 3, 1]])
    row.pop("frame_range")
    row["lead_frames"] = 5
    windows = segment_windows(row)
    assert windows[0] == [("N", 5, 8), ("C", 8, 9)]


def test_segment_windows_gaps_len_n_minus_1_pad():
    windows = segment_windows(_row(durations=[3, 4, 2], gaps=[2, 3], frame_range=[0, 12],
                                   seg_durations=[[1, 0, 1, 1], [0, 0, 2, 2], [0, 0, 1, 1]]))
    starts = [entry[0][1] for entry in windows]
    assert starts == [0, 5, 12]


def test_segment_shape_hang_so():
    f0 = np.full(6, 150.0)
    vuv = np.ones(6)
    shape, n = segment_shape(f0, vuv, 4.8, 0.2)
    assert n == 6 and shape.size == 4
    assert abs(shape[0] - (np.log(150.0) - 4.8) / 0.2) < 1e-9  # s0 = mean z
    assert abs(shape[1]) < 1e-9 and abs(shape[2]) < 1e-9


def test_segment_shape_ngan_chi_co_k_du():
    f0 = np.array([150.0, 140.0])
    shape, n = segment_shape(f0, np.ones(2), 4.8, 0.2)
    assert n == 2 and shape.size == 2  # n < 4 → chỉ k = n hệ số


def test_segment_shape_vuv_loc_bo_frame_vo_thanh():
    f0 = np.array([150.0, 150.0, 0.5, 150.0, 150.0])
    vuv = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    shape, n = segment_shape(f0, vuv, 4.8, 0.2)
    assert n == 4 and shape.size == 4  # frame vuv=0 (và f0≈0) bị loại


def test_segment_shape_toan_vo_thanh():
    shape, n = segment_shape(np.zeros(5), np.zeros(5), 4.8, 0.2)
    assert n == 0 and shape.size == 0


def test_mask_f0_partition():
    voiceless = {"PH", "T", "TH", "X", "S", "CH", "TR", "K", "KH", "H", "P"}
    assert F0_VOICED_ONSETS.isdisjoint(voiceless)
    assert {"M", "N", "NG", "NH", "L", "V", "B", "D", "G", "GI", "R"} == F0_VOICED_ONSETS
    assert F0_VOICED_CODAS.isdisjoint({"P", "T", "C", "CH"})


def test_pause_histogram():
    row = {
        "syllables": [
            {"boundaries": {"word_end": True}, "punctuation": ""},
            {"boundaries": {"word_end": True, "phrase_end": True}, "punctuation": ""},
            {"boundaries": {"sentence_end": True}, "punctuation": "..."},
        ],
        "gaps": [0, 30, 500],
    }
    summary = pause_histogram([row], hop=256, sample_rate=24000)
    assert summary["WORD"]["n"] == 1 and summary["WORD"]["zero_frac"] == 1.0
    assert summary["PHRASE"]["n"] == 1 and summary["PHRASE"]["ge3_frac"] == 1.0
    assert summary["SENTENCE_ELLIPSIS"]["n"] == 1
    expected = 500 * 256 / 24000 * 1000
    assert abs(summary["SENTENCE_ELLIPSIS"]["median_ms"] - expected) < 0.1
