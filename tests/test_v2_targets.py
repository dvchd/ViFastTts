"""Test `data/v2_targets.py` — cửa sổ segment, ratio/pause target, s_k + mask F0 (T2.2)."""

from __future__ import annotations

import math

import torch

from vifasttts.data.v2_targets import (
    f0_targets,
    pause_targets,
    ratio_targets,
    segment_pools,
    segment_targets,
    segment_windows,
)

# Row tổng hợp: "các" (onset K vô thanh) d=5 seg [2,0,2,1] (C chỉ 1 frame → bỏ s_k);
# "ma" (onset M hữu thanh) d=6 seg [2,0,2,2]. frame_range [10, 23] → span 5 + gap 2 + 6 = 13.
ROW = {
    "id": "clip__s00", "parent": "clip", "frame_range": [10, 23],
    "durations": [5, 6], "gaps": [2],
    "seg_durations": [[2, 0, 2, 1], [2, 0, 2, 2]],
    "syllables": [
        {"raw": "các", "abstract": {"onset": "K", "medial": "NONE", "nucleus": "A",
                                    "coda": "C", "tone": "SAC"}},
        {"raw": "ma", "abstract": {"onset": "M", "medial": "NONE", "nucleus": "A",
                                   "coda": "N", "tone": "NGANG"}},
    ],
}


def _row(**overrides) -> dict:
    row = {**ROW, "syllables": [dict(s) for s in ROW["syllables"]]}
    row.update(overrides)
    return row


def test_segment_windows_relative_khop_tuyet_doi():
    row = _row()
    relative = segment_windows(row, relative=True)
    assert relative[0] == [("O", 0, 2), ("N", 2, 4), ("C", 4, 5)]
    assert relative[1] == [("O", 7, 9), ("N", 9, 11), ("C", 11, 13)]  # 5 + gap 2 = 7
    absolute = segment_windows(row)
    assert absolute[0][0] == ("O", 10, 12)  # frame_range[0] = 10


def test_ratio_targets_va_mask_b5():
    target, mask = ratio_targets(_row())
    assert torch.allclose(target[0], torch.tensor([0.4, 0.0, 0.4, 0.2]))
    assert torch.allclose(target[1], torch.tensor([1 / 3, 0.0, 1 / 3, 1 / 3]))
    assert mask[0].tolist() == [True, False, True, True]
    assert mask[1].tolist() == [True, False, True, True]

    short = _row(durations=[2, 6], seg_durations=[[1, 0, 1, 0], [2, 0, 2, 2]])
    _, short_mask = ratio_targets(short)
    assert not short_mask[0].any()  # d_syl < 3 → mask hết (B5)


def test_pause_targets_gate_va_mask_cuoi_row():
    """A2 (§43.1): vị trí CUỐI row được gán nhãn khi manifest ghi đủ `gaps` (len == N)."""
    gate, log_gap, mask = pause_targets(_row())
    assert gate.tolist() == [0.0, 0.0]  # gap 2 < 3
    assert abs(float(log_gap[0]) - math.log1p(2)) < 1e-6
    assert mask.tolist() == [True, False]  # row cũ: len(gaps) == N-1 ⇒ vị trí cuối không có nhãn

    loud = _row(gaps=[5])
    gate, log_gap, _ = pause_targets(loud)
    assert gate[0] == 1.0 and abs(float(log_gap[0]) - math.log1p(5)) < 1e-6

    full = _row(gaps=[5, 30])  # manifest A2: gap cuối = pause đuôi ⇒ CÓ nhãn (trước đây bị mask)
    full_gate, full_log_gap, full_mask = pause_targets(full)
    assert full_mask.tolist() == [True, True]
    assert full_gate.tolist() == [1.0, 1.0] and abs(float(full_log_gap[1]) - math.log1p(30)) < 1e-6


def test_f0_targets_s0_va_mask_addendum6():
    row = _row()
    f0 = torch.full((13,), 150.0)
    vuv = torch.ones(13)
    shape, weight, ratio, k_mask = f0_targets(row, f0, vuv, f0_mean=4.8, f0_std=0.2)
    z_constant = (math.log(150.0) - 4.8) / 0.2
    assert abs(float(shape[0, 2, 0]) - z_constant) < 1e-5  # N: s0 == mean z
    assert abs(float(shape[0, 2, 1])) < 1e-6
    # "các": O (onset K vô thanh) = 0; N = ratio; C chỉ 1 frame → không đủ tính s_k → 0
    assert weight[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    # "ma": O (onset M hữu thanh) = 1; N = ratio; C 2 frame đủ → 1
    assert weight[1].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert torch.allclose(ratio[0, 2], torch.tensor(1.0))
    # k_mask: N "các" 2 frame → 2 hệ số; C 1 frame → không đủ tính s_k
    assert k_mask[0, 2].tolist() == [True, True, False, False]
    assert not k_mask[0, 3].any()


def test_f0_targets_vuv_loc_bo_segment():
    row = _row()
    f0 = torch.full((13,), 150.0)
    vuv = torch.ones(13)
    vuv[11:13] = 0.0  # C của "ma" vô thanh
    _, weight, ratio, _ = f0_targets(row, f0, vuv, f0_mean=4.8, f0_std=0.2)
    assert weight[1, 3] == 0.0 and ratio[1, 3] == 0.0


def test_segment_pools():
    row = _row()
    energy = torch.full((13,), 2.5)
    vuv = torch.ones(13)
    energy_pool, voiced_ratio = segment_pools(row, energy, vuv)
    assert abs(float(energy_pool[0, 0]) - 2.5) < 1e-6
    assert abs(float(voiced_ratio[1, 2]) - 1.0) < 1e-6


def test_segment_targets_gop_day_du():
    row = _row()
    features = {"f0": torch.full((13,), 150.0), "vuv": torch.ones(13), "energy": torch.full((13,), 2.5)}
    stats = {"f0_log_mean": 4.8, "f0_log_std": 0.2}
    targets = segment_targets(row, features, stats)
    assert set(targets) == {"seg_ratio", "seg_ratio_mask", "f0_shape", "f0_weight", "f0_k_mask",
                            "seg_energy", "seg_vuv", "seg_phonation", "seg_phonation_mask",
                            "pause_gate", "pause_log_gap", "pause_mask"}
    assert targets["seg_phonation"].shape == (2, 4, 3)  # features test không có kênh → zeros
    assert not targets["seg_phonation_mask"].any()
    assert targets["seg_ratio"].shape == (2, 4)
    assert targets["f0_shape"].shape == (2, 4, 4)
    assert targets["pause_mask"].tolist() == [True, False]
