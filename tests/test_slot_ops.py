"""Test `models/slot_ops.py` — bất biến B2, vị trí, contour F0, parity ONNX (R4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vifasttts.data.features import dct4_frames, idct4_frames
from vifasttts.frontend.segments import distribute_frames, expand_syllable
from vifasttts.frontend.syllable import parse_syllable
from vifasttts.models.slot_ops import build_f0_contour, distribute_slots, slot_positions


def _random_batch(rows: int, syllables: int, generator: torch.Generator
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = torch.zeros(rows, syllables, 4)
    active[..., 2] = 1.0  # nucleus luôn active
    extra = torch.rand(rows, syllables, 3, generator=generator) < 0.5
    active[..., 0] = extra[..., 0].float()
    active[..., 1] = extra[..., 1].float()
    active[..., 3] = extra[..., 2].float()
    ratio = torch.rand(rows, syllables, 4, generator=generator) * active * 3.0
    d_syl = torch.randint(0, 60, (rows, syllables), generator=generator)
    return d_syl, ratio, active


def test_distribute_slots_bat_bien_1e5():
    generator = torch.Generator().manual_seed(0)
    processed = 0
    for _ in range(20):
        d_syl, ratio, active = _random_batch(5000, 1, generator)
        d_seg = distribute_slots(d_syl, ratio, active)
        assert d_seg.shape == (5000, 1, 4) and d_seg.dtype == torch.long
        assert torch.equal(d_seg.sum(dim=-1), d_syl)  # Σ == d_syl
        assert (d_seg[active < 0.5] == 0).all()  # inactive = 0
        enough = d_syl >= (active > 0.5).sum(dim=-1)
        assert (d_seg[enough][active[enough] > 0.5] >= 1).all()  # active ≥ 1
        processed += 5000
    assert processed == 100_000


def test_distribute_slots_bat_bien_nhieu_am_tiet():
    generator = torch.Generator().manual_seed(7)
    d_syl, ratio, active = _random_batch(2000, 6, generator)
    d_seg = distribute_slots(d_syl, ratio, active)
    assert torch.equal(d_seg.sum(dim=-1), d_syl)
    assert (d_seg[active < 0.5] == 0).all()


def test_distribute_slots_uu_tien_khi_thieu_frame():
    ratio = torch.ones(1, 1, 4)
    active = torch.ones(1, 1, 4)
    assert distribute_slots(torch.tensor([[1]]), ratio, active).tolist() == [[[0, 0, 1, 0]]]  # N
    assert distribute_slots(torch.tensor([[2]]), ratio, active).tolist() == [[[1, 0, 1, 0]]]  # N, O
    assert distribute_slots(torch.tensor([[3]]), ratio, active).tolist() == [[[1, 0, 1, 1]]]  # N, O, C
    assert distribute_slots(torch.tensor([[4]]), ratio, active).tolist() == [[[1, 1, 1, 1]]]
    assert distribute_slots(torch.tensor([[0]]), ratio, active).tolist() == [[[0, 0, 0, 0]]]


def test_distribute_slots_hoa_pha_theo_chi_so_role():
    ratio = torch.ones(1, 1, 4)
    active = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]])  # O, N, C
    # base 1 mỗi slot; dư 2 = 2/3 → O và N (chỉ số nhỏ hơn) nhận thêm, hòa phá tất định
    assert distribute_slots(torch.tensor([[5]]), ratio, active).tolist() == [[[2, 0, 2, 1]]]


def test_distribute_slots_ratio_chua_softmax_van_bat_bien():
    d_syl = torch.tensor([[9, 4]])
    ratio = torch.tensor([[[3.0, 1.0, 5.0, 1.0], [0.0, 0.0, 1.0, 0.0]]])
    active = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]]])
    d_seg = distribute_slots(d_syl, ratio, active)
    # base 1 mỗi active + floor([1,5, 0,5, 2,5, 0,5] = [1,0,2,0]) + 2 frame dư cho O, M (hòa theo chỉ số)
    assert d_seg.tolist() == [[[3, 2, 3, 1], [0, 0, 4, 0]]]
    assert (d_seg.sum(dim=-1) == d_syl).all()


def test_slot_positions_trong_tung_slot_va_gap():
    durations = torch.tensor([[2, 3, 2]])
    gap = torch.tensor([[False, True, False]])
    pos = slot_positions(durations, gap)
    expected = [0.25, 0.75, 0.0, 0.0, 0.0, 0.25, 0.75]
    assert torch.allclose(pos[0], torch.tensor(expected), atol=1e-6)


def test_build_f0_contour_s0_bang_mean_va_khop_idct4():
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(9, generator=generator)
    coefficients = dct4_frames(values)
    shape = coefficients / math.sqrt(9)  # s_k = c_k/√n
    durations = torch.tensor([[9]])
    pos = slot_positions(durations)
    contour = build_f0_contour(shape.view(1, 1, 4), durations, pos)
    reference = idct4_frames(coefficients, 9)
    assert torch.allclose(contour[0], reference, atol=1e-5)
    assert abs(float(contour.mean()) - float(shape[0])) < 1e-5  # s0 == mean (B1)


def test_distribute_frames_frontend_cung_uu_tien_b2():
    segments = expand_syllable(parse_syllable("quốc"))
    distribution = distribute_frames(2, segments)
    roles = {segment.role: distribution[index] for index, segment in enumerate(segments)}
    assert sum(distribution) == 2 and distribution.count(1) == 2
    assert roles.get("nucleus") == 1 and roles.get("onset") == 1
    assert roles.get("medial", 0) == 0 and roles.get("coda", 0) == 0


class _OpsModule(torch.nn.Module):
    """Bọc 3 hàm slot_ops để export ONNX."""

    def forward(self, d_syl: torch.Tensor, ratio: torch.Tensor, active: torch.Tensor,
                slot_durations: torch.Tensor, gap_mask: torch.Tensor,
                shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_seg = distribute_slots(d_syl, ratio, active)
        pos = slot_positions(slot_durations, gap_mask)
        contour = build_f0_contour(shape, slot_durations, pos)
        return d_seg, pos, contour


def test_slot_ops_onnx_parity(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    generator = torch.Generator().manual_seed(3)
    module = _OpsModule().eval()
    d_syl = torch.tensor([[7, 4, 1]])
    ratio = torch.tensor([[[0.2, 0.1, 0.6, 0.1], [0.4, 0.0, 0.4, 0.2], [0.0, 0.0, 1.0, 0.0]]])
    active = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]]])
    slot_durations = torch.tensor([[0, 2, 3, 2, 0, 1, 3]])  # G_lead + 1 âm tiết, có slot G
    gap_mask = torch.tensor([[True, False, False, False, True, False, False]])
    shape = torch.randn(1, 7, 4, generator=generator)

    with torch.no_grad():
        expected = module(d_syl, ratio, active, slot_durations, gap_mask, shape)
    path = tmp_path / "slot_ops.onnx"
    torch.onnx.export(
        module, (d_syl, ratio, active, slot_durations, gap_mask, shape), str(path),
        input_names=["d_syl", "ratio", "active", "slot_durations", "gap_mask", "shape"],
        output_names=["d_seg", "pos", "f0"], opset_version=18, dynamo=False,
    )
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {"d_syl": d_syl.numpy(), "ratio": ratio.numpy(), "active": active.numpy(),
             "slot_durations": slot_durations.numpy(), "gap_mask": gap_mask.numpy(),
             "shape": shape.numpy()}
    first = session.run(None, feeds)
    second = session.run(None, feeds)  # byte-identical 2 lần chạy
    assert (first[0] == expected[0].numpy()).all()
    assert np.allclose(first[1], expected[1].numpy(), atol=1e-6)
    assert np.allclose(first[2], expected[2].numpy(), atol=1e-5)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
