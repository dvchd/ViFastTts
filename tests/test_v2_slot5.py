"""Test model `v2_slot5` + factory/warm-start (task 5 — docs/ACOUSTIC_V2.md §1–§11)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vifasttts.models.factory import (
    ARCH_V1,
    ARCH_V2,
    arch_of_checkpoint,
    build_model,
    warm_start_v2_from_v1,
)
from vifasttts.models.v2_slot5 import ViFastTtsModelV2
from vifasttts.models.vifasttts import ViFastTtsModel

TINY = {
    "seed": 0,
    "sample_rate": 24000,
    "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
    "model": {
        "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
        "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
        "duration_layers": 1, "variance_layers": 1, "use_phonation": False, "use_postnet": True,
        "role_dim": 4, "segment_mlp_layers": 2,
    },
}
VOCAB = [24, 2, 15, 11, 6, 4, 6]

# 4 âm tiết: "các" (K, C), "ma" (M, N), "qua" (K+W), "ông" (NONE+N+NG, sentence end)
FEATURES = torch.tensor([[
    [18, 0, 0, 9, 2, 1, 0],
    [1, 0, 0, 2, 0, 1, 0],
    [18, 1, 0, 0, 1, 2, 1],
    [0, 0, 0, 3, 5, 3, 2],
]])
D_SYL = torch.tensor([[8, 6, 7, 9]])
SEG = torch.tensor([[[3, 0, 3, 2], [2, 0, 2, 2], [0, 0, 4, 3], [3, 0, 4, 2]]])
GAPS = torch.tensor([[2, 0, 5, 0]])
LEAD = torch.tensor([4])


def _model(use_phonation: bool = False) -> ViFastTtsModelV2:
    config = {**TINY, "model": {**TINY["model"], "use_phonation": use_phonation}}
    torch.manual_seed(0)
    return ViFastTtsModelV2(config, VOCAB).eval()


def test_tf_bat_bien_slot_va_do_dai():
    model = _model(use_phonation=True)
    with torch.no_grad():
        out = model(FEATURES, durations=D_SYL, seg_durations=SEG, gaps=GAPS, lead=LEAD)
    expected_slots = [4] + [value for n in range(4) for value in
                            [*SEG[0, n].tolist(), int(GAPS[0, n])]]
    assert out["slot_durations"][0].tolist() == expected_slots
    total = int(out["slot_durations"].sum())
    expected_total = int(LEAD) + int(D_SYL.sum()) + int(GAPS.sum())
    assert total == expected_total == out["mel"].shape[1]  # G_lead + Σ slot == T
    assert out["f0"].shape == (1, expected_total)
    assert out["vuv_logits"].shape == (1, 4) and out["phonation"].shape == (1, 4, 3)
    assert out["durations"].tolist() == D_SYL.tolist()


def test_free_chia_slot_bang_ratio_bat_bien():
    model = _model()
    with torch.no_grad():
        out = model(FEATURES)
    slot = out["slot_durations"]
    assert int(slot.sum()) == out["mel"].shape[1]
    d_seg = slot[:, 1:].reshape(1, 4, 5)[..., :4]
    assert torch.equal(d_seg.sum(dim=-1), out["durations"])  # Σ d_seg == d_syl (B2)
    active = model._active_mask(FEATURES)
    assert (d_seg[active < 0.5] == 0).all()  # inactive = 0


def test_speed_bat_bien_va_khong_scale_khi_teacher_forced():
    model = _model()
    with torch.no_grad():
        base = model(FEATURES, speed=torch.tensor([1.0]))
        no_speed = model(FEATURES)
        slow = model(FEATURES, speed=torch.tensor([0.5]))
        fast = model(FEATURES, speed=torch.tensor([1.25]))
        forced = model(FEATURES, durations=D_SYL, seg_durations=SEG, gaps=GAPS, lead=LEAD,
                       speed=torch.tensor([0.5]))
        forced_plain = model(FEATURES, durations=D_SYL, seg_durations=SEG, gaps=GAPS, lead=LEAD)
    assert torch.equal(base["mel"], no_speed["mel"])  # speed=1.0 byte-identical
    assert torch.equal(forced["slot_durations"], forced_plain["slot_durations"])  # tf không scale
    for other in (slow, fast):
        assert torch.equal(other["pause_gate"] > 0.5, base["pause_gate"] > 0.5)  # vị trí pause bất biến
    assert slow["mel"].shape[1] >= base["mel"].shape[1]  # speed < 1 → dài hơn


def test_gate_ep_buoc_gap_min_1():
    model = _model()
    model.pause.out.weight.data.zero_()
    model.pause.out.bias.data.copy_(torch.tensor([10.0, math.log1p(10.0)]))  # gate=1, log_gap=log1p(10)
    with torch.no_grad():
        for speed, expected in ((0.8, None), (1.0, 10), (1.25, 8), (2.0, 5)):
            out = model(FEATURES, speed=torch.tensor([speed]))
            gap_slots = out["slot_durations"][:, 1:].reshape(1, 4, 5)[..., 4]
            assert (out["pause_gate"] > 0.5).all()
            if expected is None:
                assert (gap_slots >= 1).all()  # gate=1 → ≥ 1 frame kể cả speed 0.8
            else:
                assert torch.equal(gap_slots, torch.full_like(gap_slots, expected))
    model.pause.out.bias.data.copy_(torch.tensor([-10.0, math.log1p(10.0)]))  # gate=0
    with torch.no_grad():
        out = model(FEATURES, speed=torch.tensor([0.8]))
        gap_slots = out["slot_durations"][:, 1:].reshape(1, 4, 5)[..., 4]
    assert (gap_slots == 0).all()  # gate=0 → 0 frame kể cả speed 0.8


def test_warm_start_v2_tu_v1():
    torch.manual_seed(1)
    model_v1 = ViFastTtsModel({**TINY, "model": {**TINY["model"], "use_phonation": False}}, VOCAB)
    model_v2 = _model()
    report = warm_start_v2_from_v1(model_v2, {"model": model_v1.state_dict(), "arch": ARCH_V1})
    assert "embedding.proj.weight" in report["loaded"]
    assert all(not key.startswith(("role_emb", "ratio_mlp", "variance_mlp", "pause", "slot_expander"))
               for key in report["loaded"])
    assert not report["skipped"]
    state_v1 = model_v1.state_dict()
    state_v2 = model_v2.state_dict()
    for key in report["loaded"]:
        assert torch.equal(state_v1[key], state_v2[key])
    # module mới của v2 phải khác v1 (không bị copy ngầm)
    assert not any("role_emb" in key or "ratio_mlp" in key for key in report["loaded"])


def test_factory_arch():
    assert arch_of_checkpoint({"arch": ARCH_V2}) == ARCH_V2
    assert arch_of_checkpoint({}) == ARCH_V1
    assert isinstance(build_model(TINY, VOCAB, ARCH_V1), ViFastTtsModel)
    assert isinstance(build_model(TINY, VOCAB, ARCH_V2), ViFastTtsModelV2)
    try:
        build_model(TINY, VOCAB, "v3")
        raise AssertionError("phải raise")
    except ValueError as error:
        assert "arch không hợp lệ" in str(error)


def test_slot_durations_khong_phu_thuoc_speed_khi_co_seg():
    model = _model()
    with torch.no_grad():
        a = model(FEATURES, durations=D_SYL, seg_durations=SEG, gaps=GAPS, lead=LEAD)
        b = model(FEATURES, durations=D_SYL, seg_durations=SEG, gaps=GAPS, lead=LEAD)
    assert torch.equal(a["slot_durations"], b["slot_durations"])  # tất định


def _export_wrapper(model: ViFastTtsModelV2) -> torch.nn.Module:
    """Bọc model để export đúng 10 output (thứ tự cũ + `slot_durations`, `pause_gate` cuối)."""
    order = ("mel", "log_duration", "durations", "f0", "vuv_logits", "energy",
             "tone_logits", "lengths", "slot_durations", "pause_gate")

    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model  # phải là submodule — closure sẽ làm tham số thành constant lúc export

        def forward(self, features: torch.Tensor, speed: torch.Tensor) -> tuple[torch.Tensor, ...]:
            outputs = self.model(features, speed=speed)
            return tuple(outputs[name] for name in order)

    return _Wrapper().eval()


def test_v2_onnx_export_va_parity(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = _model(use_phonation=False)
    speed = torch.tensor([1.0], dtype=torch.float32)
    with torch.no_grad():
        expected = model(FEATURES, speed=speed)
    output_names = ["mel", "log_duration", "durations", "f0", "vuv_logits", "energy",
                    "tone_logits", "lengths", "slot_durations", "pause_gate"]
    path = tmp_path / "v2_slot5.onnx"
    with torch.no_grad():  # bắt buộc: v2 slice/index Parameter trong forward (baked thành constant)
        torch.onnx.export(
            _export_wrapper(model), (FEATURES, speed), str(path),
            input_names=["features", "speed"], output_names=output_names,
            dynamic_axes={"features": {0: "batch", 1: "syllables"}},
            opset_version=18, dynamo=False,
        )
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = [item.name for item in session.get_inputs()]
    assert inputs == ["features", "speed"]  # thứ tự input giữ nguyên + speed ở cuối
    assert session.get_inputs()[1].type == "tensor(float)"
    assert [item.name for item in session.get_outputs()] == output_names  # output cũ trước, mới cuối
    feeds = {"features": FEATURES.numpy(), "speed": speed.numpy()}
    outputs = session.run(None, feeds)
    named = dict(zip(output_names, outputs))
    assert np.allclose(named["mel"], expected["mel"].numpy(), atol=1e-5)
    assert (named["durations"] == expected["durations"].numpy()).all()
    short = [int(value) for value in expected["slot_durations"][0]]
    assert named["slot_durations"][0][:len(short)].tolist() == short
    assert np.allclose(named["pause_gate"], expected["pause_gate"].numpy(), atol=1e-6)

    # câu dài hơn (N = 6) — kiểm tra trục động thật sự chạy
    longer = FEATURES.repeat(1, 2, 1)[:, :6]
    with torch.no_grad():
        expected_long = model(longer, speed=speed)
    outputs_long = session.run(None, {"features": longer.numpy(), "speed": speed.numpy()})
    named_long = dict(zip(output_names, outputs_long))
    assert named_long["mel"].shape[1] == expected_long["mel"].shape[1]
    assert np.allclose(named_long["mel"], expected_long["mel"].numpy(), atol=1e-4)

def test_f0_output_dung_thang_log_tu_nhien():
    """`f0` output = f0_log_mean + f0_log_std × contour(s_k) — bug đã gặp: thiếu affine ngoài
    → f0 ra ~1,0 log (~3 Hz) thay vì ~4,8 log (~140 Hz); khoá bằng ca đơn giản (s_k = hằng số)."""
    model = _model()
    with torch.no_grad():  # variance head ra 0 → shape_z = 0 → contour = shape_mean (raw)
        model.variance_mlp.net[-1].weight.zero_()
        model.variance_mlp.net[-1].bias.fill_(0.0)
        model.shape_mean.copy_(torch.tensor([0.5, 0.0, 0.0, 0.0]))
        model.shape_std.copy_(torch.tensor([2.0, 1.0, 1.0, 1.0]))
        model.f0_log_mean.copy_(torch.tensor(4.8))
        model.f0_log_std.copy_(torch.tensor(0.2))
    with torch.no_grad():
        out = model(FEATURES, durations=D_SYL, seg_durations=SEG, gaps=GAPS, lead=LEAD)
    expected = 4.8 + 0.2 * 0.5  # log-F0 tự nhiên = mean + std × s0 (các s_k bậc cao = 0)
    total = int(out["slot_durations"].sum())
    lead = int(LEAD)
    # frame đầu tiên của âm tiết 0 (sau lead) = trong slot O → đúng thang; frame lead/G = 0
    assert abs(float(out["f0"][0, lead]) - expected) < 1e-4
    assert float(out["f0"][0, 0]) == 0.0
    gap_start = lead + int(SEG[0, 0].sum())
    assert float(out["f0"][0, gap_start]) == 0.0  # slot G giữa âm tiết 0 và 1 → 0
    next_start = gap_start + int(GAPS[0, 0])
    assert abs(float(out["f0"][0, next_start]) - expected) < 1e-4
    assert total == out["mel"].shape[1]


def test_energy_output_khong_nhan_lai_thang():
    """Head energy học thô (loss so raw) → output `energy` phải bằng đúng giá trị head (nucleus)."""
    model = _model()
    with torch.no_grad():
        out = model(FEATURES, durations=D_SYL, seg_durations=SEG, gaps=GAPS, lead=LEAD)
    assert torch.allclose(out["energy"], out["seg_energy"][..., 2])


def test_load_model_state_tha_buffer_moi_nhung_van_chat():
    """Checkpoint cũ thiếu `tone_prior_z` (buffer suy từ f0_stats) → nạp được; thiếu weight thật → raise."""
    from vifasttts.models.factory import load_model_state

    model = _model()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state.pop("tone_prior_z")  # checkpoint thời trước T3.2
    load_model_state(model, state)  # OK
    state.pop("embedding.proj.weight")
    try:
        load_model_state(model, state)
        raise AssertionError("phải raise khi thiếu weight thật")
    except ValueError as error:
        assert "embedding.proj.weight" in str(error)


def test_cross_slot_context_item7():
    """Item 7 (v3.5): slot N thấy embedding coda, slot C thấy embedding nucleus."""
    config = {**TINY, "model": {**TINY["model"], "cross_slot_context": True}}
    torch.manual_seed(0)
    model = ViFastTtsModelV2(config, VOCAB).eval()
    assert hasattr(model, "coda_to_nucleus") and hasattr(model, "nucleus_to_coda")
    segments = model._segment_features(FEATURES, torch.randn(1, 4, 16))
    assert segments.shape == (1, 4, 4, 16 + 8 + 4)
    out = model(FEATURES, speed=torch.tensor([1.0]), durations=D_SYL, gaps=GAPS, lead=LEAD)
    assert torch.isfinite(out["mel"]).all() and out["mel"].shape[0] == 1
    # tắt cờ thì model cũ vẫn dựng được (tương thích checkpoint/ONNX cũ)
    plain = ViFastTtsModelV2(TINY, VOCAB).eval()
    assert not plain.cross_slot_context


def test_cross_slot_context_grad_chay_ve_embedding():
    """Hai projection phải NHẬN gradient (không phải tham số chết)."""
    config = {**TINY, "model": {**TINY["model"], "cross_slot_context": True}}
    torch.manual_seed(0)
    model = ViFastTtsModelV2(config, VOCAB).train()
    segments = model._segment_features(FEATURES, torch.randn(1, 4, 16))
    segments.sum().backward()
    assert model.coda_to_nucleus.weight.grad is not None
    assert model.coda_to_nucleus.weight.grad.abs().sum() > 0
