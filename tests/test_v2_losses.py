"""Test `models/v2_losses.py` — công thức + mask (không cần model/dataset thật)."""

from __future__ import annotations

import torch

from vifasttts.models.v2_losses import DEFAULT_WEIGHTS, slot_losses, structural_active

B, N = 2, 3
FEATURES = torch.tensor([
    [[18, 0, 0, 9, 2, 1, 0], [18, 1, 0, 0, 1, 2, 1], [0, 0, 0, 3, 5, 3, 2]],
    [[0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 2, 0, 1, 0], [18, 1, 2, 10, 4, 3, 5]],
])
SHAPE_MEAN = torch.zeros(4)
SHAPE_STD = torch.ones(4)


def _only(key: str) -> dict[str, float]:
    """Trọng số 0 hết, chỉ giữ `key` = 1 để soi từng loss."""
    weights = {name: 0.0 for name in DEFAULT_WEIGHTS}
    weights[key] = 1.0
    return weights


def _fixture() -> tuple[dict, dict]:
    """(outputs hoàn hảo, batch) — loss phải ≈ 0 khi dự đoán khớp target."""
    torch.manual_seed(0)
    active = structural_active(FEATURES)
    durations = torch.tensor([[5, 6, 4], [3, 6, 8]])
    ratio_mask = (active > 0.5) & (durations >= 3).unsqueeze(-1)
    # phân bố thật trên slot active: O 0,2 · N 0,6 · C 0,2 (chuẩn hoá theo active)
    base = torch.tensor([0.2, 0.0, 0.6, 0.2]).view(1, 1, 4).expand(B, N, 4) * active
    normalized = base / base.sum(-1, keepdim=True).clamp_min(1e-6)
    ratio = torch.where(ratio_mask, normalized, torch.zeros(B, N, 4))
    noise = torch.randn(B, N, 4, 4) * 0.1
    f0_weight = torch.where(active > 0.5, torch.ones_like(active), torch.zeros_like(active))
    f0_k_mask = torch.ones(B, N, 4, 4, dtype=torch.bool)
    f0_k_mask[:, :, 3, 2:] = False  # C thiếu hệ số bậc cao
    seg_vuv = torch.where(active > 0.5, torch.full_like(active, 0.9), torch.zeros_like(active))
    seg_energy = torch.randn(B, N, 4) * 0.5
    seg_phonation = torch.randn(B, N, 4, 3) * 0.2
    phonation_mask = (active > 0.5) & f0_k_mask[..., 0]
    pause_mask = torch.zeros(B, N, dtype=torch.bool)
    pause_mask[0, 0] = pause_mask[0, 2] = pause_mask[1, 1] = True
    pause_gate = torch.where(pause_mask, torch.ones(B, N), torch.zeros(B, N))
    pause_gate[1, 1] = 0.0  # gate = 0 (không có pause) nhưng vẫn là vị trí hợp lệ
    log_gap = torch.rand(B, N) * 2.0
    batch = {
        "features": FEATURES,
        "frame_lengths": torch.tensor([12, 8]),
        "mel": torch.randn(B, 12, 20),
        "durations": durations,
        "seg_ratio": ratio,
        "seg_ratio_mask": ratio_mask,
        "f0_shape": noise,
        "f0_weight": f0_weight,
        "f0_k_mask": f0_k_mask,
        "seg_vuv": seg_vuv,
        "seg_energy": seg_energy,
        "seg_phonation": seg_phonation,
        "seg_phonation_mask": phonation_mask,
        "pause_gate": pause_gate,
        "pause_log_gap": log_gap,
        "pause_mask": pause_mask,
    }
    outputs = {
        "mel": batch["mel"].clone(),
        "mel_pre": batch["mel"].clone(),
        "log_duration": torch.log1p(durations.float()),
        # logits = log(q) trên slot hợp lệ → softmax khớp q; CE tối thiểu = entropy(q)
        "seg_ratio_logits": torch.where(ratio > 0, torch.log(ratio.clamp_min(1e-6)),
                                        torch.full_like(ratio, -1e6)),
        "seg_shape": batch["f0_shape"].clone(),
        "seg_vuv_logits": torch.where(seg_vuv >= 0.5, torch.full_like(seg_vuv, 20.0),
                                      torch.full_like(seg_vuv, -20.0)),
        "seg_energy": seg_energy.clone(),
        "seg_phonation": seg_phonation.clone(),
        "tone_logits": torch.nn.functional.one_hot(FEATURES[..., 4], 6).float() * 20.0,
        "pause_logits": torch.where(pause_gate >= 0.5, torch.full((B, N), 20.0),
                                    torch.full((B, N), -20.0)),
        "pause_log_gap": log_gap.clone(),
    }
    return outputs, batch


def _ratio_entropy(batch) -> float:
    """CE tối thiểu khi dự đoán khớp target mềm: entropy(q) gộp **theo âm tiết**."""
    q = batch["seg_ratio"]
    valid = batch["seg_ratio_mask"].any(dim=-1)
    entropy = -(q * torch.where(q > 0, torch.log(q.clamp_min(1e-12)), torch.zeros_like(q))).sum(dim=-1)
    return float((entropy * valid).sum() / valid.sum().clamp_min(1))


def test_hoan_hao_tong_xap_xi_0():
    outputs, batch = _fixture()
    total, losses = slot_losses(outputs, batch, shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    for key, value in losses.items():
        if key == "ratio":  # target mềm → CE tối thiểu = entropy(q), không phải 0
            assert abs(float(value) - _ratio_entropy(batch)) < 1e-4
        else:
            assert float(value) < 1e-5, key
    expected = sum(float(value) * DEFAULT_WEIGHTS[key] for key, value in losses.items())
    assert abs(float(total) - expected) < 1e-5


def test_tong_dung_bang_trong_so():
    outputs, batch = _fixture()
    outputs["mel"] = batch["mel"] + 0.5
    outputs["mel_pre"] = batch["mel"] + 0.5
    weights = {name: 0.0 for name in DEFAULT_WEIGHTS}
    weights["mel"] = 2.0
    total, losses = slot_losses(outputs, batch, weights=weights,
                                shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    expected = 2.0 * losses["mel"]
    assert abs(float(total) - float(expected)) < 1e-6


def test_mel_bo_padding():
    outputs, batch = _fixture()
    total_before, _ = slot_losses(outputs, batch, shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    outputs["mel"] = outputs["mel"].clone()
    outputs["mel"][1, 8:] += 10.0  # frame ngoài frame_lengths[1] = 8
    total_after, _ = slot_losses(outputs, batch, shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert abs(float(total_after) - float(total_before)) < 1e-6


def test_ratio_mask_bo_inactive():
    outputs, batch = _fixture()
    outputs["seg_ratio_logits"] = outputs["seg_ratio_logits"].clone()
    baseline = _ratio_entropy(batch)
    masked = ~batch["seg_ratio_mask"]
    outputs["seg_ratio_logits"][masked] = -5.0  # đổi dự đoán ở slot bị mask → không ảnh hưởng
    _total, losses = slot_losses(outputs, batch, weights=_only("ratio"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert abs(float(losses["ratio"]) - baseline) < 1e-4
    batch["seg_ratio"][masked] = 0.0  # target ở slot bị mask phải bị bỏ qua
    old = outputs["seg_ratio_logits"][masked].clone()
    outputs["seg_ratio_logits"][masked] = 1e9
    _total, losses = slot_losses(outputs, batch, weights=_only("ratio"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert abs(float(losses["ratio"]) - baseline) < 1e-4
    outputs["seg_ratio_logits"][masked] = old  # hoàn tác để đổi tiếp slot hợp lệ
    outputs["seg_ratio_logits"][batch["seg_ratio_mask"]] = -5.0
    _total, losses = slot_losses(outputs, batch, weights=_only("ratio"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert float(losses["ratio"]) > baseline + 0.05


def test_f0_k_mask_bo_he_so_thieu():
    outputs, batch = _fixture()
    outputs["seg_shape"] = outputs["seg_shape"].clone()
    outputs["seg_shape"][..., 3] += 5.0  # k=3 chỉ hợp lệ ở một phần vị trí
    _total, losses = slot_losses(outputs, batch, weights=_only("f0"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    baseline = float(losses["f0"])
    assert baseline > 0.0  # có vị trí k=3 hợp lệ (f0_k_mask True) → thay đổi phải tăng loss
    # đổi ở đúng vị trí bị mask (C thiếu hệ số) → không đổi gì thêm
    before = baseline
    outputs["seg_shape"][:, :, 3, 2:] += 5.0
    _total, losses = slot_losses(outputs, batch, weights=_only("f0"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert abs(float(losses["f0"]) - before) < 1e-6


def test_pause_gap_chi_khi_gate_1():
    outputs, batch = _fixture()
    outputs["pause_log_gap"] = outputs["pause_log_gap"].clone()
    outputs["pause_log_gap"][1, 1] += 10.0  # gate target = 0 → không tính
    _total, losses = slot_losses(outputs, batch, weights=_only("pause_gap"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert float(losses["pause_gap"]) < 1e-6
    outputs["pause_log_gap"][0, 0] += 10.0  # gate target = 1 → tính
    _total, losses = slot_losses(outputs, batch, weights=_only("pause_gap"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert float(losses["pause_gap"]) > 1.0


def test_vuv_energy_phonation_bo_inactive():
    outputs, batch = _fixture()
    inactive = ~(structural_active(FEATURES) > 0.5)
    outputs["seg_vuv_logits"] = outputs["seg_vuv_logits"].clone()
    outputs["seg_energy"] = outputs["seg_energy"].clone()
    outputs["seg_phonation"] = outputs["seg_phonation"].clone()
    outputs["seg_vuv_logits"][inactive] = -20.0
    outputs["seg_energy"][inactive] = 100.0
    outputs["seg_phonation"][inactive] = 50.0
    _total, losses = slot_losses(outputs, batch, shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    for key in ("vuv", "energy", "phonation"):
        assert float(losses[key]) < 1e-5, key

def test_f0_delta_chi_phu_thuoc_hinh_dang():
    """Delta loss = 0 khi khớp target; đổi `s0` (mức trung bình) không ảnh hưởng, đổi `s1` thì tăng."""
    outputs, batch = _fixture()
    _total, losses = slot_losses(outputs, batch, weights=_only("f0_delta"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    baseline = float(losses["f0_delta"])
    assert baseline < 1e-5  # dự đoán khớp target → 0
    shifted = outputs["seg_shape"].clone()
    shifted[..., 0] += 3.0  # chỉ đổi s0
    outputs_shifted = {**outputs, "seg_shape": shifted}
    _total, losses = slot_losses(outputs_shifted, batch, weights=_only("f0_delta"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert abs(float(losses["f0_delta"]) - baseline) < 1e-6
    tilted = outputs["seg_shape"].clone()
    tilted[..., 1] += 1.0  # đổi s1 (độ dốc)
    _total, losses = slot_losses({**outputs, "seg_shape": tilted}, batch, weights=_only("f0_delta"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD)
    assert float(losses["f0_delta"]) > 0.5


def test_f0_prior_mem_theo_thanh_va_mask_nucleus():
    """Prior: 0 khi nucleus khớp prior; bỏ qua slot không phải nucleus và segment không hữu thanh."""
    outputs, batch = _fixture()
    tones = batch["features"][..., 4]
    # prior giả theo thanh; ép dự đoán nucleus = prior để kiểm tra "khớp → loss 0"
    tone_prior = torch.randn(6, 4, generator=torch.Generator().manual_seed(7))
    shape = outputs["seg_shape"].clone()
    for b in range(B):
        for n in range(N):
            if batch["f0_weight"][b, n, 2] > 0:
                shape[b, n, 2] = tone_prior[tones[b, n]]
    outputs = {**outputs, "seg_shape": shape}
    _total, losses = slot_losses(outputs, batch, weights=_only("f0_prior"),
                                 shape_mean=SHAPE_MEAN, shape_std=SHAPE_STD, tone_prior=tone_prior)
    assert float(losses["f0_prior"]) < 1e-6
    # sửa hình dạng của slot O (không phải nucleus) → prior không đổi
    polluted = outputs["seg_shape"].clone()
    polluted[..., 0, :] += 5.0
    _total, losses = slot_losses({**outputs, "seg_shape": polluted}, batch,
                                 weights=_only("f0_prior"), shape_mean=SHAPE_MEAN,
                                 shape_std=SHAPE_STD, tone_prior=tone_prior)
    assert float(losses["f0_prior"]) < 1e-6
    # sửa nucleus → prior tăng
    polluted_n = outputs["seg_shape"].clone()
    polluted_n[..., 2, :] += 1.0
    _total, losses = slot_losses({**outputs, "seg_shape": polluted_n}, batch,
                                 weights=_only("f0_prior"), shape_mean=SHAPE_MEAN,
                                 shape_std=SHAPE_STD, tone_prior=tone_prior)
    assert float(losses["f0_prior"]) > 0.5
