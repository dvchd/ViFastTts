"""Test rẽ nhánh `arch` cho export + runtime ONNX (task 5 đợt 2 — docs/ACOUSTIC_V2.md §10, §12)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from vifasttts.export import export_onnx
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.factory import ARCH_V2, build_model
from vifasttts.models.vifasttts import ViFastTtsModel
from vifasttts.runtime.onnx_engine import OnnxAcousticModel, read_metadata

BASE_MODEL = {
    "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
    "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
    "duration_layers": 1, "variance_layers": 1, "use_postnet": True,
    "role_dim": 4, "segment_mlp_layers": 2,
}


def _config(use_phonation: bool) -> dict:
    """Config tiny như `TINY_CONFIG` của `test_onnx_runtime` + `role_dim`/`segment_mlp_layers`."""
    return {
        "seed": 0,
        "sample_rate": 24000,
        "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20,
                "f_min": 0, "f_max": 12000},
        "model": {**BASE_MODEL, "use_phonation": use_phonation},
    }


def _sizes() -> list[int]:
    vocabs = ComponentVocabs.default()
    return [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
            len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]


# 4 âm tiết với ID hợp lệ (copy từ tests/test_v2_slot5.py).
FEATURES = np.array([[
    [18, 0, 0, 9, 2, 1, 0],
    [1, 0, 0, 2, 0, 1, 0],
    [18, 1, 0, 0, 1, 2, 1],
    [0, 0, 0, 3, 5, 3, 2],
]], dtype=np.int64)

F0_STATS = {
    "shape_mean": [0.0, 0.0, 0.0, 0.0], "shape_std": [1.0, 1.0, 1.0, 1.0],
    "f0_log_mean": 5.0, "f0_log_std": 0.3, "energy_mean": 0.0, "energy_std": 1.0,
}

V2_OUTPUTS_NO_PHON = ["mel", "log_duration", "durations", "f0", "vuv_logits", "energy",
                      "tone_logits", "lengths", "slot_durations", "pause_gate"]


def _expected_v2_outputs(use_phonation: bool) -> list[str]:
    names = list(V2_OUTPUTS_NO_PHON)
    if use_phonation:
        names.insert(6, "phonation")  # sau "energy", trước "tone_logits" (khớp v1)
    return names


def _save_v2_checkpoint(config: dict, path) -> None:
    torch.manual_seed(0)
    model = build_model(config, _sizes(), arch=ARCH_V2)
    torch.save({
        "model": model.state_dict(), "arch": ARCH_V2, "f0_stats": dict(F0_STATS),
        "seg_method": "hybrid", "seg_method_version": "t2.1-hybrid-v1",
        "seg_weights_hash": "abc",
    }, str(path))


@pytest.mark.parametrize("use_phonation", [False, True])
def test_export_v2_input_output_va_metadata(tmp_path, use_phonation: bool):
    config = _config(use_phonation)
    ckpt = tmp_path / "v2.pt"
    _save_v2_checkpoint(config, ckpt)
    out = tmp_path / "v2.onnx"
    export_onnx(config, str(ckpt), str(out), precision="fp32")

    acoustic = OnnxAcousticModel(out, provider="cpu")
    assert acoustic.input_names == ["features", "speed"]
    assert acoustic.output_names == _expected_v2_outputs(use_phonation)

    meta = read_metadata(out)
    assert meta.get("arch") == "v2_slot5"
    assert meta.get("seg_method") == "hybrid"
    assert meta.get("seg_method_version") == "t2.1-hybrid-v1"
    assert meta.get("seg_weights_hash") == "abc"
    assert meta.get("vifasttts.model") == "acoustic"
    assert meta.get("use_phonation") == str(bool(use_phonation))


@pytest.mark.parametrize("use_phonation", [False, True])
def test_runtime_v2_speed_va_tat_dinh(tmp_path, use_phonation: bool):
    config = _config(use_phonation)
    ckpt = tmp_path / "v2.pt"
    _save_v2_checkpoint(config, ckpt)
    out = tmp_path / "v2.onnx"
    export_onnx(config, str(ckpt), str(out), precision="fp32")

    acoustic = OnnxAcousticModel(out, provider="cpu")
    base = acoustic.run(FEATURES, speed=1.0)
    assert base["mel"].shape[0] == 1 and base["mel"].shape[2] == 20
    assert "slot_durations" in base and "pause_gate" in base
    slow = acoustic.run(FEATURES, speed=0.5)
    assert slow["mel"].shape[1] >= base["mel"].shape[1]  # speed < 1 → dài hơn
    again = acoustic.run(FEATURES, speed=1.0)
    assert np.array_equal(again["mel"], base["mel"])  # tất định


def test_export_v2_chua_ho_tro_fp16_int8(tmp_path):
    config = _config(False)
    ckpt = tmp_path / "v2.pt"
    _save_v2_checkpoint(config, ckpt)
    for precision in ("fp16", "int8"):
        with pytest.raises(NotImplementedError):
            export_onnx(config, str(ckpt), str(tmp_path / f"v2.{precision}.onnx"),
                        precision=precision)


@pytest.mark.parametrize("use_postnet", [False, True])
def test_export_v1_khong_doi(tmp_path, use_postnet: bool):
    """Checkpoint v1 (thiếu `arch` → v1_syl): 1 input, tên/thứ tự output cũ, **tên khớp giá trị**.

    Ca `use_postnet=True` từng lệch tên 1 slot do dict output có `mel_pre` (đã đo trên torch 2.14);
    wrapper `_select_outputs_module` giữ đúng hợp đồng.
    """
    config = _config(False)
    config["model"]["use_postnet"] = use_postnet
    torch.manual_seed(0)
    model = ViFastTtsModel(config, _sizes()).eval()
    ckpt = tmp_path / "v1.pt"
    torch.save({"model": model.state_dict()}, str(ckpt))
    out = tmp_path / "v1.onnx"
    export_onnx(config, str(ckpt), str(out), precision="fp32")

    acoustic = OnnxAcousticModel(out, provider="cpu")
    names = ["mel", "log_duration", "durations", "f0", "vuv_logits", "energy", "tone_logits", "lengths"]
    assert acoustic.input_names == ["features"]
    assert acoustic.output_names == names
    meta = read_metadata(out)
    assert meta.get("vifasttts.model") == "acoustic"
    assert "arch" not in meta  # v1 giữ nguyên metadata cũ, không thêm `arch`
    with torch.no_grad():
        reference = model(torch.as_tensor(FEATURES))
    outputs = acoustic.run(FEATURES)  # call-site cũ (không `speed`) vẫn chạy
    assert outputs["mel"].shape[0] == 1 and outputs["mel"].shape[2] == 20
    for name in names:  # tên phải khớp đúng tensor (bug lệch tên với mel_pre)
        assert np.allclose(outputs[name], reference[name].numpy(), atol=1e-4), name
