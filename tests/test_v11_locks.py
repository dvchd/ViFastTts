"""V11.A5/A7: khóa `gate_threshold` (mặc định không đổi hành vi) + `cer_prod` (đệm lặng 150 ms)."""

from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.test_export_v2 import FEATURES, _config, _sizes
from vifasttts.models.factory import ARCH_V2, build_model
from vifasttts.runtime.onnx_engine import OnnxAcousticModel


def test_gate_threshold_mac_dinh_giu_nguyen_dau_ra():
    """V11.A7: không truyền `gate_threshold` == truyền 0,5 (từng bit) — mặc định không đổi hành vi."""
    model = build_model(_config(False), _sizes(), arch=ARCH_V2).eval()
    features = torch.tensor(FEATURES, dtype=torch.long)
    speed = torch.tensor([1.0], dtype=torch.float32)
    with torch.no_grad():
        default = model(features, speed=speed)["mel"]
        explicit = model(features, speed=speed, gate_threshold=0.5)["mel"]
        strict = model(features, speed=speed, gate_threshold=0.9)["mel"]
    assert torch.equal(default, explicit)
    # ngưỡng cao hơn chỉ có thể BỎ bớt gap ⇒ số frame không tăng (đổi là đúng chức năng)
    assert strict.shape[1] <= default.shape[1]


class _FakeSession:
    """Session giả: chỉ ghi lại feeds + trả một output."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.feeds: dict | None = None

    def get_inputs(self) -> list:
        return [types.SimpleNamespace(name=name) for name in self.names]

    def get_outputs(self) -> list:
        return [types.SimpleNamespace(name="mel")]

    def get_modelmeta(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(custom_metadata_map={"arch": "v2_slot5"})

    def run(self, _outputs, feeds):
        self.feeds = feeds
        return [np.zeros(1, dtype=np.float32)]


def _engine(names: list[str]) -> OnnxAcousticModel:
    engine = object.__new__(OnnxAcousticModel)
    engine.session = _FakeSession(names)
    engine.input_names = names
    engine.input_name = names[0]
    engine.speed_input_name = names[1] if len(names) > 1 else None
    engine.gate_input_name = names[2] if len(names) > 2 else None
    engine.output_names = ["mel"]
    engine.metadata = {"arch": "v2_slot5"}
    engine.is_v2 = True
    return engine


def test_engine_truyen_gate_threshold_khi_graph_co_input_thu_ba():
    """V11.A7: graph 3 input nhận `gate_threshold`; graph cũ 2 input chạy y như trước."""
    features = np.zeros((1, 4, 7), dtype=np.int64)
    engine = _engine(["features", "speed", "gate_threshold"])
    engine.run(features, speed=1.0, gate_threshold=0.6)
    assert list(engine.session.feeds) == ["features", "speed", "gate_threshold"]
    assert engine.session.feeds["gate_threshold"].tolist() == [pytest.approx(0.6)]
    old = _engine(["features", "speed"])
    old.run(features, speed=1.0, gate_threshold=0.6)
    assert list(old.session.feeds) == ["features", "speed"]


def test_pad_silence_cer_prod():
    """V11.A5: đệm lặng cố định 150 ms hai đầu (24 kHz ⇒ 3.600 mẫu mỗi đầu)."""
    from tools.eval_acoustic import CER_PROD_PAD_MS, _pad_silence

    assert CER_PROD_PAD_MS == 150.0
    wav = torch.ones(1000)
    padded = _pad_silence(wav, 24000)
    n = round(24000 * 150 / 1000)
    assert n == 3600
    assert padded.shape[0] == 1000 + 2 * n
    assert torch.equal(padded[:n], torch.zeros(n))
    assert torch.equal(padded[-n:], torch.zeros(n))
    assert torch.equal(padded[n:n + 1000], wav)


def test_probe_chan_khi_da_co_last_pt():
    """V11S.A4: chế độ probe chặn nếu thư mục probe đã có last.pt (trừ --overwrite-probe)."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if not (root / "runs/acoustic_v2_A2_fixtail_probe/last.pt").exists():
        pytest.skip("thư mục probe là dữ liệu local (runs/)")
    proc = subprocess.run(["bash", str(root / "scripts/run_A2_fixgap.sh"), "probe"],
                          cwd=root, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "CHẶN" in proc.stdout


def test_sharpen_mel_sh25_trong_source():
    """sh25 = `runtime.postprocess.sharpen_mel` — α=0 không đổi, α=1,5 đổi giá trị, giữ shape."""
    from vifasttts.runtime.postprocess import sharpen_mel

    mel = torch.randn(40, 20)
    assert torch.equal(sharpen_mel(mel, 0.0), mel)
    sharp = sharpen_mel(mel, 1.5)
    assert sharp.shape == mel.shape
    assert not torch.allclose(sharp, mel)
    assert sharp.abs().mean() > mel.abs().mean()  # làm nét ⇒ biên độ tăng (RMS thực đo 0,031→0,039)
