"""Test chế độ "đọc đều": cắt tại dấu câu + khoảng nghỉ CỐ ĐỊNH (không cần model thật)."""

from __future__ import annotations

import numpy as np

from vifasttts.frontend.chunking import STEADY_PAUSE_COMMA, STEADY_PAUSE_PERIOD
from vifasttts.infer import Synthesizer


class _Stub(Synthesizer):
    """Synthesizer giả: bỏ qua `__init__` (không nạp model), `_render` trả 10 mẫu."""

    def __init__(self) -> None:
        self.sample_rate = 1000  # để đếm mẫu = đếm ms

    def _render(self, text, dialect, speed, options, gate_threshold=0.5, sharpen_alpha=0.0,
                chunk_loudness=False, f0_scale=1.0):
        return np.ones(10, dtype=np.float32)


def test_steady_pauses_by_punctuation():
    """4 mảnh + 3 khoảng lặng: chấm 0,55s · phẩy 0,25s · chấm 0,55s."""
    engine = _Stub()
    pieces = [audio for audio, _ in engine.iter_synthesize_steady("Một. Hai, ba. Bốn")]
    assert len(pieces) == 7
    assert all(np.allclose(piece, 1.0) for piece in pieces[::2])  # 4 mảnh "tiếng"
    assert pieces[1].size == int(1000 * STEADY_PAUSE_PERIOD)
    assert pieces[3].size == int(1000 * STEADY_PAUSE_COMMA)
    assert pieces[5].size == int(1000 * STEADY_PAUSE_PERIOD)


def test_steady_custom_pauses_and_gate_passthrough():
    """Tham số pause tuỳ chỉnh có tác dụng; gate mặc định 0,999 được truyền xuống `_render`."""
    engine = _Stub()
    gates: list[float] = []
    original = engine._render

    def spy(text, dialect, speed, options, gate_threshold=0.5, sharpen_alpha=0.0,
            chunk_loudness=False, f0_scale=1.0):
        gates.append(gate_threshold)
        return original(text, dialect, speed, options, gate_threshold, sharpen_alpha)

    engine._render = spy  # type: ignore[method-assign]
    pieces = [audio for audio, _ in engine.iter_synthesize_steady(
        "A, b. C", pause_comma=0.3, pause_period=0.7, pause_default=0.12)]
    assert pieces[1].size == 300
    assert pieces[3].size == 700
    assert gates and all(gate == 0.999 for gate in gates)


def test_steady_empty_text_raises():
    """Văn bản rỗng → ValueError như `iter_synthesize`."""
    engine = _Stub()
    try:
        list(engine.iter_synthesize_steady("   "))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Phải raise ValueError cho văn bản rỗng")


def test_steady_trims_model_edge_silence():
    """Chunk có lặng thừa hai đầu (model đệm pause) bị cắt sát tiếng trước khi ghép."""
    engine = _Stub()
    padded = np.concatenate([np.zeros(200, dtype=np.float32), np.ones(100, dtype=np.float32),
                             np.zeros(150, dtype=np.float32)])
    engine._render = lambda *a, **k: padded  # type: ignore[method-assign]
    pieces = [audio for audio, _ in engine.iter_synthesize_steady("Một. Hai")]
    # 2 chunk + 1 pause; mỗi chunk còn ~100 mẫu tiếng + margin 2×10 ms = 120 mẫu
    assert pieces[0].size <= 140 and pieces[0].size >= 100
    assert pieces[1].size == int(1000 * 0.55)
