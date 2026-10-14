"""Test cổng của `tools/bench_checkpoint.py` (X3 §44): None = trượt; ngưỡng tương đối đúng chiều."""

from __future__ import annotations

from tools.bench_checkpoint import guard_gates, hard_gates, target_gates

BASE = {"mel_l1_z": 1.14, "mel_gap": 3.1, "tone_C": 0.80, "f0_nga": 177.0,
        "pause_f1_all": 0.50, "gap_sum_mae_all": 200.0, "cer": 0.21}


def _row(**overrides):
    row = {"mel_l1_z": 1.14, "mel_db_speech": 8.5, "rough": 0.42, "tone_C": 0.80,
           "cer": 0.22, "mel_gap": 3.0, "f0_nga": 178.0, "pause_f1_all": 0.58,
           "gap_sum_mae_all": 140.0, "length_ratio": 1.02}
    row.update(overrides)
    return row


def test_cong_cung_none_la_truot():
    """None ở cổng cứng ⇒ TRƯỢT kèm 'thiếu dữ liệu' (không được coi là đạt)."""
    gates = hard_gates(_row(rough=None), BASE)
    assert gates["rough≥0.40"][0] is False and "thiếu dữ liệu" in gates["rough≥0.40"][1]
    assert hard_gates(_row(cer=None), BASE)["cer≤mốc+0.03"][0] is False


def test_cong_cung_do_dung_chieu():
    assert hard_gates(_row(), BASE)["mel_l1_z≤1.16"][0] is True
    assert hard_gates(_row(mel_l1_z=1.2), BASE)["mel_l1_z≤1.16"][0] is False
    assert hard_gates(_row(cer=0.25), BASE)["cer≤mốc+0.03"][0] is False  # 0,25 > 0,21+0,03


def test_cong_muc_tieu_tuong_doi():
    """pause_f1_all ≥ mốc+0,073; gap_all ≤ mốc×0,714; length_ratio ±0,05."""
    good = target_gates(_row(pause_f1_all=0.58, gap_sum_mae_all=140.0, length_ratio=1.02), BASE)
    assert all(ok for ok, _ in good.values())
    bad = target_gates(_row(pause_f1_all=0.55, gap_sum_mae_all=150.0, length_ratio=1.2), BASE)
    assert not any(ok for ok, _ in bad.values())


def test_rao_chan_so_moc():
    assert guard_gates(_row(), BASE)["mel_gap không tăng"][0] is True
    assert guard_gates(_row(mel_gap=3.2), BASE)["mel_gap không tăng"][0] is False
    assert guard_gates(_row(f0_nga=183.0), BASE)["F0 NGÃ≤mốc+5c"][0] is False
    assert guard_gates(_row(tone_C=0.795), BASE)["tone_C≥mốc−0.01"][0] is True
    assert guard_gates(_row(tone_C=0.785), BASE)["tone_C≥mốc−0.01"][0] is False
