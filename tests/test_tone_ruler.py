"""Test `tools/tone_ruler.py` — đặc trưng cố định độ dài, cửa sổ nucleus, API chấm âm tiết."""

from __future__ import annotations

import numpy as np
import pytest

from tools.tone_ruler import (
    CONTOUR_MAX,
    FEATURE_SPEC,
    TONES,
    ToneMLP,
    ToneRuler,
    assemble_features,
    class_weights,
    confusion_matrix,
    features_from_spec,
    featurize,
    item_records,
    raw_record,
    records_matrix,
    report,
    seg_durations_from_slot_durations,
    seg_windows,
    valid_mask,
)

F0_STATS = {"f0_log_mean": 4.8, "f0_log_std": 0.17}


def _item(count: int = 3, length: int = 60) -> dict:
    """Item giả: `count` âm tiết, mỗi âm tiết O/M/N/C = 2/1/8/3 frame + gap 4 giữa các tiếng."""
    seg = np.tile(np.array([2, 1, 8, 3]), (count, 1))
    durations = seg.sum(axis=1)
    gaps = np.array([4] * (count - 1) + [0])
    generator = np.random.default_rng(0)
    f0 = 120.0 + 30.0 * np.sin(np.linspace(0, 6.0, length))
    vuv = (np.arange(length) % 7 != 0).astype(np.float64)
    return {
        "f0": f0, "vuv": vuv,
        "energy": generator.normal(size=length).astype(np.float64),
        "hnr": generator.normal(size=length), "jitter": generator.normal(size=length),
        "glottalization": generator.normal(size=length),
        "seg_durations": seg, "durations": durations, "gaps": gaps, "lead": 0,
    }


def test_features_from_spec_matches_assemble_shape():
    names = features_from_spec(FEATURE_SPEC)
    records, _windows, _context = item_records(_item(), f0_stats=F0_STATS)
    matrix = assemble_features(records, spec=FEATURE_SPEC)
    assert matrix.shape == (3, len(names))
    assert len(names) == len(set(names))  # không trùng tên (baseline chọn cột theo tên)
    assert "contour_nuc[0]" in names and "contour_syl[0]" in names


def test_seg_windows_nucleus_va_syllable():
    item = _item(count=3)
    nucleus = seg_windows(item, "nucleus")
    syllable = seg_windows(item, "syllable")
    # âm tiết 0: O 2 + M 1 → nucleus bắt đầu ở frame 3, dài 8; cả tiếng dài 14.
    assert nucleus[0] == (3, 11)
    assert syllable[0] == (0, 14)
    # âm tiết 1: dịch thêm 14 (độ dài tiếng) + 4 (gap) = 18.
    assert syllable[1] == (18, 32)
    assert nucleus[1] == (21, 29)
    assert nucleus[-1][1] - nucleus[-1][0] == 8


def test_seg_windows_slot_durations_layout():
    """Layout phẳng `[lead, (O,M,N,C,G)×N]` của model → cùng cửa sổ với `seg_durations`."""
    lead, seg, gaps = seg_durations_from_slot_durations([5, 2, 1, 8, 3, 4, 3, 0, 6, 2, 9])
    assert lead == 5
    assert seg.tolist() == [[2, 1, 8, 3], [3, 0, 6, 2]]
    assert gaps.tolist() == [4, 9]  # G sau mỗi âm tiết (gap âm tiết cuối là rìa, không dùng)
    item = {"f0": np.zeros(80), "vuv": np.zeros(80), "slot_durations": [5, 2, 1, 8, 3, 4,
                                                                        3, 0, 6, 2, 9]}
    windows = seg_windows(item, "nucleus")
    assert windows[0] == (5 + 3, 5 + 11)  # lead + O + M
    assert windows[1] == (5 + 14 + 4 + 3, 5 + 14 + 4 + 3 + 6)


def test_featurize_dang_va_do_dai_co_dinh():
    item = _item()
    window = seg_windows(item, "nucleus")[0]
    vector = featurize(item["f0"], item["vuv"], window, hnr=item["hnr"],
                       jitter=item["jitter"], glottalization=item["glottalization"],
                       energy=item["energy"], spec=FEATURE_SPEC, f0_stats=F0_STATS)
    assert vector.shape == (len(features_from_spec(FEATURE_SPEC)),)
    assert np.isfinite(vector).all()
    # Cùng cửa sổ nhưng thiếu phonation → vẫn đúng độ dài, cờ phon_ok = 0.
    no_phonation = featurize(item["f0"], item["vuv"], window, spec=FEATURE_SPEC,
                             f0_stats=F0_STATS)
    assert no_phonation.shape == vector.shape
    names = features_from_spec(FEATURE_SPEC)
    assert no_phonation[names.index("phon_ok")] == 0.0
    assert vector[names.index("phon_ok")] == 1.0


def test_featurize_chiu_duoc_khong_co_frame_huu_thanh():
    f0 = np.linspace(100.0, 200.0, 40)
    vuv = np.zeros(40)
    window = (10, 18)
    vector = featurize(f0, vuv, window, spec=FEATURE_SPEC, f0_stats=F0_STATS)
    assert vector.shape == (len(features_from_spec(FEATURE_SPEC)),)
    assert np.isfinite(vector).all()
    names = features_from_spec(FEATURE_SPEC)
    assert vector[names.index("log_voiced")] == 0.0
    assert float(np.abs(vector[[names.index(f"contour_nuc[{i}]") for i in range(16)]]).sum()) \
        == 0.0
    # Cửa sổ ngoài mảng cũng không được vỡ (clamp).
    outside = featurize(f0, vuv, (35, 60), spec=FEATURE_SPEC, f0_stats=F0_STATS)
    assert outside.shape == vector.shape and np.isfinite(outside).all()


def test_featurize_mot_frame_huu_thanh_duy_nhat():
    f0 = np.full(30, 150.0)
    vuv = np.zeros(30)
    vuv[12] = 1.0
    vector = featurize(f0, vuv, (8, 20), spec=FEATURE_SPEC, f0_stats=F0_STATS)
    assert np.isfinite(vector).all()
    names = features_from_spec(FEATURE_SPEC)
    assert vector[names.index("log_voiced")] == pytest.approx(np.log1p(1.0))
    # Contour hằng (một frame) → điểm nào cũng bằng z của 150 Hz.
    z = (np.log(150.0) - F0_STATS["f0_log_mean"]) / F0_STATS["f0_log_std"]
    assert vector[names.index("contour_nuc[5]")] == pytest.approx(z, abs=1e-6)


def test_raw_record_contour_co_dinh_va_resample():
    item = _item()
    window = seg_windows(item, "nucleus")[0]
    record = raw_record(item["f0"], item["vuv"], window, f0_log_mean=F0_STATS["f0_log_mean"],
                        f0_log_std=F0_STATS["f0_log_std"])
    assert record["contour"].shape == (CONTOUR_MAX,)
    assert record["span"] == 8.0
    assert record["n_voiced"] >= 2


def test_valid_mask_dem_dung_ly_do():
    records = records_matrix([
        {"span": 8.0, "n_voiced": 4.0},   # giữ
        {"span": 3.0, "n_voiced": 3.0},   # quá ngắn
        {"span": 9.0, "n_voiced": 1.0},   # quá ít hữu thanh
    ])
    mask, reasons = valid_mask(records, spec={"min_span": 4, "min_voiced": 2})
    assert mask.tolist() == [True, False, False]
    assert reasons == {"short": 1, "few_voiced": 1, "kept": 1, "total": 3}


def test_class_weights_va_confusion():
    tone_ids = np.array([0] * 90 + [1] * 10)
    weights = class_weights(tone_ids)
    assert weights[1] > weights[0]
    present = weights[weights > 0]
    assert present.mean() == pytest.approx(1.0)  # chuẩn hoá trên các lớp có mặt
    matrix = confusion_matrix(np.array([0, 1, 1]), np.array([0, 0, 1]))
    assert matrix[0, 0] == 1 and matrix[1, 0] == 1 and matrix[1, 1] == 1
    summary = report(np.array([0, 1]), np.array([0, 0]), label="test")
    assert summary["total"] == 2 and summary["correct"] == 1
    assert set(summary["per_tone"]) == set(TONES)


def test_tone_ruler_score_items_va_luu_nap(tmp_path):
    records, windows, _context = item_records(_item(), f0_stats=F0_STATS)
    matrix = assemble_features(records, spec=FEATURE_SPEC)
    model = ToneMLP(matrix.shape[1], hidden=8, dropout=0.0)
    ruler = ToneRuler([model], FEATURE_SPEC, F0_STATS, {"n": 0})
    item = _item()
    item["tones"] = ["NGANG", "SAC", "NANG"]
    scored = ruler.score_items([item])
    assert len(scored) == 1 and len(scored[0]) == 3
    for row, (start, end) in zip(scored[0], windows, strict=True):
        assert (row["start"], row["end"]) == (start, end)
        assert row["tone"] in TONES and row["tone_true"] in TONES
        assert sum(row["probs"].values()) == pytest.approx(1.0, abs=1e-3)
    # Âm tiết quá ngắn → valid False, tone None.
    short = _item(count=1)
    short["seg_durations"] = np.array([[0, 0, 2, 0]])
    short["durations"] = np.array([2])
    short["gaps"] = np.array([0])
    short["vuv"] = np.ones(60)
    short_rows = ruler.score_items([short])[0]
    assert short_rows[0]["valid"] is False and short_rows[0]["tone"] is None
    assert short_rows[0]["reason"] == "ngắn"

    from tools.tone_ruler import save_model

    path = tmp_path / "ruler.pt"
    save_model([model], FEATURE_SPEC, path=path, metrics={"val": {"accuracy": 0.0}},
               f0_stats=F0_STATS)
    loaded = ToneRuler.load(path)
    assert len(loaded.models) == 1
    np.testing.assert_allclose(loaded.score_records(records), ruler.score_records(records),
                               rtol=1e-4, atol=1e-5)


def test_spec_rut_gon_van_chay():
    """Spec ít nhóm (vd chỉ contour) vẫn cho vector hợp lệ — tool khác có thể tự chọn."""
    spec = dict(FEATURE_SPEC)
    spec["groups"] = ("contour",)
    spec["contour_source"] = "nucleus"
    names = features_from_spec(spec)
    assert names == [f"contour_nuc[{i}]" for i in range(spec["contour_points"])]
    records, _windows, _context = item_records(_item(), spec=spec, f0_stats=F0_STATS)
    matrix = assemble_features(records, spec=spec)
    assert matrix.shape == (3, spec["contour_points"])
    assert np.isfinite(matrix).all()
