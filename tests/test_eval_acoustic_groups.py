"""Test phần MỚI của `tools/eval_acoustic.py` (14/10/2026) — cache giả + model ngẫu nhiên, CPU.

Ba việc của kiểm toán "hiện tượng mù":
  1. `--group-by` → metric cấp âm tiết tách theo nhóm (`groups` trong JSON);
  2. nhịp nghỉ free-run: `gap_sum_mae_frames`, `pause_f1`, `mel_gap`;
  3. ngữ điệu: `intonation.slope_*` (thật vs model) + declination thanh NGANG.

Nguyên tắc: **không truyền `group_by` ⇒ JSON y như cũ** (không có khoá `groups`).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import eval_acoustic as ev

from vifasttts.data.cache import feature_cache_key
from vifasttts.data.dataset_v2 import SlotDataset
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.factory import build_model

MEL_DIM = 8
CONFIG = {
    "sample_rate": 24000,
    "mel": {"hop_length": 256, "n_fft": 256, "n_mels": MEL_DIM},
    "f0": {"method": "pyworld"},
    "phonation": {"enabled": False, "channels": []},
    "model": {
        "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
        "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
        "duration_layers": 1, "variance_layers": 1, "use_phonation": False,
        "use_postnet": False,
    },
}
F0_STATS = {
    "f0_log_mean": 4.8, "f0_log_std": 0.2,
    "shape_mean": [0.0, 0.0, 0.0, 0.0], "shape_std": [1.0, 1.0, 1.0, 1.0],
    "energy_mean": 0.0, "energy_std": 1.0,
}


def _syllable(raw: str, *, onset: str, nucleus: str, coda: str, tone: str,
              punctuation: str | None = None, status: str | None = "VALID") -> dict:
    """Âm tiết giả trong manifest (đủ khoá `abstract` cho `encode_syllables`)."""
    syllable = {
        "raw": raw, "normalized": raw,
        "orthography": {"onset": None, "medial": None, "nucleus": "a", "coda": None},
        "abstract": {"onset": onset, "medial": "NONE", "nucleus": nucleus, "coda": coda,
                     "tone": tone},
        "boundaries": {"word_end": True, "phrase_end": False,
                       "sentence_end": punctuation in (".", "?")},
        "confidence": 1.0, "punctuation": punctuation,
    }
    if status is not None:
        syllable["status"] = status
    return syllable


#: Row có 5 âm tiết: coda tắc (T/C) vs coda mở (NONE/N/NG), 2 câu (`.` rồi `?`),
#: thanh NGANG ở nhiều vị trí (để đo declination), nhãn `status` ĐỦ (⇒ nhóm hợp lệ).
ROW1 = {
    "id": "clip__s00",
    "audio_path": "/khong/ton/tai_s00.wav",
    "frame_range": [10, 50],
    "durations": [8, 8, 8, 8, 8],
    "gaps": [0, 6, 0, 0],
    "lead_frames": 0,
    "text": "các bạn hát vang nhé",
    "seg_durations": [[2, 0, 4, 2], [2, 0, 4, 2], [2, 0, 4, 2], [2, 0, 4, 2], [2, 0, 4, 2]],
    "syllables": [
        _syllable("các", onset="K", nucleus="A", coda="C", tone="SAC", punctuation="."),
        _syllable("bạn", onset="B", nucleus="A", coda="N", tone="NANG"),
        _syllable("hát", onset="H", nucleus="A", coda="T", tone="SAC"),
        _syllable("vang", onset="V", nucleus="A", coda="NG", tone="NGANG", punctuation="?"),
        _syllable("nhé", onset="NH", nucleus="E", coda="NONE", tone="NGANG"),
    ],
}
#: Row KHÔNG có khoá `status` trong âm tiết ⇒ `--group-by status` phải bị bỏ và ghi rõ.
ROW2 = {
    "id": "clip__s01",
    "audio_path": "/khong/ton/tai_s01.wav",
    "frame_range": [30, 70],
    "durations": [8, 8, 8, 8, 8],
    "gaps": [0, 0, 5, 0],
    "lead_frames": 0,
    "text": "mưa rơi trên phố nhỏ",
    "seg_durations": [[2, 0, 4, 2], [2, 0, 4, 2], [2, 0, 4, 2], [2, 0, 4, 2], [2, 0, 4, 2]],
    "syllables": [
        _syllable("mưa", onset="M", nucleus="ƯƠ", coda="NONE", tone="NGANG", status=None),
        _syllable("rơi", onset="R", nucleus="Ơ", coda="NONE", tone="NGANG", status=None),
        _syllable("trên", onset="TR", nucleus="Ê", coda="N", tone="NGANG", status=None),
        _syllable("phố", onset="PH", nucleus="Ô", coda="NONE", tone="SAC", status=None,
                  punctuation="."),
        _syllable("nhỏ", onset="NH", nucleus="O", coda="NONE", tone="HOI", status=None),
    ],
}


def _write_cache(cache_dir: Path, config: dict, row: dict) -> None:
    """Cache giả: f0/vuv có hình dạng F0 giảm dần để hồi quy độ dốc có nghĩa."""
    start = row["frame_range"][0]
    length = start + (row["frame_range"][1] - start)
    frames = torch.arange(length, dtype=torch.float32)
    f0 = 180.0 * torch.exp(-0.002 * frames)  # logF0 dốc xuống đều (declination thật)
    torch.save({
        "mel": torch.randn(length, MEL_DIM),
        "energy": torch.full((length,), 2.5),
        "f0": f0,
        "vuv": torch.ones(length),
        "hnr": torch.zeros(length),
        "glottalization": torch.full((length,), 0.1),
        "jitter": torch.zeros(length),
    }, cache_dir / f"{feature_cache_key(row, config)}.pt")


@pytest.fixture()
def setup(tmp_path):
    torch.manual_seed(0)
    manifest = tmp_path / "slot4.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in (ROW1, ROW2)),
                        encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    mel_stats = tmp_path / "mel_stats.json"
    mel_stats.write_text(json.dumps({"mean": [0.0] * MEL_DIM, "std": [1.0] * MEL_DIM}),
                         encoding="utf-8")
    config = {**CONFIG, "mel": {**CONFIG["mel"], "normalize_path": str(mel_stats)}}
    for row in (ROW1, ROW2):
        _write_cache(cache_dir, config, row)
    dataset = SlotDataset(manifest, config, F0_STATS, cache_dir=cache_dir)
    sizes = [len(getattr(ComponentVocabs.default(), name)) for name in
             ("onset", "medial", "nucleus", "coda", "tone", "boundary", "punctuation")]
    model = build_model(config, sizes, arch="v2_slot5")
    model.load_f0_stats(F0_STATS)
    with torch.no_grad():
        # Kéo d_syl tự dự đoán về cỡ dài hơn câu thật: nếu model ngẫu nhiên đoán 1 frame/âm tiết
        # thì vùng overlap rỗng và mọi metric free-run (kể cả `mel_gap`) thành vô nghĩa.
        model.duration.out.bias.fill_(math.log1p(40.0))
    checkpoint = tmp_path / "v2.pt"
    torch.save({"model": model.state_dict(), "arch": "v2_slot5", "f0_stats": F0_STATS},
               checkpoint)
    return config, dataset, checkpoint


# ── việc 1: `--group-by` ───────────────────────────────────────────────────────
def test_group_labels_tu_features_va_manifest(setup):
    _config, dataset, _ = setup
    vocabs = ComponentVocabs.default()
    labels, skipped = ev._group_labels(ROW1, dataset[0]["features"], vocabs,
                                       ["coda", "tone", "status"])
    assert labels["coda"] == ["C", "N", "T", "NG", "NONE"]
    assert labels["tone"] == ["SAC", "NANG", "SAC", "NGANG", "NGANG"]
    assert labels["status"] == ["VALID"] * 5
    assert skipped == []
    _, skipped_missing = ev._group_labels(ROW2, dataset[1]["features"], vocabs, ["status"])
    assert skipped_missing == ["status"]  # manifest không ghi `status` ⇒ bỏ nhóm, không bịa


def test_group_by_khong_doi_mac_dinh(setup):
    config, dataset, checkpoint = setup
    model, _ = ev.load_eval_model(str(checkpoint), config, f0_stats=F0_STATS)
    plain = ev.evaluate_tf_v2(model, dataset, config, use_seg=True, device="cpu")
    assert "groups" not in plain  # mặc định y như cũ
    grouped = ev.evaluate_tf_v2(model, dataset, config, use_seg=True, device="cpu",
                                group_by=["coda"])
    assert "groups" in grouped
    # mọi metric cũ phải giữ nguyên giá trị khi bật nhóm (chỉ THÊM khoá)
    for key in ("mel_l1_z", "mel_mae_db", "vuv_accuracy", "f0_rmse_cents_by_tone"):
        assert grouped[key] == plain[key]


def test_evaluate_tf_seg_v2_group_by(setup):
    config, dataset, checkpoint = setup
    model, _ = ev.load_eval_model(str(checkpoint), config, f0_stats=F0_STATS)
    report = ev.evaluate_tf_v2(model, dataset, config, use_seg=True, device="cpu",
                               group_by=["coda", "tone", "status"])
    assert report["groups_by"] == ["coda", "tone", "status"]
    assert report["groups_note"]
    # `status` có ở ROW1, thiếu ở ROW2 ⇒ vẫn gộp phần có nhãn + ghi rõ `groups_partial`
    assert set(report["groups"]) == {"coda", "tone", "status"}
    assert report["groups_partial"] == ["status"]
    assert set(report["groups"]["tone"]) == {"NGANG", "SAC", "NANG", "HOI"}
    assert set(report["groups"]["status"]) == {"VALID"}
    assert report["groups"]["status"]["VALID"]["n_syllables"] == 5  # chỉ row có nhãn
    for metrics in report["groups"]["coda"].values():
        assert metrics["n_syllables"] == metrics["mel_z"]["n"]
        assert metrics["n_syllables"] > 0
        for name in ("mel_z", "mel_db", "dur_mae", "f0_cents"):
            stats = metrics[name]
            assert stats.get("n", 0) >= 0
            if stats.get("n"):
                assert math.isfinite(stats["mean"]) and math.isfinite(stats["median"])
        assert 0.0 <= metrics["vuv_acc"] <= 1.0
    json.dumps(report)  # JSON-serializable


def test_group_labels_ghep_coda_tone(setup):
    _config, dataset, _ = setup
    vocabs = ComponentVocabs.default()
    labels, skipped = ev._group_labels(ROW1, dataset[0]["features"], vocabs, ["coda:tone"])
    assert labels["coda:tone"][0] == "C|SAC"
    assert skipped == []
    # thiếu `status` ở ROW2 ⇒ nhóm ghép chứa nó cũng bị bỏ (không bịa nhãn); `status` cũng bị bỏ
    _, skipped_missing = ev._group_labels(ROW2, dataset[1]["features"], vocabs,
                                          ["coda:status"])
    assert skipped_missing == ["coda:status", "status"]
    assert ev.parse_group_by("coda:tone,tone") == ["coda:tone", "tone"]


def test_parse_group_by_cli():
    assert ev.parse_group_by(None) is None
    assert ev.parse_group_by("coda, tone") == ["coda", "tone"]
    with pytest.raises(SystemExit):
        ev.parse_group_by("khong-co-nhom-nay")


# ── việc 2 + 3: hàm thuần (nhịp nghỉ / ngữ điệu) ───────────────────────────────
def test_gap_frame_mask_from_khop_quy_uoc_train():
    mask = ev._gap_frame_mask_from([4, 4, 4], [3, 0, 5], lead=2, frames=20)
    assert mask[:2].all()                       # lead cũng là "khoảng nghỉ" (như `_gap_frame_mask`)
    assert mask[6:9].all() and not mask[2:6].any()   # gap 3 frame sau âm tiết 1 (2+4)
    assert mask[17:20].all() and not mask[9:17].any()  # gap 5 frame cuối, cắt theo `frames`
    assert mask.sum() == 2 + 3 + 3


def test_pause_counts_va_f1():
    pred = np.array([True, False, True, False])
    true = np.array([True, True, False, False])
    hits, false, missed = ev._pause_counts(pred, true)
    assert (hits, false, missed) == (1, 1, 1)
    assert ev._f1(hits, false, missed) == pytest.approx(0.5)
    assert ev._f1(0, 0, 0) is None  # không có dương tính nào ⇒ không có F1 (không bịa 0)


def test_f0_slope_cents_per_second():
    frame_rate = 24000 / 256
    frames = np.arange(100)
    # logF0 giảm 0,001/frame ⇒ −0,001 · 93,75 · 1731,2 ≈ −162 cent/giây
    values = 4.8 - 0.001 * frames
    slope = ev._log_f0_slope_cents_per_second(frames, values, frame_rate)
    assert slope == pytest.approx(-1200 / math.log(2) * 0.001 * frame_rate, rel=1e-6)
    assert ev._log_f0_slope_cents_per_second(frames[:5], values[:5], frame_rate) is None


def test_ngang_declination_va_slope_block():
    windows = [(0, 10), (10, 20), (20, 30), (30, 40)]
    positions = np.arange(40)
    series = (positions, 5.0 - 0.001 * positions)
    declination = ev._ngang_declination(series, windows, ["NGANG"] * 4)
    assert declination is not None and declination > 0  # đầu câu cao hơn cuối câu
    assert ev._ngang_declination((positions, 5.0 - 0.001 * positions), [],
                                 ["NGANG"] * 4) is None
    punct_codes = {"QUESTION": 3, "PERIOD": 2}
    block = ev._slope_block(series, series, windows, windows, ["NGANG"] * 4,
                            [0, 2, 0, 0], punct_codes, 24000 / 256)
    assert set(block) == {"slope_all", "slope_all_ngang", "slope_final_question",
                          "slope_final_period", "ngang_declination_cents"}
    assert block["slope_final_period"]["true"] is not None  # có `.` ở âm tiết 1
    assert block["slope_final_question"]["true"] is None    # không có `?` ⇒ bỏ qua, không bịa


def test_evaluate_free_v2_them_nhip_nghi_va_ngu_dieu(setup):
    config, dataset, checkpoint = setup
    model, _ = ev.load_eval_model(str(checkpoint), config, f0_stats=F0_STATS)
    report = ev.evaluate_free_v2(model, dataset, config, device="cpu", skip_cer=True,
                                 group_by=["coda", "tone"])
    assert report["cer"] is None and "no-cer" in report["cer_note"]
    # nhịp nghỉ (5 âm tiết/row ⇒ 4 vị trí có nhãn; bỏ vị trí gap cuối row)
    assert report["gap_sum_mae_frames"]["n"] == 2
    assert report["gap_sum_mae_frames"]["mean"] >= 0
    assert report["pause_f1"] is None or 0.0 <= report["pause_f1"] <= 1.0
    assert set(report["pause_counts"]) >= {"tp", "fp", "fn"}
    assert report["pause_note"]
    mel_gap = report["mel_gap"]
    assert mel_gap["n"] == 2 and math.isfinite(mel_gap["mean"])
    # ngữ điệu: đủ 5 metric × 2 timeline, có mẫu cho `slope_all`
    intonation = report["intonation"]
    assert set(intonation) == set(ev._SLOPE_METRICS)
    for metric in ev._SLOPE_METRICS:
        assert set(intonation[metric]) == {"true", "model"}
    assert intonation["slope_all"]["true"]["n"] == 2
    assert intonation["slope_all"]["true"]["mean"] < 0  # cache giả dốc xuống
    assert report["intonation_note"]
    # nhóm
    assert set(report["groups"]) == {"coda", "tone"}
    json.dumps(report)
