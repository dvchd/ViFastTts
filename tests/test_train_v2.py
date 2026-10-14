"""Test `train_v2.train_slot_v2` — smoke 3 bước (cache giả, CPU) + metadata + freeze giai đoạn A."""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from vifasttts.data.cache import feature_cache_key
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.vifasttts import ViFastTtsModel
from vifasttts.train_v2 import train_slot_v2

MODEL = {
    "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
    "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
    "duration_layers": 1, "variance_layers": 1, "use_phonation": True, "use_postnet": True,
    "role_dim": 4, "segment_mlp_layers": 2,
}
CONFIG = {
    "seed": 0,
    "sample_rate": 24000,
    "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
    "f0": {"method": "pyworld"},
    "phonation": {"enabled": True, "channels": ["hnr", "glottalization", "jitter"]},
    "model": MODEL,
    "loss_weights": {"mel": 1.0, "mel_pre": 0.2, "duration": 1.0, "ratio": 1.0, "f0": 1.0,
                     "vuv": 0.5, "energy": 0.5, "phonation": 0.5, "tone": 0.1,
                     "pause_gate": 0.5, "pause_gap": 0.5},
    "training": {
        "batch_size": 1, "learning_rate": 2e-3, "weight_decay": 0.0, "max_steps": 3,
        "phase_a_steps": 5, "phase_a_lr_scale": 0.25, "phase_b_warmup_steps": 1,
        "grad_clip": 1.0, "log_every": 1, "validate_every": 2, "validate_batches": 1,
        "save_every": 1, "num_workers": 0, "device": "cpu",
    },
}
F0_STATS = {
    "f0_log_mean": 4.8, "f0_log_std": 0.2,
    "shape_mean": [0.0, 0.0, 0.0, 0.0], "shape_std": [1.0, 1.0, 1.0, 1.0],
    "energy_mean": 0.0, "energy_std": 1.0,
}

SYLLABLES = [
    {"raw": "các", "normalized": "các",
     "orthography": {"onset": "c", "medial": None, "nucleus": "a", "coda": "c", "tone_location": "a"},
     "abstract": {"onset": "K", "medial": "NONE", "nucleus": "A", "coda": "C", "tone": "SAC"},
     "boundaries": {"word_end": True, "phrase_end": False, "sentence_end": False},
     "status": "VALID", "confidence": 1.0, "punctuation": None},
    {"raw": "ma", "normalized": "ma",
     "orthography": {"onset": "m", "medial": None, "nucleus": "a", "coda": None, "tone_location": "a"},
     "abstract": {"onset": "M", "medial": "NONE", "nucleus": "A", "coda": "N", "tone": "NGANG"},
     "boundaries": {"word_end": True, "phrase_end": False, "sentence_end": True},
     "status": "VALID", "confidence": 1.0, "punctuation": None},
]


def _row(row_id: str, start: int, durations: list[int], gaps: list[int],
         seg: list[list[int]]) -> dict:
    return {
        "id": row_id, "audio_path": f"/khong/ton/tai_{row_id}.wav",
        "frame_range": [start, start + sum(durations) + sum(gaps)],
        "durations": durations, "gaps": gaps, "seg_durations": seg, "lead_frames": 0,
        "seg_method": "hybrid", "seg_method_version": "t2.1-hybrid-v1", "seg_weights_hash": "abc",
        "syllables": SYLLABLES,
    }


def _write_cache(cache_dir, row: dict, config: dict) -> None:
    start = row["frame_range"][0]
    frames = row["frame_range"][1] - start
    length = start + frames
    torch.manual_seed(0)
    torch.save({
        "mel": torch.randn(length, 20),
        "energy": torch.full((length,), 2.5),
        "f0": torch.full((length,), 150.0),
        "vuv": torch.ones(length),
        "hnr": torch.zeros(length),
        "glottalization": torch.full((length,), 0.1),
        "jitter": torch.zeros(length),
    }, cache_dir / f"{feature_cache_key(row, config)}.pt")


def test_train_v2_smoke_va_metadata(tmp_path):
    rows = [_row("clip__s00", 5, [5, 6], [2], [[2, 0, 2, 1], [2, 0, 2, 2]]),
            _row("clip__s01", 0, [4, 5], [3], [[1, 0, 2, 1], [2, 0, 1, 2]])]
    manifest = tmp_path / "train_slot4.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                        encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for row in rows:
        _write_cache(cache_dir, row, CONFIG)
    stats_path = tmp_path / "f0_stats_v2.json"
    stats_path.write_text(json.dumps(F0_STATS), encoding="utf-8")

    config = {**CONFIG, "f0_stats_path": str(stats_path),
              "training": {**CONFIG["training"], "cache_dir": str(cache_dir)}}

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    torch.manual_seed(1)
    model_v1 = ViFastTtsModel({**config, "model": {**MODEL, "use_phonation": False}}, sizes)
    warm_path = tmp_path / "b0.pt"
    torch.save({"model": model_v1.state_dict(), "arch": "v1_syl"}, str(warm_path))

    run_dir = tmp_path / "run"
    train_slot_v2(config, str(manifest), str(manifest), str(run_dir),
                  warm_start=str(warm_path), max_steps=3, device="cpu")

    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state["arch"] == "v2_slot5"
    assert state["seg_method"] == "hybrid" and state["seg_method_version"] == "t2.1-hybrid-v1"
    assert state["seg_weights_hash"] == "abc"
    assert state["f0_stats_hash"] and state["f0_stats"] == F0_STATS
    assert state["warm_start_from"]  # hash B0
    assert "optimizer" in state and int(state["step"]) == 3
    assert (run_dir / "best.pt").exists() and (run_dir / "history.csv").exists()
    # Giai đoạn A (phase_a_steps=5 > 3 bước): embedding phải giữ nguyên từ warm-start
    reference = model_v1.state_dict()
    assert torch.equal(state["model"]["embedding.proj.weight"], reference["embedding.proj.weight"])
    # module mới của v2 có trọng số khác 0 (đã khởi tạo, không copy từ v1)
    assert float(state["model"]["role_emb.weight"].abs().sum()) > 0


def test_train_v2_early_stop_patience(tmp_path):
    """Với LR = 0 (model không đổi) và patience = 1 → val không cải thiện lần 2 → dừng sớm."""
    rows = [_row("clip__es0", 0, [5, 6], [2], [[2, 0, 2, 1], [2, 0, 2, 2]])]
    manifest = tmp_path / "es.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                        encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for row in rows:
        _write_cache(cache_dir, row, CONFIG)
    stats_path = tmp_path / "f0_stats.json"
    stats_path.write_text(json.dumps(F0_STATS), encoding="utf-8")
    config = {
        **CONFIG,
        "f0_stats_path": str(stats_path),
        "training": {**CONFIG["training"], "cache_dir": str(cache_dir), "learning_rate": 0.0,
                     "phase_a_steps": 0, "phase_b_warmup_steps": 1, "log_every": 100,
                     "validate_every": 1, "validate_batches": 1, "save_every": 100,
                     "early_stop_patience": 1},
    }
    run_dir = tmp_path / "run_es"
    train_slot_v2(config, str(manifest), str(manifest), str(run_dir), max_steps=20, device="cpu")
    early = json.loads((run_dir / "early_stop.json").read_text(encoding="utf-8"))
    assert early["step"] < 20 and early["patience"] == 1  # dừng TRƯỚC max_steps


def test_train_v2_ghi_danh_tinh_tap_val(tmp_path):
    """§40: checkpoint + early_stop.json phải ghi `val_manifest_hash` và `validate_batches`."""
    rows = [_row("clip__vm0", 0, [5, 6], [2], [[2, 0, 2, 1], [2, 0, 2, 2]])]
    manifest = tmp_path / "vm.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for row in rows:
        _write_cache(cache_dir, row, CONFIG)
    stats_path = tmp_path / "f0_stats.json"
    stats_path.write_text(json.dumps(F0_STATS), encoding="utf-8")
    config = {
        **CONFIG,
        "f0_stats_path": str(stats_path),
        "training": {**CONFIG["training"], "cache_dir": str(cache_dir), "learning_rate": 0.0,
                     "phase_a_steps": 0, "phase_b_warmup_steps": 1, "log_every": 100,
                     "validate_every": 1, "validate_batches": 1, "save_every": 100,
                     "early_stop_patience": 1},
    }
    run_dir = tmp_path / "run_vm"
    train_slot_v2(config, str(manifest), str(manifest), str(run_dir), max_steps=5, device="cpu")
    import hashlib
    expected = hashlib.sha1(manifest.read_bytes()).hexdigest()
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state["val_manifest_hash"] == expected
    assert state["validate_batches"] == 1
    early = json.loads((run_dir / "early_stop.json").read_text(encoding="utf-8"))
    assert early["val_manifest_hash"] == expected and early["validate_batches"] == 1


def _run_tiny(tmp_path, name, val_manifest=None, steps=3, resume=None, reset=False):
    """Train/resume 3 bước với manifest nhỏ; trả (run_dir, manifest_path)."""
    rows = [_row(f"clip__{name}0", 0, [5, 6], [2], [[2, 0, 2, 1], [2, 0, 2, 2]])]
    manifest = tmp_path / f"{name}.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    for row in rows:
        _write_cache(cache_dir, row, CONFIG)
    stats_path = tmp_path / "f0_stats.json"
    stats_path.write_text(json.dumps(F0_STATS), encoding="utf-8")
    config = {
        **CONFIG,
        "f0_stats_path": str(stats_path),
        "training": {**CONFIG["training"], "cache_dir": str(cache_dir), "learning_rate": 0.0,
                     "phase_a_steps": 0, "phase_b_warmup_steps": 1, "log_every": 100,
                     "validate_every": 1, "validate_batches": 1, "save_every": 100,
                     "early_stop_patience": 0},
    }
    run_dir = tmp_path / name
    train_slot_v2(config, str(manifest), str(val_manifest or manifest), str(run_dir),
                  max_steps=steps, resume=resume, reset_val_state=reset, device="cpu")
    return run_dir, manifest


def test_x3_chan_khi_val_hash_khac_va_khong_co_co(tmp_path):
    """X3 (0.1): resume bằng tập val KHÁC mà không có cờ ⇒ lỗi, không ghi gì."""
    run_dir, manifest = _run_tiny(tmp_path, "x3a", steps=3)
    # khác hash nhưng nội dung hàng y hệt: thêm 1 dòng trắng (dataset bỏ qua dòng trắng)
    other = tmp_path / "val_khac.jsonl"
    other.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tập val KHÁC"):
        _run_tiny(tmp_path, "x3a", val_manifest=other, steps=5, resume=run_dir / "last.pt")


def test_x3_reset_val_state_va_ghi_chu_history(tmp_path):
    """X3 (0.1): có cờ ⇒ chạy tiếp, best_val = inf, history có dòng resume_note."""
    run_dir, manifest = _run_tiny(tmp_path, "x3b", steps=3)
    other = tmp_path / "val_khac.jsonl"
    other.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    new_dir, _ = _run_tiny(tmp_path, "x3b_new", val_manifest=other, steps=5,
                           resume=run_dir / "last.pt", reset=True)
    state = torch.load(new_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 5
    first_line = (new_dir / "history.csv").read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("# X3 resume: resumed_from=")
    assert "val_hash_old=" in first_line and "val_hash_new=" in first_line


def test_x3_reset_val_state_cung_thu_muc_nguon_thi_loi(tmp_path):
    """X3 (0.1): có cờ mà thư mục run trùng thư mục nguồn ⇒ lỗi."""
    run_dir, _manifest = _run_tiny(tmp_path, "x3c", steps=3)
    with pytest.raises(ValueError, match="trùng thư mục nguồn"):
        _run_tiny(tmp_path, "x3c", steps=5, resume=run_dir / "last.pt", reset=True)


def test_x3_loi_resume_khong_de_lai_dau_vet(tmp_path):
    """A1 (Lượt 1b): đường lỗi (hash val khác, không cờ) KHÔNG được tạo/ghi tệp trong thư mục đích."""
    run_dir, manifest = _run_tiny(tmp_path, "x3d", steps=3)
    src_cfg = run_dir / "resolved_config.json"
    sha_before = hashlib.sha256(src_cfg.read_bytes()).hexdigest()
    other = tmp_path / "val_khac.jsonl"
    other.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    target = tmp_path / "x3d_target"
    with pytest.raises(ValueError, match="tập val KHÁC"):
        _run_tiny(tmp_path, "x3d_target", val_manifest=other, steps=5, resume=run_dir / "last.pt")
    assert not target.exists() or not any(target.iterdir()), "đường lỗi đã để lại dấu vết trên đĩa"
    assert hashlib.sha256(src_cfg.read_bytes()).hexdigest() == sha_before


def test_stop_at_step_dung_dung_buoc_va_khong_cham_lr(tmp_path):
    """C1: --stop-at-step dừng ở bước TUYỆT ĐỐI; LR từng bước khớp lr_at với max_steps=60000."""
    import math

    run_dir, manifest = _run_tiny(tmp_path, "sas", steps=3)
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    state["step"] = 40000
    torch.save(state, run_dir / "last.pt")
    config = {
        **CONFIG,
        "f0_stats_path": str(tmp_path / "f0_stats.json"),
        "training": {**CONFIG["training"], "cache_dir": str(tmp_path / "cache"),
                     "learning_rate": 2e-4, "phase_a_steps": 0, "phase_b_warmup_steps": 1,
                     "log_every": 100, "validate_every": 1, "validate_batches": 1,
                     "save_every": 100, "early_stop_patience": 0},
    }
    out = tmp_path / "sas_stop"
    train_slot_v2(config, str(manifest), str(manifest), str(out), max_steps=60000,
                  resume=str(run_dir / "last.pt"), stop_at_step=40003, device="cpu")
    saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == 40003  # dừng đúng bước tuyệt đối
    all_lines = (out / "history.csv").read_text(encoding="utf-8").splitlines()
    header = next(line for line in all_lines if line.startswith("step"))
    lr_idx = header.split(",").index("lr")  # V10.A8: cột thành phần thêm SAU cột lr
    rows = [line.split(",") for line in all_lines if line and not line.startswith("step")]
    lr_col = [float(row[lr_idx]) for row in rows]
    assert len(lr_col) == 3
    for step, lr in zip((40001, 40002, 40003), lr_col):
        progress = min(1.0, (step - 0 - 1) / (60000 - 0 - 1))
        expected = 2e-4 * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        assert abs(lr - expected) / expected < 1e-3  # LR không bị --stop-at-step chạm vào


def test_validate_at_stop_report_only(tmp_path):
    """B6: --validate-at-stop ghi 1 dòng history nhưng KHÔNG chạm best_val/no_improve."""
    run_dir, manifest = _run_tiny(tmp_path, "vas", steps=3)
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    state["step"] = 40000
    state["best_val"] = 0.1234
    state["no_improve"] = 7
    torch.save(state, run_dir / "last.pt")
    config = {**CONFIG, "f0_stats_path": str(tmp_path / "f0_stats.json"),
              "training": {**CONFIG["training"], "cache_dir": str(tmp_path / "cache"),
                           "learning_rate": 2e-4, "phase_a_steps": 0, "phase_b_warmup_steps": 1,
                           "log_every": 100, "validate_every": 1000, "validate_batches": 1,
                           "save_every": 1000, "early_stop_patience": 1}}
    out = tmp_path / "vas_stop"
    train_slot_v2(config, str(manifest), str(manifest), str(out), max_steps=60000,
                  resume=str(run_dir / "last.pt"), stop_at_step=40003, validate_at_stop=True,
                  device="cpu")
    saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == 40003
    assert saved["best_val"] == 0.1234 and saved["no_improve"] == 7  # không bị chạm
    rows = [l for l in (out / "history.csv").read_text(encoding="utf-8").splitlines() if l and not l.startswith("step")]
    assert len(rows) == 1  # đúng 1 dòng report-only


def test_loss_components_khop_v2_losses():
    """V10.A8: danh sách thành phần ghi history == mọi thành phần của v2_losses."""
    from vifasttts.models.v2_losses import DEFAULT_WEIGHTS
    from vifasttts.train_v2 import LOSS_COMPONENTS

    assert set(LOSS_COMPONENTS) == set(DEFAULT_WEIGHTS)


def test_history_header_du_cot_thanh_phan():
    """V10.A8: 10 cột cũ giữ nguyên thứ tự; cột thành phần + tỉ phần pause thêm vào cuối."""
    from vifasttts.train_v2 import LOSS_COMPONENTS, history_header

    header = history_header()
    assert header[:10] == ["step", "phase", "train_total", "train_mel", "mel_gap", "mel_voiced",
                           "val_total", "val_mel_voiced", "val_rough", "lr"]
    assert "train_pause_frac" in header and "val_pause_frac" in header
    for key in LOSS_COMPONENTS:
        assert f"train_{key}" in header and f"val_{key}" in header
    assert len(header) == 10 + 2 + 2 * len(LOSS_COMPONENTS)


def test_pause_fraction_dung_trong_so():
    """V10.A8: tỉ phần pause = (0,5·gate + 0,5·gap)/tổng theo trọng số mặc định; tổng = 0 ⇒ nan."""
    import math

    from vifasttts.train_v2 import _pause_fraction

    frac = _pause_fraction({"pause_gate": 0.4, "pause_gap": 0.2}, 1.0, None)
    assert abs(frac - (0.5 * 0.4 + 0.5 * 0.2)) < 1e-9
    assert math.isnan(_pause_fraction({"pause_gate": 0.1}, 0.0, None))
