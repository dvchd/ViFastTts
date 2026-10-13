"""Test T4.1 — mel discriminator đa thang + trainer adversarial (D tách khỏi checkpoint)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from tests.test_train_v2 import CONFIG, F0_STATS, _row, _write_cache
from vifasttts.models.discriminators import (
    MultiScaleMelDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)


def test_multi_scale_mel_discriminator_shapes_va_grad():
    discriminator = MultiScaleMelDiscriminator(n_mels=100, scales=(1, 2, 4), channels=16, layers=2)
    real = torch.randn(1, 100, 100)
    fake = torch.randn(1, 100, 100, requires_grad=True)
    out_real = discriminator(real)
    out_fake = discriminator(fake)
    assert len(out_real) == 3
    assert all(len(features) == 2 and logits.dim() == 3 for logits, features in out_real)
    loss_d = discriminator_loss(out_real, [(logits.detach(), [f.detach() for f in features])
                                           for logits, features in out_fake], mode="lsgan")
    assert torch.isfinite(loss_d)
    adv = generator_adversarial_loss(out_fake, mode="lsgan")
    matched = feature_matching_loss(out_real, out_fake)
    (adv + matched).backward()
    assert fake.grad is not None and float(fake.grad.abs().sum()) > 0


def test_train_v2_adversarial_smoke(tmp_path):
    """3 bước có adversarial: chạy xong, checkpoint G **không** chứa D, có `d_last.pt`."""
    from vifasttts.train_v2 import train_slot_v2

    rows = [_row("clip__adv0", 0, [5, 6], [2], [[2, 0, 2, 1], [2, 0, 2, 2]]),
            _row("clip__adv1", 0, [4, 5], [3], [[1, 0, 2, 1], [2, 0, 1, 2]])]
    manifest = tmp_path / "adv.jsonl"
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
        "training": {**CONFIG["training"], "cache_dir": str(cache_dir), "max_steps": 3,
                     "phase_a_steps": 5, "log_every": 1, "validate_every": 2, "save_every": 1},
        "adversarial": {"enabled": True, "d_warmup_steps": 1, "channels": 8, "layers": 1,
                        "scales": [1, 2]},
    }
    run_dir = tmp_path / "run_adv"
    train_slot_v2(config, str(manifest), str(manifest), str(run_dir), max_steps=3, device="cpu")
    state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state["arch"] == "v2_slot5"
    assert not any("discriminator" in key for key in state["model"])  # D tách khỏi checkpoint
    assert (run_dir / "d_last.pt").exists()
    discriminant = torch.load(run_dir / "d_last.pt", map_location="cpu", weights_only=False)
    assert "discriminator" in discriminant and "d_optimizer" in discriminant
