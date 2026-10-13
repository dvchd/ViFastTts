"""T1.1 — Hợp đồng mel với checkpoint Vocos thật (`charactr/vocos-mel-24khz`).

Mục tiêu (Plan §3, Addendum 1 A5): mel **thô trước normalize** của pipeline phải
khớp feature extractor của checkpoint — sai norm/floor/log sẽ lộ ra ở đây (R7-1).
Không phụ thuộc torchaudio: tham chiếu dùng chính
`feature_extractor.mel_spec.mel_scale.fb` [513, 100] +
`feature_extractor.mel_spec.spectrogram.window` [1024] trong checkpoint.

Ngưỡng đo trên M6 (3 clip `eval_clips_v1`, 13/10/2026):
- bank vs fb:  max|Δ| ≤ 1e-5  (đo 6,6e-6)
- mel:         max|Δ| < 1e-3  — lỗi Slaney/floor/log sẽ ≫ 1e-2; dư lượng fp32 đo được
               ≤ 1,2e-4, và nhiễu fp32 của chính phép so (ref fp32 vs fp64) ~2e-4 nên
               ngưỡng "pass < 1e-4" là bất khả thi trên fp32.
               mean|Δ| < 1e-5 (đo 1,3–1,6e-6); mean|Δ| theo bin < 2e-5 (đo ≤ 5,5e-6).
- round-trip (mel z-score → Vocos → mel thô): melL1 < 0,25 (đo 0,145–0,154),
  len_ratio ∈ [0,99; 1,01] (đo 0,9992–0,9994).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin"
CONFIG_PATH = ROOT / "configs/small_codelung.yaml"
MANIFEST = ROOT / "data/codelung/eval_clips_v1.jsonl"
N_CLIPS = 3

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="chưa có checkpoint Vocos (tools/fetch_vocos_pretrained.py)"
)


def _config() -> dict:
    """Config acoustic hiện hành (mel + vocoder_training)."""
    import yaml

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _raw_mel_config() -> dict:
    """Mục mel đã canonical, bỏ `normalize_path` — so ở không gian log-magnitude thô."""
    from vifasttts.data.features import canonical_mel_config

    raw = canonical_mel_config(_config())
    return {key: value for key, value in raw.items() if key != "normalize_path"}


def _clips() -> list[dict]:
    """Vài dòng đầu eval_clips_v1 có file audio tồn tại (skip nếu thiếu dữ liệu)."""
    if not MANIFEST.exists():
        pytest.skip("chưa có eval_clips_v1.jsonl")
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = [row for row in rows[:N_CLIPS] if (ROOT / row["audio_path"]).exists()]
    if not picked:
        pytest.skip("không tìm thấy file audio của eval_clips_v1")
    return picked


def _checkpoint_tensors():
    """(filterbank [bins, mels], window [n_fft]) từ checkpoint pretrained."""
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    return (
        state["feature_extractor.mel_spec.mel_scale.fb"],
        state["feature_extractor.mel_spec.spectrogram.window"],
    )


def _reference_mel(wav, fb, window):
    """Mel log-magnitude theo đúng fb/window của checkpoint: log(clamp(mag_mel, 1e-7))."""
    spec = torch.stft(wav, n_fft=1024, hop_length=256, win_length=1024,
                      window=window, return_complex=True, center=True)
    return torch.log((fb.T @ spec.abs()).T.clamp_min(1e-7))


def test_filterbank_khop_checkpoint():
    """Bộ lọc mel của pipeline khớp fb trong checkpoint (gốc rễ của hợp đồng)."""
    from vifasttts.data.features import mel_filterbank

    fb, _ = _checkpoint_tensors()
    bank = mel_filterbank(24000, 1024, 100, 0, 12000, norm="none")
    assert bank.shape == fb.T.shape
    assert float((bank - fb.T).abs().max()) <= 1e-5


def test_mel_tho_khop_feature_extractor():
    """Mel thô của pipeline khớp feature extractor checkpoint trên clip thật."""
    from vifasttts.data.features import extract_mel_energy, load_wav

    fb, window = _checkpoint_tensors()
    raw = _raw_mel_config()
    for row in _clips():
        wav = load_wav(str(ROOT / row["audio_path"]), 24000)
        ours, _ = extract_mel_energy(wav, raw)
        reference = _reference_mel(wav, fb, window)
        frames = min(ours.size(0), reference.size(0))
        diff = (ours[:frames] - reference[:frames]).abs()
        assert float(diff.max()) < 1e-3, (
            f"{row['id']}: max|Δ| = {float(diff.max()):.2e} — nghi sai norm/floor/log"
        )
        assert float(diff.mean()) < 1e-5, f"{row['id']}: mean|Δ| = {float(diff.mean()):.2e}"
        assert float(diff.mean(dim=0).max()) < 2e-5, (
            f"{row['id']}: lệch hệ thống theo bin (mean|Δ| bin lớn nhất = "
            f"{float(diff.mean(dim=0).max()):.2e})"
        )


def test_round_trip_vocos():
    """mel (z-score) → Vocos zero-shot → mel thô: self-consistency + độ dài."""
    from vifasttts.data.features import extract_mel_energy, load_wav
    from vifasttts.models.istftnet import build_generator

    config = _config()
    raw = _raw_mel_config()
    stats = json.loads((ROOT / config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"]).clamp_min(1e-4)
    generator = build_generator(config["vocoder_training"], config["mel"])
    generator.eval()
    for row in _clips():
        wav = load_wav(str(ROOT / row["audio_path"]), 24000)
        mel, _ = extract_mel_energy(wav, raw)
        with torch.no_grad():
            synth = generator(((mel - mean) / std).unsqueeze(0))[0]
        ratio = synth.numel() / max(1, mel.size(0) * 256)
        assert 0.99 <= ratio <= 1.01, f"{row['id']}: len_ratio = {ratio:.4f}"
        synth_mel, _ = extract_mel_energy(synth, raw)
        frames = min(synth_mel.size(0), mel.size(0))
        l1 = float((synth_mel[:frames] - mel[:frames]).abs().mean())
        assert l1 < 0.25, f"{row['id']}: round-trip melL1 = {l1:.4f}"
