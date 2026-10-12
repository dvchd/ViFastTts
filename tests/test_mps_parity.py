"""MPS vs CPU: mel loss + MR-STFT loss (giá trị & gradient) — chênh tương đối ≤ ~5e-4.

Yêu cầu R7: MPS "chạy được" chưa chắc "đúng" (FFT từng có lỗi số học), nên khóa lại
bằng test trước khi bỏ vòng copy mel sang CPU. Số đo nền trên Mac mini M6 (12/10/2026):
mel rel ~1,8e-5; grad mel rel ~5,9e-5; MR-STFT rel ~4,6e-7; grad MR-STFT rel ~2,0e-5.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vifasttts.data.features import extract_mel_energy_batched
from vifasttts.models.discriminators import multi_resolution_stft_loss

MEL = {
    "n_fft": 1024, "win_length": 1024, "hop_length": 256, "n_mels": 100,
    "f_min": 0, "f_max": 12000, "sample_rate": 24000, "norm": "none",
    "log_magnitude": True, "log_floor": 1e-7, "log_floor_mode": "clamp",
}


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    """max|a-b| / max(|a|,|b|) — thang tương đối, an toàn khi giá trị lớn (đưa về CPU trước)."""
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    diff = (a - b).abs().max()
    scale = torch.maximum(a.abs().max(), b.abs().max()).clamp_min(1e-12)
    return float(diff / scale)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS không khả dụng")
def test_mps_matches_cpu_for_mel_and_stft_losses():
    torch.manual_seed(0)
    wav = torch.randn(24000) * 0.1

    def compute(device: str):
        waveform = wav.clone().to(device).requires_grad_(True)
        mel, _ = extract_mel_energy_batched(waveform.unsqueeze(0), MEL)
        mel_loss = (mel - torch.zeros_like(mel)).abs().mean()
        mel_grad = torch.autograd.grad(mel_loss, waveform, retain_graph=True)[0]
        pred = torch.tanh(waveform)
        target = torch.tanh(waveform.detach() * 0.9)
        stft_loss = multi_resolution_stft_loss(pred.unsqueeze(0), target.unsqueeze(0))
        stft_grad = torch.autograd.grad(stft_loss, waveform)[0]
        return mel.detach(), mel_loss.detach(), mel_grad, stft_loss.detach(), stft_grad

    mel_cpu, loss_cpu, grad_cpu, stft_cpu, stft_grad_cpu = compute("cpu")
    mel_mps, loss_mps, grad_mps, stft_mps, stft_grad_mps = compute("mps")

    assert _relative(mel_cpu, mel_mps) < 5e-4
    assert _relative(loss_cpu, loss_mps) < 1e-4
    assert _relative(grad_cpu, grad_mps) < 5e-4
    assert _relative(stft_cpu, stft_mps) < 1e-4
    assert _relative(stft_grad_cpu, stft_grad_mps) < 5e-4
    assert torch.isfinite(mel_mps).all() and torch.isfinite(grad_mps).all()
