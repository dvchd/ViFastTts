from __future__ import annotations

import torch
from torch import nn

from vifasttts.models.components import ComponentEmbedding, ConvPredictor
from vifasttts.models.conformer import ConformerLite


class LengthRegulator(nn.Module):
    """Mở rộng chuỗi cấp âm tiết thành cấp frame, không dùng vòng lặp Python.

    Vector hoá bằng phép so sánh cumulative-sum để có thể export ONNX với
    đầu ra dài động theo `durations` (đã kiểm chứng với onnxruntime).
    """

    def forward(self, x: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = durations.sum(dim=1)
        max_frames = lengths.max()
        # 0-dim tensor làm biên arange: giữ độ dài động khi export ONNX.
        steps = torch.arange(max_frames, device=x.device)  # type: ignore[call-overload]
        cumulative = durations.cumsum(dim=1)
        index = (cumulative.unsqueeze(2) <= steps.view(1, 1, -1)).sum(dim=1)
        index = index.clamp(max=x.size(1) - 1)
        expanded = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        mask = steps.view(1, -1) < lengths.unsqueeze(1)
        return expanded * mask.unsqueeze(-1), lengths, mask


class ViFastTtsModel(nn.Module):
    """Reference acoustic model.

    Input shape: [B, syllables, 7] integer structured features
    (onset, medial, nucleus, coda, tone, boundary, punctuation).
    Output: mel, durations, f0, vuv, energy and tone logits.

    `input_lengths` dùng cho training có padding; khi export ONNX chạy
    một câu hoàn chỉnh thì không truyền (mask = None).
    """

    def __init__(self, config: dict, vocab_sizes: list[int]) -> None:
        super().__init__()
        m = config["model"]
        h = m["hidden_size"]
        self.embedding = ComponentEmbedding(vocab_sizes, m["component_dim"], h)
        self.encoder = ConformerLite(m["encoder_layers"], h, m["attention_heads"], m["ffn_size"], m["conv_kernel_size"], m["dropout"])
        self.duration = ConvPredictor(h, 1, m["duration_layers"], m["dropout"])
        self.f0 = ConvPredictor(h, 1, m["variance_layers"], m["dropout"])
        self.vuv = ConvPredictor(h, 1, m["variance_layers"], m["dropout"])
        self.energy = ConvPredictor(h, 1, m["variance_layers"], m["dropout"])
        self.tone_classifier = nn.Linear(h, 6)
        self.length_regulator = LengthRegulator()
        self.variance_proj = nn.Linear(3, h)
        self.position_proj = nn.Linear(1, h)
        self.decoder = ConformerLite(m["decoder_layers"], h, m["attention_heads"], m["ffn_size"], m["conv_kernel_size"], m["dropout"])
        self.mel = nn.Linear(h, config["mel"]["n_mels"])

    @staticmethod
    def _frame_positions(durations: torch.Tensor, max_frames: int) -> torch.Tensor:
        """Vị trí tương đối 0..1 của mỗi frame trong âm tiết chứa nó."""
        frames = torch.arange(max_frames, device=durations.device)  # type: ignore[call-overload]
        cumulative = durations.cumsum(dim=1)
        index = (cumulative.unsqueeze(2) <= frames.view(1, 1, -1)).sum(dim=1)
        index = index.clamp(max=durations.size(1) - 1)
        starts = cumulative - durations
        start = torch.gather(starts, 1, index)
        span = torch.gather(durations, 1, index).clamp_min(1)
        return (frames.view(1, -1) - start).clamp_min(0).float() / span.float()

    def forward(
        self,
        features: torch.Tensor,
        durations: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        padding_mask = None
        if input_lengths is not None:
            steps = torch.arange(features.size(1), device=features.device)
            padding_mask = steps.view(1, -1) < input_lengths.unsqueeze(1)
        x = self.encoder(self.embedding(features), padding_mask)
        log_d = self.duration(x).squeeze(-1)
        syllable_f0 = self.f0(x).squeeze(-1)
        syllable_vuv = self.vuv(x).squeeze(-1)
        syllable_energy = self.energy(x).squeeze(-1)
        if durations is None:
            # exp - 1 tương đương expm1 nhưng export ONNX được (aten::expm1 không có op tương ứng).
            durations = torch.round((torch.exp(log_d) - 1).clamp_min(1)).long()
        expanded, lengths, frame_mask = self.length_regulator(x, durations)
        exp_f0, _, _ = self.length_regulator(syllable_f0.unsqueeze(-1), durations)
        exp_vuv, _, _ = self.length_regulator(syllable_vuv.unsqueeze(-1), durations)
        exp_energy, _, _ = self.length_regulator(syllable_energy.unsqueeze(-1), durations)
        variance = torch.cat([exp_f0, exp_vuv, exp_energy], dim=-1)
        positions = self._frame_positions(durations, expanded.size(1))
        decoded = self.decoder(
            expanded + self.variance_proj(variance) + self.position_proj(positions.unsqueeze(-1)),
            frame_mask,
        )
        return {
            "mel": self.mel(decoded), "log_duration": log_d, "durations": durations,
            "f0": syllable_f0, "vuv_logits": syllable_vuv, "energy": syllable_energy,
            "tone_logits": self.tone_classifier(x), "lengths": lengths,
        }
