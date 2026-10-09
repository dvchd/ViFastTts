from __future__ import annotations

import torch
from torch import nn
from vifasttts.models.components import ComponentEmbedding, ConvPredictor
from vifasttts.models.conformer import ConformerLite


class LengthRegulator(nn.Module):
    def forward(self, x: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        lengths = []
        for batch_x, batch_d in zip(x, durations):
            repeated = torch.repeat_interleave(batch_x, batch_d.clamp_min(0), dim=0)
            sequences.append(repeated)
            lengths.append(repeated.size(0))
        max_len = max(lengths) if lengths else 0
        output = x.new_zeros((len(sequences), max_len, x.size(-1)))
        for i, seq in enumerate(sequences):
            output[i, : seq.size(0)] = seq
        return output, torch.tensor(lengths, device=x.device)


class ViFastTtsModel(nn.Module):
    """Reference acoustic model.

    Input shape: [B, syllables, 8] integer structured features.
    Output: mel, durations, f0, vuv, energy and tone logits.
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
        self.decoder = ConformerLite(m["decoder_layers"], h, m["attention_heads"], m["ffn_size"], m["conv_kernel_size"], m["dropout"])
        self.mel = nn.Linear(h, config["mel"]["n_mels"])

    def forward(self, features: torch.Tensor, durations: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.encoder(self.embedding(features))
        log_d = self.duration(x).squeeze(-1)
        syllable_f0 = self.f0(x).squeeze(-1)
        syllable_vuv = self.vuv(x).squeeze(-1)
        syllable_energy = self.energy(x).squeeze(-1)
        if durations is None:
            durations = torch.round(torch.expm1(log_d).clamp_min(1)).long()
        expanded, lengths = self.length_regulator(x, durations)
        exp_f0, _ = self.length_regulator(syllable_f0.unsqueeze(-1), durations)
        exp_vuv, _ = self.length_regulator(syllable_vuv.unsqueeze(-1), durations)
        exp_energy, _ = self.length_regulator(syllable_energy.unsqueeze(-1), durations)
        variance = torch.cat([exp_f0, exp_vuv, exp_energy], dim=-1)
        decoded = self.decoder(expanded + self.variance_proj(variance))
        return {
            "mel": self.mel(decoded), "log_duration": log_d, "durations": durations,
            "f0": syllable_f0, "vuv_logits": syllable_vuv, "energy": syllable_energy,
            "tone_logits": self.tone_classifier(x), "lengths": lengths,
        }
