"""Acoustic v2 — 5 slot `[O, M, N, C, G]` + `G_lead` (docs/ACOUSTIC_V2.md §1–§6, §10–§11).

Khác v1 (`vifasttts.py`): chuỗi hidden cấp **segment** (5 slot/âm tiết) thay vì cấp âm tiết;
`ratio` + `pause` head; hệ số hình dạng F0 `s_k` theo segment; `speed` nằm **trong graph**
(float32 `[1]`); thêm output `slot_durations` + `pause_gate`.

Bất biến:
- `G_lead + Σ(slot) == T` cho mọi row (segment: `G_lead = 0`);
- slot inactive duration = 0 (LengthRegulator tự bỏ qua — F2);
- chia slot bằng `slot_ops.distribute_slots` (B2) khi không có `seg_durations`;
- vị trí + variance (f0/energy/phonation) của slot G = 0; `vuv = 0`.

Quy ước tham số (khớp dataset v2):
- `durations` [B,N] = `d_syl` **hữu thanh** cấp âm tiết (khác v1 truyền spans);
- `gaps` [B,N] = gap **sau** mỗi âm tiết, phần tử cuối = 0 cho row segment;
- `lead` [B] = frame lặng đầu clip (train segment: 0).
"""

from __future__ import annotations

import torch
from torch import nn

from vifasttts.models.components import ComponentEmbedding, ConvPredictor
from vifasttts.models.conformer import ConformerLite
from vifasttts.models.slot_ops import build_f0_contour, distribute_slots, slot_positions
from vifasttts.models.vifasttts import LengthRegulator

ARCH_V2 = "v2_slot5"
ROLE_DIM = 16
SEGMENT_OUT = 9  # 4 hệ số hình dạng + vuv logit + energy + 3 phonation
NUCLEUS = 2  # chỉ số role N trong [O, M, N, C]


class SegmentMLP(nn.Module):
    """MLP trên `concat(h_syl, role_emb, emb thành phần)` cho từng role (§1) — không Conv trên 5N."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 2, dropout: float = 0.1) -> None:
        """`layers` lớp ẩn (mỗi lớp Linear + SiLU + Dropout), rồi Linear ra `out_dim`."""
        super().__init__()
        blocks: list[nn.Module] = []
        current = in_dim
        for _ in range(max(1, layers)):
            blocks += [nn.Linear(current, hidden), nn.SiLU(), nn.Dropout(dropout)]
            current = hidden
        blocks.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [.., in_dim] → [.., out_dim]."""
        output: torch.Tensor = self.net(x)
        return output


class ViFastTtsModelV2(nn.Module):
    """Reference acoustic v2 (`arch="v2_slot5"`). Xem docstring module cho quy ước tham số."""

    def __init__(self, config: dict, vocab_sizes: list[int]) -> None:
        """`vocab_sizes`: 7 vocab size theo thứ tự ID (xem `ComponentVocabs`)."""
        super().__init__()
        m = config["model"]
        h = int(m["hidden_size"])
        self.arch = ARCH_V2
        self.hidden_size = h
        self.role_dim = int(m.get("role_dim", ROLE_DIM))
        self.mlp_layers = int(m.get("segment_mlp_layers", 2))
        self.use_phonation = bool(m.get("use_phonation", False))

        # ── tái dùng kết cấu v1 để warm-start 1-1 (§11) ─────────────────────────
        self.embedding = ComponentEmbedding(vocab_sizes, m["component_dim"], h)
        self.encoder = ConformerLite(m["encoder_layers"], h, m["attention_heads"], m["ffn_size"],
                                     m["conv_kernel_size"], m["dropout"])
        self.duration = ConvPredictor(h, 1, m["duration_layers"], m["dropout"])
        self.decoder = ConformerLite(m["decoder_layers"], h, m["attention_heads"], m["ffn_size"],
                                     m["conv_kernel_size"], m["dropout"])
        self.mel = nn.Linear(h, config["mel"]["n_mels"])
        self.tone_classifier = nn.Linear(h, 6)
        self.use_postnet = bool(m.get("use_postnet", False))
        if self.use_postnet:
            mel_dim = int(config["mel"]["n_mels"])
            self.postnet = nn.Sequential(
                nn.Conv1d(mel_dim, 256, 5, padding=2), nn.LeakyReLU(0.1),
                nn.Conv1d(256, 256, 5, padding=2), nn.LeakyReLU(0.1),
                nn.Conv1d(256, 256, 5, padding=2), nn.LeakyReLU(0.1),
                nn.Conv1d(256, 256, 5, padding=2), nn.LeakyReLU(0.1),
                nn.Conv1d(256, mel_dim, 5, padding=2),
            )

        # ── module mới của v2 ───────────────────────────────────────────────────
        component_dim = int(m["component_dim"])
        self.role_emb = nn.Embedding(5, self.role_dim)  # O, M, N, C, G
        segment_in = h + self.role_dim + component_dim
        self.slot_expander = SegmentMLP(segment_in, h, h, self.mlp_layers, m["dropout"])
        self.ratio_mlp = SegmentMLP(segment_in, h, 1, self.mlp_layers, m["dropout"])
        self.variance_mlp = SegmentMLP(segment_in, h, SEGMENT_OUT, self.mlp_layers, m["dropout"])
        self.pause = ConvPredictor(h, 2, m["variance_layers"], m["dropout"])  # gate logit + log_gap
        self.silence_emb = nn.Parameter(torch.zeros(h))
        self.g_token = nn.Linear(2 * component_dim, h)  # boundary_emb + punct_emb → vector G
        self.variance_proj = nn.Linear(3 + (3 if self.use_phonation else 0), h)
        self.position_proj = nn.Linear(1, h)
        self.length_regulator = LengthRegulator()
        self.max_positions = int(m.get("max_positions", 4096))

        # hằng số chuẩn hoá (nạp từ `f0_stats_v2` ở trainer; mặc định trung tính cho test)
        self.register_buffer("shape_mean", torch.zeros(4))
        self.register_buffer("shape_std", torch.ones(4))
        self.register_buffer("f0_log_mean", torch.zeros(()))
        self.register_buffer("f0_log_std", torch.ones(()))
        self.register_buffer("energy_mean", torch.zeros(()))
        self.register_buffer("energy_std", torch.ones(()))

    def load_f0_stats(self, stats: dict) -> None:
        """Nạp hằng số từ `f0_stats_v2.json` (§6: hai pass trên s_k, chuẩn hoá theo nucleus)."""
        with torch.no_grad():
            self.shape_mean.copy_(torch.tensor(stats["shape_mean"], dtype=torch.float32))
            self.shape_std.copy_(torch.tensor(stats["shape_std"], dtype=torch.float32).clamp_min(1e-6))
            self.f0_log_mean.copy_(torch.tensor(float(stats["f0_log_mean"]), dtype=torch.float32))
            self.f0_log_std.copy_(torch.tensor(float(stats["f0_log_std"]), dtype=torch.float32).clamp_min(1e-6))
            self.energy_mean.copy_(torch.tensor(float(stats["energy_mean"]), dtype=torch.float32))
            self.energy_std.copy_(torch.tensor(float(stats["energy_std"]), dtype=torch.float32).clamp_min(1e-6))

    # ── tiện ích tensor ────────────────────────────────────────────────────────
    def _active_mask(self, features: torch.Tensor) -> torch.Tensor:
        """[B,N,7] long → [B,N,4] float {0,1} theo cấu trúc (NONE = id 0; nucleus luôn active)."""
        onset = (features[..., 0] != 0).to(torch.float32)
        medial = (features[..., 1] != 0).to(torch.float32)
        coda = (features[..., 3] != 0).to(torch.float32)
        nucleus = torch.ones_like(onset)
        return torch.stack([onset, medial, nucleus, coda], dim=-1)

    def _segment_features(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """concat(h_syl, role_emb, emb thành phần của role) → [B,N,4,in_dim]."""
        tables = self.embedding.tables
        batch, syllables = x.shape[0], x.shape[1]
        components = torch.stack([tables[0](features[..., 0]), tables[1](features[..., 1]),
                                  tables[2](features[..., 2]), tables[3](features[..., 3])], dim=2)
        role = self.role_emb.weight[:4].view(1, 1, 4, self.role_dim).expand(batch, syllables, -1, -1)
        hidden = x.unsqueeze(2).expand(-1, -1, 4, -1)
        return torch.cat([hidden, components, role], dim=-1)

    def _gap_vectors(self, features: torch.Tensor) -> torch.Tensor:
        """Slot G: `silence_emb + Linear(boundary_emb, punct_emb)` của âm tiết đứng trước (§1)."""
        tables = self.embedding.tables
        tokens = torch.cat([tables[5](features[..., 5]), tables[6](features[..., 6])], dim=-1)
        return self.g_token(tokens) + self.silence_emb.view(1, 1, -1)

    def _lead_vector(self) -> torch.Tensor:
        """`G_lead`: chung `silence_emb`, boundary = SENTENCE (id 3), punct = NONE (id 0)."""
        tables = self.embedding.tables
        tokens = torch.cat([tables[5].weight[3], tables[6].weight[0]]).view(1, -1)
        return self.g_token(tokens) + self.silence_emb.view(1, -1)

    def _frame_slot_index(self, slot_durations: torch.Tensor, frames: int) -> torch.Tensor:
        """Chỉ số slot của từng frame (clamp slot cuối) — cùng mẹo cumsum như LengthRegulator."""
        cumulative = slot_durations.cumsum(dim=1)
        steps = torch.arange(frames, device=slot_durations.device)
        return (cumulative.unsqueeze(2) <= steps.view(1, 1, -1)).sum(dim=1).clamp(max=slot_durations.size(1) - 1)

    # ── forward ────────────────────────────────────────────────────────────────
    def forward(
        self,
        features: torch.Tensor,
        durations: torch.Tensor | None = None,
        seg_durations: torch.Tensor | None = None,
        gaps: torch.Tensor | None = None,
        lead: torch.Tensor | None = None,
        speed: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """features [B,N,7] → dict đầu ra (docstring module: quy ước tham số + output)."""
        batch, syllables = features.shape[0], features.shape[1]
        padding_mask = None
        if input_lengths is not None:
            steps = torch.arange(syllables, device=features.device)
            padding_mask = steps.view(1, -1) < input_lengths.unsqueeze(1)
        x = self.encoder(self.embedding(features), padding_mask)

        # ── ratio (softmax có mask active) ─────────────────────────────────────
        active = self._active_mask(features)
        segment_in = self._segment_features(features, x)
        ratio_logits = self.ratio_mlp(segment_in).squeeze(-1)
        ratio = torch.softmax(ratio_logits.masked_fill(active < 0.5, -1e9), dim=-1) * active
        ratio = ratio / ratio.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # ── d_syl (+ speed: chỉ scale khi tự dự đoán — §5) ─────────────────────
        log_d = self.duration(x).squeeze(-1)
        free = durations is None
        speed_value: torch.Tensor | None = None
        if speed is not None:
            speed_value = speed.to(torch.float32).reshape(()).clamp_min(0.1)
        if free:
            predicted = torch.round((torch.exp(log_d) - 1.0).clamp_min(1)).long()
            if speed_value is not None:
                predicted = torch.round(predicted.to(torch.float32) / speed_value).clamp_min(1).long()
        else:
            predicted = durations.long()
        d_syl = predicted

        # ── d_seg (B2 khi không có seg_durations) ──────────────────────────────
        if seg_durations is not None:
            d_seg = seg_durations.long()
        else:
            d_seg = distribute_slots(d_syl, ratio, active)

        # ── pause (gate quyết định ở speed gốc — §5) ───────────────────────────
        pause_heads = self.pause(x)
        gate_prob = torch.sigmoid(pause_heads[..., 0])
        log_gap = pause_heads[..., 1]
        if gaps is None:
            gap = torch.round((torch.exp(log_gap) - 1.0).clamp_min(1)).long()
            if speed_value is not None:
                gap = torch.round(gap.to(torch.float32) / speed_value).clamp_min(1).long()
            gap = torch.where(gate_prob > 0.5, gap, torch.zeros_like(gap))
        else:
            gap = gaps.long()

        # ── slot [B, 1+5N] + hidden/vị trí ────────────────────────────────────
        lead_frames = torch.zeros(batch, 1, dtype=torch.long, device=features.device)
        if lead is not None:
            lead_frames = lead.view(batch, 1).long()
        slot_5 = torch.cat([d_seg, gap.unsqueeze(-1)], dim=-1).reshape(batch, -1)  # [B,5N]
        slot_durations = torch.cat([lead_frames, slot_5], dim=1)  # [B,1+5N]
        gap_flat = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=features.device),
                              (torch.arange(5 * syllables, device=features.device) % 5 == 4)
                              .view(1, -1).expand(batch, -1)], dim=1)

        gap_vectors = self._gap_vectors(features)
        slots = torch.cat([self.slot_expander(segment_in), gap_vectors.unsqueeze(2)], dim=2)
        hidden_slots = torch.cat([self._lead_vector().view(1, 1, -1).expand(batch, -1, -1),
                                  slots.reshape(batch, -1, self.hidden_size)], dim=1)
        expanded, lengths, frame_mask = self.length_regulator(hidden_slots, slot_durations)
        positions = slot_positions(slot_durations, gap_flat)
        index = self._frame_slot_index(slot_durations, expanded.size(1))
        gap_expanded = torch.gather(gap_flat.to(torch.float32), 1, index).unsqueeze(-1)

        # ── variance cấp segment → slot → frame ───────────────────────────────
        segment_out = self.variance_mlp(segment_in)  # [B,N,4,9]
        shape_z = segment_out[..., :4]
        shape_natural = shape_z * self.shape_std.view(1, 1, 1, 4) + self.shape_mean.view(1, 1, 1, 4)

        def segment_to_slot(values: torch.Tensor) -> torch.Tensor:
            """[B,N,4(,k)] → [B,1+5N(,k)] với **G_lead + slot G = 0** (khớp `slot_durations`)."""
            shape = list(values.shape)
            zeros = torch.zeros(*shape[:2], 1, *shape[3:], dtype=values.dtype, device=values.device)
            padded = torch.cat([values, zeros], dim=2).reshape(batch, -1, *shape[3:])
            head = torch.zeros(batch, 1, *shape[3:], dtype=values.dtype, device=values.device)
            return torch.cat([head, padded], dim=1)

        shape_slots = segment_to_slot(shape_natural)
        # `s_k` (raw) tính trên log-F0 **đã z-score** (f0_stats) → contour cần thêm biến đổi ngoài
        # `f0_log_mean + f0_log_std ×` để về log-F0 tự nhiên (§6); thiếu bước này f0 ra ~1,0 log.
        contour_shape = build_f0_contour(shape_slots, slot_durations, positions)
        f0_frames = (self.f0_log_mean + self.f0_log_std * contour_shape) * (1.0 - gap_expanded.squeeze(-1))

        variance_parts = [f0_frames.unsqueeze(-1)]
        for values in (segment_out[..., 4], segment_out[..., 5]):
            expanded_values, _, _ = self.length_regulator(segment_to_slot(values).unsqueeze(-1),
                                                          slot_durations)
            variance_parts.append(expanded_values * (1.0 - gap_expanded))
        if self.use_phonation:
            phonation_slots = segment_to_slot(segment_out[..., 6:9])
            expanded_phonation, _, _ = self.length_regulator(phonation_slots, slot_durations)
            variance_parts.append(expanded_phonation * (1.0 - gap_expanded))
        variance = torch.cat(variance_parts, dim=-1)

        decoded = self.decoder(
            expanded + self.variance_proj(variance) + self.position_proj(positions.unsqueeze(-1)),
            frame_mask,
        )
        mel_pre = self.mel(decoded)
        mel_refined = mel_pre
        if self.use_postnet:
            mel_refined = mel_pre + self.postnet(mel_pre.transpose(1, 2)).transpose(1, 2)

        # energy: head học **thô** (đơn vị thật, loss so raw) — không nhân lại thang (khác shape head).
        seg_energy_natural = segment_out[..., 5]
        outputs: dict[str, torch.Tensor] = {
            "mel": mel_refined,
            "log_duration": log_d,
            "durations": d_syl,
            "f0": f0_frames,  # [B,T] log-F0 (đơn vị tự nhiên) từ contour s_k
            "vuv_logits": segment_out[..., 4][..., NUCLEUS],
            "energy": seg_energy_natural[..., NUCLEUS],
            "tone_logits": self.tone_classifier(x),
            "lengths": lengths,
            "slot_durations": slot_durations,
            "pause_gate": gate_prob,
            "pause_logits": pause_heads[..., 0],
            "pause_log_gap": log_gap,
            "seg_ratio": ratio,
            "seg_ratio_logits": ratio_logits,
            "seg_shape": shape_z,
            "seg_shape_natural": shape_natural,
            "seg_vuv_logits": segment_out[..., 4],
            "seg_energy": segment_out[..., 5],
        }
        if self.use_postnet:
            outputs["mel_pre"] = mel_pre
        if self.use_phonation:
            outputs["phonation"] = segment_out[..., 6:9][..., NUCLEUS, :]
            outputs["seg_phonation"] = segment_out[..., 6:9]
        return outputs
