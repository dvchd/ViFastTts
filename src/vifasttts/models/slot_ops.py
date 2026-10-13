"""Phép toán slot v2 — interface **đóng băng** (B2 §2 + speed §5 + F0 §6, docs/ACOUSTIC_V2.md).

Ba hàm, chỉ dùng phép toán export ONNX được (không vòng lặp động, không argsort):

- `distribute_slots(d_syl, ratio, active) -> d_seg` — B2: `k = min(d_syl, n_act)` frame đầu
  cấp cho slot active **ưu tiên cao nhất** (N, O, C, M), phần còn lại chia theo `ratio`
  bằng floor + largest remainder; hòa phá theo **chỉ số role cố định** (O, M, N, C).
  Bất biến: inactive = 0 · active ≥ 1 khi `d_syl ≥ n_act` · `Σ d_seg == d_syl`.
- `slot_positions(slot_durations, gap_mask) -> pos` — `(t + 0,5)/n` trong từng slot;
  slot G (`gap_mask`) trả 0.
- `build_f0_contour(s, slot_durations, pos) -> f0` — §6: `f = s0 + √2 Σ_{k≥1} s_k cos(πk·pos)`
  (`s` là hệ số hình dạng z-score theo slot; khớp `idct4_frames` khi `s = c/√n`).

Thứ tự cột role: **O, M, N, C** (khớp `seg_durations` manifest). Runtime tăng tốc speed =
`max(1, round(d_syl/speed))` rồi gọi lại `distribute_slots` với **cùng ratio** (§5).
"""

from __future__ import annotations

import math

import torch

ROLE_ORDER = ("O", "M", "N", "C")
# Thứ hạng ưu tiên khi thiếu frame (B2): N > O > C > M → role O ưu tiên 1, M 3, N 0, C 2.
_PRIORITY_BY_INDEX = (1, 3, 0, 2)
SLOT_GAP = 4  # chỉ số slot G trong layout [O, M, N, C, G] nếu cần


def _frame_index(slot_durations: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """Chỉ số slot của từng frame — cùng mẹo cumsum như `LengthRegulator` (clamp slot cuối)."""
    cumulative = slot_durations.cumsum(dim=1)
    index = (cumulative.unsqueeze(2) <= steps.view(1, 1, -1)).sum(dim=1)
    return index.clamp(max=slot_durations.size(1) - 1)


def distribute_slots(d_syl: torch.Tensor, ratio: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """[B,N] long + [B,N,4] float + [B,N,4] (float/bool) → `d_seg` [B,N,4] long (B2)."""
    active = active > 0.5
    ratio = ratio.float() * active
    ratio = ratio / ratio.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    d_syl = d_syl.to(torch.long).clamp_min(0)
    n_act = active.to(torch.long).sum(dim=-1)  # [B,N]
    k = torch.minimum(d_syl, n_act)

    priority = torch.tensor(_PRIORITY_BY_INDEX, dtype=torch.long, device=active.device)
    # số slot active có ưu tiên cao hơn role i (j = trục cuối → mask active_j = unsqueeze(-2))
    higher = ((priority.view(1, 1, 1, 4) < priority.view(1, 1, 4, 1)) & active.unsqueeze(-2)).sum(dim=-1)
    base = (active & (higher < k.unsqueeze(-1))).to(torch.long)

    remainder = (d_syl - k).clamp_min(0)
    raw = ratio * remainder.unsqueeze(-1)
    floor = raw.floor()
    fraction = raw - floor
    extra = (remainder - floor.sum(dim=-1)).clamp_min(0)  # [B,N] ∈ [0, 3]

    role = torch.arange(4, device=active.device)
    before = (
        (fraction.unsqueeze(-2) > fraction.unsqueeze(-1))
        | ((fraction.unsqueeze(-2) == fraction.unsqueeze(-1))
           & (role.view(1, 1, 1, 4) < role.view(1, 1, 4, 1)))
    ) & active.unsqueeze(-2)  # role j active đứng trước i?
    rank = before.sum(dim=-1)
    add = (active & (rank < extra.unsqueeze(-1))).to(torch.long)
    return (base + floor.to(torch.long) + add) * active.to(torch.long)


def slot_positions(slot_durations: torch.Tensor, gap_mask: torch.Tensor | None = None,
                   max_frames: torch.Tensor | None = None) -> torch.Tensor:
    """[B,S] long (+ [B,S] gap, + max_frames) → `pos` [B,T] float = `(t+0,5)/n` từng slot.

    Slot G (`gap_mask=True`) trả 0; frame ngoài tổng độ dài (padding) trả > 1 và bị
    `frame_mask` che ở downstream (giống v1).
    """
    lengths = slot_durations.sum(dim=1)
    if max_frames is None:
        max_frames = lengths.max()
    steps = torch.arange(max_frames, device=slot_durations.device)  # type: ignore[call-overload]
    index = _frame_index(slot_durations, steps)
    starts = slot_durations.cumsum(dim=1) - slot_durations
    start = torch.gather(starts, 1, index)
    span = torch.gather(slot_durations, 1, index).clamp_min(1).float()
    local = (steps.view(1, -1) - start).clamp_min(0).float()
    pos = (local + 0.5) / span
    if gap_mask is not None:
        gap = torch.gather(gap_mask.to(torch.float32), 1, index)
        pos = pos * (1.0 - gap)
    return pos


def build_f0_contour(shape: torch.Tensor, slot_durations: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """[B,S,4] hệ số `s_k` + [B,S] long + [B,T] pos → contour `f0` [B,T] (§6).

    `s_k` z-score theo slot (T3.2 chuẩn hoá); hệ số thiếu (n < 4) caller đặt 0.
    """
    steps = torch.arange(pos.size(1), device=pos.device)  # type: ignore[call-overload]
    index = _frame_index(slot_durations, steps)
    per_frame = torch.gather(shape, 1, index.unsqueeze(-1).expand(-1, -1, shape.size(-1)))
    order = torch.arange(shape.size(-1), device=pos.device, dtype=pos.dtype).view(1, 1, -1)
    gains = torch.where(
        order == 0,
        torch.ones_like(pos).unsqueeze(-1),
        math.sqrt(2.0) * torch.cos(math.pi * order * pos.unsqueeze(-1)),
    )
    return (per_frame * gains).sum(dim=-1)
