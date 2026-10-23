"""vifasttts.eval_stats — đếm thắng/thua/hòa theo đúng CHIỀU của từng thước.

Vì sao: báo cáo v18.33 in cột "% câu G1 tốt hơn" nhưng tính `delta > 0` cho mọi thước;
với flat_V/CER (thấp hơn tốt hơn) cột đó bị ngược nghĩa. Hàm này chuẩn hoá một chỗ.
"""

from __future__ import annotations

import math


def outcome_counts(values, *, higher_better: bool = True) -> dict:
    """Đếm win/loss/tie của `values` (delta = B − A) theo chiều `higher_better`.

    Bỏ NaN. `tie` = đúng bằng 0 (không có dung sai).
    """
    arr = [float(value) for value in values if not math.isnan(float(value))]
    wins = losses = ties = 0
    for value in arr:
        if value == 0:
            ties += 1
        elif (value > 0) == higher_better:
            wins += 1
        else:
            losses += 1
    total = len(arr)
    return {
        "n": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_pct": round(100.0 * wins / total, 2) if total else None,
        "loss_pct": round(100.0 * losses / total, 2) if total else None,
        "tie_pct": round(100.0 * ties / total, 2) if total else None,
    }
