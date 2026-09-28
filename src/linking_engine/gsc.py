"""GSC helpers shared by keyword resolution and feature assembly: term normalisation and the
tenant's own CTR curve. Curves differ by vertical and SERP mix, so no industry table is used."""

from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING, Final

from linking_engine.models import CtrCurve

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.models import GscQueryStats

# Below either, the curve is too noisy to rank queries by, and the GSC rung is skipped.
MIN_CURVE_ROWS: Final = 100
MIN_CURVE_IMPRESSIONS: Final = 10_000
MAX_POSITION: Final = 100


def normalise_term(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _position(position: float) -> int:
    if not position >= 1:
        raise ValueError(f"a position is at least 1, got {position}")
    return int(min(position, MAX_POSITION) + 0.5)


def fit_ctr_curve(rows: Sequence[GscQueryStats]) -> CtrCurve | None:
    """Impression-weighted CTR per rounded position, made non-increasing by weighted isotonic
    regression; None when the tenant's data is too thin."""
    impressions = sum(row.impressions for row in rows)
    if len(rows) < MIN_CURVE_ROWS or impressions < MIN_CURVE_IMPRESSIONS:
        return None
    shown = [0] * MAX_POSITION
    clicked = [0] * MAX_POSITION
    for row in rows:
        slot = _position(row.position) - 1
        shown[slot] += row.impressions
        clicked[slot] += row.clicks
    observed = [
        (slot, min(clicked[slot] / shown[slot], 1.0), shown[slot])
        for slot in range(MAX_POSITION)
        if shown[slot]
    ]

    # Pool adjacent violators: merge any block whose CTR rises above the block before it.
    blocks: list[tuple[float, int, int]] = []
    for _, rate, weight in observed:
        value, total, count = rate, weight, 1
        while blocks and blocks[-1][0] < value:
            previous, previous_weight, previous_count = blocks.pop()
            value = (previous * previous_weight + value * total) / (previous_weight + total)
            total += previous_weight
            count += previous_count
        blocks.append((value, total, count))
    pooled: list[float] = []
    for value, _, count in blocks:
        pooled.extend([value] * count)
    fitted = {slot: value for (slot, _, _), value in zip(observed, pooled, strict=True)}

    # Positions without data take the previous position's value; any before the first
    # observed position take its value.
    ctr: list[float] = []
    value = fitted[observed[0][0]]
    for slot in range(observed[-1][0] + 1):
        value = fitted.get(slot, value)
        ctr.append(value)
    return CtrCurve(ctr=tuple(ctr), rows=len(rows), impressions=impressions)


def ctr_at(curve: CtrCurve, position: float) -> float:
    """The curve's CTR at the rounded position, clamped to its last position."""
    return curve.ctr[min(_position(position), len(curve.ctr)) - 1]


def opportunity_value(curve: CtrCurve, impressions: int, position: float) -> float:
    """Estimated click upside of reaching position 1: impressions x (CTR@1 - CTR@position)."""
    if impressions < 0:
        raise ValueError("impressions cannot be negative")
    return max(0.0, impressions * (ctr_at(curve, 1) - ctr_at(curve, position)))
