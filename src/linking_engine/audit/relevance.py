"""Existing-link relevance: how a tenant's scores spread and where they split into two modes.

The split is derived per tenant from its own scores: thresholds do not carry over between
embedding models or sites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from sklearn.mixture import GaussianMixture

from linking_engine.models import ScoreDistribution
from linking_engine.models.relevance import HISTOGRAM_BINS

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

# Fewer scores than this give no split: two modes cannot be told apart from noise.
MIN_SPLIT_SCORES: Final = 50
# Two mixture means closer than this are one mode.
MIN_MODE_GAP: Final = 0.05
SPLIT_SEED: Final = 0
_BISECTIONS: Final = 60


def mixture_split(scores: npt.NDArray[np.float64]) -> float | None:
    """Where a two-component Gaussian mixture's posterior flips from the low mode to the high
    one, between the two means; None for too few scores or modes too close to separate."""
    if len(scores) < MIN_SPLIT_SCORES or np.ptp(scores) < MIN_MODE_GAP:
        return None
    mixture = GaussianMixture(n_components=2, random_state=SPLIT_SEED).fit(scores.reshape(-1, 1))
    means = mixture.means_.ravel()
    low, high = int(np.argmin(means)), int(np.argmax(means))
    lo, hi = float(means[low]), float(means[high])
    if hi - lo < MIN_MODE_GAP:
        return None

    def high_side(x: float) -> bool:
        return bool(mixture.predict_proba(np.array([[x]]))[0, high] >= 0.5)

    if high_side(lo) or not high_side(hi):
        return None
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if high_side(mid) else (mid, hi)
    return hi


def score_distribution(scores: Sequence[float]) -> ScoreDistribution | None:
    """Mean, percentiles, histogram and mixture split of scores in [0, 1]; None without scores."""
    if not scores:
        return None
    values = np.asarray(scores, dtype=np.float64)
    p10, p25, p50, p75, p90 = (float(p) for p in np.percentile(values, [10, 25, 50, 75, 90]))
    counts, _ = np.histogram(values, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
    split = mixture_split(values)
    return ScoreDistribution(
        count=len(values),
        mean=min(1.0, max(0.0, float(values.mean()))),
        p10=p10,
        p25=p25,
        p50=p50,
        p75=p75,
        p90=p90,
        histogram=tuple(int(n) for n in counts),
        split=split,
        low_share=None if split is None else float((values < split).mean()),
    )
