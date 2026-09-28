"""Existing-link relevance maths: score distributions and the two-mode mixture split."""

from __future__ import annotations

import numpy as np
import pytest

from linking_engine.audit.relevance import MIN_SPLIT_SCORES, mixture_split, score_distribution


def bimodal(
    low: float, high: float, sizes: tuple[int, int] = (200, 100), sd: float = 0.03
) -> np.ndarray:
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(low, sd, sizes[0]), rng.normal(high, sd, sizes[1])])
    clipped: np.ndarray = np.clip(values, 0.0, 1.0)
    return clipped


def test_two_separated_modes_split_where_the_posterior_flips() -> None:
    scores = bimodal(0.3, 0.7)

    split = mixture_split(scores)

    assert split is not None
    assert 0.45 < split < 0.55
    assert np.mean(scores < split) == pytest.approx(2 / 3, abs=0.01)


def test_the_split_is_deterministic() -> None:
    scores = bimodal(0.2, 0.6, (120, 180))

    assert mixture_split(scores) == mixture_split(scores.copy())


@pytest.mark.parametrize(
    "scores",
    [
        bimodal(0.3, 0.7)[: MIN_SPLIT_SCORES - 1],
        # Two clean modes, but their means are closer than MIN_MODE_GAP.
        bimodal(0.48, 0.52, sd=0.006),
        np.full(80, 0.5),
    ],
    ids=["too-few", "modes-too-close", "constant"],
)
def test_scores_without_two_modes_have_no_split(scores: np.ndarray) -> None:
    assert mixture_split(scores) is None


def test_a_distribution_has_numpy_percentiles_and_a_histogram_over_the_unit_interval() -> None:
    scores = [0.0, 0.1, 0.12, 0.5, 0.52, 0.8, 1.0]

    found = score_distribution(scores)

    assert found is not None
    expected = np.percentile(scores, [10, 25, 50, 75, 90])
    assert [found.p10, found.p25, found.p50, found.p75, found.p90] == pytest.approx(expected)
    assert found.count == 7
    assert found.mean == pytest.approx(np.mean(scores))
    # 1.0 falls in the last bin, 0.0 in the first.
    assert found.histogram == (1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1)
    assert (found.split, found.low_share) == (None, None)


def test_a_bimodal_distribution_reports_its_split_and_low_share() -> None:
    found = score_distribution(bimodal(0.3, 0.7).tolist())

    assert found is not None
    assert found.split is not None
    assert found.low_share == pytest.approx(2 / 3, abs=0.01)


def test_no_scores_have_no_distribution() -> None:
    assert score_distribution([]) is None
