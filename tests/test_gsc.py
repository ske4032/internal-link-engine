"""The tenant's own CTR curve and the opportunity value it prices GSC queries with.

Curves are fitted from hand-built rows whose expected values are worked out in the test:
impression-weighted CTR per rounded position, pooled where it rises, gaps carried forward.
"""

from __future__ import annotations

import pytest

from linking_engine.gsc import (
    MIN_CURVE_IMPRESSIONS,
    MIN_CURVE_ROWS,
    ctr_at,
    fit_ctr_curve,
    normalise_term,
    opportunity_value,
)
from linking_engine.models import CtrCurve, GscQueryStats


def rows_at(position: float, impressions: int, clicks: int, count: int = 50) -> list[GscQueryStats]:
    """``count`` rows at one position whose totals are ``impressions`` and ``clicks``."""
    each, extra = divmod(impressions, count)
    click_each, click_extra = divmod(clicks, count)
    return [
        GscQueryStats(
            url=f"example.com/p{i}",
            query=f"query {position} {i}",
            impressions=each + (i < extra),
            clicks=click_each + (i < click_extra),
            position=position,
        )
        for i in range(count)
    ]


def curve(*ctr: float) -> CtrCurve:
    return CtrCurve(ctr=ctr, rows=MIN_CURVE_ROWS, impressions=MIN_CURVE_IMPRESSIONS)


def test_the_thresholds_are_the_contracts() -> None:
    assert (MIN_CURVE_ROWS, MIN_CURVE_IMPRESSIONS) == (100, 10_000)


# ── normalise_term ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "normalised"),
    [
        ("Trail Shoes", "trail shoes"),
        ("  trail \t\n  shoes  ", "trail shoes"),
        ("trail\u00a0shoes", "trail shoes"),
        ("\uff34\uff52\uff41\uff49\uff4c", "trail"),
        ("Stra\u00dfe", "strasse"),
        ("Cafe\u0301", "caf\u00e9"),
        (" \t ", ""),
        ("", ""),
    ],
)
def test_normalisation_is_the_signal_stages_rule(raw: str, normalised: str) -> None:
    """One rule for queries everywhere, or overlap and keyword resolution disagree."""
    assert normalise_term(raw) == normalised


def test_normalisation_folds_case_whitespace_and_compatibility_forms() -> None:
    assert normalise_term("  TRAIL\u00a0\uff33hoes \n") == "trail shoes"


# ── fit_ctr_curve ───────────────────────────────────────────────────────────


def test_ctr_is_impression_weighted_per_position() -> None:
    # Position 1: 1500 + 500 clicks over 5000 + 1000 impressions is 1/3, not the 0.4 mean
    # of the two rows' CTRs.
    rows = [
        *rows_at(1, 5000, 1500, count=25),
        *rows_at(1, 1000, 500, count=25),
        *rows_at(2, 3000, 450),
        *rows_at(3, 2000, 100),
    ]

    fitted = fit_ctr_curve(rows)

    assert fitted is not None
    assert fitted.ctr == pytest.approx((1 / 3, 0.15, 0.05))
    assert (fitted.rows, fitted.impressions) == (150, 11_000)


def test_positions_are_rounded_to_the_nearest_integer() -> None:
    rows = [*rows_at(1.4, 6000, 1800), *rows_at(1.6, 4000, 400), *rows_at(2.2, 2000, 400)]

    fitted = fit_ctr_curve(rows)

    # 1.4 is position 1 (0.3); 1.6 and 2.2 are position 2: 800 / 6000.
    assert fitted is not None
    assert fitted.ctr == pytest.approx((0.3, 800 / 6000))


def test_a_rising_ctr_is_pooled_by_impressions() -> None:
    rows = [*rows_at(1, 1000, 100), *rows_at(2, 3000, 900), *rows_at(3, 8000, 400)]

    fitted = fit_ctr_curve(rows)

    # Positions 1 and 2 violate the order: pooled to (100 + 900) / (1000 + 3000).
    assert fitted is not None
    assert fitted.ctr == pytest.approx((0.25, 0.25, 0.05))


def test_pooling_cascades_back_through_earlier_positions() -> None:
    rows = [*rows_at(1, 3000, 600), *rows_at(2, 3000, 300), *rows_at(3, 6000, 1800)]

    fitted = fit_ctr_curve(rows)

    # 2 and 3 pool to 2100 / 9000 = 0.233, which now exceeds position 1's 0.2, so all three
    # pool to 2700 / 12000.
    assert fitted is not None
    assert fitted.ctr == pytest.approx((0.225, 0.225, 0.225))


def test_positions_without_data_take_the_previous_value() -> None:
    rows = [*rows_at(1, 5000, 1500), *rows_at(2, 4000, 800), *rows_at(5, 3000, 150)]

    fitted = fit_ctr_curve(rows)

    assert fitted is not None
    assert fitted.ctr == pytest.approx((0.3, 0.2, 0.2, 0.2, 0.05))


def test_rows_with_no_impressions_carry_no_weight() -> None:
    rows = [*rows_at(1, 6000, 1800), *rows_at(2, 0, 0), *rows_at(3, 6000, 600)]

    fitted = fit_ctr_curve(rows)

    assert fitted is not None
    assert fitted.ctr == pytest.approx((0.3, 0.3, 0.1))


def test_positions_beyond_100_count_as_100() -> None:
    rows = [*rows_at(1, 8000, 1600), *rows_at(150, 4000, 40)]

    fitted = fit_ctr_curve(rows)

    assert fitted is not None
    assert len(fitted.ctr) == 100
    assert (fitted.ctr[0], fitted.ctr[98], fitted.ctr[99]) == pytest.approx((0.2, 0.2, 0.01))


@pytest.mark.parametrize(
    ("count", "impressions", "enabled"),
    [
        pytest.param(99, 20_000, False, id="too-few-rows"),
        pytest.param(100, 9_999, False, id="too-few-impressions"),
        pytest.param(100, 10_000, True, id="at-both-thresholds"),
    ],
)
def test_thin_gsc_data_disables_the_curve(count: int, impressions: int, enabled: bool) -> None:
    fitted = fit_ctr_curve(rows_at(1, impressions, impressions // 10, count=count))
    assert (fitted is not None) is enabled


def test_no_rows_give_no_curve() -> None:
    assert fit_ctr_curve([]) is None


# ── ctr_at and opportunity_value ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("position", "ctr"),
    [(1, 0.3), (2, 0.2), (3, 0.05), (1.4, 0.3), (2.6, 0.05), (3.0, 0.05), (4, 0.05), (97.2, 0.05)],
)
def test_ctr_is_read_at_the_rounded_position_and_clamped_to_the_last(
    position: float, ctr: float
) -> None:
    assert ctr_at(curve(0.3, 0.2, 0.05), position) == pytest.approx(ctr)


@pytest.mark.parametrize(
    ("impressions", "position", "value"),
    [
        pytest.param(1000, 1, 0.0, id="already-first"),
        pytest.param(1000, 2, 100.0, id="one-step"),
        pytest.param(1000, 40, 250.0, id="beyond-the-curve"),
        pytest.param(0, 3, 0.0, id="no-demand"),
    ],
)
def test_opportunity_is_the_click_upside_of_reaching_position_one(
    impressions: int, position: float, value: float
) -> None:
    assert opportunity_value(curve(0.3, 0.2, 0.05), impressions, position) == pytest.approx(value)


def test_opportunity_is_never_negative() -> None:
    assert opportunity_value(curve(0.25, 0.25, 0.25), 5000, 3) == 0.0


def test_positions_before_the_first_observed_one_take_its_value() -> None:
    rows = [*rows_at(3, 6000, 600), *rows_at(4, 6000, 300)]

    fitted = fit_ctr_curve(rows)

    assert fitted is not None
    assert fitted.ctr == pytest.approx((0.1, 0.1, 0.1, 0.05))


def test_positions_below_one_and_negative_impressions_are_refused() -> None:
    fitted = curve(0.3, 0.2)
    with pytest.raises(ValueError, match="at least 1"):
        ctr_at(fitted, 0.5)
    with pytest.raises(ValueError, match="negative"):
        opportunity_value(fitted, -1, 2)
