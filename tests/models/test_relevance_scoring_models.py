"""Validators of the #16/#17 models: each rejection beside the boundary that passes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from linking_engine.models import (
    FeatureWeight,
    LinkRelevance,
    LinkRelevanceReport,
    ScoreDistribution,
    ScoreReport,
    ScorerWeights,
)

WHEN = datetime(2026, 9, 28, tzinfo=UTC)
KEY = "f" * 64


def only_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> dict[str, object]:
    errors = exc_info.value.errors()
    assert len(errors) == 1, f"expected exactly one validation error, got {errors}"
    return dict(errors[0])


# ── LinkRelevance ───────────────────────────────────────────────────────────


def link(**fields: object) -> LinkRelevance:
    values: dict[str, object] = {
        "source_url": "example.com/a",
        "position": 0,
        "target_url": "example.com/b",
        **fields,
    }
    return LinkRelevance.model_validate(values)


def test_a_generic_anchor_has_no_fit() -> None:
    assert link(anchor_generic=True, context_relevance=0.4).anchor_target_fit is None
    assert link(anchor_target_fit=0.7).anchor_target_fit == 0.7
    with pytest.raises(ValidationError, match="generic anchor has no anchor_target_fit"):
        link(anchor_generic=True, anchor_target_fit=0.7)


@pytest.mark.parametrize("field", ["context_relevance", "anchor_target_fit"])
def test_link_scores_are_normalised_cosines(field: str) -> None:
    assert link(**{field: 0.0}).model_dump()[field] == 0.0
    assert link(**{field: 1.0}).model_dump()[field] == 1.0
    for value in (-0.01, 1.01):
        with pytest.raises(ValidationError) as exc_info:
            link(**{field: value})
        assert only_error(exc_info)["loc"] == (field,)


@pytest.mark.parametrize(
    ("field", "value"), [("position", -1), ("source_url", ""), ("target_url", "")]
)
def test_link_identity_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        link(**{field: value})
    assert only_error(exc_info)["loc"] == (field,)


# ── ScoreDistribution ───────────────────────────────────────────────────────


def bins(count: int) -> tuple[int, ...]:
    """20 bins holding ``count`` in the first."""
    return (count, *([0] * 19))


def distribution(**fields: object) -> ScoreDistribution:
    count = int(str(fields.get("count", 60)))
    values: dict[str, object] = {
        "count": count,
        "histogram": bins(count) if count >= 0 else bins(0),
        "mean": 0.5,
        "p10": 0.1,
        "p25": 0.25,
        "p50": 0.5,
        "p75": 0.75,
        "p90": 0.9,
        **fields,
    }
    return ScoreDistribution.model_validate(values)


def test_equal_percentiles_are_allowed() -> None:
    flat = distribution(p10=0.4, p25=0.4, p50=0.4, p75=0.4, p90=0.4, mean=0.4)
    assert flat.p90 == flat.p10


@pytest.mark.parametrize(
    "fields",
    [{"p25": 0.05}, {"p50": 0.2}, {"p75": 0.45}, {"p90": 0.7}],
    ids=["p25", "p50", "p75", "p90"],
)
def test_decreasing_percentiles_are_refused(fields: dict[str, float]) -> None:
    with pytest.raises(ValidationError, match="percentiles must not decrease"):
        distribution(**fields)


@pytest.mark.parametrize(
    "histogram",
    [
        pytest.param((60, *([0] * 18)), id="19-bins"),
        pytest.param((59, *([0] * 19)), id="short"),
        pytest.param((61, -1, *([0] * 18)), id="negative"),
    ],
)
def test_the_histogram_has_twenty_bins_adding_up_to_the_count(histogram: tuple[int, ...]) -> None:
    assert sum(distribution(histogram=(30, 30, *([0] * 18))).histogram) == 60
    with pytest.raises(ValidationError):
        distribution(histogram=histogram)


def test_split_and_low_share_come_together() -> None:
    assert distribution(split=0.42, low_share=0.3).split == 0.42
    assert distribution().split is None
    for fields in ({"split": 0.42}, {"low_share": 0.3}):
        with pytest.raises(ValidationError, match="set together"):
            distribution(**fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [("mean", 1.1), ("split", -0.1), ("low_share", 1.5)],
)
def test_distribution_bounds(field: str, value: object) -> None:
    paired = {"split": 0.5, "low_share": 0.5}
    with pytest.raises(ValidationError) as exc_info:
        distribution(**{**paired, field: value})
    assert only_error(exc_info)["loc"] == (field,)


def test_a_distribution_describes_at_least_one_score() -> None:
    with pytest.raises(ValidationError) as exc_info:
        distribution(count=0)
    assert only_error(exc_info)["loc"] == ("count",)


# ── LinkRelevanceReport ─────────────────────────────────────────────────────


def relevance_report(**fields: object) -> LinkRelevanceReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "links": 10,
        "scored": 8,
        "generic_anchors": 2,
        "without_anchor_vector": 1,
        "context": distribution(count=8),
        "anchor": distribution(count=5),
        "seconds": 0.1,
        "finished_at": WHEN,
        **fields,
    }
    return LinkRelevanceReport.model_validate(values)


def test_the_relevance_report_fixture_is_valid() -> None:
    assert relevance_report().scored == 8


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"scored": 11}, "more scored links than links"),
        ({"generic_anchors": 5, "without_anchor_vector": 4}, "more unscored anchors"),
        ({"scored": 0, "generic_anchors": 0, "without_anchor_vector": 0}, "context distribution"),
        ({"context": None}, "context distribution"),
    ],
)
def test_relevance_report_counts_must_be_consistent(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        relevance_report(**fields)


def test_the_relevance_report_boundaries_pass() -> None:
    assert relevance_report(scored=10).scored == relevance_report().links
    assert relevance_report(generic_anchors=4, without_anchor_vector=4).scored == 8
    empty = relevance_report(
        scored=0, generic_anchors=0, without_anchor_vector=0, context=None, anchor=None
    )
    assert (empty.context, empty.anchor) == (None, None)
    assert relevance_report(anchor=None).anchor is None


# ── FeatureWeight and ScorerWeights ─────────────────────────────────────────


def test_a_feature_weight_defaults_to_higher_identity() -> None:
    weight = FeatureWeight(column="content_cosine", weight=0.15)
    assert (weight.direction, weight.normalisation) == ("higher", "identity")


@pytest.mark.parametrize(
    ("fields", "loc"),
    [
        ({"weight": 0}, ("weight",)),
        ({"weight": -0.1}, ("weight",)),
        ({"column": ""}, ("column",)),
        ({"direction": "up"}, ("direction",)),
        ({"normalisation": "minmax"}, ("normalisation",)),
    ],
)
def test_feature_weight_bounds(fields: dict[str, object], loc: tuple[str]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        FeatureWeight.model_validate({"column": "content_cosine", "weight": 0.15, **fields})
    assert only_error(exc_info)["loc"] == loc


def weights(**fields: object) -> ScorerWeights:
    values: dict[str, object] = {
        "version": "test-1",
        "features": (
            FeatureWeight(column="content_cosine", weight=0.5),
            FeatureWeight(column="same_hub", weight=0.5),
        ),
        **fields,
    }
    return ScorerWeights.model_validate(values)


def test_default_tier_shares_are_ten_and_thirty_percent() -> None:
    assert weights().tier_shares == (0.10, 0.30)
    assert weights(tier_shares=(0.49, 0.5)).tier_shares == (0.49, 0.5)


def test_a_column_is_weighted_once() -> None:
    twice = (
        FeatureWeight(column="same_hub", weight=0.5),
        FeatureWeight(column="same_hub", weight=0.2, direction="lower"),
    )
    with pytest.raises(ValidationError, match="weighted twice"):
        weights(features=twice)


@pytest.mark.parametrize(
    "shares",
    [(0.0, 0.3), (0.1, 1.0), (0.5, 0.5), (0.6, 0.6), (-0.1, 0.3)],
    ids=["zero", "one", "no-tier-3", "over-one", "negative"],
)
def test_tier_shares_leave_room_for_tier_three(shares: tuple[float, float]) -> None:
    with pytest.raises(ValidationError, match="tier shares"):
        weights(tier_shares=shares)


def test_weights_need_a_feature_and_a_version() -> None:
    for fields, loc in (({"features": ()}, ("features",)), ({"version": ""}, ("version",))):
        with pytest.raises(ValidationError) as exc_info:
            weights(**fields)
        assert only_error(exc_info)["loc"] == loc


# ── ScoreReport ─────────────────────────────────────────────────────────────


def score_report(**fields: object) -> ScoreReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "pairs": 10,
        "weights": weights(),
        "weights_hash": KEY,
        "score_histogram": (1, 1, *([0] * 16), 3, 5),
        "tiers": {1: 1, 2: 3, 3: 6},
        "score_p10": 5.0,
        "score_p50": 50.0,
        "score_p90": 95.0,
        "top_contributors": {"content_cosine": 6, "target_is_orphan": 4},
        "missing_share": {"target_ctr_gap": 0.8, "content_cosine": 0.0},
        "feature_cache_key": KEY,
        "seconds": 0.2,
        "finished_at": WHEN,
        **fields,
    }
    return ScoreReport.model_validate(values)


def test_the_score_report_fixture_is_valid() -> None:
    assert score_report().pairs == 10


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"tiers": {1: 1, 2: 3, 4: 6}}, "tiers are 1, 2 and 3"),
        ({"tiers": {1: -1, 2: 5, 3: 6}}, "tiers are 1, 2 and 3"),
        ({"tiers": {1: 1, 2: 3, 3: 5}}, "must add up to the pairs"),
        ({"score_p50": None}, "set exactly when there are pairs"),
        ({"top_contributors": {"content_cosine": 11}}, "more top contributors than pairs"),
        ({"missing_share": {"target_ctr_gap": 1.2}}, r"missing shares are in \[0, 1\]"),
    ],
)
def test_score_report_counts_must_be_consistent(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        score_report(**fields)


@pytest.mark.parametrize(
    "histogram",
    [
        pytest.param((10, *([0] * 18)), id="19-bins"),
        pytest.param((9, *([0] * 19)), id="short"),
        pytest.param((11, -1, *([0] * 18)), id="negative"),
    ],
)
def test_the_score_histogram_has_twenty_bins_adding_up_to_the_pairs(
    histogram: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError):
        score_report(score_histogram=histogram)


def test_an_empty_run_has_no_score_percentiles() -> None:
    empty = score_report(
        pairs=0,
        score_histogram=(0,) * 20,
        tiers={},
        score_p10=None,
        score_p50=None,
        score_p90=None,
        top_contributors={},
        missing_share={},
    )
    assert empty.score_p50 is None
    with pytest.raises(ValidationError, match="set exactly when there are pairs"):
        score_report(
            pairs=0, score_histogram=(0,) * 20, tiers={}, top_contributors={}, missing_share={}
        )


@pytest.mark.parametrize(("field", "value"), [("score_p90", 100.5), ("score_p10", -1.0)])
def test_scores_are_zero_to_one_hundred(field: str, value: float) -> None:
    assert score_report(score_p10=0.0, score_p90=100.0).score_p90 == 100.0
    with pytest.raises(ValidationError) as exc_info:
        score_report(**{field: value})
    assert only_error(exc_info)["loc"] == (field,)
