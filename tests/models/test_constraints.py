"""Field bounds have to bite, and bite on the right field.

Every rejection here asserts the error `loc` and `type` as well as the exception class.
`pytest.raises(ValidationError)` on its own passes when the payload was wrong for a
different reason — a renamed field, a typo in the factory — and that is exactly how a
bounds test ends up asserting nothing.

The pair that matters: `anchor_quality_score` is 0-100 and the four audit dimensions are
0-1. A model that put all five on the same scale still passes a test that only checks
0.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from factories import (
    ANCHOR_SPEC,
    AUDIT_SPEC,
    LINK_SPEC,
    RECOMMENDATION_SPEC,
    TARGET_URL,
    UNIT_INTERVAL_FIELDS,
)
from pydantic import ValidationError

from linking_engine.models import (
    ActionType,
    ContentGapFinding,
    OrphanRescue,
    OrphanSlotReason,
    PageProfile,
    Recommendation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

CONTENT_GAP_KWARGS = RECOMMENDATION_SPEC.kwargs_with(
    action_type=ActionType.CONTENT_GAP,
    label="content gap: add copy first",
    finding=ContentGapFinding.NO_TOPICAL_MENTION,
    advice="Write a sentence about the topic first.",
    proposed_anchors=None,
    suggested=False,
)
AUDIT_VERDICT_KWARGS = RECOMMENDATION_SPEC.kwargs_with(
    action_type=ActionType.REMOVE,
    label="remove this link",
    score=None,
    tier=None,
    rank_in_source=None,
    best_rank=None,
    position=3,
    proposed_anchors=None,
    suggested=False,
)


def _single_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> Mapping[str, object]:
    """The one error pydantic raised, asserting the rest of the payload was valid."""
    errors = exc_info.value.errors()
    assert len(errors) == 1, f"expected exactly one validation error, got {errors}"
    return errors[0]


@pytest.mark.parametrize("value", [100.1, 101, 1000.0])
def test_anchor_quality_score_rejects_above_100(value) -> None:
    with pytest.raises(ValidationError) as exc_info:
        AUDIT_SPEC.model(**AUDIT_SPEC.kwargs_with(anchor_quality_score=value))
    error = _single_error(exc_info)
    assert error["loc"] == ("anchor_quality_score",)
    assert error["type"] == "less_than_equal"


@pytest.mark.parametrize("value", [-0.1, -1])
def test_anchor_quality_score_rejects_below_zero(value) -> None:
    with pytest.raises(ValidationError) as exc_info:
        AUDIT_SPEC.model(**AUDIT_SPEC.kwargs_with(anchor_quality_score=value))
    error = _single_error(exc_info)
    assert error["loc"] == ("anchor_quality_score",)
    assert error["type"] == "greater_than_equal"


@pytest.mark.parametrize("value", [0, 50.5, 100])
def test_anchor_quality_score_spans_the_full_0_to_100_range(value) -> None:
    """100 is the load-bearing case: a 0-1 scale would reject it."""
    result = AUDIT_SPEC.model(**AUDIT_SPEC.kwargs_with(anchor_quality_score=value))
    assert result.anchor_quality_score == value


@pytest.mark.parametrize("field", UNIT_INTERVAL_FIELDS)
@pytest.mark.parametrize("value", [1.1, 100])
def test_unit_interval_dimensions_reject_above_one(field, value) -> None:
    with pytest.raises(ValidationError) as exc_info:
        AUDIT_SPEC.model(**AUDIT_SPEC.kwargs_with(**{field: value}))
    error = _single_error(exc_info)
    assert error["loc"] == (field,), f"{field} is not the field that rejected {value}"
    assert error["type"] == "less_than_equal"


@pytest.mark.parametrize("field", UNIT_INTERVAL_FIELDS)
@pytest.mark.parametrize("value", [-0.1, -1])
def test_unit_interval_dimensions_reject_below_zero(field, value) -> None:
    with pytest.raises(ValidationError) as exc_info:
        AUDIT_SPEC.model(**AUDIT_SPEC.kwargs_with(**{field: value}))
    error = _single_error(exc_info)
    assert error["loc"] == (field,)
    assert error["type"] == "greater_than_equal"


@pytest.mark.parametrize("field", UNIT_INTERVAL_FIELDS)
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_unit_interval_dimensions_accept_the_closed_interval(field, value) -> None:
    result = AUDIT_SPEC.model(**AUDIT_SPEC.kwargs_with(**{field: value}))
    assert getattr(result, field) == value


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [("", "string_too_short"), ("x" * 121, "string_too_long")],
    ids=["empty", "121-chars"],
)
def test_anchor_text_length_is_bounded(value, expected_type) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ANCHOR_SPEC.model(**ANCHOR_SPEC.kwargs_with(text=value))
    error = _single_error(exc_info)
    assert error["loc"] == ("text",)
    assert error["type"] == expected_type


@pytest.mark.parametrize("length", [1, 120])
def test_anchor_text_accepts_both_inclusive_bounds(length) -> None:
    candidate = ANCHOR_SPEC.model(**ANCHOR_SPEC.kwargs_with(text="a" * length))
    assert len(candidate.text) == length


@pytest.mark.parametrize("value", ["SYNTHESISED", "extracted", ""])
def test_anchor_source_rejects_anything_outside_the_literal(value) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ANCHOR_SPEC.model(**ANCHOR_SPEC.kwargs_with(source=value))
    error = _single_error(exc_info)
    assert error["loc"] == ("source",)
    assert error["type"] == "literal_error"


@pytest.mark.parametrize("value", ["EXTRACTED", "GENERATED"])
def test_anchor_source_accepts_both_documented_values(value) -> None:
    """GENERATED should not occur post-v5, but acceptance is tracked separately."""
    candidate = ANCHOR_SPEC.model(**ANCHOR_SPEC.kwargs_with(source=value))
    assert candidate.source == value


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [(1.1, "less_than_equal"), (-0.1, "greater_than_equal")],
)
def test_anchor_score_is_a_unit_interval(value, expected_type) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ANCHOR_SPEC.model(**ANCHOR_SPEC.kwargs_with(score=value))
    error = _single_error(exc_info)
    assert error["loc"] == ("score",)
    assert error["type"] == expected_type


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_anchor_score_accepts_the_closed_unit_interval(value) -> None:
    candidate = ANCHOR_SPEC.model(**ANCHOR_SPEC.kwargs_with(score=value))
    assert candidate.score == value


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [(100.1, "less_than_equal"), (-0.1, "greater_than_equal")],
)
def test_recommendation_score_is_0_to_100(value, expected_type) -> None:
    """The opportunity score is 0-100, on the same scale as anchor_quality_score."""
    with pytest.raises(ValidationError) as exc_info:
        RECOMMENDATION_SPEC.model(**RECOMMENDATION_SPEC.kwargs_with(score=value))
    error = _single_error(exc_info)
    assert error["loc"] == ("score",)
    assert error["type"] == expected_type


@pytest.mark.parametrize("value", [0, 100])
def test_recommendation_score_accepts_both_inclusive_bounds(value) -> None:
    recommendation = RECOMMENDATION_SPEC.model(**RECOMMENDATION_SPEC.kwargs_with(score=value))
    assert recommendation.score == value


def _assert_rejected_by_the_model(
    exc_info: pytest.ExceptionInfo[ValidationError], message: str
) -> None:
    error = _single_error(exc_info)
    assert error["loc"] == ()
    assert error["type"] == "value_error"
    assert error["msg"] == f"Value error, {message}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {**AUDIT_VERDICT_KWARGS, "best_rank": 1},
        RECOMMENDATION_SPEC.kwargs_with(best_rank=None),
        {**CONTENT_GAP_KWARGS, "best_rank": None},
    ],
    ids=["audit-verdict-with-one", "add-link-without", "content-gap-without"],
)
def test_best_rank_is_set_exactly_for_new_link_actions(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Recommendation.model_validate(kwargs)
    _assert_rejected_by_the_model(exc_info, "best_rank is set exactly for new-link actions")


@pytest.mark.parametrize(
    "kwargs",
    [{**CONTENT_GAP_KWARGS, "suggested": True}, {**AUDIT_VERDICT_KWARGS, "suggested": True}],
    ids=["content-gap", "audit-verdict"],
)
def test_only_an_add_link_can_be_suggested(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Recommendation.model_validate(kwargs)
    _assert_rejected_by_the_model(exc_info, "suggested and orphan_slot are ADD_LINK only")


def test_an_orphan_slot_that_is_not_suggested_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Recommendation.model_validate(
            RECOMMENDATION_SPEC.kwargs_with(suggested=False, orphan_slot=True)
        )
    _assert_rejected_by_the_model(exc_info, "an orphan slot is always suggested")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (RECOMMENDATION_SPEC.kwargs_with(orphan_slot=True), (5, True, True)),
        (RECOMMENDATION_SPEC.kwargs_with(suggested=False), (5, False, False)),
        (CONTENT_GAP_KWARGS, (5, False, False)),
        (AUDIT_VERDICT_KWARGS, (None, False, False)),
    ],
    ids=["orphan-slot", "reserve", "content-gap", "audit-verdict"],
)
def test_best_rank_suggested_and_orphan_slot_accept_every_valid_combination(
    kwargs: dict[str, object], expected: tuple[int | None, bool, bool]
) -> None:
    recommendation = Recommendation.model_validate(kwargs)
    assert (
        recommendation.best_rank,
        recommendation.suggested,
        recommendation.orphan_slot,
    ) == expected


def _orphan_rescue(
    suggested_in: int, guaranteed: int, unmet_reason: OrphanSlotReason | None
) -> OrphanRescue:
    return OrphanRescue(
        profile=PageProfile(url=TARGET_URL, word_count=400, inbound=0, outbound=1),
        guaranteed=guaranteed,
        suggested_in=suggested_in,
        sources=(),
        unmet_reason=unmet_reason,
    )


@pytest.mark.parametrize(
    ("suggested_in", "guaranteed", "unmet_reason"),
    [
        (2, 2, OrphanSlotReason.NO_ANCHOR),
        (3, 2, OrphanSlotReason.NO_RELEVANT_SOURCE),
        (1, 2, None),
        (0, 1, None),
    ],
    ids=["met-with-a-reason", "exceeded-with-a-reason", "short-without", "none-in-without"],
)
def test_orphan_rescue_gives_a_reason_exactly_when_the_guarantee_is_unmet(
    suggested_in: int, guaranteed: int, unmet_reason: OrphanSlotReason | None
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _orphan_rescue(suggested_in, guaranteed, unmet_reason)
    _assert_rejected_by_the_model(
        exc_info, "unmet_reason is set exactly when fewer links came in than guaranteed"
    )


@pytest.mark.parametrize(
    ("suggested_in", "guaranteed", "unmet_reason"),
    [(2, 2, None), (3, 2, None), (1, 2, OrphanSlotReason.NO_ANCHOR), (0, 0, None)],
    ids=["met-exactly", "exceeded", "short", "nothing-guaranteed"],
)
def test_orphan_rescue_accepts_a_reason_only_below_the_guarantee(
    suggested_in: int, guaranteed: int, unmet_reason: OrphanSlotReason | None
) -> None:
    rescue = _orphan_rescue(suggested_in, guaranteed, unmet_reason)
    assert rescue.unmet_reason is unmet_reason


def test_link_position_defaults_to_body() -> None:
    kwargs = {key: value for key, value in LINK_SPEC.kwargs.items() if key != "link_position"}
    assert LINK_SPEC.model(**kwargs).link_position == "body"


@pytest.mark.parametrize("value", ["footer", "nav", "sidebar", "header", "BODY", 1])
def test_link_position_rejects_anything_but_body(value) -> None:
    """ADR-004 captures body links only, so any other value means extraction broke.

    The Literal turns that into a loud validation error instead of silent bad data
    that would skew PageRank.
    """
    with pytest.raises(ValidationError) as exc_info:
        LINK_SPEC.model(**LINK_SPEC.kwargs_with(link_position=value))
    error = _single_error(exc_info)
    assert error["loc"] == ("link_position",)
    assert error["type"] == "literal_error"
