"""Validators of the #22/#23 anchor selection models: each rejection beside the boundary that
passes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from linking_engine.models import (
    AnchorChoice,
    AnchorMatch,
    AnchorRung,
    AnchorScore,
    AnchorSelectionReport,
    AnchorType,
    AnchorTypeProfile,
    ContentGapFinding,
    KeywordSource,
    SemanticThreshold,
    UnanchoredPair,
    UnanchoredReason,
)
from linking_engine.models.anchors import (
    SCORE_HISTOGRAM_BINS,
    UNANCHORED_ADVICE,
    content_gap_finding,
)

NO_MENTION = UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC
AWKWARD = UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE
NO_KEYWORD = UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD
NO_TEXT = UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE
NOT_SEARCHED = UnanchoredReason.MEANING_SEARCH_NOT_RUN


def locs(exc_info: pytest.ExceptionInfo[ValidationError]) -> list[tuple[int | str, ...]]:
    return [error["loc"] for error in exc_info.value.errors()]


def test_scores_and_similarities_share_twenty_bins() -> None:
    assert SCORE_HISTOGRAM_BINS == 20


# ── SemanticThreshold ───────────────────────────────────────────────────────


def test_a_threshold_is_derived_unbounded_without_positives_by_default() -> None:
    found = SemanticThreshold(value=0.62, quantile=0.99, negatives=400, positives=0)
    assert (found.positive_recall, found.bounded, found.fallback) == (None, False, False)
    edges = SemanticThreshold(
        value=-1.0, quantile=0.01, negatives=0, positives=3, positive_recall=0.0, fallback=True
    )
    assert (edges.value, edges.positive_recall) == (-1.0, 0.0)
    assert SemanticThreshold(
        value=1.0, quantile=0.99, negatives=0, positives=1, positive_recall=1.0
    )


def test_an_overridden_threshold_is_the_configured_value_alone() -> None:
    found = SemanticThreshold(value=0.7, quantile=0.99, negatives=0, positives=4, overridden=True)
    assert (found.overridden, found.fallback, found.bounded) == (True, False, False)
    assert SemanticThreshold(value=0.7, quantile=0.99, negatives=0, positives=0).overridden is False


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"fallback": True}, id="fallback"),
        pytest.param({"bounded": True}, id="bounded"),
        pytest.param({"negatives": 1}, id="derived-from-negatives"),
    ],
)
def test_an_overridden_threshold_is_never_derived(fields: dict[str, object]) -> None:
    values: dict[str, object] = {
        "value": 0.7,
        "quantile": 0.99,
        "negatives": 0,
        "positives": 0,
        "overridden": True,
    }
    with pytest.raises(ValidationError, match="overridden threshold is neither"):
        SemanticThreshold.model_validate({**values, **fields})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("value", 1.01),
        ("value", -1.01),
        ("quantile", 0.0),
        ("quantile", 1.0),
        ("negatives", -1),
        ("positives", -1),
        ("positive_recall", 1.01),
        ("positive_recall", -0.01),
    ],
)
def test_threshold_bounds(field: str, value: float) -> None:
    fields: dict[str, object] = {"value": 0.6, "quantile": 0.99, "negatives": 300, "positives": 5}
    with pytest.raises(ValidationError) as exc_info:
        SemanticThreshold.model_validate({**fields, field: value})
    assert locs(exc_info) == [(field,)]


# ── AnchorScore and AnchorChoice ────────────────────────────────────────────


def score(**fields: object) -> AnchorScore:
    values: dict[str, object] = {
        "semantic": 0.8,
        "keyword": 0.9,
        "diversity": 1.0,
        "length": 1.0,
        "rank_weight": 1.0,
        "profile_bonus": 0.0,
        "total": 0.915,
        **fields,
    }
    return AnchorScore.model_validate(values)


def test_score_parts_may_sit_on_either_end_of_the_unit_interval() -> None:
    low = score(semantic=0.0, keyword=0.0, diversity=0.0, length=0.0, total=0.0)
    high = score(semantic=1.0, keyword=1.0, profile_bonus=1.0, total=2.0)
    assert (low.total, high.total) == (0.0, 2.0)
    assert score(rank_weight=0.7).rank_weight == 0.7


def test_the_semantic_part_is_empty_without_vectors() -> None:
    assert score(semantic=None).semantic is None
    assert (
        AnchorScore(
            keyword=0.9, diversity=1.0, length=1.0, rank_weight=1.0, profile_bonus=0.0, total=0.9
        ).semantic
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic", 1.01),
        ("keyword", -0.01),
        ("diversity", 1.01),
        ("length", -0.01),
        ("rank_weight", 0.0),
        ("rank_weight", 1.01),
        ("profile_bonus", -0.01),
        ("profile_bonus", 1.01),
        ("total", -0.01),
    ],
)
def test_score_bounds(field: str, value: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        score(**{field: value})
    assert locs(exc_info) == [(field,)]


def anchor_match() -> AnchorMatch:
    return AnchorMatch(
        source_url="example.com/guide",
        target_url="example.com/shoes",
        keyword="trail shoes",
        keyword_rank=1,
        keyword_source=KeywordSource.CLIENT_STRATEGIC,
        rung=AnchorRung.EXACT,
        phrase="trail shoes",
        start=4,
        end=15,
        sentence="Our trail shoes grip.",
        sentence_index=0,
        sentence_start=0,
    )


def choice(**fields: object) -> AnchorChoice:
    values: dict[str, object] = {
        "rank": 1,
        "match": anchor_match(),
        "anchor_type": AnchorType.EXACT,
        "score": score(),
        "context_relevance": 0.7,
        "anchor_target_fit": 0.8,
        **fields,
    }
    return AnchorChoice.model_validate(values)


def test_a_choice_ranks_one_to_three_with_unit_features() -> None:
    assert [choice(rank=rank).rank for rank in (1, 2, 3)] == [1, 2, 3]
    edges = choice(context_relevance=0.0, anchor_target_fit=1.0)
    assert (edges.context_relevance, edges.anchor_target_fit) == (0.0, 1.0)


def test_placement_features_are_empty_without_vectors() -> None:
    empty = choice(context_relevance=None, anchor_target_fit=None)
    assert (empty.context_relevance, empty.anchor_target_fit) == (None, None)
    values = choice().model_dump()
    del values["context_relevance"], values["anchor_target_fit"]
    assert AnchorChoice.model_validate(values).context_relevance is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", 0),
        ("rank", 4),
        ("context_relevance", 1.01),
        ("context_relevance", -0.01),
        ("anchor_target_fit", 1.01),
        ("anchor_target_fit", -0.01),
    ],
)
def test_choice_bounds(field: str, value: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        choice(**{field: value})
    assert locs(exc_info) == [(field,)]


# ── UnanchoredPair ──────────────────────────────────────────────────────────


def unanchored(reason: UnanchoredReason, best_score: float | None = None) -> UnanchoredPair:
    return UnanchoredPair(
        source_url="example.com/guide",
        target_url="example.com/shoes",
        reason=reason,
        best_score=best_score,
    )


def test_a_best_score_goes_with_a_mentioned_topic_only() -> None:
    for reason in (NO_MENTION, NO_KEYWORD, NO_TEXT, NOT_SEARCHED):
        assert unanchored(reason).best_score is None
    assert unanchored(AWKWARD, 0.0).best_score == 0.0


@pytest.mark.parametrize(
    ("reason", "best_score"),
    [
        pytest.param(NO_MENTION, 0.2, id="no-mention-with-score"),
        pytest.param(NO_KEYWORD, 0.2, id="no-keyword-with-score"),
        pytest.param(NO_TEXT, 0.2, id="no-text-with-score"),
        pytest.param(NOT_SEARCHED, 0.2, id="not-searched-with-score"),
        pytest.param(AWKWARD, None, id="no-good-phrase-without-score"),
    ],
)
def test_a_best_score_is_refused_elsewhere_and_required_there(
    reason: UnanchoredReason, best_score: float | None
) -> None:
    with pytest.raises(ValidationError, match="best_score is set when the topic is mentioned"):
        unanchored(reason, best_score)


def test_a_negative_best_score_is_refused() -> None:
    with pytest.raises(ValidationError) as exc_info:
        unanchored(AWKWARD, -0.01)
    assert ("best_score",) in locs(exc_info)


def test_every_reason_has_plain_advice() -> None:
    assert set(UNANCHORED_ADVICE) == set(UnanchoredReason)
    assert all(advice.strip().endswith(".") for advice in UNANCHORED_ADVICE.values())


def test_only_a_fully_searched_source_gap_becomes_a_content_gap_finding() -> None:
    assert {reason: content_gap_finding(reason) for reason in UnanchoredReason} == {
        NO_MENTION: ContentGapFinding.NO_TOPICAL_MENTION,
        AWKWARD: ContentGapFinding.AWKWARD_PHRASING,
        NO_KEYWORD: None,
        NO_TEXT: None,
        NOT_SEARCHED: None,
    }


# ── AnchorSelectionReport ───────────────────────────────────────────────────


def report(**fields: object) -> AnchorSelectionReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "pairs": 10,
        "lexical_pairs": 6,
        "semantic_invocations": 3,
        "semantic_matched": 2,
        "zero_overlap_matches": 1,
        "threshold": SemanticThreshold(
            value=0.62, quantile=0.99, negatives=400, positives=20, positive_recall=0.85
        ),
        "sentences_embedded": 30,
        "sentences_cached": 10,
        "phrases_embedded": 90,
        "phrases_cached": 0,
        "chosen": 7,
        "alternatives": 9,
        "unanchored": {NO_MENTION: 1, AWKWARD: 1, NO_KEYWORD: 1},
        "chosen_types": {
            AnchorType.EXACT: 2,
            AnchorType.PARTIAL: 2,
            AnchorType.NATURAL: 2,
            AnchorType.BRANDED: 1,
        },
        "chosen_ranks": {1: 5, 2: 2},
        "profile": AnchorTypeProfile(),
        "targets": 4,
        "targets_with_anchor": 3,
        "features_filled": 6,
        "score_histogram": (*([0] * 12), 3, 4, *([0] * 6)),
        "semantic_histogram": (*([0] * 13), 2, *([0] * 6)),
        "seconds": 1.5,
        "finished_at": datetime(2026, 9, 28, tzinfo=UTC),
        **fields,
    }
    return AnchorSelectionReport.model_validate(values)


def test_the_selection_report_boundaries_pass() -> None:
    assert report().semantic_skipped_reason is None
    equal = report(
        lexical_pairs=7,
        semantic_matched=3,
        zero_overlap_matches=3,
        alternatives=14,
        targets_with_anchor=4,
        features_filled=7,
        semantic_histogram=(*([0] * 13), 3, *([0] * 6)),
    )
    assert (equal.lexical_pairs + equal.semantic_invocations, equal.alternatives) == (10, 14)
    skipped = report(
        semantic_invocations=0,
        semantic_matched=0,
        zero_overlap_matches=0,
        semantic_histogram=(0,) * 20,
        semantic_skipped_reason="no Voyage key",
        embedding_skipped_reason="no Voyage key",
    )
    assert skipped.semantic_skipped_reason == skipped.embedding_skipped_reason == "no Voyage key"
    assert report().embedding_skipped_reason is None
    empty = report(
        pairs=0,
        lexical_pairs=0,
        semantic_invocations=0,
        semantic_matched=0,
        zero_overlap_matches=0,
        chosen=0,
        alternatives=0,
        unanchored={},
        chosen_types={},
        chosen_ranks={},
        targets=0,
        targets_with_anchor=0,
        features_filled=0,
        score_histogram=(0,) * 20,
        semantic_histogram=(0,) * 20,
    )
    assert empty.pairs == 0


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param({"lexical_pairs": 8}, "cannot exceed the pairs", id="lexical-and-semantic"),
        pytest.param(
            {"zero_overlap_matches": 3}, "more zero-overlap matches", id="zero-overlap-over-matched"
        ),
        pytest.param(
            {
                "semantic_matched": 4,
                "zero_overlap_matches": 0,
                "semantic_histogram": (*([0] * 13), 4, *([0] * 6)),
            },
            "more semantic matches than invocations",
            id="matched-over-invocations",
        ),
        pytest.param(
            {"unanchored": {NO_MENTION: 2}}, "an anchor or an unanchored reason", id="unaccounted"
        ),
        pytest.param(
            {"chosen_types": {AnchorType.EXACT: 7, AnchorType.NATURAL: 1}},
            "by type and by keyword rank",
            id="types-sum",
        ),
        pytest.param({"chosen_ranks": {1: 5, 2: 1}}, "by type and by keyword rank", id="ranks-sum"),
        pytest.param({"targets_with_anchor": 5}, "targets with an anchor", id="targets"),
        pytest.param({"features_filled": 8}, "features filled", id="features-over-chosen"),
        pytest.param({"alternatives": 15}, "at most two alternatives", id="alternatives"),
        pytest.param(
            {"score_histogram": (*([0] * 12), 3, 4, *([0] * 5))}, "score histogram", id="score-bins"
        ),
        pytest.param(
            {"score_histogram": (*([0] * 12), 3, 3, *([0] * 6))}, "score histogram", id="score-sum"
        ),
        pytest.param(
            {"semantic_histogram": (*([0] * 13), 2, *([0] * 5))},
            "semantic histogram",
            id="semantic-bins",
        ),
        pytest.param(
            {"semantic_histogram": (*([0] * 13), 1, *([0] * 6))},
            "semantic histogram",
            id="semantic-sum",
        ),
        pytest.param(
            {
                "unanchored": {NO_MENTION: 4, AWKWARD: -1},
            },
            "cannot be negative",
            id="negative-gap-count",
        ),
        pytest.param({"chosen_ranks": {0: 5, 2: 2}}, "keyword ranks start at 1", id="rank-zero"),
    ],
)
def test_selection_report_counts_must_be_consistent(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        report(**fields)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", ""),
        ("pairs", -1),
        ("semantic_invocations", -1),
        ("sentences_embedded", -1),
        ("sentences_cached", -1),
        ("phrases_embedded", -1),
        ("phrases_cached", -1),
        ("semantic_rejected_identifier", -1),
        ("semantic_rejected_other_target", -1),
        ("seconds", -0.1),
    ],
)
def test_selection_report_field_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        report(**{field: value})
    assert (field,) in locs(exc_info)


def test_the_selection_report_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        report(chosenAnchors=7)
    assert [error["type"] for error in exc_info.value.errors()] == ["extra_forbidden"]
