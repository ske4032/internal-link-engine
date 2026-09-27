"""Every validator of the pair signal models, with the boundary that passes beside the one
that is refused."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from linking_engine.models import (
    ClusterAgreement,
    GscQuery,
    PageSignals,
    PairSignals,
    SignalReport,
    StrategicKeyword,
)

ZERO = dict.fromkeys(ClusterAgreement, 0)


def only_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> dict[str, object]:
    errors = exc_info.value.errors()
    assert len(errors) == 1, f"expected exactly one validation error, got {errors}"
    return dict(errors[0])


def page(**fields: object) -> PageSignals:
    values: dict[str, object] = {
        "url": "example.com/a",
        "queries": frozenset({"trail shoes"}),
        "keywords": frozenset({"trail shoes", "rain jacket"}),
        "keyword_gap": 1,
        "link_community_id": 0,
        "keyword_community_id": 2,
        "content_community_id": None,
        "hub_id": -1,
        **fields,
    }
    return PageSignals.model_validate(values)


def pair(**fields: object) -> PairSignals:
    values: dict[str, object] = {
        "query_overlap": 0.5,
        "keyword_overlap": 0.0,
        "same_link_community": True,
        "same_keyword_community": None,
        "same_content_community": False,
        "same_hub": False,
        "cluster_agreement": ClusterAgreement.SAME_TOPIC_OTHER_LINKS,
        "content_agreement": ClusterAgreement.UNKNOWN_TOPIC,
        **fields,
    }
    return PairSignals.model_validate(values)


def report(**fields: object) -> SignalReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "pages": 4,
        "pages_with_queries": 2,
        "pages_with_keywords": 3,
        "pages_with_gap": 1,
        "unmatched_query_urls": 5,
        "pairs": 3,
        "query_overlap_pairs": 1,
        "query_overlap_mean": 0.25,
        "keyword_overlap_pairs": 2,
        "keyword_overlap_mean": 0.5,
        "same_hub_pairs": 1,
        "noise_pairs": 2,
        "cluster_agreement": {**ZERO, ClusterAgreement.SAME_TOPIC_OTHER_LINKS: 3},
        "content_agreement": {
            **ZERO,
            ClusterAgreement.UNKNOWN_TOPIC: 2,
            ClusterAgreement.OTHER_TOPIC_SAME_LINKS: 1,
        },
        "seconds": 0.01,
        **fields,
    }
    return SignalReport.model_validate(values)


def test_the_valid_fixtures_build() -> None:
    """Guards every rejection below: each changes one field of a payload that is valid."""
    assert page().keyword_gap == 1
    assert pair().query_overlap == 0.5
    assert report().pairs == 3


@pytest.mark.parametrize("model", [page(), pair(), report()], ids=lambda m: type(m).__name__)
def test_models_are_frozen_and_forbid_extra_fields(model: BaseModel) -> None:
    field = next(iter(type(model).model_fields))
    with pytest.raises(ValidationError, match="frozen"):
        setattr(model, field, getattr(model, field))
    with pytest.raises(ValidationError) as exc_info:
        type(model).model_validate({**model.model_dump(), "surprise": 1})
    assert only_error(exc_info)["type"] == "extra_forbidden"


def test_the_agreement_values_are_the_five_buckets() -> None:
    assert [a.value for a in ClusterAgreement] == [
        "SAME_TOPIC_SAME_LINKS",
        "SAME_TOPIC_OTHER_LINKS",
        "OTHER_TOPIC_SAME_LINKS",
        "OTHER_TOPIC_OTHER_LINKS",
        "UNKNOWN_TOPIC",
    ]


# ── PageSignals ─────────────────────────────────────────────────────────────


def test_the_gap_may_equal_the_targeted_keywords() -> None:
    assert page(keyword_gap=2).keyword_gap == 2


def test_a_gap_beyond_the_targeted_keywords_is_refused() -> None:
    with pytest.raises(ValidationError, match="keyword_gap cannot exceed"):
        page(keyword_gap=3)


def test_an_unknown_gap_is_none_not_zero() -> None:
    assert page(keyword_gap=None).keyword_gap is None


def test_a_page_without_queries_or_keywords_is_valid() -> None:
    bare = page(queries=frozenset(), keywords=frozenset(), keyword_gap=0, hub_id=None)
    assert (bare.queries, bare.keywords, bare.hub_id) == (frozenset(), frozenset(), None)


def test_noise_is_a_hub_label_but_nothing_below_it() -> None:
    assert page(hub_id=-1).hub_id == -1
    with pytest.raises(ValidationError) as exc_info:
        page(hub_id=-2)
    assert only_error(exc_info)["loc"] == ("hub_id",)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("url", "", "string_too_short"),
        ("keyword_gap", -1, "greater_than_equal"),
        ("link_community_id", -1, "greater_than_equal"),
        ("keyword_community_id", -1, "greater_than_equal"),
        ("content_community_id", -1, "greater_than_equal"),
    ],
)
def test_page_signal_field_bounds(field: str, value: object, error: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        page(**{field: value})
    assert (only_error(exc_info)["loc"], only_error(exc_info)["type"]) == ((field,), error)


# ── PairSignals ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("field", ["query_overlap", "keyword_overlap"])
@pytest.mark.parametrize(
    ("value", "error"), [(-0.01, "greater_than_equal"), (1.01, "less_than_equal")]
)
def test_overlaps_are_shares(field: str, value: float, error: str) -> None:
    assert pair(**{field: 0.0}).model_dump()[field] == 0.0
    assert pair(**{field: 1.0}).model_dump()[field] == 1.0
    with pytest.raises(ValidationError) as exc_info:
        pair(**{field: value})
    assert (only_error(exc_info)["loc"], only_error(exc_info)["type"]) == ((field,), error)


def test_an_agreement_outside_the_buckets_is_refused() -> None:
    with pytest.raises(ValidationError) as exc_info:
        pair(cluster_agreement="SAME_TOPIC")
    assert only_error(exc_info)["type"] == "enum"


# ── SignalReport ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("field", ["pages_with_queries", "pages_with_keywords", "pages_with_gap"])
def test_page_counts_cannot_exceed_the_pages(field: str) -> None:
    assert report(**{field: 4}).pages == 4
    with pytest.raises(ValidationError, match="page counts cannot exceed the pages"):
        report(**{field: 5})


@pytest.mark.parametrize(
    "field", ["query_overlap_pairs", "keyword_overlap_pairs", "same_hub_pairs", "noise_pairs"]
)
def test_pair_counts_cannot_exceed_the_pairs(field: str) -> None:
    assert report(**{field: 3}).pairs == 3
    with pytest.raises(ValidationError, match="pair counts cannot exceed the pairs"):
        report(**{field: 4})


@pytest.mark.parametrize("field", ["query_overlap_mean", "keyword_overlap_mean"])
def test_means_are_required_when_there_are_pairs(field: str) -> None:
    with pytest.raises(ValidationError, match="set exactly when there are pairs"):
        report(**{field: None})


@pytest.mark.parametrize("field", ["query_overlap_mean", "keyword_overlap_mean"])
def test_means_are_refused_without_pairs(field: str) -> None:
    empty = {
        "pairs": 0,
        "query_overlap_pairs": 0,
        "keyword_overlap_pairs": 0,
        "same_hub_pairs": 0,
        "noise_pairs": 0,
        "query_overlap_mean": None,
        "keyword_overlap_mean": None,
        "cluster_agreement": ZERO,
        "content_agreement": ZERO,
    }
    assert report(**empty).query_overlap_mean is None
    with pytest.raises(ValidationError, match="set exactly when there are pairs"):
        report(**{**empty, field: 0.0})


@pytest.mark.parametrize("name", ["cluster_agreement", "content_agreement"])
@pytest.mark.parametrize(
    "counts",
    [
        pytest.param({ClusterAgreement.UNKNOWN_TOPIC: 2}, id="short"),
        pytest.param({ClusterAgreement.UNKNOWN_TOPIC: 4}, id="over"),
        pytest.param(
            {ClusterAgreement.UNKNOWN_TOPIC: 4, ClusterAgreement.SAME_TOPIC_SAME_LINKS: -1},
            id="negative",
        ),
    ],
)
def test_agreement_counts_must_add_up_to_the_pairs(
    name: str, counts: dict[ClusterAgreement, int]
) -> None:
    with pytest.raises(ValidationError, match=f"{name} counts must add up to the pairs"):
        report(**{name: {**ZERO, **counts}})


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("tenant_id", "", "string_too_short"),
        ("unmatched_query_urls", -1, "greater_than_equal"),
        ("query_overlap_mean", 1.5, "less_than_equal"),
        ("keyword_overlap_mean", -0.5, "greater_than_equal"),
        ("seconds", -1.0, "greater_than_equal"),
    ],
)
def test_report_field_bounds(field: str, value: object, error: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        report(**{field: value})
    assert (only_error(exc_info)["loc"], only_error(exc_info)["type"]) == ((field,), error)


# ── GscQuery ───────────────────────────────────────────────────────────────


def test_a_gsc_row_keeps_url_and_query_verbatim() -> None:
    row = GscQuery.model_validate({"url": "example.com/a", "query": " Trail Shoes "})
    assert (row.url, row.query) == ("example.com/a", " Trail Shoes ")


@pytest.mark.parametrize(
    ("data", "loc"),
    [
        pytest.param({"url": "", "query": "q"}, ("url",), id="empty-url"),
        pytest.param({"url": 7, "query": "q"}, ("url",), id="numeric-url"),
        pytest.param({"url": "example.com/a", "query": 7}, ("query",), id="numeric-query"),
        pytest.param({"url": "example.com/a", "query": None}, ("query",), id="null-query"),
        pytest.param({"url": "example.com/a"}, ("query",), id="missing-query"),
        pytest.param({"url": "example.com/a", "query": "q", "clicks": 1}, ("clicks",), id="extra"),
    ],
)
def test_a_gsc_row_needs_string_url_and_query(data: dict[str, object], loc: tuple[str]) -> None:
    with pytest.raises(ValidationError) as exc_info:
        GscQuery.model_validate(data)
    assert only_error(exc_info)["loc"] == loc


# ── StrategicKeyword ────────────────────────────────────────────────────────


def test_a_strategic_keyword_defaults_to_no_priority_and_not_primary() -> None:
    row = StrategicKeyword(url="example.com/a", keyword="Trail Shoes", language="en")
    assert (row.keyword, row.priority, row.is_primary) == ("Trail Shoes", None, False)


@pytest.mark.parametrize("priority", [1, 5])
def test_priority_runs_from_one_to_five(priority: int) -> None:
    row = StrategicKeyword(url="example.com/a", keyword="tents", language="en", priority=priority)
    assert row.priority == priority


@pytest.mark.parametrize(
    ("fields", "loc"),
    [
        pytest.param({"url": ""}, ("url",), id="empty-url"),
        pytest.param({"keyword": ""}, ("keyword",), id="empty-keyword"),
        pytest.param({"language": ""}, ("language",), id="empty-language"),
        pytest.param({"priority": 0}, ("priority",), id="priority-zero"),
        pytest.param({"priority": 6}, ("priority",), id="priority-six"),
        pytest.param({"group": "core"}, ("group",), id="extra"),
    ],
)
def test_a_strategic_keyword_is_validated(fields: dict[str, object], loc: tuple[str]) -> None:
    data: dict[str, object] = {"url": "example.com/a", "keyword": "tents", "language": "en"}
    with pytest.raises(ValidationError) as exc_info:
        StrategicKeyword.model_validate({**data, **fields})
    assert only_error(exc_info)["loc"] == loc
