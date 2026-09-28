"""Validators of the #15/#19 models: each rejection beside the boundary that passes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from factories import PAIR_SPEC
from pydantic import ValidationError

from linking_engine.models import (
    CtrCurve,
    FeatureReport,
    GscMetrics,
    GscQueryStats,
    KeywordReport,
    KeywordRung,
    KeywordSource,
    KeywordTarget,
    LanguageRules,
    PageStructure,
    PairFeatures,
    ResolvedKeyword,
)

WHEN = datetime(2026, 9, 28, tzinfo=UTC)


def only_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> dict[str, object]:
    errors = exc_info.value.errors()
    assert len(errors) == 1, f"expected exactly one validation error, got {errors}"
    return dict(errors[0])


# ── ResolvedKeyword and KeywordTarget ───────────────────────────────────────


@pytest.mark.parametrize("rung", list(KeywordRung))
def test_opportunity_value_is_set_exactly_for_the_gsc_rung(rung: KeywordRung) -> None:
    gsc = rung is KeywordRung.GSC
    kept = ResolvedKeyword(
        url="example.com/a",
        text="tents",
        language="en",
        rung=rung,
        opportunity_value=12.5 if gsc else None,
    )
    assert kept.rung is rung
    with pytest.raises(ValidationError, match="exactly for the GSC rung"):
        ResolvedKeyword(
            url="example.com/a",
            text="tents",
            language="en",
            rung=rung,
            opportunity_value=None if gsc else 0.0,
        )


@pytest.mark.parametrize(
    ("fields", "loc"),
    [
        ({"language": "e"}, ("language",)),
        ({"text": ""}, ("text",)),
        ({"opportunity_value": -1.0, "rung": KeywordRung.GSC}, ("opportunity_value",)),
    ],
)
def test_resolved_keyword_bounds(fields: dict[str, object], loc: tuple[str]) -> None:
    values: dict[str, object] = {
        "url": "example.com/a",
        "text": "tents",
        "language": "en",
        "rung": KeywordRung.H1,
        **fields,
    }
    with pytest.raises(ValidationError) as exc_info:
        ResolvedKeyword.model_validate(values)
    assert only_error(exc_info)["loc"] == loc


def target(**fields: object) -> KeywordTarget:
    values: dict[str, object] = {
        "url": "example.com/a",
        "text": "tents",
        "language": "en",
        "source": KeywordSource.CLIENT_STRATEGIC,
        **fields,
    }
    return KeywordTarget.model_validate(values)


def test_a_resolved_target_defaults_to_rank_one() -> None:
    chosen = target(rung=KeywordRung.STRATEGIC)
    assert (chosen.resolved, chosen.rank) == (True, 1)
    assert (target().resolved, target().rank) == (False, None)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param(
            {"rung": KeywordRung.GSC, "rank": 2}, "resolved keyword has rank 1", id="resolved-2"
        ),
        pytest.param({"rank": 1}, "rank 1 is reserved", id="secondary-1"),
    ],
)
def test_rank_one_is_the_resolved_keyword_and_only_it(
    fields: dict[str, object], message: str
) -> None:
    assert target(rank=2).rank == 2
    assert target(rung=KeywordRung.H1, rank=1).rank == 1
    with pytest.raises(ValidationError, match=message):
        target(**fields)


def test_a_rank_starts_at_one() -> None:
    with pytest.raises(ValidationError) as exc_info:
        target(rank=0)
    assert only_error(exc_info)["loc"] == ("rank",)


@pytest.mark.parametrize("priority", [0, 6])
def test_a_keyword_target_priority_is_one_to_five(priority: int) -> None:
    for valid in (1, 5):
        assert (
            KeywordTarget(
                url="example.com/a",
                text="tents",
                language="en",
                source=KeywordSource.CLIENT_STRATEGIC,
                priority=valid,
            ).priority
            == valid
        )
    with pytest.raises(ValidationError) as exc_info:
        KeywordTarget(
            url="example.com/a",
            text="tents",
            language="en",
            source=KeywordSource.CLIENT_STRATEGIC,
            priority=priority,
        )
    assert only_error(exc_info)["loc"] == ("priority",)


# ── KeywordReport ───────────────────────────────────────────────────────────


def keyword_report(**fields: object) -> KeywordReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "pages": 4,
        "resolved": 3,
        "by_rung": {KeywordRung.STRATEGIC: 1, KeywordRung.H1: 2},
        "gsc_enabled": False,
        "gsc_rows": 0,
        "gsc_rejected": 0,
        "fallbacks_rejected": {"h1_repeated": 2},
        "edges_written": {KeywordSource.CLIENT_STRATEGIC: 1},
        "stale_edges_deleted": {},
        "skipped_rows": 0,
        "by_language": {"en": 3, "de": 1},
        "seconds": 0.1,
        "finished_at": WHEN,
        **fields,
    }
    return KeywordReport.model_validate(values)


def test_the_keyword_report_fixture_is_valid() -> None:
    assert keyword_report().resolved == 3


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"resolved": 5, "by_rung": {KeywordRung.H1: 5}},
            "more resolved pages than pages",
        ),
        ({"by_rung": {KeywordRung.H1: 2}}, "by_rung must add up"),
        ({"by_rung": {KeywordRung.H1: 4, KeywordRung.TITLE: -1}}, "by_rung must add up"),
        ({"fallbacks_rejected": {"h1_generic": -1}}, "rejection counts cannot be negative"),
        ({"by_language": {"en": 3}}, "by_language must add up"),
    ],
)
def test_keyword_report_counts_must_add_up(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        keyword_report(**fields)


def test_only_resolved_pages_have_secondary_keywords() -> None:
    assert (
        keyword_report(secondary_keywords=5, pages_with_secondaries=3).pages_with_secondaries == 3
    )
    with pytest.raises(ValidationError, match="only resolved pages have secondary keywords"):
        keyword_report(secondary_keywords=5, pages_with_secondaries=4)


def test_every_page_may_be_resolved() -> None:
    full = keyword_report(resolved=4, by_rung={KeywordRung.H1: 4})
    assert full.resolved == full.pages


# ── GSC rows, metrics and the CTR curve ─────────────────────────────────────


@pytest.mark.parametrize(
    ("fields", "loc"),
    [
        ({"position": 0.99}, ("position",)),
        ({"impressions": -1}, ("impressions",)),
        ({"clicks": -1}, ("clicks",)),
        ({"url": ""}, ("url",)),
    ],
)
def test_gsc_query_stats_bounds(fields: dict[str, object], loc: tuple[str]) -> None:
    values: dict[str, object] = {
        "url": "example.com/a",
        "query": "tents",
        "impressions": 10,
        "position": 1.0,
        **fields,
    }
    boundary = GscQueryStats(url="example.com/a", query="", impressions=0, clicks=0, position=1.0)
    assert (boundary.position, boundary.impressions) == (1.0, 0)
    with pytest.raises(ValidationError) as exc_info:
        GscQueryStats.model_validate(values)
    assert only_error(exc_info)["loc"] == loc


@pytest.mark.parametrize(
    ("fields", "loc"),
    [
        ({"avg_position": 0.5}, ("avg_position",)),
        ({"impressions_28d": -1}, ("impressions_28d",)),
        ({"clicks_28d": -1}, ("clicks_28d",)),
        ({"query_count": -1}, ("query_count",)),
    ],
)
def test_gsc_metrics_bounds(fields: dict[str, object], loc: tuple[str]) -> None:
    values: dict[str, object] = {
        "url": "example.com/a",
        "impressions_28d": 0,
        "clicks_28d": 0,
        "avg_position": None,
        "query_count": 0,
    }
    assert GscMetrics.model_validate(values).avg_position is None
    with pytest.raises(ValidationError) as exc_info:
        GscMetrics.model_validate({**values, **fields})
    assert only_error(exc_info)["loc"] == loc


def curve(*ctr: float) -> CtrCurve:
    return CtrCurve(ctr=ctr, rows=100, impressions=10_000)


def test_a_flat_or_falling_curve_within_zero_and_one_is_valid() -> None:
    assert curve(1.0, 0.5, 0.5, 0.0).ctr == (1.0, 0.5, 0.5, 0.0)


@pytest.mark.parametrize(
    ("ctr", "message"),
    [
        ((0.3, 0.31), "must not increase"),
        ((1.01, 0.5), r"shares in \[0, 1\]"),
        ((0.3, -0.01), r"shares in \[0, 1\]"),
    ],
)
def test_a_rising_or_out_of_range_curve_is_refused(ctr: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        curve(*ctr)


def test_a_curve_has_at_least_one_position() -> None:
    with pytest.raises(ValidationError) as exc_info:
        CtrCurve(ctr=(), rows=100, impressions=10_000)
    assert only_error(exc_info)["loc"] == ("ctr",)


# ── LanguageRules ───────────────────────────────────────────────────────────


def test_language_rules_default_to_english_without_prefixes() -> None:
    assert LanguageRules() == LanguageRules(default_language="en", prefixes=())
    assert LanguageRules(prefixes=(("/de/", "de"), ("/de/at/", "de"))).prefixes[1][0] == "/de/at/"


@pytest.mark.parametrize(
    "prefixes",
    [
        pytest.param((("/de/", "de"), ("/de/", "fr")), id="duplicate"),
        pytest.param((("de/", "de"),), id="no-leading-slash"),
        pytest.param((("/de/", "d"),), id="short-language"),
    ],
)
def test_malformed_language_prefixes_are_refused(prefixes: tuple[tuple[str, str], ...]) -> None:
    with pytest.raises(ValidationError, match="prefix"):
        LanguageRules(prefixes=prefixes)


def test_a_default_language_is_at_least_two_letters() -> None:
    with pytest.raises(ValidationError) as exc_info:
        LanguageRules(default_language="e")
    assert only_error(exc_info)["loc"] == ("default_language",)


# ── PageStructure and PairFeatures ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("fields", "loc"),
    [
        ({"hub_id": -2}, ("hub_id",)),
        ({"page_rank_percentile": 1.0}, ("page_rank_percentile",)),
        ({"inbound": -1}, ("inbound",)),
        ({"crawl_depth": -1}, ("crawl_depth",)),
    ],
)
def test_page_structure_bounds(fields: dict[str, object], loc: tuple[str]) -> None:
    base: dict[str, object] = {"url": "example.com/a", "inbound": 0, "outbound": 0}
    assert PageStructure.model_validate({**base, "hub_id": -1, "page_rank_percentile": 0.99})
    with pytest.raises(ValidationError) as exc_info:
        PageStructure.model_validate({**base, **fields})
    assert only_error(exc_info)["loc"] == loc


def test_a_pair_never_links_a_page_to_itself() -> None:
    with pytest.raises(ValidationError, match="its own source"):
        PairFeatures.model_validate(
            PAIR_SPEC.kwargs_with(target_url=PAIR_SPEC.kwargs["source_url"])
        )


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"target_hub_size": None}, id="coverage-without-hub"),
        pytest.param({"target_hub_coverage": None}, id="hub-without-coverage"),
    ],
)
def test_hub_coverage_is_set_exactly_when_the_target_is_in_a_hub(fields: dict[str, object]) -> None:
    outside = PairFeatures.model_validate(
        PAIR_SPEC.kwargs_with(target_hub_size=None, target_hub_coverage=None)
    )
    assert outside.target_hub_coverage is None
    with pytest.raises(ValidationError, match="hub coverage is set exactly"):
        PairFeatures.model_validate(PAIR_SPEC.kwargs_with(**fields))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_link_equity_share", 0.0),
        ("target_position_band", 5),
        ("target_ctr_gap", 1.5),
        ("target_hub_size", 0),
        ("target_page_rank_percentile", 1.0),
    ],
)
def test_pair_feature_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        PairFeatures.model_validate(PAIR_SPEC.kwargs_with(**{field: value}))
    assert only_error(exc_info)["loc"] == (field,)


# ── FeatureReport ───────────────────────────────────────────────────────────


def feature_report(**fields: object) -> FeatureReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "pairs": 2,
        "columns": ("a", "b"),
        "chunks": 1,
        "all_null_columns": ("b",),
        "constant_columns": ("a",),
        "null_share": {"a": 0.0, "b": 1.0},
        "has_gsc_data_share": 0.5,
        "cache_key": "k",
        "cache_hit": False,
        "seconds": 0.1,
        "finished_at": WHEN,
        **fields,
    }
    return FeatureReport.model_validate(values)


def test_the_feature_report_fixture_is_valid() -> None:
    assert feature_report().columns == ("a", "b")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"columns": ("a", "a")}, "duplicate columns"),
        ({"all_null_columns": ("c",)}, "unknown columns"),
        ({"constant_columns": ("c",)}, "unknown columns"),
        ({"null_share": {"c": 0.5}}, "unknown columns"),
        ({"null_share": {"a": 1.5}}, r"null shares are in \[0, 1\]"),
    ],
)
def test_feature_report_columns_must_be_known_and_shares_bounded(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        feature_report(**fields)


def test_a_feature_report_needs_a_column() -> None:
    with pytest.raises(ValidationError) as exc_info:
        feature_report(columns=(), all_null_columns=(), constant_columns=(), null_share={})
    assert only_error(exc_info)["loc"] == ("columns",)
