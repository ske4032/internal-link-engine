"""Validators of the #18 hub-bridge models: each rejection beside the boundary that passes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from linking_engine.models import BridgeLink, BridgeReason, BridgeReport, HubPair, KeywordRung

TREE, NEAREST, GAP = BridgeReason.SPANNING_TREE, BridgeReason.NEAREST_HUB, BridgeReason.BRIDGE_GAP


def only_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> dict[str, object]:
    errors = exc_info.value.errors()
    assert len(errors) == 1, f"expected exactly one validation error, got {errors}"
    return dict(errors[0])


# ── HubPair ─────────────────────────────────────────────────────────────────


def pair(**fields: object) -> HubPair:
    values: dict[str, object] = {
        "language": "en",
        "hub_a": 1,
        "hub_b": 4,
        "size_a": 10,
        "size_b": 20,
        "pages_ab": 2,
        "pages_ba": 0,
        "link_density": 0.01,
        "centroid_cosine": 0.8,
        "query_jaccard": 0.3,
        "bridge_gap": -0.18,
        "reasons": (TREE, NEAREST),
        **fields,
    }
    return HubPair.model_validate(values)


def test_a_valid_pair_builds_with_a_negative_gap_and_no_gsc() -> None:
    assert pair().bridge_gap == -0.18
    assert pair(query_jaccard=None, language=None, reasons=()).query_jaccard is None


@pytest.mark.parametrize(("hub_a", "hub_b"), [(4, 4), (5, 4)])
def test_hub_a_is_the_lower_id(hub_a: int, hub_b: int) -> None:
    with pytest.raises(ValidationError, match="hub_a must be lower than hub_b"):
        pair(hub_a=hub_a, hub_b=hub_b)


def test_no_more_linking_pages_than_hub_pages() -> None:
    assert pair(pages_ab=10, pages_ba=20).pages_ba == 20
    for fields in ({"pages_ab": 11}, {"pages_ba": 21}):
        with pytest.raises(ValidationError, match="more linking pages than hub pages"):
            pair(**fields)


def test_a_reason_is_recorded_once() -> None:
    with pytest.raises(ValidationError, match="duplicate reasons"):
        pair(reasons=(TREE, GAP, TREE))


def test_shared_queries_are_distinct_and_need_gsc_data() -> None:
    assert pair(shared_queries=("trail shoes", "tents")).shared_queries == ("trail shoes", "tents")
    assert pair(query_jaccard=None).shared_queries == ()
    with pytest.raises(ValidationError, match="duplicate shared queries"):
        pair(shared_queries=("tents", "trail shoes", "tents"))
    with pytest.raises(ValidationError, match="shared queries need GSC data"):
        pair(query_jaccard=None, shared_queries=("tents",))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hub_a", -1),
        ("size_a", 0),
        ("size_b", 0),
        ("pages_ab", -1),
        ("link_density", -0.1),
        ("centroid_cosine", 1.01),
        ("centroid_cosine", -1.01),
        ("query_jaccard", 1.1),
    ],
)
def test_hub_pair_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        pair(**{field: value})
    assert only_error(exc_info)["loc"] == (field,)


# ── BridgeLink ──────────────────────────────────────────────────────────────


def link(**fields: object) -> BridgeLink:
    values: dict[str, object] = {
        "language": "en",
        "hub_from": 1,
        "hub_to": 4,
        "slot": 1,
        "rank": 1,
        "source_url": "example.com/a",
        "target_url": "example.com/b",
        "similarity": 0.7,
        "source_page_rank_percentile": 0.9,
        "anchor_keyword": "trail shoes",
        "anchor_rung": KeywordRung.STRATEGIC,
        "reasons": (TREE,),
        **fields,
    }
    return BridgeLink.model_validate(values)


def test_a_link_joins_two_hubs_and_two_pages() -> None:
    assert link(hub_from=4, hub_to=1).hub_from == 4
    with pytest.raises(ValidationError, match="two different hubs"):
        link(hub_to=1)
    with pytest.raises(ValidationError, match="cannot link to itself"):
        link(target_url="example.com/a")


def test_an_anchor_keyword_comes_with_its_rung() -> None:
    assert link(anchor_keyword=None, anchor_rung=None).anchor_keyword is None
    for fields in ({"anchor_keyword": None}, {"anchor_rung": None}):
        with pytest.raises(ValidationError, match="set together"):
            link(**fields)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_rank_one_is_the_proposal_and_two_or_three_its_alternatives(rank: int) -> None:
    assert link(rank=rank).rank == rank


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", 0),
        ("rank", 4),
        ("slot", 0),
        ("similarity", 1.01),
        ("source_page_rank_percentile", 1.0),
        ("reasons", ()),
        ("source_url", ""),
        ("hub_to", -1),
    ],
)
def test_bridge_link_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        link(**{field: value})
    assert only_error(exc_info)["loc"] == (field,)


# ── BridgeReport ────────────────────────────────────────────────────────────


def report(**fields: object) -> BridgeReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "floor_share": 0.05,
        "hubs": 5,
        "noise_pages": 3,
        "hub_pairs": 10,
        "components_before": 3,
        "components_after": 1,
        "directions_below_floor": 4,
        "links_needed": 6,
        "bridge_links": 5,
        "alternatives": 8,
        "directions_short": 1,
        "by_reason": {TREE: 4, NEAREST: 5, GAP: 3},
        "gsc_used": True,
        "seconds": 0.2,
        "finished_at": datetime(2026, 9, 28, tzinfo=UTC),
        **fields,
    }
    return BridgeReport.model_validate(values)


def test_the_report_boundaries_pass() -> None:
    assert report(components_after=3).components_after == 3
    assert report(bridge_links=6).bridge_links == 6
    assert report(directions_short=4).directions_short == 4
    assert report(by_reason={}).by_reason == {}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"components_after": 4}, "cannot split the hub graph"),
        ({"bridge_links": 7}, "more bridge links than needed"),
        ({"directions_short": 5}, "more short directions"),
        ({"by_reason": {TREE: -1}}, "reason counts cannot be negative"),
    ],
)
def test_report_counts_must_be_consistent(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        report(**fields)


@pytest.mark.parametrize("share", [0.0, 1.0, -0.05])
def test_the_floor_share_is_strictly_between_zero_and_one(share: float) -> None:
    with pytest.raises(ValidationError) as exc_info:
        report(floor_share=share)
    assert only_error(exc_info)["loc"] == ("floor_share",)
