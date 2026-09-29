"""Every validator of the candidate retrieval models, with the boundary that passes beside
the one that is refused, so a validator that refuses everything cannot pass either."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from linking_engine.models import (
    CandidateReport,
    CandidateSet,
    CandidateTarget,
    TargetCandidates,
    TargetSelection,
)

URL = "example.com/target"


def only_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> dict[str, object]:
    errors = exc_info.value.errors()
    assert len(errors) == 1, f"expected exactly one validation error, got {errors}"
    return dict(errors[0])


def target(url: str = "example.com/a", *, assumed: bool = False) -> CandidateTarget:
    return CandidateTarget(url=url, indexable_assumed=assumed)


def selection(**fields: object) -> TargetSelection:
    values: dict[str, object] = {
        "crawled_pages": 4,
        "not_indexable": 1,
        "without_vector": 1,
        "targets": (target("example.com/a"), target("example.com/b")),
        **fields,
    }
    return TargetSelection.model_validate(values)


def candidates(**fields: object) -> TargetCandidates:
    values: dict[str, object] = {
        "target_url": URL,
        "sources": ("example.com/a", "example.com/b"),
        "similarities": (0.9, 0.5),
        "eligible": 3,
        "linked": 2,
        "linked_nearer": 1,
        **fields,
    }
    return TargetCandidates.model_validate(values)


REPORT: dict[str, object] = {
    "tenant_id": "acme",
    "index": "page_content",
    "per_target": 2,
    "chunk_size": 512,
    "crawled_pages": 5,
    "not_indexable": 1,
    "without_vector": 0,
    "targets": 3,
    "indexable_assumed": 1,
    "source_pages": 4,
    "candidates": 3,
    "full_targets": 1,
    "short_targets": 1,
    "empty_targets": 1,
    "min_per_target": 0,
    "median_per_target": 1.0,
    "max_per_target": 2,
    "linked_pairs": 4,
    "linked_nearer": 2,
    "drop_rate": 0.4,
    "load_seconds": 0.25,
    "search_seconds": 0.125,
    "seconds": 0.5,
    "finished_at": datetime(2026, 9, 27, tzinfo=UTC),
}


def report(**fields: object) -> CandidateReport:
    return CandidateReport.model_validate({**REPORT, **fields})


def candidate_set(**fields: object) -> CandidateSet:
    values: dict[str, object] = {
        "report": report(),
        "targets": (
            candidates(target_url="example.com/full"),
            candidates(
                target_url="example.com/short",
                sources=("example.com/a",),
                similarities=(0.9,),
                eligible=1,
            ),
            candidates(
                target_url="example.com/empty",
                sources=(),
                similarities=(),
                eligible=0,
                linked=0,
                linked_nearer=0,
            ),
        ),
        **fields,
    }
    return CandidateSet.model_validate(values)


def test_the_valid_fixtures_build() -> None:
    """Guards every rejection below: each changes one field of a payload that is valid."""
    assert selection().targets[1].url == "example.com/b"
    assert candidates().sources == ("example.com/a", "example.com/b")
    assert report().drop_rate == 0.4
    assert candidate_set().report.candidates == 3


@pytest.mark.parametrize(
    "model",
    [target(), selection(), candidates(), report(), candidate_set()],
    ids=lambda m: type(m).__name__,
)
def test_models_are_frozen_and_forbid_extra_fields(model: BaseModel) -> None:
    field = next(iter(type(model).model_fields))
    with pytest.raises(ValidationError, match="frozen"):
        setattr(model, field, getattr(model, field))
    with pytest.raises(ValidationError) as exc_info:
        type(model).model_validate({**model.model_dump(), "surprise": 1})
    assert only_error(exc_info)["type"] == "extra_forbidden"


# ── CandidateTarget ─────────────────────────────────────────────────────────


def test_a_target_needs_a_url() -> None:
    with pytest.raises(ValidationError) as exc_info:
        target("")
    assert only_error(exc_info)["loc"] == ("url",)


def test_a_bare_path_url_is_kept_verbatim() -> None:
    assert target("/pricing").url == "/pricing"


# ── TargetSelection ─────────────────────────────────────────────────────────


def test_duplicate_target_urls_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate target urls"):
        selection(targets=(target("example.com/a"), target("example.com/a", assumed=True)))


def test_targets_and_exclusions_add_up_to_the_crawled_pages() -> None:
    assert selection(crawled_pages=4).crawled_pages == 4


@pytest.mark.parametrize("crawled_pages", [3, 5])
def test_targets_and_exclusions_that_miss_the_crawled_pages_are_refused(
    crawled_pages: int,
) -> None:
    with pytest.raises(ValidationError, match="must add up to the crawled pages"):
        selection(crawled_pages=crawled_pages)


def test_an_empty_tenant_is_a_valid_selection() -> None:
    empty = selection(crawled_pages=0, not_indexable=0, without_vector=0, targets=())
    assert empty.targets == ()


@pytest.mark.parametrize("field", ["crawled_pages", "not_indexable", "without_vector"])
def test_selection_counts_are_never_negative(field: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        selection(**{field: -1})
    assert only_error(exc_info)["loc"] == (field,)


# ── TargetCandidates ────────────────────────────────────────────────────────


@pytest.mark.parametrize("similarities", [(0.9,), (0.9, 0.5, 0.1)])
def test_one_similarity_per_source(similarities: tuple[float, ...]) -> None:
    with pytest.raises(ValidationError, match="one similarity per source"):
        candidates(similarities=similarities)


def test_duplicate_source_urls_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate source urls"):
        candidates(sources=("example.com/a", "example.com/a"), similarities=(0.9, 0.9))


def test_the_target_is_never_its_own_source() -> None:
    with pytest.raises(ValidationError, match="its own source"):
        candidates(sources=(URL, "example.com/b"), similarities=(0.99, 0.5))


@pytest.mark.parametrize("similarity", [1.0001, -1.0001])
def test_a_similarity_outside_the_cosine_range_is_refused(similarity: float) -> None:
    with pytest.raises(ValidationError, match=r"cosine in \[-1, 1\]"):
        candidates(sources=("example.com/a",), similarities=(similarity,))


def test_the_cosine_range_is_closed() -> None:
    kept = candidates(similarities=(1.0, -1.0))
    assert kept.similarities == (1.0, -1.0)


def test_sources_out_of_similarity_order_are_refused() -> None:
    with pytest.raises(ValidationError, match="ordered best first"):
        candidates(similarities=(0.5, 0.9))


def test_ties_must_be_url_ascending() -> None:
    with pytest.raises(ValidationError, match="url ascending on ties"):
        candidates(sources=("example.com/b", "example.com/a"), similarities=(0.5, 0.5))
    tied = candidates(similarities=(0.5, 0.5))
    assert tied.sources == ("example.com/a", "example.com/b")


def test_more_sources_than_eligible_pages_are_refused() -> None:
    assert candidates(eligible=2).eligible == 2
    with pytest.raises(ValidationError, match="more sources than eligible pages"):
        candidates(eligible=1)


def test_linked_nearer_cannot_exceed_linked() -> None:
    assert candidates(linked_nearer=2).linked_nearer == 2
    with pytest.raises(ValidationError, match="linked_nearer cannot exceed linked"):
        candidates(linked_nearer=3)


def test_a_target_with_no_candidates_is_valid() -> None:
    empty = candidates(sources=(), similarities=(), eligible=0, linked=0, linked_nearer=0)
    assert (empty.sources, empty.eligible) == ((), 0)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("eligible", -1, "greater_than_equal"),
        ("linked", -1, "greater_than_equal"),
        ("linked_nearer", -1, "greater_than_equal"),
        ("target_url", "", "string_too_short"),
    ],
)
def test_target_candidates_field_bounds(field: str, value: object, error: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TargetCandidates.model_validate({**candidates().model_dump(), field: value})
    assert (error, (field,)) in {(e["type"], e["loc"]) for e in exc_info.value.errors()}


# ── CandidateReport ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("field", "value"), [("full_targets", 2), ("short_targets", 2), ("empty_targets", 0)]
)
def test_full_short_and_empty_must_partition_the_targets(field: str, value: int) -> None:
    with pytest.raises(ValidationError, match=r"full \+ short \+ empty targets"):
        report(**{field: value})


def test_more_assumed_indexable_targets_than_targets_are_refused() -> None:
    assert report(indexable_assumed=3).indexable_assumed == 3
    with pytest.raises(ValidationError, match="more assumed-indexable targets"):
        report(indexable_assumed=4)


def test_every_target_is_also_a_source_page() -> None:
    assert report(source_pages=3).source_pages == 3
    with pytest.raises(ValidationError, match="also a source page"):
        report(source_pages=2)


def test_linked_nearer_cannot_exceed_the_linked_pairs() -> None:
    assert report(linked_nearer=4).linked_nearer == 4
    with pytest.raises(ValidationError, match="linked_nearer cannot exceed linked_pairs"):
        report(linked_nearer=5)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_drop_rate_is_a_closed_share(value: float) -> None:
    assert report(drop_rate=value).drop_rate == value


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("drop_rate", 1.01, "less_than_equal"),
        ("drop_rate", -0.01, "greater_than_equal"),
        ("per_target", 0, "greater_than_equal"),
        ("chunk_size", 0, "greater_than_equal"),
        ("source_pages", -1, "greater_than_equal"),
        ("linked_pairs", -1, "greater_than_equal"),
        ("min_per_target", -1, "greater_than_equal"),
        ("median_per_target", -0.5, "greater_than_equal"),
        ("load_seconds", -0.1, "greater_than_equal"),
        ("search_seconds", -0.1, "greater_than_equal"),
        ("seconds", -0.1, "greater_than_equal"),
        ("tenant_id", "", "string_too_short"),
        ("index", "page_title", "literal_error"),
    ],
)
def test_report_field_bounds(field: str, value: object, error: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        report(**{field: value})
    assert only_error(exc_info)["loc"] == (field,)
    assert only_error(exc_info)["type"] == error


def test_an_empty_run_reports_no_stats() -> None:
    empty = report(
        crawled_pages=0,
        not_indexable=0,
        targets=0,
        indexable_assumed=0,
        source_pages=0,
        candidates=0,
        full_targets=0,
        short_targets=0,
        empty_targets=0,
        min_per_target=None,
        median_per_target=None,
        max_per_target=None,
        linked_pairs=0,
        linked_nearer=0,
        drop_rate=None,
    )
    assert (empty.drop_rate, empty.median_per_target, empty.max_per_target) == (None, None, None)


@pytest.mark.parametrize("field", ["min_per_target", "median_per_target", "max_per_target"])
def test_per_target_stats_are_set_whenever_there_are_targets(field: str) -> None:
    with pytest.raises(ValidationError, match="set exactly when there are targets"):
        report(**{field: None})


def test_a_run_without_targets_has_no_per_target_stats() -> None:
    with pytest.raises(ValidationError, match="set exactly when there are targets"):
        report(
            crawled_pages=0,
            not_indexable=0,
            targets=0,
            indexable_assumed=0,
            source_pages=0,
            candidates=0,
            full_targets=0,
            short_targets=0,
            empty_targets=0,
            min_per_target=0,
            median_per_target=None,
            max_per_target=None,
            linked_pairs=0,
            linked_nearer=0,
            drop_rate=None,
        )


def test_no_target_keeps_more_than_per_target() -> None:
    assert report(max_per_target=2).max_per_target == 2
    with pytest.raises(ValidationError, match="more than per_target"):
        report(max_per_target=3)


def test_drop_rate_is_set_exactly_when_the_neighbourhood_is_not_empty() -> None:
    with pytest.raises(ValidationError, match="drop_rate is set exactly"):
        report(drop_rate=None)
    with pytest.raises(ValidationError, match="drop_rate is set exactly"):
        report(candidates=0, linked_nearer=0, drop_rate=0.0)


def test_the_gnn_index_is_accepted() -> None:
    assert report(index="page_gnn").index == "page_gnn"


# ── CandidateSet ────────────────────────────────────────────────────────────


def test_one_entry_per_reported_target() -> None:
    full = candidate_set()
    with pytest.raises(ValidationError, match="one entry per reported target"):
        candidate_set(targets=full.targets[:2])


def test_duplicate_target_entries_are_refused() -> None:
    full = candidate_set()
    with pytest.raises(ValidationError, match="duplicate target urls"):
        candidate_set(targets=(full.targets[0], full.targets[1], full.targets[0]))


def test_the_candidate_count_must_match_the_report() -> None:
    with pytest.raises(ValidationError, match="candidate count does not match"):
        candidate_set(report=report(candidates=2))


def test_no_entry_keeps_more_than_per_target() -> None:
    full = candidate_set()
    over = candidates(
        target_url="example.com/full",
        sources=("example.com/a", "example.com/b", "example.com/c"),
        similarities=(0.9, 0.5, 0.4),
    )
    with pytest.raises(ValidationError, match="more than per_target"):
        candidate_set(report=report(candidates=4), targets=(over, *full.targets[1:]))


def test_linked_counts_must_match_the_report() -> None:
    with pytest.raises(ValidationError, match="linked counts do not match"):
        candidate_set(report=report(linked_pairs=5))
    with pytest.raises(ValidationError, match="linked counts do not match"):
        candidate_set(report=report(linked_nearer=1, drop_rate=0.25))


def test_full_short_and_empty_must_match_the_entries() -> None:
    with pytest.raises(ValidationError, match="full, short and empty targets do not match"):
        candidate_set(report=report(full_targets=2, short_targets=0))


# ── hub-main-page channel ───────────────────────────────────────────────────


def with_pillar_pair() -> TargetCandidates:
    """The full target plus one channel source, weaker than its nearest ones."""
    return candidates(
        target_url="example.com/full",
        sources=("example.com/a", "example.com/b", "example.com/c"),
        similarities=(0.9, 0.5, 0.45),
        pillar_pairs=1,
    )


FLOORS: dict[str, object] = {
    "pillar_floors": {"en": 0.4, "*": 0.3},
    "pillar_floor_basis": {"en": "existing_links", "*": "candidate_pairs"},
    "pillar_floor_links": {"en": 60, "*": 12},
}


def test_channel_sources_follow_the_nearest_ones_each_part_in_order() -> None:
    entry = with_pillar_pair()
    assert (entry.nearest, entry.pillar_pairs) == (2, 1)
    # Past the cap, a channel source may be more similar than the last nearest one.
    stronger = candidates(
        sources=("example.com/a", "example.com/b", "example.com/c"),
        similarities=(0.9, 0.5, 0.6),
        pillar_pairs=1,
    )
    assert stronger.nearest == 2
    with pytest.raises(ValidationError, match="ordered best first"):
        candidates(
            sources=("example.com/a", "example.com/b", "example.com/c", "example.com/d"),
            similarities=(0.9, 0.5, 0.4, 0.45),
            eligible=4,
            pillar_pairs=2,
        )
    with pytest.raises(ValidationError, match="more pillar pairs than sources"):
        candidates(pillar_pairs=3)


def test_the_cap_holds_for_the_nearest_sources_and_channel_pairs_are_counted() -> None:
    full = candidate_set()
    targets = (with_pillar_pair(), *full.targets[1:])

    found = candidate_set(
        report=report(candidates=4, pillar_pairs=1, drop_rate=2 / 5, **FLOORS), targets=targets
    )

    assert found.report.candidates == 4
    with pytest.raises(ValidationError, match="candidate count does not match"):
        candidate_set(report=report(pillar_pairs=1, drop_rate=2 / 4, **FLOORS), targets=targets)
    with pytest.raises(ValidationError, match="pillar pair count does not match"):
        candidate_set(report=report(candidates=4, drop_rate=2 / 6), targets=targets)


def test_the_drop_rate_is_over_the_nearest_candidates_only() -> None:
    assert report(candidates=4, pillar_pairs=1, drop_rate=0.4, **FLOORS).drop_rate == 0.4
    with pytest.raises(ValidationError, match="drop_rate is set exactly"):
        report(
            candidates=1,
            pillar_pairs=1,
            linked_nearer=0,
            linked_pairs=0,
            drop_rate=0.0,
            **FLOORS,
        )
    with pytest.raises(ValidationError, match="more pillar pairs than candidates"):
        report(pillar_pairs=4, **FLOORS)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({**FLOORS, "pillar_floor_links": {"en": 60}}, "its basis and link count"),
        ({**FLOORS, "pillar_floor_basis": {"en": "existing_links"}}, "its basis and link count"),
        ({**FLOORS, "pillar_floors": {"en": 1.5, "*": 0.3}}, r"cosine in \[-1, 1\]"),
        ({**FLOORS, "pillar_floor_links": {"en": -1, "*": 12}}, "cannot be negative"),
        ({"pillar_pairs": 1, "candidates": 4, "drop_rate": 2 / 5}, "pillar pairs need a floor"),
    ],
)
def test_pillar_floors_are_reported_whole(fields: dict[str, object], message: str) -> None:
    assert report(**FLOORS).pillar_floor_basis["*"] == "candidate_pairs"
    with pytest.raises(ValidationError, match=message):
        report(**fields)


def test_an_unknown_floor_basis_is_refused() -> None:
    with pytest.raises(ValidationError) as exc_info:
        report(**{**FLOORS, "pillar_floor_basis": {"en": "guessed", "*": "candidate_pairs"}})
    assert only_error(exc_info)["type"] == "literal_error"
