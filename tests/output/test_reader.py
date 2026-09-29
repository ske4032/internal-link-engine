"""The output reader: the latest complete run, keyset paging on ordinals and best ranks, exact
filters, and every read scoped to one tenant and run."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from pymongo import AsyncMongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseReadError,
    DatabaseUnavailableError,
    DatabaseWriteError,
)
from linking_engine.models import (
    ActionType,
    AnchorCandidate,
    AnchorType,
    BridgePair,
    ContentGapFinding,
    DuplicateGroup,
    ExcludedPage,
    ExclusionReason,
    HubSummary,
    IssueFlag,
    OrphanLabel,
    OrphanRescue,
    OrphanSlotReason,
    PageProfile,
    Recommendation,
    RecommendationStatus,
    RescueSource,
    RunInfo,
    ScorerName,
    SiteSummary,
    TargetFix,
    UnanchoredOut,
    UnanchoredReason,
)
from linking_engine.output.collections import (
    BRIDGES,
    DUPLICATES,
    EXCLUDED_PAGES,
    HUBS,
    ORPHANS,
    PAGES,
    RECOMMENDATIONS,
    RUNS,
    TARGET_FIXES,
    UNANCHORED,
    recommendation_id,
    to_document,
)
from linking_engine.output.reader import (
    BridgeFilter,
    ExcludedFilter,
    InvalidCursorError,
    OrphanFilter,
    OutputReader,
    PageFilter,
    RecommendationFilter,
    UnanchoredFilter,
    store_errors,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic import BaseModel
    from pymongo.asynchronous.database import AsyncDatabase

    from linking_engine.ingest.mongo_repo import MongoRepo

Document = dict[str, object]

STARTED = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
A, B, C, D = (f"example.com/guides/{name}" for name in ("a", "b", "c", "d"))


@pytest.fixture
async def db(mongo_uri: str) -> AsyncIterator[AsyncDatabase[Document]]:
    client: AsyncMongoClient[Document] = AsyncMongoClient(mongo_uri, tz_aware=True)
    yield client["linking_engine_test"]
    await client.close()


@pytest.fixture
def reader(db: AsyncDatabase[Document]) -> OutputReader:
    return OutputReader(db)


@pytest.fixture
def other_tenant() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


def summary() -> SiteSummary:
    return SiteSummary(
        pages=4,
        dead_end_pages=0,
        duplicate_groups=0,
        duplicate_copies=0,
        hubs=1,
        bridge_pairs=0,
        bridge_links=0,
        sources_with_recommendations=1,
        sources_below_limit=1,
        links_audited=2,
        unverified_links=0,
        target_fixes=0,
    )


def run_info(tenant: str, run_id: str, *, completed: datetime | None) -> RunInfo:
    return RunInfo(
        tenant_id=tenant,
        run_id=run_id,
        status="writing" if completed is None else "complete",
        started_at=STARTED,
        completed_at=completed,
        scorer=ScorerName.BASELINE,
        weights_version="w1",
        feature_code="f1",
        package_version="0.1.0",
        limit_per_source=10,
        content_gap_limit=3,
        words_per_link=200,
        guaranteed_inbound_links=2,
        guaranteed_inbound_below=1,
        max_suggested_inbound=5,
        summary=None if completed is None else summary(),
    )


def new_link(
    tenant: str,
    run_id: str,
    source: str,
    target: str,
    rank: int,
    *,
    action: ActionType = ActionType.ADD_LINK,
    tier: int = 1,
    best: int = 1,
    suggested: bool = False,
    orphan_slot: bool = False,
) -> Recommendation:
    gap = action is ActionType.CONTENT_GAP
    return Recommendation(
        id=recommendation_id(tenant, action, source, target, None),
        run_id=run_id,
        source_url=source,
        target_url=target,
        action_type=action,
        label="content gap: add copy first" if gap else "add a link",
        finding=ContentGapFinding.NO_TOPICAL_MENTION if gap else None,
        advice="Write a sentence about the topic first." if gap else None,
        score=90.0 - rank,
        tier=tier,
        rank_in_source=rank,
        best_rank=best,
        suggested=suggested,
        orphan_slot=orphan_slot,
        status=RecommendationStatus.PENDING,
        proposed_anchors=None
        if gap
        else (
            AnchorCandidate(text="trail shoes", anchor_type=AnchorType.PARTIAL, source="EXTRACTED"),
        ),
        rationale="baseline scorer, rank 1",
        signals=(("content_cosine", 0.5),),
        created_at=STARTED,
    )


def verdict(tenant: str, run_id: str, source: str, target: str, position: int) -> Recommendation:
    return Recommendation(
        id=recommendation_id(tenant, ActionType.REMOVE, source, target, position),
        run_id=run_id,
        source_url=source,
        target_url=target,
        action_type=ActionType.REMOVE,
        label="review this link",
        position=position,
        status=RecommendationStatus.PENDING,
        issue_flags=(IssueFlag.OFF_TOPIC,),
        rationale="the link is off topic",
        signals=(("context_relevance", 0.1),),
        created_at=STARTED,
    )


def profile(url: str, **fields: object) -> PageProfile:
    return PageProfile.model_validate(
        {"url": url, "word_count": 300, "inbound": 2, "outbound": 3, **fields}
    )


async def store(
    db: AsyncDatabase[Document],
    collection: str,
    tenant: str,
    run_id: str,
    models: Sequence[BaseModel],
) -> None:
    await db[collection].insert_many(
        [
            to_document(model, tenant_id=tenant, run_id=run_id, ordinal=ordinal)
            for ordinal, model in enumerate(models)
        ]
    )


async def store_run(db: AsyncDatabase[Document], run: RunInfo) -> None:
    await db[RUNS].insert_one(
        to_document(run, tenant_id=run.tenant_id, run_id=run.run_id, ordinal=None)
    )


@pytest.mark.integration
async def test_the_latest_run_is_the_newest_complete_one(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str, other_tenant: str
) -> None:
    older = run_info(tenant, "run-old", completed=STARTED + timedelta(minutes=5))
    newer = run_info(tenant, "run-new", completed=STARTED + timedelta(hours=1))
    for run in (older, newer, run_info(tenant, "run-writing", completed=None)):
        await store_run(db, run)
    await store_run(db, run_info(other_tenant, "run-other", completed=None))

    assert await reader.latest_run(tenant) == newer
    assert await reader.latest_run(other_tenant) is None
    assert await reader.latest_run(f"test-{uuid.uuid4().hex[:12]}") is None


@pytest.mark.integration
async def test_recommendations_page_on_ordinals_within_one_tenant_and_run(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str, other_tenant: str
) -> None:
    served = [
        new_link(tenant, "run-1", A, target, rank, best=rank)
        for rank, target in enumerate((B, C, D), 1)
    ]
    served += [verdict(tenant, "run-1", A, B, position) for position in range(4)]
    await store(db, RECOMMENDATIONS, tenant, "run-1", served)
    await store(db, RECOMMENDATIONS, tenant, "run-0", served[:2])
    # Same urls and run id under another tenant: never read.
    await store(db, RECOMMENDATIONS, other_tenant, "run-1", served)

    everything = await reader.recommendations(tenant, "run-1", RecommendationFilter(), limit=500)
    pages = []
    after = None
    while True:
        page = await reader.recommendations(tenant, "run-1", RecommendationFilter(), after, 3)
        pages.append(page)
        if page.next_cursor is None:
            break
        after = page.next_cursor

    assert everything.items == tuple(served)
    assert everything.total == 7
    assert everything.next_cursor is None
    assert [len(page.items) for page in pages] == [3, 3, 1]
    assert [page.next_cursor for page in pages] == ["2", "5", None]
    assert {page.total for page in pages} == {7}
    assert tuple(item for page in pages for item in page.items) == tuple(served)


@pytest.mark.integration
async def test_recommendation_filters_are_exact_matches(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str
) -> None:
    served = [
        new_link(tenant, "run-1", A, B, 1, best=1, suggested=True, orphan_slot=True),
        new_link(tenant, "run-1", A, C, 2, action=ActionType.CONTENT_GAP, tier=2, best=3),
        new_link(tenant, "run-1", B, C, 1, tier=2, best=2, suggested=True),
        verdict(tenant, "run-1", C, A, 0),
    ]
    await store(db, RECOMMENDATIONS, tenant, "run-1", served)

    async def matching(**filters: object) -> tuple[Recommendation, ...]:
        listing = await reader.recommendations(
            tenant, "run-1", RecommendationFilter.model_validate(filters), limit=500
        )
        assert listing.total == len(listing.items)
        return listing.items

    assert await matching(source="https://www.example.com/guides/a/") == tuple(served[:2])
    assert await matching(target=C) == (served[1], served[2])
    assert await matching(action_types=(ActionType.CONTENT_GAP, ActionType.REMOVE)) == (
        served[1],
        served[3],
    )
    assert await matching(tier=2) == (served[1], served[2])
    assert await matching(source=A, tier=2, target=C) == (served[1],)
    assert await matching(source=D) == ()
    assert await matching(suggested=True) == (served[0], served[2])
    assert await matching(suggested=False) == (served[1], served[3])
    assert await matching(orphan_slot=True) == (served[0],)
    assert await matching(orphan_slot=False, suggested=True) == (served[2],)


@pytest.mark.integration
async def test_best_first_order_pages_new_link_actions_on_their_best_rank(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str, other_tenant: str
) -> None:
    served = [
        new_link(tenant, "run-1", A, B, 1, best=3, suggested=True),
        new_link(tenant, "run-1", A, C, 1, action=ActionType.CONTENT_GAP, best=5),
        verdict(tenant, "run-1", A, D, 0),
        new_link(tenant, "run-1", B, C, 1, best=1, suggested=True),
        new_link(tenant, "run-1", B, D, 2, best=4),
        verdict(tenant, "run-1", C, A, 1),
        new_link(tenant, "run-1", D, A, 1, best=2, suggested=True, orphan_slot=True),
    ]
    await store(db, RECOMMENDATIONS, tenant, "run-1", served)
    await store(db, RECOMMENDATIONS, tenant, "run-0", served[3:4])
    # Another tenant's records rank first in their own order, and are never read.
    await store(db, RECOMMENDATIONS, other_tenant, "run-1", [served[1], served[0]])
    best_first = (served[3], served[6], served[0], served[4], served[1])

    everything = await reader.recommendations(
        tenant, "run-1", RecommendationFilter(), limit=500, order="best"
    )
    pages = []
    after = None
    while True:
        page = await reader.recommendations(
            tenant, "run-1", RecommendationFilter(), after, 2, order="best"
        )
        pages.append(page)
        if page.next_cursor is None:
            break
        after = page.next_cursor
    suggested = await reader.recommendations(
        tenant, "run-1", RecommendationFilter(suggested=True), limit=500, order="best"
    )
    from_a = await reader.recommendations(
        tenant, "run-1", RecommendationFilter(source=A), limit=500, order="best"
    )

    assert everything.items == best_first
    assert (everything.total, everything.next_cursor) == (5, None)
    assert [page.items for page in pages] == [best_first[:2], best_first[2:4], best_first[4:]]
    assert [page.next_cursor for page in pages] == ["2", "4", None]
    assert {page.total for page in pages} == {5}
    assert suggested.items == (served[3], served[6], served[0])
    assert suggested.total == 3
    assert from_a.items == (served[0], served[1])
    assert (await reader.recommendations(tenant, "run-1", RecommendationFilter())).items == tuple(
        served
    )


@pytest.mark.integration
async def test_one_recommendation_is_found_by_id_in_its_tenant_and_run_only(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str, other_tenant: str
) -> None:
    mine = new_link(tenant, "run-1", A, B, 1)
    theirs = new_link(other_tenant, "run-1", A, B, 1)
    await store(db, RECOMMENDATIONS, tenant, "run-1", [mine])
    await store(db, RECOMMENDATIONS, other_tenant, "run-1", [theirs])

    assert mine.id != theirs.id
    assert await reader.recommendation(tenant, "run-1", mine.id) == mine
    assert await reader.recommendation(tenant, "run-2", mine.id) is None
    assert await reader.recommendation(tenant, "run-1", theirs.id) is None


@pytest.mark.integration
async def test_page_filters_are_exact_matches(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str
) -> None:
    profiles = [
        profile(A, hub_id=1, duplicate_group=0, is_canonical=True),
        profile(B, hub_id=1, is_orphan=True, orphan_label=OrphanLabel.MENUS_ONLY),
        profile(C, hub_id=2, is_orphan=True, orphan_label=OrphanLabel.NOT_LINKED, is_dead_end=True),
        profile(D, duplicate_group=0, is_canonical=False),
    ]
    await store(db, PAGES, tenant, "run-1", profiles)

    async def urls(**filters: object) -> list[str]:
        listing = await reader.pages(tenant, "run-1", PageFilter.model_validate(filters), limit=500)
        assert listing.total == len(listing.items)
        return [page.url for page in listing.items]

    assert await urls() == [A, B, C, D]
    assert await urls(hub=1) == [A, B]
    assert await urls(orphan=True) == [B, C]
    assert await urls(orphan=False) == [A, D]
    assert await urls(orphan=True, orphan_label=OrphanLabel.NOT_LINKED) == [C]
    assert await urls(dead_end=True) == [C]
    assert await urls(duplicate=True) == [A, D]
    assert await urls(duplicate=False) == [B, C]


@pytest.mark.integration
async def test_orphans_list_rescues_by_url_filtered_by_label_and_unmet(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str, other_tenant: str
) -> None:
    link = new_link(tenant, "run-1", A, B, 1, suggested=True, orphan_slot=True)
    rescues = [
        OrphanRescue(
            profile=profile(
                B, inbound=0, is_orphan=True, orphan_label=OrphanLabel.MENUS_ONLY, link_budget=1
            ),
            guaranteed=1,
            suggested_in=1,
            sources=(
                RescueSource(
                    source_url=A,
                    score=89.0,
                    tier=1,
                    anchor="trail shoes",
                    source_page_rank_percentile=0.5,
                    recommendation_id=link.id,
                ),
                RescueSource(source_url=D, score=40.0, tier=2),
            ),
        ),
        OrphanRescue(
            profile=profile(C, inbound=0, is_orphan=True, orphan_label=OrphanLabel.NOT_LINKED),
            guaranteed=1,
            suggested_in=0,
            sources=(RescueSource(source_url=A, score=60.0, tier=2),),
            unmet_reason=OrphanSlotReason.NO_ANCHOR,
        ),
        OrphanRescue(
            profile=profile(D, inbound=0, is_orphan=True, orphan_label=OrphanLabel.NOT_LINKED),
            guaranteed=1,
            suggested_in=0,
            sources=(),
            unmet_reason=OrphanSlotReason.NO_RELEVANT_SOURCE,
        ),
    ]
    await store(db, ORPHANS, tenant, "run-1", rescues)
    await store(db, ORPHANS, other_tenant, "run-1", rescues[:1])

    async def urls(after: str | None = None, limit: int = 500, **filters: object) -> list[str]:
        listing = await reader.orphans(
            tenant, "run-1", OrphanFilter.model_validate(filters), after, limit
        )
        if after is None and limit == 500:
            assert listing.total == len(listing.items)
        return [rescue.profile.url for rescue in listing.items]

    everything = await reader.orphans(tenant, "run-1", OrphanFilter(), limit=2)
    assert everything.items == tuple(rescues[:2])
    assert (everything.next_cursor, everything.total) == ("1", 3)
    assert await urls(after="1") == [D]
    assert await urls(orphan_label=OrphanLabel.NOT_LINKED) == [C, D]
    assert await urls(unmet=True) == [C, D]
    assert await urls(unmet=False) == [B]
    assert await urls(unmet=False, orphan_label=OrphanLabel.NOT_LINKED) == []
    assert await urls(orphan_label=OrphanLabel.FOOTER_ONLY) == []
    assert (await reader.orphans(other_tenant, "run-1", OrphanFilter())).total == 1
    assert (await reader.orphans(tenant, "run-2", OrphanFilter())).total == 0


@pytest.mark.integration
async def test_page_detail_lists_its_records_and_counts_new_links_in(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str, other_tenant: str
) -> None:
    await store(db, PAGES, tenant, "run-1", [profile(A), profile(B)])
    served = [
        new_link(tenant, "run-1", A, B, 1),
        new_link(tenant, "run-1", A, C, 2, action=ActionType.CONTENT_GAP),
        verdict(tenant, "run-1", A, D, 3),
        new_link(tenant, "run-1", B, A, 1),
        new_link(tenant, "run-1", C, A, 1, action=ActionType.CONTENT_GAP),
        verdict(tenant, "run-1", D, A, 0),
    ]
    await store(db, RECOMMENDATIONS, tenant, "run-1", served)
    await store(db, RECOMMENDATIONS, other_tenant, "run-1", served)

    detail = await reader.page(tenant, "run-1", A)

    assert detail is not None
    assert detail.profile == profile(A)
    assert detail.outgoing == tuple(served[:3])
    assert detail.incoming_total == 2
    assert await reader.page(tenant, "run-1", C) is None
    assert await reader.page(tenant, "run-2", A) is None


@pytest.mark.integration
async def test_hubs_bridges_duplicates_unanchored_and_target_fixes_are_listed(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str
) -> None:
    hubs = [
        HubSummary(hub_id=hub, size=5, orphan_pages=0, dead_end_pages=0, recommendations_in=hub)
        for hub in range(3)
    ]
    bridges = [
        BridgePair(
            hub_a=a,
            hub_b=b,
            size_a=5,
            size_b=5,
            pages_ab=0,
            pages_ba=0,
            link_density=0.0,
            centroid_cosine=0.4,
            bridge_gap=0.2,
        )
        for a, b in ((0, 1), (0, 2), (1, 2))
    ]
    duplicates = [DuplicateGroup(group_id=0, canonical=A, copies=(B,))]
    unanchored = [
        UnanchoredOut(
            source_url=A,
            target_url=B,
            reason=UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC,
            advice="Write about it.",
            rank_in_source=1,
            recommended=True,
        ),
        UnanchoredOut(
            source_url=A,
            target_url=C,
            reason=UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD,
            advice="Give the target a keyword.",
            rank_in_source=2,
        ),
        UnanchoredOut(
            source_url=B,
            target_url=C,
            reason=UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD,
            advice="Give the target a keyword.",
            rank_in_source=1,
        ),
    ]
    fixes = [
        TargetFix(
            target_url=C, fix="Give the target a keyword.", waiting_sources=2, best_sources=(A, B)
        )
    ]
    for collection, models in (
        (HUBS, hubs),
        (BRIDGES, bridges),
        (DUPLICATES, duplicates),
        (UNANCHORED, unanchored),
        (TARGET_FIXES, fixes),
    ):
        await store(db, collection, tenant, "run-1", models)

    assert (await reader.hubs(tenant, "run-1", limit=2)).items == tuple(hubs[:2])
    assert (await reader.hubs(tenant, "run-1", "1")).items == (hubs[2],)
    assert (await reader.bridges(tenant, "run-1", BridgeFilter())).total == 3
    assert (await reader.bridges(tenant, "run-1", BridgeFilter(hub=2))).items == tuple(bridges[1:])
    assert (await reader.bridges(tenant, "run-1", BridgeFilter(hub=0))).items == tuple(bridges[:2])
    assert (await reader.duplicates(tenant, "run-1")).items == tuple(duplicates)
    no_keyword = UnanchoredFilter(reason=UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD)
    assert (await reader.unanchored(tenant, "run-1", no_keyword)).items == tuple(unanchored[1:])
    assert (await reader.unanchored(tenant, "run-1", UnanchoredFilter(source=B))).items == (
        unanchored[2],
    )
    assert (await reader.unanchored(tenant, "run-1", UnanchoredFilter(target=B))).items == (
        unanchored[0],
    )
    assert (await reader.target_fixes(tenant, "run-1")).items == tuple(fixes)
    assert (await reader.target_fixes(tenant, "run-2")).total == 0


@pytest.mark.integration
async def test_excluded_pages_page_on_url_whatever_the_run(
    mongo: MongoRepo, reader: OutputReader, tenant: str, other_tenant: str
) -> None:
    excluded = [
        ExcludedPage(url=url, reason=reason, label="excluded", words=100, link_words=90, links=12)
        for url, reason in (
            (D, ExclusionReason.SITEMAP),
            (A, ExclusionReason.INSUFFICIENT_CONTENT),
            (C, ExclusionReason.SITEMAP),
            (B, ExclusionReason.TENANT_EXCLUDED),
        )
    ]
    await mongo.replace_excluded_pages(tenant, excluded)
    await mongo.replace_excluded_pages(other_tenant, excluded[:1])

    first = await reader.excluded_pages(tenant, ExcludedFilter(), limit=3)
    rest = await reader.excluded_pages(tenant, ExcludedFilter(), first.next_cursor, 3)
    sitemap = await reader.excluded_pages(tenant, ExcludedFilter(reason=ExclusionReason.SITEMAP))

    by_url = sorted(excluded, key=lambda page: page.url)
    assert first.items == tuple(by_url[:3])
    assert first.next_cursor == C
    assert first.total == 4
    assert rest.items == (by_url[3],)
    assert rest.next_cursor is None
    assert sitemap.items == (excluded[2], excluded[0])
    assert sitemap.total == 2
    assert (await reader.excluded_pages(other_tenant, ExcludedFilter())).total == 1


@pytest.mark.integration
async def test_a_document_that_does_not_fit_its_model_is_a_read_error(
    db: AsyncDatabase[Document], reader: OutputReader, tenant: str
) -> None:
    await db[HUBS].insert_one({"tenantId": tenant, "runId": "run-1", "ordinal": 0, "hub_id": -1})
    await db[EXCLUDED_PAGES].insert_one({"tenantId": tenant, "url": A, "reason": "UNKNOWN"})

    with pytest.raises(DatabaseReadError, match="HubSummary"):
        await reader.hubs(tenant, "run-1")
    with pytest.raises(DatabaseReadError, match="ExcludedPage"):
        await reader.excluded_pages(tenant, ExcludedFilter())


async def test_bad_cursors_limits_and_tenants_are_rejected_before_any_read() -> None:
    client: AsyncMongoClient[Document] = AsyncMongoClient("mongodb://127.0.0.1:1", connect=False)
    reader = OutputReader(client["linking_engine_test"])
    try:
        for cursor in ("", "-1", "01", "1.5", "abc", " 1", "1" * 19, "٣", "1\n"):
            with pytest.raises(InvalidCursorError):
                await reader.hubs("test-t", "run-1", cursor)
            with pytest.raises(InvalidCursorError):
                await reader.recommendations(
                    "test-t", "run-1", RecommendationFilter(), cursor, order="best"
                )
        assert issubclass(InvalidCursorError, ValueError)
        with pytest.raises(ValueError, match="limit"):
            await reader.hubs("test-t", "run-1", limit=0)
        with pytest.raises(ValueError, match="limit"):
            await reader.excluded_pages("test-t", ExcludedFilter(), limit=0)
        with pytest.raises(ValueError, match="tenant_id"):
            await reader.latest_run(" ")
        with pytest.raises(ValueError, match="tenant_id"):
            await reader.recommendation("", "run-1", "0" * 16)
    finally:
        await client.close()


async def test_an_unreachable_store_is_reported_as_unavailable() -> None:
    client: AsyncMongoClient[Document] = AsyncMongoClient(
        "mongodb://127.0.0.1:1", serverSelectionTimeoutMS=100
    )
    try:
        with pytest.raises(DatabaseUnavailableError, match="server unavailable"):
            await OutputReader(client["linking_engine_test"]).latest_run("test-t")
    finally:
        await client.close()


@pytest.mark.parametrize(
    ("error", "write", "expected"),
    [
        (ConnectionFailure("down"), False, DatabaseUnavailableError),
        (OperationFailure("denied", code=13), False, DatabaseAuthError),
        (OperationFailure("bad auth", code=18), True, DatabaseAuthError),
        (OperationFailure("bad query", code=2), False, DatabaseReadError),
        (OperationFailure("bad update", code=2), True, DatabaseWriteError),
    ],
)
def test_driver_errors_are_translated(
    error: Exception, write: bool, expected: type[Exception]
) -> None:
    with pytest.raises(expected) as raised, store_errors("read things", write=write):
        raise error
    assert raised.value.__cause__ is error
