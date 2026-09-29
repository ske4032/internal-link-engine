"""Every output API route over a tenant's stored output: listings paged on their cursors, exact
filters on normalised urls, and only the key's tenant's latest complete run served. Needs
Docker."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

import httpx
import pytest
from pydantic import SecretStr
from pymongo import AsyncMongoClient

from linking_engine.api.app import ApiSettings, create_app
from linking_engine.models import (
    ActionType,
    AnchorCandidate,
    AnchorPlacement,
    AnchorType,
    BridgeLinkOut,
    BridgePair,
    BridgeReason,
    ContentGapFinding,
    DuplicateGroup,
    ExcludedPage,
    ExclusionReason,
    HubSummary,
    IssueFlag,
    Listing,
    OrphanLabel,
    PageDetail,
    PageProfile,
    Recommendation,
    RecommendationStatus,
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
    HUBS,
    PAGES,
    RECOMMENDATIONS,
    RUNS,
    TARGET_FIXES,
    UNANCHORED,
    recommendation_id,
    to_document,
)
from linking_engine.output.keys import KeyStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from pydantic import BaseModel
    from pymongo.asynchronous.database import AsyncDatabase

    from linking_engine.ingest.mongo_repo import MongoRepo

pytestmark = pytest.mark.integration

Document = dict[str, object]

DATABASE: Final = "linking_engine_test"
STARTED: Final = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
A, B, C, D = (f"example.com/guides/{name}" for name in ("a", "b", "c", "d"))
NOT_FOUND: Final = {"detail": "not found"}

type Params = Mapping[str, str | int | Sequence[str]]


@dataclass(frozen=True)
class Output:
    run: RunInfo
    recommendations: tuple[Recommendation, ...]
    pages: tuple[PageProfile, ...]
    hubs: tuple[HubSummary, ...]
    bridges: tuple[BridgePair, ...]
    duplicates: tuple[DuplicateGroup, ...]
    unanchored: tuple[UnanchoredOut, ...]
    target_fixes: tuple[TargetFix, ...]


@dataclass(frozen=True)
class Served:
    tenant: str
    key: str
    output: Output
    excluded: tuple[ExcludedPage, ...]


def run_info(tenant: str, run_id: str, *, complete: bool, pages: int) -> RunInfo:
    return RunInfo(
        tenant_id=tenant,
        run_id=run_id,
        status="complete" if complete else "writing",
        started_at=STARTED if complete else STARTED + timedelta(days=1),
        completed_at=STARTED + timedelta(minutes=3) if complete else None,
        scorer=ScorerName.BASELINE,
        weights_version="w1",
        feature_code="f1",
        package_version="0.1.0",
        limit_per_source=10,
        content_gap_limit=3,
        summary=SiteSummary(
            pages=pages,
            dead_end_pages=1,
            duplicate_groups=1,
            duplicate_copies=1,
            hubs=2,
            bridge_pairs=1,
            bridge_links=1,
            sources_with_recommendations=2,
            sources_below_limit=2,
            links_audited=5,
            unverified_links=1,
            target_fixes=1,
        )
        if complete
        else None,
    )


def new_link(
    tenant: str, run_id: str, source: str, target: str, rank: int, *, gap: bool = False
) -> Recommendation:
    action = ActionType.CONTENT_GAP if gap else ActionType.ADD_LINK
    anchor = AnchorCandidate(
        text="trail shoes",
        anchor_type=AnchorType.PARTIAL,
        source="EXTRACTED",
        score=0.8,
        keyword="trail shoes",
        placement=AnchorPlacement(sentence="Pick trail shoes.", sentence_index=0, start=5, end=16),
    )
    return Recommendation(
        id=recommendation_id(tenant, action, source, target, None),
        run_id=run_id,
        source_url=source,
        target_url=target,
        action_type=action,
        label="content gap: add copy first" if gap else "add a link",
        finding=ContentGapFinding.NO_TOPICAL_MENTION if gap else None,
        advice="Write a sentence about the topic first." if gap else None,
        score=80.0 - rank,
        tier=rank,
        rank_in_source=rank,
        status=RecommendationStatus.PENDING,
        proposed_anchors=None if gap else (anchor,),
        rationale="baseline scorer; strongest signal content cosine",
        signals=(("content_cosine", 0.4), ("target_is_orphan", -0.1)),
        created_at=STARTED,
    )


def verdict(
    tenant: str, run_id: str, source: str, target: str, position: int, action: ActionType
) -> Recommendation:
    return Recommendation(
        id=recommendation_id(tenant, action, source, target, position),
        run_id=run_id,
        source_url=source,
        target_url=target,
        action_type=action,
        label="fix this link" if action is ActionType.FIX else "review this link",
        position=position,
        status=RecommendationStatus.PENDING,
        current_anchor="read more",
        issue_flags=(IssueFlag.GENERIC,),
        fix_target=B if action is ActionType.FIX else None,
        rationale="the anchor says nothing about the target",
        signals=(("anchor_quality_score", 0.1),),
        created_at=STARTED,
    )


def profile(url: str, **fields: object) -> PageProfile:
    return PageProfile.model_validate(
        {"url": url, "word_count": 400, "inbound": 1, "outbound": 2, **fields}
    )


def output_a(tenant: str, run_id: str) -> Output:
    return Output(
        run=run_info(tenant, run_id, complete=True, pages=4),
        recommendations=(
            new_link(tenant, run_id, A, B, 1),
            new_link(tenant, run_id, A, C, 2, gap=True),
            verdict(tenant, run_id, A, D, 0, ActionType.REMOVE),
            new_link(tenant, run_id, B, A, 1),
            verdict(tenant, run_id, C, A, 2, ActionType.FIX),
        ),
        pages=(
            profile(A, hub_id=0, is_hub_pillar=True, duplicate_group=0, is_canonical=True),
            profile(B, hub_id=0, is_orphan=True, orphan_label=OrphanLabel.MENUS_ONLY),
            profile(
                C, hub_id=1, is_orphan=True, orphan_label=OrphanLabel.NOT_LINKED, is_dead_end=True
            ),
            profile(D, duplicate_group=0, is_canonical=False),
        ),
        hubs=(
            HubSummary(
                hub_id=0,
                size=2,
                pillar_url=A,
                orphan_pages=1,
                dead_end_pages=0,
                recommendations_in=2,
                bridge_hubs=(1,),
            ),
            HubSummary(
                hub_id=1,
                size=1,
                orphan_pages=1,
                dead_end_pages=1,
                recommendations_in=1,
                bridge_hubs=(0,),
            ),
        ),
        bridges=(
            BridgePair(
                hub_a=0,
                hub_b=1,
                size_a=2,
                size_b=1,
                pages_ab=0,
                pages_ba=0,
                link_density=0.0,
                centroid_cosine=0.5,
                bridge_gap=0.3,
                reasons=(BridgeReason.SPANNING_TREE,),
                links=(
                    BridgeLinkOut(
                        hub_from=0,
                        hub_to=1,
                        slot=0,
                        rank=1,
                        source_url=A,
                        target_url=C,
                        similarity=0.6,
                        reasons=(BridgeReason.SPANNING_TREE,),
                    ),
                ),
            ),
        ),
        duplicates=(DuplicateGroup(group_id=0, canonical=A, copies=(D,)),),
        unanchored=(
            UnanchoredOut(
                source_url=A,
                target_url=C,
                reason=UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC,
                advice="Write about the topic.",
                rank_in_source=2,
                recommended=True,
            ),
            UnanchoredOut(
                source_url=B,
                target_url=C,
                reason=UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD,
                advice="Give the target a keyword.",
                rank_in_source=3,
            ),
        ),
        target_fixes=(
            TargetFix(
                target_url=C,
                fix="Give the target a keyword.",
                waiting_sources=1,
                best_sources=(B,),
            ),
        ),
    )


def output_b(tenant: str, run_id: str) -> Output:
    """The same urls, other content."""
    return Output(
        run=run_info(tenant, run_id, complete=True, pages=2),
        recommendations=(new_link(tenant, run_id, A, D, 1), new_link(tenant, run_id, D, A, 1)),
        pages=(profile(A, hub_id=5), profile(D, hub_id=5)),
        hubs=(
            HubSummary(hub_id=5, size=2, orphan_pages=0, dead_end_pages=0, recommendations_in=2),
        ),
        bridges=(),
        duplicates=(),
        unanchored=(
            UnanchoredOut(
                source_url=A,
                target_url=B,
                reason=UnanchoredReason.MEANING_SEARCH_NOT_RUN,
                advice="Rerun the meaning search.",
                rank_in_source=2,
            ),
        ),
        target_fixes=(),
    )


async def write(db: AsyncDatabase[Document], output: Output) -> None:
    tenant, run_id = output.run.tenant_id, output.run.run_id
    await db[RUNS].insert_one(
        to_document(output.run, tenant_id=tenant, run_id=run_id, ordinal=None)
    )
    collections: tuple[tuple[str, Sequence[BaseModel]], ...] = (
        (RECOMMENDATIONS, output.recommendations),
        (PAGES, output.pages),
        (HUBS, output.hubs),
        (BRIDGES, output.bridges),
        (DUPLICATES, output.duplicates),
        (UNANCHORED, output.unanchored),
        (TARGET_FIXES, output.target_fixes),
    )
    for collection, models in collections:
        if models:
            await db[collection].insert_many(
                [
                    to_document(model, tenant_id=tenant, run_id=run_id, ordinal=ordinal)
                    for ordinal, model in enumerate(models)
                ]
            )


@pytest.fixture
async def db(mongo_uri: str) -> AsyncIterator[AsyncDatabase[Document]]:
    client: AsyncMongoClient[Document] = AsyncMongoClient(mongo_uri, tz_aware=True)
    yield client[DATABASE]
    await client.close()


@pytest.fixture
async def api(mongo_uri: str) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(ApiSettings(mongo_uri=SecretStr(mongo_uri), mongo_db=DATABASE))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://api.example"
        ) as client,
    ):
        yield client


@pytest.fixture
async def served(db: AsyncDatabase[Document], mongo: MongoRepo, tenant: str) -> Served:
    """Tenant A's output next to a writing rerun of its own and tenant B's output on the same
    urls, with a key for A."""
    other = f"test-{uuid.uuid4().hex[:12]}"
    output = output_a(tenant, uuid.uuid4().hex)
    await write(db, output)
    await write(db, output_b(other, output.run.run_id))
    rerun = uuid.uuid4().hex
    await db[RUNS].insert_one(
        to_document(
            run_info(tenant, rerun, complete=False, pages=0),
            tenant_id=tenant,
            run_id=rerun,
            ordinal=None,
        )
    )
    await db[RECOMMENDATIONS].insert_one(
        to_document(new_link(tenant, rerun, C, D, 1), tenant_id=tenant, run_id=rerun, ordinal=0)
    )
    excluded = tuple(
        ExcludedPage(url=url, reason=reason, label="excluded", words=50, link_words=45, links=9)
        for url, reason in (
            ("example.com/sitemap", ExclusionReason.SITEMAP),
            ("example.com/all-links", ExclusionReason.INSUFFICIENT_CONTENT),
            ("example.com/html-sitemap", ExclusionReason.SITEMAP),
        )
    )
    await mongo.replace_excluded_pages(tenant, excluded)
    await mongo.replace_excluded_pages(other, excluded[:1])
    keys = KeyStore(db)
    await keys.ensure_indexes()
    key, _ = await keys.issue(tenant, "routes")
    return Served(tenant=tenant, key=key, output=output, excluded=excluded)


async def get(
    api: httpx.AsyncClient, served: Served, route: str, params: Params | None = None
) -> httpx.Response:
    return await api.get(
        f"/v1/tenants/{served.tenant}{route}", params=params, headers={"X-API-Key": served.key}
    )


async def walk(
    api: httpx.AsyncClient, served: Served, route: str, limit: int
) -> list[dict[str, object]]:
    pages: list[dict[str, object]] = []
    params: dict[str, str | int] = {"limit": limit}
    while True:
        response = await get(api, served, route, params)
        assert response.status_code == 200
        page = response.json()
        pages.append(page)
        if page["next_cursor"] is None:
            return pages
        params = {"limit": limit, "after": page["next_cursor"]}


async def test_recommendations_are_paged_in_their_stable_order(
    api: httpx.AsyncClient, served: Served
) -> None:
    pages = await walk(api, served, "/recommendations", 2)
    everything = await get(api, served, "/recommendations", {"limit": 500})

    whole = Listing[Recommendation].model_validate(everything.json())
    assert whole.items == served.output.recommendations
    assert whole.total == 5
    assert whole.next_cursor is None
    walked = [Listing[Recommendation].model_validate(page) for page in pages]
    assert [len(page.items) for page in walked] == [2, 2, 1]
    assert {page.total for page in walked} == {5}
    assert tuple(item for page in walked for item in page.items) == whole.items
    default = Listing[Recommendation].model_validate(
        (await get(api, served, "/recommendations")).json()
    )
    assert default.items == whole.items


async def test_recommendation_filters_are_exact_on_normalised_urls(
    api: httpx.AsyncClient, served: Served
) -> None:
    served_recommendations = served.output.recommendations

    async def items(params: Params) -> tuple[Recommendation, ...]:
        response = await get(api, served, "/recommendations", params)
        assert response.status_code == 200
        listing = Listing[Recommendation].model_validate(response.json())
        assert listing.total == len(listing.items)
        return listing.items

    assert (
        await items({"source": "https://www.example.com/guides/a/"}) == served_recommendations[:3]
    )
    assert await items({"target": "http://example.com/guides/a"}) == served_recommendations[3:]
    assert await items({"action_type": ["ADD_LINK", "FIX"]}) == (
        served_recommendations[0],
        served_recommendations[3],
        served_recommendations[4],
    )
    assert await items({"tier": 2}) == (served_recommendations[1],)
    assert await items({"source": A, "action_type": "REMOVE"}) == (served_recommendations[2],)
    assert await items({"source": "https://example.com/guides/c", "tier": 1}) == ()


@pytest.mark.parametrize(
    "params",
    [
        {"source": "mailto:someone@example.com"},
        {"target": "https://"},
        {"action_type": "REPOSITION"},
        {"tier": 0},
        {"limit": 0},
        {"limit": 501},
        {"after": "-1"},
        {"after": "next"},
        {"after": "01"},
    ],
)
async def test_invalid_listing_parameters_get_422(
    api: httpx.AsyncClient, served: Served, params: dict[str, str | int]
) -> None:
    response = await get(api, served, "/recommendations", params)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


async def test_one_recommendation_is_served_by_id(api: httpx.AsyncClient, served: Served) -> None:
    wanted = served.output.recommendations[1]

    found = await get(api, served, f"/recommendations/{wanted.id}")
    missing = await get(api, served, "/recommendations/0123456789abcdef")
    malformed = await get(api, served, "/recommendations/not-an-id")

    assert found.status_code == 200
    assert Recommendation.model_validate(found.json()) == wanted
    assert (missing.status_code, missing.json()) == (404, NOT_FOUND)
    assert malformed.status_code == 422


async def test_the_latest_complete_run_and_its_summary_are_served(
    api: httpx.AsyncClient, served: Served
) -> None:
    run = await get(api, served, "/runs/latest")
    summary = await get(api, served, "/summary")

    assert RunInfo.model_validate(run.json()) == served.output.run
    assert SiteSummary.model_validate(summary.json()) == served.output.run.summary


async def test_pages_and_orphans_are_filtered(api: httpx.AsyncClient, served: Served) -> None:
    pages = served.output.pages

    async def urls(route: str, params: Params | None = None) -> list[str]:
        response = await get(api, served, route, params)
        assert response.status_code == 200
        listing = Listing[PageProfile].model_validate(response.json())
        assert listing.total == len(listing.items)
        return [page.url for page in listing.items]

    whole = Listing[PageProfile].model_validate((await get(api, served, "/pages")).json())
    assert whole.items == pages
    assert await urls("/pages", {"hub": 0}) == [A, B]
    assert await urls("/pages", {"orphan": "true"}) == [B, C]
    assert await urls("/pages", {"orphan": "false", "duplicate": "true"}) == [A, D]
    assert await urls("/pages", {"orphan_label": "NOT_LINKED"}) == [C]
    assert await urls("/pages", {"dead_end": "true"}) == [C]
    assert await urls("/pages", {"duplicate": "false"}) == [B, C]
    assert await urls("/orphans") == [B, C]
    assert await urls("/orphans", {"orphan_label": "MENUS_ONLY"}) == [B]
    assert (await get(api, served, "/pages", {"hub": -1})).status_code == 422


async def test_page_detail_is_served_by_url(api: httpx.AsyncClient, served: Served) -> None:
    detail = await get(api, served, "/page", {"url": "https://www.example.com/guides/a/"})
    unknown = await get(api, served, "/page", {"url": "https://example.com/guides/zz"})

    assert detail.status_code == 200
    assert PageDetail.model_validate(detail.json()) == PageDetail(
        profile=served.output.pages[0],
        outgoing=served.output.recommendations[:3],
        incoming_total=1,
    )
    assert (unknown.status_code, unknown.json()) == (404, NOT_FOUND)
    assert (await get(api, served, "/page", {"url": "ftp://example.com/a"})).status_code == 422
    assert (await get(api, served, "/page")).status_code == 422


async def test_hubs_bridges_duplicates_unanchored_and_target_fixes_are_served(
    api: httpx.AsyncClient, served: Served
) -> None:
    output = served.output

    async def listing[L: BaseModel](model: type[L], route: str, params: Params | None = None) -> L:
        response = await get(api, served, route, params)
        assert response.status_code == 200
        return model.model_validate(response.json())

    assert (await listing(Listing[HubSummary], "/hubs")).items == output.hubs
    hubs = await listing(Listing[HubSummary], "/hubs", {"limit": 1})
    assert (hubs.items, hubs.next_cursor, hubs.total) == (output.hubs[:1], "0", 2)
    bridges = Listing[BridgePair]
    assert (await listing(bridges, "/bridges")).items == output.bridges
    assert (await listing(bridges, "/bridges", {"hub": 1})).items == output.bridges
    assert (await listing(bridges, "/bridges", {"hub": 5})).items == ()
    assert (await listing(Listing[DuplicateGroup], "/duplicates")).items == output.duplicates
    unanchored = Listing[UnanchoredOut]
    assert (await listing(unanchored, "/unanchored")).items == output.unanchored
    no_keyword = {"reason": "TARGET_PAGE_HAS_NO_KEYWORD"}
    assert (await listing(unanchored, "/unanchored", no_keyword)).items == output.unanchored[1:]
    assert (await listing(unanchored, "/unanchored", {"source": A, "target": C})).items == (
        output.unanchored[0],
    )
    assert (await listing(Listing[TargetFix], "/target-fixes")).items == output.target_fixes


async def test_excluded_pages_are_paged_on_url(api: httpx.AsyncClient, served: Served) -> None:
    by_url = tuple(sorted(served.excluded, key=lambda page: page.url))

    pages = [
        Listing[ExcludedPage].model_validate(page)
        for page in await walk(api, served, "/excluded-pages", 2)
    ]
    sitemaps = await get(api, served, "/excluded-pages", {"reason": "SITEMAP"})

    assert [page.items for page in pages] == [by_url[:2], by_url[2:]]
    assert pages[0].next_cursor == by_url[1].url
    assert {page.total for page in pages} == {3}
    assert Listing[ExcludedPage].model_validate(sitemaps.json()).items == (
        served.excluded[2],
        served.excluded[0],
    )
