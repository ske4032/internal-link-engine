"""Per-page structure and languages read for feature assembly and candidate retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError
from linking_engine.graph.repo import GraphRepo
from linking_engine.models import Link, Page, PageStructure

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

BASE = "example.com"


def url(path: str) -> str:
    return f"{BASE}{path}"


def page(path: str, **fields: Any) -> Page:
    return Page.model_validate({"url": url(path), "status_code": 200, **fields})


def link(source: str, target: str, position: int) -> Link:
    return Link(
        source_url=url(source),
        target_url=url(target),
        position=position,
        anchor_text="anchor",
        surrounding_text="around the anchor",
    )


async def seed_links(graph: GraphRepo, tenant: str, links: list[Link]) -> None:
    sources = sorted({str(item.source_url) for item in links})
    await graph.replace_links(tenant, sources, links)


@pytest.mark.integration
async def test_structure_counts_distinct_crawled_pages_and_reads_every_label(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await graph.upsert_pages(
        tenant,
        [page("/a", language="de", word_count=120, crawl_depth=0), page("/c"), page("/b")],
    )
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    # Two links from /a to /b are one inbound page; the link to the placeholder is not counted.
    await seed_links(
        graph,
        tenant,
        [
            link("/a", "/b", 0),
            link("/a", "/b", 1),
            link("/a", "/c", 2),
            link("/b", "/a", 0),
            link("/c", "/ghost", 0),
        ],
    )
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.isOrphan = false, "
        "p.pageRankPercentile = 0.5, p.linkCommunityId = 1, p.keywordCommunityId = 2, "
        "p.contentCommunityId = 3, p.hubId = 4, p.isHubPillar = true",
        t=tenant,
        u=url("/a"),
    )
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.isOrphan = true, p.hubId = -1",
        t=tenant,
        u=url("/c"),
    )
    await graph.upsert_pages(other, [page("/a"), page("/b"), page("/c")])
    await seed_links(graph, other, [link("/b", "/c", 0), link("/c", "/b", 0)])

    structure = await graph.page_structure(tenant)

    assert structure == [
        PageStructure(
            url=url("/a"),
            language="de",
            word_count=120,
            inbound=1,
            outbound=2,
            is_orphan=False,
            page_rank_percentile=0.5,
            crawl_depth=0,
            link_community_id=1,
            keyword_community_id=2,
            content_community_id=3,
            hub_id=4,
            is_hub_pillar=True,
        ),
        PageStructure(url=url("/b"), inbound=1, outbound=1),
        PageStructure(url=url("/c"), inbound=1, outbound=0, is_orphan=True, hub_id=-1),
    ]


@pytest.mark.integration
async def test_languages_are_the_tenants_crawled_pages_only(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await graph.upsert_pages(tenant, [page("/de", language="de"), page("/none")])
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await graph.upsert_pages(other, [page("/de", language="fr"), page("/other-only")])

    assert await graph.page_languages(tenant) == {url("/de"): "de", url("/none"): None}


@pytest.mark.integration
async def test_a_tenant_without_pages_reads_empty(graph: GraphRepo, tenant: str) -> None:
    assert await graph.page_structure(tenant) == []
    assert await graph.page_languages(tenant) == {}


@pytest.mark.integration
async def test_stored_values_that_do_not_fit_fail_the_read(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a")])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) SET p.pageRankPercentile = 1.5, p.language = 5", t=tenant
    )

    with pytest.raises(DatabaseReadError, match="page structure"):
        await graph.page_structure(tenant)
    with pytest.raises(DatabaseReadError, match="language 5"):
        await graph.page_languages(tenant)


@pytest.fixture
async def offline_graph() -> AsyncIterator[GraphRepo]:
    """A repo whose server does not exist: any query would raise DatabaseUnavailableError."""
    driver = AsyncGraphDatabase.driver(
        "bolt://127.0.0.1:1", auth=("neo4j", "x"), connection_timeout=1
    )
    repo = GraphRepo(driver)
    yield repo
    await repo.close()


@pytest.mark.parametrize("read", ["page_structure", "page_languages"])
async def test_reads_reject_a_blank_tenant_before_any_query(
    offline_graph: GraphRepo, read: str
) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await getattr(offline_graph, read)(" ")
