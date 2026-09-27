"""load_page_signals: the three tenant-scoped reads become one PageSignals per crawled page."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest
from structlog.testing import capture_logs

from linking_engine.models import CommunityContext, KeywordSource, Page
from linking_engine.pipeline.signals import load_page_signals

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

STRATEGIC, OBSERVED = KeywordSource.CLIENT_STRATEGIC, KeywordSource.GSC_OBSERVED


def url(path: str) -> str:
    return f"example.com{path}"


@dataclass
class FakeStores:
    pages: list[CommunityContext]
    keywords: list[tuple[str, str, KeywordSource]]
    queries: list[tuple[str, str]]
    reads: list[tuple[str, str]] = field(default_factory=list)

    async def community_context(self, tenant_id: str) -> list[CommunityContext]:
        self.reads.append(("community_context", tenant_id))
        return self.pages

    async def keyword_edges(self, tenant_id: str) -> list[tuple[str, str, KeywordSource]]:
        self.reads.append(("keyword_edges", tenant_id))
        return self.keywords

    async def gsc_queries(self, tenant_id: str) -> list[tuple[str, str]]:
        self.reads.append(("gsc_queries", tenant_id))
        return self.queries


async def load(stores: FakeStores, tenant_id: str = "acme") -> tuple[dict[str, object], int]:
    pages, unmatched = await load_page_signals(
        cast("GraphRepo", stores), cast("MongoRepo", stores), tenant_id
    )
    return dict(pages), unmatched


async def test_every_crawled_page_gets_signals_and_stray_gsc_urls_are_counted_once() -> None:
    stores = FakeStores(
        pages=[CommunityContext(url=url("/a"), hub_id=-1), CommunityContext(url=url("/b"))],
        keywords=[(url("/a"), "Trail Shoes", STRATEGIC), (url("/gone"), "tents", STRATEGIC)],
        queries=[
            (url("/a"), "tents"),
            (url("/gone"), "tents"),
            (url("/gone"), "stoves"),
            (url("/elsewhere"), "tents"),
        ],
    )

    pages, unmatched = await load(stores)

    assert sorted(pages) == [url("/a"), url("/b")]
    assert unmatched == 2, "two distinct GSC urls are not crawled pages"
    assert sorted(stores.reads) == [
        ("community_context", "acme"),
        ("gsc_queries", "acme"),
        ("keyword_edges", "acme"),
    ]


async def test_the_log_line_counts_the_reads_without_urls() -> None:
    stores = FakeStores(
        pages=[CommunityContext(url=url("/a"))],
        keywords=[(url("/a"), "tents", OBSERVED)],
        queries=[(url("/a"), "tents"), (url("/x"), "stoves")],
    )

    with capture_logs() as logs:
        await load(stores)

    [line] = [entry for entry in logs if entry["event"] == "signals.pages"]
    assert (line["tenant_id"], line["pages"], line["unmatched_query_urls"]) == ("acme", 1, 1)
    assert not any("example.com" in str(value) for value in line.values()), "no urls in logs"


async def test_a_tenant_with_no_pages_has_no_signals() -> None:
    stores = FakeStores(pages=[], keywords=[], queries=[(url("/a"), "tents")])
    assert await load(stores) == ({}, 1)


async def seed(graph: GraphRepo, mongo: MongoRepo, tenant: str, query: str) -> None:
    await graph.upsert_pages(tenant, [Page(url=url(p), status_code=200) for p in ("/a", "/b")])
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) "
        "SET p.linkCommunityId = 0, p.keywordCommunityId = 3, p.contentCommunityId = 1, "
        "p.hubId = -1",
        t=tenant,
        u=url("/a"),
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row[0]}) "
        "MERGE (k:Keyword {tenantId: $t, text: row[1], language: 'en'}) "
        "MERGE (p)-[:TARGETS_KEYWORD {source: row[2]}]->(k)",
        t=tenant,
        rows=[
            [url("/a"), "Trail Shoes", "CLIENT_STRATEGIC"],
            [url("/a"), "rain jacket", "CLIENT_STRATEGIC"],
            [url("/a"), "tents", "GSC_OBSERVED"],
            [url("/ghost"), "stoves", "CLIENT_STRATEGIC"],
        ],
    )
    await mongo._db["gsc_queries"].insert_many(
        [
            {"tenantId": tenant, "url": url("/a"), "query": query, "impressions": 90},
            {"tenantId": tenant, "url": url("/a"), "query": "TENTS", "impressions": 40},
            {"tenantId": tenant, "url": url("/ghost"), "query": "stoves", "impressions": 5},
            {"tenantId": tenant, "url": url("/elsewhere"), "query": "tents", "impressions": 3},
        ]
    )


@pytest.mark.integration
async def test_page_signals_come_from_both_stores_of_one_tenant_only(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, mongo, tenant, "trail  SHOES")
    await seed(graph, mongo, other, "rain jacket")

    pages, unmatched = await load_page_signals(graph, mongo, tenant)

    assert sorted(pages) == [url("/a"), url("/b")], "placeholders get no signals"
    a, b = pages[url("/a")], pages[url("/b")]
    assert a.queries == {"trail shoes", "tents"}
    assert a.keywords == {"trail shoes", "rain jacket", "tents"}
    # Strategic keywords without a query: only "rain jacket" (the other tenant has that query).
    assert a.keyword_gap == 1
    assert (a.link_community_id, a.keyword_community_id, a.content_community_id, a.hub_id) == (
        0,
        3,
        1,
        -1,
    )
    assert (b.queries, b.keywords, b.keyword_gap, b.link_community_id) == (
        frozenset(),
        frozenset(),
        0,
        None,
    )
    # The placeholder and the unknown url.
    assert unmatched == 2
