"""The resolved keyword of each crawled page: its rank-1 TARGETS_KEYWORD edge with a rung."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError
from linking_engine.graph.repo import GraphRepo
from linking_engine.models import KeywordRung, KeywordSource, KeywordTarget, Page

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

STRATEGIC, OBSERVED, INFERRED = (
    KeywordSource.CLIENT_STRATEGIC,
    KeywordSource.GSC_OBSERVED,
    KeywordSource.INFERRED,
)


def url(path: str) -> str:
    return f"example.com{path}"


def target(
    path: str,
    text: str,
    source: KeywordSource,
    *,
    rung: KeywordRung | None = None,
    rank: int | None = None,
) -> KeywordTarget:
    return KeywordTarget(
        url=url(path), text=text, language="en", source=source, rung=rung, rank=rank
    )


async def seed(graph: GraphRepo, tenant: str, keyword: str) -> None:
    await graph.upsert_pages(
        tenant, [Page(url=url(p), status_code=200) for p in ("/a", "/b", "/c", "/d")]
    )
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await graph.replace_keyword_targets(
        tenant,
        STRATEGIC,
        [
            target("/a", keyword, STRATEGIC, rung=KeywordRung.STRATEGIC),
            target("/a", "tents", STRATEGIC, rank=2),
            # A strategic keyword on a page that resolved nothing: no rank, no rung.
            target("/c", "stoves", STRATEGIC),
        ],
    )
    await graph.replace_keyword_targets(
        tenant, OBSERVED, [target("/a", "waterproof trail shoes", OBSERVED, rank=3)]
    )
    await graph.replace_keyword_targets(
        tenant,
        INFERRED,
        [
            target("/b", "Four Season Tents", INFERRED, rung=KeywordRung.H1),
            target("/d", "Camp Stoves", INFERRED, rung=KeywordRung.TITLE),
        ],
    )
    # Written by hand: the repo never writes onto a placeholder.
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) "
        "MERGE (k:Keyword {tenantId: $t, text: 'ghost keyword', language: 'en'}) "
        "MERGE (p)-[:TARGETS_KEYWORD {source: 'INFERRED', rung: 'H1', rank: 1, resolved: true}]->(k)",
        t=tenant,
        u=url("/ghost"),
    )


@pytest.mark.integration
async def test_each_crawled_page_reads_its_rank_one_resolved_keyword(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, other, "Other Tenant Keyword")
    await seed(graph, tenant, "Trail Shoes")

    found = await graph.resolved_keywords(tenant)

    assert found == {
        url("/a"): ("Trail Shoes", KeywordRung.STRATEGIC),
        url("/b"): ("Four Season Tents", KeywordRung.H1),
        url("/d"): ("Camp Stoves", KeywordRung.TITLE),
    }
    assert all(type(rung) is KeywordRung for _, rung in found.values())
    assert (await graph.resolved_keywords(other))[url("/a")] == (
        "Other Tenant Keyword",
        KeywordRung.STRATEGIC,
    )


@pytest.mark.integration
async def test_a_tenant_without_keyword_edges_has_no_resolved_keywords(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(tenant, [Page(url=url("/a"), status_code=200)])
    assert await graph.resolved_keywords(tenant) == {}


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        pytest.param(
            "MERGE (k1:Keyword {tenantId: $t, text: 'tents', language: 'en'}) "
            "MERGE (k2:Keyword {tenantId: $t, text: 'stoves', language: 'en'}) "
            "MERGE (p)-[:TARGETS_KEYWORD {source: 'INFERRED', rung: 'H1', rank: 1}]->(k1) "
            "MERGE (p)-[:TARGETS_KEYWORD {source: 'CLIENT_STRATEGIC', rung: 'STRATEGIC', rank: 1}]->(k2)",
            "more than one resolved keyword",
            id="two-resolved",
        ),
        pytest.param(
            "MERGE (k:Keyword {tenantId: $t, text: 'tents', language: 'en'}) "
            "MERGE (p)-[:TARGETS_KEYWORD {source: 'INFERRED', rung: 'H2', rank: 1}]->(k)",
            "unknown keyword rung",
            id="unknown-rung",
        ),
        pytest.param(
            "MERGE (k:Keyword {tenantId: $t, text: '', language: 'en'}) "
            "MERGE (p)-[:TARGETS_KEYWORD {source: 'INFERRED', rung: 'H1', rank: 1}]->(k)",
            "keyword without text",
            id="empty-text",
        ),
        pytest.param(
            "CREATE (p)-[:TARGETS_KEYWORD {source: 'INFERRED', rung: 'H1', rank: 1}]->"
            "(:Keyword {tenantId: $t, language: 'en'})",
            "keyword without text",
            id="no-text",
        ),
    ],
)
@pytest.mark.integration
async def test_malformed_resolved_keyword_edges_fail_the_read(
    graph: GraphRepo, tenant: str, edges: str, message: str
) -> None:
    await graph.upsert_pages(tenant, [Page(url=url("/a"), status_code=200)])
    await graph._auto("MATCH (p:Page {tenantId: $t, url: $u}) " + edges, t=tenant, u=url("/a"))

    with pytest.raises(DatabaseReadError, match=message):
        await graph.resolved_keywords(tenant)


@pytest.fixture
async def offline_graph() -> AsyncIterator[GraphRepo]:
    """A repo whose server does not exist: any query would raise DatabaseUnavailableError."""
    driver = AsyncGraphDatabase.driver(
        "bolt://127.0.0.1:1", auth=("neo4j", "x"), connection_timeout=1
    )
    repo = GraphRepo(driver)
    yield repo
    await repo.close()


@pytest.mark.parametrize("tenant_id", ["", " "])
async def test_a_blank_tenant_is_rejected_before_any_query(
    offline_graph: GraphRepo, tenant_id: str
) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_graph.resolved_keywords(tenant_id)
