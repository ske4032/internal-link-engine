from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError
from linking_engine.graph.repo import GraphRepo
from linking_engine.models import KeywordSource, Page

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

STRATEGIC, OBSERVED, INFERRED = (
    KeywordSource.CLIENT_STRATEGIC,
    KeywordSource.GSC_OBSERVED,
    KeywordSource.INFERRED,
)


def url(path: str) -> str:
    return f"example.com{path}"


async def seed_pages(
    graph: GraphRepo, tenant: str, crawled: list[str], placeholders: list[str]
) -> None:
    await graph.upsert_pages(tenant, [Page(url=url(p), status_code=200) for p in crawled])
    await graph.upsert_placeholders(tenant, [url(p) for p in placeholders])


async def target(
    graph: GraphRepo,
    tenant: str,
    path: str,
    text: str,
    source: str | None,
    *,
    keyword_tenant: str | None = None,
) -> None:
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) "
        "MERGE (k:Keyword {tenantId: $kt, text: $text, language: 'en'}) "
        "MERGE (p)-[e:TARGETS_KEYWORD]->(k) SET e.source = $source",
        t=tenant,
        u=url(path),
        kt=keyword_tenant or tenant,
        text=text,
        source=source,
    )


@pytest.mark.integration
async def test_keyword_edges_are_the_crawled_pages_edges_with_their_source(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_pages(graph, tenant, ["/a", "/b", "/none"], ["/ghost"])
    for path, text, source in (
        ("/a", "trail shoes", STRATEGIC),
        ("/a", "Rain  Jacket", OBSERVED),
        ("/b", "trail shoes", INFERRED),
        ("/ghost", "tents", STRATEGIC),
    ):
        await target(graph, tenant, path, text, source)

    edges = await graph.keyword_edges(tenant)

    # Texts come back verbatim: normalising is the signal stage's job.
    assert sorted(edges) == [
        (url("/a"), "Rain  Jacket", OBSERVED),
        (url("/a"), "trail shoes", STRATEGIC),
        (url("/b"), "trail shoes", INFERRED),
    ]
    assert all(type(source) is KeywordSource for _, _, source in edges)


@pytest.mark.integration
async def test_keyword_edges_never_cross_tenants(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await seed_pages(graph, tenant, ["/a"], [])
    await seed_pages(graph, other, ["/a", "/b"], [])
    await target(graph, tenant, "/a", "trail shoes", STRATEGIC)
    await target(graph, other, "/a", "trail shoes", STRATEGIC)
    await target(graph, other, "/b", "tents", OBSERVED)
    # Edges that should not exist: each end of one belongs to the other tenant.
    await target(graph, tenant, "/a", "stoves", STRATEGIC, keyword_tenant=other)
    await target(graph, other, "/a", "lanterns", INFERRED, keyword_tenant=tenant)

    assert await graph.keyword_edges(tenant) == [(url("/a"), "trail shoes", STRATEGIC)]
    assert sorted(await graph.keyword_edges(other)) == [
        (url("/a"), "trail shoes", STRATEGIC),
        (url("/b"), "tents", OBSERVED),
    ]


@pytest.mark.integration
async def test_a_tenant_without_keywords_has_no_edges(graph: GraphRepo, tenant: str) -> None:
    await seed_pages(graph, tenant, ["/a"], [])
    assert await graph.keyword_edges(tenant) == []


@pytest.mark.parametrize("source", ["MANUAL", "client_strategic", None])
@pytest.mark.integration
async def test_an_edge_with_an_unknown_source_fails_the_read(
    graph: GraphRepo, tenant: str, source: str | None
) -> None:
    await seed_pages(graph, tenant, ["/a"], [])
    await target(graph, tenant, "/a", "trail shoes", STRATEGIC)
    await target(graph, tenant, "/a", "tents", source)

    with pytest.raises(DatabaseReadError, match="keyword edges"):
        await graph.keyword_edges(tenant)


@pytest.mark.integration
async def test_a_keyword_without_text_fails_the_read(graph: GraphRepo, tenant: str) -> None:
    await seed_pages(graph, tenant, ["/a"], [])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) "
        "CREATE (p)-[:TARGETS_KEYWORD {source: 'CLIENT_STRATEGIC'}]->"
        "(:Keyword {tenantId: $t, language: 'en'})",
        t=tenant,
        u=url("/a"),
    )

    with pytest.raises(DatabaseReadError):
        await graph.keyword_edges(tenant)


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
        await offline_graph.keyword_edges(tenant_id)
