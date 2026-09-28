"""Each crawled page's ranked keyword set, from its rank-carrying TARGETS_KEYWORD edges."""

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


async def seed(graph: GraphRepo, tenant: str, primary: str) -> None:
    await graph.upsert_pages(
        tenant, [Page(url=url(p), status_code=200) for p in ("/a", "/b", "/c")]
    )
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await graph.replace_keyword_targets(
        tenant,
        STRATEGIC,
        [
            target("/a", "tents", STRATEGIC, rank=2),
            target("/a", primary, STRATEGIC, rung=KeywordRung.STRATEGIC),
            # An unresolved page's strategic keyword has no rank: not in any set.
            target("/c", "stoves", STRATEGIC),
        ],
    )
    await graph.replace_keyword_targets(
        tenant, OBSERVED, [target("/a", "waterproof trail shoes", OBSERVED, rank=3)]
    )
    await graph.replace_keyword_targets(
        tenant, INFERRED, [target("/b", "Four Season Tents", INFERRED, rung=KeywordRung.H1)]
    )
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) "
        "MERGE (k:Keyword {tenantId: $t, text: 'ghost keyword', language: 'en'}) "
        "MERGE (p)-[:TARGETS_KEYWORD {source: 'INFERRED', rung: 'H1', rank: 1, resolved: true}]->(k)",
        t=tenant,
        u=url("/ghost"),
    )


@pytest.mark.integration
async def test_each_crawled_page_reads_its_keywords_in_rank_order(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, other, "Other Tenant Keyword")
    await seed(graph, tenant, "Trail Shoes")

    found = await graph.ranked_keywords(tenant)

    assert found == {
        url("/a"): [
            (1, "Trail Shoes", STRATEGIC),
            (2, "tents", STRATEGIC),
            (3, "waterproof trail shoes", OBSERVED),
        ],
        url("/b"): [(1, "Four Season Tents", INFERRED)],
    }
    assert all(type(source) is KeywordSource for rows in found.values() for _, _, source in rows)
    assert (await graph.ranked_keywords(other))[url("/a")][0] == (
        1,
        "Other Tenant Keyword",
        STRATEGIC,
    )


@pytest.mark.integration
async def test_two_keywords_of_one_rank_on_a_page_fail_the_read(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, "Trail Shoes")
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) "
        "MERGE (k:Keyword {tenantId: $t, text: 'camping', language: 'en'}) "
        "MERGE (p)-[:TARGETS_KEYWORD {source: 'GSC_OBSERVED', rank: 2}]->(k)",
        t=tenant,
        u=url("/a"),
    )

    with pytest.raises(DatabaseReadError):
        await graph.ranked_keywords(tenant)


@pytest.mark.integration
async def test_a_tenant_without_ranked_keywords_has_none(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [Page(url=url("/a"), status_code=200)])
    assert await graph.ranked_keywords(tenant) == {}


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
        await offline_graph.ranked_keywords(tenant_id)
