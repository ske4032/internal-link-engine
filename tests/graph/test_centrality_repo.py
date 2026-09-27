from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseWriteError
from linking_engine.graph.repo import GraphRepo
from linking_engine.models import Page, PageCentrality

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

BASE = "example.com"


def url(path: str) -> str:
    return f"{BASE}{path}"


def score(path: str, rank: float = 0.25, betweenness: float = 3.0) -> PageCentrality:
    return PageCentrality(
        url=url(path),
        page_rank=rank,
        page_rank_percentile=0.5,
        betweenness=betweenness,
        betweenness_percentile=0.25,
    )


async def seed(graph: GraphRepo, tenant: str, crawled: list[str], placeholders: list[str]) -> None:
    await graph.upsert_pages(tenant, [Page(url=url(p), status_code=200) for p in crawled])
    await graph.upsert_placeholders(tenant, [url(p) for p in placeholders])


async def stored(graph: GraphRepo, tenant: str, path: str) -> Page:
    [page] = await graph.get_pages(tenant, [url(path)])
    return page


@pytest.mark.integration
async def test_scores_are_written_in_chunks_and_read_back(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant, ["/a", "/b", "/c"], [])

    written = await graph.write_centrality(
        tenant, [score("/a", 0.5), score("/b", 0.3), score("/c", 0.2)], batch_size=2
    )

    assert written == 3
    page = await stored(graph, tenant, "/a")
    assert (page.page_rank, page.page_rank_percentile) == (0.5, 0.5)
    assert (page.betweenness, page.betweenness_percentile) == (3.0, 0.25)


@pytest.mark.parametrize("target", ["/ghost", "/missing"])
@pytest.mark.integration
async def test_a_row_for_a_placeholder_or_missing_page_rolls_back_every_chunk(
    graph: GraphRepo, tenant: str, target: str
) -> None:
    await seed(graph, tenant, ["/a"], ["/ghost"])

    with pytest.raises(DatabaseWriteError, match="rolled back"):
        await graph.write_centrality(tenant, [score("/a"), score(target)], batch_size=1)

    assert (await stored(graph, tenant, "/a")).page_rank is None


@pytest.mark.integration
async def test_a_page_that_became_a_placeholder_loses_its_scores(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a"], ["/gone"])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) "
        "SET p.pageRank = 0.4, p.pageRankPercentile = 0.9, "
        "p.betweenness = 2.0, p.betweennessPercentile = 0.8",
        t=tenant,
        u=url("/gone"),
    )

    await graph.write_centrality(tenant, [score("/a")])

    gone = await stored(graph, tenant, "/gone")
    assert (gone.page_rank, gone.page_rank_percentile) == (None, None)
    assert (gone.betweenness, gone.betweenness_percentile) == (None, None)


@pytest.mark.integration
async def test_scores_never_reach_another_tenant(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant, ["/a"], [])
    await seed(graph, other, ["/a"], [])

    await graph.write_centrality(tenant, [score("/a")])

    assert (await stored(graph, other, "/a")).page_rank is None
    with pytest.raises(DatabaseWriteError):
        await graph.write_centrality(other, [score("/b")])


@pytest.mark.integration
async def test_urls_are_matched_verbatim_as_stored(graph: GraphRepo, tenant: str) -> None:
    await graph._auto("CREATE (:Page {tenantId: $t, url: '/legacy-path'})", t=tenant)

    assert (
        await graph.write_centrality(
            tenant, [score("/a").model_copy(update={"url": "/legacy-path"})]
        )
        == 1
    )


@pytest.mark.integration
async def test_no_scores_still_clears_placeholders(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant, [], ["/gone"])
    await graph._auto("MATCH (p:Page {tenantId: $t}) SET p.pageRank = 0.4", t=tenant)

    assert await graph.write_centrality(tenant, []) == 0
    assert (await stored(graph, tenant, "/gone")).page_rank is None


@pytest.fixture
async def offline_graph() -> AsyncIterator[GraphRepo]:
    """A repo whose server does not exist: any query would raise DatabaseUnavailableError."""
    driver = AsyncGraphDatabase.driver(
        "bolt://127.0.0.1:1", auth=("neo4j", "x"), connection_timeout=1
    )
    repo = GraphRepo(driver)
    yield repo
    await repo.close()


@pytest.mark.parametrize(
    ("tenant_id", "rows", "batch_size", "message"),
    [
        (" ", [score("/a")], 10, "tenant_id"),
        ("t", [score("/a"), score("/a")], 10, "duplicate urls"),
        ("t", [score("/a")], 0, "batch_size"),
    ],
    ids=["blank-tenant", "duplicate-url", "zero-batch"],
)
async def test_invalid_input_is_rejected_before_any_write(
    offline_graph: GraphRepo,
    tenant_id: str,
    rows: list[PageCentrality],
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.write_centrality(tenant_id, rows, batch_size=batch_size)
