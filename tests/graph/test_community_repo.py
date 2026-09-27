from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError, DatabaseUnavailableError, DatabaseWriteError
from linking_engine.graph.repo import GraphRepo
from linking_engine.models import OrphanLabel, Page, PageCommunities

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

BASE = "example.com"


def url(path: str) -> str:
    return f"{BASE}{path}"


def row(path: str, **fields: Any) -> PageCommunities:
    values: dict[str, Any] = {"url": url(path), "is_orphan": False, "is_dead_end": False, **fields}
    return PageCommunities(**values)


async def seed(graph: GraphRepo, tenant: str, crawled: list[str], placeholders: list[str]) -> None:
    await graph.upsert_pages(tenant, [Page(url=url(p), status_code=200) for p in crawled])
    await graph.upsert_placeholders(tenant, [url(p) for p in placeholders])


async def stored(graph: GraphRepo, tenant: str, path: str) -> Page:
    [page] = await graph.get_pages(tenant, [url(path)])
    return page


@pytest.mark.integration
async def test_communities_flags_and_labels_are_written_and_read_back(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a", "/b"], [])
    rows = [
        row("/a", link_community_id=0, content_community_id=3, is_link_pillar=True),
        row("/b", is_orphan=True, is_dead_end=True, orphan_label=OrphanLabel.FOOTER_ONLY),
    ]

    assert await graph.write_communities(tenant, rows, batch_size=1) == 2

    a, b = await stored(graph, tenant, "/a"), await stored(graph, tenant, "/b")
    assert (a.link_community_id, a.content_community_id, a.is_link_pillar) == (0, 3, True)
    assert (a.keyword_community_id, a.is_orphan) == (None, False)
    assert (b.is_orphan, b.is_dead_end, b.orphan_label) == (True, True, OrphanLabel.FOOTER_ONLY)


@pytest.mark.integration
async def test_a_page_that_lost_its_community_keeps_no_stale_id(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a"], [])
    await graph.write_communities(tenant, [row("/a", keyword_community_id=4)])

    await graph.write_communities(tenant, [row("/a")])

    assert (await stored(graph, tenant, "/a")).keyword_community_id is None


@pytest.mark.parametrize("target", ["/ghost", "/missing"])
@pytest.mark.integration
async def test_a_row_for_a_placeholder_or_missing_page_rolls_back_everything(
    graph: GraphRepo, tenant: str, target: str
) -> None:
    await seed(graph, tenant, ["/a"], ["/ghost"])

    with pytest.raises(DatabaseWriteError, match="rolled back"):
        await graph.write_communities(
            tenant, [row("/a", link_community_id=1), row(target)], batch_size=1
        )

    assert (await stored(graph, tenant, "/a")).link_community_id is None


@pytest.mark.integration
async def test_a_page_that_became_a_placeholder_loses_its_communities(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a"], ["/gone"])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.linkCommunityId = 2, p.isOrphan = true, "
        "p.orphanLabel = 'NOT_LINKED', p.isLinkPillar = true",
        t=tenant,
        u=url("/gone"),
    )

    await graph.write_communities(tenant, [row("/a")])

    gone = await stored(graph, tenant, "/gone")
    assert (gone.link_community_id, gone.is_orphan, gone.orphan_label, gone.is_link_pillar) == (
        None,
        None,
        None,
        None,
    )


@pytest.mark.integration
async def test_communities_never_reach_another_tenant(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant, ["/a"], [])
    await seed(graph, other, ["/a"], [])

    await graph.write_communities(tenant, [row("/a", link_community_id=1)])

    assert (await stored(graph, other, "/a")).link_community_id is None


@pytest.mark.integration
async def test_keyword_targets_are_the_tenants_crawled_pages_only(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant, ["/a"], ["/ghost"])
    await seed(graph, other, ["/a"], [])
    for t, path in ((tenant, "/a"), (tenant, "/ghost"), (other, "/a")):
        await graph._auto(
            "MATCH (p:Page {tenantId: $t, url: $u}) "
            "MERGE (k:Keyword {tenantId: $t, text: 'trail shoes', language: 'en'}) "
            "MERGE (p)-[:TARGETS_KEYWORD]->(k)",
            t=t,
            u=url(path),
        )

    assert await graph.keyword_targets(tenant) == [(url("/a"), "trail shoes", "en")]


@pytest.mark.integration
async def test_content_vectors_are_paged_and_skip_pages_without_one(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a", "/b", "/c", "/none"], ["/ghost"])
    for path, vec in (
        ("/a", [1.0, 0.0]),
        ("/b", [0.0, 1.0]),
        ("/c", [0.5, 0.5]),
        ("/ghost", [1.0, 1.0]),
    ):
        await graph._auto(
            "MATCH (p:Page {tenantId: $t, url: $u}) SET p.content_embedding = $v",
            t=tenant,
            u=url(path),
            v=vec,
        )

    vectors = await graph.content_vectors(tenant, batch_size=1)

    assert sorted(vectors) == [url("/a"), url("/b"), url("/c")]
    assert vectors[url("/c")].tolist() == [0.5, 0.5]


@pytest.mark.integration
async def test_community_context_reads_template_counts_and_previous_ids(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a", "/b"], ["/ghost"])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.menuInlinks = 3, p.linkCommunityId = 5",
        t=tenant,
        u=url("/a"),
    )

    context = {c.url: c for c in await graph.community_context(tenant)}

    assert sorted(context) == [url("/a"), url("/b")]
    a, b = context[url("/a")], context[url("/b")]
    assert (a.menu_inlinks, a.footer_inlinks, a.link_community_id) == (3, 0, 5)
    assert (b.menu_inlinks, b.link_community_id, b.content_community_id) == (0, None, None)


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
        (" ", [row("/a")], 10, "tenant_id"),
        ("t", [row("/a"), row("/a")], 10, "duplicate urls"),
        ("t", [row("/a")], 0, "batch_size"),
    ],
    ids=["blank-tenant", "duplicate-url", "zero-batch"],
)
async def test_invalid_writes_are_rejected_before_any_query(
    offline_graph: GraphRepo,
    tenant_id: str,
    rows: list[PageCommunities],
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.write_communities(tenant_id, rows, batch_size=batch_size)


@pytest.mark.parametrize("read", ["keyword_targets", "content_vectors", "community_context"])
async def test_reads_reject_a_blank_tenant(offline_graph: GraphRepo, read: str) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await getattr(offline_graph, read)(" ")


async def test_content_vectors_reject_a_zero_batch(offline_graph: GraphRepo) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        await offline_graph.content_vectors("t", batch_size=0)


def test_rows_keep_labels_and_pillars_consistent() -> None:
    with pytest.raises(ValueError, match="exactly when the page is an orphan"):
        row("/a", orphan_label=OrphanLabel.NOT_LINKED)
    with pytest.raises(ValueError, match="exactly when the page is an orphan"):
        row("/a", is_orphan=True)
    with pytest.raises(ValueError, match="must belong to a community"):
        row("/a", is_content_pillar=True)


async def test_an_unreachable_server_surfaces_as_unavailable(offline_graph: GraphRepo) -> None:
    with pytest.raises(DatabaseUnavailableError):
        await offline_graph.write_communities("t", [row("/a")])


@pytest.mark.integration
async def test_malformed_stored_values_fail_the_read(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant, ["/a"], [])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) SET p.content_embedding = 'x', p.menuInlinks = -1", t=tenant
    )

    with pytest.raises(DatabaseReadError, match="no stored vector"):
        await graph.content_vectors(tenant)
    with pytest.raises(DatabaseReadError, match="community context"):
        await graph.community_context(tenant)
