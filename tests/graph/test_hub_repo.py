from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError, DatabaseWriteError
from linking_engine.graph.repo import VECTOR_DIMENSIONS, GraphRepo
from linking_engine.models import HubCentroid, Page, PageHub

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

BASE = "example.com"


def url(path: str) -> str:
    return f"{BASE}{path}"


def centroid(axis: int) -> tuple[float, ...]:
    vector = np.zeros(VECTOR_DIMENSIONS)
    vector[axis] = 1.0
    return tuple(vector.tolist())


def hub(hub_id: int, size: int = 2, pillar: str | None = None) -> HubCentroid:
    return HubCentroid(hub_id=hub_id, size=size, centroid=centroid(hub_id), pillar_url=pillar)


def page(path: str, **fields: Any) -> PageHub:
    return PageHub(url=url(path), **fields)


async def seed(graph: GraphRepo, tenant: str, crawled: list[str], placeholders: list[str]) -> None:
    await graph.upsert_pages(tenant, [Page(url=url(p), status_code=200) for p in crawled])
    await graph.upsert_placeholders(tenant, [url(p) for p in placeholders])


async def stored(graph: GraphRepo, tenant: str, path: str) -> Page:
    [found] = await graph.get_pages(tenant, [url(path)])
    return found


async def hub_nodes(graph: GraphRepo, tenant: str) -> dict[int, dict[str, Any]]:
    rows = await graph._read(
        "MATCH (h:Hub {tenantId: $t}) RETURN h.hubId AS id, h.active AS active, h.size AS size, "
        "h.pillarUrl AS pillar, size(h.centroid) AS dims",
        t=tenant,
    )
    return {int(r["id"]): r for r in rows}


@pytest.mark.integration
async def test_pages_and_hubs_are_written_together(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant, ["/a", "/b", "/noise", "/no-vector"], [])
    pages = [
        page("/a", hub_id=0, is_hub_pillar=True),
        page("/b", hub_id=0),
        page("/noise", hub_id=-1),
        page("/no-vector"),
    ]

    assert await graph.write_hubs(tenant, pages, [hub(0, pillar=url("/a"))], batch_size=2) == 4

    a = await stored(graph, tenant, "/a")
    noise = await stored(graph, tenant, "/noise")
    bare = await stored(graph, tenant, "/no-vector")
    assert (a.hub_id, a.is_hub_pillar) == (0, True)
    assert noise.hub_id == -1
    assert bare.hub_id is None
    node = (await hub_nodes(graph, tenant))[0]
    assert (node["active"], node["size"], node["pillar"], node["dims"]) == (
        True,
        2,
        url("/a"),
        VECTOR_DIMENSIONS,
    )


@pytest.mark.integration
async def test_a_hub_no_longer_found_is_retired_and_its_id_never_reused(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a", "/b"], [])
    await graph.write_hubs(tenant, [page("/a", hub_id=0), page("/b", hub_id=1)], [hub(0), hub(1)])

    await graph.write_hubs(tenant, [page("/a", hub_id=0), page("/b", hub_id=-1)], [hub(0)])

    nodes = await hub_nodes(graph, tenant)
    assert (nodes[1]["active"], nodes[1]["size"], nodes[1]["pillar"]) == (False, 0, None)
    active, next_id = await graph.stored_hubs(tenant)
    assert (sorted(active), next_id) == ([0], 2)
    assert active[0].shape == (VECTOR_DIMENSIONS,)


@pytest.mark.parametrize("target", ["/ghost", "/missing"])
@pytest.mark.integration
async def test_a_row_for_a_placeholder_or_missing_page_rolls_back_pages_and_hubs(
    graph: GraphRepo, tenant: str, target: str
) -> None:
    await seed(graph, tenant, ["/a"], ["/ghost"])

    with pytest.raises(DatabaseWriteError, match="rolled back"):
        await graph.write_hubs(tenant, [page("/a", hub_id=0), page(target, hub_id=0)], [hub(0)])

    assert (await stored(graph, tenant, "/a")).hub_id is None
    assert await hub_nodes(graph, tenant) == {}


@pytest.mark.integration
async def test_placeholders_keep_no_hub_and_tenants_share_nothing(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant, ["/a"], ["/gone"])
    await seed(graph, other, ["/a"], [])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.hubId = 3, p.isHubPillar = true",
        t=tenant,
        u=url("/gone"),
    )

    await graph.write_hubs(tenant, [page("/a", hub_id=0)], [hub(0)])

    gone = await stored(graph, tenant, "/gone")
    assert (gone.hub_id, gone.is_hub_pillar) == (None, None)
    assert (await stored(graph, other, "/a")).hub_id is None
    assert await graph.stored_hubs(other) == ({}, 0)


@pytest.mark.integration
async def test_a_hub_without_a_centroid_fails_the_read(graph: GraphRepo, tenant: str) -> None:
    await graph._auto("CREATE (:Hub {tenantId: $t, hubId: 0, active: true})", t=tenant)
    with pytest.raises(DatabaseReadError, match="no centroid"):
        await graph.stored_hubs(tenant)


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
    ("tenant_id", "pages", "hubs", "batch_size", "message"),
    [
        (" ", [page("/a")], [], 10, "tenant_id"),
        ("t", [page("/a"), page("/a")], [], 10, "duplicate urls"),
        ("t", [page("/a")], [hub(0), hub(0)], 10, "duplicate hub ids"),
        ("t", [page("/a")], [HubCentroid(hub_id=0, size=1, centroid=(1.0, 0.0))], 10, "dimensions"),
        ("t", [page("/a")], [], 0, "batch_size"),
    ],
    ids=["blank-tenant", "duplicate-url", "duplicate-hub", "short-centroid", "zero-batch"],
)
async def test_invalid_writes_are_rejected_before_any_query(
    offline_graph: GraphRepo,
    tenant_id: str,
    pages: list[PageHub],
    hubs: list[HubCentroid],
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.write_hubs(tenant_id, pages, hubs, batch_size=batch_size)


async def test_stored_hubs_rejects_a_blank_tenant(offline_graph: GraphRepo) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_graph.stored_hubs(" ")


def test_a_pillar_must_sit_in_a_hub() -> None:
    for hub_id in (None, -1):
        with pytest.raises(ValueError, match="must belong to a hub"):
            page("/a", hub_id=hub_id, is_hub_pillar=True)
