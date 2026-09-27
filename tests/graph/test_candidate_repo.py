"""Candidate retrieval reads: target selection and the tenant's vector pool."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
import pytest
from neo4j import READ_ACCESS, AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError
from linking_engine.graph import repo as graph_repo
from linking_engine.graph.repo import GraphRepo
from linking_engine.models import CandidateTarget, Page, VectorIndex

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

BASE = "example.com"


def url(path: str) -> str:
    return f"{BASE}{path}"


async def add_pages(graph: GraphRepo, tenant: str, paths: Sequence[str], **fields: object) -> None:
    await graph.upsert_pages(
        tenant,
        [Page.model_validate({"url": url(p), "status_code": 200, **fields}) for p in paths],
    )


async def set_vector(
    graph: GraphRepo,
    tenant: str,
    path: str,
    vector: Sequence[float],
    prop: str = "content_embedding",
) -> None:
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) CALL db.create.setNodeVectorProperty(p, $prop, $v)",
        t=tenant,
        u=url(path),
        prop=prop,
        v=list(vector),
    )


@pytest.mark.integration
async def test_targets_are_indexable_crawled_pages_with_a_vector_ordered_by_url(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await add_pages(graph, tenant, ["/flagged", "/bare"], is_indexable=True)
    await add_pages(graph, tenant, ["/flagged-404"], is_indexable=True, status_code=404)
    await add_pages(graph, tenant, ["/noindex"], is_indexable=False)
    await add_pages(graph, tenant, ["/assumed", "/assumed-bare"])
    await add_pages(graph, tenant, ["/redirect"], status_code=301)
    await add_pages(graph, tenant, ["/unknown-status"], status_code=None)
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    for path in ("/flagged", "/flagged-404", "/noindex", "/assumed", "/redirect", "/ghost"):
        await set_vector(graph, tenant, path, [1.0, 0.0])
    await add_pages(graph, other, ["/bare", "/other-only"])
    for path in ("/bare", "/other-only"):
        await set_vector(graph, other, path, [1.0, 0.0])

    selection = await graph.candidate_targets(tenant)

    # The flag wins over the status; without a flag only a 2xx page counts as indexable.
    assert selection.targets == (
        CandidateTarget(url=url("/assumed"), indexable_assumed=True),
        CandidateTarget(url=url("/flagged"), indexable_assumed=False),
        CandidateTarget(url=url("/flagged-404"), indexable_assumed=False),
    )
    # Not indexable: /noindex, /redirect, /unknown-status. No vector: /bare, /assumed-bare.
    assert (selection.crawled_pages, selection.not_indexable, selection.without_vector) == (8, 3, 2)


@pytest.mark.integration
async def test_targets_need_a_vector_in_the_chosen_index(graph: GraphRepo, tenant: str) -> None:
    await add_pages(graph, tenant, ["/both", "/content-only", "/gnn-only"])
    await set_vector(graph, tenant, "/both", [1.0, 0.0])
    await set_vector(graph, tenant, "/both", [0.0, 1.0], prop="gnn_embedding")
    await set_vector(graph, tenant, "/content-only", [1.0, 0.0])
    await set_vector(graph, tenant, "/gnn-only", [0.0, 1.0], prop="gnn_embedding")

    content = await graph.candidate_targets(tenant)
    gnn = await graph.candidate_targets(tenant, index="page_gnn")

    assert [t.url for t in content.targets] == [url("/both"), url("/content-only")]
    assert [t.url for t in gnn.targets] == [url("/both"), url("/gnn-only")]
    assert content.without_vector == gnn.without_vector == 1


@pytest.mark.integration
async def test_a_tenant_without_pages_has_no_targets(graph: GraphRepo, tenant: str) -> None:
    selection = await graph.candidate_targets(tenant)

    assert (selection.crawled_pages, selection.targets) == (0, ())


@pytest.mark.integration
async def test_page_vectors_are_the_tenants_crawled_pages_in_the_index_property(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await add_pages(graph, tenant, ["/a", "/b", "/c", "/none"])
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await set_vector(graph, tenant, "/a", [1.0, 0.5])
    await set_vector(graph, tenant, "/b", [0.25, 1.0])
    await set_vector(graph, tenant, "/b", [0.5, 0.5], prop="gnn_embedding")
    await set_vector(graph, tenant, "/c", [0.0, 1.0], prop="gnn_embedding")
    await set_vector(graph, tenant, "/ghost", [1.0, 1.0])
    await set_vector(graph, tenant, "/ghost", [1.0, 1.0], prop="gnn_embedding")
    await add_pages(graph, other, ["/a", "/other-only"])
    await set_vector(graph, other, "/a", [-1.0, 0.0])
    await set_vector(graph, other, "/other-only", [-1.0, 0.0])

    content = await graph.page_vectors(tenant)
    gnn = await graph.page_vectors(tenant, index="page_gnn")

    assert {u: v.tolist() for u, v in content.items()} == {
        url("/a"): [1.0, 0.5],
        url("/b"): [0.25, 1.0],
    }
    assert {u: v.tolist() for u, v in gnn.items()} == {
        url("/b"): [0.5, 0.5],
        url("/c"): [0.0, 1.0],
    }
    assert all(v.dtype == np.float32 for v in (*content.values(), *gnn.values()))
    assert (await graph.content_vectors(tenant)).keys() == content.keys()


@pytest.mark.parametrize("batch_size", [1, 2, 5, 6])
@pytest.mark.integration
async def test_page_vectors_are_paged_by_url(
    graph: GraphRepo, tenant: str, batch_size: int
) -> None:
    paths = ["/e", "/a", "/d", "/b", "/c"]
    await add_pages(graph, tenant, paths)
    for index, path in enumerate(paths):
        await set_vector(graph, tenant, path, [float(index), 1.0])

    vectors = await graph.page_vectors(tenant, batch_size=batch_size)

    assert list(vectors) == sorted(url(p) for p in paths)
    assert vectors[url("/e")].tolist() == [0.0, 1.0]


@pytest.mark.integration
async def test_page_vectors_read_the_index_in_url_order_without_sorting(graph: GraphRepo) -> None:
    # A Sort or Top would load every remaining vector of the tenant for each page.
    async with graph._driver.session(default_access_mode=READ_ACCESS) as session:
        result = await session.run(
            "EXPLAIN " + graph_repo._PAGE_VECTORS,
            tenant="t",
            property="content_embedding",
            after="",
            limit=500,
        )
        summary = await result.consume()
    assert summary.plan is not None
    operators: list[str] = []
    pending = [summary.plan]
    while pending:
        step = pending.pop()
        operators.append(step["operatorType"].split("@")[0])
        pending.extend(step.get("children", []))

    assert "NodeUniqueIndexSeek" in operators, operators
    assert not {"Sort", "Top", "PartialSort", "PartialTop"} & set(operators), operators


@pytest.mark.parametrize("index", ["page_content", "page_gnn"])
@pytest.mark.integration
async def test_a_malformed_stored_vector_fails_the_read(
    graph: GraphRepo, tenant: str, index: str
) -> None:
    await add_pages(graph, tenant, ["/a"])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) SET p.content_embedding = 'x', p.gnn_embedding = 'x'",
        t=tenant,
    )

    with pytest.raises(DatabaseReadError, match="no stored vector"):
        await graph.page_vectors(tenant, index=index)  # type: ignore[arg-type]


@pytest.mark.integration
async def test_a_stored_target_that_does_not_fit_the_model_fails_the_read(
    graph: GraphRepo, tenant: str
) -> None:
    await graph._auto(
        "CREATE (:Page {tenantId: $t, url: '', statusCode: 200, content_embedding: [1.0]})",
        t=tenant,
    )

    with pytest.raises(DatabaseReadError, match="candidate targets"):
        await graph.candidate_targets(tenant)


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
    ("tenant_id", "index", "batch_size", "message"),
    [
        (" ", "page_content", 10, "tenant_id"),
        ("t", "page_sage", 10, "unknown vector index"),
        ("t", "page_gnn", 0, "batch_size"),
    ],
    ids=["blank-tenant", "unknown-index", "zero-batch"],
)
async def test_invalid_vector_reads_are_rejected_before_any_query(
    offline_graph: GraphRepo, tenant_id: str, index: str, batch_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.page_vectors(
            tenant_id,
            index=index,  # type: ignore[arg-type]
            batch_size=batch_size,
        )


@pytest.mark.parametrize(
    ("tenant_id", "index", "message"),
    [(" ", "page_content", "tenant_id"), ("t", "page_sage", "unknown vector index")],
    ids=["blank-tenant", "unknown-index"],
)
async def test_invalid_target_selections_are_rejected_before_any_query(
    offline_graph: GraphRepo, tenant_id: str, index: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.candidate_targets(tenant_id, index=index)  # type: ignore[arg-type]


def test_every_vector_index_the_models_accept_has_a_property() -> None:
    assert set(get_args(VectorIndex)) == set(graph_repo.VECTOR_PROPERTIES)
