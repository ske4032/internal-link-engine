"""Embedding reads and writes on the graph: resume selection, model guard, vector flushes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError, DatabaseUnavailableError, DatabaseWriteError
from linking_engine.graph.repo import VECTOR_DIMENSIONS, GraphRepo
from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.models import EmbeddingModelCount, EmbeddingSelection, EmbeddingTarget, Page

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import numpy.typing as npt

BASE = "example.com"
MODEL = "voyage-4-large"
OTHER_MODEL = "voyage-3-large"
DIM = VECTOR_DIMENSIONS
# Container and host clocks can drift a little; embeddedAt only has to be this run's.
CLOCK_SLACK = timedelta(minutes=2)


def url(path: str) -> str:
    return f"{BASE}{path}"


def hash_of(path: str, version: int = 1) -> str:
    return body_hash(f"body of {path}, version {version}")


def page(path: str, *, version: int | None = 1, **fields: object) -> Page:
    digest = None if version is None else hash_of(path, version)
    return Page.model_validate(
        {"url": url(path), "status_code": 200, "word_count": 10, "body_hash": digest, **fields}
    )


def unit_vectors(count: int, seed: int) -> npt.NDArray[np.float32]:
    matrix = np.random.default_rng(seed).standard_normal((count, DIM))
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def seed_of(tenant: str) -> int:
    return int(tenant.rsplit("-", 1)[1], 16)


async def embed(
    graph: GraphRepo,
    tenant: str,
    paths: list[str],
    *,
    version: int = 1,
    model: str = MODEL,
    seed: int = 0,
) -> npt.NDArray[np.float32]:
    vectors = unit_vectors(len(paths), seed)
    written = await graph.write_embeddings(
        tenant,
        [url(p) for p in paths],
        [hash_of(p, version) for p in paths],
        vectors,
        model=model,
        dimensions=DIM,
    )
    assert written == len(paths)
    return vectors


async def pages_by_url(graph: GraphRepo, tenant: str, paths: list[str]) -> dict[str, Page]:
    stored = await graph.get_pages(tenant, [url(p) for p in paths], include_vectors=True)
    return {str(p.url): p for p in stored}


# ── embedding_selection ─────────────────────────────────────────────────────


@pytest.mark.integration
async def test_selection_classifies_every_page_kind(graph: GraphRepo, tenant: str) -> None:
    # Created in reverse url order, so an unordered result would come back reversed.
    await graph.upsert_pages(
        tenant,
        [
            page("/t-z-marker-without-vector"),
            page("/t-y-status-299", status_code=299),
            page("/t-x-null-hash", version=None),
            page("/t-w-changed"),
            page("/t-v-never"),
            page("/n-3xx", status_code=301),
            page("/n-4xx", status_code=404),
            page("/n-5xx", status_code=503),
            page("/n-null-status", status_code=None),
            page("/n-199", status_code=199),
            page("/n-300", status_code=300),
            page("/n-gone-after-embedding"),
            page("/u-current"),
        ],
    )
    await graph.upsert_placeholders(tenant, [url("/p-ghost")])
    await embed(graph, tenant, ["/u-current", "/t-w-changed", "/n-gone-after-embedding"])
    await graph.upsert_pages(
        tenant,
        [page("/t-w-changed", version=2), page("/n-gone-after-embedding", status_code=410)],
    )
    # The resume marker alone, without a stored vector, is not up to date.
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.embeddedBodyHash = p.bodyHash",
        t=tenant,
        u=url("/t-z-marker-without-vector"),
    )

    selection = await graph.embedding_selection(tenant)

    assert (selection.placeholders, selection.non_2xx, selection.up_to_date) == (1, 7, 1)
    assert selection.targets == (
        EmbeddingTarget(url=url("/t-v-never"), body_hash=hash_of("/t-v-never")),
        EmbeddingTarget(url=url("/t-w-changed"), body_hash=hash_of("/t-w-changed", 2)),
        EmbeddingTarget(url=url("/t-x-null-hash"), body_hash=None),
        EmbeddingTarget(url=url("/t-y-status-299"), body_hash=hash_of("/t-y-status-299")),
        EmbeddingTarget(
            url=url("/t-z-marker-without-vector"), body_hash=hash_of("/t-z-marker-without-vector")
        ),
    )


@pytest.mark.integration
async def test_selection_targets_are_ordered_by_url(graph: GraphRepo, tenant: str) -> None:
    paths = [f"/p{i:02d}" for i in range(25)]
    for path in reversed(paths):
        await graph.upsert_pages(tenant, [page(path)])
    selection = await graph.embedding_selection(tenant)
    assert [t.url for t in selection.targets] == [url(p) for p in paths]


@pytest.mark.integration
async def test_selection_of_an_empty_tenant_is_empty(graph: GraphRepo, tenant: str) -> None:
    assert await graph.embedding_selection(tenant) == EmbeddingSelection(
        targets=(), up_to_date=0, placeholders=0, non_2xx=0
    )


@pytest.mark.integration
async def test_selection_never_counts_another_tenants_pages(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await graph.upsert_pages(other, [page("/a"), page("/b"), page("/c", status_code=404)])
    await graph.upsert_placeholders(other, [url("/ghost")])
    await embed(graph, other, ["/b"])
    await graph.upsert_pages(tenant, [page("/a")])

    assert await graph.embedding_selection(tenant) == EmbeddingSelection(
        targets=(EmbeddingTarget(url=url("/a"), body_hash=hash_of("/a")),),
        up_to_date=0,
        placeholders=0,
        non_2xx=0,
    )
    assert await graph.embedding_selection(other) == EmbeddingSelection(
        targets=(EmbeddingTarget(url=url("/a"), body_hash=hash_of("/a")),),
        up_to_date=1,
        placeholders=1,
        non_2xx=1,
    )


@pytest.mark.integration
async def test_selection_of_a_target_that_does_not_fit_raises_read_error(
    graph: GraphRepo, tenant: str
) -> None:
    await graph._auto(
        "CREATE (:Page {tenantId: $t, url: $u, statusCode: 200, bodyHash: 42})",
        t=tenant,
        u=url("/corrupt"),
    )
    with pytest.raises(DatabaseReadError, match="embedding targets do not fit the model"):
        await graph.embedding_selection(tenant)


# ── embedding_models ────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_models_of_a_tenant_without_vectors(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a")])
    assert await graph.embedding_models(tenant) == ()


@pytest.mark.integration
async def test_models_with_one_model(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a"), page("/b"), page("/c")])
    await embed(graph, tenant, ["/a", "/b"])
    assert await graph.embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=MODEL, vectors=2),
    )


@pytest.mark.integration
async def test_models_with_two_models_are_ordered_by_name(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a"), page("/b"), page("/c")])
    await embed(graph, tenant, ["/a", "/b"], model=MODEL)
    await embed(graph, tenant, ["/c"], model=OTHER_MODEL, seed=1)
    assert await graph.embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=OTHER_MODEL, vectors=1),
        EmbeddingModelCount(embedding_model=MODEL, vectors=2),
    )


@pytest.mark.integration
async def test_a_vector_without_a_model_is_reported_last_as_none(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(tenant, [page("/a"), page("/raw")])
    await embed(graph, tenant, ["/a"])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.content_embedding = $vec",
        t=tenant,
        u=url("/raw"),
        vec=unit_vectors(1, 2)[0].tolist(),
    )
    assert await graph.embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=MODEL, vectors=1),
        EmbeddingModelCount(embedding_model=None, vectors=1),
    )


@pytest.mark.integration
async def test_a_model_value_that_does_not_fit_raises_read_error(
    graph: GraphRepo, tenant: str
) -> None:
    await graph._auto(
        "CREATE (:Page {tenantId: $t, url: $u, content_embedding: [1.0, 0.0], embeddingModel: 7})",
        t=tenant,
        u=url("/corrupt"),
    )
    with pytest.raises(DatabaseReadError, match="embedding model counts do not fit the model"):
        await graph.embedding_models(tenant)


@pytest.mark.integration
async def test_models_never_count_another_tenants_vectors(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    for t in (tenant, other):
        await graph.upsert_pages(t, [page("/a")])
    await embed(graph, tenant, ["/a"], model=MODEL)
    await embed(graph, other, ["/a"], model=OTHER_MODEL)
    assert await graph.embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=MODEL, vectors=1),
    )
    assert await graph.embedding_models(other) == (
        EmbeddingModelCount(embedding_model=OTHER_MODEL, vectors=1),
    )


# ── write_embeddings ────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_write_sets_vector_marker_model_dimensions_and_time(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(tenant, [page("/a"), page("/b")])
    before = datetime.now(UTC)
    vectors = await embed(graph, tenant, ["/a", "/b"], seed=seed_of(tenant))
    after = datetime.now(UTC)

    stored = await pages_by_url(graph, tenant, ["/a", "/b"])
    for index, path in enumerate(["/a", "/b"]):
        written = stored[url(path)]
        assert written.content_embedding is not None
        assert len(written.content_embedding) == DIM
        assert np.array_equal(
            np.asarray(written.content_embedding, dtype=np.float32), vectors[index]
        )
        assert written.embedded_body_hash == written.body_hash == hash_of(path)
        assert (written.embedding_model, written.embedding_dimensions) == (MODEL, DIM)
        assert written.embedded_at is not None
        assert written.embedded_at.utcoffset() == timedelta(0), written.embedded_at
        assert before - CLOCK_SLACK <= written.embedded_at <= after + CLOCK_SLACK
        assert written.gnn_embedding is None


@pytest.mark.integration
async def test_written_vector_is_found_by_the_vector_index(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a"), page("/b")])
    vectors = await embed(graph, tenant, ["/a", "/b"], seed=seed_of(tenant))
    await graph._auto("CALL db.awaitIndexes(60)")
    for vector, path in zip(vectors, ["/a", "/b"], strict=True):
        rows = await graph._auto(
            "CALL db.index.vector.queryNodes('page_content', 10, $vec) YIELD node, score "
            "WHERE node.tenantId = $t RETURN node.url AS url, score",
            vec=vector.tolist(),
            t=tenant,
        )
        assert rows, f"the page_content index returned nothing for {path}'s own vector"
        assert rows[0]["url"] == url(path), rows
        # The index quantizes vectors, so an identical vector scores just under 1.0.
        assert rows[0]["score"] > 0.999, rows


@pytest.mark.integration
async def test_rewrite_after_a_body_change_replaces_vector_and_marker(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(tenant, [page("/a")])
    await embed(graph, tenant, ["/a"], seed=1)
    [first] = (await pages_by_url(graph, tenant, ["/a"])).values()
    await graph.upsert_pages(tenant, [page("/a", version=2)])
    second_vectors = await embed(graph, tenant, ["/a"], version=2, model=OTHER_MODEL, seed=2)

    [second] = (await pages_by_url(graph, tenant, ["/a"])).values()
    assert second.content_embedding is not None
    assert np.array_equal(np.asarray(second.content_embedding, dtype=np.float32), second_vectors[0])
    assert second.content_embedding != first.content_embedding
    assert (second.embedded_body_hash, second.embedding_model) == (hash_of("/a", 2), OTHER_MODEL)
    assert first.embedded_at is not None
    assert second.embedded_at is not None
    assert second.embedded_at >= first.embedded_at


@pytest.fixture
async def flush_pages(graph: GraphRepo, tenant: str) -> list[str]:
    """/a already embedded, /b and /c not; /stale's body changed after selection."""
    paths = ["/a", "/b", "/c", "/stale"]
    await graph.upsert_pages(tenant, [page(p) for p in paths])
    await embed(graph, tenant, ["/a"], seed=7)
    await graph.upsert_pages(tenant, [page("/stale", version=2)])
    # The bodyHash matches the row, so only the placeholder guard can drop it.
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.bodyHash = $h",
        t=tenant,
        u=url("/ghost"),
        h=hash_of("/ghost"),
    )
    return [*paths, "/ghost"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("bad_path", "bad_hash"),
    [
        pytest.param("/missing", hash_of("/missing"), id="missing-url"),
        pytest.param("/ghost", hash_of("/ghost"), id="placeholder"),
        pytest.param("/stale", hash_of("/stale", 1), id="stale-body-hash"),
    ],
)
async def test_a_short_flush_raises_and_writes_nothing(
    graph: GraphRepo, tenant: str, flush_pages: list[str], bad_path: str, bad_hash: str
) -> None:
    before = await pages_by_url(graph, tenant, flush_pages)
    urls = [url("/a"), url(bad_path), url("/b"), url("/c")]
    hashes = [hash_of("/a"), bad_hash, hash_of("/b"), hash_of("/c")]

    with pytest.raises(DatabaseWriteError, match="wrote 3 of 4 rows, rolled back"):
        await graph.write_embeddings(
            tenant, urls, hashes, unit_vectors(4, 8), model=OTHER_MODEL, dimensions=DIM
        )

    assert await pages_by_url(graph, tenant, flush_pages) == before
    assert await graph.embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=MODEL, vectors=1),
    )
    assert [t.url for t in (await graph.embedding_selection(tenant)).targets] == [
        url("/b"),
        url("/c"),
        url("/stale"),
    ]


@pytest.mark.integration
async def test_write_leaves_another_tenants_same_url_untouched(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    for t in (tenant, other):
        await graph.upsert_pages(t, [page("/a")])
    await graph.upsert_pages(other, [page("/only-other")])
    before = await pages_by_url(graph, other, ["/a", "/only-other"])

    await embed(graph, tenant, ["/a"])
    with pytest.raises(DatabaseWriteError, match="wrote 0 of 1 rows, rolled back"):
        await embed(graph, tenant, ["/only-other"])

    assert await pages_by_url(graph, other, ["/a", "/only-other"]) == before
    assert before[url("/a")].content_embedding is None
    assert (await pages_by_url(graph, tenant, ["/a"]))[url("/a")].content_embedding is not None


# ── argument checks: raised before anything is sent ─────────────────────────


def good_args() -> dict[str, object]:
    return {
        "tenant_id": "t",
        "urls": [url("/a"), url("/b")],
        "body_hashes": [hash_of("/a"), hash_of("/b")],
        "vectors": unit_vectors(2, 0),
        "model": MODEL,
        "dimensions": DIM,
    }


def with_value(row: int, value: float) -> npt.NDArray[np.float32]:
    vectors = unit_vectors(2, 0)
    vectors[row, 5] = value
    return vectors


@pytest.fixture
async def offline_graph() -> AsyncIterator[GraphRepo]:
    """A repo whose server does not exist: any query would raise DatabaseUnavailableError."""
    driver = AsyncGraphDatabase.driver(
        "bolt://127.0.0.1:1",
        auth=("neo4j", "x"),
        connection_timeout=1,
        max_transaction_retry_time=0,
    )
    repo = GraphRepo(driver)
    yield repo
    await repo.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"tenant_id": " "}, "tenant_id must be a non-empty string", id="blank-tenant"),
        pytest.param({"model": " "}, "model must be a non-empty string", id="blank-model"),
        pytest.param(
            {"dimensions": 1024, "vectors": unit_vectors(2, 0)[:, :1024]},
            "dimensions 1024 does not match the 2048d vector index",
            id="dimensions-not-2048",
        ),
        pytest.param(
            {"urls": [], "body_hashes": [], "vectors": np.zeros((0, DIM), dtype=np.float32)},
            "no embeddings to write",
            id="empty",
        ),
        pytest.param(
            {"vectors": unit_vectors(2, 0).astype(np.float64)},
            "vectors must be a float32 matrix with 2048 columns, got float64",
            id="float64",
        ),
        pytest.param(
            {"vectors": unit_vectors(2, 0)[:, :2047]},
            r"vectors must be a float32 matrix with 2048 columns, got float32 \(2, 2047\)",
            id="short-columns",
        ),
        pytest.param(
            {"vectors": unit_vectors(1, 0)[0]},
            r"vectors must be a float32 matrix with 2048 columns, got float32 \(2048,\)",
            id="one-dimensional",
        ),
        pytest.param(
            {"body_hashes": [hash_of("/a")]},
            "got 2 urls, 1 body hashes and 2 vectors",
            id="hash-count",
        ),
        pytest.param(
            {"vectors": unit_vectors(3, 0)},
            "got 2 urls, 2 body hashes and 3 vectors",
            id="vector-count",
        ),
        pytest.param(
            {"urls": [url("/a"), url("/a")]}, "duplicate urls in one flush", id="duplicate-url"
        ),
        pytest.param(
            {"vectors": with_value(1, np.nan)}, "vectors contain NaN or infinite values", id="nan"
        ),
        pytest.param(
            {"vectors": with_value(0, np.inf)}, "vectors contain NaN or infinite values", id="inf"
        ),
    ],
)
async def test_bad_arguments_raise_value_error_before_sending(
    offline_graph: GraphRepo, overrides: dict[str, object], message: str
) -> None:
    args = {**good_args(), **overrides}
    with pytest.raises(ValueError, match=message):
        await offline_graph.write_embeddings(**args)  # type: ignore[arg-type]


async def test_offline_graph_would_fail_on_a_valid_flush(offline_graph: GraphRepo) -> None:
    """Guards the test above: valid arguments do reach the (absent) server."""
    with pytest.raises(DatabaseUnavailableError, match="neo4j"):
        await offline_graph.write_embeddings(**good_args())  # type: ignore[arg-type]
