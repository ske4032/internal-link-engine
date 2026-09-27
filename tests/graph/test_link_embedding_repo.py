"""Link-embedding reads and writes: edge texts, anchor keys, Anchor nodes, sentence vectors."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError, DatabaseUnavailableError, DatabaseWriteError
from linking_engine.graph.repo import ANCHOR_KEY_MAX_BYTES, VECTOR_DIMENSIONS, GraphRepo
from linking_engine.models import (
    AnchorKeyUpdate,
    EdgeRef,
    EmbeddingModelCount,
    Link,
    LinkText,
    Page,
    SentenceTarget,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    import numpy.typing as npt

BASE = "example.com"
MODEL = "voyage-4-large"
OTHER_MODEL = "voyage-3-large"
DIM = VECTOR_DIMENSIONS
CLOCK_SLACK = timedelta(minutes=2)
TARGET = "/t"
# Two-byte characters: exactly at the Anchor key limit, and one byte over it.
KEY_AT_LIMIT = "\N{LATIN SMALL LETTER E WITH ACUTE}" * (ANCHOR_KEY_MAX_BYTES // 2)
KEY_OVER_LIMIT = KEY_AT_LIMIT + "a"


def url(path: str) -> str:
    return f"{BASE}{path}"


def unit_vectors(count: int, seed: int) -> npt.NDArray[np.float32]:
    matrix = np.random.default_rng(seed).standard_normal((count, DIM))
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def edge(source: str, position: int) -> EdgeRef:
    return EdgeRef(source_url=url(source), position=position)


def text_of(source: str, position: int) -> LinkText:
    """The LinkText seed_edges writes for (source, position), before any marker."""
    return LinkText(
        source_url=url(source),
        position=position,
        anchor_text=f"anchor {source} {position}",
        surrounding_text=f"sentence {source} {position}",
    )


async def seed_edges(graph: GraphRepo, tenant: str, edges: Sequence[tuple[str, int]]) -> None:
    """Pages for every source plus TARGET; each (source, position) links to TARGET."""
    sources = sorted({source for source, _ in edges})
    await graph.upsert_pages(
        tenant, [Page(url=url(p), status_code=200) for p in [*sources, TARGET]]
    )
    links = [
        Link(
            source_url=url(source),
            target_url=url(TARGET),
            position=position,
            anchor_text=f"anchor {source} {position}",
            surrounding_text=f"sentence {source} {position}",
        )
        for source, position in edges
    ]
    await graph.replace_links(tenant, [url(s) for s in sources], links)


async def link_texts(graph: GraphRepo, tenant: str, batch_size: int = 1000) -> list[LinkText]:
    return [
        text
        async for batch in graph.iter_link_texts(tenant, batch_size=batch_size)
        for text in batch
    ]


async def edge_props(graph: GraphRepo, tenant: str) -> dict[tuple[str, int], dict[str, object]]:
    rows = await graph._auto(
        "MATCH (s:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "RETURN s.url AS source, r.position AS position, r.anchorKey AS key, "
        "r.anchorGeneric AS generic, r.surroundingEmbedding AS vec, "
        "r.surroundingEmbeddedHash AS hash, r.surroundingEmbeddingModel AS model, "
        "keys(r) AS props",
        t=tenant,
    )
    return {(str(row["source"]), int(row["position"])): row for row in rows}  # type: ignore[call-overload]


async def anchors(graph: GraphRepo, tenant: str) -> dict[str, dict[str, object]]:
    rows = await graph._auto(
        "MATCH (a:Anchor {tenantId: $t}) RETURN a.text AS text, a.embedding AS vec, "
        "a.embeddingModel AS model, a.embeddingDimensions AS dims, a.embeddedAt AS at",
        t=tenant,
    )
    return {str(row["text"]): row for row in rows}


def assert_vector(stored: object, expected: npt.NDArray[np.float32]) -> None:
    assert isinstance(stored, list), f"expected a stored list, got {type(stored).__name__}"
    assert len(stored) == DIM
    np.testing.assert_allclose(np.asarray(stored, dtype=np.float32), expected, rtol=0, atol=1e-6)


async def write_anchors(
    graph: GraphRepo, tenant: str, keys: Sequence[str], *, model: str = MODEL, seed: int = 0
) -> npt.NDArray[np.float32]:
    vectors = unit_vectors(len(keys), seed)
    written = await graph.write_anchor_embeddings(
        tenant, keys, vectors, model=model, dimensions=DIM
    )
    assert written == len(keys)
    return vectors


# ── iter_link_texts ─────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_link_texts_are_keyset_paged_by_source_then_numeric_position(
    graph: GraphRepo, tenant: str
) -> None:
    # Positions 10 and 2 would swap under a string sort; batch 2 splits /a's edges.
    await seed_edges(graph, tenant, [("/b", 1), ("/a", 10), ("/b", 0), ("/a", 2), ("/a", 0)])

    batches = [batch async for batch in graph.iter_link_texts(tenant, batch_size=2)]

    assert [len(batch) for batch in batches] == [2, 2, 1]
    order = [("/a", 0), ("/a", 2), ("/a", 10), ("/b", 0), ("/b", 1)]
    assert [text for batch in batches for text in batch] == [text_of(s, p) for s, p in order]


@pytest.mark.integration
async def test_link_texts_carry_the_stored_markers(graph: GraphRepo, tenant: str) -> None:
    await seed_edges(graph, tenant, [("/a", 0), ("/a", 1), ("/a", 2)])
    updates = [
        AnchorKeyUpdate(source_url=url("/a"), position=0, anchor_key="trail", anchor_generic=False),
        AnchorKeyUpdate(source_url=url("/a"), position=1, anchor_key=None, anchor_generic=False),
    ]
    await graph.set_anchor_keys(tenant, updates)
    target = SentenceTarget(sentence_hash="h0", edges=(edge("/a", 0),))
    await graph.write_surrounding_embeddings(
        tenant, [target], unit_vectors(1, 0), model=MODEL, dimensions=DIM
    )

    assert await link_texts(graph, tenant) == [
        text_of("/a", 0).model_copy(
            update={
                "anchor_key": "trail",
                "anchor_generic": False,
                "surrounding_embedded_hash": "h0",
                "surrounding_embedding_model": MODEL,
            }
        ),
        text_of("/a", 1).model_copy(update={"anchor_generic": False}),
        text_of("/a", 2),
    ]


@pytest.mark.integration
async def test_link_texts_never_include_another_tenants_edges(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed_edges(graph, other, [("/a", 0), ("/b", 0)])
    assert await link_texts(graph, tenant) == []
    await seed_edges(graph, tenant, [("/b", 3)])
    assert await link_texts(graph, tenant) == [text_of("/b", 3)]


@pytest.mark.integration
async def test_an_edge_without_anchor_text_raises_read_error(graph: GraphRepo, tenant: str) -> None:
    await graph._auto(
        "CREATE (:Page {tenantId: $t, url: $s})-[:LINKS_TO {position: 0, surroundingText: 'x'}]->"
        "(:Page {tenantId: $t, url: $d})",
        t=tenant,
        s=url("/a"),
        d=url("/b"),
    )
    with pytest.raises(DatabaseReadError, match="LinkText"):
        await link_texts(graph, tenant)


# ── set_anchor_keys ─────────────────────────────────────────────────────────


def key_update(source: str, position: int, key: str | None, generic: bool) -> AnchorKeyUpdate:
    return AnchorKeyUpdate(
        source_url=url(source), position=position, anchor_key=key, anchor_generic=generic
    )


@pytest.mark.integration
async def test_anchor_keys_are_written_in_batches_and_none_removes_the_key(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_edges(graph, tenant, [("/a", 0), ("/a", 1), ("/b", 0)])
    first = [
        key_update("/a", 0, "trail shoes", False),
        key_update("/a", 1, "click here", True),
        key_update("/b", 0, "c#", False),
    ]
    assert await graph.set_anchor_keys(tenant, first, batch_size=2) == 3

    props = await edge_props(graph, tenant)
    assert {k: (row["key"], row["generic"]) for k, row in props.items()} == {
        (url("/a"), 0): ("trail shoes", False),
        (url("/a"), 1): ("click here", True),
        (url("/b"), 0): ("c#", False),
    }
    # LINK_PROPERTIES maps anchorKey onto Link.anchor_key.
    assert [link.anchor_key for link in await graph.links_from(tenant, [url("/a")])] == [
        "trail shoes",
        "click here",
    ]

    assert await graph.set_anchor_keys(tenant, [key_update("/a", 1, None, False)]) == 1
    row = (await edge_props(graph, tenant))[(url("/a"), 1)]
    assert (row["key"], row["generic"]) == (None, False)
    assert "anchorKey" not in row["props"]  # type: ignore[operator]


@pytest.mark.integration
async def test_a_key_for_an_edge_that_does_not_exist_raises(graph: GraphRepo, tenant: str) -> None:
    await seed_edges(graph, tenant, [("/a", 0)])
    updates = [key_update("/a", 0, "trail", False), key_update("/a", 99, "gone", False)]
    with pytest.raises(DatabaseWriteError, match="wrote 1 of 2 rows"):
        await graph.set_anchor_keys(tenant, updates)


@pytest.mark.integration
async def test_anchor_keys_are_scoped_to_the_tenant(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await seed_edges(graph, other, [("/a", 0)])
    with pytest.raises(DatabaseWriteError, match="wrote 0 of 1 rows"):
        await graph.set_anchor_keys(tenant, [key_update("/a", 0, "trail", False)])
    assert (await edge_props(graph, other))[(url("/a"), 0)]["key"] is None


# ── Anchor nodes ────────────────────────────────────────────────────────────


@pytest.mark.integration
async def test_anchor_write_stores_vector_model_dimensions_and_time(
    graph: GraphRepo, tenant: str
) -> None:
    before = datetime.now(UTC)
    vectors = await write_anchors(graph, tenant, ["trail shoes", "c#"], seed=3)
    after = datetime.now(UTC)

    stored = await anchors(graph, tenant)
    assert set(stored) == {"trail shoes", "c#"}
    for key, vector in zip(["trail shoes", "c#"], vectors, strict=True):
        assert_vector(stored[key]["vec"], vector)
        assert (stored[key]["model"], stored[key]["dims"]) == (MODEL, DIM)
        embedded_at = stored[key]["at"].to_native()  # type: ignore[attr-defined]
        assert before - CLOCK_SLACK <= embedded_at <= after + CLOCK_SLACK


@pytest.mark.integration
async def test_one_anchor_per_tenant_and_text_even_when_rewritten(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await write_anchors(graph, tenant, ["trail shoes"], seed=1)
    second = await write_anchors(graph, tenant, ["trail shoes"], model=OTHER_MODEL, seed=2)
    await write_anchors(graph, other, ["trail shoes"], seed=3)

    rows = await graph._auto(
        "MATCH (a:Anchor {text: 'trail shoes'}) WHERE a.tenantId IN [$t, $o] "
        "RETURN a.tenantId AS tenant, count(a) AS n ORDER BY tenant",
        t=tenant,
        o=other,
    )
    assert [(row["tenant"], row["n"]) for row in rows] == [(tenant, 1), (other, 1)]
    stored = (await anchors(graph, tenant))["trail shoes"]
    assert_vector(stored["vec"], second[0])
    assert stored["model"] == OTHER_MODEL


@pytest.mark.integration
async def test_the_anchor_constraint_exists_and_rejects_a_duplicate(
    graph: GraphRepo, tenant: str
) -> None:
    constraints = {row["name"] for row in await graph._auto("SHOW CONSTRAINTS YIELD name")}
    assert "anchor_tenant_text" in constraints
    await write_anchors(graph, tenant, ["trail shoes"])
    with pytest.raises(DatabaseWriteError, match=r"ConstraintValidationFailed|already exists"):
        await graph._auto("CREATE (:Anchor {tenantId: $t, text: 'trail shoes'})", t=tenant)


@pytest.mark.integration
async def test_keys_to_embed_are_those_without_a_vector_from_the_model_in_input_order(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await write_anchors(graph, tenant, ["b"])
    await write_anchors(graph, tenant, ["c"], model=OTHER_MODEL)
    await write_anchors(graph, other, ["d"])
    # An Anchor without a vector is not embedded yet.
    await graph._auto(
        "CREATE (:Anchor {tenantId: $t, text: 'e', embeddingModel: $m})", t=tenant, m=MODEL
    )

    pending = await graph.anchor_keys_to_embed(
        tenant, ["d", "c", "b", "a", "e"], model=MODEL, batch_size=2
    )

    assert pending == ("d", "c", "a", "e")
    assert await graph.anchor_keys_to_embed(tenant, [], model=MODEL) == ()


@pytest.mark.integration
async def test_delete_tenant_removes_its_anchor_nodes(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await seed_edges(graph, tenant, [("/a", 0)])
    await write_anchors(graph, tenant, ["trail", "boots"])
    await write_anchors(graph, other, ["trail"])

    await graph.delete_tenant(tenant)

    assert await anchors(graph, tenant) == {}
    assert await graph.anchor_embedding_models(tenant) == ()
    assert set(await anchors(graph, other)) == {"trail"}


# ── sentence vectors on the edge ────────────────────────────────────────────


@pytest.mark.integration
async def test_one_sentence_vector_reaches_every_edge_of_its_target_and_round_trips(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_edges(graph, tenant, [("/a", 0), ("/a", 1), ("/b", 0), ("/b", 1)])
    targets = [
        SentenceTarget(sentence_hash="h-shared", edges=(edge("/a", 0), edge("/b", 0))),
        SentenceTarget(sentence_hash="h-single", edges=(edge("/a", 1),)),
    ]
    vectors = unit_vectors(2, 5)

    written = await graph.write_surrounding_embeddings(
        tenant, targets, vectors, model=MODEL, dimensions=DIM
    )

    assert written == 3
    props = await edge_props(graph, tenant)
    expected = {
        (url("/a"), 0): ("h-shared", 0),
        (url("/b"), 0): ("h-shared", 0),
        (url("/a"), 1): ("h-single", 1),
    }
    for key, (digest, row) in expected.items():
        assert (props[key]["hash"], props[key]["model"]) == (digest, MODEL)
        assert_vector(props[key]["vec"], vectors[row])
    untouched = props[(url("/b"), 1)]
    assert (untouched["vec"], untouched["hash"], untouched["model"]) == (None, None, None)
    # The repo's own read maps surroundingEmbedding onto Link.surrounding_embedding.
    [first, _] = await graph.links_from(tenant, [url("/b")])
    assert first.surrounding_embedding is not None
    assert_vector(list(first.surrounding_embedding), vectors[0])
    assert await graph.surrounding_embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=MODEL, vectors=3),
    )


@pytest.mark.integration
async def test_a_missing_target_edge_rolls_back_the_whole_flush(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_edges(graph, tenant, [("/a", 0), ("/a", 1)])
    targets = [
        SentenceTarget(sentence_hash="h1", edges=(edge("/a", 0), edge("/a", 99))),
        SentenceTarget(sentence_hash="h2", edges=(edge("/a", 1),)),
    ]
    with pytest.raises(DatabaseWriteError, match="wrote 2 of 3 rows, rolled back"):
        await graph.write_surrounding_embeddings(
            tenant, targets, unit_vectors(2, 0), model=MODEL, dimensions=DIM
        )

    assert await graph.surrounding_embedding_models(tenant) == ()
    assert all(row["hash"] is None for row in (await edge_props(graph, tenant)).values())


@pytest.mark.integration
async def test_an_anchor_key_at_the_byte_limit_is_written_and_read_back(
    graph: GraphRepo, tenant: str
) -> None:
    assert len(KEY_AT_LIMIT.encode()) == ANCHOR_KEY_MAX_BYTES == 4096
    vectors = await write_anchors(graph, tenant, [KEY_AT_LIMIT])
    stored = await anchors(graph, tenant)
    assert list(stored) == [KEY_AT_LIMIT]
    assert_vector(stored[KEY_AT_LIMIT]["vec"], vectors[0])


# ── clearing stale sentence vectors ─────────────────────────────────────────

SENTENCE_PROPERTIES = {
    "surroundingEmbedding",
    "surroundingEmbeddedHash",
    "surroundingEmbeddingModel",
}


@pytest.fixture
async def embedded_edges(graph: GraphRepo, tenant: str) -> list[tuple[str, int]]:
    """/a#0, /a#1 and /b#0 each carry a sentence vector, hash and model."""
    edges = [("/a", 0), ("/a", 1), ("/b", 0)]
    await seed_edges(graph, tenant, edges)
    targets = [
        SentenceTarget(sentence_hash="h1", edges=(edge("/a", 0), edge("/b", 0))),
        SentenceTarget(sentence_hash="h2", edges=(edge("/a", 1),)),
    ]
    await graph.write_surrounding_embeddings(
        tenant, targets, unit_vectors(2, 0), model=MODEL, dimensions=DIM
    )
    return edges


@pytest.mark.integration
async def test_clearing_removes_all_three_sentence_properties_in_batches(
    graph: GraphRepo, tenant: str, embedded_edges: list[tuple[str, int]]
) -> None:
    cleared = await graph.clear_surrounding_embeddings(
        tenant, [edge("/a", 0), edge("/a", 1)], batch_size=1
    )

    assert cleared == 2
    props = await edge_props(graph, tenant)
    for key in ((url("/a"), 0), (url("/a"), 1)):
        assert not SENTENCE_PROPERTIES & set(props[key]["props"]), props[key]["props"]  # type: ignore[call-overload]
        assert "anchorText" in props[key]["props"]  # type: ignore[operator]
    assert set(props[(url("/b"), 0)]["props"]) >= SENTENCE_PROPERTIES  # type: ignore[call-overload]
    assert await graph.surrounding_embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=MODEL, vectors=1),
    )
    texts = {(t.source_url, t.position): t for t in await link_texts(graph, tenant)}
    assert texts[(url("/a"), 0)].surrounding_embedded_hash is None
    assert texts[(url("/a"), 0)].surrounding_embedding_model is None


@pytest.mark.integration
async def test_clearing_an_edge_that_does_not_exist_raises(
    graph: GraphRepo, tenant: str, embedded_edges: list[tuple[str, int]]
) -> None:
    with pytest.raises(DatabaseWriteError, match="wrote 1 of 2 rows"):
        await graph.clear_surrounding_embeddings(tenant, [edge("/a", 0), edge("/a", 99)])


@pytest.mark.integration
async def test_clearing_never_reaches_another_tenants_edge(
    graph: GraphRepo, tenant: str, embedded_edges: list[tuple[str, int]]
) -> None:
    other = f"{tenant}-other"
    with pytest.raises(DatabaseWriteError, match="wrote 0 of 1 rows"):
        await graph.clear_surrounding_embeddings(other, [edge("/a", 0)])
    assert (await edge_props(graph, tenant))[(url("/a"), 0)]["hash"] == "h1"


async def test_clearing_nothing_returns_zero_without_a_query(offline_graph: GraphRepo) -> None:
    assert await offline_graph.clear_surrounding_embeddings("t", []) == 0


@pytest.mark.parametrize(
    ("tenant_id", "edges", "batch_size", "message"),
    [
        pytest.param(" ", [edge("/a", 0)], 10, "tenant_id must be a non-empty string", id="tenant"),
        pytest.param("t", [edge("/a", 0), edge("/a", 0)], 10, "duplicate edge", id="duplicate"),
        pytest.param("t", [edge("/a", 0)], 0, "batch_size must be at least 1", id="batch-size"),
    ],
)
async def test_bad_clear_arguments_raise_before_sending(
    offline_graph: GraphRepo, tenant_id: str, edges: list[EdgeRef], batch_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.clear_surrounding_embeddings(tenant_id, edges, batch_size=batch_size)


# ── stored models per kind ──────────────────────────────────────────────────


@pytest.mark.integration
async def test_anchor_models_are_counted_per_model_with_none_last(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    assert await graph.anchor_embedding_models(tenant) == ()
    await write_anchors(graph, tenant, ["a", "b"])
    await write_anchors(graph, tenant, ["c"], model=OTHER_MODEL)
    await graph._auto(
        "CREATE (:Anchor {tenantId: $t, text: 'raw', embedding: [1.0, 0.0]})", t=tenant
    )
    await write_anchors(graph, other, ["z"], model="another-model")

    assert await graph.anchor_embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=OTHER_MODEL, vectors=1),
        EmbeddingModelCount(embedding_model=MODEL, vectors=2),
        EmbeddingModelCount(embedding_model=None, vectors=1),
    )


@pytest.mark.integration
async def test_sentence_models_are_counted_per_model_with_none_last(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed_edges(graph, tenant, [("/a", 0), ("/a", 1), ("/a", 2)])
    await seed_edges(graph, other, [("/a", 0)])
    assert await graph.surrounding_embedding_models(tenant) == ()

    shared = SentenceTarget(sentence_hash="h", edges=(edge("/a", 0), edge("/a", 1)))
    await graph.write_surrounding_embeddings(
        tenant, [shared], unit_vectors(1, 0), model=OTHER_MODEL, dimensions=DIM
    )
    await graph._auto(
        "MATCH (:Page {tenantId: $t, url: $u})-[r:LINKS_TO {position: 2}]->() "
        "SET r.surroundingEmbedding = [1.0, 0.0]",
        t=tenant,
        u=url("/a"),
    )
    await graph.write_surrounding_embeddings(
        other,
        [SentenceTarget(sentence_hash="h", edges=(edge("/a", 0),))],
        unit_vectors(1, 1),
        model=MODEL,
        dimensions=DIM,
    )

    assert await graph.surrounding_embedding_models(tenant) == (
        EmbeddingModelCount(embedding_model=OTHER_MODEL, vectors=2),
        EmbeddingModelCount(embedding_model=None, vectors=1),
    )


# ── argument checks: raised before anything is sent ─────────────────────────


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


def anchor_args(**overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "tenant_id": "t",
        "keys": ["trail", "boots"],
        "vectors": unit_vectors(2, 0),
        "model": MODEL,
        "dimensions": DIM,
    }
    return {**args, **overrides}


def sentence_args(**overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "tenant_id": "t",
        "targets": [
            SentenceTarget(sentence_hash="h1", edges=(edge("/a", 0), edge("/b", 0))),
            SentenceTarget(sentence_hash="h2", edges=(edge("/a", 1),)),
        ],
        "vectors": unit_vectors(2, 0),
        "model": MODEL,
        "dimensions": DIM,
    }
    return {**args, **overrides}


def with_nan() -> npt.NDArray[np.float32]:
    vectors = unit_vectors(2, 0)
    vectors[1, 7] = np.nan
    return vectors


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
            {"keys": [], "vectors": np.zeros((0, DIM), dtype=np.float32)},
            "no embeddings to write",
            id="empty",
        ),
        pytest.param(
            {"vectors": unit_vectors(2, 0).astype(np.float64)},
            "vectors must be a float32 matrix with 2048 columns, got float64",
            id="float64",
        ),
        pytest.param(
            {"vectors": unit_vectors(1, 0)[0]},
            r"vectors must be a float32 matrix with 2048 columns, got float32 \(2048,\)",
            id="one-dimensional",
        ),
        pytest.param({"vectors": unit_vectors(3, 0)}, "and 3 vectors", id="vector-count"),
        pytest.param({"keys": ["trail", " "]}, "blank keys", id="blank-key"),
        pytest.param({"keys": ["trail", "trail"]}, "duplicate keys", id="duplicate-key"),
        pytest.param(
            {"keys": ["trail", KEY_OVER_LIMIT]},
            "1 keys exceed 4096 UTF-8 bytes",
            id="key-over-byte-limit",
        ),
        pytest.param({"vectors": with_nan()}, "NaN or infinite", id="nan"),
    ],
)
async def test_bad_anchor_writes_raise_before_sending(
    offline_graph: GraphRepo, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.write_anchor_embeddings(**anchor_args(**overrides))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"tenant_id": ""}, "tenant_id must be a non-empty string", id="blank-tenant"),
        pytest.param(
            {"targets": [], "vectors": np.zeros((0, DIM), dtype=np.float32)},
            "no embeddings to write",
            id="empty",
        ),
        pytest.param(
            {"vectors": unit_vectors(2, 0).astype(np.float64)},
            "float32 matrix with 2048 columns",
            id="float64",
        ),
        pytest.param({"vectors": unit_vectors(1, 0)}, "and 1 vectors", id="vector-count"),
        pytest.param(
            {
                "targets": [
                    SentenceTarget(sentence_hash="h1", edges=(edge("/a", 0),)),
                    SentenceTarget(sentence_hash="h1", edges=(edge("/a", 1),)),
                ]
            },
            "duplicate sentence hashes",
            id="duplicate-hash",
        ),
        pytest.param(
            {
                "targets": [
                    SentenceTarget(sentence_hash="h1", edges=(edge("/a", 0),)),
                    SentenceTarget(sentence_hash="h2", edges=(edge("/a", 0),)),
                ]
            },
            "duplicate edge",
            id="edge-in-two-targets",
        ),
        pytest.param({"vectors": with_nan()}, "NaN or infinite", id="nan"),
    ],
)
async def test_bad_sentence_writes_raise_before_sending(
    offline_graph: GraphRepo, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.write_surrounding_embeddings(**sentence_args(**overrides))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        pytest.param(
            [key_update("/a", 0, "trail", False), key_update("/a", 0, "boots", False)],
            "duplicate edge",
            id="duplicate-edge",
        ),
        pytest.param([key_update("/a", 0, "  ", False)], "anchor_key", id="blank-key"),
    ],
)
async def test_bad_anchor_key_updates_raise_before_sending(
    offline_graph: GraphRepo, updates: list[AnchorKeyUpdate], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.set_anchor_keys("t", updates)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda g: g.set_anchor_keys(" ", []), id="set_anchor_keys"),
        pytest.param(lambda g: g.anchor_keys_to_embed(" ", ["a"], model=MODEL), id="keys_to_embed"),
        pytest.param(lambda g: g.anchor_embedding_models(" "), id="anchor_models"),
        pytest.param(lambda g: g.surrounding_embedding_models(" "), id="sentence_models"),
        pytest.param(lambda g: anext(g.iter_link_texts(" ")), id="iter_link_texts"),
    ],
)
async def test_every_read_and_key_write_rejects_a_blank_tenant(
    offline_graph: GraphRepo, call: object
) -> None:
    with pytest.raises(ValueError, match="tenant_id must be a non-empty string"):
        await call(offline_graph)  # type: ignore[operator]


async def test_link_texts_reject_a_batch_size_below_one(offline_graph: GraphRepo) -> None:
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        await anext(offline_graph.iter_link_texts("t", batch_size=0))


async def test_keys_to_embed_rejects_a_blank_model(offline_graph: GraphRepo) -> None:
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        await offline_graph.anchor_keys_to_embed("t", ["a"], model=" ")


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda g: g.write_anchor_embeddings(**anchor_args()), id="write_anchor_embeddings"
        ),
        pytest.param(
            lambda g: g.write_surrounding_embeddings(**sentence_args()),
            id="write_surrounding_embeddings",
        ),
    ],
)
async def test_offline_graph_would_fail_on_a_valid_flush(
    offline_graph: GraphRepo, call: object
) -> None:
    """Guards the argument tests above: valid arguments do reach the (absent) server."""
    with pytest.raises(DatabaseUnavailableError, match="neo4j"):
        await call(offline_graph)  # type: ignore[operator]
