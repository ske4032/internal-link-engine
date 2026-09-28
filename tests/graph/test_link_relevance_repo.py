"""Relevance of existing links: sentence and anchor against the target, written on LINKS_TO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseReadError
from linking_engine.graph.repo import GraphRepo
from linking_engine.models import Link, Page

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

BASE = "example.com"
TARGET = [1.0, 2.0, 0.0, -1.0]
SENTENCES = {0: [2.0, 1.0, 0.5, 0.0], 1: [0.0, 1.0, 3.0, 1.0], 2: [-1.0, 0.0, 1.0, 2.0]}
ANCHOR = [1.0, 1.0, 1.0, 0.0]
MODEL = "voyage-4-large"


def url(path: str) -> str:
    return f"{BASE}{path}"


def normalised_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Neo4j's vector.similarity.cosine: the cosine mapped from [-1, 1] to [0, 1]."""
    x, y = np.asarray(a), np.asarray(b)
    return float((1 + x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) / 2)


def link(source: str, target: str, position: int) -> Link:
    return Link(
        source_url=url(source),
        target_url=url(target),
        position=position,
        anchor_text=f"anchor {position}",
        surrounding_text=f"sentence {position}",
    )


async def set_node_vector(
    graph: GraphRepo, tenant: str, path: str, vector: Sequence[float]
) -> None:
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.embeddingModel = $model "
        "WITH p CALL db.create.setNodeVectorProperty(p, 'content_embedding', $v)",
        t=tenant,
        u=url(path),
        v=list(vector),
        model=MODEL,
    )


async def set_edge(
    graph: GraphRepo,
    tenant: str,
    source: str,
    position: int,
    *,
    sentence: Sequence[float] | None,
    key: str | None,
    generic: bool,
) -> None:
    await graph._auto(
        "MATCH (:Page {tenantId: $t, url: $u})-[r:LINKS_TO {position: $p}]->() "
        "SET r.anchorKey = $key, r.anchorGeneric = $generic",
        t=tenant,
        u=url(source),
        p=position,
        key=key,
        generic=generic,
    )
    if sentence is not None:
        await graph._auto(
            "MATCH (:Page {tenantId: $t, url: $u})-[r:LINKS_TO {position: $p}]->() "
            "SET r.surroundingEmbeddingModel = $model "
            "WITH r CALL db.create.setRelationshipVectorProperty(r, 'surroundingEmbedding', $v)",
            t=tenant,
            u=url(source),
            p=position,
            v=list(sentence),
            model=MODEL,
        )


async def add_anchor(graph: GraphRepo, tenant: str, key: str, vector: Sequence[float]) -> None:
    await graph._auto(
        "MERGE (a:Anchor {tenantId: $t, text: $key}) SET a.embeddingModel = $model "
        "WITH a CALL db.create.setNodeVectorProperty(a, 'embedding', $v)",
        t=tenant,
        key=key,
        v=list(vector),
        model=MODEL,
    )


async def seed(graph: GraphRepo, tenant: str) -> None:
    """/s links to /t three times (topical anchor, generic anchor, anchor without a vector), to
    /bare whose page has no vector, and to a placeholder; /s2 links to /t once."""
    await graph.upsert_pages(
        tenant, [Page(url=url(p), status_code=200) for p in ("/s", "/s2", "/t", "/bare")]
    )
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await graph.replace_links(
        tenant,
        [url("/s"), url("/s2")],
        [
            link("/s", "/t", 0),
            link("/s", "/t", 1),
            link("/s", "/t", 2),
            link("/s", "/bare", 3),
            link("/s", "/ghost", 4),
            link("/s2", "/t", 0),
        ],
    )
    await set_node_vector(graph, tenant, "/t", TARGET)
    await add_anchor(graph, tenant, "trail shoes", ANCHOR)
    await set_edge(graph, tenant, "/s", 0, sentence=SENTENCES[0], key="trail shoes", generic=False)
    await set_edge(graph, tenant, "/s", 1, sentence=SENTENCES[1], key="click here", generic=True)
    await set_edge(graph, tenant, "/s", 2, sentence=SENTENCES[2], key="no vector", generic=False)
    await set_edge(graph, tenant, "/s", 3, sentence=SENTENCES[0], key="trail shoes", generic=False)
    await set_edge(graph, tenant, "/s2", 0, sentence=None, key="trail shoes", generic=False)


async def edge_scores(
    graph: GraphRepo, tenant: str
) -> dict[tuple[str, int], tuple[object, object]]:
    rows = await graph._read(
        "MATCH (s:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "RETURN s.url AS url, r.position AS position, r.contextRelevance AS context, "
        "r.anchorTargetFit AS fit",
        t=tenant,
    )
    return {
        (str(row["url"]).removeprefix(BASE), int(str(row["position"]))): (
            row["context"],
            row["fit"],
        )
        for row in rows
    }


@pytest.mark.integration
async def test_links_are_scored_with_the_normalised_cosine_against_the_target(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)

    assert await graph.score_link_relevance(tenant, batch_size=1) == (4, 1)

    found = await graph.link_relevance(tenant)
    assert [(r.source_url, r.position, r.target_url) for r in found] == [
        (url("/s"), 0, url("/t")),
        (url("/s"), 1, url("/t")),
        (url("/s"), 2, url("/t")),
        (url("/s2"), 0, url("/t")),
    ]
    topical, generic, unknown, unembedded = found
    assert topical.context_relevance == pytest.approx(normalised_cosine(SENTENCES[0], TARGET))
    assert topical.anchor_target_fit == pytest.approx(normalised_cosine(ANCHOR, TARGET))
    assert generic.context_relevance == pytest.approx(normalised_cosine(SENTENCES[1], TARGET))
    assert (generic.anchor_target_fit, generic.anchor_generic) == (None, True)
    assert unknown.context_relevance == pytest.approx(normalised_cosine(SENTENCES[2], TARGET))
    assert unknown.anchor_target_fit is None
    # No sentence vector: only the anchor is scored.
    assert unembedded.context_relevance is None
    assert unembedded.anchor_target_fit == pytest.approx(normalised_cosine(ANCHOR, TARGET))


@pytest.mark.integration
async def test_a_rerun_converges_and_removes_scores_that_no_longer_apply(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    # Stale scores from an earlier run, on links that cannot be scored now.
    await graph._auto(
        "MATCH (:Page {tenantId: $t, url: $u})-[r:LINKS_TO]->() "
        "WHERE r.position IN [1, 3] SET r.contextRelevance = 0.5, r.anchorTargetFit = 0.5",
        t=tenant,
        u=url("/s"),
    )
    await graph.score_link_relevance(tenant)
    first = await edge_scores(graph, tenant)

    assert first[("/s", 1)][1] is None
    assert first[("/s", 3)] == (None, None)
    assert await graph.score_link_relevance(tenant) == (4, 1)
    assert await edge_scores(graph, tenant) == first

    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) REMOVE p.content_embedding", t=tenant, u=url("/t")
    )

    assert await graph.score_link_relevance(tenant) == (0, 5)
    assert set((await edge_scores(graph, tenant)).values()) == {(None, None)}
    assert await graph.link_relevance(tenant) == []


@pytest.mark.integration
async def test_another_tenant_is_never_scored_or_read(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other)
    # The same anchor key in another tenant, with another vector, must never be looked up.
    await add_anchor(graph, other, "trail shoes", [0.0, 0.0, -1.0, 3.0])
    await graph._auto(
        "MATCH (:Page {tenantId: $t})-[r:LINKS_TO]->() SET r.contextRelevance = 0.25", t=other
    )

    assert await graph.score_link_relevance(tenant) == (4, 1)

    topical = (await graph.link_relevance(tenant))[0]
    assert topical.anchor_target_fit == pytest.approx(normalised_cosine(ANCHOR, TARGET))
    assert {context for context, _ in (await edge_scores(graph, other)).values()} == {0.25}
    assert {r.source_url for r in await graph.link_relevance(tenant)} == {url("/s"), url("/s2")}
    assert len(await graph.link_relevance(other)) == 4


@pytest.mark.integration
async def test_a_link_whose_target_became_a_placeholder_loses_its_scores_uncounted(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    await graph.score_link_relevance(tenant)
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.isPlaceholder = true", t=tenant, u=url("/t")
    )

    assert await graph.score_link_relevance(tenant) == (0, 1)

    assert set((await edge_scores(graph, tenant)).values()) == {(None, None)}


@pytest.mark.integration
async def test_a_tenant_without_links_scores_nothing(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [Page(url=url("/a"), status_code=200)])

    assert await graph.score_link_relevance(tenant) == (0, 0)
    assert await graph.link_relevance(tenant) == []


@pytest.mark.integration
async def test_a_stored_score_out_of_range_fails_the_read(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant)
    await graph.score_link_relevance(tenant)
    await graph._auto(
        "MATCH (:Page {tenantId: $t})-[r:LINKS_TO]->() SET r.anchorTargetFit = 1.5", t=tenant
    )

    with pytest.raises(DatabaseReadError, match="link relevance"):
        await graph.link_relevance(tenant)


@pytest.fixture
async def offline_graph() -> AsyncIterator[GraphRepo]:
    """A repo whose server does not exist: any query would raise DatabaseUnavailableError."""
    driver = AsyncGraphDatabase.driver(
        "bolt://127.0.0.1:1", auth=("neo4j", "x"), connection_timeout=1
    )
    repo = GraphRepo(driver)
    yield repo
    await repo.close()


async def test_invalid_calls_are_rejected_before_any_query(offline_graph: GraphRepo) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_graph.score_link_relevance(" ")
    with pytest.raises(ValueError, match="batch_size"):
        await offline_graph.score_link_relevance("t", batch_size=0)
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_graph.link_relevance(" ")
