from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import HttpUrl

from linking_engine.ingest.graph_load import TargetResolver, canonical_key, load_tenant_graph
from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.models import (
    EmbeddingSelection,
    EmbeddingTarget,
    Heading,
    LinkRecord,
    PageRecord,
    TenantGraphCounts,
)

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

BASE = "https://www.example.com"


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://www.example.com/a/", "example.com/a"),
        ("http://example.com/a", "example.com/a"),
        ("https://example.com/", "example.com/"),
        ("https://example.com", "example.com/"),
        ("https://example.com:8443/a?x=1", "example.com:8443/a?x=1"),
    ],
)
def test_canonical_key(url: str, key: str) -> None:
    assert canonical_key(url) == key


def test_resolver_prefers_exact_then_canonical_then_one_placeholder_per_key() -> None:
    resolver = TargetResolver(["https://www.example.com/a/", "https://www.example.com/b"])
    assert resolver.resolve("https://www.example.com/a/") == ("https://www.example.com/a/", False)
    assert resolver.resolve("http://example.com/a") == ("https://www.example.com/a/", False)
    assert resolver.resolve("https://www.example.com/b/") == ("https://www.example.com/b", False)
    assert resolver.resolve("https://www.example.com/new/") == (
        "https://www.example.com/new/",
        True,
    )
    assert resolver.resolve("https://example.com/new") == ("https://www.example.com/new/", True)


def page(path: str, links: int, body: str = "text") -> PageRecord:
    return PageRecord(
        url=HttpUrl(f"{BASE}{path}"),
        status_code=200,
        usable=True,
        meta_title=None,
        meta_description=None,
        h1=None,
        headings=(Heading(level=1, text="H"),),
        body_text=body,
        word_count=len(body.split()),
        link_count=links,
        content_hash="h",
        body_hash=body_hash(body),
        scraped_at=None,
        source="test",
    )


def link(source: str, position: int, target: str, *, internal: bool = True) -> LinkRecord:
    return LinkRecord(
        source_url=HttpUrl(f"{BASE}{source}"),
        position=position,
        target_url=HttpUrl(target if target.startswith("http") else f"{BASE}{target}"),
        anchor_text="anchor",
        surrounding_text="around",
        is_internal=internal,
    )


@pytest.mark.integration
async def test_load_builds_the_graph_with_placeholders_and_converges(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    pages = [page("/a/", 5), page("/b", 0)]
    links = [
        link("/a/", 0, "/b"),
        link("/a/", 1, "http://example.com/b/"),  # same page as /b
        link("/a/", 2, "/uncrawled"),
        link("/a/", 3, "/a"),  # the source itself
        link("/a/", 4, "https://other.test/x", internal=False),
    ]
    await mongo.write_pages(tenant, pages, links)

    report = await load_tenant_graph(mongo, graph, tenant, batch_size=1)
    assert (report.pages, report.placeholders, report.links) == (2, 1, 3)
    assert (report.external_links_skipped, report.self_links_skipped) == (1, 1)
    assert await graph.counts(tenant) == TenantGraphCounts(pages=2, placeholders=1, links=3)
    targets = [str(lk.target_url) for lk in await graph.links_from(tenant, [f"{BASE}/a/"])]
    assert targets == [f"{BASE}/b", f"{BASE}/b", f"{BASE}/uncrawled"]

    again = await load_tenant_graph(mongo, graph, tenant)
    assert (again.links, again.stale_links_deleted) == (3, 0)

    await mongo.write_pages(tenant, [page("/a/", 1)], [link("/a/", 0, "/b")])
    shrunk = await load_tenant_graph(mongo, graph, tenant)
    assert (shrunk.links, shrunk.stale_links_deleted) == (1, 2)
    assert (await graph.counts(tenant)).links == 1


async def graph_body_hashes(graph: GraphRepo, tenant: str) -> dict[str, object]:
    rows = await graph._auto(
        "MATCH (p:Page {tenantId: $t}) WHERE NOT coalesce(p.isPlaceholder, false) "
        "RETURN p.url AS url, p.bodyHash AS hash",
        t=tenant,
    )
    return {str(row["url"]): row["hash"] for row in rows}


async def mongo_body_hashes(mongo: MongoRepo, tenant: str) -> dict[str, str]:
    return {
        str(s.url): s.body_hash async for batch in mongo.iter_page_summaries(tenant) for s in batch
    }


@pytest.mark.integration
async def test_load_carries_body_hash_and_follows_a_body_edit(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    await mongo.write_pages(tenant, [page("/a", 0, "first body"), page("/b", 0, "other body")], [])
    await load_tenant_graph(mongo, graph, tenant)
    loaded = await graph_body_hashes(graph, tenant)
    assert loaded == await mongo_body_hashes(mongo, tenant)
    assert loaded == {f"{BASE}/a": body_hash("first body"), f"{BASE}/b": body_hash("other body")}

    await mongo.write_pages(tenant, [page("/a", 0, "first body, edited")], [])
    await load_tenant_graph(mongo, graph, tenant)
    reloaded = await graph_body_hashes(graph, tenant)
    assert reloaded == await mongo_body_hashes(mongo, tenant)
    assert reloaded[f"{BASE}/a"] == body_hash("first body, edited") != loaded[f"{BASE}/a"]
    assert reloaded[f"{BASE}/b"] == loaded[f"{BASE}/b"]


@pytest.mark.integration
async def test_a_body_edit_makes_exactly_that_page_a_target_again(
    mongo: MongoRepo, graph: GraphRepo, tenant: str
) -> None:
    bodies = {"/a": "alpha body", "/b": "beta body", "/c": "gamma body"}
    await mongo.write_pages(tenant, [page(path, 0, text) for path, text in bodies.items()], [])
    await load_tenant_graph(mongo, graph, tenant)
    urls = [f"{BASE}{path}" for path in bodies]
    vectors = np.random.default_rng(0).standard_normal((3, 2048)).astype(np.float32)
    await graph.write_embeddings(
        tenant,
        urls,
        [body_hash(text) for text in bodies.values()],
        vectors,
        model="voyage-4-large",
        dimensions=2048,
    )
    assert await graph.embedding_selection(tenant) == EmbeddingSelection(
        targets=(), up_to_date=3, placeholders=0, non_2xx=0
    )

    await mongo.write_pages(tenant, [page("/b", 0, "beta body, edited")], [])
    await load_tenant_graph(mongo, graph, tenant)

    assert await graph.embedding_selection(tenant) == EmbeddingSelection(
        targets=(EmbeddingTarget(url=f"{BASE}/b", body_hash=body_hash("beta body, edited")),),
        up_to_date=2,
        placeholders=0,
        non_2xx=0,
    )
