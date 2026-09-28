"""TARGETS_KEYWORD edges: one replace per source, crawled pages only, one resolved edge per page."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.graph.repo import GraphRepo
from linking_engine.models import KeywordRung, KeywordSource, KeywordTarget, Page

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

BASE = "example.com"
STRATEGIC = KeywordSource.CLIENT_STRATEGIC
GSC = KeywordSource.GSC_OBSERVED
INFERRED = KeywordSource.INFERRED

Edge = tuple[object, ...]


def url(path: str) -> str:
    return f"{BASE}{path}"


def target(
    path: str,
    text: str,
    source: KeywordSource = STRATEGIC,
    *,
    language: str = "en",
    priority: int | None = None,
    is_primary: bool = False,
    rung: KeywordRung | None = None,
) -> KeywordTarget:
    return KeywordTarget(
        url=url(path),
        text=text,
        language=language,
        source=source,
        priority=priority,
        is_primary=is_primary,
        rung=rung,
    )


async def seed(graph: GraphRepo, tenant: str, crawled: list[str], placeholders: list[str]) -> None:
    await graph.upsert_pages(tenant, [Page(url=url(p), status_code=200) for p in crawled])
    await graph.upsert_placeholders(tenant, [url(p) for p in placeholders])


async def edges(graph: GraphRepo, tenant: str) -> set[Edge]:
    """(path, text, language, source, priority, isPrimary, rung, resolved) of every edge."""
    rows = await graph._read(
        "MATCH (p:Page {tenantId: $t})-[r:TARGETS_KEYWORD]->(k:Keyword {tenantId: $t}) "
        "RETURN p.url AS url, k.text AS text, k.language AS language, r.source AS source, "
        "r.priority AS priority, r.isPrimary AS primary, r.rung AS rung, r.resolved AS resolved",
        t=tenant,
    )
    return {
        (
            str(row["url"]).removeprefix(BASE),
            str(row["text"]),
            str(row["language"]),
            str(row["source"]),
            row["priority"],
            row["primary"],
            row["rung"],
            row["resolved"],
        )
        for row in rows
    }


async def keywords(graph: GraphRepo, tenant: str) -> dict[tuple[str, str], object]:
    rows = await graph._read(
        "MATCH (k:Keyword {tenantId: $t}) RETURN k.text AS text, k.language AS language, "
        "k.isStrategic AS strategic",
        t=tenant,
    )
    return {(str(r["text"]), str(r["language"])): r["strategic"] for r in rows}


@pytest.mark.integration
async def test_targets_become_edges_to_one_keyword_per_text_and_language(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a", "/b"], [])
    rows = [
        target("/a", "trail shoes", priority=5, is_primary=True, rung=KeywordRung.STRATEGIC),
        target("/a", "running", priority=2),
        target("/b", "trail shoes", priority=3, is_primary=True),
        target("/b", "trail shoes", language="de", priority=3),
    ]

    assert await graph.replace_keyword_targets(tenant, STRATEGIC, rows, batch_size=1) == (4, 0)

    assert await edges(graph, tenant) == {
        ("/a", "trail shoes", "en", "CLIENT_STRATEGIC", 5, True, "STRATEGIC", True),
        ("/a", "running", "en", "CLIENT_STRATEGIC", 2, False, None, None),
        ("/b", "trail shoes", "en", "CLIENT_STRATEGIC", 3, True, None, None),
        ("/b", "trail shoes", "de", "CLIENT_STRATEGIC", 3, False, None, None),
    }
    assert await keywords(graph, tenant) == {
        ("trail shoes", "en"): True,
        ("running", "en"): True,
        ("trail shoes", "de"): True,
    }


@pytest.mark.integration
async def test_a_rerun_makes_the_edges_exactly_the_current_rows(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a", "/b"], [])
    first = [
        target("/a", "trail shoes", priority=5, is_primary=True, rung=KeywordRung.STRATEGIC),
        target("/a", "running", priority=2),
        target("/b", "tents", priority=3),
    ]
    await graph.replace_keyword_targets(tenant, STRATEGIC, first)

    # The client dropped "running", reprioritised "tents" and moved the resolved keyword.
    second = [
        target("/a", "trail shoes", priority=5, is_primary=True),
        target("/b", "tents", priority=4, rung=KeywordRung.STRATEGIC),
    ]

    assert await graph.replace_keyword_targets(tenant, STRATEGIC, second) == (2, 1)
    assert await graph.replace_keyword_targets(tenant, STRATEGIC, second) == (2, 0)
    assert await edges(graph, tenant) == {
        ("/a", "trail shoes", "en", "CLIENT_STRATEGIC", 5, True, None, None),
        ("/b", "tents", "en", "CLIENT_STRATEGIC", 4, False, "STRATEGIC", True),
    }
    # The dropped keyword keeps its node but is no longer strategic.
    assert await keywords(graph, tenant) == {
        ("trail shoes", "en"): True,
        ("running", "en"): False,
        ("tents", "en"): True,
    }


@pytest.mark.integration
async def test_rows_on_placeholders_or_unknown_urls_are_not_written(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a"], ["/ghost"])
    rows = [
        target("/a", "tents", INFERRED, rung=KeywordRung.H1),
        target("/ghost", "ghost keyword", INFERRED, rung=KeywordRung.TITLE),
        target("/missing", "missing keyword", INFERRED, rung=KeywordRung.TITLE),
    ]

    assert await graph.replace_keyword_targets(tenant, INFERRED, rows) == (1, 0)

    assert await edges(graph, tenant) == {
        ("/a", "tents", "en", "INFERRED", None, False, "H1", True)
    }
    assert await keywords(graph, tenant) == {("tents", "en"): None}


@pytest.mark.integration
async def test_each_source_is_replaced_on_its_own(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant, ["/a"], [])
    await graph.replace_keyword_targets(
        tenant, STRATEGIC, [target("/a", "tents", priority=1, rung=KeywordRung.STRATEGIC)]
    )
    await graph.replace_keyword_targets(tenant, GSC, [target("/a", "tents", GSC)])
    await graph.replace_keyword_targets(tenant, INFERRED, [target("/a", "camping tents", INFERRED)])

    assert await graph.replace_keyword_targets(tenant, GSC, []) == (0, 1)

    assert await edges(graph, tenant) == {
        ("/a", "tents", "en", "CLIENT_STRATEGIC", 1, False, "STRATEGIC", True),
        ("/a", "camping tents", "en", "INFERRED", None, False, None, None),
    }


@pytest.mark.integration
async def test_an_edge_on_a_page_that_became_a_placeholder_is_deleted(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, ["/a"], [])
    rows = [target("/a", "tents", INFERRED, rung=KeywordRung.H1)]
    await graph.replace_keyword_targets(tenant, INFERRED, rows)
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: $u}) SET p.isPlaceholder = true", t=tenant, u=url("/a")
    )

    assert await graph.replace_keyword_targets(tenant, INFERRED, rows) == (0, 1)
    assert await edges(graph, tenant) == set()


@pytest.mark.integration
async def test_another_tenants_edges_and_keywords_are_never_touched(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    for t in (tenant, other):
        await seed(graph, t, ["/a"], [])
        await graph.replace_keyword_targets(t, STRATEGIC, [target("/a", "tents", priority=2)])

    assert await graph.replace_keyword_targets(tenant, STRATEGIC, []) == (0, 1)

    assert await edges(graph, tenant) == set()
    assert await edges(graph, other) == {
        ("/a", "tents", "en", "CLIENT_STRATEGIC", 2, False, None, None)
    }
    assert await keywords(graph, tenant) == {("tents", "en"): False}
    assert await keywords(graph, other) == {("tents", "en"): True}


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
        (" ", [], 10, "tenant_id"),
        ("t", [target("/a", "tents", GSC)], 10, "CLIENT_STRATEGIC"),
        ("t", [target("/a", "tents"), target("/a", "tents", priority=1)], 10, "duplicate"),
        (
            "t",
            [
                target("/a", "tents", rung=KeywordRung.STRATEGIC),
                target("/a", "camping", rung=KeywordRung.STRATEGIC),
            ],
            10,
            "more than one resolved",
        ),
        ("t", [], 0, "batch_size"),
    ],
    ids=["blank-tenant", "other-source", "duplicate-target", "two-resolved", "zero-batch"],
)
async def test_invalid_replacements_are_rejected_before_any_query(
    offline_graph: GraphRepo,
    tenant_id: str,
    rows: list[KeywordTarget],
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.replace_keyword_targets(
            tenant_id, STRATEGIC, rows, batch_size=batch_size
        )
