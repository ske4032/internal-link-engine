"""Duplicate reads and writes on a real Neo4j: the grouping inputs and the stored groups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from neo4j import AsyncGraphDatabase

from linking_engine.errors import DatabaseWriteError
from linking_engine.graph.repo import GraphRepo
from linking_engine.models import DuplicateGroup, DuplicateInput, Link, Page

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

ARTICLE = "a" * 64
UNTITLED = "b" * 64
EMPTY = "c" * 64
UNIQUE = "d" * 64


def url(path: str) -> str:
    return f"example.com{path}"


def page(
    path: str,
    body: str = ARTICLE,
    *,
    language: str | None = "en",
    status: int = 200,
    words: int = 120,
    indexable: bool | None = None,
) -> Page:
    return Page(
        url=url(path),
        status_code=status,
        body_hash=body,
        language=language,
        word_count=words,
        is_indexable=indexable,
    )


async def link(graph: GraphRepo, tenant: str, pairs: Sequence[tuple[str, str]]) -> None:
    sources = sorted({source for source, _ in pairs})
    await graph.replace_links(
        tenant,
        [url(s) for s in sources],
        [
            Link(
                source_url=url(s),
                target_url=url(t),
                position=i,
                anchor_text="x",
                surrounding_text="",
            )
            for i, (s, t) in enumerate(pairs)
        ],
    )


async def seed(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(
        tenant,
        [
            page("/blog/post"),
            page("/post"),
            page("/news/post", indexable=False),
            # The same body in another language, alone in it.
            page("/de/post", language="de"),
            page("/untitled-1", UNTITLED, language=None, indexable=True),
            page("/untitled-2", UNTITLED, language=None),
            # A redirect and a 404 serving the article's body are not copies of it.
            page("/old-post", status=301),
            page("/missing-post", status=404),
            page("/empty-1", EMPTY, words=0),
            page("/empty-2", EMPTY, words=0),
            page("/unique", UNIQUE),
            page("/hub", UNIQUE + "0"),
            page("/other", UNIQUE + "1"),
        ],
    )
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await link(
        graph,
        tenant,
        [
            ("/hub", "/blog/post"),
            ("/hub", "/blog/post"),
            ("/other", "/blog/post"),
            ("/blog/post", "/blog/post"),
            ("/ghost", "/post"),
            ("/missing-post", "/news/post"),
        ],
    )


async def flags(
    graph: GraphRepo, tenant: str, paths: Sequence[str]
) -> dict[str, tuple[object, ...]]:
    pages = await graph.get_pages(tenant, [url(p) for p in paths])
    return {str(p.url): (p.duplicate_group, p.is_canonical) for p in pages}


def group(group_id: int, canonical: str, *copies: str) -> DuplicateGroup:
    return DuplicateGroup(
        group_id=group_id, canonical=url(canonical), copies=tuple(url(c) for c in copies)
    )


GROUPS = (group(0, "/blog/post", "/news/post", "/post"), group(1, "/untitled-1", "/untitled-2"))
PATHS = ("/blog/post", "/news/post", "/post", "/untitled-1", "/untitled-2", "/unique")


@pytest.mark.integration
async def test_inputs_are_crawled_2xx_pages_with_a_body_sharing_it_within_their_language(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other)
    await link(graph, other, [("/unique", "/post"), ("/hub", "/post"), ("/other", "/post")])

    rows = await graph.duplicate_inputs(tenant)

    # /blog/post: /hub (linking twice) and /other, not its self-link. /post: only the
    # placeholder, which never counts. /news/post: the 404, a crawled page. Only /news/post is
    # flagged not indexable; a 2xx page without a flag is indexable.
    def expected(path: str, body: str, language: str | None, inbound: int) -> DuplicateInput:
        return DuplicateInput(
            url=url(path),
            body_hash=body,
            language=language,
            indexable=path != "/news/post",
            inbound=inbound,
        )

    assert rows == [
        expected("/blog/post", ARTICLE, "en", 2),
        expected("/news/post", ARTICLE, "en", 1),
        expected("/post", ARTICLE, "en", 0),
        expected("/untitled-1", UNTITLED, None, 0),
        expected("/untitled-2", UNTITLED, None, 0),
    ]
    assert [r.inbound for r in await graph.duplicate_inputs(other) if r.url == url("/post")] == [
        3
    ], "another tenant's links reach only its own pages"


@pytest.mark.integration
async def test_a_tenant_without_duplicates_has_no_inputs(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a", UNIQUE), page("/b", ARTICLE)])

    assert await graph.duplicate_inputs(tenant) == []


@pytest.mark.integration
async def test_groups_are_written_then_cleared_from_pages_that_leave_them(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other)
    assert await graph.write_duplicate_groups(other, (group(0, "/post", "/blog/post"),)) == 2

    assert await graph.write_duplicate_groups(tenant, GROUPS) == 5

    assert await flags(graph, tenant, PATHS) == {
        url("/blog/post"): (0, True),
        url("/news/post"): (0, False),
        url("/post"): (0, False),
        url("/untitled-1"): (1, True),
        url("/untitled-2"): (1, False),
        url("/unique"): (None, None),
    }
    assert await graph.non_canonical_copies(tenant) == {
        url("/news/post"),
        url("/post"),
        url("/untitled-2"),
    }

    assert await graph.write_duplicate_groups(tenant, (group(0, "/blog/post", "/post"),)) == 2

    assert await flags(graph, tenant, PATHS) == {
        url("/blog/post"): (0, True),
        url("/news/post"): (None, None),
        url("/post"): (0, False),
        url("/untitled-1"): (None, None),
        url("/untitled-2"): (None, None),
        url("/unique"): (None, None),
    }
    assert await graph.non_canonical_copies(tenant) == {url("/post")}

    assert await graph.write_duplicate_groups(tenant, ()) == 0

    assert set((await flags(graph, tenant, PATHS)).values()) == {(None, None)}
    assert await graph.non_canonical_copies(tenant) == frozenset()
    assert await flags(graph, other, ("/post", "/blog/post", "/news/post")) == {
        url("/post"): (0, True),
        url("/blog/post"): (0, False),
        url("/news/post"): (None, None),
    }, "another tenant's groups survive every write and clear"
    assert await graph.non_canonical_copies(other) == {url("/blog/post")}


@pytest.mark.integration
@pytest.mark.parametrize("stray", ["/nowhere", "/ghost"], ids=["missing-page", "placeholder"])
async def test_a_group_on_a_page_that_is_not_crawled_rolls_the_whole_write_back(
    graph: GraphRepo, tenant: str, stray: str
) -> None:
    await seed(graph, tenant)
    await graph.write_duplicate_groups(tenant, GROUPS)
    before = await flags(graph, tenant, PATHS)

    with pytest.raises(DatabaseWriteError, match="rolled back"):
        await graph.write_duplicate_groups(tenant, (group(0, "/blog/post", stray),))

    assert await flags(graph, tenant, PATHS) == before, "the clear rolled back with the write"


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
    ("tenant_id", "groups", "batch_size", "message"),
    [
        (" ", (), 10, "tenant_id"),
        ("t", (), 0, "batch_size"),
        ("t", (group(0, "/a", "/b"), group(0, "/c", "/d")), 10, "duplicate group ids"),
        ("t", (group(0, "/a", "/b"), group(1, "/c", "/a")), 10, "one group only"),
    ],
    ids=["blank-tenant", "zero-batch", "same-id", "page-in-two-groups"],
)
async def test_invalid_input_is_rejected_before_any_write(
    offline_graph: GraphRepo,
    tenant_id: str,
    groups: tuple[DuplicateGroup, ...],
    batch_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        await offline_graph.write_duplicate_groups(tenant_id, groups, batch_size=batch_size)
