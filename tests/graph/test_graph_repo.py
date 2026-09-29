from __future__ import annotations

import pytest

from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseReadError,
    DatabaseUnavailableError,
    DatabaseWriteError,
)
from linking_engine.graph.repo import (
    LINK_PROPERTIES,
    PAGE_PROPERTIES,
    VECTOR_INDEXES,
    GraphRepo,
    load_migrations,
    split_statements,
    status_issue,
)
from linking_engine.models import ActionType, IssueFlag, Link, Page, TenantGraphCounts

BASE = "example.com"


def page(path: str, **fields: object) -> Page:
    return Page.model_validate(
        {"url": f"{BASE}{path}", "status_code": 200, "word_count": 100, **fields}
    )


def link(source: str, target: str, position: int, anchor: str = "anchor") -> Link:
    return Link(
        source_url=str(f"{BASE}{source}"),
        target_url=str(f"{BASE}{target}"),
        position=position,
        anchor_text=anchor,
        surrounding_text=f"text with {anchor}",
    )


def url(path: str) -> str:
    return f"{BASE}{path}"


# ── no database ─────────────────────────────────────────────────────────────


def test_property_maps_cover_every_model_field() -> None:
    assert set(PAGE_PROPERTIES) == set(Page.model_fields)
    assert set(LINK_PROPERTIES) | {"source_url", "target_url"} == set(Link.model_fields)


def test_split_statements_drops_comments_and_blanks() -> None:
    text = "// header\nCREATE INDEX a;\n\n// note\nCREATE INDEX b\n  FOR (n) ON (n.x);\n"
    assert split_statements(text) == ("CREATE INDEX a", "CREATE INDEX b\n  FOR (n) ON (n.x)")


def test_migrations_are_packaged_in_filename_order() -> None:
    names = [m.name for m in load_migrations()]
    assert names == sorted(names)
    assert names[:3] == ["001_schema.cypher", "002_vectors.cypher", "003_status_code.cypher"]
    assert all(m.statements for m in load_migrations())


async def test_unreachable_server_raises_unavailable() -> None:
    with pytest.raises(DatabaseUnavailableError):
        await GraphRepo.connect("bolt://127.0.0.1:1", "neo4j", "x", connection_timeout=1)


async def test_invalid_uri_raises_unavailable() -> None:
    with pytest.raises(DatabaseUnavailableError, match="invalid connection settings"):
        await GraphRepo.connect("http://nowhere", "neo4j", "x")


# ── against a real Neo4j ────────────────────────────────────────────────────


@pytest.mark.integration
async def test_wrong_password_raises_auth_error(neo4j_server: tuple[str, str, str]) -> None:
    uri, user, _ = neo4j_server
    with pytest.raises(DatabaseAuthError):
        await GraphRepo.connect(uri, user, "wrong-password")


@pytest.mark.integration
async def test_migrations_apply_twice_and_every_statement_is_idempotent(graph: GraphRepo) -> None:
    assert await graph.migrate() == ()
    for migration in load_migrations():
        await graph.apply_migration(migration)
    assert await graph.migrate() == ()


@pytest.mark.integration
async def test_schema_after_migration(graph: GraphRepo) -> None:
    await graph.check_server()
    assert await graph.vector_index_dimensions() == dict.fromkeys(VECTOR_INDEXES, 2048)
    constraints = {row["name"] for row in await graph._auto("SHOW CONSTRAINTS YIELD name")}
    assert {"page_tenant_url", "keyword_tenant_text_lang"} <= constraints
    assert not {"page_url", "keyword_text_lang"} & constraints
    indexes = {row["name"] for row in await graph._auto("SHOW INDEXES YIELD name")}
    assert {
        "page_tenant",
        "keyword_tenant",
        "page_topic",
        "page_indexable",
        "page_lifecycle",
        "page_link_community",
        "page_kw_community",
        "link_audited",
        "link_anchor_type",
    } <= indexes


@pytest.mark.integration
async def test_legacy_status_properties_are_renamed(graph: GraphRepo, tenant: str) -> None:
    await graph._auto(
        "CREATE (:Page {tenantId: $t, url: 'x.test/a', httpStatus: 200})"
        "-[:LINKS_TO {position: 0, targetHttpStatus: 404}]->"
        "(:Page {tenantId: $t, url: 'x.test/b'})",
        t=tenant,
    )
    migration = next(m for m in load_migrations() if m.name == "003_status_code.cypher")
    await graph.apply_migration(migration)
    rows = await graph._auto(
        "MATCH (a:Page {tenantId: $t})-[r]->() "
        "RETURN a.statusCode AS s, a.httpStatus AS h, r.targetStatusCode AS ts, r.targetHttpStatus AS th",
        t=tenant,
    )
    assert rows == [{"s": 200, "h": None, "ts": 404, "th": None}]


@pytest.mark.integration
async def test_pages_round_trip(graph: GraphRepo, tenant: str) -> None:
    pages = [page("/a", content_hash="h1", language="en"), page("/b")]
    assert await graph.upsert_pages(tenant, pages) == 2
    assert await graph.get_pages(tenant, [url("/b"), url("/a"), url("/missing")]) == [
        pages[1],
        pages[0],
    ]


@pytest.mark.integration
async def test_upsert_keeps_computed_properties_and_clears_dropped_crawl_fields(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(tenant, [page("/a", language="en")])
    await graph._auto("MATCH (p:Page {tenantId: $t}) SET p.pageRank = 0.5", t=tenant)
    await graph.upsert_pages(tenant, [page("/a")])
    [stored] = await graph.get_pages(tenant, [url("/a")])
    assert stored.page_rank == 0.5
    assert stored.language is None


@pytest.mark.integration
async def test_vectors_are_returned_only_on_request(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a")])
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) SET p.content_embedding = [0.1, 0.2]", t=tenant
    )
    [without] = await graph.get_pages(tenant, [url("/a")])
    [with_vectors] = await graph.get_pages(tenant, [url("/a")], include_vectors=True)
    assert without.content_embedding is None
    assert with_vectors.content_embedding == (0.1, 0.2)


@pytest.mark.integration
async def test_placeholders(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/crawled")])
    assert await graph.upsert_placeholders(tenant, [url("/crawled"), url("/ghost")]) == 2
    crawled, ghost = await graph.get_pages(tenant, [url("/crawled"), url("/ghost")])
    assert not crawled.is_placeholder
    assert crawled.status_code == 200
    assert ghost.is_placeholder
    assert ghost.status_code is None

    await graph.upsert_pages(tenant, [page("/ghost")])
    [now_crawled] = await graph.get_pages(tenant, [url("/ghost")])
    assert not now_crawled.is_placeholder
    assert await graph.counts(tenant) == TenantGraphCounts(pages=2, placeholders=0, links=0)


@pytest.mark.integration
async def test_upsert_pages_rejects_placeholders(graph: GraphRepo, tenant: str) -> None:
    with pytest.raises(ValueError, match="upsert_placeholders"):
        await graph.upsert_pages(tenant, [page("/a", is_placeholder=True)])


@pytest.mark.integration
async def test_replace_links_upserts_then_prunes(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a"), page("/b"), page("/c")])
    first = [link("/a", "/b", 0), link("/a", "/c", 1), link("/a", "/b", 2), link("/b", "/c", 0)]
    assert await graph.replace_links(tenant, [url("/a"), url("/b")], first) == (4, 0)
    assert await graph.links_from(tenant, [url("/a"), url("/b")]) == [
        lk.model_copy(update={"target_status_code": 200}) for lk in first
    ]

    # /a: position 1 now points elsewhere and position 2 is gone; /b lost all links.
    second = [link("/a", "/b", 0), link("/a", "/b", 1, "moved")]
    assert await graph.replace_links(tenant, [url("/a"), url("/b")], second) == (2, 3)
    stored = await graph.links_from(tenant, [url("/a"), url("/b")])
    assert [(str(s.target_url), s.position, s.anchor_text) for s in stored] == [
        (url("/b"), 0, "anchor"),
        (url("/b"), 1, "moved"),
    ]


@pytest.mark.integration
async def test_link_to_missing_page_fails_the_write(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(tenant, [page("/a")])
    with pytest.raises(DatabaseWriteError, match="wrote 0 of 1"):
        await graph.replace_links(tenant, [url("/a")], [link("/a", "/nowhere", 0)])


@pytest.mark.integration
async def test_links_must_come_from_listed_sources(graph: GraphRepo, tenant: str) -> None:
    with pytest.raises(ValueError, match="not listed in sources"):
        await graph.replace_links(tenant, [url("/a")], [link("/b", "/a", 0)])


@pytest.mark.integration
async def test_batches_cover_everything(graph: GraphRepo, tenant: str) -> None:
    pages = [page(f"/p{i}") for i in range(5)]
    assert await graph.upsert_pages(tenant, pages, batch_size=2) == 5
    links = [link("/p0", f"/p{i}", i) for i in range(1, 5)]
    assert await graph.replace_links(tenant, [url("/p0")], links, batch_size=3) == (4, 0)
    batches = [b async for b in graph.iter_pages(tenant, batch_size=2)]
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [str(p.url) for b in batches for p in b] == sorted(str(p.url) for p in pages)
    assert len(await graph.get_pages(tenant, [str(p.url) for p in pages], batch_size=2)) == 5


@pytest.mark.integration
async def test_tenants_are_isolated(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    for t in (tenant, other):
        await graph.upsert_pages(t, [page("/a"), page("/b")])
        await graph.replace_links(t, [url("/a")], [link("/a", "/b", 0)])

    await graph.replace_links(tenant, [url("/a")], [])
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    assert await graph.counts(tenant) == TenantGraphCounts(pages=2, placeholders=1, links=0)
    assert await graph.counts(other) == TenantGraphCounts(pages=2, placeholders=0, links=1)
    assert await graph.get_pages(other, [url("/ghost")]) == []

    assert await graph.delete_tenant(tenant, batch_size=1) == 3
    assert await graph.counts(tenant) == TenantGraphCounts(pages=0, placeholders=0, links=0)
    assert await graph.counts(other) == TenantGraphCounts(pages=2, placeholders=0, links=1)
    await graph.delete_tenant(other)


@pytest.mark.integration
async def test_node_that_does_not_fit_the_model_raises_read_error(
    graph: GraphRepo, tenant: str
) -> None:
    await graph._auto("CREATE (:Page {tenantId: $t, url: 'x.test/a', statusCode: 42})", t=tenant)
    with pytest.raises(DatabaseReadError, match="does not fit the Page model"):
        await graph.get_pages(tenant, ["x.test/a"])


@pytest.mark.integration
async def test_empty_tenant_id_is_rejected(graph: GraphRepo) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await graph.counts(" ")


@pytest.mark.parametrize(
    ("code", "flag"),
    [
        (None, None),
        (200, None),
        (204, None),
        (301, IssueFlag.REDIRECTED),
        (308, IssueFlag.REDIRECTED),
        (404, IssueFlag.BROKEN),
        (429, IssueFlag.BROKEN),
        (503, IssueFlag.BROKEN),
    ],
)
def test_status_issue(code: int | None, flag: IssueFlag | None) -> None:
    assert status_issue(code) is flag


@pytest.mark.integration
async def test_links_to_non_2xx_pages_carry_the_status_and_count_as_fix(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(
        tenant,
        [
            page("/src"),
            page("/ok"),
            page("/moved", status_code=301),
            page("/gone", status_code=404),
        ],
    )
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    links = [
        link("/src", target, i) for i, target in enumerate(["/ok", "/moved", "/gone", "/ghost"])
    ]
    await graph.replace_links(tenant, [url("/src")], links)

    stored = {str(lk.target_url): lk for lk in await graph.links_from(tenant, [url("/src")])}
    targets = ("/ok", "/moved", "/gone", "/ghost")
    statuses = {target: stored[url(target)].target_status_code for target in targets}
    assert statuses == {"/ok": 200, "/moved": 301, "/gone": 404, "/ghost": None}
    # Flags and verdicts are the link audit's to write, never the load's.
    assert all(lk.issue_flags == frozenset() and lk.verdict is None for lk in stored.values())
    counts = await graph.counts(tenant)
    assert (counts.redirected_pages, counts.broken_pages, counts.fix_links) == (1, 1, 2)


@pytest.mark.integration
async def test_status_change_updates_inbound_links_and_keeps_the_audit_result(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(tenant, [page("/src"), page("/t", status_code=503)])
    await graph.replace_links(tenant, [url("/src")], [link("/src", "/t", 0)])
    await graph._auto(
        "MATCH (:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "SET r.issueFlags = ['BROKEN', 'GENERIC'], r.verdict = 'FIX'",
        t=tenant,
    )
    assert (await graph.counts(tenant)).fix_links == 1

    await graph.upsert_pages(tenant, [page("/t")])
    [recovered] = await graph.links_from(tenant, [url("/src")])
    assert recovered.target_status_code == 200
    assert recovered.issue_flags == {IssueFlag.BROKEN, IssueFlag.GENERIC}
    assert recovered.verdict is ActionType.FIX
    assert (await graph.counts(tenant)).fix_links == 0

    await graph.upsert_pages(tenant, [page("/t", status_code=410)])
    [broken] = await graph.links_from(tenant, [url("/src")])
    assert broken.target_status_code == 410
    assert (await graph.counts(tenant)).fix_links == 1


@pytest.mark.integration
async def test_link_graph_is_tenant_scoped_and_keeps_orphans_and_placeholders(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    for t in (tenant, other):
        await graph.upsert_pages(t, [page("/a"), page("/b"), page("/orphan")])
        await graph.replace_links(t, [url("/a")], [link("/a", "/b", 0), link("/a", "/b", 1)])
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await graph.replace_links(other, [url("/b")], [link("/b", "/a", 0)])

    snapshot = await graph.link_graph(tenant)

    assert snapshot.pages == tuple(sorted(url(p) for p in ("/a", "/b", "/ghost", "/orphan")))
    assert dict(zip(snapshot.pages, snapshot.placeholders, strict=True))[url("/ghost")] is True
    assert sorted(snapshot.links) == [(url("/a"), url("/b"))] * 2
    await graph.delete_tenant(other)
