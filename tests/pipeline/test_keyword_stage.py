"""resolve_tenant_keywords end to end: Mongo pages, strategic rows and GSC rows in, one keyword
per crawled 2xx page resolved and every source's TARGETS_KEYWORD edges replaced in Neo4j.

Each page is built for one rung, because a corpus where every page has a strategic keyword
never reaches the others."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest
from structlog.testing import capture_logs

from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.models import (
    AnchorRules,
    Heading,
    KeywordReport,
    KeywordRung,
    KeywordSource,
    LanguageRules,
    Page,
    PageRecord,
)
from linking_engine.pipeline.keywords import resolve_tenant_keywords, summarise_keywords

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

STRATEGIC, OBSERVED, INFERRED = (
    KeywordSource.CLIENT_STRATEGIC,
    KeywordSource.GSC_OBSERVED,
    KeywordSource.INFERRED,
)


def url(path: str) -> str:
    return f"example.com{path}"


# Each page's status, title, H1, body and stored language.
PAGES: dict[str, tuple[int, str | None, str | None, str, str | None]] = {
    "/strategic": (200, "Trail Shoes | Acme", "Trail Shoes", "Trail shoes.", "en"),
    "/gsc": (
        200,
        "Rain Jackets | Acme",
        "Waterproof Rain Jackets | Acme",
        "Our waterproof rain jackets keep you dry.",
        "en",
    ),
    "/h1": (200, "Tents | Acme", "Four Season Tents | Acme", "Tents for winter.", "en"),
    "/title": (200, "Camp Stoves | Acme", None, "", "en"),
    # No language stored: the tenant's default applies.
    "/empty": (200, None, None, "", None),
    "/missing": (404, "Not Found | Acme", None, "", "en"),
    "/de/zelte": (200, "Zelte | Acme", "Zelte", "Zelte fuer den Winter.", "de"),
}


def record(path: str) -> PageRecord:
    return page_record(path, *PAGES[path])


def page_record(
    path: str,
    status: int,
    title: str | None,
    h1: str | None,
    body: str,
    language: str | None,
    headings: tuple[Heading, ...] = (),
) -> PageRecord:
    return PageRecord.model_validate(
        {
            "url": url(path),
            "status_code": status,
            "usable": True,
            "meta_title": title,
            "meta_description": None,
            "h1": h1,
            "headings": headings,
            "body_text": body,
            "word_count": len(body.split()),
            "link_count": 0,
            "content_hash": None,
            "body_hash": body_hash(body),
            "scraped_at": None,
            "source": "test",
            "crawl_url": f"https://{url(path)}",
            "language": language,
        }
    )


def gsc_rows(tenant: str) -> list[dict[str, object]]:
    # 120 filler rows on an uncrawled url make the curve: 0.3, 0.15, 0.08, 0.04 before pooling.
    filler: list[dict[str, object]] = [
        {
            "tenantId": tenant,
            "url": url("/elsewhere"),
            "query": f"filler {i}",
            "impressions": 100,
            "clicks": (30, 15, 8, 4)[i % 4],
            "position": float(i % 4 + 1),
        }
        for i in range(120)
    ]
    page_rows = [
        ("waterproof rain jackets", 400, 20, 3.0),
        ("rain jackets", 60, 3, 2.0),
        # The most upside, but the page never mentions boots: rejected by the quality bar.
        ("hiking boots", 5000, 50, 4.0),
    ]
    on_page: list[dict[str, object]] = [
        {
            "tenantId": tenant,
            "url": url("/gsc"),
            "query": query,
            "impressions": impressions,
            "clicks": clicks,
            "position": position,
        }
        for query, impressions, clicks, position in page_rows
    ]
    return filler + on_page


def strategic_rows(tenant: str, keyword: str = "Trail Running Shoes") -> list[dict[str, object]]:
    return [
        {"url": url("/strategic"), "keyword": keyword, "priority": 3, "isPrimary": True},
        {"url": url("/strategic"), "keyword": "rain jackets", "priority": 5, "isPrimary": False},
        {"url": url("/gone"), "keyword": "tents", "priority": 1, "isPrimary": True},
        {"url": url("/ghost"), "keyword": "stoves", "priority": 1, "isPrimary": True},
    ]


async def seed(graph: GraphRepo, mongo: MongoRepo, tenant: str, keyword: str) -> None:
    await mongo.set_language_rules(tenant, LanguageRules(default_language="en"))
    await mongo.write_pages(tenant, [record(path) for path in PAGES], [])
    await mongo._db["strategic_keywords"].insert_many(
        [{**row, "tenantId": tenant, "language": "en"} for row in strategic_rows(tenant, keyword)]
    )
    await mongo._db["gsc_queries"].insert_many(gsc_rows(tenant))
    await graph.upsert_pages(
        tenant,
        [
            Page(url=url(path), status_code=status, language=language)
            for path, (status, _, _, _, language) in PAGES.items()
        ],
    )
    await graph.upsert_placeholders(tenant, [url("/ghost")])


Edge = tuple[str, str, str, str, str | None]


async def edges(graph: GraphRepo, tenant: str) -> set[Edge]:
    rows = await graph._auto(
        "MATCH (p:Page {tenantId: $t})-[e:TARGETS_KEYWORD]->(k:Keyword {tenantId: $t}) "
        "RETURN p.url AS url, k.text AS text, k.language AS language, e.source AS source, "
        "e.rung AS rung",
        t=tenant,
    )
    return {
        (str(r["url"]), str(r["text"]), str(r["language"]), str(r["source"]), r["rung"])  # type: ignore[misc]
        for r in rows
    }


EXPECTED: set[Edge] = {
    (url("/strategic"), "Trail Running Shoes", "en", STRATEGIC, KeywordRung.STRATEGIC),
    (url("/strategic"), "rain jackets", "en", STRATEGIC, None),
    (url("/gsc"), "waterproof rain jackets", "en", OBSERVED, KeywordRung.GSC),
    # The page's next usable query, ranked after the resolved one.
    (url("/gsc"), "rain jackets", "en", OBSERVED, None),
    (url("/h1"), "Four Season Tents", "en", INFERRED, KeywordRung.H1),
    (url("/title"), "Camp Stoves", "en", INFERRED, KeywordRung.TITLE),
    (url("/de/zelte"), "Zelte", "de", INFERRED, KeywordRung.H1),
}


@pytest.mark.integration
async def test_every_rung_resolves_its_page_and_writes_its_edge(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, mongo, other, "Other Tenant Keyword")
    await resolve_tenant_keywords(graph, mongo, other)
    other_edges = await edges(graph, other)
    await seed(graph, mongo, tenant, "Trail Running Shoes")

    with capture_logs() as logs:
        report = await resolve_tenant_keywords(graph, mongo, tenant)

    assert (report.pages, report.resolved) == (6, 5), "the 404 is not resolved, /empty cannot be"
    assert report.by_rung == {
        KeywordRung.STRATEGIC: 1,
        KeywordRung.GSC: 1,
        KeywordRung.H1: 2,
        KeywordRung.TITLE: 1,
    }
    assert (report.gsc_enabled, report.gsc_rows, report.gsc_rejected) == (True, 123, 1)
    assert (report.brand_suffix, report.brand_prefix) == ("Acme", None)
    assert report.fallbacks_rejected == {"h1_missing": 2, "title_missing": 1}
    assert report.long_fallbacks == 0
    assert report.by_language == {"en": 5, "de": 1}
    assert report.skipped_rows == 2, "rows on an uncrawled url and on a placeholder"
    assert report.edges_written == {STRATEGIC: 2, OBSERVED: 2, INFERRED: 3}

    assert await edges(graph, tenant) == EXPECTED
    rows = await graph._auto(
        "MATCH (p:Page {tenantId: $t})-[e:TARGETS_KEYWORD]->() WHERE e.rung IS NOT NULL "
        "RETURN p.url AS url, count(e) AS n",
        t=tenant,
    )
    assert {r["url"]: r["n"] for r in rows} == dict.fromkeys(
        [url(p) for p in ("/strategic", "/gsc", "/h1", "/title", "/de/zelte")], 1
    ), "exactly one resolved edge per resolved page"
    [primary] = await graph._auto(
        "MATCH (:Page {tenantId: $t, url: $u})-[e:TARGETS_KEYWORD]->(:Keyword {text: $k}) "
        "RETURN e.priority AS priority, e.isPrimary AS primary",
        t=tenant,
        u=url("/strategic"),
        k="Trail Running Shoes",
    )
    assert (primary["priority"], primary["primary"]) == (3, True)

    assert await edges(graph, other) == other_edges, "another tenant's run left its edges"
    assert (
        url("/strategic"),
        "Other Tenant Keyword",
        "en",
        STRATEGIC,
        KeywordRung.STRATEGIC,
    ) in other_edges

    [line] = [entry for entry in logs if entry["event"] == "keywords.resolved"]
    logged = " ".join(str(value) for value in line.values())
    leaked = [u for u in (*map(url, PAGES), url("/gone"), url("/ghost")) if u in logged]
    assert leaked == [], f"urls in the log line: {leaked}"


@pytest.mark.integration
async def test_a_rerun_replaces_edges_and_removes_only_a_dropped_keyword(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    await seed(graph, mongo, tenant, "Trail Running Shoes")
    await resolve_tenant_keywords(graph, mongo, tenant)

    again = await resolve_tenant_keywords(graph, mongo, tenant)
    assert again.stale_edges_deleted == {STRATEGIC: 0, OBSERVED: 0, INFERRED: 0}
    assert await edges(graph, tenant) == EXPECTED

    await mongo._db["strategic_keywords"].delete_one(
        {"tenantId": tenant, "keyword": "rain jackets"}
    )
    dropped = await resolve_tenant_keywords(graph, mongo, tenant)

    assert dropped.stale_edges_deleted == {STRATEGIC: 1, OBSERVED: 0, INFERRED: 0}
    assert await edges(graph, tenant) == EXPECTED - {
        (url("/strategic"), "rain jackets", "en", STRATEGIC, None)
    }


@pytest.mark.integration
async def test_a_tenant_without_gsc_or_strategic_data_resolves_from_headings(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    # A template H1 on three pages cannot tell them apart; their titles can.
    templated = [
        page_record(f"/t{i}", 200, f"{name} | Acme", "Our Products", "", "en")
        for i, name in enumerate(("Tents", "Stoves", "Boots"))
    ]
    records = [record("/gsc"), record("/title"), *templated]
    await mongo.write_pages(tenant, records, [])
    await graph.upsert_pages(tenant, [Page(url=r.url, status_code=200) for r in records])

    report = await resolve_tenant_keywords(graph, mongo, tenant)

    assert (report.gsc_enabled, report.gsc_rows) == (False, 0)
    assert report.by_rung == {
        KeywordRung.STRATEGIC: 0,
        KeywordRung.GSC: 0,
        KeywordRung.H1: 1,
        KeywordRung.TITLE: 4,
    }
    assert report.fallbacks_rejected == {"h1_missing": 1, "h1_repeated": 3}
    assert await edges(graph, tenant) == {
        (url("/gsc"), "Waterproof Rain Jackets", "en", INFERRED, KeywordRung.H1),
        (url("/title"), "Camp Stoves", "en", INFERRED, KeywordRung.TITLE),
        (url("/t0"), "Tents", "en", INFERRED, KeywordRung.TITLE),
        (url("/t1"), "Stoves", "en", INFERRED, KeywordRung.TITLE),
        (url("/t2"), "Boots", "en", INFERRED, KeywordRung.TITLE),
    }


LONG_H1 = " ".join(f"word{i}" for i in range(20))
# H1 formats not seen on any known tenant: path -> (title, H1, other headings).
UNSEEN: dict[str, tuple[str, str | None, tuple[Heading, ...]]] = {
    "/p1": ("Acme | Tents", "Products", ()),
    "/p2": ("Acme | Stoves", "Products", ()),
    "/p3": ("Acme | Boots", "products", ()),
    "/read": ("Acme | Lanterns", "Read more", ()),
    "/mehr": ("Acme | Zelte", "Mehr erfahren", ()),
    "/brand": ("Acme | Maps", "ACME", ()),
    "/pricing": ("Acme | Pricing", "Acme | Pricing Plans", ()),
    "/year": ("Acme | Calendar", "2026", ()),
    "/arrow": ("Acme | Arrows", "\u00bb", ()),
    "/second": (
        "Acme | Shoes",
        None,
        (Heading(level=2, text="Sizing"), Heading(level=1, text="Trail Running Shoes")),
    ),
    "/long": ("Acme | Guide", LONG_H1, ()),
    # The tenant's AnchorRules add "special offers" and remove "learn more".
    "/offers": ("Acme | Offers", "Special Offers", ()),
    "/learn": ("Acme | Courses", "Learn more", ()),
}


@pytest.mark.integration
async def test_unseen_h1_formats_fall_back_for_the_recorded_reason(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    records = [
        page_record(path, 200, title, h1, "", "en", headings)
        for path, (title, h1, headings) in UNSEEN.items()
    ]
    await mongo.write_pages(tenant, records, [])
    await mongo.set_anchor_rules(
        tenant,
        AnchorRules(
            generic_add=frozenset({"special offers"}), generic_remove=frozenset({"learn more"})
        ),
    )
    await graph.upsert_pages(tenant, [Page(url=r.url, status_code=200) for r in records])

    report = await resolve_tenant_keywords(graph, mongo, tenant)

    assert (report.brand_prefix, report.brand_suffix) == ("Acme", None)
    assert report.fallbacks_rejected == {"h1_repeated": 3, "h1_generic": 5, "h1_brand": 1}
    assert report.long_fallbacks == 1
    assert report.by_rung == {
        KeywordRung.STRATEGIC: 0,
        KeywordRung.GSC: 0,
        KeywordRung.H1: 4,
        KeywordRung.TITLE: 9,
    }
    title, h1 = KeywordRung.TITLE, KeywordRung.H1
    assert {(u, text, rung) for u, text, _, _, rung in await edges(graph, tenant)} == {
        (url("/p1"), "Tents", title),
        (url("/p2"), "Stoves", title),
        (url("/p3"), "Boots", title),
        (url("/read"), "Lanterns", title),
        (url("/mehr"), "Zelte", title),
        (url("/brand"), "Maps", title),
        (url("/pricing"), "Pricing Plans", h1),
        (url("/year"), "Calendar", title),
        (url("/arrow"), "Arrows", title),
        (url("/second"), "Trail Running Shoes", h1),
        (url("/long"), LONG_H1, h1),
        (url("/offers"), "Offers", title),
        (url("/learn"), "Learn more", h1),
    }


async def seed_placeholder(graph: GraphRepo, mongo: MongoRepo, tenant: str, h1: str) -> None:
    # Two titles share the suffix, the minimum for detecting it.
    records = [
        page_record("/start", 200, "Trail Shoes | Acme", h1, "", "en"),
        page_record("/other", 200, "Rain Jackets | Acme", "Rain Jackets", "", "en"),
    ]
    await mongo.write_pages(tenant, records, [])
    await graph.upsert_pages(tenant, [Page(url=r.url, status_code=200) for r in records])


@pytest.mark.parametrize("h1", ["Home", "Welcome"])
@pytest.mark.integration
async def test_placeholder_h1s_are_rejected_as_generic(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, h1: str
) -> None:
    await seed_placeholder(graph, mongo, tenant, h1)

    report = await resolve_tenant_keywords(graph, mongo, tenant)

    assert report.fallbacks_rejected == {"h1_generic": 1}
    assert {(u, text, rung) for u, text, _, _, rung in await edges(graph, tenant)} == {
        (url("/start"), "Trail Shoes", KeywordRung.TITLE),
        (url("/other"), "Rain Jackets", KeywordRung.H1),
    }


@pytest.mark.integration
async def test_a_tenant_can_keep_a_placeholder_h1_through_its_anchor_rules(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    await seed_placeholder(graph, mongo, tenant, "Home")
    await mongo.set_anchor_rules(tenant, AnchorRules(generic_remove=frozenset({"Home"})))

    report = await resolve_tenant_keywords(graph, mongo, tenant)

    assert report.fallbacks_rejected == {}
    assert (url("/start"), "Home", KeywordRung.H1) in {
        (u, text, rung) for u, text, _, _, rung in await edges(graph, tenant)
    }


def report(*, gsc: bool, brand: str | None) -> KeywordReport:
    return KeywordReport(
        tenant_id="acme",
        pages=6,
        resolved=5,
        by_rung={
            KeywordRung.STRATEGIC: 1,
            KeywordRung.GSC: 1 if gsc else 0,
            KeywordRung.H1: 2,
            KeywordRung.TITLE: 1 if gsc else 2,
        },
        gsc_enabled=gsc,
        gsc_rows=123 if gsc else 40,
        gsc_rejected=1 if gsc else 0,
        brand_suffix=brand,
        edges_written={STRATEGIC: 2, OBSERVED: 1 if gsc else 0, INFERRED: 3},
        stale_edges_deleted={STRATEGIC: 1, OBSERVED: 0, INFERRED: 0},
        skipped_rows=2,
        by_language={"en": 5, "de": 1},
        seconds=0.5,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


def test_the_summary_states_the_rungs_edges_and_why_gsc_ran() -> None:
    summary = summarise_keywords(report(gsc=True, brand="Acme"))

    for fact in (
        "acme",
        "5 of 6",
        "1 STRATEGIC, 1 GSC, 2 H1, 1 TITLE",
        "5 en",
        "1 de",
        "GSC rung enabled from 123",
        "1 queries failed the quality bar",
        "'Acme'",
        "CLIENT_STRATEGIC: 2 written, 1 stale deleted",
        "2 strategic rows",
    ):
        assert fact in summary, f"{fact!r} missing from:\n{summary}"


def test_the_summary_says_why_gsc_was_skipped_and_no_suffix_stripped() -> None:
    summary = summarise_keywords(report(gsc=False, brand=None))

    assert "GSC rung skipped: 40 query rows" in summary
    assert "none stripped" in summary


async def test_a_blank_tenant_is_refused_before_any_read() -> None:
    unused = cast("GraphRepo", object())
    with pytest.raises(ValueError, match="tenant_id"):
        await resolve_tenant_keywords(unused, cast("MongoRepo", object()), " ")
