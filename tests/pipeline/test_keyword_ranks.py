"""The ranked keyword set of each resolved page: rank 1 the resolved keyword, then the page's
other strategic keywords, then its best further GSC queries, ranks consecutive across sources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from test_keyword_stage import page_record, url

from linking_engine.models import (
    KeywordRung,
    KeywordSource,
    Page,
    ResolvedKeyword,
    StrategicKeyword,
)
from linking_engine.pipeline.keywords import ranked_targets, resolve_tenant_keywords

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

STRATEGIC, OBSERVED, INFERRED = (
    KeywordSource.CLIENT_STRATEGIC,
    KeywordSource.GSC_OBSERVED,
    KeywordSource.INFERRED,
)
BODY = (
    "Trail running shoes, waterproof trail shoes, rain jackets, tents, trail socks, trail "
    "gaiters, trail poles, trail maps and trail lights. Waterproof jackets and rain shells."
)
PAGES = {
    "/strategic": ("Trail Shoes | Acme", "Trail Shoes"),
    "/gsc": ("Rain Jackets | Acme", "Rain Jackets"),
    "/h1": ("Tents | Acme", "Four Season Tents"),
}
# Primary first, then priority 5, 2 and none: the order of ranks 1 to 4.
STRATEGIC_ROWS = [
    ("Trail Running Shoes", 3, True),
    ("tents", 2, False),
    ("rain jackets", 5, False),
    ("trail socks", None, False),
]
# All at position 3, so opportunity follows impressions.
QUERIES = {
    "/strategic": [
        ("trail running shoes", 900),
        ("Rain Jackets", 800),
        ("waterproof trail shoes", 700),
        ("trail gaiters", 600),
        ("trail poles", 500),
        ("trail maps", 400),
        ("trail lights", 300),
        ("hiking boots", 5000),
        ("trail", 40),
    ],
    "/gsc": [
        ("rain jackets", 500),
        ("waterproof jackets", 400),
        ("rain shells", 300),
        ("Rain  Jackets", 200),
    ],
}


async def seed(graph: GraphRepo, mongo: MongoRepo, tenant: str, *, curve: bool) -> None:
    records = [page_record(p, 200, title, h1, BODY, "en") for p, (title, h1) in PAGES.items()]
    await mongo.write_pages(tenant, records, [])
    await graph.upsert_pages(tenant, [Page(url=r.url, status_code=200) for r in records])
    await mongo._db["strategic_keywords"].insert_many(
        [
            {
                "tenantId": tenant,
                "url": url("/strategic"),
                "keyword": keyword,
                "language": "en",
                "priority": priority,
                "isPrimary": primary,
            }
            for keyword, priority, primary in STRATEGIC_ROWS
        ]
    )
    if not curve:
        return
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
    on_pages: list[dict[str, object]] = [
        {
            "tenantId": tenant,
            "url": url(path),
            "query": query,
            "impressions": impressions,
            "clicks": impressions // 20,
            "position": 3.0,
        }
        for path, rows in QUERIES.items()
        for query, impressions in rows
    ]
    await mongo._db["gsc_queries"].insert_many(filler + on_pages)


async def ranked(graph: GraphRepo, tenant: str) -> dict[str, list[tuple[object, ...]]]:
    """Per page, (rank, text, source, rung) of every keyword edge, in rank order."""
    rows = await graph._auto(
        "MATCH (p:Page {tenantId: $t})-[r:TARGETS_KEYWORD]->(k:Keyword {tenantId: $t}) "
        "RETURN p.url AS url, r.rank AS rank, k.text AS text, r.source AS source, "
        "r.rung AS rung ORDER BY url, rank",
        t=tenant,
    )
    found: dict[str, list[tuple[object, ...]]] = {}
    for row in rows:
        found.setdefault(str(row["url"]), []).append(
            (row["rank"], row["text"], row["source"], row["rung"])
        )
    return found


@pytest.mark.integration
async def test_resolved_pages_get_a_ranked_keyword_set(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    await seed(graph, mongo, tenant, curve=True)

    report = await resolve_tenant_keywords(graph, mongo, tenant)

    found = await ranked(graph, tenant)
    assert found[url("/strategic")] == [
        (1, "Trail Running Shoes", STRATEGIC, KeywordRung.STRATEGIC),
        (2, "rain jackets", STRATEGIC, None),
        (3, "tents", STRATEGIC, None),
        (4, "trail socks", STRATEGIC, None),
        # Duplicates of the set are skipped, the unusable are rejected, at most four are kept.
        (5, "waterproof trail shoes", OBSERVED, None),
        (6, "trail gaiters", OBSERVED, None),
        (7, "trail poles", OBSERVED, None),
        (8, "trail maps", OBSERVED, None),
    ]
    assert found[url("/gsc")] == [
        (1, "rain jackets", OBSERVED, KeywordRung.GSC),
        (2, "waterproof jackets", OBSERVED, None),
        (3, "rain shells", OBSERVED, None),
    ]
    assert found[url("/h1")] == [(1, "Four Season Tents", INFERRED, KeywordRung.H1)]
    for page, edges in found.items():
        assert [edge[0] for edge in edges] == list(range(1, len(edges) + 1)), page
    assert (report.secondary_keywords, report.pages_with_secondaries) == (9, 2)


@pytest.mark.integration
async def test_without_a_ctr_curve_only_strategic_keywords_rank_after_the_first(
    graph: GraphRepo, mongo: MongoRepo, tenant: str
) -> None:
    await seed(graph, mongo, tenant, curve=False)

    report = await resolve_tenant_keywords(graph, mongo, tenant)

    found = await ranked(graph, tenant)
    assert [(rank, source) for rank, _, source, _ in found[url("/strategic")]] == [
        (1, STRATEGIC),
        (2, STRATEGIC),
        (3, STRATEGIC),
        (4, STRATEGIC),
    ]
    assert found[url("/gsc")] == [(1, "Rain Jackets", INFERRED, KeywordRung.H1)]
    assert (report.gsc_enabled, report.secondary_keywords, report.pages_with_secondaries) == (
        False,
        3,
        1,
    )


# ── ranked_targets ──────────────────────────────────────────────────────────


def strategic_row(
    path: str, keyword: str, priority: int | None, primary: bool = False
) -> StrategicKeyword:
    return StrategicKeyword(
        url=url(path), keyword=keyword, language="en", priority=priority, is_primary=primary
    )


def resolved(path: str, text: str, rung: KeywordRung, language: str = "en") -> ResolvedKeyword:
    return ResolvedKeyword(
        url=url(path),
        text=text,
        language=language,
        rung=rung,
        opportunity_value=10.0 if rung is KeywordRung.GSC else None,
    )


def flat(targets: dict[KeywordSource, list[Any]]) -> set[tuple[object, ...]]:
    return {
        (t.url, t.rank, t.text, t.language, source, t.rung)
        for source, rows in targets.items()
        for t in rows
    }


def test_ranks_run_consecutively_per_page_across_sources() -> None:
    strategic = [
        strategic_row("/s", "Trail Shoes", 3, primary=True),
        strategic_row("/s", "rain jackets", 5),
        strategic_row("/s", "tents", None),
        strategic_row("/missing", "stoves", 5),
    ]
    keywords = {
        url("/s"): resolved("/s", "Trail Shoes", KeywordRung.STRATEGIC),
        url("/g"): resolved("/g", "rain jackets", KeywordRung.GSC, language="de"),
        url("/h"): resolved("/h", "Four Season Tents", KeywordRung.H1),
    }
    secondaries = {
        url("/s"): ("en", ("trail gaiters", "trail maps")),
        url("/g"): ("de", ("regenjacke",)),
    }

    targets = ranked_targets(strategic, keywords, secondaries)

    assert flat(targets) == {
        (url("/s"), 1, "Trail Shoes", "en", STRATEGIC, KeywordRung.STRATEGIC),
        (url("/s"), 2, "rain jackets", "en", STRATEGIC, None),
        (url("/s"), 3, "tents", "en", STRATEGIC, None),
        (url("/s"), 4, "trail gaiters", "en", OBSERVED, None),
        (url("/s"), 5, "trail maps", "en", OBSERVED, None),
        (url("/g"), 1, "rain jackets", "de", OBSERVED, KeywordRung.GSC),
        (url("/g"), 2, "regenjacke", "de", OBSERVED, None),
        (url("/h"), 1, "Four Season Tents", "en", INFERRED, KeywordRung.H1),
        # An unresolved page keeps its strategic edge, outside any ranking.
        (url("/missing"), None, "stoves", "en", STRATEGIC, None),
    }
    assert set(targets) == {STRATEGIC, OBSERVED, INFERRED}
