"""The anchor-extraction stage with the identifier guard: refused places are counted once per
source page, however many targets' keywords they were refused for."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from test_keyword_stage import page_record, url

from linking_engine.models import KeywordRung, KeywordSource, KeywordTarget, Page
from linking_engine.pipeline.anchors import extract_anchors

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

BODIES = {
    # "upgrade to widget" twice, never with version 10 or 12.
    "/g1": "Upgrade to widget today. Then upgrade to widget 11 soon.",
    # Another CVE number beside the widget.
    "/g2": "Read the CVE-2026-10002 widget notice before patching.",
}
KEYWORDS = {
    "/t1": "upgrade to widget 10",
    "/t2": "upgrade to widget 12",
    "/t3": "CVE-2026-10001 \u2013 Widget",
}


async def seed(graph: GraphRepo, mongo: MongoRepo, tenant: str) -> None:
    paths = [*BODIES, *KEYWORDS]
    await graph.upsert_pages(
        tenant,
        [Page(url=url(p), status_code=200, is_indexable=True, language="en") for p in paths],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec",
        t=tenant,
        rows=[{"url": url(p), "vec": [1.0, 0.1 * i, 0.0]} for i, p in enumerate(paths)],
    )
    inferred = KeywordSource.INFERRED
    await graph.replace_keyword_targets(
        tenant,
        inferred,
        [
            KeywordTarget(
                url=url(p), text=text, language="en", source=inferred, rung=KeywordRung.H1
            )
            for p, text in KEYWORDS.items()
        ],
    )
    await mongo.write_pages(
        tenant, [page_record(p, 200, None, None, body, "en") for p, body in BODIES.items()], []
    )


@pytest.mark.integration
async def test_places_refused_for_other_identifiers_are_counted_once_per_source(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, mongo, tenant)

    report, path = await extract_anchors(graph, mongo, tenant, cache_dir=tmp_path)

    # g1: both "upgrade to widget" places, refused for t1 and t2 alike, count once each.
    # g2: "CVE-2026-10002 widget" shares 3 of 5 stems with t3's keyword but names another CVE.
    assert report.identifier_mismatches == 3
    rows = pq.read_table(path).to_pylist()
    assert [
        (row["source_url"], row["target_url"])
        for row in rows
        if row["target_url"] in {url(p) for p in KEYWORDS}
    ] == [], "a place with other identifiers was matched"
