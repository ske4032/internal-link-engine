"""extract_anchors end to end on real Neo4j and Mongo: candidate pairs plus hub bridges, each
target's ranked keywords found in the source copy, existing anchors skipped, one Parquet file."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from structlog.testing import capture_logs
from test_keyword_stage import page_record, url

from linking_engine.anchor.extraction import DEFAULT_THRESHOLD
from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.models import (
    AnchorMatch,
    ExtractionSettings,
    KeywordRung,
    KeywordSource,
    KeywordTarget,
    Link,
    Page,
)
from linking_engine.pipeline.anchors import _SCHEMA, extract_anchors

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import PageRecord

VECTORS = {
    "/g": [1.0, 0.0, 0.0],
    "/s": [0.9, 0.1, 0.0],
    "/t": [0.8, 0.2, 0.0],
    "/j": [0.0, 1.0, 0.0],
    "/b": [0.0, 0.0, 1.0],
}
GUIDE = (
    "Read about trail shoes here. Our trail shoes guide covers running shoes. "
    "Pack four season tents too."
)
# The guide already links to /j with its first "trail shoes".
EXISTING = Link(
    source_url=url("/g"),
    target_url=url("/j"),
    position=0,
    anchor_text="trail shoes",
    surrounding_text="Read about trail shoes here.",
)
# Its surrounding text is not in the guide, so its anchor cannot be located.
UNLOCATED = Link(
    source_url=url("/g"),
    target_url=url("/b"),
    position=1,
    anchor_text="stoves",
    surrounding_text="Pack light stoves.",
)
BODIES = {
    "/g": GUIDE,
    "/s": "Trail shoes grip wet rock.",
    "/t": "Tents for winter camping.",
    # /j has an empty body; /b has no page record at all.
    "/j": "",
}
ANCHORS = [
    "source_url",
    "target_url",
    "keyword",
    "keyword_rank",
    "keyword_source",
    "rung",
    "phrase",
    "start",
    "end",
    "sentence",
    "sentence_index",
    "sentence_start",
    "stem_jaccard",
    "semantic_similarity",
]


async def seed(graph: GraphRepo, mongo: MongoRepo, tenant: str) -> None:
    await graph.upsert_pages(
        tenant,
        [Page(url=url(p), status_code=200, is_indexable=True, language="en") for p in VECTORS],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec",
        t=tenant,
        rows=[{"url": url(p), "vec": v} for p, v in VECTORS.items()],
    )
    await graph.replace_links(tenant, [url("/g")], [EXISTING, UNLOCATED])
    strategic = KeywordSource.CLIENT_STRATEGIC
    await graph.replace_keyword_targets(
        tenant,
        strategic,
        [
            KeywordTarget(
                url=url("/s"),
                text="trail shoes",
                language="en",
                source=strategic,
                rung=KeywordRung.STRATEGIC,
            ),
            KeywordTarget(
                url=url("/s"), text="running shoe", language="en", source=strategic, rank=2
            ),
        ],
    )
    inferred = KeywordSource.INFERRED
    await graph.replace_keyword_targets(
        tenant,
        inferred,
        [
            KeywordTarget(
                url=url("/t"),
                text="Four Season Tents",
                language="en",
                source=inferred,
                rung=KeywordRung.H1,
            )
        ],
    )
    await mongo.write_pages(
        tenant, [page_record(p, 200, None, None, body, "en") for p, body in BODIES.items()], []
    )


def write_bridges(cache_dir: Path, tenant: str, pairs: list[tuple[str, str]]) -> None:
    folder = cache_dir / tenant
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "source_url": [url(s) for s, _ in pairs],
                "target_url": [url(t) for _, t in pairs],
                "rank": list(range(1, len(pairs) + 1)),
                "similarity": [0.5] * len(pairs),
            }
        ),
        folder / "bridges.parquet",
    )


def entries(cache_dir: Path, tenant: str) -> list[str]:
    return sorted(p.name for p in (cache_dir / tenant).iterdir())


@pytest.mark.integration
async def test_anchors_come_from_candidates_and_bridges_and_skip_existing_anchors(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, mongo, tenant)
    candidates = {
        (source, entry.target_url)
        for entry in (await retrieve_candidates(graph, tenant)).targets
        for source in entry.sources
    }
    assert (url("/g"), url("/s")) in candidates
    assert (url("/g"), url("/j")) not in candidates, "an existing link is never a candidate"
    # One bridge repeats a candidate pair, twice; g -> j is new.
    write_bridges(tmp_path, tenant, [("/g", "/s"), ("/g", "/s"), ("/g", "/j")])

    with capture_logs() as logs:
        report, path = await extract_anchors(graph, mongo, tenant, cache_dir=tmp_path)

    assert (report.pairs, report.bridge_pairs) == (len(candidates) + 1, 2)
    assert report.stem_set_threshold == DEFAULT_THRESHOLD
    assert path == tmp_path / tenant / "anchors.parquet"
    assert not [n for n in entries(tmp_path, tenant) if n.endswith(".tmp")]
    frame = pq.read_table(path).to_pandas()
    assert list(frame.columns) == ANCHORS
    rows = {
        (row.target_url, row.keyword): row
        for row in frame.itertuples()
        if row.source_url == url("/g")
    }
    shoes = rows[(url("/s"), "trail shoes")]
    # The first "trail shoes" is an existing link's anchor: the next sentence is used.
    assert (shoes.rung, shoes.sentence_index, shoes.keyword_rank) == ("EXACT", 1, 1)
    assert GUIDE[shoes.start : shoes.end] == shoes.phrase == "trail shoes"
    running = rows[(url("/s"), "running shoe")]
    assert (running.rung, running.phrase, running.keyword_rank) == ("STEMMED", "running shoes", 2)
    tents = rows[(url("/t"), "Four Season Tents")]
    assert (tents.rung, tents.phrase, tents.keyword_source) == (
        "EXACT",
        "four season tents",
        "INFERRED",
    )
    assert report.overlapping_existing_anchors == 1
    assert (report.existing_anchors_located, report.existing_anchors_unlocated) == (1, 1)
    # /j's body is empty; /b has no page record, so it has no language either.
    assert (report.source_pages, report.sources_without_body) == (5, 2)
    assert report.stemmed_languages == {"en": 4}
    assert report.unstemmed_languages == {}

    [line] = [entry for entry in logs if entry["event"] == "anchors.extracted"]
    logged = " ".join(str(value) for value in line.values())
    leaked = [text for text in (*map(url, VECTORS), "trail shoes", GUIDE) if text in logged]
    assert leaked == []


@pytest.mark.integration
async def test_without_a_bridges_file_only_the_candidates_source_pages_are_read(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed(graph, mongo, tenant)
    await mongo.set_extraction_settings(tenant, ExtractionSettings(stem_set_threshold=1.0))
    # A stored page that no pair starts from.
    await mongo.write_pages(tenant, [page_record("/x", 200, None, None, GUIDE, "en")], [])
    sources = {
        source
        for entry in (await retrieve_candidates(graph, tenant)).targets
        for source in entry.sources
    }
    requested: list[set[str] | None] = []
    read = mongo.iter_page_records

    def recording(tenant_id: str, **options: Any) -> AsyncIterator[list[PageRecord]]:
        urls = options.get("urls")
        requested.append(None if urls is None else set(urls))
        return read(tenant_id, **options)

    monkeypatch.setattr(mongo, "iter_page_records", recording)

    report, path = await extract_anchors(graph, mongo, tenant, cache_dir=tmp_path)

    assert requested == [sources]
    assert url("/x") not in sources
    assert report.bridge_pairs == 0
    assert report.stem_set_threshold == 1.0, "the tenant's own threshold applies"
    assert path.is_file()


async def seed_branded(graph: GraphRepo, mongo: MongoRepo, tenant: str, *, suffix: str) -> None:
    """A guide whose copy names the target's keyword without the brand word that opens it."""
    pages = {"/guide": "Our hydraulic presses ship fast.", "/presses": "Presses in stock."}
    await graph.upsert_pages(
        tenant,
        [Page(url=url(p), status_code=200, is_indexable=True, language="en") for p in pages],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec",
        t=tenant,
        rows=[
            {"url": url("/guide"), "vec": [1.0, 0.1]},
            {"url": url("/presses"), "vec": [1.0, 0.0]},
        ],
    )
    strategic = KeywordSource.CLIENT_STRATEGIC
    await graph.replace_keyword_targets(
        tenant,
        strategic,
        [
            KeywordTarget(
                url=url("/presses"),
                text="Acme7 hydraulic press",
                language="en",
                source=strategic,
                rung=KeywordRung.STRATEGIC,
            )
        ],
    )
    await mongo.write_pages(
        tenant,
        [
            page_record(p, 200, f"{p[1:].title()}{suffix}", None, body, "en")
            for p, body in pages.items()
        ],
        [],
    )


@pytest.mark.integration
async def test_a_brand_with_a_digit_is_no_identifier_the_ladder_must_keep(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    branded, plain = tenant, f"{tenant}-plain"
    await seed_branded(graph, mongo, branded, suffix=" | Acme7 Tools")
    await seed_branded(graph, mongo, plain, suffix="")

    kept, path = await extract_anchors(graph, mongo, branded, cache_dir=tmp_path)
    refused, _ = await extract_anchors(graph, mongo, plain, cache_dir=tmp_path)

    # "acme7" is the brand's, so dropping it loses no identifier.
    [row] = pq.read_table(path).to_pylist()
    assert (row["rung"], row["phrase"]) == ("STEMMED", "hydraulic presses")
    assert kept.identifier_mismatches == 0
    # Without the brand, "acme7" reads as a model number the phrase lacks.
    assert (refused.matches, refused.identifier_mismatches) == (0, 1)


def test_the_parquet_schema_is_every_anchor_match_field_in_order() -> None:
    assert _SCHEMA.names == list(AnchorMatch.model_fields) == ANCHORS


@pytest.mark.parametrize("tenant_id", [" ", "../escape", "acme/other", ".."])
async def test_a_tenant_id_that_is_not_a_plain_directory_name_is_refused(
    tmp_path: Path, tenant_id: str
) -> None:
    unused = object()
    with pytest.raises(ValueError, match="tenant_id"):
        await extract_anchors(
            cast("GraphRepo", unused), cast("MongoRepo", unused), tenant_id, cache_dir=tmp_path
        )
