"""assemble_features end to end: candidate pairs from Neo4j, signals from both stores, one
Parquet matrix per tenant keyed by its inputs, streamed in chunks."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, cast

import pandas
import pyarrow.parquet as pq
import pytest
from structlog.testing import capture_logs

from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.discovery.features import FEATURE_COLUMNS, KEY_COLUMNS
from linking_engine.models import Link, Page
from linking_engine.pipeline.features import assemble_features

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

NAMES = ("a", "b", "c", "d", "e", "f")
VECTORS = {
    "a": [1.0, 0.0, 0.0, 0.0],
    "b": [0.9, 0.1, 0.0, 0.0],
    "c": [0.8, 0.3, 0.0, 0.0],
    "d": [0.0, 0.0, 1.0, 0.0],
    "e": [0.0, 0.1, 0.9, 0.0],
    "f": [0.0, 0.0, 0.7, 0.3],
}


def url(name: str) -> str:
    return f"example.com/f/{name}"


def entries(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


async def seed(graph: GraphRepo, mongo: MongoRepo, tenant: str, *, enrichment: bool = True) -> None:
    await graph.upsert_pages(
        tenant,
        [
            Page(
                url=url(n),
                status_code=200,
                is_indexable=True,
                word_count=500,
                language="en",
                crawl_depth=i,
            )
            for i, n in enumerate(NAMES)
        ],
    )
    await graph.replace_links(
        tenant,
        [url(n) for n in NAMES],
        [
            Link(
                source_url=url(s),
                target_url=url(t),
                position=0,
                anchor_text="x",
                surrounding_text="",
            )
            for s, t in (("a", "b"), ("b", "c"), ("c", "a"))
        ],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec, p.hubId = row.hub, p.isHubPillar = row.pillar, "
        "p.pageRankPercentile = row.pr",
        t=tenant,
        rows=[
            {
                "url": url(n),
                "vec": VECTORS[n],
                "hub": 0 if n in "abc" else 1,
                "pillar": n in "ad",
                "pr": i / 10,
            }
            for i, n in enumerate(NAMES)
        ],
    )
    if not enrichment:
        return
    await mongo._db["gsc_metrics"].insert_one(
        {
            "tenantId": tenant,
            "url": url("b"),
            "impressions_28d": 999,
            "clicks_28d": 150,
            "avg_position": 3.0,
            "query_count": 2,
        }
    )
    await mongo._db["gsc_queries"].insert_many(
        [
            {
                "tenantId": tenant,
                "url": url("b"),
                "query": q,
                "impressions": 500,
                "clicks": 75,
                "position": 3.0,
            }
            for q in ("tents", "stoves")
        ]
    )
    await mongo._db["strategic_keywords"].insert_one(
        {"tenantId": tenant, "url": url("c"), "keyword": "tents", "language": "en", "priority": 4}
    )


@pytest.mark.parametrize(
    ("tenant_id", "options", "message"),
    [
        pytest.param(" ", {}, "tenant_id", id="blank"),
        pytest.param("../escape", {}, "directory name", id="parent-path"),
        pytest.param("acme/other", {}, "directory name", id="nested-path"),
        pytest.param("..", {}, "directory name", id="dot-dot"),
        pytest.param("acme", {"chunk_pairs": 0}, "chunk_pairs", id="zero-chunk"),
    ],
)
async def test_invalid_arguments_are_refused_before_any_read(
    tmp_path: Path, tenant_id: str, options: dict[str, int], message: str
) -> None:
    unused = object()
    with pytest.raises(ValueError, match=message):
        await assemble_features(
            cast("GraphRepo", unused),
            cast("MongoRepo", unused),
            tenant_id,
            cache_dir=tmp_path,
            **options,
        )
    assert entries(tmp_path) == []


@pytest.mark.integration
async def test_the_matrix_has_one_row_per_candidate_pair_in_the_persisted_column_order(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, mongo, tenant)
    expected = {
        (source, target.target_url)
        for target in (await retrieve_candidates(graph, tenant)).targets
        for source in target.sources
    }

    with capture_logs() as logs:
        report, path = await assemble_features(
            graph, mongo, tenant, cache_dir=tmp_path, chunk_pairs=4
        )

    assert path == tmp_path / tenant / f"{report.cache_key}.parquet"
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata
    assert json.loads(metadata[b"feature_columns"]) == list(FEATURE_COLUMNS)
    assert metadata[b"tenant_id"].decode() == tenant
    frame = parquet.read().to_pandas()
    assert tuple(frame.columns) == (*KEY_COLUMNS, *FEATURE_COLUMNS)
    assert set(zip(frame["source_url"], frame["target_url"], strict=True)) == expected
    assert len(frame) == report.pairs == len(expected)
    assert report.chunks == parquet.num_row_groups == math.ceil(len(expected) / 4)
    assert report.cache_hit is False
    gsc = frame.groupby("target_url")["has_gsc_data"].max().to_dict()
    assert gsc == {url(n): (1.0 if n == "b" else 0.0) for n in NAMES}
    assert report.has_gsc_data_share == pytest.approx((frame["target_url"] == url("b")).mean())
    assert {"context_relevance", "anchor_target_fit"} <= set(report.all_null_columns)
    [line] = [entry for entry in logs if entry["event"] == "features.assembled"]
    logged = " ".join(str(value) for value in line.values())
    assert [u for u in map(url, NAMES) if u in logged] == [], "urls in the log line"


@pytest.mark.integration
async def test_an_unchanged_input_reuses_the_matrix_and_a_changed_one_builds_a_new_one(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, mongo, tenant)
    first, path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path, chunk_pairs=4)
    written = path.stat().st_mtime_ns
    built = pq.read_table(path).to_pandas()

    again, same = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path, chunk_pairs=4)

    assert (again.cache_hit, again.cache_key, same) == (True, first.cache_key, path)
    assert path.stat().st_mtime_ns == written, "a cache hit must not rewrite the file"
    pandas.testing.assert_frame_equal(pq.read_table(same).to_pandas(), built)
    assert (again.pairs, again.null_share, again.constant_columns) == (
        first.pairs,
        first.null_share,
        first.constant_columns,
    )

    await mongo._db["gsc_metrics"].insert_one(
        {
            "tenantId": tenant,
            "url": url("e"),
            "impressions_28d": 40,
            "clicks_28d": 1,
            "avg_position": 30.0,
            "query_count": 1,
        }
    )
    changed, new_path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)

    assert (changed.cache_hit, changed.cache_key != first.cache_key) == (False, True)
    assert new_path != path
    assert new_path.is_file()


@pytest.mark.integration
async def test_an_unreadable_cache_file_is_rebuilt(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, mongo, tenant)
    first, path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)
    path.write_bytes(b"not parquet")

    rebuilt, same = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)

    assert (rebuilt.cache_hit, rebuilt.cache_key, same) == (False, first.cache_key, path)
    assert pq.ParquetFile(path).metadata.num_rows == first.pairs
    assert entries(path.parent) == [path.name], "no temporary file left"


@pytest.mark.integration
async def test_each_tenant_has_its_own_matrix(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-other"
    await seed(graph, mongo, tenant)
    await seed(graph, mongo, other)

    mine, my_path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)
    theirs, their_path = await assemble_features(graph, mongo, other, cache_dir=tmp_path)

    assert mine.cache_key != theirs.cache_key, "identical data of two tenants never shares a key"
    assert (my_path.parent.name, their_path.parent.name) == (tenant, other)
    assert (mine.cache_hit, theirs.cache_hit) == (False, False)
    assert mine.pairs == theirs.pairs


@pytest.mark.integration
async def test_a_tenant_without_gsc_or_keywords_gets_a_full_matrix_with_its_gaps_reported(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, mongo, tenant, enrichment=False)

    report, path = await assemble_features(graph, mongo, tenant, cache_dir=tmp_path)

    assert report.pairs > 0
    assert pq.ParquetFile(path).metadata.num_rows == report.pairs
    assert {
        "source_impressions_log",
        "target_impressions_log",
        "target_ctr_gap",
        "target_query_count",
        "target_max_priority",
        "target_keyword_gap",
        "context_relevance",
        "anchor_target_fit",
    } <= set(report.all_null_columns)
    assert {"has_gsc_data", "target_position_band_null"} <= set(report.constant_columns)
    assert report.has_gsc_data_share == 0.0
    assert "content_cosine" not in report.all_null_columns
