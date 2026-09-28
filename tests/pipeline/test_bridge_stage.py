"""find_bridges end to end on real Neo4j and Mongo: a planted hub graph with one link, the
eligibility exclusions, the anchors, both files and the tenant boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas
import pyarrow.parquet as pq
import pytest
from structlog.testing import capture_logs

from linking_engine.errors import DatabaseReadError
from linking_engine.models import KeywordRung, KeywordSource, KeywordTarget, Link, Page
from linking_engine.pipeline.bridges import find_bridges, read_hub_pairs

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

CENTROIDS = {0: [1.0, 0.0, 0.0, 0.0], 1: [0.9, 0.436, 0.0, 0.0], 2: [0.0, 0.0, 1.0, 0.0]}
HUB_PAGES = {0: ("a0", "a1", "a2", "a3"), 1: ("b0", "b1", "b2", "b3"), 2: ("c0", "c1", "c2", "c3")}
# a3 and b3 are non-canonical copies, b2 is not indexable, b-de is German, n0 is noise.
NON_CANONICAL = ("a3", "b3")
NOT_INDEXABLE = ("b2",)
OUTSIDE = ("n0", "loose", "b-de")


def url(name: str) -> str:
    return f"example.com/hub/{name}"


def vector(hub: int, offset: int) -> list[float]:
    base = list(CENTROIDS[hub])
    base[3] = 0.05 * (offset + 1)
    return base


async def seed(graph: GraphRepo, tenant: str, *, links: list[tuple[str, str]]) -> None:
    names = [n for pages in HUB_PAGES.values() for n in pages] + ["n0", "loose", "b-de"]
    await graph.upsert_pages(
        tenant,
        [
            Page(
                url=url(n),
                status_code=200,
                is_indexable=n not in NOT_INDEXABLE,
                language="de" if n == "b-de" else "en",
                word_count=400,
            )
            for n in names
        ],
    )
    sources = sorted({s for s, _ in links})
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
            for i, (s, t) in enumerate(links)
        ],
    )
    rows: list[dict[str, object]] = [
        {"url": url(n), "hub": hub, "vec": vector(hub, i), "pr": 0.1 * (i + 1)}
        for hub, pages in HUB_PAGES.items()
        for i, n in enumerate(pages)
    ]
    rows += [
        {"url": url("n0"), "hub": -1, "vec": [0.5, 0.5, 0.5, 0.0], "pr": 0.9},
        {"url": url("loose"), "hub": None, "vec": [0.4, 0.4, 0.4, 0.4], "pr": 0.9},
        {"url": url("b-de"), "hub": 1, "vec": vector(1, 9), "pr": 0.9},
    ]
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.hubId = row.hub, p.content_embedding = row.vec, p.pageRankPercentile = row.pr",
        t=tenant,
        rows=rows,
    )
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) WHERE p.url IN $urls "
        "SET p.isCanonical = false, p.duplicateGroup = 1",
        t=tenant,
        urls=[url(n) for n in NON_CANONICAL],
    )
    await graph._auto(
        "UNWIND $hubs AS hub CREATE (:Hub {tenantId: $t, hubId: hub.id, active: true, "
        "centroid: hub.centroid})",
        t=tenant,
        hubs=[{"id": hub, "centroid": centroid} for hub, centroid in CENTROIDS.items()],
    )
    await graph.replace_keyword_targets(
        tenant,
        KeywordSource.INFERRED,
        [
            KeywordTarget(
                url=url("a1"),
                text="Trail Shoes",
                language="en",
                source=KeywordSource.INFERRED,
                rung=KeywordRung.H1,
            )
        ],
    )


def files(cache_dir: Path, tenant: str) -> tuple[pandas.DataFrame, pandas.DataFrame, list[str]]:
    folder = cache_dir / tenant
    names = sorted(p.name for p in folder.iterdir())
    return (
        pq.read_table(folder / "bridges.parquet").to_pandas(),
        pq.read_table(folder / "hub_pairs.parquet").to_pandas(),
        names,
    )


HUB_OF = {url(n): hub for hub, pages in HUB_PAGES.items() for n in pages}


@pytest.mark.integration
async def test_bridges_connect_the_planted_hubs_through_eligible_pages_only(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-other"
    everything = [(s, t) for s in ("a0", "b0", "c0") for t in ("a1", "b1", "c1") if s[0] != t[0]]
    await seed(graph, other, links=everything)
    await seed(graph, tenant, links=[("a0", "b0")])

    with capture_logs() as logs:
        report, path = await find_bridges(graph, mongo, tenant, cache_dir=tmp_path)

    bridges, pairs, names = files(tmp_path, tenant)
    assert path.parent == tmp_path / tenant
    assert {"bridges.parquet", "hub_pairs.parquet"} <= set(names)
    assert not [n for n in names if n.endswith(".tmp")]
    assert report.gsc_used is False, "no GSC data: the Jaccard term drops out"

    en = pairs[pairs["language"] == "en"]
    assert sorted(zip(en["hub_a"], en["hub_b"], strict=True)) == [(0, 1), (0, 2), (1, 2)]
    first = en[(en["hub_a"] == 0) & (en["hub_b"] == 1)].iloc[0]
    assert (first["size_a"], first["size_b"], first["pages_ab"], first["pages_ba"]) == (4, 4, 1, 0)
    assert en["query_jaccard"].isna().all()
    assert all(len(queries) == 0 for queries in en["shared_queries"]), "no GSC: none shared"
    assert "SPANNING_TREE" in list(first["reasons"]), "reasons are a list column"

    assert len(bridges) > 0
    excluded_sources = {url(n) for n in (*NON_CANONICAL, *OUTSIDE)}
    excluded_targets = excluded_sources | {url(n) for n in NOT_INDEXABLE}
    assert not set(bridges["source_url"]) & excluded_sources
    assert not set(bridges["target_url"]) & excluded_targets
    for row in bridges.itertuples():
        assert (HUB_OF[row.source_url], HUB_OF[row.target_url]) == (row.hub_from, row.hub_to)
        assert row.language == "en"
    # Hub 0 already reaches hub 1 through a0, so nothing is proposed from 0 to 1.
    assert bridges[(bridges["hub_from"] == 0) & (bridges["hub_to"] == 1)].empty
    assert not ((bridges["source_url"] == url("a0")) & (bridges["target_url"] == url("b0"))).any()
    assert set(bridges.loc[bridges["rank"] == 1, "hub_to"]) == {0, 1, 2}
    to_a1 = bridges[bridges["target_url"] == url("a1")]
    assert (to_a1["anchor_keyword"] == "Trail Shoes").all()
    assert (to_a1["anchor_rung"] == "H1").all()
    others = bridges[bridges["target_url"] != url("a1")]
    assert others["anchor_keyword"].isna().all()

    # en was three components, de one; the bridges join en into one.
    assert (report.components_before, report.directions_short) == (4, 0)
    assert report.components_after == 2
    assert report.bridge_links == int((bridges["rank"] == 1).sum())
    assert report.alternatives == int((bridges["rank"] > 1).sum())

    [line] = [entry for entry in logs if entry["event"] == "graph.bridges"]
    logged = " ".join(str(value) for value in line.values())
    assert [u for u in (*HUB_OF, *map(url, OUTSIDE)) if u in logged] == []


@pytest.mark.integration
async def test_the_same_graph_gives_the_same_bridges(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, tenant, links=[("a0", "b0")])

    first, _ = await find_bridges(graph, mongo, tenant, cache_dir=tmp_path / "one")
    again, _ = await find_bridges(graph, mongo, tenant, cache_dir=tmp_path / "two")

    left, left_pairs, _ = files(tmp_path / "one", tenant)
    right, right_pairs, _ = files(tmp_path / "two", tenant)
    pandas.testing.assert_frame_equal(left, right, check_exact=True)
    pandas.testing.assert_frame_equal(left_pairs, right_pairs, check_exact=True)
    assert first.model_dump(exclude={"seconds", "finished_at"}) == again.model_dump(
        exclude={"seconds", "finished_at"}
    )


@pytest.mark.integration
async def test_with_gsc_data_hub_pairs_carry_the_query_overlap_and_shared_queries(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, tenant, links=[("a0", "b0")])
    await mongo._db["gsc_queries"].insert_many(
        [
            {"tenantId": tenant, "url": url(page), "query": query}
            for page, query in (
                ("a1", "Trail Shoes"),
                ("a2", "trail shoes"),
                ("a1", "tents"),
                ("b1", "trail  shoes"),
                ("c1", "stoves"),
            )
        ]
    )

    report, _ = await find_bridges(graph, mongo, tenant, cache_dir=tmp_path)

    _, pairs, _ = files(tmp_path, tenant)
    en = pairs[pairs["language"] == "en"].set_index(["hub_a", "hub_b"])
    assert report.gsc_used is True
    # Hub 0 has {trail shoes, tents}, hub 1 {trail shoes}, hub 2 {stoves}.
    assert en.loc[(0, 1), "query_jaccard"] == pytest.approx(0.5)
    assert list(en.loc[(0, 1), "shared_queries"]) == ["trail shoes"]
    assert en.loc[(0, 2), "query_jaccard"] == 0.0
    assert list(en.loc[(0, 2), "shared_queries"]) == []


@pytest.mark.integration
async def test_hub_pairs_round_trip_with_a_separator_inside_a_query(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, tenant, links=[("a0", "b0")])
    await mongo._db["gsc_queries"].insert_many(
        [
            {"tenantId": tenant, "url": url(page), "query": "trail shoes | acme"}
            for page in ("a1", "b1")
        ]
    )

    _, path = await find_bridges(graph, mongo, tenant, cache_dir=tmp_path)

    pairs = {
        (p.language, p.hub_a, p.hub_b): p
        for p in read_hub_pairs(path.with_name("hub_pairs.parquet"))
    }
    assert pairs["en", 0, 1].shared_queries == ("trail shoes | acme",)
    assert pairs["en", 0, 2].shared_queries == ()
    assert pairs["en", 0, 1].reasons
    written = pq.read_table(path.with_name("hub_pairs.parquet")).to_pandas()
    assert list(
        written.loc[(written["hub_a"] == 0) & (written["hub_b"] == 1), "shared_queries"].iloc[0]
    ) == ["trail shoes | acme"]


@pytest.mark.parametrize("tenant_id", [" ", "../escape", "acme/other", ".."])
async def test_a_tenant_id_that_is_not_a_plain_directory_name_is_refused(
    tmp_path: Path, tenant_id: str
) -> None:
    unused = object()
    with pytest.raises(ValueError, match="tenant_id"):
        await find_bridges(
            cast("GraphRepo", unused), cast("MongoRepo", unused), tenant_id, cache_dir=tmp_path
        )


@pytest.mark.integration
async def test_a_hub_without_a_stored_centroid_fails_the_read(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed(graph, tenant, links=[("a0", "b0")])
    await graph._auto("MATCH (h:Hub {tenantId: $t, hubId: 2}) DETACH DELETE h", t=tenant)

    with pytest.raises(DatabaseReadError, match="no stored centroid"):
        await find_bridges(graph, mongo, tenant, cache_dir=tmp_path)
