from __future__ import annotations

import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest
from scipy.stats import spearmanr
from structlog.testing import capture_logs

from linking_engine.graph.algorithms import build_link_graphs, percentile_rank
from linking_engine.models import Link, LinkGraphSnapshot, Page
from linking_engine.pipeline.analytics import compute_centrality, load_link_graphs, page_centrality

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo


@pytest.mark.integration
async def test_load_link_graphs_builds_both_graphs_and_logs_the_cost(
    graph: GraphRepo, tenant: str
) -> None:
    urls = [f"example.com/{name}" for name in ("a", "b", "c", "orphan")]
    await graph.upsert_pages(tenant, [Page(url=u, status_code=200) for u in urls])
    links = [
        Link(
            source_url=urls[0], target_url=urls[1], position=0, anchor_text="b", surrounding_text=""
        ),
        Link(
            source_url=urls[1], target_url=urls[0], position=0, anchor_text="a", surrounding_text=""
        ),
        Link(
            source_url=urls[1], target_url=urls[2], position=1, anchor_text="c", surrounding_text=""
        ),
    ]
    await graph.replace_links(tenant, urls[:2], links)

    with capture_logs() as logs:
        graphs = await load_link_graphs(graph, tenant)

    assert graphs.urls == tuple(sorted(urls))
    assert (graphs.directed.ecount(), graphs.undirected.ecount()) == (3, 2)
    (event,) = [e for e in logs if e["event"] == "graph.build"]
    assert (event["pages"], event["link_rows"], event["isolated"]) == (4, 3, 1)
    assert event["pull_s"] >= 0
    assert event["build_s"] >= 0


def test_page_centrality_scores_crawled_pages_only_and_ranks_among_them() -> None:
    graphs = build_link_graphs(
        LinkGraphSnapshot(
            tenant_id="t",
            pages=("a", "b", "c", "x"),
            placeholders=(False, False, False, True),
            links=(),
        )
    )
    rows = page_centrality(graphs, np.array([0.1, 0.3, 0.2, 0.4]), np.array([0.0, 2.0, 1.0]))
    assert [row.url for row in rows] == ["a", "b", "c"]
    assert [row.page_rank for row in rows] == [0.1, 0.3, 0.2]
    assert [row.page_rank_percentile for row in rows] == [0, 2 / 3, 1 / 3]
    assert [row.betweenness_percentile for row in rows] == [0, 2 / 3, 1 / 3]


# ── independent networkx reference, 600-page synthetic corpus ───────────────

CRAWLED, PLACEHOLDERS, ISOLATED, DANGLING = 540, 60, 10, 40


def synthetic_corpus(seed: int = 8) -> tuple[list[str], list[str], list[Link]]:
    """Random body links; some crawled pages link nowhere, a few are isolated, some repeat links."""
    rng = random.Random(seed)
    crawled = [f"example.com/p{i:03d}" for i in range(CRAWLED)]
    placeholders = [f"example.com/ghost{i:02d}" for i in range(PLACEHOLDERS)]
    isolated = set(crawled[:ISOLATED])
    targets = [u for u in crawled + placeholders if u not in isolated]
    links: list[Link] = []
    for source in crawled[ISOLATED + DANGLING :]:
        chosen = [t for t in rng.sample(targets, rng.randint(1, 8)) if t != source]
        if rng.random() < 0.2:
            chosen.append(chosen[0])
        links.extend(
            Link(
                source_url=source,
                target_url=target,
                position=position,
                anchor_text="a",
                surrounding_text="",
            )
            for position, target in enumerate(chosen)
        )
    return crawled, placeholders, links


async def seed_synthetic(graph: GraphRepo, tenant: str) -> None:
    crawled, placeholders, links = synthetic_corpus()
    await graph.upsert_pages(tenant, [Page(url=u, status_code=200) for u in crawled])
    await graph.upsert_placeholders(tenant, placeholders)
    await graph.replace_links(tenant, crawled, links)


async def written_scores(graph: GraphRepo, tenant: str) -> dict[str, dict[str, object]]:
    rows = await graph._read(
        "MATCH (p:Page {tenantId: $tenant}) "
        "RETURN p.url AS url, coalesce(p.isPlaceholder, false) AS placeholder, "
        "p.pageRank AS pr, p.pageRankPercentile AS pr_pct, "
        "p.betweenness AS bc, p.betweennessPercentile AS bc_pct",
        tenant=tenant,
    )
    return {str(row["url"]): row for row in rows}


async def reference_graphs(graph: GraphRepo, tenant: str) -> tuple[nx.DiGraph, nx.Graph]:
    """Built from its own URL-keyed read, never from the pipeline's id mapping."""
    pages = await graph._read(
        "MATCH (p:Page {tenantId: $tenant}) "
        "RETURN p.url AS url, coalesce(p.isPlaceholder, false) AS placeholder",
        tenant=tenant,
    )
    links = await graph._read(
        "MATCH (a:Page {tenantId: $tenant})-[:LINKS_TO]->(b:Page {tenantId: $tenant}) "
        "RETURN a.url AS source, b.url AS target",
        tenant=tenant,
    )
    directed = nx.DiGraph()
    directed.add_nodes_from(row["url"] for row in pages)
    directed.add_edges_from((row["source"], row["target"]) for row in links)
    crawled = [row["url"] for row in pages if not row["placeholder"]]
    undirected = directed.to_undirected().subgraph(crawled).copy()
    return directed, undirected


@pytest.mark.integration
async def test_scores_match_an_independent_networkx_reference_by_url(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_synthetic(graph, tenant)

    report = await compute_centrality(graph, tenant)

    assert (report.pages, report.placeholders) == (CRAWLED, PLACEHOLDERS)
    written = await written_scores(graph, tenant)
    assert all(row["pr"] is None for row in written.values() if row["placeholder"])

    directed, undirected = await reference_graphs(graph, tenant)
    nx_pr = nx.pagerank(directed, alpha=0.85, tol=1e-12, max_iter=1000)
    nx_bc = nx.betweenness_centrality(undirected, normalized=False)
    urls = sorted(nx_bc)
    assert len(urls) == CRAWLED

    reference_pr = np.array([nx_pr[u] for u in urls])
    ours_pr = np.array([written[u]["pr"] for u in urls], dtype=np.float64)
    assert spearmanr(reference_pr, ours_pr).statistic >= 0.999
    assert np.abs(reference_pr - ours_pr).max() <= 1e-6

    reference_bc = np.array([nx_bc[u] for u in urls])
    ours_bc = np.array([written[u]["bc"] for u in urls], dtype=np.float64)
    assert reference_bc.max() > 0
    assert np.abs(reference_bc - ours_bc).max() <= 1e-9 * reference_bc.max()

    assert [written[u]["bc_pct"] for u in urls] == percentile_rank(reference_bc).tolist()
    assert [written[u]["pr_pct"] for u in urls] == percentile_rank(reference_pr).tolist()


@pytest.mark.integration
async def test_percentiles_are_stable_across_identical_runs_and_timings_are_logged(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_synthetic(graph, tenant)

    with capture_logs() as logs:
        await compute_centrality(graph, tenant)
    first = await written_scores(graph, tenant)
    await compute_centrality(graph, tenant)
    second = await written_scores(graph, tenant)

    assert first == second
    (event,) = [e for e in logs if e["event"] == "graph.centrality"]
    assert event["pages"] == CRAWLED
    assert min(event["pagerank_s"], event["betweenness_s"], event["write_s"]) >= 0
