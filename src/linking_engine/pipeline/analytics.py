"""Graph analytics stage: pull a tenant's link graph, build it in igraph and score its pages."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from linking_engine.graph.algorithms import (
    LinkGraphs,
    build_link_graphs,
    crawled_betweenness,
    page_rank,
    percentile_rank,
)
from linking_engine.models import CentralityReport, PageCentrality

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from linking_engine.graph.repo import GraphRepo

log = structlog.get_logger(__name__)


async def load_link_graphs(graph: GraphRepo, tenant_id: str) -> LinkGraphs:
    """Rebuilt every run from url keys; nothing about the mapping is cached."""
    started = time.perf_counter()
    snapshot = await graph.link_graph(tenant_id)
    pulled = time.perf_counter()
    graphs = build_link_graphs(snapshot)
    log.info(
        "graph.build",
        tenant_id=tenant_id,
        pages=len(snapshot.pages),
        placeholders=sum(snapshot.placeholders),
        link_rows=len(snapshot.links),
        directed_edges=graphs.directed.ecount(),
        undirected_edges=graphs.undirected.ecount(),
        isolated=sum(1 for degree in graphs.undirected.degree() if degree == 0),
        pull_s=round(pulled - started, 3),
        build_s=round(time.perf_counter() - pulled, 3),
    )
    return graphs


def page_centrality(
    graphs: LinkGraphs,
    ranks: npt.NDArray[np.float64],
    betweenness: npt.NDArray[np.float64],
) -> tuple[PageCentrality, ...]:
    """One row per crawled page; ``ranks`` covers every vertex, ``betweenness`` crawled ones."""
    crawled = list(graphs.crawled)
    crawled_ranks = ranks[crawled]
    rank_percentiles = percentile_rank(crawled_ranks)
    betweenness_percentiles = percentile_rank(betweenness)
    return tuple(
        PageCentrality(
            url=graphs.url_of(vertex),
            page_rank=float(crawled_ranks[i]),
            page_rank_percentile=float(rank_percentiles[i]),
            betweenness=float(betweenness[i]),
            betweenness_percentile=float(betweenness_percentiles[i]),
        )
        for i, vertex in enumerate(crawled)
    )


async def compute_centrality(graph: GraphRepo, tenant_id: str) -> CentralityReport:
    """PageRank and exact betweenness for every crawled page of the tenant, written back."""
    graphs = await load_link_graphs(graph, tenant_id)
    started = time.perf_counter()
    ranks = page_rank(graphs)
    ranked = time.perf_counter()
    betweenness = crawled_betweenness(graphs)
    measured = time.perf_counter()
    scores = page_centrality(graphs, ranks, betweenness)
    written = await graph.write_centrality(tenant_id, scores)
    report = CentralityReport(
        tenant_id=tenant_id,
        pages=written,
        placeholders=sum(graphs.placeholders),
        pagerank_s=round(ranked - started, 3),
        betweenness_s=round(measured - ranked, 3),
        write_s=round(time.perf_counter() - measured, 3),
    )
    log.info("graph.centrality", **report.model_dump())
    return report
