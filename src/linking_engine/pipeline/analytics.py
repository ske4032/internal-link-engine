"""Graph analytics stage: pull a tenant's link graph and build it in igraph."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from linking_engine.graph.algorithms import LinkGraphs, build_link_graphs

if TYPE_CHECKING:
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
