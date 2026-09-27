"""Graph analytics stage: pull a tenant's link graph, build it in igraph and score its pages."""

from __future__ import annotations

import time
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import structlog

from linking_engine.graph.algorithms import (
    LinkGraphs,
    Partition,
    agreement,
    build_link_graphs,
    content_pass_graph,
    crawled_betweenness,
    disconnected_communities,
    keyword_pass_graph,
    link_pass_graph,
    link_states,
    page_rank,
    partition,
    percentile_rank,
    pillars,
    seed_stability,
)
from linking_engine.models import (
    CentralityReport,
    CommunityReport,
    OrphanLabel,
    PageCentrality,
    PageCommunities,
    PassReport,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import igraph as ig
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


async def compute_centrality(
    graph: GraphRepo, tenant_id: str, graphs: LinkGraphs | None = None
) -> CentralityReport:
    """PageRank and exact betweenness for every crawled page of the tenant, written back."""
    graphs = graphs or await load_link_graphs(graph, tenant_id)
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


def orphan_label(menu_inlinks: int, footer_inlinks: int) -> OrphanLabel:
    """What still links to a page no body link reaches."""
    if menu_inlinks and footer_inlinks:
        return OrphanLabel.MENUS_AND_FOOTER_ONLY
    if menu_inlinks:
        return OrphanLabel.MENUS_ONLY
    if footer_inlinks:
        return OrphanLabel.FOOTER_ONLY
    return OrphanLabel.NOT_LINKED


async def compute_communities(
    graph: GraphRepo, tenant_id: str, graphs: LinkGraphs | None = None
) -> CommunityReport:
    """Link, keyword and content communities, their pillar pages and every crawled page's link
    state, written back. Seen-but-not-crawled pages take no part."""
    graphs = graphs or await load_link_graphs(graph, tenant_id)
    crawled = set(graphs.crawled)
    vertex_of = {url: v for v, url in enumerate(graphs.urls) if v in crawled}
    context = {c.url: c for c in await graph.community_context(tenant_id)}

    keyword_ids: dict[tuple[str, str], int] = {}
    page_keywords: dict[int, set[int]] = {}
    for url, text, language in await graph.keyword_targets(tenant_id):
        if url in vertex_of:
            keyword = keyword_ids.setdefault((text, language), len(keyword_ids))
            page_keywords.setdefault(vertex_of[url], set()).add(keyword)
    stored = await graph.content_vectors(tenant_id)
    vectors = {vertex_of[url]: vector for url, vector in stored.items() if url in vertex_of}
    ranks = page_rank(graphs)

    link_graph, link_vertices = link_pass_graph(graphs)
    keyword_graph, keyword_vertices, dropped = keyword_pass_graph(page_keywords)
    embedded = sorted(vectors)
    content_graph, content_vertices = content_pass_graph(
        embedded, np.stack([vectors[v] for v in embedded]) if embedded else np.empty((0, 0))
    )

    def previous(field: str) -> dict[int, int]:
        return {
            vertex_of[url]: value
            for url, item in context.items()
            if url in vertex_of and (value := getattr(item, field)) is not None
        }

    passes: dict[str, tuple[Partition, PassReport, frozenset[int]]] = {}
    for name, pass_graph, vertices, weighted in (
        ("link", link_graph, link_vertices, False),
        ("keyword", keyword_graph, keyword_vertices, True),
        ("content", content_graph, content_vertices, True),
    ):
        passes[name] = _run_pass(
            pass_graph, vertices, weighted, vectors, ranks, previous(f"{name}_community_id")
        )
        log.info(
            "graph.communities", tenant_id=tenant_id, graph=name, **passes[name][1].model_dump()
        )

    no_inbound, no_outbound = link_states(graphs)
    labels = {name: found.labels() for name, (found, _, _) in passes.items()}
    rows: list[PageCommunities] = []
    for i, vertex in enumerate(graphs.crawled):
        url = graphs.url_of(vertex)
        orphan = bool(no_inbound[i])
        item = context.get(url)
        rows.append(
            PageCommunities(
                url=url,
                link_community_id=labels["link"].get(vertex),
                keyword_community_id=labels["keyword"].get(vertex),
                content_community_id=labels["content"].get(vertex),
                is_link_pillar=vertex in passes["link"][2],
                is_keyword_pillar=vertex in passes["keyword"][2],
                is_content_pillar=vertex in passes["content"][2],
                is_orphan=orphan,
                is_dead_end=bool(no_outbound[i]),
                orphan_label=(
                    orphan_label(
                        item.menu_inlinks if item else 0, item.footer_inlinks if item else 0
                    )
                    if orphan
                    else None
                ),
            )
        )
    started = time.perf_counter()
    await graph.write_communities(tenant_id, rows)
    report = CommunityReport(
        tenant_id=tenant_id,
        crawled_pages=len(rows),
        seen_not_crawled=sum(graphs.placeholders),
        link=passes["link"][1],
        keyword=passes["keyword"][1],
        content=passes["content"][1],
        keywords=len(keyword_ids),
        keywords_dropped=len(dropped),
        pages_with_keywords=len(page_keywords),
        pages_with_embeddings=len(vectors),
        agreement_link_content=agreement(labels["link"], labels["content"]),
        agreement_link_keyword=agreement(labels["link"], labels["keyword"]),
        agreement_keyword_content=agreement(labels["keyword"], labels["content"]),
        orphans=sum(row.is_orphan for row in rows),
        dead_ends=sum(row.is_dead_end for row in rows),
        orphan_labels=dict(Counter(row.orphan_label for row in rows if row.orphan_label)),
        write_s=round(time.perf_counter() - started, 3),
    )
    log.info(
        "graph.communities.written",
        tenant_id=tenant_id,
        pages=report.crawled_pages,
        orphans=report.orphans,
        dead_ends=report.dead_ends,
        write_s=report.write_s,
    )
    return report


def _run_pass(
    pass_graph: ig.Graph,
    vertices: tuple[int, ...],
    weighted: bool,
    vectors: Mapping[int, npt.NDArray[np.float32]],
    ranks: npt.NDArray[np.float64],
    previous: dict[int, int],
) -> tuple[Partition, PassReport, frozenset[int]]:
    started = time.perf_counter()
    found = partition(pass_graph, vertices, weighted=weighted)
    ran = time.perf_counter()
    stability = seed_stability(pass_graph, weighted=weighted)
    stable = time.perf_counter()
    chosen = pillars(found, vectors, ranks)
    sizes = np.bincount(found.membership) if found.membership else np.zeros(0, dtype=np.int64)
    report = PassReport(
        pages=len(found.vertices),
        edges=found.edges,
        communities=len(sizes),
        singletons=int((sizes == 1).sum()),
        largest_community_pct=float(sizes.max() / len(found.vertices)) if len(sizes) else 0.0,
        median_community_size=float(np.median(sizes)) if len(sizes) else 0.0,
        modularity=found.modularity,
        disconnected_communities=disconnected_communities(pass_graph, found.membership),
        seed_stability_ari_mean=stability[0] if stability else None,
        seed_stability_ari_min=stability[1] if stability else None,
        drift_ari=agreement(previous, found.labels()),
        pillars=len(chosen),
        runtime_s=round(ran - started, 3),
        stability_s=round(stable - ran, 3),
    )
    return found, report, chosen


def summarise(centrality: CentralityReport, communities: CommunityReport) -> str:
    """A short prose record of one analytics run, for the MLflow run description."""

    def described(name: str, found: PassReport) -> str:
        if not found.pages:
            return f"{name}: no input, so no communities."
        stability = (
            f", seed stability {found.seed_stability_ari_mean:.2f}"
            if found.seed_stability_ari_mean is not None
            else ""
        )
        drift = (
            f", ARI {found.drift_ari:.2f} against the previous run"
            if found.drift_ari is not None
            else ""
        )
        return (
            f"{name}: {found.communities} communities over {found.pages} pages, largest "
            f"{found.largest_community_pct:.0%}, modularity {found.modularity:.3f}{stability}{drift}."
        )

    labels = ", ".join(
        f"{count} {label.value.lower().replace('_', ' ')}"
        for label, count in sorted(communities.orphan_labels.items())
    )
    return "\n".join(
        [
            f"Graph analytics for tenant {communities.tenant_id}: {communities.crawled_pages} crawled "
            f"pages, {communities.seen_not_crawled} seen but not crawled (excluded).",
            f"PageRank and exact betweenness for {centrality.pages} pages in "
            f"{centrality.pagerank_s + centrality.betweenness_s:.2f} s.",
            described("Link communities", communities.link),
            described("Content communities", communities.content),
            described("Keyword communities", communities.keyword)
            + (
                f" {communities.keywords_dropped} most shared keywords dropped."
                if communities.keywords_dropped
                else ""
            ),
            f"Orphans (no body link in): {communities.orphans}"
            + (f" ({labels})." if labels else ".")
            + f" Dead ends (no body link out): {communities.dead_ends}.",
        ]
    )
