from __future__ import annotations

import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from structlog.testing import capture_logs

from linking_engine.graph.algorithms import build_link_graphs, percentile_rank
from linking_engine.models import CentralityReport, Link, LinkGraphSnapshot, OrphanLabel, Page
from linking_engine.pipeline.analytics import (
    compute_centrality,
    compute_communities,
    load_link_graphs,
    orphan_label,
    page_centrality,
    summarise,
)

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


# ── communities ─────────────────────────────────────────────────────────────

TOPICS, TOPIC_SIZE, DIM = 4, 30, 16


def topic_url(topic: int, page: int) -> str:
    return f"example.com/t{topic}/p{page:02d}"


async def seed_planted(graph: GraphRepo, tenant: str) -> dict[str, int]:
    """Four topics, each linked mostly inside itself, with content vectors and keywords to match.
    Plus a page with no links, a page only reachable through menus, and one uncrawled target."""
    rng = random.Random(4)
    vectors = np.random.default_rng(4)
    centroids = vectors.normal(size=(TOPICS, DIM))
    planted = {topic_url(t, p): t for t in range(TOPICS) for p in range(TOPIC_SIZE)}
    crawled = [*planted, "example.com/lone", "example.com/menu-only"]
    await graph.upsert_pages(tenant, [Page(url=u, status_code=200) for u in crawled])
    await graph.upsert_placeholders(tenant, ["example.com/ghost"])
    links: list[Link] = []
    for url, topic in planted.items():
        targets = rng.sample([u for u, t in planted.items() if t == topic and u != url], 4)
        if rng.random() < 0.1:
            targets.append(rng.choice([u for u, t in planted.items() if t != topic]))
        links += [
            Link(source_url=url, target_url=t, position=i, anchor_text="a", surrounding_text="")
            for i, t in enumerate(targets)
        ]
    links.append(
        Link(
            source_url=topic_url(0, 0),
            target_url="example.com/ghost",
            position=9,
            anchor_text="a",
            surrounding_text="",
        )
    )
    links.append(
        Link(
            source_url="example.com/menu-only",
            target_url=topic_url(1, 0),
            position=0,
            anchor_text="a",
            surrounding_text="",
        )
    )
    await graph.replace_links(tenant, crawled, links)
    rows = [
        {
            "url": u,
            "vec": (centroids[t] + 0.3 * vectors.normal(size=DIM)).tolist(),
            "keywords": [f"topic {t} term {k}" for k in rng.sample(range(5), 2)],
        }
        for u, t in {**planted, "example.com/lone": 0}.items()
    ]
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec "
        "WITH p, row UNWIND row.keywords AS text "
        "MERGE (k:Keyword {tenantId: $t, text: text, language: 'en'}) "
        "MERGE (p)-[:TARGETS_KEYWORD]->(k)",
        t=tenant,
        rows=rows,
    )
    await graph._auto(
        "MATCH (p:Page {tenantId: $t, url: 'example.com/menu-only'}) SET p.menuInlinks = 2",
        t=tenant,
    )
    return planted


async def written(graph: GraphRepo, tenant: str) -> dict[str, Page]:
    rows = await graph._read(
        "MATCH (p:Page {tenantId: $tenant}) RETURN p.url AS url", tenant=tenant
    )
    pages = await graph.get_pages(tenant, [str(r["url"]) for r in rows])
    return {str(p.url): p for p in pages}


@pytest.mark.integration
async def test_three_passes_recover_planted_topics_and_label_every_crawled_page(
    graph: GraphRepo, tenant: str
) -> None:
    planted = await seed_planted(graph, tenant)

    report = await compute_communities(graph, tenant)

    pages = await written(graph, tenant)
    urls = sorted(planted)
    for field in ("link_community_id", "keyword_community_id", "content_community_id"):
        found = [getattr(pages[u], field) for u in urls]
        assert adjusted_rand_score([planted[u] for u in urls], found) > 0.7, field
    for found in (report.link, report.keyword, report.content):
        assert found.disconnected_communities == 0
        assert found.seed_stability_ari_mean is not None

    lone, menu_only, ghost = (
        pages["example.com/lone"],
        pages["example.com/menu-only"],
        pages["example.com/ghost"],
    )
    assert lone.link_community_id is None
    assert lone.content_community_id == pages[topic_url(0, 1)].content_community_id
    assert lone.keyword_community_id is not None
    assert (lone.is_orphan, lone.is_dead_end, lone.orphan_label) == (
        True,
        True,
        OrphanLabel.NOT_LINKED,
    )
    assert (menu_only.is_orphan, menu_only.is_dead_end, menu_only.orphan_label) == (
        True,
        False,
        OrphanLabel.MENUS_ONLY,
    )
    assert (ghost.link_community_id, ghost.is_orphan, ghost.content_community_id) == (None,) * 3

    link_communities = {
        pages[u].link_community_id for u in pages if pages[u].link_community_id is not None
    }
    assert sum(bool(p.is_link_pillar) for p in pages.values()) == report.link.pillars
    assert report.link.pillars == len(link_communities)
    assert (report.crawled_pages, report.seen_not_crawled) == (TOPICS * TOPIC_SIZE + 2, 1)
    assert report.orphan_labels[OrphanLabel.MENUS_ONLY] == 1
    assert report.keywords == TOPICS * 5
    assert report.pages_with_embeddings == TOPICS * TOPIC_SIZE + 1


@pytest.mark.integration
async def test_link_modularity_matches_networkx_on_the_written_labels(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_planted(graph, tenant)

    report = await compute_communities(graph, tenant)

    rows = await graph._read(
        "MATCH (a:Page {tenantId: $tenant})-[:LINKS_TO]->(b:Page {tenantId: $tenant}) "
        "WHERE a <> b AND NOT coalesce(a.isPlaceholder, false) "
        "AND NOT coalesce(b.isPlaceholder, false) "
        "RETURN a.url AS source, b.url AS target, a.linkCommunityId AS ca, b.linkCommunityId AS cb",
        tenant=tenant,
    )
    reference = nx.Graph((r["source"], r["target"]) for r in rows)
    label = {r["source"]: r["ca"] for r in rows} | {r["target"]: r["cb"] for r in rows}
    communities: dict[object, set[object]] = {}
    for node in reference.nodes:
        communities.setdefault(label[node], set()).add(node)
    expected = nx.community.modularity(reference, communities.values())
    assert abs(expected - report.link.modularity) <= 1e-9


@pytest.mark.integration
async def test_a_second_identical_run_writes_the_same_labels_with_no_drift(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_planted(graph, tenant)
    await compute_communities(graph, tenant)
    first = await written(graph, tenant)

    report = await compute_communities(graph, tenant)

    assert await written(graph, tenant) == first
    assert report.link.drift_ari == report.keyword.drift_ari == report.content.drift_ari == 1.0


@pytest.mark.integration
async def test_a_tenant_without_keywords_or_vectors_gets_link_communities_only(
    graph: GraphRepo, tenant: str
) -> None:
    urls = [f"example.com/{name}" for name in ("a", "b", "c")]
    await graph.upsert_pages(tenant, [Page(url=u, status_code=200) for u in urls])
    await graph.replace_links(
        tenant,
        urls,
        [
            Link(
                source_url=urls[0],
                target_url=urls[1],
                position=0,
                anchor_text="b",
                surrounding_text="",
            ),
            Link(
                source_url=urls[1],
                target_url=urls[2],
                position=0,
                anchor_text="c",
                surrounding_text="",
            ),
        ],
    )

    report = await compute_communities(graph, tenant)

    assert (report.keyword.pages, report.content.pages, report.link.pages) == (0, 0, 3)
    assert report.agreement_link_content is None
    assert "Keyword communities: no input" in summarise(
        CentralityReport(
            tenant_id=tenant, pages=3, placeholders=0, pagerank_s=0, betweenness_s=0, write_s=0
        ),
        report,
    )


@pytest.mark.parametrize(
    ("menu", "footer", "label"),
    [
        (0, 0, OrphanLabel.NOT_LINKED),
        (2, 0, OrphanLabel.MENUS_ONLY),
        (0, 1, OrphanLabel.FOOTER_ONLY),
        (1, 1, OrphanLabel.MENUS_AND_FOOTER_ONLY),
    ],
)
def test_orphan_labels_follow_the_template_inlinks(
    menu: int, footer: int, label: OrphanLabel
) -> None:
    assert orphan_label(menu, footer) is label
