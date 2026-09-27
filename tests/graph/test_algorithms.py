from __future__ import annotations

import random
from fractions import Fraction

import igraph as ig
import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.metrics import adjusted_rand_score

from linking_engine.graph.algorithms import (
    Partition,
    agreement,
    build_link_graphs,
    content_pass_graph,
    crawled_betweenness,
    disconnected_communities,
    keyword_pass_graph,
    knn_graph,
    link_pass_graph,
    link_states,
    page_rank,
    partition,
    percentile_rank,
    pillars,
    seed_stability,
)
from linking_engine.models import LinkGraphSnapshot


def snapshot(
    pages: list[str], links: list[tuple[str, str]], placeholders: list[bool] | None = None
) -> LinkGraphSnapshot:
    return LinkGraphSnapshot(
        tenant_id="t",
        pages=tuple(pages),
        placeholders=tuple(placeholders or [False] * len(pages)),
        links=tuple(links),
    )


def test_orphans_and_placeholders_stay_in_the_graph_at_degree_zero() -> None:
    graphs = build_link_graphs(
        snapshot(["a", "b", "orphan", "ghost"], [("a", "b")], [False, False, False, True])
    )
    assert graphs.directed.vcount() == graphs.undirected.vcount() == 4
    assert graphs.undirected.degree(2) == graphs.undirected.degree(3) == 0
    assert graphs.placeholders == (False, False, False, True)


def test_reciprocal_links_are_two_directed_edges_and_one_weighted_undirected_edge() -> None:
    graphs = build_link_graphs(snapshot(["a", "b"], [("a", "b"), ("b", "a")]))
    assert graphs.directed.ecount() == 2
    assert graphs.undirected.ecount() == 1
    assert graphs.undirected.es["weight"] == [2]


def test_repeated_links_collapse_into_one_weighted_edge_and_self_links_are_dropped() -> None:
    graphs = build_link_graphs(snapshot(["a", "b"], [("a", "b"), ("a", "b"), ("a", "a")]))
    assert graphs.directed.ecount() == 1
    assert graphs.directed.es["weight"] == [2]
    assert graphs.directed.is_directed()
    assert not graphs.undirected.is_directed()


def test_dense_index_round_trips_to_the_url() -> None:
    pages = ["x.com/a", "x.com/b", "x.com/c"]
    graphs = build_link_graphs(snapshot(pages, [("x.com/a", "x.com/c")]))
    assert [graphs.url_of(v) for v in range(graphs.directed.vcount())] == pages
    source, target = graphs.directed.es[0].tuple
    assert (graphs.url_of(source), graphs.url_of(target)) == ("x.com/a", "x.com/c")


def test_an_empty_tenant_builds_empty_graphs() -> None:
    graphs = build_link_graphs(snapshot([], []))
    assert (graphs.directed.vcount(), graphs.undirected.ecount()) == (0, 0)


def test_a_link_to_an_unknown_page_is_rejected() -> None:
    with pytest.raises(ValueError, match="'c' is not a page"):
        build_link_graphs(snapshot(["a", "b"], [("a", "c")]))


def test_duplicate_page_urls_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate page urls"):
        build_link_graphs(snapshot(["a", "a"], []))


def test_snapshot_needs_one_placeholder_flag_per_page() -> None:
    with pytest.raises(ValidationError, match="one flag per page"):
        LinkGraphSnapshot(tenant_id="t", pages=("a", "b"), placeholders=(False,), links=())


# ── PageRank and betweenness: closed forms, each under shuffled vertex ids ──

SEEDS = [0, 1, 2]


def scores(
    pages: list[str],
    links: list[tuple[str, str]],
    placeholders: list[bool] | None = None,
    *,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, float]]:
    """PageRank and betweenness by url, after shuffling the page order that assigns vertex ids."""
    flags = placeholders or [False] * len(pages)
    order = random.Random(seed).sample(range(len(pages)), len(pages))
    graphs = build_link_graphs(
        snapshot([pages[i] for i in order], links, [flags[i] for i in order])
    )
    ranks = page_rank(graphs)
    betweenness = crawled_betweenness(graphs)
    return (
        {graphs.url_of(v): float(ranks[v]) for v in range(len(pages))},
        {graphs.url_of(v): float(betweenness[i]) for i, v in enumerate(graphs.crawled)},
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_path_betweenness_is_i_times_n_minus_1_minus_i(seed: int) -> None:
    n = 7
    pages = [f"p{i}" for i in range(n)]
    _, betweenness = scores(pages, [(pages[i], pages[i + 1]) for i in range(n - 1)], seed=seed)
    for i, url in enumerate(pages):
        assert betweenness[url] == pytest.approx(i * (n - 1 - i), rel=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_star_centre_betweenness_is_n_minus_1_choose_2_and_leaves_zero(seed: int) -> None:
    n = 8
    leaves = [f"leaf{i}" for i in range(n - 1)]
    # Direction is ignored: half the leaves link in, half are linked to.
    links = [(leaf, "hub") if i % 2 else ("hub", leaf) for i, leaf in enumerate(leaves)]
    _, betweenness = scores(["hub", *leaves], links, seed=seed)
    assert betweenness["hub"] == pytest.approx((n - 1) * (n - 2) / 2, rel=1e-9)
    assert all(betweenness[leaf] == 0 for leaf in leaves)


@pytest.mark.parametrize("seed", SEEDS)
def test_directed_cycle_pagerank_is_uniform(seed: int) -> None:
    n = 5
    pages = [f"c{i}" for i in range(n)]
    ranks, _ = scores(pages, [(pages[i], pages[(i + 1) % n]) for i in range(n)], seed=seed)
    for url in pages:
        assert ranks[url] == pytest.approx(1 / n, rel=1e-9)


# a -> b, a -> c, b -> c; c is dangling. Derivation in test_dangling_derivation_is_consistent.
DANGLING_EXPECTED = {
    "a": Fraction(800, 4049),
    "b": Fraction(1140, 4049),
    "c": Fraction(2109, 4049),
}


def test_dangling_derivation_is_consistent() -> None:
    """c has no out-links, so its rank is spread over all three pages.
    With d = 17/20, n = 3 and t = (1 - d)/n:
        a = t + d*c/3
        b = t + d*(a/2 + c/3)
        c = t + d*(a/2 + b + c/3)
    Put k = d*c/3, so c = 3k/d, a = t + k, b = t + d*a/2 + k. From a + b + c = 1:
        k * (3/d + 2 + d/2) = 1 - 2t - d*t/2, so k = 11951/80980,
    giving a = 800/4049, b = 1140/4049, c = 2109/4049.
    """
    d, t = Fraction(17, 20), Fraction(1, 20)
    a, b, c = DANGLING_EXPECTED.values()
    assert d * c / 3 == Fraction(11951, 80980)
    assert a + b + c == 1
    assert a == t + d * c / 3
    assert b == t + d * (a / 2 + c / 3)
    assert c == t + d * (a / 2 + b + c / 3)


@pytest.mark.parametrize("seed", SEEDS)
def test_dangling_page_pagerank_matches_the_hand_derivation(seed: int) -> None:
    ranks, _ = scores(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c")], seed=seed)
    for url, expected in DANGLING_EXPECTED.items():
        assert ranks[url] == pytest.approx(float(expected), rel=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_placeholder_never_bridges_crawled_pages(seed: int) -> None:
    # a - b - c is the only crawled path; a and c also both link to placeholder x.
    links = [("a", "b"), ("b", "c"), ("a", "x"), ("c", "x")]
    _, betweenness = scores(["a", "b", "c", "x"], links, [False, False, False, True], seed=seed)
    assert betweenness == {"a": 0, "b": 1, "c": 0}


@pytest.mark.parametrize("seed", SEEDS)
def test_a_link_to_a_placeholder_still_takes_its_share_of_pagerank(seed: int) -> None:
    ranks, _ = scores(["a", "b", "x"], [("a", "b"), ("a", "x")], [False, False, True], seed=seed)
    assert ranks["b"] == pytest.approx(ranks["x"], rel=1e-9)
    assert sum(ranks.values()) == pytest.approx(1, rel=1e-9)


def test_repeated_links_count_as_one_vote() -> None:
    once, _ = scores(["a", "b", "c"], [("a", "b"), ("a", "c")])
    repeated, _ = scores(["a", "b", "c"], [("a", "b"), ("a", "b"), ("a", "b"), ("a", "c")])
    assert repeated == pytest.approx(once, rel=1e-12)


def test_an_isolated_page_gets_a_small_nonzero_pagerank() -> None:
    ranks, betweenness = scores(["a", "b", "lone"], [("a", "b"), ("b", "a")])
    assert 0 < ranks["lone"] < ranks["a"]
    assert betweenness["lone"] == 0


def test_an_empty_graph_has_no_scores() -> None:
    graphs = build_link_graphs(snapshot([], []))
    assert len(page_rank(graphs)) == len(crawled_betweenness(graphs)) == 0


def test_crawled_lists_every_non_placeholder_vertex_in_order() -> None:
    graphs = build_link_graphs(snapshot(["a", "b", "c", "d"], [], [True, False, True, False]))
    assert graphs.crawled == (1, 3)


# ── percentile rank ─────────────────────────────────────────────────────────


def test_percentile_is_the_share_strictly_below_and_ties_share_the_lowest() -> None:
    assert percentile_rank([3, 1, 3, 0]).tolist() == [0.5, 0.25, 0.5, 0]


def test_scores_within_the_tolerance_tie_and_wider_gaps_do_not() -> None:
    assert percentile_rank([1.0, 1.0 + 1e-12, 0.5]).tolist() == [1 / 3, 1 / 3, 0]
    assert percentile_rank([1.0, 1.0 + 1e-6]).tolist() == [0, 0.5]


@pytest.mark.parametrize(
    ("values", "expected"),
    [([], []), ([7.0], [0.0]), ([0, 0, 0], [0, 0, 0])],
    ids=["empty", "single", "all-zero"],
)
def test_percentile_edge_cases(values: list[float], expected: list[float]) -> None:
    assert percentile_rank(values).tolist() == expected


def test_percentiles_stay_below_one_and_follow_the_order() -> None:
    values = np.random.default_rng(3).random(200)
    ranks = percentile_rank(values)
    assert ranks.max() < 1
    assert (np.argsort(ranks, kind="stable") == np.argsort(values, kind="stable")).all()


# ── Leiden passes ───────────────────────────────────────────────────────────


def two_cliques(size: int = 5) -> ig.Graph:
    """Two cliques of ``size`` joined by one edge between vertex 0 and vertex ``size``."""
    left = [(a, b) for a in range(size) for b in range(a + 1, size)]
    right = [(a + size, b + size) for a, b in left]
    return ig.Graph(n=2 * size, edges=[*left, *right, (0, size)])


def groups(found: Partition) -> set[frozenset[int]]:
    members: dict[int, set[int]] = {}
    for vertex, community in zip(found.vertices, found.membership, strict=True):
        members.setdefault(community, set()).add(vertex)
    return {frozenset(m) for m in members.values()}


def test_two_cliques_joined_by_one_edge_are_exactly_the_two_cliques_every_run() -> None:
    graph = two_cliques()
    first = partition(graph, range(10), weighted=False)
    second = partition(graph, range(10), weighted=False)
    assert groups(first) == {frozenset(range(5)), frozenset(range(5, 10))}
    assert first == second
    assert seed_stability(graph, weighted=False) == (1.0, 1.0)


def test_planted_topics_are_recovered() -> None:
    rng = random.Random(9)
    topics, size = 8, 40
    planted = [v // size for v in range(topics * size)]
    edges = {
        (a, b)
        for a in range(topics * size)
        for b in range(a + 1, topics * size)
        if rng.random() < (0.25 if planted[a] == planted[b] else 0.004)
    }
    found = partition(
        ig.Graph(n=topics * size, edges=sorted(edges)), range(topics * size), weighted=False
    )
    assert adjusted_rand_score(planted, found.membership) > 0.7
    graph = ig.Graph(n=topics * size, edges=sorted(edges))
    assert disconnected_communities(graph, found.membership) == 0


def test_a_community_split_into_pieces_is_counted() -> None:
    graph = ig.Graph(n=4, edges=[(0, 1), (2, 3)])
    assert disconnected_communities(graph, [0, 0, 0, 0]) == 1
    assert disconnected_communities(graph, [0, 0, 1, 1]) == 0


def test_partition_needs_one_id_per_vertex_and_handles_an_empty_graph() -> None:
    with pytest.raises(ValueError, match="one vertex id"):
        partition(two_cliques(), range(3), weighted=False)
    assert partition(ig.Graph(), (), weighted=False) == Partition((), (), 0.0, 0)
    assert seed_stability(ig.Graph(n=1), weighted=False) is None


def test_the_link_pass_keeps_crawled_pages_with_a_crawled_link_only() -> None:
    graphs = build_link_graphs(
        snapshot(
            ["a", "b", "c", "ghost", "lone"],
            [("a", "b"), ("a", "b"), ("c", "ghost")],
            [False, False, False, True, False],
        )
    )
    graph, vertices = link_pass_graph(graphs)
    assert [graphs.url_of(v) for v in vertices] == ["a", "b"]
    assert graph.ecount() == 1


def test_keyword_projection_weights_pairs_by_shared_keywords() -> None:
    graph, vertices, dropped = keyword_pass_graph({10: {1, 2}, 11: {1, 2}, 12: {2}, 13: {3}})
    assert (vertices, dropped) == ((10, 11, 12), ())
    weights = {
        tuple(sorted((vertices[e.source], vertices[e.target]))): w
        for e, w in zip(graph.es, graph.es["weight"], strict=True)
    }
    assert weights == {(10, 11): 2.0, (10, 12): 1.0, (11, 12): 1.0}


def test_the_most_shared_keywords_are_dropped_to_fit_the_budget() -> None:
    brand = {page: {0} for page in range(6)}  # 15 pairs
    niche = {6: {1}, 7: {1}}  # 1 pair
    graph, vertices, dropped = keyword_pass_graph({**brand, **niche}, budget=10)
    assert dropped == (0,)
    assert (vertices, graph.ecount()) == ((6, 7), 1)


def test_a_projection_without_shared_keywords_is_empty() -> None:
    graph, vertices, dropped = keyword_pass_graph({1: {1}, 2: {2}})
    assert (graph.vcount(), vertices, dropped) == (0, (), ())


def test_knn_joins_nearest_rows_and_keeps_the_stronger_direction() -> None:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]])
    graph = knn_graph(vectors, k=1)
    edges = {tuple(sorted(e.tuple)) for e in graph.es}
    assert {(0, 1), (2, 3)} <= edges
    assert all(w > 0 for w in graph.es["weight"])
    # Row 4 points away from everything: its best similarity is negative, so it has no edge.
    assert graph.degree(4) == 0


def test_knn_rejects_zero_vectors_and_caps_k() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        knn_graph(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert knn_graph(np.array([[1.0, 0.0], [1.0, 0.1]]), k=10).ecount() == 1
    assert knn_graph(np.array([[1.0, 0.0]])).vcount() == 1


def test_the_content_pass_maps_rows_to_vertex_ids_and_drops_pages_without_edges() -> None:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    graph, vertices = content_pass_graph((7, 8, 9), vectors, k=1)
    assert (vertices, graph.ecount()) == ((7, 8), 1)
    with pytest.raises(ValueError, match="one vector per vertex"):
        content_pass_graph((1,), vectors)


def test_agreement_ignores_label_names_and_needs_two_shared_vertices() -> None:
    assert agreement({1: 0, 2: 0, 3: 1}, {1: 5, 2: 5, 3: 9, 4: 9}) == 1.0
    assert agreement({1: 0}, {1: 0, 2: 1}) is None


def test_the_pillar_is_the_member_nearest_the_content_centroid() -> None:
    found = Partition((1, 2, 3, 4), (0, 0, 0, 0), 0.5, 3)
    vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
        3: np.array([0.7, 0.7]),
        4: np.array([1.0, 0.1]),
    }
    assert pillars(found, vectors, np.zeros(5)) == {3}


def test_pagerank_breaks_ties_and_decides_without_vectors() -> None:
    ranks = np.array([0.0, 0.1, 0.5, 0.2, 0.0])
    tied = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    found = Partition((1, 2, 3), (0, 0, 0), 0.5, 2)
    assert pillars(found, tied, ranks) == {2}
    assert pillars(found, {}, ranks) == {2}


def test_small_communities_have_no_pillar() -> None:
    found = Partition((1, 2, 3, 4, 5), (0, 0, 1, 1, 1), 0.5, 3)
    assert pillars(found, {}, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])) == {5}


def test_link_states_flag_orphans_and_dead_ends_among_crawled_pages() -> None:
    graphs = build_link_graphs(
        snapshot(
            ["hub", "leaf", "orphan", "ghost", "self"],
            [("hub", "leaf"), ("orphan", "hub"), ("leaf", "ghost"), ("self", "self")],
            [False, False, False, True, False],
        )
    )
    no_inbound, no_outbound = link_states(graphs)
    crawled = [graphs.url_of(v) for v in graphs.crawled]
    assert dict(zip(crawled, no_inbound.tolist(), strict=True)) == {
        "hub": False,
        "leaf": False,
        "orphan": True,
        "self": True,
    }
    assert dict(zip(crawled, no_outbound.tolist(), strict=True)) == {
        "hub": False,
        "leaf": False,
        "orphan": False,
        "self": True,
    }
