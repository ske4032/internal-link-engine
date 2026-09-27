from __future__ import annotations

import pytest
from pydantic import ValidationError

from linking_engine.graph.algorithms import build_link_graphs
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
