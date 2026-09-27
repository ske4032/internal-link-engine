"""Pure graph algorithms over in-process igraph graphs; no database access (import-linter enforced)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import igraph as ig

if TYPE_CHECKING:
    from linking_engine.models import LinkGraphSnapshot


@dataclass(frozen=True, slots=True)
class LinkGraphs:
    """One tenant's link graph; vertex i is ``urls[i]`` in both graphs."""

    urls: tuple[str, ...]
    placeholders: tuple[bool, ...]
    # One edge per ordered page pair; weight counts the links. For PageRank.
    directed: ig.Graph
    # One edge per unordered page pair; weight counts links both ways. For Leiden and betweenness.
    undirected: ig.Graph

    def url_of(self, vertex: int) -> str:
        return self.urls[vertex]


def build_link_graphs(snapshot: LinkGraphSnapshot) -> LinkGraphs:
    """Dense 0..n-1 ids in page order, so orphans and placeholders stay in the graph at degree 0.
    Self-links are dropped; repeated and reciprocal links become edge weights."""
    index = {url: vertex for vertex, url in enumerate(snapshot.pages)}
    if len(index) != len(snapshot.pages):
        raise ValueError("duplicate page urls in the snapshot")
    edges: list[tuple[int, int]] = []
    for source, target in snapshot.links:
        try:
            edges.append((index[source], index[target]))
        except KeyError as error:
            raise ValueError(
                f"link endpoint {error.args[0]!r} is not a page of the snapshot"
            ) from error

    directed = ig.Graph(n=len(snapshot.pages), edges=edges, directed=True)
    directed.es["weight"] = [1] * len(edges)
    directed.simplify(multiple=True, loops=True, combine_edges={"weight": "sum"})
    undirected = directed.as_undirected(mode="collapse", combine_edges={"weight": "sum"})
    return LinkGraphs(
        urls=snapshot.pages,
        placeholders=snapshot.placeholders,
        directed=directed,
        undirected=undirected,
    )
