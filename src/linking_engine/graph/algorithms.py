"""Pure graph algorithms over in-process igraph graphs; no database access (import-linter enforced)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import igraph as ig
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from linking_engine.models import LinkGraphSnapshot

DAMPING: Final = 0.85
# Scores this close, relative to the largest, are ties: summation order must not reorder them.
TIE_TOLERANCE: Final = 1e-9


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

    @property
    def crawled(self) -> tuple[int, ...]:
        return tuple(v for v, placeholder in enumerate(self.placeholders) if not placeholder)


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


def page_rank(graphs: LinkGraphs) -> npt.NDArray[np.float64]:
    """PageRank of every vertex, one vote per linked page pair. Placeholders stay in, so links
    to them still dilute the linking page; they are dangling, so their rank spreads evenly."""
    return np.asarray(
        graphs.directed.pagerank(damping=DAMPING, weights=None, directed=True), dtype=np.float64
    )


def crawled_betweenness(graphs: LinkGraphs) -> npt.NDArray[np.float64]:
    """Exact unweighted betweenness over crawled pages only, aligned with ``graphs.crawled``.
    A placeholder has no known out-links; kept in, it would bridge every page linking to it."""
    crawled = graphs.undirected.induced_subgraph(graphs.crawled)
    return np.asarray(crawled.betweenness(directed=False, weights=None), dtype=np.float64)


def percentile_rank(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Share of values strictly below each one, in [0, 1); ties share the lowest rank."""
    scores = np.asarray(values, dtype=np.float64)
    count = len(scores)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    tolerance = TIE_TOLERANCE * float(np.abs(ordered).max())
    starts_group = np.empty(count, dtype=bool)
    starts_group[0] = True
    starts_group[1:] = np.diff(ordered) > tolerance
    below = np.maximum.accumulate(np.where(starts_group, np.arange(count), 0))
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = below / count
    return ranks
