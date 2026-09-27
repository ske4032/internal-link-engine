"""Pure graph algorithms over in-process igraph graphs; no database access (import-linter enforced)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Final

import hdbscan
import igraph as ig
import leidenalg as la
import numpy as np
from sklearn.metrics import adjusted_rand_score

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    import numpy.typing as npt

    from linking_engine.models import LinkGraphSnapshot

DAMPING: Final = 0.85
# Scores this close, relative to the largest, are ties: summation order must not reorder them.
TIE_TOLERANCE: Final = 1e-9
RESOLUTION: Final = 1.0
SEED: Final = 42
# Extra seeds whose partitions measure how stable the seeded one is.
STABILITY_SEEDS: Final = (1, 2, 3, 4)
KNN_NEIGHBOURS: Final = 10
# A keyword on k pages projects to k(k-1)/2 page pairs; the most shared keywords are dropped first.
PROJECTION_EDGE_BUDGET: Final = 5_000_000
MIN_PILLAR_COMMUNITY: Final = 3
# HDBSCAN over raw unit page vectors (hdbscan-eval studies 1-3): min_samples=1 keeps planted
# noise at -1 while halving false noise; PCA or UMAP lose the noise label.
HUB_MIN_CLUSTER_SIZE: Final = 10
HUB_MIN_SAMPLES: Final = 1
# Cosine similarity at which a new hub centroid keeps the id of a previous one.
HUB_MATCH_SIMILARITY: Final = 0.9
NOISE: Final = -1


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


@dataclass(frozen=True, slots=True)
class Partition:
    """Leiden communities; ``vertices`` are LinkGraphs vertex ids aligned with ``membership``."""

    vertices: tuple[int, ...]
    membership: tuple[int, ...]
    modularity: float
    edges: int

    def labels(self) -> dict[int, int]:
        return dict(zip(self.vertices, self.membership, strict=True))


def leiden(graph: ig.Graph, *, weighted: bool, seed: int = SEED) -> tuple[list[int], float]:
    """RBConfiguration at RESOLUTION, run until no move improves it."""
    found = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights="weight" if weighted else None,
        resolution_parameter=RESOLUTION,
        seed=seed,
        n_iterations=-1,
    )
    return list(found.membership), float(found.modularity)


def partition(
    graph: ig.Graph, vertices: Sequence[int], *, weighted: bool, seed: int = SEED
) -> Partition:
    if len(vertices) != graph.vcount():
        raise ValueError("one vertex id per graph vertex")
    if graph.vcount() == 0:
        return Partition((), (), 0.0, 0)
    membership, modularity = leiden(graph, weighted=weighted, seed=seed)
    return Partition(tuple(vertices), tuple(membership), modularity, graph.ecount())


def link_pass_graph(graphs: LinkGraphs) -> tuple[ig.Graph, tuple[int, ...]]:
    """Crawled pages with a body link to or from another crawled page, one unweighted edge per pair."""
    crawled = graphs.crawled
    graph = graphs.undirected.induced_subgraph(crawled)
    return _with_edges(graph, crawled)


def keyword_pass_graph(
    page_keywords: Mapping[int, Collection[int]], budget: int = PROJECTION_EDGE_BUDGET
) -> tuple[ig.Graph, tuple[int, ...], tuple[int, ...]]:
    """Pages joined by the number of keywords they share. Returns the graph, its vertex ids and
    the keywords dropped, most shared first, to keep the projection within ``budget`` pairs."""
    pages_of: dict[int, list[int]] = defaultdict(list)
    for page, keywords in page_keywords.items():
        for keyword in set(keywords):
            pages_of[keyword].append(page)
    pairs = sum(len(pages) * (len(pages) - 1) // 2 for pages in pages_of.values())
    dropped: list[int] = []
    for keyword in sorted(pages_of, key=lambda k: (-len(pages_of[k]), k)):
        if pairs <= budget:
            break
        size = len(pages_of[keyword])
        pairs -= size * (size - 1) // 2
        dropped.append(keyword)
    removed = set(dropped)
    groups = [
        np.array(sorted(pages), dtype=np.int64)
        for keyword, pages in pages_of.items()
        if keyword not in removed and len(pages) > 1
    ]
    if not groups:
        return ig.Graph(), (), tuple(dropped)
    base = int(max(page for pages in pages_of.values() for page in pages)) + 1
    codes = []
    for pages in groups:
        first, second = np.triu_indices(len(pages), k=1)
        codes.append(pages[first] * base + pages[second])
    pair_codes, shared = np.unique(np.concatenate(codes), return_counts=True)
    sources, targets = pair_codes // base, pair_codes % base
    vertices = sorted(set(sources.tolist()) | set(targets.tolist()))
    index = {vertex: i for i, vertex in enumerate(vertices)}
    graph = ig.Graph(
        n=len(vertices),
        edges=[
            (index[a], index[b]) for a, b in zip(sources.tolist(), targets.tolist(), strict=True)
        ],
    )
    graph.es["weight"] = shared.astype(np.float64).tolist()
    return graph, tuple(vertices), tuple(dropped)


def knn_graph(
    vectors: npt.NDArray[np.floating], k: int = KNN_NEIGHBOURS, *, chunk: int = 1024
) -> ig.Graph:
    """Each row joined to its ``k`` most cosine-similar rows. An edge found from both ends keeps
    the higher similarity as its weight; non-positive similarities are dropped."""
    count = len(vectors)
    if count < 2:
        return ig.Graph(n=count)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("a zero vector has no direction")
    unit = (vectors / norms).astype(np.float32)
    k = min(k, count - 1)
    sources, targets, weights = [], [], []
    for start in range(0, count, chunk):
        sims = unit[start : start + chunk] @ unit.T
        rows = np.arange(len(sims))
        sims[rows, start + rows] = -np.inf
        nearest = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        sources.append(np.repeat(start + rows, k))
        targets.append(nearest.ravel())
        weights.append(sims[rows[:, None], nearest].ravel())
    source, target, weight = map(np.concatenate, (sources, targets, weights))
    keep = weight > 0
    low, high, weight = (
        np.minimum(source, target)[keep],
        np.maximum(source, target)[keep],
        weight[keep],
    )
    order = np.argsort(-weight, kind="stable")
    codes = (low * count + high)[order]
    _, first = np.unique(codes, return_index=True)
    chosen = order[first]
    graph = ig.Graph(
        n=count, edges=list(zip(low[chosen].tolist(), high[chosen].tolist(), strict=True))
    )
    graph.es["weight"] = weight[chosen].astype(np.float64).tolist()
    return graph


def content_pass_graph(
    vertices: Sequence[int], vectors: npt.NDArray[np.floating], k: int = KNN_NEIGHBOURS
) -> tuple[ig.Graph, tuple[int, ...]]:
    """kNN graph over pages with a content vector; ``vectors`` rows align with ``vertices``."""
    if len(vertices) != len(vectors):
        raise ValueError("one vector per vertex")
    return _with_edges(knn_graph(vectors, k), vertices)


def seed_stability(
    graph: ig.Graph, *, weighted: bool, seeds: Sequence[int] = STABILITY_SEEDS
) -> tuple[float, float] | None:
    """Mean and minimum pairwise ARI of the seeded partition and one per extra seed."""
    if graph.vcount() < 2:
        return None
    runs = [leiden(graph, weighted=weighted, seed=seed)[0] for seed in (SEED, *seeds)]
    scores = [float(adjusted_rand_score(a, b)) for a, b in combinations(runs, 2)]
    return float(np.mean(scores)), min(scores)


def agreement(first: Mapping[int, int], second: Mapping[int, int]) -> float | None:
    """ARI of two labellings over the vertices both label; None below two shared vertices."""
    shared = sorted(first.keys() & second.keys())
    if len(shared) < 2:
        return None
    return float(adjusted_rand_score([first[v] for v in shared], [second[v] for v in shared]))


def disconnected_communities(graph: ig.Graph, membership: Sequence[int]) -> int:
    """Communities whose members do not form one connected subgraph; Leiden guarantees none."""
    members_of: dict[int, list[int]] = defaultdict(list)
    for vertex, community in enumerate(membership):
        members_of[community].append(vertex)
    return sum(
        1
        for members in members_of.values()
        if len(graph.induced_subgraph(members).connected_components()) > 1
    )


def pillars(
    found: Partition,
    vectors: Mapping[int, npt.NDArray[np.floating]],
    ranks: npt.NDArray[np.float64],
) -> frozenset[int]:
    """Per community of MIN_PILLAR_COMMUNITY or more pages, the member nearest the centroid of
    the members' unit content vectors. PageRank breaks ties and decides when none has a vector."""
    members_of: dict[int, list[int]] = defaultdict(list)
    for vertex, community in zip(found.vertices, found.membership, strict=True):
        members_of[community].append(vertex)
    chosen: set[int] = set()
    for members in members_of.values():
        if len(members) < MIN_PILLAR_COMMUNITY:
            continue
        embedded = [v for v in members if v in vectors]
        closeness: dict[int, float] = {}
        if embedded:
            unit = np.stack([vectors[v] / np.linalg.norm(vectors[v]) for v in embedded])
            scores = unit @ unit.mean(axis=0)
            closeness = {v: round(float(s), 9) for v, s in zip(embedded, scores, strict=True)}
        candidates = embedded or members
        chosen.add(max(candidates, key=lambda v: (closeness.get(v, 0.0), float(ranks[v]), -v)))
    return frozenset(chosen)


def link_states(graphs: LinkGraphs) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Per crawled page, aligned with ``graphs.crawled``: no body link in from another page,
    and no body link out to another page."""
    crawled = list(graphs.crawled)
    inbound = np.asarray(graphs.directed.degree(crawled, mode="in"))
    outbound = np.asarray(graphs.directed.degree(crawled, mode="out"))
    return inbound == 0, outbound == 0


def _with_edges(graph: ig.Graph, vertices: Sequence[int]) -> tuple[ig.Graph, tuple[int, ...]]:
    keep = [i for i, degree in enumerate(graph.degree()) if degree > 0]
    return graph.induced_subgraph(keep), tuple(vertices[i] for i in keep)


@dataclass(frozen=True, slots=True)
class Hubs:
    """HDBSCAN over unit vectors: one label per row, NOISE for rows in no dense region."""

    labels: tuple[int, ...]
    relative_validity: float | None
    # Per cluster, in label order.
    persistence: tuple[float, ...]


def find_hubs(vectors: npt.NDArray[np.floating]) -> Hubs:
    if len(vectors) <= HUB_MIN_CLUSTER_SIZE:
        return Hubs((NOISE,) * len(vectors), None, ())
    unit = _unit(vectors)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HUB_MIN_CLUSTER_SIZE,
        min_samples=HUB_MIN_SAMPLES,
        cluster_selection_method="eom",
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(unit)
    clustered = bool((labels != NOISE).any())
    return Hubs(
        labels=tuple(int(label) for label in labels),
        relative_validity=float(clusterer.relative_validity_) if clustered else None,
        persistence=tuple(float(p) for p in clusterer.cluster_persistence_),
    )


def hub_centroids(
    labels: Sequence[int], vectors: npt.NDArray[np.floating]
) -> dict[int, npt.NDArray[np.float64]]:
    """Unit mean of each cluster's unit vectors; noise has none."""
    unit = _unit(vectors)
    members: dict[int, list[int]] = defaultdict(list)
    for row, label in enumerate(labels):
        if label != NOISE:
            members[label].append(row)
    return {
        label: _unit(unit[rows].mean(axis=0, keepdims=True))[0] for label, rows in members.items()
    }


def match_hubs(
    previous: Mapping[int, npt.NDArray[np.floating]],
    current: Mapping[int, npt.NDArray[np.floating]],
    next_id: int,
    threshold: float = HUB_MATCH_SIMILARITY,
) -> dict[int, int]:
    """Stable id per current cluster: the most similar unclaimed previous hub at ``threshold``
    or above, greedily from the closest pair, otherwise a fresh id from ``next_id`` upwards."""
    pairs = sorted(
        (
            (float(np.dot(_unit(c[None, :])[0], _unit(p[None, :])[0])), label, hub)
            for label, c in current.items()
            for hub, p in previous.items()
        ),
        reverse=True,
    )
    ids: dict[int, int] = {}
    claimed: set[int] = set()
    for similarity, label, hub in pairs:
        if similarity < threshold:
            break
        if label not in ids and hub not in claimed:
            ids[label] = hub
            claimed.add(hub)
    for label in sorted(current):
        if label not in ids:
            ids[label] = next_id
            next_id += 1
    return ids


def _unit(vectors: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("a zero vector has no direction")
    return np.asarray(vectors / norms, dtype=np.float64)
