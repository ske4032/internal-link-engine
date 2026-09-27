"""retrieve_candidates on hand-made vectors through a fake repo, and the report maths.

The fake returns what the three tenant-scoped reads return: the target selection, the
pool of vectors keyed by url in url order, and the tenant's link graph. Scoring is checked
against hand-computed cosines, and on random vectors against a float64 brute force.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from structlog.testing import capture_logs

from linking_engine.discovery.candidates import (
    PER_TARGET,
    TARGET_CHUNK,
    candidate_report,
    nearest_eligible,
    retrieve_candidates,
    summarise_candidates,
)
from linking_engine.errors import DatabaseReadError
from linking_engine.models import (
    CandidateReport,
    CandidateSet,
    CandidateTarget,
    LinkGraphSnapshot,
    TargetCandidates,
    TargetSelection,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.models import VectorIndex

TENANT = "acme"


def page(name: str) -> str:
    return f"example.com/{name}"


@dataclass
class FakeGraph:
    """The pool is every page with a vector; targets default to all of them."""

    vectors: dict[str, list[float]]
    links: list[tuple[str, str]] = field(default_factory=list)
    targets: list[str] | None = None
    assumed: frozenset[str] = frozenset()
    not_indexable: int = 0
    # Crawled pages without a vector and placeholders only appear in the link graph.
    no_vector: tuple[str, ...] = ()
    placeholders: tuple[str, ...] = ()
    reads: list[tuple[str, str, str | None]] = field(default_factory=list)

    async def candidate_targets(
        self, tenant_id: str, *, index: str = "page_content"
    ) -> TargetSelection:
        self.reads.append(("candidate_targets", tenant_id, index))
        targets = sorted(self.vectors) if self.targets is None else sorted(self.targets)
        return TargetSelection(
            crawled_pages=len(targets) + len(self.no_vector) + self.not_indexable,
            not_indexable=self.not_indexable,
            without_vector=len(self.no_vector),
            targets=tuple(
                CandidateTarget(url=u, indexable_assumed=u in self.assumed) for u in targets
            ),
        )

    async def page_vectors(
        self, tenant_id: str, *, index: str = "page_content", batch_size: int = 500
    ) -> dict[str, npt.NDArray[np.float32]]:
        self.reads.append(("page_vectors", tenant_id, index))
        return {u: np.asarray(self.vectors[u], dtype=np.float32) for u in sorted(self.vectors)}

    async def link_graph(self, tenant_id: str) -> LinkGraphSnapshot:
        self.reads.append(("link_graph", tenant_id, None))
        pages = sorted({*self.vectors, *self.no_vector, *self.placeholders})
        return LinkGraphSnapshot(
            tenant_id=tenant_id,
            pages=tuple(pages),
            placeholders=tuple(p in self.placeholders for p in pages),
            links=tuple(self.links),
        )


async def retrieve(graph: FakeGraph, tenant_id: str = TENANT, **options: Any) -> CandidateSet:
    return await retrieve_candidates(cast("GraphRepo", graph), tenant_id, **options)


def by_url(found: CandidateSet) -> dict[str, TargetCandidates]:
    return {t.target_url: t for t in found.targets}


def brute_force(
    vectors: dict[str, list[float]], links: list[tuple[str, str]], target: str, per_target: int
) -> list[tuple[str, float]]:
    """The expected candidates, in float64, straight from the definition."""
    unit = {u: np.asarray(v) / np.linalg.norm(v) for u, v in vectors.items()}
    linked = {s for s, t in links if t == target}
    ranked = sorted(
        ((u, float(unit[u] @ unit[target])) for u in vectors if u != target and u not in linked),
        key=lambda hit: (-hit[1], hit[0]),
    )
    return ranked[:per_target]


def random_tenant(pages: int, seed: int) -> tuple[dict[str, list[float]], list[tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    urls = [page(f"r{i:02d}") for i in range(pages)]
    vectors = {u: rng.normal(size=8).tolist() for u in urls}
    links = [
        (urls[i], urls[j])
        for i in range(pages)
        for j in range(pages)
        if (i * 7 + j) % 5 == 0 and i != j
    ]
    return vectors, links


def test_the_defaults_are_the_contracts() -> None:
    assert (PER_TARGET, TARGET_CHUNK) == (50, 512)


# ── scoring ─────────────────────────────────────────────────────────────────


async def test_sources_rank_by_cosine_whatever_their_length_and_ties_by_url() -> None:
    t = page("t")
    graph = FakeGraph(
        {
            t: [1, 0, 0],
            page("b"): [1, 1, 0],
            page("a"): [1, 1, 0],
            page("c"): [0, 5, 0],
            page("d"): [-2, 0, 0],
            page("e"): [4, 0, 3],
        },
        targets=[t],
    )

    [found] = (await retrieve(graph)).targets

    assert found.sources == (page("e"), page("a"), page("b"), page("c"), page("d"))
    assert found.similarities == pytest.approx((0.8, 0.5**0.5, 0.5**0.5, 0.0, -1.0), abs=1e-5)
    assert (found.eligible, found.linked, found.linked_nearer) == (5, 0, 0)


async def test_the_target_is_excluded_by_position_even_beside_an_identical_vector() -> None:
    t, twin = page("t"), page("twin")
    graph = FakeGraph({t: [0.3, 0.4, 0.5], twin: [0.3, 0.4, 0.5], page("x"): [0, 0, 1]})

    targets = by_url(await retrieve(graph))

    assert targets[t].sources[0] == twin
    assert targets[t].similarities[0] == pytest.approx(1.0, abs=1e-5)
    assert targets[twin].sources[0] == t
    assert t not in targets[t].sources
    assert targets[t].eligible == 2


async def test_similarities_are_clamped_to_the_cosine_range() -> None:
    """Float32 self-products of unit vectors land a hair outside [-1, 1] for some vectors;
    the model refuses those, so unclamped scoring fails this run outright."""
    rng = np.random.default_rng(3)
    vectors: dict[str, list[float]] = {}
    for i, vector in enumerate(rng.normal(size=(40, 7)).tolist()):
        vectors[page(f"v{i:02d}")] = vector
        vectors[page(f"v{i:02d}-twin")] = vector
        vectors[page(f"v{i:02d}-anti")] = [-x for x in vector]
    graph = FakeGraph(vectors, targets=[page(f"v{i:02d}") for i in range(40)])

    found = await retrieve(graph, per_target=2)

    for target in found.targets:
        assert target.sources[0] == f"{target.target_url}-twin"
        assert target.similarities[0] == pytest.approx(1.0, abs=1e-5)
    for i in range(40):
        v, anti = page(f"v{i:02d}"), page(f"v{i:02d}-anti")
        pair = FakeGraph({v: vectors[v], anti: vectors[anti]}, targets=[v])
        [target] = (await retrieve(pair)).targets
        assert target.similarities == pytest.approx((-1.0,), abs=1e-5)


async def test_a_zero_vector_scores_zero_against_everything() -> None:
    zero = page("zero")
    graph = FakeGraph({zero: [0, 0, 0], page("a"): [1, 0, 0], page("b"): [0, 1, 0]})

    targets = by_url(await retrieve(graph))

    assert targets[zero].sources == (page("a"), page("b"))
    assert targets[zero].similarities == (0.0, 0.0)
    assert (
        dict(zip(targets[page("a")].sources, targets[page("a")].similarities, strict=True))[zero]
        == 0.0
    )


async def test_linked_sources_are_masked_and_reverse_links_are_not() -> None:
    t = page("t")
    graph = FakeGraph(
        {
            t: [1, 0, 0],
            page("s1"): [0.9, 0.1, 0],
            page("s2"): [0.8, 0.2, 0],
            page("s3"): [0.7, 0.3, 0],
            page("s4"): [0.6, 0.4, 0],
        },
        links=[
            (page("s1"), t),
            (t, page("s2")),
            (page("s3"), t),
            (page("s3"), t),
            (t, t),
            (page("novec"), t),
            (t, page("ghost")),
        ],
        no_vector=(page("novec"),),
        placeholders=(page("ghost"),),
    )

    targets = by_url(await retrieve(graph))

    # s3 links in twice and counts once; the self-link is only the self exclusion; novec
    # links in but is not in the pool.
    assert targets[t].sources == (page("s2"), page("s4"))
    assert (targets[t].eligible, targets[t].linked, targets[t].linked_nearer) == (2, 2, 2)
    # The reverse direction: t links to s2, so t is masked from s2's sources only.
    assert t not in targets[page("s2")].sources
    assert page("s2") in targets[t].sources
    assert (targets[page("s2")].eligible, targets[page("s2")].linked) == (3, 1)


# Linked pages L at 0.95, 0.85, 0.7, 0.5 interleaved with eligible e at 0.9, 0.8, 0.6.
RANKED = {
    "L1": 0.95,
    "e1": 0.9,
    "L2": 0.85,
    "e2": 0.8,
    "L3": 0.7,
    "e3": 0.6,
    "L4": 0.5,
}


def ranked_tenant() -> FakeGraph:
    t = page("t")
    vectors = {t: [1.0, 0.0]}
    vectors |= {page(name): [cos, (1 - cos**2) ** 0.5] for name, cos in RANKED.items()}
    return FakeGraph(
        vectors, links=[(page(n), t) for n in RANKED if n.startswith("L")], targets=[t]
    )


@pytest.mark.parametrize(
    ("per_target", "kept", "linked_nearer"),
    [
        pytest.param(2, ["e1", "e2"], 2, id="full-L3-and-L4-below-the-last-kept"),
        pytest.param(3, ["e1", "e2", "e3"], 3, id="exactly-full-L4-below"),
        pytest.param(5, ["e1", "e2", "e3"], 4, id="short-every-linked-counts"),
        pytest.param(1, ["e1"], 1, id="full-only-L1-above"),
    ],
)
async def test_linked_nearer_counts_linked_pages_that_would_have_been_kept(
    per_target: int, kept: list[str], linked_nearer: int
) -> None:
    [found] = (await retrieve(ranked_tenant(), per_target=per_target)).targets

    assert found.sources == tuple(page(n) for n in kept)
    assert found.similarities == pytest.approx([RANKED[n] for n in kept], abs=1e-5)
    assert (found.eligible, found.linked, found.linked_nearer) == (3, 4, linked_nearer)


async def test_a_linked_page_tied_with_the_last_kept_counts_only_if_its_url_ranks_first() -> None:
    t = page("t")
    graph = FakeGraph(
        {
            t: [1, 0],
            page("k-linked"): [1, 1],
            page("m-kept"): [1, 1],
            page("z-linked"): [1, 1],
            page("a-low"): [0, 1],
        },
        links=[(page("k-linked"), t), (page("z-linked"), t)],
        targets=[t],
    )

    [found] = (await retrieve(graph, per_target=1)).targets

    assert found.sources == (page("m-kept"),)
    assert (found.eligible, found.linked, found.linked_nearer) == (2, 2, 1)


async def test_every_target_keeps_its_own_top_n_with_no_global_cap() -> None:
    vectors, links = random_tenant(12, seed=5)
    graph = FakeGraph(vectors, links=links)

    found = await retrieve(graph, per_target=5)

    assert found.report.candidates == 12 * 5, "a global cap would drop whole targets"
    for target in found.targets:
        expected = brute_force(vectors, links, target.target_url, 5)
        assert list(target.sources) == [u for u, _ in expected], target.target_url
        assert target.similarities == pytest.approx([s for _, s in expected], abs=1e-5)
        linked = {s for s, t in links if t == target.target_url}
        assert (target.linked, target.eligible) == (len(linked), 11 - len(linked))


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 1000])
async def test_the_chunk_size_never_changes_the_candidates(chunk_size: int) -> None:
    vectors, links = random_tenant(12, seed=9)
    reference = await retrieve(FakeGraph(vectors, links=links), per_target=4)

    found = await retrieve(FakeGraph(vectors, links=links), per_target=4, chunk_size=chunk_size)

    assert found.report.chunk_size == chunk_size
    for before, after in zip(reference.targets, found.targets, strict=True):
        assert (before.target_url, before.sources) == (after.target_url, after.sources)
        assert (before.eligible, before.linked, before.linked_nearer) == (
            after.eligible,
            after.linked,
            after.linked_nearer,
        )
        assert before.similarities == pytest.approx(after.similarities, abs=1e-5)


# ── retrieve_candidates ─────────────────────────────────────────────────────


async def test_only_targets_are_scored_but_every_pool_page_is_a_source() -> None:
    graph = FakeGraph(
        {page("t"): [1, 0], page("noindex"): [1, 0.1], page("other"): [0, 1]},
        targets=[page("t")],
        not_indexable=1,
    )

    found = await retrieve(graph)

    assert [t.target_url for t in found.targets] == [page("t")]
    assert found.targets[0].sources == (page("noindex"), page("other"))
    assert (found.report.targets, found.report.source_pages) == (1, 3)


async def test_targets_come_back_in_url_order() -> None:
    urls = [page(f"p{i:02d}") for i in range(9)]
    rng = np.random.default_rng(1)
    graph = FakeGraph({u: rng.normal(size=4).tolist() for u in reversed(urls)})

    found = await retrieve(graph, chunk_size=2)

    assert [t.target_url for t in found.targets] == urls


async def test_the_tenant_and_index_reach_every_read() -> None:
    graph = FakeGraph({page("a"): [1, 0], page("b"): [0, 1]})

    found = await retrieve(graph, index="page_gnn")

    assert sorted(graph.reads) == [
        ("candidate_targets", TENANT, "page_gnn"),
        ("link_graph", TENANT, None),
        ("page_vectors", TENANT, "page_gnn"),
    ]
    assert found.report.index == "page_gnn"


@pytest.mark.parametrize(
    ("tenant_id", "options", "message"),
    [
        pytest.param(" ", {}, "tenant_id", id="blank-tenant"),
        pytest.param(TENANT, {"per_target": 0}, "per_target", id="zero-cap"),
        pytest.param(TENANT, {"chunk_size": 0}, "chunk_size", id="zero-chunk"),
    ],
)
async def test_invalid_arguments_are_rejected_before_any_read(
    tenant_id: str, options: dict[str, int], message: str
) -> None:
    graph = FakeGraph({page("a"): [1, 0]})

    with pytest.raises(ValueError, match=message):
        await retrieve(graph, tenant_id, **options)

    assert graph.reads == []


async def test_a_target_missing_from_the_pool_fails_the_read() -> None:
    """The reads are separate transactions: a vector removed in between must not pass."""
    graph = FakeGraph({page("a"): [1, 0], page("b"): [0, 1]}, targets=[page("a"), page("gone")])

    with pytest.raises(DatabaseReadError):
        await retrieve(graph)


@pytest.mark.parametrize(
    "vectors",
    [
        pytest.param({page("a"): [1, 0], page("b"): [0, 1, 0]}, id="mixed-lengths"),
        pytest.param({page("a"): [[1, 0]], page("b"): [[0, 1]]}, id="not-flat"),
        pytest.param({page("a"): [], page("b"): []}, id="empty"),
        pytest.param({page("a"): [1, 0], page("b"): [float("nan"), 1]}, id="nan"),
    ],
)
async def test_malformed_vectors_fail_the_read(vectors: dict[str, list[Any]]) -> None:
    with pytest.raises(DatabaseReadError):
        await retrieve(FakeGraph(vectors))


async def test_a_single_page_tenant_has_one_empty_target() -> None:
    found = await retrieve(FakeGraph({page("a"): [1, 0]}))

    [alone] = found.targets
    assert (alone.sources, alone.eligible, alone.linked) == ((), 0, 0)
    assert (found.report.empty_targets, found.report.drop_rate) == (1, None)


@pytest.mark.parametrize(
    ("urls", "vectors", "targets", "options", "message"),
    [
        pytest.param(["a"], [[1.0]], ["a"], {"per_target": 0}, "per_target", id="zero-cap"),
        pytest.param(["a"], [[1.0]], ["a"], {"chunk_size": 0}, "chunk_size", id="zero-chunk"),
        pytest.param(["a", "b"], [[1.0]], ["a"], {}, "row per url", id="row-count"),
        pytest.param(["a"], [1.0], ["a"], {}, "row per url", id="not-a-matrix"),
        pytest.param(["a", "b"], [[], []], ["a"], {}, "non-empty row", id="empty-rows"),
        pytest.param(["b", "a"], [[1.0], [0.0]], ["a"], {}, "ascending", id="unsorted"),
        pytest.param(["a", "a"], [[1.0], [0.0]], ["a"], {}, "unique", id="duplicate"),
        pytest.param(["a"], [[1.0]], ["z"], {}, "not in the pool", id="stray-target"),
    ],
)
def test_the_scorer_refuses_inconsistent_input(
    urls: list[str],
    vectors: list[Any],
    targets: list[str],
    options: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        nearest_eligible(urls, vectors, targets, [], **options)


@pytest.mark.parametrize(
    "vectors",
    [
        pytest.param({}, id="nothing-crawled"),
        pytest.param({page("a"): [1.0, 0.0]}, id="no-targets"),
    ],
)
async def test_a_tenant_without_targets_gets_a_valid_empty_report(
    vectors: dict[str, list[float]],
) -> None:
    graph = FakeGraph(vectors, targets=[], not_indexable=len(vectors), no_vector=(page("nv"),))

    found = await retrieve(graph)

    report = found.report
    assert found.targets == ()
    assert (report.targets, report.candidates, report.linked_pairs, report.linked_nearer) == (
        0,
        0,
        0,
        0,
    )
    assert (report.source_pages, report.without_vector) == (len(vectors), 1)
    assert (report.min_per_target, report.median_per_target, report.max_per_target) == (
        None,
        None,
        None,
    )
    assert report.drop_rate is None
    assert "None" not in summarise_candidates(report)


async def test_the_run_report_adds_up_its_targets() -> None:
    graph = ranked_tenant()
    graph.targets = None
    graph.assumed = frozenset({page("t")})

    found = await retrieve(graph, per_target=2)

    report = found.report
    assert (report.targets, report.source_pages, report.indexable_assumed) == (8, 8, 1)
    assert report.candidates == sum(len(t.sources) for t in found.targets)
    assert report.linked_pairs == sum(t.linked for t in found.targets) == 4
    assert report.linked_nearer == sum(t.linked_nearer for t in found.targets)
    kept_or_nearer = report.candidates + report.linked_nearer
    assert report.drop_rate == pytest.approx(report.linked_nearer / kept_or_nearer)
    assert 0 <= report.load_seconds <= report.seconds
    assert 0 <= report.search_seconds <= report.seconds


async def test_one_log_line_carries_the_report_and_no_urls() -> None:
    graph = ranked_tenant()

    with capture_logs() as logs:
        found = await retrieve(graph, per_target=2)

    assert len(logs) == 1, logs
    [line] = logs
    assert line["event"] == "candidates.retrieved"
    assert (line["tenant_id"], line["candidates"]) == (TENANT, found.report.candidates)
    urls = set(graph.vectors) | {s for t in found.targets for s in t.sources}
    logged = " ".join(str(value) for value in line.values())
    assert not [url for url in urls if url in logged], "no urls in logs"


# ── candidate_report and summarise_candidates ───────────────────────────────


def kept(
    url: str, sources: int, *, eligible: int | None = None, linked: int = 0, nearer: int = 0
) -> TargetCandidates:
    return TargetCandidates(
        target_url=url,
        sources=tuple(page(f"s{i}") for i in range(sources)),
        similarities=tuple(0.9 - i * 0.1 for i in range(sources)),
        eligible=sources if eligible is None else eligible,
        linked=linked,
        linked_nearer=nearer,
    )


def selection_of(*targets: TargetCandidates, assumed: int = 0) -> TargetSelection:
    return TargetSelection(
        crawled_pages=len(targets) + 4,
        not_indexable=3,
        without_vector=1,
        targets=tuple(
            CandidateTarget(url=t.target_url, indexable_assumed=i < assumed)
            for i, t in enumerate(targets)
        ),
    )


def report_of(
    *targets: TargetCandidates,
    per_target: int = 2,
    index: VectorIndex = "page_content",
    assumed: int = 0,
) -> CandidateReport:
    return candidate_report(
        TENANT,
        index,
        per_target,
        64,
        selection_of(*targets, assumed=assumed),
        len(targets) + 2,
        targets,
        load_seconds=0.75,
        search_seconds=0.5,
        seconds=1.5,
    )


THREE = (
    kept(page("a"), 2, eligible=9, linked=3, nearer=1),
    kept(page("b"), 1, linked=1, nearer=1),
    kept(page("c"), 0),
)


def test_the_report_adds_up_the_targets() -> None:
    before = datetime.now(UTC)

    report = report_of(*THREE, assumed=2)

    assert (report.crawled_pages, report.not_indexable, report.without_vector) == (7, 3, 1)
    assert (report.targets, report.indexable_assumed, report.source_pages) == (3, 2, 5)
    assert (report.per_target, report.chunk_size, report.candidates) == (2, 64, 3)
    assert (report.full_targets, report.short_targets, report.empty_targets) == (1, 1, 1)
    assert (report.min_per_target, report.median_per_target, report.max_per_target) == (0, 1.0, 2)
    assert (report.linked_pairs, report.linked_nearer) == (4, 2)
    # Linked pages that would have been kept, over those plus the kept candidates.
    assert report.drop_rate == pytest.approx(2 / (3 + 2))
    assert (report.load_seconds, report.search_seconds, report.seconds) == (0.75, 0.5, 1.5)
    assert before <= report.finished_at <= datetime.now(UTC) + timedelta(seconds=1)


def test_the_median_of_an_even_count_is_the_mean_of_the_middle_two() -> None:
    report = report_of(*(kept(page(n), c) for n, c in (("a", 2), ("b", 2), ("c", 1), ("d", 0))))
    assert report.median_per_target == 1.5


def test_links_that_would_not_have_been_kept_do_not_raise_the_drop_rate() -> None:
    report = report_of(kept(page("a"), 2, linked=5, nearer=0))
    assert (report.linked_pairs, report.drop_rate) == (5, 0.0)


def test_no_kept_or_nearer_links_leaves_the_drop_rate_undefined() -> None:
    report = report_of(kept(page("a"), 0, linked=2, nearer=0))

    assert (report.empty_targets, report.drop_rate) == (1, None)
    assert "None" not in summarise_candidates(report)


def test_the_summary_states_what_ran_and_what_it_found() -> None:
    summary = summarise_candidates(report_of(*THREE, index="page_gnn", assumed=2))

    for fact in (TENANT, "page_gnn", "3 candidates", "40.0%", "1.5"):
        assert fact in summary, f"{fact!r} missing from:\n{summary}"
    assert "example.com" not in summary


def test_targets_that_differ_from_the_selection_are_refused() -> None:
    a, b = kept(page("a"), 1), kept(page("b"), 1)
    with pytest.raises(ValueError, match="exactly the selection's targets"):
        candidate_report(
            TENANT,
            "page_content",
            2,
            64,
            selection_of(a, b),
            2,
            (a,),
            load_seconds=0.0,
            search_seconds=0.0,
            seconds=0.0,
        )
