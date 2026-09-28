"""Candidate retrieval end to end on a real Neo4j, over a planted synthetic tenant.

Three topics of 25 pages share a common direction, so a page's own topic ranks first.
Within a topic p00 links to every page and each page to the next. Beside them: an orphan
NEW page with no inbound link, a page flagged not indexable that is a near duplicate of
t1/p05, a page with no flag whose vector is identical to t2/p03, an indexable page without
a vector, a 404, and a placeholder that still carries a vector. A second tenant holds the
same urls, vectors and links, plus one page of its own.

Synthetic topics separate cleanly by construction: this proves the constraints, the reads
and the accounting, not retrieval quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.models import CandidateSet, DuplicateGroup, Link, Page
from linking_engine.pipeline.duplicates import find_duplicates

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo

DIM = 16
TOPICS, PER_TOPIC = 3, 25
NEW = "example.com/t0/new"
NOINDEX = "example.com/t1/noindex"
ASSUMED = "example.com/t2/assumed"
NO_VECTOR = "example.com/t2/no-vector"
MISSING = "example.com/gone"
GHOST = "example.com/t0/ghost"
OTHER_ONLY = "example.com/t0/other-tenant-only"


def topic_url(topic: int, page: int) -> str:
    return f"example.com/t{topic}/p{page:02d}"


TOPIC_URLS = [topic_url(t, p) for t in range(TOPICS) for p in range(PER_TOPIC)]
POOL = sorted([*TOPIC_URLS, NEW, NOINDEX, ASSUMED])


def planted_vectors() -> dict[str, list[float]]:
    rng = np.random.default_rng(12)
    common = rng.normal(size=DIM)
    centres = common + 1.5 * rng.normal(size=(TOPICS, DIM))
    raw = {
        topic_url(t, p): centres[t] + 0.3 * rng.normal(size=DIM)
        for t in range(TOPICS)
        for p in range(PER_TOPIC)
    }
    raw[NEW] = centres[0] + 0.3 * rng.normal(size=DIM)
    raw[GHOST] = centres[0] + 0.3 * rng.normal(size=DIM)
    raw[NOINDEX] = raw[topic_url(1, 5)] + 0.01 * rng.normal(size=DIM)
    raw[ASSUMED] = raw[topic_url(2, 3)]
    return {url: (vector / np.linalg.norm(vector)).tolist() for url, vector in raw.items()}


def planted_links() -> list[tuple[str, str]]:
    hubs = [(topic_url(t, 0), topic_url(t, p)) for t in range(TOPICS) for p in range(1, PER_TOPIC)]
    chains = [
        (topic_url(t, p), topic_url(t, p + 1))
        for t in range(TOPICS)
        for p in range(1, PER_TOPIC - 1)
    ]
    return [*hubs, *chains, (NOINDEX, topic_url(1, 1))]


async def seed(graph: GraphRepo, tenant: str, *, own_page: bool = False) -> None:
    indexable = [*TOPIC_URLS, NEW, NO_VECTOR, *([OTHER_ONLY] if own_page else [])]
    await graph.upsert_pages(
        tenant,
        [Page(url=u, status_code=200, is_indexable=True) for u in indexable]
        + [
            Page(url=NOINDEX, status_code=200, is_indexable=False),
            Page(url=ASSUMED, status_code=200),
            Page(url=MISSING, status_code=404),
        ],
    )
    await graph.upsert_placeholders(tenant, [GHOST])
    links = [
        Link(source_url=s, target_url=t, position=i, anchor_text="x", surrounding_text="")
        for i, (s, t) in enumerate(planted_links())
    ]
    await graph.replace_links(tenant, sorted({s for s, _ in planted_links()}), links)
    vectors = planted_vectors()
    if own_page:
        vectors[OTHER_ONLY] = vectors[NEW]
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec",
        t=tenant,
        rows=[{"url": url, "vec": vec} for url, vec in vectors.items()],
    )


async def existing_links(graph: GraphRepo, tenant: str, pairs: list[tuple[str, str]]) -> int:
    rows = await graph._auto(
        "UNWIND $pairs AS pair "
        "MATCH (:Page {tenantId: $t, url: pair[0]})-[:LINKS_TO]->(:Page {tenantId: $t, url: pair[1]}) "
        "RETURN count(*) AS n",
        t=tenant,
        pairs=[list(pair) for pair in pairs],
    )
    return int(str(rows[0]["n"]))


def pairs_of(found: CandidateSet) -> list[tuple[str, str]]:
    return [(source, t.target_url) for t in found.targets for source in t.sources]


def nearest_similarities(target: str, per_target: int) -> list[float]:
    """The best eligible cosines of a target, in float64, from the planted truth."""
    vectors = {u: np.asarray(v) for u, v in planted_vectors().items() if u in POOL}
    linked = {s for s, t in planted_links() if t == target}
    scores = [float(vectors[u] @ vectors[target]) for u in POOL if u != target and u not in linked]
    return sorted(scores, reverse=True)[:per_target]


@pytest.mark.integration
async def test_retrieval_over_a_planted_tenant_honours_every_hard_constraint(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    alone = await retrieve_candidates(graph, tenant, per_target=5)
    await seed(graph, f"{tenant}-other", own_page=True)

    found = await retrieve_candidates(graph, tenant, per_target=5)

    report, targets = found.report, {t.target_url: t for t in found.targets}
    assert (report.crawled_pages, report.not_indexable, report.without_vector) == (80, 2, 1)
    assert (report.targets, report.indexable_assumed, report.source_pages) == (77, 1, 78)
    assert sorted(targets) == sorted([*TOPIC_URLS, NEW, ASSUMED])

    assert NEW in targets, "the planted orphan is not a target: something is still filtering"
    assert len(targets[NEW].sources) == 5
    assert all(s.startswith("example.com/t0/") for s in targets[NEW].sources)

    # Per-target cap only: every target is full and nothing is capped overall.
    assert {len(t.sources) for t in found.targets} == {5}
    assert report.candidates == 5 * 77

    pairs = pairs_of(found)
    assert await existing_links(graph, tenant, [(topic_url(0, 0), topic_url(0, 1))]) == 1
    assert await existing_links(graph, tenant, pairs) == 0, "a candidate pair is already linked"
    assert report.linked_nearer > 0, "no linked neighbour was among the nearest: vacuous check"
    assert all(s != t for s, t in pairs)

    assert not {NOINDEX, MISSING, NO_VECTOR, GHOST} & set(targets)
    assert NOINDEX in targets[topic_url(1, 5)].sources, "a non-indexable page can still link out"
    assert GHOST not in {s for s, _ in pairs}, "a placeholder is never a source"
    # Position, not score: a different page with the identical vector stays a source.
    assert targets[topic_url(2, 3)].sources[0] == ASSUMED
    assert targets[topic_url(2, 3)].similarities[0] > 0.999
    assert targets[ASSUMED].sources[0] == topic_url(2, 3)
    # Only source -> target links exclude: p00 links out to every source it gets.
    hub_sources = set(targets[topic_url(0, 0)].sources) - {NEW}
    assert hub_sources <= {topic_url(0, p) for p in range(1, PER_TOPIC)}
    assert len(hub_sources) >= 4

    for url, target in targets.items():
        linked = {s for s, t in planted_links() if t == url}
        assert (target.linked, target.eligible) == (len(linked), 78 - 1 - len(linked)), url
        assert target.similarities == pytest.approx(nearest_similarities(url, 5), abs=1e-5), url

    assert OTHER_ONLY not in {s for s, _ in pairs}, "another tenant's page became a source"
    assert found.targets == alone.targets, "another tenant's identical pages changed the result"
    unchanged = {"finished_at", "load_seconds", "search_seconds", "seconds"}
    assert found.report.model_dump(exclude=unchanged) == alone.report.model_dump(exclude=unchanged)


@pytest.mark.integration
async def test_the_gnn_index_reads_only_gnn_vectors(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant)

    before = await retrieve_candidates(graph, tenant, index="page_gnn")
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) WHERE p.url IN $urls "
        "SET p.gnn_embedding = [1.0, 0.0, 0.0, 0.0]",
        t=tenant,
        urls=[topic_url(0, 1), topic_url(0, 2), NOINDEX],
    )
    after = await retrieve_candidates(graph, tenant, index="page_gnn")

    assert (before.report.targets, before.report.source_pages) == (0, 0)
    assert (before.report.without_vector, before.report.drop_rate) == (78, None)
    assert (after.report.index, after.report.targets, after.report.source_pages) == (
        "page_gnn",
        2,
        3,
    )
    by_url = {t.target_url: t for t in after.targets}
    # p01 is linked from p00 only, which has no gnn vector; p02 is linked from p01.
    assert by_url[topic_url(0, 1)].sources == (topic_url(0, 2), NOINDEX)
    assert by_url[topic_url(0, 2)].sources == (NOINDEX,)
    assert (by_url[topic_url(0, 2)].linked, by_url[topic_url(0, 2)].eligible) == (1, 1)


# One article served at three urls with one body, beside ten pages of its topic. The blog url has
# the most inbound links, so it is the canonical copy.
ARTICLE = ("example.com/blog/article", "example.com/article", "example.com/news/article")
ARTICLE_TOPIC = [f"example.com/topic/p{i:02d}" for i in range(10)]
ARTICLE_LINKS = [
    (ARTICLE_TOPIC[0], ARTICLE[0]),
    (ARTICLE_TOPIC[1], ARTICLE[0]),
    (ARTICLE_TOPIC[2], ARTICLE[1]),
    (ARTICLE_TOPIC[3], ARTICLE_TOPIC[4]),
]


async def seed_article(graph: GraphRepo, tenant: str) -> None:
    rng = np.random.default_rng(72)
    centre = rng.normal(size=DIM)
    article = centre + 0.3 * rng.normal(size=DIM)
    vectors = {u: centre + 0.3 * rng.normal(size=DIM) for u in ARTICLE_TOPIC}
    vectors |= dict.fromkeys(ARTICLE, article)
    hashes = {u: f"{i:064x}" for i, u in enumerate(ARTICLE_TOPIC, start=1)}
    hashes |= dict.fromkeys(ARTICLE, "a" * 64)
    await graph.upsert_pages(
        tenant,
        [
            Page(url=u, status_code=200, body_hash=hashes[u], language="en", word_count=200)
            for u in [*ARTICLE_TOPIC, *ARTICLE]
        ],
    )
    await graph.replace_links(
        tenant,
        sorted({s for s, _ in ARTICLE_LINKS}),
        [
            Link(source_url=s, target_url=t, position=i, anchor_text="x", surrounding_text="")
            for i, (s, t) in enumerate(ARTICLE_LINKS)
        ],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec",
        t=tenant,
        rows=[{"url": u, "vec": (v / np.linalg.norm(v)).tolist()} for u, v in vectors.items()],
    )


def joins_two_copies(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(s, t) for s, t in pairs if s in ARTICLE and t in ARTICLE]


@pytest.mark.integration
async def test_one_article_at_three_urls_is_one_target_never_linked_from_its_copies(
    graph: GraphRepo, tenant: str
) -> None:
    await seed_article(graph, tenant)
    before = await retrieve_candidates(graph, tenant, per_target=5)
    assert joins_two_copies(pairs_of(before)), "the copies were never each other's candidates"

    duplicates = await find_duplicates(graph, tenant)
    found = await retrieve_candidates(graph, tenant, per_target=5)

    assert duplicates.groups == (
        DuplicateGroup(group_id=0, canonical=ARTICLE[0], copies=(ARTICLE[1], ARTICLE[2])),
    )
    pairs = pairs_of(found)
    assert joins_two_copies(pairs) == []
    targets = {t.target_url for t in found.targets}
    assert targets & set(ARTICLE) == {ARTICLE[0]}, "only the canonical copy is a target"
    assert {s for s, _ in pairs} & set(ARTICLE) == {ARTICLE[0]}, "copies are never sources"
    report = found.report
    assert (report.targets, report.source_pages, report.non_canonical_excluded) == (11, 11, 2)
    assert (before.report.targets, before.report.non_canonical_excluded) == (13, 0)
