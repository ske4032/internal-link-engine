"""Hub bridges: the floor, the hub pair scores, the covered pairs and their reasons, the ranked
bridge links with their exclusions, and the components before and after."""

from __future__ import annotations

import random
from datetime import UTC, datetime

import pytest

from linking_engine.discovery.bridges import (
    ALTERNATIVES,
    DENSITY_WEIGHT,
    FLOOR_SHARE,
    NEAREST_HUBS,
    SHARED_QUERIES,
    TOP_GAP_PAIRS,
    bridge_links,
    bridge_report,
    components,
    covered_pairs,
    floor_pages,
    hub_pairs,
    spanning_tree,
    summarise_bridges,
)
from linking_engine.models import (
    BridgeLink,
    BridgeReason,
    BridgeReport,
    HubPair,
    KeywordRung,
    PageStructure,
)

TREE, NEAREST, GAP = BridgeReason.SPANNING_TREE, BridgeReason.NEAREST_HUB, BridgeReason.BRIDGE_GAP


def url(name: str) -> str:
    return f"example.com/b/{name}"


def page(
    name: str,
    hub: int | None,
    *,
    language: str | None = "en",
    pr: float | None = None,
    outbound: int = 0,
) -> PageStructure:
    return PageStructure(
        url=url(name),
        language=language,
        inbound=0,
        outbound=outbound,
        page_rank_percentile=pr,
        hub_id=hub,
    )


def pair(
    a: int,
    b: int,
    cosine: float,
    gap: float = 0.0,
    *,
    language: str | None = "en",
    size_a: int = 10,
    size_b: int = 10,
    pages_ab: int = 0,
    pages_ba: int = 0,
    reasons: tuple[BridgeReason, ...] = (),
) -> HubPair:
    return HubPair(
        language=language,
        hub_a=a,
        hub_b=b,
        size_a=size_a,
        size_b=size_b,
        pages_ab=pages_ab,
        pages_ba=pages_ba,
        link_density=0.0,
        centroid_cosine=cosine,
        bridge_gap=gap,
        reasons=reasons,
    )


def link(u: str, v: str) -> tuple[str, str]:
    return url(u), url(v)


def test_the_constants_are_the_contracts() -> None:
    assert (FLOOR_SHARE, NEAREST_HUBS, TOP_GAP_PAIRS, ALTERNATIVES, DENSITY_WEIGHT) == (
        0.05,
        2,
        3,
        2,
        50.0,
    )
    assert SHARED_QUERIES == 10


# ── the floor ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("size", "pages"),
    [(1, 1), (20, 1), (21, 2), (40, 2), (41, 3), (60, 3), (61, 4), (100, 5), (101, 6)],
)
def test_the_floor_is_five_percent_of_the_hub_rounded_up_with_at_least_one_page(
    size: int, pages: int
) -> None:
    """60 is the float trap: 0.05 * 60 is 3.0000000000000004, which ceil makes 4."""
    assert floor_pages(size) == pages


def test_an_empty_hub_has_no_floor() -> None:
    with pytest.raises(ValueError, match="at least one page"):
        floor_pages(0)


# ── hub_pairs ───────────────────────────────────────────────────────────────

CENTROIDS = {0: [1.0, 0.0], 1: [0.6, 0.8], 2: [0.0, 1.0]}


def two_hubs() -> list[PageStructure]:
    return [
        *(page(f"a{i}", 0) for i in range(4)),
        page("b0", 1),
        page("b1", 1),
        page("noise", -1),
        page("loose", None),
        page("b-de", 1, language="de"),
    ]


def test_a_pair_counts_linking_pages_and_distinct_links_between_the_hubs() -> None:
    links = [
        link("a0", "b0"),
        link("a0", "b0"),
        link("a0", "b1"),
        link("a1", "b0"),
        link("b0", "a0"),
        link("a0", "a1"),
        link("a0", "noise"),
        link("loose", "b0"),
        link("a2", "b-de"),
    ]

    [found] = hub_pairs(two_hubs(), CENTROIDS, links)

    assert (found.language, found.hub_a, found.hub_b, found.size_a, found.size_b) == (
        "en",
        0,
        1,
        4,
        2,
    )
    # a0 and a1 link into hub 1, b0 into hub 0; four distinct page pairs over 4 x 2.
    assert (found.pages_ab, found.pages_ba) == (2, 1)
    assert found.link_density == pytest.approx(4 / 8)
    assert found.centroid_cosine == pytest.approx(0.6)
    assert found.query_jaccard is None
    assert found.bridge_gap == pytest.approx(0.6 - 50 * 0.5)
    assert (found.shared_queries, found.reasons) == ((), ())


@pytest.mark.parametrize(
    ("queries", "jaccard", "gap"),
    [
        pytest.param(None, None, 0.6, id="no-gsc-cosine-takes-both-weights"),
        pytest.param(
            {url("a0"): frozenset({"x", "y"}), url("b0"): frozenset({"x"})},
            0.5,
            0.4 * 0.6 + 0.6 * 0.5,
            id="gsc",
        ),
        pytest.param({}, 0.0, 0.4 * 0.6, id="gsc-but-no-queries-on-either-hub"),
    ],
)
def test_the_query_overlap_only_weighs_in_with_gsc_data(
    queries: dict[str, frozenset[str]] | None, jaccard: float | None, gap: float
) -> None:
    pages = [page("a0", 0), page("b0", 1)]

    [found] = hub_pairs(pages, CENTROIDS, [], queries)

    assert found.query_jaccard == (None if jaccard is None else pytest.approx(jaccard))
    assert found.bridge_gap == pytest.approx(gap)


def test_shared_queries_are_the_ten_on_most_pages_then_by_text() -> None:
    pages = [page("a0", 0), page("a1", 0), page("a2", 0), page("b0", 1), page("b1", 1)]
    filler = {f"q{i:02d}" for i in range(1, 11)}
    queries = {
        url("a0"): frozenset({"alpha", "bravo", "charlie", "only-a", *filler}),
        url("a1"): frozenset({"alpha"}),
        url("a2"): frozenset({"alpha"}),
        url("b0"): frozenset({"alpha", "bravo", "charlie", "only-b", *filler}),
        url("b1"): frozenset({"alpha", "charlie"}),
    }

    [found] = hub_pairs(pages, CENTROIDS, [], queries)

    # alpha on 5 pages, charlie on 3, then every query on 2 pages by text, cut at ten.
    assert found.shared_queries == (
        "alpha",
        "charlie",
        "bravo",
        *(f"q{i:02d}" for i in range(1, 8)),
    )
    assert found.query_jaccard == pytest.approx(13 / 15)


def test_each_language_is_its_own_hub_graph() -> None:
    pages = [
        page("a0", 0),
        page("b0", 1),
        page("a-de", 0, language="de"),
        page("c-de", 2, language="de"),
        page("a-none", 0, language=None),
        page("b-none", 1, language=None),
        page("b-fr", 1, language="fr"),
    ]
    links = [link("a0", "c-de"), link("a-de", "c-de")]

    found = hub_pairs(pages, CENTROIDS, links)

    # One hub alone in French makes no pair; a link across languages counts for neither.
    assert [(p.language, p.hub_a, p.hub_b) for p in found] == [
        (None, 0, 1),
        ("de", 0, 2),
        ("en", 0, 1),
    ]
    de = found[1]
    assert (de.size_a, de.size_b, de.pages_ab, de.pages_ba) == (1, 1, 1, 0)
    assert found[2].pages_ab == 0


@pytest.mark.parametrize(
    ("pages", "centroids", "message"),
    [
        pytest.param([page("a0", 0), page("c0", 3)], CENTROIDS, "no stored centroid", id="missing"),
        pytest.param(
            [page("a0", 0), page("b0", 1)],
            {0: [1.0, 0.0], 1: [1.0, 0.0, 0.0]},
            "differ in length",
            id="mixed-lengths",
        ),
        pytest.param([page("a0", 0), page("a0", 1)], CENTROIDS, "duplicate page urls", id="dupes"),
    ],
)
def test_inconsistent_inputs_are_refused(
    pages: list[PageStructure], centroids: dict[int, list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        hub_pairs(pages, centroids, [])


# ── spanning_tree ───────────────────────────────────────────────────────────

# Five hubs: the tree, nearest hubs and widest gaps each pick different pairs.
FIVE = [
    pair(0, 1, 0.9, -1.0),
    pair(0, 2, 0.8, -2.0),
    pair(0, 3, 0.1, 0.5),
    pair(0, 4, 0.2, -3.0),
    pair(1, 2, 0.7, -4.0),
    pair(1, 3, 0.15, 0.4),
    pair(1, 4, 0.05, -5.0),
    pair(2, 3, 0.3, -6.0),
    pair(2, 4, 0.6, 0.3),
    pair(3, 4, 0.25, -7.0),
]


def connected(edges: set[tuple[int, int]], hubs: set[int]) -> bool:
    reached, frontier = {min(hubs)}, [min(hubs)]
    while frontier:
        hub = frontier.pop()
        for a, b in edges:
            other = b if a == hub else a if b == hub else None
            if other is not None and other not in reached:
                reached.add(other)
                frontier.append(other)
    return reached == hubs


def test_the_maximum_spanning_tree_connects_every_hub_over_centroid_cosine() -> None:
    tree = spanning_tree(FIVE)

    assert tree == {(0, 1), (0, 2), (2, 4), (2, 3)}
    assert connected(tree, {0, 1, 2, 3, 4})


def test_equal_cosines_go_to_the_lower_hub_ids() -> None:
    assert spanning_tree([pair(1, 2, 0.5), pair(0, 2, 0.5), pair(0, 1, 0.5)]) == {(0, 1), (0, 2)}


def test_a_spanning_tree_is_per_language() -> None:
    assert spanning_tree([]) == set()
    with pytest.raises(ValueError, match="one language"):
        spanning_tree([pair(0, 1, 0.5), pair(0, 1, 0.5, language="de")])


# ── covered_pairs ───────────────────────────────────────────────────────────


def test_covered_pairs_merge_tree_nearest_and_gap_reasons() -> None:
    covered = {(p.hub_a, p.hub_b): p.reasons for p in covered_pairs(random.sample(FIVE, 10))}

    assert covered == {
        (0, 1): (TREE, NEAREST),
        (0, 2): (TREE, NEAREST),
        (0, 3): (GAP,),
        (0, 4): (),
        (1, 2): (NEAREST,),
        (1, 3): (GAP,),
        (1, 4): (),
        (2, 3): (TREE, NEAREST),
        (2, 4): (TREE, NEAREST, GAP),
        (3, 4): (NEAREST,),
    }


def test_without_nearest_or_gap_pairs_only_the_tree_is_covered() -> None:
    covered = covered_pairs(FIVE, nearest=0, top_gap=0)
    assert {(p.hub_a, p.hub_b) for p in covered if p.reasons} == {(0, 1), (0, 2), (2, 4), (2, 3)}
    assert {p.reasons for p in covered if p.reasons} == {(TREE,)}
    with pytest.raises(ValueError, match="cannot be negative"):
        covered_pairs(FIVE, nearest=-1)


def test_each_language_is_covered_on_its_own() -> None:
    covered = covered_pairs([*FIVE, pair(0, 1, -0.2, -9.0, language="de")])

    de = [p for p in covered if p.language == "de"]
    assert [(p.hub_a, p.hub_b, p.reasons) for p in de] == [(0, 1, (TREE, NEAREST, GAP))]
    assert [p.language for p in covered] == ["de"] + ["en"] * 10


# ── bridge_links ────────────────────────────────────────────────────────────


def hubs_with(
    sources: list[PageStructure], targets: list[PageStructure], links: list[tuple[str, str]]
) -> list[HubPair]:
    pages = [*sources, *targets]
    return covered_pairs(hub_pairs(pages, {0: [1.0, 0.0], 1: [0.0, 1.0]}, links))


def filler(count: int) -> list[PageStructure]:
    return [page(f"f{i:02d}", 0, pr=0.1, outbound=9) for i in range(count)]


def along(cosine: float) -> list[float]:
    """A unit vector at ``cosine`` to [1, 0]."""
    return [cosine, (1 - cosine**2) ** 0.5]


def test_sources_go_by_relevance_to_the_other_hub_before_authority() -> None:
    # 41 pages: the floor from hub 0 is three pages; hub 1's only target sits at [1, 0].
    sources = [
        page("s-relevant-low-rank", 0, pr=0.1),
        page("s-close-high-rank", 0, pr=0.9),
        page("s-less-relevant-top-rank", 0, pr=0.99),
        page("s-far-high-rank", 0, pr=0.95),
        *filler(37),
    ]
    targets = [page("t0", 1)]
    vectors = {p.url: along(0.0) for p in sources}
    vectors |= {
        url("s-relevant-low-rank"): along(1.0),
        url("s-close-high-rank"): along(0.996),
        url("s-less-relevant-top-rank"): along(0.98),
        url("s-far-high-rank"): along(0.6),
        url("t0"): along(1.0),
    }
    pairs = hubs_with(sources, targets, [])

    found = bridge_links(pairs, [*sources, *targets], [], vectors, [url("t0")], [], {})

    firsts = [(b.slot, b.source_url) for b in found if b.hub_from == 0 and b.rank == 1]
    # 1.0 and 0.996 both round to 1.00, so PageRank decides between them; 0.98 comes next
    # despite the highest PageRank, and 0.6 never makes it.
    assert firsts == [
        (1, url("s-close-high-rank")),
        (2, url("s-relevant-low-rank")),
        (3, url("s-less-relevant-top-rank")),
    ]


def test_equally_relevant_sources_go_by_authority_then_fewest_outbound_links_then_url() -> None:
    sources = [
        page("s-many", 0, pr=0.5, outbound=9),
        page("s-few", 0, pr=0.5, outbound=1),
        page("s-unranked", 0, pr=None, outbound=0),
        *(page(f"f{i:02d}", 0, pr=0.5, outbound=9) for i in range(38)),
    ]
    targets = [page("t0", 1)]
    vectors = {p.url: along(1.0) for p in [*sources, *targets]}
    pairs = hubs_with(sources, targets, [])

    found = bridge_links(pairs, [*sources, *targets], [], vectors, [url("t0")], [], {})

    firsts = [b.source_url for b in found if b.hub_from == 0 and b.rank == 1]
    assert firsts == [url("s-few"), url("f00"), url("f01")]


def test_ineligible_sources_and_targets_are_never_proposed() -> None:
    sources = [
        page("s-linked", 0, pr=0.99),
        page("s-copy", 0, pr=0.98),
        page("s-novec", 0, pr=0.97),
        page("s-best", 0, pr=0.9),
        page("s-next", 0, pr=0.8),
        page("noise", -1, pr=0.99),
        *filler(36),
    ]
    targets = [
        page("t-ok", 1),
        page("t-noindex", 1),
        page("t-copy", 1),
        page("t-novec", 1),
        page("t-de", 1, language="de"),
    ]
    pages = [*sources, *targets]
    vectors = {p.url: [1.0, 0.0] for p in pages if p.url not in {url("s-novec"), url("t-novec")}}
    links = [link("s-linked", "t-ok")]
    pairs = hubs_with(sources, targets, links)
    indexable = [url(n) for n in ("t-ok", "t-copy", "t-novec", "t-de")]
    copies = [url("s-copy"), url("t-copy")]

    found = bridge_links(pairs, pages, links, vectors, indexable, copies, {})

    forward = [b for b in found if b.hub_from == 0]
    # Three pages needed, one already links: two slots, the two best eligible sources.
    assert [(b.slot, b.rank, b.source_url, b.target_url) for b in forward] == [
        (1, 1, url("s-best"), url("t-ok")),
        (2, 1, url("s-next"), url("t-ok")),
    ]
    used = {b.source_url for b in found} | {b.target_url for b in found}
    assert not {url("noise"), url("t-de")} & used


def alternatives_case() -> tuple[list[PageStructure], dict[str, list[float]], list[HubPair]]:
    sources = [page("s0", 0), page("s1", 0)]
    targets = [page(n, 1) for n in ("t1", "t2", "t3", "t4", "t5")]
    vectors = {
        url("s0"): [1.0, 0.0],
        url("s1"): [1.0, 0.0],
        url("t1"): [1.0, 0.0],
        url("t2"): [0.8, 0.6],
        url("t3"): [0.6, 0.8],
        url("t4"): [0.0, 1.0],
        url("t5"): [0.8, 0.6],
    }
    return [*sources, *targets], vectors, hubs_with(sources, targets, [])


@pytest.mark.parametrize(("alternatives", "ranks"), [(2, 3), (1, 2), (0, 1)])
def test_each_slot_ranks_its_targets_by_similarity_to_the_source_page(
    alternatives: int, ranks: int
) -> None:
    pages, vectors, pairs = alternatives_case()
    targets = [p.url for p in pages if "/t" in p.url]
    keywords = {url("t1"): ("Trail Shoes", KeywordRung.H1)}

    found = bridge_links(
        pairs, pages, [], vectors, targets, [], keywords, alternatives=alternatives
    )

    forward = [b for b in found if b.hub_from == 0]
    # t2 and t5 tie at 0.8: url order.
    expected = [("t1", 1.0), ("t2", 0.8), ("t5", 0.8), ("t3", 0.6), ("t4", 0.0)][:ranks]
    assert [(b.slot, b.rank, b.source_url) for b in forward] == [
        (1, rank, url("s0")) for rank in range(1, ranks + 1)
    ]
    assert [b.target_url for b in forward] == [url(n) for n, _ in expected]
    assert [b.similarity for b in forward] == pytest.approx([s for _, s in expected])
    assert (forward[0].anchor_keyword, forward[0].anchor_rung) == ("Trail Shoes", KeywordRung.H1)
    assert all(b.anchor_keyword is None for b in forward[1:])
    assert {b.reasons for b in found} == {(TREE, NEAREST, GAP)}


@pytest.mark.parametrize("alternatives", [-1, 3])
def test_alternatives_beyond_what_a_bridge_link_holds_are_refused(alternatives: int) -> None:
    """Rank 1 plus two alternatives is all BridgeLink allows; fail before any work, whatever
    the number of candidates."""
    pages = [page("s0", 0), page("t1", 1)]
    vectors = {p.url: [1.0, 0.0] for p in pages}
    pairs = hubs_with(pages[:1], pages[1:], [])

    with pytest.raises(ValueError, match="alternatives"):
        bridge_links(pairs, pages, [], vectors, [url("t1")], [], {}, alternatives=alternatives)


def test_several_slots_may_propose_the_same_target() -> None:
    # 21 pages: two slots, both closest to t1.
    sources = [page("s0", 0, pr=0.9), page("s1", 0, pr=0.8), *filler(19)]
    targets = [page("t1", 1), page("t4", 1)]
    vectors = {p.url: [1.0, 0.0] for p in [*sources, *targets]}
    vectors[url("t4")] = [0.0, 1.0]
    pages = [*sources, *targets]
    pairs = hubs_with(sources, targets, [])

    found = bridge_links(
        pairs, pages, [], vectors, [p.url for p in targets], [], {}, alternatives=0
    )

    assert [(b.slot, b.target_url) for b in found if b.hub_from == 0] == [
        (1, url("t1")),
        (2, url("t1")),
    ]


def test_an_uncovered_pair_gets_no_bridges() -> None:
    pages, vectors, _ = alternatives_case()
    uncovered = hub_pairs(pages, {0: [1.0, 0.0], 1: [0.0, 1.0]}, [])

    assert bridge_links(uncovered, pages, [], vectors, [p.url for p in pages], [], {}) == []


def test_the_same_inputs_in_any_order_give_the_same_links() -> None:
    pages, vectors, pairs = alternatives_case()
    targets = [p.url for p in pages]

    first = bridge_links(pairs, pages, [], vectors, targets, [], {})
    again = bridge_links(
        list(reversed(pairs)), random.sample(pages, len(pages)), [], vectors, targets[::-1], [], {}
    )

    assert first == again


# ── components and the report ───────────────────────────────────────────────


def one_page_hubs() -> list[PageStructure]:
    return [
        page("a", 0),
        page("b", 1),
        page("c", 2),
        page("a-de", 0, language="de"),
        page("n", -1),
    ]


def bridge(hub_from: int, hub_to: int, rank: int = 1) -> BridgeLink:
    return BridgeLink(
        language="en",
        hub_from=hub_from,
        hub_to=hub_to,
        slot=1,
        rank=rank,
        source_url=url(f"s{hub_from}{rank}"),
        target_url=url(f"t{hub_to}{rank}"),
        similarity=0.5,
        reasons=(TREE,),
    )


ONE_PAGE_PAIRS = [
    pair(0, 1, 0.9, size_a=1, size_b=1, pages_ab=1, pages_ba=1),
    pair(0, 2, 0.5, size_a=1, size_b=1, pages_ab=1, pages_ba=0),
    pair(1, 2, 0.1, size_a=1, size_b=1),
]


def test_components_count_hubs_joined_both_ways_summed_over_languages() -> None:
    pages = one_page_hubs()

    # en: {0, 1} and {2}; de: {0}.
    assert components(pages, ONE_PAGE_PAIRS) == 3
    assert components(pages, ONE_PAGE_PAIRS, [bridge(2, 0)]) == 2
    assert components(pages, ONE_PAGE_PAIRS, [bridge(2, 0, rank=2)]) == 3, (
        "alternatives are not made"
    )
    assert components(pages, []) == 4


def test_the_report_counts_directions_links_shortfalls_and_reasons() -> None:
    pairs = [
        pair(0, 1, 0.9, size_a=1, size_b=1, pages_ab=1, pages_ba=1, reasons=(TREE, NEAREST)),
        pair(0, 2, 0.5, size_a=1, size_b=1, pages_ab=1, pages_ba=0, reasons=(TREE,)),
        pair(1, 2, 0.1, size_a=1, size_b=1, reasons=(GAP,)),
    ]
    # 2 -> 0 is made with an alternative; 1 -> 2 is made; 2 -> 1 found nothing.
    links = [bridge(2, 0), bridge(2, 0, rank=2), bridge(1, 2)]

    report = bridge_report("acme", one_page_hubs(), pairs, links, gsc_used=False, started=0.0)

    assert (report.hubs, report.noise_pages, report.hub_pairs) == (3, 1, 3)
    assert (report.directions_below_floor, report.links_needed) == (3, 3)
    assert (report.bridge_links, report.alternatives, report.directions_short) == (2, 1, 1)
    assert (report.components_before, report.components_after) == (3, 2)
    assert report.by_reason == {TREE: 2, NEAREST: 1, GAP: 1}
    assert (report.floor_share, report.gsc_used) == (0.05, False)


def test_a_direction_at_the_floor_and_an_uncovered_pair_need_nothing() -> None:
    pairs = [
        pair(0, 1, 0.9, size_a=1, size_b=1, pages_ab=1, pages_ba=1, reasons=(TREE,)),
        pair(0, 2, 0.5, size_a=1, size_b=1),
    ]
    pages = [page("a", 0), page("b", 1), page("c", 2)]
    vectors = {p.url: [1.0, 0.0] for p in pages}

    assert bridge_links(pairs, pages, [], vectors, [p.url for p in pages], [], {}) == []
    report = bridge_report("acme", pages, pairs, [], gsc_used=True, started=0.0)
    assert (report.directions_below_floor, report.links_needed, report.directions_short) == (
        0,
        0,
        0,
    )
    assert (report.components_before, report.components_after) == (2, 2)


def test_the_summary_states_the_counts_without_urls() -> None:
    report = BridgeReport(
        tenant_id="acme",
        floor_share=0.05,
        hubs=5,
        noise_pages=3,
        hub_pairs=10,
        components_before=3,
        components_after=1,
        directions_below_floor=4,
        links_needed=6,
        bridge_links=5,
        alternatives=8,
        directions_short=1,
        by_reason={TREE: 4, NEAREST: 5, GAP: 3},
        gsc_used=False,
        seconds=0.2,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )

    summary = summarise_bridges(report)

    for fact in (
        "acme",
        "5 hubs",
        "no GSC data",
        "5%",
        "6 links",
        "5 proposed",
        "3 before, 1 after",
    ):
        assert fact in summary, f"{fact!r} missing from:\n{summary}"


# ── the planted gate (issue #18) ────────────────────────────────────────────

PLANTED = {(0, 1): 3, (2, 3): 5}
WELL_CONNECTED = (0, 4)


def planted_tenant() -> tuple[
    list[PageStructure],
    dict[int, list[float]],
    list[tuple[str, str]],
    dict[str, frozenset[str]],
    dict[str, list[float]],
]:
    """Five hubs of 20 pages with near-equal centroid cosine. Every link runs from the lower
    hub to the higher, so no pair meets the floor both ways and only bridges can join them."""
    hubs = range(5)
    pages = [page(f"h{hub}p{i:02d}", hub) for hub in hubs for i in range(20)]
    centroids = {hub: [1.0, *(0.3 if d == hub else 0.0 for d in hubs)] for hub in hubs}
    counts = {(a, b): 57 for a in hubs for b in hubs if a < b}
    counts |= PLANTED
    counts[WELL_CONNECTED] = 120
    links = [
        link(f"h{a}p{i // 20:02d}", f"h{b}p{i % 20:02d}")
        for (a, b), count in counts.items()
        for i in range(count)
    ]
    shared = {0: {"trail shoes", "running shoes"}, 1: {"trail shoes", "running shoes"}}
    shared |= {2: {"tents", "camping"}, 3: {"tents", "camping"}}
    queries = {
        url(f"h{hub}p00"): frozenset({*shared.get(hub, set()), f"only hub {hub}"}) for hub in hubs
    }
    vectors = {p.url: [*centroids[p.hub_id or 0], 0.01 * int(p.url[-2:])] for p in pages}
    return pages, {h: [*c, 0.0] for h, c in centroids.items()}, links, queries, vectors


def gap_ranks(pairs: list[HubPair]) -> list[tuple[int, int]]:
    ordered = sorted(pairs, key=lambda p: (-p.bridge_gap, p.hub_a, p.hub_b))
    return [(p.hub_a, p.hub_b) for p in ordered]


@pytest.mark.parametrize("with_gsc", [True, False], ids=["gsc", "no-gsc"])
def test_planted_bridge_gaps_rank_top_three(with_gsc: bool) -> None:
    pages, centroids, links, queries, vectors = planted_tenant()

    pairs = covered_pairs(hub_pairs(pages, centroids, links, queries if with_gsc else None))

    ranked = gap_ranks(pairs)
    well_linked = [key for key in ranked if key not in PLANTED]
    assert len(ranked) == 10
    by_key = {(p.hub_a, p.hub_b): p for p in pairs}
    # 3 and 5 distinct page pairs over 20 x 20; every well-linked pair has at least 57.
    assert [by_key[key].link_density for key in PLANTED] == pytest.approx([3 / 400, 5 / 400])
    assert min(by_key[key].link_density for key in well_linked) >= 57 / 400
    if with_gsc:
        assert set(ranked[:3]) >= set(PLANTED), ranked
        assert all(GAP in by_key[key].reasons for key in PLANTED)
        assert ranked.index(WELL_CONNECTED) >= len(ranked) // 2, ranked
        assert by_key[(0, 1)].shared_queries == ("running shoes", "trail shoes")
        assert by_key[WELL_CONNECTED].shared_queries == ()
    else:
        # The density term alone still puts both below-floor gaps above every well-linked pair.
        assert all(
            by_key[key].bridge_gap > by_key[other].bridge_gap
            for key in PLANTED
            for other in well_linked
        )

    proposed = bridge_links(pairs, pages, links, vectors, [p.url for p in pages], [], {})

    assert components(pages, pairs) == 5
    assert components(pages, pairs, proposed) == 1
