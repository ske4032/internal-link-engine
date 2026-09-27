"""Pair signals: term normalisation, null-safe Jaccard, cluster matching with hub noise, the
agreement buckets, per-page precomputation and the run report."""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from linking_engine.discovery.candidates import candidate_report
from linking_engine.discovery.signals import (
    agreement,
    build_page_signals,
    jaccard,
    normalise_term,
    pair_signals,
    same_cluster,
    signal_report,
)
from linking_engine.models import (
    CandidateSet,
    CandidateTarget,
    ClusterAgreement,
    CommunityContext,
    KeywordSource,
    PageSignals,
    TargetCandidates,
    TargetSelection,
)

A = ClusterAgreement
STRATEGIC, OBSERVED, INFERRED = (
    KeywordSource.CLIENT_STRATEGIC,
    KeywordSource.GSC_OBSERVED,
    KeywordSource.INFERRED,
)


def url(name: str) -> str:
    return f"example.com/{name}"


# ── normalise_term ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "normalised"),
    [
        pytest.param("Trail Shoes", "trail shoes", id="case"),
        pytest.param("  trail \t\n  shoes  ", "trail shoes", id="whitespace-runs"),
        pytest.param("trail\u00a0shoes", "trail shoes", id="no-break-space"),
        pytest.param("\uff34\uff52\uff41\uff49\uff4c", "trail", id="nfkc-full-width"),
        pytest.param("\ufb01t", "fit", id="nfkc-ligature"),
        pytest.param("Stra\u00dfe", "strasse", id="casefold-not-lower"),
        pytest.param("Cafe\u0301", "caf\u00e9", id="nfkc-composes"),
        pytest.param(" \t\n ", "", id="blank"),
    ],
)
def test_terms_are_normalised_before_any_set_operation(raw: str, normalised: str) -> None:
    assert normalise_term(raw) == normalised


# ── jaccard ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "overlap"),
    [
        pytest.param(set(), set(), 0.0, id="both-empty-new-pages"),
        pytest.param({"x"}, set(), 0.0, id="one-empty"),
        pytest.param({"x", "y"}, {"y", "x"}, 1.0, id="identical"),
        pytest.param({"x", "y"}, {"y", "z"}, 1 / 3, id="partial"),
        pytest.param({"x"}, {"y"}, 0.0, id="disjoint"),
        pytest.param({"x"}, {"x", "y", "z", "w"}, 0.25, id="sizes-differ"),
    ],
)
def test_jaccard_is_null_safe_and_symmetric(a: set[str], b: set[str], overlap: float) -> None:
    assert jaccard(frozenset(a), frozenset(b)) == pytest.approx(overlap)
    assert jaccard(frozenset(b), frozenset(a)) == pytest.approx(overlap)


# ── same_cluster and agreement ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "noise", "same"),
    [
        pytest.param(3, 3, None, True, id="same"),
        pytest.param(3, 4, None, False, id="different"),
        pytest.param(0, 0, None, True, id="id-zero-is-a-real-id"),
        pytest.param(None, 3, None, None, id="source-unclustered"),
        pytest.param(3, None, None, None, id="target-unclustered"),
        pytest.param(None, None, None, None, id="both-unclustered"),
        pytest.param(-1, -1, -1, False, id="noise-never-matches-noise"),
        pytest.param(-1, 2, -1, False, id="noise-and-hub"),
        pytest.param(2, -1, -1, False, id="hub-and-noise"),
        pytest.param(2, 2, -1, True, id="same-hub"),
        pytest.param(None, -1, -1, None, id="unclustered-beats-noise"),
    ],
)
def test_same_cluster(a: int | None, b: int | None, noise: int | None, same: bool | None) -> None:
    assert same_cluster(a, b, noise=noise) is same


@pytest.mark.parametrize(
    ("topics", "links", "bucket"),
    [
        pytest.param((1, 1), (5, 5), A.SAME_TOPIC_SAME_LINKS, id="well-connected"),
        pytest.param((1, 1), (5, 6), A.SAME_TOPIC_OTHER_LINKS, id="missing-link"),
        pytest.param((1, 2), (5, 5), A.OTHER_TOPIC_SAME_LINKS, id="link-to-remove"),
        pytest.param((1, 2), (5, 6), A.OTHER_TOPIC_OTHER_LINKS, id="unrelated"),
        pytest.param((0, 0), (0, 0), A.SAME_TOPIC_SAME_LINKS, id="zero-ids"),
        pytest.param((1, 1), (None, 5), A.SAME_TOPIC_OTHER_LINKS, id="orphan-source"),
        pytest.param((1, 1), (5, None), A.SAME_TOPIC_OTHER_LINKS, id="orphan-target"),
        pytest.param((1, 1), (None, None), A.SAME_TOPIC_OTHER_LINKS, id="two-orphans"),
        pytest.param((1, 2), (None, None), A.OTHER_TOPIC_OTHER_LINKS, id="two-orphans-other"),
        pytest.param((None, 1), (5, 5), A.UNKNOWN_TOPIC, id="source-no-topic"),
        pytest.param((1, None), (5, 5), A.UNKNOWN_TOPIC, id="target-no-topic"),
        pytest.param((None, None), (None, None), A.UNKNOWN_TOPIC, id="nothing-known"),
    ],
)
def test_agreement_buckets(
    topics: tuple[int | None, int | None],
    links: tuple[int | None, int | None],
    bucket: ClusterAgreement,
) -> None:
    assert agreement(topics[0], topics[1], links[0], links[1]) is bucket


# ── pair_signals ────────────────────────────────────────────────────────────


def signals(name: str, **fields: object) -> PageSignals:
    values: dict[str, object] = {
        "url": url(name),
        "queries": frozenset(),
        "keywords": frozenset(),
        "keyword_gap": 0,
        **fields,
    }
    return PageSignals.model_validate(values)


def test_a_pair_carries_both_overlaps_and_every_cluster_signal() -> None:
    source = signals(
        "s",
        queries=frozenset({"a", "b"}),
        keywords=frozenset({"k"}),
        link_community_id=5,
        keyword_community_id=1,
        content_community_id=7,
        hub_id=3,
    )
    target = signals(
        "t",
        queries=frozenset({"b", "c"}),
        keywords=frozenset({"k", "m"}),
        link_community_id=6,
        keyword_community_id=1,
        content_community_id=8,
        hub_id=3,
    )

    found = pair_signals(source, target)

    assert found.query_overlap == pytest.approx(1 / 3)
    assert found.keyword_overlap == 0.5
    assert (found.same_link_community, found.same_keyword_community) == (False, True)
    assert (found.same_content_community, found.same_hub) == (False, True)
    # Keyword communities agree against different link communities; content ones do not.
    assert found.cluster_agreement is A.SAME_TOPIC_OTHER_LINKS
    assert found.content_agreement is A.OTHER_TOPIC_OTHER_LINKS


def test_new_pages_with_no_data_get_zero_overlap_and_unknown_clusters() -> None:
    found = pair_signals(signals("new"), signals("other-new"))

    assert (found.query_overlap, found.keyword_overlap) == (0.0, 0.0)
    assert (found.same_link_community, found.same_keyword_community, found.same_hub) == (
        None,
        None,
        None,
    )
    assert (found.cluster_agreement, found.content_agreement) == (A.UNKNOWN_TOPIC, A.UNKNOWN_TOPIC)


def test_two_noise_pages_never_share_a_hub() -> None:
    assert pair_signals(signals("a", hub_id=-1), signals("b", hub_id=-1)).same_hub is False


def test_an_orphan_never_matches_on_links() -> None:
    orphan = signals("orphan", keyword_community_id=1, content_community_id=1)
    linked = signals("t", keyword_community_id=1, content_community_id=1, link_community_id=0)

    found = pair_signals(orphan, linked)

    assert found.same_link_community is None
    assert (found.cluster_agreement, found.content_agreement) == (
        A.SAME_TOPIC_OTHER_LINKS,
        A.SAME_TOPIC_OTHER_LINKS,
    )


# ── build_page_signals ──────────────────────────────────────────────────────


def context(name: str, **ids: int | None) -> CommunityContext:
    return CommunityContext.model_validate({"url": url(name), **ids})


def test_one_entry_per_crawled_page_with_its_cluster_ids() -> None:
    pages = [
        context(
            "a", link_community_id=0, keyword_community_id=2, content_community_id=4, hub_id=-1
        ),
        context("b"),
    ]

    found = build_page_signals(pages, [], [])

    assert sorted(found) == [url("a"), url("b")]
    a, b = found[url("a")], found[url("b")]
    assert (a.link_community_id, a.keyword_community_id, a.content_community_id, a.hub_id) == (
        0,
        2,
        4,
        -1,
    )
    assert (b.link_community_id, b.hub_id, b.queries, b.keywords, b.keyword_gap) == (
        None,
        None,
        frozenset(),
        frozenset(),
        None,
    )


def test_queries_and_keywords_are_normalised_and_deduplicated() -> None:
    found = build_page_signals(
        [context("a")],
        [
            (url("a"), "Trail Shoes"),
            (url("a"), " trail  shoes "),
            (url("a"), "\u3000"),
            (url("a"), ""),
        ],
        [(url("a"), "RAIN Jacket", STRATEGIC), (url("a"), "rain jacket", OBSERVED)],
    )[url("a")]

    assert found.queries == {"trail shoes"}
    assert found.keywords == {"rain jacket"}


def test_the_gap_counts_strategic_keywords_without_a_matching_query() -> None:
    queries = [(url("a"), "Trail Shoes"), (url("a"), "tents")]
    keywords = [
        (url("a"), "trail shoes", STRATEGIC),
        (url("a"), "Rain Jacket", STRATEGIC),
        (url("a"), "rain  jacket", STRATEGIC),
        (url("a"), "hiking boots", OBSERVED),
        (url("a"), "camp stoves", INFERRED),
    ]

    found = build_page_signals([context("a")], queries, keywords)[url("a")]

    # Only "rain jacket": trail shoes has a query, the other two are not strategic.
    assert found.keyword_gap == 1
    assert found.keywords == {"trail shoes", "rain jacket", "hiking boots", "camp stoves"}


def test_a_page_with_only_non_strategic_keywords_has_no_gap() -> None:
    keywords = [(url("a"), "hiking boots", OBSERVED), (url("a"), "camp stoves", INFERRED)]
    queries = [(url("b"), "tents")]
    assert build_page_signals([context("a")], queries, keywords)[url("a")].keyword_gap == 0


def test_without_any_gsc_data_the_gap_is_unknown_not_every_keyword() -> None:
    keywords = [(url("a"), "hiking boots", STRATEGIC), (url("a"), "camp stoves", STRATEGIC)]

    found = build_page_signals([context("a")], [], keywords)[url("a")]

    assert (found.keyword_gap, found.keywords) == (None, frozenset({"hiking boots", "camp stoves"}))


def test_a_page_without_queries_on_a_tenant_with_gsc_data_ranks_for_none_of_its_keywords() -> None:
    keywords = [(url("a"), "hiking boots", STRATEGIC), (url("a"), "camp stoves", STRATEGIC)]
    queries = [(url("not-crawled"), "tents")]

    assert build_page_signals([context("a")], queries, keywords)[url("a")].keyword_gap == 2


def test_rows_for_urls_that_are_not_crawled_pages_are_ignored() -> None:
    found = build_page_signals(
        [context("a")],
        [(url("a"), "tents"), (url("gone"), "tents"), (url("gone"), "stoves")],
        [(url("gone"), "stoves", STRATEGIC), (url("A"), "tents", STRATEGIC)],
    )

    assert list(found) == [url("a")]
    assert (found[url("a")].queries, found[url("a")].keywords) == (
        frozenset({"tents"}),
        frozenset(),
    )


def test_no_pages_give_no_signals() -> None:
    assert build_page_signals([], [(url("a"), "tents")], [(url("a"), "tents", STRATEGIC)]) == {}


# ── signal_report ───────────────────────────────────────────────────────────


def candidate_set(pairs: dict[str, list[str]]) -> CandidateSet:
    targets = tuple(
        TargetCandidates(
            target_url=target,
            sources=tuple(sources),
            similarities=tuple(0.9 - i * 0.1 for i in range(len(sources))),
            eligible=len(sources),
            linked=0,
            linked_nearer=0,
        )
        for target, sources in sorted(pairs.items())
    )
    selection = TargetSelection(
        crawled_pages=10,
        not_indexable=0,
        without_vector=10 - len(targets),
        targets=tuple(CandidateTarget(url=t.target_url, indexable_assumed=False) for t in targets),
    )
    report = candidate_report(
        "acme",
        "page_content",
        50,
        512,
        selection,
        10,
        targets,
        load_seconds=0.0,
        search_seconds=0.0,
        seconds=0.0,
    )
    return CandidateSet(report=report, targets=targets)


PAGES = {
    page.url: page
    for page in (
        signals(
            "t",
            queries=frozenset({"a", "b"}),
            keywords=frozenset({"k"}),
            keyword_gap=1,
            link_community_id=1,
            keyword_community_id=1,
            content_community_id=1,
            hub_id=2,
        ),
        signals(
            "s1",
            queries=frozenset({"b"}),
            link_community_id=1,
            keyword_community_id=1,
            content_community_id=2,
            hub_id=2,
        ),
        signals("s2", keywords=frozenset({"k"}), link_community_id=3, hub_id=-1),
        signals("new"),
    )
}


def test_the_report_counts_every_candidate_pair() -> None:
    candidates = candidate_set(
        {url("t"): [url("s1"), url("s2"), url("new")], url("new"): [url("t")], url("s2"): []}
    )

    report = signal_report("acme", PAGES, candidates, unmatched_query_urls=7)

    assert (report.tenant_id, report.pages, report.unmatched_query_urls, report.pairs) == (
        "acme",
        4,
        7,
        4,
    )
    assert (report.pages_with_queries, report.pages_with_keywords, report.pages_with_gap) == (
        2,
        2,
        1,
    )
    # Query overlaps: s1->t 1/2, s2->t 0, new->t 0, t->new 0. Keyword: s2->t 1.
    assert (report.query_overlap_pairs, report.keyword_overlap_pairs) == (1, 1)
    assert report.query_overlap_mean == pytest.approx(0.125)
    assert report.keyword_overlap_mean == pytest.approx(0.25)
    # s1 shares hub 2 with t; s2 is noise.
    assert (report.same_hub_pairs, report.noise_pairs) == (1, 1)
    assert report.cluster_agreement == {
        A.SAME_TOPIC_SAME_LINKS: 1,
        A.SAME_TOPIC_OTHER_LINKS: 0,
        A.OTHER_TOPIC_SAME_LINKS: 0,
        A.OTHER_TOPIC_OTHER_LINKS: 0,
        A.UNKNOWN_TOPIC: 3,
    }
    assert report.content_agreement == {
        A.SAME_TOPIC_SAME_LINKS: 0,
        A.SAME_TOPIC_OTHER_LINKS: 0,
        A.OTHER_TOPIC_SAME_LINKS: 1,
        A.OTHER_TOPIC_OTHER_LINKS: 0,
        A.UNKNOWN_TOPIC: 3,
    }


def test_an_empty_candidate_set_reports_every_bucket_at_zero() -> None:
    report = signal_report("acme", PAGES, candidate_set({}), unmatched_query_urls=0)

    assert (report.pairs, report.query_overlap_mean, report.keyword_overlap_mean) == (0, None, None)
    assert report.cluster_agreement == dict.fromkeys(ClusterAgreement, 0)
    assert report.content_agreement == dict.fromkeys(ClusterAgreement, 0)


@pytest.mark.parametrize(
    "pairs",
    [
        pytest.param({url("t"): [url("ghost")]}, id="unknown-source"),
        pytest.param({url("ghost"): [url("t")]}, id="unknown-target"),
    ],
)
def test_a_candidate_url_without_page_signals_is_refused(pairs: dict[str, list[str]]) -> None:
    with pytest.raises(ValueError, match=r"example\.com/ghost"):
        signal_report("acme", PAGES, candidate_set(pairs), unmatched_query_urls=0)


def test_one_log_line_carries_the_report_and_no_urls() -> None:
    candidates = candidate_set({url("t"): [url("s1"), url("s2")]})

    with capture_logs() as logs:
        report = signal_report("acme", PAGES, candidates, unmatched_query_urls=2)

    assert len(logs) == 1, logs
    [line] = logs
    assert line["event"] == "signals.report"
    assert (line["tenant_id"], line["pairs"], line["unmatched_query_urls"]) == ("acme", 2, 2)
    assert line["pairs"] == report.pairs
    assert not any("example.com" in str(value) for value in line.values()), "no urls in logs"
