"""Existing-link audit scoring (#99): each edge's scores, flags and verdict against the tenant's own
cut-offs. Small hand-built tenants pin the boundaries; the planted tenant
(tests/pipeline/link_audit_seed.py) carries the cases whose cut-offs need a realistic spread."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from link_audit_seed import (
    ARCHIVE,
    AUDITED,
    CANONICAL,
    GUIDE,
    HEALTHY,
    LINKS,
    MISALIGNED,
    NOISE,
    OFF,
    OVER,
    PAGE_BY_PATH,
    PAGINATED,
    SATURATED_SOURCES,
    SITEMAP,
    WASTED,
    PlantedLink,
    anchor_facts,
    audit_edges,
    planted_proposals,
    topic,
)
from test_keyword_stage import url

from linking_engine.audit.links import (
    CONTEXT_SPLIT,
    DENSITY_FENCE,
    EQUITY_TOP,
    FIT_SPLIT,
    GENERIC_CAP,
    INDEX_FENCE,
    NO_STORED_SCORES,
    OVER_OPTIMISED_MIN,
    SATURATION,
    AnchorFacts,
    Assessment,
    AuditOutcome,
    Proposal,
    anchor_quality,
    assess,
    audit_report,
    audit_scope,
    decide,
    equity_efficiency,
    is_pagination,
    is_sitemap,
    keyword_alignment,
    ladder_pairs,
)
from linking_engine.models import ActionType, AuditEdge, IssueFlag, LinkAuditReport

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from linking_engine.models import AuditCutoff, LinkAuditResult

RUN = "run-1"
AT = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
FIX, REANCHOR, REMOVE = ActionType.FIX, ActionType.REANCHOR, ActionType.REMOVE
GENERIC, BROKEN, REDIRECTED, NOINDEX, NOFOLLOW = (
    IssueFlag.GENERIC,
    IssueFlag.BROKEN,
    IssueFlag.REDIRECTED,
    IssueFlag.NOINDEX_TARGET,
    IssueFlag.NOFOLLOW,
)
Pair = tuple[str, str]


def edge(source: str, position: int, target: str, **fields: object) -> AuditEdge:
    return AuditEdge.model_validate(
        {
            "source_url": f"example.com/{source}",
            "position": position,
            "target_url": f"example.com/{target}",
            "anchor_text": "trail shoes",
            "source_page_rank_percentile": 0.5,
            "target_page_rank_percentile": 0.5,
            "target_status_code": 200,
            "target_indexable": True,
            "target_hub_id": 0,
            **fields,
        }
    )


def fact(
    key: str = "trail shoes",
    *,
    generic: bool = False,
    exact: bool = False,
    jaccard: float | None = 1.0,
) -> AnchorFacts:
    return AnchorFacts(key, generic, exact, jaccard)


def run(
    edges: Sequence[AuditEdge],
    facts: Sequence[AnchorFacts],
    proposals: Mapping[Pair, Sequence[Proposal]] | None = None,
) -> tuple[Assessment, list[LinkAuditResult]]:
    assessment = assess(edges, facts)
    outcome = decide(assessment, proposals or {}, run_id=RUN, audited_at=AT)
    return assessment, list(outcome.results)


def planted(
    *,
    a2: bool = True,
    edges: Sequence[AuditEdge] | None = None,
    proposals: Mapping[Pair, Sequence[Proposal]] | None = None,
) -> tuple[Assessment, AuditOutcome, dict[PlantedLink, LinkAuditResult]]:
    scope = audit_scope(audit_edges(a2=a2) if edges is None else edges)
    assessment = assess(scope.edges, anchor_facts(), scope=scope)
    outcome = decide(
        assessment,
        planted_proposals() if proposals is None else proposals,
        run_id=RUN,
        audited_at=AT,
    )
    return assessment, outcome, dict(zip(AUDITED, outcome.results, strict=True))


def case(
    results: Mapping[PlantedLink, LinkAuditResult], name: str
) -> list[tuple[PlantedLink, LinkAuditResult]]:
    found = [(link, result) for link, result in results.items() if link.case == name]
    assert found, f"no planted {name} link"
    return found


def planted_link(anchor: str) -> PlantedLink:
    [found] = [link for link in AUDITED if link.anchor == anchor]
    return found


def pair(link: PlantedLink) -> Pair:
    return url(link.source), url(link.target)


def cutoff(assessment: Assessment, name: str) -> AuditCutoff:
    [found] = [c for c in assessment.cutoffs if c.name == name]
    return found


def report_of(assessment: Assessment, outcome: AuditOutcome) -> LinkAuditReport:
    return audit_report(
        "test-planted",
        RUN,
        assessment,
        outcome,
        ladder_pairs=len(ladder_pairs(assessment)),
        vectors_skipped_reason=None,
        seconds=1.0,
        finished_at=AT,
    )


def said(result: LinkAuditResult, words: str) -> bool:
    return any(words in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("jaccard", "cosine", "expected"),
    [(None, 0.9, None), (0.5, None, 0.5), (0.5, 1.0, 0.65), (0.0, 0.5, 0.15), (1.0, 1.0, 1.0)],
)
def test_keyword_alignment_is_seven_tenths_jaccard_and_three_tenths_cosine(
    jaccard: float | None, cosine: float | None, expected: float | None
) -> None:
    found = keyword_alignment(jaccard, cosine)
    assert found == (None if expected is None else pytest.approx(expected))


def test_anchor_quality_score_composite_and_generic_cap() -> None:
    assert anchor_quality(0.8, 0.6, 0.5, generic=False) == pytest.approx(64.0)
    # Without a fit the weights renormalise over the two dimensions present.
    assert anchor_quality(0.8, None, 0.5, generic=False) == pytest.approx(
        100 * (0.35 * 0.8 + 0.30 * 0.5) / 0.65
    )
    assert anchor_quality(0.8, None, None, generic=False) == pytest.approx(80.0)
    assert anchor_quality(None, None, None, generic=False) is None
    assert GENERIC_CAP == 20
    assert anchor_quality(1.0, None, 0.9, generic=True) == pytest.approx(20.0)
    # The cap is a ceiling, not a floor.
    assert anchor_quality(0.0, None, 0.1, generic=True) == pytest.approx(100 * 0.03 / 0.65)

    _, _, results = planted()
    for _, result in case(results, "generic"):
        assert result.anchor_quality_score is not None
        assert result.anchor_quality_score <= 20.0
    healthy = [r for link, r in results.items() if link.a2 == HEALTHY and r.anchor_target_fit]
    assert len(healthy) > 50
    for result in healthy:
        assert result.anchor_quality_score == pytest.approx(
            anchor_quality(
                result.keyword_alignment,
                result.anchor_target_fit,
                result.context_relevance,
                generic=False,
            )
        )


def test_equity_efficiency_and_wasted_equity() -> None:
    assert equity_efficiency(0.9, 0, 4, 0.1) == pytest.approx(0.81)
    assert equity_efficiency(0.9, 2, 4, 0.1) == pytest.approx(0.405)
    assert equity_efficiency(0.9, 3, 4, 0.1) == pytest.approx(0.2025)
    assert equity_efficiency(None, 0, 4, 0.1) is None
    assert equity_efficiency(0.9, 0, 4, None) is None

    # One link per source into a noise page, equity 0.1 .. 0.4: only the one above Q3 wastes it.
    noise = [
        edge(f"s{i}", 0, f"n{i}", source_page_rank_percentile=p, target_page_rank_percentile=0.0,
             target_hub_id=-1)
        for i, p in enumerate((0.1, 0.2, 0.3, 0.4))
    ]  # fmt: skip
    assessment, results = run(noise, [fact()] * 4)
    assert cutoff(assessment, EQUITY_TOP).value == pytest.approx(0.325)
    assert [r.equity_efficiency for r in results] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert [r.issue_flags for r in results] == [set(), set(), set(), {WASTED}]
    assert results[3].verdict is None
    _, hub = run([e.model_copy(update={"target_hub_id": 3}) for e in noise], [fact()] * 4)
    assert not any(r.issue_flags for r in hub)
    _, hidden = run(
        [e.model_copy(update={"target_hub_id": 3, "target_indexable": False}) for e in noise],
        [fact()] * 4,
    )
    assert hidden[3].issue_flags == {NOINDEX, WASTED}

    # Top-quartile equity alone is no waste: plenty of healthy planted links have it.
    assessment, _, planted_results = planted()
    top = cutoff(assessment, EQUITY_TOP).value
    assert top is not None
    rich = [
        r for link, r in planted_results.items()
        if link.a2 == HEALTHY and r.equity_efficiency is not None and r.equity_efficiency > top
    ]  # fmt: skip
    assert len(rich) >= 10
    assert not any(r.issue_flags for r in rich)


def test_technical_flags_fix_with_canonical_target() -> None:
    edges = [
        edge("s1", 0, "gone", target_status_code=404),
        edge("s1", 1, "down", target_status_code=503),
        edge("s1", 2, "moved", target_status_code=301),
        edge("s2", 0, "hidden", target_indexable=False),
        edge("s2", 1, "sponsor", is_follow=False),
        edge("s2", 2, "print", target_canonical_url="example.com/original"),
        edge("s3", 0, "fine"),
        edge("s3", 1, "s3-print", target_canonical_url="example.com/s3"),
    ]
    _, results = run(edges, [fact()] * len(edges))

    assert [r.issue_flags for r in results] == [
        {BROKEN}, {BROKEN}, {REDIRECTED}, {NOINDEX}, {NOFOLLOW}, set(), set(), set()
    ]  # fmt: skip
    assert [r.verdict for r in results] == [FIX] * 6 + [None, FIX]
    assert [r.fix_target for r in results] == [None] * 5 + ["example.com/original", None, None]
    assert said(results[0], "404")
    assert said(results[1], "503")
    assert said(results[5], "canonical")
    # A copy of the source page itself: FIX, but there is no other page to point at.
    assert said(results[7], "itself")

    _, _, planted_results = planted()
    [(_, copy)] = case(planted_results, "non-canonical")
    assert (copy.verdict, copy.fix_target) == (FIX, url(CANONICAL))
    [(_, canonical)] = case(planted_results, "canonical")
    assert (canonical.verdict, canonical.fix_target) == (None, None)


def test_generic_and_over_optimised_reanchor() -> None:
    keyword = "trekking poles"
    edges: list[AuditEdge] = []
    facts: list[AnchorFacts] = []

    def into(target: str, *, exact: int, partial: int = 0, generic: int = 0) -> None:
        for kind, count, anchor, measured in (
            ("e", exact, keyword, fact(keyword, exact=True)),
            ("p", partial, f"best {keyword}", fact(f"best {keyword}")),
            ("g", generic, "click here", fact("click here", generic=True, jaccard=0.0)),
        ):
            for i in range(count):
                edges.append(edge(f"{target}-{kind}{i}", 0, target, anchor_text=anchor))
                facts.append(measured)

    assert OVER_OPTIMISED_MIN == 5
    into("five", exact=5, partial=2)
    into("half", exact=5, partial=5)
    into("four", exact=4)
    into("diluted", exact=5, partial=6)
    # Generic anchors are not descriptive: five exact of five descriptive.
    into("generic", exact=5, generic=5)
    _, results = run(edges, facts)

    over = Counter(r.target_url for r in results if OVER in r.issue_flags)
    assert over == {"example.com/five": 5, "example.com/half": 5, "example.com/generic": 5}
    assert sum(GENERIC in r.issue_flags for r in results) == 5
    for result in results:
        if result.issue_flags & {OVER, GENERIC}:
            assert result.verdict is REANCHOR
            assert result.proposed_anchor is None
            assert said(result, "no better phrase")
        else:
            assert result.verdict is None, result


def test_misaligned_uses_tenant_split() -> None:
    assessment, _, results = planted()
    split = cutoff(assessment, FIT_SPLIT)
    assert split.value is not None
    assert 0.34 < split.value < 0.74, split
    [(_, synonym)] = case(results, "synonym")
    assert (synonym.issue_flags, synonym.verdict) == (set(), None)
    for _, result in case(results, "misaligned"):
        assert MISALIGNED in result.issue_flags
        assert said(result, "weakly")
    # A generic anchor shares no stem either; GENERIC covers it.
    for _, result in case(results, "generic"):
        assert MISALIGNED not in result.issue_flags
    # A target without keywords has no alignment to miss.
    [noise] = [r for link, r in case(results, "noise") if link.target == NOISE]
    assert noise.keyword_alignment is None
    assert noise.issue_flags == set()

    # A1: without a fit, sharing no stem is enough.
    _, _, a1 = planted(a2=False)
    [(_, synonym_a1)] = case(a1, "synonym")
    assert MISALIGNED in synonym_a1.issue_flags
    assert (synonym_a1.verdict, synonym_a1.proposed_anchor) == (REANCHOR, "folding chair")

    # Too few fits to split: zero overlap alone flags, and the cut-off says why.
    edges = [
        edge(f"s{i}", 0, "t", anchor_text="portable seat", context_relevance=0.8,
             anchor_target_fit=0.8)
        for i in range(10)
    ]  # fmt: skip
    few, flagged = run(edges, [fact("portable seat", jaccard=0.0)] * 10)
    assert cutoff(few, FIT_SPLIT).value is None
    assert "fewer than 50" in cutoff(few, FIT_SPLIT).reason
    assert all(MISALIGNED in r.issue_flags for r in flagged)


def test_off_topic_remove_is_conservative() -> None:
    assessment, _, results = planted()
    by_source = {link.source: r for link, r in case(results, "off-topic")}
    alone = by_source[topic(8)]
    saturated = by_source[topic(SATURATED_SOURCES[0])]
    wasted = by_source[topic(19)]
    assert (alone.issue_flags, alone.verdict) == ({OFF, MISALIGNED}, None)
    assert saturated.verdict is REMOVE
    assert WASTED not in saturated.issue_flags
    assert said(saturated, "above the tenant's Q3")
    assert wasted.verdict is REMOVE
    assert said(wasted, "wastes equity")
    # Off topic on a saturated source, but sharing a stem with the keyword: a flag, not REMOVE.
    [(aligned_link, aligned)] = case(results, "off-topic-aligned")
    assert aligned_link.source == topic(SATURATED_SOURCES[1])
    assert (aligned.issue_flags, aligned.verdict) == ({OFF}, None)
    # Index-like, not a listing: flagged, never removed.
    [(_, guide)] = case(results, "index-like")
    assert guide.issue_flags == {OFF, MISALIGNED, WASTED}
    assert guide.verdict is None
    assert said(guide, "index-like")
    assert assessment.index_like_pages == 2
    removed = {link.source for link, r in results.items() if r.verdict is REMOVE}
    assert removed == {topic(SATURATED_SOURCES[0]), topic(19)}

    # Fewer than 50 context scores give no split: nothing is off topic, so nothing is removed.
    edges = [
        e if i < 40 else e.model_copy(update={"context_relevance": None})
        for i, e in enumerate(audit_edges())
    ]
    thin, _, thin_results = planted(edges=edges)
    assert cutoff(thin, CONTEXT_SPLIT).value is None
    assert "fewer than 50" in cutoff(thin, CONTEXT_SPLIT).reason
    assert thin.embeddings_skipped_reason is None
    assert not any(OFF in r.issue_flags or r.verdict is REMOVE for r in thin_results.values())


def test_reanchor_proposes_an_extracted_phrase_or_says_none() -> None:
    given = planted_proposals()
    _, _, results = planted()
    proposed = [r for r in results.values() if r.proposed_anchor is not None]
    assert len(proposed) == sum(link.a2.proposal is not None for link in AUDITED)
    for result in proposed:
        assert result.verdict is REANCHOR
        phrases = {p.phrase for p in given[(result.source_url, result.target_url)]}
        assert result.proposed_anchor in phrases
        assert said(result, result.proposed_anchor)
    [(_, weak)] = case(results, "weak-fit")
    assert (weak.issue_flags, weak.verdict, weak.proposed_anchor) == (
        set(),
        REANCHOR,
        "snow shovel",
    )
    assert said(weak, "weakly")
    static = results[planted_link("static line")]
    assert static.verdict is None
    assert said(static, "no better phrase")

    # The anchor itself is no proposal, nor is a phrase that neither aligns nor fits better.
    read_more, static_line = planted_link("read more"), planted_link("static line")
    block = [link for link in AUDITED if OVER in link.a2.flags]
    extra = dict(given)
    extra[pair(read_more)] = [
        Proposal("Read More", "read more", 0.0, None, None),
        Proposal("fleece vest", "fleece vest", 1.0, None, None),
    ]
    extra[pair(static_line)] = [Proposal("rope work", "rope work", 0.0, None, 0.1)]
    for link in block:
        extra[pair(link)] = [Proposal("Trekking Poles", "trekking poles", 1.0, None, None)]
    _, _, again = planted(proposals=extra)
    assert again[read_more].proposed_anchor == "fleece vest"
    assert again[static_line].verdict is None
    assert len(block) == 5
    for link in block:
        assert (again[link].verdict, again[link].proposed_anchor) == (REANCHOR, None)


def test_verdict_precedence() -> None:
    saturated = next(
        link for link in AUDITED
        if link.case == "off-topic" and link.source == topic(SATURATED_SOURCES[0])
    )  # fmt: skip
    generic = planted_link("click here")
    edges = audit_edges()
    at = {link: i for i, link in enumerate(LINKS)}
    edges[at[saturated]] = edges[at[saturated]].model_copy(update={"is_follow": False})
    edges[at[generic]] = edges[at[generic]].model_copy(update={"target_status_code": 404})
    _, _, results = planted(edges=edges)
    # FIX over REMOVE, and FIX over REANCHOR even with a phrase in the copy.
    assert results[saturated].verdict is FIX
    assert {OFF, NOFOLLOW} <= results[saturated].issue_flags
    assert results[generic].verdict is FIX
    assert {GENERIC, BROKEN} <= results[generic].issue_flags
    assert results[generic].proposed_anchor is None

    # REMOVE over REANCHOR: the copy writes the target's keyword, the link still goes.
    proposals = planted_proposals()
    proposals[pair(saturated)] = [Proposal("insect net", "insect net", 1.0, None, None)]
    _, _, results = planted(proposals=proposals)
    assert (results[saturated].verdict, results[saturated].proposed_anchor) == (REMOVE, None)
    # REANCHOR over no verdict: a weak fit with a better phrase.
    [(_, weak)] = case(results, "weak-fit")
    assert weak.verdict is REANCHOR


def test_placeholder_links_are_unverified_not_flagged() -> None:
    assessment, outcome, results = planted()
    [(_, ghost)] = case(results, "placeholder")
    assert ghost.unverified
    assert (ghost.issue_flags, ghost.verdict, ghost.proposed_anchor) == (set(), None, None)
    scores = (
        ghost.anchor_quality_score,
        ghost.keyword_alignment,
        ghost.context_relevance,
        ghost.anchor_target_fit,
        ghost.equity_efficiency,
    )
    assert scores == (None,) * 5
    assert said(ghost, "not crawled")
    assert sum(r.unverified for r in outcome.results) == 1

    # Still a body link of its source: the link before it is weighed among six, not five.
    [(link, canonical)] = case(results, "canonical")
    links_on_source = sum(other.source == link.source for other in AUDITED)
    assert links_on_source == 6
    source, target = PAGE_BY_PATH[link.source], PAGE_BY_PATH[link.target]
    assert canonical.equity_efficiency == pytest.approx(
        source.percentile * (1 - link.position / links_on_source) * (1 - target.percentile)
    )

    report = report_of(assessment, outcome)
    assert report.unverified == 1
    assert report.healthy + sum(report.by_verdict.values()) + report.unverified == report.links


def test_a1_only_when_embeddings_missing_with_reason() -> None:
    assessment, outcome, results = planted(a2=False)
    assert assessment.embeddings_skipped_reason == NO_STORED_SCORES
    for name in (CONTEXT_SPLIT, FIT_SPLIT):
        missing = cutoff(assessment, name)
        assert missing.value is None
        assert missing.reason.startswith("A1 only")
    assert not any(OFF in r.issue_flags or r.verdict is REMOVE for r in outcome.results)
    assert all(r.context_relevance is None and r.anchor_target_fit is None for r in outcome.results)
    # The composite falls back to keyword alignment alone.
    exact = [r for link, r in results.items() if link.a1 == HEALTHY and r.keyword_alignment == 1.0]
    assert exact
    assert all(r.anchor_quality_score == pytest.approx(100.0) for r in exact)

    report = report_of(assessment, outcome)
    assert not report.embeddings
    assert report.embeddings_skipped_reason == NO_STORED_SCORES
    assert (report.context_relevance, report.anchor_target_fit) == (None, None)


@pytest.mark.parametrize("a2", [True, False], ids=["A2", "A1"])
def test_the_scoring_recovers_every_planted_flag_and_verdict(a2: bool) -> None:
    _, _, results = planted(a2=a2)

    def named(links: set[PlantedLink]) -> list[str]:
        return sorted(f"{link.case} {link.source}->{link.target}" for link in links)

    for flag in IssueFlag:
        truth = {link for link in AUDITED if flag in link.expected(a2=a2).flags}
        found = {link for link, r in results.items() if flag in r.issue_flags}
        assert found == truth, (
            f"{flag}: missed {named(truth - found)}, extra {named(found - truth)}"
        )
    wrong = [
        (link.case, link.source, link.target, r.verdict, r.proposed_anchor)
        for link, r in results.items()
        if (r.verdict, r.proposed_anchor)
        != (link.expected(a2=a2).verdict, link.expected(a2=a2).proposal)
    ]
    assert not wrong
    assert {link.fix_target for link in AUDITED} - {None} == {CANONICAL}
    assert all(
        r.fix_target == (None if link.fix_target is None else url(link.fix_target))
        for link, r in results.items()
    )


def test_the_report_counts_every_link_once_and_names_each_cutoff() -> None:
    assessment, outcome, _ = planted()
    report = report_of(assessment, outcome)

    skipped = Counter(link.skipped for link in LINKS if link.skipped)
    assert (report.links, report.unverified, report.source_pages) == (len(AUDITED), 1, 22)
    assert (report.index_like_pages, report.listing_pages) == (2, 1)
    assert (report.sitemap_pages, report.sitemap_links) == (1, skipped["sitemap"])
    assert (report.paginated_pages, report.paginated_links) == (1, skipped["paginated"])
    assert report.by_flag == Counter(flag for link in AUDITED for flag in link.a2.flags)
    assert report.by_verdict == Counter(link.a2.verdict for link in AUDITED if link.a2.verdict)
    assert report.proposals == sum(link.a2.proposal is not None for link in AUDITED)
    names = {c.name for c in report.cutoffs}
    assert names == {CONTEXT_SPLIT, FIT_SPLIT, SATURATION, INDEX_FENCE, EQUITY_TOP, DENSITY_FENCE}
    assert all(c.value is not None and c.reason for c in report.cutoffs)
    assert report.embeddings
    assert report.keyword_cosines == 0

    # Only links that want a new anchor go to the ladder: neither FIX nor REMOVE, and not from a
    # listing.
    wanting = {
        pair(link) for link in AUDITED
        if link.a2.verdict not in (FIX, REMOVE)
        and link.source != ARCHIVE
        and not link.unverified
        and (
            link.a2.flags & {GENERIC, MISALIGNED, OVER}
            or (not link.generic and link.fit is not None and link.fit < 0.5)
        )
    }  # fmt: skip
    assert ladder_pairs(assessment) == wanting


@pytest.mark.parametrize(
    ("path", "sitemap", "pagination"),
    [
        ("example.com/sitemap", True, False),
        ("example.com/Sitemap.HTML", True, False),
        ("example.com/html-sitemap", True, False),
        ("example.com/en/site-map/", True, False),
        ("sitemap.example.com/gear/tents", False, False),
        ("example.com/sitemaps-guide/tents", False, False),
        ("example.com/blog/page/2", False, True),
        ("example.com/blog?page=3", False, True),
        ("example.com/blog?p=3&sort=new", False, True),
        ("example.com/blog?PG=3", False, True),
        ("example.com/page/about", False, False),
        ("example.com/pages/2", False, False),
        ("example.com/blog?pager=2", False, False),
        ("[unparsable", False, False),
        # The host never counts.
        ("sitemap-tools.example/blog/post", False, False),
        ("https://sitemap-tools.example/a", False, False),
        ("page-2.example/a", False, False),
        ("p.example/2", False, False),
        ("example.com/redirect?to=https://other.example/sitemap", False, False),
        ("//cdn.example.com/sitemap", True, False),
        ("example.com:8080/site-map/", True, False),
        ("example.com:8080/blog/page/2", False, True),
        # Amendment 7: page-number, offset and extra parameters, and three path forms.
        ("example.com/blog?p=123", False, True),
        ("example.com/blog?utm=a&start=20", False, True),
        ("example.com/blog?CurrentPage=3", False, True),
        ("example.com/blog?pageIndex=2&sort=a", False, True),
        ("example.com/blog?page_index", False, True),
        ("example.com/blog/page-2", False, True),
        ("example.com/blog/Page-12/", False, True),
        ("example.com/blog/p/3", False, True),
        ("example.com/p/about", False, False),
        ("example.com/blog/page-two", False, False),
        ("example.com/blog/pages-2", False, False),
        ("example.com/blog/p3", False, False),
        ("example.com/blog?sort=page", False, False),
        ("example.com/blog?ids=1", False, False),
    ],
)
def test_sitemap_and_pagination_urls_are_recognised_by_path_and_query_only(
    path: str, sitemap: bool, pagination: bool
) -> None:
    assert (is_sitemap(path), is_pagination(path)) == (sitemap, pagination)


def test_sitemap_and_paginated_pages_are_left_out_entirely() -> None:
    edges = audit_edges()
    scope = audit_scope(edges)

    kept = {(e.source_url, e.position) for e in scope.edges}
    assert kept == {(url(link.source), link.position) for link in AUDITED}
    assert all(url(end) not in {e.source_url for e in scope.edges} for end in (SITEMAP, PAGINATED))
    assert all(url(end) not in {e.target_url for e in scope.edges} for end in (SITEMAP, PAGINATED))
    skipped = Counter(link.skipped for link in LINKS if link.skipped)
    assert (scope.sitemap_pages, scope.sitemap_links) == (1, skipped["sitemap"])
    assert (scope.paginated_pages, scope.paginated_links) == (1, skipped["paginated"])
    # A link between the two kinds counts once, as sitemap.
    both = audit_scope([edge("sitemap", 0, "journal/page/2")])
    assert (both.sitemap_links, both.paginated_links, both.edges) == (1, 0, ())
    with pytest.raises(ValueError, match="audit_scope"):
        assess(edges, anchor_facts(LINKS))


def test_listing_sources_keep_flags_and_fix_but_are_never_reanchored_or_removed() -> None:
    assessment, _, results = planted()
    density = cutoff(assessment, DENSITY_FENCE)
    assert density.value is not None
    assert "per 100 words" in density.reason
    assert assessment.listing_pages == 1
    listing = {link.anchor: r for link, r in results.items() if link.source == ARCHIVE}
    assert listing["car wax"].issue_flags == {OFF, MISALIGNED, WASTED}
    assert listing["car wax"].verdict is None
    assert said(listing["car wax"], "listing page")
    assert (listing["See more"].issue_flags, listing["See more"].verdict) == ({GENERIC}, None)
    assert (listing["Retired Boots"].issue_flags, listing["Retired Boots"].verdict) == (
        {BROKEN},
        FIX,
    )
    # Its exact title into the over-optimised target is neither flagged nor counted.
    assert not listing["Trekking Poles"].issue_flags
    assert all(url(ARCHIVE) != source for source, _ in ladder_pairs(assessment))
    # The guide has as many links in long copy: index-like, not a listing.
    guide = [r for link, r in results.items() if link.source == GUIDE]
    assert not any(said(r, "listing page") for r in guide)


def test_an_iqr_of_zero_is_floored_so_one_link_more_is_not_index_like() -> None:
    edges: list[AuditEdge] = []
    for page, count in [*((f"p{i}", 5) for i in range(12)), ("q0", 6), ("q1", 6), ("hub", 40)]:
        edges += [edge(page, n, f"t{n}") for n in range(count)]
    assessment, results = run(edges, [fact()] * len(edges))

    fence = cutoff(assessment, INDEX_FENCE)
    assert (cutoff(assessment, SATURATION).value, fence.value) == (5.0, 8.0)
    assert "floored" in fence.reason
    assert assessment.index_like_pages == 1
    assert len(results) == len(edges)


def test_remove_needs_off_topic_misaligned_and_waste_or_saturation() -> None:
    """One saturated source per case (six links against a Q3 of five) whose last link is off
    topic by its context; only the misaligned, non-generic one from a page that is not a listing
    goes. The listing is saturated but not index-like, so only its density protects it."""
    words = {"source_word_count": 100}
    background = [
        edge(f"b{i}", n, f"t{i}-{n}", context_relevance=0.8, anchor_target_fit=0.8, **words)
        for i in range(20)
        for n in range(5)
    ]
    cases = {
        "misaligned": (fact("garden gnomes", jaccard=0.0), 0.3, 100),
        "aligned": (fact("warmer weather", jaccard=0.25), 0.3, 100),
        "generic": (fact("click here", generic=True, jaccard=0.0), None, 100),
        "listing": (fact("bike bells", jaccard=0.0), 0.3, 10),
        # A fit stored before the tenant's generic rules changed: ignored.
        "stale-generic": (fact("see the range", generic=True, jaccard=0.0), 0.3, 100),
    }
    edges, facts = list(background), [fact()] * len(background)
    for name, (measured, fit, count) in cases.items():
        edges += [
            edge(f"sat-{name}", n, f"u{n}", context_relevance=0.8, anchor_target_fit=0.8,
                 source_word_count=count)
            for n in range(5)
        ]  # fmt: skip
        edges.append(
            edge(f"sat-{name}", 5, f"off-{name}", anchor_text=measured.key or "",
                 context_relevance=0.3, anchor_target_fit=fit, source_word_count=count)
        )  # fmt: skip
        facts += [fact()] * 5 + [measured]
    assessment, results = run(edges, facts)

    assert (assessment.index_like_pages, assessment.listing_pages) == (0, 1)
    off = {r.source_url.removeprefix("example.com/sat-"): r for r in results if r.position == 5}
    assert (off["misaligned"].issue_flags, off["misaligned"].verdict) == (
        {OFF, MISALIGNED},
        REMOVE,
    )
    assert (off["aligned"].issue_flags, off["aligned"].verdict) == ({OFF}, None)
    assert (off["generic"].issue_flags, off["generic"].verdict) == ({GENERIC}, REANCHOR)
    assert (off["listing"].issue_flags, off["listing"].verdict) == ({OFF, MISALIGNED}, None)
    assert said(off["listing"], "listing page")
    stale = off["stale-generic"]
    assert (stale.issue_flags, stale.verdict, stale.anchor_target_fit) == (
        {GENERIC},
        REANCHOR,
        None,
    )
    assert stale.anchor_quality_score is not None
    assert stale.anchor_quality_score <= 20.0


def test_assess_takes_one_measurement_per_edge() -> None:
    with pytest.raises(ValueError, match="one AnchorFacts per edge"):
        assess([edge("s", 0, "t")], [])


def test_scores_in_one_mode_give_no_split_so_nothing_is_off_topic() -> None:
    edges = [
        edge(f"s{i}", 0, f"t{i}", anchor_text="garden gnomes", context_relevance=0.8 + i / 10_000,
             anchor_target_fit=0.8 + i / 10_000)
        for i in range(60)
    ]  # fmt: skip
    assessment, results = run(edges, [fact("garden gnomes", jaccard=0.0)] * 60)

    for name in (CONTEXT_SPLIT, FIT_SPLIT):
        assert cutoff(assessment, name).value is None
        assert "do not separate into two modes" in cutoff(assessment, name).reason
    # Without a fit split, sharing no stem is enough for MISALIGNED; nothing is off topic.
    assert {r.issue_flags for r in results} == {frozenset({MISALIGNED})}
    assert not any(r.verdict is REMOVE for r in results)


def test_a_mostly_partial_match_tenant_flags_no_partial_match_as_misaligned() -> None:
    """The spread that fooled a mixture split: ten anchors sharing no stem, sixty partial
    matches, thirty exact ones. Only the ten share no stem, so only they are misaligned."""
    measured = (
        [fact("garden gnomes", jaccard=0.0)] * 10
        + [fact("best trail shoes", jaccard=2 / 3)] * 60
        + [fact("trail shoes", exact=True)] * 30
    )
    edges = [edge(f"s{i}", 0, f"t{i}", anchor_text=m.key or "") for i, m in enumerate(measured)]

    _, results = run(edges, measured)

    flagged = [i for i, r in enumerate(results) if MISALIGNED in r.issue_flags]
    assert flagged == list(range(10))
