"""The planted link-audit tenant keeps its premises: each planted case sits clearly on its side of
the per-tenant cut-offs the contract defines, computed here from the contract's formulas rather
than the audit's code, so a failure names a broken fixture, not a broken audit."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from link_audit_seed import (
    ARCHIVE,
    AUDITED,
    CONTROL_TARGET,
    GUIDE,
    LINKS,
    MISALIGNED,
    OFF,
    OVER,
    OVER_OPTIMISED_TARGET,
    PAGE_BY_PATH,
    PAGES,
    SATURATED_SOURCES,
    WASTED,
    PlantedLink,
    topic,
)

from linking_engine.anchor.extraction import Stems, locate_anchors, tokens
from linking_engine.anchor.generic import normalise_anchor
from linking_engine.audit.relevance import mixture_split

# Outbound body links per source, as audited: sitemap and paginated links left out.
OUTBOUND = Counter(link.source for link in AUDITED)
WASTE_CASES = ("off-topic", "off-topic-aligned", "noise", "noindex", "index-like", "listing")
LISTINGS = {ARCHIVE}


def upper_fence(values: list[float]) -> tuple[float, float]:
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3), float(q3 + 3 * max(q3 - q1, 1.0))


def equity(link: PlantedLink) -> float:
    source, target = PAGE_BY_PATH[link.source], PAGE_BY_PATH[link.target]
    weight = 1 - link.position / OUTBOUND[link.source]
    return source.percentile * weight * (1 - target.percentile)


def stems(text: str) -> set[str]:
    stem = Stems("en")
    return {stem(token.folded) for token in tokens(text)}


def test_the_archive_and_the_guide_are_index_like_and_the_saturated_sources_are_not() -> None:
    q3, fence = upper_fence(list(map(float, OUTBOUND.values())))
    assert {path for path, n in OUTBOUND.items() if n > fence} == {ARCHIVE, GUIDE}
    for source in SATURATED_SOURCES:
        assert q3 < OUTBOUND[topic(source)] <= fence, f"page {source} must be saturated"
    [alone] = [link for link in AUDITED if link.case == "off-topic" and link.a2.verdict is None]
    assert OUTBOUND[alone.source] <= q3, "the off-topic-alone source must not be saturated"


def test_only_the_archive_is_dense_enough_to_be_a_listing() -> None:
    density = {path: 100 * n / PAGE_BY_PATH[path].words for path, n in OUTBOUND.items()}
    _, fence = upper_fence(list(density.values()))
    assert {path for path, found in density.items() if found > fence} == LISTINGS
    assert density[ARCHIVE] > 1.5 * fence
    assert max(found for path, found in density.items() if path not in LISTINGS) < 0.8 * fence


def test_planted_wasted_links_are_in_the_top_equity_quartile_and_the_others_are_not() -> None:
    q3 = float(np.percentile([equity(link) for link in AUDITED if not link.unverified], 75))
    for link in AUDITED:
        if link.case in WASTE_CASES:
            wasted = WASTED in link.a2.flags
            assert (equity(link) >= q3 * 1.5) if wasted else (equity(link) <= q3 * 0.75), (
                f"{link.case} {link.source} -> {link.target}: equity {equity(link):.3f}, Q3 {q3:.3f}"
            )


def test_the_stored_scores_split_exactly_the_planted_weak_links_off() -> None:
    with_context = [link for link in AUDITED if link.context is not None and not link.unverified]
    with_fit = [link for link in AUDITED if link.fit is not None and not link.generic]
    context_split = mixture_split(np.array([link.context for link in with_context]))
    fit_split = mixture_split(np.array([link.fit for link in with_fit]))
    assert context_split is not None
    assert fit_split is not None
    off = {link for link in AUDITED if OFF in link.a2.flags}
    assert {link for link in with_context if link.context < context_split} == off  # type: ignore[operator]
    weak = {link for link in with_fit if link.fit < fit_split}  # type: ignore[operator]
    assert weak == {link for link in with_fit if link.fit < 0.5}  # type: ignore[operator]
    assert off < weak


def test_five_identical_exact_anchors_overweight_one_target_and_four_do_not() -> None:
    exact: defaultdict[str, Counter[str]] = defaultdict(Counter)
    descriptive: Counter[str] = Counter()
    for link in AUDITED:
        if link.source in LISTINGS or link.unverified:
            continue
        keyword = PAGE_BY_PATH[link.target].keyword
        descriptive[link.target] += not link.generic
        if keyword and normalise_anchor(link.anchor) == keyword:
            exact[link.target][keyword] += 1
    most = {target: max(counts.values()) for target, counts in exact.items()}
    over, control = topic(OVER_OPTIMISED_TARGET), topic(CONTROL_TARGET)
    assert most[over] == 5
    assert most[over] * 2 >= descriptive[over]
    assert most[control] == 4
    assert max(n for target, n in most.items() if target != over) == 4
    flagged = {link for link in AUDITED if OVER in link.a2.flags}
    assert len(flagged) == 5
    assert {link.target for link in flagged} == {over}
    # The archive's exact title into the target would make six, and it must not count.
    assert any(
        link.source == ARCHIVE and link.target == over and OVER not in link.a2.flags
        for link in AUDITED
    )


def test_misaligned_links_share_no_stem_with_the_target_keyword_and_the_rest_do() -> None:
    for link in AUDITED:
        target = PAGE_BY_PATH.get(link.target)
        if target is None or target.keyword is None or link.generic:
            continue
        shared = stems(link.anchor) & stems(target.keyword)
        assert (MISALIGNED in link.a1.flags) == (not shared), link


def test_every_anchor_is_located_and_only_planted_proposals_are_in_the_copy() -> None:
    for page in PAGES:
        outgoing = [link for link in LINKS if link.source == page.path]
        spans, lost = locate_anchors(page.body, [(link.anchor, link.sentence) for link in outgoing])
        assert lost == 0, page.path
        free = page.body
        for start, end in spans:
            free = free[:start] + " " * (end - start) + free[end:]
        for link in outgoing:
            target = PAGE_BY_PATH.get(link.target)
            flagged = link.a2.flags | link.a1.flags or link.a2.verdict or link.a1.verdict
            if target is None or target.keyword is None or not flagged or link.skipped:
                continue
            proposals = {link.a2.proposal, link.a1.proposal} - {None}
            written = target.keyword in free.casefold()
            assert written == bool(proposals), f"{link.case} {link.source} -> {link.target}"
            assert all(free.count(proposal) == 1 for proposal in proposals)
