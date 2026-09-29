"""Existing-link audit (Stage 1): every body link's scores, issue flags and verdict.

Pure: the pipeline reads the edges and measures the anchors with the anchor stage's tools, then
passes both in. Every threshold is derived from the tenant's own links and reported with how it
was derived, or why there is none.
"""

from __future__ import annotations

import posixpath
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final
from urllib.parse import SplitResult, parse_qsl, urlsplit

import numpy as np

from linking_engine.audit.relevance import MIN_SPLIT_SCORES, mixture_split, score_distribution
from linking_engine.models import (
    ActionType,
    AuditCutoff,
    AuditReason,
    IssueFlag,
    LinkAuditReport,
    LinkAuditResult,
)
from linking_engine.models.audit import TECHNICAL_FLAGS
from linking_engine.urls import OFFSET_PARAMS, PAGE_NUMBER_PARAMS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from datetime import datetime

    from linking_engine.models import AuditEdge, ScoreDistribution

# Keyword alignment is the stem Jaccard; with vectors, the phrase-keyword cosine beside it.
ALIGNMENT_JACCARD_SHARE: Final = 0.7
ALIGNMENT_COSINE_SHARE: Final = 0.3
# The composite's weights, renormalised over the dimensions present.
ALIGNMENT_WEIGHT: Final = 0.35
FIT_WEIGHT: Final = 0.35
CONTEXT_WEIGHT: Final = 0.30
GENERIC_CAP: Final = 20.0
OVER_OPTIMISED_MIN: Final = 5
OVER_OPTIMISED_SHARE: Final = 0.5
# Index-like and listing pages lie this many interquartile ranges above the Q3.
FENCE_IQRS: Final = 3.0
# A count moves in whole links, so a narrower spread is floored: pages that share one template's
# link count would otherwise make every page with a link more index-like.
MIN_IQR: Final = 1.0
HUB_NOISE: Final = -1
# Pagination parameters the url keys do not keep as page numbers or offsets.
EXTRA_PAGINATION_PARAMS: Final = frozenset({"p", "currentpage", "pageindex", "page_index"})
# Query parameters that make a url a page of a paginated listing, whatever their value.
PAGINATION_PARAMS: Final = PAGE_NUMBER_PARAMS | OFFSET_PARAMS | EXTRA_PAGINATION_PARAMS
# A path segment that numbers a page: page-2, or /page/2 and /p/2 as two segments.
_PAGE_SEGMENT: Final = re.compile(r"page-\d+")
_PAGE_PREFIXES: Final = frozenset({"page", "p"})

CONTEXT_SPLIT: Final = "context_relevance_split"
FIT_SPLIT: Final = "anchor_target_fit_split"
SATURATION: Final = "outbound_q3"
INDEX_FENCE: Final = "outbound_fence"
EQUITY_TOP: Final = "equity_q3"
DENSITY_FENCE: Final = "link_density_fence"
NO_STORED_SCORES: Final = (
    "no link has a stored context relevance; embed-links and score-links give A2"
)
SITEMAP_SKIPPED: Final = (
    "sitemap pages list the site's pages by design, so neither their links nor the links into "
    "them are evaluated"
)
PAGINATED_SKIPPED: Final = (
    "paginated pages are natural link hubs, so neither their links nor the links into them are "
    "evaluated"
)
_REANCHOR_FLAGS: Final = frozenset(
    {IssueFlag.GENERIC, IssueFlag.MISALIGNED, IssueFlag.OVER_OPTIMISED}
)
# Flags that ask for a new anchor with or without a better phrase in the copy.
_ANCHOR_DEFECTS: Final = frozenset({IssueFlag.GENERIC, IssueFlag.OVER_OPTIMISED})
# Flags a listing source keeps without a REANCHOR or REMOVE verdict.
_LISTING_KEPT: Final = frozenset(
    {IssueFlag.GENERIC, IssueFlag.MISALIGNED, IssueFlag.OFF_TOPIC, IssueFlag.WASTED_EQUITY}
)
_NOT_ALNUM: Final = re.compile(r"[^0-9a-z]+")
_SCHEME: Final = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AnchorFacts:
    """What the pipeline measured of one edge's anchor with the anchor stage's tools."""

    # normalise_anchor of the text; None when it normalises to nothing.
    key: str | None
    generic: bool
    # The anchor is the words of the target's primary keyword (#23's EXACT type).
    exact_match: bool
    # The best stem Jaccard over the target's ranked keywords; None when the target has none
    # or the anchor has no key.
    keyword_jaccard: float | None
    # The anchor's cosine to that keyword on [0, 1]; None without both vectors.
    keyword_cosine: float | None = None


@dataclass(frozen=True, slots=True)
class Proposal:
    """A phrase the anchor ladder found for the target in the source copy, measured as the
    anchors are."""

    phrase: str
    key: str
    keyword_jaccard: float | None
    keyword_cosine: float | None
    anchor_target_fit: float | None


@dataclass(frozen=True, slots=True)
class AuditScope:
    """The edges to audit, and the natural link hubs left out: sitemap pages, then paginated
    pages, and the links from or into them."""

    edges: tuple[AuditEdge, ...]
    sitemap_pages: int = 0
    sitemap_links: int = 0
    paginated_pages: int = 0
    paginated_links: int = 0


@dataclass(frozen=True, slots=True)
class Assessed:
    """One edge's scores, flags and findings before the ladder's proposals."""

    edge: AuditEdge
    facts: AnchorFacts
    flags: frozenset[IssueFlag]
    reasons: tuple[tuple[AuditReason, str], ...]
    keyword_alignment: float | None
    # The stored fit, None for a generic anchor: a fit stored before the tenant's generic rules
    # changed does not count.
    anchor_target_fit: float | None
    equity: float | None
    quality: float | None
    fix: bool
    fix_target: str | None
    remove: bool
    # Why the anchor fits its target weakly; None when it does not, or without a fit split.
    weak_fit: str | None
    # Generic, misaligned, over-optimised or weakly fitting, and neither FIX nor REMOVE nor
    # from a listing source.
    reanchor: bool


@dataclass(frozen=True, slots=True)
class Assessment:
    rows: tuple[Assessed, ...]
    cutoffs: tuple[AuditCutoff, ...]
    source_pages: int
    index_like_pages: int
    # Why A2 did not run; None when it did.
    embeddings_skipped_reason: str | None
    listing_pages: int = 0
    scope: AuditScope | None = None


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    results: tuple[LinkAuditResult, ...]
    by_reason: dict[AuditReason, int]
    proposals: int


@dataclass(frozen=True, slots=True)
class _Tenant:
    """The tenant's cut-offs and per-page facts every edge is judged against."""

    context_split: float | None
    fit_split: float | None
    saturation: float | None
    fence: float | None
    equity_top: float | None
    outbound: Mapping[str, int]
    # (target, anchor key) of each over-optimised block -> (its links, the target's
    # descriptive inbound anchors), counted over links from sources that are not listings.
    blocks: Mapping[tuple[str, str], tuple[int, int]]
    # Listing sources -> why each is one.
    listings: Mapping[str, str]


def _url_parts(url: str) -> SplitResult | None:
    """The url's parts. A stored key has no scheme ("host/path"), so it is split as
    network-path reference, which keeps its host out of the path; a bare path stays a path.
    None for a url that cannot be split."""
    text = url.strip()
    try:
        return urlsplit(text if _SCHEME.match(text) or text.startswith("/") else f"//{text}")
    except ValueError:
        return None


def is_sitemap(url: str) -> bool:
    """A path segment naming a sitemap, case-insensitive: /sitemap, /sitemap.html,
    /html-sitemap, /site-map; the host never counts."""
    parts = _url_parts(url)
    if parts is None:
        return False
    for segment in parts.path.lower().split("/"):
        words = [word for word in _NOT_ALNUM.split(posixpath.splitext(segment)[0]) if word]
        if "sitemap" in words or ("site", "map") in pairwise(words):
            return True
    return False


def is_pagination(url: str) -> bool:
    """A page of a paginated listing: a PAGINATION_PARAMS query parameter with any value, or a
    /page/<n>, /page-<n> or /p/<n> path segment; the host never counts."""
    parts = _url_parts(url)
    if parts is None:
        return False
    if any(
        name.strip().lower() in PAGINATION_PARAMS
        for name, _ in parse_qsl(parts.query, keep_blank_values=True)
    ):
        return True
    segments = [segment for segment in parts.path.lower().split("/") if segment]
    return any(_PAGE_SEGMENT.fullmatch(segment) for segment in segments) or any(
        first in _PAGE_PREFIXES and second.isdigit() for first, second in pairwise(segments)
    )


def audit_scope(edges: Sequence[AuditEdge]) -> AuditScope:
    """The edges between pages that are neither sitemaps nor paginated, and how many of each
    kind of page and of the links from or into them are left out; a paginated sitemap page, and
    a link between the two kinds, count as sitemap."""
    urls = {url for edge in edges for url in (edge.source_url, edge.target_url)}
    sitemaps = {url for url in urls if is_sitemap(url)}
    paginated = {url for url in urls - sitemaps if is_pagination(url)}
    kept: list[AuditEdge] = []
    sitemap_links = paginated_links = 0
    for edge in edges:
        ends = (edge.source_url, edge.target_url)
        if any(url in sitemaps for url in ends):
            sitemap_links += 1
        elif any(url in paginated for url in ends):
            paginated_links += 1
        else:
            kept.append(edge)
    return AuditScope(tuple(kept), len(sitemaps), sitemap_links, len(paginated), paginated_links)


def _in_scope(url: str) -> bool:
    return not is_sitemap(url) and not is_pagination(url)


def keyword_alignment(jaccard: float | None, cosine: float | None) -> float | None:
    """The stem Jaccard, blended with the phrase-keyword cosine when there is one."""
    if jaccard is None:
        return None
    if cosine is None:
        return jaccard
    return min(1.0, ALIGNMENT_JACCARD_SHARE * jaccard + ALIGNMENT_COSINE_SHARE * cosine)


def anchor_quality(
    alignment: float | None, fit: float | None, context: float | None, *, generic: bool
) -> float | None:
    """The 0-100 composite over the dimensions present; a generic anchor is capped."""
    parts = [
        (weight, value)
        for weight, value in (
            (ALIGNMENT_WEIGHT, alignment),
            (FIT_WEIGHT, fit),
            (CONTEXT_WEIGHT, context),
        )
        if value is not None
    ]
    if not parts:
        return None
    score = 100 * sum(weight * value for weight, value in parts) / sum(w for w, _ in parts)
    score = min(100.0, max(0.0, score))
    return min(score, GENERIC_CAP) if generic else score


def equity_efficiency(
    source_percentile: float | None, rank: int, links: int, target_percentile: float | None
) -> float | None:
    """The source's PageRank percentile, weighted by how early the link sits among the source's
    ``links`` (``rank`` 0 first), times the target's need; None without both percentiles."""
    if source_percentile is None or target_percentile is None:
        return None
    return source_percentile * (1 - rank / links) * (1 - target_percentile)


def assess(
    edges: Sequence[AuditEdge],
    facts: Sequence[AnchorFacts],
    *,
    scope: AuditScope | None = None,
) -> Assessment:
    """Scores, flags and findings of every edge, against cut-offs from the tenant's own links.
    Sitemap and paginated links are left out beforehand by ``audit_scope``, whose counts
    ``scope`` carries into the report."""
    if len(edges) != len(facts):
        raise ValueError("one AnchorFacts per edge")
    urls = {url for edge in edges for url in (edge.source_url, edge.target_url)}
    if not all(_in_scope(url) for url in urls):
        raise ValueError(
            "sitemap and paginated links are left out by audit_scope before the assessment"
        )
    outbound = Counter(edge.source_url for edge in edges)
    listings, density = _listings(edges, outbound)
    verified = [
        (edge, fact) for edge, fact in zip(edges, facts, strict=True) if not edge.target_placeholder
    ]
    ranks = _position_ranks(edges)
    saturation, fence = _outbound_cutoffs(list(outbound.values()))
    a2 = any(edge.context_relevance is not None for edge, _ in verified)
    skipped = None if a2 else NO_STORED_SCORES
    context = _split(
        CONTEXT_SPLIT,
        [edge.context_relevance for edge, _ in verified if edge.context_relevance is not None],
        skipped,
    )
    fit = _split(
        FIT_SPLIT,
        [
            edge.anchor_target_fit
            for edge, fact in verified
            if edge.anchor_target_fit is not None and not fact.generic
        ],
        skipped,
    )
    equities = {
        (edge.source_url, edge.position): found
        for edge, _ in verified
        if (
            found := equity_efficiency(
                edge.source_page_rank_percentile,
                ranks[(edge.source_url, edge.position)],
                outbound[edge.source_url],
                edge.target_page_rank_percentile,
            )
        )
        is not None
    }
    equity = _equity_cutoff(list(equities.values()))
    tenant = _Tenant(
        context_split=context.value,
        fit_split=fit.value,
        saturation=saturation.value,
        fence=fence.value,
        equity_top=equity.value,
        outbound=outbound,
        blocks=_over_optimised(
            [(edge, fact) for edge, fact in verified if edge.source_url not in listings]
        ),
        listings=listings,
    )
    rows = tuple(
        _unverified(edge, fact)
        if edge.target_placeholder
        else _assess_one(edge, fact, equities.get((edge.source_url, edge.position)), tenant)
        for edge, fact in zip(edges, facts, strict=True)
    )
    return Assessment(
        rows=rows,
        cutoffs=(context, fit, saturation, fence, equity, density),
        source_pages=len(outbound),
        index_like_pages=sum(
            1 for count in outbound.values() if fence.value is not None and count > fence.value
        ),
        embeddings_skipped_reason=skipped,
        listing_pages=len(listings),
        scope=scope,
    )


def ladder_pairs(assessment: Assessment) -> frozenset[tuple[str, str]]:
    """(source, target) of every edge that wants a better anchor, for the ladder to search."""
    return frozenset(
        (row.edge.source_url, row.edge.target_url)
        for row in assessment.rows
        if row.reanchor and row.edge.source_url != row.edge.target_url
    )


def decide(
    assessment: Assessment,
    proposals: Mapping[tuple[str, str], Sequence[Proposal]],
    *,
    run_id: str,
    audited_at: datetime,
) -> AuditOutcome:
    """Every edge's verdict, FIX over REMOVE over REANCHOR: REANCHOR takes the first of the
    pair's ``proposals`` that is better than the anchor, and a generic or over-optimised anchor
    is REANCHOR even when the copy has none."""
    results: list[LinkAuditResult] = []
    by_reason: Counter[AuditReason] = Counter()
    proposed = 0
    for row in assessment.rows:
        reasons = list(row.reasons)
        verdict: ActionType | None = None
        phrase: str | None = None
        if row.fix:
            verdict = ActionType.FIX
        elif row.remove:
            verdict = ActionType.REMOVE
        elif row.reanchor:
            better = _better(row, proposals.get((row.edge.source_url, row.edge.target_url), ()))
            if better is not None:
                verdict, phrase = ActionType.REANCHOR, better.phrase
                if row.weak_fit is not None and not row.flags & _REANCHOR_FLAGS:
                    reasons.append((AuditReason.WEAK_FIT, f"the anchor {row.weak_fit}"))
                reasons.append(
                    (
                        AuditReason.BETTER_PHRASE,
                        f'a better phrase for the target is in the same copy: "{better.phrase}"',
                    )
                )
                proposed += 1
            elif row.flags & _REANCHOR_FLAGS:
                reasons.append(
                    (AuditReason.NO_BETTER_PHRASE, "no better phrase for the target in the copy")
                )
                if row.flags & _ANCHOR_DEFECTS:
                    verdict = ActionType.REANCHOR
        by_reason.update(reason for reason, _ in reasons)
        results.append(
            LinkAuditResult(
                source_url=row.edge.source_url,
                position=row.edge.position,
                target_url=row.edge.target_url,
                run_id=run_id,
                anchor_quality_score=row.quality,
                keyword_alignment=row.keyword_alignment,
                context_relevance=None
                if row.edge.target_placeholder
                else row.edge.context_relevance,
                anchor_target_fit=row.anchor_target_fit,
                equity_efficiency=row.equity,
                issue_flags=row.flags,
                verdict=verdict,
                reasons=tuple(words for _, words in reasons),
                proposed_anchor=phrase,
                fix_target=row.fix_target,
                unverified=row.edge.target_placeholder,
                audited_at=audited_at,
            )
        )
    return AuditOutcome(tuple(results), dict(by_reason), proposed)


def audit_report(
    tenant_id: str,
    run_id: str,
    assessment: Assessment,
    outcome: AuditOutcome,
    *,
    ladder_pairs: int,
    vectors_skipped_reason: str | None,
    seconds: float,
    finished_at: datetime,
) -> LinkAuditReport:
    """The run's counts, cut-offs and score distributions."""
    results = outcome.results
    scope = assessment.scope or AuditScope(tuple(row.edge for row in assessment.rows))
    verified = [result for result in results if not result.unverified]
    return LinkAuditReport(
        tenant_id=tenant_id,
        run_id=run_id,
        links=len(results),
        unverified=len(results) - len(verified),
        source_pages=assessment.source_pages,
        index_like_pages=assessment.index_like_pages,
        listing_pages=assessment.listing_pages,
        sitemap_pages=scope.sitemap_pages,
        sitemap_links=scope.sitemap_links,
        paginated_pages=scope.paginated_pages,
        paginated_links=scope.paginated_links,
        by_flag=dict(Counter(flag for result in results for flag in result.issue_flags)),
        by_verdict=dict(Counter(r.verdict for r in results if r.verdict is not None)),
        by_reason=outcome.by_reason,
        healthy=sum(1 for result in verified if result.verdict is None),
        ladder_pairs=ladder_pairs,
        proposals=outcome.proposals,
        cutoffs=assessment.cutoffs,
        embeddings=assessment.embeddings_skipped_reason is None,
        embeddings_skipped_reason=assessment.embeddings_skipped_reason,
        keyword_cosines=sum(
            1
            for row in assessment.rows
            if row.keyword_alignment is not None and row.facts.keyword_cosine is not None
        ),
        vectors_skipped_reason=vectors_skipped_reason,
        keyword_alignment=_distribution(r.keyword_alignment for r in verified),
        context_relevance=_distribution(r.context_relevance for r in verified),
        anchor_target_fit=_distribution(r.anchor_target_fit for r in verified),
        equity_efficiency=_distribution(r.equity_efficiency for r in verified),
        anchor_quality=_distribution(
            None if r.anchor_quality_score is None else r.anchor_quality_score / 100
            for r in verified
        ),
        seconds=seconds,
        finished_at=finished_at,
    )


def _better(row: Assessed, proposals: Sequence[Proposal]) -> Proposal | None:
    """The first proposal that is not the anchor itself: any such phrase replaces a generic or
    over-optimised anchor, and a misaligned or weakly fitting one only for a phrase that aligns
    with the target's keywords or fits the target better."""
    fit = row.anchor_target_fit
    for proposal in proposals:
        if proposal.key == row.facts.key:
            continue
        if row.flags & _ANCHOR_DEFECTS:
            return proposal
        alignment = keyword_alignment(proposal.keyword_jaccard, proposal.keyword_cosine)
        aligns = (
            alignment is not None
            and row.keyword_alignment is not None
            and alignment > row.keyword_alignment
        )
        fits = (
            proposal.anchor_target_fit is not None
            and fit is not None
            and proposal.anchor_target_fit > fit
        )
        if aligns or fits:
            return proposal
    return None


def _distribution(values: Iterable[float | None]) -> ScoreDistribution | None:
    return score_distribution([value for value in values if value is not None])


def _position_ranks(edges: Sequence[AuditEdge]) -> dict[tuple[str, int], int]:
    """Each edge's place among its source's body links, 0 first."""
    positions: defaultdict[str, list[int]] = defaultdict(list)
    for edge in edges:
        positions[edge.source_url].append(edge.position)
    return {
        (source, position): rank
        for source, found in positions.items()
        for rank, position in enumerate(sorted(found))
    }


def _upper_fence(
    name: str, values: Sequence[float], what: str, unit: str
) -> tuple[float, AuditCutoff]:
    """The Q3 of ``values`` and the fence Q3 + FENCE_IQRS x IQR above it, the IQR floored at
    MIN_IQR ``unit``."""
    q1, q3 = (float(q) for q in np.percentile(np.asarray(values, dtype=np.float64), [25, 75]))
    iqr = q3 - q1
    floored = f"; IQR {iqr:g} floored to {MIN_IQR:g} {unit}" if iqr < MIN_IQR else ""
    return q3, AuditCutoff(
        name=name,
        value=q3 + FENCE_IQRS * max(iqr, MIN_IQR),
        reason=f"Q3 + {FENCE_IQRS:g} x IQR (Q1 {q1:g}, Q3 {q3:g}) {what}{floored}",
    )


def _outbound_cutoffs(counts: Sequence[int]) -> tuple[AuditCutoff, AuditCutoff]:
    """A source above the outbound Q3 is saturated; one above Q3 + FENCE_IQRS x IQR, the IQR
    at least MIN_IQR, is index-like."""
    if not counts:
        reason = "no source page has body links"
        return (
            AuditCutoff(name=SATURATION, value=None, reason=reason),
            AuditCutoff(name=INDEX_FENCE, value=None, reason=reason),
        )
    over = f"of the outbound body links of {len(counts)} source pages"
    q3, fence = _upper_fence(INDEX_FENCE, counts, over, "link")
    return AuditCutoff(name=SATURATION, value=q3, reason=f"Q3 {over}"), fence


def _listings(
    edges: Sequence[AuditEdge], outbound: Mapping[str, int]
) -> tuple[dict[str, str], AuditCutoff]:
    """Listing and archive sources, those whose link density lies above the tenant's fence,
    with why, and the fence. Density is body links per 100 words; a source without a word
    count has none."""
    words: dict[str, int] = {}
    for edge in edges:
        if edge.source_word_count and edge.source_url not in words:
            words[edge.source_url] = edge.source_word_count
    density = {url: 100 * outbound[url] / count for url, count in words.items()}
    if density:
        _, fence = _upper_fence(
            DENSITY_FENCE,
            list(density.values()),
            f"of the body links per 100 words of {len(density)} source pages",
            "link per 100 words",
        )
    else:
        fence = AuditCutoff(
            name=DENSITY_FENCE, value=None, reason="no source page has a word count"
        )
    if fence.value is None:
        return {}, fence
    return {
        url: f"the source is a listing page ({found:.1f} links per 100 words, above the "
        f"tenant's fence of {fence.value:.1f})"
        for url, found in density.items()
        if found > fence.value
    }, fence


def _equity_cutoff(values: Sequence[float]) -> AuditCutoff:
    if not values:
        return AuditCutoff(
            name=EQUITY_TOP, value=None, reason="no link has PageRank on both of its pages"
        )
    return AuditCutoff(
        name=EQUITY_TOP,
        value=float(np.percentile(np.asarray(values, dtype=np.float64), 75)),
        reason=f"Q3 of the equity efficiency of {len(values)} links",
    )


def _split(name: str, scores: Sequence[float], skipped: str | None) -> AuditCutoff:
    """The boundary of the tenant's weak mode of one score, or why there is none."""
    if skipped is not None:
        return AuditCutoff(name=name, value=None, reason=f"A1 only: {skipped}")
    if len(scores) < MIN_SPLIT_SCORES:
        return AuditCutoff(
            name=name,
            value=None,
            reason=f"{len(scores)} scores, fewer than {MIN_SPLIT_SCORES}, give no split",
        )
    split = mixture_split(np.asarray(scores, dtype=np.float64))
    if split is None:
        return AuditCutoff(
            name=name,
            value=None,
            reason=f"the {len(scores)} scores do not separate into two modes",
        )
    return AuditCutoff(
        name=name,
        value=split,
        reason=f"two-component Gaussian mixture over {len(scores)} scores",
    )


def _over_optimised(
    verified: Sequence[tuple[AuditEdge, AnchorFacts]],
) -> dict[tuple[str, str], tuple[int, int]]:
    """Blocks of identical exact-match anchors into one target, at least OVER_OPTIMISED_MIN
    long and at least OVER_OPTIMISED_SHARE of the target's descriptive inbound anchors."""
    descriptive: Counter[str] = Counter()
    exact: Counter[tuple[str, str]] = Counter()
    for edge, fact in verified:
        if fact.key is None or fact.generic:
            continue
        descriptive[edge.target_url] += 1
        if fact.exact_match:
            exact[(edge.target_url, fact.key)] += 1
    return {
        block: (count, descriptive[block[0]])
        for block, count in exact.items()
        if count >= OVER_OPTIMISED_MIN and count >= OVER_OPTIMISED_SHARE * descriptive[block[0]]
    }


def _unverified(edge: AuditEdge, fact: AnchorFacts) -> Assessed:
    return Assessed(
        edge=edge,
        facts=fact,
        flags=frozenset(),
        reasons=(
            (
                AuditReason.UNVERIFIED_TARGET,
                "the target was not crawled, so the link is unverified",
            ),
        ),
        keyword_alignment=None,
        anchor_target_fit=None,
        equity=None,
        quality=None,
        fix=False,
        fix_target=None,
        remove=False,
        weak_fit=None,
        reanchor=False,
    )


def _technical(edge: AuditEdge) -> list[tuple[IssueFlag, AuditReason, str]]:
    found: list[tuple[IssueFlag, AuditReason, str]] = []
    status = edge.target_status_code
    if status is not None and status >= 400:
        found.append(
            (
                IssueFlag.BROKEN,
                AuditReason.BROKEN_TARGET,
                f"the target answers {status}, so the link is broken",
            )
        )
    elif status is not None and status >= 300:
        found.append(
            (
                IssueFlag.REDIRECTED,
                AuditReason.REDIRECTED_TARGET,
                f"the target redirects ({status}): link to the page it redirects to",
            )
        )
    elif edge.target_indexable is False:
        found.append(
            (IssueFlag.NOINDEX_TARGET, AuditReason.NOINDEX_TARGET, "the target is not indexable")
        )
    if not edge.is_follow:
        found.append(
            (
                IssueFlag.NOFOLLOW,
                AuditReason.NOFOLLOW,
                "the link is nofollow, so it passes no equity",
            )
        )
    return found


def _not_indexable(edge: AuditEdge) -> bool:
    status = edge.target_status_code
    return edge.target_indexable is False or (status is not None and not 200 <= status < 300)


def _canonical(edge: AuditEdge) -> tuple[str | None, str | None]:
    """The fix target and the reason when the target is a non-canonical copy."""
    canonical = edge.target_canonical_url
    if canonical is None:
        return None, None
    if canonical == edge.source_url:
        return None, "the target is a duplicate copy of this page: the link points at itself"
    return canonical, "the target is a duplicate copy: link to its canonical page instead"


def _assess_one(
    edge: AuditEdge, fact: AnchorFacts, equity: float | None, tenant: _Tenant
) -> Assessed:
    found: list[tuple[IssueFlag | None, AuditReason, str]] = list(_technical(edge))
    fix_target, copy = _canonical(edge)
    if copy is not None:
        found.append((None, AuditReason.NON_CANONICAL_TARGET, copy))
    if fact.generic:
        found.append(
            (
                IssueFlag.GENERIC,
                AuditReason.GENERIC_ANCHOR,
                "the anchor text is generic and says nothing of the target",
            )
        )

    fit = None if fact.generic else edge.anchor_target_fit
    context = edge.context_relevance
    weak_fit = (
        f"fits the target weakly ({fit:.2f}, below the tenant's split {tenant.fit_split:.2f})"
        if fit is not None and tenant.fit_split is not None and fit < tenant.fit_split
        else None
    )
    # A synonym anchor that fits its target well is not misaligned; without a fit, or without
    # a split to judge it by, sharing no stem is enough.
    misaligned = (
        not fact.generic
        and fact.keyword_jaccard == 0
        and (weak_fit is not None or fit is None or tenant.fit_split is None)
    )
    if misaligned:
        words = "the anchor shares no word stem with the target's keywords"
        found.append(
            (
                IssueFlag.MISALIGNED,
                AuditReason.MISALIGNED_ANCHOR,
                words if weak_fit is None else f"{words} and {weak_fit}",
            )
        )
    listing = tenant.listings.get(edge.source_url)
    if (
        listing is None
        and fact.key is not None
        and (block := tenant.blocks.get((edge.target_url, fact.key)))
    ):
        count, descriptive = block
        found.append(
            (
                IssueFlag.OVER_OPTIMISED,
                AuditReason.OVER_OPTIMISED_ANCHOR,
                f"{count} links into the target use this exact-match anchor, "
                f"{count / descriptive:.0%} of its descriptive anchors",
            )
        )
    off_topic = weak_fit is not None and _weak(context, tenant.context_split)
    if off_topic:
        found.append(
            (
                IssueFlag.OFF_TOPIC,
                AuditReason.OFF_TOPIC,
                f"the sentence ({context:.2f}) and the anchor ({fit:.2f}) both fall in the "
                f"tenant's weak modes, below {tenant.context_split:.2f} and "
                f"{tenant.fit_split:.2f}",
            )
        )
    wasted = _wasted(edge, equity, tenant, off_topic=off_topic)
    if wasted is not None:
        found.append((IssueFlag.WASTED_EQUITY, AuditReason.WASTED_EQUITY, wasted))

    flags = frozenset(flag for flag, _, _ in found if flag is not None)
    fix = bool(flags & TECHNICAL_FLAGS) or copy is not None
    if listing is not None and not fix and flags & _LISTING_KEPT:
        found.append(
            (
                None,
                AuditReason.LISTING_SOURCE,
                f"{listing}: its links are never reanchored or removed",
            )
        )
    removal = (
        None
        if fix or listing is not None or not (off_topic and misaligned)
        else _removal(edge, tenant, wasted=wasted is not None)
    )
    if removal is not None:
        found.append((None, *removal))
    remove = removal is not None and removal[0] is AuditReason.REMOVE_OFF_TOPIC
    alignment = keyword_alignment(fact.keyword_jaccard, fact.keyword_cosine)
    return Assessed(
        edge=edge,
        facts=fact,
        flags=flags,
        reasons=tuple((reason, words) for _, reason, words in found),
        keyword_alignment=alignment,
        anchor_target_fit=fit,
        equity=equity,
        quality=anchor_quality(alignment, fit, context, generic=fact.generic),
        fix=fix,
        fix_target=fix_target,
        remove=remove,
        weak_fit=weak_fit,
        reanchor=not fix
        and not remove
        and listing is None
        and (bool(flags & _REANCHOR_FLAGS) or weak_fit is not None),
    )


def _weak(score: float | None, split: float | None) -> bool:
    return score is not None and split is not None and score < split


def _removal(edge: AuditEdge, tenant: _Tenant, *, wasted: bool) -> tuple[AuditReason, str] | None:
    """Whether an off-topic, misaligned link goes: when it wastes equity or its source is
    saturated, and never from an index-like source; None when neither holds."""
    outbound = tenant.outbound[edge.source_url]
    saturated = tenant.saturation is not None and outbound > tenant.saturation
    if not (wasted or saturated):
        return None
    if tenant.fence is not None and outbound > tenant.fence:
        return (
            AuditReason.INDEX_LIKE_SOURCE_KEPT,
            f"the source is index-like ({outbound} outbound links, above the tenant's fence of "
            f"{tenant.fence:g}), so its links are never removed",
        )
    why = (
        "it wastes equity"
        if wasted
        else f"its source has {outbound} outbound links, above the tenant's Q3 of "
        f"{tenant.saturation:g}"
    )
    return (
        AuditReason.REMOVE_OFF_TOPIC,
        f"off-topic, sharing no stem with the target's keywords, and {why}: remove the link",
    )


def _wasted(
    edge: AuditEdge, equity: float | None, tenant: _Tenant, *, off_topic: bool
) -> str | None:
    """Why a top-quartile link wastes its equity; None when it does not."""
    if equity is None or tenant.equity_top is None or equity <= tenant.equity_top:
        return None
    into = [
        what
        for what, holds in (
            ("an off-topic target", off_topic),
            ("a non-indexable target", _not_indexable(edge)),
            ("a page in no topic hub", edge.target_hub_id == HUB_NOISE),
        )
        if holds
    ]
    if not into:
        return None
    return (
        f"equity efficiency {equity:.2f} is in the tenant's top quartile (above "
        f"{tenant.equity_top:.2f}) and goes to {' and '.join(into)}"
    )
