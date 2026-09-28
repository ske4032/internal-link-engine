"""Pair signals: overlap of GSC queries and of strategic keywords, and cluster membership.

Queries and keywords are sets of comparable size, so overlap is Jaccard, not cosine. Every
signal is a ranker input: a pair with no overlap still goes on with a weak signal.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Final

import structlog

from linking_engine.gsc import normalise_term
from linking_engine.models import (
    ClusterAgreement,
    PageSignals,
    PairSignals,
    SignalReport,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from linking_engine.models import CandidateSet, CommunityContext

log = structlog.get_logger(__name__)

HUB_NOISE: Final = -1
STAGE: Final = "pair-signals"


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def same_cluster(a: int | None, b: int | None, *, noise: int | None = None) -> bool | None:
    """None when either page is outside the clustering; noise never matches, not even noise."""
    if a is None or b is None:
        return None
    if noise is not None and noise in (a, b):
        return False
    return a == b


def agreement(
    topic_a: int | None, topic_b: int | None, link_a: int | None, link_b: int | None
) -> ClusterAgreement:
    if topic_a is None or topic_b is None:
        return ClusterAgreement.UNKNOWN_TOPIC
    same_links = link_a is not None and link_a == link_b
    if topic_a == topic_b:
        return (
            ClusterAgreement.SAME_TOPIC_SAME_LINKS
            if same_links
            else ClusterAgreement.SAME_TOPIC_OTHER_LINKS
        )
    return (
        ClusterAgreement.OTHER_TOPIC_SAME_LINKS
        if same_links
        else ClusterAgreement.OTHER_TOPIC_OTHER_LINKS
    )


def build_page_signals(
    pages: Sequence[CommunityContext],
    queries: Iterable[tuple[str, str]],
    keywords: Iterable[tuple[str, str]],
) -> dict[str, PageSignals]:
    """One entry per crawled page from its GSC queries and strategic keywords, as (url, term)
    rows; rows of any other url are ignored. Without a single GSC row for the tenant the
    keyword gap is unknown rather than every keyword."""
    known = {page.url for page in pages}
    page_queries: defaultdict[str, set[str]] = defaultdict(set)
    has_gsc = False
    for url, query in queries:
        has_gsc = True
        if url in known and (term := normalise_term(query)):
            page_queries[url].add(term)
    strategic: defaultdict[str, set[str]] = defaultdict(set)
    for url, keyword in keywords:
        if url in known and (term := normalise_term(keyword)):
            strategic[url].add(term)
    return {
        page.url: PageSignals(
            url=page.url,
            queries=frozenset(page_queries[page.url]),
            keywords=frozenset(strategic[page.url]),
            keyword_gap=len(strategic[page.url] - page_queries[page.url]) if has_gsc else None,
            link_community_id=page.link_community_id,
            keyword_community_id=page.keyword_community_id,
            content_community_id=page.content_community_id,
            hub_id=page.hub_id,
        )
        for page in pages
    }


def pair_signals(source: PageSignals, target: PageSignals) -> PairSignals:
    return PairSignals(
        query_overlap=jaccard(source.queries, target.queries),
        keyword_overlap=jaccard(source.keywords, target.keywords),
        same_link_community=same_cluster(source.link_community_id, target.link_community_id),
        same_keyword_community=same_cluster(
            source.keyword_community_id, target.keyword_community_id
        ),
        same_content_community=same_cluster(
            source.content_community_id, target.content_community_id
        ),
        same_hub=same_cluster(source.hub_id, target.hub_id, noise=HUB_NOISE),
        cluster_agreement=agreement(
            source.keyword_community_id,
            target.keyword_community_id,
            source.link_community_id,
            target.link_community_id,
        ),
        content_agreement=agreement(
            source.content_community_id,
            target.content_community_id,
            source.link_community_id,
            target.link_community_id,
        ),
    )


def signal_report(
    tenant_id: str,
    pages: Mapping[str, PageSignals],
    candidates: CandidateSet,
    *,
    unmatched_query_urls: int,
) -> SignalReport:
    """Every candidate pair through `pair_signals`, summarised."""
    started = time.perf_counter()
    pairs = 0
    query_total = keyword_total = 0.0
    query_pairs = keyword_pairs = same_hub = noise = 0
    by_cluster: Counter[ClusterAgreement] = Counter()
    by_content: Counter[ClusterAgreement] = Counter()
    for entry in candidates.targets:
        target = _page(pages, entry.target_url)
        for source_url in entry.sources:
            source = _page(pages, source_url)
            found = pair_signals(source, target)
            pairs += 1
            query_total += found.query_overlap
            keyword_total += found.keyword_overlap
            query_pairs += found.query_overlap > 0
            keyword_pairs += found.keyword_overlap > 0
            same_hub += found.same_hub is True
            noise += HUB_NOISE in (source.hub_id, target.hub_id)
            by_cluster[found.cluster_agreement] += 1
            by_content[found.content_agreement] += 1
    report = SignalReport(
        tenant_id=tenant_id,
        pages=len(pages),
        pages_with_queries=sum(1 for page in pages.values() if page.queries),
        pages_with_keywords=sum(1 for page in pages.values() if page.keywords),
        pages_with_gap=sum(1 for page in pages.values() if page.keyword_gap),
        unmatched_query_urls=unmatched_query_urls,
        pairs=pairs,
        query_overlap_pairs=query_pairs,
        query_overlap_mean=query_total / pairs if pairs else None,
        keyword_overlap_pairs=keyword_pairs,
        keyword_overlap_mean=keyword_total / pairs if pairs else None,
        same_hub_pairs=same_hub,
        noise_pairs=noise,
        cluster_agreement={kind: by_cluster[kind] for kind in ClusterAgreement},
        content_agreement={kind: by_content[kind] for kind in ClusterAgreement},
        seconds=round(time.perf_counter() - started, 3),
    )
    log.info("signals.report", stage=STAGE, **report.model_dump(mode="json"))
    return report


def _page(pages: Mapping[str, PageSignals], url: str) -> PageSignals:
    try:
        return pages[url]
    except KeyError:
        raise ValueError(f"no page signals for candidate url {url!r}") from None
