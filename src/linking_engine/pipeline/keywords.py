"""Keyword resolution stage: every crawled 2xx page's target keyword, and the tenant's keywords
written to the graph as ``TARGETS_KEYWORD`` edges tagged with their source."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import structlog

from linking_engine.anchor.generic import generic_overrides
from linking_engine.anchor.keywords import (
    BRAND_SUFFIX_SHARE,
    MAX_KEYWORD_TOKENS,
    MAX_SECONDARY_QUERIES,
    MIN_QUERY_IMPRESSIONS,
    REPEATED_FALLBACK_PAGES,
    brand_affixes,
    fallback_texts,
    is_long,
    merge_strategic,
    repeated_fallbacks,
    resolve_page,
    secondary_queries,
    usable_strategic,
)
from linking_engine.gsc import (
    MIN_CURVE_IMPRESSIONS,
    MIN_CURVE_ROWS,
    fit_ctr_curve,
    normalise_term,
)
from linking_engine.models import KeywordReport, KeywordRung, KeywordSource, KeywordTarget

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import (
        CtrCurve,
        GscQueryStats,
        PageRecord,
        ResolvedKeyword,
        StrategicKeyword,
    )

log = structlog.get_logger(__name__)

STAGE: Final = "resolve-keywords"
_RUNG_SOURCE: Final = {
    KeywordRung.STRATEGIC: KeywordSource.CLIENT_STRATEGIC,
    KeywordRung.GSC: KeywordSource.GSC_OBSERVED,
    KeywordRung.H1: KeywordSource.INFERRED,
    KeywordRung.TITLE: KeywordSource.INFERRED,
}


def _resolvable(page: PageRecord) -> bool:
    return page.status_code is not None and 200 <= page.status_code <= 299


def ranked_targets(
    strategic: Sequence[StrategicKeyword],
    resolved: Mapping[str, ResolvedKeyword],
    secondaries: Mapping[str, tuple[str, tuple[str, ...]]],
) -> dict[KeywordSource, list[KeywordTarget]]:
    """Every keyword edge to write, by source. ``strategic`` is ``merge_strategic``'s output;
    ``secondaries`` maps a resolved page to its language and further GSC queries. A resolved
    page's keywords are ranked consecutively across the sources: 1 the resolved keyword, then
    its other strategic keywords, then the GSC queries. Unresolved pages' keywords get no rank."""
    targets: dict[KeywordSource, list[KeywordTarget]] = {
        KeywordSource.CLIENT_STRATEGIC: [],
        KeywordSource.GSC_OBSERVED: [],
        KeywordSource.INFERRED: [],
    }
    strategic_of: defaultdict[str, list[StrategicKeyword]] = defaultdict(list)
    for row in strategic:
        strategic_of[row.url].append(row)
    for url in sorted(strategic_of.keys() | resolved.keys()):
        keyword = resolved.get(url)
        rank = 1
        for row in strategic_of.get(url, []):
            is_resolved = (
                keyword is not None
                and keyword.rung is KeywordRung.STRATEGIC
                and normalise_term(keyword.text) == normalise_term(row.keyword)
                and keyword.language == row.language
            )
            if keyword is not None and not is_resolved:
                rank += 1
            targets[KeywordSource.CLIENT_STRATEGIC].append(
                KeywordTarget(
                    url=url,
                    text=row.keyword,
                    language=row.language,
                    source=KeywordSource.CLIENT_STRATEGIC,
                    priority=row.priority,
                    is_primary=row.is_primary,
                    rung=KeywordRung.STRATEGIC if is_resolved else None,
                    rank=None if keyword is None else 1 if is_resolved else rank,
                )
            )
        if keyword is not None and keyword.rung is not KeywordRung.STRATEGIC:
            source = _RUNG_SOURCE[keyword.rung]
            targets[source].append(
                KeywordTarget(
                    url=url,
                    text=keyword.text,
                    language=keyword.language,
                    source=source,
                    rung=keyword.rung,
                )
            )
        language, queries = secondaries.get(url, ("", ()))
        for text in queries:
            rank += 1
            targets[KeywordSource.GSC_OBSERVED].append(
                KeywordTarget(
                    url=url,
                    text=text,
                    language=language,
                    source=KeywordSource.GSC_OBSERVED,
                    rank=rank,
                )
            )
    return targets


@dataclass(frozen=True, slots=True)
class KeywordPlan:
    """Every keyword decision of a run, before anything is written."""

    pages: int
    resolved: Mapping[str, ResolvedKeyword]
    by_source: Mapping[KeywordSource, tuple[KeywordTarget, ...]]
    by_language: Mapping[str, int]
    fallbacks_rejected: Mapping[str, int]
    gsc_rejected: int
    gsc_rows: int
    curve: CtrCurve | None
    brand_prefix: str | None
    brand_suffix: str | None
    invalid_strategic_rows: int
    generic_add: frozenset[str]
    generic_remove: frozenset[str]


async def plan_keywords(mongo: MongoRepo, tenant_id: str) -> KeywordPlan:
    """Resolve every crawled 2xx page's keyword and rank each page's keyword set; read-only."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    rules = await mongo.get_language_rules(tenant_id)
    anchor_rules = await mongo.get_anchor_rules(tenant_id)
    generic_add, generic_remove = generic_overrides(
        anchor_rules.generic_add, anchor_rules.generic_remove
    )
    # Brand affixes and template fallbacks come from every page, so they are found before
    # any page is resolved.
    titles: list[str | None] = []
    fallbacks: list[tuple[tuple[str, ...], str]] = []
    async for batch in mongo.iter_page_records(tenant_id):
        for page in batch:
            if _resolvable(page):
                titles.append(page.meta_title)
                fallbacks.append((fallback_texts(page), page.body_hash))
    prefix, suffix = brand_affixes(titles)
    repeated = repeated_fallbacks(fallbacks, suffix, prefix)
    del titles, fallbacks
    rows = await mongo.strategic_keywords(tenant_id)
    invalid = sum(1 for row in rows if not usable_strategic(row))
    strategic = merge_strategic(rows)
    gsc_rows = await mongo.gsc_query_stats(tenant_id)
    curve = fit_ctr_curve(gsc_rows)
    strategic_by_url: defaultdict[str, list[StrategicKeyword]] = defaultdict(list)
    for row in strategic:
        strategic_by_url[row.url].append(row)
    queries_by_url: defaultdict[str, list[GscQueryStats]] = defaultdict(list)
    for query in gsc_rows:
        queries_by_url[query.url].append(query)

    resolved: dict[str, ResolvedKeyword] = {}
    secondaries: dict[str, tuple[str, tuple[str, ...]]] = {}
    by_language: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    pages = rejected = 0
    async for batch in mongo.iter_page_records(tenant_id):
        for page in batch:
            if not _resolvable(page):
                continue
            pages += 1
            language = page.language or rules.default_language
            by_language[language] += 1
            keyword, failed, fallbacks_rejected = resolve_page(
                page,
                language,
                strategic_by_url.get(page.url, []),
                queries_by_url.get(page.url, []),
                curve,
                suffix,
                prefix=prefix,
                repeated=repeated,
                generic_add=generic_add,
                generic_remove=generic_remove,
            )
            rejected += failed
            reasons.update(fallbacks_rejected)
            if keyword is None:
                continue
            resolved[page.url] = keyword
            taken = [keyword.text, *(row.keyword for row in strategic_by_url.get(page.url, []))]
            found = secondary_queries(
                page,
                queries_by_url.get(page.url, []),
                curve,
                taken,
                suffix=suffix,
                prefix=prefix,
            )
            if found:
                secondaries[page.url] = (language, found)

    return KeywordPlan(
        pages=pages,
        resolved=MappingProxyType(resolved),
        by_source=MappingProxyType(
            {
                source: tuple(targets)
                for source, targets in ranked_targets(strategic, resolved, secondaries).items()
            }
        ),
        by_language=MappingProxyType(dict(by_language)),
        fallbacks_rejected=MappingProxyType(dict(sorted(reasons.items()))),
        gsc_rejected=rejected,
        gsc_rows=len(gsc_rows),
        curve=curve,
        brand_prefix=prefix,
        brand_suffix=suffix,
        invalid_strategic_rows=invalid,
        generic_add=generic_add,
        generic_remove=generic_remove,
    )


async def resolve_tenant_keywords(
    graph: GraphRepo, mongo: MongoRepo, tenant_id: str
) -> KeywordReport:
    """Resolve every crawled 2xx page's keyword and replace the tenant's keyword edges, one
    source at a time."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    started = time.perf_counter()
    plan = await plan_keywords(mongo, tenant_id)
    by_source = plan.by_source
    ranked_after = [
        target
        for targets in by_source.values()
        for target in targets
        if target.rank is not None and target.rank > 1
    ]
    # Every source is replaced, empty ones too, so a source that no longer applies loses its
    # stale edges.
    written: dict[KeywordSource, int] = {}
    deleted: dict[KeywordSource, int] = {}
    for source, targets in by_source.items():
        written[source], deleted[source] = await graph.replace_keyword_targets(
            tenant_id, source, targets
        )

    rungs = Counter(keyword.rung for keyword in plan.resolved.values())
    report = KeywordReport(
        tenant_id=tenant_id,
        pages=plan.pages,
        resolved=len(plan.resolved),
        by_rung={rung: rungs[rung] for rung in KeywordRung},
        gsc_enabled=plan.curve is not None,
        gsc_rows=plan.gsc_rows,
        gsc_rejected=plan.gsc_rejected,
        brand_suffix=plan.brand_suffix,
        brand_prefix=plan.brand_prefix,
        fallbacks_rejected=dict(sorted(plan.fallbacks_rejected.items())),
        long_fallbacks=sum(1 for keyword in plan.resolved.values() if is_long(keyword)),
        secondary_keywords=len(ranked_after),
        pages_with_secondaries=len({target.url for target in ranked_after}),
        edges_written=written,
        stale_edges_deleted=deleted,
        # The repo writes no edge on a url that is not a crawled page.
        skipped_rows=len(by_source[KeywordSource.CLIENT_STRATEGIC])
        - written[KeywordSource.CLIENT_STRATEGIC],
        by_language=dict(plan.by_language),
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )
    log.info(
        "keywords.resolved",
        stage=STAGE,
        invalid_strategic_rows=plan.invalid_strategic_rows,
        **report.model_dump(mode="json"),
    )
    return report


def summarise_keywords(report: KeywordReport) -> str:
    """A short prose record of one keyword resolution run, for the MLflow run description."""
    rungs = ", ".join(f"{report.by_rung.get(rung, 0)} {rung.value}" for rung in KeywordRung)
    languages = ", ".join(
        f"{count} {language}" for language, count in sorted(report.by_language.items())
    )
    gsc = (
        f"GSC rung enabled from {report.gsc_rows} query rows; {report.gsc_rejected} queries "
        f"failed the quality bar (at least {MIN_QUERY_IMPRESSIONS} impressions, more than the "
        "brand, every term in the page's H1, title or body)."
        if report.gsc_enabled
        else f"GSC rung skipped: {report.gsc_rows} query rows, below {MIN_CURVE_ROWS} rows or "
        f"{MIN_CURVE_IMPRESSIONS} impressions for a CTR curve."
    )
    affixes = [
        f"{kind} {affix!r}"
        for kind, affix in (("prefix", report.brand_prefix), ("suffix", report.brand_suffix))
        if affix
    ]
    brand = (
        f"Brand {' and '.join(affixes)} stripped from titles and H1s at either end."
        if affixes
        else f"No title prefix or suffix shared by {BRAND_SUFFIX_SHARE:.0%} of the titles, "
        "so none stripped."
    )
    rejections = ", ".join(
        f"{count} {reason.replace('_', ' ')}"
        for reason, count in sorted(report.fallbacks_rejected.items())
    )
    fallbacks = (
        f"H1 and title fallbacks rejected: {rejections or 'none'} (repeated means on at least "
        f"{REPEATED_FALLBACK_PAGES} pages); {report.long_fallbacks} kept fallbacks longer than "
        f"{MAX_KEYWORD_TOKENS} words."
    )
    edges = "; ".join(
        f"{source.value}: {report.edges_written.get(source, 0)} written, "
        f"{report.stale_edges_deleted.get(source, 0)} stale deleted"
        for source in (
            KeywordSource.CLIENT_STRATEGIC,
            KeywordSource.GSC_OBSERVED,
            KeywordSource.INFERRED,
        )
    )
    return "\n".join(
        [
            f"Keyword resolution for tenant {report.tenant_id}: {report.resolved} of "
            f"{report.pages} crawled 2xx pages resolved ({rungs}); languages: "
            f"{languages or 'none'}.",
            gsc,
            brand,
            fallbacks,
            f"Ranked keyword sets: {report.secondary_keywords} keywords after the resolved one "
            f"on {report.pages_with_secondaries} pages (other strategic keywords, then up to "
            f"{MAX_SECONDARY_QUERIES} further usable GSC queries).",
            f"Keyword edges: {edges}. {report.skipped_rows} strategic rows on urls that are "
            "not crawled pages skipped.",
            f"{report.seconds:.1f} s.",
        ]
    )
