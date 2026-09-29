"""Audit a tenant's existing body links (Stage 1). Every anchor is measured with the anchor
stage's tools, the ladder searches the source copy of the flagged links for a better phrase, and
the scoring in ``audit.links`` decides. Each run replaces the previous one in ``link_audit``
once complete, and its scores and verdicts are written back onto the LINKS_TO edges.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import structlog

from linking_engine.anchor.extraction import Stems
from linking_engine.anchor.generic import generic_overrides, is_generic, normalise_anchor
from linking_engine.anchor.scoring import Brand, brand_tokens, existing_type, stem_jaccard, unit
from linking_engine.audit.links import (
    NO_STORED_SCORES,
    PAGINATED_SKIPPED,
    SITEMAP_SKIPPED,
    AnchorFacts,
    Proposal,
    assess,
    audit_report,
    audit_scope,
    decide,
    ladder_pairs,
)
from linking_engine.discovery.candidates import candidate_report
from linking_engine.errors import DatabaseWriteError
from linking_engine.models import (
    AnchorType,
    CandidateSet,
    CandidateTarget,
    TargetCandidates,
    TargetSelection,
)
from linking_engine.pipeline.anchor_selection import compute_anchor_choices
from linking_engine.pipeline.anchors import AnchorView, cache_folder
from linking_engine.pipeline.semantic_anchors import AnchorVectors

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set
    from pathlib import Path

    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import KeywordSource, LinkAuditReport, ScoreDistribution

log = structlog.get_logger(__name__)

STAGE: Final = "link-audit"
# The roadmap bands of the fixable-link rate: below the first, discovery is the product and the
# audit a feature; above the second, the audit is the product.
FIXABLE_FEATURE: Final = 0.10
FIXABLE_PRODUCT: Final = 0.30


@dataclass(frozen=True, slots=True)
class _Measured:
    facts: AnchorFacts
    # The ranked keyword the text overlaps best, which its cosine is taken to.
    keyword: str | None


class _Measure:
    """Measures anchors and phrases against a target's ranked keywords, stemmed in the source
    page's language; repeated (text, target, language) triples are measured once."""

    def __init__(
        self,
        keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
        brand: Brand,
        *,
        generic_add: frozenset[str],
        generic_remove: frozenset[str],
    ) -> None:
        self._keywords = keywords
        self._brand = brand
        self._add = generic_add
        self._remove = generic_remove
        self._stems: dict[str | None, Stems] = {}
        self._found: dict[tuple[str, str, str | None], _Measured] = {}

    def __call__(self, text: str, target_url: str, language: str | None) -> _Measured:
        wanted = (text, target_url, language)
        found = self._found.get(wanted)
        if found is None:
            found = self._found[wanted] = self._measure(text, target_url, language)
        return found

    def _measure(self, text: str, target_url: str, language: str | None) -> _Measured:
        key = normalise_anchor(text) or None
        generic = is_generic(text, add=self._add, remove=self._remove)
        ranked = self._keywords.get(target_url)
        if key is None or not ranked:
            return _Measured(AnchorFacts(key, generic, False, None), None)
        if language not in self._stems:
            self._stems[language] = Stems(language)
        stems = self._stems[language]
        # Highest overlap first, the better-ranked keyword on ties.
        jaccard, _, keyword = max(
            (stem_jaccard(text, keyword, stems), -rank, keyword) for rank, keyword, _ in ranked
        )
        kind = existing_type(text, [keyword for _, keyword, _ in ranked], stems, self._brand)
        return _Measured(AnchorFacts(key, generic, kind is AnchorType.EXACT, jaccard), keyword)


async def audit_links(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    *,
    cache_dir: Path,
    voyage: VoyageClient | None,
) -> tuple[LinkAuditReport, str]:
    """Scores, flags and a verdict for every body link of the tenant, written to ``link_audit``
    as a new run and onto the edges; the report and the run id. A2 runs on #16's stored scores;
    without them A1 runs alone and the report says why. Without ``voyage``, only cached phrase
    and keyword vectors add the cosine to keyword alignment."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    cache_folder(cache_dir, tenant_id)
    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    # Sitemap and paginated links are left out before anything is measured or embedded.
    scope = audit_scope(await graph.audit_edges(tenant_id))
    edges = scope.edges
    keywords = await graph.ranked_keywords(tenant_id)
    rules = await mongo.get_anchor_rules(tenant_id)
    generic_add, generic_remove = generic_overrides(rules.generic_add, rules.generic_remove)
    measure = _Measure(
        keywords,
        brand_tokens(await mongo.page_titles(tenant_id)),
        generic_add=generic_add,
        generic_remove=generic_remove,
    )
    measured = await asyncio.to_thread(
        lambda: [measure(e.anchor_text, e.target_url, e.source_language) for e in edges]
    )
    a2 = any(edge.context_relevance is not None for edge in edges if not edge.target_placeholder)
    vectors = (
        AnchorVectors(voyage, tenant_id, {}, page_models=(), cache_dir=cache_dir) if a2 else None
    )
    facts = await _with_cosines(
        vectors, [(edge.anchor_text, found) for edge, found in zip(edges, measured, strict=True)]
    )
    assessment = await asyncio.to_thread(lambda: assess(edges, facts, scope=scope))
    pairs = ladder_pairs(assessment)
    proposals, ladder_skipped = await _proposals(
        graph,
        mongo,
        tenant_id,
        pairs,
        languages={edge.source_url: edge.source_language for edge in edges},
        measure=measure,
        vectors=vectors,
        cache_dir=cache_dir,
        voyage=voyage,
    )
    # Milliseconds, as Mongo stores them, so the rows, the edges and the marker agree.
    now = datetime.now(UTC)
    audited_at = now.replace(microsecond=now.microsecond // 1000 * 1000)
    outcome = decide(assessment, proposals, run_id=run_id, audited_at=audited_at)
    results = outcome.results
    written = await graph.write_link_audit(tenant_id, results)
    if written != len(results):
        raise DatabaseWriteError("neo4j", f"{written} of {len(results)} audited links written")
    # Every link read now carries this run, so only the links left out can hold another's.
    cleared = await graph.clear_stale_link_audit(tenant_id, run_id)
    left_out = scope.sitemap_links + scope.paginated_links
    if cleared > left_out:
        raise DatabaseWriteError(
            "neo4j",
            f"{cleared} stale audits cleared but only {left_out} links were left out of run "
            f"{run_id}; the links changed during the run",
        )
    stored = await mongo.insert_link_audit(tenant_id, run_id, results)
    if stored != len(results):
        raise DatabaseWriteError("mongodb", f"{stored} of {len(results)} audit rows stored")
    # Only a marked run is ever read as the tenant's latest; the older runs go once it is.
    await mongo.complete_link_audit(
        tenant_id, run_id, audited_at=audited_at, documents=stored, edges=written
    )
    pruned, pruned_runs = await mongo.prune_link_audit(tenant_id, run_id)
    report = audit_report(
        tenant_id,
        run_id,
        assessment,
        outcome,
        ladder_pairs=len(pairs),
        vectors_skipped_reason=(vectors.skipped_reason() if vectors else None) or ladder_skipped,
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )
    log.info(
        "audit.complete",
        stage=STAGE,
        tenant_id=tenant_id,
        run_id=run_id,
        links=report.links,
        unverified=report.unverified,
        healthy=report.healthy,
        by_flag={flag.value: n for flag, n in report.by_flag.items()},
        by_verdict={verdict.value: n for verdict, n in report.by_verdict.items()},
        index_like_pages=report.index_like_pages,
        listing_pages=report.listing_pages,
        sitemap_pages=report.sitemap_pages,
        sitemap_links=report.sitemap_links,
        paginated_pages=report.paginated_pages,
        paginated_links=report.paginated_links,
        stale_audits_cleared=cleared,
        pruned_documents=pruned,
        pruned_runs=pruned_runs,
        ladder_pairs=report.ladder_pairs,
        proposals=report.proposals,
        embeddings=report.embeddings,
        embeddings_skipped_reason=report.embeddings_skipped_reason,
        vectors_skipped_reason=report.vectors_skipped_reason,
        seconds=report.seconds,
    )
    return report, run_id


async def _with_cosines(
    vectors: AnchorVectors | None, measured: Sequence[tuple[str, _Measured]]
) -> list[AnchorFacts]:
    """Each measurement's facts, with the text's cosine to its best keyword when both vectors
    are cached or embedded."""
    if vectors is None:
        return [found.facts for _, found in measured]
    wanted = [(text.strip(), found) for text, found in measured if found.keyword is not None]
    await vectors.ensure("phrases", {text for text, _ in wanted})
    await vectors.ensure("keywords", {found.keyword for _, found in wanted if found.keyword})
    facts: list[AnchorFacts] = []
    for text, found in measured:
        cosine = (
            None if found.keyword is None else vectors.phrase_keyword(text.strip(), found.keyword)
        )
        facts.append(
            found.facts
            if cosine is None
            else AnchorFacts(
                found.facts.key,
                found.facts.generic,
                found.facts.exact_match,
                found.facts.keyword_jaccard,
                unit(cosine),
            )
        )
    return facts


async def _proposals(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    pairs: Set[tuple[str, str]],
    *,
    languages: Mapping[str, str | None],
    measure: _Measure,
    vectors: AnchorVectors | None,
    cache_dir: Path,
    voyage: VoyageClient | None,
) -> tuple[dict[tuple[str, str], list[Proposal]], str | None]:
    """The ladder's phrases for every pair in rank order, measured as the anchors are, and why
    Voyage stopped serving the ladder; nothing is searched without pairs."""
    if not pairs:
        return {}, None
    selection = await compute_anchor_choices(
        graph,
        mongo,
        tenant_id,
        cache_dir=cache_dir,
        voyage=voyage,
        view=AnchorView(_candidates(tenant_id, pairs), frozenset()),
    )
    choices = sorted(
        selection.choices,
        key=lambda c: (c.match.source_url, c.match.target_url, c.rank),
    )
    measured = [
        (
            choice,
            measure(
                choice.match.phrase,
                choice.match.target_url,
                languages.get(choice.match.source_url),
            ),
        )
        for choice in choices
    ]
    facts = await _with_cosines(vectors, [(c.match.phrase, found) for c, found in measured])
    proposals: defaultdict[tuple[str, str], list[Proposal]] = defaultdict(list)
    for (choice, _), fact in zip(measured, facts, strict=True):
        if fact.key is None:
            continue
        proposals[(choice.match.source_url, choice.match.target_url)].append(
            Proposal(
                phrase=choice.match.phrase,
                key=fact.key,
                keyword_jaccard=fact.keyword_jaccard,
                keyword_cosine=fact.keyword_cosine,
                anchor_target_fit=choice.anchor_target_fit,
            )
        )
    return dict(proposals), selection.skipped_reason


def _candidates(tenant_id: str, pairs: Iterable[tuple[str, str]]) -> CandidateSet:
    """The flagged pairs as a candidate set the ladder can search: every source of a target
    ranks level, so they are taken in url order."""
    sources: defaultdict[str, list[str]] = defaultdict(list)
    for source, target in pairs:
        sources[target].append(source)
    targets = tuple(
        TargetCandidates(
            target_url=target,
            sources=tuple(sorted(found)),
            similarities=(0.0,) * len(found),
            eligible=len(found),
            linked=0,
            linked_nearer=0,
        )
        for target, found in sorted(sources.items())
    )
    widest = max(len(entry.sources) for entry in targets)
    pages = {page for entry in targets for page in (entry.target_url, *entry.sources)}
    report = candidate_report(
        tenant_id,
        "page_content",
        widest,
        widest,
        TargetSelection(
            crawled_pages=len(targets),
            not_indexable=0,
            without_vector=0,
            targets=tuple(
                CandidateTarget(url=entry.target_url, indexable_assumed=False) for entry in targets
            ),
        ),
        len(pages),
        targets,
        load_seconds=0.0,
        search_seconds=0.0,
        seconds=0.0,
    )
    return CandidateSet(report=report, targets=targets)


def _describe(name: str, found: ScoreDistribution | None, scale: float = 1.0) -> str:
    if found is None:
        return f"{name}: no scores."
    return (
        f"{name} over {found.count} links: mean {found.mean * scale:.3f}, "
        f"p10 {found.p10 * scale:.3f}, median {found.p50 * scale:.3f}, p90 {found.p90 * scale:.3f}."
    )


def fixable_band(rate: float) -> str:
    """The roadmap band a fixable-link rate falls in."""
    if rate < FIXABLE_FEATURE:
        return f"below {FIXABLE_FEATURE:.0%}: the audit is a feature, discovery is the product"
    if rate > FIXABLE_PRODUCT:
        return f"above {FIXABLE_PRODUCT:.0%}: the audit is the product"
    return f"{FIXABLE_FEATURE:.0%}-{FIXABLE_PRODUCT:.0%}: both the audit and discovery matter"


def _fixable(report: LinkAuditReport) -> str:
    rates = [
        f"{rate:.1%} of {label} ({fixable_band(rate)})"
        for label, rate in (
            ("all audited links", report.fixable_rate),
            ("the links into crawled pages", report.verified_fixable_rate),
        )
        if rate is not None
    ]
    return f"Fixable-link rate: {'; '.join(rates) if rates else 'no audited links'}."


def summarise_link_audit(report: LinkAuditReport) -> str:
    """A short prose record of one audit run, for the MLflow run description; no urls."""
    counts = ", ".join(f"{flag.value} {n}" for flag, n in sorted(report.by_flag.items())) or "none"
    verdicts = (
        ", ".join(f"{verdict.value} {n}" for verdict, n in sorted(report.by_verdict.items()))
        or "none"
    )
    cutoffs = "; ".join(
        f"{cutoff.name} {'none' if cutoff.value is None else f'{cutoff.value:.4g}'} "
        f"({cutoff.reason})"
        for cutoff in report.cutoffs
    )
    mode = (
        "A2 on the stored context relevance and anchor-target fit"
        if report.embeddings
        else f"A1 only: {report.embeddings_skipped_reason or NO_STORED_SCORES}"
    )
    return "\n".join(
        [
            f"Link audit {report.run_id} of tenant {report.tenant_id}: {report.links} body links "
            f"from {report.source_pages} source pages, {report.unverified} into pages never "
            f"crawled (unverified), {report.healthy} healthy; {report.seconds:.1f}s.",
            f"{mode}. Keyword alignment includes the phrase-keyword cosine on "
            f"{report.keyword_cosines} links"
            + (
                f"; vectors: {report.vectors_skipped_reason}."
                if report.vectors_skipped_reason
                else "."
            ),
            f"Flags: {counts}. Verdicts: {verdicts}.",
            _fixable(report),
            f"{report.index_like_pages} index-like source pages, whose links are never removed. "
            f"{report.listing_pages} listing pages above the link density fence, whose links "
            "are never reanchored or removed.",
            f"{report.sitemap_pages} sitemap pages and {report.sitemap_links} links from or into "
            f"them skipped: {SITEMAP_SKIPPED}. {report.paginated_pages} paginated pages and "
            f"{report.paginated_links} links skipped: {PAGINATED_SKIPPED}.",
            f"{report.ladder_pairs} pairs searched for a better phrase; {report.proposals} "
            "REANCHOR verdicts propose one from the copy.",
            f"Cut-offs: {cutoffs}.",
            _describe("Keyword alignment", report.keyword_alignment),
            _describe("Equity efficiency", report.equity_efficiency),
            _describe("Anchor quality score (0-100)", report.anchor_quality, 100.0),
        ]
    )
