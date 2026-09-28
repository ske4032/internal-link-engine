"""Score a tenant's existing links against their targets and describe how the scores spread.

Both scores are stored on the LINKS_TO edge; the distribution maths lives in
``audit.relevance``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import structlog

from linking_engine.audit.relevance import MIN_MODE_GAP, MIN_SPLIT_SCORES, score_distribution
from linking_engine.errors import DatabaseReadError
from linking_engine.models import LinkRelevanceReport
from linking_engine.pipeline.embed import NO_MODEL, check_models

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.models import ScoreDistribution

log = structlog.get_logger(__name__)

STAGE: Final = "score-links"


async def score_links(graph: GraphRepo, tenant_id: str) -> LinkRelevanceReport:
    """Write both scores on every existing link of the tenant, read them back and describe
    their spread."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    started = time.perf_counter()
    # Cosines across embedding models mean nothing: sentences and anchors must share the pages'.
    pages = await graph.embedding_models(tenant_id)
    if pages:
        model = pages[0].embedding_model or NO_MODEL
        check_models(pages, model, tenant_id=tenant_id)
        check_models(await graph.anchor_embedding_models(tenant_id), model, tenant_id=tenant_id)
        check_models(
            await graph.surrounding_embedding_models(tenant_id), model, tenant_id=tenant_id
        )
    scored, cleared = await graph.score_link_relevance(tenant_id)
    rows = await graph.link_relevance(tenant_id)
    if len(rows) != scored:
        raise DatabaseReadError(
            "neo4j", f"{scored} links scored but {len(rows)} read back; links changed mid-run"
        )
    context = [row.context_relevance for row in rows if row.context_relevance is not None]
    if scored and not context:
        raise ValueError(
            f"{scored} links scored but none has a surrounding embedding; run embed-links first"
        )
    fits = [row.anchor_target_fit for row in rows if row.anchor_target_fit is not None]
    report = LinkRelevanceReport(
        tenant_id=tenant_id,
        links=scored + cleared,
        scored=scored,
        generic_anchors=sum(row.anchor_generic for row in rows),
        without_anchor_vector=sum(
            not row.anchor_generic and row.anchor_target_fit is None for row in rows
        ),
        context=score_distribution(context),
        anchor=score_distribution(fits),
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )
    log.info("links.relevance", stage=STAGE, **report.model_dump(mode="json"))
    return report


def _describe(name: str, found: ScoreDistribution | None) -> str:
    if found is None:
        return f"{name}: no scores."
    spread = (
        f"{name} over {found.count} links: mean {found.mean:.3f}, p10 {found.p10:.3f}, "
        f"p25 {found.p25:.3f}, median {found.p50:.3f}, p75 {found.p75:.3f}, p90 {found.p90:.3f}."
    )
    if found.split is None or found.low_share is None:
        return f"{spread} No split: the scores do not separate into two modes."
    return f"{spread} Split at {found.split:.3f}, {found.low_share:.1%} of the links below it."


def summarise_link_relevance(report: LinkRelevanceReport) -> str:
    """A short prose record of one link relevance run, for the MLflow run description."""
    return "\n".join(
        [
            f"Link relevance for tenant {report.tenant_id}: {report.scored} of {report.links} "
            "body links between crawled pages scored against their target's content vector "
            f"(the rest have a target without one); {report.seconds:.1f}s.",
            f"Anchors: {report.generic_anchors} generic, so no fit; "
            f"{report.without_anchor_vector} without a stored anchor vector.",
            _describe("Context relevance (sentence vs target)", report.context),
            _describe("Anchor-target fit (anchor vs target)", report.anchor),
            "Scores are Neo4j's normalised cosine in [0, 1]; the split is where a two-component "
            f"Gaussian mixture flips (at least {MIN_SPLIT_SCORES} scores, means at least "
            f"{MIN_MODE_GAP} apart).",
        ]
    )
