"""Duplicate pages stage: after the graph is loaded, group the tenant's exact duplicates and
store each page's group and whether it is the canonical copy."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import structlog

from linking_engine.discovery.duplicates import duplicate_groups
from linking_engine.models import DuplicateReport

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo

log = structlog.get_logger(__name__)

STAGE: Final = "duplicates"


async def find_duplicates(graph: GraphRepo, tenant_id: str) -> DuplicateReport:
    """Regroup the tenant from the stored graph and replace every stored group and flag."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    started = time.perf_counter()
    pages = await graph.duplicate_inputs(tenant_id)
    groups = duplicate_groups(pages)
    written = await graph.write_duplicate_groups(tenant_id, groups)
    sizes = [1 + len(group.copies) for group in groups]
    report = DuplicateReport(
        tenant_id=tenant_id,
        groups=tuple(groups),
        pages_in_groups=sum(sizes),
        non_canonical=sum(sizes) - len(groups),
        largest_group=max(sizes, default=0),
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )
    log.info(
        "graph.duplicates",
        stage=STAGE,
        tenant_id=tenant_id,
        groups=len(groups),
        pages_in_groups=report.pages_in_groups,
        non_canonical=report.non_canonical,
        largest_group=report.largest_group,
        pages_written=written,
        seconds=report.seconds,
    )
    return report


def summarise_duplicates(report: DuplicateReport) -> str:
    """A short prose record of one duplicate grouping run, for the MLflow run description."""
    scope = (
        f"Exact duplicate pages of tenant {report.tenant_id}: crawled 2xx pages with a "
        "non-empty body, grouped by identical body hash within one language."
    )
    if not report.groups:
        return "\n".join([scope, "No duplicates found.", f"{report.seconds:.1f} s."])
    return "\n".join(
        [
            scope,
            f"{len(report.groups)} groups over {report.pages_in_groups} pages, the largest "
            f"{report.largest_group} pages; {report.non_canonical} non-canonical copies are "
            "neither link targets nor sources, a consolidation finding (canonical or 301 to "
            "the canonical url).",
            "Canonical copy per group: an indexable copy first, then the most inbound body "
            "links from distinct crawled pages, the shortest url and the url. groups.json "
            "lists every group.",
            f"{report.seconds:.1f} s.",
        ]
    )
