"""Load every crawled page's pair signals once per run, so no pair ever reads a database."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from linking_engine.discovery.signals import STAGE, build_page_signals

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import PageSignals

log = structlog.get_logger(__name__)


async def load_page_signals(
    graph: GraphRepo, mongo: MongoRepo, tenant_id: str
) -> tuple[dict[str, PageSignals], int]:
    """The tenant's page signals, and how many distinct GSC urls are not crawled pages."""
    pages = await graph.community_context(tenant_id)
    queries = await mongo.gsc_queries(tenant_id)
    keywords = [(k.url, k.keyword) for k in await mongo.strategic_keywords(tenant_id)]
    signals = build_page_signals(pages, queries, keywords)
    unmatched = len({url for url, _ in queries} - signals.keys())
    log.info(
        "signals.pages",
        stage=STAGE,
        tenant_id=tenant_id,
        pages=len(signals),
        query_rows=len(queries),
        keyword_rows=len(keywords),
        unmatched_query_urls=unmatched,
    )
    return signals, unmatched
