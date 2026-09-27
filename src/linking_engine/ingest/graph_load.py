"""Load one tenant's pages and internal links from MongoDB into Neo4j."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import batched
from typing import TYPE_CHECKING, Final

from linking_engine.models import GraphLoadReport, Link, Page

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

LOAD_BATCH: Final = 500


async def load_tenant_graph(
    mongo: MongoRepo, graph: GraphRepo, tenant_id: str, *, batch_size: int = LOAD_BATCH
) -> GraphLoadReport:
    """Idempotent: re-running converges, including pages that lost links."""
    crawled: list[str] = []
    pages = 0
    async for summaries in mongo.iter_page_summaries(tenant_id, batch_size=batch_size):
        pages += await graph.upsert_pages(
            tenant_id,
            [
                Page(
                    url=summary.url,
                    status_code=summary.status_code,
                    word_count=summary.word_count,
                    content_hash=summary.content_hash,
                    body_hash=summary.body_hash,
                    menu_inlinks=summary.menu_inlinks,
                    footer_inlinks=summary.footer_inlinks,
                )
                for summary in summaries
            ],
        )
        crawled.extend(summary.url for summary in summaries)

    # Urls are normalised keys, so a link target either is a crawled page or becomes a placeholder.
    known = set(crawled)
    placeholders: set[str] = set()
    links = external = self_links = deleted = 0
    for sources in batched(crawled, batch_size):
        batch: list[Link] = []
        new_placeholders: list[str] = []
        for record in await mongo.links_for(tenant_id, sources):
            if not record.is_internal:
                external += 1
                continue
            target = record.target_url
            if target == record.source_url:
                self_links += 1
                continue
            if target not in known and target not in placeholders:
                placeholders.add(target)
                new_placeholders.append(target)
            batch.append(
                Link(
                    source_url=record.source_url,
                    target_url=target,
                    position=record.position,
                    anchor_text=record.anchor_text,
                    surrounding_text=record.surrounding_text,
                )
            )
        if new_placeholders:
            await graph.upsert_placeholders(tenant_id, new_placeholders)
        written, removed = await graph.replace_links(tenant_id, sources, batch)
        links += written
        deleted += removed

    return GraphLoadReport(
        pages=pages,
        placeholders=len(placeholders),
        links=links,
        external_links_skipped=external,
        self_links_skipped=self_links,
        stale_links_deleted=deleted,
        finished_at=datetime.now(UTC),
    )
