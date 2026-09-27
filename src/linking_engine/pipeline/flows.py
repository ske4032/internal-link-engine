"""Prefect flows for pipeline stages."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE
from prefect.runtime import flow_run
from structlog.contextvars import bound_contextvars

from linking_engine.embedding.voyage_client import VoyageClient, VoyageSettings
from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseUnavailableError,
    EmbeddingAuthError,
    EmbeddingUnavailableError,
)
from linking_engine.graph.repo import GraphRepo
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.models import EmbedRunReport, LinkEmbedReport, TenantConfig
from linking_engine.pipeline.embed import FLUSH_SIZE, embed_tenant
from linking_engine.pipeline.embed_links import embed_links

if TYPE_CHECKING:
    from prefect.client.schemas.objects import State


def voyage_client(tenant_id: str) -> VoyageClient:
    tenant = TenantConfig(tenant_id=tenant_id)
    return VoyageClient(
        VoyageSettings(), model=tenant.embedding_model, dimension=tenant.embedding_dimensions
    )


def is_transient(_task: object, _task_run: object, state: State[Any]) -> bool:
    """Retry only outages; bad input, credentials and model mismatches fail at once."""
    error = state.data
    if isinstance(error, DatabaseAuthError | EmbeddingAuthError):
        return False
    return isinstance(error, DatabaseUnavailableError | EmbeddingUnavailableError)


# A retry re-runs the whole stage: the resume query skips every flush already committed.
@task(
    name="embed-pages",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def embed_pages_task(tenant_id: str, flush_size: int) -> EmbedRunReport:
    voyage = voyage_client(tenant_id)
    async with (
        await GraphRepo.connect(
            os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]
        ) as graph,
        await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"]) as mongo,
    ):
        await graph.check_server()
        await graph.verify_vector_indexes()
        return await embed_tenant(mongo, graph, voyage, tenant_id, flush_size=flush_size)


# A retry re-runs the whole stage: cached anchors and sentences are skipped, so only the rest is embedded.
@task(
    name="embed-links",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def embed_links_task(tenant_id: str, flush_size: int) -> LinkEmbedReport:
    voyage = voyage_client(tenant_id)
    async with await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"]) as mongo:
        rules = await mongo.get_anchor_rules(tenant_id)
    async with await GraphRepo.connect(
        os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]
    ) as graph:
        await graph.check_server()
        return await embed_links(graph, voyage, tenant_id, flush_size=flush_size, rules=rules)


# Two flows, not one: a failure shows as either the page or the link flow in Prefect.
@flow(name="embed-pages")
async def embed_pages_flow(tenant_id: str, flush_size: int = FLUSH_SIZE) -> EmbedRunReport:
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("embedding pages of tenant %s, flush size %d", tenant_id, flush_size)
        pages = await embed_pages_task(tenant_id, flush_size)
    logger.info(
        "embedded %d of %d selected pages in %d flushes; skipped %d not usable, %d empty, "
        "%d missing, %d hash mismatch; %d api tokens, %.1fs",
        pages.embedded,
        pages.selected,
        pages.flushes,
        pages.skipped_not_usable,
        pages.skipped_empty_body,
        pages.skipped_missing,
        pages.skipped_hash_mismatch,
        pages.api_tokens,
        pages.elapsed_s,
    )
    return pages


@flow(name="embed-links")
async def embed_links_flow(tenant_id: str, flush_size: int = FLUSH_SIZE) -> LinkEmbedReport:
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info(
            "embedding anchors and sentences of tenant %s, flush size %d", tenant_id, flush_size
        )
        links = await embed_links_task(tenant_id, flush_size)
    logger.info(
        "embedded %d of %d anchors (%d generic, %d cached) and %d of %d sentences "
        "(%d cached, %d reused) over %d edges; %d edges written; %d api tokens, %.1fs",
        links.anchors_embedded,
        links.unique_anchors,
        links.generic_anchors,
        links.anchors_cached,
        links.sentences_embedded,
        links.unique_sentences,
        links.sentences_cached,
        links.sentences_reused,
        links.edges,
        links.surrounding_edges_written,
        links.api_tokens,
        links.elapsed_s,
    )
    return links
