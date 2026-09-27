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
from linking_engine.ingest.graph_load import load_tenant_graph
from linking_engine.ingest.mongo_repo import CrawlSource, MongoRepo
from linking_engine.ingest.prepare import BOILERPLATE_SHARE, NAV_SHARE, prepare_tenant
from linking_engine.ml.tracking import log_analytics
from linking_engine.models import (
    CentralityReport,
    CommunityReport,
    EmbedRunReport,
    GraphLoadReport,
    HubReport,
    LinkEmbedReport,
    PrepareReport,
    TenantConfig,
    TenantGraphCounts,
)
from linking_engine.pipeline.analytics import (
    compute_centrality,
    compute_communities,
    compute_hubs,
    summarise,
)
from linking_engine.pipeline.embed import FLUSH_SIZE, embed_tenant
from linking_engine.pipeline.embed_links import embed_links

if TYPE_CHECKING:
    from prefect.client.schemas.objects import State


async def neo4j() -> GraphRepo:
    return await GraphRepo.connect(
        os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]
    )


async def mongo() -> MongoRepo:
    return await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"])


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


# Each task reads its own snapshot and writes all its pages in one transaction, so a retry
# rewrites the stage whole.
@task(
    name="graph-centrality",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def centrality_task(tenant_id: str) -> CentralityReport:
    async with await neo4j() as graph:
        await graph.check_server()
        return await compute_centrality(graph, tenant_id)


@task(
    name="graph-communities",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def communities_task(tenant_id: str) -> CommunityReport:
    async with await neo4j() as graph:
        await graph.check_server()
        return await compute_communities(graph, tenant_id)


# Hub ids are matched to the previous run's hubs, so a retry keeps the same ids.
@task(
    name="graph-hubs",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def hubs_task(tenant_id: str) -> HubReport:
    async with await neo4j() as graph:
        await graph.check_server()
        return await compute_hubs(graph, tenant_id)


@task(name="mlflow-log", cache_policy=NONE)
def log_analytics_task(
    centrality: CentralityReport, communities: CommunityReport, hubs: HubReport
) -> str:
    return log_analytics(centrality, communities, hubs, summarise(centrality, communities, hubs))


@flow(name="graph-analytics")
async def graph_analytics_flow(
    tenant_id: str,
) -> tuple[CentralityReport, CommunityReport, HubReport, str]:
    """PageRank, betweenness, communities and hubs written to Neo4j, then the run logged to
    MLflow. Hubs run after communities so their agreement uses this run's communities."""
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("graph analytics of tenant %s", tenant_id)
        centrality = await centrality_task(tenant_id)
        communities = await communities_task(tenant_id)
        hubs = await hubs_task(tenant_id)
        mlflow_run = log_analytics_task(centrality, communities, hubs)
    logger.info(
        "%d pages: %d link, %d keyword and %d content communities; %d hubs, %d noise pages; "
        "%d orphans, %d dead ends; mlflow run %s",
        communities.crawled_pages,
        communities.link.communities,
        communities.keyword.communities,
        communities.content.communities,
        hubs.hubs,
        hubs.noise,
        communities.orphans,
        communities.dead_ends,
        mlflow_run,
    )
    return centrality, communities, hubs, mlflow_run


# Upserts keyed by url and position, so a retry converges on the same records.
@task(
    name="prepare-corpus",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def prepare_task(
    tenant_id: str,
    source_db: str,
    source_collection: str,
    boilerplate_share: float,
    nav_share: float,
) -> PrepareReport:
    if source_db == os.environ["MONGO_DB"]:
        raise ValueError("source and target database must differ: the source is read-only")
    async with (
        await CrawlSource.connect(os.environ["MONGO_URI"], source_db, source_collection) as source,
        await mongo() as repo,
    ):
        _, report = await prepare_tenant(
            source,
            repo,
            tenant_id,
            source_name=f"{source_db}.{source_collection}",
            boilerplate_share=boilerplate_share,
            nav_share=nav_share,
        )
        return report


# The load converges on re-run, including pages that lost links.
@task(
    name="load-graph",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def load_graph_task(tenant_id: str) -> tuple[GraphLoadReport, TenantGraphCounts]:
    logger = get_run_logger()
    async with await neo4j() as graph, await mongo() as repo:
        await graph.check_server()
        applied = await graph.migrate()
        logger.info("neo4j migrations applied: %s", list(applied) or "none pending")
        await repo.ensure_indexes()
        report = await load_tenant_graph(repo, graph, tenant_id)
        return report, await graph.counts(tenant_id)


# Separate flows, like embedding: a failure shows as either the prepare or the load flow.
@flow(name="prepare-corpus")
async def prepare_corpus_flow(
    tenant_id: str,
    source_db: str,
    source_collection: str,
    boilerplate_share: float = BOILERPLATE_SHARE,
    nav_share: float = NAV_SHARE,
) -> PrepareReport:
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("preparing the crawl of tenant %s", tenant_id)
        report = await prepare_task(
            tenant_id, source_db, source_collection, boilerplate_share, nav_share
        )
    logger.info(
        "%d documents, %d pages (%d merged urls), %d links; skipped %s; %d template lines; "
        "%d pages with menu inlinks, %d with footer inlinks; written %d pages, %d links, "
        "%d stale links deleted",
        report.documents,
        report.pages,
        report.merged_urls,
        report.links,
        report.skipped or "none",
        report.template_lines,
        report.menu_inlink_pages,
        report.footer_inlink_pages,
        report.pages_written,
        report.links_written,
        report.stale_links_deleted,
    )
    return report


@flow(name="load-graph")
async def load_graph_flow(tenant_id: str) -> tuple[GraphLoadReport, TenantGraphCounts]:
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("loading the graph of tenant %s", tenant_id)
        report, counts = await load_graph_task(tenant_id)
    logger.info(
        "%d pages, %d placeholders, %d links; skipped %d external and %d self links; "
        "%d stale links deleted; %d broken pages, %d FIX links",
        report.pages,
        report.placeholders,
        report.links,
        report.external_links_skipped,
        report.self_links_skipped,
        report.stale_links_deleted,
        counts.broken_pages,
        counts.fix_links,
    )
    return report, counts
