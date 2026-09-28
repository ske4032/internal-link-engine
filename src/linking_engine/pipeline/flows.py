"""Prefect flows for pipeline stages."""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003 - Prefect validates flow parameters at runtime
from typing import TYPE_CHECKING, Any

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE
from prefect.runtime import flow_run
from structlog.contextvars import bound_contextvars

from linking_engine.discovery.bridges import summarise_bridges
from linking_engine.discovery.candidates import retrieve_candidates, summarise_candidates
from linking_engine.discovery.features import CHUNK_PAIRS, summarise_features
from linking_engine.discovery.scoring import summarise_scores
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
from linking_engine.ml.tracking import (
    log_analytics,
    log_bridges,
    log_candidates,
    log_duplicates,
    log_features,
    log_keywords,
    log_link_relevance,
    log_scores,
)
from linking_engine.models import (
    BridgeReport,
    CandidateSet,
    CentralityReport,
    CommunityReport,
    DuplicateReport,
    EmbedRunReport,
    FeatureReport,
    GraphLoadReport,
    HubReport,
    KeywordReport,
    LinkEmbedReport,
    LinkRelevanceReport,
    PrepareReport,
    ScoreReport,
    TenantConfig,
    TenantGraphCounts,
    VectorIndex,
)
from linking_engine.pipeline.analytics import (
    compute_centrality,
    compute_communities,
    compute_hubs,
    summarise,
)
from linking_engine.pipeline.bridges import HUB_PAIRS_FILE, find_bridges, read_hub_pairs
from linking_engine.pipeline.duplicates import find_duplicates, summarise_duplicates
from linking_engine.pipeline.embed import FLUSH_SIZE, embed_tenant
from linking_engine.pipeline.embed_links import embed_links
from linking_engine.pipeline.features import CACHE_DIR, assemble_features
from linking_engine.pipeline.keywords import resolve_tenant_keywords, summarise_keywords
from linking_engine.pipeline.link_relevance import score_links, summarise_link_relevance
from linking_engine.pipeline.scoring import score_pairs

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


# Read-only against Neo4j, so a retry repeats the retrieval with nothing to undo.
@task(
    name="candidate-retrieval",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def candidates_task(tenant_id: str, index: VectorIndex) -> CandidateSet:
    async with await neo4j() as graph:
        await graph.check_server()
        return await retrieve_candidates(graph, tenant_id, index=index)


@task(name="mlflow-log-candidates", cache_policy=NONE)
def log_candidates_task(found: CandidateSet) -> str:
    return log_candidates(found, summarise_candidates(found.report))


@flow(name="candidate-retrieval")
async def candidate_retrieval_flow(
    tenant_id: str, index: VectorIndex = "page_content"
) -> tuple[CandidateSet, str]:
    """The nearest eligible sources of every indexable page, capped per target; nothing is
    written to Neo4j, the report and a per-target table are logged to MLflow."""
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("candidate retrieval of tenant %s over the %s vectors", tenant_id, index)
        found = await candidates_task(tenant_id, index)
        mlflow_run = log_candidates_task(found)
    report = found.report
    logger.info(
        "%d targets over %d source pages, %d candidates; %d full, %d short, %d empty targets; "
        "%d linked pairs excluded, %d among the nearest; %.1fs (%.1fs load, %.1fs search); "
        "mlflow run %s",
        report.targets,
        report.source_pages,
        report.candidates,
        report.full_targets,
        report.short_targets,
        report.empty_targets,
        report.linked_pairs,
        report.linked_nearer,
        report.seconds,
        report.load_seconds,
        report.search_seconds,
        mlflow_run,
    )
    return found, mlflow_run


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


# Regrouped from the stored graph and written in one transaction, so a retry converges.
@task(
    name="find-duplicates",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def duplicates_task(tenant_id: str) -> DuplicateReport:
    async with await neo4j() as graph:
        await graph.check_server()
        return await find_duplicates(graph, tenant_id)


@task(name="mlflow-log-duplicates", cache_policy=NONE)
def log_duplicates_task(report: DuplicateReport) -> str:
    return log_duplicates(report, summarise_duplicates(report))


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
async def load_graph_flow(
    tenant_id: str,
) -> tuple[GraphLoadReport, TenantGraphCounts, DuplicateReport, str]:
    """Pages and body links loaded into Neo4j, then exact duplicates grouped, which needs the
    inbound links; the duplicate groups are logged to MLflow."""
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("loading the graph of tenant %s", tenant_id)
        report, counts = await load_graph_task(tenant_id)
        duplicates = await duplicates_task(tenant_id)
        mlflow_run = log_duplicates_task(duplicates)
    logger.info(
        "%d pages, %d placeholders, %d links; skipped %d external and %d self links; "
        "%d stale links deleted; %d broken pages, %d FIX links; %d duplicate groups over %d "
        "pages, %d non-canonical copies; mlflow run %s",
        report.pages,
        report.placeholders,
        report.links,
        report.external_links_skipped,
        report.self_links_skipped,
        report.stale_links_deleted,
        counts.broken_pages,
        counts.fix_links,
        len(duplicates.groups),
        duplicates.pages_in_groups,
        duplicates.non_canonical,
        mlflow_run,
    )
    return report, counts, duplicates, mlflow_run


# Each source's keyword edges are replaced whole, so a retry converges on the same edges.
@task(
    name="resolve-keywords",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def resolve_keywords_task(tenant_id: str) -> KeywordReport:
    async with await neo4j() as graph, await mongo() as repo:
        await graph.check_server()
        return await resolve_tenant_keywords(graph, repo, tenant_id)


@task(name="mlflow-log-keywords", cache_policy=NONE)
def log_keywords_task(report: KeywordReport) -> str:
    return log_keywords(report, summarise_keywords(report))


@flow(name="resolve-keywords")
async def resolve_keywords_flow(tenant_id: str) -> tuple[KeywordReport, str]:
    """Every crawled 2xx page's target keyword and the tenant's keyword edges, after load-graph
    and before graph-analytics so the keyword communities see this run's keywords."""
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("resolving the keywords of tenant %s", tenant_id)
        report = await resolve_keywords_task(tenant_id)
        mlflow_run = log_keywords_task(report)
    logger.info(
        "%d of %d pages resolved: %s; gsc %s (%d rejected); edges written %s, stale deleted %s; "
        "%d strategic rows on uncrawled urls; %.1fs; mlflow run %s",
        report.resolved,
        report.pages,
        {rung.value: count for rung, count in report.by_rung.items()},
        "enabled" if report.gsc_enabled else "skipped",
        report.gsc_rejected,
        {source.value: count for source, count in report.edges_written.items()},
        {source.value: count for source, count in report.stale_edges_deleted.items()},
        report.skipped_rows,
        report.seconds,
        mlflow_run,
    )
    return report, mlflow_run


# Read-only against both stores; the matrix file is renamed into place only when complete, so
# a retry finds it cached or builds it again.
@task(
    name="feature-assembly",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def features_task(
    tenant_id: str, cache_dir: Path, chunk_pairs: int
) -> tuple[FeatureReport, Path]:
    async with await neo4j() as graph, await mongo() as repo:
        await graph.check_server()
        return await assemble_features(
            graph, repo, tenant_id, cache_dir=cache_dir, chunk_pairs=chunk_pairs
        )


@task(name="mlflow-log-features", cache_policy=NONE)
def log_features_task(report: FeatureReport) -> str:
    return log_features(report, summarise_features(report))


@flow(name="feature-assembly")
async def feature_assembly_flow(
    tenant_id: str, cache_dir: Path = CACHE_DIR, chunk_pairs: int = CHUNK_PAIRS
) -> tuple[FeatureReport, Path, str]:
    """The features of every candidate pair, cached as Parquet under ``cache_dir``; nothing is
    written to the stores, the report and the column order are logged to MLflow."""
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("feature assembly of tenant %s, chunks of %d pairs", tenant_id, chunk_pairs)
        report, path = await features_task(tenant_id, cache_dir, chunk_pairs)
        mlflow_run = log_features_task(report)
    logger.info(
        "%d pairs in %d chunks, %d columns (%s); %d all null, %d constant; %s; %.1fs; "
        "mlflow run %s",
        report.pairs,
        report.chunks,
        len(report.columns),
        "cache hit" if report.cache_hit else "built",
        len(report.all_null_columns),
        len(report.constant_columns),
        path,
        report.seconds,
        mlflow_run,
    )
    return report, path, mlflow_run


# Read-only against both stores; the feature matrix is reused from the cache when nothing
# changed, and the scores file is renamed into place only when complete.
@task(
    name="score-pairs",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def score_pairs_task(tenant_id: str, cache_dir: Path) -> tuple[ScoreReport, Path]:
    async with await neo4j() as graph, await mongo() as repo:
        await graph.check_server()
        return await score_pairs(graph, repo, tenant_id, cache_dir=cache_dir)


@task(name="mlflow-log-scores", cache_policy=NONE)
def log_scores_task(report: ScoreReport) -> str:
    return log_scores(report, summarise_scores(report))


@flow(name="score-pairs")
async def score_pairs_flow(
    tenant_id: str, cache_dir: Path = CACHE_DIR
) -> tuple[ScoreReport, Path, str]:
    """The baseline score, tier and top contributions of every candidate pair, written beside
    the cached feature matrix; nothing is written to the stores, the run is logged to MLflow."""
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("baseline scoring of tenant %s", tenant_id)
        report, path = await score_pairs_task(tenant_id, cache_dir)
        mlflow_run = log_scores_task(report)
    logger.info(
        "%d pairs scored with weights %s; tiers %s; p10 %s, p50 %s, p90 %s; %s; %.1fs; "
        "mlflow run %s",
        report.pairs,
        report.weights.version,
        report.tiers,
        report.score_p10,
        report.score_p50,
        report.score_p90,
        path,
        report.seconds,
        mlflow_run,
    )
    return report, path, mlflow_run


# Both scores are recomputed from the stored vectors and replace the previous ones, so a retry
# converges on the same edges.
@task(
    name="score-links",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def score_links_task(tenant_id: str) -> LinkRelevanceReport:
    async with await neo4j() as graph:
        await graph.check_server()
        return await score_links(graph, tenant_id)


@task(name="mlflow-log-link-relevance", cache_policy=NONE)
def log_link_relevance_task(report: LinkRelevanceReport) -> str:
    return log_link_relevance(report, summarise_link_relevance(report))


@flow(name="score-links")
async def score_links_flow(tenant_id: str) -> tuple[LinkRelevanceReport, str]:
    """Context relevance and anchor-target fit of every existing body link, written on the
    LINKS_TO edges after embed-links; their distributions are logged to MLflow."""
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("scoring the existing links of tenant %s", tenant_id)
        report = await score_links_task(tenant_id)
        mlflow_run = log_link_relevance_task(report)
    logger.info(
        "%d of %d links scored; %d generic anchors, %d without an anchor vector; context split "
        "%s, anchor split %s; %.1fs; mlflow run %s",
        report.scored,
        report.links,
        report.generic_anchors,
        report.without_anchor_vector,
        report.context.split if report.context else None,
        report.anchor.split if report.anchor else None,
        report.seconds,
        mlflow_run,
    )
    return report, mlflow_run


# Read-only against both stores; both files are renamed into place only when complete, so a
# retry writes them again whole.
@task(
    name="hub-bridges",
    retries=1,
    retry_delay_seconds=30,
    retry_condition_fn=is_transient,
    cache_policy=NONE,
)
async def bridges_task(tenant_id: str, cache_dir: Path) -> tuple[BridgeReport, Path]:
    async with await neo4j() as graph, await mongo() as repo:
        await graph.check_server()
        return await find_bridges(graph, repo, tenant_id, cache_dir=cache_dir)


@task(name="mlflow-log-bridges", cache_policy=NONE)
def log_bridges_task(report: BridgeReport, path: Path) -> str:
    pairs = read_hub_pairs(path.with_name(HUB_PAIRS_FILE))
    return log_bridges(report, pairs, summarise_bridges(report))


@flow(name="hub-bridges")
async def hub_bridges_flow(
    tenant_id: str, cache_dir: Path = CACHE_DIR
) -> tuple[BridgeReport, Path, str]:
    """Bridge links that keep every hub connected to the others, written with the scored hub
    pairs under ``cache_dir``; nothing is written to the stores, the run is logged to MLflow
    without page urls."""
    logger = get_run_logger()
    with bound_contextvars(run_id=str(flow_run.id)):
        logger.info("hub bridges of tenant %s", tenant_id)
        report, path = await bridges_task(tenant_id, cache_dir)
        mlflow_run = log_bridges_task(report, path)
    logger.info(
        "%d hubs, %d pairs; %d directions below the floor need %d links, %d proposed, %d short; "
        "components %d -> %d; %s; %.1fs; mlflow run %s",
        report.hubs,
        report.hub_pairs,
        report.directions_below_floor,
        report.links_needed,
        report.bridge_links,
        report.directions_short,
        report.components_before,
        report.components_after,
        path,
        report.seconds,
        mlflow_run,
    )
    return report, path, mlflow_run
