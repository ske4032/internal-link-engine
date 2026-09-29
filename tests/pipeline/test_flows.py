from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest
from link_audit_seed import AUDITED as AUDIT_LINKS
from link_audit_seed import PAGES as AUDIT_PAGES
from link_audit_seed import seed_link_audit
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text
from prefect import flow
from prefect.runtime import flow_run
from prefect.states import Failed
from pydantic import ValidationError
from pymongo import AsyncMongoClient
from quality_seed import KEYWORDS, URLS, seed_quality
from quality_seed import voyage as keyword_voyage
from ranking_seed import seed_ranking
from ranking_seed import voyage as ranking_voyage
from store_state import graph_state, mongo_state
from test_anchor_stage import GUIDE
from test_anchor_stage import seed as seed_anchors
from test_bridge_stage import HUB_PAGES
from test_bridge_stage import seed as seed_bridges
from test_bridge_stage import url as bridge_url
from test_embed import DIM, record, seed, seed_plain, url
from test_embed_links import text_vector
from test_feature_stage import seed as seed_features
from test_keyword_stage import EXPECTED as EXPECTED_KEYWORD_EDGES
from test_keyword_stage import edges as keyword_edges
from test_keyword_stage import seed as seed_keywords
from test_recommendations import A, S, T, U, seed_tenant
from voyage_fakes import FakeVoyage, client, page_index
from voyageai.error import InvalidRequestError, ServiceUnavailableError

from linking_engine.discovery.candidates import retrieve_candidates, summarise_candidates
from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseReadError,
    DatabaseUnavailableError,
    EmbeddingAuthError,
    EmbeddingModelMismatchError,
    EmbeddingRequestError,
    EmbeddingUnavailableError,
)
from linking_engine.models import (
    DuplicateGroup,
    HeldOutSettings,
    Link,
    LinkRecord,
    Page,
    StageStatus,
)
from linking_engine.pipeline import flows, recommendations
from linking_engine.pipeline.duplicates import summarise_duplicates
from linking_engine.pipeline.tenant_pipeline import NEEDS, REPORT_STAGES, PipelineFailedError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mlflow.entities import Run

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import CandidateSet
    from linking_engine.pipeline.tenant_pipeline import Runner


@pytest.fixture(scope="session")
def prefect_api() -> Iterator[None]:
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness():
        yield


@pytest.fixture
def flow_env(
    monkeypatch: pytest.MonkeyPatch,
    neo4j_server: tuple[str, str, str],
    mongo_uri: str,
    prefect_api: None,
) -> None:
    uri, user, password = neo4j_server
    monkeypatch.setenv("NEO4J_URI", uri)
    monkeypatch.setenv("NEO4J_USER", user)
    monkeypatch.setenv("NEO4J_PASSWORD", password)
    monkeypatch.setenv("MONGO_URI", mongo_uri)
    monkeypatch.setenv("MONGO_DB", "linking_engine_test")
    for name in ("embed_pages_task", "embed_links_task", "candidates_task"):
        task = getattr(flows, name)
        monkeypatch.setattr(flows, name, task.with_options(retry_delay_seconds=0))


def use(monkeypatch: pytest.MonkeyPatch, fake: FakeVoyage) -> None:
    monkeypatch.setattr(flows, "voyage_client", lambda _tenant: client(fake))


@pytest.mark.integration
async def test_page_flow_embeds_the_tenants_pages(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_plain(mongo, graph, tenant, 3)
    fake = FakeVoyage(dimension=DIM)
    use(monkeypatch, fake)

    report = await flows.embed_pages_flow(tenant, 2)

    assert (report.selected, report.embedded, report.flushes) == (3, 3, 2)
    assert fake.call_count == 2


@pytest.mark.integration
async def test_rejected_request_fails_the_run_without_a_task_retry(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_plain(mongo, graph, tenant, 2)
    fake = FakeVoyage(dimension=DIM, failures=[InvalidRequestError("bad input", http_status=400)])
    use(monkeypatch, fake)

    state = await flows.embed_pages_flow(tenant, 2, return_state=True)

    assert state.is_failed()
    assert fake.call_count == 1


@pytest.mark.integration
async def test_outage_retries_the_task_and_embeds_only_the_uncommitted_flush(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_plain(mongo, graph, tenant, 4)
    # Flush 1 succeeds; every client attempt of flush 2 (calls 2-4) hits an outage.
    outage = {n: ServiceUnavailableError("down", http_status=503) for n in (2, 3, 4)}
    fake = FakeVoyage(dimension=DIM, fail_on=outage)
    use(monkeypatch, fake)

    report = await flows.embed_pages_flow(tenant, 2)

    assert fake.call_count == 5
    assert (report.selected, report.embedded) == (2, 2)
    assert sorted(page_index(text) for text in fake.calls[-1].texts) == [2, 3]


# p000 links to p001 and p002, p001 to p002: 2 anchor keys (1 generic), 2 distinct sentences.
SHARED = "Our trail shoes grip wet rock."
OTHER = "Click here for sizing."
LINKS = (
    LinkRecord(
        source_url=url(0),
        position=0,
        target_url=url(1),
        anchor_text="Trail Shoes",
        surrounding_text=SHARED,
        is_internal=True,
    ),
    LinkRecord(
        source_url=url(0),
        position=1,
        target_url=url(2),
        anchor_text="click here",
        surrounding_text=OTHER,
        is_internal=True,
    ),
    LinkRecord(
        source_url=url(1),
        position=0,
        target_url=url(2),
        anchor_text="trail shoes",
        surrounding_text=SHARED,
        is_internal=True,
    ),
)


async def seed_linked(mongo: MongoRepo, graph: GraphRepo, tenant: str) -> None:
    await seed(mongo, graph, tenant, [record(0, links=2), record(1, links=1), record(2)], LINKS)


def any_text(**options: object) -> FakeVoyage:
    """Vectors for page bodies and for anchors or sentences alike."""
    return FakeVoyage(
        dimension=DIM,
        respond=lambda texts: [text_vector(text) for text in texts],
        **options,  # type: ignore[arg-type]
    )


@pytest.mark.integration
async def test_link_flow_embeds_anchors_and_sentences_without_page_vectors(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_linked(mongo, graph, tenant)
    fake = any_text()
    use(monkeypatch, fake)

    links = await flows.embed_links_flow(tenant, 2)

    assert (links.edges, links.unique_anchors, links.generic_anchors, links.anchors_embedded) == (
        3,
        2,
        1,
        1,
    )
    assert (links.unique_sentences, links.sentences_embedded, links.surrounding_edges_written) == (
        2,
        2,
        3,
    )
    sent = {text for call in fake.calls for text in call.texts}
    assert sent == {"trail shoes", SHARED, OTHER}, "the link flow never embeds page bodies"


@pytest.mark.integration
async def test_a_failed_page_flow_leaves_the_link_flow_unaffected(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_linked(mongo, graph, tenant)
    use(monkeypatch, any_text(failures=[InvalidRequestError("bad input", http_status=400)]))
    pages = await flows.embed_pages_flow(tenant, 2, return_state=True)
    use(monkeypatch, any_text())

    links = await flows.embed_links_flow(tenant, 2)

    assert pages.is_failed()
    assert (links.anchors_embedded, links.sentences_embedded) == (1, 2)


@pytest.mark.integration
async def test_link_outage_retries_only_the_link_task_and_embeds_only_the_rest(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_linked(mongo, graph, tenant)
    # Flush size 1: the anchor is call 1, sentences 2 and 3. Every client attempt of the
    # second sentence flush (calls 3-5) hits an outage; the task retry is call 6.
    outage = {n: ServiceUnavailableError("down", http_status=503) for n in (3, 4, 5)}
    fake = any_text(fail_on=outage)
    use(monkeypatch, fake)

    links = await flows.embed_links_flow(tenant, 1)

    assert fake.call_count == 6
    assert set(fake.calls[-1].texts) == {SHARED, OTHER} - set(fake.calls[1].texts)
    assert (links.anchors_cached, links.anchors_embedded) == (1, 0)
    assert (links.sentences_cached, links.sentences_embedded, links.keys_written) == (1, 1, 0)


def test_voyage_client_uses_the_tenant_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    voyage = flows.voyage_client("tenant-a")
    assert (voyage.model, voyage.dimension) == ("voyage-4-large", 2048)


@pytest.mark.parametrize(
    ("error", "retry"),
    [
        (EmbeddingUnavailableError("outage"), True),
        (DatabaseUnavailableError("neo4j", "down"), True),
        (EmbeddingAuthError("bad key", status_code=401), False),
        (DatabaseAuthError("mongodb", "bad password"), False),
        (EmbeddingRequestError("bad input", status_code=400), False),
        (EmbeddingModelMismatchError("mixed models"), False),
    ],
)
def test_only_outages_are_retried(error: Exception, retry: bool) -> None:
    assert flows.is_transient(None, None, Failed(data=error)) is retry


@pytest.mark.integration
async def test_graph_analytics_flow_writes_to_neo4j_and_logs_one_mlflow_run(
    graph: GraphRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    urls = [f"example.com/{name}" for name in ("a", "b", "c", "d")]
    await graph.upsert_pages(tenant, [Page(url=u, status_code=200) for u in urls])
    await graph.replace_links(
        tenant,
        urls,
        [
            Link(source_url=s, target_url=t, position=0, anchor_text="x", surrounding_text="")
            for s, t in ((urls[0], urls[1]), (urls[1], urls[2]), (urls[2], urls[0]))
        ],
    )

    centrality, communities, hubs, run_id = await flows.graph_analytics_flow(tenant)

    assert (centrality.pages, communities.crawled_pages, communities.orphans) == (4, 4, 1)
    assert (hubs.pages, hubs.hubs) == (0, 0)
    [lone] = await graph.get_pages(tenant, [urls[3]])
    assert (lone.page_rank is not None, lone.is_orphan, lone.link_community_id) == (
        True,
        True,
        None,
    )
    run = MlflowClient(uri).get_run(run_id)
    assert run.data.tags["tenant_id"] == tenant
    assert run.data.metrics["link_pages"] == 3


NAV_LINE = "- [Pricing](https://example.com/pricing)"


async def seed_crawl(mongo_uri: str, database: str, copies: tuple[str, ...] = ()) -> None:
    """A chain of six pages; each of ``copies`` serves page a's content at another path."""
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    pages = [
        *zip(("pricing", "a", "b", "c", "d", "e"), ("a", "b", "c", "d", "e", "a"), strict=True),
        *((copy, "b") for copy in copies),
    ]
    await client[database]["crawl_pages"].insert_many(
        [
            {
                "url": f"https://example.com/{path}",
                "title": path,
                "content": f"{NAV_LINE}\n\n{path.rsplit('/', 1)[-1].title()} page. Read "
                f"[the next one](https://example.com/{nxt}) as well.",
                "statusCode": 200,
                "usable": True,
            }
            for path, nxt in pages
        ]
    )
    await client.close()


@pytest.mark.integration
async def test_prepare_and_load_flows_ingest_a_crawl_into_mongo_and_neo4j(
    graph: GraphRepo,
    mongo_uri: str,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    source_db = f"crawl_{tenant.replace('-', '_')}"
    await seed_crawl(mongo_uri, source_db)

    prepared = await flows.prepare_corpus_flow(tenant, source_db, "crawl_pages")
    loaded, counts, duplicates, _ = await flows.load_graph_flow(tenant)

    assert (prepared.documents, prepared.pages, prepared.links, prepared.pages_written) == (
        6,
        6,
        6,
        6,
    )
    assert prepared.menu_inlink_pages == 1
    assert (loaded.pages, loaded.links, counts.pages) == (6, 6, 6)
    [pricing] = await graph.get_pages(tenant, ["example.com/pricing"])
    assert pricing.menu_inlinks == 5
    assert (duplicates.tenant_id, duplicates.groups) == (tenant, ())


@pytest.mark.integration
async def test_the_load_flow_groups_exact_duplicates_and_logs_them_to_mlflow(
    graph: GraphRepo,
    mongo_uri: str,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    source_db = f"crawl_{tenant.replace('-', '_')}"
    await seed_crawl(mongo_uri, source_db, copies=("blog/a", "news/a"))
    # Page a's sentence is on three of eight pages: keep it out of the boilerplate.
    await flows.prepare_corpus_flow(tenant, source_db, "crawl_pages", boilerplate_share=0.5)

    loaded, _, duplicates, run_id = await flows.load_graph_flow(tenant)

    # a is linked from pricing and e, its copies from nowhere.
    copies = ("example.com/blog/a", "example.com/news/a")
    assert loaded.pages == 8
    assert duplicates.groups == (
        DuplicateGroup(group_id=0, canonical="example.com/a", copies=copies),
    )
    assert await graph.non_canonical_copies(tenant) == set(copies)
    [canonical] = await graph.get_pages(tenant, ["example.com/a"])
    assert (canonical.duplicate_group, canonical.is_canonical) == (0, True)
    run = MlflowClient(uri).get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "duplicates")
    assert run.data.tags["mlflow.note.content"] == summarise_duplicates(duplicates)
    assert run.data.metrics["non_canonical"] == 2
    assert load_dict(f"runs:/{run_id}/groups.json") == {
        "groups": [{"group_id": 0, "canonical": "example.com/a", "copies": list(copies)}]
    }


@pytest.mark.integration
async def test_the_prepare_flow_refuses_the_project_database_as_its_source(
    tenant: str, flow_env: None
) -> None:
    state = await flows.prepare_corpus_flow(
        tenant, "linking_engine_test", "crawl_pages", return_state=True
    )

    assert state.is_failed()
    with pytest.raises(ValueError, match="source and target database must differ"):
        await state.result()


# a -> b -> c -> a; d is linked from nowhere.
CANDIDATE_VECTORS = {
    "a": [1.0, 0.0, 0.0, 0.0],
    "b": [0.9, 0.1, 0.0, 0.0],
    "c": [0.0, 1.0, 0.0, 0.0],
    "d": [0.0, 0.0, 1.0, 0.5],
}


async def seed_candidates(graph: GraphRepo, tenant: str) -> dict[str, str]:
    urls = {name: f"example.com/{name}" for name in CANDIDATE_VECTORS}
    await graph.upsert_pages(tenant, [Page(url=u, status_code=200) for u in urls.values()])
    await graph.replace_links(
        tenant,
        list(urls.values()),
        [
            Link(
                source_url=urls[s],
                target_url=urls[t],
                position=0,
                anchor_text="x",
                surrounding_text="",
            )
            for s, t in (("a", "b"), ("b", "c"), ("c", "a"))
        ],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec",
        t=tenant,
        rows=[{"url": urls[name], "vec": vec} for name, vec in CANDIDATE_VECTORS.items()],
    )
    return urls


@pytest.mark.integration
async def test_candidate_retrieval_flow_reads_neo4j_and_logs_one_mlflow_run(
    graph: GraphRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    u = await seed_candidates(graph, tenant)

    found, run_id = await flows.candidate_retrieval_flow(tenant)

    assert {t.target_url: set(t.sources) for t in found.targets} == {
        u["a"]: {u["b"], u["d"]},
        u["b"]: {u["c"], u["d"]},
        u["c"]: {u["a"], u["d"]},
        u["d"]: {u["a"], u["b"], u["c"]},
    }
    report = found.report
    assert (report.tenant_id, report.targets, report.indexable_assumed, report.candidates) == (
        tenant,
        4,
        4,
        9,
    )
    assert (report.linked_pairs, report.linked_nearer, report.drop_rate) == (3, 3, 0.25)
    run = MlflowClient(uri).get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "candidate-retrieval")
    assert run.data.tags["mlflow.note.content"] == summarise_candidates(report)
    assert run.data.params["index"] == "page_content"
    assert run.data.metrics["candidates"] == 9


def failing_first(monkeypatch: pytest.MonkeyPatch, error: Exception, *, times: int) -> list[str]:
    """Make retrieve_candidates raise ``error`` on its first ``times`` calls; returns the calls."""
    calls: list[str] = []

    async def retrieve(graph: GraphRepo, tenant_id: str, **options: object) -> CandidateSet:
        calls.append(tenant_id)
        if len(calls) <= times:
            raise error
        return await retrieve_candidates(graph, tenant_id, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(flows, "retrieve_candidates", retrieve)
    return calls


@pytest.mark.integration
async def test_a_neo4j_outage_retries_candidate_retrieval_once(
    graph: GraphRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    await seed_candidates(graph, tenant)
    calls = failing_first(monkeypatch, DatabaseUnavailableError("neo4j", "down"), times=1)

    found, _ = await flows.candidate_retrieval_flow(tenant)

    assert calls == [tenant, tenant]
    assert found.report.candidates == 9


@pytest.mark.integration
async def test_unreadable_graph_data_fails_candidate_retrieval_without_a_retry(
    tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    calls = failing_first(monkeypatch, DatabaseReadError("neo4j", "bad vector"), times=2)

    state = await flows.candidate_retrieval_flow(tenant, return_state=True)

    assert state.is_failed()
    assert calls == [tenant]


@pytest.mark.integration
async def test_resolve_keywords_flow_writes_edges_and_logs_one_mlflow_run(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    await seed_keywords(graph, mongo, tenant, "Trail Running Shoes")

    report, run_id = await flows.resolve_keywords_flow(tenant)

    assert (report.tenant_id, report.resolved) == (tenant, 5)
    assert await keyword_edges(graph, tenant) == EXPECTED_KEYWORD_EDGES
    run = MlflowClient(uri).get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "resolve-keywords")
    assert run.data.metrics["rung_gsc"] == 1


@pytest.mark.integration
async def test_feature_assembly_flow_writes_the_matrix_and_logs_one_mlflow_run(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    await seed_features(graph, mongo, tenant)

    report, path, run_id = await flows.feature_assembly_flow(tenant, tmp_path / "features", 5)

    assert path == tmp_path / "features" / tenant / f"{report.cache_key}.parquet"
    assert path.is_file()
    assert report.pairs > 0
    run = MlflowClient(uri).get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "feature-assembly")
    assert run.data.metrics["pairs"] == report.pairs


@pytest.mark.integration
async def test_score_pairs_flow_writes_the_scores_and_logs_one_mlflow_run(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    await seed_features(graph, mongo, tenant)

    report, path, run_id = await flows.score_pairs_flow(tenant, tmp_path / "features")

    assert path.parent == tmp_path / "features" / tenant
    assert path.name.endswith(f".{report.weights_hash[:12]}.scores.parquet")
    assert path.is_file()
    client = MlflowClient(uri)
    run = client.get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "score-pairs")
    assert run.data.metrics["pairs"] == report.pairs
    assert len(client.get_metric_history(run_id, "score_hist")) == 20


async def seed_link_relevance(graph: GraphRepo, tenant: str) -> dict[str, str]:
    """a -> b with a stored anchor vector, b -> c with a generic anchor, c -> a whose anchor
    has no vector, and a -> d whose target has no content vector; every link has a sentence
    vector."""
    urls = {name: f"example.com/rel/{name}" for name in ("a", "b", "c", "d")}
    await graph.upsert_pages(tenant, [Page(url=u, status_code=200) for u in urls.values()])
    await graph.replace_links(
        tenant,
        list(urls.values()),
        [
            Link(
                source_url=urls[s],
                target_url=urls[t],
                position=position,
                anchor_text="x",
                surrounding_text="",
            )
            for s, t, position in (("a", "b", 0), ("b", "c", 0), ("c", "a", 0), ("a", "d", 1))
        ],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec",
        t=tenant,
        rows=[
            {"url": urls["a"], "vec": [1.0, 0.0, 0.0]},
            {"url": urls["b"], "vec": [0.0, 1.0, 0.0]},
            {"url": urls["c"], "vec": [0.0, 0.0, 1.0]},
        ],
    )
    await graph._auto(
        "UNWIND $rows AS row "
        "MATCH (:Page {tenantId: $t, url: row.s})-[r:LINKS_TO]->(:Page {tenantId: $t, url: row.t}) "
        "SET r.surroundingEmbedding = row.vec, r.anchorKey = row.key, r.anchorGeneric = row.generic",
        t=tenant,
        rows=[
            {
                "s": urls["a"],
                "t": urls["b"],
                "vec": [0.0, 1.0, 0.0],
                "key": "trail shoes",
                "generic": False,
            },
            {
                "s": urls["b"],
                "t": urls["c"],
                "vec": [0.0, 0.6, 0.8],
                "key": "click here",
                "generic": True,
            },
            {
                "s": urls["c"],
                "t": urls["a"],
                "vec": [0.6, 0.8, 0.0],
                "key": "tents",
                "generic": False,
            },
            {
                "s": urls["a"],
                "t": urls["d"],
                "vec": [1.0, 0.0, 0.0],
                "key": "trail shoes",
                "generic": False,
            },
        ],
    )
    await graph._auto(
        "CREATE (:Anchor {tenantId: $t, text: 'trail shoes', embedding: [0.0, 0.8, 0.6]}), "
        "(:Anchor {tenantId: $t, text: 'click here', embedding: [0.0, 0.0, 1.0]})",
        t=tenant,
    )
    # Every stored vector comes from one model, as embed-pages and embed-links write them.
    await graph._auto(
        "MATCH (p:Page {tenantId: $t}) WHERE p.content_embedding IS NOT NULL "
        "SET p.embeddingModel = $m "
        "WITH count(p) AS pages "
        "MATCH (:Page {tenantId: $t})-[r:LINKS_TO]->() SET r.surroundingEmbeddingModel = $m "
        "WITH count(r) AS links "
        "MATCH (a:Anchor {tenantId: $t}) SET a.embeddingModel = $m",
        t=tenant,
        m=MODEL,
    )
    return urls


MODEL = "voyage-4-large"


@pytest.mark.parametrize(
    "mismatch",
    [
        pytest.param(
            "MATCH (a:Anchor {tenantId: $t, text: 'trail shoes'}) SET a.embeddingModel = $other",
            id="anchor-from-another-model",
        ),
        pytest.param(
            "MATCH (:Page {tenantId: $t})-[r:LINKS_TO]->() SET r.surroundingEmbeddingModel = $other",
            id="sentences-from-another-model",
        ),
        pytest.param(
            "MATCH (p:Page {tenantId: $t}) REMOVE p.embeddingModel", id="pages-without-a-model"
        ),
    ],
)
@pytest.mark.integration
async def test_vectors_from_different_models_fail_score_links_without_writing_scores(
    graph: GraphRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    await seed_link_relevance(graph, tenant)
    await graph._auto(mismatch, t=tenant, other="voyage-3-large")

    state = await flows.score_links_flow(tenant, return_state=True)

    assert state.is_failed()
    with pytest.raises(EmbeddingModelMismatchError):
        await state.aresult()
    [written] = await graph._auto(
        "MATCH (:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "WHERE r.contextRelevance IS NOT NULL OR r.anchorTargetFit IS NOT NULL "
        "RETURN count(r) AS n",
        t=tenant,
    )
    assert written["n"] == 0, "no score may be written from mixed models"


@pytest.mark.integration
async def test_score_links_flow_scores_existing_links_and_logs_one_mlflow_run(
    graph: GraphRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    urls = await seed_link_relevance(graph, tenant)

    report, run_id = await flows.score_links_flow(tenant)

    # a -> d is cleared: its target has no content vector.
    assert (report.links, report.scored) == (4, 3)
    assert (report.generic_anchors, report.without_anchor_vector) == (1, 1)
    assert report.context is not None
    assert report.anchor is not None
    assert (report.context.count, report.anchor.count) == (3, 1)
    stored = {
        (row.source_url, row.target_url): (row.context_relevance, row.anchor_target_fit)
        for row in await graph.link_relevance(tenant)
    }
    # Neo4j's normalised cosine, (1 + cos) / 2.
    assert set(stored) == {(urls["a"], urls["b"]), (urls["b"], urls["c"]), (urls["c"], urls["a"])}
    assert stored[urls["a"], urls["b"]][0] == pytest.approx(1.0)
    assert stored[urls["a"], urls["b"]][1] == pytest.approx(0.9)
    assert stored[urls["b"], urls["c"]][0] == pytest.approx(0.9)
    assert stored[urls["c"], urls["a"]][0] == pytest.approx(0.8)
    assert (stored[urls["b"], urls["c"]][1], stored[urls["c"], urls["a"]][1]) == (None, None)
    client = MlflowClient(uri)
    run = client.get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "score-links")
    assert run.data.metrics["scored"] == 3
    for name in ("context_relevance_hist", "anchor_target_fit_hist"):
        assert len(client.get_metric_history(run_id, name)) == 20, name
    assert "relevance_histogram.json" in {a.path for a in client.list_artifacts(run_id)}


@pytest.mark.integration
async def test_hub_bridges_flow_writes_both_files_and_logs_one_run_without_urls(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    await seed_bridges(graph, tenant, links=[("a0", "b0")])

    report, path, run_id = await flows.hub_bridges_flow(tenant, tmp_path / "bridges")

    assert path == tmp_path / "bridges" / tenant / "bridges.parquet"
    assert path.with_name("hub_pairs.parquet").is_file()
    assert report.bridge_links > 0
    client = MlflowClient(uri)
    run = client.get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "hub-bridges")
    assert run.data.metrics["bridge_links"] == report.bridge_links
    stored = load_dict(f"runs:/{run_id}/hub_pairs.json")
    assert len(stored["data"]) == report.hub_pairs
    logged = " ".join(
        [
            *map(str, run.data.params.values()),
            *map(str, run.data.tags.values()),
            str(stored),
            load_text(f"runs:/{run_id}/report.json"),
            load_text(f"runs:/{run_id}/summary.md"),
        ]
    )
    pages = [bridge_url(name) for names in HUB_PAGES.values() for name in names]
    assert [u for u in pages if u in logged] == [], "page urls reached the MLflow run"


@pytest.mark.integration
async def test_anchor_extraction_flow_writes_the_anchors_and_logs_one_run_without_text(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    await seed_anchors(graph, mongo, tenant)

    report, path, run_id = await flows.anchor_extraction_flow(tenant, tmp_path / "anchors")

    assert path == tmp_path / "anchors" / tenant / "anchors.parquet"
    assert path.is_file()
    assert report.matches > 0
    client = MlflowClient(uri)
    run = client.get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "anchor-extraction")
    assert run.data.metrics["matches"] == report.matches
    assert len(client.get_metric_history(run_id, "sentence_index_hist")) == 7
    logged = " ".join(
        [
            *map(str, run.data.params.values()),
            *map(str, run.data.tags.values()),
            str(load_dict(f"runs:/{run_id}/rung_by_rank.json")),
            str(load_dict(f"runs:/{run_id}/languages.json")),
            load_text(f"runs:/{run_id}/report.json"),
            load_text(f"runs:/{run_id}/summary.md"),
        ]
    )
    texts = ["example.com/g", "example.com/s", "trail shoes", "running shoes", GUIDE]
    assert [text for text in texts if text in logged] == [], "text reached the MLflow run"


# ── quality eval ────────────────────────────────────────────────────────────

# A deliberately low-entropy stand-in for a Voyage API key.
VOYAGE_KEY = "f" * 64


def test_keyword_voyage_is_none_without_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flows, "voyage_client", lambda _tenant: pytest.fail("no key, no client"))
    assert flows.keyword_voyage("test-tenant") is None


def test_keyword_voyage_is_the_tenants_client_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", VOYAGE_KEY)
    voyage = flows.keyword_voyage("test-tenant")
    assert voyage is not None
    assert (voyage.model, voyage.dimension) == ("voyage-4-large", 2048)


@pytest.mark.parametrize("with_key", [True, False], ids=["with-key", "without-key"])
def test_keyword_voyage_raises_on_any_other_settings_error(
    monkeypatch: pytest.MonkeyPatch, with_key: bool
) -> None:
    if with_key:
        monkeypatch.setenv("VOYAGE_API_KEY", VOYAGE_KEY)
    monkeypatch.setenv("VOYAGE_MAX_ATTEMPTS", "0")
    with pytest.raises(ValidationError, match="max_attempts"):
        flows.keyword_voyage("test-tenant")


def run_text(client: MlflowClient, run_id: str) -> str:
    """Every tag, param, metric name and artifact of a run, joined."""
    run = client.get_run(run_id)
    return " ".join(
        [
            *run.data.tags.values(),
            *run.data.params.values(),
            *run.data.metrics,
            *(load_text(f"runs:/{run_id}/{a.path}") for a in client.list_artifacts(run_id)),
        ]
    )


@pytest.mark.integration
async def test_anchor_selection_flow_writes_both_files_and_logs_one_run_without_text(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("VOYAGE_API_KEY", VOYAGE_KEY)
    fake = keyword_voyage()
    use(monkeypatch, fake)
    await seed_quality(graph, mongo, tenant)
    stored_graph, stored_mongo = await graph_state(graph, tenant), await mongo_state(mongo)

    report, path, run_id = await flows.anchor_selection_flow(tenant, tmp_path / "features")

    assert await graph_state(graph, tenant) == stored_graph
    assert await mongo_state(mongo) == stored_mongo
    assert path == tmp_path / "features" / tenant / "anchor_choices.parquet"
    assert (path.parent / "unanchored_pairs.parquet").is_file()
    assert (report.semantic_skipped_reason, report.embedding_skipped_reason) == (None, None)
    assert fake.call_count > 0
    assert report.chosen > 0
    client = MlflowClient(uri)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == f"analytics-{tenant}"
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "anchor-selection")
    assert run.data.metrics["chosen"] == report.chosen
    for name in ("anchor_score_hist", "semantic_similarity_hist"):
        assert len(client.get_metric_history(run_id, name)) == 20, name
    choices = pq.read_table(path).to_pandas()
    texts = {
        *URLS,
        *KEYWORDS,
        *choices["phrase"],
        *choices["sentence"],
        *choices["keyword"],
    }
    logged = run_text(client, run_id)
    assert [text for text in texts if text in logged] == [], "text reached the MLflow run"


@pytest.mark.integration
async def test_quality_eval_flow_logs_one_read_only_run_without_urls_or_keywords(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("GIT_SHA", "c" * 40)
    monkeypatch.setenv("VOYAGE_API_KEY", VOYAGE_KEY)
    fake = keyword_voyage()
    use(monkeypatch, fake)
    await seed_quality(graph, mongo, tenant)
    stored_graph, stored_mongo = await graph_state(graph, tenant), await mongo_state(mongo)

    report, run_id = await flows.quality_eval_flow(tenant, tmp_path / "features")

    assert await graph_state(graph, tenant) == stored_graph
    assert await mongo_state(mongo) == stored_mongo
    assert report.not_applicable == ()
    assert fake.call_count > 0
    client = MlflowClient(uri)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == f"analytics-{tenant}"
    assert {
        name: run.data.tags[name] for name in ("stage", "git_sha", "alerts", "baseline_run")
    } == {
        "stage": "quality-eval",
        "git_sha": "c" * 40,
        "alerts": "none",
        "baseline_run": "none",
    }
    assert run.data.metrics["recall_at_10"] == 1.0
    assert "no baseline" in load_text(f"runs:/{run_id}/summary.md")
    logged = run_text(client, run_id)
    assert [u for u in URLS if u in logged] == [], "page urls reached the MLflow run"
    folded = logged.casefold()
    assert [k for k in KEYWORDS if k.casefold() in folded] == [], "keywords reached the run"


@pytest.mark.integration
async def test_quality_eval_compares_with_the_previous_run_and_an_alert_never_fails_it(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("VOYAGE_API_KEY", VOYAGE_KEY)
    use(monkeypatch, keyword_voyage())
    await seed_quality(graph, mongo, tenant)
    cache = tmp_path / "features"
    _, first = await flows.quality_eval_flow(tenant, cache)

    report, second = await flows.quality_eval_flow(tenant, cache)

    client = MlflowClient(uri)
    tags = client.get_run(second).data.tags
    assert (tags["baseline_run"], tags["alerts"]) == (first, "none")
    assert report.keywords is not None
    assert report.keywords.relevance is not None
    assert report.keywords.relevance.embedded == 0, "cached keyword vectors were embedded again"
    assert "no headline metric moved" in load_text(f"runs:/{second}/summary.md")

    # A later finished run where held-out recall was far lower.
    experiment = client.get_experiment_by_name(f"analytics-{tenant}")
    assert experiment is not None
    doctored = client.create_run(
        experiment.experiment_id, tags={"stage": "quality-eval", "tenant_id": tenant}
    )
    client.log_metric(doctored.info.run_id, "recall_at_10", 0.25)
    client.set_terminated(doctored.info.run_id)

    moved, third = await flows.quality_eval_flow(tenant, cache)

    assert [alert.metric for alert in moved.alerts] == ["recall_at_10"]
    tags = client.get_run(third).data.tags
    assert (tags["baseline_run"], tags["alerts"]) == (doctored.info.run_id, "recall_at_10")
    assert "- recall_at_10: 0.25 -> 1 (+300.0%, band 0.2 relative)" in load_text(
        f"runs:/{third}/summary.md"
    )
    assert "alerts.json" in {a.path for a in client.list_artifacts(third)}


@pytest.mark.integration
async def test_quality_eval_without_a_voyage_key_reports_keyword_relevance_not_applicable(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(flows, "voyage_client", lambda _tenant: pytest.fail("no key, no client"))
    await seed_quality(graph, mongo, tenant)

    report, run_id = await flows.quality_eval_flow(tenant, tmp_path / "features")

    assert report.not_applicable == ("keyword_relevance",)
    run = MlflowClient(uri).get_run(run_id)
    assert run.data.tags["not_applicable"] == "keyword_relevance"
    assert not [name for name in run.data.metrics if name.startswith("keyword_relevance")]


@pytest.mark.integration
async def test_train_ranker_flow_logs_one_run_and_skips_with_too_few_groups(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("RANKER_PROMOTION", raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", VOYAGE_KEY)
    fake = ranking_voyage()
    use(monkeypatch, fake)
    await seed_ranking(graph, mongo, tenant)
    stored_graph, stored_mongo = await graph_state(graph, tenant), await mongo_state(mongo)

    report = await flows.train_ranker_flow(tenant, rounds=1, share=0.1, cache_dir=tmp_path)

    assert await graph_state(graph, tenant) == stored_graph
    assert await mongo_state(mongo) == stored_mongo
    assert fake.call_count > 0, "the flow did not pass its Voyage client to the anchor choice"
    assert report.skipped_reason is not None
    assert (report.settings.rounds, report.settings.share) == (1, 0.1)
    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name(f"ranker-{tenant}")
    assert experiment is not None
    [run] = client.search_runs([experiment.experiment_id])
    assert run.data.tags["skipped_reason"] == report.skipped_reason


@pytest.mark.integration
async def test_rank_pairs_flow_ranks_with_the_baseline_without_a_promoted_model(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    await seed_ranking(graph, mongo, tenant)

    report, path = await flows.rank_pairs_flow(tenant, tmp_path)

    assert path == tmp_path / tenant / "ranked_pairs.parquet"
    assert (report.scorer.value, report.fallback_reason) == ("baseline", "no promoted model")
    assert pq.read_metadata(path).num_rows == report.pairs > 0


def test_the_train_ranker_flow_defaults_are_the_held_out_settings_defaults() -> None:
    parameters = inspect.signature(flows.train_ranker_flow.fn).parameters
    defaults = HeldOutSettings()
    assert (parameters["rounds"].default, parameters["share"].default) == (
        defaults.rounds,
        defaults.share,
    ), "the flow hides a different share of the links than the settings it passes on"


@pytest.mark.integration
async def test_link_audit_flow_audits_the_tenant_and_logs_one_mlflow_run_without_urls(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    await seed_link_audit(graph, mongo, tenant)

    report, run_id = await flows.link_audit_flow(tenant, cache_dir=tmp_path / "cache")

    # Without a Voyage key alignment stays lexical; the planted verdicts do not depend on it.
    assert report.embeddings
    assert report.keyword_cosines == 0
    assert report.by_verdict == Counter(link.a2.verdict for link in AUDIT_LINKS if link.a2.verdict)
    assert {r.run_id for r in await mongo.latest_link_audit(tenant)} == {report.run_id}
    run = MlflowClient(uri).get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "link-audit")
    assert run.data.tags["audit_run_id"] == report.run_id
    assert run.data.metrics["links"] == len(AUDIT_LINKS)
    summary = load_text(f"runs:/{run_id}/summary.md")
    assert not [page.path for page in AUDIT_PAGES if page.path in summary]


@pytest.mark.integration
async def test_recommendations_flow_publishes_the_output_and_logs_one_run_without_urls(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    flow_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    cache = tmp_path / "cache"
    await seed_tenant(graph, mongo, tenant, cache / tenant)

    async def features(_graph: object, _mongo: object, name: str, **_: object) -> tuple[None, Path]:
        return None, cache / name / "matrix.parquet"

    monkeypatch.setattr(recommendations, "assemble_features", features)

    report, run_id = await flows.recommendations_flow(tenant, cache_dir=cache)

    served = await mongo._db["output_runs"].find_one({"tenantId": tenant, "status": "complete"})
    assert served is not None
    assert served["runId"] == report.run_id
    assert served["quality"] is None
    run = MlflowClient(uri).get_run(run_id)
    assert (run.data.tags["tenant_id"], run.data.tags["stage"]) == (tenant, "recommendations")
    assert run.data.tags["output_run_id"] == report.run_id
    assert run.data.metrics["action_add_link"] == 2
    words = {word.strip(".,;:()") for word in load_text(f"runs:/{run_id}/summary.md").split()}
    logged = {*run.data.tags.values(), *run.data.params.values(), *run.data.metrics, *words}
    assert not logged & {A, S, T, U}


CORE_STAGES = [stage for stage in NEEDS if stage not in REPORT_STAGES and stage != "train-ranker"]
SOURCE = ("crawls", "crawl_pages")


class PipelineStages:
    """Fake stage flows and stored outputs of one tenant. Each stage is a Prefect flow that
    records when it starts and ends and the flow runs it sees, then fails or writes the files
    recommendations reads; link-audit completes an audit."""

    def __init__(
        self, folder: Path, *, failing: str | None = None, until_failed: tuple[str, ...] = ()
    ) -> None:
        self.folder = folder
        self.folder.mkdir(parents=True)
        self.failing = failing
        # Stages that finish only once the failing stage's flow run has failed; it fails once
        # they have started.
        self.until_failed = until_failed
        self.begun = {stage: asyncio.Event() for stage in NEEDS}
        self.failed = asyncio.Event()
        self.events: list[tuple[str, str]] = []
        # Each stage's own flow run id and its root flow run id.
        self.runs: dict[str, tuple[str, str]] = {}
        self.audit_at: datetime | None = None

    def runners(
        self, _tenant: str, source_db: str | None, source_collection: str | None, _cache: Path
    ) -> dict[str, Runner]:
        source = source_db is not None and source_collection is not None
        return {
            stage: self.stage_flow(stage) for stage in NEEDS if source or stage != "prepare-corpus"
        }

    def stage_flow(self, stage: str) -> Runner:
        @flow(name=f"fake-{stage}")
        async def run() -> str:
            self.events.append(("start", stage))
            self.runs[stage] = (str(flow_run.id), str(flow_run.root_flow_run_id))
            self.begun[stage].set()
            await asyncio.sleep(0.01)
            if stage in self.until_failed:
                await asyncio.wait_for(self.failed.wait(), timeout=30)
            if stage == self.failing:
                begun = (self.begun[s].wait() for s in self.until_failed)
                await asyncio.wait_for(asyncio.gather(*begun), timeout=30)
            self.events.append(("end", stage))
            if stage == self.failing:
                raise RuntimeError(f"{stage} broke")
            for name, writer in recommendations.REQUIRED_FILES:
                if writer == stage:
                    (self.folder / name).write_bytes(b"")
            if stage == "link-audit":
                self.audit_at = datetime.now(UTC)
            return stage

        # The waiting stages are released once the failed flow run has returned, so the
        # scheduler holds the failure before they finish.
        async def runner() -> str:
            try:
                return await run()
            finally:
                if stage == self.failing:
                    self.failed.set()

        return runner

    async def audit_completed(self) -> datetime | None:
        return self.audit_at

    def at(self, kind: str, stage: str) -> int:
        return self.events.index((kind, stage))


@pytest.fixture
def pipeline_mlflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prefect_api: None
) -> MlflowClient:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return MlflowClient(uri)


def fake_stages(
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    tenant: str,
    *,
    failing: str | None = None,
    until_failed: tuple[str, ...] = (),
) -> PipelineStages:
    stages = PipelineStages(cache / tenant, failing=failing, until_failed=until_failed)
    monkeypatch.setattr(flows, "stage_runners", stages.runners)
    monkeypatch.setattr(flows, "StoredOutputs", lambda *_: stages)
    return stages


def pipeline_run(client: MlflowClient, tenant: str) -> Run:
    experiment = client.get_experiment_by_name(f"analytics-{tenant}")
    assert experiment is not None
    [run] = client.search_runs([experiment.experiment_id], "tags.kind = 'pipeline'")
    return run


async def test_the_tenant_pipeline_flow_runs_every_planned_stage_as_a_subflow_in_dependency_order(
    tenant: str, pipeline_mlflow: MlflowClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    stages = fake_stages(monkeypatch, cache, tenant)

    report = await flows.tenant_pipeline_flow(tenant, *SOURCE, cache_dir=cache)

    assert {r.stage: r.status for r in report.stages} == dict.fromkeys(CORE_STAGES, StageStatus.OK)
    own_runs = {own for own, _ in stages.runs.values()} - {report.pipeline_run_id}
    assert len(own_runs) == len(CORE_STAGES), "every stage runs as a flow run of its own"
    for stage in CORE_STAGES:
        for needed in (n for n in NEEDS[stage] if n in CORE_STAGES):
            assert stages.at("end", needed) < stages.at("start", stage), (
                f"{stage} started before {needed} finished"
            )
    assert pipeline_run(pipeline_mlflow, tenant).data.tags["status"] == "ok"


async def test_a_failing_stage_fails_the_flow_and_stops_its_dependants(
    tenant: str, pipeline_mlflow: MlflowClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    # embed-links breaks while embed-pages and resolve-keywords, started beside it, still run.
    running = ("embed-pages", "resolve-keywords")
    stages = fake_stages(monkeypatch, cache, tenant, failing="embed-links", until_failed=running)

    state = await flows.tenant_pipeline_flow(tenant, *SOURCE, cache_dir=cache, return_state=True)

    assert state.is_failed()
    with pytest.raises(PipelineFailedError) as failed:
        await state.result()
    assert failed.value.failed == ("embed-links",)
    for stage in running:
        assert stages.at("start", stage) < stages.at("end", "embed-links") < stages.at("end", stage)
    # Everything that reads embed-links' vectors, directly or through score-links, is skipped;
    # the stages whose inputs finished are not started after the failure.
    skipped = ("score-links", "quality-eval", "anchor-selection", "link-audit", "rank-pairs")
    expected = {
        **dict.fromkeys(CORE_STAGES, StageStatus.NOT_RUN),
        **dict.fromkeys(("prepare-corpus", "load-graph", *running), StageStatus.OK),
        "embed-links": StageStatus.FAILED,
        **dict.fromkeys((*skipped, "recommendations"), StageStatus.SKIPPED),
    }
    assert sorted(stages.runs) == sorted(("prepare-corpus", "load-graph", "embed-links", *running))
    run = pipeline_run(pipeline_mlflow, tenant)
    assert run.data.tags["status"] == "failed"
    logged = load_dict(f"runs:/{run.info.run_id}/stages.json")
    assert {stage: record["status"] for stage, record in logged.items()} == {
        stage: status.value for stage, status in expected.items()
    }
    assert logged["embed-links"]["error"] == "RuntimeError"


async def test_stage_subflows_share_the_pipelines_root_run_id(
    tenant: str, pipeline_mlflow: MlflowClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    stages = fake_stages(monkeypatch, cache, tenant)

    state = await flows.tenant_pipeline_flow(tenant, *SOURCE, cache_dir=cache, return_state=True)

    pipeline = str(state.state_details.flow_run_id)
    report = await state.result()
    assert sorted(stages.runs) == sorted(CORE_STAGES)
    assert {stage: root for stage, (_, root) in stages.runs.items()} == dict.fromkeys(
        CORE_STAGES, pipeline
    )
    assert report.pipeline_run_id == pipeline
    assert pipeline_run(pipeline_mlflow, tenant).data.tags["pipeline_run_id"] == pipeline
