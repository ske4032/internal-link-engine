from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mlflow import MlflowClient
from prefect.states import Failed
from pymongo import AsyncMongoClient
from test_embed import DIM, record, seed, seed_plain, url
from test_embed_links import text_vector
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
from linking_engine.models import Link, LinkRecord, Page
from linking_engine.pipeline import flows

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import CandidateSet


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


async def seed_crawl(mongo_uri: str, database: str) -> None:
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(mongo_uri)
    await client[database]["crawl_pages"].insert_many(
        [
            {
                "url": f"https://example.com/{name}",
                "title": name,
                "content": f"{NAV_LINE}\n\n{name.title()} page. Read [the next one]"
                f"(https://example.com/{nxt}) as well.",
                "statusCode": 200,
                "usable": True,
            }
            for name, nxt in zip(
                ("pricing", "a", "b", "c", "d", "e"), ("a", "b", "c", "d", "e", "a"), strict=True
            )
        ]
    )
    await client.close()


@pytest.mark.integration
async def test_prepare_and_load_flows_ingest_a_crawl_into_mongo_and_neo4j(
    graph: GraphRepo, mongo_uri: str, tenant: str, flow_env: None
) -> None:
    source_db = f"crawl_{tenant.replace('-', '_')}"
    await seed_crawl(mongo_uri, source_db)

    prepared = await flows.prepare_corpus_flow(tenant, source_db, "crawl_pages")
    loaded, counts = await flows.load_graph_flow(tenant)

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
