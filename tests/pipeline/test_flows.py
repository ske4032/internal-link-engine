from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from prefect.states import Failed
from test_embed import DIM, body, record, seed, seed_plain, url
from test_embed_links import text_vector
from voyage_fakes import FakeVoyage, client, page_index
from voyageai.error import InvalidRequestError, ServiceUnavailableError

from linking_engine.errors import (
    DatabaseAuthError,
    DatabaseUnavailableError,
    EmbeddingAuthError,
    EmbeddingModelMismatchError,
    EmbeddingRequestError,
    EmbeddingUnavailableError,
)
from linking_engine.models import LinkRecord, TenantEmbedReport
from linking_engine.pipeline import flows

if TYPE_CHECKING:
    from collections.abc import Iterator

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo


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
    for name in ("embed_tenant_task", "embed_links_task"):
        task = getattr(flows, name)
        monkeypatch.setattr(flows, name, task.with_options(retry_delay_seconds=0))


def use(monkeypatch: pytest.MonkeyPatch, fake: FakeVoyage) -> None:
    monkeypatch.setattr(flows, "voyage_client", lambda _tenant: client(fake))


@pytest.mark.integration
async def test_flow_embeds_the_tenant(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_plain(mongo, graph, tenant, 3)
    fake = FakeVoyage(dimension=DIM)
    use(monkeypatch, fake)

    report = await flows.embed_tenant_flow(tenant, 2)

    assert isinstance(report, TenantEmbedReport)
    assert (report.pages.selected, report.pages.embedded, report.pages.flushes) == (3, 3, 2)
    assert (report.links.tenant_id, report.links.edges, fake.call_count) == (tenant, 0, 2)


@pytest.mark.integration
async def test_rejected_request_fails_the_run_without_a_task_retry(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_plain(mongo, graph, tenant, 2)
    fake = FakeVoyage(dimension=DIM, failures=[InvalidRequestError("bad input", http_status=400)])
    use(monkeypatch, fake)

    state = await flows.embed_tenant_flow(tenant, 2, return_state=True)

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

    report = await flows.embed_tenant_flow(tenant, 2)

    assert fake.call_count == 5
    assert (report.pages.selected, report.pages.embedded) == (2, 2)
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
BODIES = {body(i) for i in range(3)}


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
async def test_flow_embeds_links_after_pages_and_returns_both_reports(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_linked(mongo, graph, tenant)
    fake = any_text()
    use(monkeypatch, fake)

    report = await flows.embed_tenant_flow(tenant, 2)

    assert (report.pages.selected, report.pages.embedded) == (3, 3)
    links = report.links
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
    sent = [text for call in fake.calls for text in call.texts]
    kinds = ["page" if text in BODIES else "link" for text in sent]
    assert kinds == ["page"] * 3 + ["link"] * 3, "every page must be embedded before any link"
    assert set(sent[3:]) == {"trail shoes", SHARED, OTHER}


@pytest.mark.integration
async def test_link_outage_retries_only_the_link_task_and_embeds_only_the_rest(
    mongo: MongoRepo, graph: GraphRepo, tenant: str, flow_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed_linked(mongo, graph, tenant)
    # Flush size 1: pages are calls 1-3, the anchor call 4, sentences 5 and 6. Every client
    # attempt of the second sentence flush (calls 6-8) hits an outage; the task retry is call 9.
    outage = {n: ServiceUnavailableError("down", http_status=503) for n in (6, 7, 8)}
    fake = any_text(fail_on=outage)
    use(monkeypatch, fake)

    report = await flows.embed_tenant_flow(tenant, 1)

    assert fake.call_count == 9
    sent = [text for call in fake.calls for text in call.texts]
    assert sum(text in BODIES for text in sent) == 3, "the page task must not re-run"
    assert report.pages.embedded == 3
    assert set(fake.calls[-1].texts) == {SHARED, OTHER} - set(fake.calls[4].texts)
    links = report.links
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
