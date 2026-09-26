from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from prefect.states import Failed
from test_embed import DIM, seed_plain
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
    monkeypatch.setattr(
        flows, "embed_tenant_task", flows.embed_tenant_task.with_options(retry_delay_seconds=0)
    )


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

    assert (report.selected, report.embedded, report.flushes) == (3, 3, 2)


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
    assert (report.selected, report.embedded) == (2, 2)
    assert sorted(page_index(text) for text in fake.calls[-1].texts) == [2, 3]


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
