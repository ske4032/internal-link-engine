"""The output API's keys and error shapes: 401 for a missing, unknown or revoked key, the same
404 for another tenant's path as for a missing resource, 503 when the store is down, and a key
that is never logged."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from pymongo import AsyncMongoClient
from starlette.requests import Request
from structlog.testing import capture_logs

from linking_engine.api.app import ApiSettings, create_app
from linking_engine.api.deps import key_store, output_reader
from linking_engine.errors import DatabaseUnavailableError
from linking_engine.models import RunInfo, ScorerName, SiteSummary
from linking_engine.output.collections import RUNS, to_document
from linking_engine.output.keys import PREFIX, KeyStore
from linking_engine.output.reader import OutputReader

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI
    from pymongo.asynchronous.database import AsyncDatabase

Document = dict[str, object]

DATABASE: Final = "linking_engine_test"
UNKNOWN_KEY: Final = PREFIX + "f" * 43
INVALID_KEY: Final = {"detail": "invalid or missing API key"}
NOT_FOUND: Final = {"detail": "not found"}
RUN_ROUTES: Final = (
    "/recommendations",
    "/recommendations/0123456789abcdef",
    "/summary",
    "/runs/latest",
    "/pages",
    "/page?url=example.com/guides/a",
    "/orphans",
    "/hubs",
    "/bridges",
    "/duplicates",
    "/unanchored",
    "/target-fixes",
)
ROUTES: Final = (*RUN_ROUTES, "/excluded-pages")


def complete_run(tenant: str) -> RunInfo:
    now = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
    return RunInfo(
        tenant_id=tenant,
        run_id=uuid.uuid4().hex,
        status="complete",
        started_at=now,
        completed_at=now,
        scorer=ScorerName.BASELINE,
        weights_version="w1",
        feature_code="f1",
        package_version="0.1.0",
        limit_per_source=10,
        content_gap_limit=3,
        summary=SiteSummary(
            pages=0,
            dead_end_pages=0,
            duplicate_groups=0,
            duplicate_copies=0,
            hubs=0,
            bridge_pairs=0,
            bridge_links=0,
            sources_with_recommendations=0,
            sources_below_limit=0,
            links_audited=0,
            unverified_links=0,
            target_fixes=0,
        ),
    )


@pytest.fixture
async def db(mongo_uri: str) -> AsyncIterator[AsyncDatabase[Document]]:
    client: AsyncMongoClient[Document] = AsyncMongoClient(mongo_uri, tz_aware=True)
    yield client[DATABASE]
    await client.close()


@pytest.fixture
async def keys(db: AsyncDatabase[Document]) -> KeyStore:
    store = KeyStore(db)
    await store.ensure_indexes()
    return store


@pytest.fixture
async def api(mongo_uri: str) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(ApiSettings(mongo_uri=SecretStr(mongo_uri), mongo_db=DATABASE))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://api.example"
        ) as client,
    ):
        yield client


def offline_app() -> FastAPI:
    return create_app(ApiSettings(mongo_uri=SecretStr("mongodb://127.0.0.1:1"), mongo_db=DATABASE))


async def test_health_needs_no_key() -> None:
    app = offline_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api.example"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
@pytest.mark.parametrize("route", ROUTES)
async def test_a_missing_unknown_or_revoked_key_gets_401(
    api: httpx.AsyncClient, keys: KeyStore, tenant: str, route: str
) -> None:
    key, info = await keys.issue(tenant, "revoked")
    await keys.revoke(tenant, info.key_id)
    prefix = f"/v1/tenants/{tenant}"

    for headers in ({}, {"X-API-Key": UNKNOWN_KEY}, {"X-API-Key": key}, {"X-API-Key": ""}):
        response = await api.get(prefix + route, headers=headers)
        assert response.status_code == 401, headers
        assert response.json() == INVALID_KEY
        assert response.headers["WWW-Authenticate"] == "APIKey"


@pytest.mark.integration
@pytest.mark.parametrize("route", ROUTES)
async def test_a_key_on_another_tenants_path_gets_the_missing_resource_404(
    api: httpx.AsyncClient, keys: KeyStore, db: AsyncDatabase[Document], tenant: str, route: str
) -> None:
    other = f"test-{uuid.uuid4().hex[:12]}"
    for owner in (tenant, other):
        run = complete_run(owner)
        await db[RUNS].insert_one(
            to_document(run, tenant_id=owner, run_id=run.run_id, ordinal=None)
        )
    key, _ = await keys.issue(tenant)
    headers = {"X-API-Key": key}

    missing = await api.get(
        f"/v1/tenants/{tenant}/recommendations/0123456789abcdef", headers=headers
    )
    for path in (f"/v1/tenants/{other}", f"/v1/tenants/test-{uuid.uuid4().hex[:12]}"):
        response = await api.get(path + route, headers=headers)
        assert response.status_code == 404
        assert response.content == missing.content
        assert response.json() == NOT_FOUND


@pytest.mark.integration
@pytest.mark.parametrize("route", RUN_ROUTES)
async def test_a_tenant_without_a_complete_run_gets_404(
    api: httpx.AsyncClient, keys: KeyStore, tenant: str, route: str
) -> None:
    key, _ = await keys.issue(tenant)

    response = await api.get(f"/v1/tenants/{tenant}{route}", headers={"X-API-Key": key})

    assert response.status_code == 404
    assert response.json() == NOT_FOUND


@pytest.mark.integration
async def test_excluded_pages_are_served_without_a_complete_run(
    api: httpx.AsyncClient, keys: KeyStore, tenant: str
) -> None:
    key, _ = await keys.issue(tenant)

    response = await api.get(f"/v1/tenants/{tenant}/excluded-pages", headers={"X-API-Key": key})

    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None, "total": 0}


async def test_an_unavailable_store_gets_503_and_the_key_is_not_logged() -> None:
    unreachable: AsyncMongoClient[Document] = AsyncMongoClient(
        "mongodb://127.0.0.1:1", serverSelectionTimeoutMS=100
    )
    app = offline_app()
    app.dependency_overrides[key_store] = lambda: KeyStore(unreachable[DATABASE])
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://api.example"
        ) as client:
            with capture_logs() as logs:
                response = await client.get(
                    "/v1/tenants/test-acme/summary", headers={"X-API-Key": UNKNOWN_KEY}
                )
    finally:
        await unreachable.close()

    assert response.status_code == 503
    assert response.json() == {"detail": "store unavailable"}
    assert logs == [
        {
            "event": "api.store_unavailable",
            "log_level": "warning",
            "path": "/v1/tenants/test-acme/summary",
            "error": "DatabaseUnavailableError",
        }
    ]


async def test_an_unexpected_error_keeps_the_detail_body() -> None:
    class Broken:
        async def tenant_for(self, key: str) -> str | None:
            raise RuntimeError(key)

    app = offline_app()
    app.dependency_overrides[key_store] = Broken
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://api.example",
    ) as client:
        with capture_logs() as logs:
            response = await client.get(
                "/v1/tenants/test-acme/hubs", headers={"X-API-Key": UNKNOWN_KEY}
            )

    assert response.status_code == 500
    assert response.json() == {"detail": "internal error"}
    assert [entry["event"] for entry in logs] == ["api.internal_error"]
    assert UNKNOWN_KEY not in {str(value) for value in logs[0].values()}


def test_the_stores_come_only_from_the_lifespan() -> None:
    request = Request({"type": "http", "app": offline_app()})

    with pytest.raises(RuntimeError, match="lifespan"):
        output_reader(request)
    with pytest.raises(RuntimeError, match="lifespan"):
        key_store(request)


async def test_the_lifespan_opens_the_stores_and_rejects_bad_settings() -> None:
    app = offline_app()
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.reader, OutputReader)
        assert isinstance(app.state.keys, KeyStore)

    broken = create_app(ApiSettings(mongo_uri=SecretStr("mongodb://"), mongo_db=DATABASE))
    with pytest.raises(DatabaseUnavailableError, match="invalid connection settings"):
        async with broken.router.lifespan_context(broken):
            pass


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:1")
    monkeypatch.setenv("MONGO_DB", DATABASE)

    app = create_app()

    assert app.title == "Internal Linking Engine API"
    assert app.version == "1"
    monkeypatch.delenv("MONGO_DB")
    with pytest.raises(ValidationError, match="mongo_db"):
        create_app()


def test_every_tenant_route_is_documented_with_its_key_and_errors() -> None:
    spec = offline_app().openapi()
    operations = {path: item["get"] for path, item in spec["paths"].items()}

    assert operations.pop("/health")["summary"]
    assert {path.removeprefix("/v1/tenants/{tenant}") for path in operations} == {
        route.split("?")[0].replace("0123456789abcdef", "{recommendation_id}") for route in ROUTES
    }
    for operation in operations.values():
        assert operation["summary"]
        assert operation["security"] == [{"APIKeyHeader": []}]
        assert {"200", "401", "404", "422", "503"} <= set(operation["responses"])
