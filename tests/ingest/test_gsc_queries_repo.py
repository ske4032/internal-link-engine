from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymongo import AsyncMongoClient

from linking_engine.errors import DatabaseReadError
from linking_engine.ingest.mongo_repo import MongoRepo

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def url(path: str) -> str:
    return f"example.com{path}"


def row(tenant: str, path: str, query: object, **fields: object) -> dict[str, object]:
    return {
        "tenantId": tenant,
        "url": url(path),
        "query": query,
        "impressions": 120,
        "clicks": 4,
        "position": 7.5,
        "dateBucket": "2026-09",
        **fields,
    }


async def insert(mongo: MongoRepo, documents: list[dict[str, object]]) -> None:
    await mongo._db["gsc_queries"].insert_many(documents)


@pytest.mark.integration
async def test_queries_are_the_tenants_rows_ordered_by_url_then_query(
    mongo: MongoRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await insert(
        mongo,
        [
            row(tenant, "/b", "tents"),
            row(tenant, "/a", "trail shoes"),
            row(other, "/a", "other tenant query"),
            row(tenant, "/a", "Rain Jacket "),
            row(tenant, "/b", "camp stoves"),
            row(other, "/c", "tents"),
        ],
    )

    # Stored text is returned verbatim: normalising is the signal stage's job.
    assert await mongo.gsc_queries(tenant) == [
        (url("/a"), "Rain Jacket "),
        (url("/a"), "trail shoes"),
        (url("/b"), "camp stoves"),
        (url("/b"), "tents"),
    ]


@pytest.mark.integration
async def test_every_row_is_read_across_batches(mongo: MongoRepo, tenant: str) -> None:
    documents = [row(tenant, f"/p{i}", f"query {i}") for i in range(5)]
    await insert(mongo, documents)

    rows = await mongo.gsc_queries(tenant, batch_size=2)

    assert rows == sorted((url(f"/p{i}"), f"query {i}") for i in range(5))


@pytest.mark.integration
async def test_a_tenant_without_gsc_data_has_no_queries(mongo: MongoRepo, tenant: str) -> None:
    await insert(mongo, [row(f"{tenant}-other", "/a", "tents")])
    assert await mongo.gsc_queries(tenant) == []


@pytest.mark.parametrize(
    "document",
    [
        pytest.param({"url": 7}, id="numeric-url"),
        pytest.param({"url": None}, id="null-url"),
        pytest.param({"query": 7}, id="numeric-query"),
        pytest.param({"query": None}, id="null-query"),
    ],
)
@pytest.mark.integration
async def test_a_row_without_string_url_and_query_fails_the_read(
    mongo: MongoRepo, tenant: str, document: dict[str, object]
) -> None:
    await insert(mongo, [row(tenant, "/a", "tents"), {**row(tenant, "/b", "stoves"), **document}])

    with pytest.raises(DatabaseReadError, match="GscQuery"):
        await mongo.gsc_queries(tenant)


@pytest.mark.integration
async def test_a_row_missing_its_query_fails_the_read(mongo: MongoRepo, tenant: str) -> None:
    broken = row(tenant, "/b", "x")
    del broken["query"]
    await insert(mongo, [broken])

    with pytest.raises(DatabaseReadError, match="GscQuery"):
        await mongo.gsc_queries(tenant)


@pytest.mark.integration
async def test_another_tenants_malformed_row_does_not_fail_the_read(
    mongo: MongoRepo, tenant: str
) -> None:
    await insert(mongo, [row(tenant, "/a", "tents"), row(f"{tenant}-other", "/a", None)])
    assert await mongo.gsc_queries(tenant) == [(url("/a"), "tents")]


@pytest.fixture
async def offline_mongo() -> AsyncIterator[MongoRepo]:
    """A repo whose server does not exist: any read would fail as unavailable."""
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
        "mongodb://127.0.0.1:1", serverSelectionTimeoutMS=200
    )
    repo = MongoRepo(client, "linking_engine_test")
    yield repo
    await repo.close()


@pytest.mark.parametrize("tenant_id", ["", " "])
async def test_a_blank_tenant_is_rejected_before_any_read(
    offline_mongo: MongoRepo, tenant_id: str
) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_mongo.gsc_queries(tenant_id)
