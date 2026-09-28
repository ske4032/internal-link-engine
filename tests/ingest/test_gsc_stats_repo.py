"""GSC query rows with their metrics, and the per-page 28-day rollup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymongo import AsyncMongoClient

from linking_engine.errors import DatabaseReadError
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.models import GscMetrics, GscQueryStats

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def url(path: str) -> str:
    return f"example.com{path}"


def query_row(tenant: str, path: str, query: str, **fields: object) -> dict[str, object]:
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


def metrics_row(tenant: str, path: str, **fields: object) -> dict[str, object]:
    # The rollup keeps its snake_case field names.
    return {
        "tenantId": tenant,
        "url": url(path),
        "impressions_28d": 900,
        "clicks_28d": 30,
        "avg_position": 6.25,
        "query_count": 3,
        **fields,
    }


@pytest.mark.integration
async def test_query_stats_are_the_tenants_rows_ordered_by_url_then_query(
    mongo: MongoRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await mongo._db["gsc_queries"].insert_many(
        [
            query_row(tenant, "/b", "tents", impressions=80, clicks=0, position=12.0),
            query_row(tenant, "/a", "trail shoes", position=1),
            query_row(other, "/a", "other tenant query"),
            query_row(tenant, "/a", "Rain Jacket "),
        ]
    )

    stats = await mongo.gsc_query_stats(tenant, batch_size=2)

    assert stats == [
        GscQueryStats(url=url("/a"), query="Rain Jacket ", impressions=120, clicks=4, position=7.5),
        GscQueryStats(url=url("/a"), query="trail shoes", impressions=120, clicks=4, position=1.0),
        GscQueryStats(url=url("/b"), query="tents", impressions=80, clicks=0, position=12.0),
    ]


@pytest.mark.integration
async def test_a_query_row_without_clicks_reads_as_zero(mongo: MongoRepo, tenant: str) -> None:
    row = query_row(tenant, "/a", "tents")
    del row["clicks"]
    await mongo._db["gsc_queries"].insert_one(row)

    [stats] = await mongo.gsc_query_stats(tenant)

    assert stats.clicks == 0


@pytest.mark.parametrize(
    "broken",
    [{"position": 0.4}, {"impressions": -1}, {"impressions": None}],
    ids=["position-below-one", "negative-impressions", "no-impressions"],
)
@pytest.mark.integration
async def test_a_query_row_that_does_not_fit_fails_the_read(
    mongo: MongoRepo, tenant: str, broken: dict[str, object]
) -> None:
    await mongo._db["gsc_queries"].insert_one(query_row(tenant, "/a", "tents", **broken))

    with pytest.raises(DatabaseReadError, match="GscQueryStats"):
        await mongo.gsc_query_stats(tenant)


@pytest.mark.integration
async def test_metrics_are_the_tenants_pages_ordered_by_url(mongo: MongoRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await mongo._db["gsc_metrics"].insert_many(
        [
            metrics_row(tenant, "/b", avg_position=None, impressions_28d=0, clicks_28d=0),
            metrics_row(other, "/a", impressions_28d=5),
            metrics_row(tenant, "/a"),
        ]
    )

    metrics = await mongo.gsc_metrics(tenant, batch_size=1)

    assert metrics == [
        GscMetrics(
            url=url("/a"), impressions_28d=900, clicks_28d=30, avg_position=6.25, query_count=3
        ),
        GscMetrics(
            url=url("/b"), impressions_28d=0, clicks_28d=0, avg_position=None, query_count=3
        ),
    ]


@pytest.mark.integration
async def test_a_tenant_without_gsc_data_reads_empty(mongo: MongoRepo, tenant: str) -> None:
    assert await mongo.gsc_query_stats(tenant) == []
    assert await mongo.gsc_metrics(tenant) == []


@pytest.mark.integration
async def test_a_metrics_row_missing_a_total_fails_the_read(mongo: MongoRepo, tenant: str) -> None:
    row = metrics_row(tenant, "/a")
    del row["query_count"]
    await mongo._db["gsc_metrics"].insert_one(row)

    with pytest.raises(DatabaseReadError, match="GscMetrics"):
        await mongo.gsc_metrics(tenant)


@pytest.fixture
async def offline_mongo() -> AsyncIterator[MongoRepo]:
    """A repo whose server does not exist: any read would fail as unavailable."""
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
        "mongodb://127.0.0.1:1", serverSelectionTimeoutMS=200
    )
    repo = MongoRepo(client, "linking_engine_test")
    yield repo
    await repo.close()


@pytest.mark.parametrize("read", ["gsc_query_stats", "gsc_metrics"])
async def test_a_blank_tenant_is_rejected_before_any_read(
    offline_mongo: MongoRepo, read: str
) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await getattr(offline_mongo, read)(" ")


async def test_page_records_reject_a_blank_tenant_before_any_read(
    offline_mongo: MongoRepo,
) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        _ = [batch async for batch in offline_mongo.iter_page_records(" ")]
