from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymongo import AsyncMongoClient

from linking_engine.errors import DatabaseReadError
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.models import StrategicKeyword

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def url(path: str) -> str:
    return f"example.com{path}"


def row(tenant: str, path: str, keyword: object, **fields: object) -> dict[str, object]:
    return {
        "tenantId": tenant,
        "url": url(path),
        "keyword": keyword,
        "language": "en",
        "priority": 3,
        "isPrimary": False,
        **fields,
    }


async def insert(mongo: MongoRepo, documents: list[dict[str, object]]) -> None:
    await mongo._db["strategic_keywords"].insert_many(documents)


@pytest.mark.integration
async def test_keywords_are_the_tenants_rows_in_order_with_their_fields(
    mongo: MongoRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await insert(
        mongo,
        [
            row(tenant, "/b", "tents", priority=5, isPrimary=True),
            row(other, "/a", "other tenant keyword"),
            # Fields the client adds beyond the model are not read.
            row(tenant, "/a", "Trail Shoes", priority=None, _group="core"),
            row(tenant, "/a", "rain jacket", language="de"),
        ],
    )

    assert await mongo.strategic_keywords(tenant) == [
        StrategicKeyword(url=url("/a"), keyword="Trail Shoes", language="en", is_primary=False),
        StrategicKeyword(url=url("/a"), keyword="rain jacket", language="de", priority=3),
        StrategicKeyword(
            url=url("/b"), keyword="tents", language="en", priority=5, is_primary=True
        ),
    ]


@pytest.mark.integration
async def test_every_row_is_read_across_batches(mongo: MongoRepo, tenant: str) -> None:
    await insert(mongo, [row(tenant, f"/p{i}", f"keyword {i}") for i in range(5)])

    rows = await mongo.strategic_keywords(tenant, batch_size=2)

    assert [(r.url, r.keyword) for r in rows] == [(url(f"/p{i}"), f"keyword {i}") for i in range(5)]


@pytest.mark.integration
async def test_a_tenant_without_keywords_has_none(mongo: MongoRepo, tenant: str) -> None:
    await insert(mongo, [row(f"{tenant}-other", "/a", "tents")])
    assert await mongo.strategic_keywords(tenant) == []


@pytest.mark.parametrize(
    "document",
    [
        pytest.param({"keyword": 7}, id="numeric-keyword"),
        pytest.param({"keyword": ""}, id="blank-keyword"),
        pytest.param({"url": None}, id="null-url"),
        pytest.param({"language": None}, id="null-language"),
        pytest.param({"priority": 9}, id="priority-out-of-range"),
    ],
)
@pytest.mark.integration
async def test_a_malformed_row_fails_the_read(
    mongo: MongoRepo, tenant: str, document: dict[str, object]
) -> None:
    await insert(mongo, [row(tenant, "/a", "tents"), {**row(tenant, "/b", "stoves"), **document}])

    with pytest.raises(DatabaseReadError, match="StrategicKeyword"):
        await mongo.strategic_keywords(tenant)


@pytest.fixture
async def offline_mongo() -> AsyncIterator[MongoRepo]:
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
        await offline_mongo.strategic_keywords(tenant_id)
