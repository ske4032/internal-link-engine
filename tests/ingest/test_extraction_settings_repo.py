"""A tenant's extraction settings on its tenant config: round trip, reset and isolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymongo import AsyncMongoClient

from linking_engine.errors import DatabaseReadError
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.models import ExtractionSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.mark.integration
async def test_settings_round_trip_for_one_tenant_only(mongo: MongoRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    stored = ExtractionSettings(stem_set_threshold=0.4)

    assert await mongo.get_extraction_settings(tenant) is None
    await mongo.set_extraction_settings(tenant, stored)

    assert await mongo.get_extraction_settings(tenant) == stored
    assert await mongo.get_extraction_settings(other) is None


@pytest.mark.integration
async def test_storing_none_restores_the_defaults(mongo: MongoRepo, tenant: str) -> None:
    await mongo.set_extraction_settings(tenant, ExtractionSettings(stem_set_threshold=0.7))
    await mongo.set_extraction_settings(tenant, None)

    assert await mongo.get_extraction_settings(tenant) is None


@pytest.mark.integration
async def test_settings_do_not_disturb_the_rest_of_the_tenant_config(
    mongo: MongoRepo, tenant: str
) -> None:
    await mongo._db["tenant_config"].insert_one({"tenantId": tenant, "defaultLanguage": "de"})

    await mongo.set_extraction_settings(tenant, ExtractionSettings(stem_set_threshold=0.6))

    assert (await mongo.get_language_rules(tenant)).default_language == "de"
    assert await mongo.get_extraction_settings(tenant) == ExtractionSettings(stem_set_threshold=0.6)


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param("0.5", id="not-a-document"),
        pytest.param({"stemSetThreshold": 0.0}, id="out-of-range"),
        pytest.param({"stem_set_threshold": 0.4}, id="snake-case-key"),
    ],
)
@pytest.mark.integration
async def test_malformed_stored_settings_fail_the_read(
    mongo: MongoRepo, tenant: str, stored: object
) -> None:
    await mongo._db["tenant_config"].update_one(
        {"tenantId": tenant}, {"$set": {"extractionSettings": stored}}, upsert=True
    )

    with pytest.raises(DatabaseReadError, match="extraction settings"):
        await mongo.get_extraction_settings(tenant)


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
        await offline_mongo.get_extraction_settings(tenant_id)
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_mongo.set_extraction_settings(tenant_id, ExtractionSettings())
