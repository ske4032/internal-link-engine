"""A tenant's anchor type profile on its tenant config: round trip, reset and isolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymongo import AsyncMongoClient

from linking_engine.errors import DatabaseReadError
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.models import AnchorTypeProfile, ExtractionSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

PROFILE = AnchorTypeProfile(exact=0.3, partial=0.3, natural=0.3, branded=0.1)


@pytest.mark.integration
async def test_the_profile_round_trips_for_one_tenant_only(mongo: MongoRepo, tenant: str) -> None:
    other = f"{tenant}-other"

    assert await mongo.get_anchor_type_profile(tenant) is None
    await mongo.set_anchor_type_profile(tenant, PROFILE)

    assert await mongo.get_anchor_type_profile(tenant) == PROFILE
    assert await mongo.get_anchor_type_profile(other) is None
    stored = await mongo._db["tenant_config"].find_one({"tenantId": tenant})
    assert stored is not None
    assert stored["anchorTypeProfile"] == {
        "exact": 0.3,
        "partial": 0.3,
        "natural": 0.3,
        "branded": 0.1,
    }


@pytest.mark.integration
async def test_storing_none_restores_the_default(mongo: MongoRepo, tenant: str) -> None:
    await mongo.set_anchor_type_profile(tenant, PROFILE)
    await mongo.set_anchor_type_profile(tenant, None)

    assert await mongo.get_anchor_type_profile(tenant) is None


@pytest.mark.integration
async def test_the_profile_leaves_the_rest_of_the_tenant_config_alone(
    mongo: MongoRepo, tenant: str
) -> None:
    await mongo._db["tenant_config"].insert_one({"tenantId": tenant, "defaultLanguage": "de"})
    await mongo.set_extraction_settings(tenant, ExtractionSettings(stem_set_threshold=0.7))

    await mongo.set_anchor_type_profile(tenant, PROFILE)
    await mongo.set_anchor_type_profile(tenant, None)

    assert (await mongo.get_language_rules(tenant)).default_language == "de"
    assert await mongo.get_extraction_settings(tenant) == ExtractionSettings(stem_set_threshold=0.7)


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param([0.15, 0.2, 0.5, 0.15], id="not-a-document"),
        pytest.param({"exact": 1.5}, id="out-of-range"),
        pytest.param({"exact": 0.2, "exactShare": 0.2}, id="unknown-key"),
    ],
)
@pytest.mark.integration
async def test_a_malformed_stored_profile_fails_the_read(
    mongo: MongoRepo, tenant: str, stored: object
) -> None:
    await mongo._db["tenant_config"].update_one(
        {"tenantId": tenant}, {"$set": {"anchorTypeProfile": stored}}, upsert=True
    )

    with pytest.raises(DatabaseReadError, match="anchor type profile"):
        await mongo.get_anchor_type_profile(tenant)


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
async def test_a_blank_tenant_is_refused_before_any_read(
    offline_mongo: MongoRepo, tenant_id: str
) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_mongo.get_anchor_type_profile(tenant_id)
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_mongo.set_anchor_type_profile(tenant_id, PROFILE)
