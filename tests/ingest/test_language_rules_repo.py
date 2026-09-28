"""Per-tenant language rules in tenant_config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymongo import AsyncMongoClient

from linking_engine.errors import DatabaseReadError
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.models import AnchorRules, LanguageRules
from linking_engine.urls import UrlRules

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.mark.integration
async def test_language_rules_round_trip_per_tenant(mongo: MongoRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    rules = LanguageRules(default_language="de", prefixes=(("/fr/", "fr"), ("/en/", "en")))

    await mongo.set_language_rules(tenant, rules)

    stored = await mongo.get_language_rules(tenant)
    assert stored.default_language == "de"
    assert sorted(stored.prefixes) == sorted(rules.prefixes)
    assert await mongo.get_language_rules(other) == LanguageRules()


@pytest.mark.integration
async def test_a_tenant_without_rules_is_english_without_prefixes(
    mongo: MongoRepo, tenant: str
) -> None:
    await mongo.set_url_rules(tenant, UrlRules(keep_params=frozenset({"pg"})))

    assert await mongo.get_language_rules(tenant) == LanguageRules()


@pytest.mark.integration
async def test_language_rules_leave_the_other_tenant_settings_alone(
    mongo: MongoRepo, tenant: str
) -> None:
    url_rules = UrlRules(keep_params=frozenset({"pg"}))
    anchor_rules = AnchorRules(generic_add=frozenset({"find out more"}))
    await mongo.set_url_rules(tenant, url_rules)
    await mongo.set_anchor_rules(tenant, anchor_rules)

    await mongo.set_language_rules(tenant, LanguageRules(default_language="nl"))
    await mongo.set_language_rules(tenant, LanguageRules(prefixes=(("/de/", "de"),)))

    assert await mongo.get_url_rules(tenant) == url_rules
    assert await mongo.get_anchor_rules(tenant) == anchor_rules
    assert await mongo.get_language_rules(tenant) == LanguageRules(prefixes=(("/de/", "de"),))


@pytest.mark.parametrize(
    "stored",
    [
        {"defaultLanguage": "x"},
        {"languagePrefixes": [{"path": "de/", "language": "de"}]},
        {"languagePrefixes": "/de/"},
    ],
    ids=["short-language", "relative-prefix", "not-a-list"],
)
@pytest.mark.integration
async def test_stored_rules_that_do_not_fit_fail_the_read(
    mongo: MongoRepo, tenant: str, stored: dict[str, object]
) -> None:
    await mongo._db["tenant_config"].insert_one({"tenantId": tenant, **stored})

    with pytest.raises(DatabaseReadError, match="language rules"):
        await mongo.get_language_rules(tenant)


@pytest.fixture
async def offline_mongo() -> AsyncIterator[MongoRepo]:
    """A repo whose server does not exist: any query would fail."""
    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
        "mongodb://127.0.0.1:1", serverSelectionTimeoutMS=100
    )
    repo = MongoRepo(client, "linking_engine_test")
    yield repo
    await repo.close()


async def test_a_blank_tenant_is_rejected_before_any_query(offline_mongo: MongoRepo) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_mongo.get_language_rules(" ")
    with pytest.raises(ValueError, match="tenant_id"):
        await offline_mongo.set_language_rules(" ", LanguageRules())
