"""Output API keys: issued once and stored only as a hash, resolved to their tenant, revoked per
tenant, and never logged."""

from __future__ import annotations

import hashlib
import itertools
import secrets
import uuid
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING

import pytest
from pymongo import AsyncMongoClient
from structlog.testing import capture_logs

from linking_engine.errors import DatabaseReadError, DatabaseUnavailableError, DatabaseWriteError
from linking_engine.output.collections import API_KEYS
from linking_engine.output.keys import PREFIX, KeyStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pymongo.asynchronous.database import AsyncDatabase

Document = dict[str, object]


@pytest.fixture
async def db(mongo_uri: str) -> AsyncIterator[AsyncDatabase[Document]]:
    client: AsyncMongoClient[Document] = AsyncMongoClient(mongo_uri, tz_aware=True)
    yield client["linking_engine_test"]
    await client.close()


@pytest.fixture
async def store(db: AsyncDatabase[Document]) -> KeyStore:
    keys = KeyStore(db)
    await keys.ensure_indexes()
    return keys


def _values(document: Document) -> set[str]:
    return {str(value) for value in document.values()}


@pytest.mark.integration
async def test_an_issued_key_is_stored_only_as_its_hash(
    store: KeyStore, db: AsyncDatabase[Document], tenant: str
) -> None:
    key, info = await store.issue(tenant, "  Acme dashboard ")

    assert key.startswith(PREFIX)
    assert len(key) == len(PREFIX) + 43
    assert info.tenant_id == tenant
    assert info.label == "Acme dashboard"
    assert info.revoked_at is None
    documents = await db[API_KEYS].find({"tenantId": tenant}).to_list()
    assert len(documents) == 1
    stored = documents[0]
    assert stored["keyHash"] == hashlib.sha256(key.encode()).hexdigest()
    assert stored["keyId"] == info.key_id
    assert key not in _values(stored)
    assert not any(key[len(PREFIX) :] in value for value in _values(stored))


@pytest.mark.integration
async def test_a_live_key_resolves_to_its_tenant_and_nothing_else_does(
    store: KeyStore, tenant: str
) -> None:
    key, _ = await store.issue(tenant)

    assert await store.tenant_for(key) == tenant
    assert await store.tenant_for(PREFIX + "f" * 43) is None
    assert await store.tenant_for(key[len(PREFIX) :]) is None
    assert await store.tenant_for(key + "x" * 200) is None
    assert await store.tenant_for("") is None


@pytest.mark.integration
async def test_a_key_is_revoked_only_by_its_own_tenant(store: KeyStore, tenant: str) -> None:
    other = f"test-{uuid.uuid4().hex[:12]}"
    key, info = await store.issue(tenant, "one")

    assert await store.revoke(other, info.key_id) is False
    assert await store.tenant_for(key) == tenant
    assert await store.revoke(tenant, "00000000") is False

    assert await store.revoke(tenant, info.key_id) is True
    assert await store.tenant_for(key) is None
    assert await store.revoke(tenant, info.key_id) is False


@pytest.mark.integration
async def test_keys_lists_the_tenants_keys_only_oldest_first(store: KeyStore, tenant: str) -> None:
    other = f"test-{uuid.uuid4().hex[:12]}"
    _, first = await store.issue(tenant, "first")
    _, second = await store.issue(tenant)
    await store.issue(other, "elsewhere")
    await store.revoke(tenant, first.key_id)

    listed = await store.keys(tenant)

    assert [info.key_id for info in listed] == [first.key_id, second.key_id]
    assert listed[0].revoked_at is not None
    assert listed[0].model_copy(update={"revoked_at": None}) == first
    assert listed[1] == second
    assert all(info.tenant_id == tenant for info in listed)
    assert await store.keys(f"test-{uuid.uuid4().hex[:12]}") == ()


@pytest.mark.integration
async def test_keys_issued_in_the_same_millisecond_keep_their_issue_order(
    store: KeyStore, tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FrozenClock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> FrozenClock:
            return cls(2026, 3, 2, 9, 0, tzinfo=tz)

    monkeypatch.setattr("linking_engine.output.keys.datetime", FrozenClock)
    issued = [(await store.issue(tenant))[1].key_id for _ in range(6)]

    assert [info.key_id for info in await store.keys(tenant)] == issued


@pytest.mark.integration
async def test_a_colliding_key_id_is_drawn_again(
    store: KeyStore, tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = itertools.chain(["0000aaaa", "0000aaaa", "0000bbbb"], itertools.repeat("0000cccc"))
    monkeypatch.setattr(secrets, "token_hex", lambda _: next(ids))

    _, first = await store.issue(tenant)
    _, second = await store.issue(tenant)

    assert (first.key_id, second.key_id) == ("0000aaaa", "0000bbbb")


@pytest.mark.integration
async def test_issue_gives_up_when_every_key_id_collides(
    store: KeyStore, tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(secrets, "token_hex", lambda _: "0000dddd")
    await store.issue(tenant)

    with pytest.raises(DatabaseWriteError, match="no free key id"):
        await store.issue(tenant)
    assert len(await store.keys(tenant)) == 1


@pytest.mark.integration
async def test_neither_a_key_nor_its_hash_is_logged(store: KeyStore, tenant: str) -> None:
    with capture_logs() as logs:
        key, info = await store.issue(tenant, "logged")
        await store.tenant_for(key)
        await store.revoke(tenant, info.key_id)

    digest = hashlib.sha256(key.encode()).hexdigest()
    assert [entry["event"] for entry in logs] == ["api_keys.issued", "api_keys.revoked"]
    for entry in logs:
        assert entry["key_id"] == info.key_id
        assert not {key, digest} & {str(value) for value in entry.values()}


@pytest.mark.integration
async def test_a_stored_key_that_does_not_fit_is_a_read_error(
    store: KeyStore, db: AsyncDatabase[Document], tenant: str
) -> None:
    await db[API_KEYS].insert_one({"keyHash": "0" * 64, "keyId": "zz", "tenantId": tenant})

    with pytest.raises(DatabaseReadError, match="ApiKeyInfo"):
        await store.keys(tenant)


async def test_a_blank_tenant_is_rejected() -> None:
    client: AsyncMongoClient[Document] = AsyncMongoClient("mongodb://127.0.0.1:1", connect=False)
    store = KeyStore(client["linking_engine_test"])
    try:
        with pytest.raises(ValueError, match="tenant_id"):
            await store.issue(" ")
        with pytest.raises(ValueError, match="tenant_id"):
            await store.revoke("", "00000000")
        with pytest.raises(ValueError, match="tenant_id"):
            await store.keys("")
    finally:
        await client.close()


async def test_an_unreachable_store_is_reported_as_unavailable() -> None:
    client: AsyncMongoClient[Document] = AsyncMongoClient(
        "mongodb://127.0.0.1:1", serverSelectionTimeoutMS=100
    )
    store = KeyStore(client["linking_engine_test"])
    try:
        with pytest.raises(DatabaseUnavailableError):
            await store.tenant_for(PREFIX + "f" * 43)
    finally:
        await client.close()
