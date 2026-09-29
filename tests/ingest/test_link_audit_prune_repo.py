"""link_audit pruning: once a run is complete, the tenant's other runs go, documents and markers,
and the latest run still reads back; another tenant's runs are never touched."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from test_link_audit_history_repo import AT, documents, ordered, rows, store

from linking_engine.errors import DatabaseReadError, DatabaseWriteError

if TYPE_CHECKING:
    from linking_engine.ingest.mongo_repo import MongoRepo


async def markers(mongo: MongoRepo, tenant: str) -> list[str]:
    cursor = mongo._db["link_audit_runs"].find({"tenantId": tenant}).sort("runId", 1)
    return [str(marker["runId"]) for marker in await cursor.to_list()]


@pytest.mark.integration
async def test_a_complete_run_prunes_every_other_run_of_its_tenant(
    mongo: MongoRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await store(mongo, other, "run-1", rows("run-1"))
    await store(mongo, tenant, "run-1", rows("run-1"))
    latest = rows("run-2", AT + timedelta(hours=1))
    await store(mongo, tenant, "run-2", latest)
    # Written but never completed: pruned all the same.
    await mongo.insert_link_audit(tenant, "run-3", rows("run-3", AT + timedelta(hours=2)))

    assert await mongo.prune_link_audit(tenant, "run-2") == (6, 1)

    assert {d["runId"] for d in await documents(mongo, tenant)} == {"run-2"}
    assert await markers(mongo, tenant) == ["run-2"]
    assert await mongo.latest_link_audit(tenant) == ordered(latest)
    found = await mongo.latest_link_audit_run(tenant)
    assert found is not None
    assert found[0] == "run-2"
    assert found[1].tzinfo is not None
    assert await mongo.prune_link_audit(tenant, "run-2") == (0, 0)
    assert await markers(mongo, other) == ["run-1"]
    assert len(await documents(mongo, other)) == 3


@pytest.mark.integration
async def test_an_incomplete_run_prunes_nothing(mongo: MongoRepo, tenant: str) -> None:
    await store(mongo, tenant, "run-1", rows("run-1"))
    await mongo.insert_link_audit(tenant, "run-2", rows("run-2"))

    with pytest.raises(DatabaseWriteError, match="not complete"):
        await mongo.prune_link_audit(tenant, "run-2")
    with pytest.raises(ValueError, match="keep"):
        await mongo.prune_link_audit(tenant, " ")
    assert len(await documents(mongo, tenant)) == 6
    assert await markers(mongo, tenant) == ["run-1"]


@pytest.mark.integration
async def test_the_latest_run_marker_reads_back_or_fails_without_its_fields(
    mongo: MongoRepo, tenant: str
) -> None:
    assert await mongo.latest_link_audit_run(tenant) is None
    await mongo._db["link_audit_runs"].insert_one(
        {"tenantId": tenant, "runId": "run-1", "auditedAt": AT}
    )
    with pytest.raises(DatabaseReadError, match="completion time"):
        await mongo.latest_link_audit_run(tenant)
