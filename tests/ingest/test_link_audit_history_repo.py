"""link_audit history: one document per edge per run, retries rewrite only their own run, and the
latest completed run of a tenant is what reads back."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from linking_engine.errors import DatabaseReadError, DatabaseWriteError
from linking_engine.models import ActionType, IssueFlag, LinkAuditResult

if TYPE_CHECKING:
    from linking_engine.ingest.mongo_repo import MongoRepo

AT = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


def url(path: str) -> str:
    return f"example.com{path}"


def rows(
    run: str, at: datetime = AT, verdict: ActionType | None = ActionType.REANCHOR
) -> list[LinkAuditResult]:
    return [
        LinkAuditResult(
            source_url=url("/b"), position=1, target_url=url("/t"), run_id=run,
            anchor_quality_score=18.5, keyword_alignment=0.0, context_relevance=0.8,
            equity_efficiency=0.2, issue_flags=frozenset({IssueFlag.GENERIC}), verdict=verdict,
            reasons=("the anchor text is generic", "a better phrase: \"map case\""),
            proposed_anchor="map case" if verdict is ActionType.REANCHOR else None,
            audited_at=at,
        ),
        LinkAuditResult(
            source_url=url("/a"), position=0, target_url=url("/ghost"), run_id=run,
            issue_flags=frozenset(), verdict=None, reasons=("not crawled",), unverified=True,
            audited_at=at,
        ),
        LinkAuditResult(
            source_url=url("/a"), position=3, target_url=url("/copy"), run_id=run,
            anchor_quality_score=90.0, keyword_alignment=1.0,
            issue_flags=frozenset({IssueFlag.NOFOLLOW, IssueFlag.BROKEN}), verdict=ActionType.FIX,
            reasons=("404", "nofollow"), fix_target=url("/canon"), audited_at=at,
        ),
    ]  # fmt: skip


def ordered(results: list[LinkAuditResult]) -> tuple[LinkAuditResult, ...]:
    return tuple(sorted(results, key=lambda r: (r.source_url, r.position)))


async def store(mongo: MongoRepo, tenant: str, run: str, results: list[LinkAuditResult]) -> None:
    assert await mongo.insert_link_audit(tenant, run, results) == len(results)
    await mongo.complete_link_audit(
        tenant, run, audited_at=results[0].audited_at, documents=len(results), edges=len(results)
    )


async def documents(mongo: MongoRepo, tenant: str) -> list[dict[str, object]]:
    cursor = (
        mongo._db["link_audit"]
        .find({"tenantId": tenant}, {"_id": 0})
        .sort([("runId", 1), ("sourceUrl", 1), ("position", 1)])
    )
    return await cursor.to_list()


@pytest.mark.integration
async def test_runs_are_kept_as_history_and_the_latest_completed_run_reads_back(
    mongo: MongoRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    assert await mongo.latest_link_audit(tenant) == ()
    await store(mongo, other, "run-x", rows("run-x", AT + timedelta(days=9)))
    await store(mongo, tenant, "run-1", rows("run-1"))
    later = rows("run-2", AT + timedelta(hours=1), verdict=None)
    await store(mongo, tenant, "run-2", later)

    assert await mongo.latest_link_audit(tenant) == ordered(later)
    stored = await documents(mongo, tenant)
    assert [(d["runId"], d["sourceUrl"], d["position"]) for d in stored] == [
        (run, source, position)
        for run in ("run-1", "run-2")
        for source, position in ((url("/a"), 0), (url("/a"), 3), (url("/b"), 1))
    ]
    assert {d["tenantId"] for d in stored} == {tenant}
    [generic] = [d for d in stored if d["runId"] == "run-1" and d["position"] == 1]
    assert generic["issueFlags"] == ["GENERIC"]
    assert generic["verdict"] == "REANCHOR"
    assert generic["proposedAnchor"] == "map case"
    assert (await mongo.latest_link_audit(other))[0].run_id == "run-x"


@pytest.mark.integration
async def test_an_uncompleted_or_miscounted_run_is_never_the_latest(
    mongo: MongoRepo, tenant: str
) -> None:
    first = rows("run-1")
    await store(mongo, tenant, "run-1", first)
    newer = rows("run-2", AT + timedelta(hours=1))
    await mongo.insert_link_audit(tenant, "run-2", newer)
    assert await mongo.latest_link_audit(tenant) == ordered(first)

    with pytest.raises(DatabaseWriteError, match="3 of 4"):
        await mongo.complete_link_audit(
            tenant, "run-2", audited_at=newer[0].audited_at, documents=4, edges=4
        )
    assert await mongo.latest_link_audit(tenant) == ordered(first)
    await mongo.complete_link_audit(
        tenant, "run-2", audited_at=newer[0].audited_at, documents=3, edges=3
    )
    assert await mongo.latest_link_audit(tenant) == ordered(newer)


@pytest.mark.integration
async def test_a_retried_run_rewrites_its_own_documents_only(mongo: MongoRepo, tenant: str) -> None:
    await store(mongo, tenant, "run-1", rows("run-1"))
    await store(mongo, tenant, "run-2", rows("run-2", AT + timedelta(hours=1)))
    retried = rows("run-2", AT + timedelta(hours=1), verdict=None)
    await store(mongo, tenant, "run-2", retried)

    stored = await documents(mongo, tenant)
    assert len(stored) == 6
    assert [d["verdict"] for d in stored if d["runId"] == "run-1" and d["position"] == 1] == [
        "REANCHOR"
    ]
    assert await mongo.latest_link_audit(tenant) == ordered(retried)


@pytest.mark.integration
async def test_an_audited_at_tie_goes_to_the_greater_run_id(mongo: MongoRepo, tenant: str) -> None:
    await store(mongo, tenant, "run-b", rows("run-b"))
    await store(mongo, tenant, "run-a", rows("run-a"))

    assert {r.run_id for r in await mongo.latest_link_audit(tenant)} == {"run-b"}


@pytest.mark.integration
async def test_invalid_audit_writes_are_rejected_before_any_document(
    mongo: MongoRepo, tenant: str
) -> None:
    results = rows("run-1")
    with pytest.raises(ValueError, match="run_id"):
        await mongo.insert_link_audit(tenant, " ", results)
    with pytest.raises(ValueError, match="belong to run"):
        await mongo.insert_link_audit(tenant, "run-2", results)
    with pytest.raises(ValueError, match="one document per edge"):
        await mongo.insert_link_audit(tenant, "run-1", [results[0], results[0]])
    with pytest.raises(ValueError, match="tenant_id"):
        await mongo.insert_link_audit("", "run-1", results)
    with pytest.raises(ValueError, match="negative"):
        await mongo.complete_link_audit(tenant, "run-1", audited_at=AT, documents=-1, edges=0)
    with pytest.raises(ValueError, match="run_id"):
        await mongo.complete_link_audit(tenant, "", audited_at=AT, documents=0, edges=0)
    assert await documents(mongo, tenant) == []
    assert await mongo.latest_link_audit(tenant) == ()


@pytest.mark.integration
async def test_a_latest_marker_without_a_run_fails_the_read(mongo: MongoRepo, tenant: str) -> None:
    await store(mongo, tenant, "run-1", rows("run-1"))
    await mongo._db["link_audit_runs"].insert_one(
        {"tenantId": tenant, "auditedAt": AT + timedelta(days=1)}
    )

    with pytest.raises(DatabaseReadError, match="has no run"):
        await mongo.latest_link_audit(tenant)
