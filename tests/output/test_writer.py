"""OutputWriter on real Mongo: a run is written in order under its own run id, completed, and
replaces the tenant's previous run; a run that stops midway leaves the previous one served; the
writes refused outside a writing run; and another tenant's identical output is never touched."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from linking_engine.errors import DatabaseUnavailableError, DatabaseWriteError
from linking_engine.models import (
    ActionType,
    PageProfile,
    RunInfo,
    ScorerName,
    SiteSummary,
    UnanchoredOut,
    UnanchoredReason,
)
from linking_engine.output.collections import (
    API_KEYS,
    PAGES,
    RUN_SCOPED,
    RUNS,
    UNANCHORED,
    from_document,
)
from linking_engine.output.reader import OutputReader
from linking_engine.output.writer import WRITE_BATCH, OutputWriter, milliseconds

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

AT = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
DATABASE = "linking_engine_test"


def url(path: str) -> str:
    return f"example.com/{path}"


def run_info(tenant: str, run_id: str) -> RunInfo:
    return RunInfo(
        tenant_id=tenant,
        run_id=run_id,
        status="writing",
        started_at=AT,
        scorer=ScorerName.BASELINE,
        weights_version="baseline-1",
        feature_code="0" * 64,
        package_version="0.1.0",
        limit_per_source=10,
        content_gap_limit=3,
    )


def summary(pages: int) -> SiteSummary:
    return SiteSummary(
        pages=pages,
        dead_end_pages=0,
        duplicate_groups=0,
        duplicate_copies=0,
        hubs=0,
        bridge_pairs=0,
        bridge_links=0,
        recommendations={ActionType.ADD_LINK: 1},
        sources_with_recommendations=1,
        sources_below_limit=1,
        links_audited=0,
        unverified_links=0,
        target_fixes=0,
    )


def profiles(*paths: str) -> list[PageProfile]:
    return [PageProfile(url=url(path), word_count=100, inbound=1, outbound=2) for path in paths]


@pytest.fixture
async def writer(mongo_uri: str) -> AsyncIterator[OutputWriter]:
    async with await OutputWriter.connect(mongo_uri, DATABASE) as found:
        await found.ensure_indexes()
        yield found


async def stored(writer: OutputWriter, collection: str, tenant: str) -> list[dict[str, object]]:
    cursor = (
        writer._db[collection]
        .find({"tenantId": tenant}, {"_id": 0})
        .sort([("runId", 1), ("ordinal", 1)])
    )
    return await cursor.to_list()


async def complete_run(writer: OutputWriter, tenant: str, run_id: str, *paths: str) -> None:
    await writer.begin(run_info(tenant, run_id))
    assert await writer.write(PAGES, tenant, run_id, profiles(*paths)) == len(paths)
    await writer.complete(tenant, run_id, summary(len(paths)), AT + timedelta(minutes=1))
    await writer.prune(tenant, run_id)


async def latest_complete(writer: OutputWriter, tenant: str) -> str | None:
    found = await writer._db[RUNS].find_one(
        {"tenantId": tenant, "status": "complete"}, sort=[("completed_at", -1)]
    )
    return None if found is None else str(found["runId"])


@pytest.mark.integration
async def test_a_run_is_written_in_order_completed_and_replaces_the_previous(
    writer: OutputWriter, tenant: str
) -> None:
    other = f"{tenant}-other"
    await complete_run(writer, other, "run-x", "a", "b")
    before = [await stored(writer, name, other) for name in (*RUN_SCOPED, RUNS)]

    await complete_run(writer, tenant, "run-1", "a", "b", "c")
    await writer.begin(run_info(tenant, "run-2"))
    written = profiles("a", "d")
    assert await writer.write(PAGES, tenant, "run-2", written) == 2
    assert await latest_complete(writer, tenant) == "run-1"

    done = await writer.complete(tenant, "run-2", summary(2), AT + timedelta(hours=1))
    assert await latest_complete(writer, tenant) == "run-2"
    assert await writer.prune(tenant, "run-2") == 4

    pages = await stored(writer, PAGES, tenant)
    assert [(d["runId"], d["ordinal"], d["url"]) for d in pages] == [
        ("run-2", 0, url("a")),
        ("run-2", 1, url("d")),
    ]
    assert [from_document(PageProfile, d) for d in pages] == written
    [run] = await stored(writer, RUNS, tenant)
    assert from_document(RunInfo, run) == done
    assert (done.status, done.completed_at, done.summary) == (
        "complete",
        AT + timedelta(hours=1),
        summary(2),
    )
    # The other tenant's identical urls were never read, rewritten or pruned.
    assert [await stored(writer, name, other) for name in (*RUN_SCOPED, RUNS)] == before


@pytest.mark.integration
async def test_a_run_that_stops_midway_leaves_the_previous_one_served(
    writer: OutputWriter, tenant: str
) -> None:
    await complete_run(writer, tenant, "run-1", "a", "b")
    await writer.begin(run_info(tenant, "run-2"))
    await writer.write(PAGES, tenant, "run-2", profiles("c"))

    assert await latest_complete(writer, tenant) == "run-1"
    assert {d["runId"] for d in await stored(writer, PAGES, tenant)} == {"run-1", "run-2"}

    # The next complete run clears the one that never completed.
    await complete_run(writer, tenant, "run-3", "d")
    assert {d["runId"] for d in await stored(writer, PAGES, tenant)} == {"run-3"}
    assert [d["runId"] for d in await stored(writer, RUNS, tenant)] == ["run-3"]


@pytest.mark.integration
async def test_large_listings_are_written_in_batches_with_contiguous_ordinals(
    writer: OutputWriter, tenant: str
) -> None:
    count = 2 * WRITE_BATCH + 5
    rows = [
        UnanchoredOut(
            source_url=url(f"s{i:05d}"),
            target_url=url("t"),
            reason=UnanchoredReason.SOURCE_PAGE_TEXT_UNAVAILABLE,
            advice="Recrawl the source page.",
            rank_in_source=1,
        )
        for i in range(count)
    ]
    await writer.begin(run_info(tenant, "run-1"))

    assert await writer.write(UNANCHORED, tenant, "run-1", iter(rows)) == count
    documents = await stored(writer, UNANCHORED, tenant)
    assert [d["ordinal"] for d in documents] == list(range(count))
    assert [d["source_url"] for d in documents] == [row.source_url for row in rows]


@pytest.mark.integration
async def test_writes_outside_a_writing_run_are_refused(writer: OutputWriter, tenant: str) -> None:
    with pytest.raises(ValueError, match="not a run-scoped"):
        await writer.write(API_KEYS, tenant, "run-1", [])
    with pytest.raises(DatabaseWriteError, match="was not begun"):
        await writer.write(PAGES, tenant, "run-1", profiles("a"))
    with pytest.raises(ValueError, match="begins as writing"):
        await writer.begin(
            run_info(tenant, "run-1").model_copy(
                update={"status": "complete", "completed_at": AT, "summary": summary(0)}
            )
        )
    with pytest.raises(ValueError, match="tenant_id"):
        await writer.write(PAGES, " ", "run-1", [])

    await writer.begin(run_info(tenant, "run-1"))
    with pytest.raises(DatabaseWriteError, match="duplicate key"):
        await writer.begin(run_info(tenant, "run-1"))
    await writer.write(PAGES, tenant, "run-1", profiles("a"))
    with pytest.raises(DatabaseWriteError, match="write output_pages"):
        await writer.write(PAGES, tenant, "run-1", profiles("b"))
    with pytest.raises(DatabaseWriteError, match="not complete"):
        await writer.prune(tenant, "run-1")

    await writer.complete(tenant, "run-1", summary(1), AT)
    with pytest.raises(DatabaseWriteError, match="is complete, not writing"):
        await writer.complete(tenant, "run-1", summary(1), AT)
    with pytest.raises(DatabaseWriteError, match="is complete, not writing"):
        await writer.write(PAGES, tenant, "run-1", profiles("c"))
    assert [d["url"] for d in await stored(writer, PAGES, tenant)] == [url("a")]


async def test_an_unreachable_or_misconfigured_store_fails_to_connect() -> None:
    with pytest.raises(DatabaseUnavailableError, match="server unavailable"):
        await OutputWriter.connect("mongodb://127.0.0.1:1", DATABASE, timeout_ms=200)
    with pytest.raises(DatabaseUnavailableError, match="invalid connection settings"):
        await OutputWriter.connect("no-scheme://", DATABASE)


@pytest.mark.integration
async def test_the_later_completion_is_served_even_when_the_earlier_has_no_fraction(
    writer: OutputWriter, tenant: str
) -> None:
    # As JSON text "...12:00:00Z" sorts after "...12:00:00.500000Z"; as dates it does not.
    for run_id, completed in (("run-1", AT), ("run-2", AT + timedelta(milliseconds=500))):
        await writer.begin(run_info(tenant, run_id))
        await writer.complete(tenant, run_id, summary(0), completed)

    assert await latest_complete(writer, tenant) == "run-2"
    served = await OutputReader(writer._db).latest_run(tenant)
    assert served is not None
    assert (served.run_id, served.completed_at) == ("run-2", AT + timedelta(milliseconds=500))
    stored = await writer._db[RUNS].find_one({"tenantId": tenant, "runId": "run-2"})
    assert stored is not None
    assert isinstance(stored["completed_at"], datetime)
    assert isinstance(stored["started_at"], datetime)


def test_moments_are_kept_to_the_millisecond_mongo_stores() -> None:
    assert milliseconds(AT + timedelta(microseconds=1_999)) == AT + timedelta(milliseconds=1)
