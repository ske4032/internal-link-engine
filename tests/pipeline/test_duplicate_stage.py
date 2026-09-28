"""The duplicate stage on a real Neo4j: read, group, write, one log line."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from structlog.testing import capture_logs

from linking_engine.models import DuplicateGroup, DuplicateReport, Link, Page
from linking_engine.pipeline.duplicates import find_duplicates, summarise_duplicates

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo

ARTICLE = "a" * 64
GUIDE = "b" * 64


def url(path: str) -> str:
    return f"example.com{path}"


PAGES = {
    "/blog/post": ARTICLE,
    "/post": ARTICLE,
    "/topics/post": ARTICLE,
    "/guide": GUIDE,
    "/guides/guide": GUIDE,
    "/home": "c" * 64,
    "/about": "d" * 64,
}
LINKS = [("/home", "/post"), ("/about", "/post"), ("/home", "/guides/guide")]


async def seed(graph: GraphRepo, tenant: str) -> None:
    await graph.upsert_pages(
        tenant,
        [
            Page(url=url(path), status_code=200, body_hash=body, language="en", word_count=80)
            for path, body in PAGES.items()
        ],
    )
    await graph.replace_links(
        tenant,
        [url("/home"), url("/about")],
        [
            Link(
                source_url=url(s),
                target_url=url(t),
                position=i,
                anchor_text="x",
                surrounding_text="",
            )
            for i, (s, t) in enumerate(LINKS)
        ],
    )


def logged_urls(line: dict[str, object]) -> list[str]:
    logged = " ".join(str(value) for value in line.values())
    return [url(path) for path in PAGES if url(path) in logged]


@pytest.mark.integration
async def test_the_stage_groups_stores_and_logs_the_tenants_duplicates(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other)
    await graph.write_duplicate_groups(
        other,
        (DuplicateGroup(group_id=0, canonical=url("/topics/post"), copies=(url("/post"),)),),
    )

    with capture_logs() as logs:
        report = await find_duplicates(graph, tenant)

    # /guides/guide and /post are linked, so they win over their shorter or earlier copies.
    assert report.groups == (
        DuplicateGroup(group_id=0, canonical=url("/guides/guide"), copies=(url("/guide"),)),
        DuplicateGroup(
            group_id=1, canonical=url("/post"), copies=(url("/blog/post"), url("/topics/post"))
        ),
    )
    assert (report.tenant_id, report.pages_in_groups, report.non_canonical) == (tenant, 5, 3)
    assert report.largest_group == 3
    assert await graph.non_canonical_copies(tenant) == {
        url("/guide"),
        url("/blog/post"),
        url("/topics/post"),
    }
    assert await graph.non_canonical_copies(other) == {url("/post")}, "other tenant untouched"

    [line] = [entry for entry in logs if entry["event"] == "graph.duplicates"]
    assert (line["tenant_id"], line["groups"], line["non_canonical"]) == (tenant, 2, 3)
    assert line["pages_written"] == 5
    assert logged_urls(line) == [], "no urls in logs"


@pytest.mark.integration
async def test_groups_and_ids_are_identical_across_runs_until_the_data_changes(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)

    first = await find_duplicates(graph, tenant)
    second = await find_duplicates(graph, tenant)

    assert second.groups == first.groups

    await graph.upsert_pages(
        tenant,
        [
            Page(
                url=url("/guide"), status_code=200, body_hash="e" * 64, language="en", word_count=90
            )
        ],
    )
    changed = await find_duplicates(graph, tenant)

    assert changed.groups == (
        DuplicateGroup(
            group_id=0, canonical=url("/post"), copies=(url("/blog/post"), url("/topics/post"))
        ),
    )
    [guide, guides] = await graph.get_pages(tenant, [url("/guide"), url("/guides/guide")])
    assert (guide.duplicate_group, guide.is_canonical) == (None, None)
    assert (guides.duplicate_group, guides.is_canonical) == (None, None)


def report_of(*groups: DuplicateGroup) -> DuplicateReport:
    sizes = [1 + len(g.copies) for g in groups]
    return DuplicateReport(
        tenant_id="acme",
        groups=groups,
        pages_in_groups=sum(sizes),
        non_canonical=sum(sizes) - len(groups),
        largest_group=max(sizes, default=0),
        seconds=1.5,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


def test_the_summary_states_the_scope_the_counts_and_the_canonical_rule() -> None:
    summary = summarise_duplicates(
        report_of(
            DuplicateGroup(group_id=0, canonical=url("/a"), copies=(url("/b"), url("/c"))),
            DuplicateGroup(group_id=1, canonical=url("/d"), copies=(url("/e"),)),
        )
    )

    for fact in ("acme", "2 groups over 5 pages", "largest 3", "3 non-canonical", "1.5 s"):
        assert fact in summary, f"{fact!r} missing from:\n{summary}"
    assert "most inbound body links" in summary


def test_the_summary_of_a_tenant_without_duplicates_says_so() -> None:
    assert "No duplicates found." in summarise_duplicates(report_of())


async def test_a_blank_tenant_is_refused_before_any_read() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await find_duplicates(None, " ")  # type: ignore[arg-type]
