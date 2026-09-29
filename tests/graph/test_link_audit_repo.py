"""The link audit's graph I/O: one read of every body edge with both pages, and the batched,
tenant-scoped write-back of a run's scores and verdicts onto LINKS_TO."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from linking_engine.errors import DatabaseReadError, DatabaseWriteError
from linking_engine.models import ActionType, AuditEdge, IssueFlag, Link, LinkAuditResult, Page

if TYPE_CHECKING:
    from linking_engine.graph.repo import GraphRepo

AT = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


def url(path: str) -> str:
    return f"example.com{path}"


def link(source: str, target: str, position: int, anchor: str, **fields: object) -> Link:
    return Link.model_validate(
        {
            "source_url": url(source),
            "target_url": url(target),
            "position": position,
            "anchor_text": anchor,
            "surrounding_text": f"About {anchor}.",
            **fields,
        }
    )


async def set_pages(graph: GraphRepo, tenant: str, rows: dict[str, dict[str, object]]) -> None:
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) SET p += row.props",
        t=tenant,
        rows=[{"url": url(path), "props": props} for path, props in rows.items()],
    )


async def seed(graph: GraphRepo, tenant: str, anchor: str = "trail shoes") -> None:
    """/s links to /t, to a copy of /canon, and to a placeholder; /s2 links to /canon and to a
    broken page. Only /s -> /t carries #16's scores."""
    await graph.upsert_pages(
        tenant,
        [
            Page(url=url("/s"), status_code=200, is_indexable=True, language="en", word_count=412),
            Page(url=url("/s2"), status_code=200, is_indexable=True, language="de"),
            Page(url=url("/t"), status_code=200, is_indexable=False, language="en"),
            Page(url=url("/canon"), status_code=200, is_indexable=True, language="en"),
            Page(url=url("/copy"), status_code=200, is_indexable=True, language="en"),
            Page(url=url("/gone"), status_code=404),
        ],
    )
    await graph.upsert_placeholders(tenant, [url("/ghost")])
    await graph.replace_links(
        tenant,
        [url("/s"), url("/s2")],
        [
            link("/s", "/t", 0, anchor, is_follow=False),
            link("/s", "/copy", 1, "dry sack"),
            link("/s", "/ghost", 2, "unlisted gear"),
            link("/s2", "/canon", 0, "dry sack"),
            link("/s2", "/gone", 1, "retired boots"),
        ],
    )
    await set_pages(
        graph,
        tenant,
        {
            "/s": {"pageRankPercentile": 0.8},
            "/s2": {"pageRankPercentile": 0.4},
            "/t": {"pageRankPercentile": 0.2, "hubId": 2},
            "/canon": {"pageRankPercentile": 0.3, "hubId": -1, "duplicateGroup": "g",
                       "isCanonical": True},
            "/copy": {"pageRankPercentile": 0.1, "duplicateGroup": "g", "isCanonical": False},
        },
    )  # fmt: skip
    await graph._auto(
        "MATCH (:Page {tenantId: $t, url: $u})-[r:LINKS_TO {position: 0}]->() "
        "SET r.contextRelevance = 0.81, r.anchorTargetFit = 0.66",
        t=tenant,
        u=url("/s"),
    )


def expected_edges() -> list[AuditEdge]:
    s = {
        "source_url": url("/s"),
        "source_language": "en",
        "source_page_rank_percentile": 0.8,
        "source_word_count": 412,
    }
    s2 = {"source_url": url("/s2"), "source_language": "de", "source_page_rank_percentile": 0.4}
    return [
        AuditEdge(**s, position=0, target_url=url("/t"), anchor_text="trail shoes", is_follow=False,
                  context_relevance=0.81, anchor_target_fit=0.66, target_status_code=200,
                  target_indexable=False, target_hub_id=2, target_page_rank_percentile=0.2),
        AuditEdge(**s, position=1, target_url=url("/copy"), anchor_text="dry sack",
                  target_status_code=200, target_indexable=True, target_page_rank_percentile=0.1,
                  target_canonical_url=url("/canon")),
        AuditEdge(**s, position=2, target_url=url("/ghost"), anchor_text="unlisted gear",
                  target_placeholder=True),
        AuditEdge(**s2, position=0, target_url=url("/canon"), anchor_text="dry sack",
                  target_status_code=200, target_indexable=True, target_hub_id=-1,
                  target_page_rank_percentile=0.3),
        AuditEdge(**s2, position=1, target_url=url("/gone"), anchor_text="retired boots",
                  target_status_code=404),
    ]  # fmt: skip


def result(source: str, position: int, target: str, run: str, **fields: object) -> LinkAuditResult:
    return LinkAuditResult.model_validate(
        {
            "source_url": url(source),
            "position": position,
            "target_url": url(target),
            "run_id": run,
            "issue_flags": frozenset(),
            "verdict": None,
            "audited_at": AT,
            **fields,
        }
    )


def run_rows(run: str) -> list[LinkAuditResult]:
    return [
        result("/s", 0, "/t", run, anchor_quality_score=41.5, keyword_alignment=1.0,
               context_relevance=0.81, anchor_target_fit=0.66, equity_efficiency=0.64,
               issue_flags=frozenset({IssueFlag.NOFOLLOW, IssueFlag.NOINDEX_TARGET}),
               verdict=ActionType.FIX, reasons=("nofollow", "noindex")),
        result("/s", 1, "/copy", run, anchor_quality_score=100.0, keyword_alignment=1.0,
               verdict=ActionType.FIX, fix_target=url("/canon"), reasons=("a copy",)),
        result("/s", 2, "/ghost", run, unverified=True, reasons=("not crawled",)),
        result("/s2", 0, "/canon", run, anchor_quality_score=100.0, keyword_alignment=1.0,
               equity_efficiency=0.28),
        result("/s2", 1, "/gone", run, issue_flags=frozenset({IssueFlag.BROKEN}),
               verdict=ActionType.FIX, reasons=("404",)),
    ]  # fmt: skip


async def audit_props(graph: GraphRepo, tenant: str) -> dict[tuple[str, int], dict[str, object]]:
    rows = await graph._read(
        "MATCH (s:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "RETURN s.url AS source, r.position AS position, r.anchorQualityScore AS quality, "
        "r.keywordAlignment AS alignment, r.equityEfficiency AS equity, r.issueFlags AS flags, "
        "r.verdict AS verdict, r.auditedAt AS at, r.auditRunId AS run, "
        "r.contextRelevance AS context",
        t=tenant,
    )
    return {(str(row["source"]), int(str(row["position"]))): dict(row) for row in rows}


@pytest.mark.integration
async def test_audit_edges_reads_every_body_link_with_both_pages_and_no_other_tenant(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other, anchor="other tenant anchor")

    assert await graph.audit_edges(tenant) == expected_edges()
    assert next(iter(await graph.audit_edges(other))).anchor_text == "other tenant anchor"
    assert await graph.audit_edges(f"{tenant}-empty") == []
    with pytest.raises(ValueError, match="tenant_id"):
        await graph.audit_edges(" ")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"/canon": {"isCanonical": False}}, "has no canonical page"),
        ({"/t": {"duplicateGroup": "g", "isCanonical": True}}, "two canonical pages"),
    ],
    ids=["no-canonical", "two-canonicals"],
)
async def test_a_duplicate_group_without_exactly_one_canonical_fails_the_read(
    graph: GraphRepo, tenant: str, change: dict[str, dict[str, object]], problem: str
) -> None:
    await seed(graph, tenant)
    await set_pages(graph, tenant, change)

    with pytest.raises(DatabaseReadError, match=problem):
        await graph.audit_edges(tenant)


@pytest.mark.integration
async def test_write_link_audit_sets_each_run_and_a_null_removes_the_property(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other)
    before_other = await audit_props(graph, other)

    assert await graph.write_link_audit(tenant, run_rows("run-1")) == 5
    first = await audit_props(graph, tenant)
    assert first[(url("/s"), 0)] | {"at": None} == {
        "source": url("/s"), "position": 0, "quality": 41.5, "alignment": 1.0, "equity": 0.64,
        "flags": ["NOFOLLOW", "NOINDEX_TARGET"], "verdict": "FIX", "at": None, "run": "run-1",
        "context": 0.81,
    }  # fmt: skip
    assert first[(url("/s"), 0)]["at"].to_native() == AT  # type: ignore[union-attr]
    assert first[(url("/s2"), 1)]["flags"] == ["BROKEN"]
    ghost = first[(url("/s"), 2)]
    assert (ghost["quality"], ghost["flags"], ghost["verdict"]) == (None, [], None)

    # A second run clears what it no longer finds; #16's score on the edge stays.
    cleared = run_rows("run-2")
    cleared[0] = result("/s", 0, "/t", "run-2")
    assert await graph.write_link_audit(tenant, cleared) == 5
    again = await audit_props(graph, tenant)
    edge = again[(url("/s"), 0)]
    assert (edge["quality"], edge["alignment"], edge["equity"]) == (None, None, None)
    assert (edge["flags"], edge["verdict"], edge["run"], edge["context"]) == (
        [],
        None,
        "run-2",
        0.81,
    )
    # Rewriting the same run is idempotent.
    assert await graph.write_link_audit(tenant, cleared) == 5
    assert await audit_props(graph, tenant) == again
    assert await audit_props(graph, other) == before_other
    assert await graph.write_link_audit(tenant, []) == 0


@pytest.mark.integration
async def test_write_link_audit_batches_and_a_row_off_its_edge_rolls_its_batch_back(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    assert await graph.write_link_audit(tenant, run_rows("run-1"), batch_size=2) == 5

    rows = run_rows("run-2")
    # (/s2, 0) points at /canon, not /t: the second batch ([/s 2, /s2 0]) rolls back.
    rows[3] = result("/s2", 0, "/t", "run-2")
    with pytest.raises(DatabaseWriteError, match="rolled back"):
        await graph.write_link_audit(tenant, rows, batch_size=2)
    runs = {key: props["run"] for key, props in (await audit_props(graph, tenant)).items()}
    assert runs == {
        (url("/s"), 0): "run-2",
        (url("/s"), 1): "run-2",
        (url("/s"), 2): "run-1",
        (url("/s2"), 0): "run-1",
        (url("/s2"), 1): "run-1",
    }


@pytest.mark.integration
async def test_write_link_audit_rejects_mixed_runs_duplicate_edges_and_empty_batches(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    before = await audit_props(graph, tenant)
    rows = run_rows("run-1")

    with pytest.raises(ValueError, match="more than one audit run"):
        await graph.write_link_audit(tenant, [rows[0], rows[1].model_copy(update={"run_id": "x"})])
    with pytest.raises(ValueError, match=r"duplicate|once"):
        await graph.write_link_audit(tenant, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="batch_size"):
        await graph.write_link_audit(tenant, rows, batch_size=0)
    with pytest.raises(ValueError, match="tenant_id"):
        await graph.write_link_audit("", rows)
    assert await audit_props(graph, tenant) == before


@pytest.mark.integration
async def test_clear_stale_link_audit_removes_only_other_runs_audits(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other)
    await graph.write_link_audit(other, run_rows("run-1"))
    before_other = await audit_props(graph, other)
    await graph.write_link_audit(tenant, run_rows("run-1"))
    # The second run no longer reads /s2's links: their run-1 audits are stale.
    await graph.write_link_audit(tenant, run_rows("run-2")[:3])

    assert await graph.clear_stale_link_audit(tenant, "run-2") == 2
    props = await audit_props(graph, tenant)
    assert {key: row["run"] for key, row in props.items()} == {
        (url("/s"), 0): "run-2",
        (url("/s"), 1): "run-2",
        (url("/s"), 2): "run-2",
        (url("/s2"), 0): None,
        (url("/s2"), 1): None,
    }
    stale = props[(url("/s2"), 1)]
    assert (stale["flags"], stale["verdict"], stale["quality"], stale["at"]) == (None,) * 4
    assert props[(url("/s"), 0)]["context"] == 0.81
    assert await graph.clear_stale_link_audit(tenant, "run-2") == 0
    assert await audit_props(graph, other) == before_other


@pytest.mark.integration
async def test_clear_stale_link_audit_clears_in_batches_and_legacy_flags_too(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other)
    await graph.write_link_audit(tenant, run_rows("run-1"))
    # Labels an older ingest left on /s2's edges, with no run id.
    for t in (tenant, other):
        await graph._auto(
            "MATCH (:Page {tenantId: $t, url: $u})-[r:LINKS_TO]->() "
            "REMOVE r.auditRunId, r.auditedAt SET r.issueFlags = ['BROKEN'], r.verdict = 'FIX'",
            t=t,
            u=url("/s2"),
        )
    before_other = await audit_props(graph, other)
    await graph.write_link_audit(tenant, run_rows("run-3")[:1])

    assert await graph.clear_stale_link_audit(tenant, "run-3", batch_size=1) == 4
    runs = {key: row["run"] for key, row in (await audit_props(graph, tenant)).items()}
    assert runs == {key: "run-3" if key == (url("/s"), 0) else None for key in runs}
    assert all(
        row["verdict"] is None and row["flags"] is None
        for key, row in (await audit_props(graph, tenant)).items()
        if key != (url("/s"), 0)
    )
    assert await audit_props(graph, other) == before_other
    assert sum(row["verdict"] == "FIX" for row in before_other.values()) == 2
    with pytest.raises(ValueError, match="run_id"):
        await graph.clear_stale_link_audit(tenant, " ")
    with pytest.raises(ValueError, match="batch_size"):
        await graph.clear_stale_link_audit(tenant, "run-3", batch_size=0)


@pytest.mark.integration
async def test_malformed_stored_values_fail_the_audit_read(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant)
    await set_pages(graph, tenant, {"/s": {"pageRankPercentile": 1.5}})

    with pytest.raises(DatabaseReadError, match="audit edges"):
        await graph.audit_edges(tenant)
