"""The link audit stage on the planted tenant, against real Neo4j and MongoDB: every body link is
scored and judged, kept in link_audit until the next run and written onto its edge, one tenant
at a time. The planted data separate cleanly by construction, so full precision and recall prove
the plumbing, not the method."""

from __future__ import annotations

import importlib.util
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import test_audit_models as audit_models
from link_audit_seed import (
    AUDITED,
    LINKS,
    PAGES,
    PLACEHOLDER,
    PlantedLink,
    link_key,
    seed_link_audit,
    voyage,
)
from store_state import graph_state
from structlog.testing import capture_logs
from test_keyword_stage import url
from voyage_fakes import client

from linking_engine.audit.links import NO_STORED_SCORES
from linking_engine.errors import DatabaseWriteError
from linking_engine.models import AnchorRules, AuditEdge, IssueFlag, LinkAuditReport
from linking_engine.pipeline import link_audit as link_audit_stage
from linking_engine.pipeline.link_audit import audit_links, fixable_band, summarise_link_audit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import LinkAuditResult

# Planted by construction: missing a planted flag, or flagging a link not planted with it, is a
# fault in the plumbing.
MIN_PRECISION = 1.0
MIN_RECALL = 1.0


async def audit(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, cache_dir: Path
) -> tuple[LinkAuditReport, str]:
    return await audit_links(graph, mongo, tenant, cache_dir=cache_dir, voyage=client(voyage()))


def by_link(results: Sequence[LinkAuditResult]) -> dict[PlantedLink, LinkAuditResult]:
    found = {(r.source_url, r.position): r for r in results}
    assert len(found) == len(results) == len(AUDITED)
    return {link: found[link_key(link)] for link in AUDITED}


def named(links: set[PlantedLink]) -> list[str]:
    return sorted(f"{link.case} {link.source}->{link.target}" for link in links)


async def tenant_documents(mongo: MongoRepo, tenant: str) -> dict[str, list[dict[str, object]]]:
    found: dict[str, list[dict[str, object]]] = {}
    for name in sorted(await mongo._db.list_collection_names()):
        cursor = mongo._db[name].find({"tenantId": tenant}).sort("_id", 1)
        found[name] = await cursor.to_list()
    return found


@pytest.mark.integration
@pytest.mark.parametrize("a2", [True, False], ids=["A2", "A1"])
async def test_planted_fixture_precision_recall_per_flag(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, a2: bool
) -> None:
    await seed_link_audit(graph, mongo, tenant, scored=a2)

    with capture_logs() as logs:
        report, run_id = await audit(graph, mongo, tenant, tmp_path)

    results = by_link(await mongo.latest_link_audit(tenant))
    assert {r.run_id for r in results.values()} == {run_id} == {report.run_id}
    planted = {flag for link in AUDITED for flag in link.expected(a2=a2).flags}
    assert (IssueFlag.OFF_TOPIC in planted) is a2
    for flag in IssueFlag:
        truth = {link for link in AUDITED if flag in link.expected(a2=a2).flags}
        found = {link for link, r in results.items() if flag in r.issue_flags}
        if flag not in planted:
            assert not found, f"{flag} was never planted: {named(found)}"
            continue
        precision = len(truth & found) / len(found) if found else 0.0
        recall = len(truth & found) / len(truth)
        assert precision >= MIN_PRECISION, (
            f"{flag}: precision {precision:.2f}, min {MIN_PRECISION}; extra {named(found - truth)}"
        )
        assert recall >= MIN_RECALL, (
            f"{flag}: recall {recall:.2f}, min {MIN_RECALL}; missed {named(truth - found)}"
        )
    wrong = [
        (link.case, link.source, link.target, r.verdict, r.proposed_anchor)
        for link, r in results.items()
        if (r.verdict, r.proposed_anchor)
        != (link.expected(a2=a2).verdict, link.expected(a2=a2).proposal)
    ]
    assert not wrong, f"verdicts or proposals off the planted truth: {wrong}"
    copies = {link.target: r.fix_target for link, r in results.items() if r.fix_target}
    assert copies == {"/gear/dry-sack-print": url("/gear/dry-sack")}

    skipped = Counter(link.skipped for link in LINKS if link.skipped)
    assert (report.links, report.unverified) == (len(AUDITED), 1)
    assert (report.index_like_pages, report.listing_pages) == (2, 1)
    assert (report.sitemap_pages, report.sitemap_links) == (1, skipped["sitemap"])
    assert (report.paginated_pages, report.paginated_links) == (1, skipped["paginated"])
    assert report.by_flag == Counter(f for link in AUDITED for f in link.expected(a2=a2).flags)
    assert report.proposals == sum(link.expected(a2=a2).proposal is not None for link in AUDITED)
    assert report.embeddings is a2
    assert report.embeddings_skipped_reason == (None if a2 else NO_STORED_SCORES)
    # A2 adds the phrase-keyword cosine to every verified link whose target has a keyword.
    with_keyword = sum(r.keyword_alignment is not None for r in results.values())
    assert report.keyword_cosines == (with_keyword if a2 else 0)

    [line] = [entry for entry in logs if entry["event"] == "audit.complete"]
    assert (line["tenant_id"], line["run_id"], line["links"]) == (tenant, run_id, len(AUDITED))
    logged = " ".join(map(str, line.values()))
    assert not [page.path for page in PAGES if page.path in logged]
    assert PLACEHOLDER not in logged


@pytest.mark.integration
async def test_audit_keeps_the_latest_run_and_writes_edges_per_tenant(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-other"
    await seed_link_audit(graph, mongo, tenant)
    await seed_link_audit(graph, mongo, other)
    left_out = {link_key(link) for link in LINKS if link.skipped}
    # An audit from before sitemaps and paginated pages were left out.
    await graph._auto(
        "UNWIND $keys AS key "
        "MATCH (:Page {tenantId: $t, url: key[0]})-[r:LINKS_TO {position: key[1]}]->() "
        "SET r.auditRunId = 'old-run', r.verdict = 'REMOVE', r.issueFlags = ['OFF_TOPIC']",
        t=tenant,
        keys=[list(key) for key in left_out],
    )
    other_graph = await graph_state(graph, other)
    other_documents = await tenant_documents(mongo, other)

    with capture_logs() as logs:
        first, first_run = await audit(graph, mongo, tenant, tmp_path)
        second, second_run = await audit(graph, mongo, tenant, tmp_path)

    # The first run clears exactly the old audits on links now left out; the second, nothing.
    cleared = [e["stale_audits_cleared"] for e in logs if e["event"] == "audit.complete"]
    assert cleared == [len(left_out), 0]

    assert first_run != second_run
    assert (first.by_flag, first.by_verdict) == (second.by_flag, second.by_verdict)
    # The second run, once complete, replaced the first.
    stored = await mongo._db["link_audit"].find({"tenantId": tenant}).to_list()
    assert Counter(doc["runId"] for doc in stored) == {second_run: len(AUDITED)}
    pruned = [
        (e["pruned_documents"], e["pruned_runs"]) for e in logs if e["event"] == "audit.complete"
    ]
    assert pruned == [(0, 0), (len(AUDITED), 1)]
    latest = await mongo.latest_link_audit(tenant)
    assert {r.run_id for r in latest} == {second_run}
    assert len(latest) == len(AUDITED)

    edges = await graph._read(
        "MATCH (s:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "RETURN s.url AS source, r.position AS position, r.auditRunId AS run, "
        "r.verdict AS verdict, r.issueFlags AS flags, r.anchorQualityScore AS quality",
        t=tenant,
    )
    on_edge = {(row["source"], row["position"]): row for row in edges}
    assert len(on_edge) == len(LINKS)
    assert {on_edge[key]["run"] for key in left_out} == {None}, (
        "a sitemap or paginated link was audited"
    )
    assert {on_edge[key]["verdict"] for key in left_out} == {None}, "a stale verdict was kept"
    for result in latest:
        row = on_edge[(result.source_url, result.position)]
        assert row["run"] == second_run
        assert row["verdict"] == (None if result.verdict is None else result.verdict.value)
        assert row["flags"] == sorted(flag.value for flag in result.issue_flags)
        assert row["quality"] == result.anchor_quality_score

    # The other tenant's identical links were neither read nor written.
    assert await graph_state(graph, other) == other_graph
    assert await tenant_documents(mongo, other) == other_documents
    _, other_run = await audit(graph, mongo, other, tmp_path)
    assert {r.run_id for r in await mongo.latest_link_audit(other)} == {other_run}
    assert {r.run_id for r in await mongo.latest_link_audit(tenant)} == {second_run}


@pytest.mark.integration
async def test_a_tenant_without_links_gets_an_empty_run(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    report, run_id = await audit(graph, mongo, tenant, tmp_path)

    assert (report.links, report.unverified, report.healthy, report.source_pages) == (0, 0, 0, 0)
    assert report.embeddings_skipped_reason == NO_STORED_SCORES
    assert await mongo.latest_link_audit(tenant) == ()
    assert run_id == report.run_id
    assert "Keyword alignment: no scores." in summarise_link_audit(report)


@pytest.mark.parametrize("tenant_id", [" ", "..", "a/b"])
async def test_an_unusable_tenant_id_is_rejected_before_any_read(
    tmp_path: Path, tenant_id: str
) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await audit_links(None, None, tenant_id, cache_dir=tmp_path, voyage=None)  # type: ignore[arg-type]


class ShortStores:
    """Both stores for audit_links, with one healthy link; either write can come up short."""

    def __init__(self, *, edges_short: int = 0, documents_short: int = 0, stale: int = 0) -> None:
        self.edges_short = edges_short
        self.documents_short = documents_short
        self.stale = stale
        self.calls: list[str] = []

    async def audit_edges(self, tenant_id: str) -> list[AuditEdge]:
        return [
            AuditEdge(
                source_url="example.com/a",
                position=0,
                target_url="example.com/b",
                anchor_text="trail shoes",
                target_status_code=200,
                target_indexable=True,
            )
        ]

    async def ranked_keywords(self, tenant_id: str) -> dict[str, list[object]]:
        return {}

    async def get_anchor_rules(self, tenant_id: str) -> AnchorRules:
        return AnchorRules()

    async def page_titles(self, tenant_id: str) -> list[str]:
        return []

    async def write_link_audit(self, tenant_id: str, rows: Sequence[object]) -> int:
        self.calls.append("edges")
        return len(rows) - self.edges_short

    async def clear_stale_link_audit(self, tenant_id: str, run_id: str) -> int:
        self.calls.append("clear")
        return self.stale

    async def insert_link_audit(self, tenant_id: str, run_id: str, rows: Sequence[object]) -> int:
        self.calls.append("documents")
        return len(rows) - self.documents_short

    async def complete_link_audit(self, tenant_id: str, run_id: str, **counts: object) -> None:
        self.calls.append("complete")

    async def prune_link_audit(self, tenant_id: str, keep: str) -> tuple[int, int]:
        self.calls.append("prune")
        return 0, 0


@pytest.mark.parametrize(
    ("short", "store", "calls"),
    [
        ({"edges_short": 1}, "neo4j.*0 of 1", ["edges"]),
        ({"stale": 1}, "neo4j.*1 stale audits cleared but only 0", ["edges", "clear"]),
        ({"documents_short": 1}, "mongodb.*0 of 1", ["edges", "clear", "documents"]),
    ],
    ids=["edges", "stale", "documents"],
)
async def test_a_short_write_fails_the_run_before_it_is_marked_complete(
    tmp_path: Path, short: dict[str, int], store: str, calls: list[str]
) -> None:
    stores = ShortStores(**short)

    with pytest.raises(DatabaseWriteError, match=store):
        await audit_links(stores, stores, "test-tenant", cache_dir=tmp_path, voyage=None)  # type: ignore[arg-type]

    assert stores.calls == calls


async def test_complete_writes_mark_the_run_after_both_stores_then_prune(tmp_path: Path) -> None:
    stores = ShortStores()
    report, run_id = await audit_links(
        stores,
        stores,
        "test-tenant",
        cache_dir=tmp_path,
        voyage=None,  # type: ignore[arg-type]
    )
    assert stores.calls == ["edges", "clear", "documents", "complete", "prune"]
    assert (report.links, report.healthy, report.run_id) == (1, 1, run_id)


@pytest.mark.integration
async def test_the_row_the_edge_and_the_marker_hold_one_millisecond(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = datetime(2026, 9, 29, 12, 30, 15, 123456, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            return started

    monkeypatch.setattr(link_audit_stage, "datetime", Clock)
    await seed_link_audit(graph, mongo, tenant)

    _, run_id = await audit(graph, mongo, tenant, tmp_path)

    stamp = started.replace(microsecond=123000)
    assert {r.audited_at for r in await mongo.latest_link_audit(tenant)} == {stamp}
    rows = await graph._read(
        "MATCH (:Page {tenantId: $t})-[r:LINKS_TO {auditRunId: $run}]->() RETURN r.auditedAt AS at",
        t=tenant,
        run=run_id,
    )
    assert len(rows) == len(AUDITED)
    assert {row["at"].to_native() for row in rows} == {stamp}
    marker = await mongo._db["link_audit_runs"].find_one({"tenantId": tenant, "runId": run_id})
    assert marker is not None
    assert marker["auditedAt"] == stamp


def test_the_script_prints_the_summary_and_the_mlflow_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "link_audit.py"
    spec = importlib.util.spec_from_file_location("link_audit_script", path)
    assert spec is not None
    assert spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    report = LinkAuditReport(
        tenant_id="test-tenant",
        run_id="run-1",
        links=0,
        unverified=0,
        source_pages=0,
        index_like_pages=0,
        by_flag={},
        by_verdict={},
        by_reason={},
        healthy=0,
        ladder_pairs=0,
        proposals=0,
        cutoffs=(),
        embeddings=False,
        embeddings_skipped_reason=NO_STORED_SCORES,
        keyword_cosines=0,
        seconds=0.0,
        finished_at=datetime(2026, 9, 29, tzinfo=UTC),
    )
    calls: list[str] = []

    async def flow(tenant_id: str) -> tuple[LinkAuditReport, str]:
        calls.append(tenant_id)
        return report, "mlflow-run-7"

    monkeypatch.setattr(script, "link_audit_flow", flow)
    monkeypatch.setattr("sys.argv", ["link_audit.py", "--tenant", "test-tenant"])

    script.main()

    out = capsys.readouterr().out
    assert calls == ["test-tenant"]
    assert summarise_link_audit(report) in out
    assert out.rstrip().endswith("mlflow run mlflow-run-7")


@pytest.mark.parametrize(
    ("rate", "band"),
    [
        (0.0, "below 10%"),
        (0.0999, "below 10%"),
        (0.10, "10%-30%"),
        (0.30, "10%-30%"),
        (0.3001, "above 30%"),
        (1.0, "above 30%"),
    ],
)
def test_the_fixable_band_is_strict_below_10_and_strict_above_30_percent(
    rate: float, band: str
) -> None:
    assert fixable_band(rate).startswith(f"{band}: "), fixable_band(rate)


BELOW = "below 10%: the audit is a feature, discovery is the product"
BETWEEN = "10%-30%: both the audit and discovery matter"


@pytest.mark.parametrize(
    ("fields", "line"),
    [
        # Two verdicts over 25 links, 15 of them into uncrawled pages.
        (
            {"links": 25, "unverified": 15, "healthy": 8},
            f"Fixable-link rate: 8.0% of all audited links ({BELOW}); 20.0% of the links into "
            f"crawled pages ({BETWEEN}).",
        ),
        (audit_models.ALL_UNVERIFIED, f"Fixable-link rate: 0.0% of all audited links ({BELOW})."),
        (audit_models.NO_LINKS, "Fixable-link rate: no audited links."),
    ],
    ids=["both-rates", "all-unverified", "no-links"],
)
def test_the_summary_gives_each_fixable_rate_with_its_band(
    fields: dict[str, object], line: str
) -> None:
    assert line in summarise_link_audit(audit_models.report(**fields))
