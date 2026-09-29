"""The link audit's MLflow run: counts by flag and verdict, the tenant's cut-offs, score
histograms as step metrics and tables, in the tenant's analytics experiment; no url anywhere."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import pytest
from link_audit_seed import (
    AUDITED,
    LINKS,
    PAGES,
    PLACEHOLDER,
    anchor_facts,
    audit_edges,
    planted_proposals,
)
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text

from linking_engine.audit.links import (
    CONTEXT_SPLIT,
    NO_STORED_SCORES,
    assess,
    audit_report,
    audit_scope,
    decide,
    ladder_pairs,
)
from linking_engine.ml.tracking import (
    EXPERIMENT_KIND,
    EXPERIMENT_KIND_TAG,
    analytics_experiment,
    link_audit_metrics,
    log_link_audit,
)
from linking_engine.models import ActionType, IssueFlag, LinkAuditReport
from linking_engine.models.relevance import HISTOGRAM_BINS
from linking_engine.pipeline.link_audit import summarise_link_audit

TENANT = "acme"
AT = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
ARTIFACTS = {
    "audit_reasons.json",
    "audit_cutoffs.json",
    "audit_histogram.json",
    "report.json",
    "summary.md",
}


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:  # type: ignore[no-untyped-def]
    """Runs go to a throwaway local store, never the remote server."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def planted_report(*, a2: bool = True) -> LinkAuditReport:
    scope = audit_scope(audit_edges(a2=a2))
    assessment = assess(scope.edges, anchor_facts(), scope=scope)
    outcome = decide(assessment, planted_proposals(), run_id="run-1", audited_at=AT)
    return audit_report(
        TENANT,
        "run-1",
        assessment,
        outcome,
        ladder_pairs=len(ladder_pairs(assessment)),
        vectors_skipped_reason=None,
        seconds=1.5,
        finished_at=AT,
    )


def table(run_id: str, name: str) -> dict[str, list[object]]:
    stored = load_dict(f"runs:/{run_id}/{name}")
    columns = stored["columns"]
    return {column: [row[i] for row in stored["data"]] for i, column in enumerate(columns)}


def everything_logged(client: MlflowClient, run_id: str) -> str:
    run = client.get_run(run_id)
    tables = [table(run_id, name) for name in ARTIFACTS if name.startswith("audit_")]
    return " ".join(
        [
            *map(str, run.data.params.values()),
            *map(str, run.data.tags.values()),
            *run.data.metrics,
            *(str(v) for rows in tables for column in rows.values() for v in column),
            load_text(f"runs:/{run_id}/report.json"),
            load_text(f"runs:/{run_id}/summary.md"),
        ]
    )


def test_mlflow_audit_run_has_counts_and_no_urls(local_mlflow: str) -> None:
    report = planted_report()
    summary = summarise_link_audit(report)

    run_id = log_link_audit(report, summary)

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == analytics_experiment(TENANT)
    assert experiment.tags[EXPERIMENT_KIND_TAG] == EXPERIMENT_KIND
    assert (run.info.run_name, run.data.tags["stage"]) == ("link audit", "link-audit")
    assert run.data.tags["audit_run_id"] == "run-1"
    assert run.data.tags["mlflow.note.content"] == summary

    metrics = run.data.metrics
    stepped = {name for name in metrics if name.endswith("_hist")}
    assert len(stepped) == 5
    assert {k: v for k, v in metrics.items() if k not in stepped} == pytest.approx(
        link_audit_metrics(report)
    )
    flags = Counter(flag for link in AUDITED for flag in link.a2.flags)
    verdicts = Counter(link.a2.verdict for link in AUDITED if link.a2.verdict)
    for flag in IssueFlag:
        assert metrics[f"flag_{flag.value.lower()}"] == flags[flag], flag
    for verdict in (ActionType.FIX, ActionType.REANCHOR, ActionType.REMOVE):
        assert metrics[f"verdict_{verdict.value.lower()}"] == verdicts[verdict], verdict
    assert (metrics["links"], metrics["unverified"], metrics["index_like_pages"]) == (
        len(AUDITED), 1, 2
    )  # fmt: skip
    assert metrics["embeddings"] == 1.0
    assert 0.34 < metrics[f"cutoff_{CONTEXT_SPLIT}"] < 0.74

    assert report.keyword_alignment is not None
    history = sorted(
        client.get_metric_history(run_id, "keyword_alignment_hist"), key=lambda m: m.step
    )
    assert [m.step for m in history] == list(range(HISTOGRAM_BINS))
    assert [m.value for m in history] == list(report.keyword_alignment.histogram)
    assert sum(m.value for m in history) == report.keyword_alignment.count

    assert {a.path for a in client.list_artifacts(run_id)} == ARTIFACTS
    reasons = table(run_id, "audit_reasons.json")
    assert dict(zip(reasons["reason"], reasons["links"], strict=True)) == {
        reason.value: n for reason, n in report.by_reason.items()
    }
    cutoffs = table(run_id, "audit_cutoffs.json")
    assert cutoffs["cutoff"] == [cutoff.name for cutoff in report.cutoffs]
    histogram = table(run_id, "audit_histogram.json")
    assert set(histogram["score"]) == {
        "keyword_alignment", "context_relevance", "anchor_target_fit", "equity_efficiency",
        "anchor_quality",
    }  # fmt: skip
    assert load_dict(f"runs:/{run_id}/report.json") == report.model_dump(mode="json")

    logged = everything_logged(client, run_id)
    assert "OVER_OPTIMISED_ANCHOR" in logged, "the reasons table was not read"
    assert "outbound_fence" in logged, "the cut-offs table was not read"
    paths = [page.path for page in PAGES] + [PLACEHOLDER]
    leaked = [path for path in paths if path in logged]
    assert not leaked, f"urls leaked into the MLflow run: {leaked}"
    anchors = [link.anchor for link in LINKS if len(link.anchor) > 10]
    assert not [anchor for anchor in anchors if anchor in logged], "anchor texts leaked"


def test_an_a1_audit_run_logs_why_it_had_no_embeddings(local_mlflow: str) -> None:
    report = planted_report(a2=False)

    run_id = log_link_audit(report, summarise_link_audit(report))

    run = MlflowClient(local_mlflow).get_run(run_id)
    assert run.data.params["embeddings_skipped_reason"] == NO_STORED_SCORES
    assert run.data.metrics["embeddings"] == 0.0
    assert run.data.metrics["flag_off_topic"] == 0.0
    assert run.data.metrics["verdict_remove"] == 0.0
    assert f"cutoff_{CONTEXT_SPLIT}" not in run.data.metrics
    assert "context_relevance_mean" not in run.data.metrics
    summary = load_text(f"runs:/{run_id}/summary.md")
    assert summary.startswith(f"Link audit run-1 of tenant {TENANT}: {len(AUDITED)} body links")
    assert f"A1 only: {NO_STORED_SCORES}" in summary
