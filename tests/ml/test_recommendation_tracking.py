"""The recommendations MLflow run and the tenant's latest quality snapshot it is served with: tags,
params, metrics and artefacts without urls, and the quality lookup that never fails the run."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import mlflow
import pytest
import quality_factories as make
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text
from mlflow.entities import Metric
from mlflow.exceptions import MlflowException
from structlog.testing import capture_logs

from linking_engine.ml.tracking import (
    analytics_experiment,
    latest_quality,
    log_quality,
    log_recommendations,
    recommendation_metrics,
)
from linking_engine.models import (
    ActionType,
    ExclusionReason,
    IssueFlag,
    OrphanLabel,
    OrphanSlotReason,
    RecommendationReport,
    ScorerName,
    SiteSummary,
    UnanchoredReason,
)

if TYPE_CHECKING:
    from pathlib import Path

TENANT = "acme"
SUMMARY = "Recommendations run run-1 of tenant acme."


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs go to a throwaway local store, never the remote server."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def report(tenant: str = TENANT) -> RecommendationReport:
    return RecommendationReport(
        tenant_id=tenant,
        run_id="run-1",
        scorer=ScorerName.LEARNED,
        model_version="4",
        limit_per_source=10,
        content_gap_limit=3,
        words_per_link=200,
        guaranteed_inbound_links=2,
        guaranteed_inbound_below=1,
        max_suggested_inbound=5,
        summary=SiteSummary(
            pages=12,
            excluded_pages={ExclusionReason.SITEMAP: 1},
            orphan_pages={OrphanLabel.MENUS_ONLY: 2, OrphanLabel.NOT_LINKED: 1},
            dead_end_pages=1,
            duplicate_groups=1,
            duplicate_copies=2,
            hubs=3,
            bridge_pairs=2,
            bridge_links=4,
            recommendations={ActionType.ADD_LINK: 20, ActionType.REMOVE: 3},
            tiers={1: 2, 2: 6, 3: 12},
            sources_with_recommendations=4,
            sources_below_limit=2,
            links_audited=40,
            unverified_links=2,
            audit_flags={IssueFlag.GENERIC: 5},
            unanchored={UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD: 7},
            target_fixes=2,
            suggested_links=14,
            reserve_links=6,
            guaranteed_pages=3,
            orphan_slots=2,
            guarantees_unmet={OrphanSlotReason.NO_ANCHOR: 1},
            orphans_reached=2,
            orphans_to_pillar=1,
            inbound_gini=0.42,
            pages_at_cap=3,
            links_moved_by_cap=4,
            links_dropped_by_cap=1,
            top10_inbound_share=0.35,
        ),
        pairs_not_assessed=1,
        seconds=2.5,
        finished_at=datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
    )


def test_a_run_logs_its_counts_report_and_description(local_mlflow: str) -> None:
    run_id = log_recommendations(report(), SUMMARY)

    run = MlflowClient(local_mlflow).get_run(run_id)
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
    assert (tags["tenant_id"], tags["stage"], tags["kind"]) == (
        TENANT,
        "recommendations",
        "pipeline",
    )
    assert (tags["output_run_id"], tags["scorer"], tags["model_version"]) == (
        "run-1",
        "learned",
        "4",
    )
    assert params == {
        "limit_per_source": "10",
        "content_gap_limit": "3",
        "words_per_link": "200",
        "guaranteed_inbound_links": "2",
        "guaranteed_inbound_below": "1",
        "max_suggested_inbound": "5",
        "scorer": "learned",
        "model_version": "4",
    }
    assert (metrics["limit_per_source"], metrics["content_gap_limit"]) == (10.0, 3.0)
    assert metrics == recommendation_metrics(report())
    assert (metrics["action_add_link"], metrics["action_fix"], metrics["recommendations"]) == (
        20.0,
        0.0,
        23.0,
    )
    assert (metrics["tier_3"], metrics["orphan_pages"], metrics["orphan_not_linked"]) == (
        12.0,
        3.0,
        1.0,
    )
    assert metrics["unanchored_target_page_has_no_keyword"] == 7.0
    assert (metrics["flag_generic"], metrics["excluded_sitemap"]) == (5.0, 1.0)
    assert (metrics["suggested_links"], metrics["reserve_links"], metrics["orphan_slots"]) == (
        14.0,
        6.0,
        2.0,
    )
    assert (metrics["guaranteed_pages"], metrics["orphans_reached"]) == (3.0, 2.0)
    assert (metrics["orphans_to_pillar"], metrics["inbound_gini"]) == (1.0, 0.42)
    assert (metrics["guarantees_unmet"], metrics["unmet_no_anchor"]) == (1.0, 1.0)
    assert (metrics["unmet_no_relevant_source"], metrics["unmet_sources_full"]) == (0.0, 0.0)
    assert (metrics["pages_at_cap"], metrics["top10_inbound_share"]) == (3.0, 0.35)
    assert (metrics["links_moved_by_cap"], metrics["links_dropped_by_cap"]) == (4.0, 1.0)
    assert load_text(f"runs:/{run_id}/summary.md") == SUMMARY
    assert load_dict(f"runs:/{run_id}/metrics.json") == metrics
    assert load_dict(f"runs:/{run_id}/report.json")["summary"]["pages"] == 12
    experiment = MlflowClient(local_mlflow).get_experiment_by_name(analytics_experiment(TENANT))
    assert experiment is not None
    assert run.info.experiment_id == experiment.experiment_id


def test_a_run_without_suggested_links_logs_no_gini_or_top_ten_share() -> None:
    unset = {"inbound_gini": None, "top10_inbound_share": None}
    quiet = report().model_copy(update={"summary": report().summary.model_copy(update=unset)})

    assert not set(unset) & set(recommendation_metrics(quiet))
    assert set(unset) <= set(recommendation_metrics(report()))


def test_no_quality_run_gives_no_snapshot_and_creates_no_experiment(local_mlflow: str) -> None:
    with capture_logs() as logs:
        assert latest_quality(TENANT) is None

    assert [e["event"] for e in logs] == ["recommendations.no_quality_run"]
    assert MlflowClient(local_mlflow).get_experiment_by_name(analytics_experiment(TENANT)) is None


def test_the_snapshot_is_the_tenants_latest_finished_quality_run(local_mlflow: str) -> None:
    log_quality(make.report(TENANT, seconds=1.0), "first")
    latest = log_quality(make.report(TENANT, seconds=2.0), "second")
    log_quality(make.report("other", seconds=3.0), "another tenant")
    MlflowClient(local_mlflow).log_batch(
        latest, metrics=[Metric("recall_at_20", math.nan, int(time.time() * 1000), 1)]
    )

    found = latest_quality(TENANT)

    assert found is not None
    run = MlflowClient(local_mlflow).get_run(latest)
    assert found.mlflow_run_id == latest
    assert found.finished_at == datetime.fromtimestamp((run.info.end_time or 0) / 1000, UTC)
    assert found.metrics["seconds"] == 2.0
    assert "recall_at_20" not in found.metrics


def test_an_unreachable_tracking_server_gives_no_snapshot(
    local_mlflow: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(_name: str) -> None:
        raise MlflowException("API request failed")

    monkeypatch.setattr(mlflow, "get_experiment_by_name", refused)
    with capture_logs() as logs:
        assert latest_quality(TENANT) is None

    [event] = logs
    assert (event["event"], event["error"]) == (
        "recommendations.quality_unavailable",
        "MlflowException",
    )
