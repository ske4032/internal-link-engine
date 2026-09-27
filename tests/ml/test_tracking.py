from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mlflow import MlflowClient

from linking_engine.ml.tracking import analytics_experiment, analytics_metrics, log_analytics
from linking_engine.models import CentralityReport, CommunityReport, OrphanLabel, PassReport

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs go to a throwaway local store, never the remote server. Only the environment is
    set: an explicit set_tracking_uri would outlive the test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def found(pages: int, drift: float | None = None) -> PassReport:
    return PassReport(
        pages=pages,
        edges=pages * 2,
        communities=3 if pages else 0,
        singletons=0,
        largest_community_pct=0.4 if pages else 0.0,
        median_community_size=10.0 if pages else 0.0,
        modularity=0.5 if pages else 0.0,
        disconnected_communities=0,
        seed_stability_ari_mean=0.9 if pages else None,
        seed_stability_ari_min=0.8 if pages else None,
        drift_ari=drift,
        pillars=3 if pages else 0,
        runtime_s=0.1,
        stability_s=0.2,
    )


def reports(tenant: str = "acme") -> tuple[CentralityReport, CommunityReport]:
    centrality = CentralityReport(
        tenant_id=tenant, pages=30, placeholders=2, pagerank_s=0.01, betweenness_s=0.02, write_s=0.1
    )
    communities = CommunityReport(
        tenant_id=tenant,
        crawled_pages=30,
        seen_not_crawled=2,
        link=found(28, drift=0.95),
        keyword=found(0),
        content=found(30),
        keywords=0,
        keywords_dropped=0,
        pages_with_keywords=0,
        pages_with_embeddings=30,
        agreement_link_content=0.3,
        orphans=4,
        dead_ends=5,
        orphan_labels={OrphanLabel.MENUS_ONLY: 3, OrphanLabel.NOT_LINKED: 1},
        write_s=0.2,
    )
    return centrality, communities


def test_metrics_are_flat_and_leave_out_what_was_not_computed() -> None:
    metrics = analytics_metrics(*reports())
    assert metrics["link_modularity"] == 0.5
    assert metrics["link_drift_ari"] == 0.95
    assert metrics["centrality_pages"] == 30
    assert metrics["orphans_menus_only"] == 3
    assert metrics["agreement_link_content"] == 0.3
    assert "keyword_seed_stability_ari_mean" not in metrics
    assert "agreement_link_keyword" not in metrics


def test_a_run_is_logged_to_the_tenants_experiment_with_its_description(local_mlflow: str) -> None:
    run_id = log_analytics(*reports(), "Graph analytics for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == analytics_experiment("acme") == "analytics-acme"
    assert run.data.tags["mlflow.note.content"] == "Graph analytics for tenant acme."
    assert run.data.tags["tenant_id"] == "acme"
    assert run.data.params["resolution"] == "1.0"
    assert run.data.metrics["content_pages"] == 30
    assert {a.path for a in client.list_artifacts(run_id)} == {"report.json", "summary.md"}


def test_reports_from_two_tenants_are_refused(local_mlflow: str) -> None:
    centrality, _ = reports("acme")
    _, communities = reports("globex")
    with pytest.raises(ValueError, match="same tenant"):
        log_analytics(centrality, communities, "x")
